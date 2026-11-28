// DLAA: NVIDIA DLSS at the scene's own size (REKARINA_DLAA, rekarina.ini dlaa). NGX is loaded at run time from the driver's
// NGX core: on Windows _nvngx.dll, found through HKLM\SOFTWARE\NVIDIA Corporation\Global\NGXCore; on Linux
// libnvidia-ngx.so.1, which the NVIDIA driver installs on the library path. NGX loads the DLSS model from the folder of
// the executable: nvngx_dlss.dll on Windows, libnvidia-ngx-dlss.so.<version> on Linux. No NVIDIA SDK code or header is
// compiled in: the few NGX types used are declared here from the SDK's documented ABI.
//
// The core's parameter object is a C++ class whose virtual functions are called through its vtable: the Set overloads
// in slots 0-7 and the Get overloads in 8-15. MSVC (the Windows core) lays out each group of overloads in reverse
// declaration order, the Itanium C++ ABI (the Linux core, built with GCC) in declaration order. vtable_ok checks the
// layout with a set-then-get round trip before anything relies on it.
//
// Where DLAA runs in the frame, the jitter and the motion vectors: see the sections below.
#ifdef _WIN32
#include <windows.h>
#include <immintrin.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <wchar.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "layout.h"
#include "nvn_api.h"
#include "renderer_mcp.h"
#include "scene_replay.h"

#include "platform/platform.h"
#include "tex.h"
#include "vk.h"
#include "vk_dlaa.h"
#include "vk_exec.h"
#include "vk_dlssg.h"
#include "vk_fg.h"
#include "vk_ffx.h"
#include "vk_rec.h"
#include "dlaa_mv_spv.h"

static void hudless_scene_target(const VkTexImage *texture, VkImage image);

typedef int NgxResult;   // NVSDK_NGX_Result: 1 success, 0xBAD00000 | n failures
#define NGX_OK(r) (((unsigned)(r) & 0xFFF00000u) != 0xBAD00000u)
#define NGX_VERSION_API 0x15
#define NGX_FEATURE_SUPERSAMPLING 1
#define NGX_PERFQ_DLAA 5   // NVSDK_NGX_PerfQuality_Value: 0 performance, 1 balanced, 2 quality, 3 ultra performance
typedef struct NgxParam NgxParam;
typedef struct NgxHandle NgxHandle;
typedef struct { void *slot[17]; } NgxParamVtbl;
struct NgxParam { const NgxParamVtbl *vt; };
typedef struct {   // NVSDK_NGX_ImageViewInfo_VK
    VkImageView view;
    VkImage image;
    VkImageSubresourceRange range;
    VkFormat format;
    unsigned width, height;
} NgxImageView;
typedef struct {   // NVSDK_NGX_Resource_VK
    union { NgxImageView img; struct { VkBuffer buffer; unsigned bytes; } buf; } r;
    int type;      // 0 image view, 1 buffer
    _Bool rw;
} NgxResourceVK;

typedef NgxResult (*PFN_Init)(unsigned long long app, const wchar_t *path, VkInstance, VkPhysicalDevice, VkDevice,
                              PFN_vkGetInstanceProcAddr, PFN_vkGetDeviceProcAddr, int version, const void *info);
typedef NgxResult (*PFN_ReqExt)(unsigned *ni, const char ***ie, unsigned *nd, const char ***de);
typedef NgxResult (*PFN_Params)(NgxParam **);
typedef NgxResult (*PFN_DestroyParams)(NgxParam *);
typedef NgxResult (*PFN_Create)(VkCommandBuffer, int feature, NgxParam *, NgxHandle **);
typedef NgxResult (*PFN_Evaluate)(VkCommandBuffer, const NgxHandle *, const NgxParam *, void *callback);
typedef NgxResult (*PFN_Release)(NgxHandle *);
// The driver export takes an output count; the public SDK wrapper supplies it.
typedef NgxResult (*PFN_Shutdown)(VkDevice, unsigned *remaining);

static struct {
    void *core;   // the NGX core library (an HMODULE on Windows)
    PFN_Init init;
    PFN_ReqExt req_ext;
    PFN_Params caps, alloc_params;
    PFN_DestroyParams destroy_params;
    PFN_Create create;
    PFN_Evaluate evaluate;
    PFN_Release release;
    PFN_Shutdown shutdown;
    int initialized;
    int tried, loaded, available;
    int wanted;   // -1 not read yet
} N = {.wanted = -1};

typedef void (*SetI)(NgxParam *, const char *, int);
typedef void (*SetU)(NgxParam *, const char *, unsigned);
typedef void (*SetF)(NgxParam *, const char *, float);
typedef void (*SetP)(NgxParam *, const char *, void *);
typedef NgxResult (*GetI)(const NgxParam *, const char *, int *);
typedef NgxResult (*GetU)(const NgxParam *, const char *, unsigned *);
typedef NgxResult (*GetF)(const NgxParam *, const char *, float *);
typedef NgxResult (*GetP)(const NgxParam *, const char *, void **);
// The SDK declares Set(u64), Set(float), Set(double), Set(u32), Set(int), Set(ID3D11Resource *), Set(ID3D12Resource *),
// Set(void *), then the Get overloads in the same order, then Reset.
#ifdef _WIN32
enum { S_P = 0, S_D12, S_D11, S_I, S_U32, S_D, S_F, S_U64, G_P, G_D12, G_D11, G_I, G_U32, G_D, G_F, G_U64 };
#else
enum { S_U64 = 0, S_F, S_D, S_U32, S_I, S_D11, S_D12, S_P, G_U64, G_F, G_D, G_U32, G_I, G_D11, G_D12, G_P };
#endif
static void p_seti(NgxParam *p, const char *n, int v) { ((SetI)p->vt->slot[S_I])(p, n, v); }
static void p_setu(NgxParam *p, const char *n, unsigned v) { ((SetU)p->vt->slot[S_U32])(p, n, v); }
static void p_setf(NgxParam *p, const char *n, float v) { ((SetF)p->vt->slot[S_F])(p, n, v); }
static void p_setp(NgxParam *p, const char *n, void *v) { ((SetP)p->vt->slot[S_P])(p, n, v); }
static int p_geti(const NgxParam *p, const char *n, int *v) { return NGX_OK(((GetI)p->vt->slot[G_I])(p, n, v)); }
static int p_getu(const NgxParam *p, const char *n, unsigned *v) { return NGX_OK(((GetU)p->vt->slot[G_U32])(p, n, v)); }
static int p_getf(const NgxParam *p, const char *n, float *v) { return NGX_OK(((GetF)p->vt->slot[G_F])(p, n, v)); }
static int p_getp(const NgxParam *p, const char *n, void **v) { return NGX_OK(((GetP)p->vt->slot[G_P])(p, n, v)); }

#ifdef _WIN32
#define DLSS_MODEL "nvngx_dlss.dll"
#else
#define DLSS_MODEL "libnvidia-ngx-dlss.so.<version>"
#endif

#ifndef GAME_OOT
// TotK obtains frame-generation inputs from its temporal upscaler.
static int fg_asked(void) {
    const char *v = cfg_value("REKARINA_FRAMEGEN");
    return v && *v && strcmp(v, "off");
}
#endif

int vk_dlaa_wanted(void) {
#ifdef GAME_OOT
    const CfgRenderValues *settings = cfg_render_values();
    return settings ? settings->dlaa : cfg_bool("REKARINA_DLAA");
#else
    static int wanted = -1;
    if (wanted < 0) wanted = cfg_bool("REKARINA_DLAA");
    return wanted;
#endif
}

// The upscaler behind dlaa (REKARINA_TAA): NVIDIA DLSS through
// NGX, or AMD FSR through the FidelityFX loader (vk_ffx.cpp), which takes the same inputs and needs no device
// extensions. The DLSS mode setting (REKARINA_DLSS) is the render scale for both.
#ifdef GAME_OOT
static int backend_available(CfgTaa backend) {
    return backend == CFG_TAA_BILINEAR ? 0 : backend == CFG_TAA_FSR1 ||
           (backend == CFG_TAA_FSR3 ? vk_ffx_upscale_available() : N.available);
}

static CfgTaa effective_taa(void) {
    const CfgRenderValues *settings = cfg_render_values();
    const CfgTaa taa = cfg_taa();
    if (taa == CFG_TAA_DLSS && (settings ? (settings->consumers & CFG_RENDER_FSR_FALLBACK) != 0 :
                               !N.available && vk_ffx_upscale_available())) return CFG_TAA_FSR3;
    return taa;
}

static int fsr3(void) { return effective_taa() == CFG_TAA_FSR3; }
static int fsr1(void) { return cfg_taa() == CFG_TAA_FSR1; }
#ifdef GAME_OOT
static int bilinear(void) { return cfg_bool("REKARINA_DLAA") && cfg_taa() == CFG_TAA_BILINEAR; }
static int spatial_scaler(void) { return bilinear() || vk_dlaa_fsr1(); }
#endif
#else
static int fsr3_on = -1;   // REKARINA_TAA = fsr3, or DLSS cannot run on this card (vk_dlaa_init)
static int fsr3(void) {
    if (fsr3_on < 0) {
        fsr3_on = !strcmp(cfg_value("REKARINA_TAA"), "fsr3");
        if (!fsr3_on && fg_asked() && !strcmp(cfg_value("REKARINA_TAA"), "fsr1")) {
            fsr3_on = 1;
            rt_log("dlaa: FSR 3.1 in place of FSR 1 for frame generation, which needs motion vectors");
        }
    }
    return fsr3_on;
}
// REKARINA_TAA = fsr1: AMD FSR 1, spatial (vk_fsr.c vk_fsr_scene), at the same point of the frame and at the same DLSS mode
// sizes, from the light target alone: no motion vectors, no jitter, no library. For cards where a temporal upscaler
// costs too much (a Radeon RX 6000, a handheld); at the dlaa mode (the scene at the picture's size) it does nothing.
static int fsr1_on = -1;
static int fsr1(void) {
    if (fsr1_on < 0) fsr1_on = !strcmp(cfg_value("REKARINA_TAA"), "fsr1") && !fg_asked();
    return fsr1_on;
}
#endif
int vk_dlaa_fsr1(void) { return vk_dlaa_wanted() && fsr1(); }
int vk_dlaa_fsr3(void) { return vk_dlaa_wanted() && fsr3(); }
int vk_dlaa_prepare_sl_sr(void) {
#ifdef GAME_OOT
    // The Vulkan device must support DLSS even if the first frame uses a different backend.
    return 1;
#else
    return vk_dlaa_wanted() && !fsr3() && !fsr1();
#endif
}

static int core_load(void) {
    if (N.tried) return N.loaded;
    N.tried = 1;
#ifdef _WIN32
    char dir[MAX_PATH] = {0}, path[MAX_PATH + 32];
    DWORD n = sizeof dir;
    if (RegGetValueA(HKEY_LOCAL_MACHINE, "SOFTWARE\\NVIDIA Corporation\\Global\\NGXCore", "FullPath", RRF_RT_REG_SZ,
                     NULL, dir, &n) != ERROR_SUCCESS) {
        rt_log("dlaa: the driver has no NGX core (registry NGXCore missing): DLAA off");
        return 0;
    }
    snprintf(path, sizeof path, "%s\\_nvngx.dll", dir);
    // The core's own dependencies sit beside it in the driver's DriverStore folder. LoadLibraryA with a full path
    // looks for them in the usual places only, not there, and a player's RTX 3060 Laptop (driver 576.88) got error 126
    // (a module not found) where his other games run DLSS. The DLL's own folder joins the search for its dependencies
    // (LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR with the default folders); a Windows that refuses those flags (error 87) gets
    // the altered search path, which also starts at the DLL's folder.
    N.core = LoadLibraryExA(path, NULL, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    DWORD err = N.core ? 0 : GetLastError();
    if (!N.core && err == ERROR_INVALID_PARAMETER) {
        N.core = LoadLibraryExA(path, NULL, LOAD_WITH_ALTERED_SEARCH_PATH);
        err = N.core ? 0 : GetLastError();
    }
    if (!N.core) {
        DWORD at = GetFileAttributesA(path);
        rt_log("dlaa: cannot load %s (error %lu; the file %s): DLAA off", path, err,
               at == INVALID_FILE_ATTRIBUTES ? "is not there" : "is there (126: a module it needs was not found)");
        return 0;
    }
#define FN(f, name) N.f = (void *)GetProcAddress((HMODULE)N.core, name)
#else
    // The driver's NGX core; the dynamic loader finds it where the driver put it (/usr/lib, /usr/lib/x86_64-linux-gnu).
    const char *path = "libnvidia-ngx.so.1";
    N.core = plat_lib_open(path);
    if (!N.core) {
        rt_log("dlaa: cannot load %s (not an NVIDIA driver, or one without NGX): DLAA off", path);
        return 0;
    }
#define FN(f, name) N.f = plat_lib_sym(N.core, name)
#endif
    FN(init, "NVSDK_NGX_VULKAN_Init_Ext2");
    FN(req_ext, "NVSDK_NGX_VULKAN_RequiredExtensions");
    FN(caps, "NVSDK_NGX_VULKAN_GetCapabilityParameters");
    FN(alloc_params, "NVSDK_NGX_VULKAN_AllocateParameters");
    FN(destroy_params, "NVSDK_NGX_VULKAN_DestroyParameters");
    FN(create, "NVSDK_NGX_VULKAN_CreateFeature");
    FN(evaluate, "NVSDK_NGX_VULKAN_EvaluateFeature");
    FN(release, "NVSDK_NGX_VULKAN_ReleaseFeature");
    FN(shutdown, "NVSDK_NGX_VULKAN_Shutdown1");
#undef FN
    if (!N.init || !N.req_ext || !N.caps || !N.alloc_params || !N.destroy_params || !N.create || !N.evaluate ||
        !N.release || !N.shutdown) {
        rt_log("dlaa: the NGX core %s lacks Vulkan entry points: DLAA off", path);
        return 0;
    }
    N.loaded = 1;
    rt_log("dlaa: NGX core %s", path);
    return 1;
}

static int listed(const char *const *list, uint32_t n, const char *s) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(list[i], s)) return 1;
    return 0;
}

void vk_dlaa_instance_exts(const char **ext, uint32_t *n, uint32_t max) {
    unsigned ni = 0, nd = 0;
    const char **ie = NULL, **de = NULL;
    if (vk_sl_sr_requested()) return;
    if (!vk_dlaa_prepare_sl_sr() || !core_load() || !NGX_OK(N.req_ext(&ni, &ie, &nd, &de))) return;
    for (unsigned i = 0; i < ni; i++)
        if (!listed(ext, *n, ie[i]) && *n < max) ext[(*n)++] = ie[i];
}

void vk_dlaa_device_exts(const VkExtensionProperties *have, uint32_t nhave, const char **ext, uint32_t *n,
                         uint32_t max) {
    unsigned ni = 0, nd = 0;
    const char **ie = NULL, **de = NULL;
    if (vk_sl_sr_requested()) return;
    if (!vk_dlaa_prepare_sl_sr() || !core_load() || !NGX_OK(N.req_ext(&ni, &ie, &nd, &de))) return;
    for (unsigned i = 0; i < nd; i++) {
        int ok = 0;
        for (uint32_t k = 0; k < nhave && !ok; k++) ok = !strcmp(have[k].extensionName, de[i]);
        if (!ok) rt_log("dlaa: the device lacks %s, which NGX asks for", de[i]);
        else if (!listed(ext, *n, de[i]) && *n < max) ext[(*n)++] = de[i];
    }
}

static int vtable_ok(void) {
    NgxParam *p = NULL;
    if (!NGX_OK(N.alloc_params(&p)) || !p) return 0;
    int i = 0;
    unsigned u = 0;
    float f = 0;
    void *v = NULL;
    p_seti(p, "TotK.Test.I", -12345);
    p_setu(p, "TotK.Test.U", 4000000000u);
    p_setf(p, "TotK.Test.F", 1.25f);
    p_setp(p, "TotK.Test.P", (void *)p);
    int ok = p_geti(p, "TotK.Test.I", &i) && i == -12345 && p_getu(p, "TotK.Test.U", &u) && u == 4000000000u &&
             p_getf(p, "TotK.Test.F", &f) && f == 1.25f && p_getp(p, "TotK.Test.P", &v) && v == (void *)p;
    if (N.destroy_params) N.destroy_params(p);
    if (!ok) rt_log("dlaa: NGX parameter layout check failed (i=%d u=%u f=%g p=%p): DLAA off", i, u, f, v);
    return ok;
}

#ifndef _WIN32
// UTF-8 to UTF-32 wchar_t, without the C library's locale (the runtime never sets one, so mbstowcs would stop at the
// first non-ASCII byte). Returns 0 on malformed input or when out has no room.
static int utf8_to_wide(const char *in, wchar_t *out, size_t cap) {
    const unsigned char *u = (const unsigned char *)in;
    size_t n = 0;
    while (*u) {
        unsigned c = *u++, more = c < 0x80 ? 0 : c >= 0xF0 ? 3 : c >= 0xE0 ? 2 : c >= 0xC0 ? 1 : 4;
        if (more == 4) return 0;
        if (more) c &= 0x3F >> more;
        for (unsigned k = 0; k < more; k++, u++) {
            if ((*u & 0xC0) != 0x80) return 0;
            c = c << 6 | (*u & 0x3F);
        }
        if (n + 1 >= cap) return 0;
        out[n++] = (wchar_t)c;
    }
    out[n] = 0;
    return 1;
}
#endif

static void dlss_init(void);

// REKARINA_DLAA_PRE_BLOOM=1 (OoT): DLSS resolves before the bloom (pre_bloom_draw). Read once at init: the chunk
// threads read it at every draw.
static int pre_bloom_on;

// taa = dlss on a card where DLSS cannot run (AMD, Intel, a GeForce GTX, an NVIDIA driver without NGX, a missing model
// file): AMD FSR, which runs on any card, takes its place. Until 2026-11-15 the DLSS / DLAA switch then did nothing, and
// the DLSS mode still made the scene smaller (frame_rate.c scene_share): a soft picture. The launcher writes
// taa = fsr3 for such cards, but "Reset all settings" and an ini older than its presets put dlss back.
void vk_dlaa_init(void) {
#ifdef GAME_OOT   // the bloom draw it looks for is OoT's
    pre_bloom_on = rt_knob("DLAA_PRE_BLOOM", "0");
    rt_log("dlaa: pre-bloom resolve %s (REKARINA_DLAA_PRE_BLOOM=%d)", pre_bloom_on ? "on" : "off", pre_bloom_on);
    dlss_init();
    cfg_render_publish_present(vk_dlaa_resolve_consumers);
#else
    if (!vk_dlaa_wanted()) return;
    if (fsr1()) {   // nothing to load: the shaders are the runtime's own
        N.available = 1;
        rt_log("dlaa: the upscaler is AMD FSR 1 (spatial, EASU + RCAS)");
        return;
    }
    if (fsr3()) {
        N.available = vk_ffx_load();
        rt_log("dlaa: the temporal upscaler is AMD FSR%s", N.available ? "" : ", which is missing: the game's own scaling");
        return;
    }
    dlss_init();
    if (!N.available && vk_ffx_load()) {
        fsr3_on = 1, N.available = 1;
        rt_log("dlaa: DLSS cannot run here: AMD FSR takes its place");
    }
#endif
}

static int use_streamline_sr;

static void dlss_init(void) {
    use_streamline_sr = 0;
#ifdef _WIN32
    if (vk_sl_sr_requested()) {
        use_streamline_sr = 1;
        N.available = vk_sl_sr_available();
        rt_log("dlaa: Streamline DLSS %s", N.available ? "available" : "unavailable");
        return;
    }
#endif
    if (!core_load()) return;
#ifdef _WIN32
    wchar_t wdir[MAX_PATH];
    DWORD len = GetModuleFileNameW(NULL, wdir, MAX_PATH);
    while (len && wdir[len - 1] != L'\\') len--;
    wdir[len] = 0;
#else
    // The executable's folder, where NGX looks for libnvidia-ngx-dlss.so and writes its logs, as a wchar_t string:
    // 32-bit code points here, as the Linux core expects.
    char dir[PLAT_PATH_MAX];
    wchar_t wdir[PLAT_PATH_MAX];
    char *cut = plat_exe_path(dir, sizeof dir) ? plat_last_sep(dir) : NULL;
    if (!cut || !utf8_to_wide((cut[1] = 0, dir), wdir, PLAT_PATH_MAX)) {
        rt_log("dlaa: cannot name the executable's folder for NGX: DLAA off");
        return;
    }
#endif
    // 231313132: the application id NVIDIA's DLSS samples use for builds without an NVIDIA-issued id
    const VkCtx *c = vk_ctx();
    NgxResult r = N.init(231313132ull, wdir, c->instance, c->phys, c->device, vkGetInstanceProcAddr,
                         vkGetDeviceProcAddr, NGX_VERSION_API, NULL);
    if (!NGX_OK(r)) {
        rt_log("dlaa: NGX init failed (%08x): DLAA off", (unsigned)r);
        return;
    }
    N.initialized = 1;
    if (!vtable_ok()) return;
    NgxParam *caps = NULL;
    int avail = 0, driver = 0;
    if (NGX_OK(N.caps(&caps)) && caps) {
        p_geti(caps, "SuperSampling.Available", &avail);
        p_geti(caps, "SuperSampling.NeedsUpdatedDriver", &driver);
        r = N.destroy_params(caps);
        if (!NGX_OK(r)) rt_log("dlaa: NGX capability parameter release failed (%08x)", (unsigned)r);
    }
    N.available = avail != 0;
    rt_log("dlaa: NGX up; DLSS %s%s", avail ? "available" : "NOT available (is " DLSS_MODEL " next to the executable?)",
           driver ? "; the driver needs an update" : "");
}

int vk_dlaa_backend_available(void) {
#ifdef GAME_OOT
    return backend_available(effective_taa());
#else
    return N.available;
#endif
}

int vk_dlaa_available(void) {
#ifdef GAME_OOT
    const CfgRenderValues *settings = cfg_render_values();
    if (settings) return (settings->consumers & CFG_RENDER_UPSCALER) != 0;
#endif
    return vk_dlaa_wanted() && vk_dlaa_backend_available();
}

// ---------------------------------------------------------------- where DLAA runs in the game's frame
// Measured (REKARINA_DUMP, frame 2100 at the first town, 2560x1440 scene): the G-buffer passes draw into five targets with
// the scene depth (d32f at the scene size), one of them the R11G11B10F light target; lighting and the transparents
// then draw into that light target alone, the last time with the scene depth attached; the first pass narrower than a
// quarter of the scene starts the bloom chain, which reads it. After the bloom come the tonemap and post passes, then
// the HUD is drawn into the same target and a last pass copies it to the 8-bit output.
// So: after the G-buffer, a pass at the scene size with exactly one colour target (R11G11B10F or RGBA16F) and the D32F
// depth arms the frame with that pair; the next pass at most a quarter as wide ends the scene (no jitter after it) and,
// when the scene is as wide as the output, runs DLAA on the armed target first. Bloom, tonemap, post and HUD then all
// work on the anti-aliased picture.
// Upscaling (a scene smaller than the output: REKARINA_DLSS quality/balanced/performance, or the game's own 5/6): measured
// at a 960x536 scene, 1920x1080 output, the bloom and its composite write back into the light target at the scene size,
// then one full-screen draw (the FXAA shader below, which also scales) reads it into an output-sized R11G11B10F target
// with no depth, and the HUD is drawn over that target at the output size. DLSS runs at the begin of that pass: it
// reads the light target and its result is blitted into the pass's target, whose scaling draw is then skipped. The
// HUD, drawn afterwards into the same target, is never jittered nor passed through DLSS.
// The scaling draw is not always that FXAA shader: at a 1280x720 scene and 1920x1080 output (1.4.3, dlss = quality)
// the game's tonemap (guest fs 7e114c4330dc212a) reads the light target and writes the output-sized target itself,
// over DLSS's picture, so the jittered scene reached the screen and the whole picture shook. So from DLSS's evaluation
// to the frame's end, draws in passes larger than the light target sample DLSS's output in its place
// (vk_dlaa_swap_img, applied by vk_exec_draw.c tex_info): whichever shader scales, it scales DLSS's picture.
enum { PRIVATE_BLOOM_TARGET_COUNT = 16 };
struct PrivateBloomTarget {
    VkImage image;
    VkImageView view;
    VmaAllocation allocation;
    uint32_t width, height;
};

enum ScenePhase { SCENE_DRAWING, SCENE_CLOSED, SCENE_RESOLVED, SCENE_SCALED, SCENE_PREBLOOM };
enum SceneEvaluation { EVALUATE_AT_PASS, EVALUATE_AT_DISPATCH, EVALUATE_PREBLOOM, EVALUATE_NATIVE_PREBLOOM };

static struct {
    struct {
        VkExecTarget color, depth;
        enum ScenePhase phase;
        int armed, gbuf, early_attempted;
        struct BloomGuestSubresource {
            uint32_t image_serial, level, layer;
        } bloom_guest_subresources[PRIVATE_BLOOM_TARGET_COUNT];
        VkImage tonemapped, packed;
        SceneReplay replay;
        SceneReplayKey capture_key;
        VkImage capture_source, capture_copy;
        uint64_t capture_copy_address;
        // A checked scaler output can be the final picture if nothing writes it again before the encode.
        VkImage clean_output;
        uint64_t clean_output_address;
        int capture_hdr;
        uint32_t tonemap_serial, packed_serial;
        const VkTexImage *stock_pack, *stock_output;
#ifdef GAME_OOT
        const VkTexImage *rcas_output;
        const op_hdr *skip_aa[3];
        const VkTexImage *skip_aa_output;
#endif
        struct {
            const VkTexImage *texture;
            VkImage image;
            uint64_t address;
            uint32_t width, height;
            int pending;
        } hudless;
        // Present validates the submitted frame and HUD-less generation after replay resets.
        struct {
            SceneReplayKey key;
            uint64_t hudless_generation;
            VkImage target, copy;
            uint64_t target_address, copy_address;
            const char *rejection;
            int hdr;
        } coverage;
    } scene;
    struct Size {                // a DLSS feature and its work images per scene size (dynamic resolution flips it)
        uint32_t w, h, ow, oh;   // render size, output size
        struct PrivateBloomTarget bloom_targets[PRIVATE_BLOOM_TARGET_COUNT];
        VkImage out, mv, tonemap;
        VkImageView tonemap_view;
        VmaAllocation tonemap_mem;
        VkFormat tonemap_format;
        VmaAllocation out_mem, mv_mem;
        VkImageView out_view, mv_view;
        VkImage dist[2];         // each pixel's view distance, this and last frame (R32F, dlaa_mv.comp depth_out)
        VmaAllocation dist_mem[2];
        VkImageView dist_view[2];
        uint64_t pose_distance_frame;
        float pose_history[4];
        int pose_distance_valid;
        int dist_cur, dist_ok;   // next write slot and validity of the previous slot
        NgxHandle *feat;
        uint64_t used;           // evals count when it last ran
    } sizes[5], *last;           // last: the one the previous DLAA frame ran with
    VkImage last_color;          // the target DLAA ran on last frame (light_target_check)
    NgxParam *params;
    int failed, reset;
    int early_full;              // the pause screens' full-size feature was made
    uint64_t evals, frames_missed;
    const char *nvidia_scene_aa;
    uint32_t jw, jh, ev_w, ev_h, ev_ow, ev_oh;
    int stepped;
    // jitter (vk_dlaa_viewport)
    VkImage jdepth;              // the scene depth of the last frame DLAA ran on: passes drawing with it are jittered
    float jx, jy;                // the offset the next scene draws take, render pixels (window x right, y down)
    float ux, uy;                // the offset this frame's scene draws took (DLSS and the motion vectors get these)
    uint32_t jframe;
    int jit_ok, jit_used, hold;  // jitter allowed this frame / a draw used it / frames left without it
    // projection jitter (vk_dlaa_pjitter): the scene size DLSS last ran at, the blocks jittered this frame
    uint32_t pj_w, pj_h, pj_n;
    struct { uint64_t addr; float rows[8], dx, dy; } pj[8];
    uint64_t holds;
    // camera (vk_dlaa_draw) and motion vectors (mv_pass)
    uint64_t cam_addr;           // this frame's scene camera block (vk_dlaa_draw), 0: none seen
    p_scene_projection projection;
    SceneProjection projection_history;
    uint64_t projection_history_frame;
    int projection_conflict, projection_resolved;
    uint32_t cam_size, cam_vp;   // its size and the offset of its view-projection rows
    struct { uint64_t addr; uint32_t n; } views[8];   // OoT: this frame's scene contexts of light-target draws
    int nviews;
    VkBuffer camb;               // two CAM_SLOT slots: this frame's block and last frame's (deferred engine)
    VmaAllocation camb_mem;
    VkBuffer mvm;                // MVM_SLOTS reprojection matrices for dlaa_mv.comp
    VmaAllocation mvm_mem;
    uint8_t *mvm_map;
    uint32_t mvm_slot;
    double cam_prev[16];         // last frame's view-projection (rows 3-6 of its block), for the reprojection
    int cam_slot, cam_prev_ok;   // the slot this frame writes / the other slot holds last frame's camera
    VkDescriptorSetLayout mv_dsl;
    VkPipelineLayout mv_layout;
    VkPipeline mv_pipe;
    VkSampler mv_samp;
    int mv_ok;                   // -1 not built, 0 failed, 1 ready
    uint64_t mv_frames, mv_none;
    uint64_t why[3];             // history resets: scene size changed, no camera, scene target changed
    uint64_t target_moves_kept;  // scene target changed and the history carried on (light_target_check)
    uint64_t out_addr;           // the storage of the game's output picture, where DLSS ran at the scaling dispatch
    uint64_t at_dispatch, at_pass;   // evaluations by where they ran (vk_dlaa_dispatch / vk_dlaa_pass)
    struct {                     // OoT's resolve before the bloom (pre_bloom_draw)
        int last;                // the last evaluation ran there: its history is of scene light
        int tried;               // this frame's bloom draw was inspected
        int resolved;            // DLSS resolved before the bloom this frame
        VkImage stage;           // tone-mapped color staged at SMAA compose (pre_bloom_stage)
#ifdef GAME_OOT
        VkImageView stage_view;
#endif
        VmaAllocation stage_mem;
        VkFormat stage_fmt;
        uint32_t stage_w;
        uint32_t stage_h;
        int staged;              // this frame's tone-mapped color is valid in stage
        const char *stage_why;   // why pre_bloom_stage skipped copying this frame (for fallback logs)
        uint32_t packed_serial;  // serial of the RGB9E5 image SMAA compose wrote, matched at the scaler
        uint64_t evals;          // evaluations that ran before the bloom
    } pre_bloom;
    int on;                      // the dlaa setting, read at each present (the menu switches it live)
    int preset, preset_dirty;    // the DLSS model features are created with (preset_value); a change pending
    unsigned cfg_seen;           // cfg_version when preset was read
    uint64_t last_eval_ns;       // plat_now_ns of the previous evaluation (FrameTimeDeltaInMsec)
    VkFfxImage fg_depth, fg_mv;  // the last evaluation's depth and motion vectors, for frame generation
    VkImageView fg_depth_view, fg_mv_view;
    VkDlaaCamera fg_cam;         // this frame's camera (mv_pass), fg_cam_ok when known
    int fg_cam_ok;
    float fg_viewport_jitter[2]; // legacy paths without a committed CPU projection
    int fg_fresh, fg_reset;      // fresh: not handed out yet (vk_dlaa_fg_inputs); reset: this evaluation reset its history
    uint64_t presents;           // vk_dlaa_frame calls so far
    // the picture without the HUD for frame generation (vk_dlaa_fg_hudless): the game's finished picture before its HUD
    // (hudless_at_pass) encoded to sRGB as the game's last pass encodes its picture, then copied bit for bit into an RGBA8
    // UNORM image, the format the present reads; with HDR on, as RGBA16F, as the game's float picture the present encodes
    // (hl_srgb unused)
    VkImage hl_srgb, hl;
    VmaAllocation hl_srgb_mem, hl_mem;
    VkImageView hl_view;
    uint32_t hl_w, hl_h;
    VkFormat hl_fmt;
    int hl_pq, hl_fresh;
    uint64_t hl_taken, hl_window, hl_early, hl_missed;   // copies at the HUD's pass, at the window copy, at DLSS, none
    // presents (the present thread) that found a later frame's HUD-less copy or depth and motion already recorded by the
    // replay: hl_fresh and fg_fresh count the frames recorded since the present before
    uint64_t hl_ahead, fg_ahead;
    // the game's pause screens (vk_dlaa_fg_inputs): oot_pass is set while OoT's frame runs the upscaler at an output pass;
    // fg_menu: the evaluation handed to frame generation ran there; fg_menu_on, fg_menu_at, fg_menu_n: the stretch now
    int oot_pass, fg_menu, fg_menu_on;
    uint64_t fg_menu_at, fg_menu_n, fg_menu_frames;
} D = {.mv_ok = -1, .cfg_seen = ~0u};

static struct {
    SceneReplay replay;
    int hudless;
    SceneUiAlpha coverage;
} sealed_scene;

static uint64_t replay_submission_frame(void) {
    return __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
}

static int replay_scene_live(void) {
    return !D.projection_conflict && scene_replay_matches(D.scene.replay.key,
        (SceneReplayKey){replay_submission_frame(), D.projection.sample.token});
}

static void replay_scene_start(void) {
    if (replay_scene_live()) return;
    D.scene.replay = (SceneReplay){.key = {replay_submission_frame(), D.projection.sample.token},
        .rw = D.scene.color.width, .rh = D.scene.color.height,
        .uw = D.scene.color.width, .uh = D.scene.color.height,
        .light = D.scene.color.img ? D.scene.color.img->serial : 0};
}

static void replay_scene_complete(const VkExec *ex, const VkExecTarget *target) {
    if (!replay_scene_live() || !target->img) return;
    D.scene.replay.completed = 1;
    D.scene.replay.output = target->img->serial;
    if (ex->interface_coverage) {
        D.scene.clean_output = target->image;
        D.scene.clean_output_address = target->img->addr;
    }
}

static int scene_closed(void) { return D.scene.phase != SCENE_DRAWING; }
static int scene_resolved(void) { return D.scene.phase >= SCENE_RESOLVED; }
static int scene_scaled(void) { return D.scene.phase == SCENE_SCALED; }
static int scene_prebloom(void) { return D.scene.phase == SCENE_PREBLOOM; }

void vk_dlaa_seal_scene(uint64_t frame, VkImage guest, VkImage source, uint32_t serial) {
    sealed_scene.replay = D.scene.replay.key.frame == frame && !D.projection_conflict ?
        D.scene.replay : (SceneReplay){.key.frame = frame};
    int direct = scene_replay_direct_present(&sealed_scene.replay, serial);
    if (direct) {
        sealed_scene.replay.completed = 1;
        sealed_scene.replay.output = sealed_scene.replay.window = serial;
    }
    // Match the checked output directly or carry the capture through the verified guest encode.
    int source_matches = source && (source == D.scene.capture_source ||
        (source == guest && source == D.scene.capture_copy));
    sealed_scene.hudless = !direct && source_matches && sealed_scene.replay.completed && D.scene.capture_hdr == vk_hdr_active() &&
        scene_replay_matches(sealed_scene.replay.key, D.scene.capture_key);
    if (source_matches) sealed_scene.replay.window = serial;
    int coverage_matches = scene_replay_matches(sealed_scene.replay.key, D.scene.coverage.key) &&
        (source == D.scene.coverage.target || source == D.scene.coverage.copy);
    int empty = coverage_matches && D.scene.clean_output && D.scene.clean_output == D.scene.capture_source;
    sealed_scene.coverage = scene_replay_ui_alpha(sealed_scene.hudless, D.scene.coverage.rejection != NULL,
        empty, coverage_matches && !empty);
    if (vk_interface_coverage_enabled()) {
        const char *reason = direct ? "scene already composited" :
            !sealed_scene.replay.completed ? "no completed scene" :
            D.scene.coverage.rejection ? D.scene.coverage.rejection :
            !sealed_scene.hudless ? "no matching HUD-less checkpoint" :
            !sealed_scene.coverage ? "no coverage for present source" : NULL;
        static const char *previous;
        static uint64_t last_log;
        if (reason != previous && (vk_dlaa_pathlog() || !last_log || frame - last_log >= 120)) {
            rt_log("interface coverage: frame %llu %s%s", (unsigned long long)frame,
                   reason ? "withheld: " : "ready to fit", reason ? reason : "");
            previous = reason;
            last_log = frame;
        }
    }
    renderer_mcp_scene(&sealed_scene.replay, sealed_scene.hudless, sealed_scene.coverage != SCENE_UI_ALPHA_NONE);
    if (vk_dlaa_pathlog())
        rt_log("PATHLOG scene seal f %llu key %llu/%llu route %d complete %d source match %d hudless %d coverage %d | tonemap %u packed %u output %u present %u",
               (unsigned long long)frame, (unsigned long long)sealed_scene.replay.key.frame,
               (unsigned long long)sealed_scene.replay.key.view, (int)sealed_scene.replay.route,
               sealed_scene.replay.completed, source_matches, sealed_scene.hudless, sealed_scene.coverage != SCENE_UI_ALPHA_NONE,
               sealed_scene.replay.tonemap, sealed_scene.replay.packed, sealed_scene.replay.output, serial);
}

int vk_dlaa_fg_hudless(uint64_t frame, VkImage *img, VkImageView *view, uint32_t *w, uint32_t *h, VkFormat *fmt, int *pq) {
    int n = __atomic_exchange_n(&D.hl_fresh, 0, __ATOMIC_ACQ_REL);
    if (!n || (D.failed && !vk_dlaa_fg_only())) return 0;
    if (n > 1) D.hl_ahead++;
#ifdef GAME_OOT
    if ((vk_dlaa_prebloom_enabled() || vk_interface_coverage_enabled()) &&
        (sealed_scene.replay.key.frame != frame || !sealed_scene.hudless)) return 0;
#endif
    *img = D.hl, *view = D.hl_view, *w = D.hl_w, *h = D.hl_h, *fmt = D.hl_fmt, *pq = D.hl_pq;
    return 1;
}

int vk_dlaa_fg_views(VkImageView *depth, VkImageView *mv) {
    *depth = D.fg_depth_view, *mv = D.fg_mv_view;
    return D.fg_depth_view && D.fg_mv_view;
}

int vk_dlaa_pose_depth(uint64_t frame, uint32_t w, uint32_t h, VkDescriptorImageInfo *image, float history[4]) {
    const struct Size *last = D.last;
    if (!last || !last->pose_distance_valid || !last->dist_ok || last->w != w || last->h != h ||
        !frame || last->pose_distance_frame != frame - 1 || fabsf(last->pose_history[2]) < 1e-6f) return 0;
    *image = (VkDescriptorImageInfo){D.mv_samp, last->dist_view[last->dist_cur ^ 1], VK_IMAGE_LAYOUT_GENERAL};
    memcpy(history, last->pose_history, sizeof last->pose_history);
    return 1;
}

int vk_dlaa_fg_camera(VkDlaaCamera *c) {
    if (!D.fg_cam_ok) return 0;
    *c = D.fg_cam;
    return 1;
}

// OoT's play frames run the upscaler at the game's scaling dispatch (vk_dlaa_dispatch); its pause screens draw their 3D
// (the hero on the Equipment tab, the items turning) into a scene the game ends at an output pass (vk_dlaa_pass_),
// with a camera of their own. Frame generation given those frames blended the tabs into each other and ghosted the
// turning items (owner, 2026-11-20, DLSS-G), so they give it no inputs: both frame generations go off there (DLSS-G
// keeps its resources, vk_dlssg.c) and the first play frame after them starts generation's history anew.
// REKARINA_FG_MENU_OFF=0: such frames are handed on as before (test).
static int fg_menu_frame(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("FG_MENU_OFF", "1");
    int menu = on && D.fg_menu;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (menu != D.fg_menu_on) {
        if (menu) rt_log("dlaa: frame generation off from frame %llu: the scene is drawn outside the game's scaling dispatch "
                         "(a pause screen)", (unsigned long long)f);
        else rt_log("dlaa: frame generation back at frame %llu after %llu frames of a pause screen (stretch %llu)",
                    (unsigned long long)f, (unsigned long long)(f - D.fg_menu_at), (unsigned long long)D.fg_menu_n);
        if (menu) D.fg_menu_at = f, D.fg_menu_n++;
        else D.fg_reset = 1;   // the frame before is the last one before the pause screen
        D.fg_menu_on = menu;
    }
    if (menu) D.fg_menu_frames++;
    return menu;
}

int vk_dlaa_fg_inputs(VkFfxImage *depth, VkFfxImage *mv, float *jx, float *jy, int *reset) {
    int n = __atomic_exchange_n(&D.fg_fresh, 0, __ATOMIC_ACQ_REL);
    if (!n) return 0;
    if (n > 1) D.fg_ahead++;
    if (fg_menu_frame()) return 0;
    // the upscaler's history resets (another scene size, no camera, another scene target) are frame generation's too:
    // its last frame's depth and motion are of another picture
    const float *jitter = D.projection_history.token ? D.projection_history.jitter : D.fg_viewport_jitter;
    *depth = D.fg_depth, *mv = D.fg_mv, *jx = jitter[0], *jy = jitter[1], *reset = D.fg_reset;
    D.fg_reset = 0;
    return 1;
}

const VkTexImage *vk_dlaa_swap_img;
VkImageView vk_dlaa_swap_view;

static int dl_active(void) { return !D.failed && D.on; }
int vk_dlaa_active(void) { return dl_active(); }
const char *vk_dlaa_nvidia_scene_aa(void) { return D.nvidia_scene_aa; }
static int fg_motion(void) {
#ifdef GAME_OOT
    const CfgRenderValues *settings = cfg_render_values();
    return settings && (settings->consumers & CFG_RENDER_FRAMEGEN_MOTION);
#else
    return 0;
#endif
}

#ifndef GAME_OOT
int vk_dlaa_hudless_wanted;
#endif

static int hudless_wanted(void) {
#ifdef GAME_OOT
    return fg_motion();
#else
    return vk_dlaa_hudless_wanted;
#endif
}

unsigned vk_dlaa_resolve_consumers(CfgTaa taa, int wanted) {
#ifdef GAME_OOT
    const char *fg = cfg_value("REKARINA_FRAMEGEN");
    const int dlss = !strcmp(fg, "dlss") && vk_dlssg_swapchain_ready();
    const int generate = dlss || (!strcmp(fg, "fsr3") && vk_fg_active());
    const int fallback = wanted && taa == CFG_TAA_DLSS && !N.available;
    if (wanted && (taa == CFG_TAA_FSR3 || fallback)) vk_ffx_load();
    const int use_fallback = fallback && vk_ffx_upscale_available();
    return (generate ? CFG_RENDER_FRAMEGEN_MOTION : 0) |
           (dlss && vk_dlssg_interface_coverage_available() && cfg_bool("REKARINA_DLSS_INTERFACE_RECOMPOSITION") ?
               CFG_RENDER_INTERFACE_COVERAGE : 0) |
           (use_fallback ? CFG_RENDER_FSR_FALLBACK : 0) |
           (wanted && (use_fallback || backend_available(taa)) ? CFG_RENDER_UPSCALER : 0);
#else
    (void)taa;
    return wanted && N.available ? CFG_RENDER_UPSCALER : 0;
#endif
}

int vk_dlaa_fg_only(void) { return fg_motion() && (!dl_active() || fsr1()); }

static void dl_fail(enum SceneEvaluation evaluation, const char *what, NgxResult r) {
    if (evaluation == EVALUATE_PREBLOOM || evaluation == EVALUATE_NATIVE_PREBLOOM) {
        static unsigned failures;
        if (failures++ < 8) rt_log("dlaa: pre-bloom %s failed (%08x); retaining late fallback", what, (unsigned)r);
        D.reset = 1;
        return;
    }
    rt_log("dlaa: %s failed (%08x): DLAA off for this run", what, (unsigned)r);
    D.failed = 1;
    scene_projection_feedback(NULL, 0, 0, 0);
}

static int image2d(uint32_t w, uint32_t h, VkFormat f, VkImageUsageFlags usage, VkImage *img, VmaAllocation *mem,
                   VkImageView *view) {
    const VkCtx *c = vk_ctx();
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ai, img, mem, NULL) != VK_SUCCESS) return 0;
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = *img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = f,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    return vkCreateImageView(c->device, &vi, NULL, view) == VK_SUCCESS;
}

static void destroy_size(struct Size *e) {
    const VkCtx *c = vk_ctx();
    vk_queue_lock(), vkDeviceWaitIdle(c->device), vk_queue_unlock();   // a third scene size: the evicted images may still be in flight
    for (unsigned index = 0; index < PRIVATE_BLOOM_TARGET_COUNT; ++index) {
        const struct PrivateBloomTarget *target = &e->bloom_targets[index];
        if (target->view) vkDestroyImageView(c->device, target->view, NULL);
        if (target->image) vmaDestroyImage(c->vma, target->image, target->allocation);
    }
    if (e->tonemap_view) vkDestroyImageView(c->device, e->tonemap_view, NULL);
    if (e->tonemap) vmaDestroyImage(c->vma, e->tonemap, e->tonemap_mem);
    if (e->out_view) vkDestroyImageView(c->device, e->out_view, NULL);
    if (e->mv_view) vkDestroyImageView(c->device, e->mv_view, NULL);
    if (e->out) vmaDestroyImage(c->vma, e->out, e->out_mem);
    if (e->mv) vmaDestroyImage(c->vma, e->mv, e->mv_mem);
    for (int i = 0; i < 2; i++) {
        if (e->dist_view[i]) vkDestroyImageView(c->device, e->dist_view[i], NULL);
        if (e->dist[i]) vmaDestroyImage(c->vma, e->dist[i], e->dist_mem[i]);
    }
    if (e->feat && N.release) N.release(e->feat);
    memset(e, 0, sizeof *e);
}

void vk_dlaa_release_ngx(VkDevice device) {
    if (!N.initialized) return;
    for (size_t i = 0; i < sizeof D.sizes / sizeof *D.sizes; ++i) {
        if (!D.sizes[i].feat) continue;
        NgxResult r = N.release(D.sizes[i].feat);
        if (!NGX_OK(r)) rt_log("dlaa: NGX feature release failed (%08x)", (unsigned)r);
        D.sizes[i].feat = NULL;
    }
    if (D.params) {
        NgxResult r = N.destroy_params(D.params);
        if (!NGX_OK(r)) rt_log("dlaa: NGX parameter release failed (%08x)", (unsigned)r);
        D.params = NULL;
    }
    unsigned remaining = 0;
    NgxResult r = N.shutdown(device, &remaining);
    if (!NGX_OK(r)) rt_log("dlaa: NGX shutdown failed (%08x)", (unsigned)r);
    N.initialized = N.available = 0;
}

static void free_size(struct Size *e) {
    vk_queue_lock(), vkDeviceWaitIdle(vk_ctx()->device), vk_queue_unlock();   // evicted images may still be in flight
    destroy_size(e);
}

// The entry for a scene and output size: its output (RGBA16F at the output size, written by DLSS, then blitted into
// the target) and motion vectors (RG16F at the scene size, render pixels), both kept in GENERAL like every image of the
// back end. the game's dynamic resolution switches between a few sizes; each keeps its feature, the least recently used
// goes for a new one.
static struct Size *size_for(VkCommandBuffer cb, uint32_t w, uint32_t h, uint32_t ow, uint32_t oh) {
    struct Size *e = NULL, *lru = &D.sizes[0];
    for (int i = 0; i < 5; i++) {
        if (D.sizes[i].w == w && D.sizes[i].h == h && D.sizes[i].ow == ow && D.sizes[i].oh == oh) return &D.sizes[i];
        if (!D.sizes[i].out && !e) e = &D.sizes[i];
        if (D.sizes[i].used < lru->used) lru = &D.sizes[i];
    }
    if (!e) {
        if (D.last == lru) D.last = NULL;
        free_size(e = lru);
    }
    VkImageUsageFlags u = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                          VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    VkImageUsageFlags u8 = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!image2d(ow, oh, VK_FORMAT_R16G16B16A16_SFLOAT, u, &e->out, &e->out_mem, &e->out_view) ||
        !image2d(w, h, VK_FORMAT_R16G16_SFLOAT, u | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT, &e->mv, &e->mv_mem, &e->mv_view) ||
        !image2d(w, h, VK_FORMAT_R32_SFLOAT, u8, &e->dist[0], &e->dist_mem[0], &e->dist_view[0]) ||
        !image2d(w, h, VK_FORMAT_R32_SFLOAT, u8, &e->dist[1], &e->dist_mem[1], &e->dist_view[1])) {
        destroy_size(e);
        return NULL;
    }
    VkImageMemoryBarrier2 ib[4];
    VkImage im[4] = {e->out, e->mv, e->dist[0], e->dist[1]};
    for (int i = 0; i < 4; i++)
        ib[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = im[i],
                                        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 4,
                           .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(cb, &di);
    e->w = w, e->h = h, e->ow = ow, e->oh = oh;
    return e;
}

static void full_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static NgxResourceVK res_image(VkImageView v, VkImage img, VkFormat f, uint32_t w, uint32_t h, VkImageAspectFlags a,
                               int rw) {
    NgxResourceVK r;
    memset(&r, 0, sizeof r);
    r.r.img = (NgxImageView){v, img, {a, 0, 1, 0, 1}, f, w, h};
    r.type = 0;
    r.rw = rw;
    return r;
}

// ---------------------------------------------------------------- timing (REKARINA_DLAA_TIMES=1)
// GPU timestamps around camera motion, object motion, and temporal reconstruction; CPU time spent draining the
// recorder, calling the upscaler, and recording motion. Reports accumulate over TIMES_SUMMARY frames.
// Each frame takes its own slot of TIMES_SLOTS; a slot's results are read when it comes round again (by then the GPU is
// long done with it; not ready: that frame is left out).
enum { T_MV0, T_MV1, T_OBJ, T_DL0, T_DL1, T_N };
#define TIMES_SLOTS 8u
#define TIMES_SUMMARY 240
static struct {
    int on;                      // -1 not read
    VkQueryPool pool;
    uint32_t slot;
    int used[TIMES_SLOTS];       // the slot holds a frame's timestamps (all T_N written)
    double gpu[3];               // ms summed: camera motion, object motion, upscaler
    uint64_t frames, cpu_drain, cpu_upscale, cpu_mv;   // ns summed
} T = {.on = -1};

static int times_on(void) {
    if (T.on < 0) {
        T.on = rt_knob("DLAA_TIMES", "0");
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                                    .queryType = VK_QUERY_TYPE_TIMESTAMP, .queryCount = TIMES_SLOTS * T_N};
        if (T.on && vkCreateQueryPool(vk_ctx()->device, &qi, NULL, &T.pool) != VK_SUCCESS) T.on = 0;
    }
    return T.on;
}

static void times_mark(VkCommandBuffer cb, int which) {
    if (times_on()) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, T.pool, T.slot * T_N + (uint32_t)which);
}

// At the start of a frame's timing (mv_pass): the slot's last results are read and it is reset.
static void times_begin(VkCommandBuffer cb) {
    if (!times_on()) return;
    T.slot = (T.slot + 1) % TIMES_SLOTS;
    if (T.used[T.slot]) {
        uint64_t q[T_N];
        if (vkGetQueryPoolResults(vk_ctx()->device, T.pool, T.slot * T_N, T_N, sizeof q, q, sizeof q[0],
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            double ns = vk_ctx()->props.limits.timestampPeriod / 1e6;   // ticks to ms
            T.gpu[0] += (double)(q[T_MV1] - q[T_MV0]) * ns, T.gpu[1] += (double)(q[T_OBJ] - q[T_MV1]) * ns;
            T.gpu[2] += (double)(q[T_DL1] - q[T_DL0]) * ns;
            if (++T.frames % TIMES_SUMMARY == 0) {
                rt_log("dlaa times: gpu camera motion %.3f ms, object motion %.3f ms, upscaler %.3f ms (%llu frames timed); "
                       "cpu: recorder drain %.3f ms, upscaler call %.3f ms, motion pass recording %.3f ms",
                       T.gpu[0] / TIMES_SUMMARY, T.gpu[1] / TIMES_SUMMARY, T.gpu[2] / TIMES_SUMMARY,
                       (unsigned long long)T.frames, T.cpu_drain / 1e6 / TIMES_SUMMARY, T.cpu_upscale / 1e6 / TIMES_SUMMARY,
                       T.cpu_mv / 1e6 / TIMES_SUMMARY);
                T.gpu[0] = T.gpu[1] = T.gpu[2] = 0, T.cpu_drain = T.cpu_upscale = T.cpu_mv = 0;
            }
        }
    }
    T.used[T.slot] = 0;
    vkCmdResetQueryPool(cb, T.pool, T.slot * T_N, T_N);
}

// ---------------------------------------------------------------- motion vectors
// Reproject depth through the two cameras, then let model replay overwrite matching pixels.
static const float *pj_clean(uint64_t addr, const float *cr, float *out);   // below, with the projection jitter
#define CAM_BYTES 256u   // vec4 rows 0-15: view, view-projection (3-6), projection, inverse view
#define OCAM_BLOCK 0x3500u   // OoT's scene context block (vk_dlaa_draw)
#define OCAM_BYTES 0xb0u     // its rows 0-10: the view and the projection
#define CAM_BLOCK 0x940u   // the whole block, kept for the per-object motion pass's vertex shaders (vk_exec_draw.c)
#define CAM_SLOT 0xa00u    // camb slot stride
#define MVM_SLOTS 16u      // frames the GPU may still be reading the reprojection blocks of

static MotionVectorsMode frame_motion_mode(void) {
    const CfgRenderValues *settings = cfg_render_values();
    return settings ? settings->motion_mode : MOTION_VECTORS_OFF;
}

static const char *const motion_names[] = {"off", "camera", "models", "hybrid"};   // MotionVectorsMode order

static void motion_history_check(void) {
    static MotionVectorsMode was = MOTION_VECTORS_HYBRID;
    const MotionVectorsMode mode = frame_motion_mode();
    if (mode != was) {
        rt_log("dlaa: motion vectors %s (were %s): a new history", motion_names[mode], motion_names[was]);
        was = mode;
        D.reset = 1;
    }
}

// dlaa_mv.comp's push constants in its layout (std430: dz at offset 16)
typedef struct {
    float size[2], jitter[2];
    float dz[4];   // the projection's [2][2], [2][3], [3][2], [3][3]
    uint32_t write_distance;
    uint32_t camera_mv;
} MvPush;

static int mv_init(void) {
    if (D.mv_ok >= 0) return D.mv_ok;
    D.mv_ok = 0;
    const VkCtx *c = vk_ctx();
    VkDescriptorSetLayoutBinding b[4] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                           .bindingCount = 4, .pBindings = b};
    VkPushConstantRange pr = {VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MvPush)};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
                              .minFilter = VK_FILTER_NEAREST, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = sizeof dlaa_mv_spv,
                                   .pCode = dlaa_mv_spv};
    VkShaderModule m = VK_NULL_HANDLE;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 2 * CAM_SLOT,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vkCreateDescriptorSetLayout(c->device, &dci, NULL, &D.mv_dsl) != VK_SUCCESS) goto fail;
    VkBufferCreateInfo mbi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = MVM_SLOTS * CAM_BYTES,
                              .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT};
    VmaAllocationCreateInfo mai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                   .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                            VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo minfo;
    if (vmaCreateBuffer(c->vma, &mbi, &mai, &D.mvm, &D.mvm_mem, &minfo) != VK_SUCCESS) goto fail;
    D.mvm_map = minfo.pMappedData;
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &D.mv_dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    if (vkCreatePipelineLayout(c->device, &li, NULL, &D.mv_layout) != VK_SUCCESS ||
        vkCreateSampler(c->device, &si, NULL, &D.mv_samp) != VK_SUCCESS ||
        vkCreateShaderModule(c->device, &mi, NULL, &m) != VK_SUCCESS ||
        vmaCreateBuffer(c->vma, &bi, &ai, &D.camb, &D.camb_mem, NULL) != VK_SUCCESS)
        goto fail;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = D.mv_layout};
    VkResult r = vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &D.mv_pipe);
    vkDestroyShaderModule(c->device, m, NULL);
    if (r != VK_SUCCESS) goto fail;
    return D.mv_ok = 1;
fail:
    rt_log("dlaa: the motion-vector pass could not be set up: DLAA runs with zero motion");
    return 0;
}

// The reprojection from this frame's clip space to last frame's, last * inverse(this), in double precision: the shader
// took each pixel back to world space through a float inverse of the view-projection, and with the game's world
// coordinates in the thousands and its far plane the result was noise of a fraction of a pixel and more on distant
// surfaces, which DLSS followed into a shimmer like heat haze (roofs and trees, most of all in the upscaling modes).
// The shader is unchanged: its "this frame" block gets the identity and its "last frame" block this matrix, so it
// applies just the one matrix to each pixel. A still camera gives the identity, and zero motion exactly.
static int inv4(const double m[16], double o[16]) {
    double a[4][8];
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 8; k++) a[r][k] = k < 4 ? m[r * 4 + k] : (k - 4 == r);
    for (int col = 0; col < 4; col++) {
        int piv = col;
        for (int r = col + 1; r < 4; r++)
            if (fabs(a[r][col]) > fabs(a[piv][col])) piv = r;
        if (fabs(a[piv][col]) < 1e-300) return 0;
        for (int k = 0; k < 8; k++) { double t = a[col][k]; a[col][k] = a[piv][k]; a[piv][k] = t; }
        double d = a[col][col];
        for (int k = 0; k < 8; k++) a[col][k] /= d;
        for (int r = 0; r < 4; r++)
            if (r != col) {
                double f = a[r][col];
                for (int k = 0; k < 8; k++) a[r][k] -= f * a[col][k];
            }
    }
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 4; k++) o[r * 4 + k] = a[r][k + 4];
    return 1;
}

// OoT's camera from its scene context block (vk_dlaa_draw): its scene vertex shaders take the world position to
// the camera through rows 0-2 (a 3x4 view) and then to clip space through rows 7-10 (the projection, 4x4), so the
// view-projection is rows 7-10 times rows 0-2.
static void ocam_vp(const float *v, double out[16]) {
    const float *P = v + 7 * 4;
    double V[16] = {v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11], 0, 0, 0, 1};
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 4; k++) {
            double acc = 0;
            for (int j = 0; j < 4; j++) acc += (double)P[r * 4 + j] * V[j * 4 + k];
            out[r * 4 + k] = acc;
        }
}

// For the shadow mask's temporal filter (vk_maskfilter.c): the same view-projection and inverse.
void vk_dlaa_ocam_vp(const float *rows, double out[16]) { ocam_vp(rows, out); }
int vk_dlaa_inv4(const double m[16], double out[16]) { return inv4(m, out); }

// Frame generation's camera (vk_dlaa_fg_camera) from the OoT block: the projection is rows 7-10, the view rows 0-2
// (rows: right, up, back; the camera looks down -z), m the reprojection from this frame's clip space to last frame's.
// Each matrix is handed out transposed, for row vectors.
static void fg_store(float out[16], const double m[16]) {
    for (int r = 0; r < 4; r++)
        for (int k = 0; k < 4; k++) out[r * 4 + k] = (float)m[k * 4 + r];
}
// The view distance in front of the camera at which the projection P (row-major, column vectors, looking down -z) gives
// clip depth z_ndc = k: (P[10] z + P[11]) / (P[14] z + P[15]) = k, solved for z. INFINITY where no distance gives it.
static double plane_dist(const double P[16], double k) {
    double den = P[10] - k * P[14];
    if (fabs(den) <= 1e-9 * (fabs(P[10]) + fabs(P[14]))) return INFINITY;
    return -(k * P[15] - P[11]) / den;
}

// Extract near/far planes and depth orientation from the frame's projection and NVN depth mode.
static int proj_planes(const double P[16], VkDlaaCamera *c) {
    int minus_w = nvn_depth_minus_w();
    double at0 = plane_dist(P, minus_w ? -1.0 : 0.0), at1 = plane_dist(P, 1.0);
    int inverted = !(at0 < at1);
    double n = inverted ? at1 : at0, f = inverted ? at0 : at1;
    if (!(n > 0 && n < VK_DLAA_FAR_INFINITE && f > n)) return 0;
    const int infinite = f >= VK_DLAA_FAR_INFINITE;
    const float near_plane = (float)n, far_plane = infinite ? VK_DLAA_FAR_INFINITE : (float)f;
    if (!(near_plane > 0 && far_plane > near_plane)) return 0;
    c->inverted = inverted;
    c->znear = near_plane;
    c->infinite = infinite;
    c->zfar = far_plane;
    static float logged_n, logged_f;
    static int logged_inv = -1, lines;
    if (lines < 20 && (inverted != logged_inv || fabsf(c->znear - logged_n) > 1e-4f * logged_n ||
                       fabsf(c->zfar - logged_f) > 1e-4f * logged_f)) {
        lines++, logged_inv = inverted, logged_n = c->znear, logged_f = c->zfar;
        char fz[32];
        if (c->infinite) snprintf(fz, sizeof fz, "infinite");
        else snprintf(fz, sizeof fz, "%.6g", c->zfar);
        rt_log("dlaa: the scene's projection: near %.6g, far %s, depth %s, clip depth %s (z row %.6g %.6g %.6g %.6g, "
               "w row %.6g %.6g %.6g %.6g)", c->znear, fz, inverted ? "inverted (1 near)" : "0 near, 1 far",
               minus_w ? "-w..w" : "0..w", P[8], P[9], P[10], P[11], P[12], P[13], P[14], P[15]);
    }
    return 1;
}

static void fg_camera(const float *v, const double m[16]) {
    double P[16], Pi[16], mi[16];
    for (int i = 0; i < 16; i++) P[i] = v[28 + i];
    if (!inv4(P, Pi) || !inv4(m, mi)) return;
    VkDlaaCamera *c = &D.fg_cam;
    fg_store(c->view_to_clip, P), fg_store(c->clip_to_view, Pi);
    fg_store(c->clip_to_prev_clip, m), fg_store(c->prev_clip_to_clip, mi);
    for (int i = 0; i < 3; i++) {
        c->right[i] = v[i], c->up[i] = v[4 + i], c->fwd[i] = -v[8 + i];
        // the position: -R^T t, R the view's 3x3 and t its last column
        c->pos[i] = -(v[i] * v[3] + v[4 + i] * v[7] + v[8 + i] * v[11]);
    }
    {   // DEBUG-CAMLOG: REKARINA_CAMLOG=1: each frame's camera direction (yaw from +z toward +x, pitch up, degrees) and
        // position, from the view matrix (how the camera answers input: docs/oot-notes.md, "The mouse camera, raw")
        static int on = -1;
        if (on < 0) on = cfg_get("REKARINA_CAMLOG") != NULL;
        if (on)
            rt_log("CAMLOG %llu %.4f %.4f %.3f %.3f %.3f t %.3f", (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED),
                   atan2(c->fwd[0], c->fwd[2]) * 57.29578, asin(fmax(-1.0, fmin(1.0, c->fwd[1]))) * 57.29578, c->pos[0], c->pos[1], c->pos[2],
                   plat_now_ns() / 1e6);
    }
    c->fov_y = P[5] != 0 ? 2.0f * (float)atan(1.0 / fabs(P[5])) : 1.0f;
    c->aspect = P[0] != 0 ? (float)fabs(P[5] / P[0]) : 1.777f;
    if (!(c->znear > 0)) c->znear = 0.1f, c->zfar = VK_DLAA_FAR_INFINITE, c->infinite = 1;
    c->depth_planes_valid = proj_planes(P, c);
    D.fg_cam_ok = 1;
}

typedef struct { double vp[16]; int valid; } ProjectionHistory;
enum { MV_DISABLED = 3, MV_DISABLED_CONTINUOUS };

// The scene target the upscaler reads this frame (image), against the one it read last frame. OoT draws its scene into
// one of a few pooled light targets of the same size and moves to another whenever its pass list changes, while the
// camera and the scene carry on: 571 moves in a minute standing at one dialogue, about one every six frames. The
// upscaler's history lives in its feature, not in the target, so a move whose motion vectors reach back to last
// frame's camera (mv_pass 1, or MV_DISABLED_CONTINUOUS) keeps it. A new history at each move never let fine dithered
// detail such as hair gather the whole 16-frame jitter cycle, and it stayed grainy. Without that camera, on the first
// frame, or with REKARINA_DLAA_TARGET_KEEP=0 (test: the old behaviour), the move starts a new history.
#ifdef GAME_OOT
static void stock_fg_motion(VkExec *ex, uint32_t ow, uint32_t oh);

static void sharpen_hud_input(VkExec *ex, const VkTexImage *output, int stock_scaler) {
    const int strength = cfg_scene_sharpness(stock_scaler);
    static int last_strength = -1;
    if (ex->gpu_cond) return;
    if (strength <= 0) {
        if (last_strength > 0) rt_log("vk: pre-HUD RCAS disabled");
        last_strength = 0;
        return;
    }
    VkImageView view = VK_NULL_HANDLE;
    if ((output->usage & (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT)) ==
        (VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        view = vk_exec_target_view(output, output->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0);
    if (view) {
        vk_exec_barrier(ex);
        full_barrier(ex->cb);
        if (vk_scene_rcas(ex->cb, output->image, view, output->format, output->width, output->height,
                          strength > 100 ? 100 : strength)) {
            ex->recorded++;
            vk_exec_draw_new_cb();
            full_barrier(ex->cb);
            if (strength != last_strength || vk_dlaa_pathlog())
                rt_log("vk: pre-HUD RCAS at %ux%u (strength %d)", output->width, output->height, strength);
            last_strength = strength;
            return;
        }
    }
    static int warned;
    if (!warned++) rt_log("vk: pre-HUD RCAS unavailable on the checked scene output");
}
#endif

void vk_dlaa_dispatch_recorded(VkExec *ex) {
    const VkTexImage *pack = D.scene.stock_pack, *output = D.scene.stock_output;
    D.scene.stock_pack = D.scene.stock_output = NULL;
#ifdef GAME_OOT
    const VkTexImage *rcas_output = output ? output : D.scene.rcas_output;
    D.scene.rcas_output = NULL;
    if (rcas_output) sharpen_hud_input(ex, rcas_output, 1);
#endif
    if (!replay_scene_live()) return;
    if (pack) {
        D.scene.packed = pack->image;
        D.scene.packed_serial = D.scene.replay.packed = pack->serial;
    }
    if (!output) return;
    const VkExecTarget target = {.image = output->image, .img = output, .width = output->width, .height = output->height};
    replay_scene_complete(ex, &target);
    D.scene.tonemapped = D.scene.packed = VK_NULL_HANDLE;
    if (hudless_wanted()) {
        D.scene.hudless.pending = 1;
        D.scene.hudless.width = target.width;
        D.scene.hudless.height = target.height;
        hudless_scene_target(output, target.image);
    }
#ifdef GAME_OOT
    if (!ex->gpu_cond && vk_dlaa_fg_only()) stock_fg_motion(ex, target.width, target.height);
#else
    (void)ex;
#endif
}

static void light_target_check(VkImage image, int mv) {
    if (D.last_color == image) return;
    static int keep = -1;
    if (keep < 0) keep = rt_knob("DLAA_TARGET_KEEP", "1");
    if (keep && D.last_color && (mv == 1 || mv == MV_DISABLED_CONTINUOUS)) D.target_moves_kept++;
    else D.reset = 1, D.why[2]++;
    D.last_color = image;
}

// The pass, recorded through the recorder like the replay's own commands. 0: no camera this frame (the vectors stay
// zero and DLSS starts a new history).
static int mv_pass(VkExec *ex, struct Size *e, VkImageView zv, float jx, float jy,
                   const float *projection_camera, ProjectionHistory *next) {
    VkCommandBuffer cb = ex->cb;
    uint32_t w = e->w, h = e->h;
    VkDescriptorBufferInfo cam = {0};
    MotionVectorsMode mode = frame_motion_mode();
    int disabled = mode == MOTION_VECTORS_OFF;
    if (disabled) {
        D.fg_cam_ok = 0;
        e->dist_ok = e->pose_distance_valid = 0;
    }
    int found = (disabled || mv_init()) && (projection_camera || (D.cam_addr &&
        vk_usnap_find(D.cam_addr, D.cam_size == OCAM_BLOCK ? OCAM_BYTES : D.cam_size, 4, &cam)));
    if (disabled && !found) {
        D.cam_prev_ok = 0;
        return MV_DISABLED;
    }
    if (!disabled && !found) {
        D.mv_none++;
        D.cam_prev_ok = 0;
        e->dist_ok = e->pose_distance_valid = 0;
        return 0;
    }
    // this frame's view-projection as the snapshot holds it (or from CPU projection), for the reprojection
    const float *cr = projection_camera ? projection_camera
                      : D.cam_size == OCAM_BLOCK ? (const float *)vk_usnap_bytes(D.cam_addr, OCAM_BYTES)
                                                 : (const float *)vk_usnap_bytes(D.cam_addr + D.cam_vp, 64);
    double cur_vp[16], m[16], ci[16];
    int have_m = 0;
    float clean[OCAM_BYTES / 4];
    if (cr && !projection_camera && D.cam_size == OCAM_BLOCK) cr = pj_clean(D.cam_addr, cr, clean);
    {   // DEBUG-CAMLOG: REKARINA_DLAA_CAMLOG=1: the camera block's rows 0-10 and the depth convention, a few times
        static int on = -1;
        if (on < 0) on = rt_knob("DLAA_CAMLOG", "0");
        if (on && cr && D.cam_size == OCAM_BLOCK && (D.evals == 300 || D.evals == 301 || D.evals == 900)) {
            rt_log("CAMLOG eval %llu addr %#llx scene %ux%u depth format %d depth_minus_w %d", (unsigned long long)D.evals,
                   (unsigned long long)D.cam_addr, w, h, (int)D.scene.depth.format, nvn_depth_minus_w());
            for (int r = 0; r < 11; r++)
                rt_log("CAMLOG row %2d: %14.7g %14.7g %14.7g %14.7g", r, cr[r * 4], cr[r * 4 + 1], cr[r * 4 + 2], cr[r * 4 + 3]);
        }
    }
    if (cr && D.cam_size == OCAM_BLOCK) ocam_vp(cr, cur_vp);
    else if (cr)
        for (int i = 0; i < 16; i++) cur_vp[i] = cr[i];
    if (cr) {
        if (D.cam_prev_ok && inv4(cur_vp, ci)) {
            for (int r = 0; r < 4; r++)
                for (int k = 0; k < 4; k++) {
                    double v = 0;
                    for (int j = 0; j < 4; j++) v += D.cam_prev[r * 4 + j] * ci[j * 4 + k];
                    m[r * 4 + k] = v;
                }
            have_m = 1;
        }
    }
    D.fg_cam_ok = 0;
    if (projection_camera && !have_m)
        for (int i = 0; i < 16; i++) m[i] = i % 5 == 0;
    if (cr && (have_m || projection_camera) && D.cam_size == OCAM_BLOCK) fg_camera(cr, m);
    if (projection_camera && D.reset) {
        for (int i = 0; i < 16; i++) m[i] = i % 5 == 0;
        fg_camera(cr, m);
        have_m = 0;
    }
    int s = D.cam_slot, had_prev = D.cam_prev_ok;
    int models = motion_vectors_models(mode);
    if (D.cam_size == CAM_BLOCK && !disabled) {   // the whole block, for the per-object motion pass (the deferred engine's only)
        VkBufferCopy bc = {cam.offset, (VkDeviceSize)s * CAM_SLOT, CAM_BLOCK};
        vkCmdCopyBuffer(cb, cam.buffer, D.camb, 1, &bc);
        full_barrier(cb);
    }
    if (next) {
        next->valid = cr != NULL;
        if (cr) memcpy(next->vp, cur_vp, sizeof cur_vp);
    } else {
        if (D.cam_size == CAM_BLOCK) D.cam_slot ^= 1;
        D.cam_prev_ok = cr != NULL;
        if (cr) memcpy(D.cam_prev, cur_vp, sizeof cur_vp);
    }
    if (disabled) return had_prev && have_m ? MV_DISABLED_CONTINUOUS : MV_DISABLED;
    if (!had_prev || !have_m) {   // nothing to reproject into: zero motion, new history
        e->dist_ok = e->pose_distance_valid = 0;
        return 0;
    }
    VkDescriptorImageInfo zi = {D.mv_samp, zv, VK_IMAGE_LAYOUT_GENERAL};
    VkDescriptorImageInfo oi = {VK_NULL_HANDLE, e->mv_view, VK_IMAGE_LAYOUT_GENERAL};
    uint32_t ms = D.mvm_slot++ % MVM_SLOTS;
    VkDeviceSize matrix_offset = (VkDeviceSize)ms * CAM_BYTES;
    float *blk = (float *)(D.mvm_map + matrix_offset);
    for (int i = 0; i < 16; i++) blk[i] = (float)m[i];
    vmaFlushAllocation(vk_ctx()->vma, D.mvm_mem, matrix_offset, 16 * sizeof(float));
    VkDescriptorBufferInfo prev = {D.mvm, matrix_offset, 16 * sizeof(float)};
    VkDescriptorImageInfo doi = {VK_NULL_HANDLE, e->dist_view[e->dist_cur], VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr[4] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &zi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &oi},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 2, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, .pBufferInfo = &prev},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 3, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &doi}};
    times_begin(cb);
    times_mark(cb, T_MV0);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.mv_pipe);
    rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_COMPUTE, D.mv_layout, 0, 4, wr);
    int write_distance = cr && D.cam_size == OCAM_BLOCK;
    MvPush pc = {{(float)w, (float)h}, {jx, jy},
                 {write_distance ? cr[38] : 0.0f, write_distance ? cr[39] : 0.0f,
                  write_distance ? cr[42] : 0.0f, write_distance ? cr[43] : 0.0f},
                 (uint32_t)write_distance, (uint32_t)motion_vectors_camera(mode)};
    vkCmdPushConstants(cb, D.mv_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cb, (w + 7) / 8, (h + 7) / 8, 1);
    full_barrier(cb);
    times_mark(cb, T_MV1);
    vk_mvcheck_camera(cb, e->mv, w, h, jx, jy);   // REKARINA_MV_CHECK (test)
    // the characters' own motion over the camera's (vk_exec_draw.c, per-object motion)
    uint64_t c0 = T.on > 0 ? kern_now_ns() : 0;
    uint64_t token = projection_camera ? D.projection.sample.token : 0;
    int object_motion = D.cam_size == CAM_BLOCK;
    if (models && D.cam_size == OCAM_BLOCK)
        object_motion = vk_fg_pose_prepare(ex, D.cam_addr, D.scene.depth.image, w, h, token) && !D.reset && !D.stepped;
    if (models && object_motion && vk_mvobj_execute(ex, D.cam_addr, e->mv_view, zv, D.scene.depth.format, w, h, token)) full_barrier(cb);
    if (T.on > 0) T.cpu_mv += kern_now_ns() - c0;
    times_mark(cb, T_OBJ);
    if (T.on > 0) T.used[T.slot] = 1;   // the DLSS marks follow in evaluate
    vk_mvcheck_final(cb, ex->fence, e->mv, D.scene.color.image, D.scene.color.format, D.scene.color.level, D.scene.color.base_layer, w, h, jx, jy);
    D.mv_frames++;
    e->pose_distance_valid = write_distance && D.fg_cam_ok;
    e->pose_distance_frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    e->pose_history[0] = jx, e->pose_history[1] = jy;
    e->pose_history[2] = D.fg_cam.view_to_clip[11];
    e->pose_history[3] = D.fg_cam.view_to_clip[15];
    e->dist_cur ^= 1, e->dist_ok = write_distance;
    return 1;
}

// Last frame's camera block for the per-object motion pass, recorded before this frame's mv_pass: that pass writes
// this frame's into slot cam_slot and leaves the other slot, last frame's, as it is. 0: none (no history).
int vk_dlaa_prev_camera(VkDescriptorBufferInfo *out, float jitter[2]) {
    if (!dl_active() || !D.cam_prev_ok || !D.camb) return 0;
    if (D.cam_size != CAM_BLOCK) return 0;   // the per-object pass knows the deferred engine's block only
    *out = (VkDescriptorBufferInfo){D.camb, (VkDeviceSize)(D.cam_slot ^ 1) * CAM_SLOT, CAM_BLOCK};
    jitter[0] = D.jit_ok ? D.jx : 0.0f, jitter[1] = D.jit_ok ? D.jy : 0.0f;
    return 1;
}

// ---------------------------------------------------------------- jitter
// The scene's draws (every pass drawing with the scene depth, until the DLAA point) have their viewport moved by a
// sub-pixel Halton(2,3) offset, which shifts what they rasterise as a jittered projection would; full-screen passes
// that shade by window position are unaffected. Jitter only follows a frame DLSS ran on: a jittered frame that DLSS
// does not resolve shakes on screen, so a missed DLAA point stops the jitter for a second.
// ---------------------------------------------------------------- scene jitter in guest memory (REKARINA_DLAA_PJITTER=2)
// OoT's frame has no jitter of its own, and DLSS without one repeats the same sample of every pixel while the camera
// stands: the picture goes soft, and sharp again at the next camera motion (the "sharpness jump" in dialogues).
// The game copies its camera into many uniform blocks, each renderer its own: the scene context (0x3500 bytes at c3:
// view rows 0-2, view-projection rows 3-6, projection rows 7-10, more copies further down), a 0x3c0-byte view block with
// the projection and its inverse, 0x210-byte blocks with the six matrices transposed (characters, water, particles),
// 0x700-byte blocks of the mesh stages with the view-projection, and others. A jitter in one of them moves one part of
// the picture: the scene context alone left the water, the fog sheets and the characters where they were, and DLSS,
// told that everything had moved, drew a grid over them.
// So the offset goes into every copy, in the game's own memory, before the submission that draws the scene takes its
// uniform and mirror copies: the scan below lists the submission's uniform blocks and finds the frame's camera (the
// scene context of a draw into a D32F depth target of the scene's size); every 16 floats equal to that camera's
// projection, view-projection or one of their inverses, as stored or transposed, get the offset in the form their
// kind takes (J M for a matrix that ends in clip space, M inverse(J) for one that starts there). Shaders that reach the
// bytes through pointers see the same camera, since the bytes themselves change.
// Where the matrices sit is learned per block type (size and binding point): a block of a type is searched whole a few
// times, afterwards only the places found are compared. A slot is jittered from what the game wrote; when it still holds
// what was written here (the game left the block alone since), the clean matrix kept from then is the base again, and
// with jitter off the clean matrices go back.
enum { CJ_P, CJ_VP, CJ_IP, CJ_IVP, CJ_T = 4, CJ_PATTERNS = 8 };   // | CJ_T: stored transposed
#define CJ_TYPES 512u
#define CJ_SLOTS 16384u
#define CJ_BINDS 4096
#define CJ_SEARCHES 6   // whole-block searches per type before only the known places are compared
typedef struct { uint32_t key, searches; uint16_t n; struct { uint16_t off; uint8_t kind; } s[14]; } CjType;
typedef struct { uint64_t at, stamp; float clean[16], wrote[16]; uint8_t kind, have; } CjSlot;
typedef struct { uint64_t addr; uint32_t size, key; } CjBind;
static struct {
    int mode;            // REKARINA_DLAA_PJITTER: 0 off, 1 the scene context's uniform copy only (vk_dlaa_pjitter), 2 this; -1 not read
    int ys;              // REKARINA_DLAA_PJ_YS: the sign of clip y against window y (-1: y up)
    unsigned kinds;      // REKARINA_DLAA_PJ_KINDS: bit per CJ_P..CJ_IVP (default 3: CJ_P and CJ_VP)
    CjType type[CJ_TYPES];
    CjSlot *slot;        // CJ_SLOTS, by address
    CjBind *bind;        // this submission's uniform blocks
    int nbind;
    uint64_t *seen;      // CJ_BINDS * 2 words: (addr, size) already listed
    uint64_t stamp;      // counts frames (vk_dlaa_frame)
    float dx, dy;        // this frame's offset in NDC
    uint64_t ctx[4];     // this frame's scene contexts
    int nctx;
    float pat[CJ_PATTERNS][16], scale[CJ_PATTERNS];
    uint64_t applied, restored, searched, slots_live;
    uint64_t front[256][2];   // the scan's last (addr, size) per bucket: most binds repeat a few blocks
    uint64_t ns_scan, ns_apply, timed;   // time spent, per frame with a scene (PJLOG)
} SJ = {.mode = -1};

static void sj_read(void) {
    if (SJ.mode >= 0) return;
    const char *m = cfg_get("REKARINA_DLAA_PJITTER"), *y = cfg_get("REKARINA_DLAA_PJ_YS"), *k = cfg_get("REKARINA_DLAA_PJ_KINDS");
#ifdef GAME_OOT
    SJ.mode = m && *m ? atoi(m) : 2;   // OoT: every copy of the camera, in the game's memory
#else
    SJ.mode = m && *m ? atoi(m) : 0;
#endif
    SJ.ys = y && *y ? atoi(y) : -1;
    // 3: the projection and the view-projection only. With their inverses jittered too (15) the fog and what lies under
    // water, which rebuild positions through an inverse and blend with their own last frame, changed from frame to
    // frame under a standing camera: mean second difference of consecutive frames in the upper half of the picture
    // 0.44 against 0.21 without jitter (the player saw it as flicker under water and in the fog), 0.31 with 3; the
    // detail measure of the still picture is the same (11.6).
    SJ.kinds = k && *k ? (unsigned)strtoul(k, NULL, 0) : 3u;
    if (SJ.mode == 2) {
        SJ.slot = calloc(CJ_SLOTS, sizeof *SJ.slot), SJ.bind = malloc(sizeof *SJ.bind * CJ_BINDS);
        SJ.seen = malloc(sizeof *SJ.seen * CJ_BINDS * 2 * 2);
        if (!SJ.slot || !SJ.bind || !SJ.seen) SJ.mode = 0;
    }
    if (SJ.mode) rt_log("dlaa: projection jitter mode %d (y sign %d, kinds %#x)", SJ.mode, SJ.ys, SJ.kinds);
}

// The offset applied to 16 floats as stored. To clip (projection, view-projection): rows 0 and 1 take d times row 3.
// From clip (their inverses): column 3 loses d times columns 0 and 1. Transposed storage swaps rows and columns.
static void cj_jit(float *m, int kind, float dx, float dy) {
    int t = kind & CJ_T, k = kind & 3;
    if (k == CJ_P || k == CJ_VP) {
        if (!t) for (int j = 0; j < 4; j++) m[j] += dx * m[12 + j], m[4 + j] += dy * m[12 + j];
        else for (int i = 0; i < 4; i++) m[i * 4] += dx * m[i * 4 + 3], m[i * 4 + 1] += dy * m[i * 4 + 3];
    } else {
        if (!t) for (int i = 0; i < 4; i++) m[i * 4 + 3] -= dx * m[i * 4] + dy * m[i * 4 + 1];
        else for (int j = 0; j < 4; j++) m[12 + j] -= dx * m[j] + dy * m[4 + j];
    }
}

static int cj_match(const float *v, int kind) {
    const float *p = SJ.pat[kind];
    float rel = (kind & 3) >= CJ_IP ? 2e-3f : 3e-5f, abs_ = ((kind & 3) >= CJ_IP ? 2e-4f : 1e-6f) * SJ.scale[kind];
    for (int i = 0; i < 16; i++)
        if (!(fabsf(v[i] - p[i]) <= rel * fabsf(p[i]) + abs_)) return 0;
    return 1;
}

static CjSlot *cj_slot(uint64_t at, int add) {
    uint32_t h = (uint32_t)((at >> 2) * 0x9E3779B97F4A7C15ull >> 50) & (CJ_SLOTS - 1);
    CjSlot *old = NULL;
    for (uint32_t k = 0; k < 24; k++, h = (h + 1) & (CJ_SLOTS - 1)) {
        CjSlot *e = &SJ.slot[h];
        if (e->at == at) return e;
        if (!e->at) { old = e; break; }
        if (!old || e->stamp < old->stamp) old = e;
    }
    if (!add || !old) return NULL;
    VkMemSpan sp;
    if (old->at && old->have && vk_mem_resolve(old->at, 64, &sp)) {   // an evicted slot the game left jittered goes back first
        float *g = (float *)(uintptr_t)old->at;
        if (!memcmp(g, old->wrote, sizeof old->wrote)) memcpy(g, old->clean, sizeof old->clean), SJ.restored++;
    }
    memset(old, 0, sizeof *old);
    old->at = at;
    return old;
}

static CjType *cj_type(uint32_t key) {
    uint32_t h = (key * 2654435761u) >> 23 & (CJ_TYPES - 1);
    for (uint32_t k = 0; k < CJ_TYPES; k++, h = (h + 1) & (CJ_TYPES - 1)) {
        if (SJ.type[h].key == key) return &SJ.type[h];
        if (!SJ.type[h].key) { SJ.type[h].key = key; return &SJ.type[h]; }
    }
    return NULL;
}

// Every place in the block that holds one of the frame's camera matrices joins the type's list.
static void cj_search(CjType *t, const float *v, uint32_t nf) {
    t->searches++, SJ.searched++;
    for (int kind = 0; kind < CJ_PATTERNS; kind++) {
        const float *p = SJ.pat[kind];
        int piv = 0;
        for (int i = 1; i < 16; i++)
            if (fabsf(p[i]) > fabsf(p[piv])) piv = i;
        float tol = ((kind & 3) >= CJ_IP ? 2e-3f : 3e-5f) * fabsf(p[piv]) + 1e-12f;
        for (uint32_t i = 0; i + 16 <= nf; i++) {
            if (!(fabsf(v[i + (uint32_t)piv] - p[piv]) <= tol) || !cj_match(v + i, kind)) continue;
            int k = 0;
            while (k < t->n && !(t->s[k].off == i && t->s[k].kind == kind)) k++;
            if (k == t->n && t->n < 14) t->s[t->n].off = (uint16_t)i, t->s[t->n++].kind = (uint8_t)kind;
        }
    }
}

static void cj_restore_all(void) {
    if (!SJ.slots_live) return;
    for (uint32_t i = 0; i < CJ_SLOTS; i++) {
        CjSlot *e = &SJ.slot[i];
        if (!e->at || !e->have) continue;
        float *g = (float *)(uintptr_t)e->at;
        VkMemSpan sp;
        if (vk_mem_resolve(e->at, 64, &sp) && !memcmp(g, e->wrote, sizeof e->wrote)) memcpy(g, e->clean, sizeof e->clean), SJ.restored++;
        e->have = 0;
    }
    SJ.slots_live = 0;
}

typedef struct { const tex_t *depth; uint64_t c3[6]; } SjScan;
static void sj_note(SjScan *s, const p_bind_buf *b) {
    (void)s;
    uint64_t *fr = SJ.front[(b->addr >> 6 ^ b->addr >> 14) & 255];
    if (fr[0] == b->addr && fr[1] == b->size) return;
    fr[0] = b->addr, fr[1] = b->size;
    if (!b->addr || b->size < 64 || b->size > (64u << 10) || (b->addr & 3) || nvn_gpu_only(b->addr) || SJ.nbind == CJ_BINDS) return;
    uint32_t h = (uint32_t)((b->addr ^ b->size << 40) * 0x9E3779B97F4A7C15ull >> 51) & (CJ_BINDS * 2 - 1);
    for (;; h = (h + 1) & (CJ_BINDS * 2 - 1)) {
        if (SJ.seen[h * 2] == b->addr && SJ.seen[h * 2 + 1] == b->size) return;
        if (!SJ.seen[h * 2]) break;
    }
    SJ.seen[h * 2] = b->addr, SJ.seen[h * 2 + 1] = b->size;
    SJ.bind[SJ.nbind++] = (CjBind){b->addr, (uint32_t)b->size, (uint32_t)b->size << 8 | (uint32_t)(b->stage & 7) << 5 | (uint32_t)(b->index & 31)};
}
static void sj_walk(const op_hdr *h, int depth, SjScan *s) {
    for (;;) {
        const void *a = nvn_op_payload(h);
        switch (h->op) {
        case OP_END: return;
        case OP_JUMP: h = (const op_hdr *)(uintptr_t)((const p_jump *)a)->target; continue;
        case OP_CALL:
            if (depth < 16) sj_walk(nvn_op_first(((const p_jump *)a)->target), depth + 1, s);
            break;
        case OP_SET_RT: s->depth = ((const p_set_rt *)a)->depth; break;
        case OP_BIND_UBO: {
            const p_bind_buf *b = a;
            if (b->index == 0 && b->stage >= 0 && b->stage < 6) s->c3[b->stage] = b->size == OCAM_BLOCK ? b->addr : 0;
            sj_note(s, b);
            break;
        }
        case OP_DRAW: case OP_DRAW_IDX: case OP_DRAW_IDX_IND:
#ifdef GAME_NS2
        case OP_DRAW_MESH: case OP_DRAW_MESH_IND:
#endif
            if (SJ.nctx == 4 || !s->depth || s->depth->d.format != NVN_FORMAT_DEPTH32F || s->depth->d.width != D.pj_w ||
                s->depth->d.height != D.pj_h)
                break;
            for (int st = 0; st < 6; st++) {
                uint64_t ad = s->c3[st];
                int k = 0;
                while (k < SJ.nctx && SJ.ctx[k] != ad) k++;
                if (ad && k == SJ.nctx && SJ.nctx < 4 && !nvn_gpu_only(ad)) SJ.ctx[SJ.nctx++] = ad;
            }
            break;
        default: break;
        }
        h = nvn_op_next(h);
    }
}

static int cj_inv4f(const float *m, float *o) {
    double a[16], r[16];
    for (int i = 0; i < 16; i++) a[i] = m[i];
    if (!inv4(a, r)) return 0;
    for (int i = 0; i < 16; i++) o[i] = (float)r[i];
    return 1;
}

// The game's bytes at a slot as the game wrote them: what is there, or the clean copy when what is there was written here.
static const float *cj_clean(uint64_t at) {
    const float *g = (const float *)(uintptr_t)at;
    const CjSlot *e = cj_slot(at, 0);
    return e && e->have && !memcmp(g, e->wrote, sizeof e->wrote) ? e->clean : g;
}

// vk_exec.c vk_exec_record, before the submission's mirror and uniform copies are taken.
void vk_dlaa_scene_jitter(VkExec *ex, const uint64_t *handles, int n) {
    sj_read();
    if (scene_projection_enabled()) {
        cj_restore_all();
        return;
    }
    if (SJ.mode != 2) return;
    if (!dl_active() || D.hold || !D.pj_w || fsr1()) {   // FSR 1 has no history: a jittered frame would shake on screen
        cj_restore_all();
        return;
    }
    if (scene_closed() || n <= 0 || SJ.nctx) return;   // one submission a frame draws the scene: the first with a scene context
    SjScan s = {.depth = ex->rt.depth};
    uint64_t t0 = plat_now_ns();
    SJ.nbind = 0;
    memset(SJ.seen, 0, sizeof *SJ.seen * CJ_BINDS * 2 * 2);
    memset(SJ.front, 0, sizeof SJ.front);
    for (int st = 0; st < 6; st++) {
        s.c3[st] = ex->ubo[st][0].size == OCAM_BLOCK ? ex->ubo[st][0].addr : 0;
        for (int k = 0; k < 18; k++) sj_note(&s, &ex->ubo[st][k]);   // bindings that carry over from the queue's last submission
    }
    for (int i = 0; i < n && handles[i]; i++) sj_walk(nvn_op_first(handles[i]), 0, &s);
    uint64_t t1 = plat_now_ns();
    SJ.ns_scan += t1 - t0;
    if (!SJ.nctx) return;
    // the frame's camera, as the game wrote it
    const uint64_t c0 = SJ.ctx[0];
    memcpy(SJ.pat[CJ_VP], cj_clean(c0 + 12 * 4), 64);
    memcpy(SJ.pat[CJ_P], cj_clean(c0 + 28 * 4), 64);
    if (!cj_inv4f(SJ.pat[CJ_P], SJ.pat[CJ_IP]) || !cj_inv4f(SJ.pat[CJ_VP], SJ.pat[CJ_IVP])) { SJ.nctx = 0; return; }
    for (int k = 0; k < 4; k++) {
        float mx = 0;
        for (int i = 0; i < 16; i++) {
            SJ.pat[k | CJ_T][i] = SJ.pat[k][(i & 3) * 4 + (i >> 2)];
            if (fabsf(SJ.pat[k][i]) > mx) mx = fabsf(SJ.pat[k][i]);
        }
        SJ.scale[k] = SJ.scale[k | CJ_T] = mx;
    }
    SJ.dx = 2.0f * D.jx / (float)D.pj_w, SJ.dy = (float)SJ.ys * 2.0f * D.jy / (float)D.pj_h;
    for (int i = 0; i < SJ.nbind; i++) {
        const CjBind *b = &SJ.bind[i];
        CjType *t = cj_type(b->key);
        if (!t) continue;
        float *g = (float *)(uintptr_t)b->addr;
        uint32_t nf = b->size / 4;
        // a block whose slots were written here is searched on its clean matrices: put them back for the search
        if (t->searches < CJ_SEARCHES || !((SJ.stamp + (b->addr >> 8)) & 1023)) {
            for (int k = 0; k < t->n; k++) {
                CjSlot *e = t->s[k].off + 16u <= nf ? cj_slot(b->addr + t->s[k].off * 4u, 0) : NULL;
                if (e && e->have && !memcmp(g + t->s[k].off, e->wrote, 64)) memcpy(g + t->s[k].off, e->clean, 64), e->have = 0;
            }
            cj_search(t, g, nf);
        }
        for (int k = 0; k < t->n; k++) {
            uint32_t off = t->s[k].off;
            int kind = t->s[k].kind;
            if (off + 16 > nf || !(SJ.kinds >> (kind & 3) & 1)) continue;
            CjSlot *e = cj_slot(b->addr + off * 4u, 1);
            if (!e || (e->have && e->stamp == SJ.stamp)) continue;   // (two kinds can share a place only by coincidence)
            float w[16];
            if (e->have && !memcmp(g + off, e->wrote, 64)) memcpy(w, e->clean, 64);       // the game left it
            else if (cj_match(g + off, kind)) memcpy(w, g + off, 64), memcpy(e->clean, w, 64);   // written anew: this frame's camera
            else { e->have = 0; continue; }   // another camera's block of the same type, or last frame's matrices
            cj_jit(w, kind, SJ.dx, SJ.dy);
            memcpy(g + off, w, 64);
            memcpy(e->wrote, w, 64);
            e->kind = (uint8_t)kind, e->have = 1, e->stamp = SJ.stamp;
            SJ.applied++, SJ.slots_live = 1;
        }
    }
    D.ux = D.jx, D.uy = D.jy, D.jit_used = 1;
    SJ.ns_apply += plat_now_ns() - t1, SJ.timed++;
}

// DEBUG-UBODUMP: REKARINA_UBODUMP=<frame>:<dir> writes every uniform block the draws of that frame bind (as the draw sees
// it: this submission's copy, else guest memory) to <dir>/ubo_<addr>_<size>.bin, and one line per draw to
// <dir>/index.txt: its number, the pass, the targets, the shaders' input ids and the blocks as stage:index:addr:size.
void vk_dlaa_ubodump(VkExec *ex, const GfxShader *vs, const GfxShader *ms, const GfxShader *fs, int mesh) {
    static int state = -1;
    static uint64_t frame;
    static char dir[400];
    static FILE *idx;
    static struct { uint64_t addr, size; } seen[8192];
    static int nseen;
    static uint32_t draw_no;
    if (state < 0) {
        const char *e = cfg_get("REKARINA_UBODUMP");
        char *end = NULL;
        state = 0;
        if (e && *e) {
            frame = strtoull(e, &end, 10);
            if (end && *end == ':') snprintf(dir, sizeof dir, "%s", end + 1), state = 1;
        }
    }
    if (!state) return;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (f < frame) return;
    if (f > frame) {
        if (idx) fclose(idx), idx = NULL, rt_log("ubodump: frame %llu: %u draws, %d blocks in %s", (unsigned long long)frame, draw_no, nseen, dir);
        state = 0;
        return;
    }
    if (!idx) {
        char p[512];
        snprintf(p, sizeof p, "%s/index.txt", dir);
        if (!(idx = fopen(p, "w"))) { state = 0; return; }
    }
    unsigned long long iv = 0, im = 0, iz = 0;
    const uint8_t *a = vs ? gfx_shader_input_id(vs) : NULL, *b = ms ? gfx_shader_input_id(ms) : NULL, *c = fs ? gfx_shader_input_id(fs) : NULL;
    for (int i = 0; i < 8; i++) iv = iv << 8 | (a ? a[i] : 0), im = im << 8 | (b ? b[i] : 0), iz = iz << 8 | (c ? c[i] : 0);
    fprintf(idx, "%u pass %u %ux%u ncolor %d c0 %d %ux%u depth %d %ux%u %s vs %016llx ms %016llx fs %016llx armed %d ended %d :", ++draw_no,
            (unsigned)ex->pass_serial, ex->pass_w, ex->pass_h, ex->pass_ncolor, ex->color[0].image ? (int)ex->color[0].format : 0,
            ex->color[0].width, ex->color[0].height, ex->depth.image ? (int)ex->depth.format : 0, ex->depth.width, ex->depth.height,
            mesh ? "mesh" : "draw", iv, im, iz, D.scene.armed, scene_closed());
    for (int st = 0; st < 6; st++)
        for (int k = 0; k < 18; k++) {
            const p_bind_buf *u = &ex->ubo[st][k];
            if (!u->addr || !u->size) continue;
            fprintf(idx, " %d:%d:%llx:%llx", st, k, (unsigned long long)u->addr, (unsigned long long)u->size);
            int j = 0;
            while (j < nseen && !(seen[j].addr == u->addr && seen[j].size == u->size)) j++;
            if (j < nseen || nseen == 8192 || nvn_gpu_only(u->addr)) continue;
            seen[nseen].addr = u->addr, seen[nseen++].size = u->size;
            uint64_t nb = u->size < (64u << 10) ? u->size : (64u << 10);
            const uint8_t *src = vk_usnap_bytes(u->addr, nb);
            if (!src) src = (const uint8_t *)(uintptr_t)u->addr;
            char p[512];
            snprintf(p, sizeof p, "%s/ubo_%llx_%llx.bin", dir, (unsigned long long)u->addr, (unsigned long long)u->size);
            FILE *o = fopen(p, "wb");
            if (o) fwrite(src, 1, nb, o), fclose(o);
        }
    fputc('\n', idx);
}

static float halton(uint32_t i, uint32_t b) {
    float f = 1, r = 0;
    for (; i; i /= b) f /= (float)b, r += f * (float)(i % b);
    return r;
}

void vk_dlaa_viewport(VkExec *ex, VkViewport *v) {
    if (scene_projection_enabled() || ex->scene_projection.sample.token) return;
    if (!D.jit_ok || scene_closed() || !ex->depth.image || ex->depth.image != D.jdepth) return;
    v->x += D.jx, v->y += D.jy;
    D.ux = D.jx, D.uy = D.jy, D.jit_used = 1;
}

static int scene_color(VkFormat f);

const SceneProjection *vk_dlaa_previous_projection(uint64_t frame) {
    return frame && D.projection_history.token && D.projection_history_frame == frame - 1 ? &D.projection_history : NULL;
}

void vk_dlaa_scene_projection(const p_scene_projection *p) {
    if (vk_dlaa_pathlog())
        rt_log("PATHLOG   projection complete token %llx color %llu depth %llu", (unsigned long long)p->sample.token,
               (unsigned long long)p->color_serial, (unsigned long long)p->depth_serial);
    if (D.projection.sample.token && D.projection.sample.token != p->sample.token)
        D.projection_conflict = 1;
    if (!D.projection.sample.token) D.projection = *p;
}

void vk_dlaa_draw(VkExec *ex) {
    if (scene_projection_enabled() || !dl_active() || scene_closed()) return;
    static int survey = -1;   // REKARINA_DLAA_SURVEY=1 (test): what is drawn into the armed light target
    if (survey < 0) survey = rt_knob("DLAA_SURVEY", "0");
    static uint64_t dfn[16];   // scene draws writing depth, by depth test (reversed depth: GREATER/GEQUAL)
    if (survey && ex->depth.image && ex->depth.image == D.jdepth && ex->pipe.ds.dt && ex->pipe.ds.dw)
        dfn[ex->pipe.ds.df & 15]++;
    if (survey && D.scene.armed && ex->color[0].image == D.scene.color.image) {
        static uint64_t n[2][2][2], arms, frames, last_frame;   // [blend][depth write][depth test]
        const VkPipeState *s = &ex->pipe;
        n[s->color.blend_en & 1][s->ds.dw][s->ds.dt]++;
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (f != last_frame) last_frame = f, frames++;
        static uint64_t ps;
        if (ex->pass_serial != ps) ps = ex->pass_serial, arms++;
        if (frames % 300 == 0) {
            static uint64_t logged;
            if (logged != frames) {
                logged = frames;
                rt_log("dlaa survey: scene draws writing depth by depth test (NVN func 1-8: never less equal lequal "
                       "greater notequal gequal always): %llu %llu %llu %llu %llu %llu %llu %llu",
                       (unsigned long long)dfn[1], (unsigned long long)dfn[2], (unsigned long long)dfn[3],
                       (unsigned long long)dfn[4], (unsigned long long)dfn[5], (unsigned long long)dfn[6],
                       (unsigned long long)dfn[7], (unsigned long long)dfn[8]);
                rt_log("dlaa survey: %llu frames, %.1f passes a frame into the light target after arming; draws a frame "
                       "(blend/depth write/depth test): 000 %.1f 001 %.1f 010 %.1f 011 %.1f 100 %.1f 101 %.1f 110 %.1f 111 %.1f",
                       (unsigned long long)frames, (double)arms / frames, (double)n[0][0][0] / frames, (double)n[0][0][1] / frames,
                       (double)n[0][1][0] / frames, (double)n[0][1][1] / frames, (double)n[1][0][0] / frames,
                       (double)n[1][0][1] / frames, (double)n[1][1][0] / frames, (double)n[1][1][1] / frames);
            }
        }
    }
    // The scene camera of a draw into the scene depth: the deferred engine binds a 0x940-byte block at c4 with the
    // view-projection in rows 3-6 (0x30); OoT's opaque scene draws bind a 0x3500-byte scene context at c3, whose
    // rows 0-2 and 7-10 their vertex shaders use as the view and the projection (ocam_vp). Its full-screen passes bind
    // a 0x200-byte block at c3 instead, which is not the camera. Without a camera there is no camera motion and no
    // jitter (next_jitter), so DLSS runs on each frame alone without shaking the picture.
    const p_bind_buf *b = &ex->ubo[0][1];   // c4: the replay keeps user banks from c3 on
    const VkExecTarget *z = &ex->depth;
    if (!z->image || z->width < DLAA_SCENE_MIN_W || z->format != VK_FORMAT_D32_SFLOAT || (D.jdepth && z->image != D.jdepth))
        return;
    if (b->size == CAM_BLOCK && b->addr) D.cam_addr = b->addr, D.cam_size = CAM_BLOCK, D.cam_vp = 0x30;
    else if (ex->ubo[0][0].size == OCAM_BLOCK && ex->ubo[0][0].addr && ex->color[0].image &&
             scene_color(ex->color[0].format)) {   // OoT: the scene context most light-target draws use this frame
        uint64_t a = ex->ubo[0][0].addr;
        int i = 0, best = 0;
        // the count is one for the frame, whichever thread records the draw (vk_exec.c par_try); evaluate reads it
        // after the chunks are in
        static char views_lock;
        while (__atomic_test_and_set(&views_lock, __ATOMIC_ACQUIRE)) _mm_pause();
        while (i < D.nviews && D.views[i].addr != a) i++;
        if (i == D.nviews && D.nviews < 8) D.views[D.nviews].addr = a, D.views[D.nviews++].n = 0;
        if (i < D.nviews) D.views[i].n++;
        for (int k = 1; k < D.nviews; k++)
            if (D.views[k].n > D.views[best].n) best = k;
        D.cam_addr = D.views[best].addr, D.cam_size = OCAM_BLOCK, D.cam_vp = 0;
        __atomic_clear(&views_lock, __ATOMIC_RELEASE);
    }
}

// The jitter moves on right after DLAA ran, not at the present: the replay can record the next frame's scene before
// the present of this one is handled, and a step taken there landed in the middle of a frame (worse after an off/on
// switch, which shifted that order). Jitter needs a frame DLSS ran on; a frame without the DLAA point stops it.
// The sequence is as long as NVIDIA's DLSS guide asks for the upscaling ratio, 8 x (output / scene)^2 phases (18 in
// Quality, 24 in Balanced, 32 in Performance, 72 in Ultra Performance), and 16 at least, the length DLAA has used from the start.
// REKARINA_DLAA_PJITTER=1: OoT's frame gets its sub-pixel offsets through the projection in its scene context block
// (rows 7-10 of the 0x3500-byte block at c3), in this submission's uniform copy, so every pass that takes the camera from
// the block sees the same offset: the depth pre-pass, the lit passes that test EQUAL against it and the material passes
// that shade from the visibility data. A viewport offset moved only what the viewport rasterises: the texture detail the
// material passes shade stayed where it was, DLSS was told it had moved, and the standing picture went soft. Only blocks
// drawn into a depth target of the scene's size are touched, so the shadow cascades keep their own camera. The row
// form Pn += d * P3 shifts clip space by d NDC units whatever the projection's convention.
void vk_dlaa_pjitter(VkExec *ex, int stage) {
    if (scene_projection_enabled()) return;
    sj_read();
    int on = SJ.mode == 1, ys = SJ.ys;
    if (!on || !dl_active() || fsr1() || scene_closed() || D.hold || !D.pj_w) return;
    const VkExecTarget *z = &ex->depth;
    const p_bind_buf *b = &ex->ubo[stage][0];
    if (b->size != OCAM_BLOCK || !b->addr || !z->image || z->format != VK_FORMAT_D32_SFLOAT || z->width != D.pj_w || z->height != D.pj_h)
        return;
    const float *cr = (const float *)vk_usnap_bytes(b->addr, OCAM_BYTES);
    if (!cr) return;
    uint32_t i = 0;
    while (i < D.pj_n && D.pj[i].addr != b->addr) i++;
    if (i < D.pj_n && !memcmp(cr + 28, D.pj[i].rows, sizeof D.pj[i].rows)) return;   // this copy has it already
    if (i == D.pj_n) {
        if (D.pj_n == 8) return;
        D.pj[D.pj_n++].addr = b->addr;
    }
    float dx = 2.0f * D.jx / (float)z->width, dy = (float)ys * 2.0f * D.jy / (float)z->height, r[8];
    for (int j = 0; j < 4; j++) r[j] = cr[28 + j] + dx * cr[40 + j], r[4 + j] = cr[32 + j] + dy * cr[40 + j];
    if (!vk_usnap_poke(b->addr + 28 * 4, r, sizeof r)) return;
    memcpy(D.pj[i].rows, r, sizeof r);
    D.pj[i].dx = dx, D.pj[i].dy = dy;
    D.ux = D.jx, D.uy = D.jy, D.jit_used = 1;
}

// The block without this frame's projection jitter: the camera for the motion vectors and frame generation.
static const float *pj_clean(uint64_t addr, const float *cr, float *out) {
    if (SJ.mode == 2) {   // the scene context's rows 3-6 and 7-10 as the game wrote them (vk_dlaa_scene_jitter)
        const CjSlot *vp = cj_slot(addr + 12 * 4, 0), *p = cj_slot(addr + 28 * 4, 0);
        int jv = vp && vp->have && vp->stamp == SJ.stamp, jp = p && p->have && p->stamp == SJ.stamp;
        if (!jv && !jp) return cr;
        memcpy(out, cr, OCAM_BYTES);
        if (jv) memcpy(out + 12, vp->clean, 64);
        if (jp) memcpy(out + 28, p->clean, 64);
        return out;
    }
    for (uint32_t i = 0; i < D.pj_n; i++)
        if (D.pj[i].addr == addr) {
            memcpy(out, cr, OCAM_BYTES);
            for (int j = 0; j < 4; j++) out[28 + j] -= D.pj[i].dx * cr[40 + j], out[32 + j] -= D.pj[i].dy * cr[40 + j];
            return out;
        }
    return cr;
}

static void next_jitter(uint32_t w, uint32_t ow) {
    if (scene_projection_enabled()) { D.jit_ok = 0; return; }
    static int nj = -1;
    if (nj < 0) nj = !rt_knob("DLAA_JITTER", "1");   // REKARINA_DLAA_JITTER=0: no jitter (test)
    double r = w ? (double)ow / w : 1.0;
    uint32_t phases = (uint32_t)ceil(8.0 * r * r - 1e-6);
    if (phases < 16) phases = 16;
    if (phases > 72) phases = 72;
    uint32_t i = D.jframe++ % phases + 1;
    D.jx = halton(i, 2) - 0.5f, D.jy = halton(i, 3) - 0.5f;
    // Jitter only where the camera is known (vk_dlaa_draw), and not on OoT's frame without CPU scene projection.
    D.jit_ok = !D.hold && !nj && D.cam_size == CAM_BLOCK && !fsr1();   // FSR 1 keeps no history to fill
}

// DEBUG-FEATFLAGS: REKARINA_DLAA_FEAT_OR=<bits> adds NGX feature flags (4: the motion vectors include the jitter)
static int dbg_feat_flags(void) {
    static int f = -1;
    if (f < 0) { const char *v = cfg_get("REKARINA_DLAA_FEAT_OR"); f = v ? atoi(v) : 0; }
    return f;
}

// The DLSS model (REKARINA_DLSS_PRESET, live): default takes NVIDIA's model per mode (evaluate: K for DLAA, Quality and
// Balanced, M for Performance, L for Ultra Performance), 0 here. A letter a..m takes that model in every mode: k pins
// K, j is the other transformer model, which NVIDIA describes as slightly less ghosting at the cost of extra flicker, e
// the older CNN model, cheaper on slower RTX cards. The value is an NVSDK_NGX_DLSS_Hint_Render_Preset (1 A ... 13 M),
// read when a feature is created; a change recreates the features (preset_check).
static int preset_value(void) {
    const char *v = cfg_get("REKARINA_DLSS_PRESET");
    int c = v && v[0] && !v[1] ? v[0] | 0x20 : 0;
    return c >= 'a' && c <= 'm' ? c - 'a' + 1 : 0;
}

// The quality setting DLSS is created with, from the scene's share of the output width.
static int perf_quality(uint32_t w, uint32_t ow) {
    if (w >= ow) return NGX_PERFQ_DLAA;
    double k = (double)w / ow;
    return k >= 0.66 ? 2 : k >= 0.57 ? 1 : k >= 0.49 ? 0 : 3;
}

// The HUD-less picture's copy of src (w x h, a colour image in GENERAL), and the log of the copies, below.
static void hudless_scene_target(const VkTexImage *texture, VkImage image) {
    D.scene.hudless.texture = texture;
    D.scene.hudless.image = image;
    D.scene.hudless.address = texture ? texture->addr : 0;
}

static void hudless(VkCommandBuffer cb, VkImage src, uint32_t w, uint32_t h);
static void hudless_log(void);
static uint64_t hudless_generation;

// Creates the DLSS feature of a size; 0 when NGX refuses. The parameter object is the frame's (evaluate).
static int feature_make(VkCommandBuffer cb, NgxParam *p, struct Size *e, uint32_t w, uint32_t h, uint32_t ow, uint32_t oh,
                        VkFormat cf, VkFormat zf) {
    p_setu(p, "CreationNodeMask", 1);
    p_setu(p, "VisibilityNodeMask", 1);
    p_setu(p, "Width", w);
    p_setu(p, "Height", h);
    p_setu(p, "OutWidth", ow);
    p_setu(p, "OutHeight", oh);
    p_seti(p, "PerfQualityValue", perf_quality(w, ow));
    // IsHDR | MVLowRes | AutoExposure: linear scene light, motion vectors at the render size, no exposure texture
    p_seti(p, "DLSS.Feature.Create.Flags", (1 << 0) | (1 << 1) | (1 << 6) | dbg_feat_flags());
    p_seti(p, "DLSS.Enable.Output.Subrects", 0);
    // The model per mode (NVSDK_NGX_DLSS_Hint_Render_Preset: 1 A ... 11 K, 12 L, 13 M), as NVIDIA recommends for
    // DLSS 4.5 (310.x): K (transformer) for DLAA, Quality and Balanced, M for Performance, L for Ultra Performance.
    // M and L cost more on cards without FP8 (RTX 20/30). dlss_preset (D.preset, preset_check) takes one model for
    // every mode instead.
    int pk = D.preset ? D.preset : 11, pm = D.preset ? D.preset : 13, pl = D.preset ? D.preset : 12;
    p_setu(p, "DLSS.Hint.Render.Preset.DLAA", (unsigned)pk);
    p_setu(p, "DLSS.Hint.Render.Preset.UltraQuality", (unsigned)pk);
    p_setu(p, "DLSS.Hint.Render.Preset.Quality", (unsigned)pk);
    p_setu(p, "DLSS.Hint.Render.Preset.Balanced", (unsigned)pk);
    p_setu(p, "DLSS.Hint.Render.Preset.Performance", (unsigned)pm);
    p_setu(p, "DLSS.Hint.Render.Preset.UltraPerformance", (unsigned)pl);
    NgxResult r = N.create(cb, NGX_FEATURE_SUPERSAMPLING, p, &e->feat);
    if (!NGX_OK(r) || !e->feat) return 0;
    D.reset = 1;
    int q = perf_quality(w, ow), pr = q == 0 ? pm : q == 3 ? pl : pk;
    rt_log("dlaa: DLSS feature created at %ux%u -> %ux%u, quality value %d, preset %c (scene format %d, depth format "
           "%d)", w, h, ow, oh, q, 'A' + pr - 1, (int)cf, (int)zf);
    return 1;
}

static int deferred;   // a scene-size G-buffer was seen: the deferred shape

static void projection_prepare(const SceneProjection *projection, uint32_t w, uint32_t h) {
    D.cam_addr = projection->camera, D.cam_size = OCAM_BLOCK, D.cam_vp = 0;
    const SceneProjection *last = &D.projection_history;
    if (!last->token || last->owner != projection->owner || last->generation != projection->generation ||
        last->settings != projection->settings ||
        (last->token & ~SCENE_PROJECTION_SHIFTED) + 1 != (projection->token & ~SCENE_PROJECTION_SHIFTED) ||
        w != D.jw || h != D.jh) D.cam_prev_ok = 0, D.reset = 1;
}

static void projection_commit(const SceneProjection *projection, const ProjectionHistory *next) {
    D.cam_prev_ok = next->valid;
    if (next->valid) memcpy(D.cam_prev, next->vp, sizeof next->vp);
    D.projection_history = *projection;
    D.projection_history_frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    D.projection_resolved = 1;
}

static void publish_fg(const struct Size *e, const VkExecTarget *z, VkImageView zv, float jx, float jy) {
    D.fg_depth = (VkFfxImage){z->image, z->format, e->w, e->h, 0};
    D.fg_mv = (VkFfxImage){e->mv, VK_FORMAT_R16G16_SFLOAT, e->w, e->h, 1};
    D.fg_depth_view = zv, D.fg_mv_view = e->mv_view;
    D.fg_viewport_jitter[0] = jx, D.fg_viewport_jitter[1] = jy;
    __atomic_add_fetch(&D.fg_fresh, 1, __ATOMIC_ACQ_REL);
}

static int projection_target(const VkExecTarget *t, uint64_t serial, const view_t *view) {
    return t->tex && serial && t->tex->serial == serial && t->layers == 1 &&
           t->level == (view->set_levels ? view->base_level : 0) &&
           t->base_layer == (view->set_layers ? view->min_layer : 0);
}

static const float *projection_rows(const p_scene_projection *p, const VkExecTarget *color,
                                    const VkExecTarget *depth, float *rows) {
    const SceneProjection *s = &p->sample;
    const CfgRenderValues *settings = cfg_render_values();
    if (!s->token || D.projection_conflict || s->settings != (settings ? settings->projection_revision : 0) ||
        !projection_target(depth, p->depth_serial, &p->dv) ||
        depth->width != color->width || depth->height != color->height) return NULL;
    if (color->layers != 1) return NULL;
    if (!scene_projection_extent_matches(s, color->width, color->height)) return NULL;
    memset(rows, 0, OCAM_BYTES);
    memcpy(rows, s->view, sizeof s->view);
    memcpy(rows + SCENE_PROJECTION_WORD, s->unjittered, sizeof s->unjittered);
    return rows;
}

// ---- OoT: DLSS before the bloom (REKARINA_DLAA_PRE_BLOOM=1).
//
// In native DLAA, resolving into the scene light target right before the bloom chain's first draw lets the bloom,
// tone mapping, and post-processing read the anti-aliased scene. Any frame that cannot resolve there falls back to
// the scaling dispatch. Because the bloom draw's pass is a single draw (below REKARINA_PAR_MIN), it always replays
// in order on the replay thread.
#define OOT_BLOOM_FS        UINT64_C(0x865a04d320227ae3)   // bloom chain's first fragment shader input-id prefix
#define OOT_SMAA_COMPOSE_CS UINT64_C(0x1f0ccd15c28b87b8)   // SMAA compose compute shader input-id prefix
#define OOT_SCALER_CS       UINT64_C(0xdf55fc291b0f86ba)   // post-SMAA scaler compute shader input-id prefix

enum {
    OOT_STAGE_FS           = 1,   // NVN_STAGE_FS
    OOT_STAGE_CS           = 5,   // NVN_STAGE_CS
    OOT_BLOOM_SCENE_SLOT   = 0,   // FS texture slot 0: scene light target
    OOT_SMAA_COLOR_SLOT    = 3,   // CS texture slot 3: tone-mapped float color read by SMAA compose
    OOT_SMAA_PACKED_SLOT   = 2,   // CS image slot 2: RGB9E5 output packed via R32UI view
    OOT_SCALER_PACKED_SLOT = 0,   // CS texture slot 0: RGB9E5 input read by the scaler
    OOT_SCALER_OUTPUT_SLOT = 1,   // CS image slot 1: HUD destination R11G11B10F image
};

static uint64_t shader_input_prefix(const GfxShader *shader) {
    const uint8_t *id = gfx_shader_input_id(shader);
    uint64_t prefix = 0;
    for (int i = 0; i < 8; i++) prefix = (prefix << 8) | id[i];
    return prefix;
}

// The bloom chain's first pass draws into two R11G11B10F color targets without depth.
static int pre_bloom_pass(const VkExec *ex) {
    if (ex->pass_ncolor != 2 || ex->depth.image) return 0;
    if (!ex->color[0].image || ex->color[0].format != VK_FORMAT_B10G11R11_UFLOAT_PACK32) return 0;
    if (!ex->color[1].image || ex->color[1].format != VK_FORMAT_B10G11R11_UFLOAT_PACK32) return 0;
    return 1;
}

// Returns 1 when fragment texture slot 0 samples the armed scene subresource in its native format.
static int pre_bloom_reads_scene(const VkExec *ex) {
    uint64_t handle = ex->tex[OOT_STAGE_FS][OOT_BLOOM_SCENE_SLOT];
    if (!handle) return 0;
    view_t view = {0};
    int has_view = 0;
    TexVkFormat mapped;
    vk_texture_lock();
    const VkTexImage *img = vk_texture_by_id_locked(NVN_HANDLE_TEX(handle), &view, &has_view);
    uint32_t base_level = (has_view && view.set_levels) ? view.base_level : 0;
    uint32_t min_layer = (has_view && view.set_layers) ? view.min_layer : 0;
    int format_ok = !has_view || !view.set_format || (tex_vk_format(view.format, &mapped) && mapped.vk == D.scene.color.format);
    int reads = img && !img->gone && img->image == D.scene.color.image &&
                base_level == D.scene.color.level && min_layer == D.scene.color.base_layer && format_ok;
    vk_texture_unlock();
    return reads;
}

// DEBUG-PATHLOG: logs every fragment texture slot sampling the armed scene image.
static void pre_bloom_trace(const VkExec *ex, const GfxShader *fs) {
    const GfxShaderRes *res;
    int count = gfx_shader_resources(fs, GFX_RES_TEX, &res);
    vk_texture_lock();
    for (int i = 0; i < count; i++) {
        int slot = res[i].slot;
        if (slot < 0 || slot >= 128) continue;
        uint64_t handle = ex->tex[OOT_STAGE_FS][slot];
        if (!handle) continue;
        view_t view = {0};
        int has_view = 0;
        const VkTexImage *img = vk_texture_by_id_locked(NVN_HANDLE_TEX(handle), &view, &has_view);
        if (!img || img->gone || img->image != D.scene.color.image) continue;
        rt_log("PATHLOG   scene sample pass %u %ux%u colors %u fs %016llx slot %d addr %llx mip %u layer %u token %llx target %llu/%llu",
               ex->pass_serial, ex->pass_w, ex->pass_h, ex->pass_ncolor,
               (unsigned long long)shader_input_prefix(fs), slot, (unsigned long long)img->addr,
               has_view && view.set_levels ? view.base_level : 0, has_view && view.set_layers ? view.min_layer : 0,
               (unsigned long long)D.projection.sample.token,
               (unsigned long long)(D.scene.color.tex ? D.scene.color.tex->serial : 0),
               (unsigned long long)(D.scene.depth.tex ? D.scene.depth.tex->serial : 0));
    }
    vk_texture_unlock();
}

// DLSS writes RGBA16F and blits the resolved result back into the R11G11B10F scene light target.
static int pre_bloom_blit_supported(void) {
    static int supported = -1;
    if (supported < 0) {
        VkFormatProperties src_props, dst_props;
        vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, VK_FORMAT_R16G16B16A16_SFLOAT, &src_props);
        vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, VK_FORMAT_B10G11R11_UFLOAT_PACK32, &dst_props);
        supported = (src_props.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT) != 0 &&
                    (dst_props.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_DST_BIT) != 0;
    }
    return supported;
}

// Returns NULL when the temporal upscaler can resolve into the scene at this bloom draw.
static const char *pre_bloom_blocked(const VkExec *ex) {
    if (!dl_active() || fsr1())
        return "no temporal upscaler is active";
    if (res_dlss_upscales())
        return "the temporal upscaler is not in native AA mode";
    if (!scene_projection_enabled())
        return "the CPU projection jitter is off";
#ifdef GAME_NS2
    if (ex->gpu_cond)
        return "the bloom draw is under a condition the GPU decides";
#endif
    const VkExecTarget *color = &D.scene.color;
    const VkExecTarget *depth = &D.scene.depth;
    unsigned output_w = 0, output_h = 0;
    wide_picture(&output_w, &output_h);
    if (color->width != output_w || color->height != output_h)
        return "the scene is not at the output's size";

    float rows[OCAM_BYTES / sizeof(float)];
    if (!projection_rows(&D.projection, color, depth, rows) ||
        !projection_target(color, D.projection.color_serial, &D.projection.cv))
        return "the projection sample is not this scene's";
    if (color->level || color->base_layer || depth->level || depth->base_layer)
        return "the scene is not a whole image";
    if (!color->img || color->img->gone || !depth->img || depth->img->gone ||
        color->img->levels != 1 || color->img->layers != 1 ||
        color->image != color->img->image || depth->image != depth->img->image)
        return "the scene is a view of a larger image";
    if (color->format != VK_FORMAT_B10G11R11_UFLOAT_PACK32 || color->format != color->img->format)
        return "the scene is not R11G11B10F";
    if (!(color->img->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT) || !pre_bloom_blit_supported())
        return "DLSS's result cannot be blitted into the scene";
    if (color->image == ex->color[0].image || color->image == ex->color[1].image)
        return "the bloom pass draws into the scene";
    return NULL;
}

#ifdef GAME_OOT
static void stock_fg_motion(VkExec *ex, uint32_t ow, uint32_t oh) {
    const VkExecTarget *color = &D.scene.color, *depth = &D.scene.depth;
    const SceneProjection *sample = &D.projection.sample;
    float rows[OCAM_BYTES / sizeof(float)];
    if (D.projection_resolved || (sample->token & SCENE_PROJECTION_SHIFTED) || !depth->img || depth->img->gone ||
        depth->level || depth->base_layer || !projection_rows(&D.projection, color, depth, rows)) return;
    vk_exec_par_join();
    vk_exec_barrier(ex);
    struct Size *e = size_for(ex->cb, color->width, color->height, ow, oh);
    VkImageView view = vk_exec_target_view(depth->img, depth->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT,
                                           0, 0, 1, 0);
    if (!e || !view) {
        static int logged;
        if (!logged++) rt_log("fg: could not prepare motion resources for the stock picture");
        return;
    }
    projection_prepare(sample, e->w, e->h);
    if (ow != D.ev_ow || oh != D.ev_oh) D.cam_prev_ok = 0, D.reset = 1;
    ProjectionHistory next = {0};
    int mv = mv_pass(ex, e, view, 0, 0, rows, &next);
    projection_commit(sample, &next);
    D.jw = e->w, D.jh = e->h, D.ev_ow = ow, D.ev_oh = oh;
    D.fg_reset |= D.reset || mv != 1;
    D.reset = !next.valid;
    if (mv == 1 && D.fg_cam_ok) {
        D.fg_menu = 0;
        publish_fg(e, depth, view, 0, 0);
        static int logged;
        if (!logged++) rt_log("fg: stock scaler provides motion at %ux%u for the %ux%u picture", e->w, e->h, ow, oh);
    }
    ex->recorded++;
    vk_exec_draw_new_cb();
}
#endif

static void evaluate(VkExec *ex, const VkExecTarget *dst, enum SceneEvaluation evaluation) {
#ifdef GAME_OOT
    if (fsr1()) return;
    // Replay can see a new provider before the present path updates activation.
    if (fsr3() && !vk_ffx_upscale_available()) return;
#endif
    const VkExecTarget *t = &D.scene.color, *z = &D.scene.depth;
    const int pre_bloom = evaluation == EVALUATE_NATIVE_PREBLOOM;
    const int early = pre_bloom || evaluation == EVALUATE_PREBLOOM;
    if (!dst) dst = t;
    uint32_t w = t->width, h = t->height, ow = dst->width, oh = dst->height;
    const int spatial = fsr1();
    const int projection_attempt = scene_projection_enabled() && !spatial && !D.oot_pass && !deferred;
    const int stock_up = scene_scaled();
    if (projection_attempt && stock_up) D.scene.phase = SCENE_RESOLVED;   // only a successful resolve may replace the game's scaling draw
    // the scene's draws may still be recorded by the chunk threads (vk_exec.c par_try): the camera they count
    // (vk_dlaa_draw) is read from here on
    vk_exec_par_join();
    VkCommandBuffer cb = ex->cb;
    vk_exec_barrier(ex);   // what drew the scene is done (no pass is open here)
    if (D.preset_dirty) {   // another model: features are created with it, so every size gets a new one
        for (int i = 0; i < 5; i++)
            if (D.sizes[i].out) free_size(&D.sizes[i]);
        D.last = NULL, D.preset_dirty = 0;
    }
    struct Size *e = size_for(cb, w, h, ow, oh);
    if (!e) return dl_fail(evaluation, "work image creation", 0);
    float rows[OCAM_BYTES / sizeof(float)];
    const int spatial_motion = spatial && fg_motion() && evaluation == EVALUATE_AT_DISPATCH &&
                               !(D.projection.sample.token & SCENE_PROJECTION_SHIFTED);
    const float *projection_camera = projection_attempt || spatial_motion ? projection_rows(&D.projection, t, z, rows) : NULL;
    const SceneProjection *projection = &D.projection.sample;
    ProjectionHistory next = {0};
    if (projection_attempt && (!projection_camera || D.jit_used)) {
        static unsigned rejected;
        if (rejected++ < 12)
            rt_log("dlaa: projection unavailable or stale: token=%#llx input=%ux%u depth match=%d",
                   (unsigned long long)projection->token, w, h,
                   projection_target(z, D.projection.depth_serial, &D.projection.dv));
        if (!pre_bloom) scene_projection_feedback(NULL, 0, 0, 0);   // before the bloom the scaling dispatch tries again
        D.cam_prev_ok = 0, D.reset = 1;
        return;
    }
    if (D.pre_bloom.last != pre_bloom) D.reset = 1;   // scene light and the finished picture keep separate histories
    if (projection_camera) {
        projection_prepare(projection, w, h);
        if (ow != D.ev_ow || oh != D.ev_oh) D.cam_prev_ok = 0, D.reset = 1;
    }
    D.stepped = 0;
    if (e != D.last) {   // another size's history is from another moment
        D.reset = 1, D.why[0]++, e->dist_ok = 0;
        rt_log("dlaa: the scene is %ux%u now (output %ux%u, %llu frames so far)", w, h, ow, oh, (unsigned long long)D.evals);
    }
    if (spatial && dst == t) return;   // FSR 1 at the picture's own size: nothing to scale, the scene stays as drawn
    const int sl = !fsr3() && !spatial && use_streamline_sr && vk_sl_sr_available();
    if (!fsr3() && !spatial && !sl && !D.params && !NGX_OK(N.alloc_params(&D.params))) return dl_fail(evaluation, "parameter allocation", 0);
    NgxParam *p = D.params;
    VkImageView zv = vk_exec_target_view(z->img, z->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_DEPTH_BIT, z->level,
                                         z->base_layer, 1, 0);
    if (pre_bloom && !zv) return dl_fail(evaluation, "scene depth view", 0);
    float jx = D.jit_used ? D.ux : 0.0f, jy = D.jit_used ? D.uy : 0.0f;
    if (projection_attempt) {
        jx = projection_camera ? projection->jitter[0] : 0.0f;
        jy = projection_camera ? projection->jitter[1] : 0.0f;
    }
    D.pj_w = w, D.pj_h = h;
    // REKARINA_DLAA_JITTER_TEST=<x>,<y> (test, read once): scales jitter given to upscaler on legacy path.
    static float jt[2];
    static int jt_read;
    if (!jt_read) {
        jt_read = 1;
        const char *v = cfg_get("REKARINA_DLAA_JITTER_TEST");
        if (v && sscanf(v, "%f,%f", &jt[0], &jt[1]) == 2)
            rt_log("dlaa: the upscaler is told the jitter times %g, %g (REKARINA_DLAA_JITTER_TEST, a test)", jt[0], jt[1]);
        else jt[0] = jt[1] = 1.0f;
    }
    const float tjx = projection_camera ? jx : jx * jt[0], tjy = projection_camera ? jy : jy * jt[1];
    // FSR 1 reads no motion; frame generation still does (vk_dlaa_fg_inputs)
#ifdef GAME_OOT
    int need_mv = !spatial || spatial_motion;
#else
    int need_mv = !spatial || vk_fg_wanted() || vk_dlssg_wanted();
#endif
    D.fg_cam_ok = 0;
    int mv = need_mv ? mv_pass(ex, e, zv, jx, jy, projection_camera, projection_camera ? &next : NULL) : MV_DISABLED;
    if (need_mv && mv != 1) {
        VkClearColorValue zero = {{0, 0, 0, 0}};
        VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, e->mv, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &all);
        full_barrier(cb);
        if (mv == 0) D.reset = 1, D.why[1]++;   // no camera: the history cannot be moved along
    }
    VkImageView cv = vk_exec_target_view(t->img, t->format, VK_IMAGE_VIEW_TYPE_2D, VK_IMAGE_ASPECT_COLOR_BIT, t->level,
                                         t->base_layer, 1, 0);
    if (pre_bloom && !cv) return dl_fail(evaluation, "scene color view", 0);
    int timed = T.on > 0 && T.used[T.slot];   // mv_pass began this frame's slot
    if (timed) times_mark(cb, T_DL0);
    uint64_t c0 = timed ? kern_now_ns() : 0;
#ifndef GAME_OOT
    if (spatial) {   // FSR 1 (vk_fsr.c vk_fsr_scene): through the recorder like the replay's own commands
        if (!vk_fsr_scene(cb, cv, w, h, e->out_view, ow, oh)) return dl_fail(evaluation, "FSR 1", 0);
        goto evaluated;
    }
#endif
    rec_drain();   // NGX / Streamline records into the command buffer itself: the recorder thread must have made every call so far
    if (timed) T.cpu_drain += kern_now_ns() - c0;
    if (fsr3()) {   // FSR 3.1 (vk_ffx.cpp) in DLSS's place: the same images, then the same blit below
        if (t->level || t->base_layer || z->level || z->base_layer) return dl_fail(evaluation, "FSR 3.1 on a target that is not level 0, layer 0", 0);
        light_target_check(t->image, mv);
        uint64_t now = plat_now_ns();
        double dt = D.last_eval_ns ? (now - D.last_eval_ns) / 1e6 : 0;
        D.last_eval_ns = now;
        VkFfxImage fc = {t->image, t->format, w, h, 0}, fz = {z->image, z->format, w, h, 0},
                   fm = {e->mv, VK_FORMAT_R16G16_SFLOAT, w, h, 1}, fo = {e->out, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh, 1};
        c0 = timed ? kern_now_ns() : 0;
        VkDlaaCamera camera;
        float fov = vk_dlaa_fg_camera(&camera) ? camera.fov_y : 0.0f;
        int ok = vk_ffx_upscale(cb, &fc, &fz, &fm, &fo, tjx, tjy, D.reset, dt > 0 && dt < 100 ? (float)dt : 16.67f, fov);
        if (timed) T.cpu_upscale += kern_now_ns() - c0;
        if (!ok)
            return dl_fail(evaluation, "FSR upscaling", 0);
        static int told;
        if (!told++) rt_log("dlaa: FSR upscaling at %ux%u -> %ux%u (scene format %d, depth format %d)", w, h, ow, oh,
                            (int)t->format, (int)z->format);
        goto evaluated;
    }
    if (sl) {
        VkDlaaCamera camera;
        if (!vk_dlaa_fg_camera(&camera) || t->level || t->base_layer || z->level || z->base_layer) {
            D.reset = 1;
            return;
        }
        Nx2SlImage color = {t->image, cv, t->format, w, h, VK_IMAGE_LAYOUT_GENERAL, t->img->usage};
        Nx2SlImage depth = {z->image, zv, z->format, w, h, VK_IMAGE_LAYOUT_GENERAL, z->img->usage};
        Nx2SlImage motion = {e->mv, e->mv_view, VK_FORMAT_R16G16_SFLOAT, w, h, VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT};
        Nx2SlImage output = {e->out, e->out_view, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh, VK_IMAGE_LAYOUT_GENERAL,
                             VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT};
        int quality = perf_quality(w, ow);
        Nx2SlSrMode mode = quality == 5 ? NX2SL_SR_DLAA : quality == 2 ? NX2SL_SR_QUALITY
                         : quality == 1 ? NX2SL_SR_BALANCED : quality == 3 ? NX2SL_SR_ULTRA_PERFORMANCE
                         : NX2SL_SR_PERFORMANCE;
        light_target_check(t->image, mv);
        c0 = timed ? kern_now_ns() : 0;
        int ok = vk_sl_sr_evaluate(cb, &color, &depth, &motion, &output, &camera, (uint32_t)D.presents,
                                   preset_value(), D.reset, tjx, tjy, mode);
        if (timed) T.cpu_upscale += kern_now_ns() - c0;
        renderer_mcp_dlss(RENDERER_MCP_STREAMLINE, w, h, ow, oh, ok, 0);
        if (!ok) return dl_fail(evaluation, "Streamline DLSS evaluation", 0);
        goto evaluated;
    }
    if (!e->feat && !feature_make(cb, p, e, w, h, ow, oh, t->format, z->format)) return dl_fail(evaluation, "DLSS feature creation", 0);
    if (!D.early_full && w < ow && !fsr3() && !spatial) {
        // The game's pause screens draw the scene at the picture's full size (2560x1440 under a 1707x960 scene): that
        // feature is made now, in the scene's first frame behind the loading screen, not when the inventory first opens.
        D.early_full = 1;
        struct Size *full = size_for(cb, ow, oh, ow, oh);
        if (full && !full->feat) feature_make(cb, p, full, ow, oh, ow, oh, t->format, z->format);
    }
    light_target_check(t->image, mv);
    NgxResourceVK rc = res_image(cv, t->image, t->format, w, h, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    NgxResourceVK ro = res_image(e->out_view, e->out, VK_FORMAT_R16G16B16A16_SFLOAT, ow, oh, VK_IMAGE_ASPECT_COLOR_BIT, 1);
    NgxResourceVK rz = res_image(zv, z->image, z->format, w, h, VK_IMAGE_ASPECT_DEPTH_BIT, 0);
    NgxResourceVK rm = res_image(e->mv_view, e->mv, VK_FORMAT_R16G16_SFLOAT, w, h, VK_IMAGE_ASPECT_COLOR_BIT, 0);
    p_setp(p, "Color", &rc);
    p_setp(p, "Output", &ro);
    p_setp(p, "Depth", &rz);
    p_setp(p, "MotionVectors", &rm);
    {   // DEBUG-JSIGN: REKARINA_DLAA_JSIGN bit 0 flips the x offset given to DLSS, bit 1 the y offset
        static int js = -1;
        if (js < 0) { const char *v = cfg_get("REKARINA_DLAA_JSIGN"); js = v ? atoi(v) : 0; }
        p_setf(p, "Jitter.Offset.X", js & 1 ? -tjx : tjx);
        p_setf(p, "Jitter.Offset.Y", js & 2 ? -tjy : tjy);
    }
    p_setf(p, "Sharpness", 0.0f);
    p_seti(p, "Reset", D.reset);
    p_setp(p, "DLSS.Input.Bias.Current.Color.Mask", NULL);
    p_setf(p, "MV.Scale.X", 1.0f);
    p_setf(p, "MV.Scale.Y", 1.0f);
    p_setu(p, "DLSS.Render.Subrect.Dimensions.Width", w);
    p_setu(p, "DLSS.Render.Subrect.Dimensions.Height", h);
    p_setf(p, "DLSS.Pre.Exposure", 1.0f);
    p_setf(p, "DLSS.Exposure.Scale", 1.0f);
    // The time since the previous evaluation, which DLSS weighs its history by; 60 fps before the first and after a
    // gap (a pause, a load), where no history carries over anyway.
    uint64_t now = plat_now_ns();
    double dt = D.last_eval_ns ? (now - D.last_eval_ns) / 1e6 : 0;
    D.last_eval_ns = now;
    p_setf(p, "FrameTimeDeltaInMsec", dt > 0 && dt < 100 ? (float)dt : 16.67f);
    c0 = timed ? kern_now_ns() : 0;
    NgxResult r = N.evaluate(cb, e->feat, p, NULL);
    if (timed) T.cpu_upscale += kern_now_ns() - c0;
    renderer_mcp_dlss(RENDERER_MCP_NGX, w, h, ow, oh, NGX_OK(r), (uint32_t)r);
    if (!NGX_OK(r)) return dl_fail(evaluation, "DLSS evaluation", r);
evaluated:
    if (projection_attempt && stock_up) D.scene.phase = SCENE_SCALED;
    if (projection_camera) {
        projection_commit(projection, &next);
        double ratio = w ? (double)ow / w : 1.0;
        uint32_t phases = (uint32_t)ceil(8.0 * ratio * ratio - 1e-6);
        if (!spatial) scene_projection_feedback(projection, w, h, phases < 16 ? 16 : phases > 72 ? 72 : phases);
        static unsigned accepted, shifted;
        int applied = !!(projection->token & SCENE_PROJECTION_SHIFTED);
        if ((!applied && accepted++ < 3) || (applied && shifted++ < 6))
            rt_log("dlaa: projection token=%#llx view=%u input=%ux%u raster jitter=(%.5f,%.5f) SDK=(%.5f,%.5f)",
                   (unsigned long long)projection->token, projection->index, w, h, jx, jy, tjx, tjy);
    }
    if (timed) times_mark(cb, T_DL1);   // through the recorder, after NGX's own commands
    D.fg_reset |= D.reset;
    D.fg_menu = D.oot_pass;
    if (vk_dlaa_pathlog())
        rt_log("PATHLOG   DLAA resolved%s, frame %llu evaluation %llu reset %d, scene %ux%u output %ux%u",
               early ? " before the bloom" : "", (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED),
               (unsigned long long)(D.evals + 1), D.reset, w, h, ow, oh);
    D.reset = 0;
    D.last = e, e->used = D.evals + 1;
    D.jw = w, D.jh = h, D.ev_w = w, D.ev_h = h, D.ev_ow = ow, D.ev_oh = oh;
    full_barrier(cb);
    {   // DEBUG-RAW: REKARINA_DLAA_RAW=1 shows the light target itself, scaled bilinearly, in place of DLSS's picture (what
        // the jitter moves and what it leaves); REKARINA_DLAA_PJLOG=<frame>:<frames> logs each frame's offset
        static int raw = -1;
        static uint64_t lf, ln;
        if (raw < 0) {
            raw = rt_knob("DLAA_RAW", "0");
            const char *l = cfg_get("REKARINA_DLAA_PJLOG");
            if (l && sscanf(l, "%llu:%llu", (unsigned long long *)&lf, (unsigned long long *)&ln) != 2) ln = 0;
        }
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (ln && f >= lf && f < lf + ln)
            rt_log("PJLOG frame %llu jitter %.5f %.5f used %d scene %ux%u out %ux%u applied %llu restored %llu searched %llu blocks %d contexts %d; "
                   "per frame: scan %.1f us, apply %.1f us, %.1f slots; DLSS ran at the scaling dispatch %llu times, at a pass %llu",
                   (unsigned long long)f, jx, jy, D.jit_used, w, h, ow, oh, (unsigned long long)SJ.applied,
                   (unsigned long long)SJ.restored, (unsigned long long)SJ.searched, SJ.nbind, SJ.nctx,
                   SJ.timed ? SJ.ns_scan / 1e3 / (double)SJ.timed : 0.0, SJ.timed ? SJ.ns_apply / 1e3 / (double)SJ.timed : 0.0,
                   SJ.timed ? (double)SJ.applied / (double)SJ.timed : 0.0, (unsigned long long)D.at_dispatch, (unsigned long long)D.at_pass);
        if (raw) {
            VkImageBlit rb = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, t->level, t->base_layer, 1},
                              .srcOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}},
                              .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                              .dstOffsets = {{0, 0, 0}, {(int32_t)ow, (int32_t)oh, 1}}};
            vkCmdBlitImage(cb, t->image, VK_IMAGE_LAYOUT_GENERAL, e->out, VK_IMAGE_LAYOUT_GENERAL, 1, &rb, VK_FILTER_LINEAR);
            full_barrier(cb);
        }
    }
    vk_mvcheck_output(cb, e->out, ow, oh);   // REKARINA_MV_CHECK (test)
    if (evaluation != EVALUATE_PREBLOOM) {
        // the result into the scene target (DLAA: the bloom, the post passes and the HUD read it and draw over it) or into
        // the scaling pass's target (the HUD draws over it)
        VkImageBlit b = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                         .srcOffsets = {{0, 0, 0}, {(int32_t)ow, (int32_t)oh, 1}},
                         .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, dst->level, dst->base_layer, 1},
                         .dstOffsets = {{0, 0, 0}, {(int32_t)ow, (int32_t)oh, 1}}};
        // REKARINA_DLAA_MVDEBUG=1 (test): the motion vectors in place of the picture, x in red and y in green, render pixels
        // as scene light (the post passes still run over them)
        static int mvdebug = -1;
        if (mvdebug < 0) mvdebug = rt_knob("DLAA_MVDEBUG", "0");
        VkImage shown = mvdebug ? e->mv : e->out;
        if (shown != e->out) b.srcOffsets[1] = (VkOffset3D){(int32_t)w, (int32_t)h, 1};
        vkCmdBlitImage(cb, shown, VK_IMAGE_LAYOUT_GENERAL, dst->image, VK_IMAGE_LAYOUT_GENERAL, 1, &b, VK_FILTER_NEAREST);
        full_barrier(cb);
        if (dst != t) vk_dlaa_swap_img = t->img, vk_dlaa_swap_view = e->out_view;   // the scaling draw reads DLSS's picture
#ifndef GAME_OOT
        if (hudless_wanted() && !pre_bloom) {   // frame generation keeps the HUD out of its generated frames
            static int at_dlss = -1;
            if (at_dlss < 0) at_dlss = rt_knob("FG_HUDLESS_AT_DLSS", "0");
            if (evaluation == EVALUATE_AT_DISPATCH && !at_dlss) {
                D.scene.hudless.pending = 1, D.scene.hudless.width = ow, D.scene.hudless.height = oh;
                hudless_scene_target(dst != t ? dst->img : NULL, dst != t ? dst->image : VK_NULL_HANDLE);
            } else {
                hudless(cb, e->out, ow, oh);
                D.hl_early++, D.scene.hudless.width = ow, D.scene.hudless.height = oh;
                hudless_log();
            }
        }
    } else if (hudless_wanted()) {
        D.scene.hudless.pending = 1;
        wide_picture(&D.scene.hudless.width, &D.scene.hudless.height);
#endif
    }
    D.pre_bloom.last = pre_bloom;
    if (early) D.pre_bloom.evals++;
    ex->recorded += 4;
    vk_exec_draw_new_cb();   // NGX bound pipelines and descriptor sets of its own: the next draw binds everything again
    D.jdepth = z->image;
    // for frame generation at this frame's present (vk_dlaa_fg_inputs)
    if (!projection_camera) D.projection_history.token = 0;
    if (!spatial || (projection_camera && zv && mv == 1 && D.fg_cam_ok))
        publish_fg(e, z, zv, projection_camera ? 0.0f : jx, projection_camera ? 0.0f : jy);
    next_jitter(w, ow);
#ifdef GAME_OOT
    if (!spatial && !fsr3()) D.nvidia_scene_aa = w == ow && h == oh ? "DLAA" : "DLSS";
#endif
    if (++D.evals == 1 || D.evals == 300 || D.evals % 3600 == 0)   // 300: one early line with the frames counted
        rt_log("dlaa: %llu frames anti-aliased (scene %ux%u, output %ux%u; %llu frames without the DLAA point, %llu with camera "
               "motion, %llu without a camera, %llu jitter holds; resets: %llu size, %llu camera, %llu target; %llu target "
               "moves kept the history; %llu before the bloom)",
               (unsigned long long)D.evals, w, h, ow, oh, (unsigned long long)D.frames_missed,
               (unsigned long long)D.mv_frames, (unsigned long long)D.mv_none,
               (unsigned long long)D.holds, (unsigned long long)D.why[0], (unsigned long long)D.why[1],
               (unsigned long long)D.why[2], (unsigned long long)D.target_moves_kept,
               (unsigned long long)D.pre_bloom.evals);
}

// Called from vk_dlaa_skip_draw on the replay thread before a draw binds its state.
// At the bloom chain's first draw, suspends the pass, resolves DLSS into the scene target, and resumes the pass.
static void pre_bloom_draw(VkExec *ex, const GfxShader *fs) {
    if (!fs || !D.scene.armed) return;
    int is_bloom = !scene_resolved() && !D.pre_bloom.tried && D.projection.sample.token &&
                   pre_bloom_pass(ex) && shader_input_prefix(fs) == OOT_BLOOM_FS;
    if (vk_dlaa_pathlog()) pre_bloom_trace(ex, fs);
    if (!is_bloom || !pre_bloom_reads_scene(ex)) return;

    D.pre_bloom.tried = 1;
    const char *blocked = pre_bloom_blocked(ex);
    static const char *last_reason = "";   // log only when the status/reason changes
    static unsigned reason_logs;
    if (blocked != last_reason && reason_logs++ < 32) {
        last_reason = blocked;
        rt_log("dlaa: pre-bloom %s: %s", blocked ? "waiting" : "armed",
               blocked ? blocked : "temporal AA resolves before the bloom");
    }
    if (blocked) return;

    uint64_t evals_before = D.evals;
    vk_exec_par_join();
    vk_exec_pass_end(ex);
    evaluate(ex, NULL, EVALUATE_NATIVE_PREBLOOM);

    if (D.evals != evals_before) {
        D.pre_bloom.resolved = 1;
        D.scene.phase = SCENE_RESOLVED;
        replay_scene_start();
    } else {
        // Evaluation failed; finish any partial work so the scaling dispatch can resolve this frame instead.
        full_barrier(ex->cb);
        ex->recorded++;
        vk_exec_draw_new_cb();
    }
    ex->dirty = VK_EXEC_DIRTY_ALL;   // rebind viewport, scissor, and descriptor state for the bloom draw
    vk_exec_pass_begin(ex);
}

// The HUD-less picture (vk_dlaa_fg_hudless): the game's picture before its HUD (hudless_at_pass; DLSS's output where
// the HUD follows in the same pass), through the encode the game's last pass gives its picture (linear to sRGB), as
// RGBA8 UNORM bytes for SDR. For PQ presentation keep a linear RGBA16F copy for the host's HUD-less encode.
// Presentation rejects a checkpoint captured against the old swapchain during a mode change.
static void hudless(VkCommandBuffer cb, VkImage src, uint32_t w, uint32_t h) {
    const VkCtx *c = vk_ctx();
    int hdr = vk_hdr_active();
    VkFormat fmt = hdr ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
    if (D.hl && (D.hl_w != w || D.hl_h != h || D.hl_fmt != fmt)) {
        vk_queue_lock(), vkDeviceWaitIdle(c->device), vk_queue_unlock();
        vkDestroyImageView(c->device, D.hl_view, NULL);
        vmaDestroyImage(c->vma, D.hl, D.hl_mem);
        if (D.hl_srgb) vmaDestroyImage(c->vma, D.hl_srgb, D.hl_srgb_mem);
        D.hl = D.hl_srgb = VK_NULL_HANDLE;
    }
    if (!D.hl) {
        VkImageView v2 = VK_NULL_HANDLE;
        VkImageUsageFlags u = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        if ((!hdr && !image2d(w, h, VK_FORMAT_R8G8B8A8_SRGB, u, &D.hl_srgb, &D.hl_srgb_mem, &v2)) ||
            !image2d(w, h, fmt, u | VK_IMAGE_USAGE_STORAGE_BIT, &D.hl, &D.hl_mem, &D.hl_view)) {
            rt_log("dlaa: the HUD-less picture's images could not be made: frame generation sees the HUD");
#ifndef GAME_OOT
            vk_dlaa_hudless_wanted = 0;
#endif
            return;
        }
        if (v2) vkDestroyImageView(c->device, v2, NULL);
        D.hl_w = w, D.hl_h = h, D.hl_fmt = fmt;
        VkImageMemoryBarrier2 ib[2];
        VkImage im[2] = {D.hl, D.hl_srgb};
        int nim = D.hl_srgb ? 2 : 1;
        for (int i = 0; i < nim; i++)
            ib[i] = (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                            .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                            .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                            .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                            .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = im[i],
                                            .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = (uint32_t)nim,
                               .pImageMemoryBarriers = ib};
        vkCmdPipelineBarrier2(cb, &di);
    }
    VkImageBlit b = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}},
                     .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}};
    vkCmdBlitImage(cb, src, VK_IMAGE_LAYOUT_GENERAL, hdr ? D.hl : D.hl_srgb, VK_IMAGE_LAYOUT_GENERAL, 1, &b,
                   VK_FILTER_NEAREST);
    full_barrier(cb);
    D.hl_pq = hdr;
    if (hdr) {
        ++hudless_generation;
        __atomic_add_fetch(&D.hl_fresh, 1, __ATOMIC_ACQ_REL);
        return;
    }
    VkImageCopy cp = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                      .extent = {w, h, 1}};
    vkCmdCopyImage(cb, D.hl_srgb, VK_IMAGE_LAYOUT_GENERAL, D.hl, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
    full_barrier(cb);
    ++hudless_generation;
    __atomic_add_fetch(&D.hl_fresh, 1, __ATOMIC_ACQ_REL);
}

#define INTERFACE_COVERAGE_PRESENT_USAGE (VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | \
                                          VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT)

static struct InterfaceCoverageImage {
    VkImage image;
    VkImageView view;
    VmaAllocation allocation;
    VkExtent2D extent;
    VkFormat format;
    VkImageUsageFlags usage;
} interface_coverage_target, interface_coverage_output, interface_overlay;

int vk_interface_coverage_enabled(void) {
    const CfgRenderValues *settings = cfg_render_values();
    return settings && (settings->consumers & CFG_RENDER_INTERFACE_COVERAGE);
}

static int interface_coverage_current(void) {
    return replay_scene_live() && scene_replay_matches(D.scene.coverage.key, D.scene.replay.key);
}

static int interface_coverage_draw_target(VkImage image) {
    return image && !D.scene.coverage.rejection && !D.scene.coverage.copy && image == D.scene.coverage.target;
}

static int interface_checkpoint_target(const VkExecTarget *target) {
    return replay_scene_live() && D.scene.replay.completed && target->img && !target->level && !target->base_layer &&
        target->image == D.scene.hudless.image && target->img->serial == D.scene.replay.output;
}

int vk_interface_coverage_serial_required(const VkExec *execution) {
    if (D.scene.capture_source && replay_scene_live()) return 1;
    if (!execution->interface_coverage) return 0;
    // Keep checkpoint and presentation-source tracking on the replay thread.
    return D.scene.hudless.pending || D.scene.replay.completed || interface_coverage_current();
}

int vk_interface_coverage_candidate(const VkExec *execution) {
    const VkExecTarget *target = &execution->color[0];
    if (!execution->interface_draw || !execution->interface_coverage || !target->image) return 0;
    if (interface_coverage_current())
        return interface_coverage_draw_target(target->image);
    return interface_checkpoint_target(target);
}

void vk_interface_coverage_reject(const char *reason) {
    if (!interface_coverage_current()) {
        memset(&D.scene.coverage, 0, sizeof D.scene.coverage);
        D.scene.coverage.key = D.scene.replay.key;
    }
    if (!D.scene.coverage.rejection) {
        static unsigned logged;
        if (logged++ < 8) rt_log("interface coverage: rejected frame %llu (%s)", (unsigned long long)D.scene.coverage.key.frame, reason);
        D.scene.coverage.rejection = reason;
    }
}

static int interface_coverage_matches(VkImage image, uint64_t address) {
    return (image && (image == D.scene.coverage.target || image == D.scene.coverage.copy)) ||
           (address && (address == D.scene.coverage.target_address || address == D.scene.coverage.copy_address));
}

static void interface_coverage_invalidate(VkImage image, uint64_t address) {
    if (D.scene.clean_output && (image == D.scene.clean_output || (address && address == D.scene.clean_output_address)))
        D.scene.clean_output = VK_NULL_HANDLE;
    if (interface_coverage_current() && interface_coverage_matches(image, address)) vk_interface_coverage_reject("tracked image overwritten");
    if (D.scene.capture_copy && (image == D.scene.capture_copy || (address && address == D.scene.capture_copy_address)))
        D.scene.capture_copy = VK_NULL_HANDLE;
}

void vk_interface_coverage_write(VkImage image, uint64_t address) {
#ifdef GAME_OOT
    if (!vk_par_worker) vk_hdr_output_overwrite(image, address);
#endif
    interface_coverage_invalidate(image, address);
}

void vk_interface_coverage_check_draw(VkExec *execution) {
    if (!D.scene.clean_output && !D.scene.capture_copy &&
        (!interface_coverage_current() || !D.scene.coverage.target || D.scene.coverage.rejection)) return;
    vk_exec_resolve_targets(execution);
    for (unsigned attachment = 0; attachment < execution->pass_ncolor; ++attachment) {
        const VkExecTarget *target = &execution->color[attachment];
        if (execution->interface_draw && attachment == 0 && interface_coverage_draw_target(target->image)) continue;
        interface_coverage_invalidate(target->image, target->img ? target->img->addr : 0);
    }
}

int vk_interface_coverage_source(VkImage image) {
    return (interface_coverage_current() && interface_coverage_draw_target(image)) ||
        (image && image == D.scene.capture_source && replay_scene_live() &&
         scene_replay_matches(D.scene.capture_key, D.scene.replay.key));
}

void vk_interface_coverage_copy(VkImage source, VkImage destination, uint64_t address, int full_copy) {
    vk_interface_coverage_write(destination, address);
    if (full_copy && source && source == D.scene.capture_source && !D.scene.capture_copy) {
        D.scene.capture_copy = destination;
        D.scene.capture_copy_address = address;
    }
    if (!interface_coverage_current() || !D.scene.coverage.target || D.scene.coverage.rejection) return;
    if (!D.scene.coverage.rejection && source == D.scene.coverage.target && full_copy && !D.scene.coverage.copy) {
        D.scene.coverage.copy = destination;
        D.scene.coverage.copy_address = address;
    }
}

static void interface_coverage_destroy(struct InterfaceCoverageImage *image) {
    const VkCtx *context = vk_ctx();
    if (image->view) vkDestroyImageView(context->device, image->view, NULL);
    if (image->image) vmaDestroyImage(context->vma, image->image, image->allocation);
    *image = (struct InterfaceCoverageImage){0};
}

static void ui_extract_destroy(void);

void vk_interface_coverage_release(void) {
    ui_extract_destroy();
    interface_coverage_destroy(&interface_coverage_target);
    interface_coverage_destroy(&interface_coverage_output);
    interface_coverage_destroy(&interface_overlay);
    memset(&D.scene.coverage, 0, sizeof D.scene.coverage);
}

static int interface_coverage_allocate(struct InterfaceCoverageImage *destination, VkExtent2D extent, VkFormat format,
                                       VkImageUsageFlags usage, VkFormatFeatureFlags features) {
    const VkCtx *context = vk_ctx();
    if (destination->view && destination->extent.width == extent.width && destination->extent.height == extent.height &&
        destination->format == format && (destination->usage & usage) == usage) return 1;
    VkFormatProperties properties;
    vkGetPhysicalDeviceFormatProperties(context->phys, format, &properties);
    if ((properties.optimalTilingFeatures & features) != features) return 0;
    if (destination->image) {
        if (vk_exec_recording_fence()) {
            vk_defer_view(destination->view);
            vk_defer_image(destination->image, destination->allocation);
            *destination = (struct InterfaceCoverageImage){0};
        } else {
            vk_queue_lock(), vkDeviceWaitIdle(context->device), vk_queue_unlock();
            interface_coverage_destroy(destination);
        }
    }
    if (!image2d(extent.width, extent.height, format, usage,
                 &destination->image, &destination->allocation, &destination->view)) {
        if (destination->image) vmaDestroyImage(context->vma, destination->image, destination->allocation);
        *destination = (struct InterfaceCoverageImage){0};
        return 0;
    }
    destination->extent = extent;
    destination->format = format;
    destination->usage = usage;
    return 1;
}

static void interface_coverage_discard(VkCommandBuffer command_buffer, VkImage image,
                                       VkPipelineStageFlags2 stage, VkAccessFlags2 access) {
    VkImageMemoryBarrier2 discard = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = stage, .dstAccessMask = access,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL, .image = image,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &discard};
    vkCmdPipelineBarrier2(command_buffer, &dependency);
}

VkImageView vk_interface_coverage_overlay(VkCommandBuffer cb, uint64_t frame, VkExtent2D extent, VkFormat format) {
    if (!sealed_scene.coverage || sealed_scene.replay.key.frame != frame) return VK_NULL_HANDLE;
    // Stock ImGui's pipeline must see its usual color format, with full precision alpha.
    if (format != VK_FORMAT_R8G8B8A8_UNORM && format != VK_FORMAT_B8G8R8A8_UNORM &&
        format != VK_FORMAT_R8G8B8A8_SRGB && format != VK_FORMAT_B8G8R8A8_SRGB) return VK_NULL_HANDLE;
    if (!interface_coverage_allocate(&interface_overlay, extent, format,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) return VK_NULL_HANDLE;
    interface_coverage_discard(cb, interface_overlay.image, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                               VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    return interface_overlay.view;
}

static void interface_coverage_clear(VkCommandBuffer command_buffer, VkImage image) {
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VkImageMemoryBarrier2 transition = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL, .image = image,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .subresourceRange = range};
    VkDependencyInfo dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &transition};
    vkCmdPipelineBarrier2(command_buffer, &dependency);
    VkClearColorValue transparent = {0};
    vkCmdClearColorImage(command_buffer, image, VK_IMAGE_LAYOUT_GENERAL, &transparent, 1, &range);
    transition.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    transition.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    transition.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    if (image == interface_coverage_target.image) {
        transition.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        transition.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    }
    vkCmdPipelineBarrier2(command_buffer, &dependency);
}

static int capture_hudless_checkpoint(VkCommandBuffer cb, VkImage source, uint32_t w, uint32_t h) {
    uint64_t previous_generation = hudless_generation;
    hudless(cb, source, w, h);
    if (hudless_generation == previous_generation || !D.hl_view) return 0;
    if (replay_scene_live() && D.scene.replay.completed && D.scene.hudless.texture &&
        D.scene.hudless.texture->serial == D.scene.replay.output && source == D.scene.hudless.image) {
        D.scene.capture_key = D.scene.replay.key;
        D.scene.capture_source = source;
        D.scene.capture_hdr = vk_hdr_active();
    }
    return 1;
}

static int interface_checkpoint_capture(VkCommandBuffer cb, const VkTexImage *target, uint32_t w, uint32_t h) {
    if (!target || !replay_scene_live() || !(target->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) ||
        (target->format != VK_FORMAT_B10G11R11_UFLOAT_PACK32 && target->format != VK_FORMAT_R16G16B16A16_SFLOAT) ||
        (interface_coverage_current() && D.scene.coverage.rejection)) return 0;
    if (!interface_coverage_allocate(&interface_coverage_target, (VkExtent2D){w, h}, VK_FORMAT_R8G8B8A8_UNORM,
            VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                VK_IMAGE_USAGE_SAMPLED_BIT,
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT | VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT))
        return 0;
    if (!capture_hudless_checkpoint(cb, target->image, w, h)) return 0;

    interface_coverage_clear(cb, interface_coverage_target.image);
    memset(&D.scene.coverage, 0, sizeof D.scene.coverage);
    D.scene.coverage.key = D.scene.capture_key;
    D.scene.coverage.target = target->image;
    D.scene.coverage.target_address = target->addr;
    D.scene.coverage.hudless_generation = hudless_generation;
    D.scene.coverage.hdr = D.scene.capture_hdr;
    static unsigned logged;
    if (logged++ < 8) rt_log("interface coverage: captured frame %llu at %ux%u", (unsigned long long)D.scene.coverage.key.frame, w, h);
    return 1;
}

VkImageView vk_interface_coverage_begin(VkExec *execution) {
    const VkExecTarget *target = &execution->color[0];
    if (interface_coverage_current())
        return interface_coverage_draw_target(target->image) ? interface_coverage_target.view : VK_NULL_HANDLE;
    if (!execution->interface_coverage || !interface_checkpoint_target(target) ||
        target->width != D.scene.hudless.width || target->height != D.scene.hudless.height) return VK_NULL_HANDLE;
    vk_exec_pass_end(execution);
    VkPassMark *mark = vk_pass_mark_begin(execution->cb, VK_MARK_UI_CAPTURE, execution->fence);
    vk_exec_barrier(execution);
    int captured = interface_checkpoint_capture(execution->cb, target->img, target->width, target->height);
    vk_pass_mark_end(execution->cb, mark);
    if (!captured) return VK_NULL_HANDLE;
    D.scene.hudless.pending = 0;
    __atomic_store_n(&D.hl_fresh, 1, __ATOMIC_RELEASE);
    execution->recorded += 5;
    return interface_coverage_target.view;
}

#include "ui_replay_extract_spv.h"

static struct {
    VkDescriptorSetLayout descriptors;
    VkPipelineLayout layout;
    VkPipeline pipeline;
    VkSampler sampler;
    PFN_vkCmdPushDescriptorSetKHR push;
} ui_extract;

static void ui_extract_destroy(void) {
    VkDevice device = vk_ctx()->device;
    vkDestroyPipeline(device, ui_extract.pipeline, NULL);
    vkDestroyPipelineLayout(device, ui_extract.layout, NULL);
    vkDestroyDescriptorSetLayout(device, ui_extract.descriptors, NULL);
    vkDestroySampler(device, ui_extract.sampler, NULL);
    memset(&ui_extract, 0, sizeof ui_extract);
}

static int ui_extract_init(void) {
    if (ui_extract.pipeline) return 1;
    VkDevice device = vk_ctx()->device;
    ui_extract.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR");
    VkDescriptorSetLayoutBinding bindings[] = {
        {0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL},
        {2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, NULL}};
    VkDescriptorSetLayoutCreateInfo descriptors = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, .bindingCount = 3, .pBindings = bindings};
    VkPushConstantRange range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, 52};
    VkPipelineLayoutCreateInfo layout = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &ui_extract.descriptors, .pushConstantRangeCount = 1, .pPushConstantRanges = &range};
    VkSamplerCreateInfo sampler = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
        .magFilter = VK_FILTER_LINEAR, .minFilter = VK_FILTER_LINEAR,
        .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE};
    VkShaderModuleCreateInfo module = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof ui_replay_extract_spv, .pCode = ui_replay_extract_spv};
    VkShaderModule shader = VK_NULL_HANDLE;
    if (!ui_extract.push || vkCreateDescriptorSetLayout(device, &descriptors, NULL, &ui_extract.descriptors) != VK_SUCCESS ||
        vkCreatePipelineLayout(device, &layout, NULL, &ui_extract.layout) != VK_SUCCESS ||
        vkCreateSampler(device, &sampler, NULL, &ui_extract.sampler) != VK_SUCCESS ||
        vkCreateShaderModule(device, &module, NULL, &shader) != VK_SUCCESS) {
        ui_extract_destroy();
        return 0;
    }
    VkComputePipelineCreateInfo pipeline = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shader, .pName = "main"}, .layout = ui_extract.layout};
    VkResult result = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline, NULL, &ui_extract.pipeline);
    vkDestroyShaderModule(device, shader, NULL);
    if (result != VK_SUCCESS) ui_extract_destroy();
    return result == VK_SUCCESS;
}

static int ui_extract_fit(VkCommandBuffer cb, VkExtent2D extent, const int crop[4], const int destination[4],
                          VkImageView overlay, const int preview[4], int empty) {
    if (!ui_extract_init()) return 0;
    VkDescriptorImageInfo images[] = {
        {ui_extract.sampler, interface_coverage_target.view, VK_IMAGE_LAYOUT_GENERAL},
        {VK_NULL_HANDLE, interface_coverage_output.view, VK_IMAGE_LAYOUT_GENERAL},
        {ui_extract.sampler, overlay ? overlay : interface_coverage_target.view, VK_IMAGE_LAYOUT_GENERAL}};
    VkWriteDescriptorSet writes[] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &images[0]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &images[1]},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 2, .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &images[2]}};
    int32_t fit[13] = {0};
    memcpy(fit, crop, 16);
    memcpy(fit + 4, destination, 16);
    if (preview) memcpy(fit + 8, preview, 16);
    fit[12] = (!empty ? 1 : 0) | (overlay ? 2 : 0) | (preview ? 4 : 0);
    full_barrier(cb);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, ui_extract.pipeline);
    ui_extract.push(cb, VK_PIPELINE_BIND_POINT_COMPUTE, ui_extract.layout, 0, 3, writes);
    vkCmdPushConstants(cb, ui_extract.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof fit, fit);
    vkCmdDispatch(cb, (extent.width + 7) / 8, (extent.height + 7) / 8, 1);
    return 1;
}

int vk_interface_coverage_present(VkCommandBuffer command_buffer, uint64_t frame, VkImage source, VkExtent2D source_extent,
                                  VkExtent2D presentation_extent, const int rectangle[4], const int destination[4],
                                  VkImageView overlay, const int preview[4], VkImage *image, VkImageView *view) {
    if (!sealed_scene.coverage || sealed_scene.replay.key.frame != frame) return 0;
    int empty = sealed_scene.coverage == SCENE_UI_ALPHA_EMPTY;
    int frame_matches = D.scene.coverage.key.frame == frame;
    int source_matches = source && (source == D.scene.coverage.target || source == D.scene.coverage.copy);
    int checkpoint_matches = D.scene.coverage.hudless_generation == hudless_generation && D.scene.coverage.hdr == vk_hdr_active();
    if (!empty && (!D.scene.coverage.target || !frame_matches || !source_matches || !checkpoint_matches ||
        source_extent.width != interface_coverage_target.extent.width || source_extent.height != interface_coverage_target.extent.height)) {
        static unsigned logged;
        if (logged++ < 8) rt_log("interface coverage: presentation rejected (frame %llu/%llu, source match=%d, checkpoint match=%d, "
                                 "extent %ux%u/%ux%u)", (unsigned long long)frame, (unsigned long long)D.scene.coverage.key.frame,
                                 source_matches, checkpoint_matches, source_extent.width, source_extent.height,
                                 interface_coverage_target.extent.width, interface_coverage_target.extent.height);
        return 0;
    }
    // Keep the fitted image storage-capable across UI-free frames.
    if (!interface_coverage_allocate(&interface_coverage_output, presentation_extent, VK_FORMAT_R16_SFLOAT,
            INTERFACE_COVERAGE_PRESENT_USAGE | VK_IMAGE_USAGE_STORAGE_BIT,
            VK_FORMAT_FEATURE_BLIT_DST_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT |
            VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT | VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT |
            VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT)) return 0;
    int clear = empty && !overlay;
    if (clear) {
        interface_coverage_clear(command_buffer, interface_coverage_output.image);
    } else {
        interface_coverage_discard(command_buffer, interface_coverage_output.image,
                                   VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        if (!ui_extract_fit(command_buffer, presentation_extent, rectangle, destination, overlay, preview, empty)) return 0;
    }
    VkImageMemoryBarrier2 fitted = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = clear ? VK_PIPELINE_STAGE_2_TRANSFER_BIT : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = clear ? VK_ACCESS_2_TRANSFER_WRITE_BIT : VK_ACCESS_2_SHADER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_GENERAL, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = interface_coverage_output.image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo ready = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &fitted};
    vkCmdPipelineBarrier2(command_buffer, &ready);
    *image = interface_coverage_output.image;
    *view = interface_coverage_output.view;
    static unsigned overlay_logged;
    if (overlay && overlay_logged++ < 8) rt_log("interface coverage: host overlay included via replay frame %llu", (unsigned long long)frame);
    static unsigned logged[2];
    if (logged[empty]++ < 8) rt_log("interface coverage: present frame %llu at %ux%u%s", (unsigned long long)frame,
                                    presentation_extent.width, presentation_extent.height, empty ? " (no game UI)" : "");
    return 1;
}

static int scene_color(VkFormat f) {
    return f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || f == VK_FORMAT_R16G16B16A16_SFLOAT;
}

// Two frame shapes. A deferred frame (the G-buffer, several colour targets at the scene size) arms on the light target
// drawn after it and ends the scene at the first pass a quarter as wide (the bloom), as described above. OoT's
// frame has no G-buffer: its light target (B10G11R11 at the scene size, with the D32F scene depth) is drawn into from the
// first opaque pass on, half- and quarter-size effect passes come and go in between, and the scene ends at the pass
// that tonemaps the light target into the 8-bit output target. So without a G-buffer the frame arms on the first
// scene-size light pass and ends at that output pass: at the output size and without depth when DLSS scales up
// (DLSS writes the output and the tonemap samples it through vk_dlaa_swap_img), an sRGB target of the scene size in
// DLAA (DLSS's picture goes back into the light target, which the tonemap then reads).
static int srgb8(VkFormat f) { return f == VK_FORMAT_R8G8B8A8_SRGB || f == VK_FORMAT_B8G8R8A8_SRGB; }

// The output target DLAA runs at. With the game's own HDR (vk_hdr.c vk_hdr_native) the same pass, at the same point of
// the frame, tonemaps into its PQ picture (vk_hdr.c vk_hdr_native_picture) instead of the 8-bit sRGB one; matching sRGB
// alone left DLAA off for the whole run there (no frame anti-aliased) and with it the frame generation that takes
// DLAA's depth and motion.
static int output_target(VkFormat f) { return srgb8(f) || vk_hdr_native_picture(f); }

// DEBUG-PATHLOG: REKARINA_DLAA_PATHLOG=<frame>:<frames>[,<frame>:<frames>]: the passes of those frames from the scene's
// arming on (targets, and where DLSS ran), and what the draws of passes without depth sample (vk_exec_draw.c tex_info)
int vk_dlaa_pathlog(void) {
    static int state = -1;
    static uint64_t lo[4], n[4];
    static int nw;
    if (state < 0) {
        const char *e = cfg_get("REKARINA_DLAA_PATHLOG");
        state = 0;
        for (const char *p = e; p && *p && nw < 4;) {
            char *end;
            lo[nw] = strtoull(p, &end, 10), n[nw] = *end == ':' ? strtoull(end + 1, &end, 10) : 1;
            nw++, state = 1;
            p = *end == ',' ? end + 1 : "";
        }
    }
    if (!state) return 0;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    for (int i = 0; i < nw; i++)
        if (f >= lo[i] && f < lo[i] + n[i]) return 1;
    return 0;
}

// Frame generation's picture without the HUD is the game's finished picture right before its HUD: frame generation
// tells the HUD apart by what differs between it and the presented frame, and keeps that out of the frames it makes.
// The copy at DLSS's evaluation was taken before passes that still change the picture (OoT in DLAA: the game's own
// anti-aliasing, three dispatches, and its scaler's dispatch run after DLSS, measured in a frame dump), so the
// presented frame differed from it over the whole scene and generation took parts of the scene for interface. So the
// gameplay scaler publishes its output after RCAS (D.scene.hudless.pending). The copy is taken at the first render
// pass that draws into that picture, or at the window copy in a frame without HUD. A frame that reaches neither
// has no HUD-less snapshot.
static void hudless_track(VkExec *ex, const GfxShader *cs) {
    const GfxShaderRes *r;
    int n = gfx_shader_resources(cs, GFX_RES_IMG, &r);
    vk_texture_lock();
    for (int i = 0; i < n; i++) {
        if (r[i].slot < 0 || r[i].slot >= 32) continue;
        uint64_t h = ex->img[5][r[i].slot];   // the compute stage's images (NVN_STAGE_CS)
        const VkTexImage *img = h ? vk_texture_by_id_locked((int)(h & 0xfffff), NULL, NULL) : NULL;
        // the picture is float (B10G11R11 or RGBA16F, as the scene); the game's anti-aliasing also writes its blend
        // weights into an RGBA8 image of the output's size (black with coloured edges), which taken for the picture
        // gave frame generation green and yellow slabs and black areas with dots (owner, FSR 3 FG, 2026-11-20)
        if (img && (img->aspect & VK_IMAGE_ASPECT_COLOR_BIT) && img->width == D.scene.hudless.width && img->height == D.scene.hudless.height &&
            img->image && scene_color(img->format))
            hudless_scene_target(img, img->image);
    }
    vk_texture_unlock();
}

// The HUD-less copies so far, in the log at the first and every 1800th.
static void hudless_log(void) {
    uint64_t n = D.hl_taken + D.hl_window + D.hl_early;
    if (n != 1 && n % 1800) return;
    rt_log("dlaa: frame generation's HUD-less picture %ux%u: %llu taken before the HUD's pass, %llu at the window copy, "
           "%llu at DLSS, %llu frames without one; presents that found a later frame's copy already recorded %llu, "
           "its depth and motion %llu", D.scene.hudless.width, D.scene.hudless.height, (unsigned long long)D.hl_taken,
           (unsigned long long)D.hl_window, (unsigned long long)D.hl_early, (unsigned long long)D.hl_missed,
           (unsigned long long)D.hl_ahead, (unsigned long long)D.fg_ahead);
}

static void hudless_at_pass(VkExec *ex) {
    const VkExecTarget *c = &ex->color[0];
    if (ex->pass_ncolor < 1 || !c->image) return;
    int pic = D.scene.hudless.texture && c->img && (c->img == D.scene.hudless.texture || (D.scene.hudless.address && c->img->addr == D.scene.hudless.address)) &&
              c->width == D.scene.hudless.width && c->height == D.scene.hudless.height;
    int window = !pic && (!scene_prebloom() || D.scene.hudless.texture) && ex->pass_ncolor == 1 && !ex->depth.image &&
                 output_target(c->format) && c->width == D.scene.hudless.width && c->height == D.scene.hudless.height;
    if (!pic && !window) return;
    D.scene.hudless.pending = 0;
    VkImage src = pic ? c->image : D.scene.hudless.image;
#ifndef GAME_OOT
    if (!src && !scene_prebloom() && D.last) src = D.last->out;
#endif
    if (!src) return;
    vk_exec_par_join();
    vk_exec_barrier(ex);   // the dispatches that wrote the picture are done (no pass is open here)
    VkCommandBuffer cb = ex->cb;
    full_barrier(cb);
    int paired = ex->interface_coverage && (pic ? ex->interface_draw && interface_checkpoint_target(c) :
        replay_scene_live() && src == D.scene.clean_output);
    if (paired) {
        const VkTexImage *target = pic ? c->img : D.scene.hudless.texture;
        if (!interface_checkpoint_capture(cb, target, D.scene.hudless.width, D.scene.hudless.height)) return;
        ex->recorded += 3;
    } else {
        if (!capture_hudless_checkpoint(cb, src, D.scene.hudless.width, D.scene.hudless.height)) return;
    }
    ex->recorded += 2;
    if (pic) D.hl_taken++;
    else D.hl_window++;
    if (vk_dlaa_pathlog())
        rt_log("PATHLOG   HUD-less picture copied at %s pass %u, %ux%u format %d",
               pic ? "HUD" : "window", ex->pass_serial + 1, D.scene.hudless.width, D.scene.hudless.height, (int)D.hl_fmt);
    hudless_log();
}

static int prebloom_source(uint64_t handle) {
    if (!handle || !D.scene.color.img) return 0;
    view_t view = {0};
    int explicit_view = 0;
    const VkTexImage *texture = vk_texture_by_id_locked((int)(handle & 0xfffff), &view, &explicit_view);
    if (texture != D.scene.color.img || texture->image != D.scene.color.image) return 0;
    if (explicit_view && view.set_format) {
        TexVkFormat format;
        if (!tex_vk_format(view.format, &format) || format.vk != D.scene.color.format) return 0;
    }
    return D.scene.color.level == (explicit_view && view.set_levels ? view.base_level : 0) &&
           D.scene.color.base_layer == (explicit_view && view.set_layers ? view.min_layer : 0);
}

int vk_dlaa_prebloom_active(void) { return scene_prebloom(); }

static int bloom_guest_index(uint32_t serial, uint32_t level, uint32_t layer) {
    for (unsigned index = 0; index < PRIVATE_BLOOM_TARGET_COUNT; ++index) {
        const struct BloomGuestSubresource *guest = &D.scene.bloom_guest_subresources[index];
        if (guest->image_serial == serial && guest->level == level && guest->layer == layer) return (int)index;
    }
    return -1;
}

VkImageView vk_dlaa_prebloom_view(const VkExec *execution, uint64_t handle) {
    if (!scene_prebloom() || !handle) return VK_NULL_HANDLE;
    if (execution->draw_role == VK_BLOOM_EXTRACT || execution->draw_role == VK_BLOOM_TONEMAP) {
        if (prebloom_source(handle)) return D.last->out_view;
    }
    view_t view = {0};
    int explicit_view = 0;
    const VkTexImage *texture = vk_texture_by_id_locked((int)(handle & 0xfffff), &view, &explicit_view);
    if (!texture || texture->format != VK_FORMAT_B10G11R11_UFLOAT_PACK32 ||
        (explicit_view && (view.set_swizzle || (view.set_target && view.target != NVN_TEXTURE_TARGET_2D)))) return VK_NULL_HANDLE;
    if (explicit_view && view.set_format) {
        TexVkFormat format;
        if (!tex_vk_format(view.format, &format) || format.vk != texture->format) return VK_NULL_HANDLE;
    }
    int index = bloom_guest_index(texture->serial, view.set_levels ? view.base_level : 0, view.set_layers ? view.min_layer : 0);
    if (index < 0) return VK_NULL_HANDLE;
    const struct PrivateBloomTarget *target = &D.last->bloom_targets[index];
    for (unsigned attachment = 0; attachment < execution->pass_ncolor; ++attachment)
        if (execution->color[attachment].image == target->image) return VK_NULL_HANDLE;
    return target->view;
}

int vk_dlaa_scene_sample(const VkExec *execution, unsigned role, uint64_t handle) {
    if (!handle) return 0;
    vk_texture_lock();
    int match = prebloom_source(handle);
    if (!match && replay_scene_live() && scene_prebloom() && D.scene.replay.bloom_written && role == VK_BLOOM_TONEMAP) {
        VkExec sample = {.draw_role = role};
        match = vk_dlaa_prebloom_view(&sample, handle) != VK_NULL_HANDLE;
    }
    vk_texture_unlock();
    return match;
}

void vk_dlaa_prebloom(VkExec *execution, uint64_t handle, unsigned role) {
    if (!D.scene.armed || D.projection_conflict || !D.projection.sample.token ||
        !projection_target(&D.scene.color, D.projection.color_serial, &D.projection.cv) ||
        !vk_dlaa_scene_sample(execution, role, handle)) return;
    replay_scene_start();
    int bloom = role == VK_BLOOM_EXTRACT;
    SceneReplayRoute route = scene_replay_early_route(bloom, D.scene.replay.bloom_seen);
    D.scene.replay.bloom_seen |= bloom;
    if (route == SCENE_ROUTE_ORIGINAL || D.scene.early_attempted || !dl_active() || scene_resolved() || deferred || fsr1() ||
        !scene_projection_enabled()) return;
    VkExecTarget extent = {0};
    wide_picture(&extent.width, &extent.height);
    if (extent.width < D.scene.color.width || extent.height < D.scene.color.height) return;
    D.scene.early_attempted = 1;
    vk_exec_pass_end(execution);
    uint64_t evaluations_before = D.evals;
    evaluate(execution, &extent, EVALUATE_PREBLOOM);
    if (D.evals == evaluations_before) {
        vk_exec_draw_new_cb();
        return;
    }
    D.scene.phase = SCENE_PREBLOOM;
    D.scene.replay.route = route;
    D.scene.replay.uw = extent.width;
    D.scene.replay.uh = extent.height;
    // The guest scaling draw still owns presentation until its destination has been written.
    static int first;
    if (!first++ || vk_dlaa_pathlog())
        rt_log("dlaa: pre-bloom %s %ux%u -> %ux%u", fsr3() ? "AMD FSR" : "DLSS",
               D.scene.color.width, D.scene.color.height, extent.width, extent.height);
}

static void bloom_copy(VkExec *execution, const VkExecTarget *source, const VkExecTarget *destination) {
    VkImageBlit region = {
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, source->level, source->base_layer, 1},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, destination->level, destination->base_layer, 1},
        .srcOffsets = {{0, 0, 0}, {(int32_t)source->width, (int32_t)source->height, 1}},
        .dstOffsets = {{0, 0, 0}, {(int32_t)destination->width, (int32_t)destination->height, 1}}};
    vkCmdBlitImage(execution->cb, source->image, VK_IMAGE_LAYOUT_GENERAL, destination->image, VK_IMAGE_LAYOUT_GENERAL,
                   1, &region, source->width == destination->width && source->height == destination->height ?
                   VK_FILTER_NEAREST : VK_FILTER_LINEAR);
    execution->recorded++;
}

static int private_bloom_target(VkExec *execution, unsigned index, uint32_t width, uint32_t height) {
    struct PrivateBloomTarget *target = &D.last->bloom_targets[index];
    if (target->image) return target->width == width && target->height == height;
    const VkImageUsageFlags usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (!image2d(width, height, VK_FORMAT_B10G11R11_UFLOAT_PACK32, usage,
                 &target->image, &target->allocation, &target->view)) {
        if (target->image) vmaDestroyImage(vk_ctx()->vma, target->image, target->allocation);
        *target = (struct PrivateBloomTarget){0};
        return 0;
    }
    target->width = width, target->height = height;
    VkImageMemoryBarrier2 barrier = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = target->image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier};
    vkCmdPipelineBarrier2(execution->cb, &dependency);
    execution->recorded++;
    return 1;
}

int vk_dlaa_bloom_targets(VkExec *execution, unsigned role, unsigned level, const VkExecTarget *guest_targets) {
    int target_indices[8];
    uint32_t reserved = 0;
    uint32_t guest_width = execution->pass_w, guest_height = execution->pass_h;
    if (level < 2 || level > 6 || !execution->pass_ncolor) return 0;
    uint32_t width = D.last->ow >> level, height = D.last->oh >> level;
    if (!width || !height) return 0;
    if (role == VK_BLOOM_PYRAMID) {
        // The scheduler also records unrelated work. Only follow dependencies from a known bloom draw.
        int follows_bloom = 0;
        vk_texture_lock();
        for (unsigned slot = 0; slot < 128 && !follows_bloom; ++slot)
            follows_bloom = vk_dlaa_prebloom_view(execution, execution->tex[NVN_SHADER_STAGE_FRAGMENT][slot]) != VK_NULL_HANDLE;
        vk_texture_unlock();
        for (unsigned index = 0; index < execution->pass_ncolor && !follows_bloom; ++index) {
            const VkExecTarget *guest = &guest_targets[index];
            follows_bloom = guest->img && bloom_guest_index(guest->img->serial, guest->level, guest->base_layer) >= 0;
        }
        if (!follows_bloom) return 0;
    }
    for (unsigned index = 0; index < execution->pass_ncolor; ++index) {
        const VkExecTarget *guest = &guest_targets[index];
        target_indices[index] = -1;
        if (!guest->image) continue;
        if (!guest->img || guest->format != VK_FORMAT_B10G11R11_UFLOAT_PACK32 || guest->img->format != guest->format ||
            guest->samples != 1 || guest->layers != 1 || guest->is3d || guest->blocks || guest->scale != 1 ||
            guest->width != guest_width || guest->height != guest_height || guest->image == D.scene.color.image) return 0;
        int target_index = bloom_guest_index(guest->img->serial, guest->level, guest->base_layer);
        if (target_index < 0) {
            for (unsigned candidate = 0; candidate < PRIVATE_BLOOM_TARGET_COUNT; ++candidate) {
                if (D.scene.bloom_guest_subresources[candidate].image_serial || (reserved & (1u << candidate))) continue;
                const struct PrivateBloomTarget *target = &D.last->bloom_targets[candidate];
                if (!target->image || (target->width == width && target->height == height)) {
                    target_index = (int)candidate;
                    break;
                }
            }
        }
        if (target_index < 0 || (reserved & (1u << target_index)) ||
            !private_bloom_target(execution, (unsigned)target_index, width, height)) return 0;
        target_indices[index] = target_index;
        reserved |= 1u << target_index;
    }
    if (!reserved) return 0;
    vk_exec_barrier(execution);
    full_barrier(execution->cb);
    for (unsigned index = 0; index < execution->pass_ncolor; ++index) {
        int target_index = target_indices[index];
        if (target_index < 0) continue;
        const struct PrivateBloomTarget *target = &D.last->bloom_targets[target_index];
        VkExecTarget *attachment = &execution->color[index];
        attachment->image = target->image;
        attachment->private_attachment_view = target->view;
        attachment->img = NULL;
        attachment->level = attachment->base_layer = 0;
        attachment->width = width, attachment->height = height;
        // Seed a new private subresource with the guest's clear/load contents before blending into it.
        if (!D.scene.bloom_guest_subresources[target_index].image_serial)
            bloom_copy(execution, &guest_targets[index], attachment);
    }
    full_barrier(execution->cb);
    return 1;
}

void vk_dlaa_bloom_finish(VkExec *execution, const VkExecTarget *guest_targets, int drew) {
    vk_exec_barrier(execution);
    full_barrier(execution->cb);
    for (unsigned index = 0; index < execution->pass_ncolor; ++index) {
        const VkExecTarget *guest = &guest_targets[index];
        const VkExecTarget *attachment = &execution->color[index];
        if (!attachment->private_attachment_view) continue;
        for (unsigned target_index = 0; target_index < PRIVATE_BLOOM_TARGET_COUNT; ++target_index) {
            if (D.last->bloom_targets[target_index].image != attachment->image) continue;
            if (drew) {
                D.scene.bloom_guest_subresources[target_index] = (struct BloomGuestSubresource){guest->img->serial, guest->level, guest->base_layer};
                // Unclassified reads still see the guest image, without changing its allocation or dimensions.
                bloom_copy(execution, attachment, guest);
            } else {
                D.scene.bloom_guest_subresources[target_index] = (struct BloomGuestSubresource){0};
            }
            break;
        }
    }
    full_barrier(execution->cb);
}

static int prebloom_blit_supported(const struct Size *reconstruction, VkFormat source_format, const VkExecTarget *destination) {
    if (!reconstruction || !destination->image || !destination->img || destination->image == reconstruction->tonemap ||
        !scene_color(source_format) || destination->format != source_format || destination->img->format != source_format ||
        destination->view_type != VK_IMAGE_VIEW_TYPE_2D || destination->is3d || destination->samples != 1 ||
        destination->layers != 1 || destination->level || destination->base_layer ||
        !destination->width || !destination->height || !(destination->img->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) return 0;
    VkFormatProperties properties;
    vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, source_format, &properties);
    VkFormatFeatureFlags required = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
    if (reconstruction->ow != destination->width || reconstruction->oh != destination->height)
        required |= VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
    return (properties.optimalTilingFeatures & required) == required;
}

int vk_dlaa_prebloom_tonemap(VkExec *execution, VkExecTarget *target) {
    struct Size *reconstruction = D.last;
    if (!replay_scene_live() || !scene_prebloom() || !prebloom_blit_supported(reconstruction, target->format, target) ||
        (D.scene.replay.bloom_seen && !D.scene.replay.bloom_written)) return 0;
    if (reconstruction->tonemap && reconstruction->tonemap_format != target->format) {
        vk_defer_view(reconstruction->tonemap_view);
        vk_defer_image(reconstruction->tonemap, reconstruction->tonemap_mem);
        reconstruction->tonemap = VK_NULL_HANDLE;
        reconstruction->tonemap_view = VK_NULL_HANDLE;
        reconstruction->tonemap_mem = NULL;
    }
    if (!reconstruction->tonemap) {
        if (!image2d(reconstruction->ow, reconstruction->oh, target->format, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                     &reconstruction->tonemap, &reconstruction->tonemap_mem, &reconstruction->tonemap_view)) {
            if (reconstruction->tonemap) vmaDestroyImage(vk_ctx()->vma, reconstruction->tonemap, reconstruction->tonemap_mem);
            reconstruction->tonemap = VK_NULL_HANDLE;
            return 0;
        }
        reconstruction->tonemap_format = target->format;
        VkImageMemoryBarrier2 barrier = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
            .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
            .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
            .image = reconstruction->tonemap, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkDependencyInfo dependency_info = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                               .pImageMemoryBarriers = &barrier};
        vkCmdPipelineBarrier2(execution->cb, &dependency_info);
    }
    target->image = reconstruction->tonemap;
    target->private_attachment_view = reconstruction->tonemap_view;
    target->img = NULL;
    target->width = reconstruction->ow, target->height = reconstruction->oh;
    target->level = target->base_layer = target->blocks = 0;
    target->scale = 1;
    target->view_type = VK_IMAGE_VIEW_TYPE_2D;
    return 1;
}

static void prebloom_blit(VkExec *execution, const VkExecTarget *destination) {
    const struct Size *reconstruction = D.last;
    VkImageBlit region = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .srcOffsets = {{0, 0, 0}, {(int32_t)reconstruction->ow, (int32_t)reconstruction->oh, 1}},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, destination->level, destination->base_layer, 1},
        .dstOffsets = {{0, 0, 0}, {(int32_t)destination->width, (int32_t)destination->height, 1}}};
    vk_exec_barrier(execution);
    full_barrier(execution->cb);
    vkCmdBlitImage(execution->cb, reconstruction->tonemap, VK_IMAGE_LAYOUT_GENERAL, destination->image, VK_IMAGE_LAYOUT_GENERAL, 1, &region,
                   reconstruction->ow == destination->width && reconstruction->oh == destination->height ? VK_FILTER_NEAREST : VK_FILTER_LINEAR);
    full_barrier(execution->cb);
    execution->recorded++;
}

void vk_dlaa_direct_draw(const VkExecTarget *target, uint32_t source, int encode) {
    SceneReplay *scene = &D.scene.replay;
    if (!replay_scene_live() || !scene->tonemap || !target->img || !source) return;
    uint32_t destination = target->img->serial;
    if (destination == source) return;
    if (!encode && source == scene->tonemap) {
        scene->direct_composite = destination;
        scene->direct_window = 0;
    } else if (encode && source == scene->direct_composite) {
        scene->direct_window = destination;
        scene->direct_w = target->width;
        scene->direct_h = target->height;
    }
}

static void scene_tonemap(const VkExecTarget *target, int reconstructed) {
    if (!replay_scene_live() || !target->img || target->level || target->base_layer || target->layers != 1 ||
        target->samples != 1 || target->is3d || !scene_color(target->format)) return;
    if (!reconstructed) {
        D.scene.replay.route = SCENE_ROUTE_ORIGINAL;
        D.scene.replay.uw = D.scene.replay.rw;
        D.scene.replay.uh = D.scene.replay.rh;
    }

    D.scene.replay.direct_composite = D.scene.replay.direct_window = 0;
    D.scene.tonemapped = target->image;
    D.scene.tonemap_serial = D.scene.replay.tonemap = target->img->serial;
    D.scene.packed = VK_NULL_HANDLE;
    D.scene.packed_serial = D.scene.replay.packed = 0;
}

void vk_dlaa_stock_tonemap(const VkExecTarget *target) {
    if (!vk_dlaa_fg_only()) return;
    int depth = projection_target(&D.scene.depth, D.projection.depth_serial, &D.projection.dv);
    if (vk_dlaa_pathlog())
        rt_log("PATHLOG stock FG tone map: armed %d token %#llx conflict %d depth match %d",
               D.scene.armed, (unsigned long long)D.projection.sample.token, D.projection_conflict, depth);
    // The unmodified path can shade into a different color attachment. The camera owns the depth, and the
    // game's blend and scaling dispatches must read the recorded tone-map chain before publishing any inputs.
    if (!D.scene.armed || D.projection_conflict || !D.projection.sample.token || !depth) return;
    replay_scene_start();
    scene_tonemap(target, 0);
}

void vk_dlaa_scene_bloom(const VkExec *execution, int reconstructed) {
    if (!replay_scene_live()) return;
    D.scene.replay.bloom_seen = 1;
    D.scene.replay.bloom_written = reconstructed;
    D.scene.replay.bw = execution->pass_w;
    D.scene.replay.bh = execution->pass_h;
}

void vk_dlaa_prebloom_finish_tonemap(VkExec *execution, const VkExecTarget *guest_tonemap_target) {
    // Keep the guest chain valid even when the scaler handoff cannot be used.
    prebloom_blit(execution, guest_tonemap_target);
    vk_texture_gpu_written(guest_tonemap_target->tex);
    scene_tonemap(guest_tonemap_target, 1);
    if (vk_dlaa_pathlog())
        rt_log("PATHLOG pre-bloom tonemap private %ux%u, guest_tonemap_target fallback %ux%u", D.last->ow, D.last->oh, guest_tonemap_target->width, guest_tonemap_target->height);
}

static void vk_dlaa_pass_(VkExec *ex);
void vk_dlaa_pass(VkExec *ex) {
    int armed0 = D.scene.armed, done0 = scene_resolved(), ended0 = scene_closed();
    if (D.scene.hudless.pending) hudless_at_pass(ex);
    vk_dlaa_pass_(ex);
    if (vk_dlaa_pathlog()) {
        const VkExecTarget *c = &ex->color[0], *z = &ex->depth;
        rt_log("PATHLOG f %llu pass %u ncolor %d c0 %llx %ux%u fmt %d depth %llx %ux%u fmt %d | armed %d->%d ended %d->%d done %d->%d up %d light %llx",
               (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), ex->pass_serial + 1, ex->pass_ncolor,
               c->img ? (unsigned long long)c->img->addr : 0ull, c->width, c->height, c->image ? (int)c->format : 0,
               z->img ? (unsigned long long)z->img->addr : 0ull, z->width, z->height, z->image ? (int)z->format : 0, armed0, D.scene.armed,
               ended0, scene_closed(), done0, scene_resolved(), scene_scaled(), D.scene.color.img ? (unsigned long long)D.scene.color.img->addr : 0ull);
    }
}

// Looks up an NVN texture/image handle and verifies it references a whole, single-sample, output-sized 2D R11G11B10F
// backing image viewed in `expected_format` (SMAA compose and the scaler reinterpret the backing storage as R32UI / RGB9E5).
static const VkTexImage *scene_filter_image_locked(uint64_t handle, VkFormat expected_format, int is_sampled) {
    if (!handle) return NULL;
    view_t view;
    int has_view = 0;
    const VkTexImage *image = vk_texture_by_id_locked(NVN_HANDLE_TEX(handle), &view, &has_view);
    if (!image || image->gone || !image->image) return NULL;
    const VkFormat backing = expected_format == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32 ? VK_FORMAT_B10G11R11_UFLOAT_PACK32 : expected_format;
    if (image->format != backing) return NULL;
    if (image->type != VK_IMAGE_TYPE_2D || image->view_type != VK_IMAGE_VIEW_TYPE_2D) return NULL;
    if (image->aspect != VK_IMAGE_ASPECT_COLOR_BIT || image->samples != 1) return NULL;
    if (image->levels != 1 || image->layers != 1) return NULL;

    VkFormat view_format = image->format;
    if (has_view) {
        if ((view.set_levels && view.base_level != 0) || (view.set_layers && view.min_layer != 0)) return NULL;
        if ((view.set_target && view.target != NVN_TEXTURE_TARGET_2D) || (!is_sampled && view.set_swizzle)) return NULL;
        if (view.set_format) {
            TexVkFormat mapped;
            if (!tex_vk_format(view.format, &mapped)) return NULL;
            view_format = mapped.vk;
        }
    }
    if (view_format != expected_format) return NULL;

    if (is_sampled) {
        const uint8_t *swizzle = (has_view && view.set_swizzle) ? view.swz : image->swz;
        if (image->chan[0] ||
            swizzle[0] != NVN_TEXTURE_SWIZZLE_R ||
            swizzle[1] != NVN_TEXTURE_SWIZZLE_G ||
            swizzle[2] != NVN_TEXTURE_SWIZZLE_B)
            return NULL;
    }
    return image;
}

static const VkTexImage *pre_bloom_image_locked(uint64_t handle, VkFormat expected_format, int is_sampled) {
    const VkTexImage *image = scene_filter_image_locked(handle, expected_format, is_sampled);
    return image && image->width == D.ev_ow && image->height == D.ev_oh ? image : NULL;
}

#ifdef GAME_OOT
static const VkTexImage *stock_rcas_target(const VkExec *ex) {
    const uint64_t src = ex->tex[OOT_STAGE_CS][OOT_SCALER_PACKED_SLOT], dst = ex->img[OOT_STAGE_CS][OOT_SCALER_OUTPUT_SLOT];
    vk_texture_lock();
    const VkTexImage *input = scene_filter_image_locked(src, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 1);
    const VkTexImage *output = scene_filter_image_locked(dst, VK_FORMAT_B10G11R11_UFLOAT_PACK32, 0);
    if (!input || !output) {
        input = scene_filter_image_locked(src, VK_FORMAT_R16G16B16A16_SFLOAT, 1);
        output = scene_filter_image_locked(dst, VK_FORMAT_R16G16B16A16_SFLOAT, 0);
    }
    if (!input || !output || input->image == output->image || input->addr == output->addr) output = NULL;
    vk_texture_unlock();
    return output;
}
#endif

static void pre_bloom_stage_destroy(int recorded) {
    if (!D.pre_bloom.stage) return;
#ifdef GAME_OOT
    vk_defer_view(D.pre_bloom.stage_view);
    D.pre_bloom.stage_view = VK_NULL_HANDLE;
#endif
    if (recorded) vk_defer_image(D.pre_bloom.stage, D.pre_bloom.stage_mem);
    else vmaDestroyImage(vk_ctx()->vma, D.pre_bloom.stage, D.pre_bloom.stage_mem);
    D.pre_bloom.stage = VK_NULL_HANDLE;
    D.pre_bloom.stage_mem = NULL;
    D.pre_bloom.stage_w = 0;
    D.pre_bloom.stage_h = 0;
}

// Ensures the runtime-owned staging image exists at (w, h) in GENERAL layout. Returns 1 on success.
static int pre_bloom_stage_ensure(VkCommandBuffer cb, uint32_t w, uint32_t h, VkFormat fmt) {
    if (D.pre_bloom.stage && D.pre_bloom.stage_w == w && D.pre_bloom.stage_h == h && D.pre_bloom.stage_fmt == fmt) return 1;
    pre_bloom_stage_destroy(1);

    const VkCtx *c = vk_ctx();
    VkImageCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType = VK_IMAGE_TYPE_2D,
        .format = fmt,
        .extent = {w, h, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = VK_SAMPLE_COUNT_1_BIT,
        .tiling = VK_IMAGE_TILING_OPTIMAL,
        .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };
#ifdef GAME_OOT
    ci.usage |= VK_IMAGE_USAGE_SAMPLED_BIT;
#endif
    VmaAllocationCreateInfo alloc_ci = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ci, &alloc_ci, &D.pre_bloom.stage, &D.pre_bloom.stage_mem, NULL) != VK_SUCCESS) {
        D.pre_bloom.stage = VK_NULL_HANDLE;
        D.pre_bloom.stage_mem = NULL;
        return 0;
    }
#ifdef GAME_OOT
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = D.pre_bloom.stage,
        .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &D.pre_bloom.stage_view) != VK_SUCCESS) {
        pre_bloom_stage_destroy(0);
        return 0;
    }
#endif

    VkImageMemoryBarrier2 barrier = {
        .sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
        .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
        .newLayout = VK_IMAGE_LAYOUT_GENERAL,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image = D.pre_bloom.stage,
        .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1},
    };
    VkDependencyInfo dep = {
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers = &barrier,
    };
    vkCmdPipelineBarrier2(cb, &dep);
    D.pre_bloom.stage_w = w;
    D.pre_bloom.stage_h = h;
    D.pre_bloom.stage_fmt = fmt;
    return 1;
}

// Snapshots the finished tone-mapped float picture as SMAA compose reads it, into our own staging image.
// The game's image can share physical-pool memory with later passes before the scaler runs, so copying at compose
// captures the exact image before anything else can alias or overwrite it.
static void pre_bloom_stage(VkExec *ex) {
    D.pre_bloom.staged = 0;
    D.pre_bloom.stage_why = NULL;
#ifdef GAME_NS2
    if (ex->gpu_cond) {
        D.pre_bloom.stage_why = "SMAA compose dispatch is GPU-conditional";
        return;
    }
#endif
    vk_texture_lock();
    const uint64_t color_handle = ex->tex[OOT_STAGE_CS][OOT_SMAA_COLOR_SLOT];
    const uint64_t packed_handle = ex->img[OOT_STAGE_CS][OOT_SMAA_PACKED_SLOT];
    const VkTexImage *color = pre_bloom_image_locked(color_handle,
                                                     VK_FORMAT_B10G11R11_UFLOAT_PACK32, 1);
    const VkTexImage *packed = pre_bloom_image_locked(packed_handle,
                                                      VK_FORMAT_R32_UINT, 0);
#ifdef GAME_OOT
    const int spatial = spatial_scaler();
    if (spatial) {
        color = scene_filter_image_locked(color_handle, VK_FORMAT_B10G11R11_UFLOAT_PACK32, 1);
        if (!color) color = scene_filter_image_locked(color_handle, VK_FORMAT_R16G16B16A16_SFLOAT, 1);
        view_t view;
        int viewed = 0;
        packed = packed_handle ? vk_texture_by_id_locked(NVN_HANDLE_TEX(packed_handle), &view, &viewed) : NULL;
        if (packed && (packed->gone || !packed->image || packed->levels != 1 || packed->layers != 1 || packed->samples != 1))
            packed = NULL;
    }
#else
    const int spatial = 0;
#endif
    const char *why = NULL;
    if (!color) why = "SMAA compose color slot is not a supported tone-mapped image";
    else if (!packed || (!spatial && packed->image != D.scene.color.image)) why = "SMAA compose packed output does not match the armed scene";
    else if (color->image == packed->image) why = "SMAA compose color image aliases the packed output";
    else if (!(color->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) why = "SMAA compose color image lacks transfer-source usage";

    VkImage src = why ? VK_NULL_HANDLE : color->image;
    uint32_t w = why ? 0 : color->width;
    uint32_t h = why ? 0 : color->height;
    VkFormat fmt = why ? VK_FORMAT_UNDEFINED : color->format;
    if (!why) D.pre_bloom.packed_serial = packed->serial;
    vk_texture_unlock();

    if (why) {
        D.pre_bloom.stage_why = why;
        return;
    }

    vk_exec_par_join();
    vk_exec_barrier(ex);   // ensure tone mapping finished writing `src`
    if (!pre_bloom_stage_ensure(ex->cb, w, h, fmt)) {
        D.pre_bloom.stage_why = "staging image creation failed";
        return;
    }

    VkImageSubresourceLayers layer = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    VkImageCopy region = {.srcSubresource = layer, .dstSubresource = layer, .extent = {w, h, 1}};
    vkCmdCopyImage(ex->cb, src, VK_IMAGE_LAYOUT_GENERAL, D.pre_bloom.stage, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    ex->recorded++;
    vk_exec_barrier(ex);   // complete read of `src` before SMAA compose writes `packed` (which may share memory)
    D.pre_bloom.staged = 1;
}

// At the guest's post-tonemap scaler dispatch, write the staged picture to the HUD target without using the SMAA output.
static int pre_bloom_scaler(VkExec *ex, const GfxShader *cs) {
    vk_exec_par_join();
    vk_texture_lock();
    const VkTexImage *packed = pre_bloom_image_locked(ex->tex[OOT_STAGE_CS][OOT_SCALER_PACKED_SLOT],
                                                      VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 1);
    const VkTexImage *hud_target = pre_bloom_image_locked(ex->img[OOT_STAGE_CS][OOT_SCALER_OUTPUT_SLOT],
                                                          VK_FORMAT_B10G11R11_UFLOAT_PACK32, 0);
#ifdef GAME_OOT
    const int easu = vk_dlaa_fsr1();
    const int spatial = spatial_scaler();
    if (spatial && !D.pre_bloom.staged) {
        const char *why = D.pre_bloom.stage_why;
        vk_texture_unlock();
        if (why) {
            static const char *reported;
            if (why != reported) rt_log("dlaa: %s scene scaling unavailable: %s", easu ? "FSR1" : "bilinear", why);
            reported = why;
        }
        return 0;
    }
    if (spatial) {
        const uint64_t input = ex->tex[OOT_STAGE_CS][OOT_SCALER_PACKED_SLOT], output = ex->img[OOT_STAGE_CS][OOT_SCALER_OUTPUT_SLOT];
        packed = scene_filter_image_locked(input, VK_FORMAT_E5B9G9R9_UFLOAT_PACK32, 1);
        if (!packed) packed = scene_filter_image_locked(input, VK_FORMAT_R16G16B16A16_SFLOAT, 1);
        hud_target = scene_filter_image_locked(output, D.pre_bloom.stage_fmt, 0);
    }
#else
    const int easu = 0;
    const int spatial = 0;
#endif
    const char *fallback_reason = NULL;
    if (!D.pre_bloom.staged)
        fallback_reason = D.pre_bloom.stage_why ? D.pre_bloom.stage_why : "SMAA compose was not seen this frame";
    else if (!packed || packed->serial != D.pre_bloom.packed_serial)
        fallback_reason = "the scaler reads a different RGB9E5 image";
    else if (!hud_target)
        fallback_reason = "the scaler's output view is not a native R11G11B10F image";
    else if (hud_target->image == packed->image || hud_target->addr == packed->addr)
        fallback_reason = "the HUD target is SMAA's image";
    else if (!spatial && (hud_target->width != D.pre_bloom.stage_w || hud_target->height != D.pre_bloom.stage_h))
        fallback_reason = "the HUD target is not the picture's size";
    else if (!(hud_target->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        fallback_reason = "the HUD target lacks transfer-destination usage";
#ifdef GAME_NS2
    else if (ex->gpu_cond)
        fallback_reason = "the scaler dispatch is GPU-conditional";
#endif
    if (spatial && !easu && !fallback_reason && (hud_target->width != D.pre_bloom.stage_w || hud_target->height != D.pre_bloom.stage_h)) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, hud_target->format, &props);
        const VkFormatFeatureFlags required = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT |
                                               VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
        if ((props.optimalTilingFeatures & required) != required) fallback_reason = "bilinear blit is unsupported for the picture format";
    }
    if (!fallback_reason) vk_texture_storage_bound_locked(hud_target);
    vk_texture_unlock();

    D.pre_bloom.staged = 0;
    D.pre_bloom.stage_why = NULL;
    D.scene.hudless.width = D.ev_ow;
    D.scene.hudless.height = D.ev_oh;
    if (fallback_reason) {
        hudless_track(ex, cs);
        if (D.scene.hudless.texture) {
            D.out_addr = D.scene.hudless.address;
            D.scene.hudless.pending = hudless_wanted();
        }
    }
    vk_exec_results_point(ex);

    if (fallback_reason) {
        static unsigned fallback_logs;
        if (fallback_logs++ < 8 || vk_dlaa_pathlog())
            rt_log("dlaa: %s could not replace the game's scaler: %s", easu ? "FSR1" : spatial ? "bilinear" : "pre-bloom", fallback_reason);
        return 0;
    }

    vk_exec_barrier(ex);
    VkImageSubresourceLayers layer = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    VkImageCopy region = {.srcSubresource = layer, .dstSubresource = layer, .extent = {D.pre_bloom.stage_w, D.pre_bloom.stage_h, 1}};
    const int resize = hud_target->width != region.extent.width || hud_target->height != region.extent.height;
#ifdef GAME_OOT
    if (easu) {
        if (!resize || hud_target->width < region.extent.width || hud_target->height < region.extent.height) return 0;
        if (!vk_fsr_tonemapped(ex->cb, D.pre_bloom.stage_view, region.extent.width, region.extent.height,
                               hud_target->image, hud_target->format, hud_target->width, hud_target->height,
                               vk_hdr_renodx_scene())) {
            static unsigned failures;
            if (failures++ < 8 || vk_dlaa_pathlog()) rt_log("dlaa: FSR1 could not filter the tone-mapped picture");
            return 0;
        }
        // The scene filters replaced the guest's compute pipeline and push descriptors.
        vk_exec_draw_new_cb();
    } else
#endif
    if (spatial && resize) {
        VkImageBlit blit = {.srcSubresource = layer, .dstSubresource = layer,
            .srcOffsets = {{0}, {(int32_t)region.extent.width, (int32_t)region.extent.height, 1}},
            .dstOffsets = {{0}, {(int32_t)hud_target->width, (int32_t)hud_target->height, 1}}};
        vkCmdBlitImage(ex->cb, D.pre_bloom.stage, VK_IMAGE_LAYOUT_GENERAL, hud_target->image, VK_IMAGE_LAYOUT_GENERAL, 1, &blit, VK_FILTER_LINEAR);
    } else vkCmdCopyImage(ex->cb, D.pre_bloom.stage, VK_IMAGE_LAYOUT_GENERAL, hud_target->image, VK_IMAGE_LAYOUT_GENERAL, 1, &region);
    ex->recorded++;
    vk_exec_barrier(ex);
#ifdef GAME_OOT
    sharpen_hud_input(ex, hud_target, 0);
#endif
    hudless_scene_target(hud_target, hud_target->image);
    D.scene.hudless.width = hud_target->width;
    D.scene.hudless.height = hud_target->height;
    D.out_addr = D.scene.hudless.address;
    D.scene.hudless.pending = hudless_wanted();
    if (vk_dump_armed) vk_dump_dlaa_copy(ex, hud_target);
    VkExecTarget output = {.img = hud_target, .image = hud_target->image,
                           .width = hud_target->width, .height = hud_target->height};
#ifdef GAME_OOT
    if (spatial && D.projection.sample.token && !D.projection_conflict) replay_scene_start();
#endif
    replay_scene_complete(ex, &output);
#ifdef GAME_OOT
    if (spatial && vk_dlaa_fg_only()) stock_fg_motion(ex, output.width, output.height);
#endif

    static unsigned logged;
    const unsigned route = easu ? 8u : spatial ? (resize ? 2u : 4u) : 1u;
    if (!(logged & route) || vk_dlaa_pathlog()) {
        logged |= route;
        rt_log("dlaa: %s tone-mapped picture %ux%u -> %ux%u", easu ? "FSR1 filters the perceptual" :
               spatial ? (resize ? "bilinear blits" : "bilinear copies") : "pre-bloom copies",
               region.extent.width, region.extent.height, hud_target->width, hud_target->height);
    }
    return 1;
}

static void prebloom_scaler_handoff(VkExec *ex, const VkExecTarget *dst) {
    prebloom_blit(ex, dst);
#ifdef GAME_OOT
    sharpen_hud_input(ex, dst->img, 0);
#endif
    replay_scene_complete(ex, dst);
    D.scene.stock_output = NULL;
    D.scene.tonemapped = D.scene.packed = VK_NULL_HANDLE;
    if (hudless_wanted()) {
        D.scene.hudless.pending = 1;
        D.scene.hudless.width = dst->width;
        D.scene.hudless.height = dst->height;
        hudless_scene_target(dst->img, dst->image);
    }
}

#ifdef GAME_OOT
static const VkTexImage *aa_scratch_locked(uint64_t handle, VkFormat format) {
    if (!handle) return NULL;
    view_t view;
    int viewed = 0;
    const VkTexImage *img = vk_texture_by_id_locked(NVN_HANDLE_TEX(handle), &view, &viewed);
    if (!img || img->gone || !img->image || img->type != VK_IMAGE_TYPE_2D || img->samples != 1 || img->layers != 1 ||
        (viewed && ((view.set_levels && view.base_level) || (view.set_layers && view.min_layer)))) return NULL;
    TexVkFormat mapped;
    if (viewed && view.set_format) return tex_vk_format(view.format, &mapped) && mapped.vk == format ? img : NULL;
    return img->format == format ? img : NULL;
}

// Prove the entire chain before omitting its first dispatch. A wait, unrelated GPU operation, or unknown binding
// leaves the stock chain intact so it can still serve a scaler handoff that cannot be replaced.
static const VkTexImage *prebloom_aa_preflight(const VkExec *ex, const op_hdr *start, const op_hdr *ops[3]) {
    static const uint64_t callers[] = {0x14e3a94, 0x14e3b24, 0x251bc0};
    uint64_t tex[128], img[32];
    memcpy(tex, ex->tex[OOT_STAGE_CS], sizeof tex);
    memcpy(img, ex->img[OOT_STAGE_CS], sizeof img);
    const dpool_t *pool = ex->tex_pool;
    const VkTexImage *edges = aa_scratch_locked(img[0], VK_FORMAT_R16G16_UINT);
    const VkTexImage *tone = scene_filter_image_locked(tex[3], D.last->tonemap_format, 1);
    if (!edges || !tone || tone->serial != D.scene.tonemap_serial) {
        if (vk_dlaa_pathlog()) rt_log("PATHLOG SMAA preflight: edges %d tone %d", !!edges, tone && tone->serial == D.scene.tonemap_serial);
        return NULL;
    }
    const VkTexImage *weights = NULL, *packed = NULL;
    const op_hdr *op = nvn_op_next(start);
    unsigned stage = 0;
    for (unsigned step = 0; op && step < 192; ++step) {
        const void *p = nvn_op_payload(op);
        switch (op->op) {
        case OP_JUMP: op = (const op_hdr *)(uintptr_t)((const p_jump *)p)->target; continue;
        case OP_SET_TEX_POOL: pool = ((const p_set_pool *)p)->pool; break;
        case OP_BIND_TEX: case OP_BIND_IMG: {
            const p_bind_handles *b = p;
            if (b->stage != OOT_STAGE_CS) break;
            const int cap = op->op == OP_BIND_TEX ? 128 : 32;
            uint64_t *row = op->op == OP_BIND_TEX ? tex : img;
            if (b->first < 0 || b->n < 0 || b->n > 32 || b->first > cap - b->n) return NULL;
            for (int i = 0; i < b->n; ++i) {
                const uint64_t id = NVN_HANDLE_TEX(b->h[i]);
                row[b->first + i] = (id && pool ? id + pool->base : id) & 0xfffff;
            }
            break;
        }
        case OP_BARRIER: case OP_BIND_PROGRAM: case OP_BIND_UBO: case OP_BIND_SSBO:
        case OP_BIND_SAMP: case OP_SET_SAMP_POOL: case OP_UPDATE_UB: case OP_CLEAR_BUF: case OP_PUSH_CONST:
            break;
        case OP_DISPATCH: case OP_DISPATCH_IND: {
            uint64_t caller = op->op == OP_DISPATCH ? ((const p_dispatch *)p)->caller : ((const p_dispatch_ind *)p)->caller;
            if (caller != MOD_MAIN_BASE + callers[stage]) {
                if (vk_dlaa_pathlog()) rt_log("PATHLOG SMAA preflight: unexpected dispatch at stage %u", stage);
                return NULL;
            }
            ops[stage] = op;
            if (stage == 0) {
                if (aa_scratch_locked(tex[4], VK_FORMAT_R16G16_UINT) != edges) return NULL;
                weights = aa_scratch_locked(img[1], VK_FORMAT_R32_UINT);
                if (!weights) return NULL;
            } else if (stage == 1) {
                if (scene_filter_image_locked(tex[3], D.last->tonemap_format, 1) != tone ||
                    aa_scratch_locked(tex[5], VK_FORMAT_R8G8B8A8_UNORM) != weights) return NULL;
                packed = aa_scratch_locked(img[2], VK_FORMAT_R32_UINT);
                if (!packed) packed = aa_scratch_locked(img[2], VK_FORMAT_R16G16B16A16_SFLOAT);
                if (!packed) return NULL;
            } else {
                const VkTexImage *input = aa_scratch_locked(tex[0], VK_FORMAT_E5B9G9R9_UFLOAT_PACK32);
                if (!input) input = aa_scratch_locked(tex[0], VK_FORMAT_R16G16B16A16_SFLOAT);
                const VkTexImage *out = scene_filter_image_locked(img[1], D.last->tonemap_format, 0);
                if (op->op != OP_DISPATCH || !out || input != packed || out->image == tone->image ||
                    out->image == packed->image || out->addr == tone->addr || out->addr == packed->addr) return NULL;
                VkExecTarget dst = {.img = out, .image = out->image, .format = out->format, .view_type = out->view_type,
                    .width = out->width, .height = out->height, .layers = out->layers, .samples = out->samples};
                return prebloom_blit_supported(D.last, D.last->tonemap_format, &dst) ? out : NULL;
            }
            ++stage;
            break;
        }
        default:
            if (vk_dlaa_pathlog()) rt_log("PATHLOG SMAA preflight stopped at %s (stage %u)", nvn_op_name(op->op), stage);
            return NULL;
        }
        op = nvn_op_next(op);
    }
    return NULL;
}

int vk_dlaa_skip_smaa(VkExec *ex, uint64_t caller) {
    if (D.scene.skip_aa_output) {
        if (vk_exec_cur_op == D.scene.skip_aa[0] || vk_exec_cur_op == D.scene.skip_aa[1]) return 1;
        if (vk_exec_cur_op == D.scene.skip_aa[2]) {
            const VkTexImage *out = D.scene.skip_aa_output;
            D.scene.skip_aa_output = NULL;
            vk_texture_lock();
            vk_texture_storage_bound_locked(out);
            vk_texture_unlock();
            const VkExecTarget dst = {.img = out, .image = out->image, .width = out->width, .height = out->height};
            vk_exec_pass_end(ex);
            vk_exec_results_point(ex);
            prebloom_scaler_handoff(ex, &dst);
            static unsigned count;
            if (++count == 1 || count % 300 == 0 || vk_dlaa_pathlog())
                rt_log("dlaa: skipped all three SMAA dispatches on %u reconstructed frames", count);
            return 1;
        }
        return 0;
    }
    if (caller != MOD_MAIN_BASE + 0x14e3a4c) return 0;
    if (vk_dump_armed || ex->gpu_cond || ex->render_disabled || !scene_prebloom() || !replay_scene_live() ||
        !D.scene.tonemapped || !D.last || !D.last->tonemap) return 0;
    static int on = -1;
    if (on < 0) on = rt_knob("SKIP_SMAA", "1") && rt_knob("DLAA_AT_DISPATCH", "1");
    if (!on) return 0;
    vk_texture_lock();
    D.scene.skip_aa_output = prebloom_aa_preflight(ex, vk_exec_cur_op, D.scene.skip_aa);
    vk_texture_unlock();
    return D.scene.skip_aa_output != NULL;
}
#endif

// OoT post-processes and scales its picture in compute: after the bloom composite and ACES/RenoDX tonemap write the
// armed light target (D.scene.color), three SMAA dispatches run (edges into RG16UI, weights into RGBA8, blend packing
// sqrt(color) into RGB9E5 through an R32UI storage view) and a fourth scaling dispatch reads that RGB9E5 view and
// writes the output-sized float picture (B10G11R11) that the HUD draws onto. Replacing that scaling dispatch in both
// DLAA (1:1) and DLSS/FSR3 upscaling modes keeps SMAA's blur, RGB9E5 quantization, and the game's own scaler off the
// upscaler's output. Spatial FSR 1 at 1:1 leaves the scaling dispatch alone. 1: the dispatch is replaced.
int vk_dlaa_dispatch(VkExec *ex, const GfxShader *cs, uint64_t caller) {
    static int on = -1;
    if (on < 0) on = rt_knob("DLAA_AT_DISPATCH", "1");
    D.scene.stock_pack = D.scene.stock_output = NULL;
#ifdef GAME_OOT
    D.scene.rcas_output = cs && caller == MOD_MAIN_BASE + 0x251bc0 && !ex->gpu_cond ? stock_rcas_target(ex) : NULL;
#endif
#ifdef GAME_OOT
    if (cs && spatial_scaler()) {
        if (caller == MOD_MAIN_BASE + 0x14e3b24) pre_bloom_stage(ex);
        if (caller == MOD_MAIN_BASE + 0x251bc0 && pre_bloom_scaler(ex, cs)) return 1;
    }
#endif
    if (D.pre_bloom.resolved && cs) {
        uint64_t key = shader_input_prefix(cs);
        if (key == OOT_SMAA_COMPOSE_CS) pre_bloom_stage(ex);
        if (key == OOT_SCALER_CS) return pre_bloom_scaler(ex, cs);
    }
    if (D.scene.hudless.pending && !D.scene.replay.completed && cs) hudless_track(ex, cs);   // after the evaluation: the picture the HUD-less copy is taken of
    const int checked = replay_scene_live() && D.scene.tonemapped;
    const int prebloom = checked && D.scene.replay.route != SCENE_ROUTE_ORIGINAL;
    const int packs_prebloom = checked && caller == MOD_MAIN_BASE + 0x14e3b24;
    const int scales_prebloom = checked && caller == MOD_MAIN_BASE + 0x251bc0 && D.scene.packed;
#ifdef GAME_OOT
    const int allow_late = !vk_dlaa_fsr1();
    if (!allow_late && !packs_prebloom && !scales_prebloom) return 0;
#else
    const int allow_late = 1;
#endif
    if (packs_prebloom) {
        D.scene.packed = VK_NULL_HANDLE;
        D.scene.packed_serial = D.scene.replay.packed = 0;
    }
    if (!cs || (!packs_prebloom && !scales_prebloom && (checked || !on || !dl_active() || scene_resolved() ||
        !D.scene.armed || deferred))) return 0;
#ifdef GAME_OOT
    const int spatial = 0;   // FSR1 only runs through the checked post-tonemap scaler above.
#else
    const int spatial = fsr1();
#endif
    if (scene_projection_enabled() && !spatial && !D.projection.sample.token) return 0;
    const GfxShaderRes *r;
    int nt = gfx_shader_resources(cs, GFX_RES_TEX, &r);
    int reads_light = 0, reads_e5 = 0, reads_tonemap = 0, reads_prebloom_packed = 0;
    const VkTexImage *out = NULL;
    vk_texture_lock();
    for (int i = 0; i < nt; i++) {
        if (r[i].slot < 0 || r[i].slot >= 128) continue;
        uint64_t h = ex->tex[5][r[i].slot];
        if (!h) continue;
        view_t v;
        int hv = 0;
        const VkTexImage *img = vk_texture_by_id_locked((int)(h & 0xfffff), &v, &hv);
        if (!img) continue;
        if (img == D.scene.color.img) reads_light = 1;
        const int base = !hv || ((!v.set_levels || !v.base_level) && (!v.set_layers || !v.min_layer));
        if (checked && base && img->image == D.scene.tonemapped && img->serial == D.scene.tonemap_serial) reads_tonemap = 1;
        TexVkFormat sampled_format;
        if (base && img->image == D.scene.packed && img->serial == D.scene.packed_serial &&
            img->format == VK_FORMAT_R16G16B16A16_SFLOAT &&
            (!hv || !v.set_format || (tex_vk_format(v.format, &sampled_format) && sampled_format.vk == img->format)))
            reads_prebloom_packed = 1;
        if (hv && v.set_format) {
            TexVkFormat tvf;
            if (tex_vk_format(v.format, &tvf) && tvf.vk == VK_FORMAT_E5B9G9R9_UFLOAT_PACK32) {
                reads_e5 = 1;
                if (base && img->image == D.scene.packed && img->serial == D.scene.packed_serial) reads_prebloom_packed = 1;
            }
        }
    }
    int n = gfx_shader_resources(cs, GFX_RES_IMG, &r);
    if ((packs_prebloom || scales_prebloom) && vk_dlaa_pathlog())
        rt_log("PATHLOG scene compute caller %#llx: color match %d packed match %d images %d",
               (unsigned long long)caller, reads_tonemap, reads_prebloom_packed, n);
    if (checked && n != 1) {
        vk_texture_unlock();
        return 0;
    }
    for (int i = 0; i < n && !out; i++) {
        if (r[i].slot < 0 || r[i].slot >= 32) continue;
        uint64_t h = ex->img[5][r[i].slot];   // the compute stage's images (NVN_STAGE_CS)
        if (!h) continue;
        view_t v;
        int hv = 0;
        const VkTexImage *img = vk_texture_by_id_locked((int)(h & 0xfffff), &v, &hv);
        if (!img || !(img->aspect & VK_IMAGE_ASPECT_COLOR_BIT) || (!prebloom && img == D.scene.color.img))
            continue;
        if (checked) {
            if (hv && ((v.set_levels && v.base_level) || (v.set_layers && v.min_layer) ||
                       (v.set_target && v.target != NVN_TEXTURE_TARGET_2D) || v.set_swizzle)) continue;
            TexVkFormat storage_format;
            if (packs_prebloom && reads_tonemap && img->image != D.scene.tonemapped) {
                const int packed = hv && v.set_format && tex_vk_format(v.format, &storage_format) &&
                                   storage_format.vk == VK_FORMAT_R32_UINT;
                const int float_blend = img->format == VK_FORMAT_R16G16B16A16_SFLOAT &&
                    img->type == VK_IMAGE_TYPE_2D && img->layers == 1 && img->samples == 1 &&
                    (!hv || !v.set_format || (tex_vk_format(v.format, &storage_format) && storage_format.vk == img->format));
                if (packed || float_blend) D.scene.stock_pack = img;
            }
            if (scales_prebloom && reads_prebloom_packed && (!hv || !v.set_format ||
                (tex_vk_format(v.format, &storage_format) && storage_format.vk == img->format)) &&
                img->image != D.scene.tonemapped && img->image != D.scene.packed &&
                img->type == VK_IMAGE_TYPE_2D && img->view_type == VK_IMAGE_VIEW_TYPE_2D &&
                scene_color(img->format) && img->layers == 1 && img->samples == 1) {
                D.scene.stock_output = img;
                VkExecTarget destination = {.image = img->image, .format = img->format, .view_type = img->view_type,
                    .layers = img->layers, .width = img->width, .height = img->height, .samples = img->samples, .img = img};
                if (on && ((prebloom && D.last && prebloom_blit_supported(D.last, D.last->tonemap_format, &destination)) ||
                    (!prebloom && allow_late && !scene_resolved() && dl_active() && !deferred &&
                     (!spatial || (img->width > D.scene.color.width && img->height > D.scene.color.height))))) out = img;
            }
            continue;
        }
        TexVkFormat storage_format;
        const int reinterpreted = hv && v.set_format && tex_vk_format(v.format, &storage_format) && storage_format.vk != img->format;
        if (!scene_color(img->format) || reinterpreted) continue;   // SMAA compose packs RGB9E5 words through an R32UI view
        int shape = (double)img->width * D.scene.color.height <= 1.03 * (double)img->height * D.scene.color.width &&
                    (double)img->height * D.scene.color.width <= 1.03 * (double)img->width * D.scene.color.height;
        int larger = shape && img->width > D.scene.color.width && img->height > D.scene.color.height;
        int same = !spatial && img->width == D.scene.color.width && img->height == D.scene.color.height && !reads_light &&
                   (reads_e5 || (D.out_addr && img->addr == D.out_addr));
        if (larger || same) out = img;
    }
    if (out) vk_texture_storage_bound_locked(out);   // the picture is written here, as the dispatch would have
    vk_texture_unlock();
    if (scales_prebloom && !out && vk_dlaa_pathlog())
        rt_log("PATHLOG pre-bloom scaler handoff rejected: packed source or destination did not match");
    if (!out) return 0;
    VkExecTarget dst = {.image = out->image, .format = out->format, .view_type = VK_IMAGE_VIEW_TYPE_2D,
                        .aspect = VK_IMAGE_ASPECT_COLOR_BIT, .layers = 1, .width = out->width, .height = out->height,
                        .samples = 1, .img = out};
    vk_exec_pass_end(ex);
    vk_exec_results_point(ex);   // what the CPU can see of the frame is drawn: the scaling and the HUD run beside the next frame
    if (prebloom) {
        // Both the packed and float AA variants feed the game's reconstruction dispatch. Copy the already
        // linear tone-map result directly, without crossing formats or downsampling to the scene render size.
        prebloom_scaler_handoff(ex, &dst);
        static int first;
        if (!first++ || vk_dlaa_pathlog())
            rt_log("dlaa: pre-bloom scaler handoff %ux%u -> %ux%u", D.last->ow, D.last->oh, dst.width, dst.height);
        return 1;
    }
    uint64_t evals0 = D.evals;
    D.scene.phase = out->width > D.scene.color.width ? SCENE_SCALED : SCENE_RESOLVED;
    D.out_addr = out->addr, D.at_dispatch++;
    static int told;
    if (!told++)
        rt_log("dlaa: %s replaces the game's scaling dispatch: scene %ux%u -> output %ux%u (format %d, e5=%d)",
               spatial ? "FSR 1" : fsr3() ? "FSR 3.1" : "DLSS", D.scene.color.width, D.scene.color.height, out->width,
               out->height, (int)out->format, reads_e5);
    if (vk_dlaa_pathlog())
        rt_log("PATHLOG   DLSS at the scaling dispatch: storage %llx %ux%u vkfmt %d, light %llx %ux%u", (unsigned long long)out->addr,
               out->width, out->height, (int)out->format, D.scene.color.img ? (unsigned long long)D.scene.color.img->addr : 0ull, D.scene.color.width,
               D.scene.color.height);
    evaluate(ex, &dst, EVALUATE_AT_DISPATCH);
#ifdef GAME_OOT
    if (D.evals != evals0 && (scales_prebloom || out == D.scene.rcas_output)) {
        sharpen_hud_input(ex, out, 0);
        if (hudless_wanted()) {
            // The late route can resolve without a checked bloom pass. Start its checkpoint at the verified scaler.
            if (D.projection.sample.token && !D.projection_conflict) replay_scene_start();
            D.scene.hudless.pending = 1, D.scene.hudless.width = dst.width, D.scene.hudless.height = dst.height;
            hudless_scene_target(out, dst.image);
        }
    }
#endif
    if (D.evals != evals0 && replay_scene_live()) {
        D.scene.replay.route = SCENE_ROUTE_LATE;
        D.scene.replay.uw = dst.width;
        D.scene.replay.uh = dst.height;
        replay_scene_complete(ex, &dst);
    }
    if (D.evals == evals0) {
        D.scene.phase = SCENE_DRAWING;
        return 0;
    }
    return 1;
}

static void vk_dlaa_pass_(VkExec *ex) {
    const VkExecTarget *c = &ex->color[0], *z = &ex->depth;
    if ((!dl_active() && !vk_dlaa_fg_only()) || scene_resolved()) return;
    unsigned o[2], s[2];
    res_sizes(o, s);
    // OoT's builds carry no scene or output patch sets, so res_sizes reports placeholders there: the sizes come
    // from the frame itself (the light target's, and the output pass's).
    int scene = z->image && z->format == VK_FORMAT_D32_SFLOAT && z->width >= DLAA_SCENE_MIN_W;
    // The G-buffer: several colour targets with the D32F scene depth. Only the light target drawn after it arms the
    // frame; at night the game draws into the light target alone with the scene depth before the G-buffer too, and
    // DLAA then ran at a small pass ahead of the whole scene, which reached the screen jittered.
    // A pass of that shape smaller than the light target already armed is not one: OoT draws a half-size effect
    // into three targets with a half-size D32F depth after its scene, and taking that for a G-buffer switched every
    // later frame to the deferred shape, where DLSS's and FSR 3.1's smaller modes never found their output pass.
    if (ex->pass_ncolor >= 3 && scene && !(D.scene.armed && (z->width < D.scene.color.width || z->height < D.scene.color.height))) {
        if (!deferred)
            rt_log("dlaa: a G-buffer pass (%d colour targets, depth %ux%u, target 0 format %d): the deferred frame shape",
                   ex->pass_ncolor, z->width, z->height, c->image ? (int)c->format : 0);
        D.scene.gbuf = 1, D.scene.armed = 0, deferred = 1;
        return;
    }
    // the scene depth is D32F; a D16 target of the scene size also serves one composite pass before the transparents.
    // A smaller pair never takes over an armed one: the camera rune draws a half-size RGBA16F target with a half-size
    // D32F depth after the lighting (852x480 under a 1704x960 scene), and DLSS ran on that and wrote its picture into
    // the scene's own light target, which left every object black.
    if ((D.scene.gbuf || !deferred) && ex->pass_ncolor == 1 && c->image && scene_color(c->format) && scene &&
        c->width == z->width && c->height == z->height &&
        (!D.scene.armed || (c->width >= D.scene.color.width && c->height >= D.scene.color.height)) && c->samples <= 1 && !c->is3d) {
        int first = !D.scene.armed;
        D.scene.color = *c, D.scene.depth = *z, D.scene.armed = 1;
        static int told;
        if (first && !told++)
            rt_log("dlaa: first scene armed: light target %ux%u format %d, depth %ux%u", c->width, c->height, (int)c->format,
                   z->width, z->height);
        return;
    }
#ifdef GAME_OOT
    if (fsr1()) return;
#endif
    if (!D.scene.armed || !dl_active()) return;
    if (!deferred) {   // OoT's shape: the output pass ends the scene and runs DLSS
        if (scene_projection_enabled() && !fsr1()) return;   // stock scaling dispatch consumes the scene-complete marker
        int one = ex->pass_ncolor == 1 && c->image && !z->image && c->image != D.scene.color.image && c->samples <= 1 &&
                  !c->is3d;
        D.oot_pass = 1;   // a pause screen's scene (vk_dlaa_fg_inputs)
        if (one && c->width > D.scene.color.width && c->height > D.scene.color.height) {
            D.scene.phase = SCENE_SCALED;
            D.at_pass++;
            evaluate(ex, c, EVALUATE_AT_PASS);
        } else if (one && output_target(c->format) && c->width == D.scene.color.width &&
                   c->height == D.scene.color.height) {
            D.scene.phase = SCENE_RESOLVED;
            evaluate(ex, NULL, EVALUATE_AT_PASS);
        }
        D.oot_pass = 0;
        return;
    }
    if (!scene_closed()) {
        if (!ex->pass_w || ex->pass_w * 4 > D.scene.color.width) return;
        D.scene.phase = SCENE_CLOSED;
        if (D.scene.color.width >= o[0]) D.scene.phase = SCENE_RESOLVED, evaluate(ex, NULL, EVALUATE_AT_PASS);   // DLAA in place; else wait for the scaling pass
        return;
    }
    // the game's scaling pass: one colour target of the output size, no depth
    if (ex->pass_ncolor == 1 && c->image && !z->image && scene_color(c->format) && c->width > D.scene.color.width &&
        c->width == o[0] && c->height == o[1] && c->samples <= 1 && !c->is3d) {
        D.scene.phase = SCENE_SCALED;
        evaluate(ex, c, EVALUATE_AT_PASS);
    }
}


// Without native projection jitter, OoT's standing camera repeats one sample per pixel and RCAS at 1:1 evens out the
// sharpness step (vk_fsr.c). With native projection jitter active, DLSS/DLAA accumulates genuine sub-pixel detail and
// needs no extra 1:1 present RCAS pass on top of HDR10/HUD unless explicitly requested (REKARINA_DLAA_SHARPEN=1).
int vk_dlaa_sharpens(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("REKARINA_DLAA_SHARPEN");
        on = e && *e ? atoi(e) != 0 : !scene_projection_enabled();
    }
    return on && dl_active() && !fsr3() && !fsr1();
}

// The first DLSS feature of a process costs seconds, the model loading and its shaders compiling, and it fell into the
// first frame of the scene (2.3 s measured). One feature made and dropped before the guest starts leaves the next
// creations some 100 ms (the guest is not running: the graphics queue is free, this thread submits alone).
void vk_dlaa_warm(void) {
    if (!vk_dlaa_available() || use_streamline_sr || fsr3() || fsr1() || !cfg_bool("REKARINA_DLAA") || !rt_knob("DLAA_WARM", "1")) return;
    VkCtx *c = vk_ctx();
    uint64_t t0 = plat_now_ns();
    VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                  .flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT, .queueFamilyIndex = c->gfx_family};
    VkCommandPool pool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(c->device, &pi, NULL, &pool) != VK_SUCCESS) return;
    VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer cb = VK_NULL_HANDLE;
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    NgxParam *p = NULL;
    NgxHandle *feat = NULL;
    VkFence fence = VK_NULL_HANDLE;
    VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkAllocateCommandBuffers(c->device, &ai, &cb) == VK_SUCCESS && vkBeginCommandBuffer(cb, &bi) == VK_SUCCESS &&
        NGX_OK(N.alloc_params(&p)) && vkCreateFence(c->device, &fi, NULL, &fence) == VK_SUCCESS) {
        int pv = preset_value(), pk = pv ? pv : 11;
        p_setu(p, "CreationNodeMask", 1);
        p_setu(p, "VisibilityNodeMask", 1);
        p_setu(p, "Width", 1707);
        p_setu(p, "Height", 960);
        p_setu(p, "OutWidth", 2560);
        p_setu(p, "OutHeight", 1440);
        p_seti(p, "PerfQualityValue", 2);
        p_seti(p, "DLSS.Feature.Create.Flags", (1 << 0) | (1 << 1) | (1 << 6) | dbg_feat_flags());
        p_seti(p, "DLSS.Enable.Output.Subrects", 0);
        p_setu(p, "DLSS.Hint.Render.Preset.Quality", (unsigned)pk);
        NgxResult r = N.create(cb, NGX_FEATURE_SUPERSAMPLING, p, &feat);
        vkEndCommandBuffer(cb);
        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
        if (vkQueueSubmit(c->gfx_queue, 1, &si, fence) == VK_SUCCESS)
            vkWaitForFences(c->device, 1, &fence, VK_TRUE, UINT64_MAX);
        rt_log("dlaa: DLSS warmed before the guest starts in %llu ms (result %#x)",
               (unsigned long long)((plat_now_ns() - t0) / 1000000), (unsigned)r);
        if (feat) N.release(feat);
    }
    if (fence) vkDestroyFence(c->device, fence, NULL);
    vkDestroyCommandPool(c->device, pool, NULL);
}

// FSR 1's texture bias: log2 of the scene's share of the picture, AMD's figure for FSR 1 (-0.58 at the quality mode's
// 67 %). A temporal upscaler takes one level more (res_dlss_lod_bias); FSR 1 has no history to hide the shimmer that
// would bring. OoT's share comes from frame_rate.c (res_fixed_scene_share), TotK's from resolution.c.
double res_fixed_scene_share(void);
static float fsr1_bias(void) {
    double s = res_fixed_scene_share();
    if (s > 0) return s < 1 ? (float)log2(s) : 0.0f;
    return res_dlss_upscales() ? res_dlss_lod_bias() + 1.0f : 0.0f;
}

// NVIDIA's DLSS guide asks for log2(render / output) - 1 on top of the game's texture LOD bias whenever sub-pixel
// temporal jitter is active (-1.0 at DLAA, -1.58 in Quality, -2.0 in Performance). OoT's live render share comes from
// frame_rate.c (res_fixed_scene_share); TotK's comes from resolution.c.
static float dlss_lod_bias(void) {
    if (fsr1()) return fsr1_bias();
    static float extra = 1e9f;
    if (extra > 1e8f) {
        const char *e = cfg_get("REKARINA_DLSS_BIAS_EXTRA");
        extra = e && *e ? (float)atof(e) : -1.0f;
    }
    int jit = scene_projection_enabled() || rt_knob("DLAA_JITTER", "1");
    double s = res_fixed_scene_share();
    if (s > 0) return (s < 1.0 ? (float)log2(s) : 0.0f) + (jit ? extra : 0.0f);
    return res_dlss_upscales() ? res_dlss_lod_bias() : (jit ? extra : 0.0f);
}

// REKARINA_DLSS, read at each present (the menu switches it live): the scene size of the mode (resolution.c) and the
// texture bias that goes with it while DLSS or FSR runs.
static void dlss_mode(void) {
    static char mode[32];
    static float last_bias = 1e9f;
    const char *m = cfg_value("REKARINA_DLSS");
    if (strcmp(m, mode)) {
        snprintf(mode, sizeof mode, "%s", m);
        res_dlss_mode(m);
    }
    float bias = dl_active() ? dlss_lod_bias() : 0.0f;
    if (bias != last_bias) {
        if (last_bias < 1e8f || bias != 0.0f)
            rt_log("dlaa: texture LOD bias %.2f (mode %s)", bias, mode[0] ? mode : "dlaa");
        last_bias = bias;
        vk_sampler_dlss_bias(bias);
    }
    {   // DEBUG-SHARPAT: REKARINA_SHARPNESS_AT=<frame>:<value> sets the slider at that present (an A/B in one run)
        static int at = -2, val;
        static uint64_t frame;
        if (at == -2) {
            const char *e = cfg_get("REKARINA_SHARPNESS_AT");
            if (!e || sscanf(e, "%d:%d", &at, &val) != 2) at = -1;
        }
        if (at >= 0 && ++frame == (uint64_t)at) {
            char v[16];
            snprintf(v, sizeof v, "%d", val);
            cfg_set("REKARINA_SHARPNESS", v);
        }
    }
#ifdef GAME_OOT
    const double strength = cfg_double("REKARINA_TEXTURE_LOD_STRENGTH"), full = 100;
#else
    const CfgRenderValues *settings = cfg_render_values();
    const int strength = settings ? settings->sharpness : cfg_int("REKARINA_SHARPNESS"), full = 80;
#endif
    float take = strength >= full ? 0.0f : strength <= 0 ? 0.5f : 0.5f * (float)(full - strength) / (float)full;
    static float took = -1.0f;
    if (take != took) {
        took = take;
        vk_sampler_sharpness(take);
        rt_log("dlaa: the game's texture LOD bias (negative biases moved toward zero by %.3f)", take);
    }
}


// The preset, read again when a setting changed (the menu); a different one recreates the features at the next DLAA run.
static void preset_check(void) {
    unsigned v = cfg_version();
    if (v == D.cfg_seen) return;
    D.cfg_seen = v;
    int p = preset_value();
    if (p == D.preset) return;
    if (D.evals) {
        if (p) rt_log("dlaa: DLSS preset %c in every mode; the features are created again", 'A' + p - 1);
        else rt_log("dlaa: DLSS presets per mode; the features are created again");
    }
    D.preset = p, D.preset_dirty = 1, D.reset = 1;
}

void vk_dlaa_frame_settings(void) {
    static uint64_t seen_frame = UINT64_MAX;
    static unsigned seen_settings = ~0u;
    static unsigned seen_consumers = ~0u;
    const CfgRenderValues *settings = cfg_render_values();
    const unsigned revision = cfg_version();
    const unsigned consumers = settings ? settings->consumers : 0;
    if (seen_frame == D.presents && seen_settings == revision && seen_consumers == consumers) return;
    if (seen_settings != revision) vk_sampler_anisotropy();
    seen_frame = D.presents;
    seen_settings = revision;
    seen_consumers = consumers;
    motion_history_check();
    preset_check();
    int wanted = settings ? settings->dlaa : cfg_bool("REKARINA_DLAA");
    int on = settings ? !!(settings->consumers & CFG_RENDER_UPSCALER) : wanted && vk_dlaa_backend_available();
#ifdef GAME_OOT
    static int previous_backend = -1, previous_wanted = -1;
    static char previous_fsr_version[32];
    int backend = effective_taa();
    const char *fsr_version = settings ? settings->fsr_version : cfg_value("REKARINA_FSR_VERSION");
    int changed = backend != previous_backend || (backend == CFG_TAA_FSR3 && strcmp(fsr_version, previous_fsr_version));
    if (changed) snprintf(previous_fsr_version, sizeof previous_fsr_version, "%s", fsr_version);
    if (changed || on != D.on) {
        previous_backend = backend;
        D.failed = 0;
        D.last_eval_ns = 0;
        __atomic_store_n(&D.fg_fresh, 0, __ATOMIC_RELEASE);
        D.fg_cam_ok = 0;
    }
    if (wanted && backend != CFG_TAA_BILINEAR && (changed || on != D.on || wanted != previous_wanted))
        rt_log("dlaa: %s %s", backend == CFG_TAA_DLSS ? "NVIDIA DLSS" : backend == CFG_TAA_FSR1 ? "AMD FSR 1" : "AMD FSR",
               !on ? "unavailable; no replacement upscaler" : backend != cfg_taa() ? "selected because DLSS is unavailable" : "selected");
    previous_wanted = wanted;
#else
    const int changed = 0;
#endif
    if (changed || on != D.on) {
        if (on != D.on) rt_log("dlaa: %s", on ? "on" : "off");
        D.on = on;
        scene_projection_feedback(NULL, 0, 0, 0);
        D.projection_history.token = 0;
        D.projection = (p_scene_projection){0};
        D.projection_resolved = D.projection_conflict = 0;
        // either way the next DLAA frame starts a new history from an unjittered frame
        D.reset = 1, D.cam_prev_ok = 0, D.jit_ok = D.jit_used = D.scene.armed = D.scene.gbuf = 0;
        D.scene.phase = SCENE_DRAWING;
        D.hold = 0;
        D.cam_addr = 0, D.nviews = 0;
    }
    dlss_mode();
}

void vk_dlaa_frame(void) {
    D.nvidia_scene_aa = NULL;
    D.scene.replay = (SceneReplay){0};
    D.scene.capture_key = (SceneReplayKey){0};
    D.scene.capture_source = D.scene.capture_copy = VK_NULL_HANDLE;
    D.scene.capture_copy_address = 0;
    D.scene.clean_output = VK_NULL_HANDLE;
    D.scene.clean_output_address = 0;
    D.scene.tonemap_serial = D.scene.packed_serial = 0;
#ifdef GAME_OOT
    D.scene.skip_aa_output = NULL;
#endif
    D.scene.stock_pack = D.scene.stock_output = NULL;
    if (D.scene.phase == SCENE_PREBLOOM) D.scene.phase = SCENE_RESOLVED;
    D.scene.early_attempted = 0;
    memset(D.scene.bloom_guest_subresources, 0, sizeof D.scene.bloom_guest_subresources);
    D.scene.tonemapped = D.scene.packed = VK_NULL_HANDLE;
    vk_dlaa_swap_img = NULL, vk_dlaa_swap_view = VK_NULL_HANDLE;
    __atomic_store_n(&D.presents, D.presents + 1, __ATOMIC_RELAXED);
    D.pre_bloom.tried = 0;
    D.pre_bloom.resolved = 0;
    D.pre_bloom.staged = 0;
    D.pre_bloom.stage_why = NULL;
    if (D.scene.hudless.pending) D.hl_missed++, D.scene.hudless.pending = 0;   // no pass of the HUD or the window copy came: no HUD-less picture
    hudless_scene_target(NULL, VK_NULL_HANDLE);
    if (!dl_active() && !vk_dlaa_fg_only() && !D.projection.sample.token) return;
    if (dl_active() && !scene_resolved()) D.frames_missed++;
    if (D.jit_used && !scene_resolved()) D.hold = 60, D.holds++;   // jittered but not resolved: that frame shook
    else if (D.hold) D.hold--;
    if (!scene_resolved()) D.jit_ok = 0;
    D.scene.armed = D.jit_used = D.scene.gbuf = 0;
    D.scene.phase = SCENE_DRAWING;
    if ((D.projection.sample.token || D.projection_history.token) && !D.projection_resolved) {
        scene_projection_feedback(NULL, 0, 0, 0);
        D.projection_history.token = 0;
        D.cam_prev_ok = 0, D.reset = 1;
        D.fg_fresh = D.fg_cam_ok = 0;
    }
    D.projection = (p_scene_projection){0};
    D.projection_resolved = D.projection_conflict = 0;
    D.pj_n = 0;
    SJ.stamp++, SJ.nctx = 0;
    D.cam_addr = 0, D.nviews = 0;
}

// ---------------------------------------------------------------- the game's FXAA (REKARINA_FXAA, live)
// the game's FXAA is two full-screen passes after the bloom, just before the HUD: one copies the scene target into a
// scratch target, the other draws the anti-aliased result back over the scene (found by diffing frame dumps with the
// game's own switch, ModelSceneExtensionParam EnableFXAA, on and off). With fxaa = 0 the draws of that second pass's
// fragment shader are skipped, so the scene reaches the HUD as it was. Shaders are known by the FNV-1a hash of their
// SPIR-V, as vk_dump.c names them; a translator change gives new hashes and the switch then does nothing.
// With a scene smaller than the output (the game's own 5/6 render scale, or a DLSS mode) the same shader draws the
// scaled picture into the output-sized target, in a pass without the depth attachment the in-place FXAA has: there it
// is always drawn, whatever fxaa says (skipping it would leave the 3D picture out), unless DLSS already wrote that
// target this frame.
// The stock translator's hash, the one with tools/xlate patch 0010 (PCSS, PR #89) applied, the one with patches 0010
// and 0011 (the shadow filter spread) applied, and that one with the zero_locals job flag (Mesa, vk_caps.c).
// A scaling draw this list misses still scales DLSS's picture, not
// the jittered scene: after DLSS ran, passes larger than the light target sample DLSS's output (vk_dlaa_swap_img).
static const uint64_t FXAA_FS[] = {0xf450003ebeb3dc52ull, 0x0a12f4bc2a9a4fbeull, 0x5c43ae18948d52cbull,
                                   0xaad42238801d31d8ull};

static uint64_t spirv_hash(const GfxShader *s) {
    // per thread: the draws of a stretch come from several (vk_exec.c par_try), and an entry is two words
    static _Thread_local struct { const GfxShader *s; uint64_t h; } tab[8192];
    uintptr_t k = ((uintptr_t)s >> 4) * 0x9E3779B97F4A7C15ull;
    for (uint32_t i = 0, j = (uint32_t)(k >> 51); i < 8192; i++, j = (j + 1) & 8191) {
        if (tab[j].s == s) return tab[j].h;
        if (!tab[j].s) {
            uint64_t words = 0, h = 0xcbf29ce484222325ull;
            const uint8_t *b = (const uint8_t *)gfx_shader_spirv(s, &words);
            for (uint64_t n = 0; b && n < words * 4; n++) h = (h ^ b[n]) * 0x100000001b3ull;
            tab[j].s = s, tab[j].h = h;
            return h;
        }
    }
    return 0;   // table full: no skip
}

int vk_dlaa_skip_draw(VkExec *ex, const GfxShader *fs) {
    if (pre_bloom_on && !vk_par_worker && !vk_dlaa_prebloom_enabled()) pre_bloom_draw(ex, fs);   // the replay thread's draws keep the frame's order
    static uint64_t frame_read = ~0ull;
    static int fxaa = 1;
    // fxaa = 0 reports once whether it found the game's FXAA draw: the first skip, or a minute of frames (3600) with
    // none, which is what a translator change that moves FXAA_FS looks like on every GPU, with or without DLSS
    static uint64_t fx_off_frames;
    static int fx_found, fx_reported;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (f != frame_read) {   // once a frame: the menu switches it live
        frame_read = f, fxaa = cfg_bool("REKARINA_FXAA");
        if (!fxaa && !fx_found && !fx_reported && ++fx_off_frames == 3600) {
            fx_reported = 1;
            rt_log("dlaa: fxaa = 0 found no draw of the game's FXAA in %llu frames (its fragment shader is not in "
                   "FXAA_FS): the FXAA stays on", (unsigned long long)fx_off_frames);
        }
    }
    if (fxaa && !scene_scaled()) return 0;   // nothing to skip: no shader lookup per draw at the defaults
    if (!fs) return 0;
    uint64_t h = spirv_hash(fs);
    int known = 0;
    for (size_t i = 0; i < sizeof FXAA_FS / sizeof FXAA_FS[0]; i++) known |= h == FXAA_FS[i];
    if (!known) return 0;
    if (!ex->depth.image) {   // the scaling pass, which DLSS replaced
        static int told;
        if (scene_scaled() && !told++) rt_log("dlaa: the game's scaling draw left out, DLSS wrote it (fragment shader %016llx)",
                                    (unsigned long long)h);
        return scene_scaled();
    }
    if (!fxaa && !fx_found) {
        fx_found = 1;
        rt_log("dlaa: fxaa = 0 skips the game's FXAA (fragment shader %016llx)", (unsigned long long)h);
    }
    return !fxaa;
}

