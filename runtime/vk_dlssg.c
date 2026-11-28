// DLSS Frame Generation (REKARINA_FRAMEGEN = dlss; applies after a restart) through rekarina_dlssg.dll
// (runtime/dlssg/nx2_sl.cpp), which drives NVIDIA Streamline. Like AMD's frame generation (vk_fg.cpp) it needs DLSS or
// FSR 3.1 upscaling on, whose depth, motion vectors and camera it hands over, and an RTX 40 or 50 GPU.
//
// The DLL and Streamline (sl.interposer.dll and its plugins, nvngx_dlssg.dll) sit next to the game. The instance and
// device are created through Streamline's proxies (vk_device.c), so it can add the extensions, features and queues
// DLSS-G needs; the present path calls Streamline's swapchain functions (gfx_present.c), and every present records the
// frame's constants and tags first (vk_dlssg_frame). Generation pauses while the menu is open and in the game's frames
// without a camera (its pause screens).
//
// DLSS-G needs Reflex, whose low-latency mode paces the game (REKARINA_REFLEX, default 1; 0 off, 2 on with boost): the
// game's frame sleeps in Reflex where it starts (its acquire of a window image), and each frame carries the latency
// markers from that start to the host present, all with the frame's number, the count of the game's presents up to it.
// The mode is the swapchain's, so it is set again after each one (vk_dlssg_create_swapchain).
#ifdef _WIN32
#include <windows.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "vk.h"
#include "vk_dlaa.h"
#include "vk_dlssg.h"
#include "vk_fg.h"
#include "platform/platform.h"
#include "report.h"
#include "dlssg/nx2_sl.h"

static struct {
    int wanted, loaded, active;
    int initialized;    // Streamline still needs shutdown if device creation disables interposition.
    int asked;          // REKARINA_FRAMEGEN was dlss at start (vk_dlssg_wanted)
    int retains;        // the library keeps generation's resources while it is off (interface version 4 and later)
    int avail;          // Streamline's DLSS-G plugin runs on this card (vk_dlssg_device_ready; 0 before)
    int sr_requested;   // Streamline was initialized with kFeatureDLSS (v5+)
    int sr_ready;       // Streamline DLSS Super Resolution is ready
    int reflex_applied; // last Reflex mode passed to Streamline (-1 unset)
    const char *why;    // why it is not running although asked, for the menu's row (vk_dlssg_off_why)
    PFN_nx2sl_init init;
    PFN_nx2sl_sr_available sr_available;
    PFN_nx2sl_sr_evaluate_mode sr_evaluate;
    PFN_nx2sl_instance_proc gipa;
    PFN_nx2sl_device_proc gdpa;
    PFN_nx2sl_available available;
    PFN_nx2sl_frame frame;
    PFN_nx2sl_frame_with_interface frame_with_interface;
    PFN_nx2sl_present_marker marker;
    PFN_nx2sl_sleep sleep;
    PFN_nx2sl_submit_end submit_end;
    PFN_nx2sl_shown shown;
    PFN_nx2sl_reflex reflex;
    uint32_t game;   // the game's presents (vk_dlssg_game_present)
    uint32_t cur;    // the frame the present thread blits and presents next (vk_dlssg_present_frame)
    uint32_t started;      // the last frame vk_dlssg_frame_start started
    uint64_t hosted_ns;    // kern_now_ns after the last host present
    PFN_nx2sl_shutdown shutdown;
    PFN_vkCreateSwapchainKHR create_sc;
    PFN_vkDestroySwapchainKHR destroy_sc;
    PFN_vkGetSwapchainImagesKHR images;
    PFN_vkAcquireNextImageKHR acquire;
    PFN_vkQueuePresentKHR present;
} S = {.wanted = -1, .reflex_applied = -1};

static void sl_line(const char *line) { rt_log("%s", line); }

static int reflex_mode(void) {
    int m = cfg_int("REKARINA_REFLEX");
    if (m < 0) m = 0;
    if (m > 2) m = 2;
    // DLSS Frame Generation requires Reflex enabled while active; keep Boost if chosen.
    if (vk_dlssg_active() && m == 0) m = 1;
    return m;
}

int vk_dlssg_reflex_available(void) { return S.active && S.reflex != NULL; }

void vk_dlssg_reflex_sync(void) {
    if (!S.active || !S.reflex) return;
    int m = reflex_mode();
    if (m == S.reflex_applied) return;
    S.reflex_applied = m;
    S.reflex(m);
}

int vk_dlssg_wanted(void) {
    if (S.wanted < 0) {
        const char *fg = cfg_value("REKARINA_FRAMEGEN");
        S.asked = !strcmp(fg, "dlss");
#ifdef _WIN32
        int want = S.asked || vk_dlaa_prepare_sl_sr();
#ifndef GAME_OOT
        int fsr3_fg = !strcmp(fg, "fsr3");
        want = S.asked || (!fsr3_fg && (want || (vk_dlaa_wanted() && !vk_dlaa_fsr1())));
#endif
#ifdef GAME_OOT
        if (want && fg_guard_backend_blocked(FG_GUARD_DLSS)) {
#else
        if (S.asked && fg_guard_blocked()) {
#endif
            S.why = FG_GUARD_WHY;
            want = 0;
#ifndef GAME_OOT
        } else if (S.asked && !vk_dlaa_wanted()) {
            rt_log("dlssg: frame generation needs DLSS or FSR 3.1 upscaling (any mode): off");
            S.why = "needs DLSS / DLAA on";
            want = 0;
        } else if (fg_guard_blocked()) {
            want = 0;
#endif
        }
        S.wanted = want;
#else
        S.wanted = 0;
#endif
    }
    return S.wanted;
}

int vk_dlssg_early(void) {
    if (!vk_dlssg_wanted()) return 0;
    if (S.loaded) return S.loaded > 0;
    S.loaded = -1;
#ifndef GAME_OOT
    if (S.asked) fg_guard_arm(FG_GUARD_DLSS);
#endif
#ifdef _WIN32
#ifdef GAME_OOT
    // Loading the bridge can run DLL initialization even when frame generation is off.
    fg_guard_arm(FG_GUARD_DLSS);
#endif
    wchar_t dir[MAX_PATH], path[MAX_PATH + 32];
    DWORD n = GetModuleFileNameW(NULL, dir, MAX_PATH);
    while (n && dir[n - 1] != L'\\') n--;
    dir[n ? n - 1 : 0] = 0;
    swprintf(path, MAX_PATH + 32, L"%ls\\%hs_dlssg.dll", dir, RT_FILE_STEM);
    HMODULE m = NULL;
    {   // DEBUG-DLSSGDLL: REKARINA_DLSSG_DLL=<path> loads our library from there (a test of a new build of it beside
        // a game folder whose own copy other builds still use); Streamline's files stay the game folder's
        const char *e = cfg_get("REKARINA_DLSSG_DLL");
        if (e && *e) {
            wchar_t w[MAX_PATH];
            if (MultiByteToWideChar(CP_UTF8, 0, e, -1, w, MAX_PATH) > 0 && (m = LoadLibraryW(w)))
                rt_log("dlssg: our library from %s (REKARINA_DLSSG_DLL)", e);
        }
    }
    if (!m) m = LoadLibraryW(path);
#ifdef RT_OLD_FILE_STEM   // a folder packaged before the file was renamed, or a build folder
    if (!m) {
        swprintf(path, MAX_PATH + 32, L"%ls\\%hs_dlssg.dll", dir, RT_OLD_FILE_STEM);
        m = LoadLibraryW(path);
    }
#endif
    if (!m) {
#ifdef GAME_OOT
        fg_guard_disarm(FG_GUARD_DLSS);
#endif
        if (S.asked) {
            rt_log("dlssg: " RT_FILE_STEM "_dlssg.dll is not next to the game: DLSS Frame Generation off");
            S.why = RT_FILE_STEM "_dlssg.dll is missing (run the setup again)";
        }
        return 0;
    }
    PFN_nx2sl_version ver = (PFN_nx2sl_version)(void *)GetProcAddress(m, "nx2sl_version");
    S.init = (PFN_nx2sl_init)(void *)GetProcAddress(m, "nx2sl_init");
    S.gipa = (PFN_nx2sl_instance_proc)(void *)GetProcAddress(m, "nx2sl_instance_proc");
    S.gdpa = (PFN_nx2sl_device_proc)(void *)GetProcAddress(m, "nx2sl_device_proc");
    S.available = (PFN_nx2sl_available)(void *)GetProcAddress(m, "nx2sl_available");
    S.frame = (PFN_nx2sl_frame)(void *)GetProcAddress(m, "nx2sl_frame");
    S.marker = (PFN_nx2sl_present_marker)(void *)GetProcAddress(m, "nx2sl_present_marker");
    S.sleep = (PFN_nx2sl_sleep)(void *)GetProcAddress(m, "nx2sl_sleep");
    S.submit_end = (PFN_nx2sl_submit_end)(void *)GetProcAddress(m, "nx2sl_submit_end");
    S.shown = (PFN_nx2sl_shown)(void *)GetProcAddress(m, "nx2sl_shown");
    S.shutdown = (PFN_nx2sl_shutdown)(void *)GetProcAddress(m, "nx2sl_shutdown");
    S.reflex = (PFN_nx2sl_reflex)(void *)GetProcAddress(m, "nx2sl_reflex");
    int v = ver ? ver() : 0;
    S.frame_with_interface = v >= 6 ? (PFN_nx2sl_frame_with_interface)(void *)GetProcAddress(m, "nx2sl_frame_with_interface") : NULL;
    S.retains = v >= 4;
    if (v < NX2SL_VERSION_OLDEST || v > NX2SL_VERSION || !S.init || !S.gipa || !S.gdpa || !S.available || !S.frame ||
        !S.marker || !S.sleep || !S.submit_end || !S.shown || !S.reflex) {
#ifdef GAME_OOT
        fg_guard_disarm(FG_GUARD_DLSS);
#endif
        if (S.asked) {
            rt_log("dlssg: " RT_FILE_STEM "_dlssg.dll is from another build: DLSS Frame Generation off");
            S.why = RT_FILE_STEM "_dlssg.dll is from another build (run the setup again)";
        }
        return 0;
    }
    // Older v3/v4 DLLs only support DLSS-G when explicitly requested at startup; do not interpose for framegen = off.
    if (v < 5 && !S.asked) {
#ifdef GAME_OOT
        fg_guard_disarm(FG_GUARD_DLSS);
#endif
        return 0;
    }
    int sr = v >= 5 && vk_dlaa_prepare_sl_sr();
    if (v >= 5) {
        S.sr_available = (PFN_nx2sl_sr_available)(void *)GetProcAddress(m, "nx2sl_sr_available");
        S.sr_evaluate = (PFN_nx2sl_sr_evaluate_mode)(void *)GetProcAddress(m, "nx2sl_sr_evaluate_mode");
        if (!S.sr_available || !S.sr_evaluate) sr = 0;
        // Require sl.dlss.dll beside the executable when requesting Streamline SR; otherwise fall back to direct NGX.
        if (sr) {
            wchar_t sldlss[MAX_PATH + 32];
            swprintf(sldlss, MAX_PATH + 32, L"%ls\\sl.dlss.dll", dir);
            if (GetFileAttributesW(sldlss) == INVALID_FILE_ATTRIBUTES) sr = 0;
        }
        int fg = 1;
        if (!S.init(dir, sl_line, sr, fg)) {
#ifdef GAME_OOT
            fg_guard_disarm(FG_GUARD_DLSS);
#endif
            if (S.asked) S.why = "NVIDIA Streamline did not start (the dlssg lines of the log say why)";
            return 0;
        }
        S.sr_requested = sr;
    } else {
        Nx2SlInitV4 init_v4 = (Nx2SlInitV4)(void *)S.init;
        if (!init_v4(dir, sl_line)) {
#ifdef GAME_OOT
            fg_guard_disarm(FG_GUARD_DLSS);
#endif
            S.why = "NVIDIA Streamline did not start (the dlssg lines of the log say why)";
            return 0;
        }
    }
    S.initialized = S.loaded = 1;
    return 1;
#else
    return 0;
#endif
}

int vk_dlssg_interposed(void) { return S.loaded > 0; }

VkResult vk_dlssg_create_instance(const VkInstanceCreateInfo *ci, VkInstance *out) {
    PFN_vkCreateInstance f = S.loaded > 0 ? (PFN_vkCreateInstance)S.gipa()(VK_NULL_HANDLE, "vkCreateInstance") : NULL;
    return f ? f(ci, NULL, out) : vkCreateInstance(ci, NULL, out);
}

// An instance function through the interposer while Streamline is loaded, else NULL (the caller takes the loader's).
// The window surface goes through it: Streamline learns the window a swapchain shows from vkCreateWin32SurfaceKHR, and
// without that its swapchain hook found "no window corresponding to this application" and generated no frames.
PFN_vkVoidFunction vk_dlssg_instance_proc(VkInstance inst, const char *name) {
    return S.loaded > 0 ? S.gipa()(inst, name) : NULL;
}

// The interposer's vkCreateDevice finds the device's instance in a map its own vkEnumeratePhysicalDevices fills; a
// physical device enumerated through the loader alone left it a null instance, so the plugins got an empty instance
// table and sl.common called a null GetPhysicalDeviceMemoryProperties while it set up. Enumerate through it as well.
VkResult vk_dlssg_enumerate(VkInstance inst, uint32_t *n, VkPhysicalDevice *pd) {
    PFN_vkEnumeratePhysicalDevices f =
        S.loaded > 0 ? (PFN_vkEnumeratePhysicalDevices)S.gipa()(inst, "vkEnumeratePhysicalDevices") : NULL;
    return f ? f(inst, n, pd) : vkEnumeratePhysicalDevices(inst, n, pd);
}

VkResult vk_dlssg_create_device(VkInstance inst, VkPhysicalDevice pd, const VkDeviceCreateInfo *ci, VkDevice *out) {
    PFN_vkCreateDevice f = S.loaded > 0 ? (PFN_vkCreateDevice)S.gipa()(inst, "vkCreateDevice") : NULL;
    VkResult r = f ? f(pd, ci, NULL, out) : vkCreateDevice(pd, ci, NULL, out);
    if (r != VK_SUCCESS && f) {
        // Streamline refused the device: it asks for NVIDIA's optical flow extension, which a Radeon or Intel GPU lacks
        // ("Required device extension VK_NV_optical_flow unsupported", -7), and rekarina.ini still says framegen =
        // dlss after a move from an RTX card, on a laptop running on its integrated GPU, or with another GPU chosen.
        // Streamline has to be loaded before the first Vulkan call, so the device was not known then. The game made no
        // device and ended with "no Vulkan device"; now the device is made again through the loader (the instance and
        // the physical device are the loader's: the interposer only passed them on), and AMD FSR frame generation takes
        // DLSS's place (vk_dlssg_device_ready).
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pd, &p);
        rt_log("dlssg: NVIDIA Streamline refused a device on %s (%d): Streamline off, the device is made "
               "without Streamline", p.deviceName, (int)r);
        S.loaded = -1;
        S.sr_requested = S.sr_ready = 0;
        if (S.asked) S.why = "NVIDIA Streamline refused this GPU: DLSS Frame Generation needs an RTX 40 or 50 card";
        r = vkCreateDevice(pd, ci, NULL, out);
    }
    return r;
}

// REKARINA_DLSSG_UNAVAILABLE=1 simulates unavailable DLSS Frame Generation: OoT leaves it off;
// TotK can exercise its legacy AMD fallback.
static int test_unavailable(void) {
    static int v = -1;
    if (v < 0) v = cfg_bool("REKARINA_DLSSG_UNAVAILABLE") != 0;
    return v;
}

void vk_dlssg_device_ready(VkDevice dev) {
    if (S.loaded <= 0) {
        fg_guard_disarm(FG_GUARD_DLSS);
#ifndef GAME_OOT
        if (S.asked && S.wanted) vk_fg_take_over(S.why ? S.why : "DLSS Frame Generation did not load");
#endif
        return;
    }
    S.sr_ready = S.sr_requested && S.sr_available && S.sr_evaluate && S.sr_available();
    PFN_vkGetDeviceProcAddr g = S.gdpa();
    S.present = (PFN_vkQueuePresentKHR)g(dev, "vkQueuePresentKHR");
    S.avail = S.available() && !test_unavailable();
    if (!S.avail) {   // Streamline's DLSS-G plugin refused this GPU, driver or system (or the test knob)
        fg_guard_disarm(FG_GUARD_DLSS);
        if (S.asked) {
            rt_log("dlssg: DLSS Frame Generation off: it needs an RTX 40 or 50 series card, a current NVIDIA driver and "
                   "Windows' hardware-accelerated GPU scheduling on (Settings > Display > Graphics)%s",
                   test_unavailable() ? " (REKARINA_DLSSG_UNAVAILABLE)" : "");
            S.why = "not available: it needs an RTX 40 / 50 card, a current driver and hardware-accelerated GPU scheduling";
#ifndef GAME_OOT
            vk_fg_take_over("no RTX 40 / 50 card, an older driver or GPU scheduling off");
#endif
        }
        return;
    }
    S.create_sc = (PFN_vkCreateSwapchainKHR)g(dev, "vkCreateSwapchainKHR");
    S.destroy_sc = (PFN_vkDestroySwapchainKHR)g(dev, "vkDestroySwapchainKHR");
    S.images = (PFN_vkGetSwapchainImagesKHR)g(dev, "vkGetSwapchainImagesKHR");
    S.acquire = (PFN_vkAcquireNextImageKHR)g(dev, "vkAcquireNextImageKHR");
    S.active = S.create_sc && S.destroy_sc && S.images && S.acquire && S.present;
    if (!S.active) fg_guard_disarm(FG_GUARD_DLSS);
    rt_log("dlssg: %s", S.active ? "DLSS Frame Generation ready for live selection" : "the swapchain hooks are missing: off");
    if (!S.active && S.asked) S.why = "Streamline's swapchain hooks are missing";
    if (S.active) {
        S.reflex_applied = reflex_mode();
        S.reflex(S.reflex_applied);
    }
#ifdef GAME_OOT
    if (!S.asked) fg_guard_disarm(FG_GUARD_DLSS);
#endif
}

int vk_dlssg_swapchain_ready(void) { return S.active; }
int vk_dlssg_active(void) {
    if (!S.active || vk_fg_swapchain()) return 0;
    // UI coverage checks this during draw replay; resolve the setting only after configuration changes.
    static _Thread_local unsigned seen;
    static _Thread_local int selected = -1;
    unsigned version = cfg_version();
    if (selected < 0 || seen != version) {
        selected = !strcmp(cfg_value("REKARINA_FRAMEGEN"), "dlss");
        seen = version;
    }
    return selected;
}

// The menu's frame generation row: whether DLSS Frame Generation can run here. Streamline's own answer when it was
// loaded at start; else an NVIDIA card of the RTX 40 or 50 series (Ada and later: the generation's optical flow runs
// there; a Turing Quadro RTX 4000 is not one) with our library and Streamline's beside the game. 0 with *why.
int vk_dlssg_offered(const char **why) {
    if (test_unavailable() || (S.loaded > 0 && !S.avail)) {
        *why = "DLSS frame generation needs an RTX 40 or 50 card";
        return 0;
    }
    if (S.loaded > 0) return 1;
    const VkCtx *c = vk_ctx();
    const char *name = c ? c->props.deviceName : "";
    int rtx = c && c->props.vendorID == 0x10de && !strstr(name, "Quadro") &&
              (strstr(name, "RTX 40") || strstr(name, "RTX 50") || strstr(name, "Ada") || strstr(name, "RTX PRO"));
    if (!rtx) {
        *why = "DLSS frame generation needs an RTX 40 or 50 card";
        return 0;
    }
    static int have = -1;   // the files do not change while the game runs: looked at once
    if (have < 0) {
        char dir[PLAT_PATH_MAX], path[PLAT_PATH_MAX + 64];
        int n = plat_exe_path(dir, sizeof dir);
        char *cut = n ? plat_last_sep(dir) : NULL;
        have = 1;
        if (cut) {
            *cut = 0;
            static const char *const files[] = {"sl.interposer.dll", "sl.dlss_g.dll", "nvngx_dlssg.dll"};
            for (size_t i = 0; i < sizeof files / sizeof *files && have; i++)
                snprintf(path, sizeof path, "%s/%s", dir, files[i]), have = plat_file_exists(path);
            snprintf(path, sizeof path, "%s/" RT_FILE_STEM "_dlssg.dll", dir);
            int ours = plat_file_exists(path);
#ifdef RT_OLD_FILE_STEM
            snprintf(path, sizeof path, "%s/" RT_OLD_FILE_STEM "_dlssg.dll", dir);
            ours = ours || plat_file_exists(path);
#endif
            have = have && ours;
        }
    }
    if (have) return 1;
    *why = "DLSS frame generation's libraries are missing (run the setup again)";
    return 0;
}

const char *vk_dlssg_off_why(void) {
    if ((!S.asked && !S.why) || S.active || (vk_fg_instead() && vk_fg_active())) return NULL;
    return S.why ? S.why : "it did not start (the dlssg lines of the log say why)";
}

VkResult vk_dlssg_create_swapchain(VkDevice d, const VkSwapchainCreateInfoKHR *ci, VkSwapchainKHR *sc) {
#ifdef GAME_OOT
    // Streamline owns this hook even when frame generation is off.
    const int guard_setup = !vk_dlssg_active();
    if (guard_setup) fg_guard_arm(FG_GUARD_DLSS);
#endif
    VkResult r = S.create_sc(d, ci, NULL, sc);
    if (r == VK_SUCCESS) vk_dlssg_reflex_sync();
#ifdef GAME_OOT
    if (guard_setup) fg_guard_disarm(FG_GUARD_DLSS);
#endif
    return r;
}
void vk_dlssg_destroy_swapchain(VkDevice d, VkSwapchainKHR sc) { S.destroy_sc(d, sc, NULL); }
VkResult vk_dlssg_get_images(VkDevice d, VkSwapchainKHR sc, uint32_t *n, VkImage *img) { return S.images(d, sc, n, img); }
VkResult vk_dlssg_acquire(VkDevice d, VkSwapchainKHR sc, uint64_t t, VkSemaphore sem, uint32_t *idx) {
    return S.acquire(d, sc, t, sem, VK_NULL_HANDLE, idx);
}

VkResult vk_dlssg_present(VkQueue q, const VkPresentInfoKHR *pi) {
    if (!S.active || !S.present) return vkQueuePresentKHR(q, pi);
    S.marker(0, S.cur);
    VkResult r = S.present(q, pi);
    S.marker(1, S.cur);
    __atomic_store_n(&S.hosted_ns, kern_now_ns(), __ATOMIC_RELAXED);
    return r;
}

// Reflex holds the frame's start until the time the driver computes from the measured render and queue times, so the
// game's CPU work ends about when the GPU is free for it; then the frame's first markers. Once a frame: the game may
// acquire again before it presents. No hold while no host present went out for HOLD_IDLE_NS (the window minimized:
// gfx_present.c skips the host presents while the game runs on): the driver then has no present to time the frame by,
// and the sleep would wait out Streamline's half-second timeout each frame.
#define HOLD_IDLE_NS 200000000ull
void vk_dlssg_frame_start(void) {
    if (!S.active || !S.present) return;
    vk_dlssg_reflex_sync();
    uint32_t f = __atomic_load_n(&S.game, __ATOMIC_RELAXED) + 1;
    if (f == S.started) return;
    S.started = f;
    uint64_t hosted = __atomic_load_n(&S.hosted_ns, __ATOMIC_RELAXED);
    S.sleep(f, hosted && kern_now_ns() - hosted < HOLD_IDLE_NS && (vk_dlssg_active() || reflex_mode()));
}

void vk_dlssg_game_present(void) { __atomic_add_fetch(&S.game, 1, __ATOMIC_RELAXED); }

void vk_dlssg_submitted(uint64_t frame) {
    if (S.active && S.present) S.submit_end((uint32_t)frame);
}

void vk_dlssg_present_frame(uint64_t frame) { S.cur = (uint32_t)frame; }

uint64_t vk_dlssg_shown(void) { return vk_dlssg_active() ? S.shown() : 0; }

int vk_dlssg_interface_coverage_available(void) { return S.active && S.frame_with_interface; }

int vk_dlssg_frame(VkCommandBuffer cb, VkExtent2D ext, VkImage hudless, VkImageView hudless_view, VkFormat hudless_fmt,
                   VkImage interface_coverage, VkImageView interface_coverage_view) {
    if (!S.active) return 0;
    int selected = vk_dlssg_active();
#ifndef GAME_OOT
    if (selected) vk_dlaa_hudless_wanted = 1;
#endif
    VkFfxImage d = {0}, m = {0};
    VkImageView dv = VK_NULL_HANDLE, mvv = VK_NULL_HANDLE;
    VkDlaaCamera c;
    float jx = 0, jy = 0;
    int reset = 0;
    int have = selected && vk_dlaa_fg_inputs(&d, &m, &jx, &jy, &reset) && vk_dlaa_fg_views(&dv, &mvv) && vk_dlaa_fg_camera(&c);
    // Off in every frame without the inputs: the game's pause screens have no camera, and generation kept on there
    // blended whole frames of them. The library keeps the resources while it is off (nx2_sl.h version 4+), so on again
    // costs nothing, and the first frame back starts generation's history anew (reset).
    static int went_on, was_on;
    int on = selected && (have || (went_on && !S.retains));
    if (!selected) went_on = 0;
    else if (on) went_on = 1;
    if (on && have && S.retains && !was_on) reset = 1;
    was_on = on;
    if (on) {
        fg_guard_arm(FG_GUARD_DLSS);
        if (have) fg_guard_running(FG_GUARD_DLSS);
    } else if (!selected) fg_guard_disarm(FG_GUARD_DLSS);
    Nx2SlImage zd = {d.image, dv, d.format, d.width, d.height, VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT};
    Nx2SlImage zm = {m.image, mvv, m.format, m.width, m.height, VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT};
    Nx2SlCamera k;
    memset(&k, 0, sizeof k);
    if (have) {
        memcpy(k.view_to_clip, c.view_to_clip, 64), memcpy(k.clip_to_view, c.clip_to_view, 64);
        memcpy(k.clip_to_prev_clip, c.clip_to_prev_clip, 64), memcpy(k.prev_clip_to_clip, c.prev_clip_to_clip, 64);
        memcpy(k.pos, c.pos, 12), memcpy(k.up, c.up, 12), memcpy(k.right, c.right, 12), memcpy(k.fwd, c.fwd, 12);
        k.znear = c.znear > 0 ? c.znear : 0.1f;
        k.zfar = c.zfar > 0 ? c.zfar : VK_DLAA_FAR_INFINITE;
        k.fov_y = c.fov_y, k.aspect = c.aspect;
        k.jitter[0] = jx, k.jitter[1] = jy;
        k.mv_scale[0] = 1.0f / (float)d.width, k.mv_scale[1] = 1.0f / (float)d.height;   // render pixels
        k.reset = reset;
    }
    // The picture without the HUD (gfx_present.c hudless_record): without it the generated frames blend the game's
    // interface as if it were the scene, and a dialogue box that changes between two frames blinks in the frames between.
    Nx2SlImage zh = {hudless, hudless_view, hudless_fmt, ext.width, ext.height, VK_IMAGE_LAYOUT_GENERAL,
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                         VK_IMAGE_USAGE_STORAGE_BIT};
    const int have_alpha = interface_coverage && interface_coverage_view;
    const Nx2SlImage *hudless_input = have && hudless && hudless_view &&
        (!vk_interface_coverage_enabled() || have_alpha) ? &zh : NULL;
    if (S.frame_with_interface) {
        Nx2SlImage coverage = {interface_coverage, interface_coverage_view, VK_FORMAT_R16_SFLOAT, ext.width, ext.height,
                               VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
        S.frame_with_interface(cb, have ? &zd : NULL, have ? &zm : NULL, hudless_input,
            on && hudless_input && interface_coverage && interface_coverage_view ? &coverage : NULL,
            have ? &k : NULL, ext.width, ext.height, on, S.cur);
    } else {
        S.frame(cb, have ? &zd : NULL, have ? &zm : NULL, hudless_input, have ? &k : NULL,
                ext.width, ext.height, on, S.cur);
    }
    // Even a failed SDK call may have accepted tags before failing to change options.
    return on && have;
}

void vk_dlssg_release(void) {
    if (S.initialized) {
        if (S.shutdown) S.shutdown();
        else rt_log("dlssg: the bridge does not export Streamline shutdown");
        S.initialized = 0;
    }
    S.active = 0;
    S.avail = 0;
    S.sr_requested = S.sr_ready = 0;
    if (S.loaded > 0) S.loaded = 0;
#ifdef GAME_OOT
    fg_guard_disarm(FG_GUARD_DLSS);
#endif
}

int vk_sl_sr_requested(void) { return S.loaded > 0 && S.sr_requested; }
int vk_sl_sr_available(void) { return S.loaded > 0 && S.sr_ready; }

int vk_sl_sr_evaluate(VkCommandBuffer cb, const Nx2SlImage *color, const Nx2SlImage *depth,
                      const Nx2SlImage *motion, const Nx2SlImage *output, const VkDlaaCamera *camera,
                      uint32_t frame, int preset, int reset, float jx, float jy, Nx2SlSrMode mode) {
    if (!vk_sl_sr_available()) return 0;
    Nx2SlCamera k;
    memset(&k, 0, sizeof k);
    memcpy(k.view_to_clip, camera->view_to_clip, sizeof k.view_to_clip);
    memcpy(k.clip_to_view, camera->clip_to_view, sizeof k.clip_to_view);
    memcpy(k.clip_to_prev_clip, camera->clip_to_prev_clip, sizeof k.clip_to_prev_clip);
    memcpy(k.prev_clip_to_clip, camera->prev_clip_to_clip, sizeof k.prev_clip_to_clip);
    memcpy(k.pos, camera->pos, sizeof k.pos);
    memcpy(k.up, camera->up, sizeof k.up);
    memcpy(k.right, camera->right, sizeof k.right);
    memcpy(k.fwd, camera->fwd, sizeof k.fwd);
    k.znear = camera->znear > 0 ? camera->znear : 0.1f;
    k.zfar = camera->zfar > 0 ? camera->zfar : VK_DLAA_FAR_INFINITE;
    k.fov_y = camera->fov_y, k.aspect = camera->aspect;
    k.jitter[0] = jx, k.jitter[1] = jy;
    k.mv_scale[0] = 1.0f / (float)motion->width, k.mv_scale[1] = 1.0f / (float)motion->height;
    k.reset = reset;
    return S.sr_evaluate(cb, color, depth, motion, output, &k, frame, preset, camera->inverted, mode);
}
