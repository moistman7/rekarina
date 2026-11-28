// Frame generation with AMD FSR (REKARINA_FRAMEGEN = fsr3): AMD's frame generation swapchain
// (its API library, vk_ffx.cpp loads it: with the libraries of FSR SDK 2 the library picks frame generation 4 on the
// Radeon cards that run it and 3.1 elsewhere) takes the place of gfx_present.c's swapchain. Every frame
// the game presents, the swapchain presents a generated frame between it and the one before, on its own present
// queue and pacing threads, so the screen shows twice the game's frame rate.
//
// The generated frame is built from the presented picture and the scene's depth and camera motion vectors, which
// vk_dlaa.c makes for DLSS and FSR 3.1 upscaling: frame generation needs one of those on (DLAA or Native AA included).
// The HUD is interpolated with the picture (it hardly moves); the menu is not, since generation pauses while it is
// open. The device creates two more queues of the graphics family for it (vk_device.c, vk_fg_queue_count).
#ifdef _WIN32
#include <windows.h>
#endif
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>

#ifndef _WIN32
// ffx_api.h marks its entry points __declspec(dllexport), a Windows attribute; the declarations are only read here (the
// library is loaded at run time), so on other hosts the attribute is dropped rather than the header changed
#define __declspec(x)
#endif
#include "third_party/ffx_api/ffx_api.h"
#include "third_party/ffx_api/ffx_framegeneration.h"
#include "third_party/ffx_api/vk/ffx_api_vk.h"
#include "third_party/ffx_api/vk/ffx_api_framegeneration_vk.h"

extern "C" {
#include "cfg.h"
#include "gfx_wsys.h"
#include "host.h"
#include "platform/platform.h"
#include "report.h"
#include "vk.h"
#include "vk_dlaa.h"
#include "vk_dlssg.h"
#include "vk_ffx.h"
#include "vk_fg.h"
#include "ui/menu.h"
}

namespace {

struct {
    int wanted = -1;
    int asked = 0;               // REKARINA_FRAMEGEN was fsr3 at start (vk_fg_wanted)
    int instead = 0;             // running in place of DLSS Frame Generation (vk_fg_take_over)
    const char *why = nullptr;   // why it is not running although asked, for the menu's row (vk_fg_off_why)
    VkQueue present = VK_NULL_HANDLE, acquire = VK_NULL_HANDLE;
    ffxContext sc = nullptr;   // the swapchain context
    VkSwapchainKHR handle = VK_NULL_HANDLE;   // Writes hold the presentation lock; status readers use vk_fg_swapchain.
    int remake_failed = 0;     // its re-creations that failed in a row
    ffxContext fg = nullptr;   // the frame generation context
    VkExtent2D fg_ext = {0, 0};
    VkExtent2D fg_max = {0, 0};   // the context's largest render size
    VkFormat fg_fmt = VK_FORMAT_UNDEFINED;
    bool fg_hdr = false;
    uint32_t fg_depth_flags = 0;
    const char *fg_method = "FSR FG";
    ffxQueryDescSwapchainReplacementFunctionsVK fn = {};
    uint64_t frame = 0;
    int failed = 0;
    uint64_t generated = 0, paused = 0, no_inputs = 0, hudless = 0;
    bool reported_generation = false;
    int on = 0;   // the last configured frame had generation on (vk_fg_generating)
} F;

void stop_running();

VkResult submit_game(uint32_t n, const VkSubmitInfo *si, VkFence fence) {
    vk_queue_lock();   // the game's queue is shared with the recorder and the present path
    VkResult r = vkQueueSubmit(vk_ctx()->gfx_queue, n, si, fence);
    vk_queue_unlock();
    return r;
}

ffxReturnCode_t generate(ffxDispatchDescFrameGeneration *d, void *user) {
    const ffxReturnCode_t result = vk_ffx_dispatch((ffxContext *)user, &d->header);
    if (result == FFX_API_RETURN_OK && d->numGeneratedFrames) {
        __atomic_add_fetch(&F.generated, d->numGeneratedFrames, __ATOMIC_RELAXED);
        fg_guard_running(FG_GUARD_FSR);
    } else if (result != FFX_API_RETURN_OK) {
        rt_log("fg: interpolation dispatch failed (%u); returning to ordinary presentation", (unsigned)result);
        stop_running();
    }
    return result;
}

// Frame generation stops for the rest of the run: the next present gives the window a plain swapchain again
// (gfx_present.c create_swapchain), so a failure after AMD's swapchain took the window costs generation, not the game.
void stop_running() {
    __atomic_store_n(&F.failed, 1, __ATOMIC_RELEASE);
    gfx_window_mark_dirty();
}

FfxApiResource image(const VkFfxImage *i) {
    VkImageCreateInfo ci = {};
    ci.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = i->format;
    ci.extent = {i->width, i->height, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    return ffxApiGetResourceVK((void *)i->image, ffxApiGetImageResourceDescriptionVK(i->image, ci, 0),
                               FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
}

void fail(const char *what, unsigned r) {
    rt_log("fg: %s failed (%u): frame generation off for this run", what, r);
    stop_running();
    F.why = "AMD's frame generation did not start (the fg lines of the log say why)";
}

}  // namespace

extern "C" int vk_fg_wanted(void) {
    if (F.wanted < 0) {
        F.asked = F.wanted = !strcmp(cfg_value("REKARINA_FRAMEGEN"), "fsr3");
        if (F.wanted && fg_guard_blocked()) {   // it stopped the last start while being set up (report.c)
            F.why = FG_GUARD_WHY;
            F.wanted = 0;
        }
#ifndef GAME_OOT
        else if (F.wanted && !vk_dlaa_wanted()) {
            rt_log("fg: frame generation needs DLSS or FSR 3.1 upscaling (any mode): off");
            F.why = "needs DLSS / DLAA on";
            F.wanted = 0;
        }
#endif
    }
#ifdef GAME_OOT
    return F.present && !strcmp(cfg_value("REKARINA_FRAMEGEN"), "fsr3") && !fg_guard_blocked();
#else
    return F.wanted;
#endif
}

// Reserve the extra queues at device creation so OoT can later enable AMD frame generation without a restart.
// TotK also needs them when AMD may fall back for DLSS.
extern "C" uint32_t vk_fg_queue_count(void) {
#if defined(GAME_OOT) && defined(_WIN32)
    const char *why = nullptr;
    return vk_fg_offered(&why) ? 3 : 1;
#else
    return vk_fg_wanted() || vk_dlssg_wanted() ? 3 : 1;
#endif
}

#ifdef GAME_OOT
extern "C" int vk_fg_restart_needed(void) {
    if (cfg_from_env("REKARINA_FRAMEGEN")) return 0;
    if (fg_guard_retry_pending()) return 1;
    const char *fg = cfg_value("REKARINA_FRAMEGEN");
    if (!strcmp(fg, "dlss") && !vk_dlssg_swapchain_ready() && !vk_dlssg_off_why()) {
        const char *why = nullptr;
        return vk_dlssg_offered(&why);
    }
    return 0;
}
#endif

// DLSS Frame Generation asked for where it cannot run (vk_dlssg.c vk_dlssg_device_ready: no RTX 40 or 50 card, an
// older driver, its libraries missing): AMD FSR frame generation runs in its place, as it runs on any card. Called
// after the device exists and before the first swapchain.
extern "C" void vk_fg_take_over(const char *why) {
    vk_fg_wanted();
    if (F.wanted > 0 || vk_fg_failed() || fg_guard_blocked()) return;
    F.asked = F.wanted = F.instead = 1;
    F.why = nullptr;
    rt_log("fg: AMD FSR frame generation in place of DLSS Frame Generation, which cannot run here (%s)", why);
}

extern "C" int vk_fg_instead(void) { return F.instead; }

// Check for the FidelityFX loader and frame-generation library beside the game; return 0 and set *why if missing.
extern "C" int vk_fg_offered(const char **why) {
    static int ok = -1;   // the files do not change while the game runs: looked at once
    if (ok < 0) {
        char dir[PLAT_PATH_MAX], path[PLAT_PATH_MAX + 64];
        int n = plat_exe_path(dir, sizeof dir);
        char *cut = n ? plat_last_sep(dir) : nullptr;
        ok = 1;   // nothing to look at: let the start say
        if (cut) {
            *cut = 0;
            snprintf(path, sizeof path, "%s/amd_fidelityfx_framegeneration_vk.dll", dir);
            int sdk2 = plat_file_exists(path);
            snprintf(path, sizeof path, "%s/amd_fidelityfx_loader_vk.dll", dir);
            sdk2 = sdk2 && plat_file_exists(path);
            ok = sdk2;
        }
    }
    if (!ok) *why = "AMD's FidelityFX libraries are missing (run the setup again)";
    return ok;
}

extern "C" void vk_fg_queues(VkQueue present, VkQueue acquire) { F.present = present, F.acquire = acquire; }

extern "C" int vk_fg_active(void) { return vk_fg_swapchain() && !vk_fg_failed(); }

extern "C" int vk_fg_swapchain(void) { return __atomic_load_n(&F.handle, __ATOMIC_ACQUIRE) != VK_NULL_HANDLE; }

extern "C" int vk_fg_failed(void) { return __atomic_load_n(&F.failed, __ATOMIC_ACQUIRE); }

extern "C" const char *vk_fg_off_why(void) {
#ifdef GAME_OOT
    if (strcmp(cfg_value("REKARINA_FRAMEGEN"), "fsr3") || vk_fg_active()) return nullptr;
    if (fg_guard_blocked()) return FG_GUARD_WHY;
    if (!F.present) {
        const char *why = nullptr;
        return vk_fg_offered(&why) ? "the GPU has no spare queues for it" : why;
    }
#else
    if (!F.asked || vk_fg_active()) return nullptr;
#endif
    return F.why ? F.why : "it did not start (the fg lines of the log say why)";
}

extern "C" int vk_fg_generating(void) { return vk_fg_active() && __atomic_load_n(&F.on, __ATOMIC_RELAXED); }

extern "C" const char *vk_fg_method(void) { return F.fg ? F.fg_method : "FSR FG"; }

// SDK 2 remakes the swapchain in place, passing the handle (not the context) as the last argument.
static VkResult remake(const VkSwapchainCreateInfoKHR *ci, VkSwapchainKHR *sc) {
    VkDevice dev = vk_ctx()->device;
    VkResult r = F.fn.pOutCreateSwapchainFFXAPI(dev, ci, nullptr, sc, (void *)F.handle);
    *sc = F.handle;
    if (r == VK_SUCCESS) {
        F.remake_failed = 0;
    } else if (++F.remake_failed >= 3 || !F.handle) {   // the window cannot go on without a swapchain
        rt_log("fg: AMD's swapchain could not be made again at %ux%u (%d)", ci->imageExtent.width, ci->imageExtent.height,
               (int)r);
        fail("the swapchain's re-creation", (unsigned)r);
    }
    return r;
}

// AMD's swapchain in the window's place: made at the first call, made again (in place with SDK 2) at the next ones.
// The caller has destroyed a plain swapchain of the window before (AMD's takes its surface) and passes no
// oldSwapchain. VK_ERROR_FEATURE_NOT_PRESENT when frame generation is not to run.
extern "C" VkResult vk_fg_create_swapchain(const VkSwapchainCreateInfoKHR *ci, VkSwapchainKHR *sc) {
    if (vk_fg_failed()) return VK_ERROR_FEATURE_NOT_PRESENT;
    if (F.sc) return remake(ci, sc);
    if (!vk_fg_wanted()) return VK_ERROR_FEATURE_NOT_PRESENT;
    if (!F.present || !vk_ffx_load()) {   // vk_ffx_load logs a library it could not load
        if (!F.why) {
            rt_log("fg: AMD FSR frame generation off: %s", F.present ? "AMD's FidelityFX libraries did not load"
                                                                     : "the device has no queues for it");
            F.why = F.present ? "AMD's FidelityFX libraries did not load (run the setup again)"
                              : "the GPU has no spare queues for it";
        }
        return VK_ERROR_FEATURE_NOT_PRESENT;
    }
    const VkCtx *c = vk_ctx();
    ffxCreateContextDescFrameGenerationSwapChainVK d = {};
    d.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FGSWAPCHAIN_VK;
    d.physicalDevice = c->phys;
    d.device = c->device;
    VkSwapchainKHR none = VK_NULL_HANDLE;
    d.swapchain = &none;
    d.createInfo = *ci;
    d.createInfo.oldSwapchain = VK_NULL_HANDLE;
    d.gameQueue = {c->gfx_queue, c->gfx_family, submit_game};
    d.asyncComputeQueue = {VK_NULL_HANDLE, c->gfx_family, nullptr};
    d.presentQueue = {F.present, c->gfx_family, nullptr};
    d.imageAcquireQueue = {F.acquire, c->gfx_family, nullptr};
    ffxCreateContextDescFrameGenerationSwapChainVersionVK api = {};
    api.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATIONSWAPCHAIN_VERSION_VK;
    api.version = FFX_FRAMEGENERATION_SWAPCHAIN_VK_VERSION;
    d.header.pNext = &api.header;
    fg_guard_arm(FG_GUARD_FSR);
    ffxReturnCode_t r = vk_ffx_create(&F.sc, &d.header);
    if (r != FFX_API_RETURN_OK) {
        F.sc = nullptr;
        fg_guard_disarm(FG_GUARD_FSR);
        fail("the frame generation swapchain", r);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    ffxQueryDescSwapchainReplacementFunctionsVK fn = {};
    fn.header.type = FFX_API_QUERY_DESC_TYPE_FGSWAPCHAIN_FUNCTIONS_VK;
    r = vk_ffx_query(&F.sc, &fn.header);
    if (r != FFX_API_RETURN_OK || !fn.pOutQueuePresentKHR || !fn.pOutDestroySwapchainFFXAPI) {
        fail("the swapchain functions", r);
        // A replacement uses the same provider; keep its previous destroy hook until the new query succeeds.
        auto destroy = r == FFX_API_RETURN_OK && fn.pOutDestroySwapchainFFXAPI
                     ? fn.pOutDestroySwapchainFFXAPI : F.fn.pOutDestroySwapchainFFXAPI;
        if (none && !destroy)
            rt_fatal("fg: AMD created a swapchain without a destroy hook; cannot fall back to Vulkan safely");
        r = vk_ffx_destroy(&F.sc);
        if (r != FFX_API_RETURN_OK)
            rt_fatal("fg: could not destroy the incomplete AMD swapchain context (%u)", (unsigned)r);
        F.sc = nullptr;
        if (none) destroy(c->device, none, nullptr, nullptr);
        fg_guard_disarm(FG_GUARD_FSR);
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    F.fn = fn;
    *sc = none;
    __atomic_store_n(&F.handle, none, __ATOMIC_RELEASE);
    ffxQueryGetProviderVersion pv = {};
    pv.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    vk_ffx_query(&F.sc, &pv.header);
    rt_log("fg: AMD FSR frame generation swapchain %s, %ux%u", pv.versionName ? pv.versionName : "(version unknown)",
           ci->imageExtent.width, ci->imageExtent.height);
    return VK_SUCCESS;
}

extern "C" VkResult vk_fg_get_images(VkSwapchainKHR sc, uint32_t *n, VkImage *img) {
    return F.fn.pOutGetSwapchainImagesKHR(vk_ctx()->device, sc, n, img);
}

extern "C" VkResult vk_fg_acquire(VkSwapchainKHR sc, uint64_t timeout, VkSemaphore sem, uint32_t *idx) {
    // A deferred rebuild still owns the old swapchain, but must not queue new work on it after failure.
    if (vk_fg_failed()) return VK_NOT_READY;
    return F.fn.pOutAcquireNextImageKHR(vk_ctx()->device, sc, timeout, sem, VK_NULL_HANDLE, idx);
}

extern "C" VkResult vk_fg_present(const VkPresentInfoKHR *pi) {
    return F.fn.pOutQueuePresentKHR(vk_ctx()->gfx_queue, pi);
}

extern "C" int vk_fg_wait_for_presents(void) {
    ffxDispatchDescFrameGenerationSwapChainWaitForPresentsVK d = {};
    d.header.type = FFX_API_DISPATCH_DESC_TYPE_FGSWAPCHAIN_WAIT_FOR_PRESENTS_VK;
    const ffxReturnCode_t r = vk_ffx_dispatch(&F.sc, &d.header);
    if (r != FFX_API_RETURN_OK) {
        rt_log("fg: waiting for AMD's pending presents failed (%u)", (unsigned)r);
        return 0;
    }
    return 1;
}

extern "C" uint64_t vk_fg_presents(VkSwapchainKHR sc) {
    return vk_fg_active() && F.fn.pOutGetLastPresentCountFFXAPI && sc == F.handle ? F.fn.pOutGetLastPresentCountFFXAPI(sc) : 0;
}

// The HDR10 metadata goes to AMD's swapchain through its own call (vk_hdr.c vk_hdr_metadata).
extern "C" PFN_vkSetHdrMetadataEXT vk_fg_hdr_metadata_fn(void) {
    return vk_fg_swapchain() ? F.fn.pOutSetHdrMetadataEXT : nullptr;
}

extern "C" void vk_fg_frame(VkCommandBuffer cb, VkSwapchainKHR sc, VkFormat fmt, VkExtent2D ext, VkImage hudless,
                            int hdr, const int pic[4]) {
    if (!vk_fg_active()) return;
#ifndef GAME_OOT
    vk_dlaa_hudless_wanted = 1;
#endif
    const VkCtx *c = vk_ctx();
    VkFfxImage depth, mv;
    float jx = 0, jy = 0;
    int reset = 0;
    int have = vk_dlaa_fg_inputs(&depth, &mv, &jx, &jy, &reset);
    VkDlaaCamera cam;
    have = have && vk_dlaa_fg_camera(&cam);
    if (have && !cam.depth_planes_valid) {
        static bool told;
        if (!told) {
            told = true;
            rt_log("fg: skipping interpolation without valid camera depth planes");
        }
        have = 0;
    }
    const uint32_t depth_flags = have ? (cam.infinite ? FFX_FRAMEGENERATION_ENABLE_DEPTH_INFINITE : 0) |
                                        (cam.inverted ? FFX_FRAMEGENERATION_ENABLE_DEPTH_INVERTED : 0) : 0;
    // The context's largest render size covers the depth and motion vectors it is given. They have the scene's size,
    // which DLAA and the widescreen pictures make larger than the window, and a context made for the window's size alone
    // wrote past its own resources: a crash in the AMD library, device faults, striped frames. The scene is never
    // larger than the picture (widescreen.c wide_picture), so the context is made for that once; remaking it while the
    // swapchain still generated from the old one hung the present. A frame with a larger depth anyway is not generated.
    unsigned pw, ph;
    wide_picture(&pw, &ph);
    VkExtent2D max_render = {ext.width > pw ? ext.width : pw, ext.height > ph ? ext.height : ph};
    if (F.fg && (F.fg_ext.width != ext.width || F.fg_ext.height != ext.height || F.fg_fmt != fmt || F.fg_hdr != (hdr != 0))) {
        vk_queue_lock(), vkDeviceWaitIdle(c->device), vk_queue_unlock();   // a new window size: the context is made for its display size
        vk_ffx_destroy(&F.fg);
        F.fg = nullptr;
    }
    const VkExtent2D render_limit = F.fg ? F.fg_max : max_render;
    if (have && (depth.width > render_limit.width || depth.height > render_limit.height)) {
        static int told;
        if (!told++)
            rt_log("fg: a %ux%u scene is larger than the context's %ux%u: those frames are not generated", depth.width,
                   depth.height, render_limit.width, render_limit.height);
        have = 0;
    }
    if (!F.fg) {
        if (!have || menu_is_open()) return;
        // FSR SDK 2: the version of the API these structures are (the prepare call below is its second form)
        ffxCreateContextDescFrameGenerationVersion api = {};
        api.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_VERSION;
        api.version = FFX_FRAMEGENERATION_VERSION;
        api.header.pNext = vk_ffx_backend();
        // the HUD-less picture has the swapchain's format (gfx_present.c hudless_record)
        ffxCreateContextDescFrameGenerationHudless hl = {};
        hl.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION_HUDLESS;
        hl.header.pNext = &api.header;
        hl.hudlessBackBufferFormat = ffxApiGetSurfaceFormatVK(fmt);
        ffxCreateContextDescFrameGeneration d = {};
        d.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_FRAMEGENERATION;
        d.header.pNext = &hl.header;
        d.flags = depth_flags | (hdr ? FFX_FRAMEGENERATION_ENABLE_HIGH_DYNAMIC_RANGE : 0);
        d.displaySize = {ext.width, ext.height};
        d.maxRenderSize = {max_render.width, max_render.height};
        d.backBufferFormat = ffxApiGetSurfaceFormatVK(fmt);
        ffxReturnCode_t r = vk_ffx_create(&F.fg, &d.header);
        if (r != FFX_API_RETURN_OK) return F.fg = nullptr, fail("the frame generation context", r);
        F.fg_ext = ext, F.fg_fmt = fmt, F.fg_max = max_render;
        F.fg_hdr = hdr != 0;
        F.fg_depth_flags = depth_flags;
        ffxQueryGetProviderVersion provider = {};
        provider.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
        const char *name = vk_ffx_query(&F.fg, &provider.header) == FFX_API_RETURN_OK ? provider.versionName : nullptr;
        F.fg_method = name && !strncmp(name, "4.", 2) ? "FSR4 FG" :
                      name && !strncmp(name, "3.", 2) ? "FSR3 FG" : "FSR FG";
        reset = 1;
        rt_log("fg: frame generation context for %ux%u%s, render size up to %ux%u (%s)", ext.width, ext.height,
               hdr ? " (HDR)" : "", max_render.width, max_render.height, name ? name : "provider unknown");
    }
    if (have && depth_flags != F.fg_depth_flags) {
        static bool told;
        if (!told) {
            told = true;
            rt_log("fg: skipping interpolation for a depth convention different from the AMD context");
        }
        have = 0;
    }
    int on = have && !menu_is_open();
    reset |= !__atomic_exchange_n(&F.on, on, __ATOMIC_RELAXED);
    if (!have) F.no_inputs++;
    else if (!on) F.paused++;
    F.frame++;
    ffxConfigureDescFrameGeneration cfg = {};
    cfg.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
    cfg.swapChain = (void *)sc;
    cfg.frameGenerationCallback = generate;
    cfg.frameGenerationCallbackUserContext = &F.fg;
    cfg.frameGenerationEnabled = on != 0;
    cfg.allowAsyncWorkloads = false;
    static const bool debug_view = rt_knob("FSR_FG_DEBUG_VIEW", "0") != 0;
    if (debug_view) cfg.flags |= FFX_FRAMEGENERATION_FLAG_DRAW_DEBUG_VIEW;
    // the picture's rectangle: with black bars beside or below it (a window of another shape) a generation over the
    // whole swapchain warped the picture's edges into the bars
    cfg.generationRect = pic && pic[2] > 0 && pic[3] > 0 ? FfxApiRect2D{pic[0], pic[1], pic[2], pic[3]}
                                                         : FfxApiRect2D{0, 0, (int32_t)ext.width, (int32_t)ext.height};
    if (on && hudless) {
        VkFfxImage h = {hudless, fmt, ext.width, ext.height, 1};
        cfg.HUDLessColor = image(&h);
    }
    cfg.frameID = F.frame;
    ffxReturnCode_t r = vk_ffx_configure(&F.fg, &cfg.header);
    if (r != FFX_API_RETURN_OK) return fail("frame generation configuration", r);
    if (!on) return;
    static uint64_t last_ns;
    uint64_t now = plat_now_ns();
    double dt = last_ns ? (now - last_ns) / 1e6 : 16.67;
    last_ns = now;
    float dt_ms = dt > 0 && dt < 100 ? (float)dt : 16.67f;
    {
        ffxDispatchDescFrameGenerationPrepareV2 p = {};
        p.header.type = FFX_API_DISPATCH_DESC_TYPE_FRAMEGENERATION_PREPARE_V2;
        p.frameID = F.frame;
        p.commandList = cb;
        p.renderSize = {depth.width, depth.height};
        p.jitterOffset = {jx, jy};
        p.motionVectorScale = {1.0f, 1.0f};   // render pixels, current to previous (vk_dlaa.c mv_pass), as FSR takes them
        p.frameTimeDelta = dt_ms;
        p.reset = reset != 0;
        p.cameraNear = cam.znear;
        p.cameraFar = cam.zfar;
        p.cameraFovAngleVertical = cam.fov_y > 0 ? cam.fov_y : 1.0f;
        p.viewSpaceToMetersFactor = 0.01f;
        p.depth = image(&depth);
        p.motionVectors = image(&mv);
        for (int i = 0; i < 3; i++)
            p.cameraPosition[i] = cam.pos[i], p.cameraUp[i] = cam.up[i], p.cameraRight[i] = cam.right[i],
            p.cameraForward[i] = cam.fwd[i];
        r = vk_ffx_dispatch(&F.fg, &p.header);
    }
    if (r != FFX_API_RETURN_OK) return fail("frame generation preparation", r);
    const bool first_hudless = hudless && ++F.hudless == 1;
    const uint64_t generated = __atomic_load_n(&F.generated, __ATOMIC_RELAXED);
    const bool first_generated = generated && !F.reported_generation;
    F.reported_generation |= first_generated;
    if (first_hudless || first_generated || F.frame == 2 || F.frame % 3600 == 0)
        rt_log("fg: cumulative: %llu frames configured, %llu generated by SDK, HUD-less input on %llu frames, "
               "paused %llu (menu), %llu missing required inputs; current swapchain: %llu presents",
               (unsigned long long)F.frame, (unsigned long long)generated, (unsigned long long)F.hudless,
               (unsigned long long)F.paused, (unsigned long long)F.no_inputs, (unsigned long long)vk_fg_presents(sc));
}

// Everything of AMD's frame generation, in the SDK's order (FidelityFX SDK 2.4.0, frame-interpolation-swap-chain.md):
// generation off and its context, then the swapchain context and the swapchain. The caller has waited for the device; the window's
// views of AMD's images are gone.
extern "C" void vk_fg_release(void) {
    VkDevice dev = vk_ctx()->device;
    if (F.fg && F.handle) {
        ffxConfigureDescFrameGeneration off = {};
        off.header.type = FFX_API_CONFIGURE_DESC_TYPE_FRAMEGENERATION;
        off.swapChain = (void *)F.handle;
        off.frameGenerationEnabled = false;
        vk_ffx_configure(&F.fg, &off.header);
    }
    if (F.fg) vk_ffx_destroy(&F.fg), F.fg = nullptr;
    if (F.sc) vk_ffx_destroy(&F.sc);
    if (F.handle && F.fn.pOutDestroySwapchainFFXAPI) F.fn.pOutDestroySwapchainFFXAPI(dev, F.handle, nullptr, nullptr);
    F.sc = nullptr;
    __atomic_store_n(&F.handle, VK_NULL_HANDLE, __ATOMIC_RELEASE);
    F.fg_ext = {0, 0};
    __atomic_store_n(&F.on, 0, __ATOMIC_RELAXED);
    fg_guard_disarm(FG_GUARD_FSR);
}
