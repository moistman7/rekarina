// Native NVIDIA Reflex through VK_NV_low_latency2 and VK_KHR_present_id (vk_reflex.h).
//
// Enabling both extensions at device creation costs nothing while REKARINA_REFLEX=0, so toggling Reflex in the F1 menu
// requires no restart:
//   - Off <-> On / On + Boost rebuilds the plain swapchain live with VkSwapchainLatencyCreateInfoNV.
//   - On <-> On + Boost updates vkSetLatencySleepModeNV live on the next frame start.
//
// When Streamline owns the swapchain (vk_dlssg_swapchain_ready), Streamline's sl.reflex drives Reflex on that swapchain
// and this module is never attached to it, avoiding duplicate sleeps or conflicting present IDs.
//
// Frame marker timeline (per present_id):
//   - Guest thread (vk_reflex_frame_start, once per game frame):
//       vkLatencySleepNV -> vkWaitSemaphores -> SIMULATION_START, INPUT_SAMPLE, SIMULATION_END, RENDERSUBMIT_START
//     (Simulation and command recording overlap across guest threads, so simulation/submit-start markers bracket the
//     guest frame start; all graphics queue submissions carry VkLatencySubmissionPresentIdNV with the same present_id.)
//   - Present thread (gfx_present.c):
//       blit submit tagged with present_id -> vk_reflex_submit_end (RENDERSUBMIT_END) ->
//       vk_reflex_present_begin (PRESENT_START + VkPresentIdKHR) -> vkQueuePresentKHR -> vk_reflex_present_end (PRESENT_END)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "kernel.h"
#include "platform/platform.h"
#include "vk_dlssg.h"
#include "vk_reflex.h"

#define REFLEX_SLEEP_TIMEOUT_NS 100000000ull   // 100 ms cap so a stalled driver or hidden window never hangs the game
#define REFLEX_HOST_IDLE_NS     200000000ull   // skip sleep when no host present occurred recently (e.g. minimized window)

#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME

// Serialize native Reflex's Vulkan calls with acquire and present, not with the wake semaphore wait.
static PlatMutex wsi_lock = PLAT_MUTEX_INIT;

void vk_reflex_wsi_lock(void) { plat_mutex_lock(&wsi_lock); }
void vk_reflex_wsi_unlock(void) { plat_mutex_unlock(&wsi_lock); }

static struct {
    PlatMutex lock;
    int extensions_enabled;
    int ready;
    VkDevice device;
    PFN_vkSetLatencySleepModeNV set_sleep_mode;
    PFN_vkLatencySleepNV latency_sleep;
    PFN_vkSetLatencyMarkerNV set_marker;
    PFN_vkGetLatencyTimingsNV get_timings;
    VkSemaphore wake_semaphore;
    uint64_t wake_value;

    VkSwapchainKHR swapchain;
    uint64_t swapchain_gen;
    uint64_t swapchain_first_id;   // lowest present_id valid on the current swapchain
    int latency_enabled;
    int applied_mode;

    uint64_t game_presents;        // monotonic guest frame counter (atomic)
    uint64_t started_game_frame;   // prevents duplicate sleeps if the game acquires twice before presenting
    uint64_t next_present_id;      // strictly monotonic across swapchain recreations
    uint64_t active_present_id;    // present_id currently armed by vk_reflex_frame_start (0 before first frame)
    uint64_t last_presented_id;    // highest present_id handed to vkQueuePresentKHR on the current swapchain
    uint64_t last_present_ns;      // kern_now_ns() at last host present (atomic)
    uint64_t logged_frames;
} reflex = {
    .lock = PLAT_MUTEX_INIT,
    .applied_mode = -1,
    .next_present_id = 1,
};

static const char *mode_name(int mode) {
    if (mode == 2) return "on + boost";
    if (mode == 1) return "on";
    return "off";
}

static int device_has_ext(const VkExtensionProperties *available, uint32_t count, const char *name) {
    for (uint32_t i = 0; i < count; i++) {
        if (!strcmp(available[i].extensionName, name)) return 1;
    }
    return 0;
}

static int name_in_list(const char *const *list, uint32_t count, const char *name) {
    for (uint32_t i = 0; i < count; i++) {
        if (!strcmp(list[i], name)) return 1;
    }
    return 0;
}

int vk_reflex_wanted_mode(void) {
    int mode = cfg_int("REKARINA_REFLEX");
    if (mode < 0) return 0;
    if (mode > 2) return 2;
    return mode;
}

static int timings_log_enabled(void) {
    static int enabled = -1;
    if (enabled < 0) {
        const char *env = cfg_get("REKARINA_REFLEX_LOG");
        enabled = env && *env && strcmp(env, "0") != 0;
    }
    return enabled;
}

static int active_locked(void) {
    return reflex.ready &&
           reflex.swapchain != VK_NULL_HANDLE &&
           reflex.latency_enabled &&
           vk_reflex_wanted_mode() > 0;
}

static void emit_marker_locked(VkLatencyMarkerNV marker, uint64_t present_id) {
    if (!reflex.set_marker || reflex.swapchain == VK_NULL_HANDLE || !present_id) return;
    VkSetLatencyMarkerInfoNV info = {
        .sType = VK_STRUCTURE_TYPE_SET_LATENCY_MARKER_INFO_NV,
        .presentID = present_id,
        .marker = marker,
    };
    vk_reflex_wsi_lock();
    reflex.set_marker(reflex.device, reflex.swapchain, &info);
    vk_reflex_wsi_unlock();
}

static void apply_sleep_mode_locked(void) {
    int wanted = reflex.latency_enabled ? vk_reflex_wanted_mode() : 0;
    if (wanted == reflex.applied_mode || !reflex.set_sleep_mode || reflex.swapchain == VK_NULL_HANDLE) return;

    VkLatencySleepModeInfoNV info = {
        .sType = VK_STRUCTURE_TYPE_LATENCY_SLEEP_MODE_INFO_NV,
        .lowLatencyMode = wanted >= 1 ? VK_TRUE : VK_FALSE,
        .lowLatencyBoost = wanted == 2 ? VK_TRUE : VK_FALSE,
        .minimumIntervalUs = 0,
    };
    vk_reflex_wsi_lock();
    VkResult res = reflex.set_sleep_mode(reflex.device, reflex.swapchain, &info);
    vk_reflex_wsi_unlock();
    if (res == VK_SUCCESS) {
        reflex.applied_mode = wanted;
        rt_log("vk: NVIDIA Reflex: %s", mode_name(wanted));
    } else {
        static int warned;
        if (warned++ < 8)
            rt_log("vk: vkSetLatencySleepModeNV(%s) failed (%d)", mode_name(wanted), (int)res);
    }
}

// REKARINA_REFLEX_LOG=1: periodically prints driver-measured simulation-to-present and simulation-to-GPU-end latency.
static void log_timings_locked(void) {
    if (!timings_log_enabled() || !reflex.get_timings || reflex.swapchain == VK_NULL_HANDLE) return;
    if (++reflex.logged_frames % 300 != 0) return;

    VkLatencyTimingsFrameReportNV reports[64];
    for (int i = 0; i < 64; i++) {
        reports[i] = (VkLatencyTimingsFrameReportNV){
            .sType = VK_STRUCTURE_TYPE_LATENCY_TIMINGS_FRAME_REPORT_NV,
        };
    }
    VkGetLatencyMarkerInfoNV info = {
        .sType = VK_STRUCTURE_TYPE_GET_LATENCY_MARKER_INFO_NV,
        .timingCount = 64,
        .pTimings = reports,
    };
    vk_reflex_wsi_lock();
    reflex.get_timings(reflex.device, reflex.swapchain, &info);
    vk_reflex_wsi_unlock();

    double sim_to_present_us = 0.0;
    double sim_to_gpu_us = 0.0;
    int valid_frames = 0;
    for (uint32_t i = 0; i < info.timingCount && i < 64; i++) {
        const VkLatencyTimingsFrameReportNV *frame = &reports[i];
        if (!frame->simStartTimeUs || !frame->presentEndTimeUs || frame->presentEndTimeUs < frame->simStartTimeUs)
            continue;
        sim_to_present_us += (double)(frame->presentEndTimeUs - frame->simStartTimeUs);
        if (frame->gpuRenderEndTimeUs > frame->simStartTimeUs)
            sim_to_gpu_us += (double)(frame->gpuRenderEndTimeUs - frame->simStartTimeUs);
        valid_frames++;
    }
    if (valid_frames > 0) {
        rt_log("vk: NVIDIA Reflex (%s): %d frames avg, sim->present %.1f ms, sim->gpuEnd %.1f ms",
               mode_name(reflex.applied_mode), valid_frames,
               sim_to_present_us / valid_frames / 1000.0,
               sim_to_gpu_us / valid_frames / 1000.0);
    }
}

// ---- device lifecycle

int vk_reflex_device_exts(const VkExtensionProperties *available, uint32_t available_count,
                          const char **enabled, uint32_t *enabled_count, uint32_t capacity) {
    reflex.extensions_enabled = 0;
    const char *env = cfg_env("REKARINA_REFLEX");
    if (env && !strcmp(env, "-")) return 0;   // emergency opt-out before device creation

    if (!device_has_ext(available, available_count, VK_NV_LOW_LATENCY_2_EXTENSION_NAME) ||
        !device_has_ext(available, available_count, VK_KHR_PRESENT_ID_EXTENSION_NAME))
        return 0;

    int add_low_latency = !name_in_list(enabled, *enabled_count, VK_NV_LOW_LATENCY_2_EXTENSION_NAME);
    int add_present_id = !name_in_list(enabled, *enabled_count, VK_KHR_PRESENT_ID_EXTENSION_NAME);
    if (*enabled_count + (uint32_t)(add_low_latency + add_present_id) > capacity) return 0;

    if (add_low_latency) enabled[(*enabled_count)++] = VK_NV_LOW_LATENCY_2_EXTENSION_NAME;
    if (add_present_id) enabled[(*enabled_count)++] = VK_KHR_PRESENT_ID_EXTENSION_NAME;
    reflex.extensions_enabled = 1;
    return 1;
}

void vk_reflex_device_init(VkDevice device, int present_id_supported) {
    plat_mutex_lock(&reflex.lock);
    reflex.ready = 0;
    reflex.device = device;
    reflex.swapchain = VK_NULL_HANDLE;
    reflex.latency_enabled = 0;
    reflex.applied_mode = -1;
    reflex.active_present_id = 0;
    reflex.last_presented_id = 0;
    reflex.swapchain_first_id = reflex.next_present_id;

    if (!reflex.extensions_enabled) {
        plat_mutex_unlock(&reflex.lock);
        return;
    }
    if (!present_id_supported) {
        rt_log("vk: NVIDIA Reflex unavailable: VK_KHR_present_id feature not supported");
        plat_mutex_unlock(&reflex.lock);
        return;
    }

    reflex.set_sleep_mode = (PFN_vkSetLatencySleepModeNV)vkGetDeviceProcAddr(device, "vkSetLatencySleepModeNV");
    reflex.latency_sleep = (PFN_vkLatencySleepNV)vkGetDeviceProcAddr(device, "vkLatencySleepNV");
    reflex.set_marker = (PFN_vkSetLatencyMarkerNV)vkGetDeviceProcAddr(device, "vkSetLatencyMarkerNV");
    reflex.get_timings = (PFN_vkGetLatencyTimingsNV)vkGetDeviceProcAddr(device, "vkGetLatencyTimingsNV");

    VkSemaphoreTypeCreateInfo timeline_ci = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
        .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE,
        .initialValue = 0,
    };
    VkSemaphoreCreateInfo sem_ci = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
        .pNext = &timeline_ci,
    };
    reflex.wake_value = 0;
    if (reflex.set_sleep_mode && reflex.latency_sleep && reflex.set_marker &&
        vkCreateSemaphore(device, &sem_ci, NULL, &reflex.wake_semaphore) == VK_SUCCESS) {
        reflex.ready = 1;
        rt_log("vk: NVIDIA Reflex available (VK_NV_low_latency2)");
    } else {
        rt_log("vk: NVIDIA Reflex unavailable: entry points or timeline semaphore failed");
    }
    plat_mutex_unlock(&reflex.lock);
}

void vk_reflex_device_release(VkDevice device) {
    plat_mutex_lock(&reflex.lock);
    if (reflex.wake_semaphore != VK_NULL_HANDLE && device != VK_NULL_HANDLE) {
        vkDestroySemaphore(device, reflex.wake_semaphore, NULL);
        reflex.wake_semaphore = VK_NULL_HANDLE;
    }
    reflex.ready = 0;
    reflex.extensions_enabled = 0;
    reflex.device = VK_NULL_HANDLE;
    reflex.swapchain = VK_NULL_HANDLE;
    reflex.latency_enabled = 0;
    reflex.active_present_id = 0;
    reflex.applied_mode = -1;
    plat_mutex_unlock(&reflex.lock);
}

int vk_reflex_available(void) { return __atomic_load_n(&reflex.ready, __ATOMIC_ACQUIRE); }

int vk_reflex_offered(void) { return vk_reflex_available() || vk_dlssg_reflex_available(); }

int vk_reflex_active(void) {
    if (vk_dlssg_swapchain_ready()) return vk_dlssg_active() || vk_reflex_wanted_mode() > 0;
    plat_mutex_lock(&reflex.lock);
    int active = active_locked();
    plat_mutex_unlock(&reflex.lock);
    return active;
}

// ---- swapchain lifecycle

int vk_reflex_swapchain_next(VkSwapchainCreateInfoKHR *create_info, VkSwapchainLatencyCreateInfoNV *storage) {
    if (!vk_reflex_available() || vk_reflex_wanted_mode() <= 0) return 0;
    *storage = (VkSwapchainLatencyCreateInfoNV){
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_LATENCY_CREATE_INFO_NV,
        .pNext = create_info->pNext,
        .latencyModeEnable = VK_TRUE,
    };
    create_info->pNext = storage;
    return 1;
}

int vk_reflex_swapchain_stale(void) {
    if (vk_dlssg_swapchain_ready()) return 0;
    plat_mutex_lock(&reflex.lock);
    int stale = reflex.ready &&
                reflex.swapchain != VK_NULL_HANDLE &&
                reflex.latency_enabled != (vk_reflex_wanted_mode() > 0);
    plat_mutex_unlock(&reflex.lock);
    return stale;
}

void vk_reflex_swapchain_created(VkSwapchainKHR swapchain, int latency_enabled) {
    if (!vk_reflex_available()) return;
    plat_mutex_lock(&reflex.lock);
    int was_enabled = reflex.latency_enabled;
    reflex.swapchain_gen++;
    reflex.swapchain = swapchain;
    reflex.latency_enabled = latency_enabled != 0;
    reflex.applied_mode = -1;
    // Keep next_present_id strictly monotonic so an in-flight present job from an older swapchain can never
    // overtake IDs issued for this new swapchain.
    reflex.swapchain_first_id = reflex.next_present_id;
    reflex.active_present_id = 0;
    reflex.last_presented_id = 0;
    reflex.started_game_frame = 0;
    if (reflex.latency_enabled) apply_sleep_mode_locked();
    else if (was_enabled) rt_log("vk: NVIDIA Reflex: off");
    plat_mutex_unlock(&reflex.lock);
}

void vk_reflex_swapchain_destroyed(void) {
    plat_mutex_lock(&reflex.lock);
    reflex.swapchain_gen++;
    reflex.swapchain = VK_NULL_HANDLE;
    reflex.latency_enabled = 0;
    reflex.active_present_id = 0;
    reflex.applied_mode = -1;
    plat_mutex_unlock(&reflex.lock);
}

// ---- per-frame sleep and markers

void vk_reflex_game_present(void) {
    __atomic_add_fetch(&reflex.game_presents, 1, __ATOMIC_RELAXED);
}

void vk_reflex_frame_start(void) {
    if (!vk_reflex_available()) return;

    uint64_t game_frame = __atomic_load_n(&reflex.game_presents, __ATOMIC_RELAXED) + 1;
    VkDevice device = VK_NULL_HANDLE;
    VkSemaphore wake_sem = VK_NULL_HANDLE;
    uint64_t gen = 0;
    uint64_t wait_val = 0;
    int do_sleep = 0;

    plat_mutex_lock(&reflex.lock);
    if (!active_locked() || game_frame == reflex.started_game_frame) {
        plat_mutex_unlock(&reflex.lock);
        return;
    }
    reflex.started_game_frame = game_frame;
    apply_sleep_mode_locked();

    uint64_t frame_id = reflex.next_present_id++;
    gen = reflex.swapchain_gen;
    uint64_t last_present = __atomic_load_n(&reflex.last_present_ns, __ATOMIC_RELAXED);
    int recent_present = last_present != 0 && kern_now_ns() - last_present < REFLEX_HOST_IDLE_NS;
    if (recent_present) {
        VkLatencySleepInfoNV sleep_info = {
            .sType = VK_STRUCTURE_TYPE_LATENCY_SLEEP_INFO_NV,
            .signalSemaphore = reflex.wake_semaphore,
            .value = ++reflex.wake_value,
        };
        vk_reflex_wsi_lock();
        VkResult result = reflex.latency_sleep(reflex.device, reflex.swapchain, &sleep_info);
        vk_reflex_wsi_unlock();
        if (result == VK_SUCCESS) {
            device = reflex.device;
            wake_sem = reflex.wake_semaphore;
            wait_val = sleep_info.value;
            do_sleep = 1;
        }
    }
    plat_mutex_unlock(&reflex.lock);

    if (do_sleep) {
        VkSemaphoreWaitInfo wait_info = {
            .sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO,
            .semaphoreCount = 1,
            .pSemaphores = &wake_sem,
            .pValues = &wait_val,
        };
        vkWaitSemaphores(device, &wait_info, REFLEX_SLEEP_TIMEOUT_NS);
    }

    plat_mutex_lock(&reflex.lock);
    if (!active_locked() || reflex.swapchain_gen != gen || (do_sleep && reflex.wake_value != wait_val)) {
        plat_mutex_unlock(&reflex.lock);
        return;
    }
    reflex.active_present_id = frame_id;
    emit_marker_locked(VK_LATENCY_MARKER_SIMULATION_START_NV, frame_id);
    emit_marker_locked(VK_LATENCY_MARKER_INPUT_SAMPLE_NV, frame_id);
    emit_marker_locked(VK_LATENCY_MARKER_SIMULATION_END_NV, frame_id);
    emit_marker_locked(VK_LATENCY_MARKER_RENDERSUBMIT_START_NV, frame_id);
    plat_mutex_unlock(&reflex.lock);
}

uint64_t vk_reflex_current_id(void) {
    if (!vk_reflex_available()) return 0;
    plat_mutex_lock(&reflex.lock);
    uint64_t id = (active_locked() && reflex.active_present_id > reflex.last_presented_id)
                      ? reflex.active_present_id
                      : 0;
    plat_mutex_unlock(&reflex.lock);
    return id;
}

void vk_reflex_submit_end(uint64_t present_id) {
    if (!present_id || !vk_reflex_available()) return;
    plat_mutex_lock(&reflex.lock);
    if (active_locked() && present_id >= reflex.swapchain_first_id && present_id > reflex.last_presented_id)
        emit_marker_locked(VK_LATENCY_MARKER_RENDERSUBMIT_END_NV, present_id);
    plat_mutex_unlock(&reflex.lock);
}

const void *vk_reflex_present_begin(const void *pnext, VkPresentIdKHR *present_id_info,
                                    uint64_t *present_id_out, uint64_t frame_id) {
    *present_id_out = 0;
    if (!vk_reflex_available()) return pnext;

    plat_mutex_lock(&reflex.lock);
    if (!active_locked()) {
        plat_mutex_unlock(&reflex.lock);
        return pnext;
    }
    uint64_t id = frame_id ? frame_id : reflex.active_present_id;
    if (!id || id < reflex.swapchain_first_id || id <= reflex.last_presented_id) {
        plat_mutex_unlock(&reflex.lock);
        return pnext;
    }
    emit_marker_locked(VK_LATENCY_MARKER_PRESENT_START_NV, id);
    reflex.last_presented_id = id;
    *present_id_out = id;
    *present_id_info = (VkPresentIdKHR){
        .sType = VK_STRUCTURE_TYPE_PRESENT_ID_KHR,
        .pNext = pnext,
        .swapchainCount = 1,
        .pPresentIds = present_id_out,
    };
    plat_mutex_unlock(&reflex.lock);
    return present_id_info;
}

void vk_reflex_present_end(uint64_t present_id) {
    __atomic_store_n(&reflex.last_present_ns, kern_now_ns(), __ATOMIC_RELAXED);
    if (!present_id || !vk_reflex_available()) return;
    plat_mutex_lock(&reflex.lock);
    if (active_locked() && present_id >= reflex.swapchain_first_id) {
        emit_marker_locked(VK_LATENCY_MARKER_PRESENT_END_NV, present_id);
        log_timings_locked();
    }
    plat_mutex_unlock(&reflex.lock);
}

#else

int vk_reflex_device_exts(const VkExtensionProperties *available, uint32_t available_count,
                          const char **enabled, uint32_t *enabled_count, uint32_t capacity) {
    (void)available; (void)available_count; (void)enabled; (void)enabled_count; (void)capacity;
    return 0;
}
void vk_reflex_device_init(VkDevice device, int present_id_supported) { (void)device; (void)present_id_supported; }
void vk_reflex_device_release(VkDevice device) { (void)device; }
int vk_reflex_available(void) { return 0; }
int vk_reflex_offered(void) { return vk_dlssg_reflex_available(); }
int vk_reflex_active(void) { return vk_dlssg_swapchain_ready() && (vk_dlssg_active() || vk_reflex_wanted_mode() > 0); }
int vk_reflex_wanted_mode(void) { int m = cfg_int("REKARINA_REFLEX"); return m < 0 ? 0 : m > 2 ? 2 : m; }
int vk_reflex_swapchain_stale(void) { return 0; }
void vk_reflex_swapchain_created(VkSwapchainKHR swapchain, int latency_enabled) { (void)swapchain; (void)latency_enabled; }
void vk_reflex_swapchain_destroyed(void) {}
void vk_reflex_wsi_lock(void) {}
void vk_reflex_wsi_unlock(void) {}
void vk_reflex_game_present(void) {}
void vk_reflex_frame_start(void) {}
uint64_t vk_reflex_current_id(void) { return 0; }
void vk_reflex_submit_end(uint64_t present_id) { (void)present_id; }
const void *vk_reflex_present_begin(const void *pnext, VkPresentIdKHR *present_id_info,
                                    uint64_t *present_id_out, uint64_t frame_id) {
    (void)present_id_info; (void)frame_id;
    *present_id_out = 0;
    return pnext;
}
void vk_reflex_present_end(uint64_t present_id) { (void)present_id; }

#endif
