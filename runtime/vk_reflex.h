// Native NVIDIA Reflex low-latency integration via VK_NV_low_latency2 + VK_KHR_present_id (runtime/vk_reflex.c).
//
// Works on any NVIDIA GPU exposing VK_NV_low_latency2 without requiring Streamline. When Streamline owns the swapchain
// (vk_dlssg_swapchain_ready), Streamline's sl.reflex drives Reflex instead and this module stays idle.
#pragma once

#include <stdint.h>
#include <vulkan/vulkan.h>

// Device lifecycle (vk_device.c): appends VK_NV_low_latency2 and VK_KHR_present_id when supported.
// Returns 1 when both extensions were added (caller chains VkPhysicalDevicePresentIdFeaturesKHR).
int vk_reflex_device_exts(const VkExtensionProperties *available, uint32_t available_count,
                          const char **enabled, uint32_t *enabled_count, uint32_t capacity);
void vk_reflex_device_init(VkDevice device, int present_id_supported);
void vk_reflex_device_release(VkDevice device);

// Availability and live configuration:
int vk_reflex_available(void);     // native VK_NV_low_latency2 is ready on this device
int vk_reflex_offered(void);       // native or Streamline Reflex can be toggled in the settings menu
int vk_reflex_active(void);        // native or Streamline Reflex is currently pacing frames
int vk_reflex_wanted_mode(void);   // REKARINA_REFLEX clamped to 0 (off), 1 (on), or 2 (on + boost)

// Swapchain lifecycle (gfx_present.c, plain swapchain only):
#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
int vk_reflex_swapchain_next(VkSwapchainCreateInfoKHR *create_info, VkSwapchainLatencyCreateInfoNV *storage);
#endif
int vk_reflex_swapchain_stale(void);   // 1 when REKARINA_REFLEX toggled Off <-> On/Boost since swapchain creation
void vk_reflex_swapchain_created(VkSwapchainKHR swapchain, int latency_enabled);
void vk_reflex_swapchain_destroyed(void);
void vk_reflex_wsi_lock(void);
void vk_reflex_wsi_unlock(void);

// Per-frame pacing, submission tagging, and latency markers:
void vk_reflex_game_present(void);      // guest frame boundary (vk_backend.c vk_be_present)
void vk_reflex_frame_start(void);       // frame start (nvnWindowAcquireTexture): sleep + simulation/submit markers
uint64_t vk_reflex_current_id(void);    // active Reflex present ID for GPU submissions (0 when off/unmarked)
void vk_reflex_submit_end(uint64_t present_id);
const void *vk_reflex_present_begin(const void *pnext, VkPresentIdKHR *present_id_info,
                                    uint64_t *present_id_out, uint64_t frame_id);
void vk_reflex_present_end(uint64_t present_id);

#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
// Chains VkLatencySubmissionPresentIdNV onto a VkSubmitInfo2 when present_id is non-zero.
static inline void vk_reflex_chain_submit(VkSubmitInfo2 *submit_info,
                                          VkLatencySubmissionPresentIdNV *storage,
                                          uint64_t present_id) {
    if (!present_id) return;
    *storage = (VkLatencySubmissionPresentIdNV){
        .sType = VK_STRUCTURE_TYPE_LATENCY_SUBMISSION_PRESENT_ID_NV,
        .pNext = submit_info->pNext,
        .presentID = present_id,
    };
    submit_info->pNext = storage;
}
#endif
