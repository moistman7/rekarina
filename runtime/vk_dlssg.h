// DLSS Super Resolution and Frame Generation through NVIDIA Streamline (vk_dlssg.c, rekarina_dlssg.dll).
#pragma once
#include <stdint.h>
#include <vulkan/vulkan.h>
#include "vk_dlaa.h"
#include "dlssg/nx2_sl.h"

int vk_sl_sr_requested(void);   // Streamline v5 was loaded with DLSS Super Resolution requested (skips direct NGX)
int vk_sl_sr_available(void);
int vk_sl_sr_evaluate(VkCommandBuffer cb, const Nx2SlImage *color, const Nx2SlImage *depth,
                      const Nx2SlImage *motion, const Nx2SlImage *output, const VkDlaaCamera *camera,
                      uint32_t frame, int preset, int reset, float jx, float jy, Nx2SlSrMode mode);

int vk_dlssg_wanted(void);   // Whether startup prepares Streamline (DLSS SR and/or live DLSS Frame Generation)
int vk_dlssg_early(void);    // before the instance exists: loads the DLL and Streamline; 1 when its proxies are there
int vk_dlssg_interposed(void);      // Streamline's proxies are in use
// vk_device.c: the instance and device through Streamline's proxies when vk_dlssg_early succeeded, else the loader's
VkResult vk_dlssg_create_instance(const VkInstanceCreateInfo *ci, VkInstance *out);
VkResult vk_dlssg_enumerate(VkInstance inst, uint32_t *n, VkPhysicalDevice *pd);   // through the interposer too
PFN_vkVoidFunction vk_dlssg_instance_proc(VkInstance inst, const char *name);   // the interposer's, or NULL
VkResult vk_dlssg_create_device(VkInstance inst, VkPhysicalDevice pd, const VkDeviceCreateInfo *ci, VkDevice *out);
void vk_dlssg_device_ready(VkDevice dev);   // after the device: checks DLSS SR & DLSS-G and takes the swapchain hooks
int vk_dlssg_swapchain_ready(void);         // Streamline owns the swapchain hooks (DLSS-G can toggle live)
int vk_dlssg_active(void);                  // swapchain ready AND REKARINA_FRAMEGEN == dlss
int vk_dlssg_reflex_available(void);        // Streamline's sl.reflex owns Reflex on its swapchain
void vk_dlssg_reflex_sync(void);            // applies live REKARINA_REFLEX changes to Streamline's swapchain
// Why DLSS Frame Generation is not running although REKARINA_FRAMEGEN was dlss at start (for the menu's row), else NULL
const char *vk_dlssg_off_why(void);
// 1 when DLSS Frame Generation can run here (an RTX 40 or 50 card, its libraries beside the game), else 0 with *why
int vk_dlssg_offered(const char **why);
// gfx_present.c's swapchain calls while active
VkResult vk_dlssg_create_swapchain(VkDevice d, const VkSwapchainCreateInfoKHR *ci, VkSwapchainKHR *sc);
void vk_dlssg_destroy_swapchain(VkDevice d, VkSwapchainKHR sc);
VkResult vk_dlssg_get_images(VkDevice d, VkSwapchainKHR sc, uint32_t *n, VkImage *img);
VkResult vk_dlssg_acquire(VkDevice d, VkSwapchainKHR sc, uint64_t timeout, VkSemaphore sem, uint32_t *idx);
VkResult vk_dlssg_present(VkQueue q, const VkPresentInfoKHR *pi);
int vk_dlssg_interface_coverage_available(void);
int vk_dlssg_frame(VkCommandBuffer cb, VkExtent2D ext, VkImage hudless, VkImageView hudless_view, VkFormat hudless_fmt,
                    VkImage interface_coverage, VkImageView interface_coverage_view);
// Frames are numbered by the game's presents (frame n ends with the n-th). Reflex's frame timing while active:
void vk_dlssg_frame_start(void);    // the game starts a frame (nvn_api_obj.c nvnWindowAcquireTexture): Reflex's sleep
void vk_dlssg_game_present(void);   // the game presents it (vk_backend.c vk_be_present): counts it
// its submissions went out (vk_backend.c: vk_be_present, or replay_main with asynchronous replay): the render-submit-end
// marker; frame is gfx_counters.frames of its present, the same count
void vk_dlssg_submitted(uint64_t frame);
void vk_dlssg_present_frame(uint64_t frame);   // the present thread's next blit and present are of this frame
uint64_t vk_dlssg_shown(void);   // the frames the display got, generated ones included (present thread); 0 when off
void vk_dlssg_release(void);
