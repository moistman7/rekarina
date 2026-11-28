// Frame generation with AMD FSR 3.1 (vk_fg.cpp): AMD's frame generation swapchain in gfx_present.c's place.
#pragma once
#include <stdint.h>
#include <vulkan/vulkan.h>

int vk_fg_wanted(void);                      // AMD frame generation requested and allowed on this presentation path
#ifdef GAME_OOT
int vk_fg_restart_needed(void);               // requested backend needs different startup swapchain hooks
#endif
uint32_t vk_fg_queue_count(void);            // graphics-family queues the device creates: 3 with frame generation, else 1
void vk_fg_queues(VkQueue present, VkQueue acquire);   // vk_device.c, after the device exists
int vk_fg_active(void);                      // AMD's swapchain is the window's and generation has not failed
// AMD's swapchain is the window's (frame generation may have failed since): its calls take the swapchain's presents,
// acquires and images until the next present gives the window a plain swapchain again (vk_fg_release)
int vk_fg_swapchain(void);
int vk_fg_failed(void);                      // frame generation stopped for this run (the fg lines of the log say why)
// Why AMD FSR frame generation is not running although REKARINA_FRAMEGEN was fsr3 at start (the menu's row), else NULL
const char *vk_fg_off_why(void);
// DLSS Frame Generation asked for where it cannot run: AMD FSR frame generation in its place (vk_dlssg.c, after the
// device and before the first swapchain); 1 once that happened.
void vk_fg_take_over(const char *why);
int vk_fg_instead(void);
// 1 when AMD FSR frame generation can run here (its libraries are beside the game), else 0 with *why (the menu's row)
int vk_fg_offered(const char **why);
int vk_fg_generating(void);                  // ... and the last frame had generation on (not paused for the menu)
const char *vk_fg_method(void);              // interpolation context's provider, read under the presentation lock
// gfx_present.c's swapchain calls while vk_fg_wanted: the first create makes AMD's swapchain context and swapchain,
// later ones (window resizes, the fullscreen switch) its swapchain again, in place with FSR SDK 2. Non-success from the
// first: the plain swapchain path stays; after vk_fg_failed: the caller releases AMD's and makes a plain one.
VkResult vk_fg_create_swapchain(const VkSwapchainCreateInfoKHR *ci, VkSwapchainKHR *sc);
VkResult vk_fg_get_images(VkSwapchainKHR sc, uint32_t *n, VkImage *img);
VkResult vk_fg_acquire(VkSwapchainKHR sc, uint64_t timeout, VkSemaphore sem, uint32_t *idx);
VkResult vk_fg_present(const VkPresentInfoKHR *pi);   // on the game's queue, its lock held by the caller
// Each present, before its command buffer is submitted: configures generation for this frame and records its
// preparation (the frame's depth and motion vectors, vk_dlaa_fg_inputs) into cb.
// hudless: this frame's picture without the HUD at the swapchain's size and format, in GENERAL (VK_NULL_HANDLE: none),
// from which generation tells the HUD apart and keeps it out of the generated frames.
// hdr: whether the output is HDR10.
// pic: the picture's rectangle in the swapchain {x, y, w, h} (black bars around it when the window has another shape);
// frames are generated inside it only.
void vk_fg_frame(VkCommandBuffer cb, VkSwapchainKHR sc, VkFormat fmt, VkExtent2D ext, VkImage hudless, int hdr,
                 const int pic[4]);
uint64_t vk_fg_presents(VkSwapchainKHR sc);   // frames the swapchain presented, generated ones included (0: off)
PFN_vkSetHdrMetadataEXT vk_fg_hdr_metadata_fn(void);   // AMD's replacement for vkSetHdrMetadataEXT while active, else NULL
// AMD's contexts and swapchain, in the SDK's order (after a device wait, with the window's views of its images gone)
void vk_fg_release(void);
int vk_fg_wait_for_presents(void);
