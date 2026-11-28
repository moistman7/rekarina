// Vulkan-internal header of the graphics layer. Host code only; the guest never sees it.
#pragma once
#include <vulkan/vulkan.h>
#include <stdint.h>
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnullability-completeness"
#include "vk_mem_alloc.h"
#pragma clang diagnostic pop
#include "nvn_obj.h"   // pool_t, tex_t, view_t, samp_par, NVNcopyRegion (memory and textures block)
#include "gfx.h"       // GfxShader, GfxShaderRes, GfxShaderAnswers (pipelines block)

// Required (device choice fails without them): Vulkan 1.3; VK_KHR_swapchain, VK_KHR_push_descriptor,
// VK_EXT_depth_clip_control, VK_EXT_external_memory_host; features timelineSemaphore, dynamicRendering,
// synchronization2, shaderStorageImageWriteWithoutFormat, depthClipControl, shaderInt64, samplerAnisotropy,
// independentBlend, geometryShader, textureCompressionBC.
// Optional (flag set in VkCtx when enabled): the has_* members.
typedef struct VkCtx {
    VkInstance instance;
    VkPhysicalDevice phys;
    VkDevice device;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem_props;
    uint32_t gfx_family;                    // graphics + compute + presentation on win32
    uint32_t xfer_family;                   // a transfer-only family when the GPU has one, else gfx_family
    VkQueue gfx_queue;                      // graphics + present
    VkQueue xfer_queue;                     // equals gfx_queue when xfer_shared
    VkQueue awb_queue;                      // a queue of a compute-only family (async write-back, vk_sync.c), or NULL
    uint32_t awb_family;
    int xfer_shared;                        // 1: no separate queue; serialise with gfx_queue
    VmaAllocator vma;
    VkSemaphore timeline;                   // one timeline semaphore; values are issued by vk_sync.c
    VkDebugUtilsMessengerEXT messenger;
    volatile int32_t validation_errors;        // errors seen by the messenger (REKARINA_VK_VALIDATION=1)
    int validation;                         // layer active
    int has_custom_border_color, has_eds, has_eds2, has_eds3, has_vertex_input_dynamic_state,
        has_conditional_rendering, has_transform_feedback, has_provoking_vertex, has_device_fault, has_memory_budget,
        has_robustness2, has_buffer_device_address, has_fragment_shader_interlock, has_buffer_marker,
        has_nv_checkpoints,                 // VK_NV_device_diagnostic_checkpoints (REKARINA_PASS_CRUMBS=1, vk_crumbs.c)
        has_astc_ldr,                       // textureCompressionASTC_LDR (enabled when reported)
        has_calibrated,                     // VK_KHR/EXT_calibrated_timestamps (GPU idle split in the perf line)
        has_hdr_metadata,                   // VK_EXT_hdr_metadata (vk_hdr.c vk_hdr_metadata)
        has_fragment_stores,                // fragmentStoresAndAtomics (vk_hdr.c, eye adaptation meter)
        has_extended_storage_formats,      // shaderStorageImageExtendedFormats
        has_map_placed,                     // VK_EXT_map_memory_placed with memoryMapPlaced and memoryUnmapReserve (Linux: driver-backed guest memory)
        has_atomic_float,                   // VK_EXT_shader_atomic_float with shaderBufferFloat32AtomicAdd
        has_present_timing,                 // DEBUG-PRESENTTIMING: VK_EXT_present_timing + VK_KHR_present_id2 (REKARINA_PRESENT_TIMING=1)
        fsr4_ready,                         // VK_KHR_cooperative_matrix + VK_EXT_shader_float8 on: FSR 4 can run (vk_ffx.cpp)
        has_fse;                            // VK_EXT_full_screen_exclusive: swap chains say full-screen exclusive is disallowed
#ifdef GAME_NS2
    int has_mesh_shader;                    // VK_EXT_mesh_shader with task and mesh shaders (NVN2 mesh draws)
    int has_vrs;                            // VK_KHR_fragment_shading_rate, the rate set per draw (vk_exec_draw.c mesh_shading_rate)
#endif
    // Feedback loops (maxwell.md 1.9, vk_exec_draw.c feedback_prepare): a draw samples a target of its open pass that
    // an earlier draw of the pass wrote. REKARINA_FEEDBACK=0: VK_FEEDBACK_OFF (sampled in GENERAL without a barrier, the
    // old behaviour); VK_FEEDBACK_LAYOUT with VK_EXT_attachment_feedback_loop_layout and dynamicRenderingLocalRead
    // (attachable images get feedback usage, a by-region barrier inside the pass); else (or REKARINA_FEEDBACK=restart)
    // VK_FEEDBACK_RESTART (the pass ends and restarts behind the full barrier).
    int feedback;                           // VK_FEEDBACK_*
    uint32_t min_imported_host_alignment;   // VK_EXT_external_memory_host
    uint32_t map_placed_align;              // VK_EXT_map_memory_placed: minPlacedMemoryMapAlignment
    int device_index;                       // index in vkEnumeratePhysicalDevices
} VkCtx;
enum { VK_FEEDBACK_OFF, VK_FEEDBACK_RESTART, VK_FEEDBACK_LAYOUT };

// vk_caps.c: fallbacks for GPUs other than the RX 9070, decided at device creation.
typedef struct {
    int float64;                       // shaderFloat64 enabled; 0: shaders are translated with softfloat64
    uint32_t sbuf_align;               // nonzero: storage buffers bind at their address rounded down to this
    uint32_t sg_size;                  // requiredSubgroupSize on sg_pin_stages (32)
    VkShaderStageFlags sg_pin_stages;  // 0: no pin, stages run at the device's subgroupSize
    int sg_pin_mesh;                   // task and mesh stages take the pin too (a device whose own subgroups are not 32 lanes)
    int gpl;                           // VK_EXT_graphics_pipeline_library enabled with fast linking (REKARINA_GPL=0: off)
    int amd;                           // an AMD GPU (vendor 0x1002), any driver
    int amd_sc;                        // AMD's Windows driver branch for RDNA 1 and 2 ("AMD proprietary shader compiler")
    int amd_prop;                      // AMD's own Windows driver, either branch (VK_DRIVER_ID_AMD_PROPRIETARY)
    int mesa;                          // a Mesa driver the game runs on: RADV, ANV or NVK
} VkCaps;
extern VkCaps vk_caps;
void vk_caps_driver(VkPhysicalDevice p);             // first, before vkCreateDevice: vk_caps.amd, amd_sc, mesa
void vk_caps_features(VkPhysicalDeviceFeatures *f);   // before vkCreateDevice (REKARINA_VK_NO_FLOAT64 clears shaderFloat64)
// Before vkCreateDevice, with the driver's graphics pipeline library feature and properties: sets vk_caps.gpl,
// 1 = enable the extensions (VK_KHR_pipeline_library, VK_EXT_graphics_pipeline_library) and the feature.
int vk_caps_gpl(const VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT *f,
                const VkPhysicalDeviceGraphicsPipelineLibraryPropertiesEXT *p);
void vk_caps_init(VkCtx *c);                          // after it: pretend limits, fallbacks, the boot report

VkCtx *vk_ctx(void);                        // NULL until vk_init() succeeded
int vk_init(void);                          // idempotent; 1 on success, 0 on failure (reason logged with rt_log)
void vk_shutdown(void);
int vk_selftest(void);                      // 0 on success
// Call with the result of a failed queue submit, present or wait before stopping on it: VK_ERROR_DEVICE_LOST logs
// what VK_EXT_device_fault reports (description, faulting addresses with their precision, vendor codes) once per
// process, after the GPU breadcrumbs (vk_crumbs_report). Any other result returns at once.
void vk_device_fault_report(VkResult r);
// vk_device.c: what a GPU object is, for the fault report (address bindings on; any thread). printf arguments.
void vk_addr_label(uint64_t handle, const char *fmt, ...);
void vk_addr_note(uint64_t handle, const char *fmt, ...);   // appended to the label (vk_device.c)
void vk_addr_label_print(uint64_t handle);   // DEBUG-IMGWATCH
void vk_mem_fault_report(uint64_t addr, uint64_t precision);   // vk_memory.c: the pool buffer a fault address lies in

// ---- GPU breadcrumbs (vk_crumbs.c): with VK_AMD_buffer_marker, each crumb has the GPU write its code into host memory
// twice, when the command processor reaches it and when all work before it has finished. After a lost device the two
// codes bracket the work that hung. Without the extension every call returns at once.
enum { VK_CRUMB_SUBMIT = 1, VK_CRUMB_SUBMIT_END, VK_CRUMB_UPLOAD, VK_CRUMB_UPLOAD_END, VK_CRUMB_ASTC, VK_CRUMB_PRESENT,
       VK_CRUMB_PRESENT_END, VK_CRUMB_CHUNK, VK_CRUMB_CHUNK_END, VK_CRUMB_AFTER_STRETCH,   // the last three: DEBUG-PARCRUMB
       // DEBUG-COPYCRUMB: the copies the replay records between passes (vk_texture.c alias_copy, vk_exec_draw.c mesh_ind_batch)
       VK_CRUMB_ALIAS, VK_CRUMB_ALIAS_END, VK_CRUMB_MESHCNT, VK_CRUMB_MESHCNT_END,
       VK_CRUMB_DRAW };   // DEBUG-DRAWCRUMB: before every draw op (vk_exec_draw.c, REKARINA_DRAW_CRUMBS=1)
void vk_exec_par_describe(uint32_t stretch, char *o, size_t size);   // DEBUG-PARCRUMB (vk_exec.c)
void vk_crumbs_init(VkCtx *c);       // after the allocator exists
void vk_crumbs_release(void);
// Records crumb kind with number n (a timeline value, a batch number; the low 28 bits are kept) into cb. Only the
// thread that records cb may call it: never into a replay part while the recorder thread owns it (vk_rec.h).
void vk_crumb(VkCommandBuffer cb, int kind, uint64_t n);
void vk_crumbs_report(void);         // logs the two codes and what they name

// ---- pass marks (vk_crumbs.c): the replay (vk_exec.c vk_exec_pass_begin / vk_exec_pass_end, vk_exec_draw.c
// op_dispatch) marks where each render pass and dispatch starts and where it ends, so that a device loss names the
// pass the GPU was in. A mark is a number; a ring of VK_PASS_MARKS records keeps what it was. With
// VK_AMD_buffer_marker the GPU writes the number of the last mark it started (TOP_OF_PIPE, before vkCmdBeginRendering
// or the dispatch) and of the last it finished (BOTTOM_OF_PIPE, after vkCmdEndRendering or the dispatch) into the
// breadcrumb buffer; with VK_NV_device_diagnostic_checkpoints the same two points are checkpoints, read with
// vkGetQueueCheckpointData(2)NV after the loss. REKARINA_PASS_CRUMBS: unset, AMD buffer marks only; 1, also NVIDIA
// checkpoints (the extension is enabled at device creation); 0, none.
#define VK_PASS_MARKS 4096   // power of two
enum { VK_MARK_PASS = 1, VK_MARK_DISPATCH, VK_MARK_PROLOGUE, VK_MARK_WRITEBACK, VK_MARK_MERGE, VK_MARK_WBCOPY, VK_MARK_UI_CAPTURE };   // the last two: mirror copies in and out
enum { VK_MARK_IMG_SET = 1, VK_MARK_IMG_BOUND = 2, VK_MARK_IMG_MIRRORED = 4, VK_MARK_IMG_GPU_OWNED = 8, VK_MARK_IMG_STORAGE = 16 };
typedef struct {
    uint64_t addr;
    uint32_t w, h, levels, layers;
    uint32_t serial;
    VkFormat format;
    uint8_t flags;                      // VK_MARK_IMG_*
} VkMarkImage;
typedef struct {
    uint32_t id;                        // the mark this record belongs to (0: never written)
    uint8_t kind;                       // VK_MARK_*
    uint8_t ncolor;                     // colour attachments
    uint32_t step;                      // index of the pass, or of the dispatch, within its submission, from 0
    uint32_t w, h, d;                   // render area and layers; a dispatch: group counts (0 0 0: indirect)
    uint32_t draws;                     // draws recorded in the pass; UINT32_MAX until its end is recorded
    uint64_t fence;                     // the submission: the timeline value it signals
    VkFormat color[8], depth;           // attachment formats (VK_FORMAT_UNDEFINED: none)
    const GfxShader *first[2], *last[2];// vertex and fragment shader of the first and last draw; a dispatch: first[0]
    // A pass only, for the device loss report (vk_pass_mark_detail): what it renders into and what its first draw
    // samples, so that a loss names the images the GPU stopped on, not only their formats.
    VkMarkImage tgt;                    // the first colour target, else the depth target (flags 0: none)
    VkMarkImage tex[3];                 // the first draw's first fragment textures at bound slots (flags 0: none)
    int32_t count;                      // the first draw's elements; 0 until a draw is prepared
} VkPassMark;
// Pass marks wanted and possible (REKARINA_PASS_CRUMBS and the extensions): set by vk_crumbs_init.
extern int vk_pass_marks_on;
// vk_device.c create_device, before vkCreateDevice: 1 when REKARINA_PASS_CRUMBS asks for NVIDIA checkpoints.
int vk_pass_marks_want_nv(void);
// The next mark: records its start into cb (through the recorder thread when it owns cb) and returns its ring record,
// id, kind and fence set, for the caller to fill. NULL when pass marks are off. Replay thread only.
VkPassMark *vk_pass_mark_begin(VkCommandBuffer cb, int kind, uint64_t fence);
void vk_pass_mark_end(VkCommandBuffer cb, const VkPassMark *m);   // records the end of mark m into cb
// The recorder thread's side of both (vk_rec.c): the marker and checkpoint commands themselves.
void vk_pass_mark_cmd(VkCommandBuffer cb, uint32_t id, int end);
// AMD only: the last mark the GPU started and the last it finished (0: none); 0 without buffer marks.
int vk_pass_marks_read(uint32_t *started, uint32_t *finished);
const VkPassMark *vk_pass_mark_record(uint32_t id);   // the ring record of mark id; NULL once a later mark took its slot
// The mark's copy of an image record, a const VkTexImage * (vk_exec.c at a pass begin, vk_exec_draw.c at its first
// draw); NULL: none.
VkMarkImage vk_pass_mark_image(const void *tex_image);
// One line for the report: the images of pass mark id (VkPassMark.tgt, tex); empty for other marks and lost records.
void vk_pass_mark_detail(uint32_t id, char *o, size_t size);
// The report's text for a started and a finished mark, three lines without a trailing newline: which marks (source:
// the extension they came from), then each with its ring record. vk_pass_marks_report logs it; the probe checks it.
void vk_pass_marks_format(const char *source, uint32_t started, uint32_t finished, char *o, size_t size);
void vk_pass_marks_report(void);    // vk_device_fault_report: the AMD words and the NVIDIA checkpoints, formatted

// ---- window and present (gfx_present.c). Host threads only.
// vk_present_image: blits the crop rectangle {x, y, w, h} of src (NULL, or w/h <= 0: the whole w x h image), scaled
// to the swapchain extent, into the acquired swapchain image and presents it. src is in layout, which is
// VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL (its writes made visible to the transfer stage by the caller's barrier) or
// VK_IMAGE_LAYOUT_GENERAL (vk_texture.c images: the present makes earlier writes visible and orders later writes after
// the blit). All its writes must be submitted to gfx_queue before the call (queue order does the rest). Starts the
// window on first use. A minimised window or a lost swapchain skips the frame. Calls serialise on an internal lock.
void vk_present_image(VkImage src, VkImageLayout layout, VkFormat fmt, int w, int h, const int crop[4]);
// vk_present_image split in two: vk_present_blit submits the blit and leaves the present waiting, vk_present_flip
// queues it. The next blit or a swapchain rebuild queues a waiting present first.
typedef struct { float row[3][4]; int valid; } VkGuestPq;
int vk_present_blit(VkImage src, VkImageLayout layout, VkFormat fmt, int w, int h, const int crop[4],
                    const VkGuestPq *pq, const char *nvidia_scene_aa); // nonzero: DLSS-G inputs must survive the flip
// FSR 1 on the present (vk_fsr.c): records EASU (+ RCAS) of src's crop rectangle {x, y, w, h} into the rectangle out
// {x, y, w, h} of dst, a swapchain image in TRANSFER_DST; src is in GENERAL and fmt its format. 0 = nothing recorded
// (the setting says bilinear, no upscale is needed, or FSR is not available): the caller blits.
int vk_fsr_record(VkCommandBuffer cb, unsigned slot, VkImage src, VkFormat fmt, int sw, int sh, const int crop[4],
                  VkImage dst, const int out[4]);
#ifdef GAME_OOT
int vk_fsr_tonemapped(VkCommandBuffer cb, VkImageView src, uint32_t w, uint32_t h, VkImage dst, VkFormat format,
                      uint32_t ow, uint32_t oh, int hdr_bt2020);
int vk_scene_rcas(VkCommandBuffer cb, VkImage image, VkImageView view, VkFormat format, uint32_t w, uint32_t h, int strength);
#else
int vk_fsr_scene(VkCommandBuffer cb, VkImageView src, uint32_t w, uint32_t h, VkImageView out, uint32_t ow, uint32_t oh);
#endif
void vk_fsr_release(void);   // before the device goes away
// Colour adjustment on the present (vk_color.c): with REKARINA_COLOR_* away from their defaults, records a pass that writes
// src's crop rectangle {x, y, w, h}, adjusted, into an RGBA16F image of that size in GENERAL and returns it (the caller
// scales that one instead, crop {0, 0, w, h}); VK_NULL_HANDLE = nothing recorded. src is in GENERAL, fmt its format.
VkImage vk_color_record(VkCommandBuffer cb, unsigned slot, VkImage src, VkFormat fmt, const int crop[4]);
void vk_color_release(void);   // before the device goes away
// On an HDR10 swapchain vk_hdr_encode grades instead: vk_color_hdr fills adj (brightness, contrast, saturation, gamma)
// and returns 1 when a profile is set (logged per picture size); vk_color_not_drawn_hdr says once when it could not.
int vk_color_hdr(float adj[4], const int crop[4]);
void vk_color_not_drawn_hdr(void);
// HDR output (vk_hdr.c). vk_hdr_armed enables the Vulkan extensions and tone-map support. vk_hdr_wanted follows the
// live mode on OoT. vk_hdr_native selects the guest scene mapper; both OoT HDR modes use the guest's PQ output.
// vk_hdr_surface_format selects the HDR10 surface format for TotK. vk_hdr_set_active records the swapchain encoding.
// vk_hdr_encode returns src's crop as PQ code values (RGBA16F, crop {0, 0, w, h}).
// vk_hdr_ui_target / vk_hdr_ui_compose: the in-game menu's RGBA8 target, then blended onto the swapchain image.
// vk_hdr_to_srgb8: 10-bit PQ pixels to sRGB bytes. vk_hdr_images_release: drops present pass images when leaving HDR.
int vk_hdr_armed(void);
int vk_hdr_native(void);
#ifdef GAME_OOT
int vk_hdr_renodx_scene(void);
#endif
int vk_hdr_native_picture(VkFormat fmt);
int vk_hdr_wanted(void);
void vk_hdr_off(void);   // probes that read back 8-bit swapchain pixels: SDR whatever the setting
int vk_hdr_surface_format(const VkSurfaceFormatKHR *f, uint32_t n, VkSurfaceFormatKHR *out);
void vk_hdr_metadata(VkSwapchainKHR sc, int new_sc);
void vk_hdr_set_active(int on);
int vk_hdr_active(void);
#ifdef GAME_OOT
int gfx_hdr_surface_capable(void);
int vk_hdr_output_pending(void);
void vk_hdr_output_overwrite(VkImage image, uint64_t address);
void vk_hdr_record_output(uint32_t serial, VkImage image, uint64_t address, GfxOutputEncoding encoding, const VkGuestPq *pq);
GfxOutputEncoding vk_hdr_written_output(uint32_t serial, uint64_t frame, VkGuestPq *pq);
#endif
VkImage vk_hdr_encode(VkCommandBuffer cb, unsigned slot, VkImage src, VkFormat fmt, const int crop[4], const VkGuestPq *pq);
VkImageView vk_hdr_ui_target(VkCommandBuffer cb, uint32_t w, uint32_t h);
void vk_hdr_ui_compose(VkCommandBuffer cb, VkImageView dst_view, VkFormat fmt, uint32_t w, uint32_t h);
void vk_hdr_to_srgb8(const uint8_t *in, uint8_t *out, size_t px, int a2r);
void vk_hdr_images_release(void);   // swapchain moved to SDR: drop transient HDR images
void vk_hdr_release(void);   // before the device goes away
// The HDR tone mapper's parameters: VK_HDR_PARAMS vec4 after the translator's own support buffer fields, from byte
// VK_HDR_PARAMS_AT (vk_exec_draw.c support_fill writes them; the spliced tone mapper and the game's bloom term read
// them, shaders/hdr_tonemap.frag lists them).
#define VK_HDR_PARAMS_AT 1440u
#define VK_HDR_PARAMS 6u
void vk_hdr_support(float out[4 * VK_HDR_PARAMS]);
#ifdef GAME_OOT
// Eye adaptation is available when the Vulkan device can atomically write the fragment meter.
int vk_hdr_adaptation_supported(void);
int vk_hdr_renodx_effects(void);
int vk_hdr_sdr_scene(void);
int vk_hdr_scene_grade(void);
// Binds the HDR eye adaptation meter for a draw. The caller binds its zero buffer if no meter was acquired.
struct VkExec;
VkDescriptorBufferInfo vk_hdr_meter_bind(struct VkExec *ex);
void vk_hdr_meter_end_pass(struct VkExec *ex);
#endif
// vk_contact.c: screen-space contact shadows on the game's composed scene colour (REKARINA_CONTACT_SHADOWS)
int vk_contact_strength(void);   // percent, 0 = off
void vk_contact_release(void);   // before the device goes away
void vk_present_flip(void);
// Every submission to ctx->gfx_queue (and xfer_queue when shared) takes this lock; vk_present_image and the swapchain
// rebuild do. Recursive.
void vk_queue_lock(void);
void vk_queue_unlock(void);
// Copy of the last presented swapchain image, for probes: on, the next presents read the image back into a host buffer
// and wait for the GPU. *px stays valid until the next present; the format is the swapchain's. Returns 0 with no frame.
// The rectangle {x, y, w, h} of a dw x dh swapchain image a cw x ch picture is presented in (gfx_present.c,
// REKARINA_ASPECT); the rest is black.
void gfx_fit_rect(int cw, int ch, int dw, int dh, int out[4]);
void vk_present_capture(int on);
int vk_present_captured(const uint8_t **px, int *w, int *h, VkFormat *fmt);

// The window system's part of device creation (gfx_window_win.c with Win32, gfx_window_sdl.c with SDL): the instance
// extensions a surface needs (their count, at most max), and whether a queue family of pd can present to it.
int gfx_window_vk_extensions(const char **out, int max);
int gfx_window_vk_presentable(VkInstance inst, VkPhysicalDevice pd, uint32_t family);
// The swap chain's pNext from the window system: with Win32 and VK_EXT_full_screen_exclusive (has_fse), full-screen
// exclusive disallowed (gfx_window_win.c); NULL otherwise.
const void *gfx_window_swapchain_next(int has_fse);
void gfx_window_resize(int client_w, int client_h);   // programmatic resize; returns after the swapchain is rebuilt
void gfx_vsync_bind_counter(volatile int32_t *counter);  // host-side stand-in for a bound event: incremented per tick
unsigned gfx_window_focus_messages(void);          // OE_MSG_FOCUS_STATE_CHANGED pushes made by the window thread
void gfx_window_release_vk(void);                  // vk_shutdown calls it first: destroys swapchain, surface, present objects
long vk_validation_errors_total(void);             // validation errors since the first vk_init, across vk_shutdown

// ---- memory and textures (vk_memory.c, vk_texture.c)
// Guest or host threads; every call needs vk_init(). The first nine have the GfxBackend signatures of gfx.h; vk_backend.c builds
// gfx_vulkan_backend from them.
int  vk_pool_init(pool_t *p);              // CPU-visible: guest storage imported (VK_EXT_external_memory_host) as one
                                           // VkBuffer; physical: device-local memory; virtual: a 64 KiB page table
void vk_pool_fini(pool_t *p);              // drops images cut from the pool; waits for the GPU if it used the pool
int  vk_pool_map_virtual(pool_t *v, int n, const void *requests);   // nvn_mapping_t[n], checked by the front end
void vk_texture_register(int id, const tex_t *t, const view_t *v);  // texture descriptor slot (copied)
void vk_sampler_register(int id, const samp_par *s);                // sampler descriptor slot (copied)
void vk_texture_slot_prefetch(int id);          // a handle was bound: its slot is looked up soon
const uint64_t *vk_texture_slot_regs(void);   // vk_texpool.c: registration serial of slot id = word id >> 16
void vk_texture_premake(const tex_t *t);   // TextureInitialize: an image bound into pool memory is made ahead of its first use
void vk_texture_release(const tex_t *t);   // TextureFinalize: the image stays for slots that still name its storage
void vk_read_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row, uint64_t img);
void vk_write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src, uint64_t row, uint64_t img);
void vk_guest_written(uint64_t addr, uint64_t size);   // a host write write-watch may miss: textures there re-upload

// GPU address -> backing (section 3.1). avail: bytes from offset that stay contiguous in buffer. alloc/alloc_offset:
// the device-local memory behind physical and (through the page table) virtual pool addresses, NULL for imported pools.
// cls: the pool class of the backing (VK_MCLS_*), for the traffic census.
#define VK_MEM_TWIN_SHIFT 512
VmaAllocation vk_mem_phys_twin(VmaAllocation main);   // vk_memory.c: a physical pool's second block, NULL when there is none
typedef struct {
    VkBuffer buffer;
    VkDeviceSize offset, avail;
    VmaAllocation alloc;
    VkDeviceSize alloc_offset;
    int cls;
    void *pool;   // the imported pool record behind the span (NULL for device memory), for VK_MEM_BOUND
} VkMemSpan;
// Traffic census (vk_memory.c vk_mem_census): bytes the replay binds or copies, by pool class and use. Imported pools by
// their CPU access flag; DEVICE is physical memory (directly or through a virtual pool); SCRATCH the replay's transient
// copies. CPU_READ counts bytes host code reads from pool storage in place of the GPU (misaligned binds, byte indices,
// counter words); GPU_WRITE what copies, fills, counter reports and uniform updates write.
enum { VK_MCLS_NONE, VK_MCLS_HOST_NOACC, VK_MCLS_HOST_UNC, VK_MCLS_HOST_CACHED, VK_MCLS_DEVICE, VK_MCLS_SCRATCH, VK_MCLS_N };
enum { VK_MUSE_VB, VK_MUSE_IB, VK_MUSE_UBO, VK_MUSE_SSBO, VK_MUSE_IND, VK_MUSE_COPY_SRC, VK_MUSE_GPU_WRITE,
       VK_MUSE_CPU_READ, VK_MUSE_N };
void vk_mem_count(int cls, int use, uint64_t bytes);
// REKARINA_TRACE=perf dirty census (vk_memory.c): the replay notes each imported pool it binds as a vertex, index, uniform,
// storage or indirect buffer; at the start of every submit the pages the CPU wrote in such pools since the previous
// submit are counted from the shared page generations and logged as `mem: dirty` every 60 frames. Nothing when tracing
// is off: VK_MEM_BOUND is one flag test at the call site.
void vk_mem_bound_note(void *pool);
#define VK_MEM_BOUND(sp) do { if ((rt_trace & RT_TRACE_PERF) && (sp).pool) vk_mem_bound_note((sp).pool); } while (0)
void vk_mem_dirty_submit(void);                   // vk_sync_submit, under its lock, before the replay
// VRAM mirror of buffer-bound pools (vk_memory.c). vk_mirror_on: REKARINA_MEMORY=mirror|verify, or the
// device class is discrete and REKARINA_MEMORY is unset or auto; set at device selection. The replay (submitting thread
// under vk_sync_submit's lock) calls:
// - VK_MEM_USE at a vertex, index, uniform, storage or indirect binding of a span vk_mem_resolve returned: the pool's
//   mirror is created on first use and sp.buffer becomes it (same offsets); a pool that stays imported keeps its buffer.
// - VK_MEM_PICK where a copy, fill or update reads or writes a span: the mirror if the pool has one, never a new one.
// - vk_mem_gpu_wrote after recording a GPU write of [sp.offset, sp.offset + size) into sp.buffer: a mirror's bytes go
//   back to guest memory at the end of the submission. No-op for imported buffers.
// - vk_mem_mirror_begin at the start of a submission's recording, vk_mem_mirror_end (records the write-backs; returns
//   the commands recorded) after its last barrier.
// vk_sync.c records the copies the mirrors' first accesses queued (vk_mem_mirror_pending) into a prologue command
// buffer with vk_mem_mirror_prologue and submits it right before each part.
extern int vk_mirror_on;
void vk_mem_use(VkMemSpan *sp);
void vk_mem_pick(VkMemSpan *sp);
#define VK_MEM_USE(sp) do { if (vk_mirror_on && (sp).pool) vk_mem_use(&(sp)); } while (0)
#define VK_MEM_PICK(sp) do { if (vk_mirror_on && (sp).pool) vk_mem_pick(&(sp)); } while (0)
// Both return 1 when the write lands in a VRAM mirror and goes back to guest memory whole (a plain copy at the end of
// the submission): the mirror then holds those bytes, and the fence's report is vk_gpu_written_mirror.
int vk_mem_gpu_wrote(const VkMemSpan *sp, uint64_t size);
// A shader's storage-buffer write: goes back through the write-back merge (vk_memory.c), which keeps the words the GPU
// did not change.
int vk_mem_gpu_wrote_shader(const VkMemSpan *sp, uint64_t size);
// A shader adds into the range with float atomics (GFX_SSBO_FADD): words the CPU wrote while the GPU ran get the GPU's
// additions on top instead of the GPU's value (vk_memory.c, write-back merge). Returns as vk_mem_gpu_wrote_shader.
int vk_mem_gpu_wrote_adds(const VkMemSpan *sp, uint64_t size);
// Uniform blocks as of the submit call (vk_usnap.c): vk_exec_record copies the uniform ranges a submission binds before
// it records, ubo bindings take the copy (0: not copied, bind memory), GPU writes of the replay drop the copies they touch.
void vk_usnap_begin(const uint64_t *handles, int n, uint64_t fence);
uint64_t vk_sync_reserved(void);   // vk_sync.c: the newest timeline value handed out (issued or reserved)
int vk_exec_ub_nojoin(void);              // vk_exec.c: an inline uniform update does not wait for a stretch's chunks
uint32_t vk_exec_draw_upd_count(uint64_t fence);   // vk_exec_draw.c: inline uniform updates noted in that submission so far
void vk_exec_draw_upd_limit(uint32_t n);  // this thread reads only the first n of them (UINT32_MAX: all)
uint32_t vk_usnap_update_serial(void);    // vk_usnap.c: snapshot drops by inline uniform updates so far
void vk_usnap_view_limit(uint32_t useq);  // this chunk thread counts only those up to useq (UINT32_MAX: all)
void vk_usnap_drop_update(uint64_t addr, uint64_t n);
void vk_tpool_slot_registered(int id);   // vk_texpool.c: vk_texture_register changed the slot (texture lock held)
int vk_pipeline_compute_pending(void);   // vk_pipeline.c: this thread's last vk_pipeline_compute is still being built
int vk_pipeline_stages_pending(void);    // vk_pipeline.c: this thread's last vk_pipeline_stages is still being built
void vk_usnap_end(uint64_t fence);
// After a wait for the CPU in the command stream: the uniform ranges bound from op record `from` on (to the end of its
// recording) and in the recordings rest[0..nrest) are copied again; the draws recorded before keep the earlier copy.
void vk_usnap_resnap(const void *from, const uint64_t *rest, int nrest);
void *vk_usnap_hold(void);                 // vk_usnap.c: a held submission's snapshot, out of the running state
void vk_usnap_held_enter(void *held);
void vk_usnap_held_leave(void *held);
void vk_usnap_held_done(void *held, uint64_t fence);
void vk_usnap_refresh(void);   // a deferred wait is met: the current arena takes the CPU's bytes again
// The same for the VRAM mirrors: each syncs again at its next access in this submission (vk_memory.c).
void vk_mem_resync_after_wait(void);
void vk_mem_resync_accessed(void);   // a deferred wait is met (vk_exec_xfer.c vk_exec_wait_resolve)
int vk_usnap_find(uint64_t addr, uint64_t n, uint64_t align, VkDescriptorBufferInfo *out);
const uint8_t *vk_usnap_map(void);   // a host copy of the current arena, at the offsets vk_usnap_find gives
int vk_usnap_poke(uint64_t addr, const void *p, uint64_t n);   // replaces bytes of this submission's copy
const uint8_t *vk_usnap_bytes(uint64_t addr, uint64_t n);   // the copy of [addr, addr+n) as of the submit, NULL: none
void vk_usnap_drop(uint64_t addr, uint64_t n);   // the replay makes the GPU write [addr, addr+n) of guest memory
void vk_usnap_gpuw(uint64_t addr, uint64_t n);   // DEBUG-USNAP-GPUW
void vk_usnap_par_view(void);                     // vk_exec.c: the snapshot the chunks of a draw stretch read from here on
uint64_t vk_usnap_gen(void);                      // changes whenever a binding may resolve differently
void vk_usnap_shutdown(void);                     // vk_exec_draw_shutdown: frees the arenas
// vk_mvprev.c: last frame's copy of the scene's object (c7) and bone (c5) blocks, for per-object motion vectors.
// vk_mvprev_step: the distance of the game's other copy of a vertex-stage block the snapshot also copies, 0: none.
// vk_mvprev_find: last frame's block of that object (prev, its address; how, MVPREV_*), 0: none.
enum { MVPREV_NONE, MVPREV_SELF, MVPREV_SIDE, MVPREV_NEARER };
uint64_t vk_mvprev_step(int stage, int index, uint64_t size);
int vk_mvprev_on(void);
void vk_mvprev_want(uint32_t index_mask);   // also keep these binding indices' last-frame copies (bit n: index n)   // DLAA on and REKARINA_DLAA_OBJMV not 0, as of the last present
int vk_mvprev_find(int index, uint64_t addr, uint64_t size, uint64_t *prev, int *how);
void vk_mvprev_frame(void);   // each present, beside vk_dlaa_frame
void vk_mem_mirror_begin(void);
void vk_mem_mirror_begin_ex(int presnap);   // presnap 0: no batched write-watch queries ahead (vk_memory.c)
int vk_mem_mirror_end(VkCommandBuffer cb, uint64_t fence);
int vk_mem_mirror_pending(void);
void vk_mem_mirror_prologue(VkCommandBuffer cb);
#ifdef GAME_NS2
// NVN2 global memory at loaded addresses (vk_memory.c, docs/oot-xlate.md "gmem_map"): the pools as disjoint guest ranges
// sorted by start, each with delta = the device address holding its bytes minus its guest start; cpu: the bytes the GPU
// writes there are guest memory (vk_gpu_written reports them), else a device copy (a VRAM mirror or device memory).
// vk_mem_gmem_table fills e[1..) (e[0] empty) and returns the entry count, *gen the generation it was made at (pools and
// mirrors); vk_mem_gmem_gen: the generation now. vk_mem_gmem_sync (replay thread): the mirrors take the CPU's writes
// before a submission's first access through the table.
typedef struct { uint64_t lo, hi, delta; int cpu, mirror; } VkMemGmemEnt;
int vk_mem_gmem_table(VkMemGmemEnt *e, int cap, uint64_t *gen);
void vk_mem_host_read_note(void);     // the GPU reads a pool's host memory in place this frame
int vk_mem_host_read_recent(void);    // it did in the last 120 frames
uint64_t vk_mem_gmem_gen(void);
uint64_t vk_mem_gmem_mgen(void);   // its part that the replay itself moves (a mirror made or given up)
void vk_exec_gmem_unpin(void);     // vk_exec_draw.c: the submission's pool table is checked again at its next use
void vk_mem_gmem_sync(void);
void vk_mem_gmem_written(uint64_t lo, uint64_t hi);   // completion thread: a log says the GPU wrote this mirror piece
void vk_mem_gmem_writeback(void);   // replay thread: this submission may write too; its hot pieces go back to guest memory
#endif
// Shadow storage (vk_memory.c): vk_sync.c calls vk_mem_shadow_sync right before each submit; the replay notes every GPU
// write into CPU-visible memory at record time (vk_mem_gpu_pending) and reports it at the fence with vk_gpu_written,
// which copies the bytes back into guest memory before telling the texture cache (vk_guest_written).
// Mesh draws on a GPU without VK_EXT_mesh_shader (vk_exec_draw.c mesh_emulate): the emulation buffer's device address
// (0 until it exists). Kernels of translator patch 0047 read it from push constants (VK_MESH_EMU_PUSH_BYTES at offset 0,
// pushed before each emulated dispatch); older kernels hold the marker constants in its place (tools/xlate MeshStage),
// which vk_pipeline.c meshc_module replaces.
uint64_t vk_mesh_emu_address(void);
#define VK_MESH_EMU_PUSH_BYTES 8u
void vk_pipeline_dynamic_forget(void);   // vk_pipeline.c: the dynamic state the replay last set is unknown now
#define VK_MESH_EMU_ADDRESS_LO 0xC0DE5A71u
#define VK_MESH_EMU_ADDRESS_HI 0xC0DE5A72u
#define VK_MESH_EMU_BYTES (64ull << 20)
#define VK_MESH_EMU_MESH_ARGS_OFFSET 32ull   // under a task: the mesh kernel's {x, y, z}, written by the task kernel
#define VK_MESH_EMU_COUNTER_OFFSET 48ull     // under a task: the meshlets taken
#define VK_MESH_EMU_VERTEX_OFFSET (8ull << 20)
#define VK_MESH_EMU_INDEX_OFFSET (48ull << 20)
void vk_mem_shadow_sync(void);
void vk_mem_gpu_pending(uint64_t addr, uint64_t size);
int vk_mem_shadow_on(void);
void vk_mem_driver_memory(int placed);   // Linux (vk_device.c create_device): driver-backed guest memory, if asked for
// Global-memory writes (vk_exec_draw.c gmem_*): a submission that binds a write log records the pieces such writes
// reached before as pending (replay thread); its fence reports each piece its log shows written, then the rest goes.
void vk_mem_shadow_gmem_record(uint64_t fence);
void vk_mem_shadow_gmem_written(uint64_t lo, uint64_t hi, uint64_t fence);
void vk_mem_shadow_gmem_done(uint64_t fence);
void vk_gpu_written(uint64_t addr, uint64_t size);
// The same for bytes a VRAM mirror wrote and copied back whole (vk_mem_gpu_wrote returned 1): the pages are stamped for
// the texture cache as written, but the mirror's sync leaves them out until the CPU writes them again (it holds them,
// and with frames in flight guest memory may be older than what later GPU work already left in the mirror).
void vk_gpu_written_mirror(uint64_t addr, uint64_t size);
int vk_mem_flush_add_probe(void);        // --probe-vkexec: the flush range list; 0 or the number of the failed check
uint64_t vk_mem_mirror_uploaded(void);   // --probe-vkexec: bytes copied into mirrors since the last summary line
int vk_mem_mirror_probe(uint64_t addr);   // --probe-vkexec: the pool at addr gets its mirror as on a first buffer use;
                                          // 1 when it has one
void vk_mem_class_log(const VkPhysicalDeviceProperties *p, const VkPhysicalDeviceMemoryProperties *m);   // once
int vk_mem_class_of(uint64_t addr);               // VK_MCLS_* of the pool at addr (NONE when unbacked)
void vk_mem_census(uint64_t frame);               // at each present: pool census and traffic, logged at frame 1800 and,
                                                  // with REKARINA_TRACE=perf, every 1800 frames
// VRAM in use and its budget, summed over the DEVICE_LOCAL heaps. exact 1: VK_EXT_memory_budget (the whole process),
// 0: this allocator's own allocations against the heap sizes. 0 when unknown.
int vk_mem_vram(uint64_t *used, uint64_t *budget, int *exact);
void vk_mem_heaps_log(const char *why);   // every heap's usage / budget / size and the imported guest memory
void vk_mem_footprint_log(const char *when);   // one "mem: footprint" line: the driver's count against VMA's
void vk_vma_name(VmaAllocation a, const char *fmt, ...);   // DEBUG-VMANAME: a name in VMA's map under REKARINA_VMA_DUMP
// VRAM headroom (vk_headroom.c): the VMA callbacks that see new device memory, and the thread that allocates the next
// block of each memory type in use ahead of the render thread.
const VmaDeviceMemoryCallbacks *vk_headroom_callbacks(void);
void vk_headroom_start(void);
void vk_headroom_stop(void);   // before vmaDestroyAllocator
void vk_mem_late_apply(uint64_t before);   // late sums of the submissions before that value, into guest memory
int vk_mem_mirror_flush(VkCommandBuffer cb, uint64_t fence);   // the write-back collected so far, mid-submission
void vk_mem_wb_pending(uint64_t *plain, uint64_t *merge);   // bytes of write-back the submission being recorded has collected
int vk_mem_early_pending(void);   // small results written after the results point, still to go back
int vk_mem_early_results(VkCommandBuffer cb, int *bound);   // REKARINA_WB_EARLY_RESULTS: they go back now
void vk_completion_add_result(uint64_t fence, void (*fn)(void *), void *arg);   // a CPU-visible side effect (vk_sync.c)
int vk_sync_results_sent(void);          // the submission being recorded has issued its results point
VkCommandBuffer vk_sync_chain(int n, VkCommandBuffer *cbs);   // the part goes on in n command buffers for other threads and a new one
int vk_sync_chain_pending(void);         // command buffers the part being recorded holds beside the current one
void vk_exec_par_join(void);             // vk_exec.c: every chained command buffer is recorded and ended
uint64_t vk_sync_cb_serial(void);        // changes whenever the replay starts a command buffer (its state is unset)
int vk_sync_results_on(void);            // REKARINA_RESULTS_FENCE: the game's waits take a submission's results as its completion
int vk_sync_results_point(void);         // the submission being recorded has its results: 1 when the part should end here
void vk_exec_flush_req(int on);
int vk_exec_held(void);
void vk_replay_flush_held(void);   // vk_backend.c: asynchronous replay: an empty submission that records every held rest, waited for
int gfx_replay_async(void);
void vk_sync_submits_drain(void);   // the driver has every batch the replay queued (vk_queue_lock calls it)
uint64_t vk_sync_submit_upload(VkCommandBuffer cb, VkFence fence);   // 0 if the caller must submit directly
int vk_sync_upload_submitted(uint64_t ticket, int wait);
uint64_t vk_sync_results_done(void);
int vk_sync_results_wait(uint64_t value, uint64_t timeout_ns);
void vk_mem_reap_drain(void);   // returns once every object handed to the reaper thread is destroyed
int vk_mem_resolve(uint64_t addr, uint64_t size, VkMemSpan *out);   // 1 when all of [addr, addr+size) is backed
// Changes whenever vk_mem_resolve plus VK_MEM_USE may give another span for an address: a pool added, finalized or
// mapped, a mirror created, refused or forgotten, a submission started.
uint64_t vk_mem_gen(void);
// Page write generations over guest memory (mem_written_pages plus vk_guest_written). Returns 1 when a page of the
// range was written after generation `since`, 0 when not, -1 when the range is not write-watched (treat as written);
// *now is the generation to pass next time.
int vk_mem_changed(uint64_t addr, uint64_t size, uint32_t since, uint32_t *now);
// The same answer from per-epoch write-watch snapshots of 2 MiB regions: no system call once the regions were queried in
// this epoch (a guest submission). For checks that may see writes made during the epoch one epoch late.
int vk_mem_changed_snap(uint64_t addr, uint64_t size, uint32_t since, uint32_t epoch, uint32_t *now);
void vk_mem_snap_stats(uint64_t *checks, uint64_t *calls, uint64_t *untracked);   // since the last call

// Upload batch: one command buffer on gfx_queue that texture uploads and texel copies record into. begin/end bracket
// the recording (recursive lock); vk_staging (between them) returns a mapped host buffer that lives until the batch
// completes, or, with alloc non-NULL, until the caller's vmaDestroyBuffer after vk_upload_flush(1). The replay calls
// vk_upload_flush(0) before each submission that may read uploaded images; queue order and the batch's closing
// barrier make the uploads visible to it. Returns the batch sequence number; wait=1 blocks until it completed.
// REKARINA_FIX_ZERO (default 1): physical pools are zeroed once at creation and GPU-only images with memory of their own
// are cleared when created; 0 leaves both as the driver hands them out.
int vk_fix_zero(void);
VkCommandBuffer vk_upload_begin(void);
void vk_upload_end(void);
void *vk_staging(VkDeviceSize size, VkBuffer *buf, VmaAllocation *alloc);
void vk_defer_call(void (*fn)(void *), void *arg);   // vk_memory.c: fn(arg) once the GPU is past what is recorded now
uint64_t vk_upload_flush(int wait);
// Deferred destruction: freed once the upload batches recorded so far and the timeline value last noted have completed.
void vk_defer_image(VkImage img, VmaAllocation alloc);   // alloc NULL: vkDestroyImage only
// vk_memory.c: an image bound into a pool's memory (alloc NULL, mem the pool's block or its twin) whose texture went is
// held for REKARINA_POOL_IMAGE_HOLD_MS (a depth image: REKARINA_POOL_IMAGE_HOLD_DEPTH_MS, 500) before its deferral, or until
// that memory goes; 1: held (the caller defers nothing)
int vk_mem_pool_image_hold(VkImage img, VmaAllocation mem, int depth);
void vk_defer_view(VkImageView view);
// free(p) once the GPU is past the submission being recorded, as vk_defer_image: a host record other threads may still
// read (a destroyed texture's, vk_texture.c destroy_tex).
void vk_defer_free(void *p);
// The deferrals this thread makes until vk_defer_past_reserved(0) wait for the GPU to pass every timeline value handed
// out so far (vk_sync_reserved): textures evicted or dropped under memory pressure (vk_texture.c sweep_pressure).
void vk_defer_past_reserved(int on);
int vk_exec_image_gone(VkImage image);   // texture lock held: the replay's views of image go with it; returns how many
int vk_exec_views_gone(VkImage image);   // the same for the views alone (a new image whose handle a destroyed one had)
void vk_defer_buffer(VkBuffer buf, VmaAllocation alloc);
// ---- ASTC (vk_astc.c). REKARINA_ASTC = auto | gpu | native | cpu | check; see vk_astc.c.
enum { VK_ASTC_CPU, VK_ASTC_GPU, VK_ASTC_NATIVE, VK_ASTC_CHECK };
int vk_astc_mode(void);                 // decided once per device (logged); VK_ASTC_CPU before vk_init
int vk_astc_native(VkFormat astc_fmt);  // 1: images of this ASTC format are created as such and get the raw blocks
int vk_astc_gpu(void);                  // 1: ASTC images decoded to RGBA8 by the compute decoder (vk_astc_record)
int vk_astc_bc(void);                   // 1: those images are BC3 instead, encoded after the decode (REKARINA_ASTC_BC=1)
// One image region decoded from raw blocks: tight rows of 16-byte blocks at byte src of the source buffer (slices one
// after the other), w x h texels and slices deep, written at offset of level / layer.
typedef struct { VkDeviceSize src; uint32_t w, h, slices, level, layer; VkOffset3D offset; } VkAstcRegion;
// Records into cb, the open upload batch (between vk_upload_begin and vk_upload_end): src (TRANSFER_SRC, src_size
// bytes of blocks) goes to a device-local scratch buffer, a dispatch per region decodes it, and the texels are copied
// into img (VK_IMAGE_LAYOUT_GENERAL). The caller records its usual barriers around an image copy before and after the
// call. readback (optional, host-visible, TRANSFER_DST): also gets the decoded texels, the regions one after the other.
// bc: img is BC3 (TexVkFormat.astc_bc); the texels are encoded to BC3 blocks before the copy, and readback is ignored.
// 0 when the decoder is unavailable or the scratch buffer cannot be allocated (nothing recorded).
int vk_astc_record(VkCommandBuffer cb, VkBuffer src, VkDeviceSize src_size, const VkAstcRegion *rg, int n, uint32_t bw,
                   uint32_t bh, int srgb, int bc, VkImage img, VkImageAspectFlags aspect, VkBuffer readback);
// The next vk_astc_record of this thread with bc also copies the BC3 blocks it makes into buf (host-visible,
// TRANSFER_DST), region after region, each vk_astc_bc_bytes long: what the texture cache keeps (vk_texture.c).
void vk_astc_bc_readback(VkBuffer buf);
// vk_texture.c, for tex_prepare.c: the texture cache's key of a storage (its file name begins with it) and its folder
// (NULL with tex_cache off)
uint64_t vk_texture_content_hash(const void *p, uint64_t n);
const char *vk_texture_cache_dir(void);
// 1 when every BC3 block (n bytes) has the opaque alpha half the encoder writes; the colour halves (BC1 blocks, n / 2
// bytes) are then written to out, which may be blocks itself (vk_texture.c, the BC1 section)
int vk_texture_bc3_to_bc1(const uint8_t *blocks, uint64_t n, uint8_t *out);
VkDeviceSize vk_astc_bc_bytes(uint32_t w, uint32_t h, uint32_t slices);
void vk_astc_stats(uint64_t *dispatches, uint64_t *texels);   // since start
// REKARINA_ASTC_TIME=1: GPU timestamps around each vk_astc_record's dispatches (logged per footprint and the largest one)
void vk_astc_time_enable(int on);
int vk_astc_time_get(uint32_t bw, uint32_t bh, uint64_t *texels, uint64_t *calls, double *us);   // totals so far
void vk_astc_time_summary(void);        // logs the totals
void vk_astc_release(void);             // vk_device.c, before the device goes: frees the decoder's objects
void vk_mem_note_submit(uint64_t timeline_value);  // vk_sync.c: after each submit that signals ctx->timeline with this value
void vk_mem_collect(void);                         // frees what has completed
void vk_mem_collect_idle(void);                    // a process without frames: every deferral its finished uploads left
typedef struct { uint64_t imported_pools, physical_pools, virtual_pools, import_failures, upload_batches; } VkMemStats;
void vk_mem_stats(VkMemStats *s);
void vk_mem_shutdown(void);                        // before vk_shutdown: waits idle, frees every memory and texture object

// Textures. Images are keyed by storage GPU address plus descriptor and stay in VK_IMAGE_LAYOUT_GENERAL. A mirrored
// texture (CPU-written pool storage, MNG tex_mirrored) is uploaded on first use and again when its storage pages were
// written; once the GPU wrote it (vk_texture_gpu_written) the image is its content. A returned pointer stays valid
// until the next call that may drop textures (release, pool fini); copy what outlives that.
typedef struct {
    VkImage image;
    VkFormat format;               // after the fallbacks of tex_vk_format (ASTC -> RGBA8, RGB -> RGBA)
    VkImageType type;
    VkImageViewType view_type;     // view of the whole image
    VkImageAspectFlags aspect;
    VkImageUsageFlags usage;
    uint32_t width, height, depth, levels, layers, samples;
    uint64_t addr;                 // storage GPU address
    uint8_t alpha_one;             // X format: sample alpha as 1 through the view swizzle
    uint8_t chan[4];               // TexVkFormat.chan: component order of a stand-in format, applied by the view swizzle
    uint8_t mirrored, gpu_owned;
    uint8_t gone;                  // destroyed (vk_texture.c destroy_tex, texture lock held): the record stays readable until
                                   // the GPU is past the submission being recorded (vk_defer_free), as the replay and DLSS
                                   // keep pointers to render targets across a command buffer; no view of it is cached
    uint8_t bound;                 // aliases physical-pool memory (physical and virtual pools)
    uint8_t block_views;           // BC image that renders through uncompressed views of its blocks (BLOCK_TEXEL_VIEW_
                                   // COMPATIBLE + EXTENDED_USAGE: only those views may use COLOR_ATTACHMENT)
    uint8_t swz[4];               // NVN texture swizzle selectors (nvnTextureBuilderSetSwizzle); a view swizzle replaces them
    uint8_t fb_usage;              // created with ATTACHMENT_FEEDBACK_LOOP usage (VK_FEEDBACK_LAYOUT, attachable images)
    uint8_t fb_learned;            // a draw sampled it while it was a written attachment of the open pass: its passes
                                   // attach it in ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT (vk_exec.c vk_exec_pass_begin)
    uint8_t scale;                 // 1, or vk_shadow_scale() for a shadow map made larger than the game's (vk_texture.c
                                   // shadow_scale_of): width and height above are the larger ones, the game's regions
                                   // are multiplied by it
    uint32_t serial;               // new for every image this record is given (vk_texture_image_serial): the view cache keys
                                   // by it with the handle, as Vulkan hands a destroyed image's handle to a later image
} VkTexImage;
#ifdef GAME_OOT
int vk_hdr_ui_guest_compose(VkCommandBuffer cb, const VkTexImage *source, VkImageView target,
                            int preview[4], const VkGuestPq *pq);
#endif
uint32_t vk_texture_image_serial(void);   // vk_texture.c: the next image serial (never 0)
// vk_texture.c: REKARINA_SHADOW_SCALE as the shadow maps are made with it, 1 to 4 (auto decided for the card), read
// once with the device up
int vk_shadow_scale(void);
// Views made ahead with an image (vk_exec_draw.c "Views made ahead with their image", REKARINA_VIEWS_AHEAD): the
// whole-image sampled view (0), attachment view (1) and 2D array view of a 2D image (2), each with the view cache key it
// was made for (a ViewKey).
#define VK_VIEWS_AHEAD 3
typedef struct { VkImage image; VkImageView view[VK_VIEWS_AHEAD]; uint64_t key[VK_VIEWS_AHEAD][6]; } VkViewsAhead;
int vk_exec_views_ahead_on(void);
void vk_exec_views_ahead(const VkTexImage *img, VkViewsAhead *va);    // any thread, no lock: makes them
void vk_exec_views_adopt(const VkTexImage *img, VkViewsAhead *va);    // texture lock held: into the view cache
void vk_exec_views_ahead_drop(VkViewsAhead *va);                      // never handed out: destroyed at once
void vk_exec_views_stats(int64_t *replay, int64_t *adopted, int64_t *dropped);   // since the last call
// gfx_present.c "driver turns": a background thread that makes or destroys Vulkan objects (images made ahead, the
// reaper) calls this before each such call, so that a thread holding or waiting for the queue lock goes first.
int vk_drv_turns_on(void);
void vk_drv_bg_turn(void);
void vk_drv_reap_turn(void);   // the reaper before each destruction: queue calls and images made ahead go first
int vk_texture_premake_pending(void);   // vk_texture.c: an image is queued or being made ahead
void vk_drv_urgent(int d);   // +1 before a queue call (submission, present), -1 after it
// The window texture the backend presents (vk_backend.c): *image and *format become the float picture the game copied
// into it, when the last draw into it was that copy and the swapchain is HDR10.
void vk_hdr_present_source(const VkTexImage *win, VkImage *image, VkFormat *format);
void vk_hdr_image_gone(VkImage image);
const VkTexImage *vk_texture_get(const tex_t *t);   // created and made current; NULL for BUFFER or unmapped formats
// A texture slot; a slot whose texture was finalized follows the live texture at the same storage (hle_nvn.c).
const VkTexImage *vk_texture_by_id(int id, view_t *view, int *has_view);
// The replay looks up many slots per draw: it takes the texture lock once around them (vk_texture_lock) and uses the
// _locked lookups; vk_texture_slots_gen changes whenever a slot may resolve to another image or view.
void vk_texture_lock(void);
void vk_texture_lock_from(const char *fn);   // the same, naming the caller for REKARINA_SECTIONS' LOCKHOLD lines
#define vk_texture_lock() vk_texture_lock_from(__func__)
void vk_texture_unlock(void);
const VkTexImage *vk_texture_by_id_locked(int id, view_t *view, int *has_view);
// The same without the texture lock when the slot's kept resolution is current (*hit 1); *hit 0: ask the locked one.
const VkTexImage *vk_texture_by_id_fast(int id, view_t *view, int *has_view, int *hit);
VkSampler vk_sampler_compare_by_id_locked(int id);   // the slot's sampler with its compare on (vk_texture.c), for a depth-compare read
VkSampler vk_sampler_by_id_locked(int id);
// The slot's sampler with nearest filtering and no anisotropy: a depth texture read without a compare lands on one texel
// on the Switch although the game binds a linear sampler, and MacNativeGames samples it the same way. VK_NULL_HANDLE
// for an empty slot.
VkSampler vk_sampler_point_by_id_locked(int id);
void vk_debug_samp3d_locked(int id, int separate, int format);   // DEBUG-SAMP3D
void vk_debug_image_where_locked(int id, const char *what);       // DEBUG-IMGWHERE
uint32_t vk_texture_slots_gen(void);
// What vk_texture_by_id(id) returns can change only when this does: the slot is registered again, or a texture entry at
// its storage address is created, dropped, released or taken back into use. Unlocked; read it before the lookup it
// guards (a lookup may create an entry). 0 when the slot was never registered: nothing is kept under 0.
uint64_t vk_texture_slot_token(int id);
void vk_sampler_dlss_bias(float bias);           // texture LOD bias added for DLSS upscaling (material samplers)
void vk_sampler_sharpness(float b);              // the game's negative material texture biases moved b towards 0
void vk_sampler_anisotropy(void);
uint32_t vk_sampler_gen(void);                      // changes whenever a sampler slot is registered
void vk_texture_gpu_written(const tex_t *t);        // rendered or copied into by the GPU: no re-upload from storage
#ifdef GAME_NS2
VkFormat vk_storage_format(VkFormat f);   // the UNORM format of an 8-bit sRGB one (storage views of NVN2 images), else f
#endif
void vk_texture_storage_bound_locked(const VkTexImage *img);   // bound as a storage image in the recording submission
int vk_texture_fill_colour_locked(const VkTexImage *img, uint8_t rgba[4]);   // 1: a small CPU-filled RGBA8 texture, its first texel
uint64_t vk_texture_last_write(const tex_t *t);    // timeline value of the newest submission recorded to write it; 0: none
size_t vk_texture_texel_span(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint64_t row, uint64_t img);   // bytes ReadTexels spans; 0 for a bad region
int vk_texture_small_dirty(void);                   // a small mirrored target holds GPU texels its storage lacks
int vk_texture_fence_readback(VkCommandBuffer cb, uint64_t fence);
void vk_texture_readback_tick(uint64_t frames);   // a present (gfx_counters.frames now): staged readbacks that are due   // a submission's end: its small targets go back at its fence
void vk_texture_write_back_small(uint64_t fence);
// A copy into a tiny CPU-visible texture was recorded (a readback): its bytes reach guest memory one presented frame
// later than our GPU could deliver them, as on the console, whose GPU runs about a frame behind the CPU.
void vk_texture_readback_hold(const tex_t *t);   // writes those rendered by work up to fence back to storage
void vk_texture_debug(const VkTexImage *img, uint32_t *uploads, int *last_change, int *pages_written);   // vk_dump
int vk_texture_slot_swz(int id, uint8_t swz[4]);    // vk_dump: swizzle of the descriptor in slot id
int vk_texture_read_image(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row,
                          uint64_t img);            // always from the image (GPU copy, waited for); 0 on failure
int vk_texture_cpu_stale(uint64_t addr, uint64_t size);    // a GPU-written mirrored texture there has stale storage
uint64_t vk_texture_dirty_fence(uint64_t addr, uint64_t size);   // newest recorded writer of such a texture; 0: none
void vk_texture_write_back(uint64_t addr, uint64_t size);   // writes those images back into their storage (waits)
void vk_texture_range_gone(uint64_t lo, uint64_t size, VmaAllocation alloc);   // vk_pool_fini: drop the images there
void vk_texture_stats(uint64_t *images, uint64_t *uploads, uint64_t *reuploads);
// Entries now, live and released (TextureFinalize, kept until sweep_released), with the bytes of their own memory
// (images not bound into pool memory).
typedef struct { uint64_t live, released, own_live, own_released, bytes_live, bytes_released; } VkTexCensus;
void vk_texture_census(VkTexCensus *s);
void vk_texture_shutdown(void);                     // vk_mem_shutdown calls it
void vk_texture_new_submit(void);                   // a queue submit starts: draw bindings check storage again
void vk_texture_room_for(uint64_t bytes);           // a buffer pool's mirror waits for this much room (vk_memory.c)
void vk_texture_vram_history_report(void);         // vk_device_fault_report: the last minute of video memory
uint32_t vk_texture_epoch(void);                    // the write-watch snapshot epoch vk_texture_new_submit started
// vk_texture.c "lookups as of a place in the command stream": a draw stretch's mark, a thread's limit (returns the one
// before), the limit in force, and whether slot id resolves under it as it does without one.
uint64_t vk_texture_asof_mark(void);
uint64_t vk_texture_asof(uint64_t limit);
uint64_t vk_texture_asof_limit(void);
int vk_texture_asof_clean(int id);
VkSampler vk_sampler_by_id(int id);                 // VK_NULL_HANDLE for an empty slot
VkSampler vk_sampler_for(const samp_par *s);        // cached by content
int vk_sampler_params(int id, samp_par *out);       // parameters of a sampler slot; 0 for an empty slot

// 8-bit RGBA PNG with stored deflate blocks (boot.c); stops with rt_fatal when the file cannot be written.
void rt_write_png(const char *path, const uint8_t *rgba, uint32_t w, uint32_t h);

// ---- submission, synchronisation, completion (vk_sync.c, vk_backend.c)
// Fence values are ctx->timeline values: each submission signals the next one, in submit order. A value counts as done
// once the GPU reached it and the completion thread ran its side effects. Guest or host threads.
void vk_sync_start(void);                  // vk_init (fatal without a device) and the completion thread; idempotent
// Async compute (vk_sync.c, vk_exec.c "Queues interleaved"): on; a compute command buffer and the cp value it will
// signal; its submission after graphics value gp; the last cp value submitted; the most any graphics part waits for;
// the graphics part being recorded waits for cp >= v; vk_sync_split whose part also signals the next graphics value.
int vk_sync_ac_on(void);
VkCommandBuffer vk_sync_ac_begin(uint64_t *value);
void vk_sync_ac_submit(VkCommandBuffer cb, uint64_t gp);
uint64_t vk_sync_ac_last(void);
uint64_t vk_sync_ac_covered(void);
void vk_sync_ac_need(uint64_t v);
VkCommandBuffer vk_sync_split_gp(uint64_t *gp);
// DEBUG-SPIKE: REKARINA_SPIKE_MS=<ms>: where a slow submission's time went (ns; replay thread, reset per submission)
typedef struct { uint64_t ev, usnap, mirb, mire, gmem, tex, ntex, op[96], max_ns; uint32_t max_op; } VkSpike;
extern int vk_spike_on;
extern VkSpike vk_spike;
uint64_t vk_sync_submit(const uint64_t *handles, int n);   // records with vk_exec_record, submits; returns the value
uint64_t vk_sync_reserve(void);            // asynchronous replay: the value the next queued submission signals
int vk_sync_fence_wait(uint64_t value, uint64_t timeout_ns);   // 1 when done; 0 on timeout or a value never submitted
uint64_t vk_sync_fence_done(void);         // highest done value
uint64_t vk_sync_fence_issued(void);       // highest submitted value
// Async write-back (vk_sync.c, REKARINA_ASYNC_WB): 1 when on; the command buffer the write-back of the submission of value
// fence is recorded into (begun; replay thread); the value up to which write-backs are done (UINT64_MAX when off); the
// part being recorded must wait for the write-back of value fence (it overwrites what that one reads).
int vk_sync_awb_on(void);
VkCommandBuffer vk_sync_awb_begin(uint64_t fence);
uint64_t vk_sync_awb_done(void);
void vk_sync_awb_need(uint64_t fence);
int vk_mem_gpu_write_pending(uint64_t addr, uint64_t size);
// vk_sync.c: the prologue timeline (completed value, wait for a value) that frees the ring of submit-time mirror copies.
void vk_mem_prologue_timeline(uint64_t (*done)(void), int (*wait)(uint64_t));
void vk_mem_prologue_submitted(uint64_t value);   // vk_sync.c: the last prologue went out, signalling value (copy guards)   // replay: a recorded GPU write there is not done yet
uint64_t vk_sync_issued(void);             // newest submitted value (the one being recorded is this + 1)
int vk_sync_epoch_early(void);             // DEBUG-EPOCHEARLY (vk_sync.c)
int vk_sync_sub_work(uint64_t value);      // DEBUG-LIVECHECK (vk_sync.c)
int vk_live_check_on(void);                // DEBUG-LIVECHECK (vk_memory.c): REKARINA_LIVE_CHECK
int vk_live_check_use(uint64_t handle, const char *what, uint64_t id);
void vk_live_check_born(uint64_t handle);
void vk_live_check_dead(uint64_t handle);   // an image whose texture went lives on (vk_texture.c park_image)
void vk_live_check_log(void);
void vk_sync_finish(void);                 // every submitted value done
void vk_sync_shutdown(void);               // finish, stop the completion thread, free the command buffers
// The op-stream replay (vk_exec.c): records the command handles of one submission into cb, which the caller has
// begun and ends. fence is the value this submission will signal. Returns the number of commands recorded into the
// command buffer it holds at the end (0: that one stays empty and is not submitted). A weak vk_sync.c version accepts
// empty streams only.
int vk_exec_record(VkCommandBuffer cb, const uint64_t *handles, int n, uint64_t fence);
// Inside vk_exec_record, outside a render pass: ends the command buffer recorded so far, submits it (after the upload
// batch) without a signal, and returns a new begun one for the rest of the submission. The submission's timeline value
// is signalled by its last part, whose signal covers every earlier part in submission order.
VkCommandBuffer vk_sync_split(void);
// Runs fn(arg) on the completion thread once fence is done on the GPU (event writes, counter reports, sync state).
// Callbacks run in the order they were added, after those of earlier values; a value already done runs at once there.
void vk_completion_add(uint64_t fence, void (*fn)(void *), void *arg);
// gfx_vulkan_backend (vk_backend.c): presents go to a present thread and run after the submissions made before them.
void rdoc_init(void);                     // runtime/rdoc.c: REKARINA_RDOC debug capture; before vkCreateInstance
int rdoc_active(void);
void rdoc_after_present(uint64_t frame);
void vk_present_drain(void);               // waits for the blit submission, or the flip if DLSS-G inputs are tagged
void vk_replay_captured(void);         // vk_exec.c -> vk_backend.c: the submission being recorded has taken its copies
int vk_replay_async(void);                 // REKARINA_REPLAY_ASYNC (default 1): submissions replay on the replay thread
void vk_program_release(const prog_t *p);  // GfxBackend.program_release: vk_pipeline.c; weak no-op in vk_backend.c
void vk_backend_shutdown(void);            // presents, vk_sync_shutdown, vk_mem_shutdown, vk_shutdown in that order

// ---- pipelines (vk_pipeline.c)
// Binding model (comment block at the top of vk_pipeline.c): one descriptor set, set 0, per pipeline. Binding 0 is
// Ryujinx's support buffer (a uniform buffer every stage shares); the vertex stage keeps its own bindings, the fragment
// stage's follow them (its SPIR-V is renumbered). Set 0 is a push-descriptor set unless the pipeline needs more than
// maxPushDescriptors (32 here) descriptors; VkPipeInfo.bindings says what goes where. No push constants.
// Everything a graphics pipeline depends on, as the op stream carries it. Fields covered by dynamic state are read by
// vk_pipeline_set_dynamic only and left out of the pipeline key.
typedef struct VkPipeState {
    const GfxShader *vs, *fs;              // gfx_shader_get results; fs NULL: no fragment stage (depth only)
    int32_t color_fmt[8];                  // NVNformat of colour target i (OP_SET_RT), 0 none
    int32_t depth_fmt;                     // NVNformat of the depth/stencil target, 0 none
    uint8_t samples;                       // sample count of the targets; 0 or 1 single-sampled
    uint8_t prim;                          // NVN_DRAW_PRIMITIVE_* of the draw
    uint8_t depth_minus_w;                 // nvn_depth_minus_w(): clip-space depth is -w..w
    uint8_t depth_clamp, raster_discard;   // OP_DEPTH_CLAMP, OP_RASTER_DISCARD
    uint8_t y_flip;                        // the replay's viewports have negative heights (upper-left origin): the
                                           // front face is not mirrored
    uint8_t prim_restart;                  // the draw's indices restart its strip or fan at the all-ones value
                                           // (OP_PRIM_RESTART, set by op_draw_idx); dynamic state only
    uint8_t feedback;                      // the fragment shader samples an attachment of the pass in the feedback
                                           // layout: 1 colour, 2 depth/stencil (VK_PIPELINE_CREATE_*_FEEDBACK_LOOP_BIT_EXT)
    blend_st blend[8];                    // OP_BIND_STATE kind 0, indexed by blend_st.target
    color_st color;                        // kinds 1-5
    chmask_st chmask;
    ms_st ms;
    poly_st poly;
    ds_st ds;
    uint32_t vattr[16];                    // OP_BIND_VATTR: va_st words; attribute i feeds vertex shader Location i
    vs_st vstream[16];                     // OP_BIND_VSTREAM
} VkPipeState;
void vk_pipe_state_defaults(VkPipeState *s);   // NVN defaults of every state object (shaders and formats left 0)
enum { VK_PIPE_SUPPORT = GFX_RES_KINDS };      // VkPipeBinding.kind of the support buffer
#define VK_PIPE_ZERO_BINDING 16                // vertex binding of shader inputs without an attribute: bind a zeroed
                                               // buffer of at least 16 bytes there (stride 0)
typedef struct {
    uint16_t binding;                      // in its set
    uint8_t stage;                         // GFX_STAGE_* that declared it (support buffer: the first stage)
    uint8_t kind;                          // GFX_RES_* or VK_PIPE_SUPPORT
    VkDescriptorType type;
    uint16_t count;
    VkShaderStageFlags stages;
    const GfxShaderRes *res;               // the stage's reflection entry; NULL for the support buffer
    uint8_t set;                           // descriptor set: 0, or 1 for the fragment stage with independent layouts
    // Filled at the info's first use by a draw (vk_exec_draw.c pipe_prep): res->slot and the NVN stage, so the build of a
    // draw's descriptors reads the bindings in a row and not one reflection entry per binding, each on its own cache line
    int8_t nst;
    int16_t slot;
} VkPipeBinding;
typedef struct {
    VkPipeline pipeline;                   // VK_NULL_HANDLE when creation failed
    VkPipelineLayout layout;
    VkDescriptorSetLayout set_layout;      // set 0
    int push;                              // 1: write set 0 with vkCmdPushDescriptorSetKHR; 0: allocate and bind a set
    VkPipelineBindPoint bind_point;
    uint32_t n_bindings;
    const VkPipeBinding *bindings;
    uint32_t vs_inputs;                    // Locations the vertex shader reads
    // Independent layouts (vk_caps.gpl): set 0 holds the vertex stage's bindings, set 1 the fragment stage's; bindings
    // are ordered by set. One of the two sets at most is a push set. n_sets 1: everything in set 0 (REKARINA_GPL=0, compute).
    VkDescriptorSetLayout set_layout1;
    int push1;
    uint32_t n_sets;
    // Descriptor update template per set (vk_pipeline.c make_tpl): reads one VkDescSlot per descriptor, in binding order,
    // from the start of the set's descriptors. VK_NULL_HANDLE: the set is written with VkWriteDescriptorSet structs.
    VkDescriptorUpdateTemplate tpl[2];
    int pool;                              // the layout has the texture pool's set at GFX_POOL_SET (vk_texpool.c)
    int mesh_standin;                      // the fast-linked mesh pipeline has no dynamic shading rate
    // pipe_prep (vk_exec_draw.c), once per info: bindings [set_lo, set_hi) per set, and whether a set has no template
    uint8_t prep, need_w;
    uint32_t set_lo[2], set_hi[2];
} VkPipeInfo;
// One descriptor of the packed array a descriptor update template reads (VkPipeInfo.tpl), 24 bytes for every type; the
// bytes a type leaves unused are zero, so equal descriptors compare equal byte for byte.
typedef union { VkDescriptorImageInfo image; VkDescriptorBufferInfo buffer; VkBufferView view; uint64_t w[3]; } VkDescSlot;
// The texture pool's descriptor set (vk_texpool.c, gfx.h GFX_POOL_SET): every registered texture and sampler by id, for
// shaders that index the pool with handles they read from memory.
VkDescriptorSetLayout vk_tpool_layout(void);        // the set's layout; VK_NULL_HANDLE when the device cannot hold it
VkDescriptorSetLayout vk_tpool_empty_layout(void);  // set 1 of a layout without independent sets that has the pool set
// Brings the set up to date for the recording submission (once per vk_texture_epoch) and binds it at GFX_POOL_SET of
// layout. Replay thread, texture lock not held.
void vk_tpool_bind(VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout layout);
void vk_tpool_pump(void);                           // a submission with no scene on screen: resolves slots within a budget
void vk_tpool_shutdown(void);
void vk_tpool_prepare(void);                        // the submission's update now, ahead of binds on other threads
void vk_tpool_slot_dirty(int id);                   // texture lock held: slot id resolves again at the next update
// vk_texture.c "handle tables": a storage buffer of a draw whose pipeline reads the texture pool (any recording thread)
void vk_texture_handle_table(uint64_t addr, uint64_t size);
void vk_tpool_image_gone(VkImage image);            // destroy_tex (texture lock held): its pool entries resolve again
int vk_texture_slot_count(void);                    // texture slot ids below this may be registered
// vk_texture_by_id_locked for vk_texpool.c: NULL as well when the slot's storage is no longer in a live memory pool.
const VkTexImage *vk_texture_pool_slot_locked(int id, view_t *view, int *has_view);
// vk_texpool.c, texture lock held: 1 when the texture stream is still preparing slot id's texture (asked less than 30
// frames ago); the update leaves a slot that shows the stand-in for a later one then.
int vk_texture_stream_defer(int id);
// vk_backend.c, the game's thread registering t in its texture pool: the texture stream prepares t (REKARINA_TEX_STREAM).
void vk_texture_stream_ask(const tex_t *t);
// vk_texpool.c (REKARINA_TPOOL_HIDE, a test of what shaders read through the pool): 1 a texture decoded from ASTC, 2 another
// single-layer texture the CPU provides, 4 an array or a volume the CPU provides, 0 anything else; with 8 added when
// a draw has bound the texture. Texture lock held.
int vk_texture_pool_class(const VkTexImage *img);
uint64_t vk_mem_vma_free(void);   // vk_memory.c: free room inside the allocator's device-local blocks
// vk_memory.c: the allocator's device-local blocks (count, bytes) and the bytes allocated in them, e.g. before and after a
// bulk free (blocks emptied = memory back to the driver; the rest is holes)
void vk_mem_vma_blocks(uint64_t *blocks, uint64_t *block_bytes, uint64_t *alloc_bytes);
uint64_t vk_mem_vram_cap(void);   // DEBUG-VRAMCAP (vk_memory.c)
uint32_t vk_pipeline_compute_trim(void);   // vk_pipeline.c: drops the prebuilt compute pipelines nothing dispatched
// vk_exec_draw.c for vk_texpool.c: a sampled view of a registered texture for a resource type (GfxShaderRes.type),
// VK_NULL_HANDLE when the image cannot serve it (texture lock held); the replay's stand-in views and sampler.
VkImageView vk_exec_pool_view(const VkTexImage *img, const view_t *v, int hv, uint16_t rtype);
void vk_exec_pool_dummies(VkImageView *view_2d, VkImageView *view_2d_array, VkSampler *samp);
// Both return the cached pipeline (created on first use; failures are cached and logged once). info: may be NULL.
// Valid until vk_pipeline_shutdown; thread-safe.
VkPipeline vk_pipeline_graphics(const VkPipeState *s, const VkPipeInfo **info);
// The replay's variant: after vk_pipeline_async_enable a missing pipeline is built by background workers and this
// returns VK_NULL_HANDLE with *pending 1 until it is ready (the draw is skipped); otherwise as vk_pipeline_graphics.
// wait_ms > 0 waits up to that long for a pending pipeline first. Pipelines built in a run are recorded to
// pipeline-keys.bin next to the pipeline cache and prebuilt at the next boot (REKARINA_PIPE_PREBUILD=0: not).
VkPipeline vk_pipeline_graphics_async(const VkPipeState *s, const VkPipeInfo **info, int *pending, uint32_t wait_ms);
// rt_boot calls it unless --shot is given; REKARINA_PIPE_SYNC=1 makes it a no-op. Probes never call it.
void vk_pipeline_async_enable(void);
int vk_pipeline_async_on(void);   // 1 after vk_pipeline_async_enable: graphics shaders are translated in the background too
// Boot prewarm, before the guest runs (rt_boot): loads the shaders pipeline-keys.bin names, translating the ones not
// cached under the current translator from their stored inputs, and builds every recorded pipeline, on all cores.
// No-op without vk_pipeline_async_enable, with REKARINA_PIPE_PREBUILD=0 or with REKARINA_GPU=null.
void vk_pipeline_prewarm(void);
// A normal exit while the entry step builds: its journal is cleared, so the next start does not take the builds in
// flight for a driver crash (vk_pipeline.c, "the entry step"; prep_report).
void vk_pipeline_prewarm_quit(void);
VkPipeline vk_pipeline_compute(const GfxShader *cs, const VkPipeInfo **info);
#ifdef GAME_NS2
// NVN2 (GAME_NS2): a pipeline with stages the library path does not know, built whole on the calling thread and cached.
// mesh 0: sh = {vertex, geometry, fragment or NULL}; mesh 1: sh = {task or NULL, mesh, fragment or NULL}. s->vs and
// s->fs are ignored (sh names the shaders); the rest of s as for vk_pipeline_graphics. VK_NULL_HANDLE on failure (cached).
VkPipeline vk_pipeline_stages(const VkPipeState *s, const GfxShader *const sh[3], int mesh, const VkPipeInfo **info);
int vk_pipeline_mesh_on(void);   // 1: the device has task and mesh shaders
void vk_pipeline_draw_mesh(VkCommandBuffer cb, uint32_t groups);                        // vkCmdDrawMeshTasksEXT(groups, 1, 1)
void vk_pipeline_draw_mesh_indirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off);   // one {x, y, z} record
#endif
// Records every dynamic state that VkPipeState determines, after binding a graphics pipeline: cull mode, front face,
// topology, depth and stencil tests and ops, depth bias enable, rasterizer discard, vertex input, and, where the GPU
// has them dynamic, depth clamp, polygon mode, blend enables, equations, write masks and alpha to coverage. The replay
// sets the rest: vkCmdSetViewportWithCount, vkCmdSetScissorWithCount, depth bias, blend constants, stencil compare and
// write masks and references; vertex buffers with vkCmdBindVertexBuffers2 (strides NULL when
// vk_pipeline_vertex_input_dynamic(), else from VkPipeState.vstream).
void vk_pipeline_set_dynamic(VkCommandBuffer cb, const VkPipeState *s);
void vk_pipeline_set_dynamic_no_assembly(VkCommandBuffer cb, const VkPipeState *s);
int vk_pipeline_vertex_input_dynamic(void);    // 1: vertex input is VK_EXT_vertex_input_dynamic_state
// Translator answers from the bound state: attr_type from the vertex attribute formats (vs), frag_out_type from the
// colour target formats (fs). Other fields are left as they are; either pointer may be NULL.
void vk_pipeline_answers(const VkPipeState *s, GfxShaderAnswers *vs, GfxShaderAnswers *fs);
int vk_pipeline_tex_type(int nvn_target);      // NVN_TEXTURE_TARGET_* -> GFX_TEX_* (GfxShaderAnswers.tex_type)
uint32_t vk_pipeline_target_wmask(const VkPipeState *s, int i);   // colour write mask of target i the pipeline uses
VkPrimitiveTopology vk_pipeline_topology(int nvn_prim);   // quads -> triangle list, line loop -> strip: the replay
                                                          // rewrites those draws; polygon -> fan
VkFormat vk_vertex_format(int nvn_format, int *type);     // attribute format (supported fallback), GFX_TYPE_* in *type
PFN_vkCmdPushDescriptorSetKHR vk_pipeline_push_fn(void);  // after the first vk_pipeline_* call
// Descriptor update templates (after the first vk_pipeline_* call): NULL when templates are off (REKARINA_DESC_TEMPLATE=0,
// a device below Vulkan 1.1) or, for the push variant, vkCmdPushDescriptorSetWithTemplateKHR is missing.
PFN_vkCmdPushDescriptorSetWithTemplateKHR vk_pipeline_push_tpl_fn(void);
PFN_vkUpdateDescriptorSetWithTemplate vk_pipeline_update_tpl_fn(void);
// Probe only: 1 selects stage-independent layouts for pipelines looked up from now on, 0 the shared set 0; returns
// the previous mode. 1 needs vk_caps.gpl (the independent-sets flag comes with the extension).
int vk_pipeline_set_indep(int on);
typedef struct { uint64_t graphics, compute, failures, hits, layouts, push_layouts, async_built, pending_skips; } VkPipeStats;
void vk_pipeline_stats(VkPipeStats *s);
void vk_pipeline_save(void);               // writes the VkPipelineCache to pipeline-cache.bin next to rekarina.exe
void vk_pipeline_autosave(void);           // vk_pipeline_save at most every 20 s, when pipelines were built since
void vk_pipeline_shutdown(void);           // waits for background builds, saves, destroys pipelines, layouts and the
                                           // cache; before vk_shutdown
int vk_pipeline_selftest(void);            // one graphics and one compute pipeline from census shaders; 0 on success

// vk_movienet.c: the movies' network without its U-Net where a movie frame does not fit the movie's frame time: whether
// it is left out now, whether a dispatch of this program is skipped, whether it is pass 12, pass 12's module that adds
// nothing of the U-Net (NULL for another program), the GPU timing a movie needs meanwhile (perf.c), the per-present
// measurement (vk_backend.c); vk_pipeline.c builds pass 12's pass-through pipeline
int vk_movienet_on(void);
int vk_movienet_skip(const GfxShader *cs);
int vk_movienet_is_pass12(const GfxShader *cs);
uint32_t *vk_movienet_module(const GfxShader *cs, const uint32_t *code, uint64_t words, uint64_t *out_words);
int vk_movienet_timing(void);
void vk_movienet_frame(void);
VkPipeline vk_pipeline_compute_passthrough(const GfxShader *cs, const VkPipeInfo **info);

#include "vk_rec.h"   // last: its macros route the vkCmd* calls of every file that includes vk.h
