// DLAA (vk_dlaa.c): NVIDIA DLSS at the scene's own size, behind REKARINA_DLAA (rekarina.ini dlaa = 1).
#pragma once
#include "cfg.h"
#include "motion_vectors.h"
#include "scene_projection.h"
#include "vk.h"
#include "vk_ffx.h"

// The narrowest scene the DLSS pass and the motion passes recognise by its D32F depth: Ultra Performance at a 1080p
// output renders 640 wide, smaller windows less.
#define DLAA_SCENE_MIN_W 320u

int vk_dlaa_wanted(void);   // the setting is on (the device may still lack DLSS)
int vk_dlaa_prepare_sl_sr(void);   // prepare NVIDIA SR at Vulkan creation, independently of live activation
int vk_dlaa_backend_available(void);   // Selected backend readiness, independent of per-frame consumer flags.
int vk_dlaa_fsr3(void);
int vk_dlaa_fsr1(void);     // the upscaler behind dlaa is FSR 1, spatial (REKARINA_TAA = fsr1)
// Frame generation: the scene's depth, motion vectors, jitter, and history reset; 1 once per publication.
// The images stay valid until the next frame's scene draws over them, after this present.
int vk_dlaa_fg_inputs(VkFfxImage *depth, VkFfxImage *mv, float *jx, float *jy, int *reset);
// ... their views, and this frame's camera for DLSS Frame Generation (vk_dlssg.c): matrices for row vectors (v * M, the
// rows stored in order), as Streamline takes them; 0 without a camera that frame.
#define VK_DLAA_FAR_INFINITE 1e8f
typedef struct {
    float view_to_clip[16], clip_to_view[16], clip_to_prev_clip[16], prev_clip_to_clip[16];
    float pos[3], up[3], right[3], fwd[3];
    float fov_y, aspect;
    float znear, zfar;
    int inverted, infinite, depth_planes_valid;
} VkDlaaCamera;
int vk_dlaa_fg_views(VkImageView *depth, VkImageView *mv);
// The picture without the HUD at the size of the game's picture, in *fmt: RGBA8 UNORM bytes like it (sRGB-encoded), or
// RGBA16F linear for PQ presentation; *pq is the active encoding sampled at capture, not inferred from *fmt.
// Returns a matching capture once per frame.
#ifndef GAME_OOT
extern int vk_dlaa_hudless_wanted;
#endif
void vk_dlaa_seal_scene(uint64_t frame, VkImage guest, VkImage source, uint32_t serial);
int vk_dlaa_fg_hudless(uint64_t frame, VkImage *img, VkImageView *view, uint32_t *w, uint32_t *h, VkFormat *fmt, int *pq);
int vk_dlaa_fg_camera(VkDlaaCamera *c);
// vk_device.c: NGX's instance and device extensions join the lists (room for max names); vk_dlaa_init after the
// device exists brings NGX up and checks that DLSS is available.
void vk_dlaa_instance_exts(const char **ext, uint32_t *n, uint32_t max);
void vk_dlaa_device_exts(const VkExtensionProperties *have, uint32_t nhave, const char **ext, uint32_t *n, uint32_t max);
void vk_dlaa_init(void);
void vk_dlaa_release_ngx(VkDevice device);   // After GPU idle, before destroying the device.
// 1 when dlaa is on and NGX came up with DLSS (after vk_dlaa_init): a DLSS mode's smaller scene is scaled up by DLSS.
int vk_dlaa_available(void);
int vk_dlaa_active(void);
const char *vk_dlaa_nvidia_scene_aa(void);   // replay thread: label of a successful NVIDIA resolve in this frame, or NULL
int vk_dlaa_fg_only(void);
int vk_dlaa_pose_depth(uint64_t frame, uint32_t w, uint32_t h, VkDescriptorImageInfo *out, float history[4]);
const SceneProjection *vk_dlaa_previous_projection(uint64_t frame);
void vk_dlaa_scene_projection(const p_scene_projection *p);
void vk_dlaa_warm(void);          // boot: the first DLSS feature, before the guest_tonemap_target starts
int vk_dlaa_sharpens(void);        // DLSS's picture is sharpened at the present on every frame (vk_fsr.c)
// vk_exec.c vk_exec_pass_begin, once the pass's targets are resolved and before its barrier: arms the frame on the
// scene's light target or, at the first quarter-width pass after it, runs DLAA on that target. Each present ends the
// frame (vk_dlaa_frame).
typedef struct VkExec VkExec;
typedef struct VkExecTarget VkExecTarget;
int vk_dlaa_prebloom_enabled(void);
int vk_dlaa_prebloom_tonemap(VkExec *execution, VkExecTarget *target);
void vk_dlaa_prebloom_finish_tonemap(VkExec *execution, const VkExecTarget *guest_tonemap_target);
void vk_dlaa_stock_tonemap(const VkExecTarget *target);
int vk_interface_coverage_enabled(void);
int vk_interface_coverage_serial_required(const VkExec *execution);
void vk_interface_coverage_release(void);
int vk_interface_coverage_candidate(const struct VkExec *execution);
VkImageView vk_interface_coverage_begin(struct VkExec *execution);
void vk_interface_coverage_reject(const char *reason);
void vk_interface_coverage_write(VkImage image, uint64_t address);
void vk_interface_coverage_check_draw(struct VkExec *execution);
int vk_interface_coverage_source(VkImage image);
void vk_interface_coverage_copy(VkImage source, VkImage destination, uint64_t address, int full_copy);
VkImageView vk_interface_coverage_overlay(VkCommandBuffer cb, uint64_t frame, VkExtent2D extent, VkFormat format);
int vk_interface_coverage_present(VkCommandBuffer command_buffer, uint64_t frame, VkImage source, VkExtent2D source_extent,
                                  VkExtent2D presentation_extent, const int rectangle[4], const int destination[4],
                                  VkImageView overlay, const int preview[4], VkImage *image, VkImageView *view);
void vk_dlaa_scene_bloom(const VkExec *execution, int reconstructed);
void vk_dlaa_prebloom(VkExec *execution, uint64_t handle, unsigned role);
VkImageView vk_dlaa_prebloom_view(const VkExec *execution, uint64_t handle);   // texture lock held
int vk_dlaa_bloom_targets(VkExec *execution, unsigned role, unsigned level, const VkExecTarget *guest_targets);
void vk_dlaa_bloom_finish(VkExec *execution, const VkExecTarget *guest_targets, int drew);
int vk_dlaa_prebloom_active(void);
void vk_dlaa_pass(VkExec *execution);
void vk_dlaa_frame(void);
void vk_dlaa_frame_settings(void);
unsigned vk_dlaa_resolve_consumers(CfgTaa taa, int wanted);
// vk_exec_draw.c, each draw before its fixed state is recorded: notes the scene camera block (vk_dlaa_draw), and moves
// the viewport of the scene's draws by the frame's jitter (vk_dlaa_viewport, when the viewport is recorded).
void vk_dlaa_draw(VkExec *execution);
// Row-major view-projection and its inverse for the shadow mask filter.
void vk_dlaa_ocam_vp(const float *rows, double out[16]);
int vk_dlaa_inv4(const double m[16], double out[16]);
void vk_dlaa_pjitter(VkExec *execution, int stage);   // the scene camera block's projection jitter (REKARINA_DLAA_PJITTER)
// vk_exec.c, at the start of a submission's recording: the projection jitter in the game's own scene context blocks
// (REKARINA_DLAA_PJITTER=2), before the uniform and mirror copies are taken
void vk_dlaa_scene_jitter(VkExec *execution, const uint64_t *handles, int n);
int vk_dlaa_pathlog(void);   // DEBUG-PATHLOG
// vk_exec_draw.c, a compute dispatch about to be recorded: 1 when it is the game's scaling dispatch and DLSS took its place
int vk_dlaa_dispatch(VkExec *execution, const GfxShader *cs, uint64_t caller);
#ifdef GAME_OOT
int vk_dlaa_skip_smaa(VkExec *execution, uint64_t caller);
#endif
void vk_dlaa_dispatch_recorded(VkExec *execution);
void vk_dlaa_ubodump(VkExec *execution, const GfxShader *vs, const GfxShader *ms, const GfxShader *fs, int mesh);   // DEBUG-UBODUMP
void vk_dlaa_viewport(VkExec *execution, VkViewport *v);
// vk_mvsurvey.c (REKARINA_MV_SURVEY): notes the uniform blocks of each scene draw, beside vk_dlaa_draw; each present
// compares them with the previous frames' and logs a summary now and then.
void vk_mvsurvey_draw(VkExec *execution);
void vk_mvsurvey_frame(void);
void vk_mvprev_draw(VkExec *execution);   // vk_mvprev.c: notes the scene draw's object and bone blocks
// Per-object motion (vk_exec_draw.c): the moving scene draws are listed again during the replay with
// their vertex shader's last-frame variant; vk_dlaa.c mv_pass records them over the camera motion into execution->cb (and
// leaves every binding of the replay to be set again; draws whose camera block is not cam are left out).
// vk_dlaa_prev_camera: last frame's camera block and this frame's jitter, 0 when there is no history.
int vk_fg_pose_prepare(VkExec *execution, uint64_t cam, VkImage depth, uint32_t w, uint32_t h, uint64_t token);
int vk_mvobj_execute(VkExec *execution, uint64_t cam, VkImageView mv, VkImageView depth, VkFormat depth_fmt, uint32_t w,
                     uint32_t h, uint64_t token);
int vk_dlaa_prev_camera(VkDescriptorBufferInfo *out, float jitter[2]);
// vk_mvcheck.c (REKARINA_MV_CHECK, test): mv_pass hands it the motion image after the camera pass and, after the object
// pass, the motion image again and the scene light target it ran on; it measures how well the vectors fit the picture.
void vk_mvcheck_camera(VkCommandBuffer cb, VkImage mv, uint32_t w, uint32_t h, float jx, float jy);
void vk_mvcheck_final(VkCommandBuffer cb, uint64_t fence, VkImage mv, VkImage cimg, VkFormat cfmt, uint32_t level,
                      uint32_t layer, uint32_t w, uint32_t h, float jx, float jy);
const char *vk_mvobj_frame_masks(void);   // vk_exec_draw.c: the last motion list, per position and changed banks
void vk_mvcheck_output(VkCommandBuffer cb, VkImage out, uint32_t ow, uint32_t oh);   // after DLSS on a check frame
// vk_exec_draw.c, once a draw's fragment shader is known and before its state is bound: 1 skips the draw (the game's FXAA
// while REKARINA_FXAA is 0, its scaling draw when DLSS wrote the scaled picture). It also runs OoT's resolve before the
// bloom (vk_dlaa.c pre_bloom_draw).
int vk_dlaa_skip_draw(VkExec *ex, const GfxShader *fs);
