// The op-stream replay of the Vulkan back end. vk_exec.c walks the command
// handles of a submission, keeps render targets, passes, clears and fixed-function state; draws, binds and dispatches
// (vk_exec_draw.c) and copies, counters, events and conditional rendering (vk_exec_xfer.c) plug in through the two
// hooks at the bottom.
// Runs on the submitting thread under vk_sync_submit's lock: no locking inside.
// The op loop follows MacNativeGames nvn_mtl.m (MIT, Copyright (c) 2026 jhaizhou) and the pass handling LAS Recomp
// (MIT, Copyright (c) 2026 Peiikai); vk_exec.c names the sources.
#pragma once
#include <stdint.h>

#include "nvn_api.h"
#include "vk.h"
#include "platform/platform.h"

// Dynamic state the next draw must record (VkExec.dirty). vk_exec_record sets them all at the start of each command
// buffer; the state ops set theirs; vk_exec_draw.c clears what it recorded.
enum {
    VK_EXEC_DIRTY_VIEWPORT = 1 << 0,     // viewports or depth range
    VK_EXEC_DIRTY_SCISSOR = 1 << 1,
    VK_EXEC_DIRTY_DEPTH_BIAS = 1 << 2,   // polygon offset
    VK_EXEC_DIRTY_BLEND_CONST = 1 << 3,
    VK_EXEC_DIRTY_STENCIL = 1 << 4,      // compare and write masks, references
    VK_EXEC_DIRTY_ALL = (1 << 5) - 1,
};

// One render target of OP_SET_RT, resolved to its vk_texture.c image for the current command buffer.
typedef struct VkExecTarget {
    const tex_t *tex;                    // texture pin of ex->rt (nvn_pins_sweep keeps it); NULL: no target in this slot
    VkImage image;                       // VK_NULL_HANDLE: no target or not renderable (logged once)
    VkImageView private_attachment_view;  // owned by the renderer feature that substitutes this target
    VkFormat format;                     // view format: the view's when it reinterprets a same-size format
    VkImageViewType view_type;           // 2D, or 2D_ARRAY for more than one layer
    VkImageAspectFlags aspect;
    uint32_t level, base_layer, layers;  // subresource of the view; of a 3D image: depth planes
    uint8_t is3d;                        // a 3D image (colour only), viewed through 2D views of its slices
    uint8_t blocks;                      // a BC image viewed as one texel per block (VkTexImage.block_views)
    uint32_t width, height, samples;     // extent of that level; with blocks, in blocks
    int32_t nvn_format;                  // NVNformat of the view (VkPipeState.color_fmt / depth_fmt)
    const VkTexImage *img;               // the image (fb_usage, fb_learned)
    uint8_t scale;                       // VkTexImage.scale
} VkExecTarget;

// The replay state. One instance lives for the run: NVN state is queue state and carries over from one submission to
// the next, as in MacNativeGames' and LAS Recomp's replays. The "per command buffer" fields are reset by vk_exec_record.
typedef struct VkExec {
    // ---- per command buffer
    VkCommandBuffer cb;                  // being recorded
    uint64_t fence;                      // timeline value the submission signals (vk_completion_add)
    int recorded;                        // Vulkan commands recorded so far; every handler adds the ones it records
    int in_pass;                         // inside vkCmdBeginRendering
    uint32_t dirty;                      // VK_EXEC_DIRTY_*
    // Pass marks (vk.h vk_pass_mark_begin): the open pass's record (NULL: none or marks off), vk_exec_frame.draws when
    // it began, and the passes and dispatches this submission marked so far.
    VkPassMark *mark;
    uint64_t mark_draws0;
    uint32_t mark_passes, mark_dispatches;

    // ---- render targets (OP_SET_RT) and the current pass
    p_set_rt rt;                         // last OP_SET_RT payload (a change ends the open pass); rt.n 0, no depth: none
    unsigned draw_role;                // scoped to a checked postprocess draw
    int interface_coverage;
    uint32_t direct_draw_source;
    GfxOutputEncoding output_encoding;
#ifdef GAME_OOT
    int motion_models, motion_reverse_only, prebloom_tonemap;
    VkGuestPq output_pq;
    struct {
        uint64_t frame;
        uint32_t composite;
        VkImage probe_source;
        const op_hdr *ops[4];
        const GfxShader *shader[4];
        VkPipeline pipeline[4];
        const VkPipeInfo *info[4];
        unsigned phase;
        int active;
    } hdr_aa;
#endif
    unsigned scene_output_role;
    VkImage interface_copy_source;      // confirmed window-encode input, committed only after the draw is recorded
    VkImageView interface_replay_view;
    VkPipeline interface_replay_pipeline;
    const VkPipeInfo *interface_replay_info;
    int interface_draw;                // this draw was recorded inside the guest interface renderer
    int rt_resolved;                   // color[] and depth hold this command buffer's images for rt
    VkExecTarget color[8], depth;        // resolved targets
    uint32_t pass_w, pass_h, pass_layers;            // render area and layer count of the open (or next) pass
    uint32_t pass_scale;                             // its targets' VkTexImage.scale: game viewports and scissors
                                                     // are multiplied by it
    int pass_noatt;                                   // the pass has no attachment: its size follows the draws (vk_exec_noatt_dims)
    VkFormat pass_color[8], pass_depth, pass_stencil; // attachment formats of the pass (VkPipelineRenderingCreateInfo);
                                                      // VK_FORMAT_UNDEFINED: none
    uint32_t pass_ncolor, pass_samples;               // colorAttachmentCount and rasterization samples of the pass

    // ---- fixed-function state (vk_exec.c keeps it; vk_exec_draw.c records it before draws)
    float vp[16][6];                     // OP_VIEWPORTS: x, y, w, h, near, far
    int has_vp;                          // a viewport was set; else the render area
    p_scissor scissor;                   // OP_SCISSOR (scissor 0)
    int has_scissor;                     // a scissor was set; else unbounded
    float depth_range[2];                // OP_DEPTH_RANGE: overrides the viewports' near and far
    uint32_t st_vmask[2], st_wmask[2], st_ref[2];   // OP_STENCIL per face (0 front, 1 back): compare mask, write mask, ref
    float point_size;                    // OP_BIND_STATE kind 6: NVN point size for the support buffer; 0 = never set = 1
    p_poly_offset poly_offset;           // OP_POLY_OFFSET: depth bias factor, units, clamp
    p_prim_restart prim_restart;         // OP_PRIM_RESTART: applied to indexed strip and fan draws (op_draw_idx)
    float blend_color[4];                // OP_BLEND_COLOR
    int64_t copy_stride;                 // OP_COPY_STRIDE: row stride of buffer <-> texture copies (vk_exec_xfer.c)
    const dpool_t *tex_pool, *samp_pool; // OP_SET_TEX_POOL, OP_SET_SAMP_POOL
    // Pipeline inputs. vk_exec.c writes depth_clamp (OP_DEPTH_CLAMP), raster_discard (OP_RASTER_DISCARD) and, at pass
    // begin, color_fmt, depth_fmt and samples; vk_exec_draw.c owns every other field (shaders, prim, y_flip, state
    // objects, vertex input).
    VkPipeState pipe;

    // ---- vk_exec_draw.c slots (draws, binds, dispatch): left empty by vk_exec.c
    const prog_t *prog;                  // OP_BIND_PROGRAM
    int32_t prog_stages;
#ifdef GAME_NS2
    // NVN2 (vk_exec_draw.c): BindProgram binds the stages its mask names, so a compute program bound between draws does
    // not replace the graphics one (MacNativeGames ns2-port.md, Pokemon Pokopia's black picture); dispatches use cprog.
    const prog_t *cprog;
    uint8_t push[2][NVN2_PUSH_SIZE];     // OP_PUSH_CONST: the graphics (0) and compute (1) push-constant blocks
    int64_t copy_image_stride;           // OP_COPY_IMAGE_STRIDE: bytes between the 2D images of a buffer side (0: packed)
#endif
    NVNbufferRange vbuf[16];             // OP_BIND_VBUF
    p_bind_buf ubo[6][18], ssbo[6][16];  // OP_BIND_UBO / OP_BIND_SSBO per stage and index
    uint64_t tex[6][128], samp[6][128], img[6][32];   // OP_BIND_TEX / OP_BIND_SAMP / OP_BIND_IMG handles per stage
    p_scene_projection scene_projection; // scene projection bound with stage 0 slot 0 (OP_BIND_UBO tail)
#ifdef GAME_OOT
    int hdr_meter_written;              // the pass records eye adaptation samples for host readback
    float hdr_meter_projection[2];      // horizontal and vertical perspective scales from the scene camera
    uint64_t hdr_meter_projection_frame;
#endif
    TrackedModelSample tracked_model;    // active tracked model sample for the current draw
    const p_model_scope *work_scope;     // active OP_MODEL_SCOPE record, NULL outside one
    uintptr_t work_scope_end;            // byte address where work_scope expires

    // ---- vk_exec_xfer.c slots (render enable, conditional rendering): left empty by vk_exec.c
    int render_disabled;                 // OP_RENDER_ENABLE false or a failed OP_RENDER_COND: clears and draws skip
    p_render_cond render_cond;           // last OP_RENDER_COND
    int has_render_cond;
    int dbg_zero;                        // DEBUG-OQLOG: the current condition was decided by a segment without queries
#ifdef GAME_NS2
    int gpu_cond;                        // the condition is a predicate word the GPU reads (vk_exec_cond_begin)
    int cond_active, cond_cpu;           // DEBUG: a RENDER_COND is in force; what the CPU side decided
    uint32_t gpu_cond_off, gpu_cond_flags;
    VkBuffer gpu_cond_buf;               // the buffer the predicate word is in (VK_NULL_HANDLE: vk_exec_xfer.c pred_buf)
#endif

    // ---- internal to vk_exec.c
    void *views;                         // image views created for this command buffer (freed at its fence)
    int nviews, cap_views;
    int barrier_at;                      // recorded right after the last full barrier (-1: none yet)
    int pass_end_at;                     // recorded right after the last pass end, when nothing followed it (-1: none)
    int dispatch_end_at;                 // recorded right after the last dispatch (-1: none; vk_exec_dispatch_barrier)
    int relaxed_at;                      // recorded right after a game barrier that orders fragments only (-1: none)

    // Feedback loops (vk.h VkCtx.feedback, vk_exec_draw.c feedback_prepare). Learned targets are attached in
    // ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT; they stay in it across consecutive passes and go back to GENERAL at the next
    // full barrier that is not a pass begin.
    VkImage fbl_img[9];                  // images in the feedback layout now
    VkImageAspectFlags fbl_aspect[9];
    int nfbl;
    uint32_t fbl_mask;                   // attachments of the open pass in that layout: bit i colour i, bit 8 depth
    uint32_t fb_dirty;                   // attachments of the open pass written by a draw or clear since the pass began
                                         // or the last in-pass barrier
    const void *fb_fs;                   // feedback_scan's memo: fragment shader, texture bind serial, pass serial
    uint32_t fb_serial, fb_pass, fb_sampled;
    uint32_t tex_serial, pass_serial;    // bumped by every texture or image bind and pass begin
} VkExec;

// Begins dynamic rendering with the current targets unless a pass is open (a full memory barrier precedes it). Call
// before a draw or an attachment clear.
void vk_exec_resolve_targets(VkExec *ex);
#ifdef GAME_OOT
void vk_present_guest_menu(VkExec *ex, const VkTexImage *source, VkImage destination);
#endif
void vk_exec_pass_begin(VkExec *ex);
// Ends the open pass (no-op without one) and marks its targets GPU-written. Call before copies and dispatches.
void vk_exec_pass_end(VkExec *ex);
// Full memory barrier (all commands, all memory); ends the open pass first. Copies and dispatches order themselves with
// it. Skipped when nothing was recorded since the last one.
void vk_exec_barrier(VkExec *ex);
// Before a dispatch: the full barrier, or none when it follows another dispatch directly (REKARINA_DISPATCH_BARRIER).
void vk_exec_dispatch_barrier(VkExec *ex);
// The command stream between two operations that may write memory or end the pass (vk_exec.c vk_exec_op_quiet): its
// number grows at each such operation; vk_exec_cur_op is the operation being replayed.
extern uint64_t vk_exec_op_window;
extern const op_hdr *vk_exec_cur_op;
int vk_exec_op_quiet(uint32_t op);   // 1: the op neither writes memory nor ends the stretch
// vk_texture.c (REKARINA_TEX_ALIAS): commands that must run at the replay's current position. vk_exec_inline_begin returns
// the command buffer being recorded after a full barrier (an open pass is ended; *reopen says so), or VK_NULL_HANDLE
// when this thread is not recording a replay. vk_exec_inline_end begins the pass again when it was open.
// vk_exec_inline_buffer: a device buffer for transfers, destroyed when the submission completes.
VkCommandBuffer vk_exec_inline_begin(int *reopen);
void vk_exec_inline_end(int reopen);
// vk_exec.c "Draw stretches on worker threads". vk_par_worker: this thread records a chunk of a stretch into a command
// buffer of its own (its pass, begun from the replay thread's, is the only one it may open or end: vk_exec_par_restart
// ends it and begins it again behind a full barrier). vk_par_lock: the rare writes of shared replay bookkeeping a chunk
// makes (a storage range a shader stores into).
extern _Thread_local int vk_par_worker;
extern int vk_par_walk;   // the replay thread walks a stretch: binds are stored, nothing is looked up yet
extern PlatMutex vk_par_lock;
void vk_exec_par_restart(VkExec *ex);
void vk_exec_par_join(void);   // every chunk handed out is recorded (vk_sync_split calls it)
// vk_exec.c: deterministic command-stream draw position across the replay thread and parallel chunk workers.
uint64_t vk_exec_stream_pos(void);
// vk_exec_draw.c: what the draws of a stretch read is made on the replay thread before the chunks are handed out (the
// stand-ins, the limits, the global memory table and write log of the submission).
void vk_exec_draw_par_prepare(VkExec *ex);
// vk_exec_xfer.c: the occlusion counts of a stretch (comment at oq_scan)
void vk_exec_oq_scan(int on);
int vk_exec_oq_scan_open(void);
int vk_exec_oq_scan_draw(void);
void vk_exec_oq_scan_cut(VkExec *ex);
void vk_exec_oq_worker(int on);
void vk_exec_oq_worker_slot(int q);
void vk_exec_oq_worker_close(VkExec *ex);
// vk_exec_draw.c: forget what the command buffer being recorded has bound (pipeline, sets, dynamic state, vertex
// buffers): the replay continues in a new one (vk_sync_split).
void vk_exec_draw_new_cb(void);
// vk_exec_draw.c: the vertex and fragment shader of the last draw recorded into this command buffer (NULL: none yet).
void vk_exec_draw_shaders(const GfxShader **vs, const GfxShader **fs);
VkBuffer vk_exec_inline_buffer(VkDeviceSize size);
// vk_exec.c: the stand-in a 3D BC render target is drawn through goes with its image (vk_exec_image_gone).
void vk_exec_bc_gone(VkImage image);
// vk_hdr.c: with REKARINA_HDR=1, notes a draw into a presented window texture (vk_hdr_track: set by the first present).
extern int vk_hdr_track;
void vk_hdr_draw(VkExec *ex, const VkPipeInfo *info);
int vk_hdr_window_target(VkImage image);   // a presented window texture (vk_exec.c: only its stretches stay serial)
// Scissor 0 clipped to the render area of the open pass; extent 0: nothing to draw.
VkRect2D vk_exec_scissor(const VkExec *ex);
void vk_exec_noatt_dims(const VkExec *ex, uint32_t *w, uint32_t *h);   // pass size when no attachment is set
// 1: every target of the current pass (any 1: at least one, and not only the depth target while a colour target is
// rendered for the first time) was rendered in the previous presented frame (per-frame targets); 0 otherwise, or when
// the pass has none.
int vk_exec_targets_recent(VkExec *ex, int any);
// The current pass's targets with their formats, extents and the frames they were last rendered in (diagnostics).
void vk_exec_targets_describe(VkExec *ex, char *buf, size_t cap);

// REKARINA_TRACE=gfx: replay counters of the current frame, logged and reset at each present (vk_exec_frame_log).
typedef struct {
    uint64_t passes, draws_seen, draws, clears, copies, dispatches, submits;
    uint64_t relaxed_barriers;           // passes that followed a pass with the relaxed barrier (vk_exec.c relaxed_pass_barrier)
    uint64_t skipped_barriers;           // DEBUG-BARRIER-EXPERIMENT
    uint64_t mesh_batches;               // count copies of indirect mesh draws, one per stretch (vk_exec_draw.c)
    uint64_t fb_draws, fb_barriers, fb_restarts, fb_layouts;   // feedback loops: draws sampling a written target of
                                                               // their pass, in-pass barriers, pass restarts, layout
                                                               // transitions
} VkExecFrame;
extern VkExecFrame vk_exec_frame;
void vk_exec_frame_written(const tex_t *t);                    // vk_texture_gpu_written: t rendered or copied into
uint64_t vk_exec_recording_fence(void);                        // fence of the submission this thread records, else 0
void vk_exec_frame_log(const tex_t *presented, uint64_t frame);

// REKARINA_DUMP=<frame> (vk_dump.c): between presents frame-1 and frame, every pass's targets and every dispatch's storage
// images are copied out and written as PNGs to REKARINA_DUMP_DIR (default "dump"), with passes.txt (targets, pipelines,
// textures, samplers) and images.txt (per image value statistics). The hooks run only when vk_dump_armed is set.
extern int vk_dump_armed;
void vk_dump_init(void);                 // reads the environment; vk_exec_record's first call
void vk_dump_pass_end(VkExec *ex);       // after vkCmdEndRendering
void vk_probe_pass_end(VkExec *ex);      // DEBUG-PROBE: after vkCmdEndRendering, REKARINA_PASS_PROBE=1
// The shadow mask's temporal filter (vk_maskfilter.c, REKARINA_MASK_FILTER): each draw notes the scene's camera block;
// after the sun's pass into the shadow mask the filter runs (after vkCmdEndRendering, before the dump's hooks)
void vk_maskfilter_draw(VkExec *ex);
void vk_maskfilter_pass_end(VkExec *ex);
void vk_probe_dispatch(VkExec *ex, const GfxShader *cs, const VkPipeInfo *info);   // DEBUG-PROBE: after a dispatch
VkDescriptorBufferInfo vk_exec_probe_ubo(VkExec *ex, int st, int slot);   // DEBUG-PROBE
VkDescriptorBufferInfo vk_exec_probe_support(void);                     // DEBUG-PROBE
void vk_probe_draw(VkExec *ex, uint32_t count, uint32_t instances);   // DEBUG-PROBE: an indexed draw after its binds
void vk_dump_draw(VkExec *ex, const VkPipeState *s, VkPipeline p, const VkPipeInfo *info);   // after a draw's binds
void vk_dump_dispatch(VkExec *ex, const GfxShader *cs, VkPipeline p, const VkPipeInfo *info, const p_dispatch *d);
void vk_dump_dlaa_copy(VkExec *ex, const VkTexImage *target);   // native pre-bloom DLAA's HUD target after its copy
void vk_dump_copy(VkExec *ex, uint32_t op, const void *payload);   // before a copy or texture clear is replayed
void vk_dump_present(uint64_t frame);    // vk_be_present: waits for the dump frame's copies and closes the logs

// c0, the NVN driver constant bank of one stage (vk_exec_draw.c; REKARINA_FIX_C0=0 binds zeros instead): bank is
// VK_C0_BYTES, stage an NVN stage (0 vertex, 1 fragment, 5 compute), ssbo and ubo the 16 bound slots of that stage.
#ifdef GAME_NS2
#define VK_C0_BYTES 0xA00u               // NVN2's mesh stage table runs to 0x870 (seedds/nxnative DMSH_DRV_SIZE)
#define VK_C0_MESH_FIRST 0x9F0u          // task / mesh: the draw's first workgroup (u32), which the translated shader adds
                                         // to its workgroup number (NVN2 DrawMeshTasks numbers them from `first`)
#else
#define VK_C0_BYTES 0x800u
#endif
void vk_c0_fill(uint8_t bank[VK_C0_BYTES], int stage, const p_bind_buf ssbo[16], const p_bind_buf ubo[16],
                const uint32_t dispatch[3], uint32_t base_instance);

// Handler hooks. vk_exec.c defines both WEAK: they stop with rt_fatal naming the op.
// vk_exec_draw.c: OP_DRAW, OP_DRAW_IDX, OP_DISPATCH, OP_BIND_* (program, state, vattr, vstream, vbuf, ubo, ssbo, tex,
// samp, img), OP_UPDATE_UB, OP_DRAW_IDX_IND, OP_DISPATCH_IND. payload: nvn_op_payload of the record.
void vk_exec_op_draw(VkExec *ex, uint32_t op, const void *payload);
// The aspect a sampled view of img reads (vk_exec_draw.c get_view): stencil for a view in STENCIL depth-stencil mode.
VkImageAspectFlags vk_exec_view_aspect(const VkTexImage *img, const view_t *v, int hv);
VkImageView vk_exec_target_view(const VkTexImage *img, VkFormat format, VkImageViewType type, uint32_t aspect,
                                uint32_t level, uint32_t base, uint32_t layers, int blocks);   // kept for the image's life
// vk_exec_xfer.c: OP_COPY_B2B, OP_COPY_B2T, OP_COPY_T2B, OP_COPY_T2T, OP_CLEAR_TEX, OP_SIGNAL_EVENT, OP_REPORT_COUNTER,
// OP_RESET_COUNTER, OP_RENDER_ENABLE, OP_RENDER_COND, OP_CLEAR_BUF.
void vk_exec_op_xfer(VkExec *ex, uint32_t op, const void *payload);
// OP_WAIT_EVENT: waits for the CPU's signal of the event when no GPU signal will set it; 1 when it waited.
// 0 no wait, 1 waited, 2 deferred, 3 hold here, 4 met by a compute rest's signal the graphics work does not wait for yet
// (vk_exec_wait_cp: the cp value)
int vk_exec_wait_event(VkExec *ex, const p_wait_event *p);
void vk_exec_results_point(VkExec *ex);   // vk_exec.c: the submission being recorded has drawn its scene (vk_sync.c results)
void vk_exec_oq_submit_end(uint64_t fence);   // vk_exec_xfer.c: the occlusion slots a submission's segments read are reset behind them
extern uint64_t vk_exec_wait_cp;
extern uint64_t vk_exec_copies_ns;   // vk_exec_xfer.c: when the copies the commands read now were begun
void vk_dump_arm_clears(const uint64_t *handles, int n);   // vk_dump.c DEBUG-DUMPCLEARS
int vk_dump_active_dbg(void);   // vk_dump.c: the frame being recorded is dumped or logged
int vk_exec_prewait(const uint64_t *handles, int n);   // vk_exec_xfer.c: the submission's waits for the CPU, before its copies
// vk_exec.c "Queues interleaved": 1 while the wait being replayed may hold its submission; the oldest held submission's
// number (0: none). vk_exec_xfer.c: 1 when a held wait (an OP_WAIT_EVENT payload) can go on.
int vk_exec_hold_ok(void);
uint64_t vk_exec_held_submit(void);
int vk_exec_wait_ready(const void *payload);
// The cp value of the compute rest being recorded (async compute), else 0; vk_exec_xfer.c notes it with each signal.
extern uint64_t vk_exec_ac_value;
void vk_exec_wait_resolve(void);
int vk_exec_wait_deferred(void);   // a deferred wait whose word is not met yet   // a deferred wait, before the part after it is submitted (vk_exec_xfer.c)
// vk_dlaa.c: once DLSS has scaled the scene this frame, passes larger than the scene's light target vk_dlaa_swap_img
// sample DLSS's output vk_dlaa_swap_view in its place, so the game's scaling draw scales DLSS's picture.
extern const VkTexImage *vk_dlaa_swap_img;
extern VkImageView vk_dlaa_swap_view;
// vk_contact.c: a pass about to begin (its targets resolved, no pass open), and a dispatch or copy about to be recorded
void vk_contact_pass_begin(VkExec *ex);
void vk_contact_flush(VkExec *ex);
// vk_exec_xfer.c, samples-passed counting: a draw about to be recorded in the open pass (begins an occlusion query
// when counting and none is active), and the open pass about to end (ends the active query).
void vk_exec_oq_draw(VkExec *ex);
// A draw the game made while samples were counted that the replay skips (shader or pipeline not ready, or failed): the
// count of its segment is unknown, so the draws conditioned on it are rendered rather than skipped as unseen.
void vk_exec_oq_draw_lost(VkExec *ex);
void vk_exec_oq_pass_end(VkExec *ex);
#ifdef GAME_NS2
void vk_exec_oq_after_pass(VkExec *ex);   // vk_exec_pass_end, after the pass: the predicate copies of its reports
void vk_exec_cond_begin(VkExec *ex);      // around each draw, dispatch and attachment clear under a GPU condition
void vk_exec_cond_end(VkExec *ex);
#endif
