// The op-stream replay core of the Vulkan back end: walks the command handles of
// one submission into its command buffer, keeps render targets and fixed-function state, opens and closes dynamic
// rendering passes and records clears and barriers. Draws, binds and dispatches go to vk_exec_op_draw
// (vk_exec_draw.c); copies, counters, events and conditional rendering to vk_exec_op_xfer (vk_exec_xfer.c). Images
// stay in VK_IMAGE_LAYOUT_GENERAL (vk_texture.c); full memory barriers order passes, clears and whatever those two
// record between them.
//
// The op loop (END / BEGIN / JUMP / CALL, which ops are pure state, the state defaults, render targets kept until they
// change, clears skipped while rendering is disabled) follows MacNativeGames
// games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m:10588-10700, 7409-7460, 5449-5457
// (https://github.com/ly05010419/MacNativeGames @ b75be8cfa704df3701e86130c8ced268e79d6b11, MIT, Copyright (c) 2026
// jhaizhou). Pass begin and end (GENERAL-layout attachments, LOAD/STORE, render area from the smallest attachment, a
// full barrier before each pass), the scissor clip and attachment clears follow LAS Recomp
// runtime/nvn/vulkan/exec.c:855-1149, 3553-3778 (https://github.com/peiikai/LASRecomp @
// da35b83a9611d046a9a681aae668d7085954d37f, MIT, Copyright (c) 2026 Peiikai).
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include "cfg.h"
#include "ftrace.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_tex.h"
#include "platform/platform.h"
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"
#include "vk_dlaa.h"

#define WEAK __attribute__((weak))
#define CALL_DEPTH 8

typedef struct {
    VkImage image;
    VkFormat format;
    VkImageViewType type;
    VkImageAspectFlags aspect;
    uint32_t level, base, layers;
    VkImageView view;
} ViewEnt;

static VkExec X;
static int x_ready;
static _Thread_local int x_here;   // this thread is inside vk_exec_record (vk_exec_inline_begin)

// ---- REKARINA_TRACE=gfx frame counters

VkExecFrame vk_exec_frame;
static struct { const uint8_t *mem; uint32_t w, h, fmt, n; } fw[64];   // textures written this frame
static int nfw;

static _Thread_local uint64_t par_fence;   // this thread records a chunk of the submission with this value (par_run)
uint64_t vk_exec_recording_fence(void) { return x_here ? X.fence : par_fence; }
static uint64_t x_stream_pos;
static _Thread_local uint64_t tl_stream_pos;
uint64_t vk_exec_stream_pos(void) { return tl_stream_pos; }

// ---- Draw stretches on worker threads: the types (the comment and the code are at par_try)
#define PAR_CHUNKS 12                    // chunks of one stretch, and chunk threads, at most
#define PAR_JOBS 40                      // chunks out at once at most (vk_sync.c CHAIN_MAX holds them and the replay's own)
#define PAR_MIN 12                       // draws: a shorter stretch is replayed as always
typedef struct {
    uint8_t disabled, has_cond, dbg_zero;
#ifdef GAME_NS2
    uint8_t gpu_cond, cond_active, cond_cpu;
    uint32_t off, flags;
    VkBuffer buf;
#endif
    p_render_cond rc;
} ParCond;
typedef struct {
    VkExec st;                           // the queue state at the chunk's first op
    const op_hdr *from;
    const op_hdr *volatile to;           // NULL until the walk reaches the chunk's end
    VkCommandBuffer cb;
    const int32_t *slot;                 // per draw op: the occlusion slot whose query it begins, -1: none
    const ParCond *cond;                 // per RENDER_COND and RENDER_ENABLE op: what the walk decided
    VkRenderingInfo ri;                  // the stretch's pass, begun again in the chunk's command buffer
    VkRenderingAttachmentInfo ca[8], da, sa;
    uint32_t crumb;                      // DEBUG-PARCRUMB: stretch serial * 16 + chunk; 0: no crumbs
    uint64_t asof;                       // vk_texture_asof_mark when the stretch was handed out: its lookups' limit
    uint32_t updn, useq;                 // inline uniform updates noted and snapshot drops by them at the handout
    uint64_t pos0;                       // deterministic stream draw counter at chunk start (vk_exec_stream_pos)
    const CfgRenderState *settings;
} ParJob;
static struct {
    int threads;                         // -1: not read; 0: off
    int chunks;                          // chunks per stretch at most
    int min_draws, spin;                 // REKARINA_PAR_MIN, REKARINA_PAR_SPIN (pauses before a chunk thread sleeps)
    int per;                             // REKARINA_PAR_PER: draws per chunk
    int prio;                            // REKARINA_PAR_PRIO: 0 normal, 1 above normal, 2 highest (par_on)
    ParJob job[PAR_JOBS];
    volatile int32_t jstate[PAR_JOBS];   // 0 free or being made, 1 to be recorded, 2 taken, 3 recorded
    volatile int32_t njobs;              // chunks handed out since the last join
    volatile int32_t ndone;              // and how many of them are recorded
    volatile int32_t active;             // chunks may be out: the chunk threads stay awake
    volatile int32_t asleep[PAR_CHUNKS]; // per chunk thread: it waits on its event
    PlatEvent ev[PAR_CHUNKS];            // one per chunk thread: woken side by side, not one after another through one lock
    // The walk has done every op before this one: the chunk it is in replays no further. Written at each draw, read by a
    // chunk only when it has caught up: on a line of its own, since the walk writes it thousands of times a stretch.
    _Alignas(64) const op_hdr *volatile walked;
    _Alignas(64) int walking;            // the replay thread walks a stretch: its pass is open in the bookkeeping only
    VkRenderingInfo ri;                  // the pass begun last (vk_exec_pass_begin)
    VkRenderingAttachmentInfo ca[8], da, sa;
    int32_t *slots;                      // of the chunks out: appended by each walk, emptied by the join
    ParCond *conds;
    uint32_t cap_slots, cap_conds, nslots, nconds;
    const op_hdr *no_from, *no_until;    // ops looked at already: no stretch begins among them
    uint64_t stretches, chunks_n, draws, walk_ns, join_ns, joins;   // REKARINA_TRACE=perf
} PAR = {.threads = -1};
_Thread_local int vk_par_worker;
int vk_par_walk;
PlatMutex vk_par_lock = PLAT_MUTEX_INIT;
static _Thread_local VkExec PX;          // the queue state of the chunk this thread records
static _Thread_local const VkRenderingInfo *par_ri;   // and its pass (vk_exec_par_restart)


void vk_exec_frame_written(const tex_t *t) {
    if (!(rt_trace & RT_TRACE_GFX) || !t) return;
    for (int i = 0; i < nfw; i++)
        if (fw[i].mem == t->mem) { fw[i].n++; return; }
    if (nfw == 64) return;
    fw[nfw].mem = t->mem, fw[nfw].w = t->d.width, fw[nfw].h = t->d.height, fw[nfw].fmt = (uint32_t)t->d.format;
    fw[nfw++].n = 1;
}

// DEBUG-PASSLIST: REKARINA_PASSLIST=<clears>: the render passes of a frame with that many clears or more, and of the frame
// before it, one line each (area, layers, draws, the first colour target and the depth target with their storage
// address, format and layer), at most four such frames. For the frames in which OoT redraws its shadow
// layers at once (45 clears against 23).
typedef struct { uint16_t w, h, draws; uint8_t layers, ncolor; int32_t cfmt, dfmt; uint64_t caddr, daddr; uint16_t clayer, dlayer; } PlRec;
#define PL_N 640
static PlRec pl_cur[PL_N], pl_prev[PL_N];
static int pl_n, pl_prevn, pl_min = -1, pl_logged;
static uint64_t pl_draws, pl_clears0;
static void pl_pass(const VkExec *ex) {
    if (pl_min < 0) { const char *e = cfg_get("REKARINA_PASSLIST"); pl_min = e && atoi(e) > 0 ? atoi(e) : 0; }
    if (!pl_min || pl_n == PL_N) return;
    uint64_t d = vk_exec_frame.draws;
    PlRec *r = &pl_cur[pl_n++];
    *r = (PlRec){(uint16_t)ex->pass_w, (uint16_t)ex->pass_h, (uint16_t)(d >= pl_draws ? d - pl_draws : d), (uint8_t)ex->pass_layers,
                 (uint8_t)ex->pass_ncolor, ex->color[0].image ? ex->color[0].nvn_format : -1, ex->depth.image ? ex->depth.nvn_format : -1,
                 ex->color[0].tex ? (uint64_t)(uintptr_t)ex->color[0].tex->mem : 0, ex->depth.tex ? (uint64_t)(uintptr_t)ex->depth.tex->mem : 0,
                 (uint16_t)ex->color[0].base_layer, (uint16_t)ex->depth.base_layer};
    pl_draws = d;
}
static void pl_frame(uint64_t frame) {
    if (vk_dump_armed) {   // DEBUG-DUMPAFTER
        void vk_dump_arm_after(uint64_t frame, uint64_t clears, uint64_t before);
        static uint64_t c0, last;
        uint64_t c = vk_exec_frame.clears, n = c >= c0 ? c - c0 : c;
        c0 = c;
        vk_dump_arm_after(frame, n, last);
        last = n;
    }
    if (pl_min <= 0) return;
    uint64_t c = vk_exec_frame.clears, n = c >= pl_clears0 ? c - pl_clears0 : c;
    pl_clears0 = c;
    if ((int)n >= pl_min && pl_logged < 4 && frame > 3000) {
        pl_logged++;
        for (int k = 0; k < 2; k++) {
            const PlRec *v = k ? pl_cur : pl_prev;
            int m = k ? pl_n : pl_prevn;
            rt_log("PASSLIST frame %llu (%s, %d passes, %llu clears in the later one)", (unsigned long long)(frame - 1 + k),
                   k ? "the frame with the clears" : "the frame before", m, (unsigned long long)n);
            for (int i = 0; i < m; i++)
                rt_log("PASSLIST %d %3d: %ux%u layers %u draws %u colours %u c0 %#llx fmt %#x layer %u depth %#llx fmt %#x layer %u",
                       k, i, v[i].w, v[i].h, v[i].layers, v[i].draws, v[i].ncolor, (unsigned long long)v[i].caddr,
                       (unsigned)v[i].cfmt, v[i].clayer, (unsigned long long)v[i].daddr, (unsigned)v[i].dfmt, v[i].dlayer);
        }
    }
    memcpy(pl_prev, pl_cur, sizeof(PlRec) * (size_t)pl_n);
    pl_prevn = pl_n, pl_n = 0;
}

void vk_exec_frame_log(const tex_t *p, uint64_t frame) {
    pl_frame(frame);   // DEBUG-PASSLIST
    if (!(rt_trace & RT_TRACE_GFX)) return;
    VkExecFrame *f = &vk_exec_frame;
    int hit = 0;
    char list[512] = "";
    size_t o = 0;
    for (int i = 0; i < nfw; i++) {
        if (p && fw[i].mem == p->mem) hit = 1;
        if (o < sizeof list - 48)
            o += (size_t)snprintf(list + o, sizeof list - o, " %p:%ux%u/%#x*%u", (const void *)fw[i].mem, fw[i].w,
                                  fw[i].h, fw[i].fmt, fw[i].n);
    }
    rt_log("frame %llu: submits %llu passes %llu draws %llu/%llu clears %llu copies %llu dispatches %llu; present %p "
           "%ux%u/%#x written %d; relaxed pass barriers %llu; feedback draws %llu barriers %llu restarts %llu layouts %llu; targets%s",
           (unsigned long long)frame, (unsigned long long)f->submits,
           (unsigned long long)f->passes, (unsigned long long)f->draws, (unsigned long long)f->draws_seen,
           (unsigned long long)f->clears, (unsigned long long)f->copies, (unsigned long long)f->dispatches,
           p ? (const void *)p->mem : NULL, p ? p->d.width : 0, p ? p->d.height : 0, p ? (uint32_t)p->d.format : 0,
           hit, (unsigned long long)f->relaxed_barriers, (unsigned long long)f->fb_draws, (unsigned long long)f->fb_barriers, (unsigned long long)f->fb_restarts,
           (unsigned long long)f->fb_layouts, list);
    memset(f, 0, sizeof *f);
    nfw = 0;
}

WEAK void vk_exec_op_draw(VkExec *ex, uint32_t op, const void *payload) {
    (void)ex, (void)payload;
    rt_fatal("vk: op %s is not replayed yet", nvn_op_name(op));
}

WEAK void vk_exec_op_xfer(VkExec *ex, uint32_t op, const void *payload) {
    (void)ex, (void)payload;
    rt_fatal("vk: op %s is not replayed yet", nvn_op_name(op));
}

enum { W_UNRENDERABLE, W_LEVEL, W_LAYER, W_VIEW_FORMAT, W_BLOCK_VIEW, W_BLOCK_LAYERS, W_CLEAR_TARGET, W_CLEAR_MASK, W_STENCIL_MASK, W_COUNT };

static void warn_once(int key, const char *fmt, ...) {
    static volatile int32_t seen[W_COUNT];
    if (__atomic_exchange_n(&seen[key], 1, __ATOMIC_RELAXED)) return;
    char msg[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof msg, fmt, ap);
    va_end(ap);
    rt_log("vk: %s (logged once)", msg);
}

// NVN defaults of the replay state (nvn_mtl.m).
static void state_reset(VkExec *ex) {
    memset(ex, 0, sizeof *ex);
    ex->st_vmask[0] = ex->st_vmask[1] = ex->st_wmask[0] = ex->st_wmask[1] = 0xff;
    ex->depth_range[1] = 1.0f;
    vk_pipe_state_defaults(&ex->pipe);
    ex->barrier_at = -1;
    ex->pass_end_at = -1;
    ex->dispatch_end_at = -1;
    ex->relaxed_at = -1;
}

// ---- render targets

static uint32_t level_extent(uint32_t v, uint32_t level) {
    v >>= level;
    return v ? v : 1;
}

// A view format other than the texture's is used when both map directly and have the same texel size (the image is
// created MUTABLE_FORMAT for colour formats). blocks: t is a BC texture with block_views, and the view an uncompressed
// format whose texel is one of its blocks.
static int view_format(const tex_t *t, const view_t *v, int blocks, VkFormat *out) {
    TexVkFormat a, b;
    int bw, bh, vbw, vbh;
    if (!tex_vk_format(t->d.format, &a) || a.vk != a.ideal || !tex_vk_format(v->format, &b) || b.vk != b.ideal ||
        b.depth || b.stencil)
        return 0;
    int bpb = nvn_fmt_bpb(t->d.format, &bw, &bh), vbpb = nvn_fmt_bpb(v->format, &vbw, &vbh);
    if (!bpb || bpb != vbpb || (!blocks && (bw != 1 || bh != 1)) || vbw != 1 || vbh != 1) return 0;
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, b.vk, &fp);
    if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BIT)) return 0;
    *out = b.vk;
    return 1;
}

// ---- 3D BC render targets
// A 3D BC texture has no block views (vk_texture.c create_tex, REKARINA_BLOCK_VIEWS_3D): on RDNA2 cards such an image
// made BLOCK_TEXEL_VIEW_COMPATIBLE and 2D_ARRAY_COMPATIBLE took GPU writes past the end of its memory (issue #5). Draws
// into it through an uncompressed view of its blocks (agl's BC4 encoder fills 32x32x32 and 64x64x64 volumes and their
// levels at the logo) go to a stand-in: a 2D array of the blocks of one level in the view's format, one layer per
// depth plane. It starts with the level's content, so a pass that writes part of a plane keeps the rest, and each pass
// end copies the planes it drew into the level (a copy between a compressed and a size-compatible uncompressed format).
typedef struct { VkImage bc, img; VmaAllocation alloc; VkFormat fmt; uint32_t level, w, h, d; } BcStandIn;
#define BC_STANDINS 64   // one per level drawn: 6 volumes of up to 7 levels at the logo
static BcStandIn bcs[BC_STANDINS];
static int nbcs;
static PlatMutex bcs_lock = PLAT_MUTEX_INIT;   // the replay adds and reads; destroy_tex (any thread) removes

static void image_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, uint32_t layers) {
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_NONE,
                                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                .oldLayout = from, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cb, &di);
}

static void memory_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 src, VkAccessFlags2 sa, VkPipelineStageFlags2 dst,
                           VkAccessFlags2 da) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = src, .srcAccessMask = sa,
                           .dstStageMask = dst, .dstAccessMask = da};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

// The stand-in of level level of a 3D BC image for draws through view format fmt, made on first use; 0 when none can
// be made (logged). Replay thread.
static int bc_standin(const VkTexImage *img, uint32_t level, VkFormat fmt, BcStandIn *out) {
    static int warned;
    plat_mutex_lock(&bcs_lock);
    int full = nbcs == BC_STANDINS;
    for (int i = 0; i < nbcs; i++)
        if (bcs[i].bc == img->image && bcs[i].level == level && bcs[i].fmt == fmt) {
            *out = bcs[i];
            plat_mutex_unlock(&bcs_lock);
            return 1;
        }
    plat_mutex_unlock(&bcs_lock);
    if (full) {
        if (!warned++) rt_log("vk: more than %d 3D BC render targets: the others are left out", BC_STANDINS);
        return 0;
    }
    uint32_t lw = level_extent(img->width, level), lh = level_extent(img->height, level);
    BcStandIn s = {img->image, VK_NULL_HANDLE, NULL, fmt, level, (lw + 3) / 4, (lh + 3) / 4,
                   level_extent(img->depth, level)};
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
                            .extent = {s.w, s.h, 1}, .mipLevels = 1, .arrayLayers = s.d, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                     VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(vk_ctx()->vma, &ci, &ai, &s.img, &s.alloc, NULL) != VK_SUCCESS) {
        if (!warned++) rt_log("vk: stand-in for a 3D BC render target could not be created: left out");
        return 0;
    }
    vk_addr_label((uint64_t)(uintptr_t)s.img, "BC stand-in of %#llx level %u", (unsigned long long)(uintptr_t)img->image, level);
    // In the upload batch, which goes to the queue before this part: GENERAL, then the texture's blocks.
    VkCommandBuffer cb = vk_upload_begin();
    image_barrier(cb, s.img, VK_IMAGE_LAYOUT_UNDEFINED, s.d);
    memory_barrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                   VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageCopy rg = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, level, 0, 1},
                      .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, s.d},
                      .extent = {lw, lh, s.d}};   // texels of the compressed source
    vkCmdCopyImage(cb, img->image, VK_IMAGE_LAYOUT_GENERAL, s.img, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
    memory_barrier(cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                   VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    vk_upload_end();
    plat_mutex_lock(&bcs_lock);
    bcs[nbcs++] = s;
    plat_mutex_unlock(&bcs_lock);
    rt_log("vk: 3D BC render target %ux%ux%u level %u drawn through a %ux%u x %u stand-in (format %d)", img->width,
           img->height, img->depth, level, s.w, s.h, s.d, (int)fmt);
    *out = s;
    return 1;
}

// vk_exec_draw.c vk_exec_image_gone: the stand-in of a BC image that goes, freed once the GPU is done with it.
void vk_exec_bc_gone(VkImage image) {
    plat_mutex_lock(&bcs_lock);
    for (int i = 0; i < nbcs; i++)
        if (bcs[i].bc == image) {
            vk_defer_image(bcs[i].img, bcs[i].alloc);
            bcs[i--] = bcs[--nbcs];
        }
    plat_mutex_unlock(&bcs_lock);
}

// At a pass end: the planes the pass drew into a stand-in go to its texture.
static void bc_copy_back(VkExec *ex, const VkExecTarget *t) {
    plat_mutex_lock(&bcs_lock);
    for (int i = 0; i < nbcs; i++) {
        if (bcs[i].img != t->image) continue;
        memory_barrier(ex->cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_COPY_BIT,
                       VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkImageCopy rg = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, t->base_layer, t->layers},
                          .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, bcs[i].level, 0, 1},
                          .dstOffset = {0, 0, (int32_t)t->base_layer},
                          .extent = {bcs[i].w, bcs[i].h, t->layers}};   // texels of the uncompressed source
        vkCmdCopyImage(ex->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, bcs[i].bc, VK_IMAGE_LAYOUT_GENERAL, 1, &rg);
        memory_barrier(ex->cb, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        ex->recorded++;
        break;
    }
    plat_mutex_unlock(&bcs_lock);
}

// A draw into a 3D BC image without block views through an uncompressed view of its blocks: the target is the
// image's stand-in (bc_standin). 1 when o is filled (target or left out), 0 for any other target.
static int bc_target(const tex_t *t, const view_t *v, int has_v, const VkTexImage *img, VkExecTarget *o) {
    int bw, bh;
    if (img->type != VK_IMAGE_TYPE_3D || img->block_views || !nvn_fmt_bpb(t->d.format, &bw, &bh) || bw != 4 || bh != 4)
        return 0;
    VkFormat fmt;
    BcStandIn s;
    uint32_t level = has_v && v->set_levels ? v->base_level : 0;
    if (level >= img->levels) {
        warn_once(W_LEVEL, "render target level %u of a %u-level texture: left out", level, img->levels);
        return 1;
    }
    if (!has_v || !v->set_format || !view_format(t, v, 1, &fmt) || !bc_standin(img, level, fmt, &s)) {
        warn_once(W_BLOCK_VIEW, "render target format %#x (%ux%u) through view format %#x: no block view, left out",
                  t->d.format, t->d.width, t->d.height, has_v && v->set_format ? v->format : 0);
        return 1;
    }
    uint32_t base = 0, layers = s.d;
    if (v->set_layers) {
        base = v->min_layer;
        layers = v->nlayers ? v->nlayers : (base < s.d ? s.d - base : 0);
    }
    if (base >= s.d || !layers) return 1;
    if (base + layers > s.d) layers = s.d - base;
    o->image = s.img;   // not img->image: view_of makes this command buffer's view of the stand-in
    o->img = img;
    o->format = fmt;
    o->view_type = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    o->aspect = VK_IMAGE_ASPECT_COLOR_BIT;
    o->level = 0, o->base_layer = base, o->layers = layers;
    o->width = s.w, o->height = s.h, o->samples = 1;
    o->nvn_format = v->format;
    return 1;
}

static void resolve_target(const tex_t *t, const view_t *v, int has_v, int depth_slot, VkExecTarget *o) {
    memset(o, 0, sizeof *o);
    o->tex = t;
    if (!t) return;
    const VkTexImage *img = vk_texture_get(t);
    if (img && !depth_slot && bc_target(t, v, has_v, img, o)) return;   // a 3D BC image: its stand-in
    VkImageUsageFlags need = depth_slot ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    int is3d = img && img->type == VK_IMAGE_TYPE_3D && !depth_slot;   // slices drawn through 2D views (2D_ARRAY_COMPATIBLE)
    if (!img || !(img->usage & need) || (img->type != VK_IMAGE_TYPE_2D && !is3d)) {
        warn_once(W_UNRENDERABLE, "render target format %#x target %d (%ux%u) is not renderable here: left out",
                  t->d.format, t->d.target, t->d.width, t->d.height);
        return;
    }
    uint32_t level = has_v && v->set_levels ? v->base_level : 0;
    if (level >= img->levels) {
        warn_once(W_LEVEL, "render target level %u of a %u-level texture: left out", level, img->levels);
        return;
    }
    // A 3D level's depth planes take the place of layers (Vulkan views them as the array layers of a 2D array view).
    uint32_t nlay = is3d ? level_extent(img->depth, level) : img->layers;
    uint32_t base = 0, layers = nlay;
    if (has_v && v->set_layers) {
        base = v->min_layer;
        layers = v->nlayers ? v->nlayers : (base < nlay ? nlay - base : 0);
    }
    if (base >= nlay || !layers) {
        warn_once(W_LAYER, "render target layer %u of a %u-layer texture: left out", base, nlay);
        return;
    }
    if (base + layers > nlay) layers = nlay - base;
    VkFormat fmt = img->format;
    int32_t nvn_fmt = t->d.format;
    int blocks = !depth_slot && img->block_views;
    if (blocks) {   // a BC texture takes draws only through an uncompressed view of its blocks
        if (!has_v || !v->set_format || !view_format(t, v, 1, &fmt)) {
            warn_once(W_BLOCK_VIEW, "render target format %#x (%ux%u) through view format %#x: no block view, left out",
                      t->d.format, t->d.width, t->d.height, has_v && v->set_format ? v->format : 0);
            return;
        }
        nvn_fmt = v->format;
        if (layers > 1) {   // an uncompressed view of a BC image has one layer (no maintenance6 here)
            warn_once(W_BLOCK_LAYERS, "render target format %#x: block view of %u layers, first layer only", t->d.format,
                      layers);
            layers = 1;
        }
    } else if (!depth_slot && has_v && v->set_format && v->format != t->d.format) {
        if (view_format(t, v, 0, &fmt)) nvn_fmt = v->format;
        else warn_once(W_VIEW_FORMAT, "render target view format %#x of a format %#x texture: texture format used",
                       v->format, t->d.format);
    }
    o->image = img->image;
    o->img = img;
    o->scale = img->scale ? img->scale : 1;
    o->format = fmt;
    o->view_type = layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    o->aspect = depth_slot ? img->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)
                           : VK_IMAGE_ASPECT_COLOR_BIT;
    o->level = level, o->base_layer = base, o->layers = layers;
    o->is3d = (uint8_t)is3d;
    o->blocks = (uint8_t)blocks;
    o->width = level_extent(img->width, level), o->height = level_extent(img->height, level);
    if (blocks) o->width = (o->width + 3) / 4, o->height = (o->height + 3) / 4;   // the view's extent: 4x4 blocks
    o->samples = img->samples ? img->samples : 1;
    o->nvn_format = nvn_fmt;
    if (!o->aspect) o->image = VK_NULL_HANDLE;
}

/* The size of a pass without attachments: the far corner of viewport 0, else of the scissor, 1..16384, as
   MacNativeGames and nxnative size it. TotK draws a fragment shader that only writes a storage buffer into a 1x1
   viewport; a pass sized by the viewport's width alone puts an offset viewport outside the scissor, and the cave rock
   loses its ambient light. */
void vk_exec_noatt_dims(const VkExec *ex, uint32_t *w, uint32_t *h) {
    double x1 = 0, y1 = 0;
    if (ex->has_vp) {
        x1 = ceil(fmax(ex->vp[0][0], ex->vp[0][0] + ex->vp[0][2]));
        y1 = ceil(fmax(ex->vp[0][1], ex->vp[0][1] + ex->vp[0][3]));
    }
    if ((x1 < 1 || y1 < 1) && ex->has_scissor) x1 = (double)ex->scissor.x + ex->scissor.w, y1 = (double)ex->scissor.y + ex->scissor.h;
    *w = x1 < 1 ? 1 : x1 > 16384 ? 16384 : (uint32_t)x1;
    *h = y1 < 1 ? 1 : y1 > 16384 ? 16384 : (uint32_t)y1;
}

// Resolves ex->rt to this command buffer's images and derives the pass parameters and the pipeline's target formats.
static void resolve_rt(VkExec *ex) {
    if (ex->rt_resolved) return;
    ex->rt_resolved = 1;
    const p_set_rt *r = &ex->rt;
    int n = r->n < 0 ? 0 : r->n > 8 ? 8 : r->n;
    uint32_t w = UINT32_MAX, h = UINT32_MAX, layers = UINT32_MAX, samples = 0, scale = 1;
    for (int i = 0; i < 8; i++) {
        VkExecTarget *t = &ex->color[i];
        resolve_target(i < n ? r->color[i] : NULL, &r->cview[i], i < n && r->has_cview[i], 0, t);
        if (i < n && r->color[i]) { void vk_dbg_r8(const char *, const tex_t *, long, long); vk_dbg_r8("render-target", r->color[i], i, n); }   // DEBUG-R8LOG
        ex->pass_color[i] = t->image ? t->format : VK_FORMAT_UNDEFINED;
        ex->pipe.color_fmt[i] = t->image ? t->nvn_format : 0;
        if (!t->image) continue;
        if (t->width < w) w = t->width;
        if (t->height < h) h = t->height;
        if (t->layers < layers) layers = t->layers;
        if (!samples) samples = t->samples;
        if (t->scale > scale) scale = t->scale;
    }
    VkExecTarget *d = &ex->depth;
    resolve_target(r->depth, &r->dview, r->has_dview, 1, d);
    ex->pass_depth = d->image && (d->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) ? d->format : VK_FORMAT_UNDEFINED;
    ex->pass_stencil = d->image && (d->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) ? d->format : VK_FORMAT_UNDEFINED;
    ex->pipe.depth_fmt = d->image ? d->nvn_format : 0;
    if (d->image) {
        if (d->width < w) w = d->width;
        if (d->height < h) h = d->height;
        if (d->layers < layers) layers = d->layers;
        if (!samples) samples = d->samples;
        if (d->scale > scale) scale = d->scale;
    }
    ex->pass_noatt = w == UINT32_MAX;
    if (ex->pass_noatt) {
        vk_exec_noatt_dims(ex, &w, &h);
        layers = 1;
        static int logged;
        if (logged < 8 && ex->has_vp && (ex->vp[0][0] != 0 || ex->vp[0][1] != 0 || ex->vp[0][2] != w || ex->vp[0][3] != h))
            logged++, rt_log("vk_exec: pass without attachments %ux%u for viewport %g,%g %gx%g", w, h, ex->vp[0][0],
                             ex->vp[0][1], ex->vp[0][2], ex->vp[0][3]);
    }
    ex->pass_w = w, ex->pass_h = h, ex->pass_layers = layers;
    ex->pass_scale = ex->pass_noatt ? 1 : scale;
    ex->pass_ncolor = (uint32_t)n;
    ex->pass_samples = samples ? samples : 1;
    ex->pipe.samples = (uint8_t)ex->pass_samples;
}

void vk_exec_resolve_targets(VkExec *ex) { resolve_rt(ex); }

// The attachment view of a target. A registered texture's view lives as long as its image (vk_exec_target_view); any
// other image's view is made for this command buffer and freed at its fence.
static VkImageView view_of(VkExec *ex, const VkExecTarget *t) {
    if (t->private_attachment_view) return t->private_attachment_view;
    if (t->img && t->img->image == t->image && !t->img->gone)
        return vk_exec_target_view(t->img, t->format, t->view_type, t->aspect, t->level, t->base_layer, t->layers, t->blocks);
    ViewEnt *v = ex->views;
    for (int i = 0; i < ex->nviews; i++)
        if (v[i].image == t->image && v[i].format == t->format && v[i].type == t->view_type && v[i].aspect == t->aspect &&
            v[i].level == t->level && v[i].base == t->base_layer && v[i].layers == t->layers)
            return v[i].view;
    VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->image,
                                .viewType = t->view_type, .format = t->format,
                                .subresourceRange = {t->aspect, t->level, 1, t->base_layer, t->layers}};
    VkImageViewUsageCreateInfo use = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                      .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
    if (t->blocks) ci.pNext = &use;   // not the BC image's sampling and transfer usages
    VkImageView view;
    if (vkCreateImageView(vk_ctx()->device, &ci, NULL, &view) != VK_SUCCESS)
        rt_fatal("vk: render target view (format %d, %ux%u) could not be created", (int)t->format, t->width, t->height);
    if (ex->nviews == ex->cap_views) {
        ex->cap_views = ex->cap_views ? ex->cap_views * 2 : 16;
        ex->views = realloc(ex->views, sizeof(ViewEnt) * (size_t)ex->cap_views);
        if (!ex->views) rt_fatal("vk: out of memory");
        v = ex->views;
    }
    v[ex->nviews++] = (ViewEnt){t->image, t->format, t->view_type, t->aspect, t->level, t->base_layer, t->layers, view};
    return view;
}

typedef struct { int n; VkImageView v[]; } ViewList;

static void free_views(void *arg) {   // completion thread, once the submission that used them is done
    ViewList *l = arg;
    VkDevice dev = vk_ctx()->device;
    for (int i = 0; i < l->n; i++) vkDestroyImageView(dev, l->v[i], NULL);
    free(l);
}

// ---- target history
// The presented frame (gfx_counters.frames + 1; 0: never) each target subresource was last rendered into, and the one
// before. prepare_draw lets a draw into a target the previous frame did not render (a one-shot draw: a volume or LUT
// generated once) wait for its shaders and pipeline instead of skipping it. Per subresource, not per image: a volume
// may be generated a few slices per frame. Direct-mapped: a slot another subresource took reads as not rendered, so a
// collision makes a draw wait rather than skip.
typedef struct { VkImage image; uint32_t level, layer; uint64_t last, prev; } TargetFrames;
static TargetFrames tfr[1024];

static TargetFrames *tfr_slot(const VkExecTarget *t) {
    uint64_t h = ((uint64_t)(uintptr_t)t->image ^ (uint64_t)t->level << 56 ^ (uint64_t)t->base_layer << 40) *
                 0x9E3779B97F4A7C15ull;
    return &tfr[h >> 54];
}

static int tfr_match(const TargetFrames *e, const VkExecTarget *t) {
    return e->image == t->image && e->level == t->level && e->layer == t->base_layer;
}

static void tfr_note(const VkExecTarget *t, uint64_t stamp) {
    TargetFrames *e = tfr_slot(t);
    if (!tfr_match(e, t)) *e = (TargetFrames){t->image, t->level, t->base_layer, 0, 0};
    if (e->last != stamp) e->prev = e->last, e->last = stamp;
}

static int tfr_recent(const VkExecTarget *t, uint64_t stamp) {
    const TargetFrames *e = tfr_slot(t);
    return tfr_match(e, t) && (e->last == stamp ? e->prev : e->last) == stamp - 1;
}

// No frame before the current one rendered the target (a slot another subresource took reads the same).
static int tfr_first(const VkExecTarget *t, uint64_t stamp) {
    const TargetFrames *e = tfr_slot(t);
    return !tfr_match(e, t) || !(e->last == stamp ? e->prev : e->last);
}

// With any, a recent depth target alone does not make the pass per-frame when a colour target is rendered for the
// first time: the game's item icon capture (an inventory or Fuse icon) draws into a new colour target with the
// per-frame depth buffer, once. Its draws skipped there stay missing from the icon until the item is captured again.
int vk_exec_targets_recent(VkExec *ex, int any) {
    resolve_rt(ex);
    uint64_t stamp = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    int n = 0, color = 0, depth = 0, color_first = 0;
    for (int i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if (!t->image) continue;
        n++;
        if (tfr_recent(t, stamp)) *(i < 8 ? &color : &depth) = 1;
        else if (!any) return 0;
        else if (i < 8 && tfr_first(t, stamp)) color_first = 1;
    }
    return any ? color || (depth && !color_first) : n > 0;
}

// One line naming each target of the current pass: format, extent, layer and the two frames it was last rendered in
// (prepare_draw's pipeline wait log).
void vk_exec_targets_describe(VkExec *ex, char *buf, size_t cap) {
    resolve_rt(ex);
    size_t n = 0;
    buf[0] = 0;
    for (int i = 0; i < 9 && n < cap; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if (!t->image) continue;
        const TargetFrames *e = tfr_slot(t);
        int m = tfr_match(e, t);
        int w = snprintf(buf + n, cap - n, "%s%c%d fmt %d %ux%u L%u seen %llu/%llu", n ? "; " : "", i < 8 ? 'c' : 'd',
                         i < 8 ? i : 0, (int)t->format, t->width, t->height, t->base_layer,
                         (unsigned long long)(m ? e->last : 0), (unsigned long long)(m ? e->prev : 0));
        if (w < 0) break;
        n += (size_t)w;
    }
    if (!n) snprintf(buf, cap, "no targets");
}

// ---- passes and barriers

static int has_image(const VkImage *list, int n, VkImage img) {
    for (int i = 0; i < n; i++)
        if (list[i] == img) return 1;
    return 0;
}

static VkImageMemoryBarrier2 layout_barrier(VkImage img, VkImageAspectFlags aspect, VkImageLayout from, VkImageLayout to) {
    return (VkImageMemoryBarrier2){.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                   .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                                   .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                   .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                   .oldLayout = from, .newLayout = to, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                   .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
                                   .subresourceRange = {aspect, 0, VK_REMAINING_MIP_LEVELS, 0, VK_REMAINING_ARRAY_LAYERS}};
}

// Pass after pass. NVN orders the results of rendering for later reads only through nvnCommandBufferBarrier (OP_BARRIER:
// a full barrier at once), and the console's next pass starts its vertex work while the last one's fragments still
// run. A pass that follows another with nothing recorded in between (no game barrier, copy, clear, dispatch or query
// copy: those record their own full barriers) therefore waits only with its fragment work: the previous pass's
// vertex, mesh and fragment stages and attachment writes before this pass's fragment stages and attachment loads.
// Attachment reuse (write after write, a target the last pass sampled) stays ordered. A full barrier before every
// pass (LAS Recomp's rule) drained the GPU about 265 times a frame in the forest; nxnative measured GPU 25 -> 18 ms
// on OoT with the game's barriers alone (nvn_mtl.m barrier_game). REKARINA_PASS_BARRIER=full: the old rule.
static int relaxed_passes(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("REKARINA_PASS_BARRIER");
        on = !(e && !strcmp(e, "full"));
        rt_log("vk: pass after pass: %s", on ? "fragment work waits for the previous pass (game barriers stay full)"
                                              : "full barrier before every pass (REKARINA_PASS_BARRIER=full)");
    }
    return on;
}

static void relaxed_pass_barrier(VkExec *ex) {
    const VkPipelineStageFlags2 src = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_TESSELLATION_CONTROL_SHADER_BIT |
                                      VK_PIPELINE_STAGE_2_TESSELLATION_EVALUATION_SHADER_BIT |
                                      VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT |
                                      VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                                      VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
#ifdef GAME_NS2
                                      | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT
#endif
        ;
    const VkPipelineStageFlags2 dst = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    memory_barrier(ex->cb, src, VK_ACCESS_2_MEMORY_WRITE_BIT, dst, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    ex->recorded++;
    vk_exec_frame.relaxed_barriers++;
}

// Full barrier. The images in the feedback layout become want[] (pass begin) or none (every other barrier): the others
// go back to GENERAL and the new ones leave it, in the same command. pass: called for a pass begin, which may take
// the relaxed barrier above instead.
static void barrier_to_(VkExec *ex, const VkImage *want, const VkImageAspectFlags *wasp, int nwant, int pass);
static int barrier_game_now;   // DEBUG-BARRIER-EXPERIMENT: the barrier being recorded is the game's or the submission end's
static void barrier_to(VkExec *ex, const VkImage *want, const VkImageAspectFlags *wasp, int nwant) {
    barrier_to_(ex, want, wasp, nwant, 0);
}
static void barrier_to_(VkExec *ex, const VkImage *want, const VkImageAspectFlags *wasp, int nwant, int pass) {
    vk_exec_pass_end(ex);
    VkImageMemoryBarrier2 ib[18];
    uint32_t n = 0;
    for (int i = 0; i < ex->nfbl; i++)
        if (!has_image(want, nwant, ex->fbl_img[i]))
            ib[n++] = layout_barrier(ex->fbl_img[i], ex->fbl_aspect[i], VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT,
                                     VK_IMAGE_LAYOUT_GENERAL);
    for (int i = 0; i < nwant; i++)
        if (!has_image(ex->fbl_img, ex->nfbl, want[i]))
            ib[n++] = layout_barrier(want[i], wasp[i], VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT);
    for (int i = 0; i < nwant; i++) ex->fbl_img[i] = want[i], ex->fbl_aspect[i] = wasp[i];
    ex->nfbl = nwant;
    if (ex->barrier_at == ex->recorded && !n) return;
    {   // DEBUG-BARRIER-EXPERIMENT: REKARINA_BARRIER_EXPERIMENT=1 keeps only the game's barriers, layout changes and the
        // end of a submission (a measurement of what the other full barriers cost; the picture may break)
        static int exp = -1;
        if (exp < 0) { const char *e = cfg_get("REKARINA_BARRIER_EXPERIMENT"); exp = e && *e == '1'; }
        if (exp && !n && !barrier_game_now) { vk_exec_frame.skipped_barriers++; return; }
    }
    if (pass && !n && ex->relaxed_at == ex->recorded) return;   // a game barrier ordered the fragments just now
    if (pass && !n && ex->pass_end_at == ex->recorded && relaxed_passes()) {   // pass after pass, no layout change
        relaxed_pass_barrier(ex);
        return;
    }
    vk_exec_frame.fb_layouts += n;
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb,
                           .imageMemoryBarrierCount = n, .pImageMemoryBarriers = ib};
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->barrier_at = ++ex->recorded;
}

// The game's barrier (nvnCommandBufferBarrier). One that orders fragments only (ORDER_FRAGMENTS / _TILED, cache
// invalidations; neither ORDER_PRIMITIVES nor ORDER_INDIRECT_DATA): later fragment work waits for all earlier work,
// later vertex work may start at once, as on the console. ~65 of the ~117 game barriers of a forest frame (render to
// a texture, sample it in the next pass's fragment shader). A dispatch, copy or clear after it still records its full
// barrier; the next pass begin records none. REKARINA_GAME_BARRIER=full: every game barrier full.
static void game_barrier(VkExec *ex, int bits) {
    static int relax = -1;
    if (relax < 0) {
        const char *e = cfg_get("REKARINA_GAME_BARRIER");
        relax = !(e && !strcmp(e, "full"));
    }
    if (relax && !(bits & (NVN_BARRIER_ORDER_PRIMITIVES_BIT | NVN_BARRIER_ORDER_INDIRECT_DATA_BIT)) && !ex->nfbl &&
        !vk_exec_ac_value) {   // a compute rest's queue has no fragment stages
        vk_exec_pass_end(ex);
        const VkPipelineStageFlags2 dst = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                          VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT |
                                          VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        memory_barrier(ex->cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, dst,
                       VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        ex->relaxed_at = ++ex->recorded;
        vk_exec_frame.relaxed_barriers++;
        return;
    }
    barrier_game_now = 1, vk_exec_barrier(ex), barrier_game_now = 0;
}

void vk_exec_barrier(VkExec *ex) { barrier_to(ex, NULL, NULL, 0); }

// Dispatch after dispatch. NVN orders a dispatch after the work before it only through nvnCommandBufferBarrier
// (OP_BARRIER, a full barrier here): the game puts one where a dispatch reads what an earlier one wrote. A dispatch that
// follows another with nothing recorded in between can then run beside it; the full barrier before every dispatch
// drained the GPU ~140 times at the start of each forest frame (GPU 20.3 -> 19.7 ms on the porch, the picture
// unchanged). REKARINA_DISPATCH_BARRIER=full: the barrier before every dispatch, as before.
static int dispatch_barrier_mode(void) {
    static int m = -1;
    if (m < 0) {
        const char *e = cfg_get("REKARINA_DISPATCH_BARRIER");
        m = !(e && !strcmp(e, "full"));
        rt_log("vk: dispatch after dispatch: %s", m ? "ordered by the game's barriers only"
                                                    : "full barrier before every dispatch (REKARINA_DISPATCH_BARRIER=full)");
    }
    return m;
}
void vk_exec_dispatch_barrier(VkExec *ex) {
    if (dispatch_barrier_mode() && !ex->in_pass && ex->dispatch_end_at == ex->recorded && !ex->nfbl) {
        vk_exec_frame.skipped_barriers++;
        return;
    }
    vk_exec_barrier(ex);
}

static void results_point(VkExec *ex);   // below, with the parts

// A chunk owns its rendering scope; the replay thread only walks that scope for bookkeeping.
static void par_pass_begin(VkExec *ex) {
    vkCmdBeginRendering(ex->cb, par_ri);
    ex->in_pass = 1;
    ex->recorded++;
}

static void par_pass_end(VkExec *ex) {
    if (!ex->in_pass) return;
    vk_exec_oq_worker_close(ex);
    vkCmdEndRendering(ex->cb);
    ex->in_pass = 0;
    ex->recorded++;
#ifdef GAME_OOT
    if (ex->hdr_meter_written) vk_hdr_meter_end_pass(ex);
#endif
}

void vk_exec_pass_begin(VkExec *ex) {
    if (ex->in_pass) return;
    if (vk_par_worker) {
        par_pass_begin(ex);
        return;
    }
    resolve_rt(ex);
    results_point(ex);
    vk_contact_pass_begin(ex);
    vk_dlaa_pass(ex);   // DLAA runs before the bloom chain's first pass (vk_dlaa.c)
    // Learned targets (VkTexImage.fb_learned) are attached in the feedback layout
    VkImage want[9];
    VkImageAspectFlags wasp[9];
    int nwant = 0;
    uint32_t fbl = 0;
    for (uint32_t i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if ((i < 8 && i >= ex->pass_ncolor) || !t->image || !t->img || !t->img->fb_learned) continue;
        fbl |= 1u << i;
        if (has_image(want, nwant, t->image)) continue;
        want[nwant] = t->image, wasp[nwant++] = i < 8 ? VK_IMAGE_ASPECT_COLOR_BIT : t->img->aspect;
    }
    barrier_to_(ex, want, wasp, nwant, 1);   // passes, clears, copies and dispatches see each other's results
    ex->fbl_mask = fbl;
    ex->fb_dirty = 0;
    ex->pass_serial++;
    VkRenderingAttachmentInfo ca[8], da = {0}, sa = {0};
    for (uint32_t i = 0; i < ex->pass_ncolor; i++) {
        const VkExecTarget *t = &ex->color[i];
        ca[i] = (VkRenderingAttachmentInfo){.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        if (!t->image) continue;
        ca[i].imageView = view_of(ex, t);
        ca[i].imageLayout = fbl >> i & 1 ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT : VK_IMAGE_LAYOUT_GENERAL;
        ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;    }
    const VkExecTarget *d = &ex->depth;
    if (d->image) {
        VkRenderingAttachmentInfo a = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = view_of(ex, d),
                                       .imageLayout = fbl >> 8 & 1 ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
                                                                   : VK_IMAGE_LAYOUT_GENERAL,
                                       .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
        if (d->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) da = a;
        if (d->aspect & VK_IMAGE_ASPECT_STENCIL_BIT) sa = a;
    }
    // The start mark goes between the barrier and vkCmdBeginRendering, its end right after vkCmdEndRendering
    // (vk_exec_pass_end): outside the pass, and without breaking it.
    VkPassMark *m = ex->mark = vk_pass_mark_begin(ex->cb, VK_MARK_PASS, ex->fence);
    if (m) {
        m->step = ex->mark_passes++;
        m->w = ex->pass_w, m->h = ex->pass_h, m->d = ex->pass_layers;
        m->ncolor = (uint8_t)(ex->pass_ncolor < 8 ? ex->pass_ncolor : 8);
        for (int i = 0; i < m->ncolor; i++) m->color[i] = ex->pass_color[i];
        m->depth = ex->pass_depth ? ex->pass_depth : ex->pass_stencil;
        // The shaders bound now are the first draw's when it repeats the pipeline state; when it does not,
        // vk_exec_draw.c prepare_draw replaces them as it records the new state.
        vk_exec_draw_shaders(&m->first[0], &m->first[1]);
        ex->mark_draws0 = vk_exec_frame.draws;
        m->tgt = vk_pass_mark_image(ex->color[0].image ? ex->color[0].img : d->image ? d->img : NULL);
        m->count = 0;
    }
    {   // DEBUG-OQLOG: render pass sizes (shadow cascades: depth only)
        int vk_oqlog_on(void);
        if (!ex->pass_ncolor && ex->pass_depth && vk_oqlog_on())
            rt_log("OQ pass f %llu %ux%u layers %u depth %d", (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED),
                   ex->pass_w, ex->pass_h, ex->pass_layers, (int)ex->pass_depth);
    }
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {ex->pass_w, ex->pass_h}},
                          .layerCount = ex->pass_layers, .colorAttachmentCount = ex->pass_ncolor,
                          .pColorAttachments = ca,
                          .pDepthAttachment = ex->pass_depth ? &da : NULL,
                          .pStencilAttachment = ex->pass_stencil ? &sa : NULL};
    vkCmdBeginRendering(ex->cb, &ri);
    {   // kept for the chunks of a draw stretch, which begin this pass again in their own command buffers
        memcpy(PAR.ca, ca, sizeof ca[0] * ex->pass_ncolor);
        PAR.da = da, PAR.sa = sa, PAR.ri = ri;
        PAR.ri.pColorAttachments = PAR.ca;
        if (ri.pDepthAttachment) PAR.ri.pDepthAttachment = &PAR.da;
        if (ri.pStencilAttachment) PAR.ri.pStencilAttachment = &PAR.sa;
    }
    vk_exec_frame.passes++;
    ex->recorded++;
    ex->in_pass = 1;
}

static void pass_end(VkExec *ex, int walked) {
    if (!ex->in_pass) return;
    { void vk_exec_drawtime_pass_end(VkExec *ex); vk_exec_drawtime_pass_end(ex); }   // DEBUG-DRAWTIME
    vk_exec_oq_pass_end(ex);
    if (!walked) {
        vkCmdEndRendering(ex->cb);
        ex->recorded++;
    }
    ex->in_pass = 0;
    int end_at = ex->recorded;
#ifdef GAME_OOT
    if (ex->hdr_meter_written) vk_hdr_meter_end_pass(ex);
#endif
#ifdef GAME_NS2
    vk_exec_oq_after_pass(ex);   // the predicate words of the samples-passed reports of the pass
#endif
    if (ex->mark) {
        // vk_exec_frame_log resets the frame counters at a present (REKARINA_TRACE=gfx): then only the draws since it count
        uint64_t d = vk_exec_frame.draws, d0 = ex->mark_draws0;
        ex->mark->draws = (uint32_t)(d >= d0 ? d - d0 : d);
        if (ex->mark->draws) vk_exec_draw_shaders(&ex->mark->last[0], &ex->mark->last[1]);
        {   // DEBUG-MARKDETAIL: REKARINA_MARK_DETAIL=<width>: the line a device loss would print for this pass
            // (vk_pass_mark_detail), for the first 400 passes with draws that are at most that wide
            static int wmax = -1, shown;
            if (wmax < 0) { const char *e = cfg_env("REKARINA_MARK_DETAIL"); wmax = e ? atoi(e) : 0; }
            if (wmax && ex->mark->draws && ex->mark->w <= (uint32_t)wmax && shown < 400) {
                char t[1024];
                shown++;
                vk_pass_mark_detail(ex->mark->id, t, sizeof t);
                if (t[0]) rt_log("%s (frame %llu, %u draws)", t,
                                 (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), ex->mark->draws);
            }
        }
        vk_pass_mark_end(ex->cb, ex->mark);
        ex->mark = NULL;
    }
    for (int i = 0; i < ex->pass_ncolor && i < 8; i++)   // stand-ins of 3D BC targets go to their textures
        if (ex->color[i].image && ex->color[i].img && ex->color[i].img->image != ex->color[i].image)
            bc_copy_back(ex, &ex->color[i]);
    if (!walked) vk_maskfilter_pass_end(ex);   // the shadow mask's temporal filter (REKARINA_MASK_FILTER, on in OoT)
    if (vk_dump_armed) vk_dump_pass_end(ex);
    vk_probe_pass_end(ex);   // DEBUG-PROBE
    pl_pass(ex);             // DEBUG-PASSLIST
    uint64_t stamp = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    for (int i = 0; i < 8; i++)
        if (ex->color[i].image) vk_texture_gpu_written(ex->color[i].tex), tfr_note(&ex->color[i], stamp);
    if (ex->depth.image) vk_texture_gpu_written(ex->depth.tex), tfr_note(&ex->depth, stamp);
    ex->pass_end_at = ex->recorded == end_at ? end_at : -1;   // nothing recorded after the pass: the next may relax
    {   // DEBUG-BIGLOG: a pass into a depth target 8192 or more wide ended
        static int on = -1, n;
        if (on < 0) on = cfg_get("REKARINA_BIGLOG") != NULL;
        if (on && ex->rt.depth && ex->rt.depth->d.width >= 8192 && n++ < 20)
            rt_log("BIGLOG replay pass end depth %p image %p frame %llu", (const void *)ex->rt.depth->mem, (void *)ex->depth.image,
                   (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));
    }
}

void vk_exec_pass_end(VkExec *ex) {
    if (vk_par_worker) par_pass_end(ex);
    else pass_end(ex, 0);
}

// DEBUG-WHONAMES (the device fault report): the queue state's targets
void vk_exec_who_names(uint64_t image) {
    for (int i = 0; i < 8; i++)
        if ((uint64_t)(uintptr_t)X.color[i].image == image) rt_log("  the replay's colour target %d is it", i);
    if ((uint64_t)(uintptr_t)X.depth.image == image) rt_log("  the replay's depth target is it");
}

uint64_t vk_exec_ac_value;
enum { INLINE_NO_PASS, INLINE_REOPEN, INLINE_PAR_REOPEN, INLINE_PAR_NO_PASS };
// Inside a chunk of a draw stretch the commands go into the chunk's own command buffer, between two halves of its
// pass (as vk_exec_par_restart): that is where the draw that asked for them is recorded. The replay thread records
// chunks too (vk_exec_par_join takes what no thread took), and there this function handed out the replay's own command
// buffer, which lies behind every chunk of the part, and ended and began again a pass that was open only in the
// bookkeeping: the copy ran after the draws that needed it and a pass began inside a pass (found from the stack of a
// driver access violation in the field at a run, 2026-11-13: exec_op_, par_try, vk_sync_chain).
VkCommandBuffer vk_exec_inline_begin(int *reopen) {
    if (vk_exec_ac_value) return VK_NULL_HANDLE;   // a compute rest is recorded: the caller submits its own
    *reopen = INLINE_NO_PASS;
    if (vk_par_worker) {
        VkExec *px = &PX;
        if (!px->cb) return VK_NULL_HANDLE;
        *reopen = px->in_pass ? INLINE_PAR_REOPEN : INLINE_PAR_NO_PASS;
        par_pass_end(px);
        return px->cb;
    }
    VkExec *ex = &X;
    if (!x_here || !ex->cb || PAR.walking) return VK_NULL_HANDLE;   // a stretch is walked: its pass is open in the bookkeeping only
    *reopen = ex->in_pass ? INLINE_REOPEN : INLINE_NO_PASS;
    vk_exec_barrier(ex);   // ends the pass; feedback-layout targets go back to GENERAL
    return ex->cb;
}

void vk_exec_inline_end(int reopen) {
    if (reopen == INLINE_PAR_REOPEN || reopen == INLINE_PAR_NO_PASS) {
        VkExec *px = &PX;
        if (reopen == INLINE_PAR_REOPEN) {
            memory_barrier(px->cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
            par_pass_begin(px);
            px->fb_dirty = 0;
        }
        px->recorded++;
        return;
    }
    X.recorded++;
    if (reopen == INLINE_REOPEN) vk_exec_pass_begin(&X);
}

typedef struct { VkBuffer b; VmaAllocation a; } InlineBuf;
static void free_inline_buf(void *arg) {   // completion thread
    InlineBuf *t = arg;
    vmaDestroyBuffer(vk_ctx()->vma, t->b, t->a);
    free(t);
}

VkBuffer vk_exec_inline_buffer(VkDeviceSize size) {
    InlineBuf *t = calloc(1, sizeof *t);
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (!t || vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &t->b, &t->a, NULL) != VK_SUCCESS) { free(t); return VK_NULL_HANDLE; }
    vk_completion_add(X.fence, free_inline_buf, t);
    return t->b;
}

VkRect2D vk_exec_scissor(const VkExec *ex) {
    int64_t w = ex->pass_w, h = ex->pass_h;
    if (!ex->has_scissor) return (VkRect2D){{0, 0}, {(uint32_t)w, (uint32_t)h}};
    int64_t k = ex->pass_scale > 1 ? ex->pass_scale : 1;
    int64_t x0 = ex->scissor.x * k, y0 = ex->scissor.y * k;
    int64_t x1 = x0 + (int64_t)ex->scissor.w * k, y1 = y0 + (int64_t)ex->scissor.h * k;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > w) x1 = w;
    if (y1 > h) y1 = h;
    if (x1 <= x0 || y1 <= y0) return (VkRect2D){{0, 0}, {0, 0}};
    return (VkRect2D){{(int32_t)x0, (int32_t)y0}, {(uint32_t)(x1 - x0), (uint32_t)(y1 - y0)}};
}

// ---- clears

// Outside a pass, a clear of a whole target (no narrower scissor, the target as large as the render area) clears the
// image directly instead of opening a pass for it.
static int whole_target(const VkExec *ex, const VkExecTarget *t) {
    // vkCmdClear*Image addresses a 3D level whole, not one depth plane: slices of a 3D target clear in a pass; it
    // cannot clear a compressed image: views of BC blocks clear in a pass too
    if (ex->in_pass || t->is3d || t->blocks || t->width != ex->pass_w || t->height != ex->pass_h) return 0;
#ifdef GAME_NS2
    if (ex->gpu_cond) return 0;   // a transfer clear ignores conditional rendering: in a pass, ClearAttachments
#endif
    if (!ex->has_scissor) return 1;
    const p_scissor *s = &ex->scissor;
    return s->x <= 0 && s->y <= 0 && (int64_t)s->x + s->w >= t->width && (int64_t)s->y + s->h >= t->height;
}

static void clear_attachment(VkExec *ex, const VkClearAttachment *a) {
    vk_exec_pass_begin(ex);
    VkClearRect cr = {vk_exec_scissor(ex), 0, ex->pass_layers};
    if (!cr.rect.extent.width || !cr.rect.extent.height) return;
#ifdef GAME_NS2
    vk_exec_cond_begin(ex);   // a clear under a GPU condition (vk_exec_xfer.c)
    vkCmdClearAttachments(ex->cb, 1, a, 1, &cr);
    vk_exec_cond_end(ex);
#else
    vkCmdClearAttachments(ex->cb, 1, a, 1, &cr);
#endif
    ex->recorded++;
    ex->fb_dirty |= a->aspectMask & VK_IMAGE_ASPECT_COLOR_BIT ? 1u << a->colorAttachment : 0x100u;
}

static void clear_color(VkExec *ex, const p_clear_color *c) {
    vk_exec_frame.clears++;
    {   // DEBUG-SKIPLOG
        extern int vk_dbg_cond_unknown;
        static int on = -1;
        static unsigned long long lo, hi;
        if (on < 0) { const char *e = cfg_env("REKARINA_SKIPLOG"); on = e && sscanf(e, "%llu-%llu", &lo, &hi) == 2; }
        unsigned long long f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (on && f >= lo && f <= hi && ex->cond_active)
            rt_log("SKIPLOG clear frame %llu disabled %d unknown %d gpu %d cond %#llx mode %d target %d rgba %g %g %g %g", f,
                   ex->render_disabled, vk_dbg_cond_unknown, ex->gpu_cond, (unsigned long long)ex->render_cond.addr, (int)ex->render_cond.mode,
                   c->index, c->rgba[0], c->rgba[1], c->rgba[2], c->rgba[3]);
    }
    if (ex->render_disabled) return;
    resolve_rt(ex);
    int i = c->index;
    if (i < 0 || i >= (int)ex->pass_ncolor || !ex->color[i].image) {
        warn_once(W_CLEAR_TARGET, "colour clear of target %d, which is not bound or not renderable: skipped", i);
        return;
    }
    uint32_t mask = (uint32_t)c->mask & 0xf;
    if (!mask) return;
    if (mask != 0xf) {   // vkCmdClearAttachments ignores write masks; a masked clear needs a draw (vk_exec_draw.c)
        warn_once(W_CLEAR_MASK, "colour clear with channel mask %#x is not replayed: skipped", mask);
        return;
    }
    const VkExecTarget *t = &ex->color[i];
    vk_interface_coverage_write(t->image, t->img ? t->img->addr : 0);
    VkClearColorValue cv;
    memcpy(cv.float32, c->rgba, sizeof cv.float32);   // integer targets: the payload's bits as the front end stored them
    if (whole_target(ex, t)) {
        vk_exec_barrier(ex);
        VkImageSubresourceRange r = {VK_IMAGE_ASPECT_COLOR_BIT, t->level, 1, t->base_layer, t->layers};
        vkCmdClearColorImage(ex->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &r);
        ex->recorded++;
        vk_texture_gpu_written(t->tex);
        return;
    }
    VkClearAttachment a = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .colorAttachment = (uint32_t)i};
    a.clearValue.color = cv;
    clear_attachment(ex, &a);
}

static void clear_ds(VkExec *ex, const p_clear_ds *c) {
    {   // DEBUG-SKIPLOG
        static int on = -1;
        static unsigned long long lo, hi;
        if (on < 0) { const char *e = cfg_env("REKARINA_SKIPLOG"); on = e && sscanf(e, "%llu-%llu", &lo, &hi) == 2; }
        unsigned long long f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (on && f >= lo && f <= hi && ex->cond_active)
            rt_log("SKIPLOG clear_ds frame %llu disabled %d gpu %d cond %#llx mode %d depth %#llx d %g mask %d", f, ex->render_disabled,
                   ex->gpu_cond, (unsigned long long)ex->render_cond.addr, (int)ex->render_cond.mode,
                   ex->depth.img ? (unsigned long long)ex->depth.img->addr : 0ull, c->depth, (int)c->depth_mask);
    }
    if (ex->render_disabled) return;
    resolve_rt(ex);
    const VkExecTarget *t = &ex->depth;
    if (!t->image) return;
    VkImageAspectFlags asp = 0;
    if (c->depth_mask && (t->aspect & VK_IMAGE_ASPECT_DEPTH_BIT)) asp |= VK_IMAGE_ASPECT_DEPTH_BIT;
    uint32_t sm = (uint32_t)c->stencil_mask & 0xff;
    if (sm && (t->aspect & VK_IMAGE_ASPECT_STENCIL_BIT)) {
        if (sm == 0xff) asp |= VK_IMAGE_ASPECT_STENCIL_BIT;
        else warn_once(W_STENCIL_MASK, "stencil clear with mask %#x is not replayed: skipped", sm);
    }
    if (!asp) return;
    float d = c->depth < 0.0f ? 0.0f : c->depth > 1.0f ? 1.0f : c->depth;
    VkClearDepthStencilValue v = {d, (uint32_t)c->stencil & 0xff};
    if (whole_target(ex, t)) {
        vk_exec_barrier(ex);
        VkImageSubresourceRange r = {asp, t->level, 1, t->base_layer, t->layers};
        vkCmdClearDepthStencilImage(ex->cb, t->image, VK_IMAGE_LAYOUT_GENERAL, &v, 1, &r);
        ex->recorded++;
        vk_texture_gpu_written(t->tex);
        return;
    }
    VkClearAttachment a = {.aspectMask = asp};
    a.clearValue.depthStencil = v;
    clear_attachment(ex, &a);
}

// ---- the op loop

static void set_rt(VkExec *ex, const p_set_rt *r) {
    {   // DEBUG-BIGLOG: the replay reaching a SetRenderTargets with a depth target 8192 or more wide
        static int on = -1, n;
        if (on < 0) on = cfg_get("REKARINA_BIGLOG") != NULL;
        if (on && r->depth && r->depth->d.width >= 8192 && n++ < 20)
            rt_log("BIGLOG replay set-rt depth %p %ux%u frame %llu same %d", (const void *)r->depth->mem, r->depth->d.width,
                   r->depth->d.height, (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED),
                   !memcmp(&ex->rt, r, sizeof *r));
    }
    if (!memcmp(&ex->rt, r, sizeof *r)) return;   // the same targets again: the pass goes on
    vk_exec_pass_end(ex);
    ex->rt = *r;
    ex->rt_resolved = 0;
}

// ---- command buffer parts
// NVN starts the GPU as soon as commands are submitted; one Vulkan command buffer per submission starts it only after
// the whole replay. Between passes, once REKARINA_SPLIT_DRAWS (16; 0 off) draws and dispatches were recorded since the
// last part, the part goes to the queue (vk_sync_split) and the replay continues in a new command buffer. Only
// command-buffer state is lost there (bindings and dynamic state: vk_exec_draw_new_cb, ex->dirty); barriers, layouts and
// queries are not: a barrier's first scope is everything earlier in submission order, image layouts belong to the image,
// and occlusion queries end with their pass. Fences, completion callbacks and reports stay per submission: only the last
// part signals the value. nxnative commits every 64 draws between passes; MacNativeGames after 30, then every 40.
// A part also spans at least REKARINA_SPLIT_US (150, measured on the forest porch: 75 to 81 fps against 1000) of replay time: a split costs about 40 us of the render thread
// (vkQueueSubmit2, a command buffer, the mirror prologue), and a fight frame of 2,000 draws made 15 parts on the 7900X;
// a millisecond of recording is a fraction of a millisecond of GPU work, so the GPU still starts early.
// Inside a long pass (OoT's forest: one pass of ~1,200 draws, ~5 ms of recording, while the GPU waited
// with nothing queued) the part also goes to the queue after REKARINA_SPLIT_IN_PASS (60; 0 off) draws: the pass ends
// there (attachments are stored) and begins again in the next part (they are loaded; the replay's passes always load
// and store), and occlusion queries are cut into segments at the pass end as at any other.
static int split_in_pass = -1;
static int split_n = -1, split_draws, part_start;
static uint64_t split_ns, part_t0;
static uint64_t n_parts, n_subs;

// The write-back before the submission's end. What the GPU wrote into VRAM mirrors goes back to guest memory with
// copies and word merges (vk_memory.c), about 2 ms of GPU time a frame in OoT, and they were recorded at the end of
// each submission: for the frame's main one behind its last, heaviest passes, where the present waits for them. But
// the shaders that write those ranges (culling, light lists, fog) are recorded in the first two milliseconds of that
// submission, and for the six after it the GPU waits for the replay between parts of a fraction of a millisecond. So
// once the collected ranges have stopped growing from one part to the next, they go back right there, at the start of
// a part, where the GPU has time; whatever a later shader writes is collected again and goes back at the end.
// REKARINA_WB_EARLY=0: only at the end, as before.
static void wb_early(VkExec *ex) {
    static int on = -1;
    static uint64_t last_p, last_m, last_fence, min_bytes;
    if (on < 0) {
        on = rt_knob("WB_EARLY", "1");
        const char *e = cfg_get("REKARINA_WB_EARLY_KB");
        min_bytes = (uint64_t)(e && atoi(e) > 0 ? atoi(e) : 1024) << 10;
    }
    if (!on || vk_exec_ac_value) return;
    // The small results shaders wrote after the results point go back first, on their own and at once (vk_memory.c
    // vk_mem_early_results): the game reads them early in its next frame.
    if (vk_mem_early_pending()) {
        int bound = 0;
        ex->recorded += (uint64_t)vk_mem_early_results(ex->cb, &bound);
        if (bound) ex->dirty = VK_EXEC_DIRTY_ALL, vk_exec_draw_new_cb();   // the merge bound a pipeline and sets of its own
    }
    uint64_t p, m;
    vk_mem_wb_pending(&p, &m);
    if (ex->fence != last_fence) last_fence = ex->fence, last_p = last_m = UINT64_MAX;
    int stable = p == last_p && m == last_m;
    last_p = p, last_m = m;
    if (!stable || p + m < min_bytes) return;
    int rec = vk_mem_mirror_flush(ex->cb, ex->fence);
    if (!rec) return;
    ex->recorded += (uint64_t)rec;
    ex->dirty = VK_EXEC_DIRTY_ALL;
    vk_exec_draw_new_cb();   // the merges bound a pipeline and sets of their own
    last_p = last_m = 0;
}

static void maybe_split(VkExec *ex) {
    if (split_n < 0) {
        split_n = cfg_int("REKARINA_SPLIT_DRAWS");
        if (split_n < 0) split_n = 0;
        int us = cfg_int("REKARINA_SPLIT_US");
        split_ns = us > 0 ? (uint64_t)us * 1000 : 0;
        rt_log("vk: replay command buffer split every %d draws between passes, parts of at least %d us%s", split_n,
               us > 0 ? us : 0, split_n ? "" : " (off)");
    }
    if (split_in_pass < 0) {
        split_in_pass = cfg_int("REKARINA_SPLIT_IN_PASS");
        if (split_in_pass < 0) split_in_pass = 200;
    }
    if (!split_n || split_draws < (ex->in_pass ? (split_in_pass ? split_in_pass : INT32_MAX) : split_n)) return;
    if (vk_exec_ac_value) return;   // a compute rest is recorded (async compute): its own command buffer
    if (split_ns && kern_now_ns() - part_t0 < split_ns) return;   // split_draws stays: the next draw asks again
    if (vk_exec_wait_deferred()) return;   // the CPU has not signalled a deferred wait: the next draw asks again
    // A split waits for the chunks of draw stretches that are still being recorded: ~7 waits of 0.1 ms a frame, 13 % of
    // the replay thread's time in the forest. The part goes on instead, for up to four times its usual draws, and the
    // op loop submits it the moment the chunks are done, inside a pass too (the first try let it go between passes
    // only: in the field, whose long passes of indirect mesh draws are not cut into chunks, parts then reached the
    // GPU late: 88.8 fps and 60 frames over 25 ms against 97.2 and 4).
    if (PAR.active && __atomic_load_n(&PAR.ndone, __ATOMIC_ACQUIRE) != PAR.njobs && split_draws < 4 * split_n &&
        vk_sync_chain_pending() < 36)
        return;
    split_draws = 0;
    if (ex->recorded == part_start) return;
    vk_exec_pass_end(ex);
    vk_exec_wait_resolve();   // a deferred wait for the CPU: before the part after it goes
    ex->cb = vk_sync_split();
    part_start = ex->recorded;
    part_t0 = kern_now_ns();
    ex->dirty = VK_EXEC_DIRTY_ALL;
    vk_exec_draw_new_cb();
    n_parts++;
    wb_early(ex);
}

// The submission being recorded: its recordings, and the one walk() is in (vk_usnap_resnap after OP_WAIT_EVENT).
static const uint64_t *x_handles;
static int x_n, x_cur;

// Stretches of the command stream with no memory writes in them (vk_exec.h): binds, state and draws only. Indirect mesh
// draws copy their counts once per stretch (vk_exec_draw.c op_draw_mesh).
uint64_t vk_exec_op_window = 1;
const op_hdr *vk_exec_cur_op;
int vk_exec_op_quiet(uint32_t op) {
    switch (op) {
    case OP_BEGIN: case OP_VIEWPORTS: case OP_SCISSOR: case OP_DEPTH_RANGE: case OP_STENCIL: case OP_POLY_OFFSET:
    case OP_BLEND_COLOR: case OP_DEPTH_CLAMP: case OP_RASTER_DISCARD: case OP_PRIM_RESTART: case OP_COPY_STRIDE:
    case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL: case OP_DRAW: case OP_DRAW_IDX: case OP_DRAW_IDX_IND:
    case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF:
    case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_SET_RT:
    case OP_RENDER_ENABLE: case OP_RENDER_COND: case OP_MODEL_SCOPE:
#ifdef GAME_NS2
    case OP_DRAW_MESH: case OP_DRAW_MESH_IND: case OP_PUSH_CONST: case OP_COPY_IMAGE_STRIDE:
#endif
        return 1;
    default: return 0;
    }
}

static void walk(VkExec *ex, const op_hdr *h, int depth);
static void held_check(VkExec *ex);
static void ac_graphics_waits(VkExec *ex, uint64_t cp);
static int nheld, resuming, x_holdable;

static void bind_ubo_op(VkExec *ex, const op_hdr *h) {
    const void *a = nvn_op_payload(h);
    const p_bind_buf *b = a;
    if (!b->stage && !b->index) {
        const p_projection_bind *p = h->size >= sizeof *h + sizeof(p_projection_bind) ? a : NULL;
        ex->scene_projection = p ? p->scene : (p_scene_projection){0};
#ifdef GAME_OOT
        if (ex->scene_projection.sample.token) {
            const float *projection = ex->scene_projection.sample.unjittered;
            ex->hdr_meter_projection[0] = projection[0];
            ex->hdr_meter_projection[1] = projection[5];
            ex->hdr_meter_projection_frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
        }
#endif
    }
    vk_exec_op_draw(ex, OP_BIND_UBO, a);
}

// A state op (fixed-function state, the descriptor pools in use): written into ex. 0: h is not one.
static int state_op(VkExec *ex, const op_hdr *h) {
    const void *a = nvn_op_payload(h);
    switch (h->op) {
    case OP_BEGIN: ex->work_scope = NULL; break;
    case OP_MODEL_SCOPE:
        if (!ex->work_scope) {
            ex->work_scope = a;
            ex->work_scope_end = (uintptr_t)nvn_op_next(h) + ex->work_scope->bytes;
        }
        break;
    case OP_VIEWPORTS: {
        const p_viewports *v = a;
        for (int i = 0; i < v->n && v->first + i < 16; i++)
            if (v->first + i >= 0) memcpy(ex->vp[v->first + i], v->v[i], sizeof ex->vp[0]);
        ex->has_vp = 1;
        ex->dirty |= VK_EXEC_DIRTY_VIEWPORT;
        break;
    }
    case OP_SCISSOR:
        ex->scissor = *(const p_scissor *)a;
        ex->has_scissor = 1;
        ex->dirty |= VK_EXEC_DIRTY_SCISSOR;
        break;
    case OP_DEPTH_RANGE: {
        const p_depth_range *r = a;
        ex->depth_range[0] = r->n, ex->depth_range[1] = r->f;
        ex->dirty |= VK_EXEC_DIRTY_VIEWPORT;
        break;
    }
    case OP_STENCIL: {
        const p_stencil *s = a;
        for (int f = 0; f < 2; f++) {
            if (!(s->face & (1 << f))) continue;
            if (s->which == 0) ex->st_vmask[f] = (uint32_t)s->value_mask;
            else if (s->which == 1) ex->st_wmask[f] = (uint32_t)s->mask;
            else ex->st_ref[f] = (uint32_t)s->ref;
        }
        ex->dirty |= VK_EXEC_DIRTY_STENCIL;
        break;
    }
    case OP_POLY_OFFSET:
        ex->poly_offset = *(const p_poly_offset *)a;
        ex->dirty |= VK_EXEC_DIRTY_DEPTH_BIAS;
        break;
    case OP_BLEND_COLOR:
        memcpy(ex->blend_color, ((const p_blend_color *)a)->rgba, sizeof ex->blend_color);
        ex->dirty |= VK_EXEC_DIRTY_BLEND_CONST;
        break;
    case OP_DEPTH_CLAMP: ex->pipe.depth_clamp = ((const p_flag *)a)->value != 0; break;
    case OP_RASTER_DISCARD: ex->pipe.raster_discard = ((const p_flag *)a)->value != 0; break;
    case OP_PRIM_RESTART: ex->prim_restart = *(const p_prim_restart *)a; break;
    case OP_COPY_STRIDE: ex->copy_stride = ((const p_copy_stride *)a)->stride; break;
    case OP_SET_TEX_POOL: ex->tex_pool = ((const p_set_pool *)a)->pool, ex->tex_serial++; break;
    case OP_SET_SAMP_POOL: ex->samp_pool = ((const p_set_pool *)a)->pool; break;
#ifdef GAME_NS2
    case OP_COPY_IMAGE_STRIDE: ex->copy_image_stride = ((const p_copy_stride *)a)->stride; break;
#endif
    default: return 0;
    }
    return 1;
}

// ---- Draw stretches on worker threads (replay_threads in rekarina.ini: the threads, the replay thread among them; 1: off)
// A frame of OoT is one submission of ~30,000 ops, replayed by one thread while the game's other threads
// already record the next frame (docs/oot-architecture.md). About 1,600 of its 2,150 draws lie in a few long stretches
// inside one pass: the depth prepass and the lit pass (each object's draw between a samples-passed reset and report,
// the lit ones under a render condition) and the shadow cascades. A stretch holds only ops that write no memory (binds,
// state, draws: vk_exec_op_quiet), counter resets and reports and render conditions, with one set of render targets.
// Such a stretch is cut into chunks, each recorded by another thread into a command buffer of its own that is chained
// into the part being recorded (vk_sync_chain), with the pass begun again in each (load and store, as at any part
// boundary):
// - The replay thread walks the stretch once (par_try): state and bind ops are applied to its queue state as always,
//   resets, reports and render conditions run their bookkeeping in order (the pending reports, their predicate words,
//   the segments for the completion thread, the conditions' decisions), and the draws are only counted. Where a draw
//   would begin an occlusion query it is given the slot (vk_exec_xfer.c oq_scan). At each chunk's first op the queue
//   state is copied for the chunk, which starts at once and follows the walk (PAR.walked).
// - A chunk (par_run) starts from that copy with nothing bound in its command buffer, replays its ops itself (state and
//   binds into its copy, draws through vk_exec_op_draw), takes the conditions' decisions and the query slots from the
//   walk, and ends its queries at the resets and reports. The draw path's memos and per-command-buffer state are per
//   thread (vk_exec_draw.c); what a chunk shares takes locks (the texture registry on a lookup, the pipeline cache,
//   mirror creation and sync).
// - The replay thread does not wait: the pass ends in its bookkeeping (targets written, the predicate copies of the
//   stretch's reports, recorded behind the chunks) and it goes on in a new command buffer of the same part. It waits
//   for the chunks (vk_exec_par_join, recording those nobody took) where the part is submitted, so the uploads and
//   mirror copies their recording asked for precede the part as they would have, and before an op that changes what
//   they read or needs it unchanged (par_join_op: waits, signals, dispatches, copies, uniform updates).
// Measured in the forest, 2026-11-12: a stretch of ~1,100 draws took 3.8 ms on the replay thread and takes 0.4 ms of it
// (the walk); with a wait at each stretch's end, stretches under 100 draws cost more than they saved.
// Not cut: a stretch of fewer than REKARINA_PAR_MIN draws, one whose pass has no attachment, a target in the feedback layout
// or a 3D BC stand-in, indirect mesh draws (their count copies end the pass), a held rest or compute rest being
// recorded, per-object motion (its lists follow the draws in order), DLSS outside OoT (the viewport jitter and the
// camera block are noted draw by draw; OoT's DLSS hooks keep no order: vk_dlaa.c vk_dlaa_draw counts under a lock
// and evaluate waits for the chunks, REKARINA_PAR_DLSS=0 keeps them off as before), a frame dump.
static int par_dlss_bars(void) {
#ifdef GAME_OOT
    static int ok = -1;
    if (ok < 0) ok = rt_knob("PAR_DLSS", "1");
    if (ok) return 0;
#endif
    return vk_dlaa_wanted();
}

int vk_exec_mesh_emulated(void);   // vk_exec_draw.c: mesh draws run emulated, each ending the pass (mesh_emulate)
// REKARINA_HDR_SERIAL=1: with the RenoDX present, every stretch replays serially (the old way); 0, the default, only
// the stretches drawing into a presented window texture (vk_hdr_window_target).
static int par_hdr_serial(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("HDR_SERIAL", "0");
    return on;
}

static int par_is_draw(uint32_t op) {
    return op == OP_DRAW || op == OP_DRAW_IDX || op == OP_DRAW_IDX_IND
#ifdef GAME_NS2
           || (op == OP_DRAW_MESH && !vk_exec_mesh_emulated())
#endif
        ;
}

static void par_fragment_barrier(VkCommandBuffer cb) {   // as relaxed_pass_barrier: the pass goes on after the part before
    const VkPipelineStageFlags2 src = VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_2_GEOMETRY_SHADER_BIT |
                                      VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT
#ifdef GAME_NS2
                                      | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT | VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT
#endif
        ;
    const VkPipelineStageFlags2 dst = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT |
                                      VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    static int full = -1;   // DEBUG-PARBARRIER: REKARINA_PAR_BARRIER=full: every stage of a chunk waits for the chunk before
    if (full < 0) { const char *e = cfg_env("REKARINA_PAR_BARRIER"); full = e && !strcmp(e, "full"); }
    if (full) {
        memory_barrier(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                       VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
        return;
    }
    memory_barrier(cb, src, VK_ACCESS_2_MEMORY_WRITE_BIT, dst, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
}

void vk_exec_par_restart(VkExec *ex) {
    par_pass_end(ex);
    memory_barrier(ex->cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                   VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    ex->recorded++;
    par_pass_begin(ex);
    ex->fb_dirty = 0;
}

// Records one chunk into its command buffer; any thread.
static void par_run(ParJob *j) {
    const CfgRenderState *settings = cfg_render_use(j->settings);
    VkExec *ex = &PX;
    *ex = j->st;
    ex->cb = j->cb;
    ex->recorded = 0;
    ex->mark = NULL;
    ex->dirty = VK_EXEC_DIRTY_ALL;
    ex->in_pass = 0;
    ex->fb_dirty = 0;
    ex->fb_fs = NULL;
    ex->barrier_at = ex->pass_end_at = ex->dispatch_end_at = ex->relaxed_at = -1;
    int was_worker = vk_par_worker;
    uint64_t was_fence = par_fence;
    vk_par_worker = 1, par_fence = ex->fence;
    uint64_t was_asof = vk_texture_asof(j->asof);   // textures as they were at the stretch's place, not where the replay is now
    vk_exec_draw_upd_limit(j->updn), vk_usnap_view_limit(j->useq);   // and the inline uniform updates
    vk_exec_draw_new_cb();
    vk_exec_oq_worker(1);
    par_ri = &j->ri;
    if (j->crumb) vk_crumb(ex->cb, VK_CRUMB_CHUNK, j->crumb);   // DEBUG-PARCRUMB
    par_fragment_barrier(ex->cb);
    ex->recorded++;
    par_pass_begin(ex);
    uint32_t di = 0, ci = 0;
    uintptr_t lim = 0;
    for (const op_hdr *h = j->from;; h = nvn_op_next(h)) {
        // The chunk follows the walk, which may still be inside it: it gives each op's slot or decision and says where
        // the chunk ends. The position first, then the end: an end the walk has passed is set before the position says so.
        while ((uintptr_t)h >= lim) {
            uintptr_t w = (uintptr_t)__atomic_load_n(&PAR.walked, __ATOMIC_ACQUIRE);
            const op_hdr *to = __atomic_load_n(&j->to, __ATOMIC_ACQUIRE);
            if (to && h == to) goto done;
            lim = to ? (uintptr_t)to : w;   // an end that is set: the walk is past the whole chunk (and may be in a later stretch)
            if ((uintptr_t)h < lim) break;
            _mm_pause();
        }
        if (ex->work_scope && (uintptr_t)h >= ex->work_scope_end) ex->work_scope = NULL;
        const void *a = nvn_op_payload(h);
        uint32_t op = h->op;
        if (par_is_draw(op)) {
            tl_stream_pos = j->pos0 + di + 1;
            vk_exec_oq_worker_slot(j->slot[di++]);
            vk_exec_op_draw(ex, op, a);
        } else if (op == OP_RENDER_COND || op == OP_RENDER_ENABLE) {
            const ParCond *c = &j->cond[ci++];
            ex->render_disabled = c->disabled, ex->has_render_cond = c->has_cond, ex->render_cond = c->rc, ex->dbg_zero = c->dbg_zero;
#ifdef GAME_NS2
            ex->gpu_cond = c->gpu_cond, ex->cond_active = c->cond_active, ex->cond_cpu = c->cond_cpu;
            ex->gpu_cond_off = c->off, ex->gpu_cond_flags = c->flags, ex->gpu_cond_buf = c->buf;
#endif
        } else if (op == OP_RESET_COUNTER || op == OP_REPORT_COUNTER) {
            vk_exec_oq_worker_close(ex);
        } else if (op == OP_BIND_UBO) {
            bind_ubo_op(ex, h);
        } else if (!state_op(ex, h)) {
            vk_exec_op_draw(ex, op, a);   // a bind op
        }
    }
done:
    par_pass_end(ex);
    if (j->crumb) vk_crumb(ex->cb, VK_CRUMB_CHUNK_END, j->crumb);   // DEBUG-PARCRUMB
    vkEndCommandBuffer(ex->cb);
    vk_exec_oq_worker(0);
    vk_texture_asof(was_asof);
    vk_exec_draw_upd_limit(UINT32_MAX), vk_usnap_view_limit(UINT32_MAX);
    vk_par_worker = was_worker, par_fence = was_fence;
    cfg_render_use(settings);
}

// prio: the chunk thread's raised priority while it records a chunk (PAR.prio), 0 for the thread that replays, which
// keeps its own
static int par_take_run(int prio) {
    int n = __atomic_load_n(&PAR.njobs, __ATOMIC_ACQUIRE);
    for (int i = 0; i < n; i++) {
        int32_t one = 1;
        if (__atomic_load_n(&PAR.jstate[i], __ATOMIC_RELAXED) != 1 ||
            !__atomic_compare_exchange_n(&PAR.jstate[i], &one, 2, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED))
            continue;
        if (prio > 0) plat_thread_priority(prio > 1 ? PLAT_PRIO_HIGHEST : PLAT_PRIO_ABOVE_NORMAL);
        par_run(&PAR.job[i]);
        if (prio > 0) plat_thread_priority(PLAT_PRIO_NORMAL);
        __atomic_fetch_add(&PAR.ndone, 1, __ATOMIC_RELEASE);
        __atomic_store_n(&PAR.jstate[i], 3, __ATOMIC_RELEASE);
        return 1;
    }
    return 0;
}

static void par_thread(void *arg) {
    int me = (int)(intptr_t)arg;
    sec_thread_off = 1;   // REKARINA_SECTIONS times the replay thread
    {   // DEBUG-PROFPAR
        extern volatile uint32_t prof_par_tid[16];
        static volatile int32_t n;
        int i = __atomic_fetch_add(&n, 1, __ATOMIC_RELAXED);
        if (i < 16) prof_par_tid[i] = plat_thread_id();
    }
    for (int idle = 0;;) {
        if (par_take_run(PAR.prio)) { idle = 0; continue; }
        // chunks are out or a stretch is being walked: the next comes within tens of microseconds; the thread sleeps
        // only after a while without one
        if (__atomic_load_n(&PAR.active, __ATOMIC_ACQUIRE) || idle < PAR.spin) { idle++; _mm_pause(); continue; }
        __atomic_store_n(&PAR.asleep[me], 1, __ATOMIC_SEQ_CST);
        if (!__atomic_load_n(&PAR.active, __ATOMIC_SEQ_CST)) plat_event_wait_ns(&PAR.ev[me], 100000000);
        __atomic_store_n(&PAR.asleep[me], 0, __ATOMIC_SEQ_CST);
        idle = 0;
    }
}

static int par_on(void) {
    if (PAR.threads >= 0) return PAR.chunks > 0;
    const char *c = cfg_get("REKARINA_PAR_CHUNKS");
    int n = cfg_int("REKARINA_REPLAY_THREADS");
    if (n <= 0) {   // by the processor: the game's own threads record the next frame meanwhile and keep their cores
#ifdef GAME_OOT
        uint32_t cpus = plat_cpu_weighted();   // a slower core (Intel's efficient cores) counts half
        // 8 logical processors (4 cores / 8 threads, as a Steam Deck): 3 measured better than 4 in the forest
        // (1 % low 51 against 44 fps, 7 against 17 frames over 25 ms) and no worse in the Market
        // 4 or 5 (four cores, as a small desktop CPU): 2. Four processors with the core budget (kernel.c), 1080p,
        // 2026-11-19: the square's replay took 16.3 ms a frame alone and 10.5 with a second thread, 48.5 against 54.2
        // fps (1 % low 26.8 / 31.4), the village unchanged (62.7 / 62.9 fps); a second pair at the square (ABBA) gave
        // a median replay of 15.4 and 17.4 ms against 10.6 and 12.4, 51.0 and 45.7 against 52.8 and 47.0 fps, the
        // 1 % low unchanged (31.8 and 23.8 against 31.2 and 24.8), frames over 25 ms 480 against 319
        n = cpus >= 12 ? 6 : cpus >= 6 ? 3 : cpus >= 4 ? 2 : 1;
#else
        n = 1;   // measured and checked on OoT only so far
#endif
    }
    if (n > PAR_CHUNKS) n = PAR_CHUNKS;
    if (rt_knob("VK_RECORDER", "0")) n = 1;   // the recorder thread makes the driver calls of one command buffer at a time
    // chunks of one stretch at most: a few more than threads, so a thread that is done finds another
    // (REKARINA_PAR_CHUNKS with one thread: chunks recorded by the replay thread alone, a test of the mechanism)
    PAR.chunks = c && *c ? atoi(c) : n > 1 ? n + 4 : 0;
    if (PAR.chunks > PAR_CHUNKS) PAR.chunks = PAR_CHUNKS;
    if (PAR.chunks < 1) PAR.chunks = 0;
    for (int i = 0; i < PAR_CHUNKS; i++) plat_event_init(&PAR.ev[i], 0, 0);
    const char *mn = cfg_get("REKARINA_PAR_MIN"), *sp = cfg_get("REKARINA_PAR_SPIN");
    PAR.min_draws = mn && atoi(mn) > 0 ? atoi(mn) : PAR_MIN;
    PAR.spin = (int)plat_spin_count(sp && *sp ? (uint32_t)atoi(sp) : 3000);   // as long on any processor (platform.h)
    const char *pe = cfg_get("REKARINA_PAR_PER");
    PAR.per = pe && atoi(pe) > 0 ? atoi(pe) : 64;
    // REKARINA_PAR_PRIO: the host priority of a chunk thread while it records a chunk: 0 normal, 1 above normal, 2 highest.
    // Unset, it is the priority of the thread that replays, so that thread does not wait at the join (vk_exec_par_join)
    // for a chunk thread the scheduler left out for a whole quantum (two 15.6 ms clock ticks on a desktop Windows):
    // REKARINA_REPLAY_PRIO (replay_main in vk_backend.c, above normal by default) with async replay, else
    // REKARINA_SUBMIT_PRIO (nvnQueueSubmitCommands in nvn_api_core.c). The lake, 8 cores, 60 fps limit, with the
    // chunk threads raised for good: 3 frames of 40 ms in 41 s at normal priority, each with the replay thread 32 ms off
    // its core (SUB_OFFCPU); none in 46 s with both at 1, 2026-11-25. Raised only while a chunk is recorded, not while the
    // thread spins for the next one or sleeps, so it does not take cores from the game's own threads in between.
    {
        const char *pp = cfg_get("REKARINA_PAR_PRIO");
        if (pp && *pp) PAR.prio = atoi(pp);
        else if (gfx_replay_async()) {
            const char *rp = cfg_get("REKARINA_REPLAY_PRIO");
            PAR.prio = rp && *rp ? atoi(rp) : 1;
        } else
            PAR.prio = rt_knob("SUBMIT_PRIO", "0");
        if (PAR.prio < 0) PAR.prio = 0;
    }
    int started = 1;
    for (int i = 1; i < n; i++)
        if (plat_thread_spawn(par_thread, (void *)(intptr_t)(i - 1), "vk replay chunk", 0)) started++;
    PAR.threads = started;
    if (PAR.chunks)
        rt_log("vk: draw stretches of %d draws or more are recorded in up to %d chunks by %d threads (replay_threads)",
               PAR.min_draws, PAR.chunks, PAR.threads);
    else
        rt_log("vk: draws are recorded by the replay thread alone (replay_threads = 1)");
    return PAR.chunks > 0;
}

// DEBUG-PARCRUMB: what the last stretches were, for the breadcrumb report of a lost device
static struct { uint32_t serial, w, h, nd, ncolor; int cfmt, dfmt; uint64_t fence, frame; } par_log[256];
static uint32_t par_serial;
static int par_crumbs = -1;
void vk_exec_par_describe(uint32_t stretch, char *o, size_t size) {
    const typeof(par_log[0]) *l = &par_log[stretch & 255];
    if (l->serial != stretch) { snprintf(o, size, "no record"); return; }
    snprintf(o, size, "%u draws, pass %ux%u, %u colour targets, colour format %d, depth format %d, submission %llu, frame %llu", l->nd,
             l->w, l->h, l->ncolor, l->cfmt, l->dfmt, (unsigned long long)l->fence, (unsigned long long)l->frame);
}

// Every chunk handed out is recorded and its command buffer ended. The replay thread records what nobody took.
static int par_why;                  // DEBUG-PARWHY: who asks for the join: 0 a split, 1 + op an op, 100 the go, 101 room
static uint64_t par_why_n[128], par_why_ns[128], par_why_open[128];   // DEBUG-PARWHY
void vk_exec_par_join(void) {
    if (!PAR.active) return;
    uint64_t t0 = plat_now_ns();
    int why = par_why & 127, open = PAR.njobs - __atomic_load_n(&PAR.ndone, __ATOMIC_ACQUIRE);   // DEBUG-PARWHY
    par_why = 0;
    uint64_t s0 = sec_begin();
    while (par_take_run(0)) {}
    uint64_t s1 = s0 ? sec_clock() : 0;
    int n = PAR.njobs;
    for (int i = 0; i < n; i++)
        while (__atomic_load_n(&PAR.jstate[i], __ATOMIC_ACQUIRE) != 3) _mm_pause();
    if (s0) sec_add(SEC_PAR_RUN, s1 - s0), sec_add(SEC_PAR_SPIN, sec_clock() - s1);
    void vk_exec_draw_tchk_drain(void);
    vk_exec_draw_tchk_drain();   // the chunks' storage checks, before the part and its uploads go
    for (int i = 0; i < n; i++) PAR.jstate[i] = 0;
    __atomic_store_n(&PAR.ndone, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&PAR.njobs, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&PAR.active, 0, __ATOMIC_SEQ_CST);
    PAR.nslots = PAR.nconds = 0;
    PAR.join_ns += plat_now_ns() - t0, PAR.joins++;
    par_why_n[why]++, par_why_ns[why] += plat_now_ns() - t0, par_why_open[why] += (uint64_t)open;   // DEBUG-PARWHY
}

// 1: op needs what the chunks out read to stay as it is, or changes it: they are waited for first. State and bind ops,
// draws, the counters' ops, barriers and attachment clears do neither. Nor does a dispatch: its commands go into the
// replay thread's command buffer, behind the chunks in queue order, and what its binding notes (ranges a shader
// writes, the write-back) is bookkeeping for the submission's end, under vk_par_lock where a chunk may note the same.
// Nor a signal, though it may let a held rest go on (held_resume: the chunks read copies of the queue state and their
// own view of the uniform snapshot). A wait does when the CPU's late writes are taken in after it (the uniform snapshot
// and the mirrors are made again). Measured in the
// forest: with every one of them waiting, 1.2 ms of a frame went into ~12 waits of 0.06 to 0.13 ms.
// Nor an inline uniform update (REKARINA_PAR_UB_JOIN=1: it waits, as before): the chunks read the snapshot's regions
// and the update ring as of their stretch's handout (vk_usnap_drop_update, vk_exec_draw_upd_limit), its copy goes into
// the replay thread's command buffer behind them, and its write-back note is taken under vk_par_lock. Its wait was the
// largest: ~5 a frame at 0.19 ms, a millisecond of the replay thread's frame in the forest and the field.
int vk_exec_ub_nojoin(void) {
    static int on = -1;
    if (on < 0) on = !rt_knob("PAR_UB_JOIN", "0");
    return on;
}

static int par_join_op(uint32_t op) {
    return !(vk_exec_op_quiet(op) || op == OP_RESET_COUNTER || op == OP_REPORT_COUNTER || op == OP_BARRIER ||
             op == OP_CLEAR_COLOR || op == OP_CLEAR_DS || op == OP_END || op == OP_DISPATCH || op == OP_DISPATCH_IND ||
             op == OP_SIGNAL_EVENT || op == OP_WAIT_EVENT || (op == OP_UPDATE_UB && vk_exec_ub_nojoin()));
}

static void part_split(VkExec *ex);
static const op_hdr *exec_op_(VkExec *ex, const op_hdr *h, int depth, int holdable, int *stop);

// A stretch that begins at the draw op h: handed out in chunks, and the op after it returned; NULL: h is replayed as
// always.
static const op_hdr *par_try(VkExec *ex, const op_hdr *h) {
#ifdef GAME_OOT
    // Later draws may overwrite a checked output through an attachment or a storage image.
    if (vk_hdr_output_pending()) return NULL;
#endif
    if (h >= PAR.no_from && h < PAR.no_until) return NULL;
    uint32_t nd = 0, nc = 0;
#ifdef GAME_OOT
    int has_img_bind = 0;
#endif
    const op_hdr *e = h;
    for (;; e = nvn_op_next(e)) {
        uint32_t op = e->op;
        if (e->size < sizeof *e || (e->size & 7) || op >= OP_COUNT_) break;
        if (par_is_draw(op)) {
            const void *draw_payload = nvn_op_payload(e);
            unsigned role = op == OP_DRAW ? ((const p_draw *)draw_payload)->draw_role :
                            op == OP_DRAW_IDX ? ((const p_draw_idx *)draw_payload)->draw_role :
                            op == OP_DRAW_IDX_IND ? ((const p_draw_idx_ind *)draw_payload)->draw_role : 0;
            // Scene output and bloom draws publish ordered resource handoffs on the replay thread.
            if ((role & ~DRAW_ROLE_INTERFACE) || (role && ex->interface_coverage)) break;
            nd++;
            continue;
        }
        if (op == OP_RENDER_COND || op == OP_RENDER_ENABLE) { nc++; continue; }
        if (op == OP_RESET_COUNTER) {
            if (((const p_reset_counter *)nvn_op_payload(e))->type != NVN_COUNTER_TYPE_SAMPLES_PASSED) break;
            continue;
        }
        if (op == OP_REPORT_COUNTER) {
            if (((const p_report_counter *)nvn_op_payload(e))->type != NVN_COUNTER_TYPE_SAMPLES_PASSED) break;
            continue;
        }
#ifdef GAME_OOT
        if (op == OP_BIND_IMG) has_img_bind = 1;
#endif
        if (op == OP_SET_RT || op == OP_END || !vk_exec_op_quiet(op)) break;
#ifdef GAME_NS2
        if (op == OP_DRAW_MESH_IND || op == OP_DRAW_MESH) break;   // (a direct one reaches here only emulated)
#endif
    }
    PAR.no_from = h, PAR.no_until = e;   // whatever comes of it, the draws up to e do not ask again
    // the frame dump lists draws in order: no chunks in the frames it logs (armed and waiting, it leaves them be: the
    // faults it is armed to catch can depend on them)
    if (nd < (uint32_t)PAR.min_draws || vk_exec_ac_value || resuming || (vk_dump_armed && vk_dump_active_dbg()) ||
        (vk_hdr_track && (par_hdr_serial() || vk_hdr_window_target(ex->color[0].image))) || par_dlss_bars() ||
        vk_mvprev_on() || vk_exec_wait_deferred())
        return NULL;
    if (!ex->in_pass) {
        ex->dirty |= VK_EXEC_DIRTY_VIEWPORT | VK_EXEC_DIRTY_SCISSOR;
        vk_exec_pass_begin(ex);
    }
    if (ex->pass_noatt || ex->nfbl || ex->fbl_mask || vk_interface_coverage_serial_required(ex)) return NULL;
    for (uint32_t i = 0; i < ex->pass_ncolor && i < 8; i++)
        if (ex->color[i].image && ex->color[i].img && ex->color[i].img->image != ex->color[i].image) return NULL;
    uint64_t t0 = plat_now_ns();
    // Chunks of about PAR.per draws: each begins the pass again behind a fragment barrier, a point where the GPU's
    // fragment work of the chunk before must be done. With 12 draws a chunk the GPU took 0.4 ms longer per frame in
    // The field (9.4 against 9.0 ms busy) and the frame rate fell although the replay was 2 ms shorter.
    int want = (int)(nd / (uint32_t)PAR.per);
    if (want > PAR.chunks) want = PAR.chunks;
    if (want < 1) want = 1;
    uint32_t share = (nd + (uint32_t)want - 1) / (uint32_t)want;
    vk_exec_pass_end(ex);                // the pass so far, whole in the command buffer it began in
    // room for the chunks: in the part's chain, the job table and the walk's lists; else the part goes first
    if (PAR.njobs + want > PAR_JOBS || PAR.nslots + nd > PAR.cap_slots || PAR.nconds + nc > PAR.cap_conds ||
        vk_sync_chain_pending() + want + 1 > 44) {
        par_why = 101;
        if (ex->recorded != part_start || vk_sync_chain_pending()) part_split(ex);
        vk_exec_par_join();
        if (nd > PAR.cap_slots || !PAR.slots) {
            PAR.cap_slots = nd + nd / 2 + 4096;
            PAR.slots = realloc(PAR.slots, sizeof *PAR.slots * PAR.cap_slots);
        }
        if (nc > PAR.cap_conds || !PAR.conds) {
            PAR.cap_conds = nc + nc / 2 + 4096;
            PAR.conds = realloc(PAR.conds, sizeof *PAR.conds * PAR.cap_conds);
        }
        if (!PAR.slots || !PAR.conds) rt_fatal("vk: out of memory (draw stretch)");
    }
    if (!PAR.active) {
        __atomic_store_n(&PAR.active, 1, __ATOMIC_SEQ_CST);
        for (int i = 0; i + 1 < PAR.threads; i++)   // awake by the time the first chunk is out
            if (__atomic_load_n(&PAR.asleep[i], __ATOMIC_SEQ_CST)) plat_event_set(&PAR.ev[i]);
    }
    vk_exec_draw_par_prepare(ex);
    VkCommandBuffer cbs[PAR_CHUNKS];
    VkCommandBuffer next = vk_sync_chain(want, cbs);
    if (!next) rt_fatal("vk: no room in the part for %d chunks", want);
    ex->cb = next;
    ex->dirty = VK_EXEC_DIRTY_ALL;
    vk_exec_draw_new_cb();
    uint32_t crumb0 = 0;   // DEBUG-PARCRUMB
    if (par_crumbs < 0) par_crumbs = cfg_get("REKARINA_PAR_CRUMBS") != NULL;
    if (par_crumbs) {
        uint32_t sn = ++par_serial & 0xffffff;
        if (!sn) sn = ++par_serial & 0xffffff;
        par_log[sn & 255] = (typeof(par_log[0])){sn, ex->pass_w, ex->pass_h, nd, ex->pass_ncolor, (int)ex->pass_color[0],
                                                 (int)ex->pass_depth, ex->fence, __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED)};
        crumb0 = sn << 4;
        vk_crumb(ex->cb, VK_CRUMB_AFTER_STRETCH, crumb0 | (uint32_t)(want & 15));
    }
    __atomic_store_n(&PAR.walked, h, __ATOMIC_RELEASE);
    ex->in_pass = 1;                     // in the bookkeeping: the reports' predicate copies wait for the pass's end
    PAR.walking = 1;
    vk_exec_oq_scan(1);
    vk_par_walk = 1;
    uint32_t in_chunk = 0;
    int k = 0, ji = PAR.njobs;
#define PAR_JOB_OUT(first_op)                                                                                               \
    do {                                                                                                                    \
        j = &PAR.job[ji];                                                                                                   \
        j->st = *ex, j->from = (first_op), j->to = NULL, j->cb = cbs[k];                                                    \
        j->crumb = crumb0 ? crumb0 | (uint32_t)(k & 15) : 0;                                                                \
        j->asof = asof_mark, j->updn = upd_mark, j->useq = useq_mark, j->pos0 = x_stream_pos;                                \
        j->settings = settings;                                                                                             \
        j->slot = PAR.slots + PAR.nslots, j->cond = PAR.conds + PAR.nconds;                                                 \
        memcpy(j->ca, PAR.ca, sizeof j->ca), j->da = PAR.da, j->sa = PAR.sa, j->ri = PAR.ri;                                \
        j->ri.pColorAttachments = j->ca;                                                                                    \
        if (j->ri.pDepthAttachment) j->ri.pDepthAttachment = &j->da;                                                        \
        if (j->ri.pStencilAttachment) j->ri.pStencilAttachment = &j->sa;                                                    \
        __atomic_store_n(&PAR.jstate[ji], 1, __ATOMIC_RELEASE);   /* taken at once: the chunk follows the walk */          \
        __atomic_store_n(&PAR.njobs, ++ji, __ATOMIC_RELEASE);                                                               \
    } while (0)
    ParJob *j;
    const CfgRenderState *settings = cfg_render_current();
    // GPU writes the replay records from here on come after the stretch: its chunks' lookups leave them out
    uint64_t asof_mark = vk_texture_asof_mark();
    uint32_t upd_mark = vk_exec_draw_upd_count(ex->fence), useq_mark = vk_usnap_update_serial();   // no update inside a stretch
    PAR_JOB_OUT(h);
    for (const op_hdr *o = h; o != e; o = nvn_op_next(o)) {
        if (ex->work_scope && (uintptr_t)o >= ex->work_scope_end) ex->work_scope = NULL;
        uint32_t op = o->op;
        if (par_is_draw(op)) {
            x_stream_pos++;
            vk_exec_frame.draws_seen++;
            PAR.slots[PAR.nslots++] = ex->render_disabled ? -1 : vk_exec_oq_scan_draw();
            in_chunk++;
            __atomic_store_n(&PAR.walked, nvn_op_next(o), __ATOMIC_RELEASE);
            continue;
        }
        // a chunk ends where no query is open once it has its share of the draws, or anywhere at twice its share
        if (k + 1 < want && in_chunk >= share && (!vk_exec_oq_scan_open() || in_chunk >= 2 * share)) {
            vk_exec_oq_scan_cut(ex);
            __atomic_store_n(&j->to, o, __ATOMIC_RELEASE);
            k++;
            PAR_JOB_OUT(o);
            in_chunk = 0;
        }
        // state and binds into the queue state, the counters' and conditions' bookkeeping in order
        if (op == OP_RESET_COUNTER || op == OP_REPORT_COUNTER || op == OP_RENDER_COND || op == OP_RENDER_ENABLE)
            vk_exec_op_xfer(ex, op, nvn_op_payload(o));
        else if (op == OP_BIND_UBO)
            bind_ubo_op(ex, o);
        else if (!state_op(ex, o))
            vk_exec_op_draw(ex, op, nvn_op_payload(o));
        if (op == OP_RENDER_COND || op == OP_RENDER_ENABLE) {
            ParCond *c = &PAR.conds[PAR.nconds++];
            c->disabled = (uint8_t)ex->render_disabled, c->has_cond = (uint8_t)ex->has_render_cond, c->rc = ex->render_cond;
            c->dbg_zero = (uint8_t)ex->dbg_zero;
#ifdef GAME_NS2
            c->gpu_cond = (uint8_t)ex->gpu_cond, c->cond_active = (uint8_t)ex->cond_active, c->cond_cpu = (uint8_t)ex->cond_cpu;
            c->off = ex->gpu_cond_off, c->flags = ex->gpu_cond_flags, c->buf = ex->gpu_cond_buf;
#endif
        }
    }
#undef PAR_JOB_OUT
    vk_exec_oq_scan_cut(ex);
    __atomic_store_n(&j->to, e, __ATOMIC_RELEASE);
    __atomic_store_n(&PAR.walked, e, __ATOMIC_RELEASE);
    k++;
    for (int i = k; i < want; i++) vkEndCommandBuffer(cbs[i]);   // not used by the walk: empty, but part of the batch
    vk_par_walk = 0;
    vk_exec_op_window++;                 // resets and reports end a stretch of ops that write no memory
    // The pass ends in the bookkeeping: targets written, the predicate copies of the stretch's reports, which go into
    // the new command buffer, behind the chunks.
    ex->fb_dirty = 0;
    pass_end(ex, 1);
    vk_exec_oq_scan(0);
    PAR.walking = 0;
    ex->recorded += (int)nd;             // the part is not empty
    uint64_t t1 = plat_now_ns();
    PAR.stretches++, PAR.chunks_n += (uint64_t)k, PAR.draws += nd, PAR.walk_ns += t1 - t0;
    if ((rt_trace & RT_TRACE_PERF) && PAR.stretches >= 8000) {
        rt_log("perf: draw stretches: %.1f draws in %.1f chunks each, the walk %.3f ms each; %.2f waits for the chunks per stretch, "
               "%.3f ms each", (double)PAR.draws / (double)PAR.stretches, (double)PAR.chunks_n / (double)PAR.stretches,
               PAR.walk_ns / 1e6 / (double)PAR.stretches, (double)PAR.joins / (double)PAR.stretches,
               PAR.joins ? PAR.join_ns / 1e6 / (double)PAR.joins : 0.0);
        for (int w = 0; w < 128; w++)   // DEBUG-PARWHY
            if (par_why_n[w])
                rt_log("perf: chunk waits asked by %s: %llu, %.3f ms each, %.1f chunks still out", w == 0 ? "a split" : w == 100 ? "the go" :
                       w == 101 ? "room" : nvn_op_name((uint32_t)w - 1), (unsigned long long)par_why_n[w],
                       par_why_ns[w] / 1e6 / (double)par_why_n[w], (double)par_why_open[w] / (double)par_why_n[w]);
        memset(par_why_n, 0, sizeof par_why_n), memset(par_why_ns, 0, sizeof par_why_ns), memset(par_why_open, 0, sizeof par_why_open);
        void vk_exec_draw_tchk_log(void);
        vk_exec_draw_tchk_log();
        PAR.stretches = PAR.chunks_n = PAR.draws = PAR.walk_ns = PAR.join_ns = PAR.joins = 0;
    }
    return e;
}

// Replays the op at h and returns the next one. *stop: 0 go on, 1 OP_END, 2 held at this wait (holdable: the
// submission may be held, "Queues interleaved" below).
static char heldlog[1024];   // DEBUG-HELDLOG
static int heldlog_n;
static const op_hdr *exec_op_(VkExec *ex, const op_hdr *h, int depth, int holdable, int *stop);
int vk_spike_on = -1;   // DEBUG-SPIKE: time per op kind
static const op_hdr *exec_op(VkExec *ex, const op_hdr *h, int depth, int holdable, int *stop) {
    if (vk_spike_on < 0) vk_spike_on = cfg_get("REKARINA_SPIKE_MS") != NULL;
    if (!vk_spike_on) return exec_op_(ex, h, depth, holdable, stop);
    uint32_t kind = h->op;
    uint64_t a = plat_now_ns();
    const op_hdr *r = exec_op_(ex, h, depth, holdable, stop);
    uint64_t d = plat_now_ns() - a;
    if (kind < 96) vk_spike.op[kind] += d;
    if (d > vk_spike.max_ns) vk_spike.max_ns = d, vk_spike.max_op = kind;
    return r;
}
static const op_hdr *exec_op_(VkExec *ex, const op_hdr *h, int depth, int holdable, int *stop) {
    *stop = 0;
    if (heldlog_n >= 0 && resuming && heldlog_n < (int)sizeof heldlog - 24)   // DEBUG-HELDLOG
        heldlog_n += snprintf(heldlog + heldlog_n, sizeof heldlog - (size_t)heldlog_n, " %s", nvn_op_name(h->op));
    {
        if (h->size < sizeof *h || (h->size & 7) || h->op >= OP_COUNT_)
            rt_fatal("vk: corrupt command record at %p: op %u size %u", (const void *)h, h->op, h->size);
        if (ex->work_scope && (uintptr_t)h >= ex->work_scope_end) ex->work_scope = NULL;
        const void *a = nvn_op_payload(h);
        vk_exec_cur_op = h;
        if (PAR.active && par_join_op(h->op)) par_why = 1 + (int)h->op, vk_exec_par_join();
        if (!depth && par_is_draw(h->op) && x_here && par_on() && !vk_interface_coverage_serial_required(ex)) {
            const op_hdr *after = par_try(ex, h);
            if (after) return after;
        }
        if (h->op != OP_JUMP && !vk_exec_op_quiet(h->op)) vk_exec_op_window++;
        switch (h->op) {
        case OP_END: ex->work_scope = NULL; *stop = 1; return h;
        case OP_BEGIN: ex->work_scope = NULL; break;
        case OP_JUMP: return (const op_hdr *)(uintptr_t)((const p_jump *)a)->target;
        case OP_CALL:
            if (depth >= CALL_DEPTH) rt_fatal("vk: command buffer calls nested deeper than %d", CALL_DEPTH);
            walk(ex, nvn_op_first(((const p_jump *)a)->target), depth + 1);
            break;
        case OP_SET_RT: set_rt(ex, a); break;
        case OP_CLEAR_COLOR: clear_color(ex, a); break;
        case OP_CLEAR_DS: clear_ds(ex, a); break;
        case OP_VIEWPORTS: case OP_SCISSOR: case OP_DEPTH_RANGE: case OP_STENCIL: case OP_POLY_OFFSET: case OP_BLEND_COLOR:
        case OP_DEPTH_CLAMP: case OP_RASTER_DISCARD: case OP_PRIM_RESTART: case OP_COPY_STRIDE: case OP_SET_TEX_POOL:
        case OP_SET_SAMP_POOL: case OP_MODEL_SCOPE:
            state_op(ex, h);
            break;
        case OP_SCENE_PROJECTION:
            vk_dlaa_scene_projection(a);
            break;
        case OP_BIND_UBO:
            bind_ubo_op(ex, h);
            break;
        case OP_BARRIER:
            game_barrier(ex, ((const p_barrier *)a)->bits);
            break;
        case OP_DRAW: case OP_DRAW_IDX: case OP_DISPATCH: case OP_UPDATE_UB: case OP_DRAW_IDX_IND: case OP_DISPATCH_IND:
            if (h->op == OP_DRAW || h->op == OP_DRAW_IDX || h->op == OP_DRAW_IDX_IND)
                tl_stream_pos = ++x_stream_pos, vk_exec_frame.draws_seen++, split_draws++;
            if (h->op == OP_DISPATCH || h->op == OP_DISPATCH_IND) vk_exec_frame.dispatches++, split_draws++;
        case OP_BIND_PROGRAM: case OP_BIND_STATE: case OP_BIND_VATTR: case OP_BIND_VSTREAM: case OP_BIND_VBUF:
        case OP_BIND_SSBO: case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG:
            vk_exec_op_draw(ex, h->op, a);
            break;
        case OP_COPY_B2B: case OP_COPY_B2T: case OP_COPY_T2B: case OP_COPY_T2T: case OP_CLEAR_TEX:
            if (h->op != OP_CLEAR_TEX) vk_exec_frame.copies++;
            if (vk_dump_armed) vk_dump_copy(ex, h->op, a);
            // fall through
        case OP_SIGNAL_EVENT: case OP_REPORT_COUNTER: case OP_RESET_COUNTER: case OP_RENDER_ENABLE: case OP_RENDER_COND:
        case OP_CLEAR_BUF:
            vk_exec_op_xfer(ex, h->op, a);
            break;
        case OP_WAIT_EVENT: {   // the CPU's late writes are in (or deferred: vk_exec_wait_resolve before the next part)
            x_holdable = holdable;
            int r = vk_exec_wait_event(ex, a);
            x_holdable = 0;
            if (r == 3) {   // its signal comes with a later submission of another queue: held here
                *stop = 2;
                return h;
            }
            if (r == 4) {   // met by a compute rest that runs beside (async compute): the work after it waits for that rest
                ac_graphics_waits(ex, vk_exec_wait_cp);
                break;
            }
            if (r && !vk_exec_ac_value) {
                par_why = 1 + OP_WAIT_EVENT, vk_exec_par_join();   // the chunks out read the snapshot and mirrors as they are
                vk_exec_copies_ns = plat_now_ns();   // the copies taken below are as of now (vk_exec_xfer.c wait_event_)
                // a held rest recorded inside another submission reads that submission's copies or live memory
                if (!resuming) vk_usnap_resnap(nvn_op_next(h), x_handles + x_cur + 1, x_n - x_cur - 1);
                vk_exec_gmem_unpin();   // what the CPU wrote before its signal may name a pool made since
                vk_mem_resync_after_wait();
                vk_exec_pass_end(ex);   // a part of its own, so the new copies run before the commands after the wait
                if (ex->recorded != part_start) {
                    ex->cb = vk_sync_split();
                    part_start = ex->recorded;
                    part_t0 = kern_now_ns();
                    ex->dirty = VK_EXEC_DIRTY_ALL;
                    vk_exec_draw_new_cb();
                    n_parts++;
                }
            }
            break;
        }
#ifdef GAME_NS2
        case OP_DRAW_MESH: case OP_DRAW_MESH_IND:   // NVN2 (vk_exec_draw.c)
            vk_exec_frame.draws_seen++, split_draws++;
            // fall through
        case OP_PUSH_CONST:
            vk_exec_op_draw(ex, h->op, a);
            break;
        case OP_FENCE_SYNC: vk_exec_op_xfer(ex, h->op, a); break;
        case OP_COPY_IMAGE_STRIDE: state_op(ex, h); break;
#endif
        default: rt_fatal("vk: op %s has no replay handler", nvn_op_name(h->op));
        }
        if (h->op == OP_SIGNAL_EVENT && nheld) held_check(ex);   // a held submission may go on after this signal
        if (split_draws >= split_n) maybe_split(ex);
        // Chunks of draw stretches are out and all recorded: the part goes to the GPU now, without the wait a split at
        // its usual place would mean while they are being recorded.
        if (PAR.active && __atomic_load_n(&PAR.ndone, __ATOMIC_ACQUIRE) == PAR.njobs && !vk_exec_ac_value &&
            !vk_exec_wait_deferred()) {
            vk_exec_pass_end(ex);   // an open pass begins again in the next part, as at any split inside a pass
            vk_exec_wait_resolve();
            par_why = 100;
            part_split(ex);
            split_draws = 0;
            wb_early(ex);
        }
        return nvn_op_next(h);
    }
}

static void walk(VkExec *ex, const op_hdr *h, int depth) {
    vk_exec_op_window++;
    for (;;) {
        int stop;
        const op_hdr *next = exec_op(ex, h, depth, 0, &stop);
        if (stop) return;
        h = next;
    }
}

// The results point of the frame's main submission (vk_sync.c "Results before the frame's end"): where the scene is
// drawn and only its scaling, the HUD and the copy to the window remain (vk_dlaa.c calls it at the game's scaling
// dispatch). Measured in OoT (REKARINA_WBWHO): shaders write CPU-visible memory up to the last post-processing
// dispatches before that point (the exposure's sums, culling records the game reads back, a few status words) and
// nothing after it, and the game reads those results right after its wait for the frame, so an earlier point (the
// first lighting pass was tried: 85 fps against 72, and a whole-frame brightness pulse every hundred frames) hands it
// last frame's values now and then. Here the part before the point goes first, then a part of its own with the
// write-back collected since the early one and the results signal; the occlusion slots read so far go back with it.
// REKARINA_RESULTS_FENCE=2 (test): the first pass into a scene-sized float target with the scene depth instead.
static int x_res_point;   // this submission has had its results point
static void part_split(VkExec *ex) {
    ex->cb = vk_sync_split();
    part_start = ex->recorded;
    part_t0 = kern_now_ns();
    ex->dirty = VK_EXEC_DIRTY_ALL;
    vk_exec_draw_new_cb();
    n_parts++;
}
void vk_exec_results_point(VkExec *ex) {
    if (vk_par_worker || x_res_point || !x_here || vk_exec_ac_value || !vk_sync_results_on()) return;
    {   // not while the GPU reads the game's live memory (vk_memory.c vk_mem_host_read_note); REKARINA_RESULTS_HOST=1: anyway
        static int anyway = -1;
        static uint64_t left_out, taken;
        if (anyway < 0) anyway = cfg_get("REKARINA_RESULTS_HOST") != NULL;
        int host = vk_mem_host_read_recent();
        if (host) left_out++;
        else taken++;
        if ((rt_trace & RT_TRACE_PERF) && !((left_out + taken) & 2047))
            rt_log("perf: results point: asked %llu times with every buffer in a mirror, %llu times while buffers were bound in host memory (left out)",
                   (unsigned long long)taken, (unsigned long long)left_out);
        if (host && !anyway) return;
    }
    x_res_point = 1;
    vk_exec_pass_end(ex);
    vk_exec_wait_resolve();
    if (ex->recorded != part_start) part_split(ex);   // every snapshot copy of the ranges collected is recorded now
    int rec = vk_mem_mirror_flush(ex->cb, ex->fence);
    ex->recorded += (uint64_t)rec;
    vk_exec_oq_submit_end(ex->fence);
    if (!vk_sync_results_point()) return;
    part_split(ex);   // the part with the write-back (empty when there was none) carries the signal
}
static void results_point(VkExec *ex) {
    static int mode = -1;
    // the same default as vk_sync.c's: read from the environment alone, the point was never taken without the variable
    // set, and the base picture (no DLSS, so no scaling dispatch to take it at) waited for the whole frame at each present
    // (rt_knob only says on or off: the number is read here)
    if (mode < 0) {
        const char *e = cfg_get("REKARINA_RESULTS_FENCE");
#ifdef GAME_OOT
        mode = e && *e ? atoi(e) : 2;
#else
        mode = e && *e ? atoi(e) : 0;
#endif
    }
    {   // DEBUG-RESWHY: why the point is or is not taken, every 8192 passes
        static uint64_t n, nmode, ndone, nres, nfire;
        n++, nmode += mode != 2, ndone += mode == 2 && x_res_point, nres += mode == 2 && !x_res_point && resuming;
        const VkExecTarget *c0 = &ex->color[0], *z0 = &ex->depth;
        int shape = ex->pass_ncolor == 1 && c0->image && z0->image && z0->format == VK_FORMAT_D32_SFLOAT && c0->width >= 960 &&
                    c0->width == z0->width && c0->height == z0->height &&
                    (c0->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || c0->format == VK_FORMAT_R16G16B16A16_SFLOAT);
        nfire += mode == 2 && !x_res_point && !resuming && shape;
        if (!(n & 8191))
            rt_log("RESWHY passes %llu: mode off %llu, point already sent %llu, resuming %llu, taken %llu; this pass: %d colours "
                   "%ux%u fmt %d depth fmt %d", (unsigned long long)n, (unsigned long long)nmode, (unsigned long long)ndone,
                   (unsigned long long)nres, (unsigned long long)nfire, ex->pass_ncolor, c0->width, c0->height, (int)c0->format,
                   (int)z0->format);
    }
    if (mode != 2 || x_res_point || resuming) return;
    const VkExecTarget *c = &ex->color[0], *z = &ex->depth;
    if (ex->pass_ncolor != 1 || !c->image || !z->image || z->format != VK_FORMAT_D32_SFLOAT || c->width < 960 ||
        c->width != z->width || c->height != z->height ||
        (c->format != VK_FORMAT_B10G11R11_UFLOAT_PACK32 && c->format != VK_FORMAT_R16G16B16A16_SFLOAT))
        return;
    vk_exec_results_point(ex);
}

// Replays handles[*i..n) from *at (NULL: the start of handles[*i]). 1: held at a wait (*i and *at say where), 0: done.
static int run(VkExec *ex, const uint64_t *handles, int n, int *i, const op_hdr **at, int holdable) {
    vk_exec_op_window++;
    for (; *i < n; ++*i, *at = NULL) {
        x_cur = *i;
        const op_hdr *h = *at ? *at : nvn_op_first(handles[*i]);
        for (;;) {
            int stop;
            const op_hdr *next = exec_op(ex, h, 0, holdable, &stop);
            if (stop == 2) {
                *at = h;
                return 1;
            }
            if (stop) break;
            h = next;
        }
    }
    return 0;
}

// ---- Queues interleaved on one Vulkan queue
// The console runs the game's NVN queues side by side and orders them with GPU event words: WaitEvent stops a queue
// until another queue's SignalEvent writes the word. The replay records submissions one after the other into one
// Vulkan queue, and passed over a wait whose signal a later submission of another queue records: the commands after
// it read what that queue wrote a frame earlier. OoT submits its compute queue (~400 commands: culling,
// light lists) before its graphics queue (~22,000); the compute queue waits for the graphics queue's early passes from
// its 89th command on, and the graphics queue for the compute results from its ~7,800th (REKARINA_EVLOG2,
// docs/oot-architecture.md). With the compute work a frame behind, what the camera turned to was culled against the
// last frame and missed a frame: objects and grass flickered in motion.
// So a submission that reaches a wait no recorded signal meets yet, for a word GPU work signals (vk_exec_xfer.c
// vk_exec_wait_event, 3), is held there: its recording so far is a submission of its own, and the rest waits with its
// handles, where it stopped and its queue state (VkExec is NVN queue state). When the submission being recorded
// records a signal that meets the held wait, the rest is recorded right there, in the same command buffer, between
// two full barriers, until it ends or reaches the next such wait. A held rest is recorded without holding (as before)
// when its queue submits again, when it is HELD_AGE submits old, when the CPU waits on a sync fenced after it
// (gfx_sync_fence) and at nvnQueueFinish. Its recording copies stay meanwhile (vk_backend.c gfx_replayed_submits).
// Synchronous replay only; REKARINA_QUEUE_INTERLEAVE=1 turns it on. Off by default: with it (forest path, 2026-11-08) every
// second frame's mesh lists were wrong (buffer A of the game's two per-frame buffers: counts of 4096, 1084, 1600 where
// the other frames have 244, 54, 208; 15,000 mesh groups instead of 5,000), which flickered foliage and its shadow.
#define HELD_MAX 4
#define HELD_AGE 64
typedef struct {
    int used;
    const void *queue;           // the NVNqueue it was submitted to
    uint64_t submit_no, fence;   // nvn_submit_count and timeline value of its submission
    uint64_t *handles;           // its command handles
    int n, i;                    // their count and the one it stopped in
    const op_hdr *at;            // the wait it stopped at
    int compute_only;            // its rest holds dispatches and their binds only (async compute)
    void *usnap;                 // its uniform snapshot (vk_usnap_hold)
    const CfgRenderState *settings;
    VkExec st;                   // its queue state there
} Held;
static Held held[HELD_MAX];
static int recheck, interleave = -1;
static _Thread_local int flush_req;                   // gfx_sync_fence's submission: every held rest goes
static volatile uint64_t held_min_submit, held_min_fence;   // the oldest held submission (0: none), for other threads
static struct { uint64_t from, done; } fmap[16];      // a rest held from `from` was recorded in `done`
static unsigned nfmap;
static PlatMutex fmap_lock = PLAT_MUTEX_INIT;
static struct { uint64_t holds, resumes, finished, flush_queue, flush_age, flush_sync, ac_rests, ac_waits; } hst;

int vk_exec_hold_ok(void) { return x_holdable; }
uint64_t vk_exec_held_submit(void) { return __atomic_load_n(&held_min_submit, __ATOMIC_ACQUIRE); }

static void held_note(void) {
    uint64_t ms = 0, mf = 0;
    for (int k = 0; k < HELD_MAX; k++) {
        if (!held[k].used) continue;
        if (!ms || held[k].submit_no < ms) ms = held[k].submit_no;
        if (!mf || held[k].fence < mf) mf = held[k].fence;
    }
    __atomic_store_n(&held_min_submit, ms, __ATOMIC_RELEASE);
    __atomic_store_n(&held_min_fence, mf, __ATOMIC_RELEASE);
}

// The queue state of the running submission and of a held one change places; what belongs to the command buffer being
// recorded stays (vk_exec.h VkExec: the "per command buffer" and "internal" fields). No pass is open.
static void ctx_swap(VkExec *ex, VkExec *other) {
    static VkExec cur;
    cur = *ex;
    *ex = *other;
    *other = cur;
    ex->cb = cur.cb, ex->fence = cur.fence, ex->recorded = cur.recorded, ex->in_pass = 0;
    ex->dirty = VK_EXEC_DIRTY_ALL;
    ex->mark = cur.mark, ex->mark_draws0 = cur.mark_draws0;
    ex->mark_passes = cur.mark_passes, ex->mark_dispatches = cur.mark_dispatches;
    ex->rt_resolved = 0;
    ex->views = cur.views, ex->nviews = cur.nviews, ex->cap_views = cur.cap_views;
    ex->barrier_at = cur.barrier_at, ex->pass_end_at = cur.pass_end_at;
    ex->dispatch_end_at = cur.dispatch_end_at, ex->relaxed_at = cur.relaxed_at;
    memcpy(ex->fbl_img, cur.fbl_img, sizeof ex->fbl_img), memcpy(ex->fbl_aspect, cur.fbl_aspect, sizeof ex->fbl_aspect);
    ex->nfbl = cur.nfbl, ex->fbl_mask = cur.fbl_mask, ex->fb_dirty = cur.fb_dirty;
    // the memos keyed by serials compare again (vk_exec_draw.c memo_match, feedback_scan)
    ex->fb_fs = NULL, ex->tex_serial = cur.tex_serial + 1, ex->pass_serial = cur.pass_serial + 1;
    vk_exec_draw_new_cb();
    vk_exec_op_window++;
}

static void held_add(VkExec *ex, const uint64_t *handles, int n, int i, const op_hdr *at) {
    int k = 0;
    while (k < HELD_MAX && held[k].used) k++;
    if (k == HELD_MAX) rt_fatal("vk: no slot for a held submission");
    Held *d = &held[k];
    if (!(d->handles = malloc(sizeof *d->handles * (size_t)n))) rt_fatal("vk: out of memory (held submission)");
    memcpy(d->handles, handles, sizeof *d->handles * (size_t)n);
    d->n = n, d->i = i, d->at = at;
    d->queue = nvn_submit_queue, d->submit_no = nvn_submit_count(), d->fence = ex->fence;
    d->st = *ex;
    d->usnap = vk_usnap_hold();
    d->settings = cfg_render_retain(cfg_render_current());
    d->compute_only = 1;   // async compute: every op of the rest must be one a compute queue records
    for (int k = i; k < n && d->compute_only; k++)
        for (const op_hdr *o = k == i ? at : nvn_op_first(handles[k]); o->op != OP_END; o = nvn_op_next(o)) {
            switch (o->op) {
            case OP_BEGIN: case OP_WAIT_EVENT: case OP_SIGNAL_EVENT: case OP_BIND_PROGRAM: case OP_BIND_UBO: case OP_BIND_SSBO:
            case OP_BIND_TEX: case OP_BIND_SAMP: case OP_BIND_IMG: case OP_SET_TEX_POOL: case OP_SET_SAMP_POOL:
            case OP_DISPATCH: case OP_DISPATCH_IND: case OP_BARRIER: case OP_UPDATE_UB:
#ifdef GAME_NS2
            case OP_PUSH_CONST:
#endif
                continue;
            default: d->compute_only = 0;
            }
            break;
        }
    d->used = 1, nheld++;
    hst.holds++;
    FT(FT_HOLD, d->fence, (uintptr_t)d->queue, 0);
    held_note();
}

// The graphics part about to be recorded waits for compute rest cp (async compute): the part so far goes first.
static void ac_graphics_waits(VkExec *ex, uint64_t cp) {
    vk_exec_pass_end(ex);
    if (ex->recorded != part_start) {
        ex->cb = vk_sync_split();
        part_start = ex->recorded;
        part_t0 = kern_now_ns();
        ex->dirty = VK_EXEC_DIRTY_ALL;
        vk_exec_draw_new_cb();
        n_parts++;
    }
    vk_sync_ac_need(cp);
    hst.ac_waits++;
}

// Records the rest of held[k] here (force: waits are passed over as before, no holding again).
static void held_resume(VkExec *ex, int k, int force) {
    Held *d = &held[k];
    // no wait for the chunks of draw stretches here: they read copies of the queue state, and the uniform snapshot
    // through a view of their own (vk_usnap_par_view), which the exchange below leaves alone
    vk_exec_pass_end(ex);
    vk_exec_barrier(ex);   // what was recorded before the signal is visible to the held commands
    // async compute: a rest of dispatches goes to the compute queue, after the graphics so far (vk_sync.c)
    int ac = !force && d->compute_only && vk_sync_ac_on();
    if (!ac && vk_sync_ac_last() > vk_sync_ac_covered()) ac_graphics_waits(ex, vk_sync_ac_last());
    ctx_swap(ex, &d->st);
    static VkExec g;   // (static: large; the replay is one thread) the graphics command buffer's own fields while the rest records into the compute one
    VkCommandBuffer ccb = VK_NULL_HANDLE;
    if (ac) {
        g = *ex;
        ccb = vk_sync_ac_begin(&vk_exec_ac_value);
        ex->cb = ccb;
        ex->barrier_at = ex->pass_end_at = ex->dispatch_end_at = ex->relaxed_at = -1;
        ex->nfbl = 0, ex->fbl_mask = ex->fb_dirty = 0, ex->mark = NULL;
        vk_exec_draw_new_cb();
        hst.ac_rests++;
    }
    const uint64_t *sh = x_handles;
    int sn = x_n, sc = x_cur;
    x_handles = d->handles, x_n = d->n;
    vk_usnap_held_enter(d->usnap);
    static int64_t hl_frame = -2;   // DEBUG-HELDLOG: REKARINA_HELDLOG=<frame> logs the ops of each held rest recorded then
    if (hl_frame == -2) { const char *e = cfg_get("REKARINA_HELDLOG"); hl_frame = e ? atoll(e) : -1; }
    uint64_t hl_f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    int hl = hl_frame >= 0 && hl_f >= (uint64_t)hl_frame && hl_f <= (uint64_t)hl_frame + 1;
    heldlog_n = hl ? 0 : -1, heldlog[0] = 0;
    int i0 = d->i;
    resuming++;
    const CfgRenderState *previous_settings = cfg_render_use(d->settings);
    FT(FT_RESUME0, d->fence, ex->fence, force);
    int again = run(ex, d->handles, d->n, &d->i, &d->at, !force);
    cfg_render_use(previous_settings);
    FT(FT_RESUME1, d->fence, ex->fence, again);
    resuming--;
    if (ac) {   // the graphics command buffer again; the part so far goes now and signals what the rest waits for
        vk_exec_pass_end(ex);
        ex->cb = g.cb, ex->recorded = g.recorded, ex->barrier_at = g.barrier_at, ex->pass_end_at = g.pass_end_at;
        ex->dispatch_end_at = g.dispatch_end_at, ex->relaxed_at = g.relaxed_at, ex->mark = g.mark;
        memcpy(ex->fbl_img, g.fbl_img, sizeof ex->fbl_img), memcpy(ex->fbl_aspect, g.fbl_aspect, sizeof ex->fbl_aspect);
        ex->nfbl = g.nfbl, ex->fbl_mask = g.fbl_mask, ex->fb_dirty = g.fb_dirty;
        vk_exec_ac_value = 0;
        uint64_t gp = 0;
        ex->cb = vk_sync_split_gp(&gp);
        part_start = ex->recorded;
        part_t0 = kern_now_ns();
        ex->dirty = VK_EXEC_DIRTY_ALL;
        vk_exec_draw_new_cb();
        n_parts++;
        vk_sync_ac_submit(ccb, gp);
    }
    if (hl)
        rt_log("HELDLOG frame %llu fence %llu held from %llu (handle %d -> %d of %d)%s%s:%s", (unsigned long long)hl_f,
               (unsigned long long)ex->fence, (unsigned long long)d->fence, i0, d->i, d->n, force ? " forced" : "",
               again ? " held again" : " done", heldlog);
    heldlog_n = -1;
    vk_usnap_held_leave(d->usnap);
    if (!ac) {
        vk_exec_pass_end(ex);
        vk_exec_barrier(ex);   // and what they wrote to the commands after
    }
    x_handles = sh, x_n = sn, x_cur = sc;
    ctx_swap(ex, &d->st);
    if (again) return;
    plat_mutex_lock(&fmap_lock);
    fmap[nfmap++ % 16] = (typeof(fmap[0])){d->fence, ex->fence};
    plat_mutex_unlock(&fmap_lock);
    vk_usnap_held_done(d->usnap, ex->fence);
    d->usnap = NULL;
    free(d->handles);
    d->handles = NULL, d->used = 0, nheld--;
    cfg_render_release(d->settings);
    d->settings = NULL;
    hst.finished++;
    held_note();
}

static void held_check(VkExec *ex) {
    if (resuming) { recheck = 1; return; }
    do {
        recheck = 0;
        for (int k = 0; k < HELD_MAX; k++)
            if (held[k].used && vk_exec_wait_ready(nvn_op_payload(held[k].at))) hst.resumes++, held_resume(ex, k, 0);
    } while (recheck && nheld);
}

// nvn_api_core.c nvnSyncWait, nvnQueueFinish: the timeline value that completes what a sync fenced at f covers. Held
// rests of submissions at or before f are recorded first, in a submission of their own; a rest recorded in a later
// submission moves f to that one.
void vk_exec_flush_req(int on) { flush_req = on; }   // the replay thread, around the submission vk_replay_flush_held queued
int vk_exec_held(void) { return __atomic_load_n(&held_min_fence, __ATOMIC_ACQUIRE) != 0; }

uint64_t gfx_sync_fence(uint64_t f) {
    if (interleave <= 0) return f;
    uint64_t mf = __atomic_load_n(&held_min_fence, __ATOMIC_ACQUIRE);
    if (mf && mf <= f) {
        if (gfx_replay_async()) {
            // With the asynchronous replay the empty submission takes its place in the replay queue (a value handed out
            // here, outside the queue's order, made the next present signal another value than it was queued with) and
            // this thread waits for it.
            vk_replay_flush_held();
        } else {
            vk_present_drain();
            flush_req = 1;
            vk_sync_submit(NULL, 0);
            flush_req = 0;
        }
    }
    plat_mutex_lock(&fmap_lock);
    for (int again = 1; again;) {
        again = 0;
        for (unsigned k = 0; k < 16 && k < nfmap; k++)
            if (fmap[k].from <= f && f < fmap[k].done) f = fmap[k].done, again = 1;
    }
    plat_mutex_unlock(&fmap_lock);
    return f;
}

// Submissions being recorded right now (the headroom thread allocates device memory only between them, vk_headroom.c).
volatile int32_t vk_exec_busy;
static int exec_record_(VkCommandBuffer cb, const uint64_t *handles, int n, uint64_t fence);
int vk_exec_record(VkCommandBuffer cb, const uint64_t *handles, int n, uint64_t fence) {
    const CfgRenderState *settings = cfg_render_current() ? NULL : cfg_render_frame_acquire();
    const CfgRenderState *previous = settings ? cfg_render_use(settings) : cfg_render_current();
    __atomic_fetch_add(&vk_exec_busy, 1, __ATOMIC_ACQ_REL);
    if (n) vk_dlaa_frame_settings();
    int r = exec_record_(cb, handles, n, fence);
    __atomic_fetch_sub(&vk_exec_busy, 1, __ATOMIC_ACQ_REL);
    cfg_render_use(previous);
    cfg_render_release(settings);
    return r;
}

// DEBUG-RUNLOG: REKARINA_RUNLOG=<frame>: for each submission of that frame with 200 draws or more, the runs of draws that no
// other kind of op interrupts (state and bind ops do not), and what interrupts them
static void runlog(const uint64_t *handles, int n) {
    static int64_t want = -1;
    if (want < 0) { const char *e = cfg_get("REKARINA_RUNLOG"); want = e ? atoll(e) : 0; }
    if (!want || __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) != (uint64_t)want) return;
    uint32_t hist[8] = {0}, dhist[8] = {0}, breaks[64] = {0}, run = 0, draws = 0, ops = 0, runs = 0;
    static const uint32_t edge[8] = {1, 4, 16, 64, 128, 256, 512, 0xffffffffu};
    for (int i = 0; i < n; i++)
        for (const op_hdr *h = nvn_op_first(handles[i]); h->op != OP_END; h = nvn_op_next(h)) {
            ops++;
            int draw = h->op == OP_DRAW || h->op == OP_DRAW_IDX;
#ifdef GAME_NS2
            draw |= h->op == OP_DRAW_MESH || h->op == OP_DRAW_MESH_IND;
#endif
            int simple = draw || h->op == OP_BEGIN || h->op == OP_VIEWPORTS || h->op == OP_SCISSOR || h->op == OP_DEPTH_RANGE ||
                         (h->op >= OP_BIND_PROGRAM && h->op <= OP_POLY_OFFSET) || h->op == OP_SET_TEX_POOL ||
                         h->op == OP_SET_SAMP_POOL || h->op == OP_BLEND_COLOR || h->op == OP_BIND_IMG || h->op == OP_DEPTH_CLAMP ||
                         h->op == OP_RASTER_DISCARD;
#ifdef GAME_NS2
            simple |= h->op == OP_PUSH_CONST;
#endif
            if (draw) run++, draws++;
            if (!simple) {
                if (run) {
                    int k = 0;
                    while (run > edge[k]) k++;
                    hist[k]++, dhist[k] += run, runs++;
                }
                run = 0;
                if (h->op < 64) breaks[h->op]++;
            }
        }
    if (draws < 200) return;
    char line[512];
    int o = 0;
    for (int k = 0; k < 8; k++) o += snprintf(line + o, sizeof line - (size_t)o, " <=%u: %u runs %u draws;", edge[k], hist[k], dhist[k]);
    rt_log("RUNLOG %u ops, %u draws in %u runs:%s", ops, draws, runs, line);
    o = 0;
    for (int k = 0; k < 64; k++)
        if (breaks[k]) o += snprintf(line + o, sizeof line - (size_t)o, " op %d: %u;", k, breaks[k]);
    rt_log("RUNLOG interrupted by:%s", line);
}

static int exec_record_(VkCommandBuffer cb, const uint64_t *handles, int n, uint64_t fence) {
    runlog(handles, n);
    VkExec *ex = &X;
    if (!x_ready) state_reset(ex), vk_dump_init(), x_ready = 1;
#ifdef GAME_OOT
    const CfgRenderValues *settings = cfg_render_values();
    ex->motion_models = settings && settings->motion_models;
    ex->motion_reverse_only = settings && settings->motion_reverse_only;
    ex->prebloom_tonemap = vk_dlaa_prebloom_enabled() && !vk_dlaa_fsr1();
    ex->interface_coverage = settings && (settings->consumers & CFG_RENDER_INTERFACE_COVERAGE);
#endif
    if (interleave < 0) {
        interleave = rt_knob("QUEUE_INTERLEAVE", "1");
        rt_log("vk: queues interleaved at their event waits: %s (REKARINA_QUEUE_INTERLEAVE)", interleave ? "on" : "off");
    }
    const tex_t *keep[9 * (1 + HELD_MAX)];   // the render targets carry over (set_rt compares them by pointer)
    int nkeep = 0;
    for (int k = -1; k < HELD_MAX; k++) {
        const VkExec *s = k < 0 ? ex : held[k].used ? &held[k].st : NULL;
        if (!s) continue;
        memcpy(keep + nkeep, s->rt.color, sizeof s->rt.color);
        keep[nkeep + 8] = s->rt.depth;
        nkeep += 9;
    }
    nvn_pins_sweep(nvn_submit_count(), keep, nkeep);
    ex->cb = cb;
    ex->fence = fence;
    ex->recorded = 0;
    ex->in_pass = 0;
    ex->mark = NULL;
    ex->mark_passes = ex->mark_dispatches = 0;
    ex->dirty = VK_EXEC_DIRTY_ALL;
    ex->rt_resolved = 0;   // images may have changed since the last command buffer
    ex->nviews = 0;
    ex->barrier_at = -1;
    ex->pass_end_at = -1;
    ex->dispatch_end_at = -1;
    ex->relaxed_at = -1;
    ex->nfbl = 0;   // the previous command buffer ended with a full barrier: every image is in GENERAL
    vk_exec_frame.submits++;
    split_draws = part_start = 0;
    part_t0 = kern_now_ns();
    PAR.no_from = PAR.no_until = NULL;   // the ops of another submission
    x_here = 1;
    x_res_point = 0;
    int work = n || (flush_req && nheld);   // a submission of the game's, or gfx_sync_fence's with held rests
    if (work && n) vk_exec_prewait(handles, n);   // the CPU's late writes are in before the copies are taken
    if (work && n && vk_dump_armed) vk_dump_arm_clears(handles, n);   // DEBUG-DUMPCLEARS
    {   // DEBUG-SPIKE
        uint64_t a = plat_now_ns();
        vk_exec_copies_ns = a;   // the uniform snapshot and the mirrors' syncs of this submission begin here
        if (work && n) vk_dlaa_scene_jitter(ex, handles, n);
        if (work) {
            // REKARINA_PRESNAP_MIN_KB (default 512; 0: always): a submission whose recordings hold fewer KiB than this
            // skips the mirrors' batched pre-snapshot (vk_memory.c vk_mem_mirror_begin_ex). At the Market's square the
            // main thread's host time on the game's cycle fell from 1.00 to 0.22 ms a cycle with it (critical-path
            // recorder, 2026-11-18), the replay's per-texture checks taking ~0.4 ms of what the batch did.
            static int64_t pmin = -1;
            if (pmin < 0) {
                const char *e = cfg_get("REKARINA_PRESNAP_MIN_KB");
                pmin = (int64_t)(e && *e ? atoll(e) : 512) << 10;
            }
            uint64_t bytes = 0;
            for (int i = 0; pmin > 0 && i < n && bytes < (uint64_t)pmin; i++) bytes += nvn_handle_bytes(handles[i]);
            vk_mem_mirror_begin_ex(pmin <= 0 || bytes >= (uint64_t)pmin);
        }
        uint64_t b = plat_now_ns();
        if (work) vk_usnap_begin(handles, n, fence);
        vk_spike.mirb += b - a, vk_spike.usnap += plat_now_ns() - b;
        vk_replay_captured();   // asynchronous replay: nvnQueueSubmitCommands returns to the game from here on
    }
    for (int k = 0; k < HELD_MAX; k++) {   // held rests that wait no longer go first
        if (!held[k].used) continue;
        if (flush_req) hst.flush_sync++, held_resume(ex, k, 1);
        else if (n && held[k].queue == nvn_submit_queue) hst.flush_queue++, held_resume(ex, k, 1);
        else if (n && nvn_submit_count() - held[k].submit_no > HELD_AGE) hst.flush_age++, held_resume(ex, k, 1);
    }
    if (work) vk_tpool_pump();
    x_handles = handles, x_n = n;
    for (int i = 0; i < n; i++)
        if (!handles[i]) rt_fatal("vk: command handle %d of a submission is null", i);
    {
        int i = 0;
        const op_hdr *at = NULL;
        if (n && run(ex, handles, n, &i, &at, interleave > 0 && nheld < HELD_MAX)) held_add(ex, handles, n, i, at);
    }
    x_handles = NULL, x_n = x_cur = 0;
    if (n && hst.holds && !(n_subs % 3600)) {
        rt_log("vk: queue interleave: %llu submissions held at a wait, %llu resumed by a signal, %llu finished there; "
               "recorded without waiting: %llu at their queue's next submit, %llu by age, %llu for a sync; async compute: "
               "%llu rests on the compute queue, %llu graphics waits for them",
               (unsigned long long)hst.holds, (unsigned long long)hst.resumes, (unsigned long long)hst.finished,
               (unsigned long long)hst.flush_queue, (unsigned long long)hst.flush_age, (unsigned long long)hst.flush_sync,
               (unsigned long long)hst.ac_rests, (unsigned long long)hst.ac_waits);
        memset(&hst, 0, sizeof hst);
    }
    {   // DEBUG-EVLOG2: where each submission of the logged frames ends
        static int on = -1;
        static uint64_t f0 = 900;
        if (on < 0) { const char *e = cfg_get("REKARINA_EVLOG2"); on = e != NULL; if (e && atoll(e) > 1) f0 = (uint64_t)atoll(e); }
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (on && f >= f0 && f <= f0 + 5)
            rt_log("EVLOG2 end fence %llu frame %llu rec %llu pass %u", (unsigned long long)fence, (unsigned long long)f,
                   (unsigned long long)ex->recorded, (unsigned)ex->pass_serial);
    }
    vk_exec_wait_resolve();   // a deferred wait for the CPU: before the last part goes
    vk_exec_pass_end(ex);
    if (vk_sync_chain_pending()) part_split(ex);   // chunks of draw stretches: the last part is one command buffer
    vk_exec_par_join();
    vk_exec_oq_submit_end(fence);
    // async compute: the write-back below reads what the compute rests wrote, and the fence says they ran
    if (work && vk_sync_ac_last() > vk_sync_ac_covered()) ac_graphics_waits(ex, vk_sync_ac_last());
    x_here = 0;
    barrier_game_now = 1;
    if (ex->recorded) vk_exec_barrier(ex);   // later submissions see this one's writes
    barrier_game_now = 0;
    if (work) ex->recorded += vk_texture_fence_readback(ex->cb, fence);   // small targets back to storage at the fence
    if (work) {   // VRAM mirror bytes the GPU wrote go back to guest memory
        VkPassMark *wm = vk_pass_mark_begin(ex->cb, VK_MARK_WRITEBACK, fence);
        uint64_t spk_a = plat_now_ns();   // DEBUG-SPIKE
        ex->recorded += vk_mem_mirror_end(ex->cb, fence), vk_usnap_end(fence);
        vk_spike.mire += plat_now_ns() - spk_a;
        vk_pass_mark_end(ex->cb, wm);
    }
    if (ex->nviews) {
        ViewList *l = malloc(sizeof *l + sizeof(VkImageView) * (size_t)ex->nviews);
        if (!l) rt_fatal("vk: out of memory");
        l->n = ex->nviews;
        for (int i = 0; i < ex->nviews; i++) l->v[i] = ((ViewEnt *)ex->views)[i].view;
        vk_completion_add(fence, free_views, l);
        ex->nviews = 0;
    }
    ex->cb = VK_NULL_HANDLE;
    if (work) n_subs++;
    if ((rt_trace & RT_TRACE_PERF) && n_subs >= 240) {
        rt_log("perf: replay command buffer parts per submission %.2f", (double)(n_subs + n_parts) / (double)n_subs);
        n_subs = n_parts = 0;
    }
    return ex->recorded - part_start;   // what the last part holds
}
