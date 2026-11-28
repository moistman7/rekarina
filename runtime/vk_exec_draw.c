// Draws, binds and dispatches of the Vulkan op-stream replay: the strong
// vk_exec_op_draw behind vk_exec.c's weak hook. The bind ops fill the VkExec slots; a draw opens the pass, resolves the
// bound program to translated shaders (the translator's answers come from the bound state), asks vk_pipeline.c for the
// pipeline, pushes (or allocates) set 0 as VkPipeInfo.bindings says and records the dynamic state, vertex and index
// buffers and the draw. A dispatch ends the pass and does the same for the compute shader. UPDATE_UB is a buffer copy
// from transient memory between full barriers.
//
// Per command buffer the replay keeps transient memory (support buffers, copied uniform ranges, generated indices),
// descriptor pools, image views and the storage-buffer ranges the GPU may write; all of it is freed, and the ranges are
// reported with vk_gpu_written, from one completion callback once the submission's fence is done (the write watch
// does not see GPU writes). Nothing here locks: it runs on the submitting thread under vk_sync_submit's lock.
//
// Provenance: the op semantics (bind ops, program and slot conventions, draw and dispatch flow, vertex streams with a
// zeroed buffer for unbound ones) follow MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m:7409-7460,
// 9365-9488, 9489-9921, 10013-10079 @ b75be8cf (MIT, Copyright (c) 2026 jhaizhou); the uniform-buffer spans, texture and
// sampler descriptor lookups and the dispatch flow follow LAS Recomp runtime/nvn/vulkan/exec.c:1851-1908, 2300-2516
// @ da35b83a (MIT, Copyright (c) 2026 Peiikai). The support buffer layout is Ryujinx's SupportBuffer.cs (MIT).
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "tex.h"
#include "vk.h"
#include "vk_exec.h"
#include "platform/platform.h"
#include "vk_dlaa.h"
#ifdef GAME_OOT
#include "layout.h"
#endif

#define MAX_BINDINGS 128
#define MAX_DESC 1024
#define CHUNK_BYTES (4u << 20)
#define ZERO_BYTES 65536u
// Ryujinx SupportBuffer (std140): alpha_test, is_bgra[8], viewport_inverse, viewport_size, ...; 1408 pcss and 1424
// shadow_spread (translator patches 0010, 0011), then the HDR tone mapper's VK_HDR_PARAMS vec4 (vk_hdr.c vk_hdr_support)
#define SUPPORT_BYTES (VK_HDR_PARAMS_AT + 16u * VK_HDR_PARAMS)
#define NVN_STAGE_VS 0
#define NVN_STAGE_FS 1
#define NVN_STAGE_CS 5
#ifdef GAME_NS2
// NVN2's geometry stage and the replay's slots of its task and mesh stages (nvn_ctl.h nvn_stage_slot)
#define NVN_STAGE_GS 2
#define NVN_STAGE_TS NVN2_SLOT_TASK
#define NVN_STAGE_MS NVN2_SLOT_MESH
#define C0_SLOTS 6                       // a c0 bank per stage slot
#define C0_K(st) (st)
#else
#define C0_SLOTS 3                       // vertex, fragment, compute
#define C0_K(st) ((st) == NVN_STAGE_VS ? 0 : (st) == NVN_STAGE_FS ? 1 : 2)
#endif
#define ONESHOT_WAIT_MS 30000u   // prepare_draw: how long a one-shot draw waits for its pipeline before it is skipped

enum { W_PROGRAM, W_SHADER, W_PIPELINE, W_NO_VS, W_INDEX, W_UB, W_BIND, W_SSBO_ALIGN, W_SSBO_SLOT, W_RESTART, W_INDIRECT, W_COUNT };

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

// What the replay leaves out while a shader is translated or a pipeline built beside it. Counted per frame for the
// flight recorder (mesh_skip, cs_skip in frames.tsv; the other draws are draws_skip_shader and draws_skip_pipe); the
// first of each kind in a frame gets a log line naming the program, at most 400 in a session. A frame where the
// game's grass or its culling dispatch is left out shows the grass missing: these lines say which frames.
enum { LO_MESH_SHADER, LO_MESH_PIPE, LO_CS_SHADER, LO_CS_PIPE, LO_N };
static void left_out(int kind, uint64_t prog, const GfxShader *sh) {
    perf_count(kind <= LO_MESH_PIPE ? PERF_CNT_MESH_SKIP : PERF_CNT_CS_SKIP);
    static volatile uint64_t last[LO_N];
    static volatile int32_t logs;
    uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    if (__atomic_exchange_n(&last[kind], f, __ATOMIC_RELAXED) == f || __atomic_fetch_add(&logs, 1, __ATOMIC_RELAXED) >= 400)
        return;
    static const char *const what[LO_N] = {"mesh or geometry draw: a shader is being translated",
                                           "mesh or geometry draw: its pipeline is being built",
                                           "dispatch: its shader is being translated",
                                           "dispatch: its pipeline is being built"};
    char hx[17] = "-";
    if (sh) {
        const uint8_t *k8 = gfx_shader_key(sh);
        for (int j = 0; j < 8; j++) snprintf(hx + 2 * j, 3, "%02x", k8[j]);
    }
    rt_log("vk_exec: frame %llu: left out: a %s (program %llu, shader %s)", (unsigned long long)f, what[kind],
           (unsigned long long)prog, hx);
}

static void *grow(void *p, int *cap, size_t elem) {
    *cap = *cap ? *cap * 2 : 8;
    p = realloc(p, elem * (size_t)*cap);
    if (!p) rt_fatal("vk: out of memory");
    return p;
}

static void vk_exec_slot_cache_drop(void);   // "descriptors kept per bound slot": none outlives a command buffer

// ---- resources of one command buffer, freed by its completion callback

typedef struct { VkBuffer buf; VmaAllocation alloc; uint8_t *map; uint64_t size, used; } Chunk;
// The view cache's key: no padding (the bytes are hashed and compared whole), and the image's serial beside its handle,
// so a view of a destroyed image is never the answer for a later image the driver gives the same handle.
typedef struct {
    VkImage image;
    VkFormat format;
    VkImageViewType type;
    uint32_t aspect, level0, levels, layer0, layers, swz, serial, usage;
} ViewKey;
_Static_assert(sizeof(ViewKey) == 48, "ViewKey is hashed as bytes: no padding");
typedef struct { uint64_t addr, size; int owned; } Range;   // owned: a mirror wrote it and copies it back whole
typedef struct {
    Chunk *chunks;
    int nchunks, cchunks;
    VkDescriptorPool *pools;
    int npools, cpools;
    Range *written;
    int nwritten, cwritten;
} CbRes;

static _Thread_local CbRes *cur;
static _Thread_local uint64_t cur_fence;

// Descriptor pools and full-size transient chunks go back to these lists when their command buffer completes, instead
// of being destroyed and created again for the next one: each creation is a driver memory allocation (on Windows an
// allocation, a map and a paging-fence wait on the CPU), about ten of them per frame inside the replay.
#define SPARE_POOLS 64
#define SPARE_CHUNKS 64                  // each thread that records a chunk of a stretch has its own
static PlatMutex spare_lock = PLAT_MUTEX_INIT;
static VkDescriptorPool spare_pool[SPARE_POOLS];
static Chunk spare_chunk[SPARE_CHUNKS];
static int nspare_pool, nspare_chunk;

// State of the command buffer being recorded that the Vulkan command buffer itself holds.
static _Thread_local struct {
    VkPipeline pipe[2];                  // bound per bind point: 0 graphics, 1 compute
    VkPipeState st;                      // the state vk_pipeline_set_dynamic last recorded
    int st_valid;
    VkBuffer vb[17];                     // vertex buffers last bound
    VkDeviceSize vo[17], vsz[17], vstr[17];
    int vcls[17];                        // their pool class and pool record, for the census
    void *vpool[17];
    NVNbufferRange vin[16];              // the ranges they were resolved from, under vk_mem_gen vgen
    uint64_t vgen;
    int vb_valid;
    uint8_t support[SUPPORT_BYTES];      // last support buffer and where it lives
    float sup_in[3];                     // the support_fill arguments it was filled from
    VkBuffer sup_buf;
    VkDeviceSize sup_off;
    int sup_valid;
    uint8_t c0[C0_SLOTS][VK_C0_BYTES];   // last c0 bank per stage (C0_K: vertex, fragment, compute; NVN2: per slot)
    VkBuffer c0_buf[C0_SLOTS];           // and where it lives
    VkDeviceSize c0_off[C0_SLOTS];
    int c0_valid[C0_SLOTS];
    struct { p_bind_buf ssbo[16], ubo[16]; uint32_t dim[3], binst; } c0_in[C0_SLOTS];   // what vk_c0_fill read for c0[k]
    VkPipelineLayout ds_layout;          // graphics layout the sets below were last pushed or bound with
    VkPipelineLayout pool_layout[2];     // per bind point: the layout the texture pool's set is bound with (NULL: not bound)
    VkDescriptorSet ds_bound[2];         // allocated set bound per set number (0: none or pushed since)
    VkDescriptorSet ds_last[2];          // last allocated and written set per set number, contents in DS
    VkDescriptorSetLayout ds_last_layout[2];
    int ds_pushed[2];                    // set number holds the push of DS.psig under ds_layout
    int res_valid;                       // the graphics sets bound hold what the inputs in RS give (bind_resources)
} B;

// Contents of B.ds_last: the set's packed descriptors (VkDescSlot), which with its layout determine every write. A draw
// whose allocated set would hold the same reuses the set instead of allocating and writing one (a per-draw
// cost). psig: the same for the last push per set number; a push equal to it under the same layout is not recorded
// again (the descriptors stay bound across pipeline binds of that layout).
#define DS_SIG_BYTES (MAX_DESC * sizeof(VkDescSlot))
static _Thread_local struct { uint32_t n[2], pn[2]; uint8_t sig[2][DS_SIG_BYTES], psig[2][DS_SIG_BYTES]; } DS;

// REKARINA_BIND_STATS counters (bind_stats).
static _Thread_local struct {
    int on;
    uint64_t draws, bind[2], desc[2], push, push_skip, set_new, set_reuse, ac_miss, same_prog, same_all, bind_skip;
} BS = {-1};

// The inputs of the last graphics bind_resources (bind_sig) and what its build counted for the traffic census, which a
// skipped build counts again. RS_BYTES bounds the bound-slot bytes a pipeline reads (RsList): per binding at most one
// c0 storage table, otherwise one bound range or handle per descriptor.
#define RS_BYTES (MAX_BINDINGS * sizeof(((VkExec *)0)->ssbo[0]) + MAX_DESC * sizeof(p_bind_buf))
typedef struct { int cls, use; uint64_t bytes; } RsCount;
static _Thread_local struct {
    int on;                              // REKARINA_BIND_SKIP: 1 skip (default), 2 check, 0 off
    uint32_t hn;                         // words of h the last signature filled; 0: none
    uint64_t h[25];                      // generations, the lookup limit, feedback images, support inputs, base instance
    const VkPipeInfo *info;              // the pipeline whose RsList ranges body holds, as of bind op generation gen
    uint64_t gen;
    uint8_t body[RS_BYTES];
    int rec;                             // a graphics build is running: count_use records, scratch copies mark volatile
    int volatile_;                       // the build copied guest bytes into transient memory: a rebuild differs
    RsCount cnt[2 * VK_MCLS_N], pcnt[2 * VK_MCLS_N];
    int ncnt, pncnt, cnt_over;
    int cmds;                            // check mode: pushes, binds and set writes the build recorded
    uint32_t ckn;                        // check mode: the last build's packed descriptors (bytes in ck)
    uint8_t ck[DS_SIG_BYTES];
} RS = {-1};

// The bound-slot bytes of VkExec a pipeline's descriptors read, as ranges merged where they touch, built once per
// pipeline info: c0's storage table, the uniform and storage ranges of the bound slots and the texture and image
// handles. sup and c0: the support inputs and the base instance count too. p: storage buffers whose address the shader
// loads from a user uniform block; the address read from the block counts too. ok 0: never skipped (too many
// descriptors or such buffers).
#define RS_PTRS 4
typedef struct { uint32_t off, len; } RsRange;
typedef struct { int st; const GfxShaderRes *res; } RsPtr;
typedef struct { uint32_t n, bytes; uint8_t sup, c0, ok, np; RsPtr p[RS_PTRS]; RsRange r[]; } RsList;
static _Thread_local nvn_hmap rs_lists;                // VkPipeInfo pointer -> RsList, until shutdown
static _Thread_local const VkPipeInfo *rs_last;        // the last lookup
static _Thread_local const RsList *rs_last_l;

static _Thread_local uint32_t c0_dim[3], c0_binst;     // the dispatch size and base instance of the draw or dispatch being bound
#ifdef GAME_NS2
static _Thread_local uint32_t c0_mesh_first;           // the first workgroup of the mesh draw being bound (VK_C0_MESH_FIRST)
#endif
static _Thread_local uint64_t bind_gen = 1;            // bumped by every bind op that writes ex->prog, ubo, ssbo, tex or img

static void cb_done(void *arg) {   // completion thread
    CbRes *r = arg;
    VkCtx *c = vk_ctx();
    for (int i = 0; i < r->nwritten; i++)
        if (r->written[i].owned) vk_gpu_written_mirror(r->written[i].addr, r->written[i].size);
        else vk_gpu_written(r->written[i].addr, r->written[i].size);
    for (int i = 0; i < r->npools; i++) {
        vkResetDescriptorPool(c->device, r->pools[i], 0);
        plat_mutex_lock(&spare_lock);
        int kept = nspare_pool < SPARE_POOLS;
        if (kept) spare_pool[nspare_pool++] = r->pools[i];
        plat_mutex_unlock(&spare_lock);
        if (!kept) vkDestroyDescriptorPool(c->device, r->pools[i], NULL);
    }
    for (int i = 0; i < r->nchunks; i++) {
        plat_mutex_lock(&spare_lock);
        int kept = r->chunks[i].size == CHUNK_BYTES && nspare_chunk < SPARE_CHUNKS;
        if (kept) spare_chunk[nspare_chunk++] = r->chunks[i];
        plat_mutex_unlock(&spare_lock);
        if (!kept) vmaDestroyBuffer(c->vma, r->chunks[i].buf, r->chunks[i].alloc);
    }
    free(r->written), free(r->pools), free(r->chunks), free(r);
}

// bind_gen: a chunk of a draw stretch starts from a copy of the walk's state (vk_exec.c par_run), so the bound slots
// this thread last compared (RS.body) are another chunk's, changed by bind ops this thread never ran. Without the bump
// the chunk's first draw took them for current and left them stale, and a later draw whose slots equalled the stale
// ones skipped its build and drew with the first draw's sets: another object's textures and uniforms for that draw.
void vk_exec_draw_new_cb(void) { memset(&B, 0, sizeof B); vk_exec_slot_cache_drop(); bind_gen++; }

static void dummies_init(void);
static void limits_init(void);
#ifdef GAME_NS2
static VkDescriptorBufferInfo gmem_info(VkExec *ex, int which);
static VkDescriptorBufferInfo gmem_par[2];
#endif
static CbRes *cb_res(VkExec *ex);
void vk_exec_draw_par_prepare(VkExec *ex) {
    cb_res(ex);
    dummies_init();
    limits_init();
    vk_tpool_prepare();
    vk_usnap_par_view();
#ifdef GAME_NS2
    gmem_par[0] = gmem_info(ex, 0), gmem_par[1] = gmem_info(ex, 1);
#endif
}

void vk_exec_draw_shaders(const GfxShader **vs, const GfxShader **fs) {
    *vs = B.st_valid ? B.st.vs : NULL;
    *fs = B.st_valid ? B.st.fs : NULL;
}

// Marks a dispatch about to be recorded (vk.h vk_pass_mark_begin); the caller ends it after the dispatch.
static VkPassMark *dispatch_mark(VkExec *ex, const GfxShader *cs, uint32_t x, uint32_t y, uint32_t z) {
    VkPassMark *m = vk_pass_mark_begin(ex->cb, VK_MARK_DISPATCH, ex->fence);
    if (!m) return NULL;
    m->step = ex->mark_dispatches++;
    m->w = x, m->h = y, m->d = z, m->draws = 0, m->ncolor = 0;
    m->first[0] = cs, m->first[1] = m->last[0] = m->last[1] = NULL;
    return m;
}

static CbRes *cb_res(VkExec *ex) {
    if (cur && cur_fence == ex->fence) return cur;
    cur = calloc(1, sizeof *cur);
    if (!cur) rt_fatal("vk: out of memory");
    cur_fence = ex->fence;
    memset(&B, 0, sizeof B);
    vk_completion_add(ex->fence, cb_done, cur);
    return cur;
}

static uint8_t *scratch(VkExec *ex, uint64_t size, VkBuffer *buf, VkDeviceSize *off) {
    CbRes *r = cb_res(ex);
    size = (size + 255) & ~255ull;
    Chunk *c = r->nchunks ? &r->chunks[r->nchunks - 1] : NULL;
    if (!c || c->used + size > c->size) {
        if (r->nchunks == r->cchunks) r->chunks = grow(r->chunks, &r->cchunks, sizeof *r->chunks);
        c = &r->chunks[r->nchunks];
        uint64_t n = size > CHUNK_BYTES ? size : CHUNK_BYTES;
        int reused = 0;
        if (n == CHUNK_BYTES) {
            plat_mutex_lock(&spare_lock);
            if (nspare_chunk) *c = spare_chunk[--nspare_chunk], reused = 1;
            plat_mutex_unlock(&spare_lock);
        }
        if (reused) {
            c->used = 0;
            r->nchunks++;
            *buf = c->buf;
            *off = 0;
            c->used = size;
            return c->map;
        }
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = n,
                                 .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                          VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
#ifdef GAME_NS2
        bi.usage |= VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;   // NVN2 indirect mesh draws read their arguments from here
#endif
        // HOST_COHERENT: the CPU fills these chunks and never flushes them. Intel's integrated GPUs (ANV, Meteor Lake)
        // offer host-cached memory only without coherence, which VMA picks for random host access: the GPU then read
        // what an earlier submission left in the chunk (an UpdateUniformBuffer payload of another draw: a lighting
        // shader's step count of 0x3f800000 instead of 128, and a loop of a billion steps).
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                      .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
        VmaAllocationInfo info;
        VkResult res = vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &c->buf, &c->alloc, &info);
        if (res != VK_SUCCESS) rt_fatal("vk: transient buffer of %llu bytes failed (%d)", (unsigned long long)n, (int)res);
        c->map = info.pMappedData, c->size = n, c->used = 0;
        r->nchunks++;
    }
    *buf = c->buf;
    *off = c->used;
    uint8_t *p = c->map + c->used;
    c->used += size;
    return p;
}

static _Thread_local const GfxShader *gl_cur_vs;   // DEBUG-GRASSLOG: the vertex shader of the draw prepare_draw set up last
static void grass_log(VkExec *ex, const char *kind, int count, int inst, int first, int basev, uint64_t ibuf, int isz,
                      uint64_t ind);   // DEBUG-GRASSLOG
#ifdef GAME_NS2   // draws and dispatches under a GPU condition (vk_exec_xfer.c "GPU conditions")
#define GPU_COND_BEGIN(ex) vk_exec_cond_begin(ex)
#define GPU_COND_END(ex) vk_exec_cond_end(ex)
#else
#define GPU_COND_BEGIN(ex) ((void)0)
#define GPU_COND_END(ex) ((void)0)
#endif

void vk_mem_shadow_fadd(uint64_t addr, uint64_t size);   // vk_memory.c

static void note_written(VkExec *ex, uint64_t addr, uint64_t size, int owned, int fadd) {
    CbRes *r = cb_res(ex);
    for (int i = 0; i < r->nwritten; i++)
        if (r->written[i].addr == addr && r->written[i].size == size) {
            r->written[i].owned &= owned;
            return;
        }
    if (r->nwritten == r->cwritten) r->written = grow(r->written, &r->cwritten, sizeof *r->written);
    r->written[r->nwritten++] = (Range){addr, size, owned};
    plat_mutex_lock(&vk_par_lock);   // a chunk of a draw stretch and the replay thread may both note one
    vk_mem_gpu_pending(addr, size);
    if (fadd) vk_mem_shadow_fadd(addr, size);
    plat_mutex_unlock(&vk_par_lock);
}

// ---- stand-ins for what a binding names but the game did not bind (zeros, a 1x1 image, the default sampler)

enum { DV_1D, DV_1D_ARRAY, DV_2D, DV_2D_ARRAY, DV_3D, DV_CUBE, DV_CUBE_ARRAY, DV_2D_MS, DV_2D_MS_ARRAY, DV_COUNT };

static struct {
    int ready;
    VkBuffer zbuf;
    VmaAllocation zalloc;
    VkBufferView zview;
    VkSampler samp, samp_shadow;
    struct { VkImage image; VmaAllocation alloc; VkImageView view, sview; } img[DV_COUNT];
} Z;

static const struct { VkImageType t; VkImageViewType vt; uint32_t layers; VkSampleCountFlagBits s; VkImageCreateFlags f; } dinfo[DV_COUNT] = {
    {VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_1D, VK_IMAGE_VIEW_TYPE_1D_ARRAY, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_3D, VK_IMAGE_VIEW_TYPE_3D, 1, VK_SAMPLE_COUNT_1_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE, 6, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_CUBE_ARRAY, 6, VK_SAMPLE_COUNT_1_BIT, VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D, 1, VK_SAMPLE_COUNT_4_BIT, 0},
    {VK_IMAGE_TYPE_2D, VK_IMAGE_VIEW_TYPE_2D_ARRAY, 1, VK_SAMPLE_COUNT_4_BIT, 0}};

// A 1x1 stand-in image of view type dinfo[i] in `fmt`, moved to GENERAL and filled with `colour` (NULL: zeros) by the
// upload batch (which reaches the queue before the submission that uses it). Before the fill was added a sampler the
// game left empty read whatever the memory held. *sview is the view a storage descriptor takes (a cube becomes a 2D
// array); sampled_only leaves the storage usage out (an sRGB format has none).
static void make_dummy(int i, VkFormat fmt, const float *colour, int sampled_only, VkImage *image, VmaAllocation *alloc,
                       VkImageView *view, VkImageView *sview) {
    VkCtx *c = vk_ctx();
    int ms = dinfo[i].s != VK_SAMPLE_COUNT_1_BIT;
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .flags = dinfo[i].f, .imageType = dinfo[i].t,
                            .format = fmt, .extent = {1, 1, 1}, .mipLevels = 1, .arrayLayers = dinfo[i].layers,
                            .samples = dinfo[i].s, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                     (ms || sampled_only ? 0 : VK_IMAGE_USAGE_STORAGE_BIT),
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ia = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ia, image, alloc, NULL) != VK_SUCCESS) rt_fatal("vk: stand-in image %d failed", i);
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = *image,
                                .viewType = dinfo[i].vt, .format = fmt,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, dinfo[i].layers}};
    if (vkCreateImageView(c->device, &vi, NULL, view) != VK_SUCCESS) rt_fatal("vk: stand-in view failed");
    *sview = *view;
    if (!ms && (dinfo[i].vt == VK_IMAGE_VIEW_TYPE_CUBE || dinfo[i].vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY)) {
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        if (vkCreateImageView(c->device, &vi, NULL, sview) != VK_SUCCESS) rt_fatal("vk: stand-in view failed");
    }
    // DEBUG-LIVECHECK: a stand-in made during play (a fill-in colour's, colour_dummy) can get the handle of a view
    // destroyed before; unannounced, the live check took every binding of it for a destroyed view (slot 258, the 4x4
    // fill-in texture of colour 000000ff bound to a sampler of another kind: ec3 and the stress runs, 2026-11-16)
    vk_live_check_born((uint64_t)(uintptr_t)*image);
    vk_live_check_born((uint64_t)(uintptr_t)*view);
    vk_live_check_born((uint64_t)(uintptr_t)*sview);
    VkImageMemoryBarrier2 bar = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                 .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                 .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                 .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                 .image = *image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, dinfo[i].layers}};
    VkCommandBuffer cb = vk_upload_begin();
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &bar};
    vkCmdPipelineBarrier2(cb, &di);
    VkClearColorValue cv;
    memset(&cv, 0, sizeof cv);   // all-zero bits are zero for float, signed and unsigned formats alike
    if (colour) memcpy(cv.float32, colour, sizeof cv.float32);
    vkCmdClearColorImage(cb, *image, VK_IMAGE_LAYOUT_GENERAL, &cv, 1, &bar.subresourceRange);
    vk_upload_end();
}

// Storage image stand-ins in the format a shader declares (GfxShaderRes.format: Ryujinx TextureFormat order); Vulkan wants
// the view format to match the declaration. Created on first use.
static const VkFormat tex_format[40] = {
    VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8_UNORM, VK_FORMAT_R8_SNORM, VK_FORMAT_R8_UINT, VK_FORMAT_R8_SINT,
    VK_FORMAT_R16_SFLOAT, VK_FORMAT_R16_UNORM, VK_FORMAT_R16_SNORM, VK_FORMAT_R16_UINT, VK_FORMAT_R16_SINT,
    VK_FORMAT_R32_SFLOAT, VK_FORMAT_R32_UINT, VK_FORMAT_R32_SINT, VK_FORMAT_R8G8_UNORM, VK_FORMAT_R8G8_SNORM,
    VK_FORMAT_R8G8_UINT, VK_FORMAT_R8G8_SINT, VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R16G16_UNORM, VK_FORMAT_R16G16_SNORM,
    VK_FORMAT_R16G16_UINT, VK_FORMAT_R16G16_SINT, VK_FORMAT_R32G32_SFLOAT, VK_FORMAT_R32G32_UINT, VK_FORMAT_R32G32_SINT,
    VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_R8G8B8A8_SNORM, VK_FORMAT_R8G8B8A8_UINT, VK_FORMAT_R8G8B8A8_SINT,
    VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_UNORM, VK_FORMAT_R16G16B16A16_SNORM, VK_FORMAT_R16G16B16A16_UINT,
    VK_FORMAT_R16G16B16A16_SINT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R32G32B32A32_UINT, VK_FORMAT_R32G32B32A32_SINT,
    VK_FORMAT_A2B10G10R10_UNORM_PACK32, VK_FORMAT_A2B10G10R10_UINT_PACK32, VK_FORMAT_B10G11R11_UFLOAT_PACK32};
static struct { int used, di; VkFormat fmt; VkImage image; VmaAllocation alloc; VkImageView view, sview; } SD[48];

static VkImageView storage_dummy(int di, int code) {
    VkFormat fmt = code > 0 && code < 40 ? tex_format[code] : VK_FORMAT_R8G8B8A8_UNORM;
    if (fmt == VK_FORMAT_R8G8B8A8_UNORM || dinfo[di].s != VK_SAMPLE_COUNT_1_BIT) return Z.img[di].sview;
    for (int i = 0; i < 48; i++) {
        if (!SD[i].used) {
            SD[i].used = 1, SD[i].di = di, SD[i].fmt = fmt;
            make_dummy(di, fmt, NULL, 0, &SD[i].image, &SD[i].alloc, &SD[i].view, &SD[i].sview);
            return SD[i].sview;
        }
        if (SD[i].di == di && SD[i].fmt == fmt) return SD[i].sview;
    }
    return Z.img[di].sview;
}

// Stand-ins of one colour, for a fill-in texture of the game bound to a sampler its image cannot serve: OoT binds its
// 4x4 white, black and clear 2D textures to the 3D and cube samplers a material leaves empty (the hero's eyes read a
// 3D sampler that way: the light it scales came out as whatever the unfilled stand-in held, and the whites of the
// eyes were cyan). The console's sampler reads the texture it is given, so the stand-in has that texture's colour.
#define COLOUR_DUMMIES 32
static struct { int used, di; VkFormat fmt; uint32_t rgba; VkImage image; VmaAllocation alloc; VkImageView view, sview; } CD[COLOUR_DUMMIES];

static float unorm_linear(uint8_t v, int srgb) {   // the value a clear must name for the image to store byte v
    float x = (float)v / 255.0f;
    return !srgb ? x : x <= 0.04045f ? x / 12.92f : powf((x + 0.055f) / 1.055f, 2.4f);
}

static VkImageView colour_dummy(int di, VkFormat fmt, const uint8_t rgba[4]) {
    uint32_t key;
    memcpy(&key, rgba, 4);
    for (int i = 0; i < COLOUR_DUMMIES; i++) {
        if (!CD[i].used) {
            int srgb = fmt == VK_FORMAT_R8G8B8A8_SRGB;
            float col[4] = {unorm_linear(rgba[0], srgb), unorm_linear(rgba[1], srgb), unorm_linear(rgba[2], srgb),
                            (float)rgba[3] / 255.0f};
            CD[i].used = 1, CD[i].di = di, CD[i].fmt = fmt, CD[i].rgba = key;
            make_dummy(di, fmt, col, 1, &CD[i].image, &CD[i].alloc, &CD[i].view, &CD[i].sview);
            rt_log("vk: a fill-in texture of colour %02x%02x%02x%02x is bound to a sampler of another kind (stand-in %d, format %d)",
                   rgba[0], rgba[1], rgba[2], rgba[3], di, (int)fmt);
            return CD[i].view;
        }
        if (CD[i].di == di && CD[i].fmt == fmt && CD[i].rgba == key) return CD[i].view;
    }
    return Z.img[di].view;
}

static void dummies_init(void) {
    if (Z.ready) return;
    VkCtx *c = vk_ctx();
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = ZERO_BYTES,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_TEXEL_BUFFER_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT,
                                  .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};   // the memset below is not flushed
    VmaAllocationInfo info;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &Z.zbuf, &Z.zalloc, &info) != VK_SUCCESS) rt_fatal("vk: zero buffer failed");
    memset(info.pMappedData, 0, ZERO_BYTES);
    VkBufferViewCreateInfo bv = {.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO, .buffer = Z.zbuf,
                                 .format = VK_FORMAT_R32G32B32A32_SFLOAT, .range = ZERO_BYTES};
    if (vkCreateBufferView(c->device, &bv, NULL, &Z.zview) != VK_SUCCESS) rt_fatal("vk: zero buffer view failed");
    samp_par sp;
    memset(&sp, 0, sizeof sp);
    sp.lod_max = 1000.0f;
    Z.samp = vk_sampler_for(&sp);
    sp.cmp_mode = 1, sp.cmp_func = NVN_DEPTH_FUNC_ALWAYS;
    Z.samp_shadow = vk_sampler_for(&sp);

    for (int i = 0; i < DV_COUNT; i++)
        make_dummy(i, VK_FORMAT_R8G8B8A8_UNORM, NULL, 0, &Z.img[i].image, &Z.img[i].alloc, &Z.img[i].view, &Z.img[i].sview);
    Z.ready = 1;
}

static void twins_release(VkCtx *c);

// Releases the stand-ins and the depth twins. The device must be idle; the game path leaves them to process exit like
// the pipelines.
#ifdef GAME_NS2
static void gmem_shutdown(VkCtx *c);
#endif
#ifdef GAME_OOT
static void skin_shutdown(void);
#endif
void vk_exec_draw_shutdown(void) {
    if (!Z.ready) return;
    VkCtx *c = vk_ctx();
    twins_release(c);
#ifdef GAME_NS2
    gmem_shutdown(c);
#endif
#ifdef GAME_OOT
    skin_shutdown();
#endif
    for (int i = 0; i < DV_COUNT; i++) {
        if (Z.img[i].sview != Z.img[i].view) vkDestroyImageView(c->device, Z.img[i].sview, NULL);
        vkDestroyImageView(c->device, Z.img[i].view, NULL);
        vmaDestroyImage(c->vma, Z.img[i].image, Z.img[i].alloc);
    }
    for (int i = 0; i < 48 && SD[i].used; i++) {
        if (SD[i].sview != SD[i].view) vkDestroyImageView(c->device, SD[i].sview, NULL);
        vkDestroyImageView(c->device, SD[i].view, NULL);
        vmaDestroyImage(c->vma, SD[i].image, SD[i].alloc);
    }
    memset(SD, 0, sizeof SD);
    for (int i = 0; i < COLOUR_DUMMIES && CD[i].used; i++) {
        if (CD[i].sview != CD[i].view) vkDestroyImageView(c->device, CD[i].sview, NULL);
        vkDestroyImageView(c->device, CD[i].view, NULL);
        vmaDestroyImage(c->vma, CD[i].image, CD[i].alloc);
    }
    memset(CD, 0, sizeof CD);
    nvn_hm_clear(&rs_lists, free);
    rs_last = NULL, rs_last_l = NULL;
    for (int i = 0; i < nspare_pool; i++) vkDestroyDescriptorPool(c->device, spare_pool[i], NULL);
    for (int i = 0; i < nspare_chunk; i++) vmaDestroyBuffer(c->vma, spare_chunk[i].buf, spare_chunk[i].alloc);
    nspare_pool = nspare_chunk = 0;
    vk_usnap_shutdown();
    vkDestroyBufferView(c->device, Z.zview, NULL);
    vmaDestroyBuffer(c->vma, Z.zbuf, Z.zalloc);
    memset(&Z, 0, sizeof Z);
}

// ---- resource types

static VkImageViewType res_view_type(uint16_t t) {
    int arr = t & 0x100;
    switch (t & 0xff) {
    case 1: return arr ? VK_IMAGE_VIEW_TYPE_1D_ARRAY : VK_IMAGE_VIEW_TYPE_1D;
    case 4: return VK_IMAGE_VIEW_TYPE_3D;
    case 5: return arr ? VK_IMAGE_VIEW_TYPE_CUBE_ARRAY : VK_IMAGE_VIEW_TYPE_CUBE;
    default: return arr ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D;
    }
}

static int dummy_index(VkImageViewType vt, int ms) {
    switch (vt) {
    case VK_IMAGE_VIEW_TYPE_1D: return DV_1D;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: return DV_1D_ARRAY;
    case VK_IMAGE_VIEW_TYPE_3D: return DV_3D;
    case VK_IMAGE_VIEW_TYPE_CUBE: return DV_CUBE;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: return DV_CUBE_ARRAY;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY: return ms ? DV_2D_MS_ARRAY : DV_2D_ARRAY;
    default: return ms ? DV_2D_MS : DV_2D;
    }
}

// GFX_TEX_* of a bound texture slot (the translator's tex_type / img_type answers).
static int gfx_type_of(const VkTexImage *img, const view_t *v, int hv) {
    if (hv && v->set_target) return vk_pipeline_tex_type(v->target);
    int ms = img->samples > 1;
    switch (img->view_type) {
    case VK_IMAGE_VIEW_TYPE_1D: return GFX_TEX_1D;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: return GFX_TEX_1D_ARRAY;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY: return ms ? GFX_TEX_2D_MS_ARRAY : GFX_TEX_2D_ARRAY;
    case VK_IMAGE_VIEW_TYPE_3D: return GFX_TEX_3D;
    case VK_IMAGE_VIEW_TYPE_CUBE: return GFX_TEX_CUBE;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: return GFX_TEX_CUBE_ARRAY;
    default: return ms ? GFX_TEX_2D_MS : GFX_TEX_2D;
    }
}

static VkComponentSwizzle swz_of(uint8_t s) {
    switch (s) {
    case NVN_TEXTURE_SWIZZLE_ZERO: return VK_COMPONENT_SWIZZLE_ZERO;
    case NVN_TEXTURE_SWIZZLE_ONE: return VK_COMPONENT_SWIZZLE_ONE;
    case NVN_TEXTURE_SWIZZLE_R: return VK_COMPONENT_SWIZZLE_R;
    case NVN_TEXTURE_SWIZZLE_G: return VK_COMPONENT_SWIZZLE_G;
    case NVN_TEXTURE_SWIZZLE_B: return VK_COMPONENT_SWIZZLE_B;
    default: return VK_COMPONENT_SWIZZLE_A;
    }
}

// A view in STENCIL mode samples the stencil plane (TextureViewSetDepthStencilMode); otherwise depth wins. The
// translator declares every sampled image float and the shader takes the raw bits, as for integer formats (nxnative
// nvn_mtl.m).
VkImageAspectFlags vk_exec_view_aspect(const VkTexImage *img, const view_t *v, int hv) {
    if (hv && v->set_dsmode && v->dsmode == NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL && (img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT))
        return VK_IMAGE_ASPECT_STENCIL_BIT;
    return img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT     ? VK_IMAGE_ASPECT_DEPTH_BIT
           : img->aspect & VK_IMAGE_ASPECT_STENCIL_BIT ? VK_IMAGE_ASPECT_STENCIL_BIT
                                                       : VK_IMAGE_ASPECT_COLOR_BIT;
}

// Image views of registered textures, kept for their image's lifetime: made per command buffer they were created again
// every frame (vkCreateImageView took about 8% of the fight replay on the 7900X). Used under the texture lock, as
// vk_exec_image_gone is (destroy_tex), which hands the views of a destroyed image to the deferred release with it.
typedef struct ViewNode { struct ViewNode *next; ViewKey k; VkImageView view; } ViewNode;
// DEBUG-LASTVIEW: the sampled view being created or created last (the profiler's hang watch logs it: a process that
// stopped inside vkCreateImageView says for which image)
static struct { VkImageViewCreateInfo ci; VkTexImage img; volatile int in_call; } last_view;
void vk_exec_last_view_log(void) {
    const VkImageViewCreateInfo *c = &last_view.ci;
    const VkTexImage *m = &last_view.img;
    rt_log("vk_exec: the last sampled view (%s): image %#llx addr %#llx %ux%ux%u, %u levels, %u layers, %u samples, format %d, "
           "usage %#x, gone %d; view type %d format %d aspect %#x levels %u+%u layers %u+%u",
           last_view.in_call ? "vkCreateImageView has not returned" : "created", (unsigned long long)(uintptr_t)m->image,
           (unsigned long long)m->addr, m->width, m->height, m->depth, m->levels, m->layers, m->samples, (int)m->format,
           (unsigned)m->usage, (int)m->gone, (int)c->viewType, (int)c->format, (unsigned)c->subresourceRange.aspectMask,
           c->subresourceRange.baseMipLevel, c->subresourceRange.levelCount, c->subresourceRange.baseArrayLayer,
           c->subresourceRange.layerCount);
}
static nvn_hmap view_of;                 // ViewKey -> VkImageView
static nvn_hmap views_by_image;          // VkImage -> its first ViewNode
// DEBUG-VIEWWATCH: REKARINA_VIEWWATCH=<hex addr>: the sampled views made for images of the texture at that address, and
// their purges (a stale view the live check found handed out for a 4x4 texture, 2026-11-16)
static uint64_t vw_addr = 1;
static VkImage vw_img[64];
static int vw_n;

#ifdef GAME_NS2
// Bytes of one texel of the uncompressed formats NVN2 views reinterpret between; 0 for any other format.
static uint32_t texel_bytes(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: return 1;
    case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R8G8_SNORM: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_SNORM: case VK_FORMAT_R16_UINT: case VK_FORMAT_R16_SINT:
    case VK_FORMAT_R16_SFLOAT: return 2;
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SNORM: case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_R8G8B8A8_SRGB: case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB:
    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: case VK_FORMAT_A2B10G10R10_UINT_PACK32:
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: case VK_FORMAT_E5B9G9R9_UFLOAT_PACK32:
    case VK_FORMAT_R16G16_UNORM: case VK_FORMAT_R16G16_SNORM: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT:
    case VK_FORMAT_R16G16_SFLOAT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32_SFLOAT: return 4;
    case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_SNORM: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32_SFLOAT: return 8;
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT: case VK_FORMAT_R32G32B32A32_SFLOAT: return 16;
    default: return 0;
    }
}

// The format of an NVN2 texture view that reinterprets its texture's texels (nvnTextureViewSetFormat): OoT's 3D frame
// packs RGB9E5 words into r11g11b10f and rgba8 images through R32UI image views, and samples them back through
// RGB9E5 views. Used when the view format maps directly to a Vulkan format with the image's texel size and the GPU can
// sample it (sampled) or store to it (storage); the image is MUTABLE_FORMAT (vk_texture.c). 0: the image's format.
static VkFormat reinterpret_format(const VkTexImage *img, const view_t *v, int hv, int storage) {
    TexVkFormat b;
    if (!hv || !v->set_format || img->block_views || img->aspect != VK_IMAGE_ASPECT_COLOR_BIT ||
        !tex_vk_format(v->format, &b) || b.vk != b.ideal || b.depth || b.stencil || b.vk == img->format)
        return VK_FORMAT_UNDEFINED;   // depth images are not MUTABLE_FORMAT
    uint32_t n = texel_bytes(b.vk);
    if (!n || n != texel_bytes(img->format)) return VK_FORMAT_UNDEFINED;
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(vk_ctx()->phys, b.vk, &fp);
    VkFormatFeatureFlags need = storage ? VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT : VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT;
    if (!(fp.optimalTilingFeatures & need)) return VK_FORMAT_UNDEFINED;
    return b.vk;
}
#endif

// The view cache's key of a sampled or storage view of img for a shader resource of type `rtype`, and the create info
// of that view (ci->pNext points at *use when the view needs a usage of its own). 0: the image cannot serve it. Touches
// no table: get_view and the views made ahead (vk_exec_views_ahead) share it.
static int sampled_view_setup(const VkTexImage *img, const view_t *v, int hv, uint16_t rtype, int storage, ViewKey *kout,
                              VkImageViewCreateInfo *ciout, VkImageViewUsageCreateInfo *useout) {
    VkImageViewType vt = res_view_type(rtype);
    int ms = (rtype & 0x200) != 0;
    if ((img->samples > 1) != ms || (storage && ms)) return 0;
    VkImageUsageFlags purpose = storage ? VK_IMAGE_USAGE_STORAGE_BIT : VK_IMAGE_USAGE_SAMPLED_BIT;
    if (!(img->usage & purpose)) return 0;
    int cube = vt == VK_IMAGE_VIEW_TYPE_CUBE || vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY;
    int ok = 0;
    switch (vt) {
    case VK_IMAGE_VIEW_TYPE_1D:
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: ok = img->type == VK_IMAGE_TYPE_1D; break;
    case VK_IMAGE_VIEW_TYPE_3D: ok = img->type == VK_IMAGE_TYPE_3D; break;
    case VK_IMAGE_VIEW_TYPE_CUBE:
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY:
        ok = img->type == VK_IMAGE_TYPE_2D && (img->view_type == VK_IMAGE_VIEW_TYPE_CUBE || img->view_type == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY);
        break;
    default: ok = img->type == VK_IMAGE_TYPE_2D; break;
    }
    if (!ok) return 0;
    uint32_t l0 = hv && v->set_levels ? v->base_level : 0;
    if (l0 >= img->levels) return 0;
    uint32_t nl = hv && v->set_levels && v->nlevels ? v->nlevels : img->levels - l0;
    if (l0 + nl > img->levels) nl = img->levels - l0;
    if (storage) nl = 1;
    uint32_t y0 = 0, ny = 1;
    if (vt != VK_IMAGE_VIEW_TYPE_3D) {
        y0 = hv && v->set_layers ? v->min_layer : 0;
        if (y0 >= img->layers) return 0;
        uint32_t rest = hv && v->set_layers && v->nlayers ? v->nlayers : img->layers - y0;
        if (y0 + rest > img->layers) rest = img->layers - y0;
        if (vt == VK_IMAGE_VIEW_TYPE_1D || vt == VK_IMAGE_VIEW_TYPE_2D) ny = 1;
        else if (vt == VK_IMAGE_VIEW_TYPE_CUBE) ny = 6;
        else if (vt == VK_IMAGE_VIEW_TYPE_CUBE_ARRAY) ny = rest / 6 * 6;
        else ny = rest;
        if (!ny || y0 + ny > img->layers) return 0;
    }
    if (storage && cube) vt = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    uint32_t aspect = vk_exec_view_aspect(img, v, hv);
    uint8_t sw[4] = {NVN_TEXTURE_SWIZZLE_R, NVN_TEXTURE_SWIZZLE_G, NVN_TEXTURE_SWIZZLE_B, NVN_TEXTURE_SWIZZLE_A};
    if (!storage) {   // MNG nvn_mtl.m tex_swizzle_sel: view selectors replace the texture's, then the format remap
        memcpy(sw, hv && v->set_swizzle ? v->swz : img->swz, 4);
        if (img->alpha_one || img->chan[0])
            for (int i = 0; i < 4; i++) {
                if (img->alpha_one && sw[i] == NVN_TEXTURE_SWIZZLE_A) sw[i] = NVN_TEXTURE_SWIZZLE_ONE;
                else if (img->chan[0] && sw[i] >= NVN_TEXTURE_SWIZZLE_R && sw[i] <= NVN_TEXTURE_SWIZZLE_A)
                    sw[i] = img->chan[sw[i] - NVN_TEXTURE_SWIZZLE_R];
            }
    }
    uint32_t swz = (uint32_t)sw[0] | (uint32_t)sw[1] << 8 | (uint32_t)sw[2] << 16 | (uint32_t)sw[3] << 24;
#ifdef GAME_NS2
    // NVN2 colour images have storage usage (vk_texture.c): a storage view of an sRGB one is its UNORM format, which the
    // image slot's own view names too (OoT's movie composition writes its sRGB output through a 0x25 view). A view
    // format of the same texel size replaces the image's (reinterpret_format).
    VkFormat rf = reinterpret_format(img, v, hv, storage);
    VkFormat vf = rf ? rf : storage ? vk_storage_format(img->format) : img->format;
    *kout = (ViewKey){img->image, vf, vt, aspect, l0, nl, y0, ny, swz, img->serial, 0};
#else
    *kout = (ViewKey){img->image, img->format, vt, aspect, l0, nl, y0, ny, swz, img->serial, 0};
#endif
    VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img->image, .viewType = vt,
                                .format = img->format,
                                .components = {swz_of(sw[0]), swz_of(sw[1]), swz_of(sw[2]), swz_of(sw[3])},
                                .subresourceRange = {aspect, l0, nl, y0, ny}};
    *useout = (VkImageViewUsageCreateInfo){.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                           .usage = purpose};
    if (img->block_views) ci.pNext = useout;   // the image's COLOR_ATTACHMENT usage is its block views' (vk_texture.c)
#ifdef GAME_NS2
    ci.format = vf;
    if (!storage && (img->usage & VK_IMAGE_USAGE_STORAGE_BIT) && vk_storage_format(vf) != vf) {
        useout->usage = VK_IMAGE_USAGE_SAMPLED_BIT | (img->usage & VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT);
        ci.pNext = useout;   // an sRGB view has no storage usage to inherit
    }
    if (rf) {   // a reinterpreting view has only the usage it is made for: RGB9E5 is no attachment or storage format
        useout->usage = storage ? VK_IMAGE_USAGE_STORAGE_BIT
                                : VK_IMAGE_USAGE_SAMPLED_BIT | (img->usage & VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT);
        ci.pNext = useout;
    }
#endif
    if (ci.pNext) kout->usage = useout->usage;
    *ciout = ci;
    return 1;
}

// Views the replay thread created (vkCreateImageView in get_view and vk_exec_target_view), views taken from those made
// ahead with their image (vk_exec_views_adopt) and made-ahead views no lookup would have named (dropped): the counters of
// REKARINA_VIEWS_AHEAD, logged with the made-ahead images (vk_texture.c).
static volatile int64_t n_views_replay, n_views_adopted, n_views_dropped;
void vk_exec_views_stats(int64_t *replay, int64_t *adopted, int64_t *dropped) {
    *replay = __atomic_exchange_n(&n_views_replay, 0, __ATOMIC_RELAXED);
    *adopted = __atomic_exchange_n(&n_views_adopted, 0, __ATOMIC_RELAXED);
    *dropped = __atomic_exchange_n(&n_views_dropped, 0, __ATOMIC_RELAXED);
}

static void view_cache_put(const VkTexImage *img, const ViewKey *k, VkImageView view) {   // texture lock held
    if (!view_of.klen) nvn_hm_init(&view_of, sizeof(ViewKey)), nvn_hm_init(&views_by_image, sizeof(VkImage));
    ViewNode *n = malloc(sizeof *n);
    if (!n) rt_fatal("vk: out of memory");
    *n = (ViewNode){nvn_hm_get(&views_by_image, &img->image), *k, view};
    nvn_hm_put(&views_by_image, &img->image, n);
    nvn_hm_put(&view_of, k, (void *)view);
}

// An image view of a registered texture for a shader resource of type `rtype` (GfxShaderRes.type); VK_NULL_HANDLE when
// the image cannot serve it (the caller binds a stand-in). Texture lock held.
static VkImageView get_view(VkExec *ex, const VkTexImage *img, const view_t *v, int hv, uint16_t rtype, int storage) {
    (void)ex;
    ViewKey k;
    VkImageViewCreateInfo ci;
    VkImageViewUsageCreateInfo use;
    if (!sampled_view_setup(img, v, hv, rtype, storage, &k, &ci, &use)) return VK_NULL_HANDLE;
    if (!view_of.klen) nvn_hm_init(&view_of, sizeof(ViewKey)), nvn_hm_init(&views_by_image, sizeof(VkImage));
    if (__builtin_expect(vw_addr == 1, 0)) { const char *e = cfg_get("REKARINA_VIEWWATCH"); vw_addr = e && *e ? strtoull(e, NULL, 16) : 0; }
    if (__builtin_expect(vk_live_check_on(), 0) && !img->gone)   // DEBUG-LIVECHECK: a live record's image
        vk_live_check_use((uint64_t)(uintptr_t)img->image, "a sampled view's image", img->addr);
    if (__builtin_expect(vk_live_check_on(), 0)) {   // DEBUG-VIEWLOCK: the cache's tables are the texture lock's
        int vk_texture_lock_held(void);
        static volatile int32_t n_unlocked;
        if (!vk_texture_lock_held() && plat_inc32(&n_unlocked) <= 8)
            rt_log("LIVE CHECK: the view cache is used without the texture lock (caller %p, worker %d, image %#llx addr %#llx)",
                   __builtin_return_address(0), vk_par_worker, (unsigned long long)(uintptr_t)img->image,
                   (unsigned long long)img->addr);
    }
    VkImageView found = (VkImageView)nvn_hm_get(&view_of, &k);
    if (found) return found;
    VkImageView view;
    last_view = (typeof(last_view)){ci, *img, 1};   // DEBUG-LASTVIEW
    vk_drv_urgent(1);   // the frame waits for this call: the image-ahead thread and the reaper let it go first
    VkResult vr = vkCreateImageView(vk_ctx()->device, &ci, NULL, &view);
    vk_drv_urgent(-1);
    if (vr != VK_SUCCESS) return VK_NULL_HANDLE;
    last_view.in_call = 0;
    __atomic_fetch_add(&n_views_replay, 1, __ATOMIC_RELAXED);
    vk_live_check_born((uint64_t)(uintptr_t)view);   // DEBUG-LIVECHECK
    // An image that is already gone (destroy_tex keeps its record for the command buffer being recorded) gets a view for
    // this use only. Its views were purged when it went (vk_exec_views_gone); one cached now would stay under the image's
    // handle for good, and the next image the driver gives that handle would be sampled through it: a view of freed
    // memory. vk_exec_target_view below has had the same rule since the white-scene fix. The view differs on every
    // lookup, so a draw built with it is never taken as the same as the next (RS.volatile_).
    if (img->gone) {
        vk_defer_view(view);
        RS.volatile_ = 1;
        return view;
    }
    view_cache_put(img, &k, view);
    if (__builtin_expect(vw_addr > 1, 0) && img->addr == vw_addr) {   // DEBUG-VIEWWATCH
        if (vw_n < 64) vw_img[vw_n++] = img->image;
        rt_log("VIEWWATCH frame %llu: view %#llx made for image %#llx serial %u addr %#llx (worker %d)",
               (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), (unsigned long long)(uintptr_t)view,
               (unsigned long long)(uintptr_t)img->image, img->serial, (unsigned long long)img->addr, vk_par_worker);
    }
    return view;
}

// The key and create info of an attachment view (vk_exec_target_view, vk_exec_views_ahead). Attachment keys carry a
// swizzle word no sampled key has (those hold four NVN selectors, each below 8).
static void target_view_setup(const VkTexImage *img, VkFormat format, VkImageViewType type, uint32_t aspect, uint32_t level,
                              uint32_t base, uint32_t layers, int blocks, ViewKey *kout, VkImageViewCreateInfo *ciout,
                              VkImageViewUsageCreateInfo *useout) {
    *kout = (ViewKey){img->image, format, type, aspect, level, 1, base, layers, blocks ? 0xfffffffeu : 0xffffffffu, img->serial, 0};
    VkImageViewCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = img->image,
                                .viewType = type, .format = format, .subresourceRange = {aspect, level, 1, base, layers}};
    *useout = (VkImageViewUsageCreateInfo){.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_USAGE_CREATE_INFO,
                                           .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT};
    if (blocks) ci.pNext = useout;   // not the BC image's sampling and transfer usages
#ifdef GAME_NS2
    if (!blocks && (img->usage & VK_IMAGE_USAGE_STORAGE_BIT) && vk_storage_format(format) != format) {
        useout->usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | (img->usage & VK_IMAGE_USAGE_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT);
        ci.pNext = useout;   // an sRGB view of an NVN2 image with storage usage (vk_texture.c) leaves that usage out
    }
#endif
    *ciout = ci;
}

// An attachment view of a registered texture, kept with its sampled views for the image's lifetime: made per command
// buffer, every render target's views were created again each frame (vkCreateImageView under vk_exec_pass_begin, 2%
// of the fight replay on the 7900X). The texture lock guards the tables, as for get_view; vk_exec_pass_begin may already
// hold it.
VkImageView vk_exec_target_view(const VkTexImage *img, VkFormat format, VkImageViewType type, uint32_t aspect,
                                uint32_t level, uint32_t base, uint32_t layers, int blocks) {
    vk_texture_lock();   // destroy_tex sets gone and purges the image's views with it held
    if (__builtin_expect(vk_live_check_on(), 0) && !img->gone)   // DEBUG-LIVECHECK
        vk_live_check_use((uint64_t)(uintptr_t)img->image, "a render target's image", img->addr);
    ViewKey k;
    VkImageViewCreateInfo ci;
    VkImageViewUsageCreateInfo use;
    target_view_setup(img, format, type, aspect, level, base, layers, blocks, &k, &ci, &use);
    if (!view_of.klen) nvn_hm_init(&view_of, sizeof(ViewKey)), nvn_hm_init(&views_by_image, sizeof(VkImage));
    // A texture destroyed while a command buffer or DLSS still held it (img->gone): its image lives on until the GPU is
    // past the submission being recorded (vk_defer_image), so a view for this use is made, and goes the same way. Cached,
    // it outlived the image: Vulkan hands the handle to a later image, which then got the view of the dead image's memory
    // and the GPU faulted reading it at the start of a pass.
    VkImageView view = img->gone ? VK_NULL_HANDLE : (VkImageView)nvn_hm_get(&view_of, &k);
    if (!view) {
        vk_drv_urgent(1);
        VkResult vr = vkCreateImageView(vk_ctx()->device, &ci, NULL, &view);
        vk_drv_urgent(-1);
        if (vr != VK_SUCCESS) rt_fatal("vk: render target view (format %d) could not be created", (int)format);
        __atomic_fetch_add(&n_views_replay, 1, __ATOMIC_RELAXED);
        vk_live_check_born((uint64_t)(uintptr_t)view);   // DEBUG-LIVECHECK
        if (img->gone) {
            vk_defer_view(view);
            vk_texture_unlock();
            return view;
        }
        view_cache_put(img, &k, view);
    }
    vk_texture_unlock();
    return view;
}

// ---- Views made ahead with their image (REKARINA_VIEWS_AHEAD)
// A new image is cheap for the replay since the images in pool memory are made ahead (vk_texture.c pm_main), but its
// views were still made by the replay thread at their first use (get_view, vk_exec_target_view), and while the thread
// that makes the images ahead and the reaper that destroys the old ones are inside the driver, vkCreateImageView waits
// for them: NVIDIA's driver makes kernel calls for an image's address space with its object lock held. While the hero
// turns, the game gives its transient render targets new sizes and addresses every frame (~15 new images a frame) and
// the replay waited there for ~8 ms a frame, with the texture lock held (the game's registrations waited behind it):
// scripted half turns 46 frames over 25 ms in 1,300 (2026-11-17, profile slow-frame dumps). So the thread that makes an
// image ahead also makes the views its first uses ask for: the whole-image sampled view of its own type (with the
// texture's swizzle) and, for an attachable image, the whole-image attachment view. The replay takes them into the
// view cache when it takes the image (vk_exec_views_adopt), keyed as get_view and vk_exec_target_view would key them;
// one whose key the image's record does not give is destroyed (never handed out). No table is touched ahead.

// The shader resource type (GfxShaderRes.type) of an image's own view type.
static uint16_t rtype_of(const VkTexImage *img) {
    uint16_t ms = img->samples > 1 ? 0x200 : 0;
    switch (img->view_type) {
    case VK_IMAGE_VIEW_TYPE_1D: return 1;
    case VK_IMAGE_VIEW_TYPE_1D_ARRAY: return 0x101;
    case VK_IMAGE_VIEW_TYPE_3D: return 4;
    case VK_IMAGE_VIEW_TYPE_CUBE: return 5;
    case VK_IMAGE_VIEW_TYPE_CUBE_ARRAY: return 0x105;
    case VK_IMAGE_VIEW_TYPE_2D_ARRAY: return (uint16_t)(0x102 | ms);
    default: return (uint16_t)(2 | ms);
    }
}

// The default views of img: 0 the sampled one, 1 the attachment one (as vk_exec.c resolve_target names a target bound
// without a texture view), 2 for a single-sample 2D image its whole-image 2D array view, which the texture pool's
// array set asks for (vk_texpool.c resolve_slot: vk_exec_pool_view of type 0x103; after the first two the views the
// replay still made for these images, 2026-11-17 REKARINA_VIEWLOG). 0: that view is not made ahead.
static int ahead_setup(const VkTexImage *img, int which, ViewKey *k, VkImageViewCreateInfo *ci, VkImageViewUsageCreateInfo *use) {
    if (!which) return (img->usage & VK_IMAGE_USAGE_SAMPLED_BIT) && sampled_view_setup(img, NULL, 0, rtype_of(img), 0, k, ci, use);
    if (which == 2)
        return (img->usage & VK_IMAGE_USAGE_SAMPLED_BIT) && img->view_type == VK_IMAGE_VIEW_TYPE_2D && img->samples <= 1 &&
               sampled_view_setup(img, NULL, 0, 0x102, 0, k, ci, use);
    int depth = (img->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT)) != 0;
    VkImageUsageFlags need = depth ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT : VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (!(img->usage & need) || img->block_views || img->type != VK_IMAGE_TYPE_2D || !img->layers) return 0;
    uint32_t aspect = depth ? img->aspect & (VK_IMAGE_ASPECT_DEPTH_BIT | VK_IMAGE_ASPECT_STENCIL_BIT) : VK_IMAGE_ASPECT_COLOR_BIT;
    target_view_setup(img, img->format, img->layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, aspect, 0, 0,
                      img->layers, 0, k, ci, use);
    return 1;
}

int vk_exec_views_ahead_on(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("VIEWS_AHEAD", "1");
    return on;
}

// The thread that makes images ahead (vk_texture.c pm_main), right after the bind: img is the record the image will get
// (serial 0). Fills va; no lock, no table.
void vk_exec_views_ahead(const VkTexImage *img, VkViewsAhead *va) {
    memset(va, 0, sizeof *va);
    if (!vk_exec_views_ahead_on() || !img->image) return;
    va->image = img->image;
    for (int w = 0; w < VK_VIEWS_AHEAD; w++) {
        ViewKey k;
        VkImageViewCreateInfo ci;
        VkImageViewUsageCreateInfo use;
        if (!ahead_setup(img, w, &k, &ci, &use)) continue;
        if (vkCreateImageView(vk_ctx()->device, &ci, NULL, &va->view[w]) != VK_SUCCESS) {
            va->view[w] = VK_NULL_HANDLE;
            continue;
        }
        vk_live_check_born((uint64_t)(uintptr_t)va->view[w]);   // DEBUG-LIVECHECK
        _Static_assert(sizeof va->key[w] == sizeof k, "VkViewsAhead.key holds a ViewKey");
        memcpy(va->key[w], &k, sizeof k);
    }
}

// Views made ahead that nobody took: destroyed at once (never handed out, so no submission names them).
void vk_exec_views_ahead_drop(VkViewsAhead *va) {
    VkCtx *c = vk_ctx();
    for (int w = 0; w < VK_VIEWS_AHEAD; w++)
        if (va->view[w]) {
            if (c) vkDestroyImageView(c->device, va->view[w], NULL);
            va->view[w] = VK_NULL_HANDLE;
            __atomic_fetch_add(&n_views_dropped, 1, __ATOMIC_RELAXED);
        }
    va->image = VK_NULL_HANDLE;
}

// create_tex took the image made ahead and gave img its record (texture lock held): the views go into the cache under
// the keys get_view and vk_exec_target_view compute for that record, where they match what was made; the rest is dropped.
void vk_exec_views_adopt(const VkTexImage *img, VkViewsAhead *va) {
    if (!view_of.klen) nvn_hm_init(&view_of, sizeof(ViewKey)), nvn_hm_init(&views_by_image, sizeof(VkImage));
    if (va->image != img->image || img->gone) {
        vk_exec_views_ahead_drop(va);
        return;
    }
    for (int w = 0; w < VK_VIEWS_AHEAD; w++) {
        if (!va->view[w]) continue;
        ViewKey k, made;
        VkImageViewCreateInfo ci;
        VkImageViewUsageCreateInfo use;
        memcpy(&made, va->key[w], sizeof made);
        made.serial = img->serial;   // the record's serial is given when the image is taken
        if (!ahead_setup(img, w, &k, &ci, &use) || memcmp(&k, &made, sizeof k) || nvn_hm_get(&view_of, &k)) continue;
        view_cache_put(img, &k, va->view[w]);
        va->view[w] = VK_NULL_HANDLE;
        __atomic_fetch_add(&n_views_adopted, 1, __ATOMIC_RELAXED);
    }
    vk_exec_views_ahead_drop(va);
}

int vk_exec_image_gone(VkImage image) {
    vk_exec_bc_gone(image);
    vk_hdr_image_gone(image);
    vk_tpool_image_gone(image);
    return vk_exec_views_gone(image);
}

int vk_exec_views_gone(VkImage image) {
    if (!views_by_image.klen) return 0;
    int purged = 0, watched = 0;
    for (int i = 0; vw_addr > 1 && i < vw_n && !watched; i++) watched = vw_img[i] == image;   // DEBUG-VIEWWATCH
    for (ViewNode *n = nvn_hm_del(&views_by_image, &image), *next; n; n = next, purged++) {
        next = n->next;
        if (watched)
            rt_log("VIEWWATCH frame %llu: view %#llx of image %#llx (key image %#llx serial %u) purged",
                   (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), (unsigned long long)(uintptr_t)n->view,
                   (unsigned long long)(uintptr_t)image, (unsigned long long)(uintptr_t)n->k.image, n->k.serial);
        nvn_hm_del(&view_of, &n->k);
        vk_defer_view(n->view);
        free(n);
    }
    return purged;
}

// ---- program to shaders

// (program, stage, answers) -> shader, 4-way set-associative, most recently used first. A miss costs a SHA-256 of the
// whole shader code (gfx_shader.c req_init), slow on CPUs without SHA instructions: a field frame looks up ~3000 and
// the 2048-slot direct-mapped table this replaced missed 3% of them on conflicts.
#define SHADER_SETS 2048
#define SHADER_WAYS 4
typedef struct { uint64_t serial; int stage; int used; GfxShaderAnswers a; const GfxShader *s; } ShaderEnt;
static _Thread_local ShaderEnt shader_tab[SHADER_SETS][SHADER_WAYS];

#ifdef GAME_NS2
// The replay's stage slot of a program's shader i (nvn_ctl.h: NVN reads control +0x714; NVN2 the stage byte or the SPH,
// task and mesh in the tessellation slots, tessellation itself not replayed), or -1.
static int stage_slot_of(const prog_t *p, uint32_t i) {
    const uint8_t *c = p->sd[i].control;
    if (!c || !p->sd[i].data) return -1;
    int st = nvn_ctl_stage(c, (const uint8_t *)(uintptr_t)p->sd[i].data);
#ifdef GAME_NS2
    if (nvn2_ctl(c) && (st == 3 || st == 4)) {
        static uint8_t once;
        if (!__atomic_exchange_n(&once, 1, __ATOMIC_RELAXED)) rt_log("vk_exec: NVN2 tessellation shaders are not replayed");
        return -1;
    }
#endif
    return nvn_stage_slot(st);
}

static int find_stage_scan(const prog_t *p, int nvn_stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    for (uint32_t i = 0; i < p->nstage && i < 6; i++) {
        if (stage_slot_of(p, i) != nvn_stage) continue;
        const uint8_t *c = p->sd[i].control;
        *len = nvn_ctl_code_len(c);
        *ctl = c, *code = (const uint8_t *)(uintptr_t)p->sd[i].data;
        return 1;
    }
    return 0;
}
#else
static int find_stage_scan(const prog_t *p, int nvn_stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    for (uint32_t i = 0; i < p->nstage && i < 6; i++) {
        const uint8_t *c = p->sd[i].control;
        if (!c || !p->sd[i].data) continue;
        uint32_t st;
        memcpy(&st, c + 0x714, 4);
        if ((int)st != nvn_stage) continue;
        memcpy(len, c + 0x704, 4);
        *ctl = c, *code = (const uint8_t *)(uintptr_t)p->sd[i].data;
        return 1;
    }
    return 0;
}
#endif

// Which shader of a program serves each NVN stage, and its code length, per program serial (new at every
// nvnProgramSetShaders): the scan reads two words of every stage's control block, cold guest cache lines, and runs
// several times a draw. Direct-mapped; a collision only scans again.
#define STAGEM_N 512
static _Thread_local struct { uint64_t serial; int8_t idx[6]; uint32_t len[6]; } stagem[STAGEM_N];

static int find_stage(const prog_t *p, int nvn_stage, const uint8_t **ctl, const uint8_t **code, uint32_t *len) {
    if (!p->serial || (unsigned)nvn_stage >= 6) return find_stage_scan(p, nvn_stage, ctl, code, len);
    typeof(stagem[0]) *m = &stagem[p->serial * 0x9E3779B97F4A7C15ull >> 55];
    if (m->serial != p->serial) {
        m->serial = p->serial;
        memset(m->idx, -1, sizeof m->idx);
        for (uint32_t i = 0; i < p->nstage && i < 6; i++) {
#ifdef GAME_NS2
            int st = stage_slot_of(p, i);
            if (st >= 0 && st < 6 && m->idx[st] < 0) m->idx[st] = (int8_t)i, m->len[st] = nvn_ctl_code_len(p->sd[i].control);
#else
            const uint8_t *c = p->sd[i].control;
            if (!c || !p->sd[i].data) continue;
            uint32_t st;
            memcpy(&st, c + 0x714, 4);
            if (st < 6 && m->idx[st] < 0) m->idx[st] = (int8_t)i, memcpy(&m->len[st], c + 0x704, 4);
#endif
        }
    }
    int i = m->idx[nvn_stage];
    if (i < 0) return 0;
    *ctl = p->sd[i].control, *code = (const uint8_t *)(uintptr_t)p->sd[i].data, *len = m->len[nvn_stage];
    return 1;
}

// Texture and image types of a stage's first 32 slots, each kept with the handle and the generation and slot token it
// was looked up under: a slot is looked up again, a hash probe, only when the handle changed or its slot may resolve
// otherwise (slot_still).
static _Thread_local struct { uint64_t tex[32], img[32], ttok[32], itok[32]; uint32_t tgen[32], igen[32]; uint8_t tt[32], it[32], ifmt[32]; } AC[6];

// vk_texture_slot_token of the slot a texture or image handle names; an empty handle always resolves to nothing.
static uint64_t token_of(uint64_t h) { return h ? vk_texture_slot_token((int)(h & 0xfffff)) : 1; }

// Whether what handle h resolved to under the global generation *egen and its slot token etok (both read before that
// lookup) still holds at global generation gen: nothing at all changed since, or not its slot (the token, read only when
// the global generation moved, about every 16 draws). A token hit moves *egen to gen.
static int slot_still(uint32_t *egen, uint64_t etok, uint64_t h, uint32_t gen) {
    if (*egen == gen) return 1;
    uint64_t t = token_of(h);
    if (!t || t != etok) return 0;
    *egen = gen;
    return 1;
}
static int shader_memo_on(void);

// ---- shared lookups (REKARINA_TEX_MEMO_SHARED, with the shared descriptor memo TDS below): small results of texture slot
// lookups that each recording thread kept in a memo of its own (a slot's texture and image types for answers_for, the
// sampled subresource for feedback_scan), shared between threads the same way: written under the texture lock, read
// without it under a sequence word, valid while the slot's token and the generation hold (slot_still).
#define SHM_N 8192
typedef struct { volatile uint32_t seq; uint32_t gen; uint64_t key, tok; uint8_t data[24]; } Shm;
static Shm shm_ac[SHM_N], shm_fm[SHM_N];
static int tds_mode(void);
static inline Shm *shm_at(Shm *t, uint64_t key) { return &t[(key * 0x9E3779B97F4A7C15ull) >> 51]; }   // 13 bits: SHM_N
static int shm_get(Shm *t, uint64_t key, uint64_t h, uint32_t gen, uint64_t *tok, void *out, size_t n) {
    Shm *s = shm_at(t, key);
    uint32_t q = __atomic_load_n(&s->seq, __ATOMIC_ACQUIRE);
    if (!q || (q & 1)) return 0;
    uint32_t g = s->gen;
    uint64_t k = s->key, tk = s->tok;
    uint8_t d[24];
    memcpy(d, s->data, n);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    if (__atomic_load_n(&s->seq, __ATOMIC_RELAXED) != q || k != key || !slot_still(&g, tk, h, gen)) return 0;
    memcpy(out, d, n);
    *tok = tk;
    return 1;
}
static void shm_put(Shm *t, uint64_t key, uint64_t tok, uint32_t gen, const void *data, size_t n) {   // texture lock held
    Shm *s = shm_at(t, key);
    uint32_t q = __atomic_load_n(&s->seq, __ATOMIC_RELAXED);
    __atomic_store_n(&s->seq, q + 1, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    s->gen = gen, s->key = key, s->tok = tok;
    memcpy(s->data, data, n);
    __atomic_store_n(&s->seq, q + 2, __ATOMIC_RELEASE);
}

// The slots of (program, stage) whose types its translation reads (gfx_shader_slot_mask), learned from its first
// shader; ~0u while unknown. The types of other slots select the same shader, so answers_for leaves them zero instead
// of looking up all 64 handles. Direct-mapped by serial and stage; a collision only relearns.
#define SLOTM_N 4096
static _Thread_local struct { uint64_t serial; int st; uint32_t m; } slotm[SLOTM_N];

static uint32_t slotm_idx(uint64_t serial, int st) { return (uint32_t)((serial * 8 + (uint64_t)st) * 0x9E3779B97F4A7C15ull >> 52); }

static uint32_t slot_mask_of(uint64_t serial, int st) {
    const typeof(slotm[0]) *e = &slotm[slotm_idx(serial, st)];
    return serial && e->serial == serial && e->st == st ? e->m : ~0u;
}

static void slot_mask_learn(uint64_t serial, int st, const GfxShader *s) {
    if (!serial || slot_mask_of(serial, st) != ~0u) return;
    slotm[slotm_idx(serial, st)] = (typeof(slotm[0])){serial, st, gfx_shader_slot_mask(s)};
}

// Slot lookup for the helpers below: without the texture lock when the slot's resolution is current
// (vk_texture_by_id_fast), else under it, taken here once (*locked).
static const VkTexImage *by_id_try(uint64_t h, view_t *v, int *hv, int *locked) {
    int hit;
    const VkTexImage *img = vk_texture_by_id_fast((int)(h & 0xfffff), v, hv, &hit);
    if (hit) return img;
    if (!*locked) vk_texture_lock(), *locked = 1;
    return vk_texture_by_id_locked((int)(h & 0xfffff), v, hv);
}

static uint8_t type_of_handle_(uint64_t h, int *locked) {
    view_t v;
    int hv;
    const VkTexImage *img = h ? by_id_try(h, &v, &hv, locked) : NULL;
    return img ? (uint8_t)gfx_type_of(img, &v, hv) : 0;
}
static uint8_t type_of_handle(uint64_t h) {   // texture lock held
    int locked = 1;
    return type_of_handle_(h, &locked);
}

#ifdef GAME_NS2
// The format an image slot stores through, as a GfxShaderAnswers.img_format code (Ryujinx TextureFormat order,
// gfx_shader.c fmt_names): the view's format when it reinterprets the texture (reinterpret_format), else the image's.
// Without it the translator takes every formatted store as R8G8B8A8Unorm: OoT's 3D frame stores RGB9E5 words through
// R32UI views of r11g11b10f and rgba8 images (translator patch 0027 declares the image from this answer). Texture lock.
static uint8_t img_format_of_handle_(uint64_t h, int *locked) {
    view_t v;
    int hv;
    const VkTexImage *img = h ? by_id_try(h, &v, &hv, locked) : NULL;
    if (!img) return 0;
    VkFormat f = reinterpret_format(img, &v, hv, 1);
    if (!f) f = vk_storage_format(img->format);
    switch (f) {
    case VK_FORMAT_R8_UNORM: return 1;              case VK_FORMAT_R8_SNORM: return 2;
    case VK_FORMAT_R8_UINT: return 3;               case VK_FORMAT_R8_SINT: return 4;
    case VK_FORMAT_R16_SFLOAT: return 5;            case VK_FORMAT_R16_UNORM: return 6;
    case VK_FORMAT_R16_SNORM: return 7;             case VK_FORMAT_R16_UINT: return 8;
    case VK_FORMAT_R16_SINT: return 9;              case VK_FORMAT_R32_SFLOAT: return 10;
    case VK_FORMAT_R32_UINT: return 11;             case VK_FORMAT_R32_SINT: return 12;
    case VK_FORMAT_R8G8_UNORM: return 13;           case VK_FORMAT_R8G8_SNORM: return 14;
    case VK_FORMAT_R8G8_UINT: return 15;            case VK_FORMAT_R8G8_SINT: return 16;
    case VK_FORMAT_R16G16_SFLOAT: return 17;        case VK_FORMAT_R16G16_UNORM: return 18;
    case VK_FORMAT_R16G16_SNORM: return 19;         case VK_FORMAT_R16G16_UINT: return 20;
    case VK_FORMAT_R16G16_SINT: return 21;          case VK_FORMAT_R32G32_SFLOAT: return 22;
    case VK_FORMAT_R32G32_UINT: return 23;          case VK_FORMAT_R32G32_SINT: return 24;
    case VK_FORMAT_R8G8B8A8_UNORM: case VK_FORMAT_R8G8B8A8_SRGB:
    case VK_FORMAT_B8G8R8A8_UNORM: case VK_FORMAT_B8G8R8A8_SRGB: return 25;
    case VK_FORMAT_R8G8B8A8_SNORM: return 26;       case VK_FORMAT_R8G8B8A8_UINT: return 27;
    case VK_FORMAT_R8G8B8A8_SINT: return 28;        case VK_FORMAT_R16G16B16A16_SFLOAT: return 29;
    case VK_FORMAT_R16G16B16A16_UNORM: return 30;   case VK_FORMAT_R16G16B16A16_SNORM: return 31;
    case VK_FORMAT_R16G16B16A16_UINT: return 32;    case VK_FORMAT_R16G16B16A16_SINT: return 33;
    case VK_FORMAT_R32G32B32A32_SFLOAT: return 34;  case VK_FORMAT_R32G32B32A32_UINT: return 35;
    case VK_FORMAT_R32G32B32A32_SINT: return 36;    case VK_FORMAT_A2B10G10R10_UNORM_PACK32: return 37;
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: return 38;
    case VK_FORMAT_B10G11R11_UFLOAT_PACK32: return 39;
    default: return 0;
    }
}
#endif

// The answers of stage st for the slots in mask; the types of other slots stay zero.
static void answers_for(VkExec *ex, int st, const uint8_t *ctl, uint32_t mask, GfxShaderAnswers *a) {
    memset(a, 0, sizeof *a);
    if (st == NVN_STAGE_VS) vk_pipeline_answers(&ex->pipe, a, NULL);
    else if (st == NVN_STAGE_FS) vk_pipeline_answers(&ex->pipe, NULL, a);
#ifndef GAME_NS2
    else
        for (int k = 0; k < 3; k++) {
            uint32_t l;
            memcpy(&l, ctl + 0x748 + 4 * k, 4);
            a->local[k] = (uint16_t)(l ? l : 1);
        }
#else
    else if (st == NVN_STAGE_CS) {
        uint32_t l[3];
        nvn_ctl_local(ctl, l);
        for (int k = 0; k < 3; k++) a->local[k] = (uint16_t)(l[k] ? l[k] : 1);
        uint32_t sh = nvn2_ctl_shared(ctl);   // NVN2 compute's shared memory (MacNativeGames: the request's "shared")
        if (sh) a->shared_bytes = (sh + 15) / 16 * 16;
    }
#endif
    typeof(AC[0]) *c = &AC[st];
    // The generation and tokens are read before the lookups (one that creates an entry changes them) and unlocked: a
    // registration racing with them is ordered after this draw, and a slot stored under older ones is only looked up
    // again.
    uint32_t gen = vk_texture_slots_gen();
    int locked = 0, check = shader_memo_on() == 2;
    for (uint32_t m = mask; m; m &= m - 1) {
        int slot = __builtin_ctz(m);
        uint64_t h = ex->tex[st][slot], hi = ex->img[st][slot];
        int tmiss = c->tex[slot] != h || !c->ttok[slot] || !slot_still(&c->tgen[slot], c->ttok[slot], h, gen);
        int imiss = c->img[slot] != hi || !c->itok[slot] || !slot_still(&c->igen[slot], c->itok[slot], hi, gen);
        if ((tmiss || imiss) && !check && tds_mode()) {   // what another thread looked up (shm_ac), without the lock
            uint8_t d[2];
            uint64_t tk;
            if (tmiss && h && shm_get(shm_ac, h << 1, h, gen, &tk, d, 1))
                c->tt[slot] = d[0], c->tex[slot] = h, c->ttok[slot] = tk, c->tgen[slot] = gen, tmiss = 0;
            if (imiss && hi && shm_get(shm_ac, hi << 1 | 1, hi, gen, &tk, d, 2)) {
                c->it[slot] = d[0], c->img[slot] = hi, c->itok[slot] = tk, c->igen[slot] = gen, imiss = 0;
#ifdef GAME_NS2
                c->ifmt[slot] = d[1];
#endif
            }
        }
        uint64_t tt = tmiss ? token_of(h) : 0, it = imiss ? token_of(hi) : 0;
        if ((tmiss || imiss || check) && !locked) BS.ac_miss++;
        if (check && !locked) vk_texture_lock(), locked = 1;
        // (the lookups take the lock only when a slot's resolution is not current: by_id_try)
        if (tmiss) c->tt[slot] = type_of_handle_(h, &locked), c->tex[slot] = h, c->ttok[slot] = tt, c->tgen[slot] = gen;
        if (imiss) c->it[slot] = type_of_handle_(hi, &locked), c->img[slot] = hi, c->itok[slot] = it, c->igen[slot] = gen;
#ifdef GAME_NS2
        if (imiss) c->ifmt[slot] = c->it[slot] ? img_format_of_handle_(hi, &locked) : 0;
#endif
        if (tds_mode() && locked && tmiss && h && tt) shm_put(shm_ac, h << 1, tt, gen, &c->tt[slot], 1);   // (writers hold the lock)
        if (tds_mode() && locked && imiss && hi && it) {
            uint8_t d[2] = {c->it[slot], 0};
#ifdef GAME_NS2
            d[1] = c->ifmt[slot];
#endif
            shm_put(shm_ac, hi << 1 | 1, it, gen, d, 2);
        }
        if (check && (type_of_handle(h) != c->tt[slot] || type_of_handle(hi) != c->it[slot]))   // REKARINA_SHADER_MEMO=check
            rt_fatal("vk_exec: kept texture types of stage %d slot %d differ from a lookup", st, slot);
        a->tex_type[slot] = c->tt[slot], a->img_type[slot] = c->it[slot];
#ifdef GAME_NS2
        a->img_format[slot] = c->ifmt[slot];
#endif
    }
    if (locked) vk_texture_unlock();
}

// *present: the program has this stage. Returns NULL when it has not or the translation failed. pending (graphics
// stages, vk_pipeline_async_on): a shader not translated yet is queued and NULL returned with *pending = 1. await: a
// shader queued earlier is waited for (gfx_shader_await) instead.
static _Thread_local int sf_mvprev;   // shader_for asks for the vertex shader's last-frame variant (per-object motion)
static int sf_msc;      // shader_for asks for the mesh stage as a compute kernel (mesh_emulate)

static const GfxShader *shader_for(VkExec *ex, int st, int *present, int *pending, int await) {
    const uint8_t *ctl, *code;
    uint32_t len;
    *present = 0;
    if (pending) *pending = 0;
    if (!ex->prog || !find_stage(ex->prog, st, &ctl, &code, &len)) return NULL;
    *present = 1;
    GfxShaderAnswers a;
    uint64_t serial = ex->prog->serial;
    answers_for(ex, st, ctl, slot_mask_of(serial, st), &a);
    a.mvprev = (uint8_t)(sf_mvprev && st == NVN_STAGE_VS);
    int msc = sf_msc && (st == NVN_STAGE_MS || st == NVN_STAGE_TS);
    int stk = st | (msc ? 0x100 : 0);   // the variant's own entries in the table
    ShaderEnt *set = NULL;
    if (serial) {
        uint64_t h = (serial * 0x9E3779B97F4A7C15ull ^ (uint64_t)stk) ^ nvn_hm_hash(&a, sizeof a);   // a is zero-padded
        set = shader_tab[(h ^ h >> 29) & (SHADER_SETS - 1)];
        for (int w = 0; w < SHADER_WAYS && set[w].used; w++) {
            ShaderEnt *e = &set[w];
            if (e->serial != serial || e->stage != stk || memcmp(&e->a, &a, sizeof a)) continue;
            if (w) {   // to the front
                ShaderEnt hit = *e;
                memmove(&set[1], &set[0], w * sizeof *set);
                set[0] = hit;
            }
            return set[0].s;
        }
    }
#ifdef GAME_NS2
    int gs = st == NVN_STAGE_VS ? GFX_STAGE_VS : st == NVN_STAGE_FS ? GFX_STAGE_FS : st == NVN_STAGE_GS ? GFX_STAGE_GS
           : st == NVN_STAGE_TS ? (msc ? GFX_STAGE_TSC : GFX_STAGE_TS) : st == NVN_STAGE_MS ? (msc ? GFX_STAGE_MSC : GFX_STAGE_MS) : GFX_STAGE_CS;
#else
    int gs = st == NVN_STAGE_VS ? GFX_STAGE_VS : st == NVN_STAGE_FS ? GFX_STAGE_FS : GFX_STAGE_CS;
#endif
    const GfxShader *s = await ? gfx_shader_await(gs, ctl, code, len, &a)
                       : pending && (vk_pipeline_async_on() || a.mvprev) ? gfx_shader_get_async(gs, ctl, code, len, &a, pending)
                                                           : gfx_shader_get(gs, ctl, code, len, NULL, &a);
    if (pending && *pending) return NULL;
    if (s) slot_mask_learn(serial, st, s);
    if (!s) warn_once(W_SHADER, "a stage %d shader of program %llu failed to translate: draws with it are skipped", st,
                      (unsigned long long)serial);
    else if (set) {   // in front, the least recently used way drops out
        memmove(&set[1], &set[0], (SHADER_WAYS - 1) * sizeof *set);
        set[0] = (ShaderEnt){.serial = serial, .stage = stk, .used = 1, .a = a, .s = s};
    }
    return s;
}

// ---- descriptors

static int nvn_stage_of(int gfx_stage) {
#ifdef GAME_NS2
    if (gfx_stage == GFX_STAGE_GS) return NVN_STAGE_GS;
    if (gfx_stage == GFX_STAGE_TS) return NVN_STAGE_TS;
    if (gfx_stage == GFX_STAGE_MS || gfx_stage == GFX_STAGE_MSC) return NVN_STAGE_MS;   // MSC: the mesh stage's resources
    if (gfx_stage == GFX_STAGE_TSC) return NVN_STAGE_TS;
#endif
    return gfx_stage == GFX_STAGE_VS ? NVN_STAGE_VS : gfx_stage == GFX_STAGE_FS ? NVN_STAGE_FS : NVN_STAGE_CS;
}

static VkDescriptorBufferInfo zero_info(uint64_t max) {
    return (VkDescriptorBufferInfo){Z.zbuf, 0, max < ZERO_BYTES ? max : ZERO_BYTES};
}

// ---- descriptors kept per bound slot (REKARINA_SLOT_CACHE: 0 off, 1 on, check)
// A graphics draw resolves each of its bindings from the slot it names: a uniform range through the submission's
// snapshot or its pool, a texture through the descriptor memo (TD). Between two draws only a few slots are bound
// again (OoT: 3.3 bind ops for the dozen descriptors of a draw), yet every draw looked every slot up. Here
// the descriptor made for a slot is kept beside it and taken again while the slot's value is the same and nothing it
// was made from has moved: the stamp stands for the texture slot generation and write-watch epoch, the sampler
// generation, the memory and snapshot generations, and the command buffer with its submission (the notes a lookup
// leaves for the part's prologue, the census and the mirror sync are per command buffer and repeat as no-ops inside
// one: the argument of the whole-draw skip above bind_skip_on). Kept: uniform slots 3 and up (not the driver's own
// banks) and sampled textures while no image is in the feedback layout and DLSS swaps no light target. Not kept:
// storage buffers and images (they note their writes), a range copied into transient memory (RS.volatile_), per-object
// motion's overrides. A hit repeats the census count and takes no texture lock.
// check: every hit is resolved again and compared; a difference stops the run.
typedef struct { uint64_t stamp, h; uint16_t type; uint8_t separate; VkDescriptorImageInfo out; } ScTex;
typedef struct { uint64_t stamp, addr, size; VkDescriptorBufferInfo out; int cls, use; uint64_t n; } ScBuf;
static _Thread_local struct {
    int on;
    uint64_t stamp, key[4], hits, misses;
    ScTex tex[6][128];
    ScBuf ubo[6][18];
} SC = {.on = -1, .stamp = 1};
static _Thread_local struct { int on, cls, use; uint64_t n; } sc_cap;   // count_use's arguments while a slot's descriptor is made

static int slot_cache_on(void) {
    if (SC.on < 0) {
        const char *e = cfg_get("REKARINA_SLOT_CACHE");
        SC.on = !e || !*e ? 0 : !strcmp(e, "check") ? 2 : atoi(e) != 0;
        if (SC.on) rt_log("vk_exec: descriptors kept per bound slot%s (REKARINA_SLOT_CACHE)", SC.on == 2 ? ", every hit checked" : "");
    }
    return SC.on;
}

// The stamp of this build's generations; a change of any makes every kept descriptor stale.
static void slot_cache_stamp(const VkExec *ex) {
    uint64_t k[4] = {(uint64_t)vk_texture_slots_gen() << 32 | vk_texture_epoch(), vk_sampler_gen(), vk_mem_gen() + vk_usnap_gen(),
                     (uint64_t)(uintptr_t)ex->cb ^ ex->fence << 1};
    if (memcmp(k, SC.key, sizeof k)) memcpy(SC.key, k, sizeof k), SC.stamp++;
    if (!((SC.hits + SC.misses + 1) & 0x3fffff))
        rt_log("vk_exec: slot descriptors: %llu taken again, %llu made", (unsigned long long)SC.hits, (unsigned long long)SC.misses);
}

static void vk_exec_slot_cache_drop(void) { SC.stamp++; }

// vk_mem_count, also kept per (class, use) while a graphics build runs (RS).
static void count_use(int cls, int use, uint64_t n) {
    vk_mem_count(cls, use, n);
    if (sc_cap.on) sc_cap.cls = cls, sc_cap.use = use, sc_cap.n += n;
    if (!RS.rec) return;
    for (int i = 0; i < RS.ncnt; i++)
        if (RS.cnt[i].cls == cls && RS.cnt[i].use == use) {
            RS.cnt[i].bytes += n;
            return;
        }
    if (RS.ncnt < (int)(sizeof RS.cnt / sizeof RS.cnt[0])) RS.cnt[RS.ncnt++] = (RsCount){cls, use, n};
    else RS.cnt_over = 1;
}

// A buffer descriptor for [addr, addr+size) of guest memory: its backing (a pool's VRAM mirror when it has one) when
// aligned, a copy in transient memory when the backing is host memory with an unaligned offset. 0 when neither works.
// written: the shader stores into it (a mirror's range goes back to guest memory at the end of the submission); 2: it
// adds into it with float atomics (GFX_SSBO_FADD: the range goes back as additions).
static _Thread_local int bind_owned;   // the last bind_range of a written range: a mirror holds it (vk_mem_gpu_wrote returned 1)
static int bind_range(VkExec *ex, uint64_t addr, uint64_t size, uint64_t align, uint64_t max, int use, int written,
                      VkDescriptorBufferInfo *out) {
    VkMemSpan sp;
    bind_owned = 0;
    if (!addr || !vk_mem_resolve(addr, 1, &sp)) return 0;
    uint64_t n = size && size < sp.avail ? size : sp.avail;
    if (n > max) n = max;
    if (!n) return 0;
    if (sp.offset % align == 0) {
        VK_MEM_USE(sp);
        *out = (VkDescriptorBufferInfo){sp.buffer, sp.offset, n};
        if (written) {
            plat_mutex_lock(&vk_par_lock);   // a chunk of a draw stretch and the replay thread may both note one
            bind_owned = written == 2 ? vk_mem_gpu_wrote_adds(&sp, n) : vk_mem_gpu_wrote_shader(&sp, n);
            vk_usnap_drop(addr, n);
            plat_mutex_unlock(&vk_par_lock);
        }
        count_use(sp.cls, use, n);
        VK_MEM_BOUND(sp);
        return 1;
    }
    if (sp.alloc) return 0;
    VkBuffer b;
    VkDeviceSize o;
    RS.volatile_ = 1;
    memcpy(scratch(ex, n, &b, &o), (const void *)(uintptr_t)addr, n);
    vk_mem_count(sp.cls, VK_MUSE_CPU_READ, n);
    vk_mem_count(VK_MCLS_SCRATCH, use, n);
    *out = (VkDescriptorBufferInfo){b, o, n};
    return 1;
}

static void c0_table(uint8_t *at, const p_bind_buf b[16]) {
    for (int r = 0; r < 16; r++) {
        uint64_t a = b[r].addr;
        uint32_t n = (uint32_t)b[r].size;
        memcpy(at + 16 * r, &a, 8);
        memcpy(at + 16 * r + 8, &n, 4);
    }
}

// The driver constant bank c0 as GLSLC-compiled shaders read it. Storage-buffer tables {u64 address, u32 size} every
// 16 bytes at 0x110 (vertex), 0x510 (fragment), 0x310 (compute): Ryujinx's GlobalToStorage fallback loads each
// candidate base from here and compares the global pointer against it. Compute: dispatch size at 0x200, the uniform
// buffer table at 0x210 + 16 r. Graphics: the largest point size (Maxwell's limit, GLSLC clamps gl_PointSize to it) at
// 0x610, the base instance at 0x644. Other stages: zeros. Unbound slots read zero, since the replay keeps them zero.
// Layout: nxnative dm/gfx/nvn/nvn_mtl.m (MIT).
void vk_c0_fill(uint8_t bank[VK_C0_BYTES], int stage, const p_bind_buf ssbo[16], const p_bind_buf ubo[16],
                const uint32_t dispatch[3], uint32_t base_instance) {
    memset(bank, 0, VK_C0_BYTES);
    int base = stage == NVN_STAGE_VS ? 0x110 : stage == NVN_STAGE_FS ? 0x510 : stage == NVN_STAGE_CS ? 0x310 : -1;
#ifdef GAME_NS2
    // seedds/nxnative dmsh_abi.h (MIT): the geometry stage's table at 0x410 (NieR:Automata), NVN2's task stage's at
    // 0x670 and its mesh stage's at 0x770 (Mario Kart World's task and mesh shaders)
    if (stage == NVN_STAGE_GS) base = 0x410;
    else if (stage == NVN_STAGE_TS) base = 0x670;
    else if (stage == NVN_STAGE_MS) base = 0x770;
#endif
    if (base < 0) return;
    c0_table(bank + base, ssbo);
    if (stage == NVN_STAGE_CS) {
        memcpy(bank + 0x200, dispatch, 12);
#ifdef GAME_NS2
        // NVN2 compute also reads the group counts below its per-shader layout at 0x10: a light culling shader's cluster
        // is (z * c0[8] + y) * c0[4] + x (seedds/nxnative DMSH_C0_DISPATCH_NVN2, Fire Emblem Fortune's Weave)
        {   // DEBUG-C0DIM: REKARINA_C0_DIM_AT=<byte>: where the group counts go (4 as above; 0: the other reading, tried for OoT's fog)
            static int at = -1;
            if (at < 0) { const char *e = cfg_get("REKARINA_C0_DIM_AT"); at = e ? atoi(e) : 4; }
            memcpy(bank + at, dispatch, 12);
        }
#endif
        c0_table(bank + 0x210, ubo);
        return;
    }
    const float max_point = 2047.0f;
    memcpy(bank + 0x610, &max_point, 4);
    memcpy(bank + 0x644, &base_instance, 4);
}

static int c0_on(void) {   // REKARINA_FIX_C0=0: c0 reads zeros instead of the driver constants
    static int on = -1;
    if (on < 0) {
        on = cfg_bool("REKARINA_FIX_C0");
    }
    return on;
}

// c0 from replay scratch; a bank equal to the stage's previous one reuses its range, so it costs no copy.
static VkDescriptorBufferInfo c0_info(VkExec *ex, int st) {
    int k = C0_K(st);
    // vk_c0_fill reads the storage slots, and the uniform slots and dispatch size for compute or the base instance for
    // graphics: equal inputs give the same bank, so neither the fill nor the 2 KiB compare is needed.
    // Compared field by field against the slots themselves: building a copy of the inputs to compare it (784 bytes
    // zeroed, filled and compared) was 1,700 times a frame in the forest, 4 MiB of memory traffic.
    typeof(B.c0_in[0]) *pin = &B.c0_in[k];
    int cs = st == NVN_STAGE_CS;
    uint32_t dim0 = cs ? c0_dim[0] : 0;
#ifdef GAME_NS2
    if (st == NVN_STAGE_TS || st == NVN_STAGE_MS) dim0 = c0_mesh_first;   // the mesh draw's first workgroup
#endif
    int same = B.c0_valid[k] && !memcmp(pin->ssbo, ex->ssbo[st], sizeof pin->ssbo) && pin->dim[0] == dim0 &&
               (cs ? !memcmp(pin->ubo, ex->ubo[st], sizeof pin->ubo) && pin->dim[1] == c0_dim[1] && pin->dim[2] == c0_dim[2]
                   : pin->binst == c0_binst);
    if (same) return (VkDescriptorBufferInfo){B.c0_buf[k], B.c0_off[k], VK_C0_BYTES};
    memcpy(pin->ssbo, ex->ssbo[st], sizeof pin->ssbo);
    pin->dim[0] = dim0;
    if (cs) memcpy(pin->ubo, ex->ubo[st], sizeof pin->ubo), pin->dim[1] = c0_dim[1], pin->dim[2] = c0_dim[2];
    else pin->binst = c0_binst;
    uint8_t tmp[VK_C0_BYTES];
    vk_c0_fill(tmp, st, ex->ssbo[st], ex->ubo[st], c0_dim, c0_binst);
#ifdef GAME_NS2
    if (st == NVN_STAGE_TS || st == NVN_STAGE_MS) memcpy(tmp + VK_C0_MESH_FIRST, &c0_mesh_first, 4);
#endif
    if (!B.c0_valid[k] || memcmp(tmp, B.c0[k], VK_C0_BYTES)) {
        memcpy(scratch(ex, VK_C0_BYTES, &B.c0_buf[k], &B.c0_off[k]), tmp, VK_C0_BYTES);
        memcpy(B.c0[k], tmp, VK_C0_BYTES);
        B.c0_valid[k] = 1;
    }
    return (VkDescriptorBufferInfo){B.c0_buf[k], B.c0_off[k], VK_C0_BYTES};
}

// The buffer range limits, read once: the device's properties are a cold cache line on every binding of every draw.
static struct { uint64_t ubo_max, ubo_align, sbo_max, sbo_align; } LIM;

static void limits_init(void) {
    const VkPhysicalDeviceLimits *l = &vk_ctx()->props.limits;
    LIM.ubo_align = l->minUniformBufferOffsetAlignment, LIM.sbo_max = l->maxStorageBufferRange;
    LIM.sbo_align = l->minStorageBufferOffsetAlignment;
    __atomic_store_n(&LIM.ubo_max, l->maxUniformBufferRange, __ATOMIC_RELEASE);
}

// Per-object motion (section at the end): while it records a draw again, these replace the uniform bindings of the
// vertex stage's slots 18-20 (last frame's camera, bones and place) and the fragment stage's slot 3 (its parameters).
static _Thread_local int mo_ovr_on;
static _Thread_local VkDescriptorBufferInfo mo_ovr[2][32];   // [vertex, fragment][slot]; buffer 0: not replaced

static VkDescriptorBufferInfo ubo_info(VkExec *ex, int st, int slot) {
    if (mo_ovr_on && (st == NVN_STAGE_VS || st == NVN_STAGE_FS) && slot >= 0 && slot < 32 && mo_ovr[st][slot].buffer)
        return mo_ovr[st][slot];
    if (!__atomic_load_n(&LIM.ubo_max, __ATOMIC_ACQUIRE)) limits_init();
    uint64_t max = LIM.ubo_max, align = LIM.ubo_align;
    VkDescriptorBufferInfo bi;
    if (slot == 0 && c0_on()) return c0_info(ex, st);
#ifdef GAME_NS2
    if (slot == NVN2_PUSH_BANK) {   // NVN2 push constants: this draw's or dispatch's copy of the stage group's block
        VkBuffer pb;
        VkDeviceSize po;
        memcpy(scratch(ex, NVN2_PUSH_SIZE, &pb, &po), ex->push[st == NVN_STAGE_CS], NVN2_PUSH_SIZE);
        return (VkDescriptorBufferInfo){pb, po, NVN2_PUSH_SIZE};
    }
#endif
    if (slot == 1 && ex->prog) {   // compiler constants stored after the shader code
        const uint8_t *ctl, *code;
        uint32_t len, c1;
        if (find_stage(ex->prog, st, &ctl, &code, &len)) {
            c1 = nvn_ctl_c1_off(ctl);
            if (c1 < len && bind_range(ex, (uint64_t)(uintptr_t)code + c1, len - c1, align, max, VK_MUSE_UBO, 0, &bi))
                return bi;
        }
    } else if (slot >= 3 && slot - 3 < 18) {
        const p_bind_buf *b = &ex->ubo[st][slot - 3];
        if (b->size && vk_usnap_find(b->addr, b->size < max ? b->size : max, align, &bi)) {   // as of the submit call
            count_use(VK_MCLS_SCRATCH, VK_MUSE_UBO, bi.range);
            return bi;
        }
        if (bind_range(ex, b->addr, b->size, align, max, VK_MUSE_UBO, 0, &bi)) return bi;
    }
    return zero_info(max);
}

// DEBUG-PROBE: the buffer a stage's uniform slot binds right now (vk_dump.c vk_probe_draw), and the support buffer.
VkDescriptorBufferInfo vk_exec_probe_ubo(VkExec *ex, int st, int slot) { return ubo_info(ex, st, slot); }
VkDescriptorBufferInfo vk_exec_probe_support(void) {
    return B.sup_valid ? (VkDescriptorBufferInfo){B.sup_buf, B.sup_off, SUPPORT_BYTES} : (VkDescriptorBufferInfo){0};
}

// The NVN slot of a storage binding. Ryujinx numbers a shader's storage buffers in the order it meets them
// (GfxShaderRes.slot); the NVN slot is the entry of the driver constant buffer c0 that holds the buffer's address
// (cbuf_slot 0, handle = word offset): 16-byte entries at byte 0x110 (vertex), 0x510 (fragment) and 0x310 (compute).
// A compute address at c0 0x210..0x30f is uniform block (offset - 0x210) / 16 read as storage: *ubo is set. -1: the
// address is not in c0 (the shader loads it from a user uniform block: ubo_ptr) or the stage has no known table.
// Mapping: MacNativeGames games/totk/tools/totk_xlate/Service.cs:206-222 @ b75be8cf (MIT, Copyright (c) 2026 jhaizhou).
static int ssbo_slot(const GfxShaderRes *r, int st, int *ubo) {
    *ubo = 0;
    if (r->cbuf_slot != 0) return -1;
    int off = r->handle * 4, base = st == NVN_STAGE_VS ? 0x110 : st == NVN_STAGE_FS ? 0x510 : st == NVN_STAGE_CS ? 0x310 : -1;
#ifdef GAME_NS2
    if (st == NVN_STAGE_GS) base = 0x410;   // vk_c0_fill
    else if (st == NVN_STAGE_TS) base = 0x670;
    else if (st == NVN_STAGE_MS) base = 0x770;
#endif
    if (st == NVN_STAGE_CS && off >= 0x210 && off < 0x310 && (off - 0x210) % 16 == 0) {
        *ubo = 1;
        return (off - 0x210) / 16;
    }
    return base >= 0 && off >= base && off < base + 0x100 && (off - base) % 16 == 0 ? (off - base) / 16 : -1;
}

// 1 when the shader loads the storage buffer's address from a user uniform block (cbuf_slot 3.., NVN uniform slot
// cbuf_slot - 3, byte offset handle * 4). The Depths terrain reads its vertices through one, about 500 draws a frame;
// with a zero buffer they collapse and the ground is missing. MacNativeGames resolves these the same way (NVN_RB_SBUFU,
// games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m @ b75be8cf).
static int ubo_ptr_res(const GfxShaderRes *r) {
    return r->cbuf_slot >= 3 && r->cbuf_slot - 3 < 18;
}

// The inline uniform updates of the submission being replayed, newest last (a ring of UPD_N). op_update_ub writes them
// into the buffer's GPU copy; guest memory gets them only from the write-back when the submission ends. The terrain
// rewrites its pointer block this way before every draw, all at one address, so ubo_ptr reads the latest update that
// covers the pointer, else memory. Payloads stay in the recorded command buffer for the whole replay.
#define UPD_N 4096
static struct {
    uint64_t fence;                      // the submission the entries belong to
    uint32_t n;                          // updates noted in it
    struct { uint64_t lo, hi; const uint8_t *data; } u[UPD_N];
} UPD;

// A chunk of a draw stretch reads the updates noted before its stretch was handed out (vk_exec_draw_upd_limit): the
// replay thread may note later ones meanwhile (it no longer waits for the chunks at an update).
static _Thread_local uint32_t upd_lim = UINT32_MAX;
// (the ring holds another submission's updates until this one notes its first: none of this one's then)
uint32_t vk_exec_draw_upd_count(uint64_t fence) { return UPD.fence == fence ? UPD.n : 0; }
void vk_exec_draw_upd_limit(uint32_t n) { upd_lim = n; }

static void upd_note(const VkExec *ex, const p_update_ub *u) {
    if (UPD.fence != ex->fence) UPD.fence = ex->fence, UPD.n = 0;
    UPD.u[UPD.n % UPD_N] = (typeof(UPD.u[0])){u->dst, u->dst + u->size, u->data};
    __atomic_store_n(&UPD.n, UPD.n + 1, __ATOMIC_RELEASE);
}

// 1 when an update of this submission supplied [a, a+n).
static int upd_read(const VkExec *ex, uint64_t a, void *out, size_t n) {
    if (UPD.fence != ex->fence) return 0;
    uint32_t top = __atomic_load_n(&UPD.n, __ATOMIC_ACQUIRE), win = UPD_N;
    if (upd_lim != UINT32_MAX) {
        if (top - upd_lim >= UPD_N / 2) return 0;   // the ring went round meanwhile (never seen): memory
        top = upd_lim;
        win = UPD_N / 2;   // the replay thread writes the entries above the limit: those of the older half are kept
    }
    for (uint32_t i = top, end = top > win ? top - win : 0; i-- > end;) {
        const typeof(UPD.u[0]) *e = &UPD.u[i % UPD_N];
        uint64_t lo = e->lo, hi = e->hi;
        const uint8_t *d = e->data;
        if (a >= hi || a + n <= lo) continue;
        if (a < lo || a + n > hi) return 0;   // partly covered: memory
        memcpy(out, d + (a - lo), n);
        // (a chunk: the entry it read was not written again meanwhile)
        return upd_lim == UINT32_MAX || __atomic_load_n(&UPD.n, __ATOMIC_ACQUIRE) - upd_lim < UPD_N / 2;
    }
    return 0;
}

// That address as the bound block holds it at this point of the replay; 0 when the block is unbound or too short. The
// block has no size field.
static uint64_t ubo_ptr(const VkExec *ex, int st, const GfxShaderRes *r) {
    const p_bind_buf *u = &ex->ubo[st][r->cbuf_slot - 3];
    uint64_t off = (uint64_t)r->handle * 4, a = 0;
    VkMemSpan sp;
    if (!u->addr || u->size < off + 8 || upd_read(ex, u->addr + off, &a, 8)) return a;
    if (vk_mem_resolve(u->addr + off, 8, &sp)) memcpy(&a, (const void *)(uintptr_t)(u->addr + off), 8);
    return a;
}

#ifdef GAME_NS2
// ---- NVN2 global memory at loaded addresses (docs/oot-xlate.md "gmem_map", "Reply")
// An Ampere shader that loads or stores at an address it read from memory, which the translator tied to no storage
// buffer, looks the address up in the pool table and goes through a PhysicalStorageBuffer pointer at address + delta
// (vk_memory.c vk_mem_gmem_table). Its reflection lists two storage buffers under cb_slot 255 (GFX_CB_GMEM): the table
// (cb_offset 0: 8192 entries of 8 u32 {start lo, hi, end lo, hi, delta lo, hi, 0, 0}, sorted by start, entry 0
// empty, unused entries starting at ~0, then the index the shaders find an entry by: GMEM_L1_WORDS chunk words, one
// per 2 MiB below 2^40, the entry whose range holds the whole chunk or GMEM_SPLIT | n for a chunk several entries
// share, n naming the n-th page table after them, GMEM_L2_WORDS words, one per 4 KiB page, the page's entry; 0 none:
// translator patch 0032) and its write log (cb_offset 1: a bitmap of 8192 bits, a store or atomic through
// entry e sets bit e). Ported from seedds/nxnative dm/gfx/nvn/nvn_mtl.m gmem_map / gml_take @ ee9b776 (MIT; the hint
// rule and the bitmap, docs/reference/frame-rate.md "A full-frame shader overflows the raw write log"), written for
// this replay. The table lives in a buffer of its own per generation of the pools and their mirrors, filled once, in
// device-local memory where the device has host-visible VRAM (every access reads it: with the table and logs in host
// memory, OoT's movie network took 120-140 ms of GPU time a frame, 86 ms with them in VRAM, 9 ms with its pool's
// accesses in the mirror too; vk_memory.c). Every submission gets one zeroed log per table it uses from a ring (in VRAM
// too: the stores' atomics stay on the GPU), and once the submission is done the entries whose bits are set count as
// written by the GPU: in guest memory (an import buffer) vk_gpu_written tells the texture cache and the mirrors, as
// copies and storage writes report theirs; in a device copy (a mirror, device memory) a write is only counted. Guest
// entries are split at 2 MiB boundaries (larger when the table would overflow), so a write marks its own 2 MiB rather
// than a whole pool.
#define GMEM_ENTRIES 8192             // the translator's GmemMapLog2 (patch 0029): the forest has some 4,140 pools
#define GMEM_ENTRY_WORDS 8
#define GMEM_TABLE_BYTES (GMEM_ENTRIES * GMEM_ENTRY_WORDS * 4)
#define GMEM_LOG_BYTES (GMEM_ENTRIES / 8)
#define GMEM_LOGS 256                    // logs in flight at most: one per submission and table, freed at its fence
#define GMEM_NO_LOG GMEM_LOGS            // every log was in flight: a transient one that no one reads
#define GMEM_HINT_SHIFT 21               // a chunk word per 2 MiB (the translator's GmemHintShift)
#define GMEM_INDEX_END (1ull << 40)      // the index covers addresses below this
#define GMEM_L1_BASE (GMEM_ENTRIES * GMEM_ENTRY_WORDS)              // in words, after the entries
#define GMEM_L1_WORDS ((uint32_t)(GMEM_INDEX_END >> GMEM_HINT_SHIFT))
#define GMEM_L2_BASE (GMEM_L1_BASE + GMEM_L1_WORDS)
#define GMEM_L2_WORDS (1u << (GMEM_HINT_SHIFT - 12))
#define GMEM_SPLIT 0x80000000u           // a chunk several entries share: the low bits name its page table
#define GMEM_PIECE (2ull << 20)          // guest pools are split at multiples of this, at least

// A table's buffer, kept for a later table once no submission uses it: the pools change every few frames while a scene
// loads and plays (512 tables in 30 s on the forest porch), and a new 2.6 MiB device-local buffer each time cost 2 ms a
// table, up to 112 ms when VMA took a new block. Only the chunk words it or the new table set are written (nz: the
// chunk words it holds that are not 0); the entries and page tables are copied whole.
typedef struct GmemBuf {
    struct GmemBuf *next;
    VkBuffer buf;
    VmaAllocation alloc;
    uint8_t *map;
    uint64_t cap;                        // bytes
    uint32_t *nz, nnz, nzcap;
} GmemBuf;
static PlatMutex gmem_free_lock = PLAT_MUTEX_INIT;
static GmemBuf *gmem_free;               // kept buffers (gmem_unref, any thread)
static int gmem_nfree;

typedef struct GmemSnap {                // one table: its buffer, and its entries for reading the logs that used it
    uint64_t gen;
    int n;
    volatile int32_t refs;               // the current table's own reference, plus one per submission using it
    GmemBuf *gb;
    uint64_t tbytes;                     // the entries and the index
    VkMemGmemEnt e[GMEM_ENTRIES];
} GmemSnap;
typedef struct { uint32_t slot; GmemSnap *snap; VkBuffer sbuf; VkDeviceSize soff; } GmemUse;
typedef struct { int n, cap; GmemUse *v; uint64_t fence; } GmemPend;   // the tables and logs of one submission, read at its fence

static struct {
    GmemSnap *cur;
    VkBuffer lbuf;                       // GMEM_LOGS logs, host-visible and coherent
    VmaAllocation lalloc;
    uint8_t *lmap;
    volatile uint8_t busy[GMEM_LOGS];
    uint32_t next;
    int failed;
    GmemPend *pend;                      // the submission being recorded
    uint64_t pend_fence;
    uint64_t logs, writes, dev_writes, full;   // statistics (completion thread for writes)
} GM;

// A mapped storage buffer, device-local where the device has host-visible VRAM, else in host memory. *where: which.
static VkBuffer gmem_buffer(VkDeviceSize size, VmaAllocation *a, uint8_t **map, const char **where) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                                  .flags = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                                  .requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
    VmaAllocationInfo info;
    VkBuffer b = VK_NULL_HANDLE;
    VmaAllocator vma = vk_ctx()->vma;
    if (vmaCreateBuffer(vma, &bi, &ai, &b, a, &info) != VK_SUCCESS) return VK_NULL_HANDLE;
    if (!info.pMappedData) {
        vmaDestroyBuffer(vma, b, *a);
        return VK_NULL_HANDLE;
    }
    VkMemoryPropertyFlags f = 0;
    vmaGetAllocationMemoryProperties(vma, *a, &f);
    *where = (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "device-local" : "host memory";
    *map = info.pMappedData;
    return b;
}

static void gmem_unref(GmemSnap *s) {
    if (!s || __atomic_sub_fetch(&s->refs, 1, __ATOMIC_ACQ_REL)) return;
    GmemBuf *gb = s->gb;
    if (gb) {
        plat_mutex_lock(&gmem_free_lock);
        if (gmem_nfree < 8) gb->next = gmem_free, gmem_free = gb, gmem_nfree++, gb = NULL;
        plat_mutex_unlock(&gmem_free_lock);
    }
    if (gb) {
        vmaDestroyBuffer(vk_ctx()->vma, gb->buf, gb->alloc);
        free(gb->nz);
        free(gb);
    }
    free(s);
}

static GmemSnap *gmem_snapshot_build(uint64_t gen);
// The table a submission binds is the one of its first use, until the replay itself changes a mirror or a wait for the
// CPU is met (vk_exec_gmem_unpin). The game's other threads create pools while the replay runs, 3-5 a frame for
// some 200 frames after a scene appears, and each made the next draw rebuild the table (0.5 ms): 1-2 ms of every
// such frame. A pool created after the submit call cannot be named by commands recorded before it, and one that goes
// keeps its buffer until the submission's fence. REKARINA_GMEM_PIN=0: rebuilt at every pool change, as before.
static struct { uint64_t fence, mgen; int on, unpin; } GP = {.on = -1};
void vk_exec_gmem_unpin(void) { GP.unpin = 1; }

static GmemSnap *gmem_snapshot(VkExec *ex) {
    uint64_t mg = vk_mem_gmem_mgen(), gen = vk_mem_gmem_gen();
    if (GP.on < 0) GP.on = rt_knob("GMEM_PIN",
#ifdef GAME_OOT
                                   "1"
#else
                                   "0"
#endif
                                   );
    if (GM.cur && (GM.cur->gen == gen || (GP.on && GP.fence == ex->fence && GP.mgen == mg && !GP.unpin))) {
        if (GP.fence != ex->fence) GP.fence = ex->fence, GP.mgen = mg, GP.unpin = 0;   // the submission's first use
        return GM.cur;
    }
    GP.fence = ex->fence, GP.mgen = mg, GP.unpin = 0;
    {   // DEBUG-GMEMTIME: the cost of the table's rebuilds
        static uint64_t n, ns, mx;
        uint64_t t0 = plat_now_ns();
        GmemSnap *s = gmem_snapshot_build(gen);
        uint64_t dt = plat_now_ns() - t0;
        vk_spike.gmem += dt;   // DEBUG-SPIKE
        n++, ns += dt;
        if (dt > mx) mx = dt;
        if (n % 64 == 0) rt_log("GMEMTIME %llu rebuilds, %.3f ms each, %.3f ms at most", (unsigned long long)n, ns / 1e6 / (double)n, mx / 1e6);
        return s;
    }
}
static GmemSnap *gmem_snapshot_build(uint64_t gen) {
    GmemSnap *s = calloc(1, sizeof *s);
    VkMemGmemEnt *raw = malloc(sizeof *raw * GMEM_ENTRIES);
    if (!s || !raw) rt_fatal("vk: out of memory (global memory pool table)");
    uint64_t gt0 = plat_now_ns();   // DEBUG-GMEMTIME
    int n = vk_mem_gmem_table(raw, GMEM_ENTRIES, &s->gen);
    uint64_t gt1 = plat_now_ns();
    // The smallest piece size, from 2 MiB up, at which the split guest pools fit the table; none: left whole.
    uint64_t piece = 0;
    for (uint64_t g = GMEM_PIECE; g <= (1ull << 40) && !piece; g <<= 1) {
        uint64_t c = (uint64_t)n;
        for (int k = 1; k < n; k++)
            if (raw[k].cpu || raw[k].mirror) c += (raw[k].hi - 1) / g - raw[k].lo / g;
        if (c <= GMEM_ENTRIES) piece = g;
    }
    s->e[0] = raw[0];
    s->n = 1;
    for (int k = 1; k < n; k++)
        for (uint64_t lo = raw[k].lo, hi; lo < raw[k].hi && s->n < GMEM_ENTRIES; lo = hi) {
            hi = raw[k].hi;
            if ((raw[k].cpu || raw[k].mirror) && piece && (lo / piece + 1) * piece < hi) hi = (lo / piece + 1) * piece;
            s->e[s->n++] = (VkMemGmemEnt){lo, hi, raw[k].delta, raw[k].cpu, raw[k].mirror};
        }
    s->refs = 1;
    // The index: the chunk words (entries are disjoint, so a chunk one entry holds whole has no other), giving each
    // shared chunk a page table, then the pages of the shared chunks. l1 is a scratch of the whole chunk range, zero
    // between tables; nz lists the words this table sets.
    static uint32_t *l1, *nz, *ew;
    if (!l1 && (!(l1 = calloc(GMEM_L1_WORDS, sizeof *l1)) || !(nz = malloc(sizeof *nz * GMEM_L1_WORDS)) ||
                !(ew = malloc(GMEM_TABLE_BYTES))))
        rt_fatal("vk: out of memory (global memory pool table)");
    uint32_t nnz = 0, nsplit = 0;
    for (int k = 1; k < s->n; k++) {
        uint64_t lo = s->e[k].lo, hi = s->e[k].hi < GMEM_INDEX_END ? s->e[k].hi : GMEM_INDEX_END;
        for (uint64_t c = lo >> GMEM_HINT_SHIFT; lo < hi && c <= (hi - 1) >> GMEM_HINT_SHIFT; c++) {
            uint64_t cs = c << GMEM_HINT_SHIFT, ce = cs + (1ull << GMEM_HINT_SHIFT);
            if (lo <= cs && hi >= ce) l1[c] = (uint32_t)k, nz[nnz++] = (uint32_t)c;
            else if (!(l1[c] & GMEM_SPLIT)) l1[c] = GMEM_SPLIT | nsplit++, nz[nnz++] = (uint32_t)c;
        }
    }
    s->tbytes = ((uint64_t)GMEM_L2_BASE + (uint64_t)nsplit * GMEM_L2_WORDS) * 4;
    uint32_t *pt = calloc((size_t)nsplit * GMEM_L2_WORDS + 1, sizeof *pt);
    if (!pt) rt_fatal("vk: out of memory (global memory pool table)");
    uint64_t unaligned = 0;
    for (int k = 1; k < s->n; k++) {
        uint64_t lo = s->e[k].lo, hi = s->e[k].hi < GMEM_INDEX_END ? s->e[k].hi : GMEM_INDEX_END;
        for (uint64_t c = lo >> GMEM_HINT_SHIFT; lo < hi && c <= (hi - 1) >> GMEM_HINT_SHIFT; c++) {
            if (!(l1[c] & GMEM_SPLIT)) continue;
            uint64_t cs = c << GMEM_HINT_SHIFT, ce = cs + (1ull << GMEM_HINT_SHIFT);
            uint64_t a = lo > cs ? lo : cs, e = hi < ce ? hi : ce;
            uint32_t *t = pt + (size_t)(l1[c] & ~GMEM_SPLIT) * GMEM_L2_WORDS;
            for (uint64_t pg = a >> 12; pg <= (e - 1) >> 12; pg++) t[pg & (GMEM_L2_WORDS - 1)] = (uint32_t)k;
            if ((a | e) & 0xfff) unaligned++;   // a page two entries share goes to the later one
        }
    }
    uint64_t gt2 = plat_now_ns();   // DEBUG-GMEMTIME
    memset(ew, 0, GMEM_TABLE_BYTES);
    for (int k = 1; k < GMEM_ENTRIES; k++) ew[k * GMEM_ENTRY_WORDS] = ew[k * GMEM_ENTRY_WORDS + 1] = ~0u;
    for (int k = 1; k < s->n; k++) {
        uint32_t *w = &ew[k * GMEM_ENTRY_WORDS];
        w[0] = (uint32_t)s->e[k].lo, w[1] = (uint32_t)(s->e[k].lo >> 32);
        w[2] = (uint32_t)s->e[k].hi, w[3] = (uint32_t)(s->e[k].hi >> 32);
        w[4] = (uint32_t)s->e[k].delta, w[5] = (uint32_t)(s->e[k].delta >> 32);
    }
    // the buffer: a kept one large enough, else a new one with room for 256 more shared chunks
    GmemBuf *gb = NULL;
    plat_mutex_lock(&gmem_free_lock);
    for (GmemBuf **pp = &gmem_free; *pp; pp = &(*pp)->next)
        if ((*pp)->cap >= s->tbytes) { gb = *pp, *pp = gb->next, gmem_nfree--; break; }
    plat_mutex_unlock(&gmem_free_lock);
    const char *where = "kept";
    if (!gb) {
        if (!(gb = calloc(1, sizeof *gb))) rt_fatal("vk: out of memory (global memory pool table)");
        gb->cap = s->tbytes + 256ull * GMEM_L2_WORDS * 4;
        if (!(gb->buf = gmem_buffer(gb->cap, &gb->alloc, &gb->map, &where)))
            rt_fatal("vk: the global memory pool table (%llu bytes) could not be allocated", (unsigned long long)gb->cap);
        memset(gb->map + (size_t)GMEM_L1_BASE * 4, 0, (size_t)GMEM_L1_WORDS * 4);   // every chunk word: none
    }
    uint32_t *m = (uint32_t *)gb->map;
    uint64_t gt3 = plat_now_ns();   // DEBUG-GMEMTIME
    memcpy(m, ew, GMEM_TABLE_BYTES);
    for (uint32_t i = 0; i < gb->nnz; i++) m[GMEM_L1_BASE + gb->nz[i]] = 0;   // the chunk words of its last table
    for (uint32_t i = 0; i < nnz; i++) m[GMEM_L1_BASE + nz[i]] = l1[nz[i]];
    if (nsplit) memcpy(m + GMEM_L2_BASE, pt, sizeof *pt * (size_t)nsplit * GMEM_L2_WORDS);
    if (gb->nzcap < nnz && !(gb->nz = realloc(gb->nz, sizeof *gb->nz * (gb->nzcap = nnz))))
        rt_fatal("vk: out of memory (global memory pool table)");
    memcpy(gb->nz, nz, sizeof *nz * nnz);
    gb->nnz = nnz;
    for (uint32_t i = 0; i < nnz; i++) l1[nz[i]] = 0;
    s->gb = gb;
    {   // DEBUG-GMEMTIME: where a rebuild's time goes
        static uint64_t tn, ta, tb, tc, td;
        uint64_t gt4 = plat_now_ns();
        tn++, ta += gt1 - gt0, tb += gt2 - gt1, tc += gt3 - gt2, td += gt4 - gt3;
        if (tn % 64 == 0)
            rt_log("GMEMSPLIT %llu rebuilds: pool list %.3f ms, index %.3f, entries %.3f, copy to the buffer %.3f (%llu KiB)",
                   (unsigned long long)tn, ta / 1e6 / (double)tn, tb / 1e6 / (double)tn, tc / 1e6 / (double)tn, td / 1e6 / (double)tn,
                   (unsigned long long)(s->tbytes >> 10));
    }
    {   // DEBUG-GMEMDUMP: REKARINA_GMEM_DUMP=<path>: the first table's words, for checking the index offline
        static int done;
        const char *dp = cfg_get("REKARINA_GMEM_DUMP");
        if (dp && !done && s->n > 1000) {
            FILE *f = fopen(dp, "wb");
            if (f) fwrite(m, 1, s->tbytes, f), fclose(f), done = 1;
        }
    }
    free(pt);
    free(raw);
    if (GM.cur) gmem_unref(GM.cur);
    GM.cur = s;
    static int said;
    if (!said++)
        rt_log("vk: global memory pool table: %d entries from %d ranges, guest pools split every %llu KiB (pool generation "
               "%llu), %u chunks shared by entries, %llu boundaries inside a page, %s; rebuilt when the pools change",
               s->n - 1, n - 1, (unsigned long long)(piece >> 10), (unsigned long long)s->gen, nsplit,
               (unsigned long long)unaligned, where);
    return s;
}

// Completion thread: the entries of table s whose bits are set in acc count as written by the GPU; adjacent guest
// entries are reported as one range.
// Under shadow storage each guest entry goes alone (vk_mem_shadow_gmem_written merges the pieces a submission recorded
// as pending, which are entries).
static void gmem_report(GmemSnap *s, const uint32_t acc[GMEM_ENTRIES / 32], uint64_t fence) {
    uint64_t lo = 0, hi = 0;
    int shadow = vk_mem_shadow_on();
    for (int w = 0; w <= GMEM_ENTRIES / 32; w++)
        for (uint32_t m = w < GMEM_ENTRIES / 32 ? acc[w] : 1; m; m &= m - 1) {
            int e = w < GMEM_ENTRIES / 32 ? w * 32 + __builtin_ctz(m) : 0;   // past the end: report the last run
            if (e > 0 && e < s->n && !s->e[e].cpu) {
                __atomic_fetch_add(&GM.dev_writes, 1, __ATOMIC_RELAXED);
                if (s->e[e].mirror) vk_mem_gmem_written(s->e[e].lo, s->e[e].hi);
                static int logged;
                if (logged++ < ((rt_trace & RT_TRACE_GFX) ? 256 : 4))
                    rt_log("vk: global memory writes reached the device copy of guest range %#llx..%#llx",
                           (unsigned long long)s->e[e].lo, (unsigned long long)s->e[e].hi);
                continue;
            }
            if (!shadow && e > 0 && e < s->n && s->e[e].lo == hi && hi) {   // continues the run
                hi = s->e[e].hi;
                continue;
            }
            if (hi) {
                if (shadow) vk_mem_shadow_gmem_written(lo, hi, fence);
                else vk_gpu_written(lo, hi - lo);
                __atomic_fetch_add(&GM.writes, 1, __ATOMIC_RELAXED);
                static int logged;   // the first ones always, more with REKARINA_TRACE=gfx
                if (logged++ < ((rt_trace & RT_TRACE_GFX) ? 256 : 16))
                    rt_log("vk: global memory writes reached guest range %#llx..%#llx", (unsigned long long)lo,
                           (unsigned long long)hi);
            }
            lo = hi = 0;
            if (e > 0 && e < s->n) lo = s->e[e].lo, hi = s->e[e].hi;
        }
}

// Completion thread: the logs of one submission, one per table it used; then they go back to the ring.
static void gmem_done(void *arg) {
    GmemPend *p = arg;
    for (int i = 0; i < p->n; i++) {
        GmemUse *u = &p->v[i];
        if (u->slot != GMEM_NO_LOG) {
            uint32_t acc[GMEM_ENTRIES / 32];
            memcpy(acc, GM.lmap + (size_t)u->slot * GMEM_LOG_BYTES, sizeof acc);   // a few wide reads of the log
            gmem_report(u->snap, acc, p->fence);
            __atomic_store_n(&GM.busy[u->slot], 0, __ATOMIC_RELEASE);
        }
        gmem_unref(u->snap);
    }
    vk_mem_shadow_gmem_done(p->fence);   // shadow storage: the pieces recorded as pending that no log showed written
    free(p->v);
    free(p);
}

static int gmem_logs_init(void) {
    if (GM.lmap) return 1;
    if (GM.failed) return 0;
    const char *where = "";
    if (!(GM.lbuf = gmem_buffer((VkDeviceSize)GMEM_LOGS * GMEM_LOG_BYTES, &GM.lalloc, &GM.lmap, &where))) {
        GM.failed = 1;
        rt_log("vk: global memory write logs could not be allocated: writes through loaded addresses are not reported");
        return 0;
    }
    rt_log("vk: global memory write logs: %d in a ring, %s", GMEM_LOGS, where);
    return 1;
}

static void gmem_shutdown(VkCtx *c) {   // after the last submission completed (vk_backend_shutdown order)
    if (GM.lbuf) vmaDestroyBuffer(c->vma, GM.lbuf, GM.lalloc);
    if (GM.cur) gmem_unref(GM.cur);
    if (GM.logs) rt_log("vk: global memory: %llu write logs, %llu guest ranges and %llu device entries reported written, %llu logs unread",
                        (unsigned long long)GM.logs, (unsigned long long)GM.writes, (unsigned long long)GM.dev_writes,
                        (unsigned long long)GM.full);
    GM.lbuf = VK_NULL_HANDLE, GM.lalloc = NULL, GM.lmap = NULL, GM.cur = NULL;
}

// The submission's use of table s: its log (zeroed when the submission first uses s) and a reference to s until its
// fence.
static GmemUse *gmem_use(VkExec *ex, GmemSnap *s) {
    if (!GM.pend || GM.pend_fence != ex->fence) {   // the first use of this submission: read at its fence
        GM.pend = calloc(1, sizeof *GM.pend);
        if (!GM.pend) rt_fatal("vk: out of memory (global memory write logs)");
        GM.pend_fence = GM.pend->fence = ex->fence;
        vk_completion_add(ex->fence, gmem_done, GM.pend);
    }
    GmemPend *p = GM.pend;
    for (int i = p->n - 1; i >= 0; i--)
        if (p->v[i].snap == s) return &p->v[i];
    if (p->n == p->cap) {
        p->cap = p->cap ? p->cap * 2 : 4;
        p->v = realloc(p->v, sizeof *p->v * (size_t)p->cap);
        if (!p->v) rt_fatal("vk: out of memory (global memory write logs)");
    }
    GmemUse u = {GMEM_NO_LOG, s, VK_NULL_HANDLE, 0};
    if (gmem_logs_init()) {
        for (uint32_t k = 0; k < GMEM_LOGS; k++) {
            uint32_t i = (GM.next + k) % GMEM_LOGS;
            if (!__atomic_load_n(&GM.busy[i], __ATOMIC_ACQUIRE)) { u.slot = i; break; }
        }
        if (u.slot == GMEM_NO_LOG) {
            static int said;
            if (!said++) rt_log("vk: all %d global memory write logs are in flight: some writes are not reported", GMEM_LOGS);
            GM.full++;
        }
    }
    if (u.slot == GMEM_NO_LOG) {
        memset(scratch(ex, GMEM_LOG_BYTES, &u.sbuf, &u.soff), 0, GMEM_LOG_BYTES);
    } else {
        GM.next = u.slot + 1;
        GM.busy[u.slot] = 1;
        memset(GM.lmap + (size_t)u.slot * GMEM_LOG_BYTES, 0, GMEM_LOG_BYTES);
    }
    __atomic_add_fetch(&s->refs, 1, __ATOMIC_RELAXED);
    p->v[p->n++] = u;
    if (!GM.logs++) rt_log("vk: first draw or dispatch with a global memory write log (submission %llu)", (unsigned long long)ex->fence);
    return &p->v[p->n - 1];
}

// The descriptor of the pool table (which 0) or of the submission's write log (which 1) for the draw or dispatch
// being bound.
static VkDescriptorBufferInfo gmem_info(VkExec *ex, int which) {
    if (vk_par_worker) {
        RS.volatile_ = 1;
        return gmem_par[which];
    }
    vk_mem_gmem_sync();   // the mirrors the table points into hold the CPU's writes (once per submission)
    GmemSnap *s = gmem_snapshot(ex);
    GmemUse *u = gmem_use(ex, s);
    RS.volatile_ = 1;   // the table and the log are this submission's: never skipped as a repeat of another build
    if (which == 0) return (VkDescriptorBufferInfo){s->gb->buf, 0, s->tbytes};
    vk_mem_shadow_gmem_record(ex->fence);   // shadow storage: the pieces such stores reached before are pending
    vk_mem_gmem_writeback();   // the shader may store: the mirror pieces stores reached before go back to guest memory
    if (u->slot == GMEM_NO_LOG) return (VkDescriptorBufferInfo){u->sbuf, u->soff, GMEM_LOG_BYTES};
    return (VkDescriptorBufferInfo){GM.lbuf, (VkDeviceSize)u->slot * GMEM_LOG_BYTES, GMEM_LOG_BYTES};
}
#endif

// DEBUG-POOLTAB (REKARINA_POOLTAB=1): the storage buffers of draws whose shaders read the texture pool, once per address:
// where the buffer is, and its first words. The shaders take their texture handles from such tables (vk_texpool.c).
static void pooltab_log(VkExec *ex, int st, const GfxShaderRes *r, uint32_t binding) {
    int ubo, slot = ssbo_slot(r, st, &ubo);
    uint64_t addr = 0, size = 0;
    if (slot >= 0 && slot < 16) {
        const p_bind_buf *b = ubo ? &ex->ubo[st][slot] : &ex->ssbo[st][slot];
        addr = b->addr, size = b->size;
    } else if (ubo_ptr_res(r)) addr = ubo_ptr(ex, st, r);
    static uint64_t seen[512];
    static volatile int32_t n;
    int m = n < 512 ? n : 512;
    for (int i = 0; i < m; i++)
        if (seen[i] == (addr ^ (uint64_t)binding << 48)) return;
    if (m == 512) return;
    seen[__atomic_fetch_add(&n, 1, __ATOMIC_RELAXED) & 511] = addr ^ (uint64_t)binding << 48;
    VkMemSpan sp;
    char line[1024];
    size_t o = 0;
    line[0] = 0;
    if (addr && vk_mem_resolve(addr, 256, &sp)) {
        const uint32_t *w = (const uint32_t *)(uintptr_t)addr;
        for (int i = 0; i < 64; i++) o += (size_t)snprintf(line + o, sizeof line - o, " %x", w[i]);
    }
    rt_log("pooltab: frame %llu stage %d binding %u slot %d ubo %d cb %d+%#x addr %#llx size %#llx:%s",
           (unsigned long long)gfx_counters.frames, st, binding, slot, ubo, r->cbuf_slot, r->handle * 4,
           (unsigned long long)addr, (unsigned long long)size, o ? line : " not backed");
}

static VkDescriptorBufferInfo ssbo_info(VkExec *ex, int st, const GfxShaderRes *r, int written) {
    if (!__atomic_load_n(&LIM.ubo_max, __ATOMIC_ACQUIRE)) limits_init();
    uint64_t max = LIM.sbo_max, align = LIM.sbo_align;
    VkDescriptorBufferInfo bi;
#ifdef GAME_NS2
    if (r->cbuf_slot == GFX_CB_GMEM) return gmem_info(ex, r->handle == 1);
#endif
    int ubo, slot = ssbo_slot(r, st, &ubo);
    p_bind_buf up = {0};
    const p_bind_buf *b = NULL;
    if (slot >= 0 && slot < 16) b = ubo ? &ex->ubo[st][slot] : &ex->ssbo[st][slot];
    else if (ubo_ptr_res(r)) up.addr = ubo_ptr(ex, st, r), b = &up;   // size 0: up to the end of the backing
    if (b) {
        // vk_caps.c: shaders translated for sbuf_align address the buffer from its address rounded down to it.
        uint64_t lead = vk_caps.sbuf_align ? b->addr & (vk_caps.sbuf_align - 1) : 0;
        if (bind_range(ex, b->addr - lead, b->size ? b->size + lead : 0, align, max, ubo ? VK_MUSE_UBO : VK_MUSE_SSBO,
                       written, &bi)) {
            if (written && bi.buffer != VK_NULL_HANDLE) note_written(ex, b->addr - lead, bi.range, bind_owned, written == 2);
            return bi;
        }
        if (b->addr) warn_once(W_SSBO_ALIGN, "storage buffer %#llx is not backed or not aligned: a zero buffer is bound",
                               (unsigned long long)b->addr);
    } else {
        warn_once(W_SSBO_SLOT, "a storage buffer whose address is at c%d+%#x (stage %d) is not replayed: a zero buffer is bound",
                  r->cbuf_slot, r->handle * 4, st);
    }
    return zero_info(max);
}

// ---- feedback loops (maxwell.md 1.9, vk.h VkCtx.feedback)
// TotK's G-buffer and terrain shaders sample targets of the pass that renders them, at their own pixel, after earlier
// draws of the pass wrote them (albedo, normals, R32F linear depth). Maxwell reads see those writes; in Vulkan the read
// is undefined without a barrier, and on the RX 9070 it showed as dark screen-aligned blocks on the island ground
// and hid a companion's sage spirit. Before a draw whose fragment shader samples an attachment of
// the open pass that a draw or clear of the pass wrote since the pass began or the last barrier:
// - VK_FEEDBACK_LAYOUT: the image is learned (VkTexImage.fb_learned; it has the feedback usage since its creation), the
//   pass restarts once with it in ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT, and from then on its passes attach it in that
//   layout, its descriptors carry that layout, the draws that sample it use a pipeline with the feedback flag
//   (VkPipeState.feedback) and a by-region barrier precedes such a draw.
// - VK_FEEDBACK_RESTART: the pass ends and restarts behind the full barrier (vk_exec_pass_begin).
static int in_fbl(const VkExec *ex, VkImage img) {
    for (int i = 0; i < ex->nfbl; i++)
        if (ex->fbl_img[i] == img) return 1;
    return 0;
}

// The image and the levels [l0, l1) and layers [a0, a1) a texture's view covers.
typedef struct { VkImage image; uint32_t l0, l1, a0, a1; } SubRange;

static SubRange sub_of(const VkTexImage *img, const view_t *v, int hv) {
    SubRange r = {img->image, 0, img->levels, 0, img->layers};
    if (hv && v->set_levels) r.l0 = v->base_level, r.l1 = v->base_level + (v->nlevels ? v->nlevels : img->levels);
    if (hv && v->set_layers) r.a0 = v->min_layer, r.a1 = v->min_layer + (v->nlayers ? v->nlayers : img->layers);
    return r;
}

// Attachment slots of the open pass (bit 8 depth) whose subresource r covers.
static uint32_t attached_as(const VkExec *ex, const SubRange *r) {
    uint32_t m = 0;
    if (!r->image) return 0;
    for (uint32_t i = 0; i < 9; i++) {
        const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
        if ((i < 8 && i >= ex->pass_ncolor) || t->image != r->image) continue;
        if (t->level < r->l0 || t->level >= r->l1) continue;
        if (!t->is3d && (t->base_layer + t->layers <= r->a0 || t->base_layer >= r->a1)) continue;
        m |= 1u << i;
    }
    return m;
}

// REKARINA_TEX_MEMO (tex_info, feedback_scan): 0 off, 1 on, 2 (=check) computes every hit again and stops on a difference.
static int td_mode(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_value("REKARINA_TEX_MEMO");
        on = !strcmp(e, "check") ? 2 : strcmp(e, "0") != 0;
    }
    return on;
}

// The subresource a texture handle samples, per handle, kept while its slot resolves the same (slot_still): a draw's
// texture bindings change the whole-scan memo below, but the handles they bind were mostly looked up before, in this
// pass or an earlier one. Direct-mapped.
#define FM_N 8192
static _Thread_local struct { uint64_t h, tok; uint32_t gen; SubRange r; } FM[FM_N];

static SubRange sampled_sub(uint64_t h) {   // texture lock held
    view_t v;
    int hv;
    const VkTexImage *img = vk_texture_by_id_locked((int)(h & 0xfffff), &v, &hv);
    return img ? sub_of(img, &v, hv) : (SubRange){0};
}

// Attachments of the open pass the fragment shader fs samples; kept while the shader, the texture bindings and the
// pass stay the same.
static uint32_t feedback_scan(VkExec *ex, const GfxShader *fs) {
    if (ex->fb_fs == fs && ex->fb_serial == ex->tex_serial && ex->fb_pass == ex->pass_serial) return ex->fb_sampled;
    const GfxShaderRes *res;
    int n = gfx_shader_resources(fs, GFX_RES_TEX, &res);
    uint32_t m = 0, gen = vk_texture_slots_gen();   // before any lookup
    int memo = td_mode(), locked = 0;
    for (int r = 0; r < n; r++) {
        // Bound slots (0..127) only. The bindless pool arrays have slot -1 and tens of thousands of descriptors: walking
        // them is many empty iterations per scan and finds nothing, since no bound slot backs them.
        if (res[r].slot < 0) continue;
        int count = res[r].count ? res[r].count : 1;
        if (count > 128 - res[r].slot) count = 128 - res[r].slot;
        for (int k = 0; k < count; k++) {
            int slot = res[r].slot + k;
            uint64_t h = ex->tex[NVN_STAGE_FS][slot];
            if (!h) continue;
            typeof(FM[0]) *f = &FM[(h * 0x9E3779B97F4A7C15ull) >> 51];
            // (a chunk whose limit leaves out a write at the slot's storage neither takes nor keeps a memo: vk_texture.c
            // "lookups as of a place")
            int clean = vk_texture_asof_clean((int)(h & 0xfffff));
            if (memo && clean && f->h == h && slot_still(&f->gen, f->tok, h, gen)) {
                if (memo == 2) {
                    if (!locked) vk_texture_lock(), locked = 1;
                    SubRange c = sampled_sub(h);
                    if (memcmp(&c, &f->r, sizeof c))
                        rt_fatal("vk_exec: feedback memo of handle %#llx differs from a lookup", (unsigned long long)h);
                }
                m |= attached_as(ex, &f->r);
                continue;
            }
            if (memo == 1 && clean && tds_mode()) {   // what another thread looked up (shm_fm), without the lock
                SubRange sr;
                uint64_t tk;
                if (shm_get(shm_fm, h, h, gen, &tk, &sr, sizeof sr)) {
                    m |= attached_as(ex, &sr);
                    *f = (typeof(*f)){.h = h, .tok = tk, .gen = gen, .r = sr};
                    continue;
                }
            }
            uint64_t tok = token_of(h);   // before the lookup
            view_t sv;
            int shv;
            const VkTexImage *simg = by_id_try(h, &sv, &shv, &locked);
            SubRange sr = simg ? sub_of(simg, &sv, shv) : (SubRange){0};
            m |= attached_as(ex, &sr);
            if (tok && clean) *f = (typeof(*f)){.h = h, .tok = tok, .gen = gen, .r = sr};
            else f->h = 0;
            if (tok && clean && tds_mode() && locked) shm_put(shm_fm, h, tok, gen, &sr, sizeof sr);   // (writers hold the lock)
        }
    }
    if (locked) vk_texture_unlock();
    ex->fb_fs = fs, ex->fb_serial = ex->tex_serial, ex->fb_pass = ex->pass_serial, ex->fb_sampled = m;
    return m;
}

// Before the pipeline lookup of a draw with fragment shader fs: learns and restarts as the mode says. Returns the
// sampled attachments in the feedback layout (the pipeline's feedback flag and feedback_barrier).
static uint32_t feedback_prepare(VkExec *ex, const GfxShader *fs) {
    int mode = vk_ctx()->feedback;
    if (mode == VK_FEEDBACK_OFF || !fs || !ex->in_pass) return 0;
    uint32_t m = feedback_scan(ex, fs);
    uint32_t d = m & ex->fb_dirty;
    if (d) vk_exec_frame.fb_draws++;
    if (d & ~(mode == VK_FEEDBACK_LAYOUT ? ex->fbl_mask : 0)) {
        if (mode == VK_FEEDBACK_LAYOUT)
            for (uint32_t i = 0; i < 9; i++) {
                const VkExecTarget *t = i < 8 ? &ex->color[i] : &ex->depth;
                if ((d >> i & 1) && t->img && t->img->fb_usage) ((VkTexImage *)t->img)->fb_learned = 1;
            }
        if (vk_par_worker) {
            vk_exec_par_restart(ex);   // a chunk's own pass, in the layout it has: the learned layout comes with the next pass
        } else {
            vk_exec_pass_end(ex);
            vk_exec_pass_begin(ex);   // learned targets now in the feedback layout; the full barrier cleared fb_dirty
        }
        vk_exec_frame.fb_restarts++;
    }
    return mode == VK_FEEDBACK_LAYOUT ? m & ex->fbl_mask : 0;
}

// By-region barrier inside the pass: earlier attachment writes become visible to this draw's fragment shader reads.
static void feedback_barrier(VkExec *ex) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT |
                                           VK_PIPELINE_STAGE_2_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_2_LATE_FRAGMENT_TESTS_BIT,
                           .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                           .dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT,
                           .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(ex->cb, &di);
    ex->recorded++;
    ex->fb_dirty = 0;
    vk_exec_frame.fb_barriers++;
}

// Marks the attachments the draw with state s writes: colour targets with a nonzero effective write mask, depth with
// depth writes or the stencil test on.
static void feedback_written(VkExec *ex, const VkPipeState *s) {
    if (vk_ctx()->feedback == VK_FEEDBACK_OFF) return;
    if (ex->depth.image && !(ex->fb_dirty & 0x100) && ((s->ds.dt && s->ds.dw) || s->ds.st)) ex->fb_dirty |= 0x100;
    for (uint32_t i = 0; i < ex->pass_ncolor; i++)
        if (ex->color[i].image && !(ex->fb_dirty >> i & 1) && vk_pipeline_target_wmask(s, (int)i)) ex->fb_dirty |= 1u << i;
}

static void tex_info_full(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                          VkDescriptorImageInfo *out, VkImage *image);

VkImageView vk_exec_pool_view(const VkTexImage *img, const view_t *v, int hv, uint16_t rtype) {
    return get_view(NULL, img, v, hv, rtype, 0);
}

void vk_exec_pool_dummies(VkImageView *view_2d, VkImageView *view_2d_array, VkSampler *samp) {
    dummies_init();
    *view_2d = Z.img[DV_2D].view, *view_2d_array = Z.img[DV_2D_ARRAY].view, *samp = Z.samp;
}

// Sampled-texture descriptors per (handle, resource type, separate), kept while nothing they derive from changed: the
// slot's resolution and texture entry (slot_still) and the samplers (vk_sampler_gen); the layout is set again on every
// hit from the images in the feedback layout (in_fbl of the image kept with it). The first binding of a slot in a submission (vk_texture_epoch) still makes the slot lookup
// for its storage check; a re-upload writes the same image, so the descriptor stays. Draws bind the same textures
// thousands of times a frame (15,000 lookups a frame in the monsters fight); a hit skips the slot lookup, the view key
// with its hash probe and the sampler lookup. Storage images, which note their binding per submission, are
// not kept. Direct-mapped; texture lock held. REKARINA_TEX_MEMO=0 turns it off; =check computes every hit again and stops
// on a difference.
#define TD_N 8192
static _Thread_local struct {
    uint64_t h, tok;
    uint32_t gen, epoch, sgen;
    uint16_t type;
    uint8_t separate, valid;
    VkImage image;
    VkDescriptorImageInfo out;
} TD[TD_N];

// bind_resources_ leaves the texture lock to the lookups (td_lock: its flag): a descriptor taken from the memo below
// needs none (the slot's token and the generations are read unlocked, as bind_sig and answers_for read them; a slot
// registered again meanwhile is ordered after this draw, and what it replaced lives until the submission being
// recorded is done). Every draw took the lock for its texture bindings, so the threads that record the chunks of a
// draw stretch (vk_exec.c) queued up behind one another on it. Other callers hold the lock themselves (td_lock NULL).
static typeof(TD[0]) *td_all[32];   // DEBUG-WHONAMES: every thread's memo
static volatile int32_t td_all_n;
static _Thread_local int td_listed;
static _Thread_local int *td_lock;
#define TD_LOCK() do { if (td_lock && !*td_lock) vk_texture_lock(), *td_lock = 1; } while (0)

// DEBUG-WHONAMES (the device fault report): cached views and descriptor memos of the image
void vk_exec_draw_who_names(uint64_t image) {
    VkImage im = (VkImage)(uintptr_t)image;
    rt_log("  view cache: %s", views_by_image.klen && nvn_hm_get(&views_by_image, &im) ? "holds views of it" : "none");
    int n = td_all_n < 32 ? td_all_n : 32, hits = 0;
    for (int t = 0; t < n; t++)
        for (int i = 0; i < TD_N; i++) {
            const typeof(TD[0]) *m = &td_all[t][i];
            if (!m->valid || (uint64_t)(uintptr_t)m->image != image) continue;
            if (hits++ < 8)
                rt_log("  descriptor memo of thread %d: handle %#llx token %#llx (now %#llx) generation %u (now %u) epoch %u (now %u)",
                       t, (unsigned long long)m->h, (unsigned long long)m->tok, (unsigned long long)token_of(m->h), m->gen,
                       vk_texture_slots_gen(), m->epoch, vk_texture_epoch());
        }
    rt_log("  descriptor memos: %d entries of %d threads hold it", hits, n);
}

// A memo hit's per-submission storage check (the slot's first binding in a submission) on a thread that records the
// chunks of a draw stretch: the slot is queued here, lock-free, and the replay thread checks the queued slots at the
// join (vk_exec_par_join, before the part and its uploads are submitted), under one hold of the texture lock and with
// each one's "as of" limit. The draw keeps the memo's descriptor either way: the check re-uploads into the same image.
// So the chunk threads no longer take the texture lock for these (about a thousand checks a frame; the replay thread
// waited 0.3 ms a frame for chunk threads' holds, RL_TEX). REKARINA_CHUNK_TEXCHECK=0: the chunk checks for itself.
#define TCHK_N 16384
static struct { uint32_t id; uint64_t asof; } tchk[TCHK_N];
static volatile uint32_t tchk_n;
static volatile uint64_t tchk_queued, tchk_full, chunk_miss_locks;   // REKARINA_TRACE=perf
static volatile uint64_t tds_hits, tds_misses, tds_hold_cuts;   // the shared memo and the short holds (below), racy counts
static uint64_t tchk_drained, tchk_drains;
static int tchk_on(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("CHUNK_TEXCHECK", "1");
    return on;
}
static int tchk_push(uint64_t h) {
    uint32_t i = __atomic_fetch_add(&tchk_n, 1, __ATOMIC_RELAXED);
    if (i >= TCHK_N) {
        __atomic_fetch_add(&tchk_full, 1, __ATOMIC_RELAXED);
        return 0;
    }
    tchk[i].id = (uint32_t)(h & 0xfffff), tchk[i].asof = vk_texture_asof_limit();
    __atomic_fetch_add(&tchk_queued, 1, __ATOMIC_RELAXED);
    return 1;
}
// vk_exec_par_join, on the replay thread once every chunk is done (their pushes are ordered before their done flags)
void vk_exec_draw_tchk_drain(void) {
    uint32_t n = __atomic_load_n(&tchk_n, __ATOMIC_ACQUIRE);
    if (!n) return;
    if (n > TCHK_N) n = TCHK_N;
    vk_texture_lock();
    uint64_t was = vk_texture_asof_limit();
    for (uint32_t i = 0; i < n; i++) {
        vk_texture_asof(tchk[i].asof);
        vk_texture_by_id_locked((int)tchk[i].id, NULL, NULL);
    }
    vk_texture_asof(was);
    vk_texture_unlock();
    __atomic_store_n(&tchk_n, 0, __ATOMIC_RELEASE);
    tchk_drained += n, tchk_drains++;
}
void vk_exec_draw_tchk_log(void) {
    rt_log("perf: chunk texture lookups: storage checks left to the replay thread %llu (%.1f per join, %llu queue full), "
           "texture lock taken by a chunk for a memo miss or an uncached binding %llu; shared memo %llu hits, %llu misses; "
           "holds ended after a texture binding %llu",
           (unsigned long long)tchk_queued, tchk_drains ? (double)tchk_drained / (double)tchk_drains : 0.0,
           (unsigned long long)tchk_full, (unsigned long long)chunk_miss_locks, (unsigned long long)tds_hits,
           (unsigned long long)tds_misses, (unsigned long long)tds_hold_cuts);
    tchk_queued = tchk_full = chunk_miss_locks = 0, tchk_drained = tchk_drains = 0, tds_hits = tds_misses = tds_hold_cuts = 0;
}

// ---- the shared descriptor memo (TDS; REKARINA_TEX_MEMO_SHARED=0: each thread its own, as before)
// A second level behind each thread's memo (TD): what one thread resolved under the texture lock, another takes without
// the lock while the same conditions hold (the slot's token and generation, the sampler generation; the per-submission
// storage check stays per thread, or queued for a chunk thread, as on a TD hit). The threads that record the chunks of a
// draw stretch each kept a memo of their own and resolved under the lock what another thread had resolved a moment
// before, and the replay thread waited for those holds: 4.5 s of a 175 s tour behind chunk threads in tex_info_lookup_
// (the lock-holder record, 2026-11-17). Entries are written with the texture lock held (writers are serialized by it) and read
// under a sequence word: an odd word (a write in progress) or one that moved during the read is a miss.
#define TDS_N 16384
typedef struct {
    uint64_t h, tok;
    uint32_t gen, epoch, sgen;
    uint16_t type;
    uint8_t separate;
    VkImage image;
    VkDescriptorImageInfo out;
} TdsEnt;
static struct { volatile uint32_t seq; TdsEnt e; } TDS[TDS_N];

static int tds_mode(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("TEX_MEMO_SHARED", "1");
    return on;
}

static inline uint32_t tds_idx(uint64_t h, uint16_t type, int separate) {
    return (uint32_t)((h * 0x9E3779B97F4A7C15ull ^ (uint64_t)type << 7 ^ (uint64_t)separate) >> 50);   // 14 bits: TDS_N
}

// 1 with *o filled when the entry for (h, type, separate) holds under generation gen and sampler generation sg.
static int tds_get(uint64_t h, uint16_t type, int separate, uint32_t sg, uint32_t gen, TdsEnt *o) {
    typeof(TDS[0]) *s = &TDS[tds_idx(h, type, separate)];
    uint32_t q = __atomic_load_n(&s->seq, __ATOMIC_ACQUIRE);
    if (!q || (q & 1)) return 0;
    memcpy(o, (const void *)&s->e, sizeof *o);
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    if (__atomic_load_n(&s->seq, __ATOMIC_RELAXED) != q) return 0;
    return o->h == h && o->type == type && o->separate == separate && o->sgen == sg && slot_still(&o->gen, o->tok, h, gen);
}

static void tds_put(uint64_t h, uint16_t type, int separate, uint64_t tok, uint32_t gen, uint32_t ep, uint32_t sg, VkImage image,
                    const VkDescriptorImageInfo *out) {   // texture lock held
    typeof(TDS[0]) *s = &TDS[tds_idx(h, type, separate)];
    uint32_t q = __atomic_load_n(&s->seq, __ATOMIC_RELAXED);
    __atomic_store_n(&s->seq, q + 1, __ATOMIC_RELAXED);
    __atomic_thread_fence(__ATOMIC_RELEASE);
    TdsEnt e = {.h = h, .tok = tok, .gen = gen, .epoch = ep, .sgen = sg, .type = type, .separate = (uint8_t)separate,
                .image = image, .out = *out};
    memcpy((void *)&s->e, &e, sizeof e);
    __atomic_store_n(&s->seq, q + 2, __ATOMIC_RELEASE);
}

static void tex_info_lookup_(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                             VkDescriptorImageInfo *out, int *memo_hit);

// A memo hit's per-submission storage check of slot h: nothing to do when the slot was already checked and noted as
// drawn in this submission (by any thread: vk_texture_by_id_fast, no lock); else a chunk leaves it to the replay
// thread (tchk) and the replay thread does it under the texture lock. Each slot's first binding in a submission took
// the lock in every thread that bound it, and the chunks queued it for the replay thread again and again: at the
// Market's square the replay waited behind tex_info_lookup_ 753 times in 5,200 frames (REKARINA_LOCKMAP, 2026-11-18).
static void memo_storage_check(uint64_t h) {
    int hit;
    vk_texture_by_id_fast((int)(h & 0xfffff), NULL, NULL, &hit);
    if (hit) return;
    if (vk_par_worker && tchk_on() && tchk_push(h)) return;
    TD_LOCK();
    vk_texture_by_id_locked((int)(h & 0xfffff), NULL, NULL);
}

static void tex_info_lookup(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                            VkDescriptorImageInfo *out) {
    int hit = 0;
    tex_info_lookup_(ex, r, h, storage, separate, out, &hit);
    // DEBUG-LIVECHECK: the view a draw binds, from the memo or from a lookup
    if (__builtin_expect(vk_live_check_on(), 0) && h && out->imageView &&
        vk_live_check_use((uint64_t)(uintptr_t)out->imageView, hit ? "a draw's texture binding (descriptor memo)"
                                                                     : "a draw's texture binding (lookup)", h)) {
        extern _Thread_local int vk_par_worker;
        void vk_texture_slot_log(int id);
        rt_log("LIVE CHECK: the binding: handle %#llx type %#x storage %d, %s, slot token now %#llx, generation %u, epoch %u",
               (unsigned long long)h, r->type, storage, vk_par_worker ? "recorded in a chunk" : "recorded by the replay",
               (unsigned long long)token_of(h), vk_texture_slots_gen(), vk_texture_epoch());
        vk_texture_slot_log((int)(h & 0xfffff));
    }
}

static void tex_info_lookup_(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                             VkDescriptorImageInfo *out, int *memo_hit) {
    int td_on = td_mode();
    VkImage image;
    if (!td_listed) {   // DEBUG-WHONAMES
        td_listed = 1;
        int i = __atomic_fetch_add(&td_all_n, 1, __ATOMIC_RELAXED);
        if (i < 32) td_all[i] = TD;
    }
    // (a chunk whose limit leaves out a GPU write at the slot's storage resolves for itself and keeps nothing:
    // vk_texture.c "lookups as of a place")
    if (!h || storage || !td_on || !vk_texture_asof_clean((int)(h & 0xfffff))) {
        if (h) {
            if (vk_par_worker && td_lock && !*td_lock) __atomic_fetch_add(&chunk_miss_locks, 1, __ATOMIC_RELAXED);
            TD_LOCK();
        }
        return tex_info_full(ex, r, h, storage, separate, out, &image);
    }
    uint32_t gen = vk_texture_slots_gen(), ep = vk_texture_epoch(), sg = vk_sampler_gen();   // before any lookup
    typeof(TD[0]) *m = &TD[(h * 0x9E3779B97F4A7C15ull ^ (uint64_t)r->type << 7 ^ (uint64_t)separate) >> 51];
    if (m->valid && m->h == h && m->type == r->type && m->separate == separate && m->sgen == sg &&
        slot_still(&m->gen, m->tok, h, gen)) {
        *memo_hit = 1;
        if (m->epoch != ep) {   // the slot's first binding in this submission: only its storage check is due
            memo_storage_check(h);
            m->epoch = ep;
        }
        *out = m->out;
        out->imageLayout = m->image && ex->nfbl && in_fbl(ex, m->image) ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
                                                                          : VK_IMAGE_LAYOUT_GENERAL;
        if (td_on == 2) {
            VkDescriptorImageInfo full;
            TD_LOCK();
            tex_info_full(ex, r, h, storage, separate, &full, &image);
            if (full.sampler != out->sampler || full.imageView != out->imageView || full.imageLayout != out->imageLayout) {
                // DEBUG-MEMOCHECK: a slot registered again between the memo's test and the lookup is no fault of the memo
                uint64_t tnow = token_of(h);
                static volatile int32_t n_raced;
                if (tnow != m->tok) {
                    if (__atomic_fetch_add(&n_raced, 1, __ATOMIC_RELAXED) < 8)
                        rt_log("memo check: handle %#llx registered again during the check (token %#llx -> %#llx)",
                               (unsigned long long)h, (unsigned long long)m->tok, (unsigned long long)tnow);
                    *out = full;
                    return;
                }
                rt_log("memo check: handle %#llx type %#x worker %d: token %#llx (same now), generation %u (now %u, memo test saw %u), "
                       "epoch %u (now %u); view %#llx -> %#llx, sampler %#llx -> %#llx, layout %d -> %d, image %#llx -> %#llx",
                       (unsigned long long)h, r->type, vk_par_worker, (unsigned long long)m->tok, m->gen, vk_texture_slots_gen(), gen,
                       m->epoch, vk_texture_epoch(), (unsigned long long)(uintptr_t)out->imageView,
                       (unsigned long long)(uintptr_t)full.imageView, (unsigned long long)(uintptr_t)out->sampler,
                       (unsigned long long)(uintptr_t)full.sampler, (int)out->imageLayout, (int)full.imageLayout,
                       (unsigned long long)(uintptr_t)m->image, (unsigned long long)(uintptr_t)image);
                {
                    void vk_texture_who_names(uint64_t image);
                    vk_texture_who_names((uint64_t)(uintptr_t)m->image);
                    vk_texture_who_names((uint64_t)(uintptr_t)image);
                }
                rt_fatal("vk_exec: texture descriptor memo of handle %#llx type %#x differs from a lookup",
                         (unsigned long long)h, r->type);
            }
        }
        return;
    }
    if (tds_mode()) {   // what another thread resolved under the same conditions, without the lock (TDS)
        TdsEnt s;
        if (tds_get(h, r->type, separate, sg, gen, &s)) {
            *memo_hit = 1;
            if (s.epoch != ep) memo_storage_check(h);   // this submission's storage check
            *m = (typeof(*m)){.h = h, .tok = s.tok, .gen = s.gen, .epoch = ep, .sgen = sg, .type = r->type,
                              .separate = (uint8_t)separate, .valid = 1, .image = s.image, .out = s.out};
            *out = s.out;
            out->imageLayout = s.image && ex->nfbl && in_fbl(ex, s.image) ? VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT
                                                                          : VK_IMAGE_LAYOUT_GENERAL;
            __atomic_fetch_add(&tds_hits, 1, __ATOMIC_RELAXED);
            return;
        }
        __atomic_fetch_add(&tds_misses, 1, __ATOMIC_RELAXED);
    }
    uint64_t tok = token_of(h);   // before the lookup
    if (vk_par_worker && td_lock && !*td_lock) __atomic_fetch_add(&chunk_miss_locks, 1, __ATOMIC_RELAXED);
    TD_LOCK();
    tex_info_full(ex, r, h, storage, separate, out, &image);
    *m = (typeof(*m)){.h = h, .tok = tok, .gen = gen, .epoch = ep, .sgen = sg, .type = r->type,
                      .separate = (uint8_t)separate, .valid = 1, .image = image, .out = *out};
    if (tds_mode()) tds_put(h, r->type, separate, tok, gen, ep, sg, image, out);   // texture lock held (TD_LOCK)
}

static void tex_info_full(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate,
                          VkDescriptorImageInfo *out, VkImage *image) {
    int ms = (r->type & 0x200) != 0;
    *image = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkSampler samp = VK_NULL_HANDLE;
    VkImageLayout layout = VK_IMAGE_LAYOUT_GENERAL;
    uint8_t fill[4];
    int filled = 0;   // the slot holds one of the game's fill-in textures and its image cannot serve this sampler
    if (h) {
        view_t v;
        int hv;
        const VkTexImage *img = vk_texture_by_id_locked((int)(h & 0xfffff), &v, &hv);
        if (img) view = get_view(ex, img, &v, hv, r->type, storage), *image = img->image;
        if (img && !view && !storage && !ms && vk_texture_fill_colour_locked(img, fill)) {
            const uint8_t *sw = hv && v.set_swizzle ? v.swz : img->swz;   // the selectors get_view would give the view
            uint8_t c[4];
            for (int i = 0; i < 4; i++)
                c[i] = sw[i] == NVN_TEXTURE_SWIZZLE_ZERO ? 0
                     : sw[i] == NVN_TEXTURE_SWIZZLE_ONE || (sw[i] == NVN_TEXTURE_SWIZZLE_A && img->alpha_one) ? 255
                     : sw[i] >= NVN_TEXTURE_SWIZZLE_R && sw[i] <= NVN_TEXTURE_SWIZZLE_A ? fill[sw[i] - NVN_TEXTURE_SWIZZLE_R]
                     : fill[i];
            memcpy(fill, c, 4);
            filled = 1 + (img->format == VK_FORMAT_R8G8B8A8_SRGB);
        }
        if (img && storage) {
            vk_interface_coverage_write(img->image, img->addr);
            vk_texture_storage_bound_locked(img);
        }
        if (img && !storage && ex->nfbl && in_fbl(ex, img->image)) layout = VK_IMAGE_LAYOUT_ATTACHMENT_FEEDBACK_LOOP_OPTIMAL_EXT;
        if (!storage && !separate) {
            // A depth texture sampled without a compare reads one texel on the Switch, whatever the sampler's filter:
            // blending neighbouring depths puts reconstructed positions in mid-air at object edges (shadow masks).
            int sid = (int)(h >> 20 & 0xfff);
            int point = img && (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) && !(r->type & 0x400) &&
                        vk_exec_view_aspect(img, &v, hv) != VK_IMAGE_ASPECT_STENCIL_BIT;
            samp = point ? vk_sampler_point_by_id_locked(sid)
                 : r->type & 0x400 ? vk_sampler_compare_by_id_locked(sid) : vk_sampler_by_id_locked(sid);
            if ((r->type & 0xff) == 4 && img) vk_debug_samp3d_locked(sid, 0, (int)img->format);   // DEBUG-SAMP3D
        }
    }
    if (!view) {
        int di = dummy_index(res_view_type(r->type), ms);
        view = storage ? storage_dummy(di, r->format)
             : filled ? colour_dummy(di, filled == 2 ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM, fill) : Z.img[di].view;
    }
    if (!samp) samp = r->type & 0x400 ? Z.samp_shadow : Z.samp;
    *out = (VkDescriptorImageInfo){separate || storage ? VK_NULL_HANDLE : samp, view, layout};
}

// ---- depth twins: colour images under a compare sampler in a compute shader
// TotK's compute passes sample an R32_SFLOAT texture through a shadow sampler. Maxwell compares any single-channel
// float or unorm texel; Vulkan only formats with SAMPLED_IMAGE_DEPTH_COMPARISON (VUID-vkCmdDispatch-None-06479), which
// AMD's drivers do not give R32_SFLOAT, so the result there is undefined. Such an image gets a depth twin of the same
// size and bits (R32_SFLOAT -> D32_SFLOAT, R16_UNORM -> D16_UNORM), copied through a buffer right before each dispatch
// that samples it (a dispatch is outside any render pass), and the dispatch samples the twin. Draws keep the colour
// image: a copy cannot go inside their pass. Design: Savage Wind Recomp runtime/nvn/vulkan/exec.c depth_twin_update
// (https://github.com/Revivedx/SavageWindRecomp, MIT); the code is ours. REKARINA_DEPTH_TWIN=0: off.
#define TWINS 8
typedef struct { VkImage src, img; VmaAllocation mem; VkImageView view[2]; uint32_t w, h, layers; VkFormat fmt; } Twin;
static Twin twins[TWINS];
static unsigned twin_next;
static _Thread_local int twin_dispatch;  // op_dispatch*: the resources this thread binds are a dispatch's
static VkBuffer twin_buf;
static VmaAllocation twin_buf_mem;
static VkDeviceSize twin_buf_cap;

static void twin_free(void *arg) {   // completion thread, after the last submission that may sample it
    Twin *t = arg;
    VkCtx *c = vk_ctx();
    for (int k = 0; k < 2; k++)
        if (t->view[k]) vkDestroyImageView(c->device, t->view[k], NULL);
    vmaDestroyImage(c->vma, t->img, t->mem);
    free(t);
}

typedef struct { VkBuffer buf; VmaAllocation mem; } TwinBuf;
static void twin_buf_free(void *arg) {   // completion thread, after the last submission that may copy through it
    TwinBuf *b = arg;
    vmaDestroyBuffer(vk_ctx()->vma, b->buf, b->mem);
    free(b);
}

static VkFormat twin_format(const VkTexImage *img) {
    static int on = -1;
    if (on < 0) on = rt_knob("DEPTH_TWIN", "1");
    if (!on || (img->aspect & VK_IMAGE_ASPECT_DEPTH_BIT) || img->samples > 1 || img->depth > 1) return VK_FORMAT_UNDEFINED;
    VkFormat df = img->format == VK_FORMAT_R32_SFLOAT ? VK_FORMAT_D32_SFLOAT
                : img->format == VK_FORMAT_R16_UNORM  ? VK_FORMAT_D16_UNORM : VK_FORMAT_UNDEFINED;
    if (!df) return df;
    static int compares[2] = {-1, -1};   // per source format: the driver compares it as it is
    int i = df == VK_FORMAT_D16_UNORM;
    if (compares[i] < 0) {
        VkFormatProperties3 p3 = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_3};
        VkFormatProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2, .pNext = &p3};
        vkGetPhysicalDeviceFormatProperties2(vk_ctx()->phys, img->format, &p2);
        compares[i] = (p3.optimalTilingFeatures & VK_FORMAT_FEATURE_2_SAMPLED_IMAGE_DEPTH_COMPARISON_BIT) != 0;
        rt_log("vk_exec: %s under a compare sampler: %s", i ? "R16_UNORM" : "R32_SFLOAT",
               compares[i] ? "compared by the driver" : "sampled through a depth twin in compute passes");
    }
    return compares[i] ? VK_FORMAT_UNDEFINED : df;
}

static void twins_release(VkCtx *c) {
    for (int i = 0; i < TWINS; i++) {
        for (int k = 0; k < 2; k++)
            if (twins[i].view[k]) vkDestroyImageView(c->device, twins[i].view[k], NULL);
        if (twins[i].img) vmaDestroyImage(c->vma, twins[i].img, twins[i].mem);
    }
    memset(twins, 0, sizeof twins);
    if (twin_buf) vmaDestroyBuffer(c->vma, twin_buf, twin_buf_mem);
    twin_buf = VK_NULL_HANDLE, twin_buf_mem = NULL, twin_buf_cap = 0;
}

// The twin of img holding img's current texels, the copies recorded into ex->cb; NULL when img needs none or it cannot
// be made.
static Twin *twin_update(VkExec *ex, const VkTexImage *img) {
    VkFormat df = twin_format(img);
    if (!df) return NULL;
    VkCtx *c = vk_ctx();
    uint32_t bpp = df == VK_FORMAT_D32_SFLOAT ? 4 : 2, layers = img->layers ? img->layers : 1;
    VkDeviceSize bytes = (VkDeviceSize)img->width * img->height * layers * bpp;
    if (bytes > (64u << 20)) return NULL;
    if (bytes > twin_buf_cap) {   // one copy buffer, grown; the old one goes after the submission being recorded
        VkDeviceSize want = twin_buf_cap ? twin_buf_cap * 2 : 1u << 16;
        while (want < bytes) want *= 2;
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = want,
                                 .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        VkBuffer nb;
        VmaAllocation nm;
        if (vmaCreateBuffer(c->vma, &bi, &ai, &nb, &nm, NULL) != VK_SUCCESS) return NULL;
        if (twin_buf) {
            TwinBuf *old = malloc(sizeof *old);
            if (!old) rt_fatal("vk: out of memory (depth twin)");
            *old = (TwinBuf){twin_buf, twin_buf_mem};
            vk_completion_add(vk_sync_issued() + 1, twin_buf_free, old);
        }
        twin_buf = nb, twin_buf_mem = nm, twin_buf_cap = want;
    }
    Twin *t = NULL;
    for (int i = 0; i < TWINS && !t; i++)
        if (twins[i].img && twins[i].src == img->image && twins[i].w == img->width && twins[i].h == img->height &&
            twins[i].layers == layers && twins[i].fmt == df)
            t = &twins[i];
    if (!t) {
        t = &twins[twin_next++ % TWINS];
        if (t->img) {   // the submission being recorded may still sample the replaced twin
            Twin *old = malloc(sizeof *old);
            if (!old) rt_fatal("vk: out of memory (depth twin)");
            *old = *t;
            vk_completion_add(vk_sync_issued() + 1, twin_free, old);
        }
        *t = (Twin){.src = img->image, .w = img->width, .h = img->height, .layers = layers, .fmt = df};
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = df,
                                .extent = {img->width, img->height, 1}, .mipLevels = 1, .arrayLayers = layers,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ci, &ai, &t->img, &t->mem, NULL) != VK_SUCCESS) {
            *t = (Twin){0};
            return NULL;
        }
        vk_addr_label((uint64_t)(uintptr_t)t->img, "depth twin of %#llx %ux%u", (unsigned long long)(uintptr_t)img->image,
                      img->width, img->height);
        for (int k = 0; k < 2; k++) {
            VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = t->img,
                                        .viewType = k ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D, .format = df,
                                        .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, k ? layers : 1}};
            if (vkCreateImageView(c->device, &vi, NULL, &t->view[k]) != VK_SUCCESS) t->view[k] = VK_NULL_HANDLE;
        }
        static int nlog;
        if (nlog++ < 8)
            rt_log("vk_exec: depth twin %ux%ux%u %s for a compute pass's compare sampler", img->width, img->height,
                   layers, df == VK_FORMAT_D32_SFLOAT ? "D32_SFLOAT" : "D16_UNORM");
    }
    // The source's earlier writes and the twin's earlier readers before the copies; the twin's old texels are dropped.
    VkMemoryBarrier2 m0 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = t->img, .subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, layers}};
    VkDependencyInfo d0 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m0,
                           .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(ex->cb, &d0);
    VkBufferImageCopy src = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, layers},
                             .imageExtent = {img->width, img->height, 1}};
    vkCmdCopyImageToBuffer(ex->cb, img->image, VK_IMAGE_LAYOUT_GENERAL, twin_buf, 1, &src);
    VkMemoryBarrier2 m1 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT};
    VkDependencyInfo d1 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m1};
    vkCmdPipelineBarrier2(ex->cb, &d1);
    VkBufferImageCopy dst = {.imageSubresource = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 0, layers},
                             .imageExtent = {img->width, img->height, 1}};
    vkCmdCopyBufferToImage(ex->cb, twin_buf, t->img, VK_IMAGE_LAYOUT_GENERAL, 1, &dst);
    // The twin's texels before the dispatch; the source's next writers and the buffer's next copy after these copies.
    VkMemoryBarrier2 m2 = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo d2 = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &m2};
    vkCmdPipelineBarrier2(ex->cb, &d2);
    ex->recorded += 5;
    return t;
}

// The image descriptor of slot h; in a dispatch, a compare-sampled colour image the driver cannot compare is swapped
// for its depth twin; in a pass larger than it, the scene's light target, once DLSS scaled it this frame, for DLSS's
// output (vk_dlaa.c).
static void tex_info(VkExec *ex, const GfxShaderRes *r, uint64_t h, int storage, int separate, VkDescriptorImageInfo *out) {
    tex_info_lookup(ex, r, h, storage, separate, out);
    {   // DEBUG-R8LOG: the textures a draw into a subtitle line target (R8, 256+ wide, 400 or less high) samples
        static int on = -1, n;
        if (on < 0) on = cfg_get("REKARINA_R8LOG") != NULL;
        if (on && h && n < 300 && ex->in_pass && ex->pipe.color_fmt[0] == 0x1 && ex->pass_w >= 256 && ex->pass_h <= 400) {
            TD_LOCK();
            const VkTexImage *img = vk_texture_by_id_locked((int)(h & 0xfffff), NULL, NULL);
            uint32_t up = 0;
            int lc = 0, pw = 0;
            if (img) vk_texture_debug(img, &up, &lc, &pw);
            n++;
            rt_log("R8LOG frame %llu draw into %ux%u samples handle %#llx: %s addr %#llx %ux%u vkfmt %d uploads %u last_change %d "
                   "pages_written %d", (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), ex->pass_w,
                   ex->pass_h, (unsigned long long)h, img ? "image" : "none", img ? (unsigned long long)img->addr : 0ull,
                   img ? img->width : 0, img ? img->height : 0, img ? (int)img->format : 0, up, lc, pw);
        }
    }
    if (h && !ex->in_pass && vk_dlaa_pathlog()) {   // DEBUG-PATHLOG: a dispatch's large images
        TD_LOCK();
        const VkTexImage *img = vk_texture_by_id_locked((int)(h & 0xfffff), NULL, NULL);
        if (img && img->width >= 1200)
            rt_log("PATHLOG   dispatch %s handle %#llx: %llx %ux%u vkfmt %d", storage ? "STORAGE" : "samples", (unsigned long long)h,
                   (unsigned long long)img->addr, img->width, img->height, (int)img->format);
    }
    if (h && ex->in_pass && !ex->depth.image && ex->pass_w >= 600 && vk_dlaa_pathlog()) {   // DEBUG-PATHLOG
        TD_LOCK();
        const VkTexImage *img = vk_texture_by_id_locked((int)(h & 0xfffff), NULL, NULL);
        rt_log("PATHLOG   pass %u %ux%u samples handle %#llx%s: %llx %ux%u vkfmt %d%s", (unsigned)ex->pass_serial, ex->pass_w, ex->pass_h,
               (unsigned long long)h, storage ? " (storage)" : "", img ? (unsigned long long)img->addr : 0ull, img ? img->width : 0,
               img ? img->height : 0, img ? (int)img->format : 0,
               img && img == vk_dlaa_swap_img ? (ex->pass_w > vk_dlaa_swap_img->width + 1 && !storage ? " = light target, SWAPPED for DLSS's" : " = light target, not swapped") : "");
    }
    if (h && !storage && vk_dlaa_swap_img && ex->in_pass && ex->pass_w > vk_dlaa_swap_img->width + 1) {
        TD_LOCK();
        if (vk_texture_by_id_locked((int)(h & 0xfffff), NULL, NULL) == vk_dlaa_swap_img)
            out->imageView = vk_dlaa_swap_view, out->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    if (!twin_dispatch || storage || separate || !h || !(r->type & 0x400)) return;
    VkImageViewType vt = res_view_type(r->type);
    if (vt != VK_IMAGE_VIEW_TYPE_2D && vt != VK_IMAGE_VIEW_TYPE_2D_ARRAY) return;
    TD_LOCK();
    const VkTexImage *img = vk_texture_by_id_locked((int)(h & 0xfffff), NULL, NULL);
    Twin *t = img ? twin_update(ex, img) : NULL;
    VkImageView v = t ? t->view[vt == VK_IMAGE_VIEW_TYPE_2D_ARRAY] : VK_NULL_HANDLE;
    if (v) out->imageView = v, out->imageLayout = VK_IMAGE_LAYOUT_GENERAL;
}

static VkDescriptorSet alloc_set(VkExec *ex, VkDescriptorSetLayout layout) {
    CbRes *r = cb_res(ex);
    VkCtx *c = vk_ctx();
    for (int attempt = 0; attempt < 2; attempt++) {
        if (r->npools) {
            VkDescriptorSetAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                              .descriptorPool = r->pools[r->npools - 1], .descriptorSetCount = 1,
                                              .pSetLayouts = &layout};
            VkDescriptorSet set;
            VkResult res = vkAllocateDescriptorSets(c->device, &ai, &set);
            if (res == VK_SUCCESS) return set;
            if (res != VK_ERROR_OUT_OF_POOL_MEMORY && res != VK_ERROR_FRAGMENTED_POOL) rt_fatal("vk: descriptor set allocation failed (%d)", (int)res);
        }
        static const VkDescriptorPoolSize sizes[] = {
            {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4096}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2048},
            {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096}, {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1024},
            {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1024}, {VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 256},
            {VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 256}};
        VkDescriptorPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 256,
                                         .poolSizeCount = sizeof sizes / sizeof sizes[0], .pPoolSizes = sizes};
        if (r->npools == r->cpools) r->pools = grow(r->pools, &r->cpools, sizeof *r->pools);
        VkDescriptorPool pool = VK_NULL_HANDLE;
        plat_mutex_lock(&spare_lock);
        if (nspare_pool) pool = spare_pool[--nspare_pool];
        plat_mutex_unlock(&spare_lock);
        if (!pool && vkCreateDescriptorPool(c->device, &pi, NULL, &pool) != VK_SUCCESS) rt_fatal("vk: descriptor pool failed");
        r->pools[r->npools++] = pool;
    }
    rt_fatal("vk: descriptor set allocation failed twice");
    return VK_NULL_HANDLE;
}

static void support_in(const VkExec *ex, float in[3]) {   // what support_fill reads: viewport size, point size
    float k = ex->pass_scale > 1 ? (float)ex->pass_scale : 1.0f;
    in[0] = ex->in_pass && ex->has_vp ? ex->vp[0][2] * k : (float)ex->pass_w;
    in[1] = ex->in_pass && ex->has_vp ? ex->vp[0][3] * k : (float)ex->pass_h;
    in[2] = ex->point_size > 0 ? ex->point_size : 1.0f;
}

// REKARINA_PCSS (percent, 0 = off): the pcss field of the support buffer, which translator patch 0010 reads in every
// explicit-LOD compare on a 2D array texture (the sun cascades). x: penumbra width in UV per unit of light-space depth
// between receiver and blocker (0 keeps the game's single compare), y and z: the narrowest and widest disc radius,
// w: the blocker search radius, all in UV of the cascade map. Radii are in UV, so they hold at any REKARINA_SHADOW_SCALE.
// REKARINA_PCSS_K overrides the depth-to-penumbra factor at 100 percent, for tuning.
static const float *pcss_params(void) {
    static float p[4] = {-1, 0, 0, 0};
    if (p[0] < 0) {
        int pct = cfg_int("REKARINA_PCSS");
        pct = pct < 0 ? 0 : pct > 200 ? 200 : pct;
        const char *k = cfg_get("REKARINA_PCSS_K");
        float k100 = k && *k ? (float)atof(k) : 0.5f;
        p[0] = k100 * (float)pct / 100.0f;
        p[1] = 0.35f / 960.0f, p[2] = 5.0f / 960.0f, p[3] = 4.0f / 960.0f;
        if (pct) rt_log("vk: soft shadows (PCSS) at %d%% (depth factor %g)", pct, p[0]);
    }
    return p;
}

// The shadow_spread field of the support buffer: the factor translator patch 0011 applies to the lighting shaders' sun
// shadow filter offset (element 56 .xy of their lighting block), 1 / REKARINA_SHADOW_SCALE. The game spaces its four
// compares for its own shadow map texels; with maps REKARINA_SHADOW_SCALE times larger, the same offset in UV spans that
// many more texels and the near cascade shows four separate copies of each edge. Dividing by the scale keeps the game's
// spacing in texels. The offset is in UV of the shadow map, so the output resolution plays no part. The scale is the
// one the maps were made with (vk_texture.c vk_shadow_scale, auto included).
static const float *shadow_spread_params(void) {
    static float p[4] = {0, 0, 0, 0};
    if (p[0] == 0) p[0] = 1.0f / (float)vk_shadow_scale();
    return p;
}

static void support_fill(uint8_t *b, float w, float h, float point_size) {
    memset(b, 0, SUPPORT_BYTES);
    float inv[4] = {w > 0 ? 2.0f / w : 0, h > 0 ? 2.0f / h : 0, 0, 0}, size[4] = {w, h, 0, 0};
    memcpy(b + 144, inv, 16);
    memcpy(b + 160, size, 16);
    const float one = 1.0f;
    memcpy(b + 1392, &point_size, 4);   // point_size (translator patch 0008): vertex shaders that write none output it
    memcpy(b + 1408, pcss_params(), 16);   // pcss (translator patch 0010): soft sun shadows in the cascade compares
    memcpy(b + 1424, shadow_spread_params(), 16);    // shadow_spread (translator patch 0011): sun shadow filter width
    for (int i = 0; i < 73; i++) memcpy(b + 192 + 16 * i, &one, 4);   // render scale 1: the host renders at guest size
    if (vk_hdr_armed()) {   // read by the HDR tonemap (shaders/hdr_tonemap.frag), past the translator's own fields
        float hdr[4 * VK_HDR_PARAMS];
        vk_hdr_support(hdr);
        memcpy(b + VK_HDR_PARAMS_AT, hdr, sizeof hdr);
    }
}

// REKARINA_BIND_STATS=1: bindings and descriptors per draw of each stage (Task 2.2's choice of the pushed stage), and how
// often a draw repeats the previous one, logged every 20000 draws: pushes recorded and skipped, allocated sets written
// and reused, stage texture-type recomputations (answers_for), draws with the previous draw's program, bound state
// and textures (shaders and pipeline would be the same) and those that also have its buffers and viewport (all
// descriptors would be the same). BS is declared beside DS.
static int bind_stats_on(void) {
    if (BS.on < 0) {
        const char *e = cfg_get("REKARINA_BIND_STATS");
        BS.on = e && *e && strcmp(e, "0");
    }
    return BS.on;
}

static void bind_stats(const VkPipeInfo *info) {
    if (!bind_stats_on() || info->bind_point != VK_PIPELINE_BIND_POINT_GRAPHICS) return;
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *b = &info->bindings[i];
        int fs = info->n_sets > 1 ? b->set : !(b->stages & VK_SHADER_STAGE_VERTEX_BIT);
        BS.bind[fs]++, BS.desc[fs] += b->count ? b->count : 1;
    }
    if (++BS.draws % 20000 == 0) {
        double d = (double)BS.draws;
        rt_log("vk_exec: per draw over %llu draws: vertex %.2f bindings %.2f descriptors, fragment %.2f bindings %.2f descriptors",
               (unsigned long long)BS.draws, BS.bind[0] / d, BS.desc[0] / d, BS.bind[1] / d, BS.desc[1] / d);
        rt_log("vk_exec: per draw: pushes %.3f skipped %.3f, sets written %.3f reused %.3f, texture types %.3f; "
               "same program and state as the draw before %.3f, also same buffers %.3f; descriptor builds skipped %.3f",
               BS.push / d, BS.push_skip / d, BS.set_new / d, BS.set_reuse / d, BS.ac_miss / d, BS.same_prog / d,
               BS.same_all / d, BS.bind_skip / d);
    }
}

// The bound state a draw reads, compared with the previous draw's (REKARINA_BIND_STATS only).
static void repeat_stats(const VkExec *ex, const VkPipeInfo *info) {
    static struct {
        const prog_t *prog;
        VkPipeState pipe;
        uint64_t tex[2][128], img[2][32];
        const VkPipeInfo *info;
        p_bind_buf ubo[2][18], ssbo[2][16];
        float vp[6], point_size;
    } last, cur;
    if (!bind_stats_on()) return;
    memset(&cur, 0, sizeof cur);
    cur.prog = ex->prog, cur.pipe = ex->pipe, cur.info = info;
    const int st[2] = {NVN_STAGE_VS, NVN_STAGE_FS};
    for (int k = 0; k < 2; k++) {
        memcpy(cur.tex[k], ex->tex[st[k]], sizeof cur.tex[k]), memcpy(cur.img[k], ex->img[st[k]], sizeof cur.img[k]);
        memcpy(cur.ubo[k], ex->ubo[st[k]], sizeof cur.ubo[k]), memcpy(cur.ssbo[k], ex->ssbo[st[k]], sizeof cur.ssbo[k]);
    }
    memcpy(cur.vp, ex->vp[0], sizeof cur.vp), cur.point_size = ex->point_size;
    size_t prog_part = offsetof(typeof(cur), info);
    if (!memcmp(&cur, &last, prog_part)) {
        BS.same_prog++;
        if (!memcmp((const char *)&cur + prog_part, (const char *)&last + prog_part, sizeof cur - prog_part)) BS.same_all++;
    }
    last = cur;
}

static int push_skip_on(void) {   // REKARINA_PUSH_SKIP=0: every graphics draw pushes its set again
    static int on = -1;
    if (on < 0) {
        on = cfg_bool("REKARINA_PUSH_SKIP");
    }
    return on;
}

// A graphics draw whose descriptor inputs equal the previous graphics draw's in this command buffer skips building its
// sets: every write would be the same, the push would be skipped and the allocated set reused without a bind (about a
// quarter of field and monsters2 draws, REKARINA_BIND_STATS). The inputs, compared and stored in place, are what
// bind_resources and its helpers read beyond the pipeline info's fixed bindings and Z's stand-ins:
// - the info (bindings, layouts), the program and its serial (the compiler-constant range of UBO slot 1);
// - vk_texture_slots_gen, vk_texture_epoch (a slot's image, view and once-per-submit storage check), vk_sampler_gen;
// - vk_mem_gen (buffer and offset of a span, mirror creation and per-submission sync);
// - the images in the feedback layout (the sampled layout);
// - per binding: the support inputs, c0's storage table and base instance, the bound uniform and storage ranges, the
//   texture and image handles (RsList; compared only when a bind op ran since the last draw, bind_gen);
// - storage buffer addresses read from user uniform blocks (RsList.p, ubo_ptr): compared on every draw, since the
//   block's contents change without a bind op.
// The side effects of a build repeat as no-ops in the same command buffer and submission: c0 and support buffers kept
// in B, note_written's ranges, vk_mem_gpu_wrote's write-backs, the per-submission mirror sync and dirty census note,
// the views cached per submission. The census counts are added again. A build that copied guest bytes into transient
// memory (an unaligned host range) is never skipped after: the copy may differ. B.res_valid says the sets bound hold
// the last build; vk_exec_draw_new_cb's memset clears it, and every build clears it until it ends.
// REKARINA_BIND_SKIP=0 builds every draw; =check builds every draw and stops when a draw with equal inputs gets other
// writes, census counts or descriptor commands.
static int bind_skip_on(void) {
    if (RS.on < 0) {
        const char *e = cfg_value("REKARINA_BIND_SKIP");
        RS.on = !strcmp(e, "check") ? 2 : strcmp(e, "0") != 0;
        if (RS.on != 1) rt_log("vk_exec: descriptor build skip %s", RS.on ? "check" : "off");
    }
    return RS.on;
}

static void rs_add(RsRange *r, uint32_t *n, const VkExec *ex, const void *at, size_t len) {
    uint32_t off = (uint32_t)((const uint8_t *)at - (const uint8_t *)ex);
    if (*n && r[*n - 1].off + r[*n - 1].len == off) r[*n - 1].len += (uint32_t)len;
    else r[(*n)++] = (RsRange){off, (uint32_t)len};
}

static const RsList *rs_list(const VkExec *ex, const VkPipeInfo *info) {
    if (info == rs_last) return rs_last_l;
    if (!rs_lists.klen) nvn_hm_init(&rs_lists, sizeof info);
    RsList *l = nvn_hm_get(&rs_lists, &info);
    if (!l) {
        // per thread: the chunks of a draw stretch build their lists at the same time (vk_exec.c par_run). Shared, two
        // threads wrote their ranges over each other's, and a pipeline's list, kept until shutdown, named the wrong
        // slots: its draws were taken for repeats of the draw before them and drew with that draw's sets.
        static _Thread_local RsRange tmp[MAX_DESC];
        RsPtr ptr[RS_PTRS];
        uint32_t n = 0, nd = 0, np = 0;
        int sup = 0, c0 = 0, ok = 1;
        for (uint32_t i = 0; i < info->n_bindings && ok; i++) {
            const VkPipeBinding *b = &info->bindings[i];
            const GfxShaderRes *r = b->res;
            uint32_t cnt = b->count ? b->count : 1;
            if ((nd += cnt) > MAX_DESC) { ok = 0; break; }   // the build stops on it
            int st = nvn_stage_of(b->stage);
            if (b->kind == VK_PIPE_SUPPORT) {
                sup = 1;
            } else if (b->kind == GFX_RES_UBO) {
                for (uint32_t k = 0; k < cnt; k++) {
                    int slot = r->slot + (int)k;
                    if (slot == 0 && c0_on()) c0 = 1, rs_add(tmp, &n, ex, ex->ssbo[st], sizeof ex->ssbo[st]);
                    else if (slot >= 3 && slot - 3 < 18) rs_add(tmp, &n, ex, &ex->ubo[st][slot - 3], sizeof(p_bind_buf));
#ifdef GAME_NS2
                    else if (slot == NVN2_PUSH_BANK) rs_add(tmp, &n, ex, ex->push[st == NVN_STAGE_CS], NVN2_PUSH_SIZE);
#endif
                }   // slot 1: the program (bind_sig); others: the zero buffer
            } else if (b->kind == GFX_RES_SSBO) {   // every descriptor of the binding is the same range
                if (r->cbuf_slot == GFX_CB_HDR_METER) { ok = 0; break; }   // a separate readback slot per draw
                int ubo, slot = ssbo_slot(r, st, &ubo);
                if (slot >= 0 && slot < 16) {
                    rs_add(tmp, &n, ex, ubo ? &ex->ubo[st][slot] : &ex->ssbo[st][slot], sizeof(p_bind_buf));
                } else if (ubo_ptr_res(r)) {
                    if (np == RS_PTRS) { ok = 0; break; }
                    rs_add(tmp, &n, ex, &ex->ubo[st][r->cbuf_slot - 3], sizeof(p_bind_buf));
                    ptr[np++] = (RsPtr){st, r};
                }
            } else if ((b->kind == GFX_RES_TEX || b->kind == GFX_RES_IMG) && b->type != VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER &&
                       b->type != VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
                int img = b->kind == GFX_RES_IMG, cap = img ? 32 : 128;
                for (uint32_t k = 0; k < cnt; k++) {
                    int slot = r->slot + (int)k;
                    if (slot >= 0 && slot < cap) rs_add(tmp, &n, ex, img ? &ex->img[st][slot] : &ex->tex[st][slot], 8);
                }
            }
        }
        uint64_t bytes = 0;
        for (uint32_t i = 0; i < n; i++) bytes += tmp[i].len;
        if (bytes > RS_BYTES) ok = 0;
        l = malloc(sizeof *l + sizeof *tmp * n);
        if (!l) rt_fatal("vk: out of memory");
        l->n = n, l->bytes = (uint32_t)bytes, l->sup = (uint8_t)sup, l->c0 = (uint8_t)c0, l->ok = (uint8_t)ok;
        l->np = (uint8_t)np;
        memcpy(l->p, ptr, sizeof *ptr * np);
        memcpy(l->r, tmp, sizeof *tmp * n);
        nvn_hm_put(&rs_lists, &info, l);
    }
    rs_last = info, rs_last_l = l;
    return l;
}

// The signature of a graphics draw's descriptor inputs into RS; 1 when it equals the stored one. The bound-slot bytes
// are compared only when a bind op ran since they were stored or the pipeline changed: otherwise they are the same.
// Check mode compares them anyway and stops when the bind op generation said they could not differ but they do.
static int bind_sig(const VkExec *ex, const VkPipeInfo *info) {
    const RsList *l = rs_list(ex, info);
    if (!l->ok) return RS.hn = 0, RS.info = NULL, 0;
    uint64_t *h = RS.h, diff = 0;
    uint32_t n = 0;
#define PUT(v) do { uint64_t v_ = (uint64_t)(v); diff |= h[n] ^ v_; h[n++] = v_; } while (0)
    PUT((uintptr_t)info);
    PUT((uintptr_t)ex->prog);
    PUT(ex->prog->serial);
    PUT((uint64_t)vk_texture_slots_gen() << 32 | vk_texture_epoch());
    PUT((uint64_t)vk_sampler_gen() << 32 | (uint32_t)ex->nfbl);
    PUT(vk_mem_gen() + vk_usnap_gen());   // both only grow
    PUT(vk_texture_asof_limit());         // a chunk of another stretch may resolve the same slots otherwise
    for (int i = 0; i < ex->nfbl && i < 9; i++) PUT((uintptr_t)ex->fbl_img[i]);
    if (l->sup) {
        float in[3];
        support_in(ex, in);
        uint64_t a;
        uint32_t ps;
        memcpy(&a, in, 8), memcpy(&ps, &in[2], 4);
        PUT(a);
        PUT(ps);
    }
    if (l->c0) PUT(c0_binst);
#ifdef GAME_NS2
    if (l->c0) PUT(c0_mesh_first);
#endif
    for (int i = 0; i < l->np; i++) PUT(ubo_ptr(ex, l->p[i].st, l->p[i].res));
#undef PUT
    diff |= n != RS.hn;
    RS.hn = n;
    int fast = RS.info == info && RS.gen == bind_gen;
    if (fast && RS.on != 2) return !diff;
    const uint8_t *e = (const uint8_t *)ex;
    uint8_t *d = RS.body;
    int bd = 0;
    for (uint32_t i = 0; i < l->n; i++) {
        const RsRange *r = &l->r[i];
        if (!bd && memcmp(d, e + r->off, r->len)) bd = 1;
        if (bd) memcpy(d, e + r->off, r->len);
        d += r->len;
    }
    if (fast && bd) rt_fatal("vk_exec: descriptor build skip check: bound slots changed without a bind op");
    RS.info = info, RS.gen = bind_gen;
    return !diff && !bd;
}

// Writes allocated set s from its packed descriptors d: through the layout's update template, or with the write
// structs w[i0..i1) when it has none.
static void write_set(const VkPipeInfo *info, uint32_t s, VkDescriptorSet set, const VkDescSlot *d, VkWriteDescriptorSet *w,
                      uint32_t i0, uint32_t i1) {
    if (i1 == i0) return;
    if (info->tpl[s]) {
        vk_pipeline_update_tpl_fn()(vk_ctx()->device, set, info->tpl[s], d);
        return;
    }
    for (uint32_t k = i0; k < i1; k++) w[k].dstSet = set;
    vkUpdateDescriptorSets(vk_ctx()->device, i1 - i0, w + i0, 0, NULL);
}

// Writes the pipeline's sets from the bound state: set 0, and set 1 with independent layouts. Each is pushed into the
// command buffer, or allocated, written and bound; a graphics set whose contents equal the last one allocated for that
// set number in this command buffer is reused, and bound again only when another layout or a push came between. A
// graphics push equal to the last one at its set number, with no other layout or bind between, is not recorded.
// The descriptors go into one packed array d, a VkDescSlot per descriptor in binding order, which a set's update
// template reads directly (VkPipeInfo.tpl); write structs are filled only when a set has no template.
// What bind_resources_ derived from the info alone at every draw: the bindings of each set and each binding's slot. The
// loop over the bindings to find the sets took 8 % of bind_resources' samples in the village and the load of
// res->slot, one reflection entry per binding, 19 % (tools: REKARINA_PROFILE with llvm-objdump -d -l).
static void pipe_prep(VkPipeInfo *info) {
    uint32_t n_sets = info->n_sets ? info->n_sets : 1;
    VkPipeBinding *pb = (VkPipeBinding *)info->bindings;
    info->need_w = 0;
    for (uint32_t s = 0, i0 = 0; s < 2; s++) {
        uint32_t i1 = i0;
        if (s < n_sets)
            while (i1 < info->n_bindings && (n_sets == 1 || pb[i1].set == s)) i1++;
        info->set_lo[s] = i0, info->set_hi[s] = i1, i0 = i1;
        if (s < n_sets && i1 > info->set_lo[s] && !info->tpl[s]) info->need_w = 1;
    }
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        pb[i].nst = (int8_t)nvn_stage_of(pb[i].stage);
        pb[i].slot = (int16_t)(pb[i].res ? pb[i].res->slot : 0);
    }
    info->prep = 1;
}

static void bind_resources_(VkExec *ex, const VkPipeInfo *info);
static void bind_resources(VkExec *ex, const VkPipeInfo *info) {
    uint64_t t = sec_begin();
    bind_resources_(ex, info);
    sec_end(info->bind_point == VK_PIPELINE_BIND_POINT_COMPUTE ? SEC_BIND_CS : SEC_BIND_RES, t);   // BIND_CS: dispatches
}

static int tex_hold_short(void) {   // REKARINA_TEX_HOLD: binding (the default since 2026-11-17) or draw (as before)
    static int on = -1;
    if (on < 0) { const char *e = cfg_get("REKARINA_TEX_HOLD"); on = !(e && !strcmp(e, "draw")); }
    return on;
}

static void bind_resources_(VkExec *ex, const VkPipeInfo *info) {
    static _Thread_local VkWriteDescriptorSet w[MAX_BINDINGS];
    static _Thread_local VkDescSlot d[MAX_DESC];
    static _Thread_local VkBufferView bv[MAX_DESC];
    if (info->n_bindings > MAX_BINDINGS) rt_fatal("vk: %u descriptor bindings exceed the replay's %d", info->n_bindings, MAX_BINDINGS);
    bind_stats(info);
    int gfx = info->bind_point == VK_PIPELINE_BIND_POINT_GRAPHICS;
    int skip = gfx ? bind_skip_on() : 0, same = 0, was_valid = B.res_valid;
    if (skip) {
        same = bind_sig(ex, info);
        if (same && was_valid && skip == 1) {
            for (int i = 0; i < RS.ncnt; i++) vk_mem_count(RS.cnt[i].cls, RS.cnt[i].use, RS.cnt[i].bytes);
            BS.bind_skip++;
            ex->recorded++;
            return;
        }
        B.res_valid = 0;   // until the build below ends
        if (skip == 2) memcpy(RS.pcnt, RS.cnt, sizeof RS.cnt), RS.pncnt = RS.ncnt;
        RS.rec = 1, RS.ncnt = 0, RS.cnt_over = 0, RS.volatile_ = 0, RS.cmds = 0;
    }
    uint32_t n_sets = info->n_sets ? info->n_sets : 1;
    if (!info->prep) pipe_prep((VkPipeInfo *)info);
    // bindings [lo, hi) and first descriptor per set; need_w: a set without a template, fill the write structs
    uint32_t lo[2] = {info->set_lo[0], info->set_lo[1]}, hi[2] = {info->set_hi[0], info->set_hi[1]}, d0[2] = {0, 0};
    int need_w = info->need_w;
    uint32_t nd = 0;
    int tlocked = 0;   // the texture lock, taken at the draw's first texture lookup and held to its last
    int sc = gfx && !mo_ovr_on ? slot_cache_on() : 0;   // "descriptors kept per bound slot"
    if (sc) slot_cache_stamp(ex);
    int sc_tex = sc && !ex->nfbl && !vk_dlaa_swap_img && !ex->draw_role;
    td_lock = &tlocked;
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *b = &info->bindings[i];
        const GfxShaderRes *r = b->res;
        uint32_t cnt = b->count ? b->count : 1;
        if (nd + cnt > MAX_DESC) rt_fatal("vk: %u descriptors exceed the replay's %d", nd + cnt, MAX_DESC);
        if (n_sets == 2 && i == lo[1]) d0[1] = nd;
        VkDescSlot *o = &d[nd];
        int st = b->nst, kind = 0;   // kind: 0 buffer, 1 image, 2 texel buffer view
        if (b->kind == VK_PIPE_SUPPORT) {
            float in[3];
            support_in(ex, in);
            if (!B.sup_valid || memcmp(in, B.sup_in, sizeof in)) {   // support_fill reads only these
                uint8_t tmp[SUPPORT_BYTES];
                support_fill(tmp, in[0], in[1], in[2]);
                memcpy(B.sup_in, in, sizeof in);
                if (!B.sup_valid || memcmp(tmp, B.support, SUPPORT_BYTES)) {   // a new buffer only when the contents changed
                    VkBuffer sb;
                    VkDeviceSize so;
                    memcpy(scratch(ex, SUPPORT_BYTES, &sb, &so), tmp, SUPPORT_BYTES);
                    memcpy(B.support, tmp, SUPPORT_BYTES);
                    B.sup_buf = sb, B.sup_off = so, B.sup_valid = 1;
                }
            }
            o->buffer = (VkDescriptorBufferInfo){B.sup_buf, B.sup_off, SUPPORT_BYTES};
        } else if (b->kind == GFX_RES_UBO) {
            for (uint32_t k = 0; k < cnt; k++) {
                int slot = b->slot + (int)k;
                if (!sc || (unsigned)st >= 6 || slot < 3 || slot - 3 >= 18) {
                    o[k].buffer = ubo_info(ex, st, slot);
                    continue;
                }
                const p_bind_buf *u = &ex->ubo[st][slot - 3];
                ScBuf *e = &SC.ubo[st][slot - 3];
                if (e->stamp == SC.stamp && e->addr == u->addr && e->size == u->size) {
                    o[k].buffer = e->out;
                    if (e->n) count_use(e->cls, e->use, e->n);
                    SC.hits++;
                    if (sc == 2) {
                        VkDescriptorBufferInfo x = ubo_info(ex, st, slot);
                        if (x.buffer != e->out.buffer || x.offset != e->out.offset || x.range != e->out.range)
                            rt_fatal("vk_exec: slot descriptor check: uniform slot %d of stage %d (%#llx + %#llx) kept as offset %#llx "
                                     "range %#llx, a lookup gives offset %#llx range %#llx%s", slot, st, (unsigned long long)u->addr,
                                     (unsigned long long)u->size, (unsigned long long)e->out.offset, (unsigned long long)e->out.range,
                                     (unsigned long long)x.offset, (unsigned long long)x.range,
                                     x.buffer != e->out.buffer ? " in another buffer" : "");
                    }
                    continue;
                }
                int vol = RS.volatile_;
                RS.volatile_ = 0;
                sc_cap = (typeof(sc_cap)){.on = 1};
                o[k].buffer = ubo_info(ex, st, slot);
                sc_cap.on = 0;
                if (RS.volatile_) e->stamp = 0;   // a copy made for this draw: the next draw gets its own
                else *e = (ScBuf){SC.stamp, u->addr, u->size, o[k].buffer, sc_cap.cls, sc_cap.use, sc_cap.n};
                RS.volatile_ |= vol;
                SC.misses++;
            }
#ifdef GAME_OOT
        } else if (b->kind == GFX_RES_SSBO && r->cbuf_slot == GFX_CB_HDR_METER) {
            o->buffer = vk_hdr_meter_bind(ex);
            if (!o->buffer.buffer) o->buffer = zero_info(VK_WHOLE_SIZE);
#endif
        } else if (b->kind == GFX_RES_SSBO) {
            {   // DEBUG-POOLTAB
                static int tab = -1;
                if (tab < 0) { const char *e = cfg_get("REKARINA_POOLTAB"); tab = e && *e == '1'; }
                if (tab && info->pool && r->cbuf_slot != GFX_CB_GMEM) pooltab_log(ex, st, r, b->binding);
            }
            for (uint32_t k = 0; k < cnt; k++) o[k].buffer = ssbo_info(ex, st, r, r->flags & GFX_SSBO_FADD ? 2 : r->flags & 1);
#ifdef GAME_NS2
            // a storage buffer of a draw that reads the texture pool: one of the tables its shaders take texture
            // handles from (vk_texture.c "handle tables")
            if (info->pool && r->cbuf_slot != GFX_CB_GMEM) {
                int ubo, slot = ssbo_slot(r, st, &ubo);
                uint64_t ta = 0;
                if (slot >= 0 && slot < 16) ta = ubo ? ex->ubo[st][slot].addr : ex->ssbo[st][slot].addr;
                else if (ubo_ptr_res(r)) ta = ubo_ptr(ex, st, r);
                if (ta && o[0].buffer.range && o[0].buffer.range != VK_WHOLE_SIZE)
                    vk_texture_handle_table(ta - (vk_caps.sbuf_align ? ta & (vk_caps.sbuf_align - 1) : 0), o[0].buffer.range);
            }
#endif
        } else if (b->kind == GFX_RES_TEX || b->kind == GFX_RES_IMG) {
            int img = b->kind == GFX_RES_IMG;
            if (b->type == VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER || b->type == VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER) {
                kind = 2;
                for (uint32_t k = 0; k < cnt; k++) {   // buffer textures are not replayed yet
                    o[k].w[0] = 0, o[k].w[1] = 0, o[k].w[2] = 0;
                    o[k].view = Z.zview;
                    if (need_w) bv[nd + k] = Z.zview;
                }
            } else {
                kind = 1;
                // tex_info looks up with the texture lock held. Locking and unlocking around each binding was ~6,000
                // pairs a frame in forest village, 0.3-0.4 ms of the replay thread inside the lock's own code. A draw
                // whose textures are all kept per slot takes no lock at all.
                for (uint32_t k = 0; k < cnt; k++) {
                    int slot = b->slot + (int)k;
                    uint64_t h = 0;
                    if (img) h = slot >= 0 && slot < 32 ? ex->img[st][slot] : 0;
                    else h = slot >= 0 && slot < 128 ? ex->tex[st][slot] : 0;
                    ScTex *e = sc_tex && !img && h && (unsigned)st < 6 ? &SC.tex[st][slot] : NULL;
                    if (e && e->stamp == SC.stamp && e->h == h && e->type == r->type && e->separate == r->separate) {
                        o[k].image = e->out;
                        o[k].w[2] = (uint32_t)o[k].image.imageLayout;
                        SC.hits++;
                        if (sc == 2) {
                            VkDescriptorImageInfo x;
                            if (!tlocked) vk_texture_lock(), tlocked = 1;
                            tex_info(ex, r, h, img, r->separate, &x);
                            if (x.sampler != e->out.sampler || x.imageView != e->out.imageView || x.imageLayout != e->out.imageLayout)
                                rt_fatal("vk_exec: slot descriptor check: texture slot %d of stage %d (handle %#llx, type %#x) "
                                         "kept with another %s than a lookup gives", slot, st, (unsigned long long)h, r->type,
                                         x.imageView != e->out.imageView ? "view" : x.sampler != e->out.sampler ? "sampler" : "layout");
                        }
                        continue;
                    }
                    tex_info(ex, r, h, img, r->separate, &o[k].image);   // takes the texture lock when it looks up
                    if (!img && st == NVN_STAGE_FS && b->type != VK_DESCRIPTOR_TYPE_SAMPLER &&
                        ex->draw_role && vk_dlaa_prebloom_active()) {
                        TD_LOCK();
                        VkImageView reconstructed = vk_dlaa_prebloom_view(ex, h);
                        if (reconstructed) {
                            o[k].image.imageView = reconstructed, o[k].image.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
                            if (vk_dlaa_pathlog())
                                rt_log("PATHLOG bloom role %u FS slot %d handle %#llx uses reconstructed scene",
                                       ex->draw_role, slot, (unsigned long long)h);
                        }
                    }
                    o[k].w[2] = (uint32_t)o[k].image.imageLayout;   // zero the tail padding
                    if (e) *e = (ScTex){SC.stamp, h, r->type, r->separate, o[k].image}, SC.misses++;
                }
                // REKARINA_TEX_HOLD=binding: the lock a lookup of this binding took goes now, not at the draw's last
                // binding: the buffers resolved after it (uniform and storage ranges, the mirrors' notes) ran with the
                // texture lock held, and the replay thread waited behind chunk threads' holds (A's LOCKWHO 2026-11-17).
                if (tlocked && tex_hold_short()) vk_texture_unlock(), tlocked = 0, tds_hold_cuts++;
            }
        } else {
            rt_fatal("vk: binding %u has an unknown kind %d", (unsigned)b->binding, (int)b->kind);
        }
        if (need_w) {
            w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                                          .dstBinding = b->binding, .descriptorCount = cnt, .descriptorType = b->type};
            if (kind == 0) w[i].pBufferInfo = &o->buffer;   // a VkDescSlot is as large as either info struct
            else if (kind == 1) w[i].pImageInfo = &o->image;
            else w[i].pTexelBufferView = &bv[nd];
        }
        nd += cnt;
    }
    td_lock = NULL;
    if (tlocked) vk_texture_unlock();
    // Under the validation layer, a new graphics layout binds set 1 before set 0 is pushed or bound. The layer (SDK
    // 1.4.363, PipelineLayoutCompatDef::operator==) takes any two INDEPENDENT_SETS layouts as compatible, so
    // vkCmdPushDescriptorSetKHR keeps the push set it made for an earlier layout and drops writes to bindings that
    // layout lacks (false VUID-vkCmdDrawIndexed-None-08114, then an access violation inside the layer). A bind at set 1
    // disturbs set 0 by its canonical id, which drops the layer's push set: the push after it gets a new one.
    int set1_first = 0;
    if (gfx && info->layout != B.ds_layout) {   // another layout: sets bound before may be disturbed
        B.ds_layout = info->layout;
        B.ds_bound[0] = B.ds_bound[1] = VK_NULL_HANDLE;
        B.ds_pushed[0] = B.ds_pushed[1] = 0;
        set1_first = n_sets == 2 && vk_ctx()->validation;
    }
    for (uint32_t k = 0; k < n_sets; k++) {
        uint32_t s = set1_first ? 1 - k : k, i0 = lo[s], i1 = hi[s];
        const VkDescSlot *ds = &d[d0[s]];
        uint32_t dbytes = (uint32_t)(((s + 1 < n_sets ? d0[s + 1] : nd) - d0[s]) * sizeof *d);
        if (s && i1 == i0) {               // no fragment stage: set 1 is empty and unused
            if (set1_first) {              // bound only to drop the layer's push set (comment above)
                VkDescriptorSet set = alloc_set(ex, info->set_layout1);
                vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, 1, 1, &set, 0, NULL);
                B.ds_bound[1] = set, B.ds_pushed[1] = 0;
                RS.cmds++;
            }
            continue;
        }
        if (s ? info->push1 : info->push) {
            uint32_t n = gfx && push_skip_on() ? dbytes : 0;
            if (n && B.ds_pushed[s] && DS.pn[s] == n && !memcmp(DS.psig[s], ds, n)) {
                BS.push_skip++;
                continue;
            }
            if (info->tpl[s]) rec_push_tpl(vk_pipeline_push_tpl_fn(), ex->cb, info->tpl[s], info->layout, s, ds, dbytes);
            else rec_push(vk_pipeline_push_fn(), ex->cb, info->bind_point, info->layout, s, i1 - i0, w + i0);
            BS.push++;
            RS.cmds++;
            if (gfx) {
                B.ds_bound[s] = VK_NULL_HANDLE, B.ds_pushed[s] = n != 0;
                if (n) DS.pn[s] = n, memcpy(DS.psig[s], ds, n);
            }
        } else if (!gfx || s > 1) {
            VkDescriptorSet set = alloc_set(ex, s ? info->set_layout1 : info->set_layout);
            write_set(info, s, set, ds, w, i0, i1);
            vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, s, 1, &set, 0, NULL);
        } else {
            VkDescriptorSetLayout sl = s ? info->set_layout1 : info->set_layout;
            VkDescriptorSet set = B.ds_last[s];
            if (!set || B.ds_last_layout[s] != sl || DS.n[s] != dbytes || memcmp(DS.sig[s], ds, dbytes)) {
                set = alloc_set(ex, sl);
                write_set(info, s, set, ds, w, i0, i1);
                B.ds_last[s] = set, B.ds_last_layout[s] = sl, DS.n[s] = dbytes;
                memcpy(DS.sig[s], ds, dbytes);
                BS.set_new++;
                RS.cmds++;
            } else {
                BS.set_reuse++;
            }
            if (B.ds_bound[s] != set) {
                vkCmdBindDescriptorSets(ex->cb, info->bind_point, info->layout, s, 1, &set, 0, NULL);
                B.ds_bound[s] = set, B.ds_pushed[s] = 0;
                RS.cmds++;
            }
        }
    }
    // The texture pool's set (vk_texpool.c) after sets 0 and 1: binding those with another layout first would disturb it.
    // Once per layout and command buffer: the set stays bound while the layout it was bound with stays in use. Bound again
    // for every draw it was 2,400 vkCmdBindDescriptorSets a frame in the forest (0.28 ms inside the driver).
    // REKARINA_POOL_BIND_KEEP=0: a bind per draw.
    // A draw with another layout, with the pool or without, may disturb the set: the next one binds it again.
    {
        static int keep = -1;
        if (keep < 0) keep = rt_knob("POOL_BIND_KEEP", "1");
        int pk = info->bind_point == VK_PIPELINE_BIND_POINT_COMPUTE;
        if (!keep || B.pool_layout[pk] != info->layout) {
            B.pool_layout[pk] = VK_NULL_HANDLE;
            if (info->pool) {
                vk_tpool_bind(ex->cb, info->bind_point, info->layout);
                B.pool_layout[pk] = info->layout;
            }
        }
    }
    if (skip) {
        RS.rec = 0;
        // An equal next draw records nothing when every push set holds its push (REKARINA_PUSH_SKIP on) and nothing made
        // the build unrepeatable; allocated sets are always bound to the one written last.
        int valid = !RS.volatile_ && !RS.cnt_over && RS.hn;
        for (uint32_t s = 0; s < n_sets; s++)
            if (hi[s] > lo[s] && (s ? info->push1 : info->push) && !B.ds_pushed[s]) valid = 0;
        if (skip == 2) {   // with the pipeline info equal (bind_sig), the packed descriptors determine every write
            uint32_t n = (uint32_t)(nd * sizeof *d);
            int wr = n != RS.ckn || memcmp(RS.ck, d, n);
            int cn = RS.ncnt != RS.pncnt || memcmp(RS.cnt, RS.pcnt, sizeof RS.cnt[0] * (size_t)RS.ncnt);
            if (same && was_valid && (wr || cn || RS.cmds)) {   // DEBUG-SKIPCHECK: which descriptor differs
                const uint64_t *a = (const uint64_t *)RS.ck, *b2 = (const uint64_t *)d;
                uint32_t words = sizeof *d / 8, dn = n < RS.ckn ? n : RS.ckn;
                rt_log("skip check: wr %d cn %d cmds %d, %u descriptor bytes (were %u)", wr, cn, RS.cmds, n, RS.ckn);
                for (uint32_t i = 0, at = 0, shown = 0; i < info->n_bindings && shown < 6; i++) {
                    const VkPipeBinding *pb = &info->bindings[i];
                    uint32_t cnt = pb->count ? pb->count : 1;
                    for (uint32_t k = 0; k < cnt && shown < 6; k++, at++) {
                        if ((at + 1) * sizeof *d > dn || !memcmp(&a[at * words], &b2[at * words], sizeof *d)) continue;
                        rt_log("skip check: binding %u[%u] kind %d type %d stage %d slot %d: was %#llx %#llx %#llx, is %#llx %#llx %#llx",
                               i, k, (int)pb->kind, (int)pb->type, (int)pb->nst, (int)pb->slot + (int)k,
                               (unsigned long long)a[at * words], (unsigned long long)a[at * words + 1],
                               (unsigned long long)(words > 2 ? a[at * words + 2] : 0), (unsigned long long)b2[at * words],
                               (unsigned long long)b2[at * words + 1], (unsigned long long)(words > 2 ? b2[at * words + 2] : 0));
                        shown++;
                    }
                }
                rt_fatal("vk_exec: descriptor build skip check: a draw of program %llu with the inputs of the previous one "
                         "gets %s", (unsigned long long)ex->prog->serial,
                         RS.cmds ? "descriptor commands" : wr ? "other writes" : "other census counts");
            }
            if (same && was_valid) BS.bind_skip++;   // the draws a skip would have spared, compared instead
            RS.ckn = n;
            memcpy(RS.ck, d, n);
        }
        B.res_valid = valid;
    }
    ex->recorded++;
}

// ---- draws

static void bind_pipeline(VkExec *ex, VkPipelineBindPoint bp, VkPipeline p) {
    int k = bp == VK_PIPELINE_BIND_POINT_COMPUTE;
    if (B.pipe[k] == p) return;
    vkCmdBindPipeline(ex->cb, bp, p);
    ex->recorded++;
    B.pipe[k] = p;
    if (!k) B.st_valid = 0;
}

static VkViewport base_viewport(const VkExec *ex) {
    float x = 0, y = 0, w = (float)ex->pass_w, h = (float)ex->pass_h;
    if (ex->has_vp) {
        float k = ex->pass_scale > 1 ? (float)ex->pass_scale : 1.0f;
        x = ex->vp[0][0] * k, y = ex->vp[0][1] * k, w = ex->vp[0][2] * k, h = ex->vp[0][3] * k;
    }
    if (w < 1) w = 1;
    if (h < 1) h = 1;
    return (VkViewport){x, y + h, w, -h, ex->depth_range[0], ex->depth_range[1]};   // the y flip: negative height
}

static VkViewport draw_viewport(VkExec *ex) {
    VkViewport v = base_viewport(ex);
    vk_dlaa_viewport(ex, &v);   // the scene's draws move by the DLAA jitter
    return v;
}

static void record_fixed(VkExec *ex) {
    VkCommandBuffer cb = ex->cb;
    if (ex->dirty & VK_EXEC_DIRTY_VIEWPORT) {
        VkViewport v = draw_viewport(ex);
        vkCmdSetViewportWithCount(cb, 1, &v);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_SCISSOR) {
        VkRect2D r = vk_exec_scissor(ex);
        vkCmdSetScissorWithCount(cb, 1, &r);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_DEPTH_BIAS) {
        vkCmdSetDepthBias(cb, ex->poly_offset.units, ex->poly_offset.clamp, ex->poly_offset.factor);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_BLEND_CONST) {
        vkCmdSetBlendConstants(cb, ex->blend_color);
        ex->recorded++;
    }
    if (ex->dirty & VK_EXEC_DIRTY_STENCIL) {
        for (int f = 0; f < 2; f++) {
            VkStencilFaceFlags face = f ? VK_STENCIL_FACE_BACK_BIT : VK_STENCIL_FACE_FRONT_BIT;
            vkCmdSetStencilCompareMask(cb, face, ex->st_vmask[f]);
            vkCmdSetStencilWriteMask(cb, face, ex->st_wmask[f]);
            vkCmdSetStencilReference(cb, face, ex->st_ref[f]);
            ex->recorded += 3;
        }
    }
    ex->dirty = 0;
}

static void bind_vertex_buffers(VkExec *ex) {
    VkBuffer vb[17];
    VkDeviceSize vo[17], vsz[17], vstr[17];
    int vcls[17] = {0};
    void *vpool[17] = {0};
    int dyn = vk_pipeline_vertex_input_dynamic();
    // A stream whose range is the one its bound buffer was resolved from, with no pool or mirror change since
    // (vk_mem_gen, read before any resolve), resolves to that buffer again: most draws change one stream or none.
    uint64_t gen = vk_mem_gen();
    int same_gen = B.vb_valid && B.vgen == gen;
    for (int b = 0; b < 17; b++) {
        vb[b] = Z.zbuf, vo[b] = 0, vsz[b] = ZERO_BYTES, vstr[b] = b < 16 ? ex->pipe.vstream[b].stride : 0;
        if (b == VK_PIPE_ZERO_BINDING) continue;
        const NVNbufferRange *r = &ex->vbuf[b];
        if (same_gen && B.vin[b].address == r->address && B.vin[b].size == r->size) {
            vb[b] = B.vb[b], vo[b] = B.vo[b], vsz[b] = B.vsz[b], vcls[b] = B.vcls[b], vpool[b] = B.vpool[b];
            continue;
        }
        VkMemSpan sp;
        if (!r->address || !vk_mem_resolve(r->address, 1, &sp)) continue;
        VK_MEM_USE(sp);
        uint64_t n = r->size && r->size < sp.avail ? r->size : sp.avail;
        vb[b] = sp.buffer, vo[b] = sp.offset, vsz[b] = n, vcls[b] = sp.cls, vpool[b] = sp.pool;
    }
    memcpy(B.vcls, vcls, sizeof vcls), memcpy(B.vpool, vpool, sizeof vpool);
    memcpy(B.vin, ex->vbuf, sizeof B.vin);
    B.vgen = gen;
    if (B.vb_valid && !memcmp(B.vb, vb, sizeof vb) && !memcmp(B.vo, vo, sizeof vo) && !memcmp(B.vsz, vsz, sizeof vsz) &&
        (dyn || !memcmp(B.vstr, vstr, sizeof vstr)))
        return;
    vkCmdBindVertexBuffers2(ex->cb, 0, 17, vb, vo, vsz, dyn ? NULL : vstr);
    ex->recorded++;
    for (int b = 0; b < 17; b++)
        if (vcls[b]) vk_mem_count(vcls[b], VK_MUSE_VB, vsz[b]);
    if (rt_trace & RT_TRACE_PERF)
        for (int b = 0; b < 17; b++)
            if (vpool[b]) vk_mem_bound_note(vpool[b]);
    memcpy(B.vb, vb, sizeof vb), memcpy(B.vo, vo, sizeof vo), memcpy(B.vsz, vsz, sizeof vsz), memcpy(B.vstr, vstr, sizeof vstr);
    B.vb_valid = 1;
}

// REKARINA_PIPE_SKIP: which draws are skipped, rather than waited for, while their shaders or pipeline build.
// strict (0): only draws whose pass targets were all rendered in the previous presented frame; a pass with a target
//   that was not counts as one-shot and waits.
// any (1): also draws in a pass with at least one target rendered in the previous frame (a per-frame depth
//   buffer or g-buffer with a colour target the frame before did not use: half-resolution effect buffers in fights).
//   Passes with none (volumes, LUTs, shadow layers seen for the first time) still wait, and so do passes whose only
//   recent target is the depth buffer while a colour target is rendered for the first time (item icon captures).
// fullscreen (2, default): as any, and full-screen draws into such passes are skipped instead of waiting from their
//   budget. A full-screen pass whose stage libraries are cold in the driver cache took 86 ms on average (up to 194 ms)
//   in a fight on an RX 9070: three frames in a row each spent the 50 ms budget, and some of the draws were skipped
//   after all. Skipping costs the pass for the frames until the pipeline is ready.
static int pipe_skip_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *e = cfg_value("REKARINA_PIPE_SKIP");
        mode = !strcmp(e, "strict") || !strcmp(e, "0") ? 0 : !strcmp(e, "fullscreen") || !strcmp(e, "2") ? 2 : 1;
        static const char *const what[3] = {"only draws into per-frame passes (REKARINA_PIPE_SKIP=strict)",
                                            "draws into passes with a target the previous frame rendered",
                                            "those and full-screen draws (REKARINA_PIPE_SKIP=fullscreen)"};
        rt_log("vk_exec: draws skipped while their pipeline builds: %s", what[mode]);
    }
    return mode;
}

// The previous draw's shaders, kept while its program, bound pipeline state, the vertex and fragment stages' first 32
// texture and image handles and every slot's resolution (vk_texture_slots_gen, read unlocked: a registration racing
// with it is ordered after this draw, as answers_for's own check would order it) stay the same: answers_for reads
// nothing else, so shader_for would return the same shaders. About 65% of field draws repeat them
// (REKARINA_BIND_STATS). REKARINA_SHADER_MEMO=0 turns it off; =check looks the shaders up on every hit too, and every slot's
// texture type in answers_for, and stops on a difference.
// tex_serial (bumped by every texture and image bind) equal to the key's means the handles were not rebound since
// they were compared, so the 1 KiB compare is skipped; a rebind to the same handles still matches by the compare.
typedef struct {
    const prog_t *prog;
    uint64_t serial;
    uint32_t gen, tex_serial;
    uint64_t tex[2][32], img[2][32];
    VkPipeState pipe;
} MemoKey;
static _Thread_local struct { int on, valid, has_fs; MemoKey k; const GfxShader *vs, *fs; } SM = {-1};

static void memo_key(const VkExec *ex, uint32_t gen, MemoKey *k) {
    k->prog = ex->prog, k->serial = ex->prog->serial, k->gen = gen, k->tex_serial = ex->tex_serial;
    memcpy(k->tex[0], ex->tex[NVN_STAGE_VS], sizeof k->tex[0]), memcpy(k->tex[1], ex->tex[NVN_STAGE_FS], sizeof k->tex[1]);
    memcpy(k->img[0], ex->img[NVN_STAGE_VS], sizeof k->img[0]), memcpy(k->img[1], ex->img[NVN_STAGE_FS], sizeof k->img[1]);
    memcpy(&k->pipe, &ex->pipe, sizeof k->pipe);
}

static int memo_match(const VkExec *ex, uint32_t gen, MemoKey *k) {
    if (k->prog != ex->prog || k->serial != ex->prog->serial || k->gen != gen || memcmp(&k->pipe, &ex->pipe, sizeof k->pipe))
        return 0;
    if (k->tex_serial == ex->tex_serial) return 1;
    if (memcmp(k->tex[0], ex->tex[NVN_STAGE_VS], sizeof k->tex[0]) || memcmp(k->tex[1], ex->tex[NVN_STAGE_FS], sizeof k->tex[1]) ||
        memcmp(k->img[0], ex->img[NVN_STAGE_VS], sizeof k->img[0]) || memcmp(k->img[1], ex->img[NVN_STAGE_FS], sizeof k->img[1]))
        return 0;
    k->tex_serial = ex->tex_serial;
    return 1;
}

static int shader_memo_on(void) {
    if (SM.on < 0) {
        const char *e = cfg_value("REKARINA_SHADER_MEMO");
        SM.on = !strcmp(e, "check") ? 2 : strcmp(e, "0") != 0;
    }
    return SM.on;
}

static _Thread_local struct { int on, valid; VkPipeState s; const VkPipeInfo *info; } PM = {-1};

static int pipe_memo_on(void) {
    if (PM.on < 0) PM.on = cfg_bool("REKARINA_PIPE_MEMO");
    return PM.on;
}

// REKARINA_SECTIONS, log only: whether an 8-entry table of recent picks would have turned a memo miss into a hit.
static _Thread_local uint64_t sim_shader[8], sim_pipe[8];
static uint64_t sim_mix(uint64_t h, const void *p, size_t n) {
    const uint64_t *w = p;
    for (size_t i = 0; i < n / 8; i++) h = (h ^ w[i]) * 0x9E3779B97F4A7C15ull;
    return h | 1;
}
static void sim_mix_hit(int kind, uint64_t *t, uint64_t key) {
    int i;
    for (i = 0; i < 8 && t[i] != key; i++) {}
    if (i < 8) sec_count(kind, 1);
    else i = 7;
    memmove(t + 1, t, sizeof *t * (size_t)i);
    t[0] = key;
}

// Pass, shaders, pipeline, fixed state and resources for a draw. 0: the draw is skipped.
#ifdef GAME_NS2
static int stages_has(VkExec *ex, int st);
static int prepare_stages(VkExec *ex, int mesh, int prim, int restart);

// DEBUG-SKIPSH: REKARINA_SKIP_SH=<hash>[,<hash>...]: draws with a shader whose SPIR-V FNV-1a hash (vk_dump.c shader_hash,
// as the dump lists it) is one of these are skipped, to see what they draw.
static int skip_sh(const GfxShader *s) {
    static int n = -1;
    static uint64_t want[16];
    static struct { const GfxShader *s; int hit; } memo[4096];
    static int nm;
    if (n < 0) {
        n = 0;
        const char *e = cfg_get("REKARINA_SKIP_SH");
        while (e && *e && n < 16) {
            want[n++] = strtoull(e, (char **)&e, 16);
            while (*e == ',') e++;
        }
    }
    if (!n || !s) return 0;
    for (int i = 0; i < nm; i++)
        if (memo[i].s == s) return memo[i].hit;
    uint64_t words = 0, h = 0xcbf29ce484222325ull;
    const uint8_t *w = (const uint8_t *)gfx_shader_spirv(s, &words);
    for (uint64_t i = 0; w && i < words * 4; i++) h = (h ^ w[i]) * 0x100000001b3ull;
    int hit = 0;
    for (int i = 0; i < n; i++) hit |= want[i] == h;
    if (nm < 4096) memo[nm].s = s, memo[nm].hit = hit, nm++;
    return hit;
}
#endif
int vk_dbg_cond_unknown, vk_dbg_cond_why;   // DEBUG-SKIPLOG: the last RENDER_COND rendered because its count was not known
static void dbg_skipcount(VkExec *ex, int skipped) {   // DEBUG-SKIPLOG: REKARINA_SKIPLOG=lo-hi, per frame
    static int on = -1;
    static unsigned long long lo, hi, lastf;
    static int nskip, ncond, nall, nunk, ngpu, why[16];
    if (on < 0) { const char *e = cfg_env("REKARINA_SKIPLOG"); on = e && sscanf(e, "%llu-%llu", &lo, &hi) == 2; }
    if (!on) return;
    unsigned long long f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (f != lastf) {
        if (lastf >= lo && lastf <= hi)
            rt_log("SKIPLOG frame %llu draws %d conditional %d skipped %d gpu %d unknown %d why %d %d %d %d %d %d %d %d", lastf, nall, ncond, nskip, ngpu, nunk,
                   why[1], why[2], why[3], why[4], why[10], why[11], why[12], why[13]);
        lastf = f, nskip = ncond = nall = nunk = ngpu = 0;
        memset(why, 0, sizeof why);
    }
    nall++;
#ifdef GAME_NS2
    if (ex->gpu_cond) ngpu++;
    else
#endif
    if (ex->has_render_cond && !skipped && vk_dbg_cond_unknown) nunk++, why[vk_dbg_cond_why & 15]++;
    if (ex->has_render_cond) ncond++;
    if (skipped) nskip++;
}
static int interface_shader_writes_storage(const GfxShader *shader) {
    const GfxShaderRes *resources = NULL;
    int count = gfx_shader_resources(shader, GFX_RES_SSBO, &resources);
    for (int index = 0; index < count; ++index)
        if (resources[index].flags & 1) return 1;
    count = gfx_shader_resources(shader, GFX_RES_IMG, &resources);
    for (int index = 0; index < count; ++index)
        if (resources[index].flags & 4) return 1;
    return 0;
}

static void interface_replay_state(VkPipeState *coverage) {
    coverage->color_fmt[0] = NVN_FORMAT_RGBA8;
    coverage->chmask.mask = 8;
    coverage->blend[0].src_a = NVN_BLEND_FUNC_ONE;
    coverage->blend[0].dst_a = NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA;
    coverage->blend[0].eq_a = NVN_BLEND_EQUATION_ADD;
}

static void interface_prepare_replay(VkExec *execution, const VkPipeState *state, uint32_t feedback) {
    if (!vk_interface_coverage_candidate(execution)) return;
    const VkExecTarget *target = &execution->color[0];
    const blend_st *blend = &state->blend[0];
    // Use shader alpha as best-effort coverage for non-source-over blends without changing the game's color blend.
    if (feedback || execution->pass_ncolor != 1 || execution->pass_layers != 1 || execution->depth.image ||
        target->level || target->base_layer || target->is3d || target->samples != 1 || state->raster_discard ||
        state->ms.a2c || state->ms.cov2color_en || state->color.logic_op != NVN_LOGIC_OP_COPY ||
        state->color.alpha_func != NVN_ALPHA_FUNC_ALWAYS || !(state->color.blend_en & 1) || !state->fs ||
        vk_pipeline_target_wmask(state, 0) != 15 || interface_shader_writes_storage(state->fs)) {
        static unsigned logged;
        if (logged++ < 4)
            rt_log("interface coverage: unsupported draw: feedback=%#x colors=%u layers=%u depth=%d samples=%u format=%d "
                   "alpha-test=%u logic=%u blend=%u equation=%u source=%u destination=%u mask=%u "
                   "level=%u layer=%u discard=%u alpha-to-coverage=%u coverage-to-color=%u storage-writes=%d", feedback,
                   execution->pass_ncolor, execution->pass_layers, execution->depth.image != VK_NULL_HANDLE, target->samples,
                   target->format, state->color.alpha_func, state->color.logic_op, state->color.blend_en,
                   blend->eq_rgb, blend->src_rgb, blend->dst_rgb, vk_pipeline_target_wmask(state, 0),
                   target->level, target->base_layer, state->raster_discard, state->ms.a2c, state->ms.cov2color_en,
                   state->fs ? interface_shader_writes_storage(state->fs) : 0);
        vk_interface_coverage_reject("draw state");
        return;
    }
    if (execution->gpu_cond) {
        vk_interface_coverage_reject("conditional replay draw");
        return;
    }
    VkPipeState coverage = *state;
    interface_replay_state(&coverage);
    const VkPipeInfo *info = NULL;
    VkPipeline pipeline = vk_pipeline_graphics(&coverage, &info);
    VkImageView view = pipeline && info ? vk_interface_coverage_begin(execution) : VK_NULL_HANDLE;
    if (!view) vk_interface_coverage_reject("alpha replay pipeline or checkpoint");
    execution->interface_replay_pipeline = pipeline;
    execution->interface_replay_info = info;
    execution->interface_replay_view = view;
}

void vk_dlaa_direct_draw(const VkExecTarget *target, uint32_t source, int encode);

#ifdef GAME_OOT
static void scene_output_pq(VkExec *ex) {
    const p_bind_buf *u = &ex->ubo[NVN_STAGE_FS][0];   // the checked PQ shader reads guest cbuf 3
    const uint64_t offset = 18 * 16;
    VkGuestPq *pq = &ex->output_pq;
    if (!u->addr || u->size < offset + sizeof pq->row) return;
    uint64_t address = u->addr + offset;
    VkMemSpan span;
    if (!upd_read(ex, address, pq->row, sizeof pq->row)) {
        if (nvn_gpu_only(address) || !vk_mem_resolve(address, sizeof pq->row, &span) || span.alloc ||
            span.avail < sizeof pq->row) return;
        memcpy(pq->row, (const void *)(uintptr_t)address, sizeof pq->row);
    }
    for (unsigned r = 0; r < 3; r++)
        for (unsigned c = 0; c < 3; c++)
            if (!__builtin_isfinite(pq->row[r][c])) return;
    pq->valid = 1;
}
#endif

static int display_composite_format(VkFormat format) {
    return format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || format == VK_FORMAT_R16G16B16A16_SFLOAT;
}

static const VkTexImage *scene_output_draw(VkExec *execution, const VkPipeState *state,
                                            const VkPipeInfo *information, int count) {
    const VkTexImage *none = NULL;
    if (!(execution->scene_output_role & DRAW_ROLE_SCENE_OUTPUT)) return none;
    const int encode = execution->scene_output_role == DRAW_ROLE_SCENE_ENCODE;
    if (!encode && !vk_dlaa_prebloom_enabled()) return none;
    const VkExecTarget *destination = &execution->color[0];
    if (!encode && vk_dlaa_pathlog())
        rt_log("PATHLOG direct composite draw: count %d prim %d targets %d layers %u depth %d/%d mask %#x",
               count, state->prim, execution->pass_ncolor, execution->pass_layers,
               state->ds.dt, state->ds.dw, vk_pipeline_target_wmask(state, 0));
    if (execution->pass_ncolor != 1 || execution->pass_layers != 1 || state->ds.dt || state->ds.dw ||
        execution->gpu_cond || state->feedback || state->raster_discard || (state->color.blend_en & 1) ||
        state->color.logic_op != NVN_LOGIC_OP_COPY || vk_pipeline_target_wmask(state, 0) != 15 ||
        count != (encode ? 6 : 3)) return none;
    if (!destination->img || destination->samples != 1 || destination->level || destination->base_layer || destination->is3d ||
        (encode ? execution->depth.image || (destination->format != VK_FORMAT_R8G8B8A8_SRGB && destination->format != VK_FORMAT_B8G8R8A8_SRGB &&
                                              destination->format != VK_FORMAT_A2B10G10R10_UNORM_PACK32)
                : !display_composite_format(destination->format))) return none;
    VkViewport viewport = draw_viewport(execution);
    VkRect2D scissor = vk_exec_scissor(execution);
    if (viewport.x != 0 || viewport.y != (float)destination->height || viewport.width != (float)destination->width ||
        viewport.height != -(float)destination->height || scissor.offset.x || scissor.offset.y ||
        scissor.extent.width != destination->width || scissor.extent.height != destination->height) return none;
    const GfxShaderRes *sampled = NULL;
    for (uint32_t index = 0; index < information->n_bindings; ++index) {
        const VkPipeBinding *binding = &information->bindings[index];
        if (binding->kind != GFX_RES_TEX || binding->stage != GFX_STAGE_FS) continue;
        if (sampled || !binding->res || binding->count > 1) return none;
        sampled = binding->res;
    }
    if (!sampled || sampled->slot < 0 || sampled->slot >= 128 || sampled->type != 3 || sampled->separate) return none;
    view_t view = {0};
    int has_view = 0;
    const VkTexImage *source = vk_texture_by_id(NVN_HANDLE_TEX(execution->tex[NVN_STAGE_FS][sampled->slot]), &view, &has_view);
    if (!source || source->samples != 1 || source->image == destination->image ||
        source->width != destination->width || source->height != destination->height) return none;
    TexVkFormat sampled_format;
    if (has_view && ((view.set_levels && view.base_level) || (view.set_layers && view.min_layer) || view.set_swizzle ||
                     (view.set_format && (!tex_vk_format(view.format, &sampled_format) || sampled_format.vk != source->format)))) return none;
#ifdef GAME_OOT
    if (encode && vk_dlaa_pathlog())
        rt_log("PATHLOG guest output f %llu composite s%u fmt %d -> window s%u fmt %d",
               (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), source->serial, source->format,
               destination->img->serial, destination->format);
    if (encode && state->fs) {
        const uint8_t *id = gfx_shader_input_id(state->fs);
        static const uint8_t sdr[] = {0xea, 0x16, 0x31, 0x08, 0xd2, 0xa8, 0x2e, 0x96};
        if (gfx_shader_guest_pq(state->fs)) execution->output_encoding = GFX_OUTPUT_PQ;
        else if (id && !memcmp(id, sdr, sizeof sdr)) execution->output_encoding = GFX_OUTPUT_SDR;
        if (execution->output_encoding && (rt_trace & RT_TRACE_GFX)) {
            static VkFormat previous_format;
            if (previous_format != source->format) {
                previous_format = source->format;
                const char *name = source->format == VK_FORMAT_R16G16B16A16_SFLOAT ? "RGBA16F" :
                                   source->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 ? "R11G11B10F" : "other";
                rt_log("hdr: checked game compositor %s (Vulkan format %d)", name, source->format);
            }
        }
    }
    if (encode && execution->output_encoding == GFX_OUTPUT_PQ &&
        execution->hdr_aa.phase == 4 && execution->hdr_aa.composite == source->serial &&
        execution->hdr_aa.frame == __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1) {
        if (execution->hdr_aa.active && source->format == VK_FORMAT_R16G16B16A16_SFLOAT &&
            (source->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) execution->hdr_aa.probe_source = source->image;
    }
#endif
    if (vk_dlaa_prebloom_enabled() && (source->format == VK_FORMAT_B10G11R11_UFLOAT_PACK32 ||
                                     (encode && display_composite_format(source->format))))
        execution->direct_draw_source = source->serial;
    return encode && execution->output_encoding ? source : none;
}

static int prepare_draw(VkExec *ex, int prim, int32_t count, int restart) {
    dbg_skipcount(ex, ex->render_disabled);
    if (ex->render_disabled) {   // DEBUG-DRAWDBG
        void vk_oqlog_skip(VkExec *ex);
        vk_oqlog_skip(ex);   // DEBUG-OQLOG
        {   // DEBUG-DRAWLIST: the draws a condition skipped, with their c4 and the condition's address and kind
            static int on = -1;
            static uint64_t fr0;
            if (on < 0) { const char *e = cfg_get("REKARINA_DRAWLIST"); on = e && *e; fr0 = on ? strtoull(e, NULL, 0) : 0; }
            if (on && __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) == fr0)
                rt_log("DRAWSKIP pass %u c4 %#llx df %d count %d cond %d addr %#llx mode %d zero %d prog %llu", ex->pass_serial,
                       (unsigned long long)ex->ubo[NVN_STAGE_VS][1].addr, (int)ex->pipe.ds.df, count, ex->has_render_cond,
                       (unsigned long long)ex->render_cond.addr, (int)ex->render_cond.mode, ex->dbg_zero,
                       (unsigned long long)(ex->prog ? ex->prog->serial : 0));
        }
        int vk_dump_active_dbg(void), vk_dump_drawdbg(void);
        if (vk_dump_armed && vk_dump_drawdbg() && vk_dump_active_dbg()) {
            uint64_t a = ex->render_cond.addr, x = 0, y = 0;
            VkMemSpan sp;
            if (ex->has_render_cond && vk_mem_resolve(a, 24, &sp) && !sp.alloc) memcpy(&x, (void *)(uintptr_t)a, 8), memcpy(&y, (void *)(uintptr_t)(a + 16), 8);
            rt_log("DRAWDBG SKIP frame %llu depth %#llx cond %d addr %#llx mode %d x %llu y %llu", (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), ex->depth.img ? (unsigned long long)ex->depth.img->addr : 0ull,
                   ex->has_render_cond, (unsigned long long)a, (int)ex->render_cond.mode, (unsigned long long)x, (unsigned long long)y);
        }
        return 0;
    }
#ifdef GAME_NS2
    if (stages_has(ex, NVN_STAGE_GS)) {
        vk_exec_resolve_targets(ex);
        if (vk_interface_coverage_candidate(ex)) vk_interface_coverage_reject("geometry stage");
        return prepare_stages(ex, 0, prim, restart);
    }
#endif
    cb_res(ex);
    dummies_init();
    if (ex->pass_noatt) {   // no attachment: the pass covers this draw's viewport or scissor, reopened when it needs more
        uint32_t w, h;
        vk_exec_noatt_dims(ex, &w, &h);
        if (ex->in_pass && (w > ex->pass_w || h > ex->pass_h)) vk_exec_pass_end(ex);
        if (!ex->in_pass) ex->rt_resolved = 0;
    }
    if (!ex->in_pass) ex->dirty |= VK_EXEC_DIRTY_VIEWPORT | VK_EXEC_DIRTY_SCISSOR;
    vk_exec_pass_begin(ex);
    VkRect2D sc = vk_exec_scissor(ex);
    if (!sc.extent.width || !sc.extent.height) {
        void vk_oqlog_note(const char *what, int a, int b);
        vk_oqlog_note("emptyscissor", (int)ex->pass_w, (int)ex->pass_h);   // DEBUG-OQLOG
        return 0;
    }
    if (!ex->prog) return vk_exec_oq_draw_lost(ex), warn_once(W_PROGRAM, "a draw without a bound program is skipped"), 0;
    int has_vs, has_fs, vs_wait = 0, fs_wait = 0, memo = shader_memo_on();
    const GfxShader *vs, *fs;
    uint32_t gen = memo ? vk_texture_slots_gen() : 0;   // before shader_for, whose lookups may bump it
    if (memo && SM.valid && memo_match(ex, gen, &SM.k)) {
        vs = SM.vs, fs = SM.fs, has_vs = 1, has_fs = SM.has_fs;
        if (memo == 2) {
            int hv, hf, wv, wf;
            const GfxShader *v2 = shader_for(ex, NVN_STAGE_VS, &hv, &wv, 0), *f2 = shader_for(ex, NVN_STAGE_FS, &hf, &wf, 0);
            if (v2 != vs || f2 != fs || hf != has_fs || wv || wf)
                rt_fatal("vk_exec: shader memo of program %llu differs from the lookup", (unsigned long long)ex->prog->serial);
        }
    } else {
        SM.valid = 0;
        uint64_t sec1 = sec_begin();
        vs = shader_for(ex, NVN_STAGE_VS, &has_vs, &vs_wait, 0);
        fs = shader_for(ex, NVN_STAGE_FS, &has_fs, &fs_wait, 0);
        if (sec1) {
            sec_end(SEC_SHADER_MISS, sec1);
            uint64_t h = ex->prog->serial + ((uint64_t)gen << 32) + ((uint64_t)ex->tex_serial << 17);   // the key, not ex->tex's 6 KiB: 340 misses a frame
            sim_mix_hit(SEC_SIM8_SHADER, sim_shader, sim_mix(h, &ex->pipe, sizeof ex->pipe));
        }
        if (memo && vs && !vs_wait && !fs_wait && (!has_fs || fs))
            memo_key(ex, gen, &SM.k), SM.vs = vs, SM.fs = fs, SM.has_fs = has_fs, SM.valid = 1;
    }
    if (!has_vs) return vk_exec_oq_draw_lost(ex), warn_once(W_NO_VS, "a draw with a program that has no vertex stage is skipped"), 0;
    // A draw into a target the previous frame did not render (or into none) waits for a pending shader or pipeline:
    // one-shot draws (volumes and LUTs generated once, later only sampled) would leave their target empty for the
    // session. A draw into per-frame targets is skipped until they are ready and comes back next frame.
    int oneshot = -1, skip = pipe_skip_mode();   // -1: not looked up
    if (vs_wait || fs_wait) {
        if (!(oneshot = !vk_exec_targets_recent(ex, skip >= 1))) return perf_count(PERF_CNT_SHADER_SKIP), 0;   // skipped until ready
        perf_count(PERF_CNT_ONESHOT_WAIT);
        if (vs_wait) vs = shader_for(ex, NVN_STAGE_VS, &has_vs, NULL, 1);
        if (fs_wait) fs = shader_for(ex, NVN_STAGE_FS, &has_fs, NULL, 1);
    }
    if (!vs || (has_fs && !fs)) return vk_exec_oq_draw_lost(ex), 0;
#ifdef GAME_OOT
    if (ex->scene_output_role == DRAW_ROLE_SCENE_ENCODE && gfx_shader_guest_pq(fs)) {
        scene_output_pq(ex);
    } else if (fs && ex->scene_output_role == DRAW_ROLE_SCENE_TONEMAP) {
        if (vk_hdr_renodx_scene()) fs = gfx_shader_renodx_variant(fs, GFX_RENODX_HDR);
        else if (vk_hdr_sdr_scene()) fs = gfx_shader_renodx_variant(fs, GFX_RENODX_SDR);
        else if (vk_hdr_scene_grade()) fs = gfx_shader_renodx_variant(fs, GFX_RENODX_GRADE);
        else if (vk_hdr_renodx_effects()) fs = gfx_shader_renodx_variant(fs, GFX_RENODX_EFFECTS);
    }
#endif
    if (skip_sh(vs) || skip_sh(fs)) return 0;   // DEBUG-SKIPSH
    {   // DEBUG-VALOG: REKARINA_VALOG=<vs hash>:<frame>: that frame's draws with that vertex shader, their attribute formats
        // and the raw bytes of vertex 0 for each attribute (CPU-visible vertex buffers only), at most 8 draws
        static int on = -1, nl;
        static uint64_t want, fr0;
        static const GfxShader *hs[256];
        static uint64_t hh[256];
        static int nh;
        if (on < 0) {
            const char *e = cfg_get("REKARINA_VALOG");
            char *q = NULL;
            on = e && *e;
            want = on ? strtoull(e, &q, 16) : 0;
            fr0 = q && *q == ':' ? strtoull(q + 1, NULL, 0) : 0;
        }
        if (on && nl < 8 && __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) == fr0) {
            uint64_t h = 0;
            for (int i = 0; i < nh; i++) if (hs[i] == vs) h = hh[i];
            if (!h) {
                uint64_t words = 0;
                const uint8_t *w8 = (const uint8_t *)gfx_shader_spirv(vs, &words);
                h = 0xcbf29ce484222325ull;
                for (uint64_t i = 0; w8 && i < words * 4; i++) h = (h ^ w8[i]) * 0x100000001b3ull;
                if (nh < 256) hs[nh] = vs, hh[nh] = h, nh++;
            }
            if (h == want) {
                nl++;
                for (int a = 0; a < 16; a++) {
                    const va_st *va = (const va_st *)&ex->pipe.vattr[a];
                    if (!va->fmt) continue;
                    const NVNbufferRange *vb = &ex->vbuf[va->stream & 15];
                    uint32_t stride = ex->pipe.vstream[va->stream & 15].stride;
                    char line[512];
                    int o = snprintf(line, sizeof line, "VALOG attr %d fmt %#x offset %u stream %u stride %u vb %#llx+%#llx:", a,
                                     va->fmt, va->offset, va->stream, stride, (unsigned long long)vb->address,
                                     (unsigned long long)vb->size);
                    VkMemSpan sp;
                    if (vb->address && vk_mem_resolve(vb->address, 64, &sp) && !sp.alloc && !nvn_gpu_only(vb->address)) {
                        for (int v = 0; v < 3; v++) {
                            const uint8_t *p8 = (const uint8_t *)(uintptr_t)(vb->address + (uint64_t)v * stride + va->offset);
                            o += snprintf(line + o, sizeof line - (size_t)o, " v%d", v);
                            for (int k = 0; k < 8; k++) o += snprintf(line + o, sizeof line - (size_t)o, " %02x", p8[k]);
                        }
                    } else {
                        o += snprintf(line + o, sizeof line - (size_t)o, " (not CPU-visible)");
                    }
                    rt_log("%s", line);
                }
            }
        }
    }
    if (vk_dlaa_skip_draw(ex, fs)) return 0;   // the game's FXAA off, or its scaling replaced by DLSS (vk_dlaa.c)
    uint32_t fbl = feedback_prepare(ex, fs);
    VkPipeState s = ex->pipe;
    s.vs = vs, s.fs = fs;
    if (gfx_shader_debug_mark(fs)) s.chmask.mask = 0xffffffffu, s.color.blend_en = 0;   // DEBUG-UVDEBUG: its colour lands
    {   // DEBUG-DRAWLIST: REKARINA_DRAWLIST=<frame>: every draw of that frame: pass, shader hashes (as the dump), c4, depth state
        static int on = -1;
        static uint64_t fr0;
        if (on < 0) { const char *e = cfg_get("REKARINA_DRAWLIST"); on = e && *e; fr0 = on ? strtoull(e, NULL, 0) : 0; }
        if (on && __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) == fr0) {
            static const GfxShader *hs[1024];
            static uint64_t hh[1024];
            static int nh;
            uint64_t hv[2] = {0, 0};
            const GfxShader *ss[2] = {vs, fs};
            for (int q = 0; q < 2; q++) {
                if (!ss[q]) continue;
                for (int i = 0; i < nh; i++) if (hs[i] == ss[q]) hv[q] = hh[i];
                if (!hv[q]) {
                    uint64_t words = 0;
                    const uint8_t *w8 = (const uint8_t *)gfx_shader_spirv(ss[q], &words);
                    uint64_t h = 0xcbf29ce484222325ull;
                    for (uint64_t i = 0; w8 && i < words * 4; i++) h = (h ^ w8[i]) * 0x100000001b3ull;
                    hv[q] = h;
                    if (nh < 1024) hs[nh] = ss[q], hh[nh] = h, nh++;
                }
            }
            rt_log("DRAW pass %u vs %016llx fs %016llx c4 %#llx c3 %#llx df %d dt %d dw %d prim %d count %d cond %d interface %d", ex->pass_serial,
                   (unsigned long long)hv[0], (unsigned long long)hv[1], (unsigned long long)ex->ubo[NVN_STAGE_VS][1].addr,
                   (unsigned long long)ex->ubo[NVN_STAGE_VS][0].addr, (int)s.ds.df, (int)s.ds.dt, (int)s.ds.dw, prim, count,
                   ex->has_render_cond, ex->interface_draw);
        }
    }
    s.feedback = (uint8_t)((fbl & 0xff ? 1 : 0) | (fbl & 0x100 ? 2 : 0));
    s.prim = (uint8_t)prim;
    s.depth_minus_w = (uint8_t)(nvn_depth_minus_w() != 0);
    s.y_flip = 1;
    s.prim_restart = (uint8_t)(restart != 0);
    const VkPipeInfo *info = NULL;
    ex->interface_replay_view = VK_NULL_HANDLE;
    interface_prepare_replay(ex, &s, fbl);
    if (!ex->in_pass) vk_exec_pass_begin(ex);
    int pending;
    // A draw of at most 6 vertices over half the render area or more (a full-screen triangle or quad, a tile) waits for
    // its pipeline: skipping it can leave a black tile. The waits share 50 ms per presented frame
    // (REKARINA_PIPE_SKIP=fullscreen skips them instead). A one-shot draw waits without that budget. Everything else is
    // skipped while its pipeline is built.
    static uint64_t wait_frame;
    static uint64_t wait_left_ns;
    uint32_t wait_ms = 0;
    uint64_t t0 = 0;
    if (oneshot > 0) {
        wait_ms = ONESHOT_WAIT_MS;
    } else if (skip < 2 && count <= 6 && (uint64_t)sc.extent.width * sc.extent.height * 2 >= (uint64_t)ex->pass_w * ex->pass_h) {
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (f != wait_frame) wait_frame = f, wait_left_ns = 50000000;
        wait_ms = (uint32_t)(wait_left_ns / 1000000);
        if (wait_ms) t0 = kern_now_ns();
    }
    // The previous draw's pipeline while its state repeats (PM, REKARINA_PIPE_MEMO=0 turns it off): the cache key is made
    // from the state alone, and a pipeline once found only changes from its fast link to its optimised build, read
    // from the info the cache entry keeps.
    VkPipeline p;
    const char *why = NULL;
    uint64_t w0 = 0;
    if (pipe_memo_on() && PM.valid && !memcmp(&PM.s, &s, sizeof s)) {
        info = PM.info, pending = 0;
        p = __atomic_load_n(&info->pipeline, __ATOMIC_ACQUIRE);
    } else {
        w0 = t0 ? t0 : wait_ms ? kern_now_ns() : 0;   // the clock is read only for draws that may wait
        uint64_t sec1 = sec_begin();
        p = vk_pipeline_graphics_async(&s, &info, &pending, wait_ms);
        if (sec1) sec_end(SEC_PIPE_MISS, sec1), sim_mix_hit(SEC_SIM8_PIPE, sim_pipe, sim_mix(1, &s, sizeof s));
        if (t0) {
            uint64_t spent = kern_now_ns() - t0;
            wait_left_ns = spent < wait_left_ns ? wait_left_ns - spent : 0;
        }
        why = oneshot > 0 ? "one-shot, shader" : wait_ms ? "full-screen" : NULL;
        if (pending && oneshot < 0 && !vk_exec_targets_recent(ex, skip >= 1)) {
            perf_count(PERF_CNT_ONESHOT_WAIT);
            why = "one-shot";
            if (!w0) w0 = kern_now_ns();
            p = vk_pipeline_graphics_async(&s, &info, &pending, ONESHOT_WAIT_MS);
        }
        PM.valid = !pending && p && info;
        if (PM.valid) PM.s = s, PM.info = info;
    }
    uint64_t waited = w0 ? kern_now_ns() - w0 : 0;
    static int wait_logs, wait_all = -1;   // DEBUG-ONESHOTLOG: REKARINA_ONESHOT_LOG=1: every such draw, however short its wait
    if (wait_all < 0) wait_all = cfg_env("REKARINA_ONESHOT_LOG") != NULL;
    if (why && (waited >= 2000000 || wait_all) && wait_logs < 200) {
        char tg[512];
        const uint8_t *a = gfx_shader_input_id(s.vs), *z = s.fs ? gfx_shader_input_id(s.fs) : NULL;
        unsigned long long ia = 0, iz = 0;
        for (int i = 0; i < 8; i++) ia = ia << 8 | a[i], iz = iz << 8 | (z ? z[i] : 0);
        wait_logs++;
        vk_exec_targets_describe(ex, tg, sizeof tg);
        rt_log("vk_exec: frame %llu: a draw waited %.1f ms for its pipeline (%s%s): %u vertices, vs %016llx fs %016llx, "
               "pass %ux%u, %s", (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1,
               waited / 1e6, why, pending ? ", skipped after all" : "", count, ia, iz, ex->pass_w, ex->pass_h, tg);
    }
    if (pending) return vk_exec_oq_draw_lost(ex), 0;   // a background worker is building it: the draw is skipped until it is ready
    if (!p || !info) return vk_exec_oq_draw_lost(ex), warn_once(W_PIPELINE, "a graphics pipeline could not be created: draws with it are skipped"), 0;
    const VkTexImage *output = scene_output_draw(ex, &s, info, count);
#ifdef GAME_OOT
    if (output) vk_present_guest_menu(ex, output, ex->color[0].image);
#endif
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    if (!B.st_valid || memcmp(&B.st, &s, sizeof s)) {
        uint64_t sec1 = sec_begin();
        vk_pipeline_set_dynamic(ex->cb, &s);
        sec_end(SEC_SET_DYN, sec1);
        ex->recorded++;
        B.st = s, B.st_valid = 1;
        if (ex->mark && vk_exec_frame.draws == ex->mark_draws0) ex->mark->first[0] = vs, ex->mark->first[1] = fs;
    }
    if (ex->mark && !ex->mark->count && vk_exec_frame.draws == ex->mark_draws0) {
        // The pass mark's detail (vk.h VkPassMark.tex): the first draw's fragment textures, read without the texture lock
        // (a slot whose resolution is not current is left out). Once a pass.
        VkPassMark *m = ex->mark;
        static int timed = -1;   // DEBUG-MARKDETAIL: with REKARINA_MARK_DETAIL set, what these copies cost
        if (timed < 0) timed = cfg_env("REKARINA_MARK_DETAIL") != NULL;
        uint64_t t0 = timed ? plat_now_ns() : 0;
        m->count = count ? count : -1;
        memset(m->tex, 0, sizeof m->tex);
        const GfxShaderRes *res;
        int nres = s.fs ? gfx_shader_resources(s.fs, GFX_RES_TEX, &res) : 0, k = 0;
        for (int r = 0; r < nres && k < 3; r++) {
            if (res[r].slot < 0 || res[r].slot >= 128) continue;
            uint64_t h = ex->tex[NVN_STAGE_FS][res[r].slot];
            int hit = 0;
            const VkTexImage *img = h ? vk_texture_by_id_fast((int)(h & 0xfffff), NULL, NULL, &hit) : NULL;
            if (img && hit) m->tex[k++] = vk_pass_mark_image(img);
        }
        if (timed) {
            static uint64_t ns, calls;
            ns += plat_now_ns() - t0;
            if (!(++calls & 0x3fff))
                rt_log("MARKDETAIL first-draw copies: %llu so far, %.0f ns each (two clock reads included), frame %llu",
                       (unsigned long long)calls, (double)ns / (double)calls,
                       (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));
        }
    }
    vk_dlaa_draw(ex);
    vk_maskfilter_draw(ex);   // the shadow mask filter's camera (REKARINA_MASK_FILTER, on in OoT)
    vk_dlaa_pjitter(ex, NVN_STAGE_VS);
    vk_dlaa_ubodump(ex, s.vs, NULL, s.fs, 0);
    vk_mvsurvey_draw(ex);
    vk_mvprev_draw(ex);
    record_fixed(ex);
    repeat_stats(ex, info);
    if (output && vk_interface_coverage_source(output->image))
        ex->interface_copy_source = output->image;
    bind_resources(ex, info);
    if (fbl & ex->fb_dirty) feedback_barrier(ex);
    feedback_written(ex, &s);
    bind_vertex_buffers(ex);
    if (vk_dump_armed) vk_dump_draw(ex, &s, p, info);
    gl_cur_vs = s.vs;   // DEBUG-GRASSLOG
    if (vk_hdr_track) vk_hdr_draw(ex, info);
    vk_exec_oq_draw(ex);
    {   // DEBUG-OQLOG: a draw under a condition decided by a segment without queries (rendered with REKARINA_PRED_ZERO=0)
        int vk_oqlog_on(void);
        if (ex->has_render_cond && ex->dbg_zero && vk_oqlog_on()) {
            const uint8_t *a = gfx_shader_input_id(s.vs), *z = s.fs ? gfx_shader_input_id(s.fs) : NULL;
            unsigned long long ia = 0, iz = 0;
            for (int i = 0; i < 8; i++) ia = ia << 8 | a[i], iz = iz << 8 | (z ? z[i] : 0);
            rt_log("OQ zerodraw f %llu vs %016llx fs %016llx count %d prim %d pass %ux%u depth %d/%d/%d blend %d",
                   (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), ia, iz, count, prim,
                   ex->pass_w, ex->pass_h, (int)s.ds.dt, (int)s.ds.dw, (int)s.ds.df, (int)s.color.blend_en);
        }
    }
    return 1;
}

// ---- per-object motion
// The camera pass (vk_dlaa.c mv_pass) gives everything the camera's motion, so a character that moves with the camera,
// or on its own, gets the wrong one and DLSS throws its history away there. Each solid scene draw whose position reads
// an object's own uniform blocks (bones c5, place c7, c6; mvobj_draw_ says which are left out and why) is therefore
// drawn a second time (listed during the replay, MQ, and recorded by mv_pass) with its vertex shader's last-frame
// variant (translator patch 0015: the position again from last frame's blocks, bound at slot bank + 15, and last
// frame's camera, written to a spare output) and the fragment shader of vk_mvobj_fs.h, which writes last frame's window
// position minus this frame's. The draw keeps everything else of the original: vertex and index buffers, the other
// uniform blocks, the jittered viewport and the scene depth, tested with the game's own test and its equal case
// (LEQUAL, GEQUAL) without writes, so only the visible surface writes its vector. Last frame's blocks:
// vk_mvprev_find and mo_prev_block; the camera: vk_dlaa_prev_camera. A draw whose variant or pipeline is still
// building, or whose last-frame blocks are not in the snapshot, keeps the camera's motion. REKARINA_DLAA_OBJMV=0 turns it
// off (vk_mvprev.c).
#include "vk_mvobj_fs.h"
#include "model_motion_spv.h"
#include "scene_motion_projection.h"
static const char MO_FS_RES[] = "{\"key\":\"mvobj\",\"stage\":\"ps\",\"notes\":[],\"cbufs\":[{\"binding\":1,\"slot\":3}],"
                                "\"sbufs\":[],\"textures\":[],\"images\":[]}";
#ifdef GAME_OOT
static const char MODEL_FS_RES[] = "{\"key\":\"model-motion\",\"stage\":\"ps\",\"cbufs\":[{\"binding\":1,\"slot\":3}],"
                                   "\"textures\":[{\"binding\":2,\"type\":3}],\"sbufs\":[],\"images\":[]}";
#endif
static struct {
    const GfxShader *fs[16];     // per input Location
#ifdef GAME_OOT
    const GfxShader *guarded_fs[16];
#endif
    int failed;
    uint64_t cand, still, draws, nodata, building, ran, dropped;
    uint64_t ns_check, ns_rec, n_rec, ns_emit, ns_still;
    uint64_t ns_part[3], n_part[3], misses, cyc[3], ncyc, big;   // checks: filter, data and stillness (memo misses), variant and pipeline   // time spent: checks and lookups of every candidate; recording the draws again   // cand: draws whose vertex shader reads c5 or c7
    uint64_t why[5];             // nodata by cause: camera, bones, place, bones copy, place copy
    uint64_t no_out, storage, no_fs;   // variant without the extra output, with storage resources; no fs
    uint64_t other_cam;          // listed draws left out: their camera block is not the scene camera
    uint64_t lk[14][5];          // last-frame lookups per binding index: none, self, side, nearer, refused (another object)
    uint64_t e_draws, e_pipe, e_dyn, e_push, e_pushw, e_sets, e_vb, e_setw, e_setreuse;   // emitted commands (summary)
} MO;

// The motion draws of the frame, kept as a list while the scene replays and recorded into the replay's own command
// buffer by vk_mvobj_execute, through the recorder thread (vk_rec.h) like every other draw: recording them straight
// into a secondary command buffer made the driver's work for each one on the replay thread. What a draw shares with
// the one before it (pipeline, dynamic state, viewport, buffers, descriptors) is recorded once; a push descriptor set
// is pushed again only where a binding changed (pushes update the set in place while its layout stays bound).
typedef struct {
    VkPipeline pipe;
    const VkPipeInfo *info;
    uint32_t st, desc, vb;       // indices into MQ.st, MQ.desc (the draw's packed descriptors) and MQ.vb
    VkDescriptorSet set[2];      // allocated sets; VK_NULL_HANDLE for a pushed one
    VkViewport v;
    VkRect2D sc;
    float bias[3];
    VkBuffer ib;
    VkDeviceSize ib_off;
    VkIndexType it;
    uint32_t count, instances, base_instance;
    int32_t base_vertex;
    uint64_t projection_token;
    uint64_t cam;                // the draw's camera block, this frame's
    int skin;                    // vk_skin_motion.inc: 1 paired draw, 2 checked against last frame's depth, -1 refused
} MoDraw;
typedef struct { VkBuffer b[17]; VkDeviceSize o[17], sz[17], str[17]; } MoVb;
static struct {
    uint64_t frame, fence;       // the frame and submission the list belongs to
    VkFormat depth_fmt;
    MoDraw *d;
    VkPipeState *st;
    VkDescSlot *desc;
    MoVb *vb;
    int n, cap, nst, cst, nvb, cvb;
    uint32_t ndesc, cdesc;
    NVNbufferRange vin[17];      // the vertex ranges MQ.vb[nvb - 1] was resolved from, under vk_mem_gen vgen
    uint64_t vgen;
    float sup_in[3];             // the support buffer last filled for the list and its inputs
    VkDescriptorBufferInfo sup;
    float par_in[8];             // the fragment shader's parameters last written and their inputs
    VkDescriptorBufferInfo par;
    VkDescriptorSetLayout aset_layout[2];   // the last allocated set per set number, with its descriptors
    VkDescriptorSet aset[2];
    uint32_t aset_n[2];
    VkDescSlot aset_d[2][MAX_DESC];
} MQ;

static void *mo_grow(void *p, int *cap, int need, size_t elem) {
    if (need <= *cap) return p;
    int c = *cap ? *cap : 256;
    while (c < need) c *= 2;
    if (!(p = realloc(p, elem * (size_t)c))) rt_fatal("vk_exec: out of memory");
    *cap = c;
    return p;
}

// The position bank masks of the draws considered (gfx_shader_mvprev_banks), with counts, for the summary.
static struct { uint32_t mask; uint64_t n; int ubo, ssbo, tex; } MM[32];   // ubo, ssbo, tex: the first vertex shader's
static void mo_mask_note(uint32_t banks, const GfxShader *vs) {
    for (int i = 0; i < 32; i++)
        if (MM[i].mask == banks || !MM[i].n) {
            if (!MM[i].n) {
                const GfxShaderRes *r;
                MM[i].ubo = gfx_shader_resources(vs, GFX_RES_UBO, &r), MM[i].ssbo = gfx_shader_resources(vs, GFX_RES_SSBO, &r);
                MM[i].tex = gfx_shader_resources(vs, GFX_RES_TEX, &r);
            }
            MM[i].mask = banks, MM[i].n++;
            return;
        }
}

static void mo_drop(void) {
    if (MQ.n) MO.dropped++;
    MQ.n = MQ.nst = MQ.nvb = 0, MQ.ndesc = 0, MQ.frame = 0;
}

#ifndef GAME_OOT
static uint64_t mo_gen = 1;   // the list's snapshot (mo_prev_cached), bumped by mo_begin
#endif
static struct { uint32_t mask, chg, n; } MF[16];   // this list's draws per position banks and changed banks
static char fm_last[400];                          // the last executed list's, formatted (vk_mvobj_frame_masks)

#ifdef GAME_OOT
static void skin_begin(uint64_t frame);
#endif
// Starts this frame's list (the old one, never executed, is dropped).
static void mo_begin(VkExec *ex, uint64_t frame) {
    mo_drop();
#ifdef GAME_OOT
    skin_begin(frame);
#else
    mo_gen++;
#endif
    memset(MF, 0, sizeof MF);
    MQ.frame = frame, MQ.fence = ex->fence, MQ.depth_fmt = ex->depth.format;
    memset(MQ.vin, 0, sizeof MQ.vin);
    MQ.vgen = 0, MQ.sup.buffer = MQ.par.buffer = VK_NULL_HANDLE;
    memset(MQ.aset, 0, sizeof MQ.aset);
}

static const GfxShader *mo_fs(int loc) {
    if (loc < 0 || loc >= 16) return NULL;
    if (!MO.fs[loc]) {
        uint32_t w[sizeof mvobj_fs_spv / 4];
        memcpy(w, mvobj_fs_spv, sizeof w);
        w[MVOBJ_FS_LOCATION_WORD] = (uint32_t)loc;
        MO.fs[loc] = gfx_shader_own(w, sizeof w / 4, MO_FS_RES);
        if (!MO.fs[loc]) MO.failed = 1, rt_log("vk_exec: the per-object motion fragment shader is malformed: off");
    }
    return MO.fs[loc];
}

#ifdef GAME_OOT
static const GfxShader *mo_guarded_fs(int loc) {
    if (loc < 0 || loc >= 16) return NULL;
    if (MO.guarded_fs[loc]) return MO.guarded_fs[loc];
    uint32_t words[sizeof model_motion_spv / sizeof(uint32_t)];
    memcpy(words, model_motion_spv, sizeof words);
    for (size_t i = 5; i < sizeof words / sizeof *words; i += words[i] >> 16) {
        // OpDecorate Location 15 uniquely identifies the previous clip position in model_motion.frag.
        if (words[i] == ((4u << 16) | 71u) && words[i + 2] == 30 && words[i + 3] == 15) {
            words[i + 3] = (uint32_t)loc;
            MO.guarded_fs[loc] = gfx_shader_own(words, sizeof words / sizeof *words, MODEL_FS_RES);
            break;
        }
    }
    return MO.guarded_fs[loc];
}
#endif

// The write of binding b from its packed descriptors.
static VkWriteDescriptorSet mo_write(const VkPipeBinding *b, const VkDescSlot *d) {
    VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = b->binding,
                              .descriptorCount = b->count ? b->count : 1, .descriptorType = b->type};
    if (b->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || b->type == VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE ||
        b->type == VK_DESCRIPTOR_TYPE_SAMPLER)
        w.pImageInfo = &d->image;
    else
        w.pBufferInfo = &d->buffer;   // VkDescSlot is 24 bytes, as VkDescriptorBufferInfo: arrays line up
    return w;
}

// The descriptors of a motion draw, packed into MQ.desc at q->desc without the binding caches of the command buffer
// being replayed (B, DS, RS), which then stay true for its next draw: uniform buffers (through ubo_info, so with
// mo_ovr), the support buffer and sampled textures. An allocated set is written here (or the last one reused when it
// would hold the same); a pushed set is pushed by mo_emit. 0 for a pipeline with storage buffers, images or texel
// buffers (not drawn again).
static int mo_desc(VkExec *ex, const VkPipeInfo *info, MoDraw *q, const VkDescriptorImageInfo *pose_depth) {
    if (info->n_bindings > MAX_BINDINGS) return 0;
    uint32_t total = 0;
    for (uint32_t i = 0; i < info->n_bindings; i++) total += info->bindings[i].count ? info->bindings[i].count : 1;
    if (total > MAX_DESC) return 0;
    int cdesc = (int)MQ.cdesc;
    MQ.desc = mo_grow(MQ.desc, &cdesc, (int)(MQ.ndesc + total), sizeof *MQ.desc);
    MQ.cdesc = (uint32_t)cdesc;
    VkDescSlot *d = MQ.desc + MQ.ndesc;
    memset(d, 0, total * sizeof *d);   // unused bytes zero: equal descriptors compare equal
    for (uint32_t i = 0, nd = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *b = &info->bindings[i];
        const GfxShaderRes *r = b->res;
        uint32_t cnt = b->count ? b->count : 1;
        VkDescSlot *o = &d[nd];
        int st = nvn_stage_of(b->stage);
        if (b->kind == VK_PIPE_SUPPORT) {
            float in[3];
            support_in(ex, in);
            if (!MQ.sup.buffer || memcmp(in, MQ.sup_in, sizeof in)) {
                VkBuffer sb;
                VkDeviceSize so;
                support_fill(scratch(ex, SUPPORT_BYTES, &sb, &so), in[0], in[1], in[2]);
                MQ.sup = (VkDescriptorBufferInfo){sb, so, SUPPORT_BYTES};
                memcpy(MQ.sup_in, in, sizeof in);
            }
            o->buffer = MQ.sup;
        } else if (b->kind == GFX_RES_UBO) {
            for (uint32_t k = 0; k < cnt; k++) o[k].buffer = ubo_info(ex, st, r->slot + (int)k);
        } else if (b->kind == GFX_RES_SSBO && !(r->flags & 1)) {   // read only: instance data of foliage and such
            for (uint32_t k = 0; k < cnt; k++) o[k].buffer = ssbo_info(ex, st, r, 0);
        } else if (pose_depth && st == NVN_STAGE_FS && b->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER) {
            if (cnt != 1) return 0;
            o->image = *pose_depth;
        } else if (b->kind == GFX_RES_TEX && b->type != VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER) {
            vk_texture_lock();
            for (uint32_t k = 0; k < cnt; k++) {
                int slot = r->slot + (int)k;
                tex_info(ex, r, slot >= 0 && slot < 128 ? ex->tex[st][slot] : 0, 0, r->separate, &o[k].image);
            }
            vk_texture_unlock();
        } else {
            return 0;
        }
        nd += cnt;
    }
    uint32_t n_sets = info->n_sets ? info->n_sets : 1;
    q->set[0] = q->set[1] = VK_NULL_HANDLE;
    for (uint32_t s = 0, i0 = 0, k0 = 0; s < n_sets && s < 2; s++) {
        uint32_t i1 = i0, k1 = k0;
        while (i1 < info->n_bindings && (n_sets == 1 || info->bindings[i1].set == s))
            k1 += info->bindings[i1].count ? info->bindings[i1].count : 1, i1++;
        if (i1 > i0 && !(s ? info->push1 : info->push)) {
            VkDescriptorSetLayout sl = s ? info->set_layout1 : info->set_layout;
            if (MQ.aset[s] && MQ.aset_layout[s] == sl && MQ.aset_n[s] == k1 - k0 &&
                !memcmp(MQ.aset_d[s], d + k0, (k1 - k0) * sizeof *d)) {
                q->set[s] = MQ.aset[s];
                MO.e_setreuse++;
            } else {
                static _Thread_local VkWriteDescriptorSet w[MAX_BINDINGS];
                VkDescriptorSet set = alloc_set(ex, sl);
                for (uint32_t i = i0, k = k0; i < i1; k += info->bindings[i].count ? info->bindings[i].count : 1, i++)
                    w[i - i0] = mo_write(&info->bindings[i], d + k), w[i - i0].dstSet = set;
                vkUpdateDescriptorSets(vk_ctx()->device, i1 - i0, w, 0, NULL);
                q->set[s] = MQ.aset[s] = set, MQ.aset_layout[s] = sl, MQ.aset_n[s] = k1 - k0;
                memcpy(MQ.aset_d[s], d + k0, (k1 - k0) * sizeof *d);
                MO.e_setw++;
            }
        }
        i0 = i1, k0 = k1;
    }
    q->desc = MQ.ndesc;
    MQ.ndesc += total;
    return 1;
}

// The draw's vertex buffers as bind_vertex_buffers resolves them, into MQ.vb; the previous entry again while the
// ranges and vk_mem_gen are the same (most motion draws in a row are one model's parts).
static uint32_t mo_vertex_buffers(VkExec *ex) {
    uint64_t gen = vk_mem_gen();
    if (MQ.nvb && MQ.vgen == gen && !memcmp(MQ.vin, ex->vbuf, sizeof MQ.vin)) {
        const MoVb *l = &MQ.vb[MQ.nvb - 1];
        int same = 1;
        if (!vk_pipeline_vertex_input_dynamic())
            for (int b = 0; b < 16 && same; b++) same = l->str[b] == ex->pipe.vstream[b].stride;
        if (same) return (uint32_t)MQ.nvb - 1;
    }
    MQ.vb = mo_grow(MQ.vb, &MQ.cvb, MQ.nvb + 1, sizeof *MQ.vb);
    MoVb *v = &MQ.vb[MQ.nvb];
    for (int b = 0; b < 17; b++) {
        v->b[b] = Z.zbuf, v->o[b] = 0, v->sz[b] = ZERO_BYTES, v->str[b] = b < 16 ? ex->pipe.vstream[b].stride : 0;
        if (b == VK_PIPE_ZERO_BINDING) continue;
        const NVNbufferRange *r = &ex->vbuf[b];
        VkMemSpan sp;
        if (!r->address || !vk_mem_resolve(r->address, 1, &sp)) continue;
        VK_MEM_USE(sp);
        v->b[b] = sp.buffer, v->o[b] = sp.offset, v->sz[b] = r->size && r->size < sp.avail ? r->size : sp.avail;
    }
    memcpy(MQ.vin, ex->vbuf, sizeof MQ.vin);
    MQ.vgen = gen;
    return (uint32_t)MQ.nvb++;
}

static _Thread_local VkBuffer ibs_buf;   // (defined with bind_index below)

// Records the list into cb (the replay's command buffer, inside the motion pass's rendering), skipping what the draw
// before already set.
static void mo_emit(VkCommandBuffer cb) {
    int dyn_vi = vk_pipeline_vertex_input_dynamic();
    VkPipeline pipe = VK_NULL_HANDLE;
    uint32_t st = UINT32_MAX, vb = UINT32_MAX;
    VkPipelineLayout lay[2] = {0};
    const VkPipeInfo *pinfo[2] = {0};
    int pdesc[2] = {-1, -1};      // the pushed set's descriptors (MQ.desc index) under lay/pinfo, -1 none
    VkDescriptorSet bound[2] = {0};
    const MoDraw *p = NULL;
    for (int n = 0; n < MQ.n; p = &MQ.d[n++]) {
        const MoDraw *q = &MQ.d[n];
        const VkPipeInfo *info = q->info;
        if (q->pipe != pipe) vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, q->pipe), pipe = q->pipe, MO.e_pipe++;
        if (q->st != st) vk_pipeline_set_dynamic(cb, &MQ.st[q->st]), st = q->st, MO.e_dyn++;
        if (!p || memcmp(&q->v, &p->v, sizeof q->v)) vkCmdSetViewportWithCount(cb, 1, &q->v);
        if (!p || memcmp(&q->sc, &p->sc, sizeof q->sc)) vkCmdSetScissorWithCount(cb, 1, &q->sc);
        if (!p || memcmp(q->bias, p->bias, sizeof q->bias)) vkCmdSetDepthBias(cb, q->bias[0], q->bias[1], q->bias[2]);
        uint32_t n_sets = info->n_sets ? info->n_sets : 1;
        const VkDescSlot *d = MQ.desc + q->desc;
        for (uint32_t s = 0, i0 = 0, k0 = 0; s < n_sets && s < 2; s++) {
            uint32_t i1 = i0, k1 = k0;
            while (i1 < info->n_bindings && (n_sets == 1 || info->bindings[i1].set == s))
                k1 += info->bindings[i1].count ? info->bindings[i1].count : 1, i1++;
            if (i1 > i0) {
                if (s ? info->push1 : info->push) {
                    // the bindings that differ from the last push of this layout and pipeline (all after another)
                    int same = lay[s] == info->layout && pinfo[s] == info && pdesc[s] >= 0;
                    const VkDescSlot *pd = same ? MQ.desc + pdesc[s] : NULL;
                    VkWriteDescriptorSet w[MAX_BINDINGS];
                    uint32_t nw = 0;
                    for (uint32_t i = i0, k = k0; i < i1; i++) {
                        uint32_t cnt = info->bindings[i].count ? info->bindings[i].count : 1;
                        if (!same || memcmp(d + k, pd + k, cnt * sizeof *d)) w[nw++] = mo_write(&info->bindings[i], d + k);
                        k += cnt;
                    }
                    if (nw) rec_push(vk_pipeline_push_fn(), cb, VK_PIPELINE_BIND_POINT_GRAPHICS, info->layout, s, nw, w);
                    MO.e_push += nw != 0, MO.e_pushw += nw;
                    lay[s] = info->layout, pinfo[s] = info, pdesc[s] = (int)q->desc, bound[s] = VK_NULL_HANDLE;
                } else if (q->set[s] != bound[s] || lay[s] != info->layout) {
                    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, info->layout, s, 1, &q->set[s], 0, NULL);
                    bound[s] = q->set[s], lay[s] = info->layout, pinfo[s] = info, pdesc[s] = -1, MO.e_sets++;
                }
            }
            i0 = i1, k0 = k1;
        }
        if (q->vb != vb) {
            const MoVb *v = &MQ.vb[q->vb];
            vkCmdBindVertexBuffers2(cb, 0, 17, v->b, v->o, v->sz, dyn_vi ? NULL : v->str);
            vb = q->vb, MO.e_vb++;
        }
        if (!p || q->ib != p->ib || q->ib_off != p->ib_off || q->it != p->it) ibs_buf = VK_NULL_HANDLE, vkCmdBindIndexBuffer(cb, q->ib, q->ib_off, q->it);
        vkCmdDrawIndexed(cb, q->count, q->instances, 0, q->base_vertex, q->base_instance);
        MO.e_draws++;
    }
}

#ifdef GAME_OOT
#define POSE_BYTES (128u * 1024u)
#include "vk_skin_motion.inc"

static PlatMutex mo_lock = PLAT_MUTEX_INIT;

static int mvobj_pick(VkExec *ex, const p_draw_idx *d, const GfxShader **vs_out, uint32_t *banks_out,
                      SkinPick *skin_out, uint64_t t0) {
    const VkExecTarget *z = &ex->depth;
    if (!z->image || z->width < DLAA_SCENE_MIN_W || z->format != VK_FORMAT_D32_SFLOAT || B.st.samples > 1 ||
        B.st.raster_discard)
        return 0;
    if (!B.st.ds.dt) return 0;
    if (!B.st.ds.dw && !(B.st.ds.df == NVN_DEPTH_FUNC_EQUAL && ex->color[0].image &&
                         (vk_pipeline_target_wmask(&B.st, 0) & 7)))
        return 0;
    VkViewport v = base_viewport(ex);
    if (ex->ubo[0][0].size != SCENE_CAMERA_BYTES || d->instances > 1 || d->base_instance || B.st.color.blend_en ||
        B.st.ds.st || ex->pass_layers != 1 || z->level || z->base_layer || z->layers != 1 ||
        v.x != 0.0f || v.y != (float)z->height || v.width != (float)z->width || v.height != -(float)z->height)
        return 0;
    if (!skin_pick(ex, ex->work_scope && !ex->motion_reverse_only ? &ex->work_scope->model : NULL, skin_out))
        return 0;
    const GfxShaderRes *ur;
    int nu = gfx_shader_resources(B.st.vs, GFX_RES_UBO, &ur), reads = 0;
    for (int i = 0; i < nu; i++) reads |= ur[i].slot >= 3 && ur[i].slot <= 16 && ur[i].slot != 4;
    if (!reads) return skin_pick_left(skin_out, SKIN_BANKS_OTHER);
    __atomic_add_fetch(&MO.cand, 1, __ATOMIC_RELAXED);
    if (t0) {
        __atomic_add_fetch(&MO.ns_part[0], kern_now_ns() - t0, __ATOMIC_RELAXED);
        __atomic_add_fetch(&MO.n_part[0], 1, __ATOMIC_RELAXED);
    }
    if (!__atomic_load_n(&LIM.ubo_max, __ATOMIC_ACQUIRE)) limits_init();
    static _Thread_local struct { const GfxShader *of, *vs; uint32_t banks; } vmemo[1024];
    uint32_t vi = (uint32_t)((uintptr_t)B.st.vs * 0x9E3779B97F4A7C15ull >> 54);
    if (vmemo[vi].of != B.st.vs) {
        int has, pending = 0;
        sf_mvprev = 1;
        const GfxShader *vs_v = shader_for(ex, NVN_STAGE_VS, &has, &pending, 0);
        sf_mvprev = 0;
        if (!vs_v || pending) {
            __atomic_add_fetch(&MO.building, 1, __ATOMIC_RELAXED);
            return skin_pick_left(skin_out, SKIN_NOT_READY);
        }
        GfxShaderIo io[32];
        uint32_t extra = gfx_shader_outputs(vs_v, io) & ~gfx_shader_outputs(B.st.vs, io);
        uint32_t banks = extra ? gfx_shader_mvprev_banks(vs_v) & 0x1fff8u : 0;
        vmemo[vi].of = B.st.vs, vmemo[vi].vs = vs_v, vmemo[vi].banks = banks;
    }
    const GfxShader *vs = vmemo[vi].vs;
    uint32_t banks = vmemo[vi].banks;
    if (!banks || !(banks & ~(1u << 4))) {
        __atomic_add_fetch(&MO.no_out, 1, __ATOMIC_RELAXED);
        return skin_pick_left(skin_out, SKIN_BANKS_OTHER);
    }
    if (!(banks & (1u << 4))) {
        __atomic_add_fetch(&MO.no_out, 1, __ATOMIC_RELAXED);
        return skin_pick_left(skin_out, SKIN_BANKS_OTHER);
    }
    if (!(banks & (1u << 5 | 1u << 6 | 1u << 7))) {
        __atomic_add_fetch(&MO.no_out, 1, __ATOMIC_RELAXED);
        return skin_pick_left(skin_out, SKIN_BANKS_OTHER);
    }
    *vs_out = vs, *banks_out = banks;
    return 1;
}

static void mvobj_draw_(VkExec *ex, const p_draw_idx *d, VkBuffer ib, VkDeviceSize ib_off, uint64_t t0,
                        const GfxShader *vs, uint32_t banks, const SkinPick *skin) {
    const VkExecTarget *z = &ex->depth;
    uint64_t tp = t0 ? kern_now_ns() : 0;
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (MQ.frame != frame || MQ.fence != ex->fence || MQ.depth_fmt != z->format) mo_begin(ex, frame);
    mo_mask_note(banks, B.st.vs);
    SkinDraw skin_draw;
    SkinEntry skin_entry;
    uint64_t skin_listed = 0;
    VkDescriptorImageInfo pose_depth = {0};
    int pose_visibility_guard = 0;
    float history[4] = {0};
    if (!skin_try(ex, d, frame, vs, banks, skin, &skin_draw, &skin_entry, &skin_listed, &pose_visibility_guard)) return;
    if (pose_visibility_guard && !vk_dlaa_pose_depth(frame, z->width, z->height, &pose_depth, history))
        pose_visibility_guard = 0;
    VkViewport v = draw_viewport(ex);
    float jit[2] = {v.x, v.y - (float)z->height};
    const uint64_t token = ex->scene_projection.sample.token;
    if (token) memcpy(jit, ex->scene_projection.sample.jitter, sizeof jit);
    if (t0) MO.ns_part[1] += kern_now_ns() - tp, MO.n_part[1]++, tp = kern_now_ns();
    GfxShaderIo io[32];
    uint32_t extra = gfx_shader_outputs(vs, io) & ~gfx_shader_outputs(B.st.vs, io);
    const GfxShaderRes *r;
    if (gfx_shader_resources(vs, GFX_RES_IMG, &r)) {
        MO.storage++;
        skin_left(SKIN_STORAGE);
        return;
    }
    int loc = 31 - __builtin_clz(extra);
    const GfxShader *fs = pose_visibility_guard ? mo_guarded_fs(loc) : mo_fs(loc);
    if (!fs) {
        MO.no_fs++;
        skin_left(SKIN_NOT_READY);
        return;
    }
    VkPipeState ms = B.st;
    ms.vs = vs, ms.fs = fs, ms.feedback = 0;
    memset(ms.color_fmt, 0, sizeof ms.color_fmt);
    ms.color_fmt[0] = NVN_FORMAT_RG16F;
    ms.color.blend_en = 0, ms.color.logic_op = 0;
    ms.chmask.mask = 0x3;
    ms.ms.a2c = 0;
    ms.ds.dt = 1, ms.ds.dw = 0, ms.ds.df = NVN_DEPTH_FUNC_EQUAL, ms.ds.st = 0;
    static struct { VkPipeState st; VkPipeline p; const VkPipeInfo *info; int used; } pmemo[64];
    static int plast = -1;
    uint32_t pi;
    if (plast >= 0 && pmemo[plast].used && !memcmp(&pmemo[plast].st, &ms, sizeof ms)) {
        pi = (uint32_t)plast;
    } else {
        uint64_t ph = 0xcbf29ce484222325ull;
        for (size_t i = 0; i < sizeof ms; i += 8) {
            uint64_t w8;
            memcpy(&w8, (const uint8_t *)&ms + i, sizeof ms - i < 8 ? sizeof ms - i : 8);
            ph = (ph ^ w8) * 0x100000001b3ull;
        }
        pi = (uint32_t)(ph >> 58);
    }
    const VkPipeInfo *info = NULL;
    VkPipeline pipe;
    int pending = 0;
    if (pmemo[pi].used && !memcmp(&pmemo[pi].st, &ms, sizeof ms)) {
        info = pmemo[pi].info;
        pipe = __atomic_load_n(&info->pipeline, __ATOMIC_ACQUIRE);
        if (!pipe) {
            skin_left(SKIN_NOT_READY);
            return;
        }
    } else {
        pipe = vk_pipeline_graphics_async(&ms, &info, &pending, 0);
        if (pending || !pipe || !info) {
            MO.building += pending != 0;
            skin_left(SKIN_NOT_READY);
            return;
        }
        pmemo[pi].st = ms, pmemo[pi].p = pipe, pmemo[pi].info = info, pmemo[pi].used = 1;
    }
    plast = (int)pi;
    uint64_t t1 = kern_now_ns();
    if (t0) MO.ns_part[2] += t1 - tp, MO.n_part[2]++;
    if (t0) MO.ns_check -= t1 - t0;
    MQ.d = mo_grow(MQ.d, &MQ.cap, MQ.n + 1, sizeof *MQ.d);
    MoDraw *q = &MQ.d[MQ.n];
    float par_in[8] = {(float)z->width, (float)z->height, jit[0], jit[1], history[0], history[1], history[2], history[3]};
    if (!MQ.par.buffer || memcmp(par_in, MQ.par_in, sizeof par_in)) {
        VkBuffer pbuf;
        VkDeviceSize poff;
        memcpy(scratch(ex, sizeof par_in, &pbuf, &poff), par_in, sizeof par_in);
        MQ.par = (VkDescriptorBufferInfo){pbuf, poff, sizeof par_in};
        memcpy(MQ.par_in, par_in, sizeof par_in);
    }
    memset(mo_ovr, 0, sizeof mo_ovr);
    for (int i = 0; i < SKIN_BANKS; i++)
        if (skin_draw.block[i] >= 0) mo_ovr[NVN_STAGE_VS][18 + i] = SK.f[SK.cur].b[skin_draw.block[i]].was;
    mo_ovr[NVN_STAGE_FS][3] = MQ.par;
    mo_ovr_on = 1;
    int bound = mo_desc(ex, info, q, pose_visibility_guard ? &pose_depth : NULL);
    mo_ovr_on = 0;
    if (!bound) {
        MO.storage++;
        skin_left(SKIN_STORAGE);
        return;
    }
    q->pipe = pipe, q->info = info;
    if (!MQ.nst || memcmp(&MQ.st[MQ.nst - 1], &ms, sizeof ms)) {
        MQ.st = mo_grow(MQ.st, &MQ.cst, MQ.nst + 1, sizeof *MQ.st);
        MQ.st[MQ.nst++] = ms;
    }
    q->st = (uint32_t)MQ.nst - 1;
    q->v = draw_viewport(ex), q->sc = vk_exec_scissor(ex);
    q->bias[0] = ex->poly_offset.units, q->bias[1] = ex->poly_offset.clamp, q->bias[2] = ex->poly_offset.factor;
    q->vb = mo_vertex_buffers(ex);
    q->ib = ib, q->ib_off = ib_off;
    q->it = d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
    q->count = (uint32_t)d->count, q->instances = d->instances > 0 ? (uint32_t)d->instances : 1;
    q->base_vertex = d->base_vertex, q->base_instance = (uint32_t)d->base_instance;
    q->cam = ex->ubo[NVN_STAGE_VS][0].addr;
    q->projection_token = token;
    q->skin = 1 + pose_visibility_guard;
    skin_commit(&skin_draw, &skin_entry, skin_listed, MQ.n);
    MQ.n++;
    MO.draws++;
    if (t0) MO.ns_rec += kern_now_ns() - t1, MO.n_rec++;
}

static void mvobj_draw(VkExec *ex, const p_draw_idx *d, VkBuffer ib, VkDeviceSize ib_off) {
    if (!ex->motion_models || MO.failed || !B.st_valid) return;
    if (gfx_shader_pool_mask(B.st.vs)) return;
    static int timed = -1;
    if (timed < 0) timed = rt_knob("MVOBJ_TIMES", "0");
    uint64_t t0 = timed ? kern_now_ns() : 0;
    const GfxShader *vs;
    uint32_t banks;
    SkinPick skin = {0};
    TrackedModelSample prior = ex->tracked_model;
    int listed = mvobj_pick(ex, d, &vs, &banks, &skin, t0);
    if (listed) {
        plat_mutex_lock(&mo_lock);
        mvobj_draw_(ex, d, ib, ib_off, t0, vs, banks, &skin);
        if (t0) __atomic_add_fetch(&MO.ns_check, kern_now_ns() - t0, __ATOMIC_RELAXED);
        plat_mutex_unlock(&mo_lock);
    } else if (t0) {
        __atomic_add_fetch(&MO.ns_check, kern_now_ns() - t0, __ATOMIC_RELAXED);
    }
    ex->tracked_model = prior;
}

int vk_fg_pose_prepare(VkExec *ex, uint64_t cam, VkImage depth, uint32_t w, uint32_t h, uint64_t token) {
    if (!ex->motion_models) return 0;
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    skin_resolve(ex, frame, cam, depth, w, h, token);
    if (frame % 120 == 0) {
        char status[1200];
        skin_status(status, sizeof status);
        rt_log("model motion: %s", status);
    }
    return MQ.n != 0;
}
#else
int vk_fg_pose_prepare(VkExec *ex, uint64_t cam, VkImage depth, uint32_t w, uint32_t h, uint64_t token) {
    (void)ex; (void)cam; (void)depth; (void)w; (void)h; (void)token;
    return 0;
}

static void mvobj_draw_(VkExec *ex, const p_draw_idx *d, VkBuffer ib, VkDeviceSize ib_off, uint64_t t0);
static void mvobj_draw(VkExec *ex, const p_draw_idx *d, VkBuffer ib, VkDeviceSize ib_off) {
    if (MO.failed || !vk_mvprev_on() || !B.st_valid) return;
    if (gfx_shader_pool_mask(B.st.vs)) return;   // mo_emit binds sets 0 and 1 only, not the texture pool's set
    static int timed = -1;
    if (timed < 0) timed = rt_knob("MVOBJ_TIMES", "0");   // REKARINA_MVOBJ_TIMES=1: the summary's times (the clock reads cost)
    if (!timed) return mvobj_draw_(ex, d, ib, ib_off, 0);
    uint64_t t0 = kern_now_ns();
    mvobj_draw_(ex, d, ib, ib_off, t0);
    MO.ns_check += kern_now_ns() - t0;
}

// REKARINA_MVOBJ_BANKS=1 (test): for scene draws, the words of each other uniform bank (c3, c6, c8-c16) that changed
// since the same shader's draw last frame, logged at frame 3000: where the time that moves wind and water lives
static void mvobj_banks(VkExec *ex) {
    static int on = -1;
    if (on < 0) on = rt_knob("MVOBJ_BANKS", "0");
    if (!on || !B.st_valid) return;
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (frame < 2999 || frame > 3000) return;
    static struct { const GfxShader *vs; int bank; uint64_t frame; float w[64]; int n; } tab[4096];
    static int logged;
    const GfxShaderRes *ur;
    int nu = gfx_shader_resources(B.st.vs, GFX_RES_UBO, &ur);
    for (int i = 0; i < nu; i++) {
        int bank = ur[i].slot;
        if (bank < 3 || bank == 4 || bank == 5 || bank == 7 || bank > 16) continue;
        const p_bind_buf *u = &ex->ubo[NVN_STAGE_VS][bank - 3];
        uint32_t n = u->size < 256 ? (uint32_t)u->size / 4 : 64;
        const float *now = n ? (const float *)vk_usnap_bytes(u->addr, n * 4) : NULL;
        if (!now) continue;
        uint32_t k = (uint32_t)(((uintptr_t)B.st.vs ^ (uint64_t)bank * 0x9E3779B97F4A7C15ull) * 0xff51afd7ed558ccdull >> 52);
        if (frame == 2999) {
            tab[k].vs = B.st.vs, tab[k].bank = bank, tab[k].frame = frame, tab[k].n = (int)n;
            memcpy(tab[k].w, now, n * 4);
        } else if (tab[k].vs == B.st.vs && tab[k].bank == bank && tab[k].frame == 2999 && logged < 80) {
            char line[700];
            int o = 0, nd = 0;
            for (int j = 0; j < tab[k].n && j < (int)n && o < 600; j++)
                if (memcmp(&tab[k].w[j], &now[j], 4)) nd++, o += snprintf(line + o, sizeof line - o, " [%d] %g>%g", j, tab[k].w[j], now[j]);
            if (nd) logged++, rt_log("mvbanks: vs %p c%d size %#llx: %d words%s", (const void *)B.st.vs, bank,
                                     (unsigned long long)u->size, nd, line);
            tab[k].frame = 0;
        }
    }
}

// 1: old is cur a little changed or the same (every float within 10 %, at least 1.0 and at most 4.0); *changed: not
// the same. The cap: TotK's world positions are in the thousands, and 10 % of them let another object's block a few
// units away pass as this one moved; nothing moves 4 units in a frame.
static int mo_similar(const uint8_t *cur, const uint8_t *old, uint64_t n, int *changed) {
    *changed = 0;
    if (!memcmp(cur, old, n)) return 1;
    // word by word as integers; float arithmetic only on two normal floats (the blocks hold integers too, and
    // denormal arithmetic took this loop 1.9 million cycles a block)
    const uint32_t *x = (const uint32_t *)cur, *y = (const uint32_t *)old;
    for (uint64_t w = 0; w < n / 4; w++) {
        if (x[w] == y[w]) continue;
        uint32_t ex = x[w] >> 23 & 0xff, ey = y[w] >> 23 & 0xff;
        if (!ex || ex == 0xff || !ey || ey == 0xff) continue;   // not both normal floats: an integer or a tiny value
        float fx, fy;
        memcpy(&fx, &x[w], 4), memcpy(&fy, &y[w], 4);
        float dd = fabsf(fx - fy), tol = 0.1f * fabsf(fy);
        if (dd > (tol < 1.0f ? 1.0f : tol > 4.0f ? 4.0f : tol)) return 0;
    }
    *changed = 1;
    return 1;
}

// Last frame's copy of a block, accepted when it is this frame's block a little changed (mo_similar,
// which see): a block of another object the address rule found would make motion up. 1 and *out: found (the block itself
// when it is unchanged, *changed 0); 0: none, the draw keeps this frame's block there.
static int mo_prev_block(int index, const p_bind_buf *u, VkDescriptorBufferInfo *out, int *changed) {
    uint64_t max = LIM.ubo_max, n = u->size < max ? u->size : max, pa;
    int how;
    *changed = 0;
    // c9 and c11 keep this frame's block. With the copies the address rule finds for c11 and last frame's c9, the
    // c4 c7 c9 c11 draws (walls, cliffs, arches) got several pixels of motion under a still camera (REKARINA_MV_CHECK,
    // those draws alone: 34 % of the picture moving; with c11 kept, 0.8 %). With c11 kept, their c9 still changes every
    // frame while nothing moves (the check's per-draw line: about 120 such draws a frame changed in c9 alone), the
    // object scale the picture shows is 0.6-0.8 of theirs, and with the hero stopped against a cliff 40-77 % of the picture
    // got 1-3 px of motion that flipped direction, three times the warp error of the camera's vectors. With c9 kept too,
    // the turning run is as good (object warp error 0.46 of the camera's). Why their copies are wrong is not known.
    // REKARINA_MVPREV_SKIP=<hex bank mask> (test) replaces the default a00 (bit n: cn).
    static long skip = -2;
    if (skip == -2) {
        const char *e = cfg_get("REKARINA_MVPREV_SKIP");
        skip = e && *e ? strtol(e, NULL, 16) : 1L << 9 | 1L << 11;
    }
    if (skip >> (index + 3) & 1) return 0;
    uint64_t c0 = __builtin_ia32_rdtsc();
    int found = vk_mvprev_find(index, u->addr, u->size, &pa, &how);
    uint64_t c1 = __builtin_ia32_rdtsc();
    MO.cyc[0] += c1 - c0, MO.ncyc++, MO.big += n;
    if (index >= 0 && index < 14) MO.lk[index][found ? how : MVPREV_NONE]++;
    if (!found || !vk_usnap_find(pa, n, LIM.ubo_align, out)) return 0;
    if (how == MVPREV_SELF) return 1;   // bound at the same address last frame: not written again
    const uint8_t *map = vk_usnap_map(), *cur = vk_usnap_bytes(u->addr, n);
    uint64_t c2 = __builtin_ia32_rdtsc();
    MO.cyc[1] += c2 - c1;
    if (!cur) return 0;
    const uint8_t *old = map + out->offset;
    static long trust = -2;   // REKARINA_MVPREV_TRUST=<hex bank mask> (test): the copy found is taken without the check
    if (trust == -2) {
        const char *e = cfg_get("REKARINA_MVPREV_TRUST");
        trust = e && *e ? strtol(e, NULL, 16) : 0;
    }
    if (trust >> (index + 3) & 1) {
        *changed = memcmp(cur, old, n) != 0;
        return 1;
    }
    int r = mo_similar(cur, old, n, changed);
    MO.cyc[2] += __builtin_ia32_rdtsc() - c2;
    if (!r && index >= 0 && index < 14) MO.lk[index][4]++;
    return r;
}

// mo_prev_block once per block and list: many draws share their blocks (the scene's c14, a model's c7 and c9 across
// its parts), and the lookup with its comparison was most of the checks' time.
static int mo_prev_cached(int index, const p_bind_buf *u, VkDescriptorBufferInfo *out, int *changed) {
    static struct { uint64_t gen, addr, size; int index, r, changed; VkDescriptorBufferInfo out; } bc[4096];
    uint32_t i = (uint32_t)((u->addr ^ (uint64_t)index << 56 ^ u->size << 40) * 0x9E3779B97F4A7C15ull >> 52);
    if (bc[i].gen == mo_gen && bc[i].addr == u->addr && bc[i].size == u->size && bc[i].index == index) {
        *out = bc[i].out, *changed = bc[i].changed;
        return bc[i].r;
    }
    int r = mo_prev_block(index, u, out, changed);
    bc[i].gen = mo_gen, bc[i].addr = u->addr, bc[i].size = u->size, bc[i].index = index;
    bc[i].r = r, bc[i].changed = *changed, bc[i].out = *out;
    return r;
}

static void mvobj_draw_(VkExec *ex, const p_draw_idx *d, VkBuffer ib, VkDeviceSize ib_off, uint64_t t0) {
    mvobj_banks(ex);
    const VkExecTarget *z = &ex->depth;
    if (!z->image || z->width < DLAA_SCENE_MIN_W || z->format != VK_FORMAT_D32_SFLOAT || B.st.samples > 1 || B.st.raster_discard)
        return;
    // solid geometry only, which tests and writes the scene depth: effects, particles and screen quads write none, and
    // drawn again they would put their motion on the surfaces behind them (DLSS takes those from the surface)
    if (!B.st.ds.dt || !B.st.ds.dw) return;
    // a vertex shader that reads a user bank besides the camera's (c4) may move on its own
    const GfxShaderRes *ur;
    int nu = gfx_shader_resources(B.st.vs, GFX_RES_UBO, &ur), reads = 0;
    for (int i = 0; i < nu; i++) reads |= ur[i].slot >= 3 && ur[i].slot <= 16 && ur[i].slot != 4;
    if (!reads) return;
    MO.cand++;
    uint64_t tp = t0 ? kern_now_ns() : 0;
    if (t0) MO.ns_part[0] += tp - t0, MO.n_part[0]++;
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (MQ.frame != frame || MQ.fence != ex->fence || MQ.depth_fmt != z->format) mo_begin(ex, frame);
    if (!__atomic_load_n(&LIM.ubo_max, __ATOMIC_ACQUIRE)) limits_init();
    // the variant: the translation of the same code and answers, kept per shader once found; it names the banks the
    // position depends on (0: none, the shader is skipped from then on)
    static struct { const GfxShader *of, *vs; uint32_t banks; } vmemo[1024];
    uint32_t vi = (uint32_t)((uintptr_t)B.st.vs * 0x9E3779B97F4A7C15ull >> 54);
    if (vmemo[vi].of != B.st.vs) {
        int has, pending = 0;
        sf_mvprev = 1;
        const GfxShader *v = shader_for(ex, NVN_STAGE_VS, &has, &pending, 0);
        sf_mvprev = 0;
        if (!v || pending) {
            MO.building++;
            return;
        }
        GfxShaderIo io[32];
        uint32_t extra = gfx_shader_outputs(v, io) & ~gfx_shader_outputs(B.st.vs, io);
        uint32_t banks = extra ? gfx_shader_mvprev_banks(v) & 0x1fff8u : 0;
        vmemo[vi].of = B.st.vs, vmemo[vi].vs = v, vmemo[vi].banks = banks;
        vk_mvprev_want((banks & ~(1u << 4)) >> 3);   // binding index = bank - 3
    }
    const GfxShader *vs = vmemo[vi].vs;
    uint32_t banks = vmemo[vi].banks;
    if (!banks || !(banks & ~(1u << 4))) {
        MO.no_out++;
        return;
    }
    // a position without the scene camera (c4) has a view of its own (c8 c9 c10 [c13]: its c8 is at a new address every
    // frame, last frame's is never found): drawn again it would get this frame's view and no camera motion
    // (REKARINA_MV_CHECK while turning: 6.2 px end-point error against the camera vectors' 1.7)
    static int noc4 = -1;
    if (noc4 < 0) noc4 = rt_knob("MVOBJ_NOC4", "0");   // REKARINA_MVOBJ_NOC4=1 (test): drawn again all the same
    if (!(banks >> 4 & 1) && !noc4) {
        MO.no_out++;
        return;
    }
    // only positions that read an object's own blocks: bones (c5), place (c7) or c6. The rest (terrain and grass: c14
    // c15 c16, with or without c9) measured no better than the camera's motion (REKARINA_MV_CHECK, turning, those draws
    // alone: object warp error 0.112 against the camera's 0.113, the object part 0.4 of what the picture shows) and
    // were half the draws checked and listed. REKARINA_MVOBJ_ALL=1 (test) takes them again.
    static int all = -1;
    if (all < 0) all = rt_knob("MVOBJ_ALL", "0");
    if (!all && !(banks & (1u << 5 | 1u << 6 | 1u << 7))) {
        MO.no_out++;
        return;
    }
    // REKARINA_MVOBJ_ONLY=<hex bank mask> (test): only draws whose position reads no bank outside the mask (bit n: cn)
    static long only = -2;
    if (only == -2) {
        const char *e = cfg_get("REKARINA_MVOBJ_ONLY");
        only = e && *e ? strtol(e, NULL, 16) : -1;
    }
    if (only >= 0 && banks & ~(uint32_t)only & ~(1u << 4)) return;
    static long exact = -2;   // REKARINA_MVOBJ_EXACT=<hex bank mask> (test): only draws with exactly these position banks
    if (exact == -2) {
        const char *e = cfg_get("REKARINA_MVOBJ_EXACT");
        exact = e && *e ? strtol(e, NULL, 16) : -1;
    }
    if (exact >= 0 && banks != (uint32_t)exact) return;
    mo_mask_note(banks, B.st.vs);
    // last frame's camera and blocks; held still when none of the position's blocks changed (most of the scenery)
    VkDescriptorBufferInfo cam, prevb[17];
    float jit[2];
    if (!vk_dlaa_prev_camera(&cam, jit)) {
        MO.nodata++, MO.why[0]++;
        return;
    }
    static struct { uint64_t frame, key; int moving; VkDescriptorBufferInfo b[17]; uint32_t have; } memo[8192];
    uint64_t key = banks;
    for (int b = 3; b <= 16; b++)
        if (banks >> b & 1 && b != 4) key = (key ^ ex->ubo[NVN_STAGE_VS][b - 3].addr) * 0x100000001b3ull;
    uint32_t mi = (uint32_t)(key * 0xff51afd7ed558ccdull >> 51);
    uint32_t have = 0;
    if (memo[mi].frame == frame && memo[mi].key == key) {
        if (!memo[mi].moving) {
            if (t0) MO.ns_still += kern_now_ns() - tp;
            return (void)MO.still++;
        }
        memcpy(prevb, memo[mi].b, sizeof prevb), have = memo[mi].have;
    } else {
        MO.misses++;
        int moving = 0;
        for (int b = 3; b <= 16; b++) {
            if (!(banks >> b & 1) || b == 4) continue;
            const p_bind_buf *u = &ex->ubo[NVN_STAGE_VS][b - 3];
            int changed;
            if (u->addr && u->size && mo_prev_cached(b - 3, u, &prevb[b], &changed))
                have |= 1u << b, moving |= changed << b;   // the banks whose block changed
        }
        memo[mi].frame = frame, memo[mi].key = key, memo[mi].moving = moving;
        memcpy(memo[mi].b, prevb, sizeof prevb), memo[mi].have = have;
        if (!moving) {
            if (t0) MO.ns_still += kern_now_ns() - tp;
            return (void)MO.still++;
        }
    }
    if (t0) MO.ns_part[1] += kern_now_ns() - tp, MO.n_part[1]++, tp = kern_now_ns();
    GfxShaderIo io[32];
    uint32_t extra = gfx_shader_outputs(vs, io) & ~gfx_shader_outputs(B.st.vs, io);
    const GfxShaderRes *r;
    if (gfx_shader_resources(vs, GFX_RES_IMG, &r)) {
        MO.storage++;
        return;
    }
    const GfxShader *fs = mo_fs(31 - __builtin_clz(extra));
    if (!fs) {
        MO.no_fs++;
        return;
    }
    VkPipeState ms = B.st;
    ms.vs = vs, ms.fs = fs, ms.feedback = 0;
    memset(ms.color_fmt, 0, sizeof ms.color_fmt);
    ms.color_fmt[0] = NVN_FORMAT_RG16F;
    // blending off (blend_en below); the factors stay the game's, valid NVN values
    ms.color.blend_en = 0, ms.color.logic_op = 0;
    ms.chmask.mask = 0x3;   // r and g of target 0
    ms.ms.a2c = 0;
    // the game's own test with its equal case (LESS -> LEQUAL, GREATER -> GEQUAL): the surface the draw left in the
    // scene depth passes, anything nearer hides it (REKARINA_MVOBJ_EQUAL=1: EQUAL, for comparison)
    static int eq = -1;
    if (eq < 0) eq = rt_knob("MVOBJ_EQUAL", "0");
    int df = B.st.ds.df;
    df = eq ? NVN_DEPTH_FUNC_EQUAL : df == NVN_DEPTH_FUNC_LESS ? NVN_DEPTH_FUNC_LEQUAL
       : df == NVN_DEPTH_FUNC_GREATER ? NVN_DEPTH_FUNC_GEQUAL : df == NVN_DEPTH_FUNC_LEQUAL || df == NVN_DEPTH_FUNC_GEQUAL ? df
       : NVN_DEPTH_FUNC_EQUAL;
    ms.ds.dt = 1, ms.ds.dw = 0, ms.ds.df = (uint64_t)df, ms.ds.st = 0;
    // the pipeline per state, kept: the lookup hashes and locks the whole state
    static struct { VkPipeState st; VkPipeline p; const VkPipeInfo *info; int used; } pmemo[64];
    static int plast = -1;   // the entry of the last draw: a model's parts in a row share the state, no hash then
    uint32_t pi;
    if (plast >= 0 && pmemo[plast].used && !memcmp(&pmemo[plast].st, &ms, sizeof ms)) {
        pi = (uint32_t)plast;
    } else {
        uint64_t ph = 0xcbf29ce484222325ull;
        for (size_t i = 0; i < sizeof ms; i += 8) {
            uint64_t w8;
            memcpy(&w8, (const uint8_t *)&ms + i, sizeof ms - i < 8 ? sizeof ms - i : 8);
            ph = (ph ^ w8) * 0x100000001b3ull;
        }
        pi = (uint32_t)(ph >> 58);
    }
    const VkPipeInfo *info = NULL;
    VkPipeline pipe;
    int pending = 0;
    if (pmemo[pi].used && !memcmp(&pmemo[pi].st, &ms, sizeof ms)) {
        info = pmemo[pi].info;
        pipe = __atomic_load_n(&info->pipeline, __ATOMIC_ACQUIRE);   // its optimised build once swapped in
        if (!pipe) return;
    } else {
        pipe = vk_pipeline_graphics_async(&ms, &info, &pending, 0);
        if (pending || !pipe || !info) {
            MO.building += pending != 0;
            return;
        }
        pmemo[pi].st = ms, pmemo[pi].p = pipe, pmemo[pi].info = info, pmemo[pi].used = 1;
    }
    plast = (int)pi;
    uint64_t t1 = kern_now_ns();
    if (t0) MO.ns_part[2] += t1 - tp, MO.n_part[2]++;
    if (t0) MO.ns_check -= t1 - t0;   // counted as recording below
    // into the frame's list, past the primary's binding caches (mo_desc); mo_emit records it
    MQ.d = mo_grow(MQ.d, &MQ.cap, MQ.n + 1, sizeof *MQ.d);
    MoDraw *q = &MQ.d[MQ.n];
    float par_in[4] = {(float)z->width, (float)z->height, jit[0], jit[1]};
    if (!MQ.par.buffer || memcmp(par_in, MQ.par_in, sizeof par_in)) {
        VkBuffer pbuf;
        VkDeviceSize poff;
        memcpy(scratch(ex, 16, &pbuf, &poff), par_in, sizeof par_in);
        MQ.par = (VkDescriptorBufferInfo){pbuf, poff, 16};
        memcpy(MQ.par_in, par_in, sizeof par_in);
    }
    memset(mo_ovr, 0, sizeof mo_ovr);
    for (int bk = 3; bk <= 16; bk++)   // last frame's block, or this frame's where none was found
        mo_ovr[NVN_STAGE_VS][bk + 15] = bk == 4 ? cam : have >> bk & 1 ? prevb[bk] : ubo_info(ex, NVN_STAGE_VS, bk);
    mo_ovr[NVN_STAGE_FS][3] = MQ.par;
    mo_ovr_on = 1;
    int bound = mo_desc(ex, info, q, NULL);
    mo_ovr_on = 0;
    if (!bound) {
        MO.storage++;
        return;
    }
    q->pipe = pipe, q->info = info;
    if (!MQ.nst || memcmp(&MQ.st[MQ.nst - 1], &ms, sizeof ms)) {
        MQ.st = mo_grow(MQ.st, &MQ.cst, MQ.nst + 1, sizeof *MQ.st);
        MQ.st[MQ.nst++] = ms;
    }
    q->st = (uint32_t)MQ.nst - 1;
    q->v = draw_viewport(ex), q->sc = vk_exec_scissor(ex);
    q->bias[0] = ex->poly_offset.units, q->bias[1] = ex->poly_offset.clamp, q->bias[2] = ex->poly_offset.factor;
    q->vb = mo_vertex_buffers(ex);
    q->ib = ib, q->ib_off = ib_off;
    q->it = d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
    q->count = (uint32_t)d->count, q->instances = d->instances > 0 ? (uint32_t)d->instances : 1;
    q->base_vertex = d->base_vertex, q->base_instance = (uint32_t)d->base_instance;
    q->cam = ex->ubo[NVN_STAGE_VS][1].addr;
    MQ.n++;
    MO.draws++;
    for (int i = 0; i < 16; i++)   // the frame's draws per position banks and changed banks (vk_mvobj_frame_masks)
        if (!MF[i].n || (MF[i].mask == banks && MF[i].chg == (uint32_t)memo[mi].moving)) {
            MF[i].mask = banks, MF[i].chg = (uint32_t)memo[mi].moving, MF[i].n++;
            break;
        }
    if (t0) MO.ns_rec += kern_now_ns() - t1, MO.n_rec++;
}
#endif

int vk_mvobj_execute(VkExec *ex, uint64_t cam, VkImageView mv, VkImageView depth, VkFormat depth_fmt, uint32_t w,
                     uint32_t h, uint64_t token) {
    static uint64_t frames;
    if (++frames % 60 == 0)
        rt_log("vk_exec: per-object motion: %.2f ms a frame checking, %.2f ms listing (%.1f us a draw), %.2f ms recording",
               MO.ns_check / 1e6 / (frames ? frames : 1), MO.ns_rec / 1e6 / (frames ? frames : 1),
               MO.n_rec ? MO.ns_rec / 1e3 / MO.n_rec : 0.0, MO.ns_emit / 1e6 / (frames ? frames : 1));
    if (frames % 600 == 0) {
        char l[1400];
        int o = 0;
        for (int i = 0; i < 32 && MM[i].n && o < 1300; i++)
            o += snprintf(l + o, sizeof l - o, " %#x:%llu (u%d s%d t%d)", MM[i].mask, (unsigned long long)MM[i].n, MM[i].ubo,
                          MM[i].ssbo, MM[i].tex);
        if (o) rt_log("vk_exec: per-object motion position banks (bit n: cn) and draws:%s", l);
        o = 0;
        for (int i = 0; i < 14 && o < 1300; i++)
            if (MO.lk[i][0] + MO.lk[i][1] + MO.lk[i][2] + MO.lk[i][3])
                o += snprintf(l + o, sizeof l - o, " c%d %llu/%llu/%llu/%llu/%llu", i + 3, (unsigned long long)MO.lk[i][0],
                              (unsigned long long)MO.lk[i][1], (unsigned long long)MO.lk[i][2],
                              (unsigned long long)MO.lk[i][3], (unsigned long long)MO.lk[i][4]);
        if (o) rt_log("vk_exec: per-object motion last-frame lookups (none/self/side/nearer/refused):%s", l);
    }
    if (frames % 60 == 0 && MO.ran)
        rt_log("vk_exec: per-object motion recorded a frame: %.0f draws, %.0f pipeline binds, %.0f dynamic states, %.0f "
               "pushes of %.1f bindings, %.0f set binds, %.0f vertex buffer binds; sets written %.0f, reused %.0f",
               (double)MO.e_draws / MO.ran, (double)MO.e_pipe / MO.ran, (double)MO.e_dyn / MO.ran, (double)MO.e_push / MO.ran,
               MO.e_push ? (double)MO.e_pushw / MO.e_push : 0.0, (double)MO.e_sets / MO.ran, (double)MO.e_vb / MO.ran,
               (double)MO.e_setw / MO.ran, (double)MO.e_setreuse / MO.ran);
    if (frames % 60 == 0)
        rt_log("vk_exec: per-object motion checks: filter %.2f ms (%.0f a frame), data %.2f ms (%.0f), variant and "
               "pipeline %.2f ms (%.0f); %.0f objects looked up; still draws %.2f ms", MO.ns_part[0] / 1e6 / frames,
               (double)MO.n_part[0] / frames,
               MO.ns_part[1] / 1e6 / frames, (double)MO.n_part[1] / frames, MO.ns_part[2] / 1e6 / frames,
               (double)MO.n_part[2] / frames, (double)MO.misses / frames, MO.ns_still / 1e6 / frames);
    if (frames % 60 == 0 && MO.ncyc)
        rt_log("vk_exec: per-object motion lookups: %llu cycles finding last frame's, %llu the copies, %llu comparing, "
               "a lookup; %llu bytes a lookup", (unsigned long long)(MO.cyc[0] / MO.ncyc),
               (unsigned long long)(MO.cyc[1] / MO.ncyc), (unsigned long long)(MO.cyc[2] / MO.ncyc),
               (unsigned long long)(MO.big / MO.ncyc));
    if (frames % 60 == 0)
        rt_log("vk_exec: per-object motion: %llu draws read place or bones, %llu of them still, %llu drawn again, %llu without last frame's data, %llu "
               "waiting for a shader or pipeline; %llu frames ran, %llu dropped (no data: camera %llu, bones %llu, place %llu, "
               "bones copy %llu, place copy %llu; uniform alignment %llu; variant without its output %llu, with storage %llu, "
               "no fragment shader %llu; another camera %llu)",
               (unsigned long long)MO.cand, (unsigned long long)MO.still, (unsigned long long)MO.draws, (unsigned long long)MO.nodata,
               (unsigned long long)MO.building,
               (unsigned long long)MO.ran, (unsigned long long)MO.dropped, (unsigned long long)MO.why[0],
               (unsigned long long)MO.why[1], (unsigned long long)MO.why[2], (unsigned long long)MO.why[3],
               (unsigned long long)MO.why[4], (unsigned long long)LIM.ubo_align, (unsigned long long)MO.no_out,
               (unsigned long long)MO.storage, (unsigned long long)MO.no_fs, (unsigned long long)MO.other_cam);
    if (!MQ.n) return 0;
    static int exec = -1;
    if (exec < 0) exec = rt_knob("MVOBJ_EXEC", "1");   // REKARINA_MVOBJ_EXEC=0: listed, never recorded (test)
    if (!exec) {
        mo_drop();
        return 0;
    }
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (MQ.frame != frame || MQ.fence != ex->fence || MQ.depth_fmt != depth_fmt) {
        mo_drop();
        return 0;
    }
    int fo = 0;
    for (int i = 0; i < 16 && MF[i].n && fo < (int)sizeof fm_last - 40; i++)
        fo += snprintf(fm_last + fo, sizeof fm_last - fo, " %#x/%#x:%u", MF[i].mask, MF[i].chg, MF[i].n);
    if (!fo) fm_last[0] = 0;
    // only draws with the scene camera block (vk_dlaa.c mv_pass's): last frame's camera they get is that block's, and
    // a draw of another view would move by the difference between the two views
    int kept = 0;
#ifdef GAME_OOT
    for (int i = 0; i < MQ.n; i++)
        if (MQ.d[i].cam == cam && MQ.d[i].projection_token == token && MQ.d[i].skin > 0 && SK.resolved) {
            skin_recorded(MQ.d[i].skin == 2);
            MQ.d[kept++] = MQ.d[i];
        }
#else
    (void)token;
    for (int i = 0; i < MQ.n; i++)
        if (MQ.d[i].cam == cam) MQ.d[kept++] = MQ.d[i];
        else MO.other_cam++;
#endif
    MQ.n = kept;
    if (!MQ.n) {
        mo_drop();
        return 0;
    }
    uint64_t t0 = kern_now_ns();
    VkCommandBuffer cb = ex->cb;
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = mv,
                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingAttachmentInfo da = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = depth,
                                    .imageLayout = VK_IMAGE_LAYOUT_GENERAL, .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD,
                                    .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = 1,
                          .colorAttachmentCount = 1, .pColorAttachments = &ca, .pDepthAttachment = &da};
    vkCmdBeginRendering(cb, &ri);   // through the recorder (vk_rec.h), in order with the draws and the end
    mo_emit(cb);
    vkCmdEndRendering(cb);
    // the draws left their pipeline, descriptors, buffers and dynamic state bound: the next game draw sets everything
    vk_exec_draw_new_cb();
    ex->dirty = VK_EXEC_DIRTY_ALL;
    ex->recorded += MQ.n;
    MQ.n = MQ.nst = MQ.nvb = 0, MQ.ndesc = 0, MQ.frame = 0;
    MO.ran++;
    MO.ns_emit += kern_now_ns() - t0;
    return 1;
}

// The index buffer the replay's command buffer has bound at offset 0 (ibs_buf NULL: none, or bound at another
// offset): an indexed draw whose indices lie in that buffer needs no bind of its own, its offset becomes the draw's
// first index. OoT gives every draw its own index address, 2,700 binds a frame in the forest, nearly all
// into the same few pool buffers. REKARINA_IB_KEEP=0: a bind per draw.
static _Thread_local VkBuffer ibs_buf;
static _Thread_local VkIndexType ibs_type;
static _Thread_local uint64_t ibs_serial;
static _Thread_local VkCommandBuffer ibs_cb;

static void bind_index(VkExec *ex, const void *data, uint64_t bytes, VkIndexType type) {
    VkBuffer b;
    VkDeviceSize o;
    memcpy(scratch(ex, bytes, &b, &o), data, bytes);
    ibs_buf = VK_NULL_HANDLE;
    vkCmdBindIndexBuffer(ex->cb, b, o, type);
    ex->recorded++;
}


// ---- DEBUG-DRAWTIME: REKARINA_DRAWTIME=<frame>: a GPU timestamp before every draw of that frame and at each pass end; three
// frames later the time from one stamp to the next is put on the draw and summed per (vs, fs) pair and per pass.
#define DT_N 16384
static struct {
    int state;   // 0 not read, -1 off, 1 armed, 2 recording, 3 done
    uint64_t frame;
    VkQueryPool pool;
    uint32_t n;
    struct { uint64_t vs, fs; uint32_t pass; uint8_t end; } m[DT_N];
} DT;
static const GfxShader *dt_k[512];
static uint64_t dt_v[512];
static uint64_t dt_fnv(const GfxShader *s) {
    const GfxShader **k = dt_k;
    uint64_t *v = dt_v;
    if (!s) return 0;
    for (int i = 0; i < 512 && k[i]; i++)
        if (k[i] == s) return v[i];
    uint64_t w = 0, h = 0xcbf29ce484222325ull;
    const uint8_t *p = (const uint8_t *)gfx_shader_spirv(s, &w);
    for (uint64_t i = 0; p && i < w * 4; i++) h = (h ^ p[i]) * 0x100000001b3ull;
    for (int i = 0; i < 512; i++)
        if (!k[i]) { k[i] = s, v[i] = h; break; }
    return h;
}
typedef struct { uint64_t vs, fs; double ms; uint32_t n; } DtAgg;
static int dt_by_ms(const void *a, const void *b) {
    double x = ((const DtAgg *)a)->ms, y = ((const DtAgg *)b)->ms;
    return x < y ? 1 : x > y ? -1 : 0;
}
static void dt_report(void) {
    VkCtx *c = vk_ctx();
    uint64_t *ts = malloc(sizeof *ts * DT.n);
    if (!ts) return;
    if (vkGetQueryPoolResults(c->device, DT.pool, 0, DT.n, sizeof *ts * DT.n, ts, sizeof *ts,
                              VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) {
        rt_log("DRAWTIME: no results");
        free(ts);
        return;
    }
    double period = c->props.limits.timestampPeriod / 1e6;   // ms per tick
    static DtAgg agg[4096];
    int na = 0;
    double total = 0, passms[1024] = {0};
    uint32_t passn[1024] = {0};
    for (uint32_t i = 0; i + 1 < DT.n; i++) {
        if (DT.m[i].end || DT.m[i + 1].pass != DT.m[i].pass) continue;   // the stamp after a draw: next draw or pass end
        double ms = (double)(ts[i + 1] - ts[i]) * period;
        if (ms < 0 || ms > 50) continue;
        total += ms;
        if (DT.m[i].pass < 1024) passms[DT.m[i].pass] += ms, passn[DT.m[i].pass]++;
        int k = 0;
        while (k < na && (agg[k].vs != DT.m[i].vs || agg[k].fs != DT.m[i].fs)) k++;
        if (k == na && na < 4096) agg[na++] = (DtAgg){DT.m[i].vs, DT.m[i].fs, 0, 0};
        if (k < na) agg[k].ms += ms, agg[k].n++;
    }
    qsort(agg, (size_t)na, sizeof *agg, dt_by_ms);
    rt_log("DRAWTIME frame %llu: %u stamps, %.2f ms in draws, %d shader pairs", (unsigned long long)DT.frame, DT.n, total, na);
    const char *dir = cfg_get("REKARINA_DRAWTIME_DIR");   // the SPIR-V of the 20 dearest pairs' shaders, as <hash>.spv
    for (int k = 0; dir && k < na && k < 20; k++)
        for (int s = 0; s < 2; s++) {
            uint64_t h = s ? agg[k].fs : agg[k].vs;
            for (int i = 0; h && i < 512 && dt_k[i]; i++)
                if (dt_v[i] == h) {
                    char path[600];
                    snprintf(path, sizeof path, "%s/%016llx.spv", dir, (unsigned long long)h);
                    uint64_t w = 0;
                    const uint32_t *code = gfx_shader_spirv(dt_k[i], &w);
                    FILE *f = fopen(path, "wb");
                    if (f) fwrite(code, 4, (size_t)w, f), fclose(f);
                    break;
                }
        }
    for (int k = 0; k < na && k < 40; k++)
        rt_log("DRAWTIME %7.3f ms %5u draws  vs %016llx fs %016llx", agg[k].ms, agg[k].n, (unsigned long long)agg[k].vs,
               (unsigned long long)agg[k].fs);
    for (uint32_t p = 0; p < 1024; p++)
        if (passms[p] > 0.15) {
            uint64_t v = 0, f = 0;   // the pass's most expensive pair
            double best = -1;
            for (uint32_t i = 0; i + 1 < DT.n; i++)
                if (DT.m[i].pass == p && !DT.m[i].end && DT.m[i + 1].pass == p) {
                    double ms = (double)(ts[i + 1] - ts[i]) * period;
                    if (ms > best) best = ms, v = DT.m[i].vs, f = DT.m[i].fs;
                }
            rt_log("DRAWTIME pass %u: %.3f ms in %u draws; dearest draw %.3f ms vs %016llx fs %016llx", p, passms[p],
                   passn[p], best, (unsigned long long)v, (unsigned long long)f);
        }
    free(ts);
}
static void dt_stamp(VkExec *ex, int end) {
    if (DT.state == 0) {
        const char *e = cfg_get("REKARINA_DRAWTIME");
        DT.state = -1;
        if (!e || !*e) return;
        DT.frame = strtoull(e, NULL, 0);
        VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                    .queryCount = DT_N};
        if (vkCreateQueryPool(vk_ctx()->device, &qi, NULL, &DT.pool) != VK_SUCCESS) return;
        vkResetQueryPool(vk_ctx()->device, DT.pool, 0, DT_N);
        DT.state = 1;
    }
    if (DT.state <= 0 || DT.state == 3) return;
    uint64_t fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (DT.state == 2 && fr >= DT.frame + 3) {
        DT.state = 3;
        dt_report();
        return;
    }
    if (fr != DT.frame || DT.n >= DT_N) return;
    DT.state = 2;
    const GfxShader *vs = NULL, *fs = NULL;
    if (!end) vk_exec_draw_shaders(&vs, &fs);
    static uint64_t pass0;
    if (!DT.n) pass0 = ex->pass_serial;
    DT.m[DT.n] = (typeof(DT.m[0])){dt_fnv(vs), dt_fnv(fs), (uint32_t)(ex->pass_serial - pass0), (uint8_t)end};
    vkCmdWriteTimestamp(ex->cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, DT.pool, DT.n);
    DT.n++;
}
void vk_exec_drawtime_pass_end(VkExec *ex) { dt_stamp(ex, 1); }

static int interface_replay_begin(VkExec *ex) {
    if (!ex->interface_replay_view) return 0;
    vk_exec_pass_end(ex);
    VkMemoryBarrier2 barrier = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, .srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
        .dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT};
    VkDependencyInfo dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
    vkCmdPipelineBarrier2(ex->cb, &dependency);
    VkRenderingAttachmentInfo color = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
        .imageView = ex->interface_replay_view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
        .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo info = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO,
        .renderArea = {{0, 0}, {ex->pass_w, ex->pass_h}}, .layerCount = 1,
        .colorAttachmentCount = 1, .pColorAttachments = &color};
    vkCmdBeginRendering(ex->cb, &info);
    VkPipeState coverage = B.st;
    interface_replay_state(&coverage);
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_GRAPHICS, ex->interface_replay_pipeline);
    vk_pipeline_set_dynamic(ex->cb, &coverage);
    bind_resources(ex, ex->interface_replay_info);
    return 1;
}

static void interface_replay_end(VkExec *ex) {
    vkCmdEndRendering(ex->cb);
    ex->interface_replay_view = VK_NULL_HANDLE;
    ex->recorded += 4;
    B.st_valid = 0;
}

static inline void hdr_output_draw_written(VkExec *ex) {
#ifdef GAME_OOT
    if (vk_par_worker || !vk_hdr_output_pending()) return;
    for (int i = 0; i < ex->pass_ncolor; i++) {
        const VkExecTarget *t = &ex->color[i];
        vk_hdr_output_overwrite(t->image, t->img ? t->img->addr : 0);
    }
#else
    (void)ex;
#endif
}

static void interface_draw_arrays(VkExec *ex, uint32_t count, uint32_t instances, uint32_t first, uint32_t base) {
    vkCmdDraw(ex->cb, count, instances, first, base);
    hdr_output_draw_written(ex);
    if (interface_replay_begin(ex)) {
        vkCmdDraw(ex->cb, count, instances, first, base);
        interface_replay_end(ex);
    }
}

static void interface_draw_indexed(VkExec *ex, uint32_t count, uint32_t instances, uint32_t first, int32_t vertex, uint32_t base) {
    vkCmdDrawIndexed(ex->cb, count, instances, first, vertex, base);
    hdr_output_draw_written(ex);
    if (interface_replay_begin(ex)) {
        vkCmdDrawIndexed(ex->cb, count, instances, first, vertex, base);
        interface_replay_end(ex);
    }
}

static int op_draw(VkExec *ex, const p_draw *d) {
    c0_binst = (uint32_t)d->base_instance;
    if (d->count <= 0 || !prepare_draw(ex, d->prim, d->count, 0)) return 0;
    if (!vk_par_worker) vk_exec_frame.draws++;   // a diagnostic count: not written by every chunk's thread
    dt_stamp(ex, 0);   // DEBUG-DRAWTIME
    vk_probe_draw(ex, (uint32_t)d->count, (uint32_t)d->instances);   // DEBUG-PROBE
    grass_log(ex, "arr", d->count, d->instances, d->base_instance, d->first, 0, 0, 0);   // DEBUG-GRASSLOG
    uint32_t inst = d->instances > 0 ? (uint32_t)d->instances : 1, first = (uint32_t)d->first, n = (uint32_t)d->count;
    if (d->prim == NVN_DRAW_PRIMITIVE_QUADS && n >= 4) {   // quads become two triangles each
        uint32_t q = n / 4;
        uint32_t *ix = malloc(sizeof *ix * 6 * q);
        if (!ix) rt_fatal("vk: out of memory");
        for (uint32_t i = 0; i < q; i++) {
            uint32_t v = first + 4 * i, k[6] = {0, 1, 2, 0, 2, 3};
            for (int j = 0; j < 6; j++) ix[6 * i + j] = v + k[j];
        }
        bind_index(ex, ix, sizeof *ix * 6 * q, VK_INDEX_TYPE_UINT32);
        free(ix);
        GPU_COND_BEGIN(ex);
        interface_draw_indexed(ex, 6 * q, inst, 0, 0, (uint32_t)d->base_instance);
        GPU_COND_END(ex);
    } else if (d->prim == NVN_DRAW_PRIMITIVE_LINE_LOOP && n >= 2) {   // a strip that returns to its first vertex
        uint32_t *ix = malloc(sizeof *ix * (n + 1));
        if (!ix) rt_fatal("vk: out of memory");
        for (uint32_t i = 0; i < n; i++) ix[i] = first + i;
        ix[n] = first;
        bind_index(ex, ix, sizeof *ix * (n + 1), VK_INDEX_TYPE_UINT32);
        free(ix);
        GPU_COND_BEGIN(ex);
        interface_draw_indexed(ex, n + 1, inst, 0, 0, (uint32_t)d->base_instance);
        GPU_COND_END(ex);
    } else {
        GPU_COND_BEGIN(ex);
        interface_draw_arrays(ex, n, inst, first, (uint32_t)d->base_instance);
        GPU_COND_END(ex);
    }
    ex->recorded++;
    return 1;
}

// Primitive restart of an indexed draw. NVN restarts strips wherever an index equals the SetPrimitiveRestart value;
// Vulkan restarts only at the all-ones value of the bound index type, and only strips and fans without
// primitiveTopologyListRestart. Byte indices are widened to 16 bits, so their restart value becomes 0xffff
// (op_draw_idx). Without restart the 0xffff entries of TOTK's terrain and cliff strips fetch vertex 65535 + base and
// the triangles around them drop out.
static int restart_for(const VkExec *ex, int prim, int itype) {
    if (!ex->prim_restart.enable) return 0;
    switch (vk_pipeline_topology(prim)) {
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN:
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY: case VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY: break;
    default: return 0;   // lists: a restart index only drops the unfinished primitive
    }
    static const uint32_t all_ones[3] = {0xffu, 0xffffu, 0xffffffffu};
    if (ex->prim_restart.index == all_ones[itype]) return 1;
    warn_once(W_RESTART, "primitive restart index %#x with %d-byte indices is not replayed: strips drawn without restart",
              ex->prim_restart.index, 1 << itype);
    return 0;
}

// DEBUG-DRAWDBG: per draw of the dumped frame into the main depth targets, the bytes the vertex stage reads
static uint64_t dbg_fnv(const uint8_t *p, uint64_t n) {
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}
static void dbg_draw(VkExec *ex, int count) {
    int vk_dump_active_dbg(void), vk_dump_drawdbg(void);
    if (!vk_dump_armed || !vk_dump_drawdbg() || !ex->depth.img || !vk_dump_active_dbg()) return;
    uint64_t da = ex->depth.img->addr;
    if (da != 0x8000000000ull && da != 0x8001518e00ull) return;
    char line[1024];
    int o = snprintf(line, sizeof line, "DRAWDBG depth %#llx df %d dw %d prog %llu n %d", (unsigned long long)da,
                     (int)ex->pipe.ds.df, (int)ex->pipe.ds.dw, (unsigned long long)(ex->prog ? ex->prog->serial : 0), count);
    o += snprintf(line + o, sizeof line - (size_t)o, " vp %g,%g %gx%g z %g..%g dr %g..%g po %d %g/%g/%g", ex->vp[0][0],
                  ex->vp[0][1], ex->vp[0][2], ex->vp[0][3], ex->vp[0][4], ex->vp[0][5], ex->depth_range[0], ex->depth_range[1],
                  (int)ex->pipe.poly.offset_en, ex->poly_offset.factor, ex->poly_offset.units, ex->poly_offset.clamp);
    for (int s = 0; s < 3; s++) {
        const p_bind_buf *u = &ex->ubo[NVN_STAGE_VS][s];
        if (!u->addr || !u->size) continue;
        uint64_t n = u->size < 0x4000 ? u->size : 0x4000;
        const uint8_t *snap = vk_usnap_bytes(u->addr, n);
        VkMemSpan sp;
        uint64_t hg = vk_mem_resolve(u->addr, n, &sp) && !sp.alloc ? dbg_fnv((const uint8_t *)(uintptr_t)u->addr, n) : 0;
        VkDescriptorBufferInfo sbi;
        uint64_t mx = LIM.ubo_max;
        int live = !vk_usnap_find(u->addr, u->size < mx ? u->size : mx, LIM.ubo_align, &sbi);
        o += snprintf(line + o, sizeof line - (size_t)o, " c%d %#llx+%#llx %s snap %016llx guest %016llx", s + 3,
                      (unsigned long long)u->addr, (unsigned long long)u->size, live ? "LIVE" : "snap",
                      (unsigned long long)(snap ? dbg_fnv(snap, n) : 0), (unsigned long long)hg);
    }
    for (int i = 0; i < 2; i++) {
        const NVNbufferRange *v = &ex->vbuf[i];
        if (!v->address) continue;
        uint64_t n = v->size < (4u << 20) ? v->size : (4u << 20);
        VkMemSpan sp;
        uint64_t hg = vk_mem_resolve(v->address, n, &sp) && !sp.alloc ? dbg_fnv((const uint8_t *)(uintptr_t)v->address, n) : 0;
        o += snprintf(line + o, sizeof line - (size_t)o, " vb%d %#llx+%#llx %016llx", i, (unsigned long long)v->address,
                      (unsigned long long)v->size, (unsigned long long)hg);
    }
    rt_log("%s", line);
}

// DEBUG-GRASSLOG: REKARINA_GRASSLOG=<vs spirv hash>,... with REKARINA_GRASSLOG_FROM=<frame> (default 4400), 40 frames: per draw
// with that vertex shader, the draw's numbers and hashes of the bytes its vertex stage reads (index bytes, vertex
// streams, uniform blocks as the snapshot and guest memory hold them, indirect arguments)
static uint64_t gl_fnv(const uint8_t *p, uint64_t n) {
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < n; i++) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}
static uint64_t gl_guest(uint64_t addr, uint64_t n) {   // 0: not host memory
    VkMemSpan sp;
    if (!addr || !n || !vk_mem_resolve(addr, n, &sp) || sp.alloc || !sp.pool || nvn_gpu_only(addr)) return 0;
    return gl_fnv((const uint8_t *)(uintptr_t)addr, n);
}
static void grass_log(VkExec *ex, const char *kind, int count, int inst, int first, int basev, uint64_t ibuf, int isz,
                      uint64_t ind) {
    static int n = -1;
    static uint64_t want[8], from;
    static const GfxShader *seen_s[64];
    static uint64_t seen_h[64];
    static int nseen;
    if (n < 0) {
        n = 0;
        const char *s = cfg_get("REKARINA_GRASSLOG"), *f = cfg_get("REKARINA_GRASSLOG_FROM");
        from = f && *f ? strtoull(f, NULL, 0) : 4400;
        for (char *x; s && *s && n < 8; s = *x == ',' ? x + 1 : x) {
            want[n++] = strtoull(s, &x, 16);
            if (x == s) break;
        }
    }
    uint64_t fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    {
        static int once;
        if (!once && fr >= from) once = 1, rt_log("GRASS first call at frame %llu: %d hashes, vs %p", (unsigned long long)fr, n, (const void *)gl_cur_vs);
    }
    if (!n || !gl_cur_vs) return;
    if (fr < from || fr >= from + 40) return;
    uint64_t h = 0;
    for (int i = 0; i < nseen; i++)
        if (seen_s[i] == gl_cur_vs) h = seen_h[i];
    if (!h) {
        uint64_t words = 0;
        const uint8_t *b = (const uint8_t *)gfx_shader_spirv(gl_cur_vs, &words);
        h = 0xcbf29ce484222325ull;
        for (uint64_t i = 0; b && i < words * 4; i++) h = (h ^ b[i]) * 0x100000001b3ull;
        if (nseen < 64) seen_s[nseen] = gl_cur_vs, seen_h[nseen] = h, nseen++;
    }
    int hit = 0;
    for (int i = 0; i < n; i++) hit |= want[i] == h;
    if (!hit) return;
    char line[2048];
    int o = snprintf(line, sizeof line, "GRASS frame %llu vs %016llx %s n %d inst %d first %d basev %d", (unsigned long long)fr,
                     (unsigned long long)h, kind, count, inst, first, basev);
    if (ibuf && count > 0)
        o += snprintf(line + o, sizeof line - (size_t)o, " ib %#llx %016llx", (unsigned long long)ibuf,
                      (unsigned long long)gl_guest(ibuf, (uint64_t)count * (uint64_t)isz));
    if (ind) {
        VkMemSpan sp;
        const uint32_t *a = vk_mem_resolve(ind, 20, &sp) && !sp.alloc && !nvn_gpu_only(ind) ? (const uint32_t *)(uintptr_t)ind : NULL;
        if (a) o += snprintf(line + o, sizeof line - (size_t)o, " ind %#llx {%u %u %u %d %u}", (unsigned long long)ind, a[0], a[1],
                             a[2], (int)a[3], a[4]);
        else o += snprintf(line + o, sizeof line - (size_t)o, " ind %#llx {device}", (unsigned long long)ind);
    }
    for (int b = 0; b < 16 && o < (int)sizeof line - 120; b++) {
        const NVNbufferRange *v = &ex->vbuf[b];
        if (!v->address || !ex->pipe.vstream[b].stride) continue;
        uint64_t m = v->size < (1u << 20) ? v->size : (1u << 20);
        o += snprintf(line + o, sizeof line - (size_t)o, " vb%d %#llx+%#llx/%u %016llx", b, (unsigned long long)v->address,
                      (unsigned long long)v->size, ex->pipe.vstream[b].stride, (unsigned long long)gl_guest(v->address, m));
    }
    for (int s = 0; s < 18 && o < (int)sizeof line - 120; s++) {
        const p_bind_buf *u = &ex->ubo[NVN_STAGE_VS][s];
        if (!u->addr || !u->size) continue;
        uint64_t m = u->size < 0x10000 ? u->size : 0x10000;
        const uint8_t *snap = vk_usnap_bytes(u->addr, m);
        o += snprintf(line + o, sizeof line - (size_t)o, " c%d %#llx+%#llx s%016llx g%016llx", s + 3,
                      (unsigned long long)u->addr, (unsigned long long)u->size,
                      (unsigned long long)(snap ? gl_fnv(snap, m) : 0), (unsigned long long)gl_guest(u->addr, m));
    }
    rt_log("%s", line);
}

static int op_draw_idx(VkExec *ex, const p_draw_idx *d) {
    c0_binst = (uint32_t)d->base_instance;
    static const uint32_t isize[3] = {1, 2, 4};
    if (d->itype < 0 || d->itype > 2) return warn_once(W_INDEX, "index type %d is unknown: draw skipped", d->itype), 0;
    int restart = restart_for(ex, d->prim, d->itype);
    if (d->count <= 0 || !prepare_draw(ex, d->prim, d->count, restart)) return 0;
    if (!vk_par_worker) vk_exec_frame.draws++;   // a diagnostic count: not written by every chunk's thread
    dt_stamp(ex, 0);   // DEBUG-DRAWTIME
    dbg_draw(ex, d->count);   // DEBUG-DRAWDBG
    vk_probe_draw(ex, (uint32_t)d->count, (uint32_t)d->instances);   // DEBUG-PROBE
    grass_log(ex, "idx", d->count, d->instances, d->base_instance, d->base_vertex, d->ibuf, (int)isize[d->itype], 0);   // DEBUG-GRASSLOG
    uint64_t bytes = (uint64_t)d->count * isize[d->itype];
    uint32_t first_index = 0;
    VkMemSpan sp;
    if (!vk_mem_resolve(d->ibuf, bytes, &sp)) return warn_once(W_INDEX, "index buffer %#llx is not backed: draws with it are skipped", (unsigned long long)d->ibuf), 0;
    if (d->itype == NVN_INDEX_TYPE_UNSIGNED_BYTE) {   // no uint8 index type here: widen (host-readable backing only)
        if (sp.alloc) return warn_once(W_INDEX, "byte indices in device-local memory: draw skipped"), 0;
        vk_mem_count(sp.cls, VK_MUSE_CPU_READ, bytes);
        vk_mem_count(VK_MCLS_SCRATCH, VK_MUSE_IB, 2 * bytes);
        uint16_t *ix = malloc(sizeof *ix * (size_t)d->count);
        if (!ix) rt_fatal("vk: out of memory");
        for (int i = 0; i < d->count; i++) {
            uint8_t v = ((const uint8_t *)(uintptr_t)d->ibuf)[i];
            ix[i] = restart && v == 0xff ? 0xffff : v;
        }
        bind_index(ex, ix, sizeof *ix * (size_t)d->count, VK_INDEX_TYPE_UINT16);
        free(ix);
    } else if (sp.offset % isize[d->itype]) {
        return warn_once(W_INDEX, "index buffer %#llx is not aligned to its index size: draw skipped", (unsigned long long)d->ibuf), 0;
    } else {
        VK_MEM_USE(sp);
        VkIndexType it = d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32;
        static int keep = -1;
        if (keep < 0) keep = rt_knob("IB_KEEP", "1");
        uint64_t first = sp.offset / isize[d->itype];
        if (keep && first + (uint64_t)d->count <= UINT32_MAX) {
            uint64_t serial = vk_sync_cb_serial();
            if (ibs_buf != sp.buffer || ibs_type != it || ibs_serial != serial || ibs_cb != ex->cb) {
                ibs_buf = sp.buffer, ibs_type = it, ibs_serial = serial, ibs_cb = ex->cb;
                vkCmdBindIndexBuffer(ex->cb, sp.buffer, 0, it);
                ex->recorded++;
            }
            first_index = (uint32_t)first;
        } else {
            ibs_buf = VK_NULL_HANDLE;
            vkCmdBindIndexBuffer(ex->cb, sp.buffer, sp.offset, it);
            ex->recorded++;
        }
        vk_mem_count(sp.cls, VK_MUSE_IB, bytes);
        VK_MEM_BOUND(sp);
    }
    GPU_COND_BEGIN(ex);
    interface_draw_indexed(ex, (uint32_t)d->count, d->instances > 0 ? (uint32_t)d->instances : 1, first_index, d->base_vertex,
                     (uint32_t)d->base_instance);
    GPU_COND_END(ex);
    ex->recorded++;
    if (d->itype != NVN_INDEX_TYPE_UNSIGNED_BYTE) mvobj_draw(ex, d, sp.buffer, sp.offset);
    return 1;
}

// DrawElementsIndirect: the arguments are read by the GPU from the pool buffer (they are often written by an earlier
// dispatch), as nxnative nvn_mtl.m does. The base instance stays out of c0 (0x644): c0 cannot be patched by a transfer
// inside a pass; gl_BaseInstance of the Vulkan draw is the argument's. Byte indices need the count to be widened: skipped.
static void op_draw_idx_ind(VkExec *ex, const p_draw_idx_ind *d) {
    c0_binst = 0;
    if (d->itype != NVN_INDEX_TYPE_UNSIGNED_SHORT && d->itype != NVN_INDEX_TYPE_UNSIGNED_INT)
        return warn_once(W_INDEX, "indirect draw with index type %d: skipped", d->itype);
    VkMemSpan ib, ip;
    if (!vk_mem_resolve(d->ind, sizeof(VkDrawIndexedIndirectCommand), &ip) || ip.offset % 4)
        return warn_once(W_INDIRECT, "indirect draw arguments at %#llx are not backed or not aligned: draws skipped", (unsigned long long)d->ind);
    if (!vk_mem_resolve(d->ibuf, 1, &ib) || ib.offset % (d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 : 4))
        return warn_once(W_INDEX, "index buffer %#llx is not backed or not aligned: indirect draws skipped", (unsigned long long)d->ibuf);
    if (!prepare_draw(ex, d->prim, 0x7fffffff, restart_for(ex, d->prim, d->itype))) return;
    if (!vk_par_worker) vk_exec_frame.draws++;   // a diagnostic count: not written by every chunk's thread
    dt_stamp(ex, 0);   // DEBUG-DRAWTIME
    grass_log(ex, "idx-ind", 0, 0, 0, 0, d->ibuf, d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? 2 : 4, d->ind);   // DEBUG-GRASSLOG
    VK_MEM_USE(ip);
    VK_MEM_USE(ib);
    ibs_buf = VK_NULL_HANDLE;
    vkCmdBindIndexBuffer(ex->cb, ib.buffer, ib.offset, d->itype == NVN_INDEX_TYPE_UNSIGNED_SHORT ? VK_INDEX_TYPE_UINT16 : VK_INDEX_TYPE_UINT32);
    GPU_COND_BEGIN(ex);
    vkCmdDrawIndexedIndirect(ex->cb, ip.buffer, ip.offset, 1, sizeof(VkDrawIndexedIndirectCommand));
    GPU_COND_END(ex);
    hdr_output_draw_written(ex);
    if (interface_replay_begin(ex)) {
        vkCmdDrawIndexedIndirect(ex->cb, ip.buffer, ip.offset, 1, sizeof(VkDrawIndexedIndirectCommand));
        interface_replay_end(ex);
    }
    ex->recorded += 2;
    vk_mem_count(ip.cls, VK_MUSE_IND, sizeof(VkDrawIndexedIndirectCommand));
    VK_MEM_BOUND(ip);
    VK_MEM_BOUND(ib);
}

#ifdef GAME_NS2
// ---- NVN2 geometry and mesh draws (GAME_NS2)
// Programs with a geometry stage (vertex, geometry, fragment) and NVN2's task and mesh programs (task optional, mesh,
// fragment) are drawn with pipelines built whole by vk_pipeline_stages: the graphics pipeline library path above knows
// vertex and fragment stages only. The geometry stage runs as Vulkan's own (MacNativeGames and seedds/nxnative run it as
// compute because Metal has none); task and mesh shaders run through VK_EXT_mesh_shader. Shaders are translated
// synchronously and pipelines built on the draw's thread, so these draws are never skipped for a missing pipeline.
// REKARINA_FOLIAGE_SHADOW_CULL=<layer>[f] (0 to 3; 3 turns it off): the foliage's mesh draws into the sun's shadow
// cascades from this array layer on drop one side of their cards ("f": the front faces, else the back faces). The game
// draws them double-sided, and stacks of alpha-tested cards seen from the sun reach about 1,000 million fragments in one
// draw of the far cascade. The cascades hold nothing but foliage (buildings, terrain and trunks are in the static map),
// and a card is one plane: the cards whose culled side faces the sun cast no shadow at all, so from cascade 1 (about
// 10 m) on every tree's and bush's shadow lost about half its cards, lighter and full of holes (players: shorter shadow
// distance than other ports). Off by default on NVIDIA, where it bought no measurable frame time (vk_exec_draw.c
// foliage_shadow_default); layer 1 elsewhere. Cull mode is dynamic state, so this needs no other pipeline.
static int foliage_shadow_default(int other) { return vk_ctx()->props.vendorID == 0x10de ? 3 : other; }
static void foliage_shadow_cull(const VkExec *ex, VkPipeState *s) {
    static int from = -1, face;
    if (from < 0) {
        const char *e = cfg_get("REKARINA_FOLIAGE_SHADOW_CULL");
        from = e && *e ? atoi(e) : foliage_shadow_default(1);
        face = e && strchr(e, 'f') ? NVN_FACE_FRONT : NVN_FACE_BACK;
    }
    if (!ex->color[0].image && ex->depth.image && (int)ex->depth.base_layer >= from && s->poly.cull == NVN_FACE_NONE)
        s->poly.cull = (unsigned)face;
}

// REKARINA_FOLIAGE_SHADOW_VRS (0 to 3; 3 turns it off): the foliage's mesh draws in the shadow cascades from this array
// layer on shade one fragment per 2x2 pixels. A leaf's alpha test then runs a quarter as often, and its cut-out is
// decided per 2x2 block of texels: blocky leaf shadows from cascade 2 (about 32 m) on, and since the game fits its
// cascades to the view every frame, the blocks land elsewhere on the leaves in each frame while the camera moves, and
// on every sway of a leaf: the far trees' shadows flickered more (measured: docs/oot-notes.md, "Foliage shadows").
// Off by default on NVIDIA (foliage_shadow_default), layer 2 elsewhere. A mesh pipeline has the rate as dynamic state,
// so every mesh draw sets it.
static void mesh_shading_rate(VkExec *ex) {
    static PFN_vkCmdSetFragmentShadingRateKHR fn;
    static int looked, from;
    if (!looked) {
        looked = 1;
        const char *e = cfg_get("REKARINA_FOLIAGE_SHADOW_VRS");
        from = e && *e ? atoi(e) : foliage_shadow_default(2);
        if (vk_ctx()->has_vrs)
            fn = (PFN_vkCmdSetFragmentShadingRateKHR)vkGetDeviceProcAddr(vk_ctx()->device, "vkCmdSetFragmentShadingRateKHR");
    }
    if (!fn) return;
    rec_shading_rate(fn, ex->cb, !ex->color[0].image && ex->depth.image && (int)ex->depth.base_layer >= from ? 2 : 1);
    ex->recorded++;
}

// REKARINA_MESH_TASK=auto|0|1: whether draws through a task shader are drawn (OoT's short grass). On AMD's Windows
// driver (seen on the RDNA 2 branch, 2026.04.17) such a draw under a GPU render condition stops the GPU or ends in an
// access violation inside amdvlk64.dll; mesh_task_cond draws them without the condition there, so auto draws them
// everywhere (until 2026-11-15 auto left them out on that driver: the short grass was missing).
// REKARINA_MESH_TASK_COND=auto|0|1: whether a draw through a task shader stays under the game's GPU render condition.
// On AMD's Windows driver a task-shader draw inside conditional rendering stops the GPU or crashes the driver (the
// breadcrumb of the lost device was a direct mesh draw under a condition); outside one it draws (2026-11-15, the
// integrated Radeon: the short grass shows, no fault in the runs). auto: no condition for those draws on that driver
// (the grass the game's occlusion query would cull is drawn anyway: a little more GPU work), the condition elsewhere.
static _Thread_local int task_draw;   // prepare_stages: the draw being prepared runs a task shader
static int mesh_task_cond(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("REKARINA_MESH_TASK_COND");
        on = e && *e && strcmp(e, "auto") ? atoi(e) != 0 : !vk_caps.amd_prop;
        if (!on) rt_log("vk_exec: draws through task shaders are drawn outside the game's render conditions (REKARINA_MESH_TASK_COND)");
    }
    return on;
}

static int mesh_task_draws(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("REKARINA_MESH_TASK");
        on = e && *e && strcmp(e, "auto") ? atoi(e) != 0 : 1;
        if (!on) rt_log("vk_exec: draws through task shaders are left out (REKARINA_MESH_TASK=0): the short grass is missing");
    }
    return on;
}

static int stages_has(VkExec *ex, int st) {
    const uint8_t *ctl, *code;
    uint32_t len;
    return ex->prog && find_stage(ex->prog, st, &ctl, &code, &len);
}

// mesh: 0 vertex (and geometry) and fragment stages, 1 task and mesh stages, 2 an emulated mesh draw (mesh_emulate):
// prep_emu_vs and the fragment stage, the vertices bound by the caller.
static _Thread_local const GfxShader *prep_emu_vs;
static int prepare_stages(VkExec *ex, int mesh, int prim, int restart) {
    if (ex->render_disabled) {
        void vk_oqlog_skip(VkExec *ex);
        vk_oqlog_skip(ex);   // DEBUG-OQLOG
        return 0;
    }
    cb_res(ex);
    dummies_init();
    if (ex->pass_noatt) {   // as prepare_draw
        uint32_t w, h;
        vk_exec_noatt_dims(ex, &w, &h);
        if (ex->in_pass && (w > ex->pass_w || h > ex->pass_h)) vk_exec_pass_end(ex);
        if (!ex->in_pass) ex->rt_resolved = 0;
    }
    if (!ex->in_pass) ex->dirty |= VK_EXEC_DIRTY_VIEWPORT | VK_EXEC_DIRTY_SCISSOR;
    vk_exec_pass_begin(ex);
    VkRect2D sc = vk_exec_scissor(ex);
    if (!sc.extent.width || !sc.extent.height) return 0;
    if (!ex->prog) return vk_exec_oq_draw_lost(ex), warn_once(W_PROGRAM, "a draw without a bound program is skipped"), 0;
    int h0, h1, h2;
    const GfxShader *sh[3];
    uint64_t sec1 = sec_begin();
    // a shader not translated yet is translated beside the replay and the draw left out meanwhile, as its pipeline
    // is built beside (vk_pipeline.c mesh_queue): neither wait belongs on the replay thread
    int w0 = 0, w1 = 0, w2 = 0;
    if (mesh == 2) {
        sh[0] = prep_emu_vs, sh[1] = NULL, h0 = h1 = 1;
        sh[2] = shader_for(ex, NVN_STAGE_FS, &h2, &w2, 0);
    } else {
        sh[0] = shader_for(ex, mesh ? NVN_STAGE_TS : NVN_STAGE_VS, &h0, &w0, 0);
        sh[1] = shader_for(ex, mesh ? NVN_STAGE_MS : NVN_STAGE_GS, &h1, &w1, 0);
        sh[2] = shader_for(ex, NVN_STAGE_FS, &h2, &w2, 0);
    }
    sec_end(SEC_SHADER_MESH, sec1);
    if (w0 || w1 || w2) return left_out(LO_MESH_SHADER, ex->prog->serial, NULL), vk_exec_oq_draw_lost(ex), 0;
    if (mesh == 2 ? !sh[0] || (h2 && !sh[2]) : !h1 || !sh[1] || (h0 && !sh[0]) || (h2 && !sh[2]) || (!mesh && !sh[0])) {
        static uint8_t once[2];
        if (!__atomic_exchange_n(&once[mesh & 1], 1, __ATOMIC_RELAXED))
            rt_log("vk_exec: a %s draw without its shaders (%s %s, %s %s, fragment %s): skipped (logged once)",
                   mesh ? "mesh" : "geometry", mesh ? "task" : "vertex", !h0 ? "-" : sh[0] ? "ok" : "failed",
                   mesh ? "mesh" : "geometry", !h1 ? "-" : sh[1] ? "ok" : "failed", !h2 ? "-" : sh[2] ? "ok" : "failed");
        return vk_exec_oq_draw_lost(ex), 0;
    }
    if (skip_sh(sh[0]) || skip_sh(sh[1]) || skip_sh(sh[2])) return 0;   // DEBUG-SKIPSH
    if (mesh == 1 && sh[0] && !mesh_task_draws()) return vk_exec_oq_draw_lost(ex), 0;
    task_draw = mesh == 1 && sh[0];
    VkPipeState s = ex->pipe;
    s.vs = mesh == 1 ? NULL : sh[0], s.fs = sh[2];
    if (mesh) foliage_shadow_cull(ex, &s);
    if (mesh == 1) vk_dlaa_pjitter(ex, NVN_STAGE_MS);
    if (mesh != 2) vk_dlaa_ubodump(ex, sh[0], sh[1], sh[2], mesh);
    s.feedback = 0;
    s.prim = (uint8_t)prim;
    s.depth_minus_w = (uint8_t)(nvn_depth_minus_w() != 0);
    s.y_flip = 1;
    s.prim_restart = (uint8_t)(restart != 0);
    const VkPipeInfo *info = NULL;
    sec1 = sec_begin();
    VkPipeline p = vk_pipeline_stages(&s, sh, mesh, &info);
    sec_end(SEC_PIPE_MESH, sec1);
    if (!p && vk_pipeline_stages_pending())
        return left_out(LO_MESH_PIPE, ex->prog->serial, sh[1] ? sh[1] : sh[0]), vk_exec_oq_draw_lost(ex), 0;
    if (!p || !info) return vk_exec_oq_draw_lost(ex), warn_once(W_PIPELINE, "a geometry or mesh pipeline could not be created: draws with it are skipped"), 0;
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_GRAPHICS, p);
    if (mesh == 2) {   // its vertex input and topology are static: the dynamic state is set without them, and forgotten after
        vk_pipeline_set_dynamic_no_assembly(ex->cb, &s);
        ex->recorded++;
        B.st_valid = 0;
    } else if (!B.st_valid || memcmp(&B.st, &s, sizeof s)) {
        uint64_t sec1 = sec_begin();
        if (mesh) vk_pipeline_set_dynamic_no_assembly(ex->cb, &s);
        else vk_pipeline_set_dynamic(ex->cb, &s);
        sec_end(SEC_SET_DYN, sec1);
        ex->recorded++;
        B.st = s, B.st_valid = 1;
    }
    record_fixed(ex);
    bind_resources(ex, info);
    if (mesh == 1 && !info->mesh_standin) mesh_shading_rate(ex);
    if (!mesh) bind_vertex_buffers(ex);
    if (vk_dump_armed && mesh != 2) {   // listed with the task (else mesh, else geometry) shader in the vertex column, prim 100 + mesh
        VkPipeState d = s;
        d.vs = mesh ? (sh[0] ? sh[0] : sh[1]) : sh[0];
        d.prim = (uint8_t)(100 + mesh);
        vk_dump_draw(ex, &d, p, info);
        if (mesh && sh[0]) { d.vs = sh[1], d.prim = 102; vk_dump_draw(ex, &d, p, info); }   // and the mesh shader
    }
    vk_exec_oq_draw(ex);
    return 1;
}

// DrawMeshTasks(first, count) and DrawMeshTasksIndirect: vkCmdDrawMeshTasksEXT with count workgroups along x. NVN2
// numbers them from first, which the translated task (or task-less mesh) shader adds to its workgroup number from c0
// (VK_C0_MESH_FIRST). The indirect record is NVN's {count, first} (assumed, nvn_obj.h p_draw_mesh_ind): its count is
// copied on the GPU into a {count, 1, 1} argument before the pass, and first is taken as 0 (said once).
// DEBUG-MESHCOUNT: REKARINA_MESHCOUNT=<frame>: for 40 frames from <frame>, the {count, first} each DrawMeshTasksIndirect
// record held when the GPU copied it, read back at the fence and logged per frame (record address and count).
// REKARINA_MESHCOUNT=first: the 40 frames from the first such copy of the run (a scene's first frames, where the
// records may not have been filled yet). Both count copy paths record: the batch (mesh_ind_copy) and the one per draw.
// With REKARINA_MESHCOUNT_PROG a batched copy carries the program and pass width of the draw that started its batch,
// not its own draw's: REKARINA_MESH_BATCH=0 names each draw's own.
typedef struct { uint64_t frame, ind, prog; uint32_t w, h; } McMeta;
static VkBuffer mc_buf;
static VmaAllocation mc_alloc;
static uint32_t *mc_map;
static McMeta mc_meta[8192];
static uint32_t mc_next;
static uint64_t mc_from;
typedef struct { uint32_t lo, hi; } McJob;
static void mc_done(void *arg) {
    McJob *j = arg;
    char line[1600];
    int o = 0;
    uint64_t fr = ~0ull;
    for (uint32_t i = j->lo; i < j->hi; i++) {
        uint32_t k = i % 8192;
        if (mc_meta[k].frame != fr || o > 1400) {
            if (o) rt_log("%s", line);
            fr = mc_meta[k].frame;
            o = snprintf(line, sizeof line, "MESHCOUNT frame %llu:", (unsigned long long)fr);
        }
        const uint32_t *w = mc_map + (size_t)k * 16;
        o += snprintf(line + o, sizeof line - (size_t)o, " %llx=%u/%u", (unsigned long long)mc_meta[k].ind, w[0], w[1]);
        if (cfg_env("REKARINA_MESHCOUNT_PROG"))   // the draw's program and the pass's width
            o += snprintf(line + o, sizeof line - (size_t)o, "(p%llu,%u)", (unsigned long long)mc_meta[k].prog, mc_meta[k].w);
        if (cfg_env("REKARINA_MESHCOUNT_WIDE"))   // the record's next 14 words too
            for (int q = 2; q < 16; q++) o += snprintf(line + o, sizeof line - (size_t)o, "%c%x", q == 2 ? '[' : ' ', w[q]);
        if (cfg_env("REKARINA_MESHCOUNT_WIDE")) o += snprintf(line + o, sizeof line - (size_t)o, "]");
    }
    if (o) rt_log("%s", line);
    free(j);
}
static uint32_t mc_sub_lo;
static uint64_t mc_sub_fence;
static void mc_record(VkExec *ex, VkBuffer src, VkDeviceSize off, uint64_t ind, uint64_t avail) {
    static int on = -1;
    uint64_t fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (on < 0) {
        const char *e = cfg_get("REKARINA_MESHCOUNT");
        on = e && *e;
        mc_from = !on ? 0 : !strcmp(e, "first") ? fr : strtoull(e, NULL, 0);
        if (on) rt_log("MESHCOUNT: the counts of indirect mesh draws are logged for 40 frames from frame %llu", (unsigned long long)mc_from);
    }
    if (!on || fr < mc_from || fr >= mc_from + 40) return;
    if (!mc_buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = 8192 * 64,
                                 .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &mc_buf, &mc_alloc, &info) != VK_SUCCESS) { on = 0; return; }
        mc_map = info.pMappedData;
    }
    if (mc_sub_fence != ex->fence) {   // a submission's records go to one callback at its fence
        if (mc_sub_fence && mc_next != mc_sub_lo) {
            McJob *j = malloc(sizeof *j);
            *j = (McJob){mc_sub_lo, mc_next};
            vk_completion_add(mc_sub_fence, mc_done, j);
        }
        mc_sub_fence = ex->fence, mc_sub_lo = mc_next;
    }
    uint32_t k = mc_next++ % 8192;
    mc_meta[k] = (McMeta){fr, ind, ex->prog ? ex->prog->serial : 0, ex->pass_w, ex->pass_h};
    VkBufferCopy cp = {off, (VkDeviceSize)k * 64, avail >= 64 ? 64 : 8};
    vkCmdCopyBuffer(ex->cb, src, mc_buf, 1, &cp);
    ex->recorded++;
}

// Indirect mesh draws: NVN's record is {count, first}, Vulkan's {x, y, z}, so the count is copied into a {count, 1, 1}
// of scratch memory first, outside the pass, between full barriers. One draw at a time that ended the pass and drained
// the GPU three times per draw (~60 foliage draws a frame in the forest: ~180 full barriers). The counts cannot change
// within a stretch of the command stream that writes no memory (vk_exec_op_quiet: a culling dispatch that fills them
// comes with the game's barrier, which ends the stretch), so the first indirect mesh draw of a stretch copies the counts
// of every indirect mesh draw up to the stretch's end at once and the others draw without leaving the pass.
// REKARINA_MESH_BATCH=0: one copy per draw.
typedef struct { uint64_t ind; VkBuffer b; VkDeviceSize o; } MeshInd;
static MeshInd mesh_ind[256];
static int nmesh_ind;
static uint64_t mesh_ind_window;

static int mesh_ind_find(uint64_t ind, VkBuffer *b, VkDeviceSize *o) {
    if (mesh_ind_window != vk_exec_op_window) return 0;
    for (int i = 0; i < nmesh_ind; i++)
        if (mesh_ind[i].ind == ind) { *b = mesh_ind[i].b, *o = mesh_ind[i].o; return 1; }
    return 0;
}

// The copy of one record's count into a fresh {count, 1, 1}. 0 when the record is not backed (the draw is skipped).
static int mesh_ind_copy(VkExec *ex, uint64_t ind, VkBuffer *ib, VkDeviceSize *io) {
    VkMemSpan ip;
    if (!vk_mem_resolve(ind, 8, &ip) || ip.offset % 4) return 0;
    static const uint32_t init[3] = {0, 1, 1};
    memcpy(scratch(ex, sizeof init, ib, io), init, sizeof init);
    VK_MEM_USE(ip);
    VkBufferCopy cp = {ip.offset, *io, 4};
    vkCmdCopyBuffer(ex->cb, ip.buffer, *ib, 1, &cp);
    mc_record(ex, ip.buffer, ip.offset, ind, ip.avail);   // DEBUG-MESHCOUNT
    VK_MEM_BOUND(ip);
    ex->recorded++;
    return 1;
}

// The counts of the indirect mesh draws from the current op to the end of its stretch, behind one barrier pair.
static void mesh_ind_batch(VkExec *ex) {
    nmesh_ind = 0;
    mesh_ind_window = vk_exec_op_window;
    vk_crumb(ex->cb, VK_CRUMB_MESHCNT, vk_exec_frame.mesh_batches);   // DEBUG-COPYCRUMB
    for (const op_hdr *h = vk_exec_cur_op; h && nmesh_ind < 256;) {
        if (h->op == OP_JUMP) { h = (const op_hdr *)(uintptr_t)((const p_jump *)nvn_op_payload(h))->target; continue; }
        if (!vk_exec_op_quiet(h->op)) break;
        if (h->op == OP_DRAW_MESH_IND) {
            uint64_t ind = ((const p_draw_mesh_ind *)nvn_op_payload(h))->ind;
            VkBuffer b;
            VkDeviceSize o;
            if (!mesh_ind_find(ind, &b, &o) && mesh_ind_copy(ex, ind, &b, &o)) mesh_ind[nmesh_ind++] = (MeshInd){ind, b, o};
        }
        h = nvn_op_next(h);
    }
    vk_crumb(ex->cb, VK_CRUMB_MESHCNT_END, vk_exec_frame.mesh_batches);   // DEBUG-COPYCRUMB
    vk_exec_frame.mesh_batches++;
}

// ---- mesh draws on a GPU without VK_EXT_mesh_shader
// The mesh shader runs as a compute kernel (GFX_STAGE_MSC, tools/xlate MeshStageWrapper.GenerateAsCompute), one
// workgroup a meshlet, with the mesh stage's resources; it writes the meshlets' vertices and triangles into the
// emulation buffer (vk.h VK_MESH_EMU_*) and the indexed draw's arguments into its head, and an indexed indirect draw
// reads them back through gfx_shader_meshc_vs and the program's fragment shader. One buffer for every such draw, so
// each one ends the render pass, waits for the previous draw's reads and starts the pass again: slow, but the foliage
// is there. Not yet: draws with a task shader (the short grass), point and line meshes, per-primitive attributes, the
// draws' GPU render conditions (drawn always). REKARINA_MESH_EMULATE=auto (on where the device has no mesh shaders and has
// buffer device addresses), 1 (also where it has them: the check against the real mesh path), 0 off.
static VkBuffer emu_buf;
static VmaAllocation emu_alloc;
static uint64_t emu_addr;
static int emu_failed;
static _Thread_local const char *emu_step;   // the step of an emulated draw this thread records (the crash record)

uint64_t vk_mesh_emu_address(void) { return __atomic_load_n(&emu_addr, __ATOMIC_ACQUIRE); }
const char *vk_exec_crash_step(void) { return emu_step; }

// The kernels take the emulation buffer's address as a run-time value (translator patch 0047): pushed before each
// dispatch into their layout's push constants (vk_pipeline.c layout_get, Bindings.emu). The two constants the runtime
// patched into the module before came out with the high word lost on AMD's driver for RDNA 1 and 2: the mesh kernel's
// first read of a task's launch count went to the buffer + 0x1000 with the high word all ones, and the device was lost
// at the first emulated frame (an RX 5700 XT, the integrated Radeon with REKARINA_MESH_EMULATE=1; 2026-11-21).
static void emu_push_address(VkExec *ex, const VkPipeInfo *pi) {
    uint64_t a = __atomic_load_n(&emu_addr, __ATOMIC_ACQUIRE);
    uint32_t w[2] = {(uint32_t)a, (uint32_t)(a >> 32)};
    vkCmdPushConstants(ex->cb, pi->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, VK_MESH_EMU_PUSH_BYTES, w);
}

static int mesh_emu_wanted(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("REKARINA_MESH_EMULATE");
        int force = e && *e == '1', off = e && *e == '0';
        on = !off && vk_ctx()->has_buffer_device_address && (force || !vk_pipeline_mesh_on());
        if (on) rt_log("vk_exec: mesh draws run emulated: the mesh shader as compute, the meshlets drawn as indexed triangles%s",
                       force ? " (REKARINA_MESH_EMULATE=1)" : " (the device has no mesh shaders)");
    }
    return on;
}

int vk_exec_mesh_emulated(void) { return mesh_emu_wanted(); }

static int mesh_emu_buffer(void) {
    if (emu_addr || emu_failed) return emu_addr != 0;
    VkCtx *c = vk_ctx();
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = VK_MESH_EMU_BYTES,
                             .usage = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT |
                                      VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    PFN_vkGetBufferDeviceAddress bda = (PFN_vkGetBufferDeviceAddress)vkGetDeviceProcAddr(c->device, "vkGetBufferDeviceAddress");
    if (!bda || vmaCreateBuffer(c->vma, &bi, &ai, &emu_buf, &emu_alloc, NULL) != VK_SUCCESS) {
        emu_failed = 1;
        rt_log("vk_exec: the mesh emulation buffer (%llu MiB) could not be made: mesh draws are skipped",
               (unsigned long long)(VK_MESH_EMU_BYTES >> 20));
        return 0;
    }
    VkBufferDeviceAddressInfo ii = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = emu_buf};
    __atomic_store_n(&emu_addr, bda(c->device, &ii), __ATOMIC_RELEASE);
    rt_log("vk_exec: mesh emulation buffer: %llu MiB at device address %#llx", (unsigned long long)(VK_MESH_EMU_BYTES >> 20),
           (unsigned long long)emu_addr);
    return emu_addr != 0;
}

static void mesh_emulate(VkExec *ex, const p_draw_mesh *d, const p_draw_mesh_ind *di) {
    if (ex->render_disabled || !ex->prog) return;
    const uint8_t *ctl, *code;
    uint32_t len;
    int task = find_stage(ex->prog, NVN_STAGE_TS, &ctl, &code, &len);
    {   // REKARINA_MESH_EMULATE_TASKS=0: draws with a task shader are skipped (as before 2026-11-15)
        static int tasks = -1;
        if (tasks < 0) tasks = rt_knob("MESH_EMULATE_TASKS", "1");
        if (task && !tasks) return vk_exec_oq_draw_lost(ex);
    }
    if (!di && !d->count) return;
    if (!mesh_emu_buffer()) return vk_exec_oq_draw_lost(ex);
    int present = 0, pending = 0;
    sf_msc = 1;
    const GfxShader *msc = shader_for(ex, NVN_STAGE_MS, &present, &pending, 0);
    int tpending = 0;
    const GfxShader *tsc = task ? shader_for(ex, NVN_STAGE_TS, &present, &tpending, 0) : NULL;
    sf_msc = 0;
    if (pending || tpending || !msc || (task && !tsc)) return vk_exec_oq_draw_lost(ex);
    uint32_t vec4s, max_meshlets;
    const uint8_t *locs;
    gfx_shader_meshc(msc, &vec4s, &max_meshlets, &locs);
    {   // DEBUG-MESHEMU: REKARINA_MESH_EMU_LOG=<frame>: the emulated draws of that frame (first: the first 64 of the run);
        // REKARINA_MESH_EMU_SKIP=<key prefix>[,...]
        static int64_t logf = -2;
        static int nfirst;
        static const char *skip;
        if (logf == -2) {
            const char *e = cfg_get("REKARINA_MESH_EMU_LOG");
            logf = e && !strcmp(e, "first") ? -3 : e && *e ? atoll(e) : -1;
            skip = cfg_get("REKARINA_MESH_EMU_SKIP");
        }
        char hx[17], thx[17] = "-";
        const uint8_t *k8 = gfx_shader_key(msc);
        for (int i = 0; i < 8; i++) snprintf(hx + 2 * i, 3, "%02x", k8[i]);
        if ((logf >= 0 && __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) == (uint64_t)logf) ||
            (logf == -3 && nfirst++ < 64)) {
            for (int i = 0; tsc && i < 8; i++) snprintf(thx + 2 * i, 3, "%02x", gfx_shader_key(tsc)[i]);
            rt_log("MESHEMU msc %s tsc %s vec4s %u locs %u max %u%s count %d first %d frame %llu", hx, thx, vec4s, vec4s - 1,
                   max_meshlets, di ? " indirect" : "", di ? -1 : (int)d->count, di ? -1 : (int)d->first,
                   (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));
        }
        if (skip && strstr(skip, hx)) return;
        static const char *save;   // REKARINA_MESH_EMU_SAVE=<key prefix>:<dir>: that shader's kernel and its mesh stage, once
        static int saved;
        if (!save) save = cfg_get("REKARINA_MESH_EMU_SAVE") ? cfg_get("REKARINA_MESH_EMU_SAVE") : "";
        const char *colon = strchr(save, ':');
        if (colon && !saved && !strncmp(hx, save, (size_t)(colon - save))) {
            saved = 1;
            int pr2 = 0, pe2 = 0;
            const GfxShader *ms = shader_for(ex, NVN_STAGE_MS, &pr2, &pe2, 0);
            const GfxShader *two[2] = {msc, ms};
            for (int i = 0; i < 2; i++) {
                if (!two[i]) continue;
                uint64_t nw;
                const uint32_t *code = gfx_shader_spirv(two[i], &nw);
                char path[600];
                snprintf(path, sizeof path, "%s/%s_%s.spv", colon + 1, i ? "ms" : "msc", hx);
                FILE *f = fopen(path, "wb");
                if (f) fwrite(code, 4, nw, f), fclose(f);
            }
            rt_log("MESHEMU saved %s (mesh stage %s)", hx, ms ? "too" : "missing");
        }
    }
    const GfxShader *vs = gfx_shader_meshc_vs(msc);
    if (!vs) return vk_exec_oq_draw_lost(ex);
    const VkPipeInfo *info = NULL;
    VkPipeline cp = vk_pipeline_compute(msc, &info);
    if (vk_pipeline_compute_pending()) return vk_exec_oq_draw_lost(ex);
    if (!cp || !info)
        return vk_exec_oq_draw_lost(ex), warn_once(W_PIPELINE, "an emulated mesh shader's compute pipeline could not be built: skipped");
    const VkPipeInfo *tinfo = NULL;
    VkPipeline tp = task ? vk_pipeline_compute(tsc, &tinfo) : VK_NULL_HANDLE;
    if (task && vk_pipeline_compute_pending()) return vk_exec_oq_draw_lost(ex);
    if (task && (!tp || !tinfo))
        return vk_exec_oq_draw_lost(ex), warn_once(W_PIPELINE, "an emulated task shader's compute pipeline could not be built: skipped");
    VkBuffer ib = VK_NULL_HANDLE;
    VkDeviceSize io = 0;
    if (di) {   // the {count, 1, 1} copies of the stretch's indirect mesh draws, as for the real draws below
        if (!mesh_ind_find(di->ind, &ib, &io)) {
            vk_exec_pass_end(ex);
            cb_res(ex);
            vk_exec_barrier(ex);
            mesh_ind_batch(ex);
            vk_exec_barrier(ex);
        }
        if (!mesh_ind_find(di->ind, &ib, &io))
            return warn_once(W_INDIRECT, "mesh draw arguments at %#llx are not backed or not aligned: skipped", (unsigned long long)di->ind);
        c0_mesh_first = 0;
    }
    cb_res(ex);
    dummies_init();
    vk_dlaa_pjitter(ex, NVN_STAGE_MS);
    // A direct draw of more workgroups than the buffer holds meshlets would lose the rest (the kernel writes nothing
    // past max_meshlets and draws only those): it runs in pieces of at most max_meshlets workgroups, each piece's
    // first workgroup moved on through c0 (VK_C0_MESH_FIRST, which the kernel adds to its workgroup number as the
    // mesh stage does) and drawn before the next piece's dispatch writes the buffer again. An indirect count is the
    // GPU's and the meshlets under a task are taken by a counter: those run as one piece.
    uint32_t total = di || task ? 1 : d->count, piece = di || task || !max_meshlets ? total : max_meshlets;
    if (total > piece) {
        static uint8_t said;
        if (!__atomic_exchange_n(&said, 1, __ATOMIC_RELAXED))
            rt_log("vk_exec: an emulated mesh draw of %u workgroups runs in %u pieces (the buffer holds %u meshlets of "
                   "%u vec4s a vertex)", total, (total + piece - 1) / piece, max_meshlets, vec4s);
    }
    for (uint32_t done = 0; done < total; done += piece) {
        uint32_t n = total - done < piece ? total - done : piece;
        if (!di) c0_mesh_first = d->first + done;
        emu_step = "end the pass";
        vk_exec_dispatch_barrier(ex);   // ends the pass; the previous emulated draw's reads and earlier writes are done
        if (di || task) {
            emu_step = "clear the arguments";
            // An indirect count of 0 (the game's own culling hid the object) runs no workgroup, so nothing writes this draw's
            // arguments and the draw took the previous emulated draw's, with that draw's vertices read at this one's
            // stride: slivers across the scene. The arguments start at no triangles; workgroup 0 writes them when it runs.
            // Under a task the mesh kernel's dispatch arguments and the meshlet counter start at zero too (64 bytes).
            vkCmdFillBuffer(ex->cb, emu_buf, 0, task ? 64 : sizeof(VkDrawIndexedIndirectCommand), 0);
            VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                                   .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                   .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
                                   .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT};
            VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
            vkCmdPipelineBarrier2(ex->cb, &dep);
            ex->recorded++;
        }
        if (task) {   // the task kernel keeps the payloads and launch counts, and the mesh kernel's arguments
            emu_step = "the task kernel";
            bind_pipeline(ex, VK_PIPELINE_BIND_POINT_COMPUTE, tp);
            twin_dispatch = 1;
            bind_resources(ex, tinfo);
            twin_dispatch = 0;
            emu_push_address(ex, tinfo);
            if (di) vkCmdDispatchIndirect(ex->cb, ib, io);
            else vkCmdDispatch(ex->cb, d->count, 1, 1);
            ex->dispatch_end_at = ++ex->recorded;
            vk_exec_barrier(ex);
        }
        emu_step = "the mesh kernel";
        bind_pipeline(ex, VK_PIPELINE_BIND_POINT_COMPUTE, cp);
        twin_dispatch = 1;
        bind_resources(ex, info);
        twin_dispatch = 0;
        emu_push_address(ex, info);
        if (task) vkCmdDispatchIndirect(ex->cb, emu_buf, VK_MESH_EMU_MESH_ARGS_OFFSET);
        else if (di) vkCmdDispatchIndirect(ex->cb, ib, io);
        else vkCmdDispatch(ex->cb, n, 1, 1);
        ex->dispatch_end_at = ++ex->recorded;
        vk_exec_barrier(ex);   // the meshlets are written before the draw reads them as vertices, indices and arguments
        emu_step = "the draw's pipeline and resources";
        prep_emu_vs = vs;
        int ok = prepare_stages(ex, 2, NVN_DRAW_PRIMITIVE_TRIANGLES, 0);
        prep_emu_vs = NULL;
        if (!ok) {
            emu_step = NULL;
            return;
        }
        emu_step = "the draw";
        VkDeviceSize vo = VK_MESH_EMU_VERTEX_OFFSET;
        vkCmdBindVertexBuffers2(ex->cb, 0, 1, &emu_buf, &vo, NULL, NULL);   // the recorder wraps this one (the stride is static)
        vkCmdBindIndexBuffer(ex->cb, emu_buf, VK_MESH_EMU_INDEX_OFFSET, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexedIndirect(ex->cb, emu_buf, 0, 1, sizeof(VkDrawIndexedIndirectCommand));
        hdr_output_draw_written(ex);
        ex->recorded++;
        B.vb_valid = 0, ibs_buf = VK_NULL_HANDLE;   // the next draw binds its own vertex and index buffers
        vk_pipeline_dynamic_forget();
    }
    emu_step = NULL;
    if (!vk_par_worker) vk_exec_frame.draws++;
    {
        static uint8_t said;
        if (!__atomic_exchange_n(&said, 1, __ATOMIC_RELAXED))
            rt_log("vk_exec: first emulated mesh draw: %u vec4s a vertex, at most %u meshlets a draw", vec4s, max_meshlets);
    }
}

static void op_draw_mesh(VkExec *ex, const p_draw_mesh *d, const p_draw_mesh_ind *di) {
    {   // DEBUG-SSBOLOG: the mesh draws of that frame, with the mesh stage's slots 0-1
        static int on = -1;
        static uint64_t from;
        if (on < 0) { const char *e = cfg_get("REKARINA_SSBOLOG"); on = e && *e; from = on ? strtoull(e, NULL, 0) : 0; }
        if (on && __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) == from)
            rt_log("SSBO mesh draw stages %#x ind %#llx ms0 %#llx+%#llx ms1 %#llx+%#llx ts0 %#llx", (unsigned)ex->prog_stages, di ? (unsigned long long)di->ind : 0ull,
                   (unsigned long long)ex->ssbo[NVN_STAGE_MS][0].addr, (unsigned long long)ex->ssbo[NVN_STAGE_MS][0].size,
                   (unsigned long long)ex->ssbo[NVN_STAGE_MS][1].addr, (unsigned long long)ex->ssbo[NVN_STAGE_MS][1].size,
                   (unsigned long long)ex->ssbo[NVN_STAGE_TS][0].addr);
    }
    if (mesh_emu_wanted()) return mesh_emulate(ex, d, di);
    if (!vk_pipeline_mesh_on())
        return warn_once(W_PIPELINE, "a mesh draw on a GPU without VK_EXT_mesh_shader task and mesh shaders: skipped");
    VkBuffer ib = VK_NULL_HANDLE;
    VkDeviceSize io = 0;
    static int batch = -1;
    if (batch < 0) batch = rt_knob("MESH_BATCH", "1");
    if (di && batch) {
        if (!mesh_ind_find(di->ind, &ib, &io)) {
            vk_exec_pass_end(ex);
            cb_res(ex);
            vk_exec_barrier(ex);   // earlier writes of the records are visible to the copies
            mesh_ind_batch(ex);
            vk_exec_barrier(ex);   // the counts are visible to the draws' indirect reads
        }
        if (!mesh_ind_find(di->ind, &ib, &io))
            return warn_once(W_INDIRECT, "mesh draw arguments at %#llx are not backed or not aligned: skipped", (unsigned long long)di->ind);
        c0_mesh_first = 0;
    } else if (di) {
        VkMemSpan ip;
        if (!vk_mem_resolve(di->ind, 8, &ip) || ip.offset % 4)
            return warn_once(W_INDIRECT, "mesh draw arguments at %#llx are not backed or not aligned: skipped", (unsigned long long)di->ind);
        static uint8_t once;
        if (!__atomic_exchange_n(&once, 1, __ATOMIC_RELAXED))
            rt_log("vk_exec: DrawMeshTasksIndirect: the record is read as {count, first} and first taken as 0 (unverified)");
        vk_exec_pass_end(ex);
        cb_res(ex);
        static const uint32_t init[3] = {0, 1, 1};
        memcpy(scratch(ex, sizeof init, &ib, &io), init, sizeof init);
        vk_exec_barrier(ex);   // earlier writes of the record are visible to the copy
        VK_MEM_USE(ip);
        {   // DEBUG-MESHLOG: REKARINA_MESHLOG=<frame>: from that frame, 400 records: the record, the buffer the copy reads it from and the pool table's entry for it
            static int on = -1, nl;
            static uint64_t from;
            if (on < 0) { const char *e = cfg_get("REKARINA_MESHLOG"); on = e && *e; from = on ? strtoull(e, NULL, 0) : 0; }
            uint64_t fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
            if (on && fr >= from && nl < 400) {
                nl++;
                const GmemSnap *s = GM.cur;
                int ent = -1;
                for (int k = 1; s && k < s->n; k++)
                    if (di->ind >= s->e[k].lo && di->ind < s->e[k].hi) { ent = k; break; }
                VkMemSpan raw;
                int rr = vk_mem_resolve(di->ind, 8, &raw);
                const uint32_t *g = rr && !raw.alloc && !nvn_gpu_only(di->ind) ? (const uint32_t *)(uintptr_t)di->ind : NULL;
                rt_log("MESH frame %llu ind %#llx copy-from %p+%#llx (import %p, %s) table entry %d %s guest {%u %u}",
                       (unsigned long long)fr, (unsigned long long)di->ind, (void *)ip.buffer, (unsigned long long)ip.offset,
                       (void *)raw.buffer, ip.buffer == raw.buffer ? "import" : "mirror", ent,
                       ent < 0 ? "NONE" : s->e[ent].mirror ? "mirror" : s->e[ent].cpu ? "import" : "device", g ? g[0] : 0,
                       g ? g[1] : 0);
            }
        }
        VkBufferCopy cp = {ip.offset, io, 4};
        vkCmdCopyBuffer(ex->cb, ip.buffer, ib, 1, &cp);
        mc_record(ex, ip.buffer, ip.offset, di->ind, ip.avail);   // DEBUG-MESHCOUNT
        VK_MEM_BOUND(ip);
        ex->recorded++;
        vk_exec_barrier(ex);   // the count is visible to the draw's indirect read
        c0_mesh_first = 0;
    } else {
        c0_mesh_first = d->first;
    }
    int big_ok = prepare_stages(ex, 1, NVN_DRAW_PRIMITIVE_TRIANGLES, 0);
    {   // DEBUG-BIGLOG: mesh draws into a depth target 8192 or more wide (OoT's static shadow map) and whether they ran
        static int on = -1, n;
        if (on < 0) on = cfg_get("REKARINA_BIGLOG") != NULL;
        if (on && ex->rt.depth && ex->rt.depth->d.width >= 8192 && n < 60) {
            int hp;
            const GfxShader *msh = shader_for(ex, NVN_STAGE_MS, &hp, NULL, 0), *fsh = shader_for(ex, NVN_STAGE_FS, &hp, NULL, 0);
            uint64_t hm = 0xcbf29ce484222325ull, hf = 0xcbf29ce484222325ull, w = 0;
            const uint8_t *q = msh ? (const uint8_t *)gfx_shader_spirv(msh, &w) : NULL;
            for (uint64_t i = 0; q && i < w * 4; i++) hm = (hm ^ q[i]) * 0x100000001b3ull;
            q = fsh ? (const uint8_t *)gfx_shader_spirv(fsh, &w) : NULL;
            for (uint64_t i = 0; q && i < w * 4; i++) hf = (hf ^ q[i]) * 0x100000001b3ull;
            rt_log("BIGLOG mesh shaders ms %016llx fs %016llx", (unsigned long long)hm, (unsigned long long)hf);
        }
        if (on && ex->rt.depth && ex->rt.depth->d.width >= 8192 && n++ < 60)
            rt_log("BIGLOG mesh draw into %ux%u: %s count %d first %d prog %llu, recorded %d, frame %llu", ex->rt.depth->d.width,
                   ex->rt.depth->d.height, di ? "indirect" : "direct", di ? -1 : (int)d->count, di ? -1 : (int)d->first,
                   (unsigned long long)(ex->prog ? ex->prog->serial : 0), big_ok,
                   (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED));
    }
    if (!big_ok) return;
    if (!vk_par_worker) vk_exec_frame.draws++;   // a diagnostic count: not written by every chunk's thread
    dt_stamp(ex, 0);   // DEBUG-DRAWTIME
    int cond = !task_draw || mesh_task_cond();
    if (cond) GPU_COND_BEGIN(ex);
    if (di) vk_pipeline_draw_mesh_indirect(ex->cb, ib, io);
    else vk_pipeline_draw_mesh(ex->cb, d->count);
    if (cond) GPU_COND_END(ex);
    if (di || d->count > 0) hdr_output_draw_written(ex);
    ex->recorded++;
}
#endif

// DEBUG-PEEK: REKARINA_PEEK=<frame>: after every dispatch of that frame, the first 64 bytes of each storage buffer bound
// to the compute stage (and an indirect dispatch's group counts) are copied on the GPU and logged at the fence, in
// order: "PEEK f <frame> d <n> caller <addr> ind x y z | s<slot> <addr>: 16 words".
#define PK_SLOTS 13                      // 12 storage slots and the indirect record
#define PK_N 1024                        // dispatches logged at most
typedef struct { uint64_t frame, caller, addr[PK_SLOTS]; uint32_t n; } PkMeta;
static VkBuffer pk_buf;
static VmaAllocation pk_alloc;
static uint32_t *pk_map;
static PkMeta pk_meta[PK_N];
static uint32_t pk_next;
static void pk_done(void *arg) {
    PkMeta *m = arg;
    char line[2048];
    int o = snprintf(line, sizeof line, "PEEK f %llu d %u caller %#llx", (unsigned long long)m->frame, m->n,
                     (unsigned long long)m->caller);
    const uint32_t *w = pk_map + (size_t)m->n * PK_SLOTS * 16;
    if (m->addr[12]) o += snprintf(line + o, sizeof line - (size_t)o, " ind %u %u %u", w[12 * 16], w[12 * 16 + 1], w[12 * 16 + 2]);
    rt_log("%s", line);
    for (int s = 0; s < 12; s++) {
        if (!m->addr[s]) continue;
        o = snprintf(line, sizeof line, "PEEK   s%d %#llx:", s, (unsigned long long)m->addr[s]);
        for (int k = 0; k < 16; k++) o += snprintf(line + o, sizeof line - (size_t)o, " %x", w[s * 16 + k]);
        rt_log("%s", line);
    }
}
void pk_dump_done(void *arg) {   // DEBUG-PEEK: REKARINA_PEEK_DUMP's file, in REKARINA_PROFILE_DIR's parent or the working directory
    struct { VkBuffer b; VmaAllocation al; void *map; int n, slot, bytes; } *pd = arg;
    char path[600];
    const char *dir = cfg_env("REKARINA_PEEK_DIR");
    snprintf(path, sizeof path, "%s/peek_d%d_s%d.bin", dir && *dir ? dir : ".", pd->n, pd->slot);
    FILE *f = fopen(path, "wb");
    if (f) fwrite(pd->map, 1, (size_t)pd->bytes, f), fclose(f);
    rt_log("PEEK dump %s (%d bytes)", path, pd->bytes);
    vmaDestroyBuffer(vk_ctx()->vma, pd->b, pd->al);
    free(pd);
}
static void pk_record(VkExec *ex, uint64_t ind, uint64_t caller) {
    static int on = -1;
    static uint64_t from;
    if (on < 0) {
        const char *e = cfg_get("REKARINA_PEEK");
        on = e && *e;
        from = on ? strtoull(e, NULL, 0) : 0;
    }
    uint64_t fr = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    if (!on || fr != from || pk_next >= PK_N) return;
    if (!pk_buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)PK_N * PK_SLOTS * 64,
                                 .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &pk_buf, &pk_alloc, &info) != VK_SUCCESS) { on = 0; return; }
        pk_map = info.pMappedData;
    }
    uint32_t n = pk_next++;
    PkMeta *m = &pk_meta[n];
    *m = (PkMeta){fr, caller, {0}, n};
    vk_exec_barrier(ex);   // the dispatch's writes are visible to the copies
    for (int s = 0; s < PK_SLOTS; s++) {
        uint64_t a = s < 12 ? ex->ssbo[NVN_STAGE_CS][s].addr : ind;
        VkMemSpan sp;
        if (!a || !vk_mem_resolve(a, 64, &sp) || sp.avail < 64) continue;
        m->addr[s] = a;
        VkBufferCopy cp = {sp.offset, ((VkDeviceSize)n * PK_SLOTS + (VkDeviceSize)s) * 64, 64};
        vkCmdCopyBuffer(ex->cb, sp.buffer, pk_buf, 1, &cp);
        ex->recorded++;
    }
    vk_completion_add(ex->fence, pk_done, m);
    {   // REKARINA_PEEK_ADDR=<addr>[:<bytes>][,...]: those GPU ranges after dispatch 0 too, as peek_a<addr>.bin
        static int na = -1;
        static uint64_t aa[16], an[16];
        if (na < 0) {
            na = 0;
            const char *e = cfg_env("REKARINA_PEEK_ADDR");
            while (e && *e && na < 16) {
                char *q;
                aa[na] = strtoull(e, &q, 16);
                an[na] = *q == ':' ? strtoull(q + 1, &q, 0) : 64;
                na++;
                e = q;
                while (*e == ',') e++;
            }
        }
        for (int i = 0; n == 0 && i < na; i++) {
            VkMemSpan sp;
            if (!vk_mem_resolve(aa[i], an[i], &sp) || sp.avail < an[i]) { rt_log("PEEK addr %#llx not backed", (unsigned long long)aa[i]); continue; }
            typedef struct { VkBuffer b; VmaAllocation al; void *map; int n, slot, bytes; } PkDump;
            PkDump *pd = calloc(1, sizeof *pd);
            VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = an[i], .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
            VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                          .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
            VmaAllocationInfo info;
            if (pd && vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &pd->b, &pd->al, &info) == VK_SUCCESS) {
                pd->map = info.pMappedData, pd->n = 1000 + i, pd->slot = 0, pd->bytes = (int)an[i];
                VkBufferCopy cp = {sp.offset, 0, an[i]};
                vkCmdCopyBuffer(ex->cb, sp.buffer, pd->b, 1, &cp);
                ex->recorded++;
                void pk_dump_done(void *arg);
                vk_completion_add(ex->fence, pk_dump_done, pd);
            }
        }
    }
    {   // REKARINA_PEEK_DUMP=<dispatch>:<slot>:<bytes>: that dispatch's storage slot written to peek_d<n>_s<slot>.bin at the fence
        static int dd = -2, ds, dn;
        if (dd == -2) {
            const char *e = cfg_env("REKARINA_PEEK_DUMP");
            char *q;
            dd = e && *e ? (int)strtol(e, &q, 0) : -1;
            if (dd >= 0) ds = *q == ':' ? (int)strtol(q + 1, &q, 0) : 0, dn = *q == ':' ? (int)strtol(q + 1, &q, 0) : 4096;
        }
        uint64_t a = dd == (int)n && ds >= 0 && ds < 16 ? ex->ssbo[NVN_STAGE_CS][ds].addr : 0;
        VkMemSpan sp;
        if (a && dn > 0 && vk_mem_resolve(a, (uint64_t)dn, &sp) && sp.avail >= (uint64_t)dn) {
            typedef struct { VkBuffer b; VmaAllocation al; void *map; int n, slot, bytes; } PkDump;
            PkDump *pd = calloc(1, sizeof *pd);
            VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = (VkDeviceSize)dn,
                                     .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
            VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                          .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
            VmaAllocationInfo info;
            if (pd && vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &pd->b, &pd->al, &info) == VK_SUCCESS) {
                pd->map = info.pMappedData, pd->n = (int)n, pd->slot = ds, pd->bytes = dn;
                VkBufferCopy cp = {sp.offset, 0, (VkDeviceSize)dn};
                vkCmdCopyBuffer(ex->cb, sp.buffer, pd->b, 1, &cp);
                ex->recorded++;
                void pk_dump_done(void *arg);
                vk_completion_add(ex->fence, pk_dump_done, pd);
            }
        }
    }
    // REKARINA_PEEK_CPU=<slot>:<byte offset>:<words>[:<per line>]: also that range of a CPU-visible storage buffer, read now
    static int cs = -2, co, cn, cl;
    if (cs == -2) {
        const char *e = cfg_get("REKARINA_PEEK_CPU");
        cs = -1;
        if (e && *e) {
            char *q;
            cs = (int)strtol(e, &q, 0);
            co = *q == ':' ? (int)strtol(q + 1, &q, 0) : 0;
            cn = *q == ':' ? (int)strtol(q + 1, &q, 0) : 16;
            cl = *q == ':' ? (int)strtol(q + 1, &q, 0) : 16;
            if (cl <= 0) cl = 16;
        }
    }
    {   // and the compute stage's c3 (the first 128 vec4) when it is CPU-visible
        const p_bind_buf *u = &ex->ubo[NVN_STAGE_CS][0];
        const char *sc = cfg_env("REKARINA_C3_SCALE");   // DEBUG: <first vec4>:<count>:<factor>, applied once per buffer
        static uint64_t scaled[64];
        static int nscaled;
        if (sc && u->addr && !nvn_gpu_only(u->addr)) {
            char *q;
            int v0 = (int)strtol(sc, &q, 0), vn = *q == ':' ? (int)strtol(q + 1, &q, 0) : 1;
            float fac = *q == ':' ? strtof(q + 1, NULL) : 1.0f;
            int seen = 0;
            for (int i = 0; i < nscaled; i++) seen |= scaled[i] == u->addr;
            if (!seen && nscaled < 64) {
                scaled[nscaled++] = u->addr;
                float *f = (float *)(uintptr_t)u->addr;
                if (u->size < (uint64_t)16 * (uint64_t)(v0 + vn)) vn = 0;   // not this buffer
                for (int i = 4 * v0; i < 4 * (v0 + vn); i++) f[i] *= fac;
                rt_log("C3_SCALE: c3 at %#llx vec4 %d..%d x %g", (unsigned long long)u->addr, v0, v0 + vn - 1, fac);
            }
        }
        VkMemSpan ur;
        const char *uk = cfg_env("REKARINA_PEEK_C3");   // =1: c3; =<n> (3 or more): c<n>
        if (uk && atoi(uk) > 3) u = &ex->ubo[NVN_STAGE_CS][atoi(uk) - 3];
        if (uk && u->addr && !nvn_gpu_only(u->addr) && vk_mem_resolve(u->addr, 16, &ur) && !ur.alloc) {
            uint64_t nv = u->size / 16 < (atoi(uk) > 3 ? 4096u : 128u) ? u->size / 16 : (atoi(uk) > 3 ? 4096u : 128u);
            const float *f = (const float *)(uintptr_t)u->addr;
            for (uint64_t i = 0; i < nv; i += 4) {
                char line[512];
                int o = snprintf(line, sizeof line, "PEEKC3 d %u [%llu]", n, (unsigned long long)i);
                const uint32_t *x = (const uint32_t *)f;
                for (uint64_t k = i; k < i + 4 && k < nv; k++)
                    o += atoi(uk) > 3 ? snprintf(line + o, sizeof line - (size_t)o, "  %08x %08x %08x %08x", x[4 * k], x[4 * k + 1],
                                                 x[4 * k + 2], x[4 * k + 3])
                                      : snprintf(line + o, sizeof line - (size_t)o, "  %.4g %.4g %.4g %.4g", f[4 * k], f[4 * k + 1],
                                                 f[4 * k + 2], f[4 * k + 3]);
                rt_log("%s", line);
            }
        }
    }
    uint64_t a = cs >= 0 && cs < 16 ? ex->ssbo[NVN_STAGE_CS][cs].addr : 0;
    VkMemSpan raw;
    if (a && !nvn_gpu_only(a) && vk_mem_resolve(a + (uint64_t)co, (uint64_t)cn * 4, &raw) && !raw.alloc) {
        const float *f = (const float *)(uintptr_t)(a + (uint64_t)co);
        char line[4096];
        for (int i = 0; i < cn; i += cl) {
            int o = snprintf(line, sizeof line, "PEEKCPU d %u +%#x:", n, co + i * 4);
            for (int k = i; k < i + cl && k < cn && o < (int)sizeof line - 32; k++) o += snprintf(line + o, sizeof line - (size_t)o, " %.4g", f[k]);
            rt_log("%s", line);
        }
    }
}

// ---- dispatch and uniform buffer updates

#ifdef GAME_NS2
// The compute program of a dispatch is the one bound with the compute bit (VkExec.cprog), else the program bound last.
#define DISPATCH_PROGRAM(ex) const prog_t *gprog_ = (ex)->prog; if ((ex)->cprog) (ex)->prog = (ex)->cprog
#ifdef GAME_OOT
#define AA_PROBE_SIDE 24
#define AA_PROBE_TEXELS (AA_PROBE_SIDE * AA_PROBE_SIDE)
typedef struct {
    VkBuffer buffer;
    VmaAllocation allocation;
    void *mapped;
    VkGuestPq pq;
    uint64_t frame;
    int synthetic;
} AaPqProbe;

static const float aa_probe_rgb[4] = {-0.29382057f, 0.56644995f, -0.05028945f, 1.0f};
static uint64_t aa_probe_injected;

static int scene_aa_probe_mode(void) {
    static int mode = -1;
    if (mode < 0) {
        const char *setting = cfg_get("REKARINA_HDR_AA_PROBE");
        mode = setting ? atoi(setting) : 0;
    }
    return mode;
}

static void scene_aa_probe_done(void *arg) {
    AaPqProbe *probe = arg;
    VmaAllocator vma = vk_ctx()->vma;
    vmaInvalidateAllocation(vma, probe->allocation, 0, VK_WHOLE_SIZE);
    const uint16_t *linear = probe->mapped;
    const uint32_t *pq = (const uint32_t *)(linear + 4 * AA_PROBE_TEXELS);
    unsigned signed_pixels = 0, valid = 0, mismatches = 0;
    double worst = 0, green_leak = 0, green_min = INFINITY, green_max = 0;
    for (unsigned i = 0; i < AA_PROBE_TEXELS; ++i) {
        float rgb[3];
        for (unsigned j = 0; j < 3; ++j) {
            uint16_t bits = linear[4 * i + j];
            _Float16 half;
            memcpy(&half, &bits, sizeof half);
            rgb[j] = half;
        }
        if (!isfinite(rgb[0]) || !isfinite(rgb[1]) || !isfinite(rgb[2])) continue;
        int is_signed = rgb[0] < 0 || rgb[1] < 0 || rgb[2] < 0;
        signed_pixels += is_signed;
        valid++;
        if (probe->synthetic) {
            double r = 0.627403896 * rgb[0] + 0.329283038 * rgb[1] + 0.043313066 * rgb[2];
            double g = 0.069097289 * rgb[0] + 0.919540396 * rgb[1] + 0.011362315 * rgb[2];
            double b = 0.016391439 * rgb[0] + 0.088013308 * rgb[1] + 0.895595253 * rgb[2];
            green_leak = fmax(green_leak, fmax(fabs(r), fabs(b)) / fmax(g, 1e-8));
            green_min = fmin(green_min, g);
            green_max = fmax(green_max, g);
        }
        double error = 0;
        for (unsigned j = 0; j < 3; ++j) {
            const float *r = probe->pq.row[j];
            double y = pow(fabs((double)r[0] * rgb[0] + (double)r[1] * rgb[1] + (double)r[2] * rgb[2]) * 0.0001,
                           2610.0 / 16384.0);
            double expected = pow((3424.0 / 4096.0 + 2413.0 / 128.0 * y) / (1.0 + 2392.0 / 128.0 * y), 2523.0 / 32.0);
            double actual = (pq[i] >> (10 * j) & 1023) / 1023.0;
            error = fmax(error, fabs(expected - actual));
        }
        if (error > 2.0 / 1023.0) mismatches++;
        worst = fmax(worst, error);
    }
    rt_log("hdr: AA PQ probe frame %llu: %u finite texels, %u signed, %u mismatches, max PQ error %.6f",
           (unsigned long long)probe->frame, valid, signed_pixels, mismatches, worst);
    if (probe->synthetic) {
        const int ok = valid == AA_PROBE_TEXELS && signed_pixels == valid &&
                       green_min >= 0.49 && green_max <= 0.51 && green_leak <= 0.01 && !mismatches;
        rt_log("hdr: AA synthetic BT.2020 green %s: %.6f..%.6f, max relative red/blue leakage %.6f",
               ok ? "PASS" : "FAIL", green_min, green_max, green_leak);
    }
    vmaDestroyBuffer(vma, probe->buffer, probe->allocation);
    free(probe);
}

static void scene_aa_probe(VkExec *ex) {
    VkImage source = ex->hdr_aa.probe_source;
    ex->hdr_aa.probe_source = VK_NULL_HANDLE;
    const VkExecTarget *dst = &ex->color[0];
    if (!source || !ex->output_pq.valid || !dst->img || dst->format != VK_FORMAT_A2B10G10R10_UNORM_PACK32 ||
        !(dst->img->usage & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) || dst->width < AA_PROBE_SIDE || dst->height < AA_PROBE_SIDE) return;
    static unsigned samples;
    int mode = scene_aa_probe_mode();
    if (!mode || !ex->hdr_aa.active || samples >= 4 ||
        (mode == 2 ? aa_probe_injected != ex->hdr_aa.frame : ex->hdr_aa.frame % 16)) return;
    AaPqProbe *probe = calloc(1, sizeof *probe);
    if (!probe) return;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = AA_PROBE_TEXELS * 12,
                             .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(vk_ctx()->vma, &bi, &ai, &probe->buffer, &probe->allocation, &info) != VK_SUCCESS) {
        free(probe);
        return;
    }
    samples++;
    probe->mapped = info.pMappedData;
    probe->pq = ex->output_pq;
    probe->frame = ex->hdr_aa.frame;
    probe->synthetic = mode == 2;
    vk_exec_barrier(ex);
    VkBufferImageCopy copy = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
        .imageOffset = {(int32_t)(dst->width - AA_PROBE_SIDE) / 2, (int32_t)(dst->height - AA_PROBE_SIDE) / 2, 0},
        .imageExtent = {AA_PROBE_SIDE, AA_PROBE_SIDE, 1}};
    vkCmdCopyImageToBuffer(ex->cb, source, VK_IMAGE_LAYOUT_GENERAL, probe->buffer, 1, &copy);
    copy.bufferOffset = AA_PROBE_TEXELS * 8;
    vkCmdCopyImageToBuffer(ex->cb, dst->image, VK_IMAGE_LAYOUT_GENERAL, probe->buffer, 1, &copy);
    VkMemoryBarrier2 host = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo dependency = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &host};
    vkCmdPipelineBarrier2(ex->cb, &dependency);
    ex->recorded += 3;
    vk_defer_call(scene_aa_probe_done, probe);
}

static void scene_aa_probe_inject(VkExec *ex, VkImage image) {
    if (!image || aa_probe_injected || scene_aa_probe_mode() != 2) return;
    static int first_frame;
    if (!first_frame) {
        const char *setting = cfg_get("REKARINA_HDR_AA_PROBE_AT");
        char *end;
        const long frame = setting ? strtol(setting, &end, 10) : 0;
        first_frame = 1100;
        if (setting) {
            if (end != setting && !*end && frame > 0 && frame <= INT32_MAX) first_frame = (int)frame;
            else rt_log("hdr: invalid REKARINA_HDR_AA_PROBE_AT; using frame %d", first_frame);
        }
    }
    if (ex->hdr_aa.frame < (uint64_t)first_frame) return;
    // Probe only: a one-frame flat BT.2020 green expressed as signed linear BT.709.
    VkClearColorValue color = {.float32 = {aa_probe_rgb[0], aa_probe_rgb[1], aa_probe_rgb[2], aa_probe_rgb[3]}};
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vk_exec_barrier(ex);
    vkCmdClearColorImage(ex->cb, image, VK_IMAGE_LAYOUT_GENERAL, &color, 1, &range);
    ex->recorded++;
    vk_exec_barrier(ex);
    aa_probe_injected = ex->hdr_aa.frame;
}

static unsigned scene_aa_stage(uint64_t caller) {
    switch (caller - MOD_MAIN_BASE) {
    case 0x14e3a4c: return 1;
    case 0x14e3a94: return 2;
    case 0x14e3b24: return 3;
    case 0x251bc0: return 4;
    default: return 0;
    }
}

static VkImage scene_aa_bindings(const VkExec *ex, unsigned stage, uint32_t serial[3], VkFormat format[3]) {
    static const int src[] = {3, 4, 3, 0}, dst[] = {0, 1, 2, 1};
    const uint64_t handles[] = {ex->tex[NVN_STAGE_CS][src[stage - 1]], stage == 3 ? ex->tex[NVN_STAGE_CS][5] : 0,
                                ex->img[NVN_STAGE_CS][dst[stage - 1]]};
    VkImage probe_image = VK_NULL_HANDLE;
    vk_texture_lock();
    for (unsigned i = 0; i < 3; ++i) {
        if (!handles[i]) continue;
        view_t view = {0};
        int has_view = 0;
        const VkTexImage *image = vk_texture_by_id_locked(NVN_HANDLE_TEX(handles[i]), &view, &has_view);
        if (!image || image->gone || image->type != VK_IMAGE_TYPE_2D || image->samples != 1 || image->layers != 1 ||
            (has_view && ((view.set_levels && view.base_level) || (view.set_layers && view.min_layer) || view.set_swizzle))) continue;
        TexVkFormat mapped;
        serial[i] = image->serial;
        format[i] = has_view && view.set_format && tex_vk_format(view.format, &mapped) ? mapped.vk : image->format;
        if (stage == 1 && i == 0 && image->format == VK_FORMAT_R16G16B16A16_SFLOAT &&
            (image->usage & VK_IMAGE_USAGE_TRANSFER_DST_BIT)) probe_image = image->image;
    }
    vk_texture_unlock();
    return probe_image;
}

static void bind_handles(uint64_t *dst, int cap, const p_bind_handles *b);
static void pool_rebase(VkExec *ex, const p_bind_handles *b, uint64_t *row, int cap, int tex, int samp);

static int scene_aa_prepare_stage(VkExec *ex, VkExec *next, unsigned stage, const GfxShader *cs) {
    static const unsigned roles[] = {GFX_HDR_AA_PREP, GFX_HDR_AA_ROLES, GFX_HDR_AA_BLEND, GFX_HDR_AA_STORE};
    if (!cs) {
        int present, pending;
        cs = shader_for(next, NVN_STAGE_CS, &present, &pending, 0);
    }
    if (!cs) return 0;
    const unsigned i = stage - 1;
    const GfxShader *variant = stage == 2 ? gfx_shader_aa_mask(cs) : gfx_shader_renodx_aa(cs, roles[i]);
    if (!variant) return 0;
    ex->hdr_aa.shader[i] = variant;
    ex->hdr_aa.pipeline[i] = vk_pipeline_compute(variant, &ex->hdr_aa.info[i]);
    return ex->hdr_aa.pipeline[i] && ex->hdr_aa.info[i];
}

// Resolve this recording's bindings and pipelines before changing the color basis of any dispatch.
static int scene_aa_preflight(VkExec *ex, const GfxShader *prep, const uint32_t first[3], const VkFormat format[3]) {
    const VkFormat fp = VK_FORMAT_R16G16B16A16_SFLOAT;
    if (!ex->cprog || !vk_exec_cur_op || !first[0] || !first[2] || first[0] == first[2] || format[0] != fp ||
        format[2] != VK_FORMAT_R16G16_UINT || !scene_aa_prepare_stage(ex, ex, 1, prep)) return 0;

    VkExec next = {.prog = ex->prog, .tex_pool = ex->tex_pool, .samp_pool = ex->samp_pool};
    const unsigned cs = NVN_STAGE_CS;
    memcpy(next.tex[cs], ex->tex[cs], sizeof next.tex[cs]);
    memcpy(next.img[cs], ex->img[cs], sizeof next.img[cs]);
    ex->hdr_aa.ops[0] = vk_exec_cur_op;
    const op_hdr *op = nvn_op_next(vk_exec_cur_op);
    unsigned stage = 2;
    uint32_t weights = 0, blend = 0;
    for (unsigned step = 0; op && step < 192; ++step) {
        const void *p = nvn_op_payload(op);
        switch (op->op) {
        case OP_JUMP: op = (const op_hdr *)(uintptr_t)((const p_jump *)p)->target; continue;
        case OP_SET_TEX_POOL: next.tex_pool = ((const p_set_pool *)p)->pool; break;
        case OP_SET_SAMP_POOL: next.samp_pool = ((const p_set_pool *)p)->pool; break;
        case OP_BIND_PROGRAM: {
            const p_bind_program *b = p;
            if (b->stages & NVN_SHADER_STAGE_COMPUTE_BIT) {
                if (!b->prog) return 0;
                next.prog = b->prog;
            }
            break;
        }
        case OP_BIND_TEX: case OP_BIND_IMG: {
            const p_bind_handles *b = p;
            if (b->stage != cs) break;
            const int sampled = op->op == OP_BIND_TEX;
            uint64_t *row = sampled ? next.tex[cs] : next.img[cs];
            const int cap = sampled ? 128 : 32;
            if (b->first < 0 || b->n < 0 || b->n > 32 || b->first > cap - b->n) return 0;
            bind_handles(row, cap, b);
            pool_rebase(&next, b, row, cap, 1, sampled);
            break;
        }
        case OP_BARRIER: case OP_BIND_UBO: case OP_BIND_SSBO: case OP_BIND_SAMP:
        case OP_UPDATE_UB: case OP_CLEAR_BUF: case OP_PUSH_CONST: break;
        case OP_DISPATCH: case OP_DISPATCH_IND: {
            uint64_t caller;
            if (op->op == OP_DISPATCH) {
                const p_dispatch *d = p;
                if (d->x <= 0 || d->y <= 0 || d->z <= 0) return 0;
                caller = d->caller;
            } else {
                const p_dispatch_ind *d = p;
                VkMemSpan args;
                if (stage != 2 || !vk_mem_resolve(d->ind, sizeof(VkDispatchIndirectCommand), &args) || args.offset % 4)
                    return 0;
                caller = d->caller;
            }
            if (scene_aa_stage(caller) != stage) return 0;
            uint32_t serial[3] = {0};
            VkFormat fmt[3] = {0};
            scene_aa_bindings(&next, stage, serial, fmt);
            if (!serial[0] || !serial[2] || serial[0] == serial[2]) return 0;
            if (stage == 2) {
                if (serial[0] != first[2]) return 0;
                weights = serial[2];
            } else if (stage == 3) {
                if (fmt[0] != fp || fmt[2] != fp || serial[0] != first[0] || serial[1] != weights) return 0;
                blend = serial[2];
            } else if (fmt[0] != fp || fmt[2] != fp || serial[0] != blend) return 0;
            if (!scene_aa_prepare_stage(ex, &next, stage, NULL)) return 0;
            ex->hdr_aa.ops[stage - 1] = op;
            if (stage++ == 4) {
                ex->hdr_aa.composite = serial[2];
                return 1;
            }
            break;
        }
        default: return 0;
        }
        op = nvn_op_next(op);
    }
    return 0;
}

static VkPipeline scene_aa_dispatch(VkExec *ex, uint64_t caller, const GfxShader **cs, const VkPipeInfo **info) {
    const unsigned stage = scene_aa_stage(caller);
    if (!stage) return VK_NULL_HANDLE;
    const int tracing = vk_dlaa_pathlog();
    static const char *const role[] = {"prep", "weights", "blend", "reconstruct"};
    const uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    typeof(ex->hdr_aa) *a = &ex->hdr_aa;
    uint32_t serial[3] = {0};
    VkFormat format[3] = {0};
    VkImage probe_image = VK_NULL_HANDLE;
    const int eligible = stage == 1 && vk_hdr_renodx_scene() && (!vk_dlaa_active() || vk_dlaa_fsr1()) && !ex->gpu_cond;
    if (eligible || tracing || stage == 2) probe_image = scene_aa_bindings(ex, stage, serial, format);
    if (tracing)
        rt_log("PATHLOG guest AA f %llu %s main+%llx src s%u fmt %d aux s%u fmt %d dst s%u fmt %d",
               (unsigned long long)(frame - 1), role[stage - 1], (unsigned long long)(caller - MOD_MAIN_BASE),
               serial[0], format[0], serial[1], format[1], serial[2], format[2]);
    if (stage == 1) {
        a->frame = frame;
        a->phase = 0;
        a->active = eligible && !cfg_env("REKARINA_SKIP_CS") &&
                    scene_aa_preflight(ex, *cs, serial, format);
        if (a->active) scene_aa_probe_inject(ex, probe_image);
    }
    if (!a->active || a->frame != frame || a->phase != stage - 1 || a->ops[stage - 1] != vk_exec_cur_op) {
        if (stage == 2 && format[0] == VK_FORMAT_R16G16_UINT) {
            const GfxShader *mask = gfx_shader_aa_mask(*cs);
            if (!mask) rt_fatal("vk: the SMAA weights shader has no valid sampler for its R16G16_UINT edge mask");
            *cs = mask;
        }
        return VK_NULL_HANDLE;
    }
    a->phase = stage;
    *cs = a->shader[stage - 1];
    *info = a->info[stage - 1];
    if (tracing && stage != 2) rt_log("PATHLOG guest AA f %llu %s signed variant", (unsigned long long)(frame - 1), role[stage - 1]);
    return a->pipeline[stage - 1];
}
#endif

#define DISPATCH_PROGRAM_END(ex) ((ex)->prog = gprog_)
static void op_dispatch_(VkExec *ex, const p_dispatch *d);
static void op_dispatch(VkExec *ex, const p_dispatch *d) {
    DISPATCH_PROGRAM(ex);
    op_dispatch_(ex, d);
    DISPATCH_PROGRAM_END(ex);
}
static void op_dispatch_(VkExec *ex, const p_dispatch *d) {
#else
static void op_dispatch(VkExec *ex, const p_dispatch *d) {
#endif
    if (d->x <= 0 || d->y <= 0 || d->z <= 0) return;
    if (!ex->prog) return warn_once(W_PROGRAM, "a dispatch without a bound program is skipped");
    vk_contact_flush(ex);
    cb_res(ex);
#ifdef GAME_OOT
    if (vk_dlaa_skip_smaa(ex, d->caller)) return;
#endif
    dummies_init();
    int present;
    // A compute shader this machine has not translated yet is translated beside the replay, and one whose pipeline is
    // not built yet gets it built beside (vk_pipeline.c "a compute pipeline no prebuild made"): the dispatch is left
    // out meanwhile. Waiting for either on the replay thread froze the game for half a second (2026-11-14).
    int cs_wait = 0;
    const GfxShader *cs = shader_for(ex, NVN_STAGE_CS, &present, &cs_wait, 0);
    if (!cs) {
        if (cs_wait) left_out(LO_CS_SHADER, ex->prog->serial, NULL);
        return;
    }
    if (vk_dlaa_dispatch(ex, cs, d->caller)) {
#ifdef GAME_OOT
        // FSR 1 writes the verified reconstruction target without running the guest's last dispatch.
        if (ex->hdr_aa.active && ex->hdr_aa.phase == 3 && ex->hdr_aa.ops[3] == vk_exec_cur_op)
            ex->hdr_aa.phase = 4;
#endif
        return;
    }
    const VkPipeInfo *info = NULL;
    VkPipeline p = VK_NULL_HANDLE;
#ifdef GAME_OOT
    p = scene_aa_dispatch(ex, d->caller, &cs, &info);
#endif
    if (!p) {
        p = vk_pipeline_compute(cs, &info);
        if (vk_pipeline_compute_pending()) return left_out(LO_CS_PIPE, ex->prog->serial, cs);
    }
    if (!p || !info) return warn_once(W_PIPELINE, "a compute pipeline could not be created: dispatches with it are skipped");
    if (vk_movienet_skip(cs)) return;   // the movies' U-Net where a movie frame does not fit its time (vk_movienet.c)
    if (vk_movienet_on() && vk_movienet_is_pass12(cs)) {   // and pass 12 then adds nothing of it
        const VkPipeInfo *pi = NULL;
        VkPipeline pp = vk_pipeline_compute_passthrough(cs, &pi);
        if (pp && pi) p = pp, info = pi;
    }
    {   // DEBUG-SKIPCS: REKARINA_SKIP_CS=<input id prefix, 16 hex>[,...]: these dispatches are not run, to measure what they cost
        static const char *want;
        static int want_read;
        if (!want_read) want = cfg_env("REKARINA_SKIP_CS"), want_read = 1;   // read once: this runs for every dispatch
        if (want) {
            char id[17];
            const uint8_t *iid = gfx_shader_input_id(cs);
            for (int i = 0; i < 8; i++) sprintf(id + 2 * i, "%02x", iid[i]);
            if (strstr(want, id)) return;
        }
    }
    {   // DEBUG-DISPLOG: REKARINA_DISPLOG=<first frame>:<frames>: one line per dispatch with the compute program's input id and size
        static int on = -1;
        static uint64_t from, cnt;
        if (on < 0) { const char *e = cfg_get("REKARINA_DISPLOG"); on = e && sscanf(e, "%llu:%llu", (unsigned long long *)&from, (unsigned long long *)&cnt) == 2; }
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (on && f >= from && f < from + cnt) {
            const uint8_t *iid = gfx_shader_input_id(cs);
            rt_log("DISPLOG %llu %02x%02x%02x%02x%02x%02x%02x%02x %d %d %d lr %#llx", (unsigned long long)f, iid[0], iid[1], iid[2], iid[3], iid[4], iid[5], iid[6], iid[7], (int)d->x, (int)d->y, (int)d->z, (unsigned long long)d->caller);
        }
    }
    vk_exec_dispatch_barrier(ex);   // ends the pass; earlier writes are visible to the shader
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    c0_dim[0] = (uint32_t)d->x, c0_dim[1] = (uint32_t)d->y, c0_dim[2] = (uint32_t)d->z;
    {   // DEBUG-IMGWHERE: REKARINA_IMGWHERE=<input id prefix, 16 hex>: where the images and textures of these dispatches live, 12 times
        static int left = 12;
        static const char *want_e;
        static int want_e_read;
        if (!want_e_read) want_e = cfg_env("REKARINA_IMGWHERE"), want_e_read = 1;   // read once: this runs for every dispatch
        const char *want = left ? want_e : NULL;
        char id[17] = "";
        const uint8_t *iid = want ? gfx_shader_input_id(cs) : NULL;
        for (int i = 0; want && i < 8; i++) sprintf(id + 2 * i, "%02x", iid[i]);
        if (want && strstr(want, id)) {
            left--;
            const GfxShaderRes *r;
            vk_texture_lock();
            rt_log("IMGWHERE dispatch %s %dx%dx%d", id, (int)d->x, (int)d->y, (int)d->z);
            for (int kind = GFX_RES_TEX; kind <= GFX_RES_IMG; kind++) {
                int n = gfx_shader_resources(cs, kind, &r);
                for (int i = 0; i < n; i++)
                    if (r[i].slot >= 0 && r[i].slot < 32)
                        vk_debug_image_where_locked((int)((kind == GFX_RES_IMG ? ex->img[NVN_STAGE_CS][r[i].slot] : ex->tex[NVN_STAGE_CS][r[i].slot]) & 0xfffff),
                                                    kind == GFX_RES_IMG ? "image" : "texture");
            }
            vk_texture_unlock();
        }
    }
    {   // DEBUG-WBWHO
        extern uint64_t vk_dbg_cur_cs;
        const uint8_t *iid = gfx_shader_input_id(cs);
        vk_dbg_cur_cs = 0;
        for (int i = 0; i < 8; i++) vk_dbg_cur_cs = vk_dbg_cur_cs << 8 | iid[i];
    }
    twin_dispatch = 1;
    bind_resources(ex, info);
    twin_dispatch = 0;
    VkPassMark *m = dispatch_mark(ex, cs, (uint32_t)d->x, (uint32_t)d->y, (uint32_t)d->z);
    GPU_COND_BEGIN(ex);
    vkCmdDispatch(ex->cb, (uint32_t)d->x, (uint32_t)d->y, (uint32_t)d->z);
    GPU_COND_END(ex);
    vk_pass_mark_end(ex->cb, m);
    ex->dispatch_end_at = ++ex->recorded;
    vk_dlaa_dispatch_recorded(ex);
    pk_record(ex, 0, d->caller);   // DEBUG-PEEK
    if (vk_dump_armed) vk_dump_dispatch(ex, cs, p, info, d);
    vk_probe_dispatch(ex, cs, info);   // DEBUG-PROBE
}

// DispatchComputeIndirect: vkCmdDispatchIndirect from the pool buffer. The shader reads the group counts from c0 0x200,
// so they are copied there on the GPU, into a c0 range no other dispatch uses, before the dispatch.
#ifdef GAME_NS2
static void op_dispatch_ind_(VkExec *ex, const p_dispatch_ind *d);
static void op_dispatch_ind(VkExec *ex, const p_dispatch_ind *d) {
    DISPATCH_PROGRAM(ex);
    op_dispatch_ind_(ex, d);
    DISPATCH_PROGRAM_END(ex);
}
static void op_dispatch_ind_(VkExec *ex, const p_dispatch_ind *d) {
#else
static void op_dispatch_ind(VkExec *ex, const p_dispatch_ind *d) {
#endif
    if (!ex->prog) return warn_once(W_PROGRAM, "a dispatch without a bound program is skipped");
#ifdef GAME_OOT
    if (vk_dlaa_skip_smaa(ex, d->caller)) return;
#endif
    VkMemSpan ip;
    if (!vk_mem_resolve(d->ind, sizeof(VkDispatchIndirectCommand), &ip) || ip.offset % 4)
        return warn_once(W_INDIRECT, "indirect dispatch arguments at %#llx are not backed or not aligned: skipped", (unsigned long long)d->ind);
    cb_res(ex);
    dummies_init();
    int present;
    int cs_wait = 0;   // as in op_dispatch: translated and built beside the replay, left out meanwhile
    const GfxShader *cs = shader_for(ex, NVN_STAGE_CS, &present, &cs_wait, 0);
    if (!cs) {
        if (cs_wait) left_out(LO_CS_SHADER, ex->prog->serial, NULL);
        return;
    }
    const VkPipeInfo *info = NULL;
    VkPipeline p = VK_NULL_HANDLE;
#ifdef GAME_OOT
    if (scene_aa_stage(d->caller) == 2) p = scene_aa_dispatch(ex, d->caller, &cs, &info);
#endif
    if (!p) {
        p = vk_pipeline_compute(cs, &info);
        if (vk_pipeline_compute_pending()) return left_out(LO_CS_PIPE, ex->prog->serial, cs);
    }
    if (!p || !info) return warn_once(W_PIPELINE, "a compute pipeline could not be created: dispatches with it are skipped");
    vk_exec_barrier(ex);
    VK_MEM_USE(ip);
    bind_pipeline(ex, VK_PIPELINE_BIND_POINT_COMPUTE, p);
    c0_dim[0] = c0_dim[1] = c0_dim[2] = 0;
    B.c0_valid[C0_K(NVN_STAGE_CS)] = 0;   // a fresh c0 range: the copy below must not change one an earlier dispatch reads
    twin_dispatch = 1;
    bind_resources(ex, info);
    twin_dispatch = 0;
    if (B.c0_valid[C0_K(NVN_STAGE_CS)]) {   // the shader binds c0
        VkBufferCopy cp = {ip.offset, B.c0_off[C0_K(NVN_STAGE_CS)] + 0x200, sizeof(VkDispatchIndirectCommand)};
        vkCmdCopyBuffer(ex->cb, ip.buffer, B.c0_buf[C0_K(NVN_STAGE_CS)], 1, &cp);
        ex->recorded++;
#ifdef GAME_NS2
        cp.dstOffset = B.c0_off[C0_K(NVN_STAGE_CS)] + 4;   // vk_c0_fill: NVN2 compute reads them there too
        vkCmdCopyBuffer(ex->cb, ip.buffer, B.c0_buf[C0_K(NVN_STAGE_CS)], 1, &cp);
        ex->recorded++;
#endif
        vk_exec_barrier(ex);
        B.c0_valid[C0_K(NVN_STAGE_CS)] = 0;   // the range no longer holds B.c0[k]
    }
    VkPassMark *m = dispatch_mark(ex, cs, 0, 0, 0);
    GPU_COND_BEGIN(ex);
    vkCmdDispatchIndirect(ex->cb, ip.buffer, ip.offset);
    GPU_COND_END(ex);
    vk_pass_mark_end(ex->cb, m);
    ex->dispatch_end_at = ++ex->recorded;
    pk_record(ex, d->ind, d->caller);   // DEBUG-PEEK
    vk_mem_count(ip.cls, VK_MUSE_IND, sizeof(VkDispatchIndirectCommand));
    VK_MEM_BOUND(ip);
    if (vk_dump_armed) vk_dump_dispatch(ex, cs, p, info, &(p_dispatch){0, 0, 0, d->caller});
    vk_probe_dispatch(ex, cs, info);   // DEBUG-PROBE
}

static void op_update_ub(VkExec *ex, const p_update_ub *u) {
    if (!u->size) return;
    VkMemSpan sp;
    if (!vk_mem_resolve(u->dst, u->size, &sp)) return warn_once(W_UB, "UpdateUniformBuffer to %#llx is not backed: dropped", (unsigned long long)u->dst);
    cb_res(ex);
    vk_exec_barrier(ex);   // earlier draws finished with the old contents
    VkBuffer b;
    VkDeviceSize o;
    memcpy(scratch(ex, u->size, &b, &o), u->data, u->size);
    VK_MEM_PICK(sp);
    VkBufferCopy cp = {o, sp.offset, u->size};
    vkCmdCopyBuffer(ex->cb, b, sp.buffer, 1, &cp);
    // Chunks of a draw stretch may still be recording (vk_exec.c par_join_op: an update no longer waits for them): the
    // write-back note under the lock they note theirs with, and the snapshot's copy of the range dropped only for
    // draws after this point (vk_usnap_drop_update).
    plat_mutex_lock(&vk_par_lock);
    int owned = vk_mem_gpu_wrote(&sp, u->size);
    if (vk_exec_ub_nojoin()) vk_usnap_drop_update(u->dst, u->size);
    else vk_usnap_drop(u->dst, u->size);
    plat_mutex_unlock(&vk_par_lock);
    ex->recorded++;
    vk_mem_count(sp.cls, VK_MUSE_GPU_WRITE, u->size);
    note_written(ex, u->dst, u->size, owned, 0);
    upd_note(ex, u);
}

// ---- bind ops

// The handles a bind op just stored are looked up by the next draw a few hundred nanoseconds later: in the descriptor
// memo (TD), the feedback memo (FM) and the texture registry's slot table, megabytes of tables a frame's handles are
// scattered over, so each lookup was a cache miss on the replay thread (the single hottest lines of tex_info,
// feedback_scan and vk_texture_by_id_locked in the village profile). Asked for here, the lines arrive while the ops
// between the bind and the draw are decoded. TD and FM share the index: the type and separate bits of TD's hash lie
// below the 51 bits its shift drops.
static void prefetch_handles(const uint64_t *row, int cap, const p_bind_handles *b) {
    if (!row || b->first < 0 || b->n <= 0 || b->n > 32 || b->first + b->n > cap) return;
    if (vk_par_walk && !vk_par_worker) return;   // the walk of a stretch: the chunks' threads look the handles up
    for (int i = 0; i < b->n; i++) {
        uint64_t h = row[b->first + i];
        if (!h) continue;
        uint32_t k = (uint32_t)((h * 0x9E3779B97F4A7C15ull) >> 51);
        __builtin_prefetch(&TD[k]);
        __builtin_prefetch((const char *)&TD[k] + sizeof TD[0] - 1);
        __builtin_prefetch(&FM[k]);
        vk_texture_slot_prefetch((int)(h & 0xfffff));
    }
}

static void bind_handles(uint64_t *dst, int cap, const p_bind_handles *b) {
    if (b->stage < 0 || b->stage >= 6 || b->first < 0 || b->n < 0 || b->n > 32 || b->first + b->n > cap) {
        warn_once(W_BIND, "a bind of %d handles at %d of stage %d is out of range: ignored", b->n, b->first, b->stage);
        return;
    }
    memcpy(&dst[b->first], b->h, sizeof(uint64_t) * (size_t)b->n);
}

#ifdef GAME_NS2
// One ID space per descriptor pool (nvn_api_obj.c nvn_dpool_base): the IDs of the handles just bound move to the part
// of the back end's tables that the bound texture (and sampler) pool owns. A game with one pool of each kind has base 0.
static void pool_rebase(VkExec *ex, const p_bind_handles *b, uint64_t *row, int cap, int tex, int samp) {
    int32_t tb = tex && ex->tex_pool ? ex->tex_pool->base : 0, sb = samp && ex->samp_pool ? ex->samp_pool->base : 0;
    if ((!tb && !sb) || !row || b->first < 0 || b->n < 0 || b->first + b->n > cap) return;
    for (int i = b->first; i < b->first + b->n; i++) {   // texture ID in bits 0-19, sampler ID (also a separate one) 20-31
        uint64_t h = row[i], tid = h & 0xfffff, sid = h >> 20 & 0xfff;
        if (tid && tex) tid += (uint64_t)tb;
        if (sid && samp) sid += (uint64_t)sb;
        row[i] = (tid & 0xfffff) | (sid & 0xfff) << 20;
    }
}
#endif

// DEBUG-DRAWCRUMB: REKARINA_DRAW_CRUMBS=1: a breadcrumb before every draw op and a ring of what each had bound. A lost
// device then names the draw the GPU reached last and the one before which everything had finished: the read that
// killed it is in a draw between the two. Each address is asked again at the report: one that resolves no more names
// a pool that went.
#define DC_N (1u << 16)
typedef struct {
    uint32_t n, op;
    uint64_t fence, prog, ibuf, vb[4], vbn[4], ubo[2][6], ubon[2][6], ssbo[2][4], tex[2][8];
    int32_t count, base;
    uint8_t worker, cond;
} DrawCrumb;
static DrawCrumb *dc_ring;
static volatile int32_t dc_seq;
static int dc_on = -1;

static void draw_crumb(VkExec *ex, uint32_t op, const void *payload) {
    extern _Thread_local int vk_par_worker;
    uint32_t n = (uint32_t)__atomic_add_fetch(&dc_seq, 1, __ATOMIC_RELAXED) & 0x0fffffff;
    DrawCrumb *d = &dc_ring[n & (DC_N - 1)];
    memset(d, 0, sizeof *d);
    d->n = n, d->op = op, d->fence = ex->fence, d->prog = ex->prog ? ex->prog->serial : 0;
    d->worker = (uint8_t)vk_par_worker;
#ifdef GAME_NS2
    d->cond = (uint8_t)ex->gpu_cond;
#endif
    if (op == OP_DRAW_IDX) {
        const p_draw_idx *p = payload;
        d->ibuf = p->ibuf, d->count = p->count, d->base = p->base_vertex;
    } else if (op == OP_DRAW) {
        d->count = ((const p_draw *)payload)->count;
    }
    for (int i = 0; i < 4; i++) d->vb[i] = ex->vbuf[i].address, d->vbn[i] = ex->vbuf[i].size;
    static const int st[2] = {NVN_STAGE_VS, NVN_STAGE_FS};
    for (int s = 0; s < 2; s++) {
        for (int i = 0; i < 6; i++) d->ubo[s][i] = ex->ubo[st[s]][i].addr, d->ubon[s][i] = ex->ubo[st[s]][i].size;
        for (int i = 0; i < 4; i++) d->ssbo[s][i] = ex->ssbo[st[s]][i].addr;
        for (int i = 0; i < 8; i++) d->tex[s][i] = ex->tex[st[s]][i];
    }
    vk_crumb(ex->cb, VK_CRUMB_DRAW, n);
}

uint64_t vk_exec_draw_crumb_fence(uint32_t n) {   // DEBUG-LIVECHECK: the submission of draw crumb n, 0 unknown
    const DrawCrumb *d = dc_ring ? &dc_ring[n & (DC_N - 1)] : NULL;
    return d && d->n == n ? d->fence : 0;
}

void vk_exec_draw_crumb_describe(uint32_t n, char *o, size_t size) {
    const DrawCrumb *d = dc_ring ? &dc_ring[n & (DC_N - 1)] : NULL;
    if (!d || d->n != n) {
        snprintf(o, size, "draw %u (no record)", n);
        return;
    }
    size_t k = (size_t)snprintf(o, size, "draw %u: %s of submission %llu, program %llu, %d elements, %s%s;", n, nvn_op_name(d->op),
                                (unsigned long long)d->fence, (unsigned long long)d->prog, d->count,
                                d->worker ? "recorded in a chunk" : "recorded by the replay", d->cond ? ", under a GPU condition" : "");
    VkMemSpan sp;
#define ADDR(what, a, len) do {         if ((a) && k < size) k += (size_t)snprintf(o + k, size - k, " %s %#llx+%#llx%s", what, (unsigned long long)(a),                                                    (unsigned long long)(len), vk_mem_resolve((a), 1, &sp) ? "" : " NOT BACKED NOW");     } while (0)
    ADDR("index", d->ibuf, 0);
    for (int i = 0; i < 4; i++) ADDR("vb", d->vb[i], d->vbn[i]);
    for (int s = 0; s < 2; s++) {
        for (int i = 0; i < 6; i++) ADDR(s ? "fs-ubo" : "vs-ubo", d->ubo[s][i], d->ubon[s][i]);
        for (int i = 0; i < 4; i++) ADDR(s ? "fs-ssbo" : "vs-ssbo", d->ssbo[s][i], 0);
    }
#undef ADDR
    for (int s = 0; s < 2; s++)
        for (int i = 0; i < 8; i++)
            if (d->tex[s][i] && k < size)
                k += (size_t)snprintf(o + k, size - k, " %s-tex%d %#llx", s ? "fs" : "vs", i, (unsigned long long)d->tex[s][i]);
}

#include "renderer_mcp.h"

static void bloom_draw(VkExec *execution, uint32_t operation, const void *payload, unsigned stage) {
    unsigned role = stage & VK_BLOOM_ROLE_MASK;
    unsigned level = stage >> VK_BLOOM_LEVEL_SHIFT;
    int rendering_enabled = !execution->render_disabled;
#ifdef GAME_NS2
    rendering_enabled &= !execution->gpu_cond;
#endif
    if (role == VK_BLOOM_EXTRACT && rendering_enabled) {
        // Match the scene image and subresource, not the numerical sampler slot.
        for (unsigned slot = 0; slot < 128 && !vk_dlaa_prebloom_active(); slot++)
            if (execution->tex[NVN_STAGE_FS][slot]) vk_dlaa_prebloom(execution, execution->tex[NVN_STAGE_FS][slot], role);
    }
    const tex_t *depth = execution->rt.depth;
    VkExecTarget guest_targets[8];
    const int had_guest_scissor = execution->has_scissor;
    float guest_viewport[6];
    memcpy(guest_viewport, execution->vp[0], sizeof guest_viewport);
    p_scissor guest_scissor = execution->scissor;
    unsigned guest_width = 0, guest_height = 0;
    int private_route = 0;
    int omit_depth = 0;
    if (rendering_enabled && vk_dlaa_prebloom_active() && execution->rt.n > 0 &&
        (!depth || (!execution->pipe.ds.dt && !execution->pipe.ds.dw && !execution->pipe.ds.st))) {
        vk_exec_pass_end(execution);
        if (depth) {
            omit_depth = 1;
            execution->rt.depth = NULL;
            execution->rt_resolved = 0;
        }
        cb_res(execution);
        vk_exec_resolve_targets(execution);
        memcpy(guest_targets, execution->color, sizeof guest_targets);
        guest_width = execution->pass_w;
        guest_height = execution->pass_h;
        if (role == VK_BLOOM_TONEMAP) {
            if (execution->pass_ncolor == 1)
                private_route = vk_dlaa_prebloom_tonemap(execution, &execution->color[0]);
        } else {
            private_route = vk_dlaa_bloom_targets(execution, role, level, guest_targets);
        }
        if (private_route) {
            unsigned attachment = 0;
            while (!execution->color[attachment].private_attachment_view) ++attachment;
            execution->pass_w = execution->color[attachment].width;
            execution->pass_h = execution->color[attachment].height;
            execution->pass_scale = 1;
            const double scale_x = (double)execution->pass_w / guest_width;
            const double scale_y = (double)execution->pass_h / guest_height;
            // Keep the normalized rectangle, including inset auxiliary draws, while changing its pixel extent.
            execution->vp[0][0] *= scale_x;
            execution->vp[0][1] *= scale_y;
            execution->vp[0][2] *= scale_x;
            execution->vp[0][3] *= scale_y;
            if (had_guest_scissor) {
                int64_t right = (int64_t)guest_scissor.x + guest_scissor.w;
                int64_t bottom = (int64_t)guest_scissor.y + guest_scissor.h;
                if (right > guest_width) right = guest_width;
                if (bottom > guest_height) bottom = guest_height;
                execution->scissor.x = (int32_t)floor(fmax(0, guest_scissor.x) * scale_x);
                execution->scissor.y = (int32_t)floor(fmax(0, guest_scissor.y) * scale_y);
                execution->scissor.w = guest_scissor.w <= 0 ? 0 : (int32_t)ceil(right * scale_x) - execution->scissor.x;
                execution->scissor.h = guest_scissor.h <= 0 ? 0 : (int32_t)ceil(bottom * scale_y) - execution->scissor.y;
            }
            execution->dirty |= VK_EXEC_DIRTY_VIEWPORT | VK_EXEC_DIRTY_SCISSOR;
            vk_exec_draw_new_cb();
        } else if (omit_depth) {
            execution->rt.depth = depth;
            execution->rt_resolved = 0;
        }
    }
    if (private_route && role == VK_BLOOM_PYRAMID && vk_dlaa_pathlog())
        rt_log("PATHLOG bloom pyramid guest level %u -> %ux%u (private)", level, execution->pass_w, execution->pass_h);
    if (role != VK_BLOOM_PYRAMID && rendering_enabled && execution->rt.n > 0 && execution->rt.color[0]) {
        const char *stage = role == VK_BLOOM_EXTRACT ? "extract" : role == VK_BLOOM_AUXILIARY ? "aux" : "tonemap";
        unsigned width = execution->rt.color[0]->d.width, height = execution->rt.color[0]->d.height;
        renderer_mcp_bloom(stage, width, height, private_route ? execution->pass_w : width,
                           private_route ? execution->pass_h : height);
        if (vk_dlaa_pathlog())
            rt_log("PATHLOG bloom %s guest %ux%u -> %ux%u (%s)", stage, width, height,
                   private_route ? execution->pass_w : width, private_route ? execution->pass_h : height,
                   private_route ? "private" : "guest");
    }
    execution->draw_role = private_route ? role : 0;
    int drew = operation == OP_DRAW ? op_draw(execution, payload) : op_draw_idx(execution, payload);
    execution->draw_role = 0;
    if (role == VK_BLOOM_EXTRACT)
        vk_dlaa_scene_bloom(execution, private_route && drew);
    if (!private_route && drew && rendering_enabled && role == VK_BLOOM_TONEMAP)
        vk_dlaa_stock_tonemap(&execution->color[0]);
    if (private_route) {
        vk_exec_pass_end(execution);
        if (role != VK_BLOOM_TONEMAP) vk_dlaa_bloom_finish(execution, guest_targets, drew);
        else if (drew) vk_dlaa_prebloom_finish_tonemap(execution, &guest_targets[0]);
        memcpy(execution->vp[0], guest_viewport, sizeof guest_viewport);
        execution->scissor = guest_scissor;
        execution->rt.depth = depth;
        execution->rt_resolved = 0;
        execution->dirty |= VK_EXEC_DIRTY_VIEWPORT | VK_EXEC_DIRTY_SCISSOR;
        vk_exec_draw_new_cb();
    }
}

void vk_exec_op_draw(VkExec *ex, uint32_t op, const void *payload) {
    if (op == OP_DRAW || op == OP_DRAW_IDX || op == OP_DRAW_IDX_IND
#ifdef GAME_NS2
        || op == OP_DRAW_MESH || op == OP_DRAW_MESH_IND
#endif
    ) {
        unsigned recorded_role = op == OP_DRAW ? ((const p_draw *)payload)->draw_role :
                                 op == OP_DRAW_IDX ? ((const p_draw_idx *)payload)->draw_role :
                                 op == OP_DRAW_IDX_IND ? ((const p_draw_idx_ind *)payload)->draw_role : 0;
        ex->interface_draw = (recorded_role & DRAW_ROLE_INTERFACE) != 0;
        ex->scene_output_role = recorded_role & (DRAW_ROLE_SCENE_OUTPUT | DRAW_ROLE_SCENE_TONEMAP);
        ex->direct_draw_source = 0;
        ex->output_encoding = GFX_OUTPUT_UNSPECIFIED;
#ifdef GAME_OOT
        ex->output_pq = (VkGuestPq){0};
        ex->hdr_aa.probe_source = VK_NULL_HANDLE;
#endif
        ex->interface_copy_source = VK_NULL_HANDLE;
        vk_interface_coverage_check_draw(ex);
    }
    if (dc_on < 0) {
        dc_on = cfg_env("REKARINA_DRAW_CRUMBS") != NULL;
        if (dc_on && !(dc_ring = calloc(DC_N, sizeof *dc_ring))) dc_on = 0;
    }
    if (dc_on && ex->cb && (op == OP_DRAW || op == OP_DRAW_IDX || op == OP_DRAW_IDX_IND
#ifdef GAME_NS2
                            || op == OP_DRAW_MESH || op == OP_DRAW_MESH_IND
#endif
                            ))
        draw_crumb(ex, op, payload);
    if (op == OP_DRAW || op == OP_DRAW_IDX || op == OP_DRAW_IDX_IND
#ifdef GAME_NS2
        || op == OP_DRAW_MESH || op == OP_DRAW_MESH_IND
#endif
    ) {   // DEBUG-OQLOG: every draw op the game made while samples were counted
        void vk_oqlog_note(const char *what, int a, int b);
        vk_oqlog_note(nvn_op_name(op), ex->render_disabled, op == OP_DRAW_IDX ? ((const p_draw_idx *)payload)->count : -1);
    }
    switch (op) {
    case OP_BIND_PROGRAM: {
        const p_bind_program *b = payload;
#ifdef GAME_NS2
        // NVN binds the stages the mask names (MacNativeGames ns2-port.md: Pokemon Pokopia binds compute programs
        // between draws). Compute goes to cprog, every other stage bit (NVN2's task and mesh bits included) to prog.
        if (b->stages & NVN_SHADER_STAGE_COMPUTE_BIT) ex->cprog = b->prog;
        if (b->stages & ~NVN_SHADER_STAGE_COMPUTE_BIT) ex->prog = b->prog, ex->prog_stages = b->stages;
#else
        ex->prog = b->prog, ex->prog_stages = b->stages;
#endif
        __builtin_prefetch(b->prog);   // its serial and stages are read at the next draw
        bind_gen++;
        break;
    }
    case OP_BIND_STATE: {
        const p_bind_state *b = payload;
        blend_st bs;
        switch (b->kind) {
        case 0:
            memcpy(&bs, b->bytes, sizeof bs);
            if (bs.target < 8) ex->pipe.blend[bs.target] = bs;
            break;
        case 1: memcpy(&ex->pipe.color, b->bytes, sizeof ex->pipe.color); break;
        case 2: memcpy(&ex->pipe.chmask, b->bytes, sizeof ex->pipe.chmask); break;
        case 3: memcpy(&ex->pipe.ms, b->bytes, sizeof ex->pipe.ms); break;
        case 4: memcpy(&ex->pipe.poly, b->bytes, sizeof ex->pipe.poly); break;
        case 5: memcpy(&ex->pipe.ds, b->bytes, sizeof ex->pipe.ds); break;
        case 6: memcpy(&ex->point_size, b->bytes, sizeof ex->point_size); break;
        }
        break;
    }
    case OP_BIND_VATTR: {
        const p_bind_vattr *b = payload;
        if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break;
        memset(ex->pipe.vattr, 0, sizeof ex->pipe.vattr);   // MNG: the new list replaces the old one
        memcpy(&ex->pipe.vattr[b->first], b->st, sizeof(uint32_t) * (size_t)b->n);
        break;
    }
    case OP_BIND_VSTREAM: {
        const p_bind_vstream *b = payload;
        if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break;
        memset(ex->pipe.vstream, 0, sizeof ex->pipe.vstream);
        memcpy(&ex->pipe.vstream[b->first], b->st, sizeof(vs_st) * (size_t)b->n);
        break;
    }
    case OP_BIND_VBUF: {
        const p_bind_vbuf *b = payload;
        if (b->first < 0 || b->n < 0 || b->first + b->n > 16) break;
        memcpy(&ex->vbuf[b->first], b->r, sizeof(NVNbufferRange) * (size_t)b->n);
        break;
    }
    case OP_BIND_UBO: {
        const p_bind_buf *b = payload;
        if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 18) ex->ubo[b->stage][b->index] = *b;
        bind_gen++;
        break;
    }
    case OP_BIND_SSBO: {
        const p_bind_buf *b = payload;
        if (b->stage >= 0 && b->stage < 6 && b->index >= 0 && b->index < 16) ex->ssbo[b->stage][b->index] = *b;
        {   // DEBUG-SSBOLOG: REKARINA_SSBOLOG=<frame>: every storage buffer bind of that frame
            static int on = -1;
            static uint64_t from;
            if (on < 0) { const char *e = cfg_get("REKARINA_SSBOLOG"); on = e && *e; from = on ? strtoull(e, NULL, 0) : 0; }
            if (on && __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) == from)
                rt_log("SSBO bind stage %d index %d addr %#llx size %#llx", b->stage, b->index, (unsigned long long)b->addr,
                       (unsigned long long)b->size);
        }
        bind_gen++;
        break;
    }
    case OP_BIND_TEX: {
        const p_bind_handles *b = payload;
        bind_handles(b->stage >= 0 && b->stage < 6 ? ex->tex[b->stage] : NULL, 128, b);
#ifdef GAME_NS2
        pool_rebase(ex, b, b->stage >= 0 && b->stage < 6 ? ex->tex[b->stage] : NULL, 128, 1, 1);
#endif
        prefetch_handles(b->stage >= 0 && b->stage < 6 ? ex->tex[b->stage] : NULL, 128, b);
        ex->tex_serial++;
        bind_gen++;
        break;
    }
    case OP_BIND_SAMP: {
        const p_bind_handles *b = payload;
        bind_handles(b->stage >= 0 && b->stage < 6 ? ex->samp[b->stage] : NULL, 128, b);
#ifdef GAME_NS2
        pool_rebase(ex, b, b->stage >= 0 && b->stage < 6 ? ex->samp[b->stage] : NULL, 128, 0, 1);
#endif
        break;
    }
    case OP_BIND_IMG: {
        const p_bind_handles *b = payload;
        bind_handles(b->stage >= 0 && b->stage < 6 ? ex->img[b->stage] : NULL, 32, b);
#ifdef GAME_NS2
        pool_rebase(ex, b, b->stage >= 0 && b->stage < 6 ? ex->img[b->stage] : NULL, 32, 1, 0);
#endif
        ex->tex_serial++;   // the shader memo's handle check (feedback_scan reads only textures: a rescan at most)
        bind_gen++;
        break;
    }
    case OP_UPDATE_UB: op_update_ub(ex, payload); break;
    case OP_DRAW: case OP_DRAW_IDX: {
        unsigned recorded_role = op == OP_DRAW ? ((const p_draw *)payload)->draw_role : ((const p_draw_idx *)payload)->draw_role;
        unsigned bloom_role = recorded_role & ~(DRAW_ROLE_INTERFACE | DRAW_ROLE_SCENE_OUTPUT | DRAW_ROLE_SCENE_TONEMAP);
#ifdef GAME_OOT
        if (!bloom_role && ex->scene_output_role == DRAW_ROLE_SCENE_TONEMAP && ex->prebloom_tonemap)
            bloom_role = VK_BLOOM_TONEMAP;
#endif
        if (bloom_role) bloom_draw(ex, op, payload, bloom_role);
        else {
            int drew = op == OP_DRAW ? op_draw(ex, payload) : op_draw_idx(ex, payload);
#ifdef GAME_OOT
            if (drew && !ex->gpu_cond && ex->scene_output_role == DRAW_ROLE_SCENE_TONEMAP)
                vk_dlaa_stock_tonemap(&ex->color[0]);
            if (drew && ex->scene_output_role == DRAW_ROLE_SCENE_ENCODE) scene_aa_probe(ex);
#endif
            if (drew && ex->direct_draw_source)
                vk_dlaa_direct_draw(&ex->color[0], ex->direct_draw_source, ex->scene_output_role == DRAW_ROLE_SCENE_ENCODE);
            if (drew && ex->interface_copy_source && ex->color[0].img)
                vk_interface_coverage_copy(ex->interface_copy_source, ex->color[0].image, ex->color[0].img->addr, 1);
#ifdef GAME_OOT
            if (drew && ex->output_encoding && ex->color[0].img) {
                const VkTexImage *img = ex->color[0].img;
                vk_hdr_record_output(img->serial, img->image, img->addr, ex->output_encoding, &ex->output_pq);
            }
#endif
        }
        ex->scene_output_role = 0;
        ex->interface_draw = 0;
    } break;
    case OP_DISPATCH: op_dispatch(ex, payload); break;
    case OP_DRAW_IDX_IND: {
        const p_draw_idx_ind *draw = payload;
        ex->interface_draw = (draw->draw_role & DRAW_ROLE_INTERFACE) != 0;
        op_draw_idx_ind(ex, draw);
        ex->interface_draw = 0;
    } break;
    case OP_DISPATCH_IND: op_dispatch_ind(ex, payload); break;
#ifdef GAME_NS2
    case OP_PUSH_CONST: {
        const p_push_const *b = payload;
        if (b->offset + (uint64_t)b->size <= NVN2_PUSH_SIZE) memcpy(ex->push[b->compute != 0] + b->offset, b->data, b->size);
        bind_gen++;
        break;
    }
    case OP_DRAW_MESH: op_draw_mesh(ex, payload, NULL); break;
    case OP_DRAW_MESH_IND: op_draw_mesh(ex, NULL, payload); break;
#endif
    default: rt_fatal("vk: op %s is not a draw op", nvn_op_name(op));
    }
}

// The last executed list's draws as " <position banks>/<changed banks>:<draws>" (bit n: cn), for REKARINA_MV_CHECK's lines.
const char *vk_mvobj_frame_masks(void) { return fm_last; }
