// Graphics and compute pipelines and descriptor layouts of the Vulkan replay. The op-stream replay (vk_exec.c)
// collects NVN state into a VkPipeState and asks for the pipeline of each draw; this file maps that state to Vulkan,
// numbers the descriptor bindings and keeps the caches.
//
// Binding model. The translator (tools/xlate/Program.cs, Ryujinx 1.3.2's SPIR-V back end) decorates every resource of a
// shader with DescriptorSet 0 and numbers the bindings of each shader from 1 in the order it meets them (Accessor
// Create*Binding); binding 0 is Ryujinx's support buffer (ResourceManager: SupportBuffer.Binding, a std140 uniform
// buffer every shader declares). The numbering is per shader, so the vertex and fragment stages of one program collide.
// The scheme here:
//   - one descriptor set, set 0, per pipeline; no push constants (Ryujinx's SPIR-V uses none);
//   - binding 0: the support buffer, one uniform buffer shared by all stages of the pipeline;
//   - vertex stage: its own bindings 1..n unchanged; fragment stage: its bindings b >= 1 move to b + n (the SPIR-V
//     Binding decorations are rewritten when the module is created); compute: its own bindings;
//   - descriptor types from the reflection: constant buffers UNIFORM_BUFFER, storage buffers STORAGE_BUFFER, textures
//     COMBINED_IMAGE_SAMPLER (UNIFORM_TEXEL_BUFFER for buffer textures, SAMPLED_IMAGE for separate ones), images
//     STORAGE_IMAGE (STORAGE_TEXEL_BUFFER for buffer images), array lengths as descriptor counts;
//   - set 0 is a push-descriptor set (VK_KHR_push_descriptor) when its descriptors fit maxPushDescriptors (32 on the RX
//     9070; a pipeline layout may hold one push set only, which rules out a set per stage); larger ones (a vertex and a
//     fragment stage near the census maxima of 15 and 27) are ordinary sets the replay allocates (VkPipeInfo.push 0).
// A fixed layout per stage combination (Ryujinx's own back end: a binding per constant buffer, storage buffer, texture
// and image slot per stage) would hold about 150 descriptors and could never be pushed, so layouts are made per distinct
// binding list and shared by every pipeline with that list; VkPipeInfo.bindings tells the replay which NVN slot feeds
// each binding.
//
// Fragment inputs the vertex stage does not write (stages are translated separately; MNG nvn_mtl.m msl_varyings) get
// outputs added to the vertex module, as Vulkan rejects such pairs, initialised to (0, 0, 0, 1) as Maxwell reads an
// unwritten output (docs/maxwell.md 1.2). The unwritten components of an output the vertex stage does write
// start at (0, 0, 0, 1) in the translator (tools/xlate/patches/0006).
//
// Pipeline key. Everything Vulkan 1.3 can take as dynamic state is dynamic: viewports and scissors with count, depth
// bias, blend constants, stencil masks and references, cull mode, front face, topology (within its class), depth and
// stencil tests and ops, depth bias enable, rasterizer discard; vertex input (VK_EXT_vertex_input_dynamic_state, else
// binding strides only); and, where VK_EXT_extended_dynamic_state3 offers them, depth clamp, polygon mode, blend enables,
// equations, write masks and alpha to coverage. The key holds the rest: shaders, target formats, sample count, topology
// class, depth clip mode, logic op and what is not dynamic on the GPU. The VkPipelineCache persists to
// pipeline-cache.bin next to rekarina.exe (REKARINA_PIPELINE_CACHE overrides the path).
//
// Provenance: the NVN -> Vulkan enum mappings (blend factors and equations, compare functions, stencil ops), the vertex
// format table with its 3-component fallback and the pipeline cache file handling follow LAS Recomp
// runtime/nvn/vulkan/exec.c and device.c @ da35b83a (MIT, Copyright (c) 2026 Peiikai); the pipeline state the key holds
// (program, target formats, blend and write masks, vertex layout, sample count) and the varying fix-up follow
// MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/nvn_mtl.m pso_key_fill / pso_desc / msl_varyings and its vfmt
// table @ b75be8cf (MIT, Copyright (c) 2026 jhaizhou); the binding conventions are those of Ryujinx's shader translator
// (MIT, tools/vendor/ryujinx).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "nvn_hmap.h"
#include "prep.h"
#include "platform/platform.h"
#include "tex.h"

#define LOG_ONCE(...) do { static volatile int32_t once_; if (!plat_xchg32(&once_, 1)) rt_log(__VA_ARGS__); } while (0)

enum { SPV_OP_ENTRY_POINT = 15, SPV_OP_TYPE_INT = 21, SPV_OP_TYPE_FLOAT = 22, SPV_OP_TYPE_VECTOR = 23,
       SPV_OP_TYPE_POINTER = 32, SPV_OP_CONSTANT = 43, SPV_OP_CONSTANT_COMPOSITE = 44,
       SPV_OP_CONSTANT_NULL = 46, SPV_OP_FUNCTION = 54, SPV_OP_VARIABLE = 59,
       SPV_OP_DECORATE = 71, SPV_DEC_BUILTIN = 11, SPV_DEC_LOCATION = 30, SPV_DEC_INDEX = 32, SPV_DEC_BINDING = 33, SPV_DEC_DESCRIPTOR_SET = 34,
       SPV_SC_OUTPUT = 3 };

#define MAX_BINDINGS 128
#define PIPE_BUCKETS 4096
#define LAYOUT_BUCKETS 256

// ---- device capabilities and the pipeline cache (set up on first use, under P.lock)

static struct {
    PlatMutex lock;
    volatile int32_t ready;
    VkDevice device;
    VkPipelineCache cache;
    uint32_t max_push;
    int indep;                         // stage-independent layouts (vk_caps.gpl): vertex set 0, fragment set 1
    int push_stage;                    // GFX_STAGE_* whose set is pushed with independent layouts
    int gpl;                           // draws whose pipeline is building get a fast-linked one (vk_caps.gpl)
    struct { uint64_t links, wlinks, dlinks, requeued, link_fail, defer, lib_n[5], lib_fail, swaps, slow, pre_used[5]; double link_ms, link_max, lib_ms[5], lib_max[5], dlink_ms, dlink_max; } g;
    int sg_fixed32;                   // the device's subgroups are always 32 lanes (min = max = 32)
    int vs_outs_all;                   // REKARINA_VS_OUTS: 0 union of paired fragment inputs, 1 (all) all 32 output locations
    int dyn_vi, dyn_clamp, dyn_poly, dyn_blend_en, dyn_blend_eq, dyn_wmask, dyn_a2c;
    int f_logic_op, f_depth_clamp, f_fill_non_solid, f_dual_src;
    int f_full_subgroups;              // computeFullSubgroups: compute pipelines can ask for full subgroups
    int pv_last;                       // provokingVertexLast enabled: NVN's default is the last vertex
    VkShaderStageFlags sg32_stages;    // stages that accept requiredSubgroupSize = 32 (0: not available)
    VkDynamicState dyn[32];
    uint32_t n_dyn;
    PFN_vkCmdPushDescriptorSetKHR push;
    int tpl;                           // descriptor update templates on (make_tpl)
    PFN_vkCreateDescriptorUpdateTemplate tpl_create;
    PFN_vkDestroyDescriptorUpdateTemplate tpl_destroy;
    PFN_vkUpdateDescriptorSetWithTemplate tpl_update;
    PFN_vkCmdPushDescriptorSetWithTemplateKHR tpl_push;
    PFN_vkCmdSetVertexInputEXT set_vi;
    PFN_vkCmdSetDepthClampEnableEXT set_clamp;
    PFN_vkCmdSetPolygonModeEXT set_poly;
    PFN_vkCmdSetColorBlendEnableEXT set_blend_en;
    PFN_vkCmdSetColorBlendEquationEXT set_blend_eq;
    PFN_vkCmdSetColorWriteMaskEXT set_wmask;
    PFN_vkCmdSetAlphaToCoverageEnableEXT set_a2c;
    char cache_path[PLAT_PATH_MAX * 2];
    int cache_unloaded;                // REKARINA_PIPELINE_CACHE_LOAD=0: the file was not handed to the driver; not saved
    int cache_fresh;                   // --warm-shaders on NVIDIA: the file is not loaded, only written (vk_pipeline_warm_shaders)
    uint64_t cache_bytes;              // the cache's size when it was loaded or last saved (autosave_in_play)
    VkPipeStats stats;
} P = {.lock = PLAT_MUTEX_INIT};

typedef struct Layout {
    struct Layout *next;
    uint64_t hash;
    VkDescriptorSetLayout set;
    VkPipelineLayout layout;
    int push;
    uint32_t n;
    VkDescriptorSetLayout set1;        // independent layouts: the fragment set (set 1); VK_NULL_HANDLE otherwise
    int push1;                         // set 1 is a push-descriptor set
    uint32_t n0;                       // independent layouts: b[0..n0) are set 0, the rest set 1
    VkPipelineLayout half[2];          // GPL libraries: {set 0, null}, {null, set 1} (made on first use)
    VkDescriptorUpdateTemplate tpl[2]; // per set (make_tpl); VK_NULL_HANDLE: written with write structs
    uint8_t pool;                      // stages reading the texture pool set (gfx.h GFX_POOL_SET): bit 0 the vertex
                                       // side (set 0's stages), bit 1 the fragment side of independent layouts
    uint8_t emu;                       // Bindings.emu: the push constant range of the mesh emulation's kernels
    VkDescriptorSetLayoutBinding b[];
} Layout;

typedef struct Pipe {
    struct Pipe *next;
    uint64_t hash;
    VkPipeInfo info;
    VkPipeBinding *bindings;
    int building;                      // a background job owns info.pipeline (under P.lock)
    struct Job *job;                   // that job while building
    const struct Layout *lay;          // graphics: the layout it was entered with
    VkPipeline fast;                   // GPL: the fast-linked pipeline (kept until shutdown: recorded draws use it)
    struct Pipe *alt;                  // a mesh pipeline building: its fast-linked stand-in, with a layout of its own
                                       // (mesh_fast; outside the table, kept until shutdown)
    uint8_t fast_tried;
    uint8_t alt_final;                 // mesh_fast: no stand-in will come (a library failed, or GPL cannot express it)
    uint8_t alt_waited;                // mesh_fast: a draw has waited for its libraries once
    uint8_t cs_used;                   // compute: a dispatch has asked for it (cs_use)
    size_t key_size;
    uint8_t key[];
} Pipe;

static Layout *layouts[LAYOUT_BUCKETS];
static Pipe *pipes[PIPE_BUCKETS];

static uint64_t fnv(uint64_t h, const void *p, size_t n) {
    const uint8_t *b = p;
    for (size_t i = 0; i < n; i++) h = (h ^ b[i]) * 0x100000001b3ull;
    return h;
}
#define FNV0 0xcbf29ce484222325ull

// Hash of a pipeline table key, 8 bytes per step: a draw hashes its whole GKey, and byte-wise FNV over it cost more
// than the lookup. Only the in-memory table uses it; the key file keeps fnv.
static uint64_t key_hash(const void *k, size_t n) { return nvn_hm_hash(k, (uint32_t)n); }

// The cache file lives next to rekarina.exe, so every build dir (the main build and each private fork) has its own:
// forks that shared build\v121's cache broke each other's runs (2026-10-22).
static void cache_path_init(void) {
    const char *cfg_path = cfg_get("REKARINA_PIPELINE_CACHE");
    if (cfg_path && *cfg_path && strlen(cfg_path) < sizeof P.cache_path) {
        strcpy(P.cache_path, cfg_path);
        return;
    }
    P.cache_path[0] = 0;
    char exe[PLAT_PATH_MAX * 2];
    int n = plat_exe_path(exe, sizeof exe);
    if (!n || n >= sizeof exe) return;
    char *cut = plat_last_sep(exe);
    if (!cut) return;
    *cut = 0;
    snprintf(P.cache_path, sizeof P.cache_path, "%s/pipeline-cache.bin", exe);
}

// Cache data whose header names another device or driver is dropped before Vulkan sees it.
// The file (626 MB after a full preparation) is handed to the driver through a read-only view of it, not a buffer it
// was read into: the driver keeps a copy of its own anyway, and the buffer was as much private memory again at the
// start (commit charge on a 16 GB PC, beside the file's pages the OS caches either way).
// DEBUG-PCLOAD: REKARINA_PIPELINE_CACHE_LOAD=0 starts with an empty cache and never writes the file (the pipelines come
// from the driver's own disk cache or are compiled), to see what the driver keeps of ours.
// The log says how large the file was and how long the driver took to take it in (a refused one starts empty).
static int warm_libraries;   // --warm-libraries (vk_pipeline_warm_libraries): pipeline-cache.bin is neither read nor written
static void cache_load(VkCtx *c) {
    cache_path_init();
    const void *data = NULL;
    uint64_t len = 0;
    size_t size = 0;
    P.cache_unloaded = !rt_knob("PIPELINE_CACHE_LOAD", "1") || warm_libraries;
    if (P.cache_unloaded) rt_log("vk_pipeline: %s not loaded and not saved (REKARINA_PIPELINE_CACHE_LOAD=0)", P.cache_path);
    if (P.cache_fresh && c->props.vendorID != 0x10de) P.cache_fresh = 0;
    if (P.cache_fresh)
        rt_log("vk_pipeline: %s not loaded: written new from what this run builds (vk_pipeline_warm_shaders)", P.cache_path);
    double t0 = (double)plat_now_ns() / 1e6;
    PlatView *v = P.cache_path[0] && !P.cache_unloaded && !P.cache_fresh ? plat_view_open(P.cache_path, &data, &len) : NULL;
    if (v && len > sizeof(VkPipelineCacheHeaderVersionOne) && len <= SIZE_MAX) {
        VkPipelineCacheHeaderVersionOne h;
        memcpy(&h, data, sizeof h);
        size = (size_t)len;
        if (h.headerVersion != VK_PIPELINE_CACHE_HEADER_VERSION_ONE || h.vendorID != c->props.vendorID ||
            h.deviceID != c->props.deviceID || memcmp(h.pipelineCacheUUID, c->props.pipelineCacheUUID, VK_UUID_SIZE)) {
            rt_log("vk_pipeline: %s was written by another device or driver; starting empty", P.cache_path);
            size = 0;
        }
    }
    // AMD's driver keeps a pipeline cache of at most about 850 MiB: the file of 1.2.4 to 1.2.7, which the setup's build
    // of every prepared stage library filled to that size on AMD, took nothing new, and every start built the entry
    // step's pipelines again (6,500 of them, four minutes on an RX 6800). No library goes into the file any more
    // (lb_warm_prepared); a file that large starts empty once, so that it holds the entry step's pipelines from then on.
    // pipeline-cache.reset beside it says it was done. REKARINA_PIPELINE_CACHE_RESET_MB sets the size (0: never).
    if (size && c->props.vendorID == 0x1002) {
        const char *rm = cfg_get("REKARINA_PIPELINE_CACHE_RESET_MB");
        uint64_t limit = (uint64_t)(rm && *rm ? atoi(rm) : 768) << 20;
        char mark[sizeof P.cache_path + 16];
        snprintf(mark, sizeof mark, "%.*s.reset", (int)(strlen(P.cache_path) - 4), P.cache_path);   // pipeline-cache.reset
        if (limit && size >= limit && !plat_file_exists(mark)) {
            rt_log("vk_pipeline: %s holds %.1f MiB, the AMD driver's limit: started empty once (%s)", P.cache_path,
                   size / 1048576.0, mark);
            size = 0;
            plat_file_write(mark, "1\n", 2);
        }
    }
    VkPipelineCacheCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO, .initialDataSize = size,
                                    .pInitialData = size ? data : NULL};
    P.cache_bytes = size;
    if (vkCreatePipelineCache(c->device, &ci, NULL, &P.cache) != VK_SUCCESS) {
        rt_log("vk_pipeline: the driver refused %s (%.1f MiB); starting empty", P.cache_path, size / 1048576.0);
        ci.initialDataSize = 0, ci.pInitialData = NULL, P.cache_bytes = 0;
        if (vkCreatePipelineCache(c->device, &ci, NULL, &P.cache) != VK_SUCCESS) P.cache = VK_NULL_HANDLE;
    }
    plat_view_close(v, 0);   // the driver copied what it keeps
    if (size)
        rt_log("vk_pipeline: pipeline cache of %.1f MiB handed to the driver in %.0f ms", size / 1048576.0,
               (double)plat_now_ns() / 1e6 - t0);
}

static void keys_open(void);
static void prebuild_start(void);
static void mk_open(void);      // mesh and geometry pipeline keys (at the mesh pipelines below)
static int skip_listed(const uint8_t iid[32]);   // variants the driver crashed on ("preparation that outlives a driver crash")
static void mk_flush(void);
static int mk_pending(void);

static int ensure_init(void) {
    if (P.ready) return 1;
    if (!vk_init()) return 0;
    plat_mutex_lock(&P.lock);
    if (P.ready) {
        plat_mutex_unlock(&P.lock);
        return 1;
    }
    VkCtx *c = vk_ctx();
    P.device = c->device;
    VkPhysicalDevicePushDescriptorPropertiesKHR pd = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &pd};
    vkGetPhysicalDeviceProperties2(c->phys, &p2);
    P.max_push = pd.maxPushDescriptors;
    P.indep = vk_caps.gpl;
    P.gpl = vk_caps.gpl;
    rt_log(P.gpl ? "vk_pipeline: graphics pipeline library on: a draw whose pipeline is building gets a fast-linked one, "
                   "the optimised build replaces it when done"
                 : "vk_pipeline: graphics pipeline library off: draws wait for or skip pipelines that are building");
    const char *ps = cfg_get("REKARINA_PUSH_STAGE"), *vo = cfg_get("REKARINA_VS_OUTS");
    P.push_stage = ps && !strcmp(ps, "fs") ? GFX_STAGE_FS : GFX_STAGE_VS;
    P.vs_outs_all = vo && !strcmp(vo, "all");
    if (P.indep) rt_log("vk_pipeline: stage-independent layouts: vertex set 0, fragment set 1, %s set pushed; vertex outputs: %s",
                        P.push_stage == GFX_STAGE_VS ? "vertex" : "fragment", P.vs_outs_all ? "all 32 locations" : "union of paired fragment inputs");
    // vk_device.c enables every feature the GPU reports in this chain, so the query tells what is enabled.
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT e3 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT};
    VkPhysicalDeviceProvokingVertexFeaturesEXT pvf = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROVOKING_VERTEX_FEATURES_EXT,
                                                      .pNext = c->has_eds3 ? &e3 : NULL};
    VkPhysicalDeviceVulkan13Features v13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES,
                                            .pNext = c->has_provoking_vertex ? (void *)&pvf : (c->has_eds3 ? (void *)&e3 : NULL)};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &v13};
    vkGetPhysicalDeviceFeatures2(c->phys, &f2);
    P.pv_last = c->has_provoking_vertex && pvf.provokingVertexLast;
    if (!P.pv_last) rt_log("vk_pipeline: provokingVertexLast unavailable: flat outputs come from the first vertex (NVN: last)");
    // Maxwell warps have 32 lanes; the translator assumes it (maxwell.md 3.4). Pin 32 where the driver allows it.
    P.f_full_subgroups = v13.computeFullSubgroups;
    if (v13.subgroupSizeControl) {
        VkPhysicalDeviceVulkan13Properties p13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
        VkPhysicalDeviceProperties2 q2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &p13};
        vkGetPhysicalDeviceProperties2(c->phys, &q2);
        P.sg32_stages = vk_caps.sg_pin_stages;   // vk_caps.c: 32 unless REKARINA_VK_SUBGROUP; 0 without the pin
        P.sg_fixed32 = p13.minSubgroupSize == 32 && p13.maxSubgroupSize == 32;
        rt_log("vk_pipeline: subgroup size %u-%u, requiredSubgroupSizeStages %#x, pinning %u on %#x", p13.minSubgroupSize,
               p13.maxSubgroupSize, (unsigned)p13.requiredSubgroupSizeStages, vk_caps.sg_size, (unsigned)P.sg32_stages);
    }
    P.f_logic_op = f2.features.logicOp;
    P.f_depth_clamp = f2.features.depthClamp;
    P.f_fill_non_solid = f2.features.fillModeNonSolid;
    P.f_dual_src = f2.features.dualSrcBlend;
#define PROC(field, name) (P.field = (void *)vkGetDeviceProcAddr(c->device, name))
    PROC(push, "vkCmdPushDescriptorSetKHR");
    // Descriptor update templates (core 1.1; the push variant comes with VK_KHR_push_descriptor on 1.1): the driver
    // walks a layout prepared when the layout is made instead of parsing write structs every draw (DXVK does so for the
    // same reason). REKARINA_DESC_TEMPLATE=0 keeps the write structs.
    if (!cfg_bool("REKARINA_DESC_TEMPLATE")) {
        rt_log("vk_pipeline: descriptor update templates off (REKARINA_DESC_TEMPLATE=0)");
    } else if (c->props.apiVersion >= VK_API_VERSION_1_1 && PROC(tpl_create, "vkCreateDescriptorUpdateTemplate") &&
               PROC(tpl_destroy, "vkDestroyDescriptorUpdateTemplate") && PROC(tpl_update, "vkUpdateDescriptorSetWithTemplate")) {
        P.tpl = 1;
        if (!PROC(tpl_push, "vkCmdPushDescriptorSetWithTemplateKHR"))
            rt_log("vk_pipeline: vkCmdPushDescriptorSetWithTemplateKHR missing: push sets are written with write structs");
    } else {
        rt_log("vk_pipeline: descriptor update templates unavailable: sets are written with write structs");
    }
    // REKARINA_VK_DYN_VERTEX=0/1 forces dynamic vertex input off or on. Default off on Intel's frozen Windows driver for its
    // older integrated GPUs (UHD 630, Ice Lake: Vulkan 1.3.215 and older), which draws nothing with it; LAS Recomp
    // turns it off there for the same reason.
    const char *dv = cfg_get("REKARINA_VK_DYN_VERTEX");   // no fixed default: it depends on the GPU, see above
    int intel_legacy = c->props.vendorID == 0x8086 && c->props.apiVersion <= VK_MAKE_API_VERSION(0, 1, 3, 215);
    int dv_on = dv && *dv ? strcmp(dv, "0") != 0 : !intel_legacy;
    P.dyn_vi = dv_on && c->has_vertex_input_dynamic_state && PROC(set_vi, "vkCmdSetVertexInputEXT");
    if (!dv_on && c->has_vertex_input_dynamic_state)
        rt_log("vk_pipeline: dynamic vertex input off (%s)", dv && *dv ? "REKARINA_VK_DYN_VERTEX=0" : "Intel's older Windows driver draws nothing with it");
    if (c->has_eds3) {
        P.dyn_clamp = e3.extendedDynamicState3DepthClampEnable && P.f_depth_clamp && PROC(set_clamp, "vkCmdSetDepthClampEnableEXT");
        P.dyn_poly = e3.extendedDynamicState3PolygonMode && PROC(set_poly, "vkCmdSetPolygonModeEXT");
        P.dyn_blend_en = e3.extendedDynamicState3ColorBlendEnable && PROC(set_blend_en, "vkCmdSetColorBlendEnableEXT");
        P.dyn_blend_eq = e3.extendedDynamicState3ColorBlendEquation && PROC(set_blend_eq, "vkCmdSetColorBlendEquationEXT");
        P.dyn_wmask = e3.extendedDynamicState3ColorWriteMask && PROC(set_wmask, "vkCmdSetColorWriteMaskEXT");
        P.dyn_a2c = e3.extendedDynamicState3AlphaToCoverageEnable && PROC(set_a2c, "vkCmdSetAlphaToCoverageEnableEXT");
    }
    // REKARINA_VK_DYN_A2C=0/1 forces dynamic alpha to coverage off or on. Default off on AMD's Windows driver branch for
    // RDNA 2 and older (vk_caps.amd_sc), which applies alpha to coverage while the dynamic state says off: on an RX 590
    // (24.9.1) TotK, which never turns it on, drew the hero and the NPCs with depth but no colour (black silhouettes) and
    // dithered other surfaces into checkerboards (issue #67, RX 470/570/580). Taken out of the dynamic state, it goes
    // into the pipeline key and the characters draw.
    const char *da = cfg_get("REKARINA_VK_DYN_A2C");   // no fixed default: it depends on the driver, see above
    int da_on = da && *da ? strcmp(da, "0") != 0 : !vk_caps.amd_sc;
    if (!da_on && P.dyn_a2c) {
        P.dyn_a2c = 0;
        rt_log("vk_pipeline: dynamic alpha to coverage off (%s)", da && *da ? "REKARINA_VK_DYN_A2C=0" : "AMD's driver for RDNA 2 and older");
    }
#undef PROC
    static const VkDynamicState core[] = {
        VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT, VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT, VK_DYNAMIC_STATE_DEPTH_BIAS,
        VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK,
        VK_DYNAMIC_STATE_STENCIL_REFERENCE, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE,
        VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY, VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE,
        VK_DYNAMIC_STATE_DEPTH_COMPARE_OP, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP,
        VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE, VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE,
        VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE};
    P.n_dyn = 0;
    for (size_t i = 0; i < sizeof core / sizeof *core; i++) P.dyn[P.n_dyn++] = core[i];
    P.dyn[P.n_dyn++] = P.dyn_vi ? VK_DYNAMIC_STATE_VERTEX_INPUT_EXT : VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE;
    if (P.dyn_clamp) P.dyn[P.n_dyn++] = VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT;
    if (P.dyn_poly) P.dyn[P.n_dyn++] = VK_DYNAMIC_STATE_POLYGON_MODE_EXT;
    if (P.dyn_blend_en) P.dyn[P.n_dyn++] = VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT;
    if (P.dyn_blend_eq) P.dyn[P.n_dyn++] = VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT;
    if (P.dyn_wmask) P.dyn[P.n_dyn++] = VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT;
    if (P.dyn_a2c) P.dyn[P.n_dyn++] = VK_DYNAMIC_STATE_ALPHA_TO_COVERAGE_ENABLE_EXT;
    cache_load(c);
    rt_log("vk_pipeline: max push descriptors %u; dynamic vertex input %d, depth clamp %d, polygon mode %d, blend enable %d, "
           "blend equation %d, write mask %d, alpha to coverage %d; cache %s",
           P.max_push, P.dyn_vi, P.dyn_clamp, P.dyn_poly, P.dyn_blend_en, P.dyn_blend_eq, P.dyn_wmask, P.dyn_a2c,
           P.cache_path[0] ? P.cache_path : "<memory only>");
    keys_open();
    mk_open();
    plat_xchg32(&P.ready, 1);
    plat_mutex_unlock(&P.lock);
    prebuild_start();
    return 1;
}

// ---- NVN -> Vulkan mappings (LAS Recomp exec.c blend_factor, blend_op, compare_fn, stencil_op)

static VkBlendFactor blend_factor(unsigned f) {
    switch (f) {
    case 1: return VK_BLEND_FACTOR_ZERO;
    case 2: return VK_BLEND_FACTOR_ONE;
    case 3: return VK_BLEND_FACTOR_SRC_COLOR;
    case 4: return VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR;
    case 5: return VK_BLEND_FACTOR_SRC_ALPHA;
    case 6: return VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
    case 7: return VK_BLEND_FACTOR_DST_ALPHA;
    case 8: return VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA;
    case 9: return VK_BLEND_FACTOR_DST_COLOR;
    case 10: return VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR;
    case 11: return VK_BLEND_FACTOR_SRC_ALPHA_SATURATE;
    case 0x10: return VK_BLEND_FACTOR_SRC1_COLOR;
    case 0x11: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR;
    case 0x12: return VK_BLEND_FACTOR_SRC1_ALPHA;
    case 0x13: return VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA;
    case 0x61: return VK_BLEND_FACTOR_CONSTANT_COLOR;
    case 0x62: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR;
    case 0x63: return VK_BLEND_FACTOR_CONSTANT_ALPHA;
    case 0x64: return VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA;
    default: LOG_ONCE("vk_pipeline: unknown NVN blend factor %#x, using ONE", f); return VK_BLEND_FACTOR_ONE;
    }
}

static VkBlendOp blend_op(unsigned e) {
    switch (e) {
    case 2: return VK_BLEND_OP_SUBTRACT;
    case 3: return VK_BLEND_OP_REVERSE_SUBTRACT;
    case 4: return VK_BLEND_OP_MIN;
    case 5: return VK_BLEND_OP_MAX;
    default: return VK_BLEND_OP_ADD;
    }
}

// NVN_DEPTH_FUNC_* / NVN_STENCIL_FUNC_* are NEVER = 1 .. ALWAYS = 8 in VkCompareOp order.
static VkCompareOp compare_op(unsigned f) { return f >= 1 && f <= 8 ? (VkCompareOp)(f - 1) : VK_COMPARE_OP_ALWAYS; }
// NVN_STENCIL_OP_* are KEEP = 1 .. DECR_WRAP = 8 in VkStencilOp order.
static VkStencilOp stencil_op(unsigned o) { return o >= 1 && o <= 8 ? (VkStencilOp)(o - 1) : VK_STENCIL_OP_KEEP; }
// NVN_POLYGON_MODE_POINT/LINE/FILL = 0/1/2, VkPolygonMode FILL/LINE/POINT = 0/1/2.
static VkPolygonMode polygon_mode(unsigned m) {
    if (!P.f_fill_non_solid || m > 2) return VK_POLYGON_MODE_FILL;
    return (VkPolygonMode)(2 - m);
}

VkPrimitiveTopology vk_pipeline_topology(int prim) {
    switch (prim) {
    case NVN_DRAW_PRIMITIVE_POINTS: return VK_PRIMITIVE_TOPOLOGY_POINT_LIST;
    case NVN_DRAW_PRIMITIVE_LINES: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST;
    case NVN_DRAW_PRIMITIVE_LINE_LOOP:
    case NVN_DRAW_PRIMITIVE_LINE_STRIP: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP;
    case NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP:
    case NVN_DRAW_PRIMITIVE_QUAD_STRIP: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP;   // same triangles, same coverage
    case NVN_DRAW_PRIMITIVE_TRIANGLE_FAN:
    case NVN_DRAW_PRIMITIVE_POLYGON: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN;
    case NVN_DRAW_PRIMITIVE_LINES_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_LINE_STRIP_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_TRIANGLES_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP_ADJACENCY: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY;
    case NVN_DRAW_PRIMITIVE_PATCHES: return VK_PRIMITIVE_TOPOLOGY_PATCH_LIST;
    default: return VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;   // TRIANGLES; QUADS after the replay's index rewrite
    }
}

// Topology class (a pipeline's static topology only fixes the class): 0 point, 1 line, 2 triangle, 3 patch.
static int topology_class(VkPrimitiveTopology t) {
    switch (t) {
    case VK_PRIMITIVE_TOPOLOGY_POINT_LIST: return 0;
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST:
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP:
    case VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY:
    case VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY: return 1;
    case VK_PRIMITIVE_TOPOLOGY_PATCH_LIST: return 3;
    default: return 2;
    }
}

int vk_pipeline_tex_type(int target) {
    switch (target) {
    case NVN_TEXTURE_TARGET_1D: return GFX_TEX_1D;
    case NVN_TEXTURE_TARGET_3D: return GFX_TEX_3D;
    case NVN_TEXTURE_TARGET_1D_ARRAY: return GFX_TEX_1D_ARRAY;
    case NVN_TEXTURE_TARGET_2D_ARRAY: return GFX_TEX_2D_ARRAY;
    case NVN_TEXTURE_TARGET_2D_MULTISAMPLE: return GFX_TEX_2D_MS;
    case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY: return GFX_TEX_2D_MS_ARRAY;
    case NVN_TEXTURE_TARGET_CUBEMAP: return GFX_TEX_CUBE;
    case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: return GFX_TEX_CUBE_ARRAY;
    case NVN_TEXTURE_TARGET_BUFFER: return GFX_TEX_BUFFER;
    default: return GFX_TEX_2D;   // 2D, RECTANGLE
    }
}

// Numeric type of a colour format as a shader sees it: GFX_TYPE_*.
static int format_type(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8B8_UINT: case VK_FORMAT_R8G8B8A8_UINT:
    case VK_FORMAT_R16_UINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16B16_UINT: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R32_UINT: case VK_FORMAT_R32G32_UINT: case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32A32_UINT:
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_B8G8R8A8_UINT: case VK_FORMAT_S8_UINT:
        return GFX_TYPE_UINT;
    case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_SINT: case VK_FORMAT_R8G8B8_SINT: case VK_FORMAT_R8G8B8A8_SINT:
    case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16_SINT: case VK_FORMAT_R16G16B16A16_SINT:
    case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32_SINT: case VK_FORMAT_R32G32B32A32_SINT:
    case VK_FORMAT_A2B10G10R10_SINT_PACK32: case VK_FORMAT_B8G8R8A8_SINT:
        return GFX_TYPE_SINT;
    default: return GFX_TYPE_FLOAT;
    }
}

// Optimal-tiling (o = 1) or buffer (o = 0) features of a core format, cached.
static VkFormatFeatureFlags format_features(VkFormat f, int optimal) {
    static VkFormatFeatureFlags known[2][256];
    static uint8_t have[2][256];
    VkCtx *c = vk_ctx();
    if (!c) return 0;
    if ((unsigned)f < 256 && have[optimal][f]) return known[optimal][f];
    VkFormatProperties p;
    vkGetPhysicalDeviceFormatProperties(c->phys, f, &p);
    VkFormatFeatureFlags r = optimal ? p.optimalTilingFeatures : p.bufferFeatures;
    if ((unsigned)f < 256) known[optimal][f] = r, have[optimal][f] = 1;
    return r;
}

// Vertex attribute formats by NVNformat (LAS Recomp exec.c vertex_format; MNG nvn_mtl.m vfmt for the TotK additions).
typedef struct { VkFormat vk; uint8_t type; } VtxFmt;
#define F(v) {VK_FORMAT_##v, GFX_TYPE_FLOAT}
#define U(v) {VK_FORMAT_##v, GFX_TYPE_UINT}
#define I(v) {VK_FORMAT_##v, GFX_TYPE_SINT}
static const VtxFmt vtx_formats[0x9B] = {
    [0x01] = F(R8_UNORM), [0x02] = F(R8_SNORM), [0x03] = U(R8_UINT), [0x04] = I(R8_SINT), [0x05] = F(R16_SFLOAT),
    [0x06] = F(R16_UNORM), [0x07] = F(R16_SNORM), [0x08] = U(R16_UINT), [0x09] = I(R16_SINT), [0x0A] = F(R32_SFLOAT),
    [0x0B] = U(R32_UINT), [0x0C] = I(R32_SINT), [0x0D] = F(R8G8_UNORM), [0x0E] = F(R8G8_SNORM), [0x0F] = U(R8G8_UINT),
    [0x10] = I(R8G8_SINT), [0x11] = F(R16G16_SFLOAT), [0x12] = F(R16G16_UNORM), [0x13] = F(R16G16_SNORM),
    [0x14] = U(R16G16_UINT), [0x15] = I(R16G16_SINT), [0x16] = F(R32G32_SFLOAT), [0x17] = U(R32G32_UINT),
    [0x18] = I(R32G32_SINT), [0x19] = F(R8G8B8_UNORM), [0x1A] = F(R8G8B8_SNORM), [0x1B] = U(R8G8B8_UINT),
    [0x1C] = I(R8G8B8_SINT), [0x1D] = F(R16G16B16_SFLOAT), [0x1E] = F(R16G16B16_UNORM), [0x1F] = F(R16G16B16_SNORM),
    [0x20] = U(R16G16B16_UINT), [0x21] = I(R16G16B16_SINT), [0x22] = F(R32G32B32_SFLOAT), [0x23] = U(R32G32B32_UINT),
    [0x24] = I(R32G32B32_SINT), [0x25] = F(R8G8B8A8_UNORM), [0x26] = F(R8G8B8A8_SNORM), [0x27] = U(R8G8B8A8_UINT),
    [0x28] = I(R8G8B8A8_SINT), [0x29] = F(R16G16B16A16_SFLOAT), [0x2A] = F(R16G16B16A16_UNORM),
    [0x2B] = F(R16G16B16A16_SNORM), [0x2C] = U(R16G16B16A16_UINT), [0x2D] = I(R16G16B16A16_SINT),
    [0x2E] = F(R32G32B32A32_SFLOAT), [0x2F] = U(R32G32B32A32_UINT), [0x30] = I(R32G32B32A32_SINT),
    [0x3D] = F(A2B10G10R10_UNORM_PACK32), [0x3E] = U(A2B10G10R10_UINT_PACK32), [0x3F] = F(B10G11R11_UFLOAT_PACK32),
    // *_UI2F / *_I2F: integer data read as float; the 8- and 16-bit ones are Vulkan's SCALED formats
    [0x51] = F(R8_USCALED), [0x52] = F(R8_SSCALED), [0x53] = F(R16_USCALED), [0x54] = F(R16_SSCALED),
    [0x57] = F(R8G8_USCALED), [0x58] = F(R8G8_SSCALED), [0x59] = F(R16G16_USCALED), [0x5A] = F(R16G16_SSCALED),
    [0x5D] = F(R8G8B8_USCALED), [0x5E] = F(R8G8B8_SSCALED), [0x5F] = F(R16G16B16_USCALED), [0x60] = F(R16G16B16_SSCALED),
    [0x63] = F(R8G8B8A8_USCALED), [0x64] = F(R8G8B8A8_SSCALED), [0x65] = F(R16G16B16A16_USCALED),
    [0x66] = F(R16G16B16A16_SSCALED),
    // 32-bit ones have no SCALED format: read as integers (values are not converted; logged when used)
    [0x55] = U(R32_UINT), [0x56] = I(R32_SINT), [0x5B] = U(R32G32_UINT), [0x5C] = I(R32G32_SINT),
    [0x61] = U(R32G32B32_UINT), [0x62] = I(R32G32B32_SINT), [0x67] = U(R32G32B32A32_UINT), [0x68] = I(R32G32B32A32_SINT),
    [0x69] = F(A2B10G10R10_SNORM_PACK32), [0x6A] = I(A2B10G10R10_SINT_PACK32), [0x6B] = F(A2B10G10R10_USCALED_PACK32),
    [0x6C] = F(A2B10G10R10_SSCALED_PACK32), [0x9A] = F(B8G8R8A8_UNORM),
};
#undef F
#undef U
#undef I

// 3-component 8/16-bit formats are not vertex formats on every GPU: read 4 components (LAS vertex_format_supported).
static VkFormat widen(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8G8B8_UNORM: return VK_FORMAT_R8G8B8A8_UNORM;
    case VK_FORMAT_R8G8B8_SNORM: return VK_FORMAT_R8G8B8A8_SNORM;
    case VK_FORMAT_R8G8B8_UINT: return VK_FORMAT_R8G8B8A8_UINT;
    case VK_FORMAT_R8G8B8_SINT: return VK_FORMAT_R8G8B8A8_SINT;
    case VK_FORMAT_R8G8B8_USCALED: return VK_FORMAT_R8G8B8A8_USCALED;
    case VK_FORMAT_R8G8B8_SSCALED: return VK_FORMAT_R8G8B8A8_SSCALED;
    case VK_FORMAT_R16G16B16_SFLOAT: return VK_FORMAT_R16G16B16A16_SFLOAT;
    case VK_FORMAT_R16G16B16_UNORM: return VK_FORMAT_R16G16B16A16_UNORM;
    case VK_FORMAT_R16G16B16_SNORM: return VK_FORMAT_R16G16B16A16_SNORM;
    case VK_FORMAT_R16G16B16_UINT: return VK_FORMAT_R16G16B16A16_UINT;
    case VK_FORMAT_R16G16B16_SINT: return VK_FORMAT_R16G16B16A16_SINT;
    case VK_FORMAT_R16G16B16_USCALED: return VK_FORMAT_R16G16B16A16_USCALED;
    case VK_FORMAT_R16G16B16_SSCALED: return VK_FORMAT_R16G16B16A16_SSCALED;
    default: return VK_FORMAT_UNDEFINED;
    }
}

VkFormat vk_vertex_format(int nvn_format, int *type) {
    if (type) *type = GFX_TYPE_FLOAT;
    if (nvn_format <= 0 || nvn_format >= (int)(sizeof vtx_formats / sizeof *vtx_formats)) return VK_FORMAT_UNDEFINED;
    const VtxFmt *v = &vtx_formats[nvn_format];
    if (v->vk == VK_FORMAT_UNDEFINED) return VK_FORMAT_UNDEFINED;
    if (type) *type = v->type;
    if ((nvn_format >= 0x55 && nvn_format <= 0x68) && v->type != GFX_TYPE_FLOAT)
        LOG_ONCE("vk_pipeline: 32-bit *_UI2F/*_I2F vertex formats are read as integers (first: %#x)", nvn_format);
    VkFormat f = v->vk;
    if (vk_ctx() && !(format_features(f, 0) & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT)) {
        VkFormat w = widen(f);
        if (w == VK_FORMAT_UNDEFINED || !(format_features(w, 0) & VK_FORMAT_FEATURE_VERTEX_BUFFER_BIT)) {
            LOG_ONCE("vk_pipeline: vertex format %d (NVN %#x) unsupported", (int)f, nvn_format);
            return VK_FORMAT_UNDEFINED;
        }
        LOG_ONCE("vk_pipeline: vertex format %d unsupported, read as 4 components (first: NVN %#x)", (int)f, nvn_format);
        f = w;
    }
    return f;
}

// Attachment format of an NVN format (tex_vk_format: the format the images are created with).
static VkFormat target_format(int nvn_format, int *depth, int *stencil) {
    TexVkFormat t;
    if (depth) *depth = 0;
    if (stencil) *stencil = 0;
    if (!nvn_format || !tex_vk_format(nvn_format, &t)) return VK_FORMAT_UNDEFINED;
    if (depth) *depth = t.depth;
    if (stencil) *stencil = t.stencil;
    return t.vk;
}

void vk_pipe_state_defaults(VkPipeState *s) {
    memset(s, 0, sizeof *s);
    s->samples = 1;
    s->prim = NVN_DRAW_PRIMITIVE_TRIANGLES;
    for (int i = 0; i < 8; i++)
        s->blend[i] = (blend_st){(uint64_t)i, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_FUNC_ONE,
                                 NVN_BLEND_FUNC_ZERO, NVN_BLEND_EQUATION_ADD, NVN_BLEND_EQUATION_ADD};
    s->color = (color_st){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_ALWAYS};
    s->chmask.mask = 0xffffffffu;
    s->ms.enable = 1;
    s->poly = (poly_st){NVN_FACE_NONE, NVN_FRONT_FACE_CCW, NVN_POLYGON_MODE_FILL, 0};
    s->ds = (ds_st){0, 1, NVN_DEPTH_FUNC_LESS, 0, NVN_STENCIL_FUNC_ALWAYS, NVN_STENCIL_FUNC_ALWAYS,
                    NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP,
                    NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP};
}

void vk_pipeline_answers(const VkPipeState *s, GfxShaderAnswers *vs, GfxShaderAnswers *fs) {
    if (vs)
        for (int i = 0; i < 16; i++) {
            va_st a;
            memcpy(&a, &s->vattr[i], sizeof a);
            int type = GFX_TYPE_FLOAT;
            if (a.fmt) vk_vertex_format((int)a.fmt, &type);
            vs->attr_type[i] = (uint8_t)type;
        }
    if (fs)
        for (int i = 0; i < 8; i++) {
            VkFormat f = target_format(s->color_fmt[i], NULL, NULL);
            fs->frag_out_type[i] = (uint8_t)(f == VK_FORMAT_UNDEFINED ? GFX_TYPE_FLOAT : format_type(f));
        }
}

// ---- vertex input (pipeline state, or vkCmdSetVertexInputEXT)

typedef struct {
    uint32_t n_attr, n_bind;
    VkVertexInputAttributeDescription2EXT a[32];
    VkVertexInputBindingDescription2EXT b[17];
} VtxInput;

// An attribute the game left without a format is a fixed one on Maxwell, a single 32-bit component of 0 (deko3d
// gpu_3d_vbo.cpp binds unused attributes as IsFixed 1x32 float): it reads (0, 0, 0, 1), as Vulkan expands a
// one-component format. TOTK's foliage shaders scale wind sway by 1 - colour.w; shapes without a vertex colour
// (Brightbloom stem and leaves) must not sway, or they bend away from the parts that have one (the bulb).
static VkFormat zero_format(int type) {
    return type == GFX_TYPE_UINT ? VK_FORMAT_R32_UINT : type == GFX_TYPE_SINT ? VK_FORMAT_R32_SINT : VK_FORMAT_R32_SFLOAT;
}

// Attributes for the Locations the vertex shader reads; one without a usable NVN attribute reads VK_PIPE_ZERO_BINDING.
// clear: the whole struct is zeroed first (for callers that hash or compare it whole); 0: only the counts, the entries
// in use are written in full (vk_pipeline_set_dynamic, 600 times a frame: 1.5 KiB each).
static void vertex_input_(const VkPipeState *s, VtxInput *v, int clear) {
    GfxShaderIo io[32];
    uint32_t used = gfx_shader_inputs(s->vs, io), streams = 0;
    int zero = 0;
    if (clear) memset(v, 0, sizeof *v);
    else v->n_attr = v->n_bind = 0;
    for (int l = 0; l < 32; l++) {
        if (!(used >> l & 1)) continue;
        VkVertexInputAttributeDescription2EXT *a = &v->a[v->n_attr++];
        *a = (VkVertexInputAttributeDescription2EXT){.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                                     .location = (uint32_t)l};
        va_st va = {0};
        if (l < 16) memcpy(&va, &s->vattr[l], sizeof va);
        VkFormat f = va.fmt ? vk_vertex_format((int)va.fmt, NULL) : VK_FORMAT_UNDEFINED;
        if (f != VK_FORMAT_UNDEFINED && va.stream < 16) {
            a->binding = va.stream, a->format = f, a->offset = va.offset;
            streams |= 1u << va.stream;
        } else {
            a->binding = VK_PIPE_ZERO_BINDING, a->format = zero_format(io[l].type);
            zero = 1;
        }
    }
    for (uint32_t b = 0; b < 16; b++) {
        if (!(streams >> b & 1)) continue;
        uint32_t div = s->vstream[b].divisor;
        if (div > 1) LOG_ONCE("vk_pipeline: instance divisor %u treated as 1 (no VK_EXT_vertex_attribute_divisor)", div);
        v->b[v->n_bind++] = (VkVertexInputBindingDescription2EXT){
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, .binding = b, .stride = s->vstream[b].stride,
            .inputRate = div ? VK_VERTEX_INPUT_RATE_INSTANCE : VK_VERTEX_INPUT_RATE_VERTEX, .divisor = 1};
    }
    if (zero)
        v->b[v->n_bind++] = (VkVertexInputBindingDescription2EXT){
            .sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, .binding = VK_PIPE_ZERO_BINDING,
            .stride = 0, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX, .divisor = 1};
}

static void vertex_input(const VkPipeState *s, VtxInput *v) { vertex_input_(s, v, 1); }

int vk_pipeline_vertex_input_dynamic(void) { return ensure_init() && P.dyn_vi; }

PFN_vkCmdPushDescriptorSetKHR vk_pipeline_push_fn(void) { return ensure_init() ? P.push : NULL; }
PFN_vkCmdPushDescriptorSetWithTemplateKHR vk_pipeline_push_tpl_fn(void) { return ensure_init() && P.tpl ? P.tpl_push : NULL; }
PFN_vkUpdateDescriptorSetWithTemplate vk_pipeline_update_tpl_fn(void) { return ensure_init() && P.tpl ? P.tpl_update : NULL; }

int vk_pipeline_set_indep(int on) {
    if (!ensure_init()) return 0;
    plat_mutex_lock(&P.lock);
    int was = P.indep;
    P.indep = on && vk_caps.gpl;
    plat_mutex_unlock(&P.lock);
    return was;
}

// ---- SPIR-V rewrites

// The ids a module decorates with DescriptorSet GFX_POOL_SET (gfx_shader.c patch_pool_arrays: the texture pool's
// arrays), at most cap; their set and bindings stay as they are when a stage's resources move.
static uint32_t spirv_pool_ids(const uint32_t *w, uint64_t n, uint32_t *ids, uint32_t cap) {
    uint32_t k = 0;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) break;
        if (op == SPV_OP_DECORATE && len >= 4 && w[i + 2] == SPV_DEC_DESCRIPTOR_SET && w[i + 3] == GFX_POOL_SET && k < cap)
            ids[k++] = w[i + 1];
        if (op == 54) break;   // OpFunction: decorations are all before it
        i += len;
    }
    return k;
}

static int id_in(uint32_t id, const uint32_t *ids, uint32_t n) {
    for (uint32_t i = 0; i < n; i++)
        if (ids[i] == id) return 1;
    return 0;
}

// Binding decorations b >= 1 become b + off (the fragment stage's share of set 0).
static uint32_t *spirv_rebind(const uint32_t *w, uint64_t n, uint32_t off) {
    uint32_t *o = malloc(n * 4), pool[16], np = spirv_pool_ids(w, n, pool, 16);
    if (!o) return NULL;
    memcpy(o, w, n * 4);
    for (uint64_t i = 5; i < n;) {
        uint32_t op = o[i] & 0xffff, len = o[i] >> 16;
        if (!len || i + len > n) break;
        if (op == SPV_OP_DECORATE && len >= 4 && o[i + 2] == SPV_DEC_BINDING && o[i + 3] >= 1 && !id_in(o[i + 1], pool, np))
            o[i + 3] += off;
        i += len;
    }
    return o;
}

// DescriptorSet decorations become set (the fragment stage's resources move to set 1 with independent layouts).
static uint32_t *spirv_set(const uint32_t *w, uint64_t n, uint32_t set) {
    uint32_t *o = malloc(n * 4);
    if (!o) return NULL;
    memcpy(o, w, n * 4);
    for (uint64_t i = 5; i < n;) {
        uint32_t op = o[i] & 0xffff, len = o[i] >> 16;
        if (!len || i + len > n) break;
        if (op == SPV_OP_DECORATE && len >= 4 && o[i + 2] == SPV_DEC_DESCRIPTOR_SET && o[i + 3] != GFX_POOL_SET) o[i + 3] = set;
        i += len;
    }
    return o;
}

// Dual-source blending (docs/maxwell.md 5.6): SRC1 blend factors read the fragment shader's colour output 1,
// which Vulkan takes from Location 0, Index 1. The Output variable at Location 1 moves there, as Ryujinx declares it
// with DualSourceBlendEnable (CodeGen/Spirv/Declarations.cs); the pipeline is keyed by GKey.dsb, so only pipelines
// that blend with SRC1 get this module. NULL when the module has no Output at Location 1.
static uint32_t *spirv_dual_source(const uint32_t *w, uint64_t n, uint64_t *out_n) {
    uint32_t outs[64];
    int nout = 0;
    uint64_t at = 0;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) return NULL;
        if (op == SPV_OP_VARIABLE && len >= 4 && w[i + 3] == SPV_SC_OUTPUT && nout < 64) outs[nout++] = w[i + 2];
        i += len;
    }
    for (uint64_t i = 5; i < n && !at;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == SPV_OP_DECORATE && len == 4 && w[i + 2] == SPV_DEC_LOCATION && w[i + 3] == 1)
            for (int k = 0; k < nout; k++)
                if (outs[k] == w[i + 1]) at = i;
        i += len;
    }
    if (!at) return NULL;
    uint32_t *o = malloc((n + 4) * 4);
    if (!o) return NULL;
    memcpy(o, w, (at + 4) * 4);
    o[at + 3] = 0;
    o[at + 4] = 4u << 16 | SPV_OP_DECORATE, o[at + 5] = w[at + 1], o[at + 6] = SPV_DEC_INDEX, o[at + 7] = 1;
    memcpy(o + at + 8, w + at + 4, (n - at - 4) * 4);
    *out_n = n + 4;
    return o;
}

// Adds an Output variable for each Location of `missing` (type and width from io) to a module, declared in the entry
// point interface and initialised as Maxwell reads an unwritten output: (0, 0, 0, 1), the w as the bits of 1.0f whatever
// the type (attribute memory is untyped); fewer than 4 components are all zero. Existing scalar, vector and pointer types
// are reused. NULL when the module does not have the expected layout.
static uint32_t *spirv_add_outputs(const uint32_t *w, uint64_t n, uint32_t missing, const GfxShaderIo io[32],
                                   uint64_t *out_n) {
    uint64_t ep_at = 0, ann_end = 0, fn_at = 0;
    uint32_t bound = w[3], scalar[3] = {0, 0, 0};          // GFX_TYPE_FLOAT, SINT, UINT
    struct { uint32_t id, comp, count; } vec[64];
    struct { uint32_t id, type; } ptr[64];
    int nvec = 0, nptr = 0;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) return NULL;
        const uint32_t *a = w + i + 1;
        if (op == SPV_OP_ENTRY_POINT && !ep_at) ep_at = i;
        else if (op == SPV_OP_DECORATE || op == 72 || op == 73 || op == 74 || op == 75 || op == 332 || op == 5632 || op == 5633)
            ann_end = i + len;
        else if (op == SPV_OP_TYPE_FLOAT && len >= 3 && a[1] == 32) scalar[GFX_TYPE_FLOAT] = a[0];
        else if (op == SPV_OP_TYPE_INT && len >= 4 && a[1] == 32) scalar[a[2] ? GFX_TYPE_SINT : GFX_TYPE_UINT] = a[0];
        else if (op == SPV_OP_TYPE_VECTOR && len >= 4 && nvec < 64) vec[nvec].id = a[0], vec[nvec].comp = a[1], vec[nvec++].count = a[2];
        else if (op == SPV_OP_TYPE_POINTER && len >= 4 && a[1] == SPV_SC_OUTPUT && nptr < 64) ptr[nptr].id = a[0], ptr[nptr++].type = a[2];
        else if (op == SPV_OP_FUNCTION) { fn_at = i; break; }
        i += len;
    }
    if (!ep_at || !ann_end || !fn_at || ann_end < ep_at || fn_at < ann_end) return NULL;
    uint32_t types[32 * 40], decs[32 * 4], vars[32], zero[3] = {0, 0, 0}, one[3] = {0, 0, 0};
    int nt = 0, nd = 0, nv = 0;
    for (int l = 0; l < 32; l++) {
        if (!(missing >> l & 1)) continue;
        int t = io[l].type <= GFX_TYPE_UINT ? io[l].type : GFX_TYPE_FLOAT;
        uint32_t comps = io[l].comps ? io[l].comps : 4;
        if (!scalar[t]) {
            scalar[t] = bound++;
            if (t == GFX_TYPE_FLOAT) types[nt++] = 3u << 16 | SPV_OP_TYPE_FLOAT, types[nt++] = scalar[t], types[nt++] = 32;
            else types[nt++] = 4u << 16 | SPV_OP_TYPE_INT, types[nt++] = scalar[t], types[nt++] = 32, types[nt++] = t == GFX_TYPE_SINT;
        }
        uint32_t type = scalar[t];
        if (comps > 1) {
            int k = 0;
            while (k < nvec && !(vec[k].comp == type && vec[k].count == comps)) k++;
            if (k == nvec) {
                if (nvec == 64) return NULL;
                vec[nvec].id = bound++, vec[nvec].comp = type, vec[nvec++].count = comps;
                types[nt++] = 4u << 16 | SPV_OP_TYPE_VECTOR, types[nt++] = vec[k].id, types[nt++] = type, types[nt++] = comps;
            }
            type = vec[k].id;
        }
        int k = 0;
        while (k < nptr && ptr[k].type != type) k++;
        if (k == nptr) {
            if (nptr == 64) return NULL;
            ptr[nptr].id = bound++, ptr[nptr++].type = type;
            types[nt++] = 4u << 16 | SPV_OP_TYPE_POINTER, types[nt++] = ptr[k].id, types[nt++] = SPV_SC_OUTPUT, types[nt++] = type;
        }
        uint32_t init = bound++, var = bound++;
        if (comps == 4) {
            if (!zero[t]) {
                zero[t] = bound++, one[t] = bound++;
                types[nt++] = 4u << 16 | SPV_OP_CONSTANT, types[nt++] = scalar[t], types[nt++] = zero[t], types[nt++] = 0;
                types[nt++] = 4u << 16 | SPV_OP_CONSTANT, types[nt++] = scalar[t], types[nt++] = one[t], types[nt++] = 0x3f800000;
            }
            types[nt++] = 7u << 16 | SPV_OP_CONSTANT_COMPOSITE, types[nt++] = type, types[nt++] = init;
            types[nt++] = zero[t], types[nt++] = zero[t], types[nt++] = zero[t], types[nt++] = one[t];
        } else {
            types[nt++] = 3u << 16 | SPV_OP_CONSTANT_NULL, types[nt++] = type, types[nt++] = init;
        }
        types[nt++] = 5u << 16 | SPV_OP_VARIABLE, types[nt++] = ptr[k].id, types[nt++] = var, types[nt++] = SPV_SC_OUTPUT,
        types[nt++] = init;
        decs[nd++] = 4u << 16 | SPV_OP_DECORATE, decs[nd++] = var, decs[nd++] = SPV_DEC_LOCATION, decs[nd++] = (uint32_t)l;
        vars[nv++] = var;
    }
    uint64_t total = n + (uint64_t)(nt + nd + nv);
    uint32_t *o = malloc(total * 4), *p = o;
    if (!o) return NULL;
    uint32_t ep_len = w[ep_at] >> 16;
    memcpy(p, w, 5 * 4), p[3] = bound, p += 5;
    memcpy(p, w + 5, (ep_at - 5) * 4), p += ep_at - 5;
    *p++ = (ep_len + (uint32_t)nv) << 16 | SPV_OP_ENTRY_POINT;
    memcpy(p, w + ep_at + 1, (ep_len - 1) * 4), p += ep_len - 1;
    memcpy(p, vars, (size_t)nv * 4), p += nv;
    memcpy(p, w + ep_at + ep_len, (ann_end - ep_at - ep_len) * 4), p += ann_end - ep_at - ep_len;
    memcpy(p, decs, (size_t)nd * 4), p += nd;
    memcpy(p, w + ann_end, (fn_at - ann_end) * 4), p += fn_at - ann_end;
    memcpy(p, types, (size_t)nt * 4), p += nt;
    memcpy(p, w + fn_at, (n - fn_at) * 4), p += n - fn_at;
    *out_n = (uint64_t)(p - o);
    return o;
}

// Repair declarations in translated shaders (NULL: nothing to change):
//  - the fragment shader interlock capability and extension, which Ryujinx adds to every fragment shader when the host
//    reports interlock support (SpirvGenerator.cs) and vk_device.c does not enable; kept when the shader interlocks;
//  - the CullDistance[1] member of gl_PerVertex next to ClipDistance[8] (Declarations.cs CreatePerVertexStructType):
//    9 distances exceed maxCombinedClipAndCullDistances (8 on the RX 9070). Ryujinx never accesses CullDistance; the
//    member goes when it is the struct's last and no access chain selects it;
//  - the capability required for the Rg16ui storage image used by the guest's edge-mask pass.
static uint32_t *spirv_fixup(const uint32_t *w, uint64_t n, uint64_t *out_n) {
    enum { OP_EXTENSION = 10, OP_CAPABILITY = 17, OP_MEMBER_NAME = 6, OP_EXECUTION_MODE = 16, OP_TYPE_ARRAY = 28,
           OP_TYPE_STRUCT = 30, OP_CONSTANT = 43, OP_MEMBER_DECORATE = 72, OP_ACCESS_CHAIN = 65, OP_IN_BOUNDS_ACCESS_CHAIN = 66,
           OP_BEGIN_INTERLOCK = 5364, OP_END_INTERLOCK = 5365, CAP_SAMPLE_INTERLOCK = 5363, CAP_PIXEL_INTERLOCK = 5378,
           CAP_RATE_INTERLOCK = 5372, BUILTIN_CLIP = 3, BUILTIN_CULL = 4, OP_TYPE_IMAGE = 25, IMAGE_RG16UI = 36,
           CAP_EXTENDED_STORAGE_FORMATS = 49 };
    if (n < 5) return NULL;
    uint32_t bound = w[3] < (1u << 22) ? w[3] : (1u << 22);
    uint32_t *val = calloc(bound, sizeof *val), *elem = calloc(bound, sizeof *elem), *len = calloc(bound, sizeof *len),
             *pointee = calloc(bound, sizeof *pointee), *var_type = calloc(bound, sizeof *var_type);
    uint32_t *o = NULL;
    int interlock_used = 0, interlock_declared = 0, cull_ok = 0;
    int rg16_storage = 0, extended_storage_declared = 0;
    uint32_t cull_struct = 0, cull_member = 0, clip_member = ~0u, members = 0, member_types[8] = {0};
    if (!val || !elem || !len || !pointee || !var_type) goto out;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, l = w[i] >> 16;
        if (!l || i + l > n) goto out;
        const uint32_t *a = w + i + 1;
        if (op == OP_BEGIN_INTERLOCK || op == OP_END_INTERLOCK) interlock_used = 1;
        else if (op == OP_EXECUTION_MODE && l >= 3 && a[1] >= 5366 && a[1] <= 5371) interlock_used = 1;
        else if (op == OP_CAPABILITY && l >= 2 && (a[0] == CAP_PIXEL_INTERLOCK || a[0] == CAP_SAMPLE_INTERLOCK || a[0] == CAP_RATE_INTERLOCK))
            interlock_declared = 1;
        else if (op == OP_CAPABILITY && l == 2 && a[0] == CAP_EXTENDED_STORAGE_FORMATS) extended_storage_declared = 1;
        else if (op == OP_TYPE_IMAGE && l >= 9 && a[6] == 2 && a[7] == IMAGE_RG16UI) rg16_storage = 1;
        else if (op == OP_CONSTANT && l == 4 && a[1] < bound) val[a[1]] = a[2];
        else if (op == OP_TYPE_ARRAY && l >= 4 && a[0] < bound) elem[a[0]] = a[1], len[a[0]] = a[2] < bound ? val[a[2]] : 0;
        else if (op == SPV_OP_TYPE_POINTER && l >= 4 && a[0] < bound) pointee[a[0]] = a[2];
        else if (op == SPV_OP_VARIABLE && l >= 4 && a[1] < bound) var_type[a[1]] = a[0] < bound ? pointee[a[0]] : 0;
        else if (op == OP_MEMBER_DECORATE && l >= 5 && a[2] == SPV_DEC_BUILTIN && a[3] == BUILTIN_CULL) cull_struct = a[0], cull_member = a[1];
        else if (op == OP_MEMBER_DECORATE && l >= 5 && a[2] == SPV_DEC_BUILTIN && a[3] == BUILTIN_CLIP) clip_member = a[1];
        else if (op == OP_TYPE_STRUCT && cull_struct && a[0] == cull_struct && l >= 2) {
            members = l - 2;
            for (uint32_t m = 0; m < members && m < 8; m++) member_types[m] = a[1 + m];
        } else if ((op == OP_ACCESS_CHAIN || op == OP_IN_BOUNDS_ACCESS_CHAIN) && l >= 5 && cull_struct && a[2] < bound) {
            uint32_t t = var_type[a[2]], k = 3;
            if (t < bound && t && elem[t] == cull_struct) k = 4;   // arrayed per-vertex block
            if ((t == cull_struct || k == 4) && k + 1 < l && a[k] < bound && val[a[k]] == cull_member) cull_struct = 0;
        }
        i += l;
    }
    VkCtx *c = vk_ctx();
    uint32_t limit = c ? c->props.limits.maxCombinedClipAndCullDistances : 8;
    if (cull_struct && members && members <= 8 && cull_member == members - 1 && clip_member < members) {
        uint32_t clip_t = member_types[clip_member], cull_t = member_types[cull_member];
        uint32_t clip_n = clip_t < bound ? len[clip_t] : 0, cull_n = cull_t < bound ? len[cull_t] : 0;
        cull_ok = clip_n + cull_n > limit && clip_n <= limit;
    }
    int strip = interlock_declared && !interlock_used;
    int add_extended = c && c->has_extended_storage_formats && rg16_storage && !extended_storage_declared;
    if (!strip && !cull_ok && !add_extended) goto out;
    o = malloc((n + 2 * add_extended) * 4);
    if (!o) goto out;
    uint64_t on = 5;
    memcpy(o, w, 5 * 4);
    if (add_extended) {
        o[on++] = 2u << 16 | OP_CAPABILITY;
        o[on++] = CAP_EXTENDED_STORAGE_FORMATS;
    }
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, l = w[i] >> 16;
        const uint32_t *a = w + i + 1;
        int drop = 0;
        if (strip && op == OP_CAPABILITY && (a[0] == CAP_PIXEL_INTERLOCK || a[0] == CAP_SAMPLE_INTERLOCK || a[0] == CAP_RATE_INTERLOCK)) drop = 1;
        if (strip && op == OP_EXTENSION && l >= 2 && !strncmp((const char *)a, "SPV_EXT_fragment_shader_interlock", (l - 1) * 4)) drop = 1;
        if (cull_ok && (op == OP_MEMBER_DECORATE || op == OP_MEMBER_NAME) && l >= 3 && a[0] == cull_struct && a[1] == cull_member) drop = 1;
        if (!drop) {
            memcpy(o + on, w + i, l * 4);
            if (cull_ok && op == OP_TYPE_STRUCT && a[0] == cull_struct) o[on] = (l - 1) << 16 | op;   // last member gone
            on += op == OP_TYPE_STRUCT && cull_ok && a[0] == cull_struct ? l - 1 : l;
        }
        i += l;
    }
    *out_n = on;
out:
    free(val), free(elem), free(len), free(pointee), free(var_type);
    return o;
}

// True when the module declares a GroupNonUniform* or SubgroupBallot-style capability (lane masks assume 32 lanes).
static int uses_subgroup_ops(const uint32_t *w, uint64_t words) {
    for (uint64_t i = 5; i < words;) {
        uint32_t op = w[i] & 0xffff, n = w[i] >> 16;
        if (!n || i + n > words) return 0;
        if (op != 17) return 0;   // OpCapability; capabilities come first, stop at the first other instruction
        uint32_t cap = w[i + 1];
        if ((cap >= 61 && cap <= 68) || cap == 4423 || cap == 4431) return 1;   // NonUniform*, SubgroupBallotKHR, SubgroupVoteKHR
        i += n;
    }
    return 0;
}

// Chain {size 32} into a stage create info when the stage uses subgroup ops and the driver allows the pin.
static void pin_subgroup32(VkPipelineShaderStageCreateInfo *st, VkPipelineShaderStageRequiredSubgroupSizeCreateInfo *rq,
                           const uint32_t *w, uint64_t words) {
    if (!(P.sg32_stages & st->stage) || !w || !uses_subgroup_ops(w, words)) return;
    *rq = (VkPipelineShaderStageRequiredSubgroupSizeCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, .requiredSubgroupSize = vk_caps.sg_size};
    st->pNext = rq;
}

// LocalSize.x of a compute module (OpExecutionMode, mode 17); 0: none found before the first function.
static uint32_t local_size_x(const uint32_t *w, uint64_t words) {
    for (uint64_t i = 5; w && i < words;) {
        uint32_t op = w[i] & 0xffff, n = w[i] >> 16;
        if (!n || i + n > words || op == 54) return 0;   // OpFunction: past the execution modes
        if (op == 16 && n >= 6 && w[i + 2] == 17) return w[i + 3];
        i += n;
    }
    return 0;
}

static VkShaderModule make_module(const uint32_t *w, uint64_t words) {
    uint64_t fixed_n = 0;
    uint32_t *fixed = spirv_fixup(w, words, &fixed_n);
    if (fixed) w = fixed, words = fixed_n;
    uint32_t *dbg = NULL;
    {   // DEBUG-SPVFMT: REKARINA_SPVFMT=<bytes>:<format>: a module of that size declares its format-less storage images with SPIR-V ImageFormat <format>
        const char *e = cfg_env("REKARINA_SPVFMT");
        unsigned bytes = 0, fmt = 0;
        if (e && sscanf(e, "%u:%u", &bytes, &fmt) == 2 && words * 4 == bytes && (dbg = malloc(words * 4))) {
            memcpy(dbg, w, words * 4);
            int nimg = 0;
            for (uint64_t i = 5; i < words;) {
                uint32_t op = dbg[i] & 0xffff, c = dbg[i] >> 16;
                if (!c || i + c > words) break;
                if (op == 17 && dbg[i + 1] == 55) dbg[i + 1] = 49;   // StorageImageReadWithoutFormat -> StorageImageExtendedFormats
                if (op == 25 && c >= 9 && dbg[i + 7] == 2 && dbg[i + 8] == 0) dbg[i + 8] = fmt, nimg++;
                i += c;
            }
            rt_log("SPVFMT: %d images of a %u-byte module declared with format %u", nimg, bytes, fmt);
            w = dbg;
        }
    }
    VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = (size_t)words * 4, .pCode = w};
    VkShaderModule m = VK_NULL_HANDLE;
    VkResult r = vkCreateShaderModule(P.device, &ci, NULL, &m);
    free(dbg);
    free(fixed);
    return r == VK_SUCCESS ? m : VK_NULL_HANDLE;
}

// ---- descriptor layouts

typedef struct {
    uint32_t n;
    VkDescriptorSetLayoutBinding b[MAX_BINDINGS];
    VkPipeBinding pb[MAX_BINDINGS];
    uint8_t pool;                      // a stage reads the texture pool set (gfx_shader_pool_mask)
    uint8_t emu;                       // a mesh emulation kernel (GFX_STAGE_MSC / TSC): the layout has VK_MESH_EMU_PUSH_BYTES
                                       // of push constants, the emulation buffer's address (compute_layout)
} Bindings;

static VkDescriptorType descriptor_type(int kind, const GfxShaderRes *r) {
    int buffer = (r->type & 0xff) == 2;
    switch (kind) {
    case GFX_RES_UBO: return VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    case GFX_RES_SSBO: return VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    case GFX_RES_TEX:
        if (buffer) return VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
        if (r->separate) LOG_ONCE("vk_pipeline: a separate texture handle: bound as SAMPLED_IMAGE");
        return r->separate ? VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE : VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    default: return buffer ? VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }
}

// ---- DEBUG-PIPESTATS: REKARINA_PIPESTATS=1: each new pipeline's statistics as the driver reports them
// (VK_KHR_pipeline_executable_properties, enabled by vk_device.c), one line per stage, keyed by the FNV-1a hash of the
// stage's SPIR-V (the hash REKARINA_DRAWTIME prints), each shader once.
static int pipestats_on(void) {
    static int on = -1;
    if (on < 0) on = cfg_get("REKARINA_PIPESTATS") != NULL;
    return on;
}
static uint64_t pipestats_fnv(const GfxShader *s) {
    if (!s) return 0;
    uint64_t w = 0, h = 0xcbf29ce484222325ull;
    const uint8_t *p = (const uint8_t *)gfx_shader_spirv(s, &w);
    for (uint64_t i = 0; p && i < w * 4; i++) h = (h ^ p[i]) * 0x100000001b3ull;
    return h;
}
static void pipestats_log(VkPipeline pipe, const GfxShader *const *sh, const VkShaderStageFlags *stage, int n) {
    if (!pipestats_on() || !pipe) return;
    static PFN_vkGetPipelineExecutablePropertiesKHR props;
    static PFN_vkGetPipelineExecutableStatisticsKHR stats;
    static PlatMutex lock = PLAT_MUTEX_INIT;
    static uint64_t seen[8192];
    static int nseen;
    if (!props) {
        props = (PFN_vkGetPipelineExecutablePropertiesKHR)vkGetDeviceProcAddr(P.device, "vkGetPipelineExecutablePropertiesKHR");
        stats = (PFN_vkGetPipelineExecutableStatisticsKHR)vkGetDeviceProcAddr(P.device, "vkGetPipelineExecutableStatisticsKHR");
        if (!props || !stats) return;
    }
    VkPipelineInfoKHR pinfo = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INFO_KHR, .pipeline = pipe};
    uint32_t ne = 0;
    if (props(P.device, &pinfo, &ne, NULL) != VK_SUCCESS || !ne) return;
    if (ne > 8) ne = 8;
    VkPipelineExecutablePropertiesKHR ep[8];
    for (uint32_t i = 0; i < ne; i++) ep[i] = (VkPipelineExecutablePropertiesKHR){.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_PROPERTIES_KHR};
    if (props(P.device, &pinfo, &ne, ep) != VK_SUCCESS) return;
    for (uint32_t e = 0; e < ne; e++) {
        uint64_t h = 0;
        for (int i = 0; i < n; i++)
            if (sh[i] && (stage[i] & ep[e].stages)) h = pipestats_fnv(sh[i]);
        plat_mutex_lock(&lock);
        int dup = 0;
        for (int i = 0; i < nseen && !dup; i++) dup = seen[i] == h;
        if (!dup && nseen < 8192) seen[nseen++] = h;
        plat_mutex_unlock(&lock);
        if (dup) continue;
        VkPipelineExecutableInfoKHR xi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_INFO_KHR, .pipeline = pipe, .executableIndex = e};
        uint32_t ns = 0;
        if (stats(P.device, &xi, &ns, NULL) != VK_SUCCESS || !ns) continue;
        if (ns > 32) ns = 32;
        VkPipelineExecutableStatisticKHR st[32];
        for (uint32_t i = 0; i < ns; i++) st[i] = (VkPipelineExecutableStatisticKHR){.sType = VK_STRUCTURE_TYPE_PIPELINE_EXECUTABLE_STATISTIC_KHR};
        if (stats(P.device, &xi, &ns, st) != VK_SUCCESS) continue;
        char line[2048];
        int o = snprintf(line, sizeof line, "PIPESTATS %s %016llx sg%u:", ep[e].name, (unsigned long long)h, ep[e].subgroupSize);
        for (uint32_t i = 0; i < ns && o < (int)sizeof line - 80; i++) {
            const VkPipelineExecutableStatisticKHR *s = &st[i];
            if (s->format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_BOOL32_KHR) o += snprintf(line + o, sizeof line - (size_t)o, " [%s]=%u", s->name, s->value.b32);
            else if (s->format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_INT64_KHR) o += snprintf(line + o, sizeof line - (size_t)o, " [%s]=%lld", s->name, (long long)s->value.i64);
            else if (s->format == VK_PIPELINE_EXECUTABLE_STATISTIC_FORMAT_UINT64_KHR) o += snprintf(line + o, sizeof line - (size_t)o, " [%s]=%llu", s->name, (unsigned long long)s->value.u64);
            else o += snprintf(line + o, sizeof line - (size_t)o, " [%s]=%.2f", s->name, s->value.f64);
        }
        rt_log("%s", line);
    }
}


static VkShaderStageFlags stage_bit(int stage) {
#ifdef GAME_NS2
    if (stage == GFX_STAGE_GS) return VK_SHADER_STAGE_GEOMETRY_BIT;
    if (stage == GFX_STAGE_TS) return VK_SHADER_STAGE_TASK_BIT_EXT;
    if (stage == GFX_STAGE_MS) return VK_SHADER_STAGE_MESH_BIT_EXT;
#endif
    return stage == GFX_STAGE_VS ? VK_SHADER_STAGE_VERTEX_BIT : stage == GFX_STAGE_FS ? VK_SHADER_STAGE_FRAGMENT_BIT
                                                                                      : VK_SHADER_STAGE_COMPUTE_BIT;
}

// One past the highest binding a shader uses (1 with no resources: binding 0 is the support buffer).
static uint32_t binding_span(const GfxShader *s) {
    uint32_t span = 1;
    for (int k = 0; k < GFX_RES_KINDS; k++) {
        const GfxShaderRes *r;
        int n = gfx_shader_resources(s, k, &r);
        for (int i = 0; i < n; i++)
            if (r[i].binding + (uint32_t)(r[i].count ? r[i].count : 1) > span) span = r[i].binding + (r[i].count ? r[i].count : 1);
    }
    return span;
}

// Adds a stage's resources with bindings b >= 1 moved by off; binding 0 (support buffer) is shared.
static int add_stage(Bindings *B, const GfxShader *s, uint32_t off) {
    int stage = gfx_shader_stage(s);
    VkShaderStageFlags bit = stage_bit(stage);
    if (gfx_shader_pool_mask(s)) B->pool = 1;
    if (!B->n) {
        B->b[0] = (VkDescriptorSetLayoutBinding){0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, 0, NULL};
        B->pb[0] = (VkPipeBinding){0, (uint8_t)stage, VK_PIPE_SUPPORT, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, 0, NULL};
        B->n = 1;
    }
    B->b[0].stageFlags |= bit;
    B->pb[0].stages |= bit;
    for (int k = 0; k < GFX_RES_KINDS; k++) {
        const GfxShaderRes *r;
        int n = gfx_shader_resources(s, k, &r);
        for (int i = 0; i < n; i++) {
            if (B->n == MAX_BINDINGS || !r[i].binding) return 0;
            VkDescriptorType t = descriptor_type(k, &r[i]);
            uint32_t binding = r[i].binding + off, count = r[i].count ? r[i].count : 1;
            B->b[B->n] = (VkDescriptorSetLayoutBinding){binding, t, count, bit, NULL};
            B->pb[B->n] = (VkPipeBinding){(uint16_t)binding, (uint8_t)stage, (uint8_t)k, t, (uint16_t)count, bit, &r[i]};
            B->n++;
        }
    }
    return 1;
}

// The first binding a module's resource variables use that the binding list b[0..n) lacks (module bindings >= 1 moved
// by off), or -1. A reflection that misses a resource would otherwise hand the driver an invalid layout.
// Only a variable the module's functions name counts: a layout has to hold what a shader statically uses, not what it
// declares. The translator declares the stage's first constant buffer (vp_c0, fp_c0: where the game keeps its storage
// buffers' addresses) whenever the shader has storage buffers, and lists it in the reflection only when the code reads
// it, which it does for the low bits of a storage buffer's offset where the device aligns such offsets (NVIDIA: 16
// bytes). On a device that does not (AMD's Radeon: 4), the buffer was declared, unread and unlisted, and every such
// pipeline was refused: 600 of them in the forest, 800 of 2,000 draws of a frame missing, the forest's ambient probe
// rendered with nothing but the sky in it and the picture washed out to white (2026-11-12).
static int module_binding_missing(const GfxShader *s, const VkDescriptorSetLayoutBinding *b, uint32_t n, uint32_t off) {
    uint64_t words = 0;
    const uint32_t *w = gfx_shader_spirv(s, &words);
    uint32_t pool[16], np = w ? spirv_pool_ids(w, words, pool, 16) : 0;
    uint32_t cand[32][2], nc = 0;   // variables with a binding the list lacks: id, binding
    uint64_t body = 0;              // the first OpFunction
    for (uint64_t i = 5; w && i < words;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > words) break;
        if (op == 54) { body = i; break; }   // OpFunction: decorations are all before it
        if (op == SPV_OP_DECORATE && len >= 4 && w[i + 2] == SPV_DEC_BINDING && !id_in(w[i + 1], pool, np)) {
            uint32_t want = w[i + 3] + (w[i + 3] >= 1 ? off : 0), j = 0;
            while (j < n && b[j].binding != want) j++;
            if (j == n) {
                if (nc == 32) return (int)w[i + 3];   // more than the table holds: as before
                cand[nc][0] = w[i + 1], cand[nc++][1] = w[i + 3];
            }
        }
        i += len;
    }
    if (!nc) return -1;
    if (!body) return (int)cand[0][1];
    // any operand word of the functions that equals the variable's id counts as a use (a literal of the same value would
    // too: the pipeline is then refused, as it was before)
    for (uint64_t i = body; i < words;) {
        uint32_t len = w[i] >> 16;
        if (!len || i + len > words) break;
        for (uint32_t k = 1; k < len; k++)
            for (uint32_t c = 0; c < nc; c++)
                if (w[i + k] == cand[c][0]) return (int)cand[c][1];
        i += len;
    }
    return -1;
}

// Checks both stages of a graphics pipeline against its layout; 0 with why filled when one uses a missing binding.
static int layout_covers(const VkPipeState *s, const Layout *l, uint32_t fs_off, char *why, size_t cap) {
    uint32_t nv = l->set1 ? l->n0 : l->n;
    int m = module_binding_missing(s->vs, l->b, nv, 0);
    const char *stage = "vertex";
    if (m < 0 && s->fs) {
        m = l->set1 ? module_binding_missing(s->fs, l->b + l->n0, l->n - l->n0, 0) : module_binding_missing(s->fs, l->b, l->n, fs_off);
        stage = "fragment";
    }
    if (m >= 0) snprintf(why, cap, "the %s shader uses binding %d, which its reflection and layout lack", stage, m);
    return m < 0;
}

// The update template of a set whose bindings are b[0..n): one entry per binding, in order, over the packed array the
// replay fills from the set's first descriptor, a VkDescSlot per descriptor. The bind point follows the stages (compute
// layouts hold the compute stage only). VK_NULL_HANDLE when templates are off, for an empty set, for a push set
// without vkCmdPushDescriptorSetWithTemplateKHR or when creation fails: the replay then writes the set with write
// structs.
static VkDescriptorUpdateTemplate make_tpl(const VkDescriptorSetLayoutBinding *b, uint32_t n, VkDescriptorSetLayout sl,
                                           int push, VkPipelineLayout pl, uint32_t set) {
    if (!P.tpl || !n || (push && !P.tpl_push)) return VK_NULL_HANDLE;
    VkDescriptorUpdateTemplateEntry e[MAX_BINDINGS];
    size_t off = 0;
    VkShaderStageFlags stages = 0;
    for (uint32_t i = 0; i < n; i++) {
        e[i] = (VkDescriptorUpdateTemplateEntry){b[i].binding, 0, b[i].descriptorCount, b[i].descriptorType, off, sizeof(VkDescSlot)};
        off += sizeof(VkDescSlot) * b[i].descriptorCount;
        stages |= b[i].stageFlags;
    }
    VkDescriptorUpdateTemplateCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO, .descriptorUpdateEntryCount = n,
        .pDescriptorUpdateEntries = e,
        .templateType = push ? VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR : VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET,
        .descriptorSetLayout = sl,
        .pipelineBindPoint = stages & VK_SHADER_STAGE_COMPUTE_BIT ? VK_PIPELINE_BIND_POINT_COMPUTE : VK_PIPELINE_BIND_POINT_GRAPHICS,
        .pipelineLayout = pl, .set = set};
    VkDescriptorUpdateTemplate t;
    if (P.tpl_create(P.device, &ci, NULL, &t) != VK_SUCCESS) {
        LOG_ONCE("vk_pipeline: descriptor update template creation failed: that set is written with write structs");
        return VK_NULL_HANDLE;
    }
    return t;
}

// Finds or creates the layout of a binding list (sorted here); NULL on failure. Caller holds P.lock.
// Sorts a binding list by binding and hashes it into *h; the descriptor total, or -1 when a binding repeats.
static int64_t bindings_sort(Bindings *B, uint64_t *h) {
    for (uint32_t i = 1; i < B->n; i++)                     // insertion sort by binding; the lists are short
        for (uint32_t j = i; j > 0 && B->b[j - 1].binding > B->b[j].binding; j--) {
            VkDescriptorSetLayoutBinding t = B->b[j]; B->b[j] = B->b[j - 1]; B->b[j - 1] = t;
            VkPipeBinding u = B->pb[j]; B->pb[j] = B->pb[j - 1]; B->pb[j - 1] = u;
        }
    int64_t total = 0;
    for (uint32_t i = 0; i < B->n; i++) {
        if (i && B->b[i].binding == B->b[i - 1].binding) return -1;
        uint32_t k[4] = {B->b[i].binding, B->b[i].descriptorType, B->b[i].descriptorCount, B->b[i].stageFlags};
        *h = fnv(*h, k, sizeof k);
        total += B->b[i].descriptorCount;
    }
    return total;
}

// The set layouts of a pipeline layout with the texture pool's set (gfx.h GFX_POOL_SET) after set0 and set1 (set1
// VK_NULL_HANDLE: vk_texpool.c's empty layout without independent sets, which take a null); the count.
static uint32_t with_pool(VkDescriptorSetLayout sets[3], VkDescriptorSetLayout set0, VkDescriptorSetLayout set1, int indep) {
    sets[0] = set0, sets[1] = set1 ? set1 : indep ? VK_NULL_HANDLE : vk_tpool_empty_layout(), sets[2] = vk_tpool_layout();
    return 3;
}

static Layout *layout_get(Bindings *B) {
    uint64_t h = FNV0;
    int64_t total = bindings_sort(B, &h);
    if (total < 0) return NULL;
    if (B->pool) h = fnv(h, "pool", 4);
    if (B->emu) h = fnv(h, "emu", 3);
    Layout **head = &layouts[h % LAYOUT_BUCKETS];
    for (Layout *l = *head; l; l = l->next)
        if (l->hash == h && l->n == B->n && !l->set1 && l->pool == B->pool && l->emu == B->emu &&
            !memcmp(l->b, B->b, B->n * sizeof *B->b))
            return l;
    if (B->pool && !vk_tpool_layout()) return NULL;
    Layout *l = calloc(1, sizeof *l + B->n * sizeof *B->b);
    if (!l) return NULL;
    l->hash = h, l->n = B->n, l->pool = B->pool, l->emu = B->emu;
    memcpy(l->b, B->b, B->n * sizeof *B->b);
    l->push = total <= (int64_t)P.max_push;
    VkDescriptorSetLayoutCreateInfo si = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .flags = l->push ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0,
                                          .bindingCount = l->n, .pBindings = l->b};
    if (vkCreateDescriptorSetLayout(P.device, &si, NULL, &l->set) != VK_SUCCESS) {
        free(l);
        return NULL;
    }
    VkDescriptorSetLayout sets[3];
    VkPipelineLayoutCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = &l->set};
    if (l->pool) pi.setLayoutCount = with_pool(sets, l->set, VK_NULL_HANDLE, 0), pi.pSetLayouts = sets;
    // the mesh emulation's kernels read the emulation buffer's address from push constants (translator patch 0047)
    static const VkPushConstantRange emu_range = {VK_SHADER_STAGE_COMPUTE_BIT, 0, VK_MESH_EMU_PUSH_BYTES};
    if (l->emu) pi.pushConstantRangeCount = 1, pi.pPushConstantRanges = &emu_range;
    if (vkCreatePipelineLayout(P.device, &pi, NULL, &l->layout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(P.device, l->set, NULL);
        free(l);
        return NULL;
    }
    l->tpl[0] = make_tpl(l->b, l->n, l->set, l->push, l->layout, 0);
    l->next = *head;
    *head = l;
    P.stats.layouts++;
    P.stats.push_layouts += (uint64_t)l->push;
    return l;
}

// Independent layouts (P.indep): the vertex stage's bindings in set 0, the fragment stage's in set 1 (empty without a
// fragment shader), each with its own support buffer at binding 0, the pipeline layout created with
// VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT so stage libraries can use it. Only the P.push_stage set is a
// push-descriptor set (a layout holds one), when it fits maxPushDescriptors; the other is allocated per draw from the
// command buffer's pools. On success *B holds both lists, set 0 first, VkPipeBinding.set filled in. Caller holds P.lock.
static Layout *layout_pair(Bindings *B, Bindings *F) {
    uint64_t h = fnv(FNV0, "indep", 5);
    int64_t tv = bindings_sort(B, &h);
    h = fnv(h, "|", 1);
    int64_t tf = bindings_sort(F, &h);
    if (tv < 0 || tf < 0 || B->n + F->n > MAX_BINDINGS) return NULL;
    uint8_t pool = (uint8_t)((B->pool ? 1 : 0) | (F->pool ? 2 : 0));
    if (pool) h = fnv(h, &pool, 1);
    if (pool && !vk_tpool_layout()) return NULL;
    uint32_t n0 = B->n, n = B->n + F->n;
    memcpy(B->b + n0, F->b, F->n * sizeof *F->b);
    memcpy(B->pb + n0, F->pb, F->n * sizeof *F->pb);
    for (uint32_t i = n0; i < n; i++) B->pb[i].set = 1;
    B->n = n;
    B->pool = pool != 0;
    Layout **head = &layouts[h % LAYOUT_BUCKETS];
    for (Layout *l = *head; l; l = l->next)
        if (l->hash == h && l->set1 && l->n == n && l->n0 == n0 && l->pool == pool && !memcmp(l->b, B->b, n * sizeof *B->b))
            return l;
    Layout *l = calloc(1, sizeof *l + n * sizeof *B->b);
    if (!l) return NULL;
    l->hash = h, l->n = n, l->n0 = n0, l->pool = pool;
    memcpy(l->b, B->b, n * sizeof *B->b);
    l->push = P.push_stage == GFX_STAGE_VS && tv <= (int64_t)P.max_push;
    l->push1 = P.push_stage == GFX_STAGE_FS && F->n && tf <= (int64_t)P.max_push;
    VkDescriptorSetLayoutCreateInfo s0 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .flags = l->push ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0,
                                          .bindingCount = n0, .pBindings = l->b};
    VkDescriptorSetLayoutCreateInfo s1 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .flags = l->push1 ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0,
                                          .bindingCount = n - n0, .pBindings = l->b + n0};
    if (vkCreateDescriptorSetLayout(P.device, &s0, NULL, &l->set) != VK_SUCCESS) {
        free(l);
        return NULL;
    }
    if (vkCreateDescriptorSetLayout(P.device, &s1, NULL, &l->set1) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(P.device, l->set, NULL);
        free(l);
        return NULL;
    }
    VkDescriptorSetLayout sets[3] = {l->set, l->set1};
    VkPipelineLayoutCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .flags = VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT, .setLayoutCount = 2,
                                     .pSetLayouts = sets};
    if (pool) pi.setLayoutCount = with_pool(sets, l->set, l->set1, 1);
    if (vkCreatePipelineLayout(P.device, &pi, NULL, &l->layout) != VK_SUCCESS) {
        vkDestroyDescriptorSetLayout(P.device, l->set, NULL);
        vkDestroyDescriptorSetLayout(P.device, l->set1, NULL);
        free(l);
        return NULL;
    }
    l->tpl[0] = make_tpl(l->b, n0, l->set, l->push, l->layout, 0);
    l->tpl[1] = make_tpl(l->b + n0, n - n0, l->set1, l->push1, l->layout, 1);
    l->next = *head;
    *head = l;
    P.stats.layouts++;
    P.stats.push_layouts += (uint64_t)(l->push | l->push1);
    return l;
}

// Vertex outputs with independent layouts: each vertex shader declares the union of the inputs of every fragment
// shader met with it (pipeline-keys.bin at the prewarm, then each pipeline built), so its module does not depend on the
// fragment shader of one pipeline. Unwritten ones read (0, 0, 0, 1) as before. REKARINA_VS_OUTS=all declares all 32.
#define VS_OUTS_CAP 8192
static struct {
    PlatMutex lock;
    const GfxShader *vs[VS_OUTS_CAP];
    uint32_t mask[VS_OUTS_CAP];
    GfxShaderIo io[VS_OUTS_CAP][32];
    uint32_t n, full;
} VO = {.lock = PLAT_MUTEX_INIT};

// Adds fs's inputs (packed ones excepted) to vs's union; returns the union, types in io (the first type met wins).
static uint32_t vs_outs(const GfxShader *vs, const GfxShader *fs, GfxShaderIo io[32]) {
    GfxShaderIo fin[32];
    uint32_t need = fs ? gfx_shader_inputs(fs, fin) : 0, mask;
    for (int i = 0; i < 32; i++)
        if (need >> i & 1 && fin[i].packed) need &= ~(1u << i);
    plat_mutex_lock(&VO.lock);
    uint32_t j = (uint32_t)(((uintptr_t)vs >> 4) * 0x9e3779b1u) & (VS_OUTS_CAP - 1);
    while (VO.vs[j] && VO.vs[j] != vs) j = (j + 1) & (VS_OUTS_CAP - 1);
    if (!VO.vs[j] && VO.n + 1 >= VS_OUTS_CAP * 3 / 4) {   // full: this pair alone
        VO.full++;
        plat_mutex_unlock(&VO.lock);
        LOG_ONCE("vk_pipeline: vertex output union table full: outputs from the pair");
        memset(io, 0, 32 * sizeof *io);
        for (int i = 0; i < 32; i++)
            if (need >> i & 1) io[i] = fin[i];
        return need;
    }
    if (!VO.vs[j]) VO.vs[j] = vs, VO.n++;
    for (int i = 0; i < 32; i++)
        if (!(need >> i & 1)) continue;
        else if (!(VO.mask[j] >> i & 1)) VO.io[j][i] = fin[i], VO.mask[j] |= 1u << i;
        else if (VO.io[j][i].type == fin[i].type && VO.io[j][i].comps < fin[i].comps) VO.io[j][i].comps = fin[i].comps;
    mask = VO.mask[j];
    memcpy(io, VO.io[j], 32 * sizeof *io);
    plat_mutex_unlock(&VO.lock);
    return mask;
}

// ---- pipeline cache table

static Pipe *pipe_find(const void *key, size_t size, uint64_t h) {
    for (Pipe *p = pipes[h % PIPE_BUCKETS]; p; p = p->next)
        if (p->hash == h && p->key_size == size && !memcmp(p->key, key, size)) return p;
    return NULL;
}

// A table entry for a result, not entered yet; NULL without memory.
static Pipe *pipe_new(const void *key, size_t size, uint64_t h, VkPipeline pipeline, const Layout *l, const Bindings *B,
                      VkPipelineBindPoint bp, uint32_t vs_inputs) {
    Pipe *p = calloc(1, sizeof *p + size);
    if (!p) return NULL;
    p->hash = h, p->key_size = size;
    memcpy(p->key, key, size);
    if (l && B) {
        p->bindings = malloc(B->n * sizeof *p->bindings);
        if (p->bindings) memcpy(p->bindings, B->pb, B->n * sizeof *p->bindings);
    }
    p->info = (VkPipeInfo){pipeline, l ? l->layout : VK_NULL_HANDLE, l ? l->set : VK_NULL_HANDLE, l ? l->push : 0, bp,
                           p->bindings ? B->n : 0, p->bindings, vs_inputs};
    p->info.set_layout1 = l ? l->set1 : VK_NULL_HANDLE;
    p->info.push1 = l ? l->push1 : 0;
    p->info.pool = l && l->pool;
    p->info.n_sets = l && l->set1 ? 2 : 1;
    if (l) p->info.tpl[0] = l->tpl[0], p->info.tpl[1] = l->tpl[1];
    p->lay = l;
    return p;
}

// Stores a result (a failed one too, so it is not retried every draw). Caller holds P.lock.
static Pipe *pipe_add(const void *key, size_t size, uint64_t h, VkPipeline pipeline, const Layout *l, const Bindings *B,
                      VkPipelineBindPoint bp, uint32_t vs_inputs) {
    Pipe *p = pipe_new(key, size, h, pipeline, l, B, bp, vs_inputs);
    if (!p) return NULL;
    Pipe **head = &pipes[h % PIPE_BUCKETS];
    p->next = *head;
    *head = p;
    return p;
}

// ---- graphics pipelines

typedef struct {                       // normalized: zero where the state is dynamic or unused
    const GfxShader *vs, *fs;
    uint32_t color[8], depth, stencil; // VkFormat
    uint8_t n_color, samples, topo_class, minus_w;
    uint8_t clamp, poly_mode, a2c, logic_op;   // logic_op: VkLogicOp + 1 when enabled
    uint8_t blend_en, dsb, fbl, pad;  // dsb: an enabled target blends with a SRC1 factor; fbl: VkPipeState.feedback
    uint32_t wmask;
    uint8_t blend[8][6];               // src, dst, op, src alpha, dst alpha, alpha op (VkBlendFactor / VkBlendOp)
    uint32_t attr_fmt[32], attr_off[32];
    uint8_t attr_bind[32];
    uint8_t rate[17];                  // per binding: 0 unused, 1 vertex, 2 instance
    uint8_t indep;                     // P.indep when built: the probe draws through both layouts
    uint8_t pad2[2];
} GKey;

static int blendable(VkFormat f) {
    return f != VK_FORMAT_UNDEFINED && (format_features(f, 1) & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT);
}

static int target_count(const VkPipeState *s) {
    int n = 0;
    for (int i = 0; i < 8; i++)
        if (s->color_fmt[i]) n = i + 1;
    return n;
}

// SRC1 blend factors read the fragment shader's second output (dual-source blending); the pipeline then needs a
// fragment module whose output 1 is Location 0, Index 1 (spirv_dual_source).
static int is_src1(unsigned f) { return f >= NVN_BLEND_FUNC_SRC1_COLOR && f <= NVN_BLEND_FUNC_ONE_MINUS_SRC1_ALPHA; }
// Runs per draw (gkey_fill, target_wmask): only the blend factors of blend-enabled targets are read until one is SRC1,
// which TotK never sets (0 of 876 pipelines in the field and fight saves).
static int uses_src1(const VkPipeState *s) {
    for (uint32_t m = s->color.blend_en & 0xffu; m; m &= m - 1) {
        int i = __builtin_ctz(m);
        const blend_st *b = &s->blend[i];
        if ((is_src1(b->src_rgb) || is_src1(b->dst_rgb) || is_src1(b->src_a) || is_src1(b->dst_a)) &&
            i < target_count(s) && blendable(target_format(s->color_fmt[i], NULL, NULL)))
            return 1;
    }
    return 0;
}

// Colour write mask of target i: NVN's channel mask, zero for a target outside the fragment shader's output map (SPH
// omap). Maxwell leaves such a target untouched (nxnative 04-graphics.md, "Targets the fragment shader does not
// write"); Vulkan writes undefined values, and on AMD they lit grass in front of the hero as flat lime. Zeros are wrong
// too: TOTK's half-res transparency pass binds its linear depth (R32F) as target 7, samples it in the same draws and
// writes only targets 0-2, 4 and 5; zeroed, the depth that later passes rebuild from it was 0 in tiles, and the fog
// over distant geometry showed rectangles of clear sky.
static uint32_t target_wmask(const VkPipeState *s, int i) {
    uint32_t m = s->chmask.mask >> (4 * i) & 0xf;
    if (s->fs && !(gfx_shader_outputs(s->fs, NULL) >> i & 1)) m = 0;
    if (i == 1 && s->fs && P.f_dual_src && uses_src1(s)) m = 0;   // output 1 is the second blend source (spirv_dual_source)
    return m;
}

uint32_t vk_pipeline_target_wmask(const VkPipeState *s, int i) { return target_wmask(s, i); }

static void gkey_fill(const VkPipeState *s, GKey *k) {
    memset(k, 0, sizeof *k);
    k->vs = s->vs, k->fs = s->fs;
    k->indep = (uint8_t)P.indep;
    k->n_color = (uint8_t)target_count(s);
    for (int i = 0; i < k->n_color; i++) k->color[i] = target_format(s->color_fmt[i], NULL, NULL);
    int d, st;
    VkFormat df = target_format(s->depth_fmt, &d, &st);
    k->depth = d ? df : VK_FORMAT_UNDEFINED;
    k->stencil = st ? df : VK_FORMAT_UNDEFINED;
    k->samples = s->samples > 1 ? s->samples : 1;
    k->topo_class = (uint8_t)topology_class(vk_pipeline_topology(s->prim));
    k->minus_w = s->depth_minus_w != 0;
    k->dsb = s->fs && P.f_dual_src && uses_src1(s);
    k->fbl = s->fs ? s->feedback : 0;
    if (!P.dyn_clamp) k->clamp = s->depth_clamp && P.f_depth_clamp;
    if (!P.dyn_poly) k->poly_mode = (uint8_t)polygon_mode(s->poly.mode);
    if (!P.dyn_a2c) k->a2c = s->ms.a2c != 0;
    if (P.f_logic_op && s->color.logic_op != NVN_LOGIC_OP_COPY) k->logic_op = (uint8_t)(s->color.logic_op + 1);
    for (int i = 0; i < k->n_color; i++) {
        int on = (s->color.blend_en >> i & 1) && blendable(k->color[i]);
        if (!P.dyn_blend_en && on) k->blend_en |= (uint8_t)(1u << i);
        if (!P.dyn_wmask) k->wmask |= target_wmask(s, i) << (4 * i);
        if (!P.dyn_blend_eq && on) {
            const blend_st *b = &s->blend[i];
            uint8_t *o = k->blend[i];
            o[0] = (uint8_t)blend_factor(b->src_rgb), o[1] = (uint8_t)blend_factor(b->dst_rgb), o[2] = (uint8_t)blend_op(b->eq_rgb);
            o[3] = (uint8_t)blend_factor(b->src_a), o[4] = (uint8_t)blend_factor(b->dst_a), o[5] = (uint8_t)blend_op(b->eq_a);
        }
    }
    if (!P.dyn_vi) {
        VtxInput v;
        vertex_input(s, &v);
        for (uint32_t i = 0; i < v.n_attr; i++) {
            uint32_t l = v.a[i].location;
            k->attr_fmt[l] = v.a[i].format, k->attr_off[l] = v.a[i].offset, k->attr_bind[l] = (uint8_t)v.a[i].binding;
        }
        for (uint32_t i = 0; i < v.n_bind; i++)
            k->rate[v.b[i].binding] = v.b[i].inputRate == VK_VERTEX_INPUT_RATE_INSTANCE ? 2 : 1;
    }
}

// Independent layouts: the outputs the vertex shader lacks from its union with s->fs's inputs added (vs_outs), types in
// fin. The vertex module (and its pre-rasterization library) depends on the vertex shader and this mask only.
static uint32_t vs_union_missing(const VkPipeState *s, GfxShaderIo fin[32]) {
    GfxShaderIo vout[32];
    uint32_t need = vs_outs(s->vs, s->fs, fin), have = gfx_shader_outputs(s->vs, vout), missing = 0;
    if (s->fs) {   // a location another fragment shader reads as another type: this pair's type (rare)
        GfxShaderIo fi[32];
        uint32_t fneed = gfx_shader_inputs(s->fs, fi);
        for (int i = 0; i < 32; i++)
            if (fneed >> i & 1 && !fi[i].packed &&
                (!(need >> i & 1) || fin[i].type != fi[i].type || fin[i].comps < fi[i].comps)) {
                if (need >> i & 1 && fin[i].type != fi[i].type)
                    LOG_ONCE("vk_pipeline: varying %d read as type %d and %d by fragment shaders of one vertex shader", i, fin[i].type, fi[i].type);
                fin[i] = fi[i], need |= 1u << i;
            }
    }
    if (P.vs_outs_all)
        for (int i = 0; i < 32; i++)
            if (!(need >> i & 1)) fin[i] = (GfxShaderIo){GFX_TYPE_FLOAT, 4, 0}, need |= 1u << i;
    for (int i = 0; i < 32; i++)
        if (need >> i & 1 && !(have >> i & 1)) missing |= 1u << i;
    return missing;
}

static VkPipeline create_graphics(const VkPipeState *s, const GKey *k, const Layout *l, uint32_t fs_off) {
    VkShaderModule vm = VK_NULL_HANDLE, fm = VK_NULL_HANDLE;
    uint64_t vw, fw;
    const uint32_t *vcode = gfx_shader_spirv(s->vs, &vw);
    uint32_t *vpatched = NULL, *fpatched = NULL;
    if (l->set1) {   // independent layouts: outputs from the vertex shader's union, not from this fragment shader
        GfxShaderIo fin[32];
        uint32_t missing = vs_union_missing(s, fin);
        if (missing) {
            uint64_t n2 = 0;
            vpatched = spirv_add_outputs(vcode, vw, missing, fin, &n2);
            if (!vpatched) LOG_ONCE("vk_pipeline: cannot add missing vertex outputs (mask %#x)", missing);
            else vcode = vpatched, vw = n2;
        }
    }
    if (s->fs) {
        GfxShaderIo fin[32], vout[32];
        uint32_t need = gfx_shader_inputs(s->fs, fin), have = gfx_shader_outputs(s->vs, vout), missing = 0;
        for (int i = 0; i < 32; i++) {
            if (!(need >> i & 1) || fin[i].packed) continue;
            if (!(have >> i & 1)) missing |= 1u << i;
            else if (vout[i].type != fin[i].type)
                LOG_ONCE("vk_pipeline: varying %d is type %d in the vertex stage, %d in the fragment stage", i, vout[i].type, fin[i].type);
        }
        if (missing && !l->set1) {
            uint64_t n2 = 0;
            vpatched = spirv_add_outputs(vcode, vw, missing, fin, &n2);
            if (!vpatched) LOG_ONCE("vk_pipeline: cannot add missing vertex outputs (mask %#x)", missing);
            else vcode = vpatched, vw = n2;
        }
        const uint32_t *fcode = gfx_shader_spirv(s->fs, &fw);
        if (k->dsb) {
            uint64_t n2 = 0;
            uint32_t *d = spirv_dual_source(fcode, fw, &n2);
            if (d) fcode = fpatched = d, fw = n2;
            else LOG_ONCE("vk_pipeline: SRC1 blend factors but the fragment shader writes no colour output 1");
        }
        if (l->set1 || fs_off) {
            uint32_t *r = !fcode ? NULL : l->set1 ? spirv_set(fcode, fw, 1) : spirv_rebind(fcode, fw, fs_off);
            free(fpatched);
            fcode = fpatched = r;
        }
        // A bound colour target outside the shader's output map gets write mask 0 (target_wmask)
        if (fcode) fm = make_module(fcode, fw);
    }
    vm = make_module(vcode, vw);
    free(vpatched), free(fpatched);
    VkPipeline pipe = VK_NULL_HANDLE;
    if (!vm || (s->fs && !fm)) goto out;

    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vm, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fm, .pName = "main"}};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rqv, rqf;
    uint64_t sw;
    const uint32_t *scode = gfx_shader_spirv(s->vs, &sw);   // vcode may be a patched copy that is freed by now
    pin_subgroup32(&stages[0], &rqv, scode, sw);
    if (s->fs) pin_subgroup32(&stages[1], &rqf, gfx_shader_spirv(s->fs, &sw), sw);
    VkVertexInputAttributeDescription va[32];
    VkVertexInputBindingDescription vb[17];
    uint32_t na = 0, nb = 0;
    if (!P.dyn_vi) {
        VtxInput v;
        vertex_input(s, &v);
        for (uint32_t i = 0; i < v.n_attr; i++) va[na++] = (VkVertexInputAttributeDescription){v.a[i].location, v.a[i].binding, v.a[i].format, v.a[i].offset};
        for (uint32_t i = 0; i < v.n_bind; i++) vb[nb++] = (VkVertexInputBindingDescription){v.b[i].binding, v.b[i].stride, v.b[i].inputRate};
    }
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                                               .vertexBindingDescriptionCount = nb, .pVertexBindingDescriptions = vb,
                                               .vertexAttributeDescriptionCount = na, .pVertexAttributeDescriptions = va};
    static const VkPrimitiveTopology class_base[4] = {VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                                                      VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_PATCH_LIST};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = class_base[k->topo_class]};
    VkPipelineViewportDepthClipControlCreateInfoEXT clip = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
                                                            .negativeOneToOne = k->minus_w};
    VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .pNext = &clip};
    // NVN's default is the last vertex (SetProvokingVertexMode, nvn_ext.c); Vulkan's is the first.
    VkPipelineRasterizationProvokingVertexStateCreateInfoEXT pvs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
        .provokingVertexMode = VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .pNext = P.pv_last ? &pvs : NULL,
                                                 .depthClampEnable = k->clamp, .polygonMode = (VkPolygonMode)k->poly_mode,
                                                 .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)k->samples,
                                               .alphaToCoverageEnable = k->a2c};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                                                .maxDepthBounds = 1.0f};
    VkPipelineColorBlendAttachmentState cba[8];
    memset(cba, 0, sizeof cba);
    for (int i = 0; i < k->n_color; i++) {
        const uint8_t *b = k->blend[i];
        cba[i].blendEnable = k->blend_en >> i & 1;
        cba[i].srcColorBlendFactor = (VkBlendFactor)b[0], cba[i].dstColorBlendFactor = (VkBlendFactor)b[1];
        cba[i].colorBlendOp = (VkBlendOp)b[2], cba[i].srcAlphaBlendFactor = (VkBlendFactor)b[3];
        cba[i].dstAlphaBlendFactor = (VkBlendFactor)b[4], cba[i].alphaBlendOp = (VkBlendOp)b[5];
        cba[i].colorWriteMask = k->wmask >> (4 * i) & 0xf;
    }
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .logicOpEnable = k->logic_op != 0,
                                              .logicOp = k->logic_op ? (VkLogicOp)(k->logic_op - 1) : VK_LOGIC_OP_COPY,
                                              .attachmentCount = k->n_color, .pAttachments = cba};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = P.n_dyn, .pDynamicStates = P.dyn};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = k->n_color,
                                        .pColorAttachmentFormats = (const VkFormat *)k->color,
                                        .depthAttachmentFormat = (VkFormat)k->depth, .stencilAttachmentFormat = (VkFormat)k->stencil};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
                                       .stageCount = s->fs ? 2 : 1, .pStages = stages, .pVertexInputState = &vi,
                                       .pInputAssemblyState = &ia, .pViewportState = &vp, .pRasterizationState = &rs,
                                       .pMultisampleState = &ms, .pDepthStencilState = &ds, .pColorBlendState = &cb,
                                       .pDynamicState = &dy, .layout = l->layout};
    if (k->fbl & 1) pi.flags |= VK_PIPELINE_CREATE_COLOR_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
    if (k->fbl & 2) pi.flags |= VK_PIPELINE_CREATE_DEPTH_STENCIL_ATTACHMENT_FEEDBACK_LOOP_BIT_EXT;
    if (pipestats_on()) pi.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;   // DEBUG-PIPESTATS
    uint64_t p0 = perf_begin();
    VkResult r = vkCreateGraphicsPipelines(P.device, P.cache, 1, &pi, NULL, &pipe);
    perf_end(PERF_PIPE, p0);   // counts on the render thread only, so background builds stay out of it
    if (r == VK_SUCCESS && pipestats_on()) {
        const GfxShader *ps[2] = {s->vs, s->fs};
        VkShaderStageFlags pst[2] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT};
        pipestats_log(pipe, ps, pst, 2);
    }
    if (r != VK_SUCCESS) {
        rt_log("vk_pipeline: vkCreateGraphicsPipelines failed (%d)", (int)r);
        pipe = VK_NULL_HANDLE;
    } else {
        perf_count(PERF_CNT_PIPE_BUILT);
    }
out:
    if (vm) vkDestroyShaderModule(P.device, vm, NULL);
    if (fm) vkDestroyShaderModule(P.device, fm, NULL);
    return pipe;
}

// ---- graphics pipeline library (vk_caps.gpl)
// A pipeline is linked from four libraries, each cached by the part of the pipeline key it depends on: vertex input
// (topology class, vertex attributes without dynamic vertex input), pre-rasterization (vertex shader, the outputs its
// union adds, static rasterization state), fragment shader (fragment shader, dual-source rewrite, samples) and fragment
// output (targets, blend, samples). With independent layouts each stage's set depends on its shader alone, so the
// pre-rasterization library uses {set 0, null} and the fragment shader library {null, set 1}. The fast link runs on the
// thread of the draw that needs it (no link-time optimisation); the optimised pipeline is the job pipe_miss queued
// (a monolithic build with the same modules), which replaces the fast one in the Pipe when done. Libraries and fast
// pipelines live until vk_pipeline_shutdown.
// Background build signalling, both with P.lock: work_cv wakes the pipeline workers, done_cv wakes threads waiting for a
// pipeline or a stage library to be published.
static PlatCond work_cv = PLAT_COND_INIT, done_cv = PLAT_COND_INIT;
#define LIB_BUCKETS 1024
enum { LIB_VI, LIB_PR, LIB_FS, LIB_FO, LIB_PRM };   // LIB_PRM: the task and mesh stages' pre-rasterization library
typedef struct Lib { struct Lib *next; uint64_t hash; VkPipeline pipe; uint32_t size; uint8_t pre, used, building, pad;
                     struct LibJob *job; uint8_t key[]; } Lib;
// pre: built by the boot prebuild ("stage library prebuild" below); used: a draw's link found it; building: a thread
// builds it or job holds it queued (pipe not set yet); entries live until libs_destroy
static Lib *libs[LIB_BUCKETS];

typedef struct { uint32_t kind, topo_class, attr_fmt[32], attr_off[32]; uint8_t attr_bind[32], rate[17], pad[3]; } ViKey;
typedef struct { uint32_t kind, missing; const GfxShader *vs; uint8_t clamp, poly_mode, minus_w, pool; GfxShaderIo fin[32]; } PrKey;
// nopin: no subgroup size pinned (a mesh pipeline's fragment stage where subgroups come in more than one size, as
// stages_create builds it); 0 for every ordinary draw's
typedef struct { uint32_t kind; uint8_t dsb, samples, a2c, nopin, pool; const GfxShader *fs; } FsKey;
typedef struct { uint32_t kind; uint8_t n_color, samples, a2c, logic_op, blend_en, pad[3]; uint32_t color[8], depth, stencil, wmask;
                 uint8_t blend[8][6]; } FoKey;
// A mesh pipeline's pre-rasterization library: its task shader (NULL: none) and mesh shader, bound as in stages_layout
// (task bindings from 0, the mesh shader's after them, both in set 0), and the static rasterization state.
typedef struct { uint32_t kind; uint8_t clamp, poly_mode, minus_w, samples, pool; const GfxShader *ts, *ms; } PrmKey;

static uint64_t qpc(void) { return plat_now_ns(); }
static double ms_since(uint64_t t0) { return (double)(qpc() - t0) / 1e6; }

static VkPipelineLayout half_layout(Layout *l, int h) {
    if (l->half[h]) return l->half[h];
    // The shared pool's bindings are visible to both stage subsets, so both libraries must declare it when either uses it.
    VkDescriptorSetLayout sets[3] = {h ? VK_NULL_HANDLE : l->set, h ? l->set1 : VK_NULL_HANDLE, vk_tpool_layout()};
    VkPipelineLayoutCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .flags = VK_PIPELINE_LAYOUT_CREATE_INDEPENDENT_SETS_BIT_EXT,
                                     .setLayoutCount = l->pool ? 3 : 2, .pSetLayouts = sets};
    if (vkCreatePipelineLayout(P.device, &pi, NULL, &l->half[h]) != VK_SUCCESS) l->half[h] = VK_NULL_HANDLE;
    return l->half[h];
}

// The pipeline cache a build goes into: pipeline-cache.bin's (P.cache), or none while this thread builds for the
// preparation on NVIDIA (lb_warm_worker). The preparation (--prepare-shaders, and --warm-shaders in the game folder at
// the setup) builds every prepared variant's stage library and compute pipeline once, so that the driver's own disk
// cache (nv-cache, kept per executable name) holds them. Built through P.cache they went into pipeline-cache.bin as
// well: some 56,000 entries play never asks the file for, since a draw takes a library missing from it from the
// driver's disk cache in about a millisecond (the note above mesh_gpl_on) and the entry step's pipelines are in the
// file anyway. Releases to 1.2.3 read the file with a 32-bit size and dropped one over 2 GiB, so their first start after
// a setup built the entry step without it and its exit wrote the small file; 1.2.4 loads it. In a copy of a 1.2.4
// game folder (RTX 5070, 2026-11-20) the setup's --warm-shaders wrote 3,057,551,174 bytes in 42 s with them and
// 626,675,543 in 16 s without; a 30 s start with the large file held 22.2 GB of commit and 17.0 GB of working set in
// play against 19.2 and 14.6, reached the game in 4.0-4.5 s against 2.1, and its exit wrote the 2.9 GiB again (1.3-1.4 s
// of the driver's copy, 5.5-5.9 s of file); with either file the entry step built all 10,847 recorded pipelines from it
// in 5.5-6.4 s of thread time and play built and translated nothing. Other drivers keep their own disk cache to a
// limited size (AMD's evicted a prepared build folder's libraries), so there the file still holds the libraries.
// REKARINA_WARM_APP_CACHE=1 keeps them in it on NVIDIA too.
static _Thread_local int build_no_app_cache;
static VkPipelineCache build_cache(void) { return build_no_app_cache ? VK_NULL_HANDLE : P.cache; }

// Builds one library from the pipeline's state; key is the library's key (PrKey: fin holds the added outputs).
static VkPipeline lib_create(int kind, const VkPipeState *s, const GKey *k, VkPipelineLayout lay, const void *key) {
    // indexed by kind: every kind needs its entry (LIB_PRM read past the end of a 4-entry table, and NVIDIA's driver
    // crashed on the garbage library flags)
    static const VkGraphicsPipelineLibraryFlagsEXT part[] = {
        VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT,
        VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT,
        VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT};
    _Static_assert(sizeof part / sizeof *part == LIB_PRM + 1, "a library flag per kind");
    VkGraphicsPipelineLibraryCreateInfoEXT gl = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT, .flags = part[kind]};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .pNext = &gl};
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = P.n_dyn, .pDynamicStates = P.dyn};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
                                       .flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR, .pDynamicState = &dy};
    VkShaderModule m = VK_NULL_HANDLE;
    VkPipelineShaderStageCreateInfo st = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pName = "main"};
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rq;
    VkVertexInputAttributeDescription va[32];
    VkVertexInputBindingDescription vb[17];
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    static const VkPrimitiveTopology class_base[4] = {VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                                                      VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_PATCH_LIST};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = class_base[k->topo_class]};
    VkPipelineViewportDepthClipControlCreateInfoEXT clip = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
                                                            .negativeOneToOne = k->minus_w};
    VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .pNext = &clip};
    VkPipelineRasterizationProvokingVertexStateCreateInfoEXT pvs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
        .provokingVertexMode = VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .pNext = P.pv_last ? &pvs : NULL, .depthClampEnable = k->clamp,
                                                 .polygonMode = (VkPolygonMode)k->poly_mode, .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)k->samples,
                                               .alphaToCoverageEnable = k->a2c};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                                                .maxDepthBounds = 1.0f};
    VkPipelineColorBlendAttachmentState cba[8];
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    uint32_t *patched = NULL;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkShaderModule mm[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkPipelineShaderStageCreateInfo mst[2];
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo mrq[2];
    VkDynamicState mdyn[32];
    if (kind == LIB_PRM) {
        // task and mesh modules as stages_create makes them (the same bindings, the same subgroup pin), none of the
        // vertex input state a mesh pipeline must not name as dynamic, and no fragment shading rate: the fragment
        // shader library is the one ordinary draws share, which has none (the optimised build sets the rate)
        const PrmKey *mk = key;
        const GfxShader *ss[2] = {mk->ts, mk->ms};
        uint32_t off = 0, n = 0;
        for (int i = 0; i < 2; i++) {
            if (!ss[i]) continue;
            uint64_t w;
            const uint32_t *code = gfx_shader_spirv(ss[i], &w);
            uint32_t *moved = off ? spirv_rebind(code, w, off) : NULL;
            if (off && !moved) goto out;
            mm[i] = make_module(moved ? moved : code, w);
            free(moved);
            if (!mm[i]) goto out;
            mst[n] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                       .stage = i ? VK_SHADER_STAGE_MESH_BIT_EXT : VK_SHADER_STAGE_TASK_BIT_EXT,
                                                       .module = mm[i], .pName = "main"};
            if (vk_caps.sg_pin_mesh) pin_subgroup32(&mst[n], &mrq[n], code, w);
            n++;
            off += binding_span(ss[i]) - 1;
        }
        uint32_t nd = 0;
        for (uint32_t i = 0; i < P.n_dyn; i++)
            if (P.dyn[i] != VK_DYNAMIC_STATE_VERTEX_INPUT_EXT && P.dyn[i] != VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE &&
                P.dyn[i] != VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY && P.dyn[i] != VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE)
                mdyn[nd++] = P.dyn[i];
        dy.dynamicStateCount = nd, dy.pDynamicStates = mdyn;
        pi.stageCount = n, pi.pStages = mst, pi.pViewportState = &vp, pi.pRasterizationState = &rs;
        pi.layout = lay;
    } else if (kind == LIB_VI) {
        if (!P.dyn_vi) {
            VtxInput v;
            vertex_input(s, &v);
            for (uint32_t i = 0; i < v.n_attr; i++)
                va[i] = (VkVertexInputAttributeDescription){v.a[i].location, v.a[i].binding, v.a[i].format, v.a[i].offset};
            for (uint32_t i = 0; i < v.n_bind; i++)
                vb[i] = (VkVertexInputBindingDescription){v.b[i].binding, v.b[i].stride, v.b[i].inputRate};
            vi.vertexAttributeDescriptionCount = v.n_attr, vi.pVertexAttributeDescriptions = va;
            vi.vertexBindingDescriptionCount = v.n_bind, vi.pVertexBindingDescriptions = vb;
        }
        pi.pVertexInputState = &vi, pi.pInputAssemblyState = &ia;
    } else if (kind == LIB_PR) {
        const PrKey *pk = key;
        uint64_t vw, sw, n2 = 0;
        const uint32_t *vcode = gfx_shader_spirv(s->vs, &vw), *scode = vcode;
        sw = vw;
        if (pk->missing && (patched = spirv_add_outputs(vcode, vw, pk->missing, pk->fin, &n2))) vcode = patched, vw = n2;
        else if (pk->missing) LOG_ONCE("vk_pipeline: cannot add missing vertex outputs (mask %#x)", pk->missing);
        if (!(m = make_module(vcode, vw))) goto out;
        st.stage = VK_SHADER_STAGE_VERTEX_BIT, st.module = m;
        pin_subgroup32(&st, &rq, scode, sw);
        pi.stageCount = 1, pi.pStages = &st, pi.pViewportState = &vp, pi.pRasterizationState = &rs;
        pi.layout = lay;
    } else if (kind == LIB_FS) {
        if (s->fs) {
            uint64_t fw, sw, n2 = 0;
            const uint32_t *fcode = gfx_shader_spirv(s->fs, &fw), *scode = fcode;
            sw = fw;
            uint32_t *d = NULL;
            if (k->dsb && fcode) {
                if ((d = spirv_dual_source(fcode, fw, &n2))) fcode = d, fw = n2;
                else LOG_ONCE("vk_pipeline: SRC1 blend factors but the fragment shader writes no colour output 1");
            }
            patched = fcode ? spirv_set(fcode, fw, 1) : NULL;
            free(d);
            if (!patched || !(m = make_module(patched, fw))) goto out;
            st.stage = VK_SHADER_STAGE_FRAGMENT_BIT, st.module = m;
            if (!((const FsKey *)key)->nopin) pin_subgroup32(&st, &rq, scode, sw);
            pi.stageCount = 1, pi.pStages = &st;
        }
        pi.pMultisampleState = &ms, pi.pDepthStencilState = &ds;
        pi.layout = lay;
    } else {
        memset(cba, 0, sizeof cba);
        for (int i = 0; i < k->n_color; i++) {
            const uint8_t *b = k->blend[i];
            cba[i].blendEnable = k->blend_en >> i & 1;
            cba[i].srcColorBlendFactor = (VkBlendFactor)b[0], cba[i].dstColorBlendFactor = (VkBlendFactor)b[1];
            cba[i].colorBlendOp = (VkBlendOp)b[2], cba[i].srcAlphaBlendFactor = (VkBlendFactor)b[3];
            cba[i].dstAlphaBlendFactor = (VkBlendFactor)b[4], cba[i].alphaBlendOp = (VkBlendOp)b[5];
            cba[i].colorWriteMask = k->wmask >> (4 * i) & 0xf;
        }
        cb.logicOpEnable = k->logic_op != 0, cb.logicOp = k->logic_op ? (VkLogicOp)(k->logic_op - 1) : VK_LOGIC_OP_COPY;
        cb.attachmentCount = k->n_color, cb.pAttachments = cba;
        ri.colorAttachmentCount = k->n_color, ri.pColorAttachmentFormats = (const VkFormat *)k->color;
        ri.depthAttachmentFormat = (VkFormat)k->depth, ri.stencilAttachmentFormat = (VkFormat)k->stencil;
        pi.pColorBlendState = &cb, pi.pMultisampleState = &ms;
    }
    if (vkCreateGraphicsPipelines(P.device, build_cache(), 1, &pi, NULL, &pipe) != VK_SUCCESS) pipe = VK_NULL_HANDLE;
out:
    if (m) vkDestroyShaderModule(P.device, m, NULL);
    for (int i = 0; i < 2; i++)
        if (mm[i]) vkDestroyShaderModule(P.device, mm[i], NULL);
    free(patched);
    return pipe;
}

// The cached library for key. Caller holds P.lock. build 0: lookup only (VK_NULL_HANDLE and *miss when absent);
// build 1: a missing one is built with P.lock released (a failed build is cached too).
// The library entry for key, NULL when absent. Caller holds P.lock.
static Lib *lib_find(uint64_t h, const void *key, uint32_t size) {
    for (Lib *b = libs[h % LIB_BUCKETS]; b; b = b->next)
        if (b->hash == h && b->size == size && !memcmp(b->key, key, size)) return b;
    return NULL;
}

// Enters a library built with P.lock released; another thread's entry made meanwhile wins and pipe is destroyed.
// Returns the entered handle. Caller holds P.lock.
static VkPipeline lib_put(uint64_t h, const void *key, uint32_t size, VkPipeline pipe) {
    Lib *b = lib_find(h, key, size);
    if (b && b->building) {   // another thread's build publishes the entry
        if (pipe) vkDestroyPipeline(P.device, pipe, NULL);
        return VK_NULL_HANDLE;
    }
    if (b) {
        if (pipe) vkDestroyPipeline(P.device, pipe, NULL);
        return b->pipe;
    }
    if (!(b = calloc(1, sizeof *b + size))) return pipe;
    b->hash = h, b->size = size, b->pipe = pipe;
    memcpy(b->key, key, size);
    b->next = libs[h % LIB_BUCKETS];
    libs[h % LIB_BUCKETS] = b;
    return pipe;
}

// Shader libraries a waiting draw needs (lib_wait): queued here ahead of every pipeline job, so an idle worker builds
// each at once and the vertex and fragment libraries of one draw build in parallel. A job no worker took is built by
// the first thread that needs its library (lib_get, lib_wait). Under P.lock.
typedef struct LibJob { struct LibJob *next; Lib *b; int kind; Layout *l; VkPipeState s; GKey k; } LibJob;
static struct {
    LibJob *head, *tail;
    int idle;                          // workers asleep on work_cv
    int running;                       // jobs workers are building
    int on;                            // REKARINA_PIPE_WAIT: 1 workers (default), 0 the drawing thread builds them (draw);
                                       // set by vk_pipeline_async_enable, before the replay runs
    uint64_t waits, queued, by_worker, stolen, timeouts;
} U;
// The entry step's library jobs (mesh_libs_ahead): only its builders take them (pw_builder), which name what they hand
// the driver in the step's journal, as for every other build of the step. Under P.lock.
static struct { LibJob *head, *tail; } MQ;

// A building entry for key, NULL when out of memory. Caller holds P.lock and found none.
static Lib *lib_new(uint64_t h, const void *key, uint32_t size) {
    Lib *b = calloc(1, sizeof *b + size);
    if (!b) return NULL;
    b->hash = h, b->size = size, b->building = 1;
    memcpy(b->key, key, size);
    b->next = libs[h % LIB_BUCKETS];
    libs[h % LIB_BUCKETS] = b;
    return b;
}

// Builds b (building, not queued) with P.lock released and publishes it; a failed build is entered as VK_NULL_HANDLE.
static VkPipeline lib_build(Lib *b, int kind, const VkPipeState *s, const GKey *k, Layout *l) {
    VkPipelineLayout lay = kind == LIB_PR || kind == LIB_PRM ? half_layout(l, 0) : kind == LIB_FS ? half_layout(l, 1) : VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    if (lay || kind == LIB_VI || kind == LIB_FO) {
        plat_mutex_unlock(&P.lock);
        uint64_t t0 = qpc();
        pipe = lib_create(kind, s, k, lay, b->key);
        double ms = ms_since(t0);
        plat_mutex_lock(&P.lock);
        P.g.lib_n[kind]++, P.g.lib_ms[kind] += ms;
        if (ms > P.g.lib_max[kind]) P.g.lib_max[kind] = ms;
        if (kind == LIB_PR || kind == LIB_FS || kind == LIB_PRM) prep_note(PREP_LIB, 0, ms);   // in play only (prep.c)
    }
    if (!pipe && ++P.g.lib_fail <= 8) rt_log("vk_pipeline: gpl: stage library %d not built", kind);
    b->pipe = pipe, b->building = 0;
    plat_cond_broadcast(&done_cv);
    return pipe;
}

// Takes b's queued job out of the queue and builds it on this thread. Caller holds P.lock.
static void lib_steal(Lib *b) {
    LibJob *j = b->job, **pp = &U.head, *prev = NULL;
    while (*pp && *pp != j) prev = *pp, pp = &(*pp)->next;
    if (*pp) {
        *pp = j->next;
        if (U.tail == j) U.tail = prev;
    } else {   // an entry step's job (MQ)
        for (pp = &MQ.head, prev = NULL; *pp && *pp != j; prev = *pp, pp = &(*pp)->next) {}
        if (*pp) *pp = j->next;
        if (MQ.tail == j) MQ.tail = prev;
    }
    b->job = NULL;
    lib_build(b, j->kind, &j->s, &j->k, j->l);
    free(j);
}

static VkPipeline lib_get(int kind, const VkPipeState *s, const GKey *k, Layout *l, const void *key, uint32_t size,
                          int build, int *miss) {
    uint64_t h = key_hash(key, size);
    Lib *have = lib_find(h, key, size);
    while (have && have->building) {   // being built (wait) or queued (build it here)
        if (!build) {
            *miss = 1;
            return VK_NULL_HANDLE;
        }
        if (have->job) U.stolen++, lib_steal(have);
        else plat_cond_wait(&done_cv, &P.lock);
    }
    if (have) {
        if (have->pre && !have->used) have->used = 1, P.g.pre_used[kind]++;
        return have->pipe;
    }
    VkPipelineLayout lay = kind == LIB_PR || kind == LIB_PRM ? half_layout(l, 0) : kind == LIB_FS ? half_layout(l, 1) : VK_NULL_HANDLE;
    if (!build || ((kind == LIB_PR || kind == LIB_FS || kind == LIB_PRM) && !lay)) {
        *miss = 1;
        return VK_NULL_HANDLE;
    }
    Lib *b = lib_new(h, key, size);   // entered first, so other threads wait for this build instead of repeating it
    if (b) return lib_build(b, kind, s, k, l);
    plat_mutex_unlock(&P.lock);
    uint64_t t0 = qpc();
    VkPipeline pipe = lib_create(kind, s, k, lay, key);
    double ms = ms_since(t0);
    plat_mutex_lock(&P.lock);
    if ((have = lib_find(h, key, size)) && !have->building) {   // another thread built it meanwhile
        if (pipe) vkDestroyPipeline(P.device, pipe, NULL);
        return have->pipe;
    }
    P.g.lib_n[kind]++, P.g.lib_ms[kind] += ms;
    if (ms > P.g.lib_max[kind]) P.g.lib_max[kind] = ms;
    if (kind == LIB_PR || kind == LIB_FS || kind == LIB_PRM) prep_note(PREP_LIB, ms, ms);   // in play only (prep.c)
    if (!pipe && ++P.g.lib_fail <= 8) rt_log("vk_pipeline: gpl: stage library %d not built", kind);
    return lib_put(h, key, size, pipe);
}

static void libs_destroy(void) {
    for (int i = 0; i < LIB_BUCKETS; i++) {
        for (Lib *b = libs[i], *n; b; b = n) {
            n = b->next;
            if (b->pipe) vkDestroyPipeline(P.device, b->pipe, NULL);
            free(b);
        }
        libs[i] = NULL;
    }
}

// The four library keys of pipeline s (key k).
static void lib_keys(const VkPipeState *s, const GKey *k, const Layout *l, ViKey *v, PrKey *pk, FsKey *fk, FoKey *ok) {
    memset(v, 0, sizeof *v);
    v->kind = LIB_VI, v->topo_class = k->topo_class;
    memcpy(v->attr_fmt, k->attr_fmt, sizeof v->attr_fmt), memcpy(v->attr_off, k->attr_off, sizeof v->attr_off);
    memcpy(v->attr_bind, k->attr_bind, sizeof v->attr_bind), memcpy(v->rate, k->rate, sizeof v->rate);
    memset(pk, 0, sizeof *pk);
    GfxShaderIo fin[32];
    pk->kind = LIB_PR, pk->vs = s->vs, pk->clamp = k->clamp, pk->poly_mode = k->poly_mode, pk->minus_w = k->minus_w;
    pk->missing = vs_union_missing(s, fin);
    for (int i = 0; i < 32; i++)
        if (pk->missing >> i & 1) pk->fin[i] = fin[i];
    memset(fk, 0, sizeof *fk);
    fk->kind = LIB_FS, fk->fs = s->fs, fk->dsb = k->dsb, fk->samples = k->samples, fk->a2c = k->a2c;
    pk->pool = fk->pool = l->pool != 0;
    memset(ok, 0, sizeof *ok);
    ok->kind = LIB_FO, ok->n_color = k->n_color, ok->samples = k->samples, ok->a2c = k->a2c, ok->logic_op = k->logic_op;
    ok->blend_en = k->blend_en, ok->depth = k->depth, ok->stencil = k->stencil, ok->wmask = k->wmask;
    memcpy(ok->color, k->color, sizeof ok->color), memcpy(ok->blend, k->blend, sizeof ok->blend);
}

// A fast-linked pipeline for s from its stage libraries. Caller holds P.lock.
// build 0 (drawing thread): links only when the shader libraries exist (vertex input and fragment output libraries,
// which hold no shader, are built here); *final 0 = not yet, a worker builds them. build 1 (worker): builds the
// missing libraries and links with P.lock released. VK_NULL_HANDLE with *final 1 when GPL cannot express the pipeline
// (shared layout, feedback loop) or a build failed.
static VkPipeline gpl_link(const VkPipeState *s, const GKey *k, const Layout *lc, int build, int *final) {
    Layout *l = (Layout *)lc;
    *final = 1;
    if (!l->set1 || k->fbl) return VK_NULL_HANDLE;
    uint64_t t0 = qpc();
    ViKey vk_;
    PrKey pk;
    FsKey fk;
    FoKey ok;
    lib_keys(s, k, l, &vk_, &pk, &fk, &ok);
    int miss = 0;
    VkPipeline lib[4];
    lib[1] = lib_get(LIB_PR, s, k, l, &pk, sizeof pk, build, &miss);   // shader libraries first: no work when absent
    lib[2] = miss ? VK_NULL_HANDLE : lib_get(LIB_FS, s, k, l, &fk, sizeof fk, build, &miss);
    if (miss) {
        if (!build) *final = 0, P.g.defer++;
        return VK_NULL_HANDLE;
    }
    lib[0] = lib_get(LIB_VI, s, k, l, &vk_, sizeof vk_, 1, &miss);
    lib[3] = lib_get(LIB_FO, s, k, l, &ok, sizeof ok, 1, &miss);
    VkPipeline pipe = VK_NULL_HANDLE;
    if (lib[0] && lib[1] && lib[2] && lib[3]) {
        VkPipelineLibraryCreateInfoKHR li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, .libraryCount = 4,
                                             .pLibraries = lib};
        VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &li,
                                           .layout = l->layout};
        if (build) plat_mutex_unlock(&P.lock);
        uint64_t t1 = qpc();
        if (vkCreateGraphicsPipelines(P.device, VK_NULL_HANDLE, 1, &pi, NULL, &pipe) != VK_SUCCESS) pipe = VK_NULL_HANDLE;
        double ms = ms_since(t1);   // the link alone: library builds are counted apart
        if (build) plat_mutex_lock(&P.lock);
        P.g.link_ms += ms;
        if (ms > P.g.link_max) P.g.link_max = ms;
    }
    if (pipe) P.g.links++;
    else if (!P.g.link_fail++) rt_log("vk_pipeline: gpl: fast link failed; such draws wait or skip as without GPL");
    double total = ms_since(t0);
    if (!build && total > 5.0 && P.g.slow++ < 16)
        rt_log("vk_pipeline: gpl: fast link on the drawing thread took %.1f ms", total);
    return pipe;
}

// ---- background graphics pipeline builds (vk_pipeline_async_enable)
// A miss in vk_pipeline_graphics_async enters the Pipe as building and queues a job; the replay skips draws with it
// until a worker publishes the handle. The job holds copies of the state and the key. What they point to outlives it:
// shaders (gfx_shader_get results) live for the process (gfx_shader_drop_memory is the shader probe's, which builds no
// pipelines), layouts until vk_pipeline_shutdown, which drains the queue first. Workers call vkCreateGraphicsPipelines
// without P.lock: P.cache is created without VK_PIPELINE_CACHE_CREATE_EXTERNALLY_SYNCHRONIZED_BIT, so the driver
// synchronises it, vkGetPipelineCacheData in vk_pipeline_save included. Compute pipelines stay synchronous: a skipped
// dispatch can change what the game computes.
//
// Two queues: draws' misses (A.head) and the boot prebuild (A.low, see "pipeline keys" below). Workers take the low
// queue only when the other is empty and run its jobs at PLAT_PRIO_BELOW_NORMAL, so guest threads win. A draw
// that finds a prebuild job still queued moves it to the draw queue; one that finds it running raises its worker.
typedef struct Job {
    struct Job *next, *prev;           // prev: low queue only
    Pipe *p;
    const Layout *l;
    uint32_t fs_off;
    uint8_t low;                       // in the low queue, or running at below-normal priority
    uint8_t prebuild, demanded;        // from the key file; a draw asked for it before it was ready
    int worker;                        // running on A.thread[worker]; -1 queued
    VkPipeState s;
    GKey k;
} Job;

#define MAX_WORKERS 4

static struct {
    int on;                            // set once by vk_pipeline_async_enable, before the replay runs
    int threads;
    PlatThread *thread[MAX_WORKERS];
    uint8_t prio_low[MAX_WORKERS];     // worker runs below normal (under P.lock)
    Job *head, *tail;                  // draws' misses
    Job *low, *low_tail;               // prebuild
    uint32_t inflight;                 // queued or running, both queues
    int low_running;                   // pool workers running a job taken from the low queue
    PlatCond idle;           // with P.lock: no job queued or running
} A = {.idle = PLAT_COND_INIT};

// ---- pipeline keys (pipeline-keys.bin next to the pipeline cache)
// Every graphics pipeline built in a game run appends a KeyRec: the VkPipeState it was built from with the shader
// pointers zeroed and the shaders named by their gfx_shader input ids instead, which do not depend on the translator,
// so the record rebuilds the pipeline without guest memory. At the next boot vk_pipeline_prewarm (rt_boot, before the
// guest runs; REKARINA_PIPE_PREBUILD=0 skips it) loads each record's shaders, translating the ones not cached under the
// current translator from their stored inputs (gfx_shader_prewarm; a record whose shader has neither is dropped),
// enters each Pipe as building under the key a draw would compute, queues it to the low queue and builds the queue on
// every core. Records are deduplicated by a hash of their bytes; they are collected under P.lock by the worker that
// built the pipeline and appended by the present thread (vk_pipeline_autosave) or at exit (vk_pipeline_save). A file
// with another magic, version or record size is ignored and rewritten. Bump KEYS_VERSION when VkPipeState or the
// record changes meaning without changing size.
#define KEYS_VERSION 3                 // 3: shaders named by input id (gfx_shader.c)
typedef struct { char magic[8]; uint32_t version, rec_size; } KeysHeader;
typedef struct {
    uint8_t vs[32], fs[32];            // gfx_shader_input_id of each stage; fs all zero with has_fs 0
    uint32_t has_fs, pad;
    VkPipeState s;                     // vs, fs zero
} KeyRec;
static struct {
    char path[PLAT_PATH_MAX * 2];
    int header_ok;                     // the file exists with our header: append
    uint64_t *seen;                    // open addressing set of record hashes (0 empty)
    uint32_t seen_cap, seen_n;
    KeyRec *pend;                      // records not written yet
    uint32_t pend_n, pend_cap;
    uint8_t *file;                     // the file as read at boot, until the prewarm is done with it
    size_t file_n;                     // records in it
    PlatMutex io;                        // one writer
    // prebuild progress (under P.lock)
    int loading;                       // the prewarm still queues records
    int warm;                          // vk_pipeline_prewarm owns the file (prebuild_start leaves it)
    int pw;                            // the entry step's builders own the low queue (pipe_worker leaves it)
    uint32_t left;                     // prebuild jobs queued or running
    uint32_t built, failed, promoted;
    double t0;                         // now_ms() when the prebuild started
} K = {.io = PLAT_MUTEX_INIT};

static uint64_t rec_hash(const KeyRec *r) {
    uint64_t h = fnv(FNV0, r, sizeof *r);
    return h ? h : 1;
}

// 1 if h was new. Caller holds P.lock.
static int seen_add(uint64_t h) {
    if ((K.seen_n + 1) * 2 > K.seen_cap) {
        uint32_t cap = K.seen_cap ? K.seen_cap * 2 : 4096;
        uint64_t *t = calloc(cap, sizeof *t);
        if (!t) return 1;
        for (uint32_t i = 0; i < K.seen_cap; i++) {
            if (!K.seen[i]) continue;
            uint32_t j = (uint32_t)K.seen[i] & (cap - 1);
            while (t[j]) j = (j + 1) & (cap - 1);
            t[j] = K.seen[i];
        }
        free(K.seen);
        K.seen = t, K.seen_cap = cap;
    }
    uint32_t j = (uint32_t)h & (K.seen_cap - 1);
    while (K.seen[j]) {
        if (K.seen[j] == h) return 0;
        j = (j + 1) & (K.seen_cap - 1);
    }
    K.seen[j] = h;
    K.seen_n++;
    return 1;
}

static void rec_fill(KeyRec *r, const VkPipeState *s) {
    memset(r, 0, sizeof *r);
    memcpy(r->vs, gfx_shader_input_id(s->vs), 32);
    if (s->fs) memcpy(r->fs, gfx_shader_input_id(s->fs), 32), r->has_fs = 1;
    r->s = *s;
    r->s.vs = r->s.fs = NULL;
    r->s.prim_restart = 0;   // dynamic state: the same pipeline either way
}

// A pipeline was built for a draw: queue its record for the file. Caller holds P.lock.
static void keys_note(const VkPipeState *s) {
    if (!A.on || !K.path[0]) return;
    static const uint8_t none[32];
    if (s->fs && !memcmp(gfx_shader_input_id(s->fs), none, 32)) return;   // the runtime's own shader (gfx_shader_own)
    KeyRec r;
    rec_fill(&r, s);
    if (!seen_add(rec_hash(&r))) return;
    if (K.pend_n == K.pend_cap) {
        uint32_t cap = K.pend_cap ? K.pend_cap * 2 : 64;
        KeyRec *t = realloc(K.pend, cap * sizeof *t);
        if (!t) return;
        K.pend = t, K.pend_cap = cap;
    }
    K.pend[K.pend_n++] = r;
}

// Appends the pending records (present thread or exit; never the render thread).
static void keys_flush(void) {
    if (!K.path[0]) return;
    plat_mutex_lock(&K.io);
    plat_mutex_lock(&P.lock);
    KeyRec *r = K.pend;
    uint32_t n = K.pend_n;
    K.pend = NULL, K.pend_n = K.pend_cap = 0;
    plat_mutex_unlock(&P.lock);
    if (n) {
        FILE *f = fopen(K.path, K.header_ok ? "ab" : "wb");
        int ok = f != NULL;
        if (ok && !K.header_ok) {
            KeysHeader h = {{'T', 'O', 'T', 'K', 'P', 'K', 'E', 'Y'}, KEYS_VERSION, sizeof(KeyRec)};
            ok = fwrite(&h, sizeof h, 1, f) == 1;
        }
        if (ok) ok = fwrite(r, sizeof *r, n, f) == n;
        if (ok) ok = plat_file_sync(f);   // appended records on the disk before the next append
        if (f) ok = !fclose(f) && ok;
        if (ok) K.header_ok = 1;
        else rt_log("vk_pipeline: cannot write %s", K.path);
    }
    free(r);
    plat_mutex_unlock(&K.io);
}

// A release's own records (pipeline-keys-seed.bin beside the key file; tools/package_oot.py puts it there): the
// pipelines its makers' play met, so a player's first run finds them built instead of building each at its first
// draw (the first 50 s at a place without them: 660 stage libraries built in play and 7 draws that waited; none
// with them). Records the player's own file holds are left out; the others join K.file for the prewarm and stay in
// the seed file only. A record whose shaders this machine cannot load is dropped by the prewarm, as any other.
static void keys_seed(void) {
    char p[sizeof K.path];
    snprintf(p, sizeof p, "%s", K.path);
    char *dot = strrchr(p, '.');
    if (!dot || (size_t)(dot - p) + sizeof "-seed.bin" > sizeof p) return;
    strcpy(dot, "-seed.bin");
    FILE *f = fopen(p, "rb");
    if (!f) return;
    KeysHeader h;
    size_t n = 0, added = 0, cap = K.file_n;
    if (fread(&h, sizeof h, 1, f) == 1 && !memcmp(h.magic, "TOTKPKEY", 8) && h.version == KEYS_VERSION &&
        h.rec_size == sizeof(KeyRec)) {
        KeyRec r;
        while (fread(&r, sizeof r, 1, f) == 1) {
            n++;
            if (!seen_add(rec_hash(&r))) continue;
            if (K.file_n == cap) {
                size_t c = cap ? cap * 2 : 1024;
                uint8_t *t = realloc(K.file, c * sizeof(KeyRec));
                if (!t) break;
                K.file = t, cap = c;
            }
            memcpy(K.file + K.file_n * sizeof(KeyRec), &r, sizeof r);
            K.file_n++, added++;
        }
        rt_log("vk_pipeline: the release's pipeline keys: %zu records, %zu of them not in this machine's own file (%s)", n, added, p);
    } else {
        rt_log("vk_pipeline: %s has another format or version: left out", p);
    }
    fclose(f);
}

// Reads the key file into K.file and its record hashes into the set. Caller holds P.lock (ensure_init).
static void keys_open(void) {
    if (!A.on || !P.cache_path[0]) return;
    snprintf(K.path, sizeof K.path, "%s", P.cache_path);
    char *slash = strrchr(K.path, '\\'), *fslash = strrchr(K.path, '/');
    if (fslash > slash) slash = fslash;
    snprintf(slash ? slash + 1 : K.path, sizeof K.path - (size_t)(slash ? slash + 1 - K.path : 0), "pipeline-keys.bin");
    gfx_shader_record_at(K.path);   // shader-answers.bin next to it (gfx_shader.c)
    FILE *f = fopen(K.path, "rb");
    if (!f) {
        keys_seed();
        return;
    }
    KeysHeader h;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (len >= (long)sizeof h && fread(&h, sizeof h, 1, f) == 1 && !memcmp(h.magic, "TOTKPKEY", 8) &&
        h.version == KEYS_VERSION && h.rec_size == sizeof(KeyRec)) {
        K.header_ok = 1;
        size_t n = ((size_t)len - sizeof h) / sizeof(KeyRec);
        if (n && (K.file = malloc(n * sizeof(KeyRec))) && fread(K.file, sizeof(KeyRec), n, f) == n) K.file_n = n;
        else free(K.file), K.file = NULL;
        for (size_t i = 0; i < K.file_n; i++) seen_add(rec_hash((const KeyRec *)K.file + i));
        if (len != (long)(sizeof h + n * sizeof(KeyRec)))   // a torn append: rewrite the file from what was read
            K.header_ok = 0;
    } else {
        rt_log("vk_pipeline: %s has another format or version; it is rewritten", K.path);
    }
    fclose(f);
    if (!K.header_ok && K.file_n) {   // keep the whole records of a torn file
        KeyRec *t = malloc(K.file_n * sizeof *t);
        if (t) memcpy(t, K.file, K.file_n * sizeof *t), K.pend = t, K.pend_n = K.pend_cap = (uint32_t)K.file_n;
    }
    keys_seed();
}

static double now_ms(void) { return (double)plat_now_ns() / 1e6; }

// Caller holds P.lock.
static void prebuild_report(void) {
    rt_log("vk_pipeline: prebuild done: %u built, %u failed, %u asked for by a draw first; %.0f ms after the device was "
           "ready", K.built, K.failed, K.promoted, now_ms() - K.t0);
}

static void pipe_worker(void *arg) {
    int me = (int)(intptr_t)arg;
    plat_mutex_lock(&P.lock);
    for (;;) {
        // With 3 or more workers one never takes low-queue work, so a waiting draw's library starts at once
        // (not with REKARINA_PIPE_WAIT=draw). While the entry step's builders own the low queue (K.pw), none takes it:
        // they note what they hand the driver in the entry step's journal (vk_pipeline_prewarm).
        while (!U.head && !A.head && !(A.low && !K.pw && (A.threads < 3 || !U.on || A.low_running < A.threads - 1))) {
            U.idle++;
            plat_cond_wait(&work_cv, &P.lock);
            U.idle--;
        }
        if (U.head) {   // a library a draw waits for
            LibJob *lj = U.head;
            if (!(U.head = lj->next)) U.tail = NULL;
            lj->b->job = NULL;
            if (A.prio_low[me]) A.prio_low[me] = 0, plat_thread_priority(PLAT_PRIO_NORMAL);
            U.by_worker++, U.running++;
            lib_build(lj->b, lj->kind, &lj->s, &lj->k, lj->l);   // releases P.lock while building
            U.running--;
            free(lj);
            continue;
        }
        Job *j = A.head;
        int lowrun = !j;
        if (j) {
            if (!(A.head = j->next)) A.tail = NULL;
        } else {
            j = A.low;
            if ((A.low = j->next)) A.low->prev = NULL;
            else A.low_tail = NULL;
            A.low_running++;
        }
        j->next = j->prev = NULL;
        j->worker = me;
        if (j->low != A.prio_low[me]) {   // under P.lock, so a draw raising this thread (promote) is not undone
            A.prio_low[me] = j->low;
            plat_thread_priority(j->low ? PLAT_PRIO_BELOW_NORMAL : PLAT_PRIO_NORMAL);
        }
        // Step 1 for a job a draw waits for: its stage libraries and the fast link. Step 2, the optimised build, goes to
        // the head of the low queue, so a burst of new pipelines gets every fast link before any optimised build.
        if (P.gpl && (!j->prebuild || j->demanded) && !j->p->info.pipeline && !j->p->fast_tried && j->p->lay) {
            int final;
            VkPipeline f = gpl_link(&j->s, &j->k, j->p->lay, 1, &final);   // releases P.lock while building
            j->p->fast_tried = 1;
            if (f && !j->p->info.pipeline) {
                j->p->fast = j->p->info.pipeline = f, P.g.wlinks++;
                plat_cond_broadcast(&done_cv);
                j->low = 1, j->worker = -1, P.g.requeued++;
                j->prev = NULL, j->next = A.low;
                if (A.low) A.low->prev = j;
                else A.low_tail = j;
                A.low = j;
                A.low_running -= lowrun;
                continue;
            } else if (f) {
                vkDestroyPipeline(P.device, f, NULL);
            }
        }
        plat_mutex_unlock(&P.lock);
        VkPipeline pipe = create_graphics(&j->s, &j->k, j->l, j->fs_off);
        plat_mutex_lock(&P.lock);
        if (pipe || !j->p->fast) j->p->info.pipeline = pipe;   // a failed optimised build keeps the fast-linked one
        if (pipe && j->p->fast) P.g.swaps++;
        j->p->building = 0;
        j->p->job = NULL;
        if (pipe) {
            P.stats.graphics++, P.stats.async_built++;
            if (!j->prebuild) keys_note(&j->s);
        } else {
            P.stats.failures++;
            rt_log("vk_pipeline: graphics pipeline not built: pipeline creation failed");
        }
        if (j->prebuild) {
            if (pipe) K.built++;
            else K.failed++;
            K.promoted += j->demanded;
            if (!--K.left && !K.loading) prebuild_report();
        }
        A.low_running -= lowrun;
        if (!--A.inflight) plat_cond_broadcast(&A.idle);
        plat_cond_broadcast(&done_cv);
        if (pipe && (!j->prebuild || j->demanded)) perf_count(PERF_CNT_PIPE_ASYNC);   // a draw waited for it
        free(j);
    }
    return;
}

// Starts the pool on first use: a quarter of the logical processors, 2 to 4. Caller holds P.lock.
static int workers_start(void) {
    if (A.threads) return 1;
    uint32_t cpus = plat_cpu_usable();
    int n = cpus / 4 < 2 ? 2 : cpus / 4 > MAX_WORKERS ? MAX_WORKERS : (int)(cpus / 4);
    for (int i = 0; i < n; i++) {
        PlatThread *t = plat_thread_new(pipe_worker, (void *)(intptr_t)A.threads, "pipe worker");
        if (!t) break;
        A.thread[A.threads++] = t;   // kept: promote() raises a worker running a prebuild job
    }
    if (A.threads) rt_log("vk_pipeline: %d background graphics pipeline workers", A.threads);
    else rt_log("vk_pipeline: cannot start pipeline workers; building synchronously");
    return A.threads != 0;
}

static void workers_drain(void) {
    plat_mutex_lock(&P.lock);
    K.loading = 0;
    for (Job *j = A.low, *n; j; j = n) {   // queued prebuild jobs are dropped, not built
        n = j->next;
        j->p->building = 0, j->p->job = NULL;
        A.inflight--, K.left--;
        free(j);
    }
    A.low = A.low_tail = NULL;
    for (LibJob *j = U.head, *n; j; j = n) {   // queued library jobs too: their entries read as failed builds
        n = j->next;
        j->b->job = NULL, j->b->building = 0;
        free(j);
    }
    U.head = U.tail = NULL;
    while (A.inflight) plat_cond_wait(&A.idle, &P.lock);
    while (U.running) plat_cond_wait(&done_cv, &P.lock);
    plat_mutex_unlock(&P.lock);
}

void vk_pipeline_async_enable(void) {
    U.on = strcmp(cfg_value("REKARINA_PIPE_WAIT"), "draw") != 0;
    const char *e = cfg_get("REKARINA_PIPE_SYNC");
    if (e && *e && strcmp(e, "0")) return;
    A.on = 1;
}

int vk_pipeline_async_on(void) { return A.on; }

// A draw needs a pipeline a prebuild job owns: the job moves to the draw queue, or its worker to normal priority.
// Caller holds P.lock.
static void promote(Pipe *p) {
    Job *j = p->job;
    if (!j || !j->low) return;
    j->low = 0, j->demanded = 1;
    if (j->worker < 0) {
        if (j->prev) j->prev->next = j->next;
        else A.low = j->next;
        if (j->next) j->next->prev = j->prev;
        else A.low_tail = j->prev;
        j->next = j->prev = NULL;
        if (A.tail) A.tail->next = j;
        else A.head = j;
        A.tail = j;
        plat_cond_signal(&work_cv);
    } else if (A.prio_low[j->worker]) {
        A.prio_low[j->worker] = 0;
        plat_thread_set_priority(A.thread[j->worker], PLAT_PRIO_NORMAL);
    }
}

// Enters a missing pipeline: built here (mode 0), or queued for a draw (1) or for the prebuild (2). Failures are
// entered too. *queued: 1 when a job owns it. Caller holds P.lock.
static Pipe *pipe_miss(const VkPipeState *s, const GKey *k, uint64_t h, int mode, int *queued) {
    *queued = 0;
    const char *why = NULL;
    Bindings B = {0};
    Layout *l = NULL;
    uint32_t span = binding_span(s->vs);
    if (gfx_shader_stage(s->vs) != GFX_STAGE_VS || (s->fs && gfx_shader_stage(s->fs) != GFX_STAGE_FS)) why = "stage mismatch";
    else if (k->topo_class == 3) why = "patches need tessellation stages, which are not translated";
    else if (P.indep) {
        Bindings F = {0};
        if (!add_stage(&B, s->vs, 0) || (s->fs && !add_stage(&F, s->fs, 0))) why = "too many bindings";
        else if (!(l = layout_pair(&B, &F))) why = "descriptor layout creation failed";
        span = 1;
    } else if (!add_stage(&B, s->vs, 0) || (s->fs && !add_stage(&B, s->fs, span - 1))) why = "too many bindings";
    else if (!(l = layout_get(&B))) why = "descriptor layout creation failed";
    char lack[128];
    if (!why && !layout_covers(s, l, span - 1, lack, sizeof lack)) why = lack, l = NULL;
    VkPipeline pipe = VK_NULL_HANDLE;
    Job *j = NULL;
    if (!why && mode && workers_start() && (j = calloc(1, sizeof *j))) {
        *j = (Job){.l = l, .fs_off = span - 1, .low = mode == 2, .prebuild = mode == 2, .worker = -1, .s = *s, .k = *k};
    } else if (!why && !(pipe = create_graphics(s, k, l, span - 1))) {
        why = "pipeline creation failed";
    }
    if (why) {
        P.stats.failures++;
        rt_log("vk_pipeline: graphics pipeline not built: %s", why);
    } else if (!j) {
        P.stats.graphics++;
        if (mode != 2) keys_note(s);
    }
    Pipe *p = pipe_add(k, sizeof *k, h, pipe, l, l ? &B : NULL, VK_PIPELINE_BIND_POINT_GRAPHICS, gfx_shader_inputs(s->vs, NULL));
    if (!j || !p) {
        free(j);
        return p;
    }
    j->p = p;
    p->building = 1, p->job = j;
    if (mode == 2) {
        j->prev = A.low_tail;
        if (A.low_tail) A.low_tail->next = j;
        else A.low = j;
        A.low_tail = j;
        K.left++;
    } else {
        if (A.tail) A.tail->next = j;
        else A.head = j;
        A.tail = j;
    }
    A.inflight++;
    plat_cond_signal(&work_cv);
    *queued = 1;
    return p;
}

// A job still in the draw queue goes to the head of the low queue: its pipeline has (or is about to have) a fast-linked
// one, so the optimised build waits behind other draws' misses. Caller holds P.lock.
static void job_lower(Job *j) {
    if (!j || j->worker >= 0 || j->low) return;
    if (A.head == j) {
        if (!(A.head = j->next)) A.tail = NULL;
    } else {
        Job *q = A.head;
        while (q && q->next != j) q = q->next;
        if (!q) return;
        if (!(q->next = j->next)) A.tail = q;
    }
    j->low = 1, P.g.requeued++;
    j->prev = NULL, j->next = A.low;
    if (A.low) A.low->prev = j;
    else A.low_tail = j;
    A.low = j;
}

// A draw waits for pipeline s: its missing pre-rasterization and fragment shader libraries go to the head of the
// worker queues (U), so they build in parallel on idle workers, and this thread sleeps until both are published or
// end (now_ms) passes. A job no worker took within 1 ms (all busy) is built here. Caller holds P.lock.
static void lib_wait(const VkPipeState *s, const GKey *k, Layout *l, double end) {
    ViKey vk_;
    PrKey pk;
    FsKey fk;
    FoKey ok;
    lib_keys(s, k, l, &vk_, &pk, &fk, &ok);
    U.waits++;
    const void *key[2] = {&pk, &fk};
    uint32_t size[2] = {sizeof pk, sizeof fk};
    Lib *b[2];
    int queued = 0;
    for (int i = 0; i < 2; i++) {
        uint64_t h = key_hash(key[i], size[i]);
        if ((b[i] = lib_find(h, key[i], size[i]))) continue;
        if (!half_layout(l, i)) return;   // gpl_link reports it
        LibJob *j = calloc(1, sizeof *j);
        if (!j || !(b[i] = lib_new(h, key[i], size[i]))) {
            free(j);
            return;
        }
        *j = (LibJob){.b = b[i], .kind = i ? LIB_FS : LIB_PR, .l = l, .s = *s, .k = *k};
        b[i]->job = j;
        if (U.tail) U.tail->next = j;
        else U.head = j;
        U.tail = j;
        U.queued++, queued++;
    }
    if (queued) plat_cond_broadcast(&work_cv);
    double steal_at = now_ms() + 1.0;
    for (;;) {
        if (!b[0]->building && !b[1]->building) return;
        double now = now_ms();
        if (now >= end) {
            U.timeouts++;
            return;
        }
        Lib *q = b[0]->job ? b[0] : b[1]->job ? b[1] : NULL;
        if (q && (now >= steal_at || !U.idle)) {
            U.stolen++;
            lib_steal(q);   // releases P.lock while building
            continue;
        }
        double until = q && steal_at < end ? steal_at : end;
        plat_cond_wait_ms(&done_cv, &P.lock, (uint32_t)(until - now) + 1);
    }
}

static VkPipeline graphics(const VkPipeState *s, const VkPipeInfo **info, int async, int *pending, uint32_t wait_ms) {
    if (info) *info = NULL;
    if (pending) *pending = 0;
    if (!s->vs || !ensure_init()) return VK_NULL_HANDLE;
    GKey k;
    gkey_fill(s, &k);
    uint64_t h = key_hash(&k, sizeof k);
    plat_mutex_lock(&P.lock);
    Pipe *p = pipe_find(&k, sizeof k, h);
    int queued;
    double miss_t0 = 0;   // a miss in play goes into the ledger with this call's wait (prep.c)
    if (p) {
        if (!p->building) P.stats.hits++;
    } else {
        if (prep_in_play()) miss_t0 = now_ms();
        p = pipe_miss(s, &k, h, async, &queued);
    }
    // A draw that would wait (wait_ms) gets its missing shader libraries built at once, the vertex and fragment ones in
    // parallel on idle workers (lib_wait; about 0.12 ms per library from the driver's disk cache, 5-180 ms cold), and
    // links them here: its own job may queue behind optimised builds of earlier pipelines. The job then only does the
    // optimised build, from the low queue. Other draws leave the libraries to the job and are skipped meanwhile.
    // REKARINA_PIPE_WAIT=draw builds them on this thread one after the other instead, as before; REKARINA_GPL_DRAW_BUILD=0
    // leaves them to the job.
    static int draw_build = -1;
    if (draw_build < 0) {
        const char *e = cfg_get("REKARINA_GPL_DRAW_BUILD");
        draw_build = !(e && !strcmp(e, "0"));
        rt_log("vk_pipeline: a draw that waits for its pipeline %s", !draw_build ? "leaves its libraries to its job"
               : U.on ? "has its libraries built in parallel on the workers, ahead of other jobs"
                      : "builds its libraries itself (REKARINA_PIPE_WAIT=draw)");
    }
    double end = wait_ms ? now_ms() + wait_ms : 0;
    if (p && p->building && P.gpl && !p->info.pipeline && !p->fast_tried && p->lay) {
        int final = 0, build = wait_ms != 0 && draw_build;
        uint64_t p0 = perf_begin(), t0 = qpc();
        if (build && U.on) {
            job_lower(p->job);   // its worker would only wait for the same libraries; it links them if this draw cannot
            lib_wait(s, &k, (Layout *)p->lay, end);   // releases P.lock while waiting
            build = 0;                                // links here when both libraries exist (gpl_link build 0)
        } else if (build) {
            p->fast_tried = 1;   // the job, if a worker takes it meanwhile, goes straight to the optimised build
        }
        VkPipeline f = p->info.pipeline ? VK_NULL_HANDLE : gpl_link(s, &k, p->lay, build, &final);   // build 1 releases P.lock
        perf_end(PERF_PIPE, p0);
        if (wait_ms && draw_build) {
            double ms = ms_since(t0);
            P.g.dlinks++, P.g.dlink_ms += ms;
            if (ms > P.g.dlink_max) P.g.dlink_max = ms;
            if (ms > 5.0 && P.g.slow++ < 16)
                rt_log("vk_pipeline: gpl: a waiting draw %s its libraries in %.1f ms", U.on ? "got" : "built", ms);
        }
        if (final) p->fast_tried = 1;
        if (f && !p->info.pipeline) {
            p->fast = p->info.pipeline = f;
            job_lower(p->job);
            plat_cond_broadcast(&done_cv);
        } else if (f) {
            vkDestroyPipeline(P.device, f, NULL);   // a worker published one while the lock was free
        }
    }
    if (p && p->building && !p->info.pipeline) {
        promote(p);
        uint64_t p0 = wait_ms ? perf_begin() : 0;   // the wait counts as the render thread's pipeline time
        double now;
        while (p->building && !p->info.pipeline && wait_ms && (now = now_ms()) < end)
            plat_cond_wait_ms(&done_cv, &P.lock, (uint32_t)(end - now) + 1);
        if (wait_ms) perf_end(PERF_PIPE, p0);
        if (wait_ms && !p->building) perf_count(PERF_CNT_PIPE_WAIT);
    }
    int wait = p && p->building && !p->info.pipeline;
    if (wait) P.stats.pending_skips++;
    VkPipeline pipe = p ? p->info.pipeline : VK_NULL_HANDLE;
    plat_mutex_unlock(&P.lock);
    if (miss_t0) prep_note(PREP_GFX, now_ms() - miss_t0, 0);
    if (wait) {
        perf_count(PERF_CNT_PIPE_SKIP);
        if (pending) *pending = 1;
        return VK_NULL_HANDLE;
    }
    if (!p) return VK_NULL_HANDLE;
    if (info) *info = &p->info;
    return pipe;
}

VkPipeline vk_pipeline_graphics(const VkPipeState *s, const VkPipeInfo **info) {
    return graphics(s, info, 0, NULL, 0);
}

VkPipeline vk_pipeline_graphics_async(const VkPipeState *s, const VkPipeInfo **info, int *pending, uint32_t wait_ms) {
    return graphics(s, info, A.on, pending, wait_ms);
}

// ---- boot prewarm from the key file (comment at "pipeline keys"): the entry step, further down

static int id_cmp(const void *x, const void *y) { return memcmp(x, y, 32); }

// ensure_init's hook: without a prewarm (REKARINA_PIPE_PREBUILD=0, or the device came up first elsewhere) the records
// are not built. Caller holds no lock.
static void prebuild_start(void) {
    plat_mutex_lock(&P.lock);
    if (K.file && !K.warm) {
        rt_log("vk_pipeline: prebuild off%s; %zu records not built",
               !cfg_bool("REKARINA_PIPE_PREBUILD") ? " (REKARINA_PIPE_PREBUILD=0)" : " (no boot prewarm)", K.file_n);
        free(K.file);
        K.file = NULL;
    }
    plat_mutex_unlock(&P.lock);
}


// A compute pipeline for cs in layout l; VK_NULL_HANDLE on failure. No lock needed (P.cache is driver-synchronised).
// A mesh shader as compute (GFX_STAGE_MSC) names the emulation buffer by two marker constants (tools/xlate MeshStage
// EmuAddressLo, EmuAddressHi): a copy of its module with the buffer's address in their place. NULL without the buffer.
static uint32_t *meshc_module(const uint32_t *code, uint64_t words) {
    uint64_t a = vk_mesh_emu_address();
    uint32_t *c = a ? malloc(words * 4) : NULL;
    if (!c) return NULL;
    memcpy(c, code, words * 4);
    int found = 0;
    for (uint64_t i = 5; i < words;) {
        uint32_t n = c[i] >> 16, op = c[i] & 0xffff;
        if (!n || i + n > words) break;
        if (op == 43 && n == 4) {   // OpConstant type id value
            if (c[i + 3] == VK_MESH_EMU_ADDRESS_LO) c[i + 3] = (uint32_t)a, found |= 1;
            else if (c[i + 3] == VK_MESH_EMU_ADDRESS_HI) c[i + 3] = (uint32_t)(a >> 32), found |= 2;
        }
        i += n;
    }
    if (found != 3 && found) free(c), c = NULL;   // none: a kernel of translator patch 0047, which takes the address from
    return c;                                     // push constants (vk_exec_draw.c mesh_emulate pushes it): used as it is
}

static VkPipeline compute_create_v(const GfxShader *cs, const Layout *l, int passthrough) {
    uint64_t words;
    const uint32_t *code = gfx_shader_spirv(cs, &words);
    int emu = gfx_shader_stage(cs) == GFX_STAGE_MSC || gfx_shader_stage(cs) == GFX_STAGE_TSC;   // task kernels name it too
    uint32_t *patched = emu ? meshc_module(code, words) : NULL;
    if (emu && !patched) return VK_NULL_HANDLE;
    if (patched) code = patched;
    uint64_t nw = 0;
    uint32_t *net = passthrough && !emu ? vk_movienet_module(cs, code, words, &nw) : NULL;   // the movies' pass 12 alone
    if (passthrough && !net) {
        free(patched);
        return VK_NULL_HANDLE;
    }
    if (net) code = net, words = nw;
    VkShaderModule m = make_module(code, words);
    if (!m) {
        free(patched), free(net);
        return VK_NULL_HANDLE;
    }
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rqc;
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = l->layout};
    {   // DEBUG-EMUPIN: REKARINA_MESH_EMU_PIN=0: the mesh emulation's kernels run at the driver's width (a test)
        static int emu_pin = -1;
        if (emu_pin < 0) emu_pin = rt_knob("MESH_EMU_PIN", "1");
        if (!emu || emu_pin) pin_subgroup32(&ci.stage, &rqc, code, words);
    }
    // A warp on the console always has its 32 lanes. Without this the driver may run partial subgroups in a workgroup that
    // is a whole number of them, and the ballots the foliage culling reserves its list slots with count differently from
    // one frame to the next (REKARINA_FULL_SUBGROUPS=0 turns it off).
    {
        static int on = -1;
        if (on < 0) on = rt_knob("FULL_SUBGROUPS", "1");
        uint32_t lx = local_size_x(code, words);
        // only with the pin (pin_subgroup32 set it: a stage it pins, a module with subgroup operations): without a
        // required size the local size would have to be a multiple of the device's largest subgroup
        if (on && P.f_full_subgroups && ci.stage.pNext == &rqc && lx && lx % vk_caps.sg_size == 0)
            ci.stage.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    }
    VkPipeline pipe = VK_NULL_HANDLE;
    if (pipestats_on()) ci.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;   // DEBUG-PIPESTATS
    if (vkCreateComputePipelines(P.device, build_cache(), 1, &ci, NULL, &pipe) != VK_SUCCESS) pipe = VK_NULL_HANDLE;
    { const GfxShader *ps[1] = {cs}; VkShaderStageFlags pst[1] = {VK_SHADER_STAGE_COMPUTE_BIT}; pipestats_log(pipe, ps, pst, 1); }
    vkDestroyShaderModule(P.device, m, NULL);
    // (the module's words are read above for the subgroup pin and the local size: freed only now, after them)
    free(patched), free(net);
    return pipe;
}
static VkPipeline compute_create(const GfxShader *cs, const Layout *l) { return compute_create_v(cs, l, 0); }
static Layout *compute_layout(const GfxShader *cs, Bindings *B, const char **why);   // below

// Pass 12 of the movies' network as vk_movienet.c runs it once the U-Net is left out: its own table entry (key compute
// 2), built at its first dispatch on this thread (one pipeline, once a session); VK_NULL_HANDLE when it cannot be.
VkPipeline vk_pipeline_compute_passthrough(const GfxShader *cs, const VkPipeInfo **info) {
    if (info) *info = NULL;
    if (!cs || !ensure_init()) return VK_NULL_HANDLE;
    struct { const GfxShader *cs; uint32_t compute; } k = {cs, 2};
    uint64_t h = key_hash(&k, sizeof k);
    plat_mutex_lock(&P.lock);
    Pipe *p = pipe_find(&k, sizeof k, h);
    if (!p) {
        const char *why = NULL;
        Bindings B = {0};
        Layout *l = compute_layout(cs, &B, &why);
        VkPipeline pipe = why ? VK_NULL_HANDLE : compute_create_v(cs, l, 1);
        rt_log("vk_pipeline: pass 12 of the movies' network without the U-Net %s", pipe ? "built" : "could not be built");
        p = pipe_add(&k, sizeof k, h, pipe, l, l && !why ? &B : NULL, VK_PIPELINE_BIND_POINT_COMPUTE, 0);
    }
    plat_mutex_unlock(&P.lock);
    if (!p || !p->info.pipeline) return VK_NULL_HANDLE;
    if (info) *info = &p->info;
    return p->info.pipeline;
}

// The layout of a compute shader in *B; NULL with *why on failure. Caller holds P.lock.
static Layout *compute_layout(const GfxShader *cs, Bindings *B, const char **why) {
    Layout *l = NULL;
    if (gfx_shader_stage(cs) != GFX_STAGE_CS && gfx_shader_stage(cs) != GFX_STAGE_MSC && gfx_shader_stage(cs) != GFX_STAGE_TSC)
        *why = "not a compute shader";
    else if (!add_stage(B, cs, 0)) *why = "too many bindings";
    else if ((B->emu = gfx_shader_stage(cs) != GFX_STAGE_CS), !(l = layout_get(B))) *why = "descriptor layout creation failed";
    return l;
}


#ifdef GAME_NS2
// ---- NVN2 geometry and mesh pipelines (GAME_NS2)
// A program with a geometry stage, or NVN2's task and mesh shaders, gets a pipeline built whole: no stage libraries, no
// background build, no vertex output fix-up. Its layout is one set as for REKARINA_GPL=0: binding 0 the support buffer,
// each stage's bindings after the previous stage's (its SPIR-V renumbered as the fragment stage's is). The mesh path
// takes no vertex input or input assembly state, and its dynamic state list leaves out what a mesh pipeline may not
// have (vertex input, topology, primitive restart). Built on the calling thread under P.lock, as compute pipelines.
static struct { PFN_vkCmdDrawMeshTasksEXT draw; PFN_vkCmdDrawMeshTasksIndirectEXT draw_ind; int ready; } MESH;

int vk_pipeline_mesh_on(void) {
    VkCtx *c = vk_ctx();
    if (!c || !c->has_mesh_shader) return 0;
    if (!__atomic_load_n(&MESH.ready, __ATOMIC_ACQUIRE)) {
        MESH.draw = (PFN_vkCmdDrawMeshTasksEXT)vkGetDeviceProcAddr(c->device, "vkCmdDrawMeshTasksEXT");
        MESH.draw_ind = (PFN_vkCmdDrawMeshTasksIndirectEXT)vkGetDeviceProcAddr(c->device, "vkCmdDrawMeshTasksIndirectEXT");
        {   // the device's task and mesh limits, once: OoT's shaders were written against NVIDIA's (output granularity 32)
            VkPhysicalDeviceMeshShaderPropertiesEXT mp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_PROPERTIES_EXT};
            VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &mp};
            vkGetPhysicalDeviceProperties2(c->phys, &p2);
            rt_log("vk_pipeline: mesh limits: task payload %u, task shared %u, payload + shared %u; mesh shared %u, payload + shared "
                   "%u, output memory %u, payload + output %u, vertices %u, primitives %u, components %u; output granularity vertex "
                   "%u primitive %u",
                   mp.maxTaskPayloadSize, mp.maxTaskSharedMemorySize, mp.maxTaskPayloadAndSharedMemorySize,
                   mp.maxMeshSharedMemorySize, mp.maxMeshPayloadAndSharedMemorySize, mp.maxMeshOutputMemorySize,
                   mp.maxMeshPayloadAndOutputMemorySize, mp.maxMeshOutputVertices, mp.maxMeshOutputPrimitives,
                   mp.maxMeshOutputComponents, mp.meshOutputPerVertexGranularity, mp.meshOutputPerPrimitiveGranularity);
        }
        __atomic_store_n(&MESH.ready, MESH.draw && MESH.draw_ind ? 1 : 2, __ATOMIC_RELEASE);
    }
    return MESH.ready == 1;
}

void vk_pipeline_draw_mesh(VkCommandBuffer cb, uint32_t groups) { rec_draw_mesh(MESH.draw, cb, groups, 1, 1); }

void vk_pipeline_draw_mesh_indirect(VkCommandBuffer cb, VkBuffer b, VkDeviceSize off) {
    rec_draw_mesh_indirect(MESH.draw_ind, cb, b, off, 1, sizeof(VkDrawMeshTasksIndirectCommandEXT));
}

// mesh: 0 a geometry pipeline (vertex, geometry, fragment), 1 a mesh pipeline (task, mesh, fragment), 2 an emulated mesh
// draw (vk_exec_draw.c mesh_emulate): gfx_shader_meshc_vs's vertex shader and the fragment shader, the vertices from
// the emulation buffer (binding 0: vec4s of 16 bytes, one a vertex input) as a triangle list.
// Test only (REKARINA_MESH_COLD=<text>): the module with an OpModuleProcessed "<text>" added after its debug names, so
// the driver's caches know it under another hash and compile it cold, as a player's driver does for a pairing met for
// the first time. NULL when there is no place for it.
static uint32_t *spirv_add_processed(const uint32_t *w, uint64_t n, const char *text, uint64_t *out_n) {
    uint64_t at = 5;
    while (at < n) {
        uint32_t op = w[at] & 0xffff, len = w[at] >> 16;
        if (!len || at + len > n) return NULL;
        if (!(op <= 7 || op == 10 || op == 11 || (op >= 14 && op <= 17) || op == 330 || op == 331)) break;
        at += len;
    }
    size_t sl = strlen(text);
    uint32_t sw = (uint32_t)(sl / 4 + 1);
    uint32_t *o = calloc(n + 1 + sw, 4);
    if (!o) return NULL;
    memcpy(o, w, at * 4);
    o[at] = (1 + sw) << 16 | 330;
    memcpy(o + at + 1, text, sl);
    memcpy(o + at + 1 + sw, w + at, (n - at) * 4);
    *out_n = n + 1 + sw;
    return o;
}

static VkPipeline stages_create(const VkPipeState *s, const GfxShader *const sh[3], int mesh, const GKey *k,
                                const Layout *l, const uint32_t off[3]) {
    static const int gst[3][3] = {{GFX_STAGE_VS, GFX_STAGE_GS, GFX_STAGE_FS}, {GFX_STAGE_TS, GFX_STAGE_MS, GFX_STAGE_FS},
                                  {GFX_STAGE_VS, GFX_STAGE_GS, GFX_STAGE_FS}};
    VkShaderModule m[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkPipelineShaderStageCreateInfo st[3];
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rq[3];
    uint32_t n = 0;
    VkPipeline pipe = VK_NULL_HANDLE;
    for (int i = 0; i < 3; i++) {
        if (!sh[i]) continue;
        uint64_t w;
        const uint32_t *code = gfx_shader_spirv(sh[i], &w);
        uint32_t *moved = off[i] ? spirv_rebind(code, w, off[i]) : NULL;
        if (off[i] && !moved) goto out;
        uint32_t *cold = NULL;
        uint64_t cw = 0;
        static const char *cold_text;   // DEBUG-MESHCOLD: REKARINA_MESH_COLD, the mesh stages of builds in play only
        static int cold_read;
        if (!__atomic_load_n(&cold_read, __ATOMIC_ACQUIRE)) cold_text = cfg_get("REKARINA_MESH_COLD"), __atomic_store_n(&cold_read, 1, __ATOMIC_RELEASE);
        if (mesh == 1 && i == 1 && cold_text && *cold_text && prep_in_play())
            cold = spirv_add_processed(moved ? moved : code, w, cold_text, &cw);
        m[i] = cold ? make_module(cold, cw) : make_module(moved ? moved : code, w);
        free(moved), free(cold);
        if (!m[i]) goto out;
        st[n] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                  .stage = (VkShaderStageFlagBits)stage_bit(gst[mesh][i]), .module = m[i],
                                                  .pName = "main"};
        // a mesh pipeline's task and mesh stages only where vk_caps.c pins them (not on a device of 32-lane subgroups)
        if (mesh != 1 || (vk_caps.sg_pin_mesh && i < 2)) pin_subgroup32(&st[n], &rq[n], code, w);
        n++;
    }
    VkVertexInputAttributeDescription va[32];
    VkVertexInputBindingDescription vb[17];
    uint32_t na = 0, nb = 0;
    if (!mesh && !P.dyn_vi) {
        VtxInput v;
        vertex_input(s, &v);
        for (uint32_t i = 0; i < v.n_attr; i++) va[na++] = (VkVertexInputAttributeDescription){v.a[i].location, v.a[i].binding, v.a[i].format, v.a[i].offset};
        for (uint32_t i = 0; i < v.n_bind; i++) vb[nb++] = (VkVertexInputBindingDescription){v.b[i].binding, v.b[i].stride, v.b[i].inputRate};
    } else if (mesh == 2) {
        GfxShaderIo io[32];
        uint32_t in = gfx_shader_inputs(sh[0], io);
        for (uint32_t i = 0; i < 32; i++)
            if (in >> i & 1) va[na++] = (VkVertexInputAttributeDescription){i, 0, VK_FORMAT_R32G32B32A32_SFLOAT, 16 * i};
        vb[nb++] = (VkVertexInputBindingDescription){0, 16 * na, VK_VERTEX_INPUT_RATE_VERTEX};
    }
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                                               .vertexBindingDescriptionCount = nb, .pVertexBindingDescriptions = vb,
                                               .vertexAttributeDescriptionCount = na, .pVertexAttributeDescriptions = va};
    static const VkPrimitiveTopology class_base[4] = {VK_PRIMITIVE_TOPOLOGY_POINT_LIST, VK_PRIMITIVE_TOPOLOGY_LINE_LIST,
                                                      VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, VK_PRIMITIVE_TOPOLOGY_PATCH_LIST};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = mesh == 2 ? VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST : class_base[k->topo_class]};
    VkPipelineViewportDepthClipControlCreateInfoEXT clip = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_DEPTH_CLIP_CONTROL_CREATE_INFO_EXT,
                                                            .negativeOneToOne = k->minus_w};
    VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO, .pNext = &clip};
    VkPipelineRasterizationProvokingVertexStateCreateInfoEXT pvs = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_PROVOKING_VERTEX_STATE_CREATE_INFO_EXT,
        .provokingVertexMode = VK_PROVOKING_VERTEX_MODE_LAST_VERTEX_EXT};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .pNext = P.pv_last ? &pvs : NULL,
                                                 .depthClampEnable = k->clamp, .polygonMode = (VkPolygonMode)k->poly_mode,
                                                 .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = (VkSampleCountFlagBits)k->samples,
                                               .alphaToCoverageEnable = k->a2c};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
                                                .maxDepthBounds = 1.0f};
    VkPipelineColorBlendAttachmentState cba[8];
    memset(cba, 0, sizeof cba);
    for (int i = 0; i < k->n_color; i++) {
        const uint8_t *b = k->blend[i];
        cba[i].blendEnable = k->blend_en >> i & 1;
        cba[i].srcColorBlendFactor = (VkBlendFactor)b[0], cba[i].dstColorBlendFactor = (VkBlendFactor)b[1];
        cba[i].colorBlendOp = (VkBlendOp)b[2], cba[i].srcAlphaBlendFactor = (VkBlendFactor)b[3];
        cba[i].dstAlphaBlendFactor = (VkBlendFactor)b[4], cba[i].alphaBlendOp = (VkBlendOp)b[5];
        cba[i].colorWriteMask = k->wmask >> (4 * i) & 0xf;
    }
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .logicOpEnable = k->logic_op != 0,
                                              .logicOp = k->logic_op ? (VkLogicOp)(k->logic_op - 1) : VK_LOGIC_OP_COPY,
                                              .attachmentCount = k->n_color, .pAttachments = cba};
    VkDynamicState dyn[32];
    uint32_t ndyn = 0;
    for (uint32_t i = 0; i < P.n_dyn; i++)
        if (!mesh || (P.dyn[i] != VK_DYNAMIC_STATE_VERTEX_INPUT_EXT && P.dyn[i] != VK_DYNAMIC_STATE_VERTEX_INPUT_BINDING_STRIDE &&
                      P.dyn[i] != VK_DYNAMIC_STATE_PRIMITIVE_TOPOLOGY && P.dyn[i] != VK_DYNAMIC_STATE_PRIMITIVE_RESTART_ENABLE))
            dyn[ndyn++] = P.dyn[i];
    if (mesh == 1 && vk_ctx()->has_vrs) dyn[ndyn++] = VK_DYNAMIC_STATE_FRAGMENT_SHADING_RATE_KHR;   // vk_exec_draw.c mesh_shading_rate
    VkPipelineDynamicStateCreateInfo dy = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = ndyn, .pDynamicStates = dyn};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO, .colorAttachmentCount = k->n_color,
                                        .pColorAttachmentFormats = (const VkFormat *)k->color,
                                        .depthAttachmentFormat = (VkFormat)k->depth, .stencilAttachmentFormat = (VkFormat)k->stencil};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
                                       .stageCount = n, .pStages = st, .pVertexInputState = mesh == 1 ? NULL : &vi,
                                       .pInputAssemblyState = mesh == 1 ? NULL : &ia, .pViewportState = &vp, .pRasterizationState = &rs,
                                       .pMultisampleState = &ms, .pDepthStencilState = &ds, .pColorBlendState = &cb,
                                       .pDynamicState = &dy, .layout = l->layout};
    uint64_t p0 = perf_begin();
    if (pipestats_on()) pi.flags |= VK_PIPELINE_CREATE_CAPTURE_STATISTICS_BIT_KHR;   // DEBUG-PIPESTATS
    VkResult r = vkCreateGraphicsPipelines(P.device, P.cache, 1, &pi, NULL, &pipe);
    perf_end(PERF_PIPE, p0);
    if (r == VK_SUCCESS && pipestats_on()) {
        VkShaderStageFlags pst[3];
        for (int i = 0; i < 3; i++) pst[i] = stage_bit(gst[mesh][i]);
        pipestats_log(pipe, sh, pst, 3);
    }
    if (r != VK_SUCCESS) {
        rt_log("vk_pipeline: vkCreateGraphicsPipelines failed (%d) for a %s pipeline", (int)r, mesh ? "mesh" : "geometry");
        pipe = VK_NULL_HANDLE;
    } else {
        perf_count(PERF_CNT_PIPE_BUILT);
    }
out:
    for (int i = 0; i < 3; i++)
        if (m[i]) vkDestroyShaderModule(P.device, m[i], NULL);
    return pipe;
}

// Background builds of mesh pipelines (OoT): NVIDIA's compiler takes seconds for the larger translated mesh shaders
// (16 s for one of 102k SPIR-V words), and vk_pipeline_stages built them on the drawing thread under P.lock, so the
// game stopped that long at each new scene. A miss now enters the Pipe as building and queues the build to threads
// of their own (MESH_THREADS: the driver compiles pipelines in parallel, and a new scene asks for a dozen at once);
// draws with it are skipped until the handle is published, as background graphics builds skip theirs.
// REKARINA_MESH_ASYNC=0 builds them on the drawing thread as before. Jobs copy the state and the key; shaders live for
// the process and layouts until vk_pipeline_shutdown, which waits for the queue first (mesh_drain).
typedef struct MeshJob {
    struct MeshJob *next;
    Pipe *p;
    VkPipeState t;
    const GfxShader *sh[3];
    int mesh;
    GKey k;
    Layout *l;
    uint32_t off[3];
    int play;                          // queued after the guest started: its build goes into the ledger (prep.c)
} MeshJob;
static void mk_note(const VkPipeState *t, const GfxShader *const sh[3], int mesh);   // the mesh keys, below
#define MESH_THREADS 4
static struct {
    int on;                            // -1 unread, 0 off, 1 on
    PlatThread *thread[MESH_THREADS];
    MeshJob *head, *tail;
    int busy;                          // queued or building
    PlatCond work, idle;
} MJ = {.on = -1, .work = PLAT_COND_INIT, .idle = PLAT_COND_INIT};

static void mesh_log_slow(const GfxShader *const sh[3], int mesh, double ms) {
    if (ms <= 250) return;
    char hx[3][17] = {"-", "-", "-"};
    uint64_t w[3] = {0, 0, 0};
    for (int i = 0; i < 3; i++) {
        if (!sh[i]) continue;
        const uint8_t *k8 = gfx_shader_key(sh[i]);
        for (int j = 0; j < 8; j++) snprintf(hx[i] + 2 * j, 3, "%02x", k8[j]);
        gfx_shader_spirv(sh[i], &w[i]);
    }
    rt_log("vk_pipeline: %s pipeline took %.0f ms (shader keys %s %s %s, SPIR-V words %llu %llu %llu)",
           mesh ? "mesh" : "geometry", ms, hx[0], hx[1], hx[2], (unsigned long long)w[0], (unsigned long long)w[1],
           (unsigned long long)w[2]);
}

static void mesh_worker(void *arg) {
    (void)arg;
    plat_thread_priority(PLAT_PRIO_BELOW_NORMAL);   // the guest's six cores keep running while the driver compiles
    plat_mutex_lock(&P.lock);
    for (;;) {
        while (!MJ.head) plat_cond_wait(&MJ.work, &P.lock);
        MeshJob *j = MJ.head;
        if (!(MJ.head = j->next)) MJ.tail = NULL;
        plat_mutex_unlock(&P.lock);
        double t0 = now_ms();
        VkPipeline pipe = stages_create(&j->t, j->sh, j->mesh, &j->k, j->l, j->off);
        double ms = now_ms() - t0;
        mesh_log_slow(j->sh, j->mesh, ms);
        if (j->play) prep_note(j->mesh ? PREP_MESH : PREP_GEOM, 0, ms);
        {   // each one built in play: its draws were left out meanwhile (vk_exec_draw.c left_out names them)
            static volatile int32_t logs;
            if (ms <= 250 && __atomic_fetch_add(&logs, 1, __ATOMIC_RELAXED) < 300)
                rt_log("vk_pipeline: frame %llu: a %s pipeline was built beside the replay in %.1f ms",
                       (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1,
                       j->mesh ? "mesh" : "geometry", ms);
        }
        plat_mutex_lock(&P.lock);
        j->p->info.pipeline = pipe;
        j->p->building = 0;
        if (pipe) {
            P.stats.graphics++;
            mk_note(&j->t, j->sh, j->mesh);
            LOG_ONCE("vk_pipeline: first %s pipeline built (in the background)", j->mesh ? "mesh" : "geometry");
        } else {
            P.stats.failures++;
            rt_log("vk_pipeline: %s pipeline not built: pipeline creation failed", j->mesh ? "mesh" : "geometry");
        }
        free(j);
        if (!--MJ.busy) plat_cond_broadcast(&MJ.idle);
    }
}

// Whether misses build in the background; starts the thread on first use. Caller holds P.lock.
static int mesh_async_on(void) {
    if (MJ.on < 0) {
        MJ.on = rt_knob("MESH_ASYNC", "1") != 0;
        int n = 0;
        for (int i = 0; MJ.on && i < MESH_THREADS; i++)
            if ((MJ.thread[n] = plat_thread_new(mesh_worker, NULL, "mesh pipelines"))) n++;
        if (!n) MJ.on = 0;
        rt_log("vk_pipeline: mesh and geometry pipelines build %s (%d threads)",
               MJ.on ? "in the background (draws skip them meanwhile)" : "on the drawing thread", n);
    }
    return MJ.on;
}

// Queues p's build; without memory p stays a failed entry. Caller holds P.lock.
static void mesh_queue(Pipe *p, const VkPipeState *t, const GfxShader *const sh[3], int mesh, const GKey *k, Layout *l,
                       const uint32_t off[3]) {
    MeshJob *j = calloc(1, sizeof *j);
    if (!j) return;
    j->p = p, j->t = *t, j->mesh = mesh, j->k = *k, j->l = l, j->play = prep_in_play();
    for (int i = 0; i < 3; i++) j->sh[i] = sh[i], j->off[i] = off[i];
    p->building = 1;
    if (MJ.tail) MJ.tail->next = j;
    else MJ.head = j;
    MJ.tail = j;
    MJ.busy++;
    plat_cond_signal(&MJ.work);
}

static void mesh_drain(void) {
    plat_mutex_lock(&P.lock);
    while (MJ.busy) plat_cond_wait(&MJ.idle, &P.lock);
    plat_mutex_unlock(&P.lock);
}

static _Thread_local int stages_pending;   // the last vk_pipeline_stages of this thread left its draw out

int vk_pipeline_stages_pending(void) { return stages_pending; }

typedef struct { char tag[8]; const GfxShader *sh[3]; uint32_t mesh; GKey k; } StagesKey;

// The table key of a mesh or geometry pipeline, and in *t the state it is built from (s with the stages in place).
static void stages_key(StagesKey *key, VkPipeState *t, const VkPipeState *s, const GfxShader *const sh[3], int mesh) {
    memset(key, 0, sizeof *key);
    memcpy(key->tag, "stages", 6);
    for (int i = 0; i < 3; i++) key->sh[i] = sh[i];
    key->mesh = (uint32_t)mesh;
    *t = *s;
    t->vs = mesh == 1 ? NULL : sh[0], t->fs = sh[2];
    gkey_fill(t, &key->k);
    key->k.indep = 0;
    if (mesh) memset(key->k.attr_fmt, 0, sizeof key->k.attr_fmt + sizeof key->k.attr_off + sizeof key->k.attr_bind + sizeof key->k.rate);
}

// The stages' layout and binding offsets; NULL with *why set when there is none. Caller holds P.lock.
static Layout *stages_layout(const GfxShader *const sh[3], Bindings *B, uint32_t off[3], const char **why) {
    uint32_t next = 0;
    for (int i = 0; i < 3 && !*why; i++) {
        if (!sh[i]) continue;
        off[i] = next;
        if (!add_stage(B, sh[i], off[i])) *why = "too many bindings";
        next += binding_span(sh[i]) - 1;
    }
    Layout *l = *why ? NULL : layout_get(B);
    if (!*why && !l) *why = "descriptor layout creation failed";
    return l;
}

static void mk_note(const VkPipeState *t, const GfxShader *const sh[3], int mesh);

// ---- mesh draws on the graphics pipeline library
// A mesh pipeline met in play cost NVIDIA's compiler 0.1 to 11 s where the driver cache did not have it (the task and
// mesh stages of the larger foliage shaders are 100k SPIR-V words, and a whole pipeline compiles them again for each
// fragment shader they are paired with), and its draws were left out the whole time: bushes gone for seconds after
// an event's camera cut (the user's session of 2026-11-17: 145 ms and 8.4 s). Every one of those met in play was a new
// pairing, mostly of a task and mesh stage seen before with a fragment shader not seen with them. So a mesh pipeline
// is also linked from libraries: a pre-rasterization library of its task and mesh stages (LIB_PRM, one per pair of
// task and mesh variants, whatever the fragment shader) and the fragment shader and fragment output libraries ordinary
// draws share. The entry step builds the libraries of every recorded mesh pipeline (mesh_libs_ahead), so
// rekarina.exe --warm-shaders at the setup puts them into the driver cache, and the fragment libraries of the
// prepared variants are there from the preparation (lb_warm_prepared); what is missing a pipe worker builds while the
// draw waits (from the driver cache, about a millisecond). The new pairing is then a link, and the draw is drawn.
// The table entry and its optimised build stay those of stages_create, with the shared layout and the same modules,
// so a later entry step finds that pipeline in the driver cache; the linked one has a layout of its own (task and
// mesh in set 0, the fragment stage in set 1: layout_pair) and lives in Pipe.alt, which draws use until the
// optimised build is published. It has no fragment shading rate state (the ordinary draws' fragment library has
// none), so shadow foliage shades at full rate until then.
// Built for every device with the graphics pipeline library and mesh shaders, and on by default where runs have shown it
// works (NVIDIA's and AMD's Windows drivers; mesh_gpl_on). The ordinary draws' fragment library pins 32
// lanes, which a whole mesh pipeline does not do for its fragment stage; where subgroups have one size (NVIDIA) that
// means nothing and the library is shared, elsewhere (AMD, Intel) a mesh pipeline's fragment stage gets a library of
// its own without the pin (FsKey.nopin), built as stages_create builds it.
// REKARINA_MESH_GPL=0 turns it off (draws are left out while the pipeline builds, as before), =1 on where it is not
// proven yet, auto (default) on where it is.
// REKARINA_MESH_WAIT_MS (default 4) is how long the first draw of a new pairing waits for missing libraries, at most
// twice that per presented frame in all; past it the draw is left out until they are there.
static struct {
    int on;                            // -1 unread
    double wait_ms;
    uint64_t frame;                    // the presented frame the budget below belongs to
    double left_ms;                    // wait budget left in it
    uint64_t links, waits, timeouts, finals, libs_ahead;
    double link_ms, link_max, wait_sum, wait_max;
} MG = {.on = -1};

static int mesh_gpl_on(void) {   // caller holds P.lock
    if (MG.on < 0) {
        const char *e = cfg_get("REKARINA_MESH_GPL"), *w = cfg_get("REKARINA_MESH_WAIT_MS");
        // auto (the default): on where runs have shown task and mesh libraries build and link (NVIDIA's driver, and
        // AMD's on an integrated Radeon: 461 libraries built, links of 0.9 ms on average, no failure); 1 on
        // wherever the graphics pipeline library and mesh shaders are there, 0 off
        int forced = e && *e && strcmp(e, "auto") ? atoi(e) != 0 : -1;
        int proven = vk_ctx()->props.vendorID == 0x10de || vk_caps.amd_prop;
        int want = forced >= 0 ? forced : proven;
        MG.on = want && P.gpl && P.indep && vk_ctx()->has_mesh_shader;
        MG.wait_ms = w && *w ? atof(w) : 4.0;
        if (MG.wait_ms < 0) MG.wait_ms = 0;
        rt_log("vk_pipeline: mesh draws %s", MG.on ? "get a fast-linked pipeline from stage libraries while theirs builds"
                                             : forced == 0 ? "are left out while their pipeline builds (REKARINA_MESH_GPL=0)"
                                             : !want ? "are left out while their pipeline builds (stage libraries of mesh "
                                                       "pipelines not proven on this driver yet: REKARINA_MESH_GPL=1 tries them)"
                                                     : "are left out while their pipeline builds (no graphics pipeline library)");
    }
    return MG.on;
}

// The stand-in's layout (task and mesh in set 0 as stages_layout numbers them, the fragment shader in set 1) and its
// library keys; NULL when there is none. *B holds both sets' bindings. Caller holds P.lock.
static Layout *mesh_lib_keys(const GfxShader *const sh[3], const GKey *k, Bindings *B, PrmKey *pm, FsKey *fk, FoKey *ok) {
    Bindings F = {0};
    uint32_t off = 0;
    memset(B, 0, sizeof *B);
    for (int i = 0; i < 2; i++) {
        if (!sh[i]) continue;
        if (!add_stage(B, sh[i], off)) return NULL;
        off += binding_span(sh[i]) - 1;
    }
    if (sh[2] && !add_stage(&F, sh[2], 0)) return NULL;
    Layout *l = layout_pair(B, &F);
    if (!l || !half_layout(l, 0) || !half_layout(l, 1)) return NULL;
    memset(pm, 0, sizeof *pm);
    pm->kind = LIB_PRM, pm->clamp = k->clamp, pm->poly_mode = k->poly_mode, pm->minus_w = k->minus_w;
    pm->ts = sh[0], pm->ms = sh[1], pm->samples = k->samples ? k->samples : 1;
    memset(fk, 0, sizeof *fk);
    fk->kind = LIB_FS, fk->fs = sh[2], fk->dsb = k->dsb, fk->samples = k->samples, fk->a2c = k->a2c;
    pm->pool = fk->pool = l->pool != 0;
    fk->nopin = !P.sg_fixed32;   // the ordinary draws' library pins 32 lanes, which only means nothing where all are 32
    memset(ok, 0, sizeof *ok);
    ok->kind = LIB_FO, ok->n_color = k->n_color, ok->samples = k->samples, ok->a2c = k->a2c, ok->logic_op = k->logic_op;
    ok->blend_en = k->blend_en, ok->depth = k->depth, ok->stencil = k->stencil, ok->wmask = k->wmask;
    memcpy(ok->color, k->color, sizeof ok->color), memcpy(ok->blend, k->blend, sizeof ok->blend);
    return l;
}

// The library for key, or a new building entry with its job queued: to the pipe workers (U, ahead of every pipeline
// job), or with pre to the entry step's builders (MQ). NULL without memory. *queued is raised for a new one. Caller
// holds P.lock.
static Lib *lib_need(int kind, const void *key, uint32_t size, Layout *l, const VkPipeState *t, const GKey *k,
                     int pre, int *queued) {
    uint64_t h = key_hash(key, size);
    Lib *b = lib_find(h, key, size);
    if (b) return b;
    LibJob *j = calloc(1, sizeof *j);
    if (!j || !(b = lib_new(h, key, size))) {
        free(j);
        return NULL;
    }
    *j = (LibJob){.b = b, .kind = kind, .l = l, .s = *t, .k = *k};
    b->job = j, b->pre = (uint8_t)pre;
    if (pre) {   // the entry step's: its builders take it (MQ), and it counts in the step's progress
        prep_progress_total(1);
        if (MQ.tail) MQ.tail->next = j;
        else MQ.head = j;
        MQ.tail = j;
    } else {
        if (U.tail) U.tail->next = j;
        else U.head = j;
        U.tail = j;
        U.queued++;
    }
    ++*queued;
    return b;
}

// The entry step: the libraries a recorded mesh pipeline would be linked from, queued so the builders make them.
// Caller holds P.lock.
static void mesh_libs_ahead(const VkPipeState *t, const GfxShader *const sh[3], const GKey *k) {
    if (k->dsb || k->fbl) return;
    Bindings B;
    PrmKey pm;
    FsKey fk;
    FoKey ok;
    Layout *l = mesh_lib_keys(sh, k, &B, &pm, &fk, &ok);
    if (!l) return;
    int queued = 0;
    lib_need(LIB_PRM, &pm, sizeof pm, l, t, k, 1, &queued);
    // REKARINA_MESH_FS_AHEAD=0: the fragment libraries are left to the first draw that needs one (from the driver
    // cache, about a millisecond), so the entry step keeps only the task and mesh libraries in video memory
    static int fs_ahead = -1;
    if (fs_ahead < 0) fs_ahead = rt_knob("MESH_FS_AHEAD", "1");
    if (fs_ahead) lib_need(LIB_FS, &fk, sizeof fk, l, t, k, 1, &queued);
    MG.libs_ahead += (uint64_t)queued;
}

// p's fast-linked stand-in (p builds), or NULL while its libraries are not there. The first draw that needs it waits
// for them within the budget; a failed library or a link GPL cannot make ends it for p (alt_final). Caller holds
// P.lock, which waits and builds release.
static Pipe *mesh_fast(Pipe *p, const VkPipeState *t, const GfxShader *const sh[3], const GKey *k) {
    if (p->alt) return p->alt;
    if (p->alt_final) return NULL;
    Bindings B;
    PrmKey pm;
    FsKey fk;
    FoKey ok;
    Layout *l = k->dsb || k->fbl ? NULL : mesh_lib_keys(sh, k, &B, &pm, &fk, &ok);   // stages_create makes neither
    if (!l) {
        p->alt_final = 1, MG.finals++;
        return NULL;
    }
    int queued = 0;
    Lib *b[2] = {lib_need(LIB_PRM, &pm, sizeof pm, l, t, k, 0, &queued), lib_need(LIB_FS, &fk, sizeof fk, l, t, k, 0, &queued)};
    if (!b[0] || !b[1]) return NULL;   // out of memory: a later draw tries again
    if (queued && workers_start()) plat_cond_broadcast(&work_cv);
    if ((b[0]->building || b[1]->building) && !p->alt_waited) {   // once per pipeline, within the frame's budget
        p->alt_waited = 1;
        uint64_t f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (f != MG.frame) MG.frame = f, MG.left_ms = 2 * MG.wait_ms;
        double w = MG.wait_ms < MG.left_ms ? MG.wait_ms : MG.left_ms;
        if (w > 0) {   // never built on this thread: a library the driver cache lacks takes it seconds
            double t0 = now_ms(), end = t0 + w;
            for (;;) {
                if (p->alt || !p->building || (!b[0]->building && !b[1]->building)) break;
                double now = now_ms();
                if (now >= end) break;
                plat_cond_wait_ms(&done_cv, &P.lock, (uint32_t)(end - now) + 1);
            }
            double ms = now_ms() - t0;
            MG.left_ms = ms < MG.left_ms ? MG.left_ms - ms : 0;
            MG.waits++, MG.wait_sum += ms;
            if (ms > MG.wait_max) MG.wait_max = ms;
            if (b[0]->building || b[1]->building) MG.timeouts++;
        }
        if (p->alt) return p->alt;   // another draw linked it while this one waited
        if (!p->building) return NULL;   // the optimised build came first: the caller takes it
    }
    if (b[0]->building || b[1]->building) return NULL;
    if (!b[0]->pipe || !b[1]->pipe) {
        p->alt_final = 1, MG.finals++;
        return NULL;
    }
    int miss = 0;
    VkPipeline fo = lib_get(LIB_FO, t, k, l, &ok, sizeof ok, 1, &miss);   // no shader in it: built here when missing
    if (p->alt) return p->alt;   // lib_get may have released P.lock
    if (!fo) {
        p->alt_final = 1, MG.finals++;
        return NULL;
    }
    VkPipeline lib[3] = {b[0]->pipe, b[1]->pipe, fo}, f = VK_NULL_HANDLE;
    VkPipelineLibraryCreateInfoKHR li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, .libraryCount = 3,
                                         .pLibraries = lib};
    VkGraphicsPipelineCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &li,
                                       .layout = l->layout};
    double t1 = now_ms();
    if (vkCreateGraphicsPipelines(P.device, VK_NULL_HANDLE, 1, &pi, NULL, &f) != VK_SUCCESS) f = VK_NULL_HANDLE;
    double ms = now_ms() - t1;
    Pipe *a = f ? pipe_new(p->key, p->key_size, p->hash, f, l, &B, VK_PIPELINE_BIND_POINT_GRAPHICS, 0) : NULL;
    if (!a) {
        if (f) vkDestroyPipeline(P.device, f, NULL);
        else rt_log("vk_pipeline: a mesh pipeline's fast link failed: its draws are left out until it is built");
        p->alt_final = 1, MG.finals++;
        return NULL;
    }
    a->fast = f;
    a->info.mesh_standin = 1;
    p->alt = a;
    MG.links++, MG.link_ms += ms;
    if (ms > MG.link_max) MG.link_max = ms;
    LOG_ONCE("vk_pipeline: first mesh draw on a fast-linked pipeline (link %.2f ms)", ms);
    static int logs;
    if (prep_in_play() && logs++ < 200)
        rt_log("vk_pipeline: frame %llu: a mesh draw got a fast-linked pipeline (link %.2f ms; its own builds beside)",
               (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1, ms);
    return a;
}

VkPipeline vk_pipeline_stages(const VkPipeState *s, const GfxShader *const sh[3], int mesh, const VkPipeInfo **info) {
    if (info) *info = NULL;
    stages_pending = 0;
    if ((mesh == 2 ? !sh[0] : !sh[1]) || !ensure_init() || (mesh == 1 && !vk_pipeline_mesh_on())) return VK_NULL_HANDLE;
    StagesKey key;
    VkPipeState t;
    stages_key(&key, &t, s, sh, mesh);
    uint64_t h = key_hash(&key, sizeof key);
    plat_mutex_lock(&P.lock);
    Pipe *p = pipe_find(&key, sizeof key, h);
    if (p) {
        P.stats.hits++;
    } else {
        const char *why = NULL;
        Bindings B = {0};
        uint32_t off[3] = {0, 0, 0};
        Layout *l = stages_layout(sh, &B, off, &why);
        VkPipeline pipe = VK_NULL_HANDLE;
        if (!why && mesh_async_on()) {
            p = pipe_add(&key, sizeof key, h, VK_NULL_HANDLE, l, &B, VK_PIPELINE_BIND_POINT_GRAPHICS, 0);
            if (p) mesh_queue(p, &t, sh, mesh, &key.k, l, off);
            if (!p) {
                plat_mutex_unlock(&P.lock);
                return VK_NULL_HANDLE;
            }
        } else {
            double t0 = now_ms();
            if (!why && !(pipe = stages_create(&t, sh, mesh, &key.k, l, off))) why = "pipeline creation failed";
            if (!why) mesh_log_slow(sh, mesh, now_ms() - t0);
            prep_note(mesh ? PREP_MESH : PREP_GEOM, now_ms() - t0, now_ms() - t0);
            if (why) {
                P.stats.failures++;
                rt_log("vk_pipeline: %s pipeline not built: %s", mesh ? "mesh" : "geometry", why);
            } else {
                P.stats.graphics++;
                mk_note(&t, sh, mesh);
                LOG_ONCE("vk_pipeline: first %s pipeline built", mesh ? "mesh" : "geometry");
            }
            p = pipe_add(&key, sizeof key, h, pipe, l, l ? &B : NULL, VK_PIPELINE_BIND_POINT_GRAPHICS, 0);
        }
    }
    if (p && p->building) {   // mesh_queue: a background build owns it; a mesh draw takes the fast-linked stand-in
        Pipe *a = mesh == 1 && mesh_gpl_on() ? mesh_fast(p, &t, sh, &key.k) : NULL;   // may release P.lock
        if (p->building) {
            VkPipeline ap = a ? a->info.pipeline : VK_NULL_HANDLE;
            plat_mutex_unlock(&P.lock);
            if (!ap) {
                stages_pending = 1;
                return VK_NULL_HANDLE;
            }
            if (info) *info = &a->info;
            return ap;
        }
    }
    static int alt_only = -1;   // DEBUG-MESHALT: REKARINA_MESH_ALT_ONLY=1: a stand-in stays after its pipeline is built,
    if (alt_only < 0) alt_only = rt_knob("MESH_ALT_ONLY", "0");   // so shots show what stand-ins draw
    if (alt_only && p && p->alt && p->alt->info.pipeline) {
        Pipe *a = p->alt;
        plat_mutex_unlock(&P.lock);
        if (info) *info = &a->info;
        return a->info.pipeline;
    }
    plat_mutex_unlock(&P.lock);
    if (!p) return VK_NULL_HANDLE;
    if (info) *info = &p->info;
    return p->info.pipeline;
}

// ---- mesh and geometry pipeline keys (pipeline-keys-mesh.bin beside pipeline-keys.bin)
// A mesh or geometry pipeline is built at its first draw, beside the replay (mesh_queue), and its draws are left out
// until it is there: the grass, bushes and tree crowns of a place met for the first time in a session were missing
// for a frame or more (54 such builds at the forest save's first frames, 0.2 to 3 ms each from the driver's cache;
// seconds where the cache is cold). Each one built in play appends a MeshKeyRec, the counterpart of the graphics
// KeyRec: the three stages' input ids, the mesh kind and the state it was built from with the shader pointers
// cleared. At the next boot the prewarm loads their shaders and queues every record to the mesh threads before the
// guest runs (the entry step, mk_queue; it waits for them), so play finds them built. A release's own
// records come as pipeline-keys-mesh-seed.bin (tools/make_seed_keys.py, tools/package_oot.py). Shares the record
// hash set and the writer lock with the graphics keys. Bump MKEYS_VERSION when VkPipeState or the record changes
// meaning without changing size.
#define MKEYS_VERSION 1
typedef struct {
    uint8_t sh[3][32];                 // gfx_shader_input_id of task (or vertex), mesh (or geometry), fragment; zero: none
    uint32_t has, mesh;                // has: bit i for sh[i]; mesh: vk_pipeline_stages' kind (0 geometry, 1 mesh)
    VkPipeState s;                     // the state built from (stages_key's t), vs and fs zero
} MeshKeyRec;
static struct {
    char path[PLAT_PATH_MAX * 2];
    int header_ok;
    MeshKeyRec *pend, *file;           // records not written yet; the file (and seed) as read at boot
    uint32_t pend_n, pend_cap;
    size_t file_n;
} MK;

static uint64_t mk_hash(const MeshKeyRec *r) {
    uint64_t h = fnv(FNV0 ^ 0x6d657368ull, r, sizeof *r);   // its own seed: one set holds both kinds of record
    return h ? h : 1;
}

static int mk_pending(void) { return *(volatile uint32_t *)&MK.pend_n != 0; }

// A mesh or geometry pipeline was built for a draw. Caller holds P.lock.
static void mk_note(const VkPipeState *t, const GfxShader *const sh[3], int mesh) {
    if (!A.on || !MK.path[0] || mesh == 2) return;   // 2: the runtime's own vertex stage (no input id)
    static const uint8_t none[32];
    MeshKeyRec r;
    memset(&r, 0, sizeof r);
    for (int i = 0; i < 3; i++) {
        if (!sh[i]) continue;
        const uint8_t *id = gfx_shader_input_id(sh[i]);
        if (!id || !memcmp(id, none, 32)) return;
        memcpy(r.sh[i], id, 32), r.has |= 1u << i;
    }
    r.mesh = (uint32_t)mesh;
    r.s = *t;
    r.s.vs = r.s.fs = NULL;
    if (!seen_add(mk_hash(&r))) return;
    if (MK.pend_n == MK.pend_cap) {
        uint32_t cap = MK.pend_cap ? MK.pend_cap * 2 : 64;
        MeshKeyRec *n = realloc(MK.pend, cap * sizeof *n);
        if (!n) return;
        MK.pend = n, MK.pend_cap = cap;
    }
    MK.pend[MK.pend_n++] = r;
}

// Appends the pending records (present thread or exit; never the render thread).
static void mk_flush(void) {
    if (!MK.path[0]) return;
    plat_mutex_lock(&K.io);
    plat_mutex_lock(&P.lock);
    MeshKeyRec *r = MK.pend;
    uint32_t n = MK.pend_n;
    MK.pend = NULL, MK.pend_n = MK.pend_cap = 0;
    plat_mutex_unlock(&P.lock);
    if (n) {
        FILE *f = fopen(MK.path, MK.header_ok ? "ab" : "wb");
        int ok = f != NULL;
        if (ok && !MK.header_ok) {
            KeysHeader h = {{'T', 'O', 'T', 'K', 'M', 'K', 'E', 'Y'}, MKEYS_VERSION, sizeof(MeshKeyRec)};
            ok = fwrite(&h, sizeof h, 1, f) == 1;
        }
        if (ok) ok = fwrite(r, sizeof *r, n, f) == n;
        if (ok) ok = plat_file_sync(f);
        if (f) ok = !fclose(f) && ok;
        if (ok) MK.header_ok = 1;
        else rt_log("vk_pipeline: cannot write %s", MK.path);
    }
    free(r);
    plat_mutex_unlock(&K.io);
}

// Reads one record file into MK.file (new records only); the count read, -1 without the file or with another layout.
static long mk_read(const char *path, int own) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    KeysHeader h;
    long n = -1;
    if (fread(&h, sizeof h, 1, f) == 1 && !memcmp(h.magic, "TOTKMKEY", 8) && h.version == MKEYS_VERSION &&
        h.rec_size == sizeof(MeshKeyRec)) {
        MeshKeyRec r;
        size_t cap = MK.file_n;
        n = 0;
        while (fread(&r, sizeof r, 1, f) == 1) {
            n++;
            if (!seen_add(mk_hash(&r))) continue;
            if (MK.file_n == cap) {
                size_t c = cap ? cap * 2 : 256;
                MeshKeyRec *t = realloc(MK.file, c * sizeof *t);
                if (!t) break;
                MK.file = t, cap = c;
            }
            MK.file[MK.file_n++] = r;
        }
        if (own) MK.header_ok = 1;
    } else {
        rt_log("vk_pipeline: %s has another format or version: %s", path, own ? "it is rewritten" : "left out");
    }
    fclose(f);
    return n;
}

// The player's records and the release's (pipeline-keys-mesh-seed.bin). Caller holds P.lock (ensure_init).
static void mk_open(void) {
    if (!A.on || !K.path[0]) return;
    snprintf(MK.path, sizeof MK.path, "%s", K.path);
    char *dot = strrchr(MK.path, '.');
    if (!dot || (size_t)(dot - MK.path) + sizeof "-mesh-seed.bin" > sizeof MK.path) {
        MK.path[0] = 0;
        return;
    }
    strcpy(dot, "-mesh.bin");
    long own = mk_read(MK.path, 1);
    size_t mine = MK.file_n;
    char seed[sizeof MK.path];
    snprintf(seed, sizeof seed, "%.*s-mesh-seed.bin", (int)(dot - MK.path), MK.path);
    long n = mk_read(seed, 0);
    if (own >= 0 || n >= 0)
        rt_log("vk_pipeline: mesh and geometry pipeline keys: %ld in this machine's file, %ld in the release's (%zu of "
               "them new)", own < 0 ? 0 : own, n < 0 ? 0 : n, MK.file_n - mine);
}

// The mesh and geometry records read at boot (the player's and the release's), handed to the entry step: mk_take moves
// them out of MK (P.lock held), mk_ids lists their input ids, mk_queue enters and queues their pipelines, mk_drop frees
// them unused.
static struct { MeshKeyRec *r; size_t n; } MKT;

static size_t mk_take(void) {
    free(MKT.r);
    MKT.r = MK.file, MKT.n = MK.file_n;
    MK.file = NULL, MK.file_n = 0;
    return MKT.n;
}

static size_t mk_ids(uint8_t (*out)[32]) {
    size_t m = 0;
    for (size_t i = 0; i < MKT.n; i++)
        for (int j = 0; j < 3; j++)
            if (MKT.r[i].has & 1u << j) memcpy(out[m++], MKT.r[i].sh[j], 32);
    return m;
}

static void mk_drop(void) { free(MKT.r), MKT.r = NULL, MKT.n = 0; }

typedef struct { size_t i; uint64_t words; const GfxShader *s3[3]; } MkOrd;
static int mk_ord_cmp(const void *a, const void *b) {
    uint64_t x = ((const MkOrd *)a)->words, y = ((const MkOrd *)b)->words;
    return x < y ? 1 : x > y ? -1 : 0;
}

// Enters every taken record's pipeline as building and queues it (mesh_queue), the one with the most SPIR-V first; the
// entry step's builders take the queue (pw_builder). One progress step for each record not queued. The mesh threads are
// not started for this: a draw starts them (mesh_async_on).
static uint32_t mk_queue(const uint8_t (*ids)[32], size_t u, const GfxShader **sh, uint32_t *known, uint32_t *dropped,
                         uint32_t *skipped) {
    MeshKeyRec *r = MKT.r;
    size_t n = MKT.n, no = 0;
    MKT.r = NULL, MKT.n = 0;
    MkOrd *o = n ? calloc(n, sizeof *o) : NULL;
    uint32_t queued = 0, cut = 0;
    // DEBUG-MESHDROP: REKARINA_MESH_KEYS_DROP=<percent>: that share of the records (by record hash) is not built here,
    // nor its libraries, so play meets those pipelines as new ones (a test of what a draw does with a pairing no seed
    // holds; the records stay in the files)
    const char *de = cfg_get("REKARINA_MESH_KEYS_DROP");
    uint32_t drop = de && *de ? (uint32_t)atoi(de) : 0;
    for (size_t i = 0; o && i < n; i++) {
        if (drop && mk_hash(&r[i]) % 100 < drop) {
            cut++, ++*dropped;
            prep_progress_step(1);
            continue;
        }
        MkOrd e = {i, 0, {NULL, NULL, NULL}};
        int ok = r[i].mesh <= 1 && (r[i].mesh != 1 || vk_pipeline_mesh_on()), skip = 0;
        for (int j = 0; ok && j < 3; j++) {
            if (!(r[i].has & 1u << j)) continue;
            const uint8_t (*f)[32] = bsearch(r[i].sh[j], ids, u, 32, id_cmp);
            uint64_t w = 0;
            if (!f || !(e.s3[j] = sh[f - ids])) ok = 0;
            else if (skip_listed(r[i].sh[j])) skip = 1;
            else gfx_shader_spirv(e.s3[j], &w), e.words += w;
        }
        if (!ok || skip) {
            ++*(ok ? skipped : dropped);
            prep_progress_step(1);
            continue;
        }
        o[no++] = e;
    }
    if (no) qsort(o, no, sizeof *o, mk_ord_cmp);
    plat_mutex_lock(&P.lock);
    for (size_t q = 0; q < no; q++) {
        size_t i = o[q].i;
        StagesKey key;
        VkPipeState t;
        stages_key(&key, &t, &r[i].s, o[q].s3, (int)r[i].mesh);
        uint64_t h = key_hash(&key, sizeof key);
        if (pipe_find(&key, sizeof key, h)) {
            ++*known;
            prep_progress_step(1);
            continue;
        }
        const char *why = NULL;
        Bindings B = {0};
        uint32_t off[3] = {0, 0, 0};
        Layout *l = stages_layout(o[q].s3, &B, off, &why);
        Pipe *p = why ? NULL : pipe_add(&key, sizeof key, h, VK_NULL_HANDLE, l, &B, VK_PIPELINE_BIND_POINT_GRAPHICS, 0);
        if (p) mesh_queue(p, &t, o[q].s3, (int)r[i].mesh, &key.k, l, off), queued++;
        else ++*dropped, prep_progress_step(1);
        if (p && r[i].mesh == 1 && mesh_gpl_on()) mesh_libs_ahead(&t, o[q].s3, &key.k);
    }
    if (cut) rt_log("vk_pipeline: %u of %zu mesh and geometry records left out of the entry step (REKARINA_MESH_KEYS_DROP=%u)",
                    cut, n, drop);
    if (MG.libs_ahead)
        rt_log("vk_pipeline: entry step: %llu stage libraries of the recorded mesh pipelines queued (task and mesh, fragment)",
               (unsigned long long)MG.libs_ahead);
    plat_mutex_unlock(&P.lock);
    free(o), free(r);
    return queued;
}
#else
static void mk_open(void) {}
static void mk_flush(void) {}
static int mk_pending(void) { return 0; }
static size_t mk_take(void) { return 0; }
static size_t mk_ids(uint8_t (*out)[32]) { (void)out; return 0; }
static void mk_drop(void) {}
static uint32_t mk_queue(const uint8_t (*ids)[32], size_t u, const GfxShader **sh, uint32_t *known, uint32_t *dropped,
                         uint32_t *skipped) {
    (void)ids, (void)u, (void)sh, (void)known, (void)dropped, (void)skipped;
    return 0;
}
#endif

// ---- stage library prebuild
// Two places build what a draw would otherwise build while the game waits:
// - rekarina.exe --prepare-shaders (lb_warm_prepared), after the translation, on every core: for each prepared variant
//   (shader-prepared.bin) the pre-rasterization library of a vertex shader, the fragment-shader library of a fragment
//   shader and the pipeline of a compute shader, each built from a private shader copy and destroyed at once. What
//   stays is the driver's own disk cache: on AMD a library built again in a later process takes 0.1 ms instead of
//   55 ms (vertex) / 28 ms (fragment). Our pipeline-cache.bin is not written there:
//   it would grow by about 10 KB per library (~240 MB) and be read into memory at every boot.
// - boot (lb_worker), after the pipeline-key prewarm, on cpus/4 threads at PLAT_PRIO_IDLE (the short P.lock
//   sections and shader loads at normal priority, so a preempted thread never holds the lock the replay needs):
//   compute pipelines of every recorded and prepared compute shader, kept (vk_pipeline_compute builds compute
//   synchronously on the replay thread; 163 prepared), and the libraries of the first REKARINA_LIB_PREBUILD_KEEP variants
//   play recorded (priority 0), kept in libs[] with their shaders. KEEP defaults to 0: a kept library costs 140-460 KB
//   of process memory (10,000 kept took 9 GB in run 1) to save a rebuild the driver cache makes 0.1 ms.
// Library state: the vertex output union known now (vs_union_peek; the prewarm fills it from pipeline-keys.bin, the
// preparation has none), the clamp, polygon mode and depth mode most recorded pipelines use (defaults without a key
// file: clamp 0, fill, -w..w), single-sampled, no dual source; the state dynamic on this GPU is 0 in every key.
// A draw that meets such a variant in a new pipeline then gets its libraries from libs[] or the driver cache and
// fast-links (gpl_link). REKARINA_LIB_PREBUILD=0 turns the boot part off, =N stops it after N variants;
// REKARINA_LIB_PREBUILD_THREADS sets its thread count.
#define LB_MAX_THREADS 16

static struct {
    GfxWarmEnt *e;
    size_t n;
    volatile int64_t next;
    volatile int32_t stop, running;
    int threads;
    PlatThread *th[LB_MAX_THREADS];
    uint8_t clamp, poly, minus_w, samples;             // library state (GKey fields)
    double t0, ms[3];                                  // ms: build time per kind (vertex, fragment, compute)
    // warmed: driver cache only; nolib: vertex and fragment variants skipped with the graphics pipeline library off
    // skipped: variants of the skip list (SK below)
    uint64_t built[3], have[3], fail[3], warmed[3], noshader, nolib, skipped, done, cs_draw;
    double cs_draw_ms, cs_draw_max;   // those the replay thread built: the time they took it
    double warm_ms[3];
    volatile int32_t kept;                                // libraries entered in libs[] so far
    int32_t keep_n;
    int idle;                                          // build at PLAT_PRIO_IDLE (boot), normal (preparation)
    PlatMutex stat;
    // Boot prebuild only: the variant each worker has handed to the driver and since when (ms, 0: none), for lb_watch
    uint8_t cur_iid[LB_MAX_THREADS][32], cur_stage[LB_MAX_THREADS], cur_said[LB_MAX_THREADS];
    volatile uint64_t cur_t0[LB_MAX_THREADS];
    PlatThread *watch;
} LB = {.stat = PLAT_MUTEX_INIT};

// ---- preparation that outlives a driver crash. A bug in the driver's shader compiler can kill the process inside
// vkCreate*Pipelines: NVIDIA 615.78.08 on an RTX 5080 crashed on the same vertex library at the same point of every
// preparation, natively and under Wine, so preparation never finished (issue #30 and a report on PR #60).
// While lb_warm_prepared builds, slot k of <dir>/shader-prepare-journal.bin holds the variant its worker k has handed to
// the driver (input id and stage; zeros between variants). The writes go straight to the OS, so they outlast the
// process, and a run that finishes deletes the file. The next preparation or boot (skip_init) records the variants a
// leftover journal names in <dir>/shader-prepare-skip.txt with this GPU and driver version.
// A death is not always the driver's: a cancelled setup (the launcher's Cancel ends the tool with a hard kill), a
// killed process or a power cut leave the same journal, and then every variant in it is innocent; after a real crash
// all but one are. So one death makes a variant a suspect (a line without a fourth field): the next preparation builds
// its suspects first, one at a time on one thread, so a crash among them names the culprit alone, and a preparation
// that finishes drops their lines (skip_clear_suspects). A death of a variant that was a suspect already makes it a
// skip (fourth field "skip"): preparation and the boot prebuild (lb_worker) skip it on this GPU and driver, and it
// builds at its first draw, where the driver can still crash. Lines written before the suspects existed have no fourth
// field, so they count as one death. Another driver version builds them again, and so does deleting the file.
// tools/prepare_shaders.py runs the build again (--warm-shaders) after a run that left a journal.
#define JR_SLOT 64    // bytes per journal slot: input id, stage, zeros
#define JR_SLOTS 64   // lb_warm_prepared's thread cap

static struct {
    PlatFile *journal;     // while lb_warm_prepared builds
    uint8_t (*iid)[32];    // the skip list's variants for this GPU and driver ("skip": two deaths)
    size_t n;
    uint8_t (*sus)[32];    // its suspects for this GPU and driver (one death): built first and alone
    size_t nsus;
    int loaded;
} SK;

static int cache_dir(char *dir, size_t n) {   // the folder of pipeline-cache.bin and shader-prepared.bin
    snprintf(dir, n, "%s", P.cache_path);
    char *cut = plat_last_sep(dir);
    if (!cut) return 0;
    *cut = 0;
    return 1;
}

static void gpu_key(char *out, size_t n) {   // vendor:device:driver version, as the skip list names a GPU and driver
    const VkPhysicalDeviceProperties *p = &vk_ctx()->props;
    snprintf(out, n, "%04x:%04x:%08x", p->vendorID, p->deviceID, p->driverVersion);
}

static const char *stage_name(int st) {
    return st == GFX_STAGE_VS ? "vertex" : st == GFX_STAGE_FS ? "fragment" : st == GFX_STAGE_CS ? "compute" : "other";
}

static int hex_digit(int c) {
    return c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : c >= 'A' && c <= 'F' ? c - 'A' + 10 : -1;
}

// One line of shader-prepare-skip.txt: input id, stage, GPU and driver, and whether it is a skip (two deaths) or a
// suspect (one).
typedef struct {
    uint8_t iid[32];
    char hex[65], stage[16], key[32];
    int skip;
} SkipLine;

static const char SKIP_HEAD[] =
    "# Shader variants the graphics driver was building when the process died during preparation\n"
    "# (runtime/vk_pipeline.c skip_init): input id, stage, GPU vendor:device:driver version, and \"skip\" once the\n"
    "# variant was in flight at two deaths. Preparation builds a variant without \"skip\" again, first and alone, and\n"
    "# drops its line when it finishes; it skips the \"skip\" ones on that GPU and driver. Delete this file to build\n"
    "# them all again.\n";

static int skip_parse(char *line, SkipLine *e) {
    char extra[16] = "";
    if (line[0] == '#') return 0;
    int k = sscanf(line, "%64s %15s %31s %15s", e->hex, e->stage, e->key, extra);
    if (k < 3 || strlen(e->hex) != 64) return 0;
    for (int b = 0; b < 32; b++) {
        int hi = hex_digit(e->hex[2 * b]), lo = hex_digit(e->hex[2 * b + 1]);
        if (hi < 0 || lo < 0) return 0;
        e->iid[b] = (uint8_t)(hi << 4 | lo);
    }
    e->skip = k == 4 && !strcmp(extra, "skip");
    return 1;
}

static SkipLine *skip_read(const char *sp, size_t *count) {
    uint64_t sn = 0;
    char *s = (char *)plat_file_read(sp, &sn);
    SkipLine *L = NULL;
    size_t n = 0;
    for (char *line = s, *next; line && *line; line = next) {
        next = strchr(line, '\n');
        if (next) *next++ = 0;
        SkipLine e;
        if (!skip_parse(line, &e)) continue;
        SkipLine *g = realloc(L, (n + 1) * sizeof *L);
        if (!g) break;
        L = g, L[n++] = e;
    }
    free(s);
    *count = n;
    return L;
}

// The list as the file's text; with no line left the file goes.
static int skip_write(const char *sp, const SkipLine *L, size_t n) {
    if (!n) {
        plat_file_delete(sp);
        return 1;
    }
    size_t cap = sizeof SKIP_HEAD + n * 128, len = sizeof SKIP_HEAD - 1;
    char *out = malloc(cap);
    if (!out) return 0;
    memcpy(out, SKIP_HEAD, len);
    for (size_t i = 0; i < n; i++)
        len += (size_t)snprintf(out + len, cap - len, "%s %s %s%s\n", L[i].hex, L[i].stage, L[i].key, L[i].skip ? " skip" : "");
    int ok = plat_file_write_atomic(sp, out, len);
    free(out);
    return ok;
}

// A journal a dead run left into the skip list (a first death makes a suspect, a second a skip), then the list's
// skips and suspects for this GPU and driver into SK.
static void skip_init(const char *dir) {
    if (SK.loaded) return;
    SK.loaded = 1;
    char jp[PLAT_PATH_MAX * 2 + 48], sp[PLAT_PATH_MAX * 2 + 48], key[32];
    snprintf(sp, sizeof sp, "%s/shader-prepare-skip.txt", dir);
    gpu_key(key, sizeof key);
    size_t n = 0;
    SkipLine *L = skip_read(sp, &n);
    // the preparation's journal (lb_warm_prepared) and the entry step's (vk_pipeline_prewarm): entries of one layout
    static const char *const journals[2] = {"shader-prepare-journal.bin", "pipeline-prewarm-journal.bin"};
    for (int q = 0; q < 2; q++) {
        snprintf(jp, sizeof jp, "%s/%s", dir, journals[q]);
        uint64_t jn = 0;
        uint8_t *j = plat_file_read(jp, &jn);
        int keep_journal = 0, seen = 0, struck = 0;
        if (j) {
            size_t n0 = n;   // a variant twice in one journal is one death
            for (uint64_t o = 0; o + 33 <= jn; o += JR_SLOT) {
                int set = 0;
                for (int b = 0; b < 32; b++) set |= j[o + b];
                if (!set) continue;
                size_t f = 0;
                while (f < n && (memcmp(L[f].iid, j + o, 32) || strcmp(L[f].key, key))) f++;
                seen++;
                if (f < n0) {
                    if (!L[f].skip) L[f].skip = 1, struck++;
                } else if (f == n) {
                    SkipLine *g = realloc(L, (n + 1) * sizeof *L);
                    if (!g) continue;
                    L = g;
                    SkipLine *e = &L[n++];
                    memset(e, 0, sizeof *e);
                    memcpy(e->iid, j + o, 32);
                    for (int b = 0; b < 32; b++) snprintf(e->hex + 2 * b, 3, "%02x", j[o + b]);
                    snprintf(e->stage, sizeof e->stage, "%s", stage_name(j[o + 32]));
                    snprintf(e->key, sizeof e->key, "%s", key);
                }
            }
            if (seen && skip_write(sp, L, n)) {
                rt_log("vk_pipeline: the last %s died while the driver built %d variants; %d of them were in flight at "
                       "an earlier death too and are skipped from now on on this GPU and driver, the others are built "
                       "again, first and one at a time (%s, %s)",
                       q ? "start's entry step" : "shader preparation", seen, struck, sp, key);
            } else if (seen) {
                rt_log("vk_pipeline: cannot write %s; %s stays for the next run", sp, jp);
                keep_journal = 1;
            }
            free(j);
        }
        if (!keep_journal) plat_file_delete(jp);
    }
    for (size_t i = 0; i < n; i++) {
        if (strcmp(L[i].key, key)) continue;
        size_t *cnt = L[i].skip ? &SK.n : &SK.nsus;
        uint8_t (**arr)[32] = L[i].skip ? &SK.iid : &SK.sus;
        uint8_t (*grown)[32] = realloc(*arr, (*cnt + 1) * sizeof **arr);
        if (grown) memcpy(grown[(*cnt)++], L[i].iid, 32), *arr = grown;
    }
    free(L);
    if (SK.n || SK.nsus)
        rt_log("vk_pipeline: %zu shader variants skipped ahead of play and %zu built first and alone at the next "
               "preparation, on this GPU and driver (%s)", SK.n, SK.nsus, sp);
}

static int skip_listed(const uint8_t iid[32]) {
    for (size_t i = 0; i < SK.n; i++)
        if (!memcmp(SK.iid[i], iid, 32)) return 1;
    return 0;
}

static int skip_suspect(const uint8_t iid[32]) {
    for (size_t i = 0; i < SK.nsus; i++)
        if (!memcmp(SK.sus[i], iid, 32)) return 1;
    return 0;
}

// A preparation that finished: every suspect of this GPU and driver was handed to the driver without a death (or had
// no shader to build), so their lines go; skips and other GPUs' lines stay.
static void skip_clear_suspects(const char *dir) {
    if (!SK.nsus) return;
    char sp[PLAT_PATH_MAX * 2 + 48], key[32];
    snprintf(sp, sizeof sp, "%s/shader-prepare-skip.txt", dir);
    gpu_key(key, sizeof key);
    size_t n = 0, kept = 0;
    SkipLine *L = skip_read(sp, &n);
    for (size_t i = 0; i < n; i++)
        if (L[i].skip || strcmp(L[i].key, key)) L[kept++] = L[i];
    if (skip_write(sp, L, kept))
        rt_log("vk_pipeline: %zu suspect shader variants built without a crash: their lines left %s", SK.nsus, sp);
    free(L);
    free(SK.sus), SK.sus = NULL, SK.nsus = 0;
}

static void skip_count(void) {
    plat_mutex_lock(&LB.stat);
    LB.skipped++;
    plat_mutex_unlock(&LB.stat);
}

// Journal slot `slot`: the variant e its worker hands to the driver next, or zeros (e NULL) when the build returned.
static void journal_mark(int slot, const GfxWarmEnt *e) {
    if (!SK.journal || slot < 0 || slot >= JR_SLOTS) return;
    uint8_t rec[JR_SLOT] = {0};
    if (e) memcpy(rec, e->iid, 32), rec[32] = e->stage;
    plat_file_pwrite(SK.journal, rec, sizeof rec, (uint64_t)slot * JR_SLOT);
}

// ---- the compute shaders play dispatched (compute-used.bin beside pipeline-keys.bin)
// The boot prebuild keeps a pipeline of every compute shader the game holds: 5,037 in OoT, where a
// session dispatches a few hundred (224 after a tour of five maps). On NVIDIA they hold about 840 MiB of video memory
// (1,179 MiB outside the allocator at the title screen against 333) and 1.1 GiB of the process's memory: 0.5 GiB of
// SPIR-V and 0.6 GiB of the driver's compiler arenas (docs/oot-notes.md, 2026-11-14). This file lists the input ids
// of those a dispatch has asked for ("TOTKCUSE", a version, then 32 bytes each, appended at a shader's first dispatch
// on this machine). Two things read it:
// - the boot prebuild (lb_start) keeps only the listed ones once the list has CS_USED_MIN entries (the release's seed
//   has about 930, from tours through the whole game); the unlisted shaders' SPIR-V is not read at all. Until
//   2026-11-17 a card with a budget of 8 GiB or more and 20 GiB of memory kept all 4,191: 290 MiB more video memory
//   for the whole session (the footprint lines: 1,190 MiB outside VMA at the title against 897, RTX 5070) and the
//   SPIR-V and compiler arenas in the process's memory, for shaders no tour dispatched;
// - vk_pipeline_compute_trim, which vk_texture.c calls when video memory stays short on any card, destroys the
//   prebuilt ones that neither a dispatch since boot nor the list names. The driver's count hardly falls then (it
//   keeps the memory for what it allocates next); not building them is what gives the memory back.
// One that was left out or dropped is built at its first dispatch, on the replay thread, from the driver's cache
// (0.2 to 0.6 ms on average in the prebuild's own count, 0.16 ms for the one the tour met; the log names any over
// 8 ms). REKARINA_CS_KEEP=all keeps every one as before, =used (and the default) the listed ones.
#define CS_USED_MIN 64
static struct {
    uint8_t (*iid)[32];
    size_t n, cap;
    char path[PLAT_PATH_MAX * 2];
    int loaded;
} CU;

static int cu_has(const uint8_t iid[32]) {
    for (size_t i = 0; i < CU.n; i++)
        if (!memcmp(CU.iid[i], iid, 32)) return 1;
    return 0;
}

static int cu_add(const uint8_t iid[32]) {
    if (CU.n == CU.cap) {
        size_t c = CU.cap ? CU.cap * 2 : 512;
        void *t = realloc(CU.iid, c * 32);
        if (!t) return 0;
        CU.iid = t, CU.cap = c;
    }
    memcpy(CU.iid[CU.n++], iid, 32);
    return 1;
}

// P.lock held. Reads the file once.
static void cu_load(void) {
    if (CU.loaded) return;
    CU.loaded = 1;
    char dir[PLAT_PATH_MAX * 2];
    if (!cache_dir(dir, sizeof dir)) return;
    snprintf(CU.path, sizeof CU.path, "%s/compute-used.bin", dir);
    // this machine's list, then the release's (compute-used-seed.bin, tools/make_seed_keys.py, tools/package_oot.py):
    // one list, which the entry step builds before play (vk_pipeline_prewarm)
    char seed[PLAT_PATH_MAX * 2 + 32];
    snprintf(seed, sizeof seed, "%s/compute-used-seed.bin", dir);
    const char *const files[2] = {CU.path, seed};
    for (int k = 0; k < 2; k++) {
        FILE *f = fopen(files[k], "rb");
        if (!f) continue;
        char head[12];
        uint8_t iid[32];
        size_t before = CU.n;
        if (fread(head, 1, 12, f) == 12 && !memcmp(head, "TOTKCUSE", 8))
            while (fread(iid, 1, 32, f) == 32)
                if (!cu_has(iid)) cu_add(iid);
        fclose(f);
        if (k) rt_log("vk_pipeline: the release's compute shaders: %zu not in this machine's list (%s)", CU.n - before, seed);
    }
}

// P.lock held: a dispatch asked for p. Notes it in the pipeline and, the first time on this machine, in the file.
static void cs_use(Pipe *p, const GfxShader *cs) {
    if (p->cs_used) return;
    p->cs_used = 1;
    cu_load();
    const uint8_t *iid = gfx_shader_input_id(cs);
    if (!iid || !CU.path[0] || cu_has(iid) || !cu_add(iid)) return;
    int fresh = plat_file_size(CU.path) < 12;   // no file yet, or not one of ours
    FILE *f = fopen(CU.path, fresh ? "wb" : "ab");
    if (!f) return;
    if (fresh) {
        static const char head[12] = {'T', 'O', 'T', 'K', 'C', 'U', 'S', 'E', 1, 0, 0, 0};
        fwrite(head, 1, sizeof head, f);
        for (size_t i = 0; i + 1 < CU.n; i++) fwrite(CU.iid[i], 1, 32, f);   // what the list held before this one
    }
    fwrite(iid, 1, 32, f);
    fclose(f);
}

static int lb_prebuild_idle(void);

// Destroys the compute pipelines the boot prebuild made that no dispatch has asked for, in this session or (by the
// list) an earlier one; returns how many. For vk_texture.c when video memory stays short. Such a pipeline was never
// bound, so nothing recorded names it. The listed ones stay: the scenes ahead dispatch them, and building one of
// those on the replay thread is the hitch the prebuild is there to spare.
uint32_t vk_pipeline_compute_trim(void) {
    static const char *keep;
    if (!keep) keep = cfg_get("REKARINA_CS_KEEP") ? cfg_get("REKARINA_CS_KEEP") : "";
    if (!strcmp(keep, "all") || !ensure_init() || !lb_prebuild_idle()) return 0;
    uint32_t n = 0;
    plat_mutex_lock(&P.lock);
    cu_load();
    for (int i = 0; i < PIPE_BUCKETS; i++)
        for (Pipe **pp = &pipes[i]; *pp;) {
            Pipe *p = *pp;
            int stays = p->info.bind_point != VK_PIPELINE_BIND_POINT_COMPUTE || p->cs_used || p->building || !p->info.pipeline;
            if (!stays && p->key_size >= sizeof(const GfxShader *)) {   // the key begins with the shader (vk_pipeline_compute)
                const GfxShader *cs;
                memcpy(&cs, p->key, sizeof cs);
                const uint8_t *iid = gfx_shader_input_id(cs);
                stays = iid && cu_has(iid);
            }
            if (stays) {
                pp = &p->next;
                continue;
            }
            *pp = p->next;
            vkDestroyPipeline(P.device, p->info.pipeline, NULL);
            free(p->bindings);
            free(p);
            n++;
        }
    plat_mutex_unlock(&P.lock);
    if (n) rt_log("vk_pipeline: %u compute pipelines no dispatch has asked for dropped: video memory is short (a later dispatch builds its own)", n);
    return n;
}

// ---- a compute pipeline no prebuild made
// A dispatch needs its pipeline at once, and one that is not in the table was built right there, on the replay
// thread. From the driver's cache that is 0.2 to 1.3 ms. A shader variant this machine has never run is compiled in
// full: the first sword hit on a bush in the forest froze the game for 525 ms (2026-11-14; the log's next line was
// a new shader answer record). No prebuild can hold a variant nobody has met, so the build goes to a thread of its
// own: the dispatch waits for it up to CS_WAIT_MS, which a cached build always makes, and after that the dispatches
// of that shader go without it until the pipeline is there (what they write stays as it was for those frames, as a
// draw whose pipeline is still being built is left out). REKARINA_CS_ASYNC=0 builds on the replay thread as before;
// REKARINA_CS_BUILD_DELAY_MS=<ms> makes every such build that much longer, for a test.
#define CS_WAIT_MS 4
typedef struct { const GfxShader *cs; const Layout *l; Pipe *p; double t0; } CsJob;
static struct { uint64_t built, late, skipped; double late_ms; } CSA;   // P.lock
static _Thread_local int cs_pending;   // the last vk_pipeline_compute of this thread left its dispatch out

int vk_pipeline_compute_pending(void) { return cs_pending; }

static void cs_build_thread(void *arg) {
    CsJob *j = arg;
    static int delay = -1;
    if (delay < 0) { const char *e = cfg_get("REKARINA_CS_BUILD_DELAY_MS"); delay = e && atoi(e) > 0 ? atoi(e) : 0; }
    if (delay) plat_sleep_ns((uint64_t)delay * 1000000ull);
    VkPipeline pipe = compute_create(j->cs, j->l);
    double ms = now_ms() - j->t0;
    prep_note(PREP_CS, ms < CS_WAIT_MS ? ms : CS_WAIT_MS, ms);   // in play only (prep.c)
    plat_mutex_lock(&P.lock);
    j->p->info.pipeline = pipe;
    j->p->building = 0;
    CSA.built++;
    if (pipe) P.stats.compute++, LB.cs_draw++, LB.cs_draw_ms += ms;
    else P.stats.failures++;
    if (ms > LB.cs_draw_max) LB.cs_draw_max = ms;
    if (ms > CS_WAIT_MS) CSA.late++, CSA.late_ms += ms;
    uint64_t skipped = CSA.skipped;
    plat_mutex_unlock(&P.lock);
    if (ms > 8.0)
        rt_log("vk_pipeline: a compute pipeline was built beside the replay in %.1f ms%s; %llu dispatches so far went "
               "without their pipeline meanwhile", ms, pipe ? "" : " and failed", (unsigned long long)skipped);
    free(j);
}

VkPipeline vk_pipeline_compute(const GfxShader *cs, const VkPipeInfo **info) {
    if (info) *info = NULL;
    cs_pending = 0;
    if (!cs || !ensure_init()) return VK_NULL_HANDLE;
    struct { const GfxShader *cs; uint32_t compute; } k = {cs, 1};
    uint64_t h = key_hash(&k, sizeof k);
    static int async = -1;
    if (async < 0) async = rt_knob("CS_ASYNC", "1");
    plat_mutex_lock(&P.lock);
    Pipe *p = pipe_find(&k, sizeof k, h);
    if (!p && async) {   // built on a thread of its own (above)
        const char *why = NULL;
        Bindings B = {0};
        Layout *l = compute_layout(cs, &B, &why);
        CsJob *j = why ? NULL : malloc(sizeof *j);
        if (j) p = pipe_add(&k, sizeof k, h, VK_NULL_HANDLE, l, &B, VK_PIPELINE_BIND_POINT_COMPUTE, 0);
        if (j && p) {
            *j = (CsJob){cs, l, p, now_ms()};
            p->building = 1;
            if (!plat_thread_spawn(cs_build_thread, j, "compute pipeline", 1024 * 1024)) {   // no thread: here, as before
                plat_mutex_unlock(&P.lock);
                cs_build_thread(j);
                plat_mutex_lock(&P.lock);
            }
        } else {
            free(j);
            P.stats.failures++;
            rt_log("vk_pipeline: compute pipeline not built: %s", why ? why : "out of memory");
            if (!p) p = pipe_add(&k, sizeof k, h, VK_NULL_HANDLE, l, l ? &B : NULL, VK_PIPELINE_BIND_POINT_COMPUTE, 0);
        }
    }
    if (p && p->building) {   // being built beside: a cached build is done within the wait
        uint64_t until = plat_now_ns() + CS_WAIT_MS * 1000000ull;
        while (p->building && plat_now_ns() < until) {
            plat_mutex_unlock(&P.lock);
            plat_sleep_ns(100000);
            plat_mutex_lock(&P.lock);
        }
        if (p->building) {
            CSA.skipped++;
            cs_pending = 1;
            plat_mutex_unlock(&P.lock);
            return VK_NULL_HANDLE;
        }
    }
    if (p) {
        P.stats.hits++;
    } else {
        const char *why = NULL;
        Bindings B = {0};
        Layout *l = compute_layout(cs, &B, &why);
        VkPipeline pipe = VK_NULL_HANDLE;
        if (!why) {
            uint64_t p0 = perf_begin();
            double t0 = now_ms();
            pipe = compute_create(cs, l);
            perf_end(PERF_PIPE, p0);
            double ms = now_ms() - t0;
            prep_note(PREP_CS, ms, ms);   // in play only (prep.c)
            LB.cs_draw_ms += ms;
            if (ms > LB.cs_draw_max) LB.cs_draw_max = ms;
            if (ms > 8.0) rt_log("vk_pipeline: a compute pipeline built at its dispatch took the replay thread %.1f ms", ms);
            if (!pipe) why = "pipeline creation failed";
        }
        if (why) {
            P.stats.failures++;
            rt_log("vk_pipeline: compute pipeline not built: %s", why);
        } else {
            P.stats.compute++;
            LB.cs_draw++;
        }
        p = pipe_add(&k, sizeof k, h, pipe, l, l ? &B : NULL, VK_PIPELINE_BIND_POINT_COMPUTE, 0);
    }
    if (p) cs_use(p, cs);
    plat_mutex_unlock(&P.lock);
    if (!p) return VK_NULL_HANDLE;
    if (info) *info = &p->info;
    return p->info.pipeline;
}

// The priority of the boot prebuild's threads. Idle priority gets no time at all while another program keeps every
// core busy: in play beside a video render the 4,999 compute pipelines were not through after 76 s (12-27 s
// otherwise), a dispatch then built its pipeline on the replay thread and the game stood for 6 s. Below normal, the
// guest's threads still win and the list ends.
#ifdef GAME_OOT
#define LB_PRIO PLAT_PRIO_BELOW_NORMAL
#else
#define LB_PRIO PLAT_PRIO_IDLE
#endif

static void lb_lock(void) {
    plat_thread_priority(PLAT_PRIO_NORMAL);
    plat_mutex_lock(&P.lock);
}

static void lb_unlock(void) {
    plat_mutex_unlock(&P.lock);
    if (LB.idle) plat_thread_priority(LB_PRIO);
}

static int lb_prebuild_idle(void) { return __atomic_load_n(&LB.running, __ATOMIC_ACQUIRE) <= 0; }

static void lb_count(int kind, int what, double ms) {   // what: 0 built, 1 already there, 2 failed, 3 cache only
    plat_mutex_lock(&LB.stat);
    if (what == 3) LB.warmed[kind]++, LB.warm_ms[kind] += ms;
    else (what == 0 ? LB.built : what == 1 ? LB.have : LB.fail)[kind]++, LB.ms[kind] += ms;
    plat_mutex_unlock(&LB.stat);
}

// vs_union_missing for vs alone, without entering vs into the union table (it holds VS_OUTS_CAP shaders).
static uint32_t vs_union_peek(const GfxShader *vs, GfxShaderIo fin[32]) {
    GfxShaderIo vout[32];
    uint32_t need = 0, have = gfx_shader_outputs(vs, vout), missing = 0;
    memset(fin, 0, 32 * sizeof *fin);
    plat_mutex_lock_shared(&VO.lock);
    uint32_t j = (uint32_t)(((uintptr_t)vs >> 4) * 0x9e3779b1u) & (VS_OUTS_CAP - 1);
    while (VO.vs[j] && VO.vs[j] != vs) j = (j + 1) & (VS_OUTS_CAP - 1);
    if (VO.vs[j]) need = VO.mask[j], memcpy(fin, VO.io[j], 32 * sizeof *fin);
    plat_mutex_unlock_shared(&VO.lock);
    if (P.vs_outs_all)
        for (int i = 0; i < 32; i++)
            if (!(need >> i & 1)) fin[i] = (GfxShaderIo){GFX_TYPE_FLOAT, 4, 0}, need |= 1u << i;
    for (int i = 0; i < 32; i++)
        if (need >> i & 1 && !(have >> i & 1)) missing |= 1u << i;
    return missing;
}

static void lb_compute(const GfxShader *cs) {
    struct { const GfxShader *cs; uint32_t compute; } k = {cs, 1};
    uint64_t h = key_hash(&k, sizeof k);
    Bindings B = {0};
    const char *why = NULL;
    lb_lock();
    Layout *l = pipe_find(&k, sizeof k, h) ? NULL : compute_layout(cs, &B, &why);
    int have = !l && !why;
    lb_unlock();
    if (!l) {
        lb_count(2, have ? 1 : 2, 0);
        return;
    }
    double t0 = now_ms();
    VkPipeline pipe = compute_create(cs, l);
    double ms = now_ms() - t0;
    lb_lock();
    if (pipe && !pipe_find(&k, sizeof k, h)) {
        pipe_add(&k, sizeof k, h, pipe, l, &B, VK_PIPELINE_BIND_POINT_COMPUTE, 0);
        P.stats.compute++;
    } else if (pipe) {
        vkDestroyPipeline(P.device, pipe, NULL);
        have = 1;
    }
    lb_unlock();
    lb_count(2, pipe ? have : 2, ms);
}

// The pre-rasterization (vertex) or fragment-shader library of one shader with the LB state.
// keep 0: build into the driver's pipeline cache only (sh is a private copy; the library is destroyed).
static void lb_library(const GfxShader *sh, int vertex, int keep) {
    int kind = vertex ? LIB_PR : LIB_FS, stat = vertex ? 0 : 1;
    VkPipeState s;
    vk_pipe_state_defaults(&s);
    GKey k;
    memset(&k, 0, sizeof k);
    k.clamp = LB.clamp, k.poly_mode = LB.poly, k.minus_w = LB.minus_w, k.samples = LB.samples;
    PrKey pk;
    FsKey fk;
    memset(&pk, 0, sizeof pk), memset(&fk, 0, sizeof fk);
    pk.pool = fk.pool = gfx_shader_pool_mask(sh) != 0;
    const void *key;
    uint32_t size;
    if (vertex) {
        GfxShaderIo fin[32];
        s.vs = sh;
        pk.kind = LIB_PR, pk.vs = sh, pk.clamp = k.clamp, pk.poly_mode = k.poly_mode, pk.minus_w = k.minus_w;
        pk.missing = vs_union_peek(sh, fin);
        for (int i = 0; i < 32; i++)
            if (pk.missing >> i & 1) pk.fin[i] = fin[i];
        key = &pk, size = sizeof pk;
    } else {
        s.fs = sh;
        fk.kind = LIB_FS, fk.fs = sh, fk.samples = k.samples;
        key = &fk, size = sizeof fk;
    }
    uint64_t h = key_hash(key, size);
    Bindings B = {0}, F = {0};
    VkPipelineLayout lay = VK_NULL_HANDLE;
    lb_lock();
    int have = keep && lib_find(h, key, size) != NULL;
    if (!have && add_stage(vertex ? &B : &F, sh, 0)) {
        Layout *l = layout_pair(&B, &F);
        if (l) lay = half_layout(l, vertex ? 0 : 1);
    }
    lb_unlock();
    if (have || !lay) {
        lb_count(stat, have ? 1 : 2, 0);
        return;
    }
    double t0 = now_ms();
    VkPipeline pipe = lib_create(kind, &s, &k, lay, key);
    double ms = now_ms() - t0;
    if (!keep) {
        if (pipe) vkDestroyPipeline(P.device, pipe, NULL);
        lb_count(stat, pipe ? 3 : 2, ms);
        return;
    }
    lb_lock();
    have = lib_find(h, key, size) != NULL;
    VkPipeline got = lib_put(h, key, size, pipe);
    if (!have && pipe) {
        Lib *b = lib_find(h, key, size);
        if (b) b->pre = 1;
    }
    lb_unlock();
    lb_count(stat, !pipe ? 2 : have || got != pipe ? 1 : 0, ms);
}

#ifdef GAME_NS2
// The task and mesh library of a prepared mesh variant (lb_warm_prepared, rekarina.exe --warm-shaders at the setup),
// so a pairing no record holds finds its mesh stage in the driver cache: a mesh program met for the first time in play
// was a quarter of the new mesh pipelines of the later seed steps (9 of 32; the rest paired a recorded task and mesh
// stage with a new fragment shader, which the entry step's libraries cover). Its task shader: the one a mesh record
// pairs it with; else, when the mesh shader reads a task payload, the task most recorded mesh stages have (179 of the
// 251 recorded pairs share one task variant); else none. The libraries are built into the driver cache and destroyed.
static struct {
    int on, log;                       // REKARINA_MESH_LIB_PREPARE: 0 off, 1 on, log: on and each pairing logged
    size_t n;
    uint8_t (*pair)[64];               // records' mesh input id, then its task's (zero: none), sorted by mesh id
    uint8_t dom[32];                   // the most recorded task
    int have_dom;
    uint8_t minus_w;                   // the depth mode most mesh records have
    uint64_t built, failed, guessed, none;
    double ms;
} LM;

static int spirv_task_payload(const uint32_t *w, uint64_t n) {   // an OpVariable in TaskPayloadWorkgroupEXT
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) break;
        if (op == SPV_OP_VARIABLE && len >= 4 && w[i + 3] == 5402) return 1;
        if (op == SPV_OP_FUNCTION) break;
        i += len;
    }
    return 0;
}

// From the mesh records read at boot (MK.file): the pairs, the most recorded task, the usual depth mode.
static void lm_init(void) {
    const char *e = cfg_get("REKARINA_MESH_LIB_PREPARE");
    plat_mutex_lock(&P.lock);
    LM.on = P.gpl && mesh_gpl_on() && vk_pipeline_mesh_on() && !(e && !strcmp(e, "0"));
    LM.log = LM.on && e && !strcmp(e, "log");
    size_t n = LM.on ? MK.file_n : 0;
    LM.pair = n ? calloc(n, 64) : NULL;
    uint32_t mw[2] = {0, 0};
    struct { uint8_t id[32]; uint32_t n; } top[16];
    int nt = 0;
    for (size_t i = 0; LM.pair && i < n; i++) {
        const MeshKeyRec *r = &MK.file[i];
        if (r->mesh != 1 || !(r->has & 2)) continue;
        memcpy(LM.pair[LM.n], r->sh[1], 32);
        if (r->has & 1) {
            memcpy(LM.pair[LM.n] + 32, r->sh[0], 32);
            int j = 0;
            while (j < nt && memcmp(top[j].id, r->sh[0], 32)) j++;
            if (j == nt && nt < 16) memcpy(top[nt].id, r->sh[0], 32), top[nt++].n = 0;
            if (j < nt) top[j].n++;
        }
        mw[r->s.depth_minus_w != 0]++;
        LM.n++;
    }
    plat_mutex_unlock(&P.lock);
    int best = -1;
    for (int j = 0; j < nt; j++)
        if (best < 0 || top[j].n > top[best].n) best = j;
    if (best >= 0) memcpy(LM.dom, top[best].id, 32), LM.have_dom = 1;
    LM.minus_w = mw[1] > mw[0];
    if (LM.n) qsort(LM.pair, LM.n, 64, id_cmp);
}

static void lb_mesh_library(const GfxShader *ms) {
    if (!LM.on) return;
    static const uint8_t zero[32];
    const uint8_t (*f)[64] = LM.n ? bsearch(gfx_shader_input_id(ms), LM.pair, LM.n, 64, id_cmp) : NULL;
    const uint8_t *tid = NULL;
    uint64_t w = 0;
    const uint32_t *code = gfx_shader_spirv(ms, &w);
    if (f) tid = memcmp(*f + 32, zero, 32) ? *f + 32 : NULL;
    else if (spirv_task_payload(code, w)) {
        if (!LM.have_dom) return;   // no record names a task (--prepare-shaders reads none): --warm-shaders builds it
        tid = LM.dom;
    }
    if (LM.log) {   // DEBUG-MESHLIB: REKARINA_MESH_LIB_PREPARE=log names each pairing before the driver gets it
        char hm[17], ht[17] = "-";
        for (int j = 0; j < 8; j++) snprintf(hm + 2 * j, 3, "%02x", gfx_shader_input_id(ms)[j]);
        for (int j = 0; tid && j < 8; j++) snprintf(ht + 2 * j, 3, "%02x", tid[j]);
        rt_log("vk_pipeline: mesh library: mesh %s task %s (%s), %llu SPIR-V words", hm, ht,
               f ? "a record's" : tid ? "the most recorded task: the mesh shader reads a payload" : "none: no payload",
               (unsigned long long)w);
    }
    int owned = 0;
    const GfxShader *ts = tid ? gfx_shader_load_private(tid, &owned) : NULL;
    if (tid && !ts) return;   // its task is not in the shader cache: nothing to pair it with
    plat_mutex_lock(&LB.stat);
    if (!f) ++*(tid ? &LM.guessed : &LM.none);
    plat_mutex_unlock(&LB.stat);
    VkPipeState s;
    vk_pipe_state_defaults(&s);
    GKey k;
    memset(&k, 0, sizeof k);
    k.clamp = P.dyn_clamp ? 0 : LB.clamp, k.poly_mode = P.dyn_poly ? 0 : LB.poly, k.minus_w = LM.minus_w, k.samples = 1;
    const GfxShader *sh3[3] = {ts, ms, NULL};
    Bindings B;
    PrmKey pm;
    FsKey fk;
    FoKey ok;
    lb_lock();
    Layout *l = mesh_lib_keys(sh3, &k, &B, &pm, &fk, &ok);
    VkPipelineLayout lay = l ? half_layout(l, 0) : VK_NULL_HANDLE;
    lb_unlock();
    double t0 = now_ms();
    VkPipeline pipe = lay ? lib_create(LIB_PRM, &s, &k, lay, &pm) : VK_NULL_HANDLE;
    double ms_ = now_ms() - t0;
    if (pipe) vkDestroyPipeline(P.device, pipe, NULL);
    plat_mutex_lock(&LB.stat);
    ++*(pipe ? &LM.built : &LM.failed), LM.ms += ms_;
    plat_mutex_unlock(&LB.stat);
    if (owned) gfx_shader_release_private(ts);
}
#endif

// A vertex or fragment variant with the graphics pipeline library off: no stage library to build ahead.
static void lb_skip_lib(void) {
    plat_mutex_lock(&LB.stat);
    LB.nolib++;
    plat_mutex_unlock(&LB.stat);
}

static void lb_report(const char *how) {
    float mem_used, mem_total, mem_proc;
    plat_mem_info(&mem_used, &mem_total, &mem_proc);
    plat_mutex_lock_shared(&LB.stat);
    rt_log("vk_pipeline: stage library prebuild %s: %llu of %zu variants in %.0f ms; kept: vertex %llu built (avg %.1f ms), "
           "%llu there; fragment %llu built (avg %.1f ms), %llu there; compute %llu built (avg %.1f ms), %llu there; driver "
           "cache only: vertex %llu (avg %.1f ms), fragment %llu (avg %.1f ms), compute %llu (avg %.1f ms); %llu failed, %llu not in the shader cache, "
           "%llu vertex/fragment skipped (graphics pipeline library off), %llu skipped (the driver crashed on them before, "
           "shader-prepare-skip.txt); process working set %.0f MB", how, (unsigned long long)LB.done, LB.n,
           now_ms() - LB.t0, (unsigned long long)LB.built[0], LB.built[0] ? LB.ms[0] / LB.built[0] : 0.0,
           (unsigned long long)LB.have[0], (unsigned long long)LB.built[1], LB.built[1] ? LB.ms[1] / LB.built[1] : 0.0,
           (unsigned long long)LB.have[1], (unsigned long long)LB.built[2], LB.built[2] ? LB.ms[2] / LB.built[2] : 0.0,
           (unsigned long long)LB.have[2], (unsigned long long)LB.warmed[0], LB.warmed[0] ? LB.warm_ms[0] / LB.warmed[0] : 0.0,
           (unsigned long long)LB.warmed[1], LB.warmed[1] ? LB.warm_ms[1] / LB.warmed[1] : 0.0,
           (unsigned long long)LB.warmed[2], LB.warmed[2] ? LB.warm_ms[2] / LB.warmed[2] : 0.0,
           (unsigned long long)(LB.fail[0] + LB.fail[1] + LB.fail[2]),
           (unsigned long long)LB.noshader, (unsigned long long)LB.nolib, (unsigned long long)LB.skipped, (double)mem_proc);
    plat_mutex_unlock_shared(&LB.stat);
    vk_mem_footprint_log("after the library prebuild");
}

// A variant the driver has been building for LB_STUCK_MS goes into the skip list, so the next start leaves it alone.
// On the player's package one compute variant of the play-recorded list never came back from the driver: its worker
// ran at a full core for every session, the list never ended, and the game shared its cores with it (2026-11-11).
// This session's build cannot be taken back (the thread is inside the driver); the list is what ends it for good.
#define LB_STUCK_MS 30000
static void lb_watch(void *arg) {
    (void)arg;
    plat_thread_priority(PLAT_PRIO_BELOW_NORMAL);
    char dir[PLAT_PATH_MAX * 2], sp[PLAT_PATH_MAX * 2 + 48], key[32];
    if (!cache_dir(dir, sizeof dir)) return;
    snprintf(sp, sizeof sp, "%s/shader-prepare-skip.txt", dir);
    gpu_key(key, sizeof key);
    while (!LB.stop && LB.running > 0) {
        for (int k = 0; k < 20 && !LB.stop && LB.running > 0; k++) plat_sleep_ms(100);
        uint64_t now = (uint64_t)now_ms();
        for (int w = 0; w < LB_MAX_THREADS; w++) {
            uint64_t t0 = LB.cur_t0[w];
            if (!t0 || now < t0 || now - t0 < LB_STUCK_MS || LB.cur_said[w]) continue;   // t0 may be newer than now
            LB.cur_said[w] = 1;
            char hex[65];
            for (int b = 0; b < 32; b++) snprintf(hex + 2 * b, 3, "%02x", LB.cur_iid[w][b]);
            uint64_t sn = 0;
            char *s = (char *)plat_file_read(sp, &sn);
            static const char head[] =
                "# Shader variants the graphics driver was building when the process died during preparation\n"
                "# (runtime/vk_pipeline.c skip_init): input id, stage, GPU vendor:device:driver version. Preparation skips\n"
                "# them on that GPU and driver. Delete this file to build them again.\n";
            size_t cap = (s ? sn : sizeof head) + 160, len = s ? sn : sizeof head - 1;
            char *out = malloc(cap);
            int ok = 0;
            if (out) {
                memcpy(out, s ? s : head, len);
                if (len && out[len - 1] != '\n') out[len++] = '\n';
                len += (size_t)snprintf(out + len, cap - len, "%s %s %s\n", hex, stage_name(LB.cur_stage[w]), key);
                ok = plat_file_write_atomic(sp, out, len);
            }
            rt_log("vk_pipeline: the driver has been building variant %s (%s) for %d s and has not come back: %s; the "
                   "prebuild of later starts skips it", hex, stage_name(LB.cur_stage[w]), LB_STUCK_MS / 1000,
                   ok ? "added to shader-prepare-skip.txt" : "shader-prepare-skip.txt could not be written");
            free(out);
            free(s);
        }
    }
}

// Boot: compute pipelines and kept libraries, all entered for the process.
static void lb_worker(void *arg) {
    int me = (int)(intptr_t)arg;
    plat_thread_priority(LB_PRIO);
    for (;;) {
        size_t i = (size_t)plat_inc64(&LB.next) - 1;
        if (i >= LB.n || LB.stop) break;
        if (skip_listed(LB.e[i].iid)) {
            skip_count();
            plat_mutex_lock(&LB.stat);
            LB.done++;
            plat_mutex_unlock(&LB.stat);
            continue;
        }
        plat_thread_priority(PLAT_PRIO_NORMAL);   // gfx_shader's cache lock, disk reads
        const GfxShader *sh = gfx_shader_load_input(LB.e[i].iid, 0);
        plat_thread_priority(LB_PRIO);
        if (sh) {   // lb_watch: what the driver gets now
            memcpy(LB.cur_iid[me], LB.e[i].iid, 32);
            LB.cur_stage[me] = (uint8_t)gfx_shader_stage(sh), LB.cur_said[me] = 0;
            __atomic_store_n(&LB.cur_t0[me], (uint64_t)now_ms() | 1, __ATOMIC_RELEASE);
        }
        if (!sh) {
            plat_mutex_lock(&LB.stat);
            LB.noshader++;
            plat_mutex_unlock(&LB.stat);
        } else if (gfx_shader_stage(sh) == GFX_STAGE_CS) {
            lb_compute(sh);
        } else if (gfx_shader_stage(sh) == GFX_STAGE_VS || gfx_shader_stage(sh) == GFX_STAGE_FS) {
            if (P.gpl) lb_library(sh, gfx_shader_stage(sh) == GFX_STAGE_VS, 1);
            else lb_skip_lib();
        }
        __atomic_store_n(&LB.cur_t0[me], 0, __ATOMIC_RELEASE);
        plat_mutex_lock(&LB.stat);
        LB.done++;
        plat_mutex_unlock(&LB.stat);
    }
    if (!plat_dec32(&LB.running)) lb_report(LB.stop ? "stopped" : "done");
    return;
}

// The library state from the key file's records (prewarm); defaults without one.
static void lb_state(const uint32_t pr_hist[16], const uint32_t fs_hist[4]) {
    // state index: clamp | poly (0 fill, 1 line, 2 point -> VkPolygonMode) << 1 | minus_w << 3 (pr); samples log2 (fs)
    uint32_t best = 0, bf = 0;
    for (uint32_t i = 1; i < 16; i++) best = pr_hist[i] > pr_hist[best] ? i : best;
    for (uint32_t i = 1; i < 4; i++) bf = fs_hist[i] > fs_hist[bf] ? i : bf;
    if (!pr_hist[best]) best = 8;   // no key file: TotK clips -w..w (all 2706 records of a recorded key file)
    LB.clamp = (uint8_t)(best & 1), LB.poly = (uint8_t)(best >> 1 & 3), LB.minus_w = (uint8_t)(best >> 3 & 1);
    LB.samples = (uint8_t)(1u << bf);
}

// Boot: compute shaders of the warm list (recorded and prepared), then the first KEEP recorded graphics variants.
static void lb_start(const uint32_t pr_hist[16], const uint32_t fs_hist[4]) {
    const char *e = cfg_value("REKARINA_LIB_PREBUILD"), *ke = cfg_get("REKARINA_LIB_PREBUILD_KEEP");
    const char *te = cfg_get("REKARINA_LIB_PREBUILD_THREADS");   // no fixed default: a quarter of the CPUs when unset
    if (!A.on || !cfg_bool("REKARINA_LIB_PREBUILD")) return;
    char dir[PLAT_PATH_MAX * 2];
    if (cache_dir(dir, sizeof dir)) skip_init(dir);
    GfxWarmEnt *all = NULL;
    size_t total = gfx_shader_warm_list(&all), n = 0, n_cs = 0;
    LB.keep_n = ke && *ke ? atoi(ke) : 0;
    LB.e = malloc((total + 1) * sizeof *LB.e);
    // the compute shaders kept: the ones a dispatch has used (compute-used.bin and the release's seed), all of them
    // with REKARINA_CS_KEEP=all or while the list is too short to stand for play
    const char *ck = cfg_get("REKARINA_CS_KEEP");
    int used_only = !(ck && !strcmp(ck, "all"));
    plat_mutex_lock(&P.lock);
    cu_load();
    if (CU.n < CS_USED_MIN) used_only = 0;   // no list yet (the first session): every one, and the trim under pressure
    size_t n_all_cs = 0;
    for (size_t i = 0; LB.e && i < total; i++)
        if (all[i].stage == GFX_STAGE_CS) {
            n_all_cs++;
            // DEBUG-CSNONE: REKARINA_CS_KEEP=none prebuilds no compute pipeline: every one is built at its first dispatch,
            // the path of a shader variant never met ("a compute pipeline no prebuild made")
            if (ck && !strcmp(ck, "none")) continue;
            if (!used_only || cu_has(all[i].iid)) LB.e[n++] = all[i];
        }
    plat_mutex_unlock(&P.lock);
    n_cs = n;
    if (used_only)
        rt_log("vk_pipeline: compute pipelines kept at boot: the %zu a dispatch has used of %zu (REKARINA_CS_KEEP=all keeps "
               "every one)", n_cs, n_all_cs);
    for (size_t i = 0; LB.e && i < total && (int32_t)(n - n_cs) < LB.keep_n; i++)
        if (all[i].stage != GFX_STAGE_CS && all[i].prio == 0) LB.e[n++] = all[i];
    free(all);
    if (atoi(e) > 1 && (size_t)atoi(e) < n) n = (size_t)atoi(e);   // a count above 1 caps the list; 1 is plain on
    if (!n) {
        free(LB.e), LB.e = NULL;
        return;
    }
    LB.n = n;
    lb_state(pr_hist, fs_hist);
    LB.idle = 1;
    int cpus = (int)plat_cpu_usable();
    int t = te && *te ? atoi(te) : cpus / 4;
    t = t < 1 ? 1 : t > LB_MAX_THREADS ? LB_MAX_THREADS : t;
    if ((size_t)t > n) t = (int)n;
    LB.t0 = now_ms();
    LB.running = t;
    int started = 0;
    for (int i = 0; i < t; i++)
        if ((LB.th[started] = plat_thread_new(lb_worker, (void *)(intptr_t)started, "lib prebuild"))) started++;
    plat_add32(&LB.running, started - t);
    LB.threads = started;
    if (started) LB.watch = plat_thread_new(lb_watch, NULL, "lib prebuild watch");
    rt_log("vk_pipeline: stage library prebuild: %zu compute pipelines and %zu recorded variants' libraries "
           "(REKARINA_LIB_PREBUILD_KEEP %d) on %d idle-priority threads; library state clamp %u, polygon mode %u, depth "
           "-w..w %u, samples %u", n_cs < n ? n_cs : n, n > n_cs ? n - n_cs : 0, LB.keep_n, started, LB.clamp, LB.poly,
           LB.minus_w, LB.samples);
}

static void lb_stop(void) {
    plat_xchg32(&LB.stop, 1);
    for (int i = 0; i < LB.threads; i++)
        if (LB.th[i]) plat_thread_join(LB.th[i]), LB.th[i] = NULL;
    LB.threads = 0;
    if (LB.watch) plat_thread_join(LB.watch), LB.watch = NULL;
}

// ---- which prepared variants this driver has: <dir>/shader-warm-done.bin
// "RKWARM01", u32 1, u32 count, a 32-byte key, then one bit per variant of shader-prepared.bin (in its order) whose
// stage library or compute pipeline was handed to the driver (built into its disk cache, failed, or skipped for a
// driver crash). The key is the device, the driver's version and cache identity, the library state they were built with
// (records_hist: the one most recorded pipelines use), the exe's path (NVIDIA keys its disk cache to the exe's file name,
// AMD to its path) and shader-prepared.bin's bytes: another driver, state, folder or list starts from nothing. lb_warm_prepared writes it every 2 s and at its end, so a run that is stopped or
// closed keeps what it built: the launcher's background build (--warm-shaders) and the game's entry step
// (vk_pipeline_prewarm) each build only what the other has not.
#define WD_HEAD 48
static struct {
    uint8_t key[32];
    uint8_t *bits;
    size_t n;
    char path[PLAT_PATH_MAX * 2 + 48];
    PlatMutex lock;
    double flushed;
    int dirty;
} WD = {.lock = PLAT_MUTEX_INIT};

static void wd_open(const char *dir, const uint8_t *list, uint64_t list_size, size_t n) {
    VkCtx *c = vk_ctx();
    struct {
        uint32_t vendor, device, driver;
        uint8_t uuid[VK_UUID_SIZE];
        uint8_t clamp, poly, minus_w, samples;   // the library state they are built with (lb_state)
    } id = {c->props.vendorID, c->props.deviceID, c->props.driverVersion, {0}, LB.clamp, LB.poly, LB.minus_w, LB.samples};
    memcpy(id.uuid, c->props.pipelineCacheUUID, VK_UUID_SIZE);
    char exe[PLAT_PATH_MAX * 2] = "";
    plat_exe_path(exe, sizeof exe);
    size_t el = strlen(exe), kn = sizeof id + el + 32;
    uint8_t *buf = malloc(kn);
    if (!buf) return;
    memcpy(buf, &id, sizeof id), memcpy(buf + sizeof id, exe, el);
    rt_sha256(list, (size_t)list_size, buf + sizeof id + el);
    rt_sha256(buf, kn, WD.key);
    free(buf);
    snprintf(WD.path, sizeof WD.path, "%s/shader-warm-done.bin", dir);
    WD.n = n, WD.dirty = 0, WD.flushed = now_ms();
    WD.bits = calloc((n + 7) / 8 + 1, 1);
    uint64_t size = 0;
    uint8_t *f = WD.bits ? plat_file_read(WD.path, &size) : NULL;
    uint32_t ver = 0, cnt = 0;
    if (f && size == WD_HEAD + (n + 7) / 8) memcpy(&ver, f + 8, 4), memcpy(&cnt, f + 12, 4);
    if (f && size == WD_HEAD + (n + 7) / 8 && !memcmp(f, "RKWARM01", 8) && ver == 1 && cnt == n && !memcmp(f + 16, WD.key, 32))
        memcpy(WD.bits, f + WD_HEAD, (n + 7) / 8);
    free(f);
}

static int wd_has(size_t i) { return WD.bits && i < WD.n && (WD.bits[i >> 3] >> (i & 7) & 1); }

static void wd_flush(int force) {
    plat_mutex_lock(&WD.lock);
    if (WD.bits && WD.dirty && (force || now_ms() - WD.flushed >= 2000)) {
        size_t nb = (WD.n + 7) / 8;
        uint8_t *out = malloc(WD_HEAD + nb);
        if (out) {
            uint32_t ver = 1, cnt = (uint32_t)WD.n;
            memcpy(out, "RKWARM01", 8), memcpy(out + 8, &ver, 4), memcpy(out + 12, &cnt, 4), memcpy(out + 16, WD.key, 32);
            memcpy(out + WD_HEAD, WD.bits, nb);
            if (!plat_file_write_atomic(WD.path, out, WD_HEAD + nb)) rt_log("vk_pipeline: cannot write %s", WD.path);
            free(out);
        }
        WD.dirty = 0, WD.flushed = now_ms();
    }
    plat_mutex_unlock(&WD.lock);
}

static void wd_set(size_t i) {
    if (!WD.bits || i >= WD.n) return;
    plat_mutex_lock(&WD.lock);
    WD.bits[i >> 3] |= (uint8_t)(1u << (i & 7)), WD.dirty = 1;
    plat_mutex_unlock(&WD.lock);
    wd_flush(0);
}

static void wd_close(void) {
    wd_flush(1);
    plat_mutex_lock(&WD.lock);
    free(WD.bits), WD.bits = NULL, WD.n = 0;
    plat_mutex_unlock(&WD.lock);
}

// Preparation: one prepared variant at a time into the driver's disk cache (private shader copy, nothing kept). arg is
// the worker's journal slot.
static int warm_no_app_cache;   // lb_warm_prepared: the builds below go past P.cache (build_cache)
static uint32_t *lb_ord;        // lb_warm_prepared: each entry's place in shader-prepared.bin (WD's bits)
static int lb_entry;            // lb_warm_prepared runs in the entry step: its count is the step's (PREP_PART_LIBS)
static void lb_warm_worker(void *arg) {
    int slot = (int)(intptr_t)arg;
    build_no_app_cache = warm_no_app_cache;
    for (;;) {
        if (__atomic_load_n(&LB.stop, __ATOMIC_ACQUIRE)) break;   // warm_stop_watch: the launcher wants the PC back
        size_t i = (size_t)plat_inc64(&LB.next) - 1;
        if (i >= LB.n) break;
        int owned = 0, skip = skip_listed(LB.e[i].iid);
        const GfxShader *sh = skip ? NULL : gfx_shader_load_private(LB.e[i].iid, &owned);
        int st = sh ? gfx_shader_stage(sh) : -1;
        if (sh) journal_mark(slot, &LB.e[i]);   // from here to its clearing below, a crash names this variant
        if (skip) {
            skip_count();
        } else if (!sh) {
            plat_mutex_lock(&LB.stat);
            LB.noshader++;
            plat_mutex_unlock(&LB.stat);
        } else if (st == GFX_STAGE_CS) {
            Bindings B = {0};
            const char *why = NULL;
            lb_lock();
            Layout *l = compute_layout(sh, &B, &why);
            lb_unlock();
            double t0 = now_ms();
            VkPipeline pipe = l ? compute_create(sh, l) : VK_NULL_HANDLE;
            if (pipe) vkDestroyPipeline(P.device, pipe, NULL);
            lb_count(2, pipe ? 3 : 2, now_ms() - t0);
        } else if (st == GFX_STAGE_VS || st == GFX_STAGE_FS) {
            if (P.gpl) lb_library(sh, st == GFX_STAGE_VS, 0);
            else lb_skip_lib();
        }
#ifdef GAME_NS2
        else if (st == GFX_STAGE_MS) lb_mesh_library(sh);
#endif
        if (sh) journal_mark(slot, NULL);
        if (owned) gfx_shader_release_private(sh);
        wd_set(lb_ord ? lb_ord[i] : i);
        if (lb_entry) prep_part_step(PREP_PART_LIBS, 1);
        plat_mutex_lock(&LB.stat);
        uint64_t d = ++LB.done;
        plat_mutex_unlock(&LB.stat);
        if (d % 4000 == 0) lb_report("preparation progress");
        static volatile int64_t said;   // the count every 2 s on stdout, for the launcher's row (installer/oot.py LIBRARIES)
        int64_t now = (int64_t)now_ms(), was = __atomic_load_n(&said, __ATOMIC_RELAXED);
        if (now - was >= 2000 && __atomic_compare_exchange_n(&said, &was, now, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
            printf("vk_pipeline: stage library prebuild preparation progress: %llu of %zu variants\n", (unsigned long long)d,
                   LB.n);
            fflush(stdout);
        }
    }
    build_no_app_cache = 0;   // the calling thread may build for play afterwards (the suspects run on the caller)
}

// The library state of the recorded graphics pipelines (K.file), as lb_state counts it: the clamp, polygon mode and
// depth mode of the pre-rasterization side, the samples of the fragment side. Read from the records' state alone, so
// that the launcher's background build and the entry step take the same state (and their done lists one key).
static void records_hist(uint32_t pr_hist[16], uint32_t fs_hist[4]) {
    memset(pr_hist, 0, 16 * sizeof *pr_hist), memset(fs_hist, 0, 4 * sizeof *fs_hist);
    plat_mutex_lock(&P.lock);
    const KeyRec *r = (const KeyRec *)K.file;
    for (size_t i = 0; r && i < K.file_n; i++) {
        const VkPipeState *st = &r[i].s;
        unsigned clamp = !P.dyn_clamp && st->depth_clamp && P.f_depth_clamp;
        unsigned poly = !P.dyn_poly ? (unsigned)polygon_mode(st->poly.mode) : 0;
        unsigned mw = st->depth_minus_w != 0, smp = st->samples > 1 ? st->samples : 1;
        pr_hist[(clamp & 1) | (poly & 3) << 1 | (mw & 1) << 3]++;
        fs_hist[smp >= 8 ? 3 : smp >= 4 ? 2 : smp >= 2 ? 1 : 0]++;
    }
    plat_mutex_unlock(&P.lock);
}

// The variants of <dir>/shader-prepared.bin this driver does not have yet (WD), into LB.e with their places in the file
// (lb_ord); their count. Taken once: the entry step counts them for its picture, lb_warm_prepared builds them.
static size_t lb_list_load(const char *dir, const uint32_t *pr_hist, const uint32_t *fs_hist) {
    if (LB.e) return LB.n;
    static const uint32_t no_pr[16], no_fs[4];
    lb_state(pr_hist ? pr_hist : no_pr, fs_hist ? fs_hist : no_fs);   // (the done list's key holds it)
    char path[PLAT_PATH_MAX * 2 + 32];
    snprintf(path, sizeof path, "%s/shader-prepared.bin", dir);
    uint64_t size = 0;
    uint8_t *f = plat_file_read(path, &size);
    size_t n = f && size >= 16 ? (size_t)(size - 16) / 36 : 0, m = 0;
    if (n) wd_open(dir, f, size, n);
    LB.e = n ? calloc(n, sizeof *LB.e) : NULL;
    lb_ord = n ? calloc(n, sizeof *lb_ord) : NULL;
    for (size_t i = 0; LB.e && lb_ord && i < n; i++)
        if (!wd_has(i)) {
            memcpy(LB.e[m].iid, f + 16 + 36 * i + 4, 32), LB.e[m].stage = f[16 + 36 * i], LB.e[m].prio = f[16 + 36 * i + 1];
            lb_ord[m++] = (uint32_t)i;
        }
    free(f);
    if (n) rt_log("vk_pipeline: %zu of %zu prepared variants have not been handed to this driver yet (shader-warm-done.bin)",
                  m, n);
    if (!m) free(LB.e), LB.e = NULL, free(lb_ord), lb_ord = NULL;
    LB.n = m;
    return m;
}

// Preparation: every variant of <dir>/shader-prepared.bin this driver does not have yet, on every core at normal
// priority. pr_hist, fs_hist: the library state of the recorded pipelines (the entry step's), NULL for the defaults.
static void lb_warm_prepared(const char *dir, const uint32_t *pr_hist, const uint32_t *fs_hist) {
    size_t n = lb_list_load(dir, pr_hist, fs_hist);
    if (!n) {
        wd_close();
        printf("vk_pipeline: stage library prebuild preparation done: 0 of 0 variants\n");   // (installer/oot.py)
        fflush(stdout);
        return;
    }
    skip_init(dir);
    char jp[PLAT_PATH_MAX * 2 + 48];
    snprintf(jp, sizeof jp, "%s/shader-prepare-journal.bin", dir);
    SK.journal = plat_fs_create(jp, NULL);   // skip_init removed a leftover one
    if (!SK.journal) rt_log("vk_pipeline: cannot create %s: a driver crash in this run will not be recorded", jp);
    LB.n = n, LB.idle = 0, LB.t0 = now_ms();
    // (the entry step's go past pipeline-cache.bin on every card: a file of the libraries filled the AMD driver's
    // limit of about 850 MiB, and the entry step's own pipelines no longer fit in it)
    warm_no_app_cache = lb_entry || (vk_ctx()->props.vendorID == 0x10de && !rt_knob("WARM_APP_CACHE", "0"));
    rt_log("vk_pipeline: the prepared variants go %s", warm_no_app_cache ? "into the driver's disk cache only (build_cache)"
                                                                            : "into the driver's disk cache and pipeline-cache.bin");
    static const uint32_t no_pr[16], no_fs[4];
    lb_state(pr_hist ? pr_hist : no_pr, fs_hist ? fs_hist : no_fs);
#ifdef GAME_NS2
    lm_init();
#endif
    // the suspects of an earlier death first, one at a time on this thread: a crash among them leaves the culprit alone
    // in the journal, and the next run skips it (skip_init); the other workers start after them
    size_t m = 0;
    for (size_t i = 0; SK.nsus && i < n; i++)
        if (skip_suspect(LB.e[i].iid)) {
            GfxWarmEnt t = LB.e[m];
            uint32_t o = lb_ord[m];
            lb_ord[m] = lb_ord[i], lb_ord[i] = o;
            LB.e[m++] = LB.e[i], LB.e[i] = t;
        }
    if (m) {
        double t1 = now_ms();
        LB.n = m;
        lb_warm_worker((void *)(intptr_t)0);
        LB.n = n, LB.next = m;   // the worker took one index past the suspects before it returned
        rt_log("vk_pipeline: %zu suspect shader variants built first and alone in %.0f ms", m, now_ms() - t1);
    }
    int cpus = (int)plat_cpu_usable(), k = 0;
    PlatThread *t[JR_SLOTS];
    for (int i = 0; i < cpus && i < JR_SLOTS; i++)
        if ((t[k] = plat_thread_new(lb_warm_worker, (void *)(intptr_t)k, "lib warm"))) k++;
    if (!k) lb_warm_worker((void *)(intptr_t)0);
    for (int i = 0; i < k; i++) plat_thread_join(t[i]);
    if (SK.journal) plat_file_close(SK.journal), SK.journal = NULL, plat_file_delete(jp);   // every build returned
    int stopped = __atomic_load_n(&LB.stop, __ATOMIC_ACQUIRE) != 0;
    if (!stopped) skip_clear_suspects(dir);   // (a stopped run did not build them all)
    lb_report(stopped ? "preparation stopped" : "preparation done");
    // (the last count on stdout too: the launcher reads it there, installer/oot.py LIBRARIES)
    printf("vk_pipeline: stage library prebuild preparation %s: %llu of %zu variants\n", stopped ? "stopped" : "done",
           (unsigned long long)LB.done, n);
    fflush(stdout);
#ifdef GAME_NS2
    if (LM.on)
        rt_log("vk_pipeline: task and mesh libraries of the prepared mesh variants: %llu built into the driver cache (avg "
               "%.1f ms), %llu failed; of those no mesh record names, %llu with the most recorded task, %llu without a task",
               (unsigned long long)LM.built, LM.built ? LM.ms / (double)LM.built : 0.0, (unsigned long long)LM.failed,
               (unsigned long long)LM.guessed, (unsigned long long)LM.none);
    free(LM.pair), memset(&LM, 0, sizeof LM);
#endif
    // with gpl off only compute variants are built: say so, rather than count the skipped ones as built
    unsigned long long skipped = LB.skipped;
    size_t tried = n - (size_t)skipped;
    if (stopped)
        printf("prepare-shaders: stopped after %llu of %zu variants on %d threads in %.0f ms\n",
               (unsigned long long)LB.done, n, k, now_ms() - LB.t0);
    else if (LB.nolib)
        printf("prepare-shaders: %zu variants on %d threads in %.0f ms, %llu vertex/fragment variants skipped (graphics "
               "pipeline library off, they build at their first draw)\n", tried, k, now_ms() - LB.t0,
               (unsigned long long)LB.nolib);
    else
        printf("prepare-shaders: %zu variants built into the driver cache on %d threads in %.0f ms\n", tried, k,
               now_ms() - LB.t0);
    if (skipped)   // installer/steps.py Prepare reads this line
        printf("prepare-shaders: %llu variants skipped: the graphics driver crashed while building them in an earlier "
               "run (shader-prepare-skip.txt); they build at their first draw\n", skipped);
    free(LB.e), LB.e = NULL, LB.n = 0, LB.next = 0, LB.done = 0, LB.skipped = 0;
    free(lb_ord), lb_ord = NULL;
    wd_close();
}

// rekarina.exe --prepare-shaders (tools/prepare_shaders.py): the device sets the translator's host answers (they are in every
// cache key), the seed next to rekarina.exe is translated on every core, then every prepared variant's stage library or
// compute pipeline is built once into the driver's disk cache (lb_warm_prepared). No game runs.
// REKARINA_PREPARE_DRIVER=0 stops after the translation (a test of a folder whose driver cache is cold: the game's entry
// step then builds everything with the driver's own compiler; --warm-shaders does the driver step later).
int vk_pipeline_prepare_shaders(void) {
    if (!ensure_init()) {
        fprintf(stderr, "prepare-shaders: no Vulkan device\n");
        return 1;
    }
    char dir[PLAT_PATH_MAX * 2];
    if (!cache_dir(dir, sizeof dir)) return 1;
    int cpus = (int)plat_cpu_usable();
    const char *pe = cfg_get("REKARINA_PREWARM_XLATE");   // no fixed default: half the CPUs, at most 16, when unset
    int procs = pe && *pe ? atoi(pe) : cpus / 2 > 16 ? 16 : cpus / 2;
    // The setup packs what this writes and checks every file as it does (tools/pack_shader_cache.py, gfx_shader.c
    // disk_load_from): no flush per file. REKARINA_SHADER_CACHE_FLUSH=1 flushes as play does.
    gfx_shader_cache_flush(rt_knob("SHADER_CACHE_FLUSH", "0"));
    int rc = gfx_shader_prepare_seed(dir, cpus, procs < 1 ? 1 : procs);
    if (!rc && cfg_bool("REKARINA_PREPARE_DRIVER")) lb_warm_prepared(dir, NULL, NULL);
    else if (!rc) printf("prepare-shaders: translated only (REKARINA_PREPARE_DRIVER=0)\n");
    return rc;
}

// rekarina.exe --warm-shaders: every variant of shader-prepared.bin built into the driver's disk cache again from the shader
// cache, without translating the seed. The AMD cache is keyed to the exe path (a relink keeps it: 29,587 variants in
// 1.8 s) and holds a limited amount: preparing another build dir evicted this one's libraries, and a library a draw
// needed then took 40-220 ms in play instead of 0.2 ms. Seconds when the libraries are still there.
// Then the entry step's work (vk_pipeline_prewarm: every pipeline the folder's key files and the release's seeds name)
// and pipeline-cache.bin written: the setup runs this in the game folder (installer/oot.py, Package), so the first start
// finds those pipelines built instead of building them (almost five minutes on 16 threads with a cold driver cache,
// 2026-11-16). REKARINA_PIPE_PREBUILD=0 leaves that part out, as it leaves the entry step out.
// On NVIDIA the file is not loaded first (P.cache_fresh, cache_load): what it is written with is what the entry step builds,
// and a file a setup of 1.2.4 filled with every prepared library (build_cache) is written small again at the next
// setup. REKARINA_WARM_FRESH_CACHE=0 loads it as before.
// REKARINA_WARM_STOP=<file>: the launcher's background run (installer/oot.py background_main, with
// REKARINA_PIPE_PREBUILD=0: the libraries alone, the game's own start does the entry step). When the file appears
// (Play was pressed), every worker finishes the library it has handed to the driver and takes no other, and the
// process ends the normal way within a moment, so the driver keeps on disk what it built and the journal holds no
// innocent variants (a killed process leaves both to chance). Exit code 3 then.
static volatile int32_t warm_watch_end;
static void warm_stop_watch(void *arg) {
    const char *path = arg;
    while (!__atomic_load_n(&warm_watch_end, __ATOMIC_ACQUIRE)) {
        if (plat_file_exists(path)) {
            rt_log("vk_pipeline: %s appeared: the build stops", path);
            plat_xchg32(&LB.stop, 1);
            return;
        }
        plat_sleep_ms(100);
    }
}

// --warm-libraries as the launcher's background job (installer/oot.py background_warm): its stdin is a pipe the
// launcher holds; its end (the launcher gone) stops the build as the stop file does.
static void warm_stdin_watch(void *arg) {
    (void)arg;
    char b[64];   // (the descriptor, not stdio: a stdio read would hold stdin's lock against the exit's flush)
    while (plat_stdin_read(b, sizeof b) > 0) {}
    if (!__atomic_load_n(&warm_watch_end, __ATOMIC_ACQUIRE)) {
        rt_log("vk_pipeline: the starter closed this process's input: the build stops");
        plat_xchg32(&LB.stop, 1);
    }
}

static int pw_no_lb;   // --warm-shaders: the entry step starts no boot prebuild after it (the process ends)
int vk_pipeline_warm_shaders(void);
// rekarina.exe --warm-libraries: the prepared variants' libraries this driver lacks, as the game's entry step would build
// them (the records' shaders and unions first), and nothing else; stops on the stop file or when its input closes.
// The launcher's background work runs it while no game runs (the game's entry step builds them in its own process).
int vk_pipeline_warm_libraries(void) {
    warm_libraries = 1;
    return vk_pipeline_warm_shaders();
}
static int pw_libs_only;   // --warm-libraries: the entry step builds no pipeline, only libraries
int vk_pipeline_warm_shaders(void) {
    // --warm-libraries (the launcher's background build): the entry step without its
    // pipelines: the records' shaders loaded and their vertex output unions taken, so that each prepared variant's
    // library is the one the game's own entry step and play build (with the union, in the records' state); then only
    // the prepared variants' libraries, into the driver's disk cache.
    int libs_only = warm_libraries;
    int pipes = cfg_bool("REKARINA_PIPE_PREBUILD") && !libs_only;
    // (only when the entry step runs below: without it nothing would fill the file again)
    P.cache_fresh = pipes && rt_knob("WARM_FRESH_CACHE", "1");
    // The launcher's background build (no entry step) still reads the records, for their library state (records_hist):
    // the entry step builds with it, and the two share the done list only under one state.
    if (pipes || libs_only || rt_knob("WARM_RECORD_STATE", "1")) {   // before ensure_init: keys_open reads the key files with A.on
        vk_pipeline_async_enable();                      // only, and prebuild_start must leave them to the entry step
        K.warm = 1;
    }
    if (!ensure_init()) {
        fprintf(stderr, "warm-shaders: no Vulkan device\n");
        return 1;
    }
    char dir[PLAT_PATH_MAX * 2];
    if (!cache_dir(dir, sizeof dir)) return 1;
    char prp[PLAT_PATH_MAX * 2 + 32];
    snprintf(prp, sizeof prp, "%s/shader-prepared.bin", dir);
    if (!plat_file_exists(prp)) {
        fprintf(stderr, "warm-shaders: no %s (tools/prepare_shaders.py writes it)\n", prp);
        return 1;
    }
    // REKARINA_WARM_STOP alone names a stop file (the launcher's background job sets it). Not the launcher's file by
    // default: Play leaves it behind, and the game's entry step starts this mode as its child right after.
    const char *stop = cfg_get("REKARINA_WARM_STOP");
    PlatThread *watch = stop && *stop ? plat_thread_new(warm_stop_watch, (void *)stop, "warm stop") : NULL;
    if (warm_libraries) plat_thread_new(warm_stdin_watch, NULL, "warm input");   // (not joined: it ends with stdin)
    if (libs_only) {
        pw_libs_only = 1, pw_no_lb = 1;
        vk_pipeline_prewarm();   // (its library part is lb_warm_prepared, with the stop above)
    } else {
        uint32_t lib_pr[16], lib_fs[4];
        records_hist(lib_pr, lib_fs);
        lb_warm_prepared(dir, lib_pr, lib_fs);
    }
    if (watch) plat_xchg32(&warm_watch_end, 1), plat_thread_join(watch);
    if (__atomic_load_n(&LB.stop, __ATOMIC_ACQUIRE)) return 3;   // stopped: no entry step, nothing saved
    if (pipes) {
        double t0 = now_ms();
        pw_no_lb = 1;   // the boot prebuild is for a game that runs on
        vk_pipeline_prewarm();
        vk_pipeline_save();
        int64_t done, total;
        prep_progress(&done, &total, NULL);
        printf("warm-shaders: %lld of %lld recorded pipelines built and saved in %.0f ms\n", (long long)done,
               (long long)total, now_ms() - t0);
    }
    return 0;
}

// ---- the entry step (vk_pipeline_prewarm, rt_boot, before the guest runs)
// Every record the player's key files and the release's seeds hold is built before the guest starts, while boot.c
// shows its progress (prep.h): the shaders they name are loaded, or translated from their stored inputs
// (gfx_shader_prewarm), then their graphics, mesh and geometry pipelines and the compute pipelines a dispatch asked for
// (compute-used.bin with the release's compute-used-seed.bin) are built on every logical processor, the mesh ones with
// the most SPIR-V first: the driver takes up to 25 s for one of those with a cold cache, and that one bounds the step.
// Until 2026-11-16 the graphics pipelines alone were waited for: the mesh pipelines went on beside the title screen and
// into play, and their draws (grass, bushes, tree crowns) were left out until they were there; the compute pipelines
// came from the boot prebuild beside play at below-normal priority, and a dispatch that came first went without its
// pipeline. The boot prebuild (lb_start) still follows the step for the compute shaders no dispatch has asked for.
// Restart-safe: while a builder hands a record to the driver, its slot of <dir>/pipeline-prewarm-journal.bin names the
// record's shaders (input id and stage, as shader-prepare-journal.bin); a process that dies there leaves them, skip_init
// moves them into shader-prepare-skip.txt for this GPU and driver, and the next start drops every record that names one
// of them (they are built at their first draw, as without a seed). A normal exit during the step clears the journal
// (vk_pipeline_prewarm_quit, from prep_report).
#define PW_SLOT 192   // journal bytes per builder: three 64-byte entries (input id, stage) of skip_init's layout
#define PW_THREADS 64

static struct {
    PlatFile *journal;
    volatile int32_t quit;
    const GfxShader **cs;              // compute shaders to build in the step
    size_t ncs;
    volatile int64_t cs_next;
    uint32_t mesh_built, mesh_failed;  // P.lock
    uint32_t libs;                     // P.lock: stage libraries of mesh pipelines built by the builders
    double lib_ms, lib_worst;          // P.lock: their build time summed and the longest
    int lib_run, lib_cap;              // P.lock: builders on them now, and at most (never every logical processor)
    double ms[3], worst[3], end[3];    // P.lock: build time summed and the longest, per kind (mesh, graphics, compute);
                                       // end: now_ms() when the kind's last build returned
} PW;

static void pw_time(int kind, double t0) {   // P.lock held
    double now = now_ms(), ms = now - t0;
    PW.ms[kind] += ms, PW.end[kind] = now;
    if (ms > PW.worst[kind]) PW.worst[kind] = ms;
    prep_progress_build(ms);   // boot.c prep_wait's stuck guard scales with it
}

static void pw_mark(int me, const GfxShader *const sh[3]) {
    if (!PW.journal || PW.quit || me < 0 || me >= PW_THREADS) return;
    uint8_t rec[PW_SLOT] = {0};
    for (int i = 0; sh && i < 3; i++)
        if (sh[i]) memcpy(rec + 64 * i, gfx_shader_input_id(sh[i]), 32), rec[64 * i + 32] = (uint8_t)gfx_shader_stage(sh[i]);
    plat_file_pwrite(PW.journal, rec, sizeof rec, (uint64_t)me * PW_SLOT);
}

void vk_pipeline_prewarm_quit(void) {
    if (!PW.journal || __atomic_exchange_n(&PW.quit, 1, __ATOMIC_ACQ_REL)) return;
    static const uint8_t zero[PW_SLOT * PW_THREADS];   // the process ends with a normal exit: nothing in it crashed
    plat_file_pwrite(PW.journal, zero, sizeof zero, 0);
}

// One builder of the step: the longest builds first, so that no long one is left for the end of the step: mesh and
// geometry jobs (up to 22 s each with a cold driver cache), then compute pipelines (0.9 s on average, up to 10 s), then
// the graphics jobs of the low queue (0.6 s, up to 4 s). Each job's bookkeeping is the one of the thread that would
// otherwise build it (mesh_worker, pipe_worker).
static void pw_builder(void *arg) {
    int me = (int)(intptr_t)arg;
    plat_mutex_lock(&P.lock);
    for (;;) {
#ifdef GAME_NS2
        if (MJ.head) {
            MeshJob *j = MJ.head;
            if (!(MJ.head = j->next)) MJ.tail = NULL;
            plat_mutex_unlock(&P.lock);
            pw_mark(me, j->sh);
            double t0 = now_ms();
            VkPipeline pipe = stages_create(&j->t, j->sh, j->mesh, &j->k, j->l, j->off);
            double ms = now_ms() - t0;
            pw_mark(me, NULL);
            mesh_log_slow(j->sh, j->mesh, ms);
            plat_mutex_lock(&P.lock);
            pw_time(0, t0);
            j->p->info.pipeline = pipe;
            j->p->building = 0;
            if (pipe) {
                P.stats.graphics++, PW.mesh_built++;
            } else {
                P.stats.failures++, PW.mesh_failed++;
                rt_log("vk_pipeline: %s pipeline not built: pipeline creation failed", j->mesh ? "mesh" : "geometry");
            }
            free(j);
            if (!--MJ.busy) plat_cond_broadcast(&MJ.idle);
            prep_progress_step(1);
            continue;
        }
        if (MQ.head && PW.lib_run < PW.lib_cap) {   // the recorded mesh pipelines' stage libraries (mesh_libs_ahead)
            LibJob *lj = MQ.head;
            PW.lib_run++;
            if (!(MQ.head = lj->next)) MQ.tail = NULL;
            lj->b->job = NULL;
            const PrmKey *pk = lj->kind == LIB_PRM ? (const PrmKey *)lj->b->key : NULL;
            const GfxShader *s3[3] = {pk ? pk->ts : lj->s.fs, pk ? pk->ms : NULL, NULL};
            plat_mutex_unlock(&P.lock);
            pw_mark(me, s3);
            double t0 = now_ms();
            plat_mutex_lock(&P.lock);
            lib_build(lj->b, lj->kind, &lj->s, &lj->k, lj->l);   // releases P.lock while building
            pw_mark(me, NULL);
            double ms = now_ms() - t0;
            PW.libs++, PW.lib_ms += ms;
            if (ms > PW.lib_worst) PW.lib_worst = ms;
            prep_progress_build(ms);
            prep_progress_step(1);
            PW.lib_run--;
            free(lj);
            continue;
        }
#endif
        if (prep_progress(NULL, NULL, NULL) == PREP_PHASE_MESH) prep_progress_phase(PREP_PHASE_PIPELINES);
        if ((size_t)__atomic_load_n(&PW.cs_next, __ATOMIC_RELAXED) < PW.ncs) {
            plat_mutex_unlock(&P.lock);
            size_t i = (size_t)plat_inc64(&PW.cs_next) - 1;
            if (i < PW.ncs) {
                const GfxShader *c3[3] = {PW.cs[i], NULL, NULL};
                pw_mark(me, c3);
                double t0 = now_ms();
                lb_compute(PW.cs[i]);
                pw_mark(me, NULL);
                prep_progress_step(1);
                plat_mutex_lock(&P.lock);
                pw_time(2, t0);
            } else {
                plat_mutex_lock(&P.lock);
            }
            continue;
        }
        if (A.low) {
            Job *j = A.low;
            if ((A.low = j->next)) A.low->prev = NULL;
            else A.low_tail = NULL;
            j->next = j->prev = NULL;
            j->low = 0;                    // not on a pool thread: promote() has nothing to raise
            plat_mutex_unlock(&P.lock);
            const GfxShader *s3[3] = {j->s.vs, j->s.fs, NULL};
            pw_mark(me, s3);
            double t0 = now_ms();
            VkPipeline pipe = create_graphics(&j->s, &j->k, j->l, j->fs_off);
            pw_mark(me, NULL);
            plat_mutex_lock(&P.lock);
            pw_time(1, t0);
            if (pipe || !j->p->fast) j->p->info.pipeline = pipe;   // a failed optimised build keeps the fast-linked one
            if (pipe && j->p->fast) P.g.swaps++;
            j->p->building = 0;
            j->p->job = NULL;
            if (pipe) {
                P.stats.graphics++, P.stats.async_built++, K.built++;
            } else {
                P.stats.failures++, K.failed++;
                rt_log("vk_pipeline: graphics pipeline not built: pipeline creation failed");
            }
            K.promoted += j->demanded;
            K.left--;
            if (!--A.inflight) plat_cond_broadcast(&A.idle);
            plat_cond_broadcast(&done_cv);
            free(j);
            prep_progress_step(1);
            continue;
        }
        plat_mutex_unlock(&P.lock);
        break;
    }
}

static int pw_skipped(const uint8_t (*iid)[32], int n) {
    for (int i = 0; i < n; i++)
        if (skip_listed(iid[i])) return 1;
    return 0;
}

void vk_pipeline_prewarm(void) {
    const char *gpu = rt_config()->gpu;
    uint32_t pr_hist[16] = {0}, fs_hist[4] = {0};   // library state of the recorded pipelines (lb_start)
    if (!A.on || (gpu && !strcmp(gpu, "null"))) return;
    if (!cfg_bool("REKARINA_PIPE_PREBUILD")) {   // prebuild_start logs the off case
        if (ensure_init()) lb_start(pr_hist, fs_hist);
        return;
    }
    double t0 = now_ms();
    K.warm = 1;
    prep_progress_begin();   // the picture's seconds count the device's start too
    if (!ensure_init()) {
        prep_progress_end();
        return;
    }
    char dir[PLAT_PATH_MAX * 2], jp[PLAT_PATH_MAX * 2 + 48];
    int have_dir = cache_dir(dir, sizeof dir);
    if (have_dir) skip_init(dir);   // a journal a crashed step left: its records' shaders join the skip list
    vk_mem_footprint_log("before the entry step");

    // the records: graphics keys (K.file), mesh and geometry keys (MK.file), the compute shaders dispatched (CU)
    plat_mutex_lock(&P.lock);
    const KeyRec *r = (const KeyRec *)K.file;
    size_t n = K.file_n, mn = mk_take();
    cu_load();
    size_t cn = CU.n;
    static const char *ck;
    if (!ck) ck = cfg_get("REKARINA_CS_KEEP") ? cfg_get("REKARINA_CS_KEEP") : "";
    if (!strcmp(ck, "none")) cn = 0;   // DEBUG-CSNONE: every compute pipeline at its first dispatch
    uint8_t (*ids)[32] = malloc((2 * n + 3 * mn + cn + 1) * 32);
    size_t m = 0;
    if (ids) {
        for (size_t i = 0; i < n; i++) {
            memcpy(ids[m++], r[i].vs, 32);
            if (r[i].has_fs) memcpy(ids[m++], r[i].fs, 32);
        }
        m += mk_ids(ids + m);
        for (size_t i = 0; i < cn; i++) memcpy(ids[m++], CU.iid[i], 32);
    }
    plat_mutex_unlock(&P.lock);
    const GfxShader **sh = calloc(m + 1, sizeof *sh);
    if (!ids || !sh) {
        free(ids), free(sh), mk_drop();
        prep_progress_end();
        return;
    }
    qsort(ids, m, 32, id_cmp);
    size_t u = 0;
    for (size_t i = 0; i < m; i++)
        if (!u || memcmp(ids[u - 1], ids[i], 32)) memcpy(ids[u++], ids[i], 32);
    // The prepared variants' stage libraries and compute pipelines this driver does not have yet (lb_list_load, WD), built
    // in this step after the records: a pairing play has not recorded then links from libraries the driver has (0.1 ms)
    // instead of building them in a frame (5-180 ms cold). The setup no longer builds them (1.2.8); the launcher's
    // background build (installer/oot.py background_main) may have built some or all; a start that finds none missing
    // costs a file read. REKARINA_ENTRY_LIBRARIES=0 leaves them out (each then builds at its first draw).
    uint32_t lib_pr[16], lib_fs[4];   // the library state of the records (records_hist), for them and their done list
    records_hist(lib_pr, lib_fs);
    size_t nlib = have_dir && rt_knob("ENTRY_LIBRARIES", "1") ? lb_list_load(dir, lib_pr, lib_fs) : 0;
    // the textures the texture cache lacks: walked by a child from now, beside the builds (tex_prepare.c); not in
    // --warm-libraries, whose launcher runs the walk after it
    if (!pw_libs_only) tex_prepare_entry_start();
    // the picture's count: the records. Their shaders load in under a second when the setup translated them, and
    // counted too they filled half the bar at once on a cold driver cache, before minutes of pipeline builds.
    prep_progress_total((int64_t)(n + mn + cn));
    int cpus = (int)plat_cpu_usable();
    prep_part_total(PREP_PART_PIPES, 0, 0.6 / (cpus > 0 ? cpus : 1));   // a graphics job takes 0.6 s cold on average
    // (0.47 s of a thread a library with a cold cache: the setup's driver phase built 56,402 in 1,650 s on 16 threads)
    if (nlib) prep_part_total(PREP_PART_LIBS, (int64_t)nlib, 0.47 / (cpus > 0 ? cpus : 1));
    else wd_close();   // (none missing: the list read is all)
    const char *pe = cfg_get("REKARINA_PREWARM_XLATE");   // no fixed default: half the CPUs, at most 16, when unset
    int procs = pe && *pe ? atoi(pe) : cpus / 2 > 16 ? 16 : cpus / 2;
    gfx_shader_prewarm((const uint8_t(*)[32])ids, u, sh, cpus, procs < 1 ? 1 : procs);
    double t1 = now_ms();
    prep_progress_phase(PREP_PHASE_MESH);   // pw_builder moves it on when the mesh queue is empty

    if (P.indep) {   // vertex output unions first: a builder may take a job before the queue is complete
        uint64_t pair = 0, uni = 0, pairs = 0;
        for (size_t i = 0; i < n; i++) {
            const uint8_t (*v)[32] = bsearch(r[i].vs, ids, u, 32, id_cmp);
            const uint8_t (*f)[32] = r[i].has_fs ? bsearch(r[i].fs, ids, u, 32, id_cmp) : NULL;
            if (!v || !f || !sh[v - ids] || !sh[f - ids]) continue;
            GfxShaderIo io[32];
            vs_outs(sh[v - ids], sh[f - ids], io);
            pair += (uint64_t)__builtin_popcount(gfx_shader_inputs(sh[f - ids], NULL)), pairs++;
        }
        for (uint32_t j = 0; j < VS_OUTS_CAP; j++) uni += VO.vs[j] ? (uint64_t)__builtin_popcount(VO.mask[j]) : 0;
        rt_log("vk_pipeline: vertex output unions: %u vertex shaders, %.2f locations each; %llu pairs read %.2f each",
               VO.n, VO.n ? (double)uni / VO.n : 0.0, (unsigned long long)pairs, pairs ? (double)pair / pairs : 0.0);
    }
    uint32_t queued = 0, dropped = 0, known = 0, skipped = 0;
    plat_mutex_lock(&P.lock);
    K.t0 = t1;
    K.loading = 1;
    K.pw = 1;   // pipe_worker leaves the low queue to the builders below
    for (size_t i = 0; i < n; i++) {
        const uint8_t (*v)[32] = bsearch(r[i].vs, ids, u, 32, id_cmp);
        const uint8_t (*f)[32] = r[i].has_fs ? bsearch(r[i].fs, ids, u, 32, id_cmp) : NULL;
        const GfxShader *vs = v ? sh[v - ids] : NULL, *fs = f ? sh[f - ids] : NULL;
        int q = 0;
        if (!vs || (r[i].has_fs && !fs)) {
            dropped++;
        } else if (pw_skipped(&r[i].vs, 1) || (r[i].has_fs && pw_skipped(&r[i].fs, 1))) {
            skipped++;
        } else {
            VkPipeState s = r[i].s;
            s.vs = vs, s.fs = fs;
            GKey k;
            gkey_fill(&s, &k);
            pr_hist[(k.clamp & 1) | (k.poly_mode & 3) << 1 | (k.minus_w & 1) << 3]++;
            fs_hist[k.samples >= 8 ? 3 : k.samples >= 4 ? 2 : k.samples >= 2 ? 1 : 0]++;
            uint64_t h = key_hash(&k, sizeof k);
            if (pipe_find(&k, sizeof k, h)) known++;
            else if (!pw_libs_only) pipe_miss(&s, &k, h, 2, &q), queued += (uint32_t)q;
        }
        if (!q) prep_progress_step(1);
    }
    K.loading = 0;
    plat_mutex_unlock(&P.lock);
    uint32_t mk_known = 0, mk_dropped = 0, mk_skipped = 0;
    uint32_t mk_queued = 0;
    if (pw_libs_only) mk_drop();
    else mk_queued = mk_queue((const uint8_t(*)[32])ids, u, sh, &mk_known, &mk_dropped, &mk_skipped);
    // the compute shaders: one progress step each now for those not built here
    PW.ncs = 0, PW.cs_next = 0;
    PW.cs = cn && !pw_libs_only ? calloc(cn, sizeof *PW.cs) : NULL;
    uint32_t cs_dropped = 0;
    plat_mutex_lock(&P.lock);
    for (size_t i = 0; i < cn; i++) {
        const uint8_t (*e)[32] = bsearch(CU.iid[i], ids, u, 32, id_cmp);
        const GfxShader *c = e ? sh[e - ids] : NULL;
        if (PW.cs && c && gfx_shader_stage(c) == GFX_STAGE_CS && !skip_listed(CU.iid[i])) PW.cs[PW.ncs++] = c;
        else cs_dropped++, prep_progress_step(1);
    }
    plat_mutex_unlock(&P.lock);
    free(ids), free(sh);

    // the builders: every logical processor, at most PW_THREADS; the journal while they run
    int work = queued || mk_queued || PW.ncs;
    if (have_dir) {
        snprintf(jp, sizeof jp, "%s/pipeline-prewarm-journal.bin", dir);
        plat_file_delete(jp);
        if (work) PW.journal = plat_fs_create(jp, NULL);
    }
    double t2 = now_ms();
    PW.lib_run = 0, PW.lib_cap = cpus - cpus / 4 < 1 ? 1 : cpus - cpus / 4;
    PlatThread *t[PW_THREADS];
    int nt = 0;
    for (int i = 0; i < cpus && i < PW_THREADS && work; i++)
        if ((t[nt] = plat_thread_new(pw_builder, (void *)(intptr_t)nt, "entry step"))) nt++;
    if (!nt && work) pw_builder((void *)(intptr_t)0);
    for (int i = 0; i < nt; i++)
        while (!plat_thread_join_ns(t[i], 5000000000u)) {
            int64_t d, tot;
            prep_progress(&d, &tot, NULL);
            rt_log("vk_pipeline: entry step: %lld of %lld, %.0f ms", (long long)d, (long long)tot, now_ms() - t0);
        }
    plat_mutex_lock(&P.lock);
    while (K.left) plat_cond_wait(&done_cv, &P.lock);   // none: pipe_worker leaves the low queue while K.pw
#ifdef GAME_NS2
    while (MJ.busy) plat_cond_wait(&MJ.idle, &P.lock);
#endif
    K.pw = 0;
    if (A.low) plat_cond_broadcast(&work_cv);
    free(K.file);
    K.file = NULL;
    plat_mutex_unlock(&P.lock);
    if (PW.journal) {
        PlatFile *f = PW.journal;
        PW.journal = NULL;
        plat_file_close(f);
        plat_file_delete(jp);   // every build returned
    }
    rt_log("vk_pipeline: entry step: %zu graphics records (%s and the release's seed): %u built, %u failed, %u already "
           "built, %u dropped (shader neither cached nor stored), %u skipped (shader-prepare-skip.txt); %zu mesh and "
           "geometry records: %u queued (%u built, %u failed), %u already built, %u dropped, %u skipped; %zu compute "
           "shaders dispatched before: %zu built here (%u not loadable); shaders %.0f ms, pipelines %.0f ms on %d "
           "threads, total %.0f ms", n, K.path, K.built, K.failed, known, dropped, skipped, mn, mk_queued, PW.mesh_built,
           PW.mesh_failed, mk_known, mk_dropped, mk_skipped, cn, PW.ncs, cs_dropped, t1 - t0, now_ms() - t2, nt,
           now_ms() - t0);
    static const char *const kn[3] = {"mesh and geometry", "graphics", "compute"};
    for (int k = 0; k < 3; k++)
        if (PW.end[k])
            rt_log("vk_pipeline: entry step: %s builds took %.0f ms of thread time (longest %.0f ms), the last returned "
                   "%.0f ms after the builders started", kn[k], PW.ms[k], PW.worst[k], PW.end[k] - t2);
#ifdef GAME_NS2
    if (MG.libs_ahead)
        rt_log("vk_pipeline: entry step: %llu stage libraries of mesh pipelines (task and mesh: %llu, avg %.1f ms, longest "
               "%.0f ms in all so far), %u of them built by the builders in %.0f ms of thread time (longest %.0f ms)",
               (unsigned long long)MG.libs_ahead, (unsigned long long)P.g.lib_n[LIB_PRM],
               P.g.lib_n[LIB_PRM] ? P.g.lib_ms[LIB_PRM] / P.g.lib_n[LIB_PRM] : 0.0, P.g.lib_max[LIB_PRM], PW.libs, PW.lib_ms,
               PW.lib_worst);
#endif
    free(PW.cs), PW.cs = NULL, PW.ncs = 0;
    // In this process, also in the game's: the drivers give each process a disk cache file of its own (NVIDIA's
    // GLCache: the first process takes the folder's file, one started while it runs a new empty one, and none reads
    // another's; AMD's VkCache keeps "_1" files beside its own the same way), so a child of the running game built
    // every library into a file the game never read (2026-11-20: 11,000 libraries from nothing at 480 ms each while
    // the game held a file with all of them). Their memory is small: the driver's footprint stayed at 297-322 MiB over
    // 8,000 builds in a process of their own and did not move over 48,000 in the game's (the 43 GiB of that run were
    // the texture walk's, which runs in a child: tex_prepare.c). The launcher's background build (--warm-libraries)
    // runs while no game does, so it fills the folder's file.
    if (nlib) {
        double tl = now_ms();
        lb_entry = 1;
        lb_warm_prepared(dir, lib_pr, lib_fs);
        lb_entry = 0;
        if (!pw_libs_only)
            rt_log("vk_pipeline: entry step: %zu prepared variants handed to the driver in %.0f ms", nlib, now_ms() - tl);
    }
    double tj = now_ms();
    tex_prepare_entry_join();   // play reads the texture cache: every texture the step walks is there before it
    if (now_ms() - tj > 100) rt_log("vk_pipeline: entry step: waited %.0f ms for the textures", now_ms() - tj);
    prep_progress_end();
    vk_mem_footprint_log("after the entry step");
    if (!pw_no_lb) lb_start(pr_hist, fs_hist);
}

// ---- dynamic state

// What the command buffer being recorded holds of each dynamic state (this thread's: the replay's, or a self test's).
// OoT changes some state about 800 times a frame in the forest and every change used to set all of them
// again: 12,800 of the frame's 36,600 Vulkan calls, most of them with the value already there (2026-11-12). A state is
// now set when its value differs or the command buffer began anew (vk_sync_cb_serial). Nothing else sets these states,
// and every graphics pipeline bound into a replay command buffer has them dynamic, so a set value stays.
// REKARINA_DYN_SHADOW=0: every state at every change, as before.
typedef struct {
    VkCommandBuffer cb;
    uint64_t serial;
    uint32_t have;
    VkCullModeFlags cull;
    VkFrontFace ff;
    VkPrimitiveTopology topo;
    VkBool32 dt, dw, st, dbe, rd, pr, clamp, a2c;
    VkCompareOp dop;
    uint32_t sop[2][4];
    VkPolygonMode poly;
    VtxInput vi;
    int n;
    VkBool32 en[8];
    VkColorBlendEquationEXT eq[8];
    VkColorComponentFlags wm[8];
} DynShadow;
static _Thread_local DynShadow DS;
enum { DY_CULL, DY_FF, DY_TOPO, DY_DT, DY_DW, DY_DOP, DY_ST, DY_SOPF, DY_SOPB, DY_DBE, DY_RD, DY_PR, DY_VI, DY_CLAMP, DY_POLY,
       DY_A2C, DY_EN, DY_EQ, DY_WM };
// 1: the state is to be set (and is noted as v).
#define DY_NEW(bit, field, v) (!(DS.have & (1u << (bit))) || DS.field != (v) ? (DS.have |= 1u << (bit), DS.field = (v), 1) : 0)

// A pipeline with this state static was bound (vk_exec_draw.c mesh_emulate): the next draw sets every dynamic state.
void vk_pipeline_dynamic_forget(void) { DS.have = 0; }

static void set_dynamic(VkCommandBuffer cb, const VkPipeState *s, int assembly) {
    if (!ensure_init()) return;
    static int shadow = -1;
    if (shadow < 0) shadow = rt_knob("DYN_SHADOW", "1");
    uint64_t serial = vk_sync_cb_serial();
    if (!shadow || DS.cb != cb || DS.serial != serial) DS.cb = cb, DS.serial = serial, DS.have = 0;
    const poly_st *pl = &s->poly;
    const ds_st *d = &s->ds;
    // The front face maps unchanged: TOTK sets the upper-left window origin, and the y-flipped viewport (y_flip) puts
    // NDC +y at row 0 as NVN does, so the framebuffer winding Vulkan tests is the one NVN tests (MacNativeGames passes
    // it to Metal unchanged; LAS Recomp exec.c mirrors it only for the lower-left origin).
    VkFrontFace ff = pl->front == NVN_FRONT_FACE_CW ? VK_FRONT_FACE_CLOCKWISE : VK_FRONT_FACE_COUNTER_CLOCKWISE;
    VkCullModeFlags cull = (VkCullModeFlags)(pl->cull & 3);   // NVN_FACE_* has VkCullModeFlagBits' values
    if (DY_NEW(DY_CULL, cull, cull)) vkCmdSetCullMode(cb, cull);
    if (DY_NEW(DY_FF, ff, ff)) vkCmdSetFrontFace(cb, ff);
    VkPrimitiveTopology topo = vk_pipeline_topology(s->prim);
    if (assembly && DY_NEW(DY_TOPO, topo, topo)) vkCmdSetPrimitiveTopology(cb, topo);
    VkBool32 dt = d->dt, dw = d->dw;
    if (DY_NEW(DY_DT, dt, dt)) vkCmdSetDepthTestEnable(cb, dt);
    {   // REKARINA_DEPTH_EQ=le|ge|always (diagnostic): what an EQUAL depth test becomes. OoT shades its scene with EQUAL
        // against its depth prepass; =always shows whether a pass's draws are there but fail the test.
        static int eq = -1;
        if (eq < 0) {
            const char *e = cfg_get("REKARINA_DEPTH_EQ");
            eq = !e ? 0 : !strcmp(e, "le") ? 1 : !strcmp(e, "ge") ? 2 : !strcmp(e, "always") ? 3 : 0;
        }
        VkCompareOp op = compare_op(d->df);
        static int alt = -1;   // REKARINA_DEPTH_EQ_ALT=1: the EQ replacement only on odd frames, for side-by-side shots
        if (alt < 0) alt = cfg_env("REKARINA_DEPTH_EQ_ALT") != NULL;
        int eq_now = eq && (!alt || (__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) & 1));
        if (op == VK_COMPARE_OP_EQUAL && eq_now) op = eq == 1 ? VK_COMPARE_OP_LESS_OR_EQUAL : eq == 2 ? VK_COMPARE_OP_GREATER_OR_EQUAL : VK_COMPARE_OP_ALWAYS;
        static int probe = -1;   // DEBUG-PROBE (vk_exec_xfer.c copy_t2t): EQUAL draws write their own depth, nearest first
        if (probe < 0) probe = cfg_env("REKARINA_DEPTH_PROBE") != NULL;
        if (probe && op == VK_COMPARE_OP_EQUAL) op = VK_COMPARE_OP_LESS, dw = VK_TRUE;
        if (DY_NEW(DY_DW, dw, dw)) vkCmdSetDepthWriteEnable(cb, dw);
        if (DY_NEW(DY_DOP, dop, op)) vkCmdSetDepthCompareOp(cb, op);
    }
    VkBool32 st = d->st;
    if (DY_NEW(DY_ST, st, st)) vkCmdSetStencilTestEnable(cb, st);
    uint32_t sf[4] = {(uint32_t)stencil_op(d->fail_f), (uint32_t)stencil_op(d->zpass_f), (uint32_t)stencil_op(d->zfail_f),
                      (uint32_t)compare_op(d->sf_f)};
    uint32_t sb[4] = {(uint32_t)stencil_op(d->fail_b), (uint32_t)stencil_op(d->zpass_b), (uint32_t)stencil_op(d->zfail_b),
                      (uint32_t)compare_op(d->sf_b)};
    if (!(DS.have & (1u << DY_SOPF)) || memcmp(DS.sop[0], sf, sizeof sf)) {
        DS.have |= 1u << DY_SOPF, memcpy(DS.sop[0], sf, sizeof sf);
        vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_BIT, (VkStencilOp)sf[0], (VkStencilOp)sf[1], (VkStencilOp)sf[2], (VkCompareOp)sf[3]);
    }
    if (!(DS.have & (1u << DY_SOPB)) || memcmp(DS.sop[1], sb, sizeof sb)) {
        DS.have |= 1u << DY_SOPB, memcpy(DS.sop[1], sb, sizeof sb);
        vkCmdSetStencilOp(cb, VK_STENCIL_FACE_BACK_BIT, (VkStencilOp)sb[0], (VkStencilOp)sb[1], (VkStencilOp)sb[2], (VkCompareOp)sb[3]);
    }
    // NVN_POLYGON_OFFSET_ENABLE_* bit of the polygon mode in use: point 1, line 2, fill 4
    unsigned mode = pl->mode <= NVN_POLYGON_MODE_FILL ? pl->mode : NVN_POLYGON_MODE_FILL;
    VkBool32 dbe = (pl->offset_en >> mode & 1) != 0, rd = s->raster_discard != 0, pr = s->prim_restart != 0;
    if (DY_NEW(DY_DBE, dbe, dbe)) vkCmdSetDepthBiasEnable(cb, dbe);
    if (DY_NEW(DY_RD, rd, rd)) vkCmdSetRasterizerDiscardEnable(cb, rd);
    if (assembly && DY_NEW(DY_PR, pr, pr)) vkCmdSetPrimitiveRestartEnable(cb, pr);
    if (!assembly) DS.have &= ~(1u << DY_TOPO | 1u << DY_PR | 1u << DY_VI);
    if (assembly && P.dyn_vi && s->vs) {
        VtxInput v;
        vertex_input_(s, &v, 0);
        if (!(DS.have & (1u << DY_VI)) || v.n_bind != DS.vi.n_bind || v.n_attr != DS.vi.n_attr ||
            memcmp(v.b, DS.vi.b, sizeof v.b[0] * v.n_bind) || memcmp(v.a, DS.vi.a, sizeof v.a[0] * v.n_attr)) {
            DS.have |= 1u << DY_VI;
            DS.vi.n_bind = v.n_bind, DS.vi.n_attr = v.n_attr;
            memcpy(DS.vi.b, v.b, sizeof v.b[0] * v.n_bind), memcpy(DS.vi.a, v.a, sizeof v.a[0] * v.n_attr);
            rec_vertex_input(P.set_vi, cb, v.n_bind, v.b, v.n_attr, v.a);
        }
    }
    VkBool32 clamp = s->depth_clamp != 0, a2c = s->ms.a2c != 0;
    VkPolygonMode poly = polygon_mode(pl->mode);
    if (P.dyn_clamp && DY_NEW(DY_CLAMP, clamp, clamp)) rec_bool(P.set_clamp, cb, clamp);
    if (P.dyn_poly && DY_NEW(DY_POLY, poly, poly)) rec_polygon_mode(P.set_poly, cb, poly);
    if (P.dyn_a2c && DY_NEW(DY_A2C, a2c, a2c)) rec_bool((PFN_vkCmdSetDepthClampEnableEXT)P.set_a2c, cb, a2c);   // the same signature
    int n = target_count(s);
    // Depth-only pipelines still declare dynamic color state. Initialize it without enabling any color writes.
    VkBool32 en[8] = {VK_FALSE};
    VkColorBlendEquationEXT eq[8] = {{0}};
    VkColorComponentFlags wm[8] = {0};
    for (int i = 0; i < n; i++) {
        const blend_st *b = &s->blend[i];
        en[i] = (s->color.blend_en >> i & 1) && blendable(target_format(s->color_fmt[i], NULL, NULL));
        eq[i] = (VkColorBlendEquationEXT){blend_factor(b->src_rgb), blend_factor(b->dst_rgb), blend_op(b->eq_rgb),
                                          blend_factor(b->src_a), blend_factor(b->dst_a), blend_op(b->eq_a)};
        wm[i] = target_wmask(s, i);
    }
    if (!n) n = 1;
    // The three are kept per attachment count: a state set for fewer attachments leaves the others as they were.
    if (DS.n != n) DS.n = n, DS.have &= ~(1u << DY_EN | 1u << DY_EQ | 1u << DY_WM);
    if (P.dyn_blend_en && (!(DS.have & (1u << DY_EN)) || memcmp(DS.en, en, sizeof en[0] * (size_t)n))) {
        DS.have |= 1u << DY_EN, memcpy(DS.en, en, sizeof en[0] * (size_t)n);
        rec_blend_enable(P.set_blend_en, cb, 0, (uint32_t)n, en);
    }
    if (P.dyn_blend_eq && (!(DS.have & (1u << DY_EQ)) || memcmp(DS.eq, eq, sizeof eq[0] * (size_t)n))) {
        DS.have |= 1u << DY_EQ, memcpy(DS.eq, eq, sizeof eq[0] * (size_t)n);
        rec_blend_equation(P.set_blend_eq, cb, 0, (uint32_t)n, eq);
    }
    if (P.dyn_wmask && (!(DS.have & (1u << DY_WM)) || memcmp(DS.wm, wm, sizeof wm[0] * (size_t)n))) {
        DS.have |= 1u << DY_WM, memcpy(DS.wm, wm, sizeof wm[0] * (size_t)n);
        rec_write_mask(P.set_wmask, cb, 0, (uint32_t)n, wm);
    }
}

void vk_pipeline_set_dynamic(VkCommandBuffer cb, const VkPipeState *s) { set_dynamic(cb, s, 1); }
void vk_pipeline_set_dynamic_no_assembly(VkCommandBuffer cb, const VkPipeState *s) { set_dynamic(cb, s, 0); }

// ---- statistics, persistence, teardown

void vk_pipeline_stats(VkPipeStats *s) {
    plat_mutex_lock_shared(&P.lock);
    *s = P.stats;
    plat_mutex_unlock_shared(&P.lock);
}

static PlatMutex save_lock = PLAT_MUTEX_INIT;   // one cache save at a time; shutdown takes it to destroy the cache

void vk_pipeline_save(void) {
    if (P.ready) keys_flush(), mk_flush();
    if (P.ready && P.gpl) {
        rt_log("vk_pipeline: gpl: %llu fast links (%llu by workers; avg %.2f ms, max %.2f ms, %llu failed), %llu draws "
               "lookups found shader libraries missing (left to a worker), %llu optimised pipelines swapped in (%llu queued "
               "low after their fast link), %llu drawing-thread links over 5 ms; %llu waiting draws built their libraries "
               "(avg %.2f ms, max %.2f ms)",
               (unsigned long long)P.g.links, (unsigned long long)P.g.wlinks, P.g.links ? P.g.link_ms / P.g.links : 0.0,
               P.g.link_max, (unsigned long long)P.g.link_fail, (unsigned long long)P.g.defer,
               (unsigned long long)P.g.swaps, (unsigned long long)P.g.requeued, (unsigned long long)P.g.slow,
               (unsigned long long)P.g.dlinks, P.g.dlinks ? P.g.dlink_ms / P.g.dlinks : 0.0, P.g.dlink_max);
        if (U.waits || U.queued)
            rt_log("vk_pipeline: gpl: library jobs for waiting draws: %llu queued, %llu built by a worker, %llu by the "
                   "thread that needed them (no worker free), %llu waits ended at their limit",
                   (unsigned long long)U.queued, (unsigned long long)U.by_worker, (unsigned long long)U.stolen,
                   (unsigned long long)U.timeouts);
        static const char *const kn[5] = {"vertex input", "pre-rasterization", "fragment shader", "fragment output",
                                          "task and mesh pre-rasterization"};
        for (int i = 0; i < 5; i++)
            rt_log("vk_pipeline: gpl: %s libraries %llu (avg %.2f ms, max %.2f ms)", kn[i], (unsigned long long)P.g.lib_n[i],
                   P.g.lib_n[i] ? P.g.lib_ms[i] / P.g.lib_n[i] : 0.0, P.g.lib_max[i]);
#ifdef GAME_NS2
        if (MG.on > 0)
            rt_log("vk_pipeline: gpl: mesh draws: %llu fast links (avg %.2f ms, max %.2f ms), %llu first draws waited for "
                   "their libraries (avg %.2f ms, max %.2f ms; %llu still without them at the limit), %llu pipelines without a "
                   "stand-in (a library or the link failed), %llu libraries queued by the entry step",
                   (unsigned long long)MG.links, MG.links ? MG.link_ms / MG.links : 0.0, MG.link_max,
                   (unsigned long long)MG.waits, MG.waits ? MG.wait_sum / MG.waits : 0.0, MG.wait_max,
                   (unsigned long long)MG.timeouts, (unsigned long long)MG.finals, (unsigned long long)MG.libs_ahead);
#endif
        if (LB.n)
            rt_log("vk_pipeline: gpl: prebuilt libraries a draw used: pre-rasterization %llu, fragment shader %llu; "
                   "compute pipelines built on the replay thread %llu (avg %.2f ms, max %.2f ms)",
                   (unsigned long long)P.g.pre_used[LIB_PR], (unsigned long long)P.g.pre_used[LIB_FS],
                   (unsigned long long)LB.cs_draw, LB.cs_draw ? LB.cs_draw_ms / (double)LB.cs_draw : 0.0, LB.cs_draw_max);
    }
    gfx_shader_record_flush(1);
    if (!P.ready || !P.cache || !P.cache_path[0] || P.cache_unloaded) return;
    // Not under P.lock: the driver synchronises P.cache, and holding P.lock through the copy and the file write (about
    // 60 ms on the 1.4.3 fight) stopped every pipeline lookup of the replay, a freeze at each 20 s autosave.
    plat_mutex_lock(&save_lock);
    if (!P.cache) {   // vk_pipeline_shutdown destroyed it
        plat_mutex_unlock(&save_lock);
        return;
    }
    // The driver writes the data straight into a view of a new file beside the cache, which then replaces it: a buffer
    // of the cache's size (626 MB) was private memory at every save, every 20 s while new pipelines came, and its
    // bytes went through the file cache a second time (--warm-shaders saving a 2.2 GB cache: the process's peak commit
    // 10,520 MiB with the buffer, 8,612 through the view, 2026-11-19). VK_INCOMPLETE (pipelines added since the size was asked, beyond
    // the room left for them) writes nothing, as before: the next save tries again.
    size_t size = 0;
    void *data = NULL;
    double t0 = now_ms(), t1 = 0, t2 = 0;
    if (vkGetPipelineCacheData(P.device, P.cache, &size, NULL) == VK_SUCCESS && size) {
        t1 = now_ms();
        // Nothing new since the file was loaded or written (the driver gives the same size back): not written. The
        // 1.2.4 check's four starts each wrote the same 3,057,551,174 bytes again, 1.3-1.4 s of the driver's copy and
        // 1.3-1.6 s of file at every exit, on an NVMe disk (5.5-5.9 s of file in the copy of the game folder above
        // build_cache). REKARINA_PIPE_SAVE_SAME=1 writes it anyway.
        static int same_skip = -1;
        if (same_skip < 0) same_skip = !rt_knob("PIPE_SAVE_SAME", "0");
        if (same_skip && (uint64_t)size == __atomic_load_n(&P.cache_bytes, __ATOMIC_RELAXED)) {
            static int said;
            if (said < 4 && ++said)
                rt_log("vk_pipeline: pipeline cache unchanged (%.1f MiB, as loaded or last written): not written again",
                       size / 1048576.0);
            plat_mutex_unlock(&save_lock);
            return;
        }
        char tmp[sizeof P.cache_path + 32];
        snprintf(tmp, sizeof tmp, "%s.%u.tmp", P.cache_path, (unsigned)plat_process_id());
        size_t room = size + (4u << 20);
        PlatView *v = plat_view_create(tmp, room, &data);
        if (v) {
            size_t got = room;
            VkResult r = vkGetPipelineCacheData(P.device, P.cache, &got, data);
            t2 = now_ms();
            int ok = plat_view_close(v, r == VK_SUCCESS ? got : 0) && r == VK_SUCCESS && got;
            if (!ok || !plat_file_replace(tmp, P.cache_path)) {
                plat_file_delete(tmp);
                if (r != VK_INCOMPLETE) rt_log("vk_pipeline: cannot write %s", P.cache_path);
            } else {
                __atomic_store_n(&P.cache_bytes, (uint64_t)got, __ATOMIC_RELAXED);
                rt_log("vk_pipeline: pipeline cache saved at frame %llu: %.1f MiB; the driver's size %.0f ms and copy %.0f "
                       "ms, the file %.0f ms", (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED),
                       got / 1048576.0, t1 - t0, t2 - t1, now_ms() - t2);
            }
        } else if ((data = malloc(size)) && vkGetPipelineCacheData(P.device, P.cache, &size, data) == VK_SUCCESS) {
            if (!plat_file_write_atomic(P.cache_path, data, size)) rt_log("vk_pipeline: cannot write %s", P.cache_path);
            else __atomic_store_n(&P.cache_bytes, (uint64_t)size, __ATOMIC_RELAXED);
            free(data);
        } else {
            free(data);
        }
    }
    plat_mutex_unlock(&save_lock);
}

// Saving in play: a player's cache holds the prepared variants, about 1 GB, and NVIDIA's driver holds every pipeline
// creation while vkGetPipelineCacheData copies it, also one that uses another cache. On a 948 MiB cache (RTX 5070,
// Ryzen 7 9700X, NVMe, 2026-11-19) a creation on another thread waited for the whole copy: 116-137 ms into a buffer,
// 344 ms into a view of a new file as above, 425-649 ms in a game run; the file then took 481-1332 ms more. That came
// every 20 s while play built new pipelines, and on a slower machine or disk it is longer, so draws that need a new
// pipeline then stand still. A cache over AUTOSAVE_MAX_MB is therefore saved when the game ends only
// (gfx_window_closed, the game's exit, vk_pipeline_shutdown): what play adds to it is in the driver's own disk cache as
// well (nv-cache), so a crash before the exit costs little. REKARINA_PIPE_AUTOSAVE=1 saves it in play anyway (the
// behaviour before, for a comparison), 0 never in play.
#define AUTOSAVE_MAX_MB 64
static int autosave_in_play(void) {
    const char *k = cfg_get("REKARINA_PIPE_AUTOSAVE");
    if (k && *k) return atoi(k) != 0;
    uint64_t mb = __atomic_load_n(&P.cache_bytes, __ATOMIC_RELAXED) >> 20;
    if (mb <= AUTOSAVE_MAX_MB) return 1;
    static int told;
    if (!told++)
        rt_log("vk_pipeline: the pipeline cache holds %llu MiB: saved when the game ends, not during play",
               (unsigned long long)mb);
    return 0;
}

// The game never shuts the device down (boot.c and the window end the process), so a small cache is saved while it
// runs: at most every 20 s, and only after new pipelines were built (autosave_in_play). New pipeline keys are appended
// every 2 s.
static void autosave_once(void) {
    static uint64_t last_n, last_ms, last_keys;
    if (!P.ready) return;
    uint64_t now = plat_now_ms();
    if ((*(volatile uint32_t *)&K.pend_n || mk_pending()) && now - last_keys >= 2000) last_keys = now, keys_flush(), mk_flush();
    gfx_shader_record_flush(0);
    if (now - last_ms < 20000) return;
    VkPipeStats s;
    vk_pipeline_stats(&s);
    if (s.graphics + s.compute == last_n) return;
    last_n = s.graphics + s.compute, last_ms = now;
    if (autosave_in_play()) vk_pipeline_save();
}

// On its own thread: the save (vkGetPipelineCacheData and the file write, about 60 ms) and the key appends (a flush to
// the disk each) ran on the present thread, and the render thread waited for it to take the next frame.
static void autosave_main(void *arg) {
    (void)arg;
    plat_thread_priority(PLAT_PRIO_BELOW_NORMAL);
    for (;;) {
        plat_sleep_ms(500);
        autosave_once();
    }
}

void vk_pipeline_autosave(void) {
    static int32_t started;
    if (!P.ready || plat_xchg32(&started, 1)) return;
    if (!plat_thread_spawn(autosave_main, NULL, "pipe autosave", 0))
        rt_log("vk_pipeline: cannot start the autosave thread; the cache is saved at exit only");
}

void vk_pipeline_shutdown(void) {
    if (!P.ready) return;
    lb_stop();
    workers_drain();   // jobs point at layouts and Pipe entries freed below
#ifdef GAME_NS2
    mesh_drain();      // so do mesh jobs
#endif
    vk_pipeline_save();
    plat_mutex_lock(&P.lock);
    vkDeviceWaitIdle(P.device);
    for (int i = 0; i < PIPE_BUCKETS; i++) {
        for (Pipe *p = pipes[i], *n; p; p = n) {
            n = p->next;
            if (p->info.pipeline) vkDestroyPipeline(P.device, p->info.pipeline, NULL);
            if (p->fast && p->fast != p->info.pipeline) vkDestroyPipeline(P.device, p->fast, NULL);
            if (p->alt) {   // a mesh pipeline's fast-linked stand-in (mesh_fast)
                if (p->alt->info.pipeline) vkDestroyPipeline(P.device, p->alt->info.pipeline, NULL);
                free(p->alt->bindings), free(p->alt);
            }
            free(p->bindings), free(p);
        }
        pipes[i] = NULL;
    }
    libs_destroy();
    for (int i = 0; i < LAYOUT_BUCKETS; i++) {
        for (Layout *l = layouts[i], *n; l; l = n) {
            n = l->next;
            for (int k = 0; k < 2; k++)
                if (l->tpl[k]) P.tpl_destroy(P.device, l->tpl[k], NULL);
            vkDestroyPipelineLayout(P.device, l->layout, NULL);
            for (int h = 0; h < 2; h++)
                if (l->half[h]) vkDestroyPipelineLayout(P.device, l->half[h], NULL);
            vkDestroyDescriptorSetLayout(P.device, l->set, NULL);
            if (l->set1) vkDestroyDescriptorSetLayout(P.device, l->set1, NULL);
            free(l);
        }
        layouts[i] = NULL;
    }
    plat_mutex_lock(&save_lock);   // not while the autosave thread copies it
    if (P.cache) vkDestroyPipelineCache(P.device, P.cache, NULL);
    P.cache = VK_NULL_HANDLE;
    plat_mutex_unlock(&save_lock);
    memset(&P.stats, 0, sizeof P.stats);
    plat_xchg32(&P.ready, 0);
    plat_mutex_unlock(&P.lock);
}

// ---- self-test: census shaders (probe_census_dir, tools/shader_census.py) through gfx_shader_get into pipelines.
// Graphics pairs are a fragment shader and a vertex shader of the same source file, spread over the census; compute
// shaders are spread the same way. Every pipeline is built twice (the second call must hit the cache) and the
// dynamic state of the first graphics pipeline is recorded into a throwaway command buffer.

typedef struct { const char *key, *c1, *src; size_t c1_len, src_len; int stage; uint32_t local[3]; } CensusEntry;

static const char *st_field(const char *from, const char *to, const char *name, size_t *len) {
    char pat[32];
    snprintf(pat, sizeof pat, "\"%s\": \"", name);
    const char *p = strstr(from, pat);
    if (!p || p >= to) return NULL;
    p += strlen(pat);
    const char *e = memchr(p, '"', (size_t)(to - p));
    if (!e) return NULL;
    *len = (size_t)(e - p);
    return p;
}

static int census_cmp(const void *x, const void *y) {
    const CensusEntry *a = x, *b = y;
    size_t n = a->src_len < b->src_len ? a->src_len : b->src_len;
    int c = memcmp(a->src, b->src, n);
    if (c) return c;
    if (a->src_len != b->src_len) return a->src_len < b->src_len ? -1 : 1;
    return a->stage - b->stage;
}

static char *census_read(const char *root, size_t *size) {
    char path[PLAT_PATH_MAX * 2 + 64];
    snprintf(path, sizeof path, "%s/jobs.json", root);
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = malloc((size_t)n + 1);
    if (b && fread(b, 1, (size_t)n, f) != (size_t)n) free(b), b = NULL;
    fclose(f);
    if (b) b[n] = 0, *size = (size_t)n;
    return b;
}

static int census_parse(const char *jobs, CensusEntry *out, int max) {
    int n = 0;
    for (const char *p = jobs; n < max && (p = strstr(p, "\"job\": {")); p += 8) {
        const char *end = strstr(p, "\"origin\": ");
        if (!end) break;
        size_t kl, sl, cl, rl;
        const char *k = st_field(p, end, "key", &kl), *s = st_field(p, end, "stage", &sl), *c = st_field(p, end, "c1", &cl),
                   *r = st_field(p, end, "source", &rl);
        if (!k || !s || !c || !r || kl != 64) continue;
        CensusEntry *e = &out[n++];
        e->key = k, e->c1 = c, e->c1_len = cl / 2, e->src = r, e->src_len = rl;
        e->stage = !strncmp(s, "vs", sl) ? GFX_STAGE_VS : !strncmp(s, "ps", sl) ? GFX_STAGE_FS : GFX_STAGE_CS;
        e->local[0] = e->local[1] = e->local[2] = 1;
        const char *l = strstr(p, "\"local\": [");
        if (l && l < end) {
            l += 10;
            for (int i = 0; i < 3; i++) {
                while (*l == ' ' || *l == '\n' || *l == '\r' || *l == ',') l++;
                e->local[i] = (uint32_t)strtoul(l, (char **)&l, 10);
            }
        }
    }
    return n;
}

static int hexv(char c) { return c <= '9' ? c - '0' : (c | 32) - 'a' + 10; }

// The program the game would hand NVN for a census entry, translated with default answers (as gfx_shader_probe.c).
static const GfxShader *census_shader(const char *root, const CensusEntry *e) {
    char path[PLAT_PATH_MAX * 2 + 128];
    static uint8_t c1[8192];
    uint8_t control[0x880] = {0};
    if (e->c1_len > sizeof c1) return NULL;
    for (size_t b = 0; b < e->c1_len; b++) c1[b] = (uint8_t)(hexv(e->c1[b * 2]) << 4 | hexv(e->c1[b * 2 + 1]));
    snprintf(path, sizeof path, "%s/code/%.64s.code", root, e->key);
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long body = ftell(f);
    fseek(f, 0, SEEK_SET);
    size_t skip = e->stage == GFX_STAGE_CS ? 0 : 0x30, total = skip + (size_t)body;
    uint8_t *code = calloc(1, total);
    int ok = code && fread(code + skip, 1, (size_t)body, f) == (size_t)body;
    fclose(f);
    if (!ok) return free(code), NULL;
    uint32_t c1_off = (uint32_t)(total - e->c1_len), length = (uint32_t)total;
    memcpy(control + 0x700, &c1_off, 4), memcpy(control + 0x704, &length, 4);
    GfxShaderAnswers a = {0};
    if (e->stage == GFX_STAGE_CS) for (int k = 0; k < 3; k++) a.local[k] = (uint16_t)e->local[k];
    const GfxShader *s = gfx_shader_get(e->stage, control, code, total, c1, &a);
    free(code);
    return s;
}

// A state for a census pair: one target per fragment output (RGBA8 of the output's type), D32S8 depth with the test on,
// alpha blending on target 0, one RGBA32 attribute per vertex input, interleaved in stream 0.
static void selftest_state(VkPipeState *s, const GfxShader *vs, const GfxShader *fs) {
    vk_pipe_state_defaults(s);
    s->vs = vs, s->fs = fs;
    GfxShaderIo io[32];
    uint32_t outs = gfx_shader_outputs(fs, io);
    for (int i = 0; i < 8; i++)
        if (outs >> i & 1)
            s->color_fmt[i] = io[i].type == GFX_TYPE_UINT ? NVN_FORMAT_RGBA8UI : io[i].type == GFX_TYPE_SINT ? NVN_FORMAT_RGBA8I
                                                                                                            : NVN_FORMAT_RGBA8;
    s->depth_fmt = NVN_FORMAT_DEPTH32F_STENCIL8;
    s->ds.dt = 1;
    s->color.blend_en = 1;
    s->blend[0].src_rgb = NVN_BLEND_FUNC_SRC_ALPHA, s->blend[0].dst_rgb = NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA;
    s->poly.cull = NVN_FACE_BACK;
    uint32_t ins = gfx_shader_inputs(vs, io), off = 0;
    for (int l = 0; l < 16; l++)
        if (ins >> l & 1) {
            va_st a = {io[l].type == GFX_TYPE_UINT ? NVN_FORMAT_RGBA32UI : io[l].type == GFX_TYPE_SINT ? NVN_FORMAT_RGBA32I
                                                                                                      : NVN_FORMAT_RGBA32F,
                       off, 0};
            memcpy(&s->vattr[l], &a, sizeof a);
            off += 16;
        }
    s->vstream[0].stride = off ? off : 16;
}

// Records the dynamic state of a pipeline into a command buffer that is never submitted (validation checks the calls).
static int selftest_record(const VkPipeState *s, VkPipeline pipe) {
    VkCtx *c = vk_ctx();
    VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = c->gfx_family};
    VkCommandPool pool;
    if (vkCreateCommandPool(c->device, &pi, NULL, &pool) != VK_SUCCESS) return 0;
    VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer cb;
    int ok = vkAllocateCommandBuffers(c->device, &ai, &cb) == VK_SUCCESS;
    if (ok) {
        VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                       .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        vkBeginCommandBuffer(cb, &bi);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
        vk_pipeline_set_dynamic(cb, s);
        ok = vkEndCommandBuffer(cb) == VK_SUCCESS;
    }
    vkDestroyCommandPool(c->device, pool, NULL);
    return ok;
}

int vk_pipeline_selftest(void) {
    enum { MAXN = 20000, WANT_PAIRS = 64, WANT_CS = 16 };
    char root[PLAT_PATH_MAX * 2];
    if (!vk_init() || !ensure_init()) { printf("vk_pipeline selftest: FAILED: no Vulkan device\n"); return 1; }
    if (!probe_census_dir(root, sizeof root)) {   // the census folder, build/<version>/shaders
        printf("vk_pipeline selftest: FAILED: no build/v%d/shaders/jobs.json above the exe\n", rt_game_version);
        return 1;
    }
    size_t size;
    char *jobs = census_read(root, &size);
    CensusEntry *e = malloc(sizeof *e * MAXN);
    int n = jobs && e ? census_parse(jobs, e, MAXN) : 0;
    if (!n) { printf("vk_pipeline selftest: FAILED: no census entries\n"); free(jobs), free(e); return 1; }
    qsort(e, (size_t)n, sizeof *e, census_cmp);
    long err0 = vk_validation_errors_total();
    VkPipeStats st0;
    vk_pipeline_stats(&st0);

    // candidate pairs: each fragment shader of a source file with one of the file's vertex shaders (round robin)
    int (*pairs)[2] = malloc(sizeof *pairs * (size_t)n), np = 0, *cs = malloc(sizeof *cs * (size_t)n), nc = 0;
    for (int i = 0; i < n;) {
        int j = i, v0 = -1, nv = 0, nf = 0;
        for (; j < n && e[j].src_len == e[i].src_len && !memcmp(e[j].src, e[i].src, e[i].src_len); j++) {
            if (e[j].stage == GFX_STAGE_VS && v0 < 0) v0 = j;   // sorted by stage within a file: vertex shaders first
            nv += e[j].stage == GFX_STAGE_VS;
            if (e[j].stage == GFX_STAGE_CS) cs[nc++] = j;
        }
        for (int k = i; k < j && nv; k++)
            if (e[k].stage == GFX_STAGE_FS) pairs[np][0] = v0 + nf++ % nv, pairs[np][1] = k, np++;
        i = j;
    }
    int tried = 0, built = 0, cached = 1, shader_fail = 0, ctried = 0, cbuilt = 0, recorded = 0, varying_fix = 0;
    for (int t = 0; t < WANT_PAIRS && t < np; t++) {
        const int *pr = pairs[(long long)t * np / (np < WANT_PAIRS ? np : WANT_PAIRS)];
        tried++;
        const GfxShader *vs = census_shader(root, &e[pr[0]]), *fs = census_shader(root, &e[pr[1]]);
        if (!vs || !fs) { shader_fail++; continue; }
        if (gfx_shader_inputs(fs, NULL) & ~gfx_shader_outputs(vs, NULL)) varying_fix++;
        VkPipeState s;
        selftest_state(&s, vs, fs);
        GfxShaderAnswers fa = {0};
        vk_pipeline_answers(&s, NULL, &fa);
        const VkPipeInfo *info;
        VkPipeline p = vk_pipeline_graphics(&s, &info);
        if (!p) continue;
        built++;
        const VkPipeInfo *again;
        if (vk_pipeline_graphics(&s, &again) != p || again != info) cached = 0;
        if (!recorded) recorded = selftest_record(&s, p);
    }
    for (int t = 0; t < WANT_CS && t < nc; t++) {
        const CensusEntry *c = &e[cs[(long long)t * nc / (nc < WANT_CS ? nc : WANT_CS)]];
        ctried++;
        const GfxShader *s = census_shader(root, c);
        if (!s) { shader_fail++; continue; }
        const VkPipeInfo *info;
        VkPipeline p = vk_pipeline_compute(s, &info);
        if (!p) continue;
        cbuilt++;
        const VkPipeInfo *again;
        if (vk_pipeline_compute(s, &again) != p || again != info) cached = 0;
    }
    vk_pipeline_save();
    VkPipeStats st;
    vk_pipeline_stats(&st);
    long errors = vk_validation_errors_total() - err0;
    printf("vk_pipeline selftest: graphics %d/%d census pairs (%d needed vertex outputs added), compute %d/%d, "
           "%d shaders failed to translate; layouts %llu (%llu push); cache hits %llu; dynamic state recorded %d; "
           "validation errors %ld\n",
           built, tried, varying_fix, cbuilt, ctried, shader_fail, (unsigned long long)(st.layouts - st0.layouts),
           (unsigned long long)(st.push_layouts - st0.push_layouts), (unsigned long long)(st.hits - st0.hits), recorded,
           errors);
    free(pairs), free(cs), free(e), free(jobs);
    int ok = built > 0 && cbuilt > 0 && cached && recorded && errors == 0;
    printf("vk_pipeline selftest: %s\n", ok ? "passed" : "FAILED");
    return ok ? 0 : 1;
}
