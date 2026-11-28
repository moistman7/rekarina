// Standalone verification of runtime/vk_hdr.c's RenoDX SPIR-V splicer (tonemap_splice), FSR unclamping
// (fsr_unclamp), and parameter packing (vk_hdr_support). Writes spliced SPIR-V binaries for external validation with
// spirv-val (--target-env vulkan1.3).
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Minimal stubs so runtime/vk_hdr.c compiles standalone as a unit test (no Vulkan device required).
#define GAME_OOT 1
#define VK_HDR_SPLICE_TEST_ONLY 1
#include "../runtime/vk.h"
#include "../runtime/cfg.h"
#include "../runtime/vk_exec.h"
#include "../runtime/vk_fg.h"

static struct { const char *k, *v; } g_cfg[64];
static int g_ncfg;
static unsigned g_cfg_ver = 1;

static void test_cfg_reset(void) { g_ncfg = 0; g_cfg_ver++; }
static void test_cfg_set(const char *k, const char *v) {
    for (int i = 0; i < g_ncfg; i++)
        if (!strcmp(g_cfg[i].k, k)) { g_cfg[i].v = v; g_cfg_ver++; return; }
    g_cfg[g_ncfg].k = k, g_cfg[g_ncfg].v = v, g_ncfg++, g_cfg_ver++;
}

unsigned cfg_version(void) { return g_cfg_ver; }
const char *cfg_get(const char *name) {
    for (int i = 0; i < g_ncfg; i++) if (!strcmp(g_cfg[i].k, name)) return g_cfg[i].v;
    return NULL;
}
const char *cfg_value(const char *name) {
    const char *v = cfg_get(name);
    if (v) return v;
    if (!strcmp(name, "REKARINA_HDR")) return "1";
    if (!strcmp(name, "REKARINA_HDR_PEAK_NITS")) return "1000";
    if (!strcmp(name, "REKARINA_HDR_GAME_NITS")) return "203";
    if (!strcmp(name, "REKARINA_HDR_UI_NITS")) return "203";
    if (!strcmp(name, "REKARINA_HDR_GAMMA")) return "off";
    if (!strcmp(name, "REKARINA_HDR_UI_EOTF")) return "2.2";
    if (!strcmp(name, "REKARINA_HDR_TONEMAP")) return "psychov35";
    if (!strcmp(name, "REKARINA_SDR_TONEMAP")) return "vanilla";
    if (!strcmp(name, "REKARINA_HDR_MIDGRAY")) return "vanilla";
    if (!strcmp(name, "REKARINA_HDR_MATCH_SLOPE")) return "1";
    if (!strcmp(name, "REKARINA_HDR_TONE_SLOPE")) return "50";
    if (!strcmp(name, "REKARINA_HDR_HUE_CLIP")) return "50";
    if (!strcmp(name, "REKARINA_HDR_EXPOSURE")) return "100";
    if (!strcmp(name, "REKARINA_HDR_HIGHLIGHTS")) return "50";
    if (!strcmp(name, "REKARINA_HDR_SHADOWS")) return "50";
    if (!strcmp(name, "REKARINA_HDR_CONTRAST")) return "50";
    if (!strcmp(name, "REKARINA_HDR_SATURATION")) return "50";
    if (!strcmp(name, "REKARINA_HDR_BLOOM")) return "50";
    if (!strcmp(name, "REKARINA_HDR_LOCAL_EXPOSURE")) return "50";
    if (!strcmp(name, "REKARINA_HDR_ADAPTATION_TARGET")) return "18";
    if (!strcmp(name, "REKARINA_HDR_ADAPTATION_MIN_STOPS")) return "-4";
    if (!strcmp(name, "REKARINA_HDR_ADAPTATION_MAX_STOPS")) return "2";
    return "0";
}
double cfg_double(const char *name) { return strtod(cfg_value(name), NULL); }
int cfg_bool(const char *name) { return strcmp(cfg_value(name), "0") != 0; }
static CfgRenderValues render_values;
const CfgRenderValues *cfg_render_values(void) { return &render_values; }
void rt_log(const char *fmt, ...) { (void)fmt; }
VkCtx *vk_ctx(void) { return NULL; }
void vk_queue_lock(void) {}
void vk_queue_unlock(void) {}
const VkTexImage *vk_texture_by_id(int id, view_t *view, int *has_view) { (void)id; (void)view; (void)has_view; return NULL; }
int vk_color_hdr(float adj[4], const int crop[4]) { (void)adj; (void)crop; return 0; }
void vk_color_not_drawn_hdr(void) {}
PFN_vkSetHdrMetadataEXT vk_fg_hdr_metadata_fn(void) { return NULL; }
void rec_push(PFN_vkCmdPushDescriptorSetKHR fn, VkCommandBuffer cb, VkPipelineBindPoint bp, VkPipelineLayout lo,
              uint32_t set, uint32_t n, const VkWriteDescriptorSet *w) {
    (void)fn; (void)cb; (void)bp; (void)lo; (void)set; (void)n; (void)w;
}

void plat_mutex_lock(PlatMutex *m) { (void)m; }
void plat_mutex_unlock(PlatMutex *m) { (void)m; }

#include "../runtime/vk_hdr.c"

GfxStats gfx_counters;
#ifdef _WIN32
void renderer_mcp_tonemap(unsigned width, unsigned height) { (void)width; (void)height; }
#endif

static int fails;
#define CHECK(cond, msg) do { if (!(cond)) { fails++; fprintf(stderr, "FAIL: %s\n", msg); } } while (0)

static void check_output_overwrite(void) {
    const VkGuestPq original = {.row = {{203, 0, 0, 0}}, .valid = 1};
    VkGuestPq result = {0};
    const VkImage image = (VkImage)(uintptr_t)1;
    gfx_counters.frames = 1;
    vk_hdr_record_output(21, image, 0x1000, GFX_OUTPUT_PQ, &original);
    vk_hdr_record_output(22, VK_NULL_HANDLE, 0x2000, GFX_OUTPUT_PQ, &original);
    vk_hdr_output_overwrite(image, 0);
    CHECK(vk_hdr_written_output(21, 2, &result) == GFX_OUTPUT_UNSPECIFIED &&
          vk_hdr_written_output(22, 2, &result) == GFX_OUTPUT_PQ && result.valid && vk_hdr_output_pending(),
          "overwriting one image invalidates its PQ proof without clearing another image's proof");
    vk_hdr_output_overwrite(VK_NULL_HANDLE, 0x2000);
    CHECK(!vk_hdr_output_pending() && vk_hdr_written_output(22, 2, &result) == GFX_OUTPUT_UNSPECIFIED,
          "an alias write invalidates the last PQ proof at the same guest address");
}

static void emit_str_op(SpvBuf *b, uint32_t op, uint32_t id, const char *s) {
    size_t slen = strlen(s) + 1;
    uint32_t sw = (uint32_t)((slen + 3) / 4);
    uint32_t hdr = ((2u + sw) << 16) | op;
    spv_emit(b, &hdr, 1);
    spv_emit(b, &id, 1);
    uint32_t tmp[16] = {0};
    memcpy(tmp, s, slen);
    spv_emit(b, tmp, sw);
}

static uint32_t fbits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

// Build a valid Vulkan 1.3 SPIR-V fragment shader matching OoT's translated ACES tonemap shape.
// shape: 1 = scene*gain (scalar stores), 2 = (scene*gain)*local_exp with interleaved scalar stores & func vars,
//        3 = fma(scene, gain, color), 4 = fma(scene*gain, local_exp, color), 5 = fallback vector store,
//        6 = translator FP16/NaN-sanitized output store (OpLoad site_var -> OpIsNan/FMax/FMin/Bitcast/Select -> OpStore),
//        7 = non-ACES shader, 8 = HDR expansion with scene bloom and direct gain,
//        9 = pause scene (raw bloom sample and sanitized ACES output), 10 = same with mismatched bloom channels.
static uint32_t *make_aces_spv(int shape, uint64_t *words_out) {
    const int pause = shape == 9 || shape == 10;
    SpvBuf b = {0};
    uint32_t hdr[5] = {SPV_MAGIC, 0x00010500u, 0, 200, 0};
    spv_emit(&b, hdr, 5);
    SPV_INS(&b, 17 /* OpCapability */, 1 /* Shader */);
    emit_str_op(&b, SPV_OP_EXT_INST_IMPORT, 1, "GLSL.std.450");
    SPV_INS(&b, 14 /* OpMemoryModel */, 0, 1);
    // OpEntryPoint Fragment %4 "main" %30 %31 %34 (support_buffer %28 omitted intentionally to verify splice adds it)
    {
        uint32_t ep[9] = {((8u + pause) << 16) | SPV_OP_ENTRY_POINT, SPV_EXEC_FRAGMENT, 4, 0x6e69616du /* "main" */, 0, 30, 31, 34, 39};
        spv_emit(&b, ep, 8 + pause);
    }
    SPV_INS(&b, 16 /* OpExecutionMode */, 4, 7 /* OriginUpperLeft */);
    emit_str_op(&b, SPV_OP_NAME, 28, "support_buffer");
    // Decorations
    SPV_INS(&b, SPV_OP_DECORATE, 26, 2 /* Block */);
    for (uint32_t m = 0; m < 11; m++)
        SPV_INS(&b, SPV_OP_MEMBER_DECORATE, 26, m, SPV_DEC_OFFSET, m == 10 ? 1424u : m * 16u);
    SPV_INS(&b, SPV_OP_DECORATE, 28, SPV_DEC_SET, 0);
    SPV_INS(&b, SPV_OP_DECORATE, 28, SPV_DEC_BINDING, 0);
    SPV_INS(&b, SPV_OP_DECORATE, 32, 2 /* Block */);
    SPV_INS(&b, SPV_OP_MEMBER_DECORATE, 32, 0, SPV_DEC_OFFSET, 0);
    SPV_INS(&b, SPV_OP_DECORATE, 34, SPV_DEC_SET, 0);
    SPV_INS(&b, SPV_OP_DECORATE, 34, SPV_DEC_BINDING, 1);
    SPV_INS(&b, SPV_OP_DECORATE, 30, 30 /* Location */, 0);
    SPV_INS(&b, SPV_OP_DECORATE, 31, 30 /* Location */, 0);
    if (pause) {
        SPV_INS(&b, SPV_OP_DECORATE, 39, SPV_DEC_SET, 0);
        SPV_INS(&b, SPV_OP_DECORATE, 39, SPV_DEC_BINDING, 2);
    }

    // Types
    SPV_INS(&b, SPV_OP_TYPE_VOID, 2);
    SPV_INS(&b, SPV_OP_TYPE_FUNCTION, 3, 2);
    SPV_INS(&b, SPV_OP_TYPE_BOOL, 5);
    SPV_INS(&b, SPV_OP_TYPE_FLOAT, 6, 32);
    SPV_INS(&b, SPV_OP_TYPE_VECTOR, 7, 6, 4);      // vec4
    SPV_INS(&b, SPV_OP_TYPE_INT, 8, 32, 1);        // i32
    SPV_INS(&b, SPV_OP_TYPE_INT, 14, 32, 0);       // u32
    SPV_INS(&b, SPV_OP_TYPE_POINTER, 9, SPV_SC_INPUT, 7);
    SPV_INS(&b, SPV_OP_TYPE_POINTER, 10, SPV_SC_OUTPUT, 7);
    SPV_INS(&b, SPV_OP_TYPE_POINTER, 11, SPV_SC_OUTPUT, 6);
    SPV_INS(&b, SPV_OP_TYPE_POINTER, 12, SPV_SC_UNIFORM, 6);
    SPV_INS(&b, SPV_OP_TYPE_POINTER, 13, SPV_SC_FUNCTION, 6);
    // SupportBuffer struct %26: 11 vec4 members (matching translator layout up to offset 1424)
    SPV_INS(&b, SPV_OP_TYPE_STRUCT, 26, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7);
    SPV_INS(&b, SPV_OP_TYPE_POINTER, 27, SPV_SC_UNIFORM, 26);
    SPV_INS(&b, SPV_OP_VARIABLE, 27, 28, SPV_SC_UNIFORM);
    // Exposure UBO struct %32 { vec4 u; }
    SPV_INS(&b, SPV_OP_TYPE_STRUCT, 32, 7);
    SPV_INS(&b, SPV_OP_TYPE_POINTER, 33, SPV_SC_UNIFORM, 32);
    SPV_INS(&b, SPV_OP_VARIABLE, 33, 34, SPV_SC_UNIFORM);
    SPV_INS(&b, SPV_OP_VARIABLE, 9, 30, SPV_SC_INPUT);
    SPV_INS(&b, SPV_OP_VARIABLE, 10, 31, SPV_SC_OUTPUT);
    if (pause) {
        SPV_INS(&b, SPV_OP_TYPE_VECTOR, 35, 6, 2);
        SPV_INS(&b, 25 /* OpTypeImage */, 36, 6, 1 /* 2D */, 0, 0, 0, 1, 0);
        SPV_INS(&b, 27 /* OpTypeSampledImage */, 37, 36);
        SPV_INS(&b, SPV_OP_TYPE_POINTER, 38, 0 /* UniformConstant */, 37);
        SPV_INS(&b, SPV_OP_VARIABLE, 38, 39, 0);
    }

    // Constants
    SPV_INS(&b, SPV_OP_CONSTANT, 8, 40, 0);
    SPV_INS(&b, SPV_OP_CONSTANT, 8, 41, 1);
    SPV_INS(&b, SPV_OP_CONSTANT, 8, 42, 2);
    SPV_INS(&b, SPV_OP_CONSTANT, 8, 43, 3);
    SPV_INS(&b, SPV_OP_CONSTANT, 6, 50, fbits(0.0f));
    SPV_INS(&b, SPV_OP_CONSTANT, 6, 51, fbits(1.0f));
    SPV_INS(&b, SPV_OP_CONSTANT, 6, 52, fbits(shape == 7 ? 1.5f : 2.51f));
    SPV_INS(&b, SPV_OP_CONSTANT, 6, 53, fbits(0.03f));
    SPV_INS(&b, SPV_OP_CONSTANT, 6, 54, fbits(2.43f));
    SPV_INS(&b, SPV_OP_CONSTANT, 6, 55, fbits(0.59f));
    SPV_INS(&b, SPV_OP_CONSTANT, 6, 56, fbits(0.14f));
    SPV_INS(&b, SPV_OP_CONSTANT, 6, 57, fbits(0.25f));   // add_color
    SPV_INS(&b, SPV_OP_CONSTANT, 14, 58, 0u);
    SPV_INS(&b, SPV_OP_CONSTANT, 6, 59, fbits(65504.0f));
    SPV_INS(&b, SPV_OP_CONSTANT, 6, 61, fbits(10000.0f));
    SPV_INS(&b, SPV_OP_CONSTANT, 6, 62, fbits(-1.0f));

    // main()
    SPV_INS(&b, SPV_OP_FUNCTION, 2, 4, 0, 3);
    SPV_INS(&b, SPV_OP_LABEL, 60);
    uint32_t next = 70;
    uint32_t fvar[3], cvar[3];
    for (int c = 0; c < 3; c++) { fvar[c] = next++; SPV_INS(&b, SPV_OP_VARIABLE, 13, fvar[c], SPV_SC_FUNCTION); }
    for (int c = 0; c < 3; c++) { cvar[c] = next++; SPV_INS(&b, SPV_OP_VARIABLE, 13, cvar[c], SPV_SC_FUNCTION); }
    uint32_t in_v = next++;
    SPV_INS(&b, SPV_OP_LOAD, 7, in_v, 30);
    uint32_t local_stops = next++, local_exp = next++;
    SPV_INS(&b, SPV_OP_COMPOSITE_EXTRACT, 6, local_stops, in_v, 3);
    SPV_INS(&b, SPV_OP_EXT_INST, 6, local_exp, 1, GLSL_EXP2, local_stops);

    // Uniform gain loaded from UBO %34 member 0, component 2
    uint32_t p_gain = next++, gain = next++;
    SPV_INS(&b, SPV_OP_ACCESS_CHAIN, 12, p_gain, 34, 40, 42);
    SPV_INS(&b, SPV_OP_LOAD, 6, gain, p_gain);

    uint32_t sample = 0;
    if (pause) {
        uint32_t coord = next++, sampler = next++;
        sample = next++;
        SPV_INS(&b, SPV_OP_VECTOR_SHUFFLE, 35, coord, in_v, in_v, 0, 1);
        SPV_INS(&b, SPV_OP_LOAD, 37, sampler, 39);
        SPV_INS(&b, SPV_OP_IMAGE_SAMPLE_IMPLICIT_LOD, 7, sample, sampler, coord);
    }

    if (shape == 11) {
        for (int c = 0; c < 3; c++) {
            uint32_t rgb = next++;
            SPV_INS(&b, SPV_OP_COMPOSITE_EXTRACT, 6, rgb, in_v, (uint32_t)c);
            SPV_INS(&b, SPV_OP_STORE, fvar[c], rgb);
        }
    }
    uint32_t cl[3];
    for (int c = 0; c < 3; c++) {
        uint32_t sc = next++;
        SPV_INS(&b, SPV_OP_COMPOSITE_EXTRACT, 6, sc, in_v, (uint32_t)c);
        if (shape == 8 || pause) {
            uint32_t low = next++, base = next++, mag = next++, log_mag = next++, power = next++, bloom = next++;
            SPV_INS(&b, SPV_OP_EXT_INST, 6, low, 1, GLSL_FMAX, sc, 50);
            SPV_INS(&b, SPV_OP_EXT_INST, 6, base, 1, GLSL_FMIN, low, 61);
            if (pause) {
                SPV_INS(&b, SPV_OP_COMPOSITE_EXTRACT, 6, bloom, sample, (uint32_t)(shape == 10 ? 0 : c));
            } else {
                SPV_INS(&b, SPV_OP_EXT_INST, 6, mag, 1, GLSL_FABS, sc);
                SPV_INS(&b, SPV_OP_EXT_INST, 6, log_mag, 1, GLSL_LOG2, mag);
                SPV_INS(&b, SPV_OP_FMUL, 6, power, log_mag, gain);
                SPV_INS(&b, SPV_OP_EXT_INST, 6, bloom, 1, GLSL_EXP2, power);
            }
            sc = next++;
            SPV_INS(&b, SPV_OP_FADD, 6, sc, base, bloom);
        }
        uint32_t x = sc;
        if (shape == 11) {
            x = next++;
            SPV_INS(&b, SPV_OP_LOAD, 6, x, fvar[c]);
        } else if (shape == 1 || shape == 6 || shape == 8 || pause) {
            x = next++;
            SPV_INS(&b, SPV_OP_FMUL, 6, x, sc, gain);
        } else if (shape == 2) {
            uint32_t sg = next++;
            SPV_INS(&b, SPV_OP_FMUL, 6, sg, sc, gain);
            SPV_INS(&b, SPV_OP_STORE, fvar[c], sg);
            uint32_t sg_ld = next++;
            SPV_INS(&b, SPV_OP_LOAD, 6, sg_ld, fvar[c]);
            x = next++;
            SPV_INS(&b, SPV_OP_FMUL, 6, x, sg_ld, local_exp);
        } else if (shape == 3) {
            x = next++;
            SPV_INS(&b, SPV_OP_EXT_INST, 6, x, 1, GLSL_FMA, sc, gain, 57);
        } else if (shape == 4) {
            uint32_t sg = next++;
            SPV_INS(&b, SPV_OP_FMUL, 6, sg, sc, gain);
            x = next++;
            SPV_INS(&b, SPV_OP_EXT_INST, 6, x, 1, GLSL_FMA, sg, local_exp, 57);
        }
        // ACES: clamp((x * fma(x, 2.51, 0.03)) * (1.0 / fma(x, fma(x, 2.43, 0.59), 0.14)), 0, 1)
        uint32_t n1 = next++, num = next++, d1 = next++, den = next++, rcp = next++, rat = next++;
        cl[c] = next++;
        SPV_INS(&b, SPV_OP_EXT_INST, 6, n1, 1, GLSL_FMA, x, 52, 53);
        SPV_INS(&b, SPV_OP_FMUL, 6, num, x, n1);
        SPV_INS(&b, SPV_OP_EXT_INST, 6, d1, 1, GLSL_FMA, x, 54, 55);
        SPV_INS(&b, SPV_OP_EXT_INST, 6, den, 1, GLSL_FMA, x, d1, 56);
        SPV_INS(&b, SPV_OP_FDIV, 6, rcp, 51, den);
        SPV_INS(&b, SPV_OP_FMUL, 6, rat, num, rcp);
        SPV_INS(&b, SPV_OP_EXT_INST, 6, cl[c], 1, shape == 2 ? GLSL_NCLAMP : GLSL_FCLAMP, rat, 50, 51);
        if (shape == 8) {
            uint32_t inner = next++, outer = next++, expanded = next++, bounded = next++;
            SPV_INS(&b, SPV_OP_EXT_INST, 6, inner, 1, GLSL_FMA, cl[c], gain, 62);
            SPV_INS(&b, SPV_OP_EXT_INST, 6, outer, 1, GLSL_FMA, inner, gain, 51);
            SPV_INS(&b, SPV_OP_FMUL, 6, expanded, outer, gain);
            SPV_INS(&b, SPV_OP_EXT_INST, 6, bounded, 1, GLSL_FMAX, expanded, 50);
            cl[c] = next++;
            SPV_INS(&b, SPV_OP_EXT_INST, 6, cl[c], 1, GLSL_FMIN, bounded, gain);
        }
        if (shape == 6 || pause) {
            SPV_INS(&b, SPV_OP_STORE, cvar[c], cl[c]);
        } else if (shape != 5) {
            uint32_t po = next++;
            SPV_INS(&b, SPV_OP_ACCESS_CHAIN, 11, po, 31, 40u + (uint32_t)c);
            SPV_INS(&b, SPV_OP_STORE, po, cl[c]);
        }
    }
    if (shape == 6 || pause) {
        // Translator FP16/NaN sanitizer: AccessChain -> Load cvar -> IsNan/FMax/FMin/Bitcast/Select/Bitcast -> Store
        for (int c = 0; c < 3; c++) {
            uint32_t po = next++, ld = next++, nan = next++, mx = next++, mn = next++, bu = next++, sel = next++, bf = next++;
            SPV_INS(&b, SPV_OP_ACCESS_CHAIN, 11, po, 31, 40u + (uint32_t)c);
            SPV_INS(&b, SPV_OP_LOAD, 6, ld, cvar[c]);
            SPV_INS(&b, 156 /* OpIsNan */, 5, nan, ld);
            SPV_INS(&b, SPV_OP_EXT_INST, 6, mx, 1, GLSL_FMAX, ld, 50);
            SPV_INS(&b, SPV_OP_EXT_INST, 6, mn, 1, 37 /* FMin */, mx, 59);
            SPV_INS(&b, 124 /* OpBitcast */, 14, bu, mn);
            SPV_INS(&b, 169 /* OpSelect */, 14, sel, nan, 58, bu);
            SPV_INS(&b, 124 /* OpBitcast */, 6, bf, sel);
            SPV_INS(&b, SPV_OP_STORE, po, bf);
        }
        uint32_t pa = next++;
        SPV_INS(&b, SPV_OP_ACCESS_CHAIN, 11, pa, 31, 43);
        SPV_INS(&b, SPV_OP_STORE, pa, 51);
    } else if (shape == 5) {
        uint32_t v = next++;
        SPV_INS(&b, SPV_OP_COMPOSITE_CONSTRUCT, 7, v, cl[0], cl[1], cl[2], 51);
        SPV_INS(&b, SPV_OP_STORE, 31, v);
    } else {
        uint32_t pa = next++;
        SPV_INS(&b, SPV_OP_ACCESS_CHAIN, 11, pa, 31, 43);
        SPV_INS(&b, SPV_OP_STORE, pa, 51);
    }
    SPV_INS(&b, 253 /* OpReturn */);
    SPV_INS(&b, SPV_OP_FUNCTION_END);
    b.w[3] = next;
    *words_out = b.n;
    return b.w;
}

static void write_spv(const char *path, const uint32_t *w, uint64_t n) {
    FILE *f = fopen(path, "wb");
    CHECK(f != NULL, path);
    if (f) { fwrite(w, 4, (size_t)n, f); fclose(f); }
}

int main(int argc, char **argv) {
    // Optional external-file mode: hdr_splice_check --file <in.spv> <out.spv> prints the result (2: exposure traced) and
    // each optional term traced in the shader.
    int mode = argc == 4 && !strcmp(argv[1], "--native-file") ? SPLICE_HDR_SCENE :
               argc == 4 && !strcmp(argv[1], "--grade-file") ? SPLICE_VANILLA_GRADE :
               argc == 4 && !strcmp(argv[1], "--effects-file") ? SPLICE_HDR_EFFECTS : SPLICE_ACES;
    int aa_stage = argc == 4 && !strcmp(argv[1], "--aa-prep") ? 1 :
                   argc == 4 && !strcmp(argv[1], "--aa-blend") ? 2 :
                   argc == 4 && !strcmp(argv[1], "--aa-store") ? 3 :
                   argc == 4 && !strcmp(argv[1], "--aa-mask") ? 4 : 0;
    if (argc == 4 && (mode != SPLICE_ACES || aa_stage || !strcmp(argv[1], "--file"))) {
        FILE *f = fopen(argv[2], "rb");
        if (!f) return 2;
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        fseek(f, 0, SEEK_SET);
        uint32_t *buf = malloc((size_t)sz);
        if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(buf); return 2; }
        fclose(f);
        uint64_t words = (uint64_t)sz / 4;
        SpliceTrace trace = {0};
        int r = aa_stage == 4 ? vk_aa_mask_splice(&buf, &words, 2) :
                aa_stage ? vk_hdr_aa_splice(&buf, &words, aa_stage == 1 ? 1 : 2,
                                            aa_stage == 1 ? 5 : aa_stage == 2 ? 6 : 18, aa_stage == 3) :
                tonemap_splice(&buf, &words, mode == SPLICE_ACES || mode == SPLICE_HDR_SCENE, mode, &trace);
        if (r > 0) write_spv(argv[3], buf, words);
        free(buf);
        printf("%d%s%s%s\n", r, r > 0 && trace.bloom ? " bloom" : "",
               r > 0 && trace.local_exposure ? " local-exposure" : "",
               r > 0 && trace.meter_binding ? " eye-adaptation" : "");
        return r < 0 || (aa_stage == 4 && !r) ? 1 : 0;
    }
    const char *out_dir = argc > 1 ? argv[1] : NULL;
    check_output_overwrite();
    VkGuestPq a = {.row = {{203, 0, 0, 0}}, .valid = 1};
    VkGuestPq b = {.row = {{300, 0, 0, 0}}, .valid = 1};
    VkGuestPq queued = {0}, current = {0};
    gfx_counters.frames = 10;
    vk_hdr_record_output(17, VK_NULL_HANDLE, 0x1000, GFX_OUTPUT_PQ, &a);
    CHECK(vk_hdr_written_output(18, 11, &current) == GFX_OUTPUT_UNSPECIFIED && !current.valid,
          "PQ transform is not inherited by another image");
    CHECK(vk_hdr_written_output(17, 11, &queued) == GFX_OUTPUT_PQ && queued.valid,
          "queued PQ frame has its writer's transform");
    gfx_counters.frames = 11;
    CHECK(vk_hdr_written_output(17, 12, &current) == GFX_OUTPUT_UNSPECIFIED && !current.valid,
          "PQ transform is not inherited by another frame");
    vk_hdr_record_output(17, VK_NULL_HANDLE, 0x1000, GFX_OUTPUT_PQ, &b);
    CHECK(vk_hdr_written_output(17, 12, &current) == GFX_OUTPUT_PQ && current.row[0][0] == 300 && queued.row[0][0] == 203,
          "a new writer cannot change a transform copied to the present queue");
    render_values.renodx_flags = CFG_HDR_RENODX_SCENE;
    CHECK(!vk_hdr_renodx_scene(), "RenoDX scene mapping waits for the HDR swapchain");
    vk_hdr_set_active(1);
    test_cfg_set("REKARINA_HDR_TONEMAP", "vanilla");
    render_values.renodx_flags = CFG_HDR_RENODX_EFFECTS;
    CHECK(vk_hdr_renodx_effects() && !vk_hdr_renodx_scene(), "Vanilla effects do not enable signed scene AA");
    test_cfg_set("REKARINA_HDR_TONEMAP", "psychov35");
    render_values.renodx_flags = CFG_HDR_RENODX_SCENE;
    CHECK(vk_hdr_renodx_scene(), "PsychoV35 enables signed scene mapping with the HDR swapchain active");
    CHECK(vk_hdr_wanted() && !vk_hdr_native(), "HDR PsychoV35 selects the signed compositor");
    test_cfg_set("REKARINA_HDR_TONEMAP", "vanilla");
    CHECK(vk_hdr_wanted() && vk_hdr_native(), "changing only the HDR tonemapper selects the native compositor");
    test_cfg_set("REKARINA_HDR", "0");
    CHECK(!vk_hdr_wanted() && !vk_hdr_native(), "the HDR tonemapper does not turn HDR on");
    test_cfg_set("REKARINA_HDR", "1");
    render_values.renodx_flags = 0;
    // 1. Test tonemap_splice across all 6 ACES shapes (including NaN-sanitized output) + non-ACES shader
    for (int shape = 1; shape <= 7; shape++) {
        uint64_t n = 0;
        uint32_t *spv = make_aces_spv(shape, &n);
        uint64_t orig_n = n;
        SpliceTrace trace;
        int r = tonemap_splice(&spv, &n, 1, SPLICE_ACES, &trace);
        if (shape <= 4 || shape == 6) CHECK(r == 2 && n > orig_n, "exposure-traced ACES shape spliced");
        else if (shape == 5) CHECK(r == 1 && n > orig_n, "fallback vector-store ACES shape spliced");
        else CHECK(r == 0 && n == orig_n, "non-ACES shader untouched");
        if (r > 0) CHECK(trace.local_exposure == (shape == 2 || shape == 4), "trace both local exposure shapes");
        if (r > 0) {
            int has_scene = shape != 5;
            CHECK(trace.meter_binding == (has_scene ? 2 : 0), "meter needs traced scene and unused binding");
        }
        if (out_dir && shape <= 6) {
            char p[256];
            snprintf(p, sizeof p, "%s/spliced_%d.spv", out_dir, shape);
            write_spv(p, spv, n);
        }
        free(spv);
    }
    uint64_t no_meter_words;
    uint32_t *no_meter = make_aces_spv(2, &no_meter_words);
    SpliceTrace no_meter_trace;
    CHECK(tonemap_splice(&no_meter, &no_meter_words, 0, SPLICE_ACES, &no_meter_trace) == 2 && !no_meter_trace.meter_binding,
          "unsupported fragment stores keep HDR and local exposure");
    if (out_dir) {
        char path[256];
        snprintf(path, sizeof path, "%s/spliced_no_meter.spv", out_dir);
        write_spv(path, no_meter, no_meter_words);
    }
    free(no_meter);

    uint64_t grade_words;
    uint32_t *grade = make_aces_spv(11, &grade_words);
    SpliceTrace grade_trace;
    CHECK(tonemap_splice(&grade, &grade_words, 0, SPLICE_VANILLA_GRADE, &grade_trace) > 0 && !grade_trace.meter_binding,
          "Vanilla grading can replace the ACES fit without installing a second tonemapper or meter");
    if (out_dir) {
        char path[256];
        snprintf(path, sizeof path, "%s/spliced_vanilla_grade.spv", out_dir);
        write_spv(path, grade, grade_words);
    }
    free(grade);

    uint64_t direct_words;
    uint32_t *direct = make_aces_spv(8, &direct_words);
    SpliceTrace direct_trace;
    CHECK(tonemap_splice(&direct, &direct_words, 1, SPLICE_HDR_SCENE, &direct_trace) == 2 &&
          direct_trace.bloom && !direct_trace.local_exposure && direct_trace.meter_binding,
          "HDR scene with direct gain keeps bloom and eye adaptation without a local-exposure term");
    if (out_dir) {
        char path[256];
        snprintf(path, sizeof path, "%s/spliced_hdr_direct.spv", out_dir);
        write_spv(path, direct, direct_words);
    }
    free(direct);

    for (int shape = 9; shape <= 10; shape++) {
        uint64_t words;
        uint32_t *spv = make_aces_spv(shape, &words);
        SpliceTrace trace;
        int result = tonemap_splice(&spv, &words, 1, SPLICE_HDR_SCENE, &trace);
        if (shape == 9) {
            CHECK(result == 2 && trace.bloom && !trace.local_exposure && !trace.meter_binding,
                  "pause scene applies the HDR mapper and bloom without metering gameplay exposure");
            if (out_dir && result > 0) {
                char path[256];
                snprintf(path, sizeof path, "%s/spliced_hdr_pause.spv", out_dir);
                write_spv(path, spv, words);
            }
        } else CHECK(result == 0, "plain ACES output with unrelated bloom channels is not replaced");
        free(spv);
    }

    // 2. Test vk_hdr_support parameter packing and SDR passthrough
    test_cfg_reset();
    float sup[4 * VK_HDR_PARAMS];
    vk_hdr_set_active(0);
    vk_hdr_support(sup);
    CHECK(sup[3] == -1.0f, "SDR swapchain sets p0.w = -1 (vanilla passthrough)");
    CHECK(sup[20] == 1.0f, "SDR swapchain keeps the game's bloom (p5.x = 1)");
    test_cfg_set("REKARINA_HDR", "0");
    CHECK(!vk_hdr_sdr_scene() && !vk_hdr_scene_grade(), "SDR defaults to the unmodified Vanilla shader");
    test_cfg_set("REKARINA_HDR_EXPOSURE", "75");
    CHECK(vk_hdr_scene_grade() && !vk_hdr_sdr_scene(), "nonneutral grade is supported with the Vanilla mapper");
    test_cfg_set("REKARINA_SDR_TONEMAP", "psychov35");
    test_cfg_set("REKARINA_HDR_GAMMA", "2.4");
    vk_hdr_support(sup);
    CHECK(vk_hdr_sdr_scene() && sup[3] == 8.0f && sup[0] == 203.0f && sup[1] == 203.0f && sup[4] == 0.75f,
          "SDR PsychoV35 uses equal peak and diffuse white and shared exposure");
    CHECK(sup[14] == 0.0f, "SDR PsychoV35 ignores the saved HDR EOTF emulation");
    test_cfg_set("REKARINA_HDR_BLOOM", "25");
    vk_hdr_support(sup);
    CHECK(vk_hdr_renodx_effects() && sup[20] == 0.5f, "SDR PsychoV35 receives the shared bloom setting");
    test_cfg_set("REKARINA_SDR_TONEMAP", "vanilla");
    vk_hdr_support(sup);
    CHECK(!vk_hdr_sdr_scene() && vk_hdr_renodx_effects() && sup[20] == 0.5f,
          "SDR Vanilla can adjust bloom without switching tonemappers");
    test_cfg_set("REKARINA_HDR", "1");
    test_cfg_set("REKARINA_HDR_TONEMAP", "vanilla");
    vk_hdr_set_active(1);
    vk_hdr_support(sup);
    CHECK(!vk_hdr_sdr_scene() && sup[3] == 0.0f && sup[4] == 0.75f,
          "HDR remembers a separate mapper and consumes the same grade");
    test_cfg_reset();
    vk_hdr_set_active(1);
    vk_hdr_support(sup);
    CHECK(sup[0] == 1000.0f && sup[1] == 203.0f && sup[2] == 203.0f && sup[3] == 9.0f, "PsychoV35 defaults");
    test_cfg_set("REKARINA_HDR_UI_NITS", "300");
    vk_hdr_support(sup);
    CHECK(sup[2] == sup[1], "menu brightness does not rescale the guest scene");
    CHECK(sup[4] == 1.0f && sup[5] == 1.0f && sup[6] == 1.0f && sup[7] == 1.0f && sup[8] == 1.0f, "neutral grading");
    test_cfg_set("REKARINA_HDR_SATURATION", "0");
    vk_hdr_support(sup);
    CHECK(sup[8] == 0.0f, "zero saturation reaches the mapper as zero purity");
    test_cfg_set("REKARINA_HDR_SATURATION", "50");
    CHECK(sup[14] == 0.0f && sup[15] == 1.0f && sup[16] == 0.5f && sup[17] == 1.0f && sup[18] == 1.0f, "PsychoV35 flags");
    CHECK(nits().ui_eotf == 1.0f, "UI EOTF defaults to 2.2 while Render EOTF is off");
    test_cfg_set("REKARINA_HDR_UI_EOTF", "2.4");
    vk_hdr_support(sup);
    CHECK(nits().ui_eotf == 2.0f && sup[14] == 0.0f, "UI EOTF does not change the render shader");
    test_cfg_set("REKARINA_HDR_UI_EOTF", "off");
    CHECK(sup[20] == 1.0f, "bloom 50 is the game's (p5.x = 1)");
    CHECK(sup[21] == 0.0f, "PsychoV35 local exposure defaults to off");
    test_cfg_set("REKARINA_HDR_BLOOM", "25");
    test_cfg_set("REKARINA_HDR_LOCAL_EXPOSURE", "0");
    vk_hdr_support(sup);
    CHECK(sup[20] == 0.5f, "bloom 25 halves the game's (p5.x = 0.5)");
    CHECK(sup[21] == 0.0f, "PsychoV35 local exposure off");
    test_cfg_set("REKARINA_HDR_LOCAL_EXPOSURE", "100");
    vk_hdr_support(sup);
    CHECK(sup[21] == 0.0f, "PsychoV35 ignores stored Vanilla local exposure strength");
    test_cfg_set("REKARINA_HDR_LOCAL_EXPOSURE", "0");
    test_cfg_set("REKARINA_HDR_MIDGRAY", "adaptation");
    vk_hdr_support(sup);
    CHECK(sup[17] == 2.0f, "PsychoV35 activates the eye adaptation meter");
    Nits settings = nits();
    CHECK(settings.adaptation_target == 0.18f && settings.adaptation_min == 0.0625f && settings.adaptation_max == 4.0f,
          "adaptation target and stop limits");

    test_cfg_set("REKARINA_HDR_GAMMA", "2.4");
    test_cfg_set("REKARINA_HDR_EXPOSURE", "120");
    vk_hdr_support(sup);
    CHECK(sup[3] == 9.0f && sup[14] == 2.0f && fabsf(sup[4] - 1.2f) < 1e-5f, "PsychoV35 + gamma 2.4 + exposure");
    CHECK(nits().ui_eotf == 0.0f, "Render EOTF does not change the UI decode");

    test_cfg_set("REKARINA_HDR_TONEMAP", "vanilla");
    test_cfg_set("REKARINA_HDR_GAMMA", "off");
    vk_hdr_support(sup);
    CHECK(sup[17] != 2.0f && sup[21] == 0.0f, "Vanilla defaults to no local exposure or eye adaptation");
    test_cfg_set("REKARINA_HDR_LOCAL_EXPOSURE", "100");
    vk_hdr_support(sup);
    CHECK(sup[3] == 0.0f && sup[14] == HDR_GAMMA_22, "Vanilla HDR keeps the native EOTF");
    CHECK(sup[21] == 2.0f, "Vanilla local exposure doubles the correction in stops");
    vk_hdr_set_active(0);
    vk_hdr_support(sup);
    CHECK(sup[21] == 1.0f, "SDR swapchain keeps the game's local exposure");

    if (fails) {
        fprintf(stderr, "hdr_splice_check: %d failures\n", fails);
        return 1;
    }
    printf("hdr_splice_check: passed (shader splices and parameter packing)\n");
    return 0;
}
