// HDR output (REKARINA_HDR=1 or native): an HDR10 swapchain (Rec. 2020 primaries, ST 2084 PQ) and a game picture tone
// mapped to the display's range (shaders/hdr_tonemap.frag on OoT, shaders/hdr_tonemap_totk.frag on TotK).
//
// On OoT (GAME_OOT), the guest selects SDR or HDR10 and writes the visible output. Native HDR leaves scene mapping
// untouched; RenoDX HDR uses a separate in-memory scene variant for the checked HDR compose.
// - Tone mappers (REKARINA_HDR_TONEMAP): psychov35 (observer-response cone mapper, default) and vanilla (guest ACES).
// - Present pass (vk_hdr_encode, shaders/hdr_encode.comp): optional colour profile (vk_color.c), RenoDX gamma correction
//   (REKARINA_HDR_GAMMA: 2.2 / 2.4 / off), Rec. 709 -> Rec. 2020, mid-grey gamut compression, UI nits scaling,
//   ratio-preserving peak clamp, and PQ encode.
//
// RenoDX algorithms are ported from MIT-licensed https://github.com/clshortfuse/renodx (Copyright Carlos Lopez Jr. /
// Carlos Lopez); this host integration and SPIR-V splicer are written for Rekarina.
#include "vk.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "hdr_encode_spv.h"
#include "hdr_fullscreen_spv.h"
#ifdef GAME_OOT
#include "hdr_tonemap_spv.h"
#else
#include "hdr_tonemap_totk_spv.h"
#endif
#include "hdr_ui_spv.h"
#include "renderer_mcp.h"
#include "host.h"
#include "nvn_api.h"
#include "vk_exec.h"
#include "vk_fg.h"
#include "platform/platform.h"

#define SLOTS 8

// ---- settings & lifecycle

#ifdef GAME_OOT
static int g_disabled;

int vk_hdr_armed(void) { return !g_disabled; }

static int hdr_mode(void) {
    static _Thread_local unsigned seen = ~0u;
    static _Thread_local int mode;
    if (!vk_hdr_armed()) return 0;
    unsigned v = cfg_version();
    if (seen != v) {
        const char *setting = cfg_value("REKARINA_HDR");
        mode = !strcmp(setting, "1") ? (!strcmp(cfg_value("REKARINA_HDR_TONEMAP"), "vanilla") ? 2 : 1) : 0;
        seen = v;
    }
    return mode;
}

int vk_hdr_native(void) { return hdr_mode() == 2; }
int vk_hdr_wanted(void) { return hdr_mode() != 0; }

static unsigned renodx_flags(void) {
    const CfgRenderValues *settings = cfg_render_values();
    return settings && vk_hdr_active() && vk_hdr_armed() ? settings->renodx_flags : 0;
}

int vk_hdr_renodx_scene(void) {
    return (renodx_flags() & CFG_HDR_RENODX_SCENE) != 0;
}

int vk_hdr_renodx_effects(void) {
    return (renodx_flags() & CFG_HDR_RENODX_EFFECTS) != 0 || cfg_double("REKARINA_HDR_BLOOM") != 50.0;
}

int vk_hdr_native_picture(VkFormat fmt) {
    return vk_hdr_wanted() &&
           (fmt == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || fmt == VK_FORMAT_A2R10G10B10_UNORM_PACK32);
}

int vk_hdr_adaptation_supported(void) {
    const VkCtx *ctx = vk_ctx();
    return ctx && ctx->has_fragment_stores;
}

void vk_hdr_off(void) { g_disabled = 1; }
#else
static int g_wanted = -1;

int vk_hdr_wanted(void) {
    if (g_wanted < 0) g_wanted = cfg_bool("REKARINA_HDR");
    return g_wanted;
}

int vk_hdr_armed(void) { return vk_hdr_wanted(); }
int vk_hdr_native(void) { return 0; }
int vk_hdr_native_picture(VkFormat fmt) { (void)fmt; return 0; }
void vk_hdr_off(void) { g_wanted = 0; }
#endif

// Gamma correction code for hdr_encode.comp / hdr_ui.frag: 0 none (sRGB decode), 1: 2.2 power, 2: 2.4 power.
#define HDR_GAMMA_22 1.0f

typedef struct {
    float peak, game, ui;   // nits
    float gamma;            // 0 off, 1: 2.2, 2: 2.4
#ifdef GAME_OOT
    float ui_eotf;
    float adaptation_target, adaptation_min, adaptation_max;
#endif
    float tm[4 * VK_HDR_PARAMS];   // support buffer parameters (from byte VK_HDR_PARAMS_AT)
} Nits;
static float clampf(float v, float lo, float hi) { return fminf(fmaxf(v, lo), hi); }
static float eotf_code(const char *setting) {
    return !strcmp(setting, "off") || !strcmp(setting, "0") ? 0.0f : !strcmp(setting, "2.4") ? 2.0f : HDR_GAMMA_22;
}

static void nits_read(Nits *n) {
    memset(n, 0, sizeof *n);
    n->peak = clampf((float)cfg_double("REKARINA_HDR_PEAK_NITS"), 80.0f, 10000.0f);
    n->game = clampf((float)cfg_double("REKARINA_HDR_GAME_NITS"), 48.0f, 500.0f);
    n->ui = clampf((float)cfg_double("REKARINA_HDR_UI_NITS"), 48.0f, 500.0f);
    n->gamma = vk_hdr_native() ? HDR_GAMMA_22 : eotf_code(cfg_value("REKARINA_HDR_GAMMA"));
#ifdef GAME_OOT
    n->ui_eotf = eotf_code(cfg_value("REKARINA_HDR_UI_EOTF"));
    const char *tm = cfg_value("REKARINA_HDR_TONEMAP");
    float mapper = !strcmp(tm, "vanilla") ? 0.0f : 9.0f;
    static const char *const keys[] = {
        "REKARINA_HDR_HIGHLIGHTS", "REKARINA_HDR_SHADOWS", "REKARINA_HDR_CONTRAST",
        "REKARINA_HDR_SATURATION", "REKARINA_HDR_TONE_SLOPE", "REKARINA_HDR_HUE_CLIP"};
    float pct[sizeof keys / sizeof *keys];
    for (unsigned i = 0; i < sizeof keys / sizeof *keys; i++) pct[i] = clampf((float)cfg_double(keys[i]), 0.0f, 100.0f);
    float *t = n->tm;
    t[0] = n->peak, t[1] = n->game, t[2] = n->game, t[3] = mapper;
    t[4] = clampf((float)cfg_double("REKARINA_HDR_EXPOSURE"), 0.0f, 800.0f) / 100.0f;
    t[5] = pct[0] / 50.0f, t[6] = pct[1] / 50.0f, t[7] = pct[2] / 50.0f;
    t[8] = pct[3] / 50.0f;
    t[14] = n->gamma, t[15] = pct[4] / 50.0f;
    t[16] = pct[5] / 100.0f;
    const char *midgray = cfg_value("REKARINA_HDR_MIDGRAY");
    t[17] = !strcmp(midgray, "render") ? 0.0f : mapper == 9.0f && !strcmp(midgray, "adaptation") ? 2.0f : 1.0f;
    n->adaptation_target = clampf((float)cfg_double("REKARINA_HDR_ADAPTATION_TARGET"), 1.0f, 100.0f) / 100.0f;
    n->adaptation_min = exp2f(clampf((float)cfg_double("REKARINA_HDR_ADAPTATION_MIN_STOPS"), -8.0f, 0.0f));
    n->adaptation_max = exp2f(clampf((float)cfg_double("REKARINA_HDR_ADAPTATION_MAX_STOPS"), 0.0f, 8.0f));
    t[18] = cfg_bool("REKARINA_HDR_MATCH_SLOPE") ? 1.0f : 0.0f, t[19] = 0.0f;
    t[20] = clampf((float)cfg_double("REKARINA_HDR_BLOOM"), 0.0f, 100.0f) / 50.0f;   // 50: the game's bloom
    t[21] = mapper == 0.0f ? clampf((float)cfg_double("REKARINA_HDR_LOCAL_EXPOSURE"), 0.0f, 100.0f) / 50.0f : 0.0f;
#else
    n->tm[0] = n->game / n->ui;
    n->tm[1] = n->peak > n->game ? n->peak / n->game : 1.0f;
    n->tm[2] = clampf((float)cfg_double("REKARINA_HDR_HUE"), 0.0f, 100.0f) / 100.0f;
    n->tm[3] = 0.0f;
#endif
}

static Nits nits(void) {
    static _Thread_local unsigned seen = ~0u;
    static _Thread_local Nits current;
    unsigned v = cfg_version();
    if (seen != v) {
        nits_read(&current);
        seen = v;
    }

    return current;
}

#ifdef GAME_OOT
int vk_hdr_sdr_scene(void) {
    return !vk_hdr_wanted() && !vk_hdr_active() && !strcmp(cfg_value("REKARINA_SDR_TONEMAP"), "psychov35");
}

int vk_hdr_scene_grade(void) {
    Nits n = nits();
    return n.tm[4] != 1.0f || n.tm[5] != 1.0f || n.tm[6] != 1.0f || n.tm[7] != 1.0f || n.tm[8] != 1.0f;
}

#endif

void vk_hdr_support(float out[4 * VK_HDR_PARAMS]) {
    Nits n = nits();
    memcpy(out, n.tm, sizeof n.tm);
#ifdef GAME_OOT
    if (vk_hdr_sdr_scene()) {
        out[0] = out[1] = out[2] = 203.0f;
        out[3] = 8.0f;   // BT.709 target with peak equal to diffuse white
        out[14] = 0.0f;
        if (!strcmp(cfg_value("REKARINA_HDR_MIDGRAY"), "adaptation")) out[17] = 2.0f;
        out[21] = 0.0f;
    } else if (!vk_hdr_active()) {
        out[3] = -1.0f;
        out[21] = 1.0f;
    }
#endif
}

// ---- 1. SPIR-V helpers and tonemap splicing / FSR unclamp

enum {
    SPV_MAGIC = 0x07230203u,
    SPV_OP_NAME = 5,
    SPV_OP_LINE = 8,
    SPV_OP_EXT_INST_IMPORT = 11,
    SPV_OP_EXT_INST = 12,
    SPV_OP_ENTRY_POINT = 15,
    SPV_OP_TYPE_VOID = 19,
    SPV_OP_TYPE_BOOL = 20,
    SPV_OP_TYPE_INT = 21,
    SPV_OP_TYPE_FLOAT = 22,
    SPV_OP_TYPE_VECTOR = 23,
    SPV_OP_TYPE_MATRIX = 24,
    SPV_OP_TYPE_STRUCT = 30,
    SPV_OP_TYPE_POINTER = 32,
    SPV_OP_TYPE_FUNCTION = 33,
    SPV_OP_CONSTANT_TRUE = 41,
    SPV_OP_CONSTANT_FALSE = 42,
    SPV_OP_CONSTANT = 43,
    SPV_OP_CONSTANT_COMPOSITE = 44,
    SPV_OP_CONSTANT_NULL = 46,
    SPV_OP_FUNCTION = 54,
    SPV_OP_FUNCTION_PARAMETER = 55,
    SPV_OP_FUNCTION_END = 56,
    SPV_OP_FUNCTION_CALL = 57,
    SPV_OP_VARIABLE = 59,
    SPV_OP_LOAD = 61,
    SPV_OP_STORE = 62,
    SPV_OP_COPY_MEMORY = 63,
    SPV_OP_ACCESS_CHAIN = 65,
    SPV_OP_DECORATE = 71,
    SPV_OP_MEMBER_DECORATE = 72,
    SPV_OP_VECTOR_SHUFFLE = 79,
    SPV_OP_COMPOSITE_CONSTRUCT = 80,
    SPV_OP_COMPOSITE_EXTRACT = 81,
    SPV_OP_COMPOSITE_INSERT = 82,
    SPV_OP_IMAGE_SAMPLE_IMPLICIT_LOD = 87,
    SPV_OP_IMAGE_SAMPLE_EXPLICIT_LOD = 88,
    SPV_OP_IMAGE_FETCH = 95,
    SPV_OP_IMAGE_WRITE = 99,
    SPV_OP_IMAGE = 100,
    SPV_OP_BITCAST = 124,
    SPV_OP_FNEGATE = 127,
    SPV_OP_FADD = 129,
    SPV_OP_FMUL = 133,
    SPV_OP_FDIV = 136,
    SPV_OP_DOT = 148,
    SPV_OP_ISNAN = 156,
    SPV_OP_SELECT = 169,
    SPV_OP_FORD_EQUAL = 180,
    SPV_OP_ATOMIC_EXCHANGE = 229,
    SPV_OP_PHI = 245,
    SPV_OP_LOOP_MERGE = 246,
    SPV_OP_SELECTION_MERGE = 247,
    SPV_OP_LABEL = 248,
    SPV_OP_BRANCH = 249,
    SPV_OP_BRANCH_CONDITIONAL = 250,
    SPV_OP_SWITCH = 251,
    SPV_OP_NO_LINE = 317,
    SPV_DEC_BINDING = 33,
    SPV_DEC_SET = 34,
    SPV_DEC_OFFSET = 35,
    SPV_SC_INPUT = 1,
    SPV_SC_UNIFORM = 2,
    SPV_SC_OUTPUT = 3,
    SPV_SC_FUNCTION = 7,
    SPV_EXEC_FRAGMENT = 4,
    GLSL_FABS = 4,
    GLSL_EXP2 = 29,
    GLSL_LOG2 = 30,
    GLSL_FMIN = 37,
    GLSL_FMAX = 40,
    GLSL_FCLAMP = 43,
    GLSL_FMA = 50,
    GLSL_NMIN = 79,
    GLSL_NMAX = 80,
    GLSL_NCLAMP = 81
};

static int name_is(const uint32_t *a, uint32_t len, const char *s) {   // OpName / OpExtInstImport operands after op
    size_t n = strlen(s);
    return len >= 2 && (len - 2) * 4 >= n + 1 && !memcmp(a + 1, s, n + 1);
}

static int is_clamp(uint32_t inst) { return inst == GLSL_FCLAMP || inst == GLSL_NCLAMP; }

#if !defined(GAME_OOT) || defined(VK_HDR_SPLICE_TEST_ONLY)
// TotK's FSR 1 pass after the tonemap ends with clamp(colour, 0.0, 1.0) per output channel, clipping HDR
// highlights back to 1.0. Replace output-stored clamps (FClamp or NClamp) with FMax/NMax(x, 0.0).
static int fsr_unclamp(uint32_t *w, uint64_t *words) {
    uint64_t n = *words;
    uint32_t bound = w[3], glsl = 0, float_type = 0, zero = 0, one = 0;
    int magic = 0, tex = 0;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) return 0;
        const uint32_t *a = w + i + 1;
        if (op == SPV_OP_EXT_INST_IMPORT && len >= 3 && name_is(a, len, "GLSL.std.450")) glsl = a[0];
        else if (op == SPV_OP_TYPE_FLOAT && len == 3 && a[1] == 32) float_type = a[0];
        else if (op == SPV_OP_CONSTANT && len == 4) {
            magic |= (a[2] == 0x7ef07ebbu) | (a[2] == 0x7ef19fffu) << 1;
            if (a[0] == float_type && float_type && a[2] == 0) zero = a[1];
            if (a[0] == float_type && float_type && a[2] == 0x3f800000u) one = a[1];
        } else if (op == SPV_OP_NAME && len >= 3 && name_is(a, len, "fp_t_tcb_8")) tex = 1;
        i += len;
    }
    if (magic != 3 || !tex) return 0;
    if (!glsl || !zero || !one || !bound) return -1;
    uint8_t *out = calloc(bound, 1);
    if (!out) return 0;
    for (uint64_t i = 5; i < n; i += w[i] >> 16) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        const uint32_t *a = w + i + 1;
        if (op == SPV_OP_VARIABLE && len >= 4 && a[2] == SPV_SC_OUTPUT && a[1] < bound) out[a[1]] = 1;
        else if (op == SPV_OP_ACCESS_CHAIN && len >= 4 && a[2] < bound && out[a[2]] == 1 && a[1] < bound) out[a[1]] = 1;
        else if (op == SPV_OP_STORE && len >= 3 && a[0] < bound && out[a[0]] == 1 && a[1] < bound) out[a[1]] = 2;
    }
    int changed = 0;
    uint64_t k = 5;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        const uint32_t *a = w + i + 1;
        if (op == SPV_OP_EXT_INST && len == 8 && a[2] == glsl && is_clamp(a[3]) && a[1] < bound && out[a[1]] == 2 &&
            a[5] == zero && a[6] == one) {
            uint32_t ins[7] = {7u << 16 | SPV_OP_EXT_INST, a[0], a[1], a[2],
                               a[3] == GLSL_NCLAMP ? GLSL_NMAX : GLSL_FMAX, a[4], zero};
            memmove(w + k, ins, sizeof ins);
            k += 7, changed++;
        } else {
            memmove(w + k, w + i, len * sizeof *w);
            k += len;
        }
        i += len;
    }
    free(out);
    *words = k;
    return changed ? changed : -1;
}
#endif

#ifdef GAME_OOT

static int name_prefix(const uint32_t *a, uint32_t len, const char *s) {
    size_t n = strlen(s);
    return len >= 2 && (len - 2) * 4 >= n && !memcmp(a + 1, s, n);
}

// Dynamic SPIR-V word buffer for splicing.
typedef struct {
    uint32_t *w;
    uint64_t n, cap;
    int err;
} SpvBuf;

static void spv_emit(SpvBuf *b, const uint32_t *words, uint32_t count) {
    if (b->err) return;
    if (b->n + count > b->cap) {
        uint64_t nc = b->cap ? b->cap * 2 : 4096;
        while (nc < b->n + count) nc *= 2;
        uint32_t *nw = realloc(b->w, nc * sizeof *nw);
        if (!nw) { b->err = 1; return; }
        b->w = nw, b->cap = nc;
    }
    memcpy(b->w + b->n, words, count * sizeof *words);
    b->n += count;
}

#define SPV_INS(b, op, ...) do {                                               \
    const uint32_t a_[] = {__VA_ARGS__};                                       \
    uint32_t c_ = (uint32_t)(sizeof a_ / sizeof *a_) + 1u;                     \
    uint32_t h_ = (c_ << 16) | (uint32_t)(op);                                 \
    spv_emit((b), &h_, 1);                                                     \
    spv_emit((b), a_, c_ - 1u);                                                \
} while (0)

static int spv_is_type(uint32_t op) { return op >= SPV_OP_TYPE_VOID && op <= 39; }
static int spv_is_const(uint32_t op) {
    return (op >= SPV_OP_CONSTANT_TRUE && op <= SPV_OP_CONSTANT_NULL && op != 45);
}

// Word index of the result ID inside an instruction (0 if none, 1 for types/labels/ext_inst_import, 2 for typed values).
static uint32_t spv_res_word(uint32_t op, uint32_t len) {
    if (spv_is_type(op) || op == SPV_OP_LABEL || op == SPV_OP_EXT_INST_IMPORT) return len >= 2 ? 1 : 0;
    if (spv_is_const(op) || op == SPV_OP_FUNCTION || op == SPV_OP_FUNCTION_PARAMETER || op == SPV_OP_FUNCTION_CALL ||
        op == SPV_OP_VARIABLE || op == SPV_OP_LOAD || op == SPV_OP_ACCESS_CHAIN || op == SPV_OP_EXT_INST ||
        op == SPV_OP_VECTOR_SHUFFLE || op == SPV_OP_COMPOSITE_CONSTRUCT || op == SPV_OP_COMPOSITE_EXTRACT ||
        op == SPV_OP_COMPOSITE_INSERT || op == SPV_OP_IMAGE_SAMPLE_IMPLICIT_LOD || op == SPV_OP_PHI || (op >= 109 && op <= 207))
        return len >= 3 ? 2 : 0;
    if (op == SPV_OP_ATOMIC_EXCHANGE) return len >= 3 ? 2 : 0;
    return 0;
}

// Returns 1 if operand word j (1-based after opcode word) is a literal rather than an ID, 0 if an ID, -1 if unsupported.
static int spv_op_literal(uint32_t op, uint32_t j) {
    switch (op) {
    case SPV_OP_TYPE_INT: case SPV_OP_TYPE_FLOAT: return j >= 2;
    case SPV_OP_TYPE_VECTOR: case SPV_OP_TYPE_MATRIX: return j == 3;
    case SPV_OP_TYPE_POINTER: return j == 2;
    case SPV_OP_CONSTANT: return j >= 3;
    case SPV_OP_FUNCTION: case SPV_OP_VARIABLE: return j == 3;
    case SPV_OP_EXT_INST: return j == 4;
    case SPV_OP_LOAD: return j >= 4;
    case SPV_OP_STORE: return j >= 3;
    case SPV_OP_COMPOSITE_EXTRACT: return j >= 4;
    case SPV_OP_COMPOSITE_INSERT: case SPV_OP_VECTOR_SHUFFLE: return j >= 5;
    case SPV_OP_LOOP_MERGE: return j >= 3;
    case SPV_OP_SELECTION_MERGE: return j == 2;
    case SPV_OP_BRANCH_CONDITIONAL: return j >= 4;
    case SPV_OP_SWITCH: case SPV_OP_LINE: case SPV_OP_NO_LINE: case SPV_OP_COPY_MEMORY: return -1;
    default: return 0;
    }
}

static int spv_emit_mapped(SpvBuf *o, const uint32_t *ins, const uint32_t *map, uint32_t lbound) {
    uint32_t op = ins[0] & 0xffff, len = ins[0] >> 16;
    if (!len || len > 64) return 0;
    uint32_t tmp[64];
    tmp[0] = ins[0];
    for (uint32_t j = 1; j < len; j++) {
        int lit = spv_op_literal(op, j);
        if (lit < 0) return 0;
        if (lit) tmp[j] = ins[j];
        else {
            if (ins[j] >= lbound || !map[ins[j]]) return 0;
            tmp[j] = map[ins[j]];
        }
    }
    spv_emit(o, tmp, len);
    return !o->err;
}

// Host module analysis tables for matching ACES tonemap, exposure extraction, and CFG dominance.
typedef struct {
    const uint32_t *w;
    uint64_t n;
    uint32_t bound, glsl;
    uint32_t *def;        // id -> word offset in w (0 if none)
    int32_t *def_blk;     // id -> basic block index in main (-1 if global / outside main)
    uint8_t *cls;         // id -> 1 output var, 2 uniform var, 3 function var
    uint32_t *sval;       // function var id -> value stored (valid when scnt == 1)
    uint8_t *scnt;        // function var id -> number of OpStores (saturated at 2)
    float *fval;          // id -> float value for f32 OpConstant
    uint8_t *has_fval;
    int32_t *ival;        // id -> int value for 32-bit OpConstant
    uint8_t *has_ival;
    uint32_t *idom;       // per main basic block: immediate dominator block index
    uint32_t nblk;
} HostSpv;

static int is_fnear(const HostSpv *h, uint32_t id, float want, float tol) {
    return id < h->bound && h->has_fval[id] && fabsf(h->fval[id] - want) <= tol;
}

static uint32_t unwrap_func_var(const HostSpv *h, uint32_t id) {
    if (id >= h->bound || !h->def[id]) return id;
    const uint32_t *ins = h->w + h->def[id];
    if ((ins[0] & 0xffff) == SPV_OP_LOAD && ins[3] < h->bound && h->cls[ins[3]] == 3 && h->scnt[ins[3]] == 1)
        return h->sval[ins[3]];
    return id;
}

static int is_uniform_load(const HostSpv *h, uint32_t id) {
    if (id >= h->bound || !h->def[id]) return 0;
    const uint32_t *l = h->w + h->def[id];
    if ((l[0] & 0xffff) != SPV_OP_LOAD || l[3] >= h->bound || !h->def[l[3]]) return 0;
    const uint32_t *ac = h->w + h->def[l[3]];
    return (ac[0] & 0xffff) == SPV_OP_ACCESS_CHAIN && ac[3] < h->bound && h->cls[ac[3]] == 2;
}

static int same_load_source(const HostSpv *h, uint32_t a, uint32_t b) {
    if (a == b) return 1;
    if (a >= h->bound || b >= h->bound || !h->def[a] || !h->def[b]) return 0;
    const uint32_t *la = h->w + h->def[a], *lb = h->w + h->def[b];
    if ((la[0] & 0xffff) != SPV_OP_LOAD || (lb[0] & 0xffff) != SPV_OP_LOAD) return 0;
    if (la[3] == lb[3]) return 1;
    if (la[3] >= h->bound || lb[3] >= h->bound || !h->def[la[3]] || !h->def[lb[3]]) return 0;
    const uint32_t *ca = h->w + h->def[la[3]], *cb = h->w + h->def[lb[3]];
    return (ca[0] & 0xffff) == SPV_OP_ACCESS_CHAIN && ca[0] == cb[0] &&
           !memcmp(ca + 3, cb + 3, ((ca[0] >> 16) - 3) * 4);
}

// Match Narkowicz ACES numerator x * (x * 2.51 + 0.03) inside clamp(num / denom, 0, 1).
// Supports both FMA and FMul+FAdd, and both FDiv(num, denom) and FMul(num, FDiv(1, denom)).
static int match_aces_inner(const HostSpv *h, uint32_t id, uint32_t x) {
    id = unwrap_func_var(h, id);
    if (id >= h->bound || !h->def[id]) return 0;
    const uint32_t *ins = h->w + h->def[id];
    uint32_t op = ins[0] & 0xffff, len = ins[0] >> 16;
    if (op == SPV_OP_EXT_INST && len == 8 && ins[3] == h->glsl && ins[4] == GLSL_FMA) {
        if (!is_fnear(h, ins[7], 0.03f, 1e-4f)) return 0;
        return (same_load_source(h, ins[5], x) && is_fnear(h, ins[6], 2.51f, 1e-4f)) ||
               (same_load_source(h, ins[6], x) && is_fnear(h, ins[5], 2.51f, 1e-4f));
    }
    if (op == SPV_OP_FADD && len == 5) {
        uint32_t mul_id = 0;
        if (is_fnear(h, ins[3], 0.03f, 1e-4f)) mul_id = ins[4];
        else if (is_fnear(h, ins[4], 0.03f, 1e-4f)) mul_id = ins[3];
        else return 0;
        mul_id = unwrap_func_var(h, mul_id);
        if (mul_id >= h->bound || !h->def[mul_id]) return 0;
        const uint32_t *m = h->w + h->def[mul_id];
        if ((m[0] & 0xffff) != SPV_OP_FMUL || (m[0] >> 16) != 5) return 0;
        return (same_load_source(h, m[3], x) && is_fnear(h, m[4], 2.51f, 1e-4f)) ||
               (same_load_source(h, m[4], x) && is_fnear(h, m[3], 2.51f, 1e-4f));
    }
    return 0;
}

static int match_aces_num(const HostSpv *h, uint32_t id, uint32_t *x_out) {
    id = unwrap_func_var(h, id);
    if (id >= h->bound || !h->def[id]) return 0;
    const uint32_t *m = h->w + h->def[id];
    if ((m[0] & 0xffff) != SPV_OP_FMUL || (m[0] >> 16) != 5) return 0;
    if (match_aces_inner(h, m[4], m[3])) { *x_out = m[3]; return 1; }
    if (match_aces_inner(h, m[3], m[4])) { *x_out = m[4]; return 1; }
    return 0;
}

static int match_aces_clamp(const HostSpv *h, uint32_t id, uint32_t *x_out) {
    id = unwrap_func_var(h, id);
    if (id >= h->bound || !h->def[id]) return 0;
    const uint32_t *c = h->w + h->def[id];
    if ((c[0] & 0xffff) != SPV_OP_EXT_INST || (c[0] >> 16) != 8 || c[3] != h->glsl || !is_clamp(c[4])) return 0;
    if (!is_fnear(h, c[6], 0.0f, 0.0f) || !is_fnear(h, c[7], 1.0f, 0.0f)) return 0;
    uint32_t ratio = unwrap_func_var(h, c[5]);
    if (ratio >= h->bound || !h->def[ratio]) return 0;
    const uint32_t *r = h->w + h->def[ratio];
    uint32_t rop = r[0] & 0xffff;
    if (rop == SPV_OP_FMUL && (r[0] >> 16) == 5)
        return match_aces_num(h, r[3], x_out) || match_aces_num(h, r[4], x_out);
    if (rop == SPV_OP_FDIV && (r[0] >> 16) == 5)
        return match_aces_num(h, r[3], x_out);
    return 0;
}

static int common_mul_factor(const HostSpv *h, const uint32_t val[3], uint32_t rest[3], uint32_t *factor) {
    const uint32_t *m[3];
    for (int c = 0; c < 3; c++) {
        if (val[c] >= h->bound || !h->def[val[c]]) return 0;
        m[c] = h->w + h->def[val[c]];
        if ((m[c][0] & 0xffff) != SPV_OP_FMUL || (m[c][0] >> 16) != 5) return 0;
    }
    for (int pos = 4; pos >= 3; pos--) {
        uint32_t f = m[0][pos];
        rest[0] = m[0][7 - pos];
        int ok = 1;
        for (int c = 1; c < 3 && ok; c++) {
            if (same_load_source(h, f, m[c][4])) rest[c] = m[c][3];
            else if (same_load_source(h, f, m[c][3])) rest[c] = m[c][4];
            else ok = 0;
        }
        if (ok) { *factor = f; return 1; }
    }
    return 0;
}

static int common_fma_factor(const HostSpv *h, const uint32_t val[3], uint32_t a[3], uint32_t t[3], uint32_t *factor) {
    const uint32_t *m[3];
    for (int c = 0; c < 3; c++) {
        if (val[c] >= h->bound || !h->def[val[c]]) return 0;
        m[c] = h->w + h->def[val[c]];
        if ((m[c][0] & 0xffff) != SPV_OP_EXT_INST || (m[c][0] >> 16) != 8 || m[c][3] != h->glsl || m[c][4] != GLSL_FMA)
            return 0;
        t[c] = m[c][7];
    }
    for (int pos = 6; pos >= 5; pos--) {
        uint32_t f = m[0][pos];
        a[0] = m[0][11 - pos];
        int ok = 1;
        for (int c = 1; c < 3 && ok; c++) {
            if (same_load_source(h, f, m[c][6])) a[c] = m[c][5];
            else if (same_load_source(h, f, m[c][5])) a[c] = m[c][6];
            else ok = 0;
        }
        if (ok) { *factor = f; return 1; }
    }
    return 0;
}

// What the 3 ACES input channels are made of (extract_exposure).
typedef struct {
    uint32_t gain;         // the uniform exposure gain
    uint32_t exposed[3];   // scene * gain, before local exposure
    uint32_t add[3];       // a colour added after local exposure (0: none)
    uint32_t scene[3];     // the scene before the gain, where the game has added its bloom (0: not traced)
    int has_local_term;    // distinguishes a direct gain from an unrecognized local correction
    uint32_t local;        // Exp2 result giving the game's local exposure factor (0: none or not traced)
    uint64_t local_at[3];  // the instruction applying it in each channel (its word offset in main())
    uint32_t local_word[3]; // and the word index of the factor there
} Exposure;

static const uint32_t *glsl_def(const HostSpv *h, uint32_t id, uint32_t inst);

// Local exposure is a shared Exp2 after the uniform gain. Record only its three uses in the ACES input.
static void note_local(const HostSpv *h, const uint32_t val[3], uint32_t first, uint32_t f, Exposure *e) {
    const uint32_t *exp2 = glsl_def(h, f, GLSL_EXP2);
    if (!exp2 || (exp2[0] >> 16) != 6 || h->def_blk[exp2[2]] < 0) return;
    for (int c = 0; c < 3; c++) {
        uint64_t at = h->def[val[c]];
        const uint32_t *ins = h->w + at;
        uint32_t word = unwrap_func_var(h, ins[first]) == exp2[2] ? first : first + 1;
        if (h->def_blk[val[c]] < 0 || unwrap_func_var(h, ins[word]) != exp2[2]) return;
        e->local_at[c] = at;
        e->local_word[c] = word;
    }
    if (e->local_at[0] == e->local_at[1] || e->local_at[1] == e->local_at[2] || e->local_at[0] == e->local_at[2]) return;
    e->local = exp2[2];
}

// Extract the uniform exposure gain and pre-local-exposure scene from the 3 ACES input channels:
// 1. x = scene * gain                          -> exposed = x, add = 0
// 2. x = (scene * gain) * local_exposure       -> exposed = scene * gain, add = 0
// 3. x = fma(scene, gain, color)               -> exposed = x, add = 0
// 4. x = fma(scene * gain, local_exp, color)   -> exposed = scene * gain, add = color
static int extract_exposure(const HostSpv *h, const uint32_t x[3], Exposure *e) {
    uint32_t val[3], inner[3], rest[3], a[3], t[3], f;
    memset(e, 0, sizeof *e);
    for (int c = 0; c < 3; c++) val[c] = unwrap_func_var(h, x[c]), e->exposed[c] = x[c];
    if (common_mul_factor(h, val, rest, &e->gain)) {
        if (is_uniform_load(h, e->gain)) {
            memcpy(e->scene, rest, sizeof rest);
            return 1;
        }
        for (int c = 0; c < 3; c++) inner[c] = unwrap_func_var(h, rest[c]);
        if (common_mul_factor(h, inner, rest, &f) && is_uniform_load(h, f)) {
            e->has_local_term = 1;
            note_local(h, val, 3, e->gain, e);
            memcpy(e->exposed, inner, sizeof inner);
            memcpy(e->scene, rest, sizeof rest);
            e->gain = f;
        }
        return 1;
    }
    if (!common_fma_factor(h, val, a, t, &e->gain)) return 0;
    if (is_uniform_load(h, e->gain)) {
        memcpy(e->scene, a, sizeof a);
        return 1;
    }
    for (int c = 0; c < 3; c++) inner[c] = unwrap_func_var(h, a[c]);
    if (common_mul_factor(h, inner, rest, &f) && is_uniform_load(h, f)) {
        e->has_local_term = 1;
        note_local(h, val, 5, e->gain, e);
        memcpy(e->exposed, inner, sizeof inner);
        memcpy(e->add, t, sizeof t);
        memcpy(e->scene, rest, sizeof rest);
        e->gain = f;
        return 1;
    }
    return 0;
}

// The definition of id (through a Function variable stored once) if it is the GLSL.std.450 instruction inst.
static const uint32_t *glsl_def(const HostSpv *h, uint32_t id, uint32_t inst) {
    id = unwrap_func_var(h, id);
    if (id >= h->bound || !h->def[id]) return NULL;
    const uint32_t *ins = h->w + h->def[id];
    return (ins[0] & 0xffff) == SPV_OP_EXT_INST && (ins[0] >> 16) >= 6 && ins[3] == h->glsl && ins[4] == inst ? ins : NULL;
}

static const uint32_t *host_def(const HostSpv *h, uint32_t id, uint32_t op, uint32_t len) {
    id = unwrap_func_var(h, id);
    if (id >= h->bound || !h->def[id]) return NULL;
    const uint32_t *ins = h->w + h->def[id];
    return ins[0] == ((len << 16) | op) ? ins : NULL;
}

static uint32_t output_before_nan_sanitize(const HostSpv *h, uint32_t value) {
    const uint32_t *cast = host_def(h, value, SPV_OP_BITCAST, 4);
    const uint32_t *sel = cast ? host_def(h, cast[3], SPV_OP_SELECT, 6) : NULL;
    const uint32_t *nan = sel ? host_def(h, sel[3], SPV_OP_ISNAN, 4) : NULL;
    return nan ? nan[3] : value;
}

// The HDR compose expands the fitted RGB around a shared luminance, then limits each output at a uniform ceiling.
static uint32_t native_hdr_fit(const HostSpv *h, uint32_t value) {
    value = output_before_nan_sanitize(h, value);
    const uint32_t *mn = glsl_def(h, value, GLSL_FMIN);
    if (!mn) mn = glsl_def(h, value, GLSL_NMIN);
    if (!mn || (mn[0] >> 16) != 7) return 0;
    for (unsigned i = 5; i < 7; i++) {
        if (!is_uniform_load(h, mn[11 - i])) continue;
        const uint32_t *mx = glsl_def(h, mn[i], GLSL_FMAX);
        if (!mx) mx = glsl_def(h, mn[i], GLSL_NMAX);
        if (!mx || (mx[0] >> 16) != 7) continue;
        for (unsigned j = 5; j < 7; j++) {
            if (!is_fnear(h, mx[11 - j], 0.0f, 0.0f)) continue;
            const uint32_t *mul = host_def(h, mx[j], SPV_OP_FMUL, 5);
            if (!mul) continue;
            for (unsigned k = 3; k < 5; k++) {
                const uint32_t *outer = glsl_def(h, mul[k], GLSL_FMA);
                if (!outer || (outer[0] >> 16) != 8 || !is_fnear(h, outer[7], 1.0f, 0.0f)) continue;
                for (unsigned m = 5; m < 7; m++) {
                    const uint32_t *inner = glsl_def(h, outer[m], GLSL_FMA);
                    if (!inner || (inner[0] >> 16) != 8 || !is_fnear(h, inner[7], -1.0f, 0.0f)) continue;
                    uint32_t x;
                    for (unsigned n = 5; n < 7; n++)
                        if (match_aces_clamp(h, inner[n], &x)) return unwrap_func_var(h, inner[n]);
                }
            }
        }
    }
    return 0;
}

static uint32_t compose_fit(const HostSpv *h, uint32_t value) {
    uint32_t fit = native_hdr_fit(h, value);
    return fit ? fit : unwrap_func_var(h, output_before_nan_sanitize(h, value));
}

// exp2(log2(|b|) * power) with a uniform power: how the composite decodes one channel of its bloom texture (the
// power is its c4[13]).
static int is_bloom_term(const HostSpv *h, uint32_t id) {
    const uint32_t *e = glsl_def(h, id, GLSL_EXP2);
    if (!e) return 0;
    uint32_t m = unwrap_func_var(h, e[5]);
    if (m >= h->bound || !h->def[m]) return 0;
    const uint32_t *mul = h->w + h->def[m];
    if ((mul[0] & 0xffff) != SPV_OP_FMUL || (mul[0] >> 16) != 5) return 0;
    for (int k = 3; k <= 4; k++) {
        const uint32_t *l = glsl_def(h, mul[k], GLSL_LOG2);
        if (l && glsl_def(h, l[5], GLSL_FABS) && is_uniform_load(h, mul[7 - k])) return 1;
    }
    return 0;
}

// min(max(0, v), 10000): one channel of the composite's colour-graded scene, which its bloom is added to. Other ACES
// passes decode textures with the same exp2(log2(|b|) * power) and add them to other things.
static int is_scene_clamp(const HostSpv *h, uint32_t id) {
    const uint32_t *mn = glsl_def(h, id, GLSL_FMIN);
    if (!mn) mn = glsl_def(h, id, GLSL_NMIN);
    if (!mn || (mn[0] >> 16) != 7) return 0;
    for (int k = 5; k <= 6; k++) {
        if (!is_fnear(h, mn[11 - k], 10000.0f, 0.0f)) continue;
        const uint32_t *mx = glsl_def(h, mn[k], GLSL_FMAX);
        if (!mx) mx = glsl_def(h, mn[k], GLSL_NMAX);
        if (mx && (mx[0] >> 16) == 7 && (is_fnear(h, mx[5], 0.0f, 0.0f) || is_fnear(h, mx[6], 0.0f, 0.0f))) return 1;
    }
    return 0;
}

// What a composite adds its bloom to: one channel of its colour-graded scene (is_scene_clamp), or that channel faded
// towards black by a factor, fma(-scene, f, scene) (the composites with fog).
static int is_scene_base(const HostSpv *h, uint32_t id) {
    if (is_scene_clamp(h, id)) return 1;
    const uint32_t *f = glsl_def(h, id, GLSL_FMA);
    if (!f || (f[0] >> 16) != 8 || !is_scene_clamp(h, f[7])) return 0;
    for (int k = 5; k <= 6; k++) {
        uint32_t n = unwrap_func_var(h, f[k]);
        const uint32_t *neg = n < h->bound && h->def[n] ? h->w + h->def[n] : NULL;
        if (neg && (neg[0] & 0xffff) == SPV_OP_FNEGATE && same_load_source(h, neg[3], f[7])) return 1;
    }
    return 0;
}

// Where one scene channel adds the game's bloom: scene + bloom (FAdd) or fma(bloom, k, scene), the scene as
// is_scene_base takes it. *at: the instruction's word offset in main(), *word: the word index of its bloom operand. 0
// when the channel is not built that way.
static int bloom_operand(const HostSpv *h, uint32_t scene, uint64_t *at, uint32_t *word) {
    scene = unwrap_func_var(h, scene);
    if (scene >= h->bound || !h->def[scene] || h->def_blk[scene] < 0) return 0;
    const uint32_t *ins = h->w + h->def[scene];
    uint32_t op = ins[0] & 0xffff, len = ins[0] >> 16;
    int fadd = op == SPV_OP_FADD && len == 5;
    int fma = op == SPV_OP_EXT_INST && len == 8 && ins[3] == h->glsl && ins[4] == GLSL_FMA;
    if (!fadd && !fma) return 0;
    uint32_t first = fadd ? 3 : 5, last = first + 1;
    for (uint32_t k = first; k <= last; k++)
        if (is_bloom_term(h, ins[k]) && is_scene_base(h, fadd ? ins[7 - k] : ins[7])) {
            *at = h->def[scene], *word = k;
            return 1;
        }

    return 0;
}

// The pause scene adds the RGB channels of one unencoded bloom sample directly to the graded scene.
static int direct_bloom_operands(const HostSpv *h, const Exposure *ex, uint64_t at[3], uint32_t word[3]) {
    uint32_t texture_sample = 0;
    for (unsigned c = 0; c < 3; c++) {
        const uint32_t *add = host_def(h, ex->scene[c], SPV_OP_FADD, 5);
        if (!add || h->def_blk[add[2]] < 0) return 0;
        unsigned k = is_scene_clamp(h, add[3]) ? 4 : is_scene_clamp(h, add[4]) ? 3 : 0;
        if (!k) return 0;
        const uint32_t *extract = host_def(h, add[k], SPV_OP_COMPOSITE_EXTRACT, 5);
        if (!extract || extract[4] != c) return 0;
        uint32_t sample = unwrap_func_var(h, extract[3]);
        const uint32_t *rgb = host_def(h, sample, SPV_OP_VECTOR_SHUFFLE, 8);
        if (rgb && rgb[3] == rgb[4] && rgb[5] == 0 && rgb[6] == 1 && rgb[7] == 2) sample = rgb[3];
        if (c == 0) {
            if (sample >= h->bound || !h->def[sample] ||
                (h->w[h->def[sample]] & 0xffff) != SPV_OP_IMAGE_SAMPLE_IMPLICIT_LOD) return 0;
            texture_sample = sample;
        } else if (sample != texture_sample) return 0;
        at[c] = (uint64_t)(add - h->w), word[c] = k;
    }
    return 1;
}

// Compute immediate dominators for main()'s CFG using Cooper-Harvey-Kennedy iterative algorithm.
static uint32_t dom_intersect(const uint32_t *idom, uint32_t a, uint32_t b) {
    while (a != b) {
        while (a > b) a = idom[a];
        while (b > a) b = idom[b];
    }
    return a;
}

static int build_dominators(HostSpv *h, uint64_t main_start, uint64_t main_end) {
    uint32_t nblk = 0;
    for (uint64_t i = main_start; i < main_end; i += h->w[i] >> 16)
        if ((h->w[i] & 0xffff) == SPV_OP_LABEL) nblk++;
    if (!nblk) return 0;
    int32_t *lbl_to_blk = malloc(h->bound * sizeof *lbl_to_blk);
    uint32_t *idom = malloc(nblk * sizeof *idom);
    uint32_t *pred_head = malloc(nblk * sizeof *pred_head);
    uint32_t edge_cap = nblk * 2 + 16, nedge = 0;
    uint32_t *edge_from = malloc(edge_cap * sizeof *edge_from);
    uint32_t *edge_next = malloc(edge_cap * sizeof *edge_next);
    if (!lbl_to_blk || !idom || !pred_head || !edge_from || !edge_next) {
        free(lbl_to_blk); free(idom); free(pred_head); free(edge_from); free(edge_next);
        return 0;
    }
    for (uint32_t i = 0; i < h->bound; i++) lbl_to_blk[i] = -1;
    for (uint32_t i = 0; i < nblk; i++) idom[i] = UINT32_MAX, pred_head[i] = UINT32_MAX;
    uint32_t blk = 0;
    for (uint64_t i = main_start; i < main_end; i += h->w[i] >> 16) {
        uint32_t op = h->w[i] & 0xffff, len = h->w[i] >> 16;
        if (op == SPV_OP_LABEL && len >= 2 && h->w[i + 1] < h->bound) lbl_to_blk[h->w[i + 1]] = (int32_t)blk++;
        uint32_t rw = spv_res_word(op, len);
        if (rw && h->w[i + rw] < h->bound && blk > 0) h->def_blk[h->w[i + rw]] = (int32_t)(blk - 1);
    }
    int32_t cur = -1;
    for (uint64_t i = main_start; i < main_end; i += h->w[i] >> 16) {
        uint32_t op = h->w[i] & 0xffff, len = h->w[i] >> 16;
        if (op == SPV_OP_LABEL) { cur++; continue; }
        if (cur < 0) continue;
        uint32_t succ[32], ns = 0;
        if (op == SPV_OP_BRANCH && len >= 2) succ[ns++] = h->w[i + 1];
        else if (op == SPV_OP_BRANCH_CONDITIONAL && len >= 4) succ[ns++] = h->w[i + 2], succ[ns++] = h->w[i + 3];
        else if (op == SPV_OP_SWITCH && len >= 3) {
            succ[ns++] = h->w[i + 2];
            for (uint32_t j = 4; j < len && ns < 32; j += 2) succ[ns++] = h->w[i + j];
        }
        for (uint32_t s = 0; s < ns; s++) {
            int32_t tb = succ[s] < h->bound ? lbl_to_blk[succ[s]] : -1;
            if (tb < 0) continue;
            if (nedge == edge_cap) {
                edge_cap *= 2;
                uint32_t *nf = realloc(edge_from, edge_cap * sizeof *nf);
                uint32_t *nn = realloc(edge_next, edge_cap * sizeof *nn);
                if (!nf || !nn) {
                    free(nf ? nf : edge_from); free(nn ? nn : edge_next);
                    free(lbl_to_blk); free(idom); free(pred_head);
                    return 0;
                }
                edge_from = nf, edge_next = nn;
            }
            edge_from[nedge] = (uint32_t)cur;
            edge_next[nedge] = pred_head[tb];
            pred_head[tb] = nedge++;
        }
    }
    idom[0] = 0;
    for (int iter = 0; iter < 64; iter++) {
        int changed = 0;
        for (uint32_t b = 1; b < nblk; b++) {
            uint32_t ni = UINT32_MAX;
            for (uint32_t e = pred_head[b]; e != UINT32_MAX; e = edge_next[e]) {
                uint32_t p = edge_from[e];
                if (idom[p] == UINT32_MAX) continue;
                ni = ni == UINT32_MAX ? p : dom_intersect(idom, p, ni);
            }
            if (ni != idom[b]) idom[b] = ni, changed = 1;
        }
        if (!changed) break;
    }
    free(lbl_to_blk); free(pred_head); free(edge_from); free(edge_next);
    h->idom = idom, h->nblk = nblk;
    return 1;
}

static int dominates(const HostSpv *h, uint32_t id, int32_t ins_blk, uint64_t ins_pos) {
    if (!id || id >= h->bound || !h->def[id]) return 0;
    int32_t b = h->def_blk[id];
    if (b < 0) return 1;   // global constant or module-scope value
    if (b == ins_blk) return h->def[id] < ins_pos;
    if (ins_blk < 0 || (uint32_t)ins_blk >= h->nblk) return 0;
    uint32_t cur = (uint32_t)ins_blk;
    for (uint32_t steps = 0; steps <= h->nblk; steps++) {
        if (cur == (uint32_t)b) return 1;
        uint32_t next = h->idom[cur];
        if (next == cur || next == UINT32_MAX) break;
        cur = next;
    }
    return 0;
}

// Find an identical non-aggregate type already emitted in `w[0..end)` for deduplication.
static uint32_t find_host_type(const uint32_t *w, uint64_t end, const uint32_t *ins, uint32_t len) {
    uint32_t op = ins[0] & 0xffff;
    if (op == SPV_OP_TYPE_STRUCT) return 0;   // structs are nominal
    for (uint64_t i = 5; i < end;) {
        uint32_t hop = w[i] & 0xffff, hlen = w[i] >> 16;
        if (!hlen || i + hlen > end) break;
        if (hop == op && hlen == len && (len <= 2 || !memcmp(w + i + 2, ins + 2, (len - 2) * 4)))
            return w[i + 1];
        i += hlen;
    }
    return 0;
}

// The parameters nx_tonemap takes, p0..p4; the support buffer holds VK_HDR_PARAMS, and the others are read where they
// act (p5.x and p5.y by the game's bloom and local exposure terms).
#define NX_TONEMAP_PARAMS 5u

// What tonemap_splice traced beside the tone mapper, for the log (vk_hdr_shader_swap) and tests/hdr_splice_check.c.
typedef struct {
    int exposure;   // the uniform gain and the pre-local-exposure scene
    int bloom;      // the game's bloom term in each channel (REKARINA_HDR_BLOOM acts there)
    int local_exposure;
    int meter_binding;   // fragment storage buffer for eye adaptation, 0 when unavailable
} SpliceTrace;

// Splice nx_tonemap from hdr_tonemap_spv into a translated OoT ACES tonemap fragment shader.
// Returns 1 if spliced, 0 if the shader is not an ACES tonemap shader, -1 on splice failure; *trace (may be NULL) says
// what else was found.
enum { SPLICE_ACES, SPLICE_HDR_SCENE, SPLICE_HDR_EFFECTS, SPLICE_VANILLA_GRADE };

static int tonemap_splice(uint32_t **spirv, uint64_t *words, int meter_binding, int mode, SpliceTrace *trace) {
    enum { DEC_BUILTIN = 11, BUILTIN_FRAG_COORD = 15, SC_STORAGE_BUFFER = 12 };
    const int native_compose = mode != SPLICE_ACES, effects_only = mode == SPLICE_HDR_EFFECTS;
    const int grade_only = mode == SPLICE_VANILLA_GRADE;
    if (trace) memset(trace, 0, sizeof *trace);
    const uint32_t *w = *spirv;
    uint64_t n = *words;
    if (n < 5 || w[0] != SPV_MAGIC) return 0;
    uint32_t bound = w[3];
    if (!bound || bound > (1u << 22)) return 0;

    HostSpv h = {0};
    h.w = w, h.n = n, h.bound = bound;
    h.def = calloc(bound, sizeof *h.def);
    h.def_blk = malloc(bound * sizeof *h.def_blk);
    h.cls = calloc(bound, 1);
    h.sval = calloc(bound, sizeof *h.sval);
    h.scnt = calloc(bound, 1);
    h.fval = calloc(bound, sizeof *h.fval);
    h.has_fval = calloc(bound, 1);
    h.ival = calloc(bound, sizeof *h.ival);
    h.has_ival = calloc(bound, 1);
    if (!h.def || !h.def_blk || !h.cls || !h.sval || !h.scnt || !h.fval || !h.has_fval || !h.ival || !h.has_ival) {
        free(h.def); free(h.def_blk); free(h.cls); free(h.sval); free(h.scnt);
        free(h.fval); free(h.has_fval); free(h.ival); free(h.has_ival);
        return 0;
    }
    for (uint32_t i = 0; i < bound; i++) h.def_blk[i] = -1;

    uint32_t f32_ty = 0, vec4_ty = 0, i32_ty = 0, u32_ty = 0, ptr_u_vec4 = 0;
    uint32_t main_id = 0, sup_var = 0, sup_ptr_ty = 0, sup_struct = 0, frag_coord = 0;
    uint64_t first_func = 0, main_start = 0, main_end = 0, annot_end = 5, ep_pos = 0;
    uint64_t vec4_pos = 0, sup_struct_pos = 0;
    int seen_251 = 0, seen_003 = 0;

    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) goto fail_0;
        const uint32_t *a = w + i + 1;
        uint32_t rw = spv_res_word(op, len);
        if (rw && w[i + rw] < bound) h.def[w[i + rw]] = (uint32_t)i;
        if (op == SPV_OP_EXT_INST_IMPORT && len >= 3 && name_is(a, len, "GLSL.std.450")) h.glsl = a[0];
        else if (op == SPV_OP_ENTRY_POINT && len >= 4 && a[0] == SPV_EXEC_FRAGMENT) main_id = a[1], ep_pos = i;
        else if (op == SPV_OP_NAME && len >= 3 && name_is(a, len, "support_buffer")) sup_var = a[0];
        else if (op == SPV_OP_DECORATE || op == SPV_OP_MEMBER_DECORATE) {
            annot_end = i + len;
            if (op == SPV_OP_DECORATE && len == 4 && a[1] == DEC_BUILTIN && a[2] == BUILTIN_FRAG_COORD)
                frag_coord = a[0];
            if (meter_binding && op == SPV_OP_DECORATE && len == 4 && a[1] == SPV_DEC_BINDING && a[2] >= (uint32_t)meter_binding)
                meter_binding = a[2] < UINT16_MAX ? (int)a[2] + 1 : 0;
        }
        else if (op == SPV_OP_TYPE_FLOAT && len == 3 && a[1] == 32) f32_ty = a[0];
        else if (op == SPV_OP_TYPE_INT && len == 4 && a[1] == 32) {
            if (a[2] == 1) i32_ty = a[0]; else u32_ty = a[0];
        } else if (op == SPV_OP_TYPE_VECTOR && len == 4 && a[1] == f32_ty && f32_ty && a[2] == 4) {
            vec4_ty = a[0], vec4_pos = i;
        } else if (op == SPV_OP_TYPE_POINTER && len == 4 && a[1] == SPV_SC_UNIFORM && a[2] == vec4_ty && vec4_ty) {
            ptr_u_vec4 = a[0];
        } else if (op == SPV_OP_CONSTANT && len == 4 && a[1] < bound) {
            if (f32_ty && a[0] == f32_ty) {
                float fv; memcpy(&fv, &a[2], 4);
                h.fval[a[1]] = fv, h.has_fval[a[1]] = 1;
                if (fabsf(fv - 2.51f) <= 1e-4f) seen_251 = 1;
                if (fabsf(fv - 0.03f) <= 1e-4f) seen_003 = 1;
            } else if ((i32_ty && a[0] == i32_ty) || (u32_ty && a[0] == u32_ty)) {
                h.ival[a[1]] = (int32_t)a[2], h.has_ival[a[1]] = 1;
            }
        } else if (op == SPV_OP_VARIABLE && len >= 4 && a[1] < bound) {
            if (a[2] == SPV_SC_OUTPUT) h.cls[a[1]] = 1;
            else if (a[2] == SPV_SC_UNIFORM) h.cls[a[1]] = 2;
            else if (a[2] == SPV_SC_FUNCTION) h.cls[a[1]] = 3;
            if (a[1] == sup_var && a[2] == SPV_SC_UNIFORM) sup_ptr_ty = a[0];
        } else if (op == SPV_OP_FUNCTION && !first_func) {
            first_func = i;
        }
        if (op == SPV_OP_FUNCTION && len >= 5 && a[1] == main_id && main_id) main_start = i;
        else if (op == SPV_OP_FUNCTION_END && main_start && !main_end) main_end = i + len;
        if (op == SPV_OP_STORE && len >= 3 && a[0] < bound && h.cls[a[0]] == 3) {
            h.sval[a[0]] = a[1];
            if (h.scnt[a[0]] < 2) h.scnt[a[0]]++;
        }
        i += len;
    }

    // Fast exit for non-ACES shaders.
    if (!seen_251 || !seen_003 || !h.glsl || !f32_ty || !vec4_ty || !main_start || !main_end || !first_func)
        goto fail_0;

    // Locate SupportBuffer struct type and verify it ends at or before byte VK_HDR_PARAMS_AT.
    if (!sup_var || !sup_ptr_ty || !h.def[sup_ptr_ty]) goto fail_0;
    {
        const uint32_t *pt = w + h.def[sup_ptr_ty];
        if ((pt[0] & 0xffff) != SPV_OP_TYPE_POINTER || (pt[0] >> 16) != 4 || pt[2] != SPV_SC_UNIFORM) goto fail_0;
        sup_struct = pt[3];
    }
    if (!sup_struct || sup_struct >= bound || !h.def[sup_struct]) goto fail_0;
    sup_struct_pos = h.def[sup_struct];
    if ((w[sup_struct_pos] & 0xffff) != SPV_OP_TYPE_STRUCT || vec4_pos >= sup_struct_pos) goto fail_0;
    uint32_t sup_members = (w[sup_struct_pos] >> 16) - 2u;
    uint32_t max_off = 0;
    for (uint64_t i = 5; i < first_func; i += w[i] >> 16) {
        if ((w[i] & 0xffff) == SPV_OP_MEMBER_DECORATE && (w[i] >> 16) == 5 &&
            w[i + 1] == sup_struct && w[i + 3] == SPV_DEC_OFFSET && w[i + 4] > max_off)
            max_off = w[i + 4];
    }
    if (max_off >= VK_HDR_PARAMS_AT) goto fail_0;   // already extended or unexpected layout

    if (!build_dominators(&h, main_start, main_end)) goto fail_0;

    // Collect all ACES clamp sites in main(): clamp(ACES(x), 0, 1) and the Function variable each stores into (if any).
    enum { MAX_ACES_SITES = 16 };
    uint64_t site_pos[MAX_ACES_SITES] = {0};
    uint32_t site_id[MAX_ACES_SITES] = {0}, site_x[MAX_ACES_SITES] = {0}, site_var[MAX_ACES_SITES] = {0};
    int nsite = 0;
    for (uint64_t i = main_start; i < main_end && nsite < MAX_ACES_SITES; i += w[i] >> 16) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op != SPV_OP_EXT_INST || len != 8 || w[i + 3] != h.glsl || !is_clamp(w[i + 4])) continue;
        uint32_t cid = w[i + 2], tx = 0;
        if (!match_aces_clamp(&h, cid, &tx)) continue;
        site_pos[nsite] = i, site_id[nsite] = cid, site_x[nsite] = tx, site_var[nsite] = 0;
        for (uint64_t j = i + len; j < main_end; j += w[j] >> 16) {
            if ((w[j] & 0xffff) == SPV_OP_STORE && (w[j] >> 16) >= 3 && w[j + 2] == cid &&
                w[j + 1] < bound && h.cls[w[j + 1]] == 3) {
                site_var[nsite] = w[j + 1];
                break;
            }
        }
        nsite++;
    }
    if (nsite < 3) goto fail_0;

    // Match the 3 ACES-clamped RGB stores into a single Output variable inside main().
    // Mode 1: three scalar OpStores through OpAccessChain(out_var, 0/1/2), either storing a site directly or loading
    //         a site's Function variable between the AccessChain and the OpStore (covers translator NaN sanitizers).
    // Mode 2: one vec4 OpStore of OpCompositeConstruct(r, g, b, a) into out_var.
    int store_mode = 0;
    uint32_t out_var = 0, x_in[3] = {0}, ptr_ch[3] = {0}, vec_alpha = 0;
    uint64_t store_pos[3] = {0}, chain_pos[3] = {0};
    int32_t store_blk[3] = {-1, -1, -1};
    int chan_site[3] = {-1, -1, -1};
    int32_t cur_blk = -1;
    uint32_t cand_var[8] = {0};
    uint64_t cand_pos[8][3] = {{0}}, cand_chain[8][3] = {{0}};
    uint32_t cand_ptr[8][3] = {{0}};
    int cand_site[8][3];
    int32_t cand_blk[8][3];
    int cand_mask[8] = {0}, ncand = 0;
    memset(cand_site, -1, sizeof cand_site);
    memset(cand_blk, -1, sizeof cand_blk);

    for (uint64_t i = main_start; i < main_end; i += w[i] >> 16) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == SPV_OP_LABEL) { cur_blk++; continue; }
        if (op != SPV_OP_STORE || len < 3) continue;
        uint32_t dst = w[i + 1], val = w[i + 2];
        if (dst >= bound || val >= bound) continue;
        // Case 2: direct vec4 OpStore to an Output variable.
        if (h.cls[dst] == 1 && h.def[val]) {
            const uint32_t *cc = w + h.def[val];
            if ((cc[0] & 0xffff) == SPV_OP_COMPOSITE_CONSTRUCT && (cc[0] >> 16) == 7) {
                int sidx[3] = {-1, -1, -1};
                for (int c = 0; c < 3; c++) {
                    uint32_t u = native_compose ? compose_fit(&h, cc[3 + c]) : unwrap_func_var(&h, cc[3 + c]);
                    for (int s = 0; s < nsite; s++)
                        if (site_id[s] == u) { sidx[c] = s; break; }
                }
                if (sidx[0] >= 0 && sidx[1] >= 0 && sidx[2] >= 0 &&
                    sidx[0] != sidx[1] && sidx[1] != sidx[2] && sidx[0] != sidx[2]) {
                    store_mode = 2, out_var = dst;
                    for (int c = 0; c < 3; c++) {
                        chan_site[c] = sidx[c];
                        x_in[c] = site_x[sidx[c]];
                        store_pos[c] = chain_pos[c] = i;
                        store_blk[c] = cur_blk;
                    }
                    vec_alpha = cc[6];
                    break;
                }
            }
        }
        // Case 1: scalar OpStore through OpAccessChain(out_var, const_0/1/2).
        if (h.def[dst]) {
            const uint32_t *ac = w + h.def[dst];
            if ((ac[0] & 0xffff) == SPV_OP_ACCESS_CHAIN && (ac[0] >> 16) == 5 &&
                ac[3] < bound && h.cls[ac[3]] == 1 && ac[4] < bound && h.has_ival[ac[4]]) {
                int32_t ch = h.ival[ac[4]];
                if (ch < 0 || ch >= 3) continue;
                int matched = -1;
                uint32_t uval = native_compose ? compose_fit(&h, val) : unwrap_func_var(&h, val);
                for (int s = 0; s < nsite; s++)
                    if (site_id[s] == uval) { matched = s; break; }
                if (matched < 0 && !native_compose) {
                    // Translator FP16/NaN sanitizer loads site_var between AccessChain and OpStore.
                    for (uint64_t j = h.def[dst]; j < i; j += w[j] >> 16) {
                        if ((w[j] & 0xffff) == SPV_OP_LOAD && (w[j] >> 16) >= 4) {
                            for (int s = 0; s < nsite; s++)
                                if (site_var[s] && w[j + 3] == site_var[s]) matched = s;
                        }
                    }
                }
                if (matched < 0) continue;
                int idx = -1;
                for (int k = 0; k < ncand; k++) if (cand_var[k] == ac[3]) idx = k;
                if (idx < 0 && ncand < 8) {
                    idx = ncand++;
                    cand_var[idx] = ac[3];
                    cand_mask[idx] = 0;
                }
                if (idx >= 0) {
                    if (cand_mask[idx] & (1 << ch)) { cand_mask[idx] = -1; continue; }   // channel stored twice
                    if (cand_mask[idx] < 0) continue;
                    cand_pos[idx][ch] = i, cand_chain[idx][ch] = h.def[dst], cand_ptr[idx][ch] = dst;
                    cand_site[idx][ch] = matched, cand_blk[idx][ch] = cur_blk, cand_mask[idx] |= 1 << ch;
                }
            }
        }
    }
    if (!store_mode) {
        for (int k = 0; k < ncand; k++) {
            if (cand_mask[k] == 7 && cand_blk[k][0] >= 0 &&
                cand_blk[k][0] == cand_blk[k][1] && cand_blk[k][1] == cand_blk[k][2] &&
                cand_site[k][0] != cand_site[k][1] && cand_site[k][1] != cand_site[k][2] &&
                cand_site[k][0] != cand_site[k][2]) {
                store_mode = 1, out_var = cand_var[k];
                for (int c = 0; c < 3; c++) {
                    chan_site[c] = cand_site[k][c];
                    x_in[c] = site_x[chan_site[c]];
                    ptr_ch[c] = cand_ptr[k][c];
                    store_pos[c] = cand_pos[k][c];
                    chain_pos[c] = cand_chain[k][c];
                    store_blk[c] = cand_blk[k][c];
                }
                break;
            }
        }
    }
    if (!store_mode) goto fail_0;

    uint64_t grade_pos = 0;
    uint32_t grade_in[3] = {0};
    if (grade_only) {
        grade_pos = site_pos[chan_site[0]];
        for (int c = 1; c < 3; c++)
            if (site_pos[chan_site[c]] < grade_pos) grade_pos = site_pos[chan_site[c]];
        const int32_t block = h.def_blk[w[grade_pos + 2]];
        for (int c = 0; c < 3; c++) {
            // The guest can reload each channel immediately before its fit, after all three values were computed.
            grade_in[c] = dominates(&h, x_in[c], block, grade_pos) ? x_in[c] : unwrap_func_var(&h, x_in[c]);
            if (!dominates(&h, grade_in[c], block, grade_pos) ||
                h.def_blk[site_id[chan_site[c]]] != block) goto fail_0;
        }
    }

    // Insertion point:
    // - If all 3 ACES clamp sites & x_in[] dominate the earliest channel AccessChain (standard layout), insert the
    //   nx_tonemap call right before that AccessChain and replace the 3 OpStore values in place (inline_replace = 1).
    // - Otherwise (interleaved compute+store), defer the first 2 stores and emit the call + all 3 stores at max(store_pos).
    int inline_replace = 0;
    uint64_t ins_pos = store_pos[0];
    if (store_pos[1] > ins_pos) ins_pos = store_pos[1];
    if (store_pos[2] > ins_pos) ins_pos = store_pos[2];
    int32_t ins_blk = store_blk[0];
    if (store_mode == 1) {
        uint64_t min_ch = chain_pos[0];
        if (chain_pos[1] < min_ch) min_ch = chain_pos[1];
        if (chain_pos[2] < min_ch) min_ch = chain_pos[2];
        int all_before = 1;
        for (int c = 0; c < 3; c++)
            if (!dominates(&h, x_in[c], ins_blk, min_ch) || site_pos[chan_site[c]] >= min_ch) all_before = 0;
        if (all_before) ins_pos = min_ch, inline_replace = 1;
    }
    for (int c = 0; c < 3; c++)
        if (!dominates(&h, x_in[c], ins_blk, ins_pos)) goto fail_err;
    if (store_mode == 1 && !inline_replace) {
        for (int c = 0; c < 3; c++)
            if (!dominates(&h, ptr_ch[c], ins_blk, ins_pos)) goto fail_err;
    } else if (store_mode == 2 && !dominates(&h, vec_alpha, ins_blk, ins_pos)) {
        goto fail_err;
    }

    // Extract uniform exposure gain and pre-local-exposure scene; fall back to vec4(x, 1.0) if not dominating.
    Exposure ex;
    int has_exp = extract_exposure(&h, x_in, &ex);
    if (has_exp) {
        if (!dominates(&h, ex.gain, ins_blk, ins_pos)) has_exp = 0;
        for (int c = 0; c < 3 && has_exp; c++) {
            if (!dominates(&h, ex.exposed[c], ins_blk, ins_pos)) has_exp = 0;
            if (ex.add[c] && !dominates(&h, ex.add[c], ins_blk, ins_pos)) has_exp = 0;
        }
    }

    // The game's bloom, in each channel where it is added to the scene: its term is multiplied there by p5.x
    // (REKARINA_HDR_BLOOM), before the gain, so everything after it (the local exposure's metering, both tone mappers)
    // sees the same bloom. Used only when all three channels are found, in three different instructions.
    uint64_t bloom_at[3] = {0};
    uint32_t bloom_word[3] = {0};
    unsigned expanded = 0;
    if (native_compose) {
        for (unsigned c = 0; c < 3; c++) {
            uint32_t value = w[store_pos[c] + 2];
            if (store_mode == 2) value = w[h.def[value] + 3 + c];
            expanded += native_hdr_fit(&h, value) != 0;
        }
        if (expanded && expanded != 3) goto fail_0;
    }
    int has_bloom = has_exp;
    for (int c = 0; c < 3 && has_bloom; c++)
        has_bloom = ex.scene[c] && bloom_operand(&h, ex.scene[c], &bloom_at[c], &bloom_word[c]);
    if (!has_bloom && !expanded && has_exp && !ex.has_local_term && direct_bloom_operands(&h, &ex, bloom_at, bloom_word)) {
        has_bloom = 1;
        meter_binding = 0;   // the pause view must not meter the gameplay eye adaptation
    }
    if (has_bloom && (bloom_at[0] == bloom_at[1] || bloom_at[1] == bloom_at[2] || bloom_at[0] == bloom_at[2]))
        has_bloom = 0;
    int has_local = has_exp && ex.local && w[h.def[ex.local] + 1] == f32_ty;
    for (int c = 0; c < 3 && has_local; c++)
        has_local = h.def[ex.local] < ex.local_at[c] &&
                    dominates(&h, ex.local, h.def_blk[w[ex.local_at[c] + 2]], ex.local_at[c]);
    // The HDR compose with direct uniform gain has bloom but no local exposure term.
    if ((effects_only || mode == SPLICE_HDR_SCENE) && (!has_bloom || (ex.has_local_term && !has_local))) goto fail_0;
    if (grade_only || !has_exp || !ex.scene[0] || !ex.scene[1] || !ex.scene[2]) meter_binding = 0;
    if (trace) *trace = (SpliceTrace){has_exp, has_bloom, has_local, meter_binding};

    // Inspect the tone mapper library (hdr_tonemap_spv) and map/deduplicate its IDs.
    const uint32_t *lw = hdr_tonemap_spv;
    uint64_t ln = sizeof hdr_tonemap_spv / 4;
    if (ln < 5 || lw[0] != SPV_MAGIC) goto fail_err;
    uint32_t lbound = lw[3];
    uint32_t *lmap = calloc(lbound, sizeof *lmap);
    uint8_t *lskip = calloc(lbound, 1);   // library type/var IDs belonging only to main()
    if (!lmap || !lskip) { free(lmap); free(lskip); goto fail_err; }

    uint32_t lmain = 0, lnx = 0, lnx_ftype = 0, lgrade = 0, lmeasure = 0, lmeter = 0, lmeter_struct = 0, lmeter_array = 0;
    uint64_t lfirst_func = 0;
    for (uint64_t i = 5; i < ln;) {
        uint32_t op = lw[i] & 0xffff, len = lw[i] >> 16;
        if (!len || i + len > ln) { free(lmap); free(lskip); goto fail_err; }
        const uint32_t *a = lw + i + 1;
        if (op == SPV_OP_EXT_INST_IMPORT && len >= 3 && name_is(a, len, "GLSL.std.450")) lmap[a[0]] = h.glsl;
        else if (op == SPV_OP_ENTRY_POINT && len >= 4) lmain = a[1];
        else if (op == SPV_OP_NAME && len >= 3 && name_prefix(a, len, "nx_tonemap(")) lnx = a[0];
        else if (op == SPV_OP_NAME && len >= 3 && name_prefix(a, len, "graded_aces(")) lgrade = a[0];
        else if (op == SPV_OP_NAME && len >= 3 && name_prefix(a, len, "hdr_measure(")) lmeasure = a[0];
        else if (op == SPV_OP_NAME && len >= 3 && name_is(a, len, "hdr_meter")) lmeter = a[0];
        else if (op == SPV_OP_NAME && len >= 3 && name_is(a, len, "HdrAdaptationMeter")) lmeter_struct = a[0];
        else if (op == SPV_OP_TYPE_STRUCT && len == 5 && a[0] == lmeter_struct) lmeter_array = a[3];
        else if (op == SPV_OP_TYPE_POINTER && len == 4 && (a[1] == SPV_SC_INPUT || a[1] == SPV_SC_OUTPUT))
            lskip[a[0]] = 1;
        else if (op == SPV_OP_VARIABLE && len >= 4 && a[2] != SPV_SC_FUNCTION)
            lskip[a[1]] = 1;
        else if (op == SPV_OP_FUNCTION && !lfirst_func) lfirst_func = i;
        if (op == SPV_OP_FUNCTION && len >= 5 && a[1] == lnx && lnx) lnx_ftype = a[3];
        i += len;
    }
    if (!lnx || !lnx_ftype || !lfirst_func || (grade_only && !lgrade)) { free(lmap); free(lskip); goto fail_err; }
    if (!lmeasure || !lmeter || !lmeter_struct || !lmeter_array) { free(lmap); free(lskip); goto fail_err; }

    // Read nx_tonemap's parameter types (vec3 return, pointer to vec3, then pointers to vec4).
    uint32_t l_vec3 = 0, l_pfunc_v3 = 0, l_pfunc_v4 = 0;
    for (uint64_t i = 5; i < lfirst_func; i += lw[i] >> 16) {
        if ((lw[i] & 0xffff) == SPV_OP_TYPE_FUNCTION && lw[i + 1] == lnx_ftype && (lw[i] >> 16) == 11) {
            l_vec3 = lw[i + 2];
            l_pfunc_v3 = lw[i + 3];
            l_pfunc_v4 = lw[i + 4];
            break;
        }
    }
    if (!l_vec3 || !l_pfunc_v3 || !l_pfunc_v4) { free(lmap); free(lskip); goto fail_err; }

    // Build the spliced module.
    uint32_t next_id = bound;
    int add_frag_coord = meter_binding && !frag_coord;
    if (add_frag_coord) frag_coord = next_id++;
    if (meter_binding) {
        lmap[lmeter] = next_id++;
        lmap[lmeter_struct] = next_id++;
        lmap[lmeter_array] = next_id++;
    } else {
        lskip[lmeter_struct] = lskip[lmeter_array] = 1;
    }
    SpvBuf out = {0};
    // Header (word 3 = bound, patched at the end).
    spv_emit(&out, w, 5);

    // 1. Copy host instructions up to the end of Annotations (appending sup_var to OpEntryPoint if needed for SPIR-V
    //    1.4+), then append the VK_HDR_PARAMS SupportBuffer MemberDecorates.
    for (uint64_t i = 5; i < annot_end;) {
        uint32_t len = w[i] >> 16;
        if (i == ep_pos && ep_pos) {
            // Find start of interface IDs after the null-terminated entry point name string.
            uint32_t j = 3;
            while (j < len) {
                uint32_t word = w[i + j++];
                if (!(word & 0xffu) || !(word & 0xff00u) || !(word & 0xff0000u) || !(word & 0xff000000u)) break;
            }
            int listed = 0;
            for (uint32_t k = j; k < len; k++)
                if (w[i + k] == sup_var) listed = 1;
            uint32_t extra[3], n_extra = 0;
            if (w[1] >= 0x00010400u) {
                if (!listed) extra[n_extra++] = sup_var;
                if (meter_binding) extra[n_extra++] = lmap[lmeter];
            }
            if (add_frag_coord) extra[n_extra++] = frag_coord;
            if (n_extra) {
                uint32_t nh = ((len + n_extra) << 16) | SPV_OP_ENTRY_POINT;
                spv_emit(&out, &nh, 1);
                spv_emit(&out, w + i + 1, len - 1);
                spv_emit(&out, extra, n_extra);
            } else {
                spv_emit(&out, w + i, len);
            }
        } else {
            spv_emit(&out, w + i, len);
        }
        i += len;
    }
    for (uint32_t k = 0; k < VK_HDR_PARAMS; k++)
        SPV_INS(&out, SPV_OP_MEMBER_DECORATE, sup_struct, sup_members + k, SPV_DEC_OFFSET, VK_HDR_PARAMS_AT + 16u * k);
    if (meter_binding) {
        for (uint64_t i = 5; i < lfirst_func; i += lw[i] >> 16) {
            uint32_t len = lw[i] >> 16, op = lw[i] & 0xffff;
            uint32_t id = lw[i + 1];
            if ((op == SPV_OP_DECORATE || op == SPV_OP_MEMBER_DECORATE) &&
                (id == lmeter_struct || id == lmeter_array)) {
                spv_emit(&out, lw + i, 1);
                spv_emit(&out, &lmap[id], 1);
                spv_emit(&out, lw + i + 2, len - 2);
            }
        }
        SPV_INS(&out, SPV_OP_DECORATE, lmap[lmeter], SPV_DEC_SET, 0u);
        SPV_INS(&out, SPV_OP_DECORATE, lmap[lmeter], SPV_DEC_BINDING, (uint32_t)meter_binding);
        if (add_frag_coord) SPV_INS(&out, SPV_OP_DECORATE, frag_coord, DEC_BUILTIN, BUILTIN_FRAG_COORD);
    }

    // 2. Copy host Types / Constants / Global Variables up to first_func, extending SupportBuffer OpTypeStruct.
    for (uint64_t i = annot_end; i < first_func;) {
        uint32_t len = w[i] >> 16;
        if (i == sup_struct_pos) {
            uint32_t nlen = len + VK_HDR_PARAMS;
            uint32_t hdr = (nlen << 16) | SPV_OP_TYPE_STRUCT;
            spv_emit(&out, &hdr, 1);
            spv_emit(&out, w + i + 1, len - 1);
            for (uint32_t k = 0; k < VK_HDR_PARAMS; k++) spv_emit(&out, &vec4_ty, 1);
        } else {
            spv_emit(&out, w + i, len);
        }
        i += len;
    }

    // 3. Merge library Types and Constants into the Types/Constants section, deduplicating non-aggregate types.
    for (uint64_t i = 5; i < lfirst_func;) {
        uint32_t op = lw[i] & 0xffff, len = lw[i] >> 16;
        if (spv_is_type(op) && len >= 2 && lw[i + 1] < lbound && !lskip[lw[i + 1]]) {
            uint32_t tmp[64];
            if (len > 64) { free(lmap); free(lskip); free(out.w); goto fail_err; }
            tmp[0] = lw[i];
            int ok = 1;
            for (uint32_t j = 2; j < len && ok; j++) {
                int lit = spv_op_literal(op, j);
                if (lit < 0) ok = 0;
                else if (lit) tmp[j] = lw[i + j];
                else if (lw[i + j] < lbound && lmap[lw[i + j]]) tmp[j] = lmap[lw[i + j]];
                else ok = 0;
            }
            if (ok) {
                uint32_t reserved = lmap[lw[i + 1]];
                uint32_t exist = reserved ? 0 : find_host_type(out.w, out.n, tmp, len);
                if (exist) {
                    lmap[lw[i + 1]] = exist;
                } else {
                    uint32_t nid = reserved ? reserved : next_id++;
                    lmap[lw[i + 1]] = nid;
                    tmp[1] = nid;
                    spv_emit(&out, tmp, len);
                }
            }
        } else if (op == SPV_OP_VARIABLE && meter_binding && lw[i + 2] == lmeter) {
            SPV_INS(&out, SPV_OP_VARIABLE, lmap[lw[i + 1]], lmap[lmeter], SC_STORAGE_BUFFER);
        } else if (spv_is_const(op) && len >= 3 && lw[i + 2] < lbound) {
            uint32_t nid = next_id++;
            lmap[lw[i + 2]] = nid;
            if (!spv_emit_mapped(&out, lw + i, lmap, lbound)) {
                free(lmap); free(lskip); free(out.w); goto fail_err;
            }
        }
        i += len;
    }

    // Ensure helper types & constants needed by the call site in main() exist in the module.
    uint32_t m_vec3 = lmap[l_vec3], m_pfunc_v3 = lmap[l_pfunc_v3], m_pfunc_v4 = lmap[l_pfunc_v4];
    if (!m_vec3 || !m_pfunc_v3 || !m_pfunc_v4) { free(lmap); free(lskip); free(out.w); goto fail_err; }
    if (!ptr_u_vec4) {
        ptr_u_vec4 = next_id++;
        SPV_INS(&out, SPV_OP_TYPE_POINTER, ptr_u_vec4, SPV_SC_UNIFORM, vec4_ty);
    }
    if (add_frag_coord) {
        uint32_t probe[] = {4u << 16 | SPV_OP_TYPE_POINTER, 0, SPV_SC_INPUT, vec4_ty};
        uint32_t ptr = find_host_type(out.w, out.n, probe, 4);
        if (!ptr) {
            ptr = next_id++;
            SPV_INS(&out, SPV_OP_TYPE_POINTER, ptr, SPV_SC_INPUT, vec4_ty);
        }
        SPV_INS(&out, SPV_OP_VARIABLE, ptr, frag_coord, SPV_SC_INPUT);
    }
    uint32_t empty_adaptation = next_id++;
    SPV_INS(&out, SPV_OP_CONSTANT_NULL, vec4_ty, empty_adaptation);
    uint32_t idx_ty = i32_ty ? i32_ty : u32_ty;
    if (!idx_ty) {
        // Check if type merging added signed 32-bit int.
        uint32_t probe[4] = {4u << 16 | SPV_OP_TYPE_INT, 0, 32, 1};
        idx_ty = find_host_type(out.w, out.n, probe, 4);
        if (!idx_ty) {
            idx_ty = next_id++;
            SPV_INS(&out, SPV_OP_TYPE_INT, idx_ty, 32u, 1u);
        }
    }
    uint32_t c_idx[VK_HDR_PARAMS];
    for (uint32_t k = 0; k < VK_HDR_PARAMS; k++) {
        c_idx[k] = next_id++;
        SPV_INS(&out, SPV_OP_CONSTANT, idx_ty, c_idx[k], sup_members + k);
    }
    uint32_t c_one = 0, c_zero = 0, bool_ty = 0;
    if (!has_exp || has_local) {
        c_one = next_id++;
        SPV_INS(&out, SPV_OP_CONSTANT, f32_ty, c_one, 0x3f800000u);
    }
    if (has_local) {
        uint32_t probe[] = {2u << 16 | SPV_OP_TYPE_BOOL, 0};
        bool_ty = find_host_type(out.w, out.n, probe, 2);
        if (!bool_ty) {
            bool_ty = next_id++;
            SPV_INS(&out, SPV_OP_TYPE_BOOL, bool_ty);
        }
        c_zero = next_id++;
        SPV_INS(&out, SPV_OP_CONSTANT, f32_ty, c_zero, 0u);
    }

    // Pre-allocate IDs for all non-main library functions and their internal result IDs (handles forward calls/labels).
    {
        int in_main = 0;
        for (uint64_t i = lfirst_func; i < ln;) {
            uint32_t op = lw[i] & 0xffff, len = lw[i] >> 16;
            if (op == SPV_OP_FUNCTION && len >= 5)
                in_main = lw[i + 2] == lmain || (!meter_binding && lw[i + 2] == lmeasure);
            if (!in_main) {
                uint32_t rw = spv_res_word(op, len);
                if (rw && lw[i + rw] < lbound && !lmap[lw[i + rw]]) lmap[lw[i + rw]] = next_id++;
            }
            if (op == SPV_OP_FUNCTION_END) in_main = 0;
            i += len;
        }
    }
    uint32_t m_nx = lmap[lnx];
    if (!m_nx) { free(lmap); free(lskip); free(out.w); goto fail_err; }

    // 4. Copy host functions, patching main():
    //    - declare 7 local Function variables after main's entry OpLabel
    //    - emit the nx_tonemap call at ins_pos and store its RGB result into the 3 output channels.
    uint32_t var_x = next_id++, var_sc = next_id++, var_p[NX_TONEMAP_PARAMS];
    uint32_t var_adapt = next_id++, var_frag = meter_binding ? next_id++ : 0;
    for (uint32_t k = 0; k < NX_TONEMAP_PARAMS; k++) var_p[k] = next_id++;
    uint32_t out_ch[3] = {0}, local_factor = 0, graded = 0;
    int entry_patched = 0;
    for (uint64_t i = first_func; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!effects_only && i > main_start && i < main_end && op == SPV_OP_LABEL && !entry_patched) {
            spv_emit(&out, w + i, len);
            SPV_INS(&out, SPV_OP_VARIABLE, m_pfunc_v3, var_x, SPV_SC_FUNCTION);
            SPV_INS(&out, SPV_OP_VARIABLE, m_pfunc_v4, var_sc, SPV_SC_FUNCTION);
            SPV_INS(&out, SPV_OP_VARIABLE, m_pfunc_v4, var_adapt, SPV_SC_FUNCTION);
            if (meter_binding) SPV_INS(&out, SPV_OP_VARIABLE, m_pfunc_v4, var_frag, SPV_SC_FUNCTION);
            for (uint32_t k = 0; k < NX_TONEMAP_PARAMS; k++)
                SPV_INS(&out, SPV_OP_VARIABLE, m_pfunc_v4, var_p[k], SPV_SC_FUNCTION);
            entry_patched = 1;
            i += len;
            continue;
        }
        if (grade_only && i == grade_pos) {
            uint32_t x = next_id++;
            SPV_INS(&out, SPV_OP_COMPOSITE_CONSTRUCT, m_vec3, x, grade_in[0], grade_in[1], grade_in[2]);
            SPV_INS(&out, SPV_OP_STORE, var_x, x);
            for (unsigned k = 1; k <= 2; k++) {
                uint32_t ptr = next_id++, value = next_id++;
                SPV_INS(&out, SPV_OP_ACCESS_CHAIN, ptr_u_vec4, ptr, sup_var, c_idx[k]);
                SPV_INS(&out, SPV_OP_LOAD, vec4_ty, value, ptr);
                SPV_INS(&out, SPV_OP_STORE, var_p[k], value);
            }
            graded = next_id++;
            SPV_INS(&out, SPV_OP_FUNCTION_CALL, m_vec3, graded, lmap[lgrade], var_x, var_p[1], var_p[2]);
        }
        if (grade_only) {
            int channel = i == site_pos[chan_site[0]] ? 0 : i == site_pos[chan_site[1]] ? 1 : i == site_pos[chan_site[2]] ? 2 : -1;
            if (channel >= 0) {
                SPV_INS(&out, SPV_OP_COMPOSITE_EXTRACT, f32_ty, site_id[chan_site[channel]], graded, (uint32_t)channel);
                i += len;
                continue;
            }
        }
        if (!effects_only && !grade_only && store_mode == 1 && !inline_replace &&
            (i == store_pos[0] || i == store_pos[1] || i == store_pos[2]) && i != ins_pos) {
            i += len;   // deferred to ins_pos
            continue;
        }
        if (!effects_only && !grade_only && i == ins_pos) {
            uint32_t x_vec = next_id++;
            SPV_INS(&out, SPV_OP_COMPOSITE_CONSTRUCT, m_vec3, x_vec, x_in[0], x_in[1], x_in[2]);
            SPV_INS(&out, SPV_OP_STORE, var_x, x_vec);

            uint32_t sc_ch[3];
            if (has_exp && !native_compose) {
                for (int c = 0; c < 3; c++) {
                    if (ex.add[c]) {
                        sc_ch[c] = next_id++;
                        SPV_INS(&out, SPV_OP_FADD, f32_ty, sc_ch[c], ex.exposed[c], ex.add[c]);
                    } else {
                        sc_ch[c] = ex.exposed[c];
                    }
                }
            } else {
                for (int c = 0; c < 3; c++) sc_ch[c] = x_in[c];
            }
            uint32_t sc_vec = next_id++;
            SPV_INS(&out, SPV_OP_COMPOSITE_CONSTRUCT, vec4_ty, sc_vec, sc_ch[0], sc_ch[1], sc_ch[2],
                    has_exp ? ex.gain : c_one);
            SPV_INS(&out, SPV_OP_STORE, var_sc, sc_vec);

            for (uint32_t k = 0; k < NX_TONEMAP_PARAMS; k++) {
                uint32_t ptr_k = next_id++, val_k = next_id++;
                SPV_INS(&out, SPV_OP_ACCESS_CHAIN, ptr_u_vec4, ptr_k, sup_var, c_idx[k]);
                SPV_INS(&out, SPV_OP_LOAD, vec4_ty, val_k, ptr_k);
                SPV_INS(&out, SPV_OP_STORE, var_p[k], val_k);
            }
            uint32_t adaptation = empty_adaptation;
            if (meter_binding) {
                uint32_t coord = next_id++;
                SPV_INS(&out, SPV_OP_LOAD, vec4_ty, coord, frag_coord);
                SPV_INS(&out, SPV_OP_STORE, var_frag, coord);
                adaptation = next_id++;
                SPV_INS(&out, SPV_OP_FUNCTION_CALL, vec4_ty, adaptation, lmap[lmeasure], var_sc, var_frag, var_p[4]);
            }
            SPV_INS(&out, SPV_OP_STORE, var_adapt, adaptation);
            uint32_t res = next_id++;
            SPV_INS(&out, SPV_OP_FUNCTION_CALL, m_vec3, res, m_nx, var_x, var_sc,
                    var_p[0], var_p[1], var_p[2], var_p[3], var_p[4], var_adapt);
            for (int c = 0; c < 3; c++) {
                out_ch[c] = next_id++;
                SPV_INS(&out, SPV_OP_COMPOSITE_EXTRACT, f32_ty, out_ch[c], res, (uint32_t)c);
            }
            if (inline_replace) {
                spv_emit(&out, w + i, len);   // emit the original AccessChain after the call
            } else if (store_mode == 1) {
                for (int c = 0; c < 3; c++) SPV_INS(&out, SPV_OP_STORE, ptr_ch[c], out_ch[c]);
            } else {
                uint32_t out_v4 = next_id++;
                SPV_INS(&out, SPV_OP_COMPOSITE_CONSTRUCT, vec4_ty, out_v4, out_ch[0], out_ch[1], out_ch[2], vec_alpha);
                SPV_INS(&out, SPV_OP_STORE, out_var, out_v4);
            }
            i += len;
            continue;
        }
        if (!effects_only && !grade_only && inline_replace) {
            int ch = i == store_pos[0] ? 0 : i == store_pos[1] ? 1 : i == store_pos[2] ? 2 : -1;
            if (ch >= 0) {
                SPV_INS(&out, SPV_OP_STORE, ptr_ch[ch], out_ch[ch]);
                i += len;
                continue;
            }
        }
        if (has_local && i == h.def[ex.local]) {
            // Scale in stops. At zero use no multiplier, and at one keep the original Exp2 exactly.
            uint32_t ptr = next_id++, p5 = next_id++, strength = next_id++, stops = next_id++, scaled = next_id++;
            uint32_t off = next_id++, neutral = next_id++, enabled = next_id++;
            local_factor = next_id++;
            spv_emit(&out, w + i, len);
            SPV_INS(&out, SPV_OP_ACCESS_CHAIN, ptr_u_vec4, ptr, sup_var, c_idx[5]);
            SPV_INS(&out, SPV_OP_LOAD, vec4_ty, p5, ptr);
            SPV_INS(&out, SPV_OP_COMPOSITE_EXTRACT, f32_ty, strength, p5, 1u);
            SPV_INS(&out, SPV_OP_FMUL, f32_ty, stops, w[i + 5], strength);
            SPV_INS(&out, SPV_OP_EXT_INST, f32_ty, scaled, h.glsl, GLSL_EXP2, stops);
            SPV_INS(&out, SPV_OP_FORD_EQUAL, bool_ty, off, strength, c_zero);
            SPV_INS(&out, SPV_OP_FORD_EQUAL, bool_ty, neutral, strength, c_one);
            SPV_INS(&out, SPV_OP_SELECT, f32_ty, enabled, neutral, ex.local, scaled);
            SPV_INS(&out, SPV_OP_SELECT, f32_ty, local_factor, off, c_one, enabled);
            i += len;
            continue;
        }
        int local_ch = !has_local ? -1 : i == ex.local_at[0] ? 0 : i == ex.local_at[1] ? 1 : i == ex.local_at[2] ? 2 : -1;
        if (local_ch >= 0) {
            uint32_t ins[8];
            memcpy(ins, w + i, len * sizeof *ins);
            ins[ex.local_word[local_ch]] = local_factor;
            spv_emit(&out, ins, len);
            i += len;
            continue;
        }
        int bloom_ch = !has_bloom ? -1 : i == bloom_at[0] ? 0 : i == bloom_at[1] ? 1 : i == bloom_at[2] ? 2 : -1;
        if (bloom_ch >= 0 && len <= 8) {   // the bloom term times p5.x, then the instruction that adds it
            uint32_t ptr = next_id++, p5 = next_id++, strength = next_id++, scaled = next_id++, ins[8];
            SPV_INS(&out, SPV_OP_ACCESS_CHAIN, ptr_u_vec4, ptr, sup_var, c_idx[5]);
            SPV_INS(&out, SPV_OP_LOAD, vec4_ty, p5, ptr);
            SPV_INS(&out, SPV_OP_COMPOSITE_EXTRACT, f32_ty, strength, p5, 0u);
            SPV_INS(&out, SPV_OP_FMUL, f32_ty, scaled, w[i + bloom_word[bloom_ch]], strength);
            memcpy(ins, w + i, len * sizeof *ins);
            ins[bloom_word[bloom_ch]] = scaled;
            spv_emit(&out, ins, len);
            i += len;
            continue;
        }
        spv_emit(&out, w + i, len);
        i += len;
    }

    // 5. Emit all non-main functions from hdr_tonemap_spv.
    if (!effects_only) {
        int in_main = 0;
        for (uint64_t i = lfirst_func; i < ln;) {
            uint32_t op = lw[i] & 0xffff, len = lw[i] >> 16;
            if (op == SPV_OP_FUNCTION && len >= 5)
                in_main = lw[i + 2] == lmain || (!meter_binding && lw[i + 2] == lmeasure);
            if (!in_main && !spv_emit_mapped(&out, lw + i, lmap, lbound)) {
                free(lmap); free(lskip); free(out.w); goto fail_err;
            }
            if (op == SPV_OP_FUNCTION_END) in_main = 0;
            i += len;
        }
    }

    free(lmap); free(lskip);
    free(h.def); free(h.def_blk); free(h.cls); free(h.sval); free(h.scnt);
    free(h.fval); free(h.has_fval); free(h.ival); free(h.has_ival); free(h.idom);
    if (out.err) { free(out.w); return -1; }
    out.w[3] = next_id;
    free(*spirv);
    *spirv = out.w;
    *words = out.n;
    return has_exp ? 2 : 1;

fail_err:
    free(h.def); free(h.def_blk); free(h.cls); free(h.sval); free(h.scnt);
    free(h.fval); free(h.has_fval); free(h.ival); free(h.has_ival); free(h.idom);
    return -1;

fail_0:
    free(h.def); free(h.def_blk); free(h.cls); free(h.sval); free(h.scnt);
    free(h.fval); free(h.has_fval); free(h.ival); free(h.has_ival); free(h.idom);
    return 0;
}

int vk_hdr_scene_splice(uint32_t **spirv, uint64_t *words, GfxRenodxVariant variant, int *meter_binding) {
    SpliceTrace trace;
    int mode = variant == GFX_RENODX_HDR ? SPLICE_HDR_SCENE : variant == GFX_RENODX_EFFECTS ? SPLICE_HDR_EFFECTS
               : variant == GFX_RENODX_SDR ? SPLICE_ACES : SPLICE_VANILLA_GRADE;
    int result = tonemap_splice(spirv, words, (variant == GFX_RENODX_HDR || variant == GFX_RENODX_SDR) &&
                                vk_hdr_adaptation_supported(), mode, &trace);
    *meter_binding = result > 0 ? trace.meter_binding : 0;
    return result;
}

// The packed AA path encodes in BT.2020. Only scene-color reads and final linear stores cross the gamut boundary.
static void aa_color_convert(SpvBuf *out, uint32_t *next, uint32_t v4, uint32_t v3, uint32_t f32,
                             uint32_t glsl, uint32_t zero3, const uint32_t rows[3], uint32_t dst, uint32_t src) {
    uint32_t rgb = (*next)++, alpha = (*next)++, converted = (*next)++, c[3];
    SPV_INS(out, SPV_OP_VECTOR_SHUFFLE, v3, rgb, src, src, 0, 1, 2);
    SPV_INS(out, SPV_OP_COMPOSITE_EXTRACT, f32, alpha, src, 3);
    for (unsigned i = 0; i < 3; ++i) {
        c[i] = (*next)++;
        SPV_INS(out, SPV_OP_DOT, f32, c[i], rgb, rows[i]);
    }
    SPV_INS(out, SPV_OP_COMPOSITE_CONSTRUCT, v3, converted, c[0], c[1], c[2]);
    if (zero3) {
        uint32_t nonnegative = (*next)++;
        SPV_INS(out, SPV_OP_EXT_INST, v3, nonnegative, glsl, GLSL_FMAX, converted, zero3);
        converted = nonnegative;
    }
    SPV_INS(out, SPV_OP_COMPOSITE_CONSTRUCT, v4, dst, converted, alpha);
}

// Only retype the edge-mask descriptor. Its fetches and gathers carry integer bits through float temporaries.
int vk_aa_mask_splice(uint32_t **spirv, uint64_t *words, unsigned binding) {
    enum { TYPE_IMAGE = 25, TYPE_SAMPLED_IMAGE = 27, IMAGE_GATHER = 96, IMAGE_QUERY_SIZE_LOD = 103 };
    const uint32_t *w = *spirv;
    if (*words < 5 || w[0] != 0x07230203 || !w[3] || w[3] > *words || w[3] > UINT32_MAX - 1024) return 0;
    uint32_t bound = w[3], next = bound, target = 0, u32 = 0, u4 = 0, f32 = 0, f4 = 0;
    uint32_t image = 0, sampled = 0, pointer = 0, sites = 0;
    const uint32_t **def = calloc(bound, sizeof *def);
    uint8_t *kind = calloc(bound, sizeof *kind);
    SpvBuf out = {0};
    if (!def || !kind) goto reject;
    spv_emit(&out, w, 5);
    for (uint64_t i = 5; i < *words;) {
        const uint32_t *a = w + i;
        uint32_t n = a[0] >> 16, op = a[0] & 0xffff;
        if (!n || i + n > *words) goto reject;
        if (op == SPV_OP_DECORATE && n == 4 && a[2] == SPV_DEC_BINDING && a[3] == binding) {
            if (target || !a[1] || a[1] >= bound) goto reject;
            target = a[1];
        }
        if (op == SPV_OP_TYPE_INT && n == 4 && a[2] == 32 && a[3] == 0) u32 = a[1];
        if (op == SPV_OP_TYPE_FLOAT && n == 3 && a[2] == 32) f32 = a[1];
        if (op == SPV_OP_TYPE_VECTOR && n == 4 && a[3] == 4) {
            if (a[2] == u32) u4 = a[1];
            if (a[2] == f32) f4 = a[1];
        }
        if ((op == TYPE_IMAGE || op == TYPE_SAMPLED_IMAGE || op == SPV_OP_TYPE_POINTER) && n >= 3 && a[1] < bound)
            def[a[1]] = a;
        if (op == SPV_OP_VARIABLE && n == 4 && target && a[2] == target) {
            const uint32_t *ptr = a[1] < bound ? def[a[1]] : NULL;
            if (!u32 || !f4 || a[3] != 0 || !ptr || ptr[0] != (4u << 16 | SPV_OP_TYPE_POINTER) || ptr[2] != 0) goto reject;
            const uint32_t *s = ptr[3] < bound ? def[ptr[3]] : NULL;
            if (!s || s[0] != (3u << 16 | TYPE_SAMPLED_IMAGE)) goto reject;
            const uint32_t *im = s[2] < bound ? def[s[2]] : NULL;
            if (!im || im[0] != (9u << 16 | TYPE_IMAGE) || im[2] != f32 || im[3] != 1 || im[4] || im[5] || im[6] ||
                im[7] != 1 || im[8]) goto reject;
            if (!u4) {
                u4 = next++;
                SPV_INS(&out, SPV_OP_TYPE_VECTOR, u4, u32, 4);
            }
            image = next++, sampled = next++, pointer = next++;
            SPV_INS(&out, TYPE_IMAGE, image, u32, 1, 0, 0, 0, 1, 0);
            SPV_INS(&out, TYPE_SAMPLED_IMAGE, sampled, image);
            SPV_INS(&out, SPV_OP_TYPE_POINTER, pointer, 0, sampled);
            SPV_INS(&out, SPV_OP_VARIABLE, pointer, target, 0);
        } else if (op == SPV_OP_LOAD && n == 4 && pointer && a[3] == target && a[2] < bound) {
            kind[a[2]] = 1;
            SPV_INS(&out, SPV_OP_LOAD, sampled, a[2], target);
        } else if (op == SPV_OP_IMAGE && n == 4 && a[2] < bound && a[3] < bound && kind[a[3]] == 1) {
            kind[a[2]] = 2;
            SPV_INS(&out, SPV_OP_IMAGE, image, a[2], a[3]);
        } else if (op >= SPV_OP_IMAGE_SAMPLE_IMPLICIT_LOD && op <= 107 && n >= 4 && a[3] < bound && kind[a[3]]) {
            if (op == IMAGE_QUERY_SIZE_LOD && kind[a[3]] == 2) spv_emit(&out, a, n);
            else {
                if (n < 5 || a[1] != f4 || next == UINT32_MAX ||
                    !((op == SPV_OP_IMAGE_FETCH && kind[a[3]] == 2) || (op == IMAGE_GATHER && kind[a[3]] == 1))) goto reject;
                uint32_t raw = next++;
                spv_emit(&out, a, 1);
                spv_emit(&out, (uint32_t[]){u4, raw}, 2);
                spv_emit(&out, a + 3, n - 3);
                SPV_INS(&out, SPV_OP_BITCAST, f4, a[2], raw);
                ++sites;
            }
        } else spv_emit(&out, a, n);
        i += n;
    }
    if (out.err || sites != 91) goto reject;
    out.w[3] = next;
    free(def);
    free(kind);
    free(*spirv);
    *spirv = out.w, *words = out.n;
    return 1;
reject:
    free(def);
    free(kind);
    free(out.w);
    return 0;
}

int vk_hdr_aa_splice(uint32_t **spirv, uint64_t *words, unsigned binding, unsigned expected, int store) {
    const uint32_t *w = *spirv;
    if (*words < 5 || w[0] != 0x07230203 || !expected || w[3] > *words || w[3] > UINT32_MAX - 1024) return 0;
    const uint32_t bound = w[3];
    uint32_t *root = calloc(bound, sizeof *root);
    if (!root) return 0;
    uint32_t f32 = 0, v3 = 0, v4 = 0, glsl = 0, image = 0, sites = 0, next = bound, rows[3] = {0}, zero3 = 0;
    int declared = 0;
    SpvBuf out = {0};
    spv_emit(&out, w, 5);
    for (uint64_t i = 5; i < *words;) {
        const uint32_t *a = w + i;
        uint32_t n = a[0] >> 16, op = a[0] & 0xffff;
        if (!n || i + n > *words) goto rejected;
        if (op == SPV_OP_EXT_INST_IMPORT && name_is(a + 1, n, "GLSL.std.450")) glsl = a[1];
        if (op == SPV_OP_DECORATE && n == 4 && a[2] == SPV_DEC_BINDING && a[3] == binding) {
            if (image || !a[1] || a[1] >= bound) goto rejected;
            image = a[1];
            root[image] = image;
        }
        if (op == SPV_OP_TYPE_FLOAT && n == 3 && a[2] == 32) f32 = a[1];
        if (op == SPV_OP_TYPE_VECTOR && n == 4 && f32 && a[2] == f32) {
            if (a[3] == 3) v3 = a[1];
            if (a[3] == 4) v4 = a[1];
        }
        if (op == SPV_OP_FUNCTION && !declared) {
            if (!f32 || !v4 || !image || (!store && !glsl)) goto rejected;
            if (!v3) {
                v3 = next++;
                SPV_INS(&out, SPV_OP_TYPE_VECTOR, v3, f32, 3);
            }
            static const float matrix[2][3][3] = {
                {{0.627403896f, 0.329283038f, 0.043313066f}, {0.069097289f, 0.919540396f, 0.011362315f},
                 {0.016391439f, 0.088013308f, 0.895595253f}},
                {{1.660491002f, -0.587641139f, -0.072849863f}, {-0.124550475f, 1.132899897f, -0.008349422f},
                 {-0.018150763f, -0.100578898f, 1.118729661f}},
            };
            for (unsigned channel = 0; channel < 3; ++channel) {
                uint32_t c[3];
                for (unsigned k = 0; k < 3; ++k) {
                    uint32_t bits;
                    memcpy(&bits, &matrix[!!store][channel][k], sizeof bits);
                    c[k] = next++;
                    SPV_INS(&out, SPV_OP_CONSTANT, f32, c[k], bits);
                }
                rows[channel] = next++;
                SPV_INS(&out, SPV_OP_CONSTANT_COMPOSITE, v3, rows[channel], c[0], c[1], c[2]);
            }
            if (!store) {
                zero3 = next++;
                SPV_INS(&out, SPV_OP_CONSTANT_NULL, v3, zero3);
            }
            declared = 1;
        }
        if ((op == SPV_OP_LOAD || op == SPV_OP_IMAGE) && n >= 4 && a[2] < bound && a[3] < bound)
            root[a[2]] = root[a[3]];
        if (!store && (op == SPV_OP_IMAGE_SAMPLE_EXPLICIT_LOD || op == SPV_OP_IMAGE_FETCH) &&
            n >= 5 && a[3] < bound && root[a[3]] == image && image) {
            if (!declared || a[1] != v4 || ++sites > expected) goto rejected;
            uint32_t raw = next++;
            spv_emit(&out, a, 2);
            spv_emit(&out, &raw, 1);
            spv_emit(&out, a + 3, n - 3);
            aa_color_convert(&out, &next, v4, v3, f32, glsl, zero3, rows, a[2], raw);
        } else if (store && op == SPV_OP_IMAGE_WRITE && n >= 4 && a[1] < bound && root[a[1]] == image && image) {
            if (!declared || ++sites > expected) goto rejected;
            uint32_t color = next++;
            aa_color_convert(&out, &next, v4, v3, f32, glsl, 0, rows, color, a[3]);
            spv_emit(&out, a, 3);
            spv_emit(&out, &color, 1);
            spv_emit(&out, a + 4, n - 4);
        } else spv_emit(&out, a, n);
        i += n;
    }
    if (out.err || sites != expected) goto rejected;
    out.w[3] = next;
    free(root);
    free(*spirv);
    *spirv = out.w, *words = out.n;
    return 1;
rejected:
    free(root);
    free(out.w);
    return 0;
}

#else

// TotK (!GAME_OOT): replace the game's 1 - e^-x tonemap shader wholesale with hdr_tonemap_totk_spv.
static const char *const tm_names[5] = {"support_buffer", "fp_t_tcb_8", "fp_t_tcb_A", "fp_c1", "fp_c3"};

int vk_hdr_shader_swap(uint32_t **spirv, uint64_t *words) {
    if (!vk_hdr_wanted() || !*spirv || *words < 5) return 0;
    int unclamped = fsr_unclamp(*spirv, words);
    if (unclamped > 0) {
        static volatile int32_t fsrs;
        rt_log("hdr: the game's FSR pass keeps values above 1 (%d output clamps; %d so far)", unclamped,
               plat_inc32(&fsrs));
        return 0;
    }
    if (unclamped < 0) {
        static volatile int32_t missed;
        if (plat_inc32(&missed) == 1)
            rt_log("hdr: the game's FSR pass was found but no clamp of its output; it still clips HDR highlights");
        return 0;
    }
    const uint32_t *w = *spirv;
    uint64_t n = *words;
    static const float want[4] = {0.298900008f, -1.44269502f, 0.666666687f, 10000.0f};
    uint32_t float_type = 0, found = 0, ids[5] = {0}, bind[5], set[5];
    int has_bind[5] = {0};
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) return 0;
        const uint32_t *a = w + i + 1;
        if (op == SPV_OP_TYPE_FLOAT && len == 3 && a[1] == 32) float_type = a[0];
        else if (op == SPV_OP_CONSTANT && len == 4 && a[0] == float_type && float_type)
            for (int k = 0; k < 4; k++) {
                float f;
                memcpy(&f, &a[2], 4);
                if (f == want[k]) found |= 1u << k;
            }
        else if (op == SPV_OP_NAME && len >= 3)
            for (int k = 0; k < 5; k++)
                if (name_is(a, len, tm_names[k])) ids[k] = a[0];
        i += len;
    }
    if (found != 15) return 0;
    for (int k = 0; k < 5; k++)
        if (!ids[k]) return 0;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        const uint32_t *a = w + i + 1;
        if (op == SPV_OP_DECORATE && len == 4)
            for (int k = 0; k < 5; k++)
                if (a[0] == ids[k]) {
                    if (a[1] == SPV_DEC_BINDING) bind[k] = a[2], has_bind[k] |= 1;
                    else if (a[1] == SPV_DEC_SET) set[k] = a[2], has_bind[k] |= 2;
                }
        i += len;
    }
    for (int k = 0; k < 5; k++)
        if (has_bind[k] != 3) return 0;
    uint64_t m = sizeof hdr_tonemap_totk_spv / 4;
    uint32_t *out = malloc(sizeof hdr_tonemap_totk_spv);
    if (!out) return 0;
    memcpy(out, hdr_tonemap_totk_spv, sizeof hdr_tonemap_totk_spv);
    uint32_t bound = out[3], *ours = calloc(bound ? bound : 1, sizeof *ours);
    if (!ours) { free(out); return 0; }
    for (uint64_t i = 5; i < m;) {
        uint32_t op = out[i] & 0xffff, len = out[i] >> 16;
        uint32_t *a = out + i + 1;
        if (op == SPV_OP_DECORATE && len == 4 && a[1] == SPV_DEC_BINDING && a[2] < 5) {
            if (a[0] < bound) ours[a[0]] = a[2] + 1;
            a[2] = bind[a[2]];
        }
        i += len;
    }
    for (uint64_t i = 5; i < m;) {
        uint32_t op = out[i] & 0xffff, len = out[i] >> 16;
        uint32_t *a = out + i + 1;
        if (op == SPV_OP_DECORATE && len == 4 && a[1] == SPV_DEC_SET && a[0] < bound && ours[a[0]])
            a[2] = set[ours[a[0]] - 1];
        i += len;
    }
    free(ours);
    free(*spirv);
    *spirv = out;
    *words = m;
    static volatile int32_t swaps;
    rt_log("hdr: the game's tonemap shader runs as shaders/hdr_tonemap_totk.frag (%d so far)", plat_inc32(&swaps));
    return 0;
}

#endif

// ---- 2. the picture the window texture is copied from

int vk_hdr_track;                   // vk_exec_draw.c calls vk_hdr_draw while an HDR10 swapchain reads float sources
static volatile int32_t g_active;   // the swapchain is HDR10 (gfx_present.c vk_hdr_set_active)
static PlatMutex rec_lock = PLAT_MUTEX_INIT;
typedef struct { VkImage win, src; VkFormat fmt; } HdrRec;
static HdrRec recs[4];   // per presented window image: what the last draw copied
static int nrecs;

void vk_hdr_set_active(int on) {
    plat_xchg32(&g_active, on);
    if (!on) vk_hdr_track = 0;
}
int vk_hdr_active(void) { return __atomic_load_n(&g_active, __ATOMIC_ACQUIRE); }

#ifdef GAME_OOT
// Both the checked output draw and present sealing run in replay order.
static struct {
    uint64_t frame;
    VkImage image;
    uint64_t address;
    uint32_t serial;
    GfxOutputEncoding encoding;
    VkGuestPq pq;
} written_outputs[8];
static unsigned next_written_output;
static uint64_t written_output_frame;

int vk_hdr_output_pending(void) {
    return written_output_frame && written_output_frame == __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
}

void vk_hdr_output_overwrite(VkImage image, uint64_t address) {
    if (!vk_hdr_output_pending()) return;
    int remaining = 0;
    for (unsigned i = 0; i < 8; i++) {
        if (written_outputs[i].frame != written_output_frame) continue;
        if ((image && written_outputs[i].image == image) || (address && written_outputs[i].address == address))
            written_outputs[i].frame = 0;
        else remaining = 1;
    }
    if (!remaining) written_output_frame = 0;
}

void vk_hdr_record_output(uint32_t serial, VkImage image, uint64_t address, GfxOutputEncoding encoding, const VkGuestPq *pq) {
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    written_output_frame = frame;
    unsigned index = 8;
    for (unsigned i = 0; i < 8; i++) {
        if (written_outputs[i].frame != frame || written_outputs[i].serial != serial) continue;
        index = i;
        break;
    }
    if (index == 8) index = next_written_output++ % 8;
    written_outputs[index] = (typeof(written_outputs[0])){frame, image, address, serial, encoding, *pq};
}

GfxOutputEncoding vk_hdr_written_output(uint32_t serial, uint64_t frame, VkGuestPq *pq) {
    for (unsigned i = 0; i < 8; i++) {
        if (written_outputs[i].frame != frame || written_outputs[i].serial != serial) continue;
        *pq = written_outputs[i].pq;
        return written_outputs[i].encoding;
    }
    return GFX_OUTPUT_UNSPECIFIED;
}
#endif

static int is_float(VkFormat f) {
    return f == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || f == VK_FORMAT_R16G16B16A16_SFLOAT ||
           f == VK_FORMAT_R32G32B32A32_SFLOAT;
}

// A presented window texture: the one target whose draws vk_hdr_draw notes. Only stretches drawing into it need the
// serial replay; every other stretch keeps the parallel chunks (turning them off for the whole frame cost ~5 ms of
// frame prep in the busy scenes, the lake and night: 47 against 60 fps).
int vk_hdr_window_target(VkImage image) {
    if (!image) return 0;
    for (int i = 0; i < 4; i++)
        if (__atomic_load_n(&recs[i].win, __ATOMIC_RELAXED) == image) return 1;
    return 0;
}

void vk_hdr_draw(VkExec *ex, const VkPipeInfo *info) {
    const VkExecTarget *t = &ex->color[0];
    if (!t->image) return;
    int known = 0;
    for (int i = 0; i < 4; i++)
        known |= __atomic_load_n(&recs[i].win, __ATOMIC_RELAXED) == t->image;
    if (!known) return;
    VkImage src = VK_NULL_HANDLE;
    VkFormat fmt = VK_FORMAT_UNDEFINED;
    int ntex = 0, slot = -1;
    for (uint32_t i = 0; i < info->n_bindings; i++) {
        const VkPipeBinding *bd = &info->bindings[i];
        if (bd->kind != GFX_RES_TEX || bd->stage != GFX_STAGE_FS || !bd->res) continue;
        ntex += bd->count ? bd->count : 1;
        if ((bd->res->type & 0x7ff) == 3 && !bd->res->separate) slot = bd->res->slot;
    }
    if (ntex == 1 && slot >= 0 && slot < 128 && ex->tex[1][slot]) {
        const VkTexImage *s = vk_texture_by_id(NVN_HANDLE_TEX(ex->tex[1][slot]), NULL, NULL);
        if (s && is_float(s->format) && s->width == t->width && s->height == t->height && s->samples == 1)
            src = s->image, fmt = s->format;
    }
    plat_mutex_lock(&rec_lock);
    for (int i = 0; i < nrecs; i++)
        if (recs[i].win == t->image) recs[i].src = src, recs[i].fmt = fmt;
    plat_mutex_unlock(&rec_lock);
}

void vk_hdr_present_source(const VkTexImage *win, VkImage *image, VkFormat *format) {
    if (!vk_hdr_wanted() || vk_hdr_native() || !vk_hdr_active()) return;
    vk_hdr_track = 1;
    plat_mutex_lock(&rec_lock);
    int r = -1;
    for (int i = 0; i < nrecs; i++)
        if (recs[i].win == win->image) r = i;
    if (r < 0 && nrecs < 4) r = nrecs++, recs[r].win = win->image, recs[r].src = VK_NULL_HANDLE;
    if (r >= 0 && recs[r].src) *image = recs[r].src, *format = recs[r].fmt;
    plat_mutex_unlock(&rec_lock);
    if (*image != win->image) renderer_mcp_tonemap(win->width, win->height);
    static int logged;
    if (!logged && *image != win->image) logged = 1, rt_log("hdr: the present reads the game's float picture");
}

void vk_hdr_image_gone(VkImage image) {
    if (!vk_hdr_track && !nrecs) return;
    plat_mutex_lock(&rec_lock);
    for (int i = 0; i < nrecs; i++) {
        if (recs[i].src == image) recs[i].src = VK_NULL_HANDLE;
        if (recs[i].win == image) recs[i] = recs[--nrecs], recs[nrecs] = (HdrRec){0}, i--;
    }
    plat_mutex_unlock(&rec_lock);
}

// ---- swapchain

int vk_hdr_surface_format(const VkSurfaceFormatKHR *f, uint32_t n, VkSurfaceFormatKHR *out) {
    if (!vk_hdr_wanted()) return 0;
    static int logged;
    if (!logged) {
        char b[1024];
        int o = 0;
        for (uint32_t i = 0; i < n && o < (int)sizeof b - 32; i++)
            o += snprintf(b + o, sizeof b - (size_t)o, " %d/%d", (int)f[i].format, (int)f[i].colorSpace);
        rt_log("hdr: surface formats (VkFormat/VkColorSpaceKHR):%s", b);
    }
    for (int pass = 0; pass < 2; pass++)
        for (uint32_t i = 0; i < n; i++)
            if (f[i].colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT &&
                f[i].format == (pass ? VK_FORMAT_A2R10G10B10_UNORM_PACK32 : VK_FORMAT_A2B10G10R10_UNORM_PACK32)) {
                *out = f[i];
                if (!logged) rt_log("hdr: HDR10 swapchain (format %d)", (int)f[i].format);
                logged = 1;
                return 1;
            }
    if (!logged)
        rt_log("hdr: the display offers no HDR10 format, so the picture stays SDR (Windows: turn on \"Use HDR\" in the "
               "display settings; Linux: a Wayland compositor with colour management and HDR on)");
    logged = 1;
    return 0;
}

#ifndef VK_HDR_SPLICE_TEST_ONLY
#ifdef GAME_OOT
#include "vk_hdr_adaptation.inc"
#endif

void vk_hdr_metadata(VkSwapchainKHR sc, int new_sc) {
    static unsigned sent = ~0u;
    VkCtx *c = vk_ctx();
    if (!c->has_hdr_metadata || (!new_sc && cfg_version() == sent)) return;
    sent = cfg_version();
    PFN_vkSetHdrMetadataEXT set = vk_fg_hdr_metadata_fn();
    if (!set) set = (PFN_vkSetHdrMetadataEXT)vkGetDeviceProcAddr(c->device, "vkSetHdrMetadataEXT");
    if (!set) return;
    Nits n = nits();
    VkHdrMetadataEXT m = {.sType = VK_STRUCTURE_TYPE_HDR_METADATA_EXT,
                          .displayPrimaryRed = {0.708f, 0.292f}, .displayPrimaryGreen = {0.170f, 0.797f},
                          .displayPrimaryBlue = {0.131f, 0.046f}, .whitePoint = {0.3127f, 0.3290f},
                          .maxLuminance = n.peak, .minLuminance = 0.0f, .maxContentLightLevel = n.peak,
                          .maxFrameAverageLightLevel = n.game};
    set(c->device, 1, &sc, &m);
}

// ---- 3. present passes

static void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 ss,
                    VkAccessFlags2 sa, VkPipelineStageFlags2 ds, VkAccessFlags2 da) {
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = ss, .srcAccessMask = sa,
                               .dstStageMask = ds, .dstAccessMask = da, .oldLayout = from, .newLayout = to,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                          .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &d);
}

typedef struct { VkImage img; VmaAllocation alloc; VkImageView view; uint32_t w, h; VkFormat format; } Img;

static void img_free(VkCtx *c, Img *m) {
    if (m->view) vkDestroyImageView(c->device, m->view, NULL);
    if (m->img) vmaDestroyImage(c->vma, m->img, m->alloc);
    *m = (Img){0};
}

static int img_ensure(VkCtx *c, Img *m, uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage) {
    if (m->img && m->w == w && m->h == h && m->format == fmt) return 1;
    if (m->img) {
        if (vk_exec_recording_fence()) {
            vk_defer_view(m->view);
            vk_defer_image(m->img, m->alloc);
            *m = (Img){0};
        } else {
            vk_queue_lock();
            vkDeviceWaitIdle(c->device);
            vk_queue_unlock();
            img_free(c, m);
        }
    }
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = fmt,
                            .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL, .usage = usage};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    if (vmaCreateImage(c->vma, &ii, &ai, &m->img, &m->alloc, NULL) != VK_SUCCESS) {
        m->img = VK_NULL_HANDLE;
        return 0;
    }
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = m->img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &m->view) != VK_SUCCESS) {
        m->view = VK_NULL_HANDLE;
        img_free(c, m);
        return 0;
    }
    m->w = w, m->h = h, m->format = fmt;
    return 1;
}

typedef struct UiPipeline {
    struct UiPipeline *next;
    VkPipeline pipeline;
    VkFormat format;
} UiPipeline;

static struct {
    int tried, ok;
    PFN_vkCmdPushDescriptorSetKHR push;
    VkSampler sampler;
    VkDescriptorSetLayout dsl_enc, dsl_ui;
    VkPipelineLayout enc_layout, ui_layout;
    VkPipeline enc;
    UiPipeline *ui;
    Img out;
    Img menu;
#ifdef GAME_OOT
    Img preview;
#endif
    VkImageView src_view[SLOTS];
} H;

static VkShaderModule module(VkCtx *c, const uint32_t *code, size_t size) {
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code};
    VkShaderModule m = VK_NULL_HANDLE;
    return vkCreateShaderModule(c->device, &mi, NULL, &m) == VK_SUCCESS ? m : VK_NULL_HANDLE;
}

static VkDescriptorSetLayout set_layout(VkCtx *c, const VkDescriptorSetLayoutBinding *b, uint32_t n) {
    VkDescriptorSetLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                          .bindingCount = n, .pBindings = b};
    VkDescriptorSetLayout l = VK_NULL_HANDLE;
    return vkCreateDescriptorSetLayout(c->device, &ci, NULL, &l) == VK_SUCCESS ? l : VK_NULL_HANDLE;
}

static VkPipelineLayout pipe_layout(VkCtx *c, VkDescriptorSetLayout *dsl, VkShaderStageFlags st, uint32_t pc) {
    VkPushConstantRange pr = {st, 0, pc};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 1,
                                     .pSetLayouts = dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    VkPipelineLayout l = VK_NULL_HANDLE;
    return vkCreatePipelineLayout(c->device, &li, NULL, &l) == VK_SUCCESS ? l : VK_NULL_HANDLE;
}

static int init(VkCtx *c) {
    H.tried = 1;
    H.push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(c->device, "vkCmdPushDescriptorSetKHR");
    VkDescriptorSetLayoutBinding enc[2] = {
        {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .descriptorCount = 1,
         .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}};
    VkDescriptorSetLayoutBinding ui = {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                       .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT};
    VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = VK_FILTER_NEAREST,
                              .minFilter = VK_FILTER_NEAREST, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                              .addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
                              .addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE, .maxLod = 0};
    if (!H.push || vkCreateSampler(c->device, &si, NULL, &H.sampler) != VK_SUCCESS ||
        !(H.dsl_enc = set_layout(c, enc, 2)) || !(H.dsl_ui = set_layout(c, &ui, 1)) ||
        !(H.enc_layout = pipe_layout(c, &H.dsl_enc, VK_SHADER_STAGE_COMPUTE_BIT, 112)) ||
        !(H.ui_layout = pipe_layout(c, &H.dsl_ui, VK_SHADER_STAGE_FRAGMENT_BIT, 16))) {
        rt_log("hdr: the present passes could not be created; the picture goes out unconverted");
        return 0;
    }
    VkShaderModule m = module(c, hdr_encode_spv, sizeof hdr_encode_spv);
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"},
                                      .layout = H.enc_layout};
    VkResult r = m ? vkCreateComputePipelines(c->device, VK_NULL_HANDLE, 1, &ci, NULL, &H.enc) : VK_ERROR_UNKNOWN;
    if (m) vkDestroyShaderModule(c->device, m, NULL);
    if (r != VK_SUCCESS) {
        rt_log("hdr: the present passes could not be created; the picture goes out unconverted");
        return 0;
    }
    return H.ok = 1;
}

VkImage vk_hdr_encode(VkCommandBuffer cb, unsigned slot, VkImage src, VkFormat fmt, const int crop[4], const VkGuestPq *pq) {
    VkCtx *c = vk_ctx();
    if (!H.tried) init(c);
    if (!H.ok || !img_ensure(c, &H.out, (uint32_t)crop[2], (uint32_t)crop[3], VK_FORMAT_R16G16B16A16_SFLOAT,
                             VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT))
        return vk_color_not_drawn_hdr(), VK_NULL_HANDLE;
    slot %= SLOTS;
    if (H.src_view[slot]) vkDestroyImageView(c->device, H.src_view[slot], NULL);
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = src,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fmt,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    if (vkCreateImageView(c->device, &vi, NULL, &H.src_view[slot]) != VK_SUCCESS) {
        H.src_view[slot] = VK_NULL_HANDLE;
        return vk_color_not_drawn_hdr(), VK_NULL_HANDLE;
    }
    Nits n = nits();
    float adj[4] = {0, 1, 1, 1};
    int grade = !pq && vk_color_hdr(adj, crop);
    int decode = fmt == VK_FORMAT_R8G8B8A8_UNORM || fmt == VK_FORMAT_B8G8R8A8_UNORM;
    const VkPipelineStageFlags2 CS = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, ALL = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    barrier(cb, src, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, ALL, VK_ACCESS_2_MEMORY_WRITE_BIT, CS,
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    barrier(cb, H.out.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, ALL, 0, CS,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    VkDescriptorImageInfo ii = {H.sampler, H.src_view[slot], VK_IMAGE_LAYOUT_GENERAL},
                          oi = {VK_NULL_HANDLE, H.out.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[2] = {
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii},
        {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 1, .descriptorCount = 1,
         .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, .pImageInfo = &oi}};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, H.enc);
    rec_push(H.push, cb, VK_PIPELINE_BIND_POINT_COMPUTE, H.enc_layout, 0, 2, w);
    struct { int32_t offset[2], size[2]; float p[4], adj[4], q[4], rows[3][4]; } pc = {
        {crop[0], crop[1]}, {crop[2], crop[3]}, {n.ui, n.peak, decode ? 1.0f : 0.0f, n.gamma},
        {adj[0], adj[1], adj[2], adj[3]}, {grade ? 1.0f : 0.0f, n.game / n.ui, 0, pq ? 1.0f : 0.0f}, {{0}}};
    if (pq) memcpy(pc.rows, pq->row, sizeof pc.rows);
    vkCmdPushConstants(cb, H.enc_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, &pc);
    vkCmdDispatch(cb, ((uint32_t)crop[2] + 7) / 8, ((uint32_t)crop[3] + 7) / 8, 1);
    barrier(cb, src, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, CS, 0, ALL, 0);
    barrier(cb, H.out.img, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, CS, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, ALL,
            VK_ACCESS_2_MEMORY_READ_BIT);
    return H.out.img;
}

VkImageView vk_hdr_ui_target(VkCommandBuffer cb, uint32_t w, uint32_t h) {
    VkCtx *c = vk_ctx();
    if (!H.tried) init(c);
    if (!H.ok || !img_ensure(c, &H.menu, w, h, VK_FORMAT_R8G8B8A8_UNORM,
                             VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT))
        return VK_NULL_HANDLE;
    barrier(cb, H.menu.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    return H.menu.view;
}

static VkPipeline ui_pipeline(VkCtx *c, VkFormat fmt) {
    for (UiPipeline *p = H.ui; p; p = p->next)
        if (p->format == fmt) return p->pipeline;
    VkShaderModule vs = module(c, hdr_fullscreen_spv, sizeof hdr_fullscreen_spv),
                   fs = module(c, hdr_ui_spv, sizeof hdr_ui_spv);
    VkPipelineShaderStageCreateInfo st[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vs,
         .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs,
         .pName = "main"}};
    VkPipelineVertexInputStateCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                 .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vp = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                            .viewportCount = 1, .scissorCount = 1};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
                                                 .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    const VkColorComponentFlags rgb = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT;
    const int guest_float = fmt == VK_FORMAT_B10G11R11_UFLOAT_PACK32 || fmt == VK_FORMAT_R16G16B16A16_SFLOAT;
    VkPipelineColorBlendAttachmentState ba = {
        .blendEnable = VK_TRUE, .srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA,
        .dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA, .colorBlendOp = VK_BLEND_OP_ADD,
        .srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE, .dstAlphaBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA,
        .alphaBlendOp = VK_BLEND_OP_ADD, .colorWriteMask = guest_float ? rgb : rgb | VK_COLOR_COMPONENT_A_BIT};
    VkPipelineColorBlendStateCreateInfo cb = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                              .attachmentCount = 1, .pAttachments = &ba};
    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                           .dynamicStateCount = 2, .pDynamicStates = dyn};
    VkPipelineRenderingCreateInfo ri = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                        .colorAttachmentCount = 1, .pColorAttachmentFormats = &fmt};
    VkGraphicsPipelineCreateInfo gi = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &ri,
                                       .stageCount = 2, .pStages = st, .pVertexInputState = &vi,
                                       .pInputAssemblyState = &ia, .pViewportState = &vp, .pRasterizationState = &rs,
                                       .pMultisampleState = &ms, .pColorBlendState = &cb, .pDynamicState = &ds,
                                       .layout = H.ui_layout};
    VkPipeline pipeline;
    VkResult r = vs && fs ? vkCreateGraphicsPipelines(c->device, VK_NULL_HANDLE, 1, &gi, NULL, &pipeline) : VK_ERROR_UNKNOWN;
    if (vs) vkDestroyShaderModule(c->device, vs, NULL);
    if (fs) vkDestroyShaderModule(c->device, fs, NULL);
    if (r != VK_SUCCESS) {
        rt_log("hdr: the menu pass could not be created; the menu is not shown");
        return VK_NULL_HANDLE;
    }
    UiPipeline *p = malloc(sizeof *p);
    if (!p) rt_fatal("hdr: out of memory caching a menu pipeline");
    *p = (UiPipeline){H.ui, pipeline, fmt};
    H.ui = p;
    return pipeline;
}

static int ui_compose(VkCommandBuffer cb, VkImageView dst_view, VkFormat fmt, uint32_t w, uint32_t h,
                      const float pc[4]) {
    VkCtx *c = vk_ctx();
    if (!H.ok || !H.menu.img) return 0;
    VkPipeline pipeline = ui_pipeline(c, fmt);
    if (!pipeline) return 0;
    barrier(cb, H.menu.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = dst_view,
                                    .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                    .loadOp = VK_ATTACHMENT_LOAD_OP_LOAD, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}}, .layerCount = 1,
                          .colorAttachmentCount = 1, .pColorAttachments = &ca};
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    VkViewport v = {0, 0, (float)w, (float)h, 0, 1};
    VkRect2D s = {{0, 0}, {w, h}};
    vkCmdSetViewport(cb, 0, 1, &v);
    vkCmdSetScissor(cb, 0, 1, &s);
    VkDescriptorImageInfo ii = {H.sampler, H.menu.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = 0, .descriptorCount = 1,
                               .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, .pImageInfo = &ii};
    rec_push(H.push, cb, VK_PIPELINE_BIND_POINT_GRAPHICS, H.ui_layout, 0, 1, &wr);
    vkCmdPushConstants(cb, H.ui_layout, VK_SHADER_STAGE_FRAGMENT_BIT, 0, sizeof(float) * 4, pc);
    vkCmdDraw(cb, 3, 1, 0, 0);
    vkCmdEndRendering(cb);
    return 1;
}

void vk_hdr_ui_compose(VkCommandBuffer cb, VkImageView dst_view, VkFormat fmt, uint32_t w, uint32_t h) {
    Nits n = nits();
#ifdef GAME_OOT
    float eotf = vk_hdr_native() ? n.gamma : n.ui_eotf;
#else
    float eotf = n.gamma;
#endif
    ui_compose(cb, dst_view, fmt, w, h, (float[4]){n.ui, eotf, 0, 0});
}

#ifdef GAME_OOT
int vk_hdr_ui_guest_compose(VkCommandBuffer cb, const VkTexImage *source, VkImageView target,
                            int preview[4], const VkGuestPq *pq) {
    float white_scale = 1.0f;
    float eotf = 0.0f;
    if (pq) {
        if (!pq->valid) return 0;
        float white = pq->row[0][0] + pq->row[0][1] + pq->row[0][2];
        if (!isfinite(white) || white <= 0.0f) return 0;
        Nits n = nits();
        white_scale = n.ui / white;
        if (!vk_hdr_native()) eotf = n.ui_eotf;
    }
    VkCtx *c = vk_ctx();
    const uint32_t w = source->width, h = source->height;
    const VkFormat format = source->format;
    VkFormatProperties properties;
    vkGetPhysicalDeviceFormatProperties(c->phys, format, &properties);
    if (!(properties.optimalTilingFeatures & VK_FORMAT_FEATURE_COLOR_ATTACHMENT_BLEND_BIT) || !ui_pipeline(c, format)) return 0;
    const uint64_t all = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, transfer = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    const uint64_t color = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    const uint64_t rw = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT;
    const uint64_t blend = VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    const VkImage image = source->image;
    const VkImageUsageFlags blit_usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    const VkFormatFeatureFlags blit_features = VK_FORMAT_FEATURE_BLIT_SRC_BIT | VK_FORMAT_FEATURE_BLIT_DST_BIT;
    if (preview[2] > 0 && preview[3] > 0 && (source->usage & blit_usage) == blit_usage &&
        (properties.optimalTilingFeatures & blit_features) == blit_features &&
        img_ensure(c, &H.preview, (uint32_t)preview[2], (uint32_t)preview[3], format,
                   blit_usage | VK_IMAGE_USAGE_SAMPLED_BIT)) {
        // Only the preview rectangle needs scratch storage: a blit cannot read and overwrite overlapping regions.
        VkImage scratch = H.preview.img;
        barrier(cb, image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                all, rw, transfer, VK_ACCESS_2_TRANSFER_READ_BIT);
        barrier(cb, scratch, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                transfer, 0, transfer, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkImageBlit region = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                              .srcOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}},
                              .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                              .dstOffsets = {{0, 0, 0}, {preview[2], preview[3], 1}}};
        vkCmdBlitImage(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, scratch, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1, &region, VK_FILTER_NEAREST);
        barrier(cb, scratch, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                transfer, VK_ACCESS_2_TRANSFER_WRITE_BIT, transfer, VK_ACCESS_2_TRANSFER_READ_BIT);
        barrier(cb, image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                transfer, VK_ACCESS_2_TRANSFER_READ_BIT, transfer, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkImageCopy copy = {.srcSubresource = region.dstSubresource, .dstSubresource = region.srcSubresource,
                            .dstOffset = {preview[0], preview[1], 0},
                            .extent = {(uint32_t)preview[2], (uint32_t)preview[3], 1}};
        vkCmdCopyImage(cb, scratch, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        barrier(cb, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                transfer, VK_ACCESS_2_TRANSFER_WRITE_BIT, color, blend);
    } else {
        memset(preview, 0, 4 * sizeof(*preview));
        barrier(cb, image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, all, rw, color, blend);
    }
    int composed = ui_compose(cb, target, format, w, h, (float[4]){white_scale, eotf, 1, 0});
    barrier(cb, image, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, all, rw, all, rw);
    return composed;
}
#endif

void vk_hdr_images_release(void) {
    VkCtx *c = vk_ctx();
    if (!c) return;
    int any = H.out.img != VK_NULL_HANDLE;
#ifndef GAME_OOT
    any |= H.menu.img != VK_NULL_HANDLE;
#endif
    for (int i = 0; i < SLOTS && !any; i++) any = H.src_view[i] != VK_NULL_HANDLE;
    if (!any) return;
    vk_queue_lock();
    vkDeviceWaitIdle(c->device);
    vk_queue_unlock();
    for (int i = 0; i < SLOTS; i++)
        if (H.src_view[i]) vkDestroyImageView(c->device, H.src_view[i], NULL), H.src_view[i] = VK_NULL_HANDLE;
    img_free(c, &H.out);
#ifndef GAME_OOT
    img_free(c, &H.menu);
#endif
}

void vk_hdr_release(void) {
    VkCtx *c = vk_ctx();
#ifdef GAME_OOT
    hdr_meter_release(c->vma);
#endif
    for (int i = 0; i < SLOTS; i++)
        if (H.src_view[i]) vkDestroyImageView(c->device, H.src_view[i], NULL), H.src_view[i] = VK_NULL_HANDLE;
    img_free(c, &H.out);
    img_free(c, &H.menu);
#ifdef GAME_OOT
    img_free(c, &H.preview);
#endif
    if (H.enc) vkDestroyPipeline(c->device, H.enc, NULL);
    while (H.ui) {
        UiPipeline *p = H.ui;
        H.ui = p->next;
        vkDestroyPipeline(c->device, p->pipeline, NULL);
        free(p);
    }
    if (H.enc_layout) vkDestroyPipelineLayout(c->device, H.enc_layout, NULL);
    if (H.ui_layout) vkDestroyPipelineLayout(c->device, H.ui_layout, NULL);
    if (H.dsl_enc) vkDestroyDescriptorSetLayout(c->device, H.dsl_enc, NULL);
    if (H.dsl_ui) vkDestroyDescriptorSetLayout(c->device, H.dsl_ui, NULL);
    if (H.sampler) vkDestroySampler(c->device, H.sampler, NULL);
    memset(&H, 0, sizeof H);
}
#endif

static float pq_nits(float e) {
    const float m1 = 0.1593017578125f, m2 = 78.84375f, c1 = 0.8359375f, c2 = 18.8515625f, c3 = 18.6875f;
    float p = powf(e, 1.0f / m2);
    return 10000.0f * powf(fmaxf(p - c1, 0.0f) / (c2 - c3 * p), 1.0f / m1);
}

void vk_hdr_to_srgb8(const uint8_t *in, uint8_t *out, size_t px, int a2r) {
    Nits n = nits();
    float ui = n.ui, inv_g = n.gamma == 1.0f ? 1.0f / 2.2f : n.gamma == 2.0f ? 1.0f / 2.4f : 0.0f;
    static const float m[9] = {1.6604910021f, -0.5876411388f, -0.0728498633f,
                               -0.1245504745f, 1.1328998971f, -0.0083494226f,
                               -0.0181507634f, -0.1005788980f, 1.1187296614f};
    for (size_t i = 0; i < px; i++) {
        uint32_t v;
        memcpy(&v, in + 4 * i, 4);
        float c[3] = {(float)(v & 1023), (float)(v >> 10 & 1023), (float)(v >> 20 & 1023)};
        if (a2r) { float t = c[0]; c[0] = c[2]; c[2] = t; }
        for (int k = 0; k < 3; k++) c[k] = pq_nits(c[k] / 1023.0f) / ui;
        for (int k = 0; k < 3; k++) {
            float l = fminf(fmaxf(m[3 * k] * c[0] + m[3 * k + 1] * c[1] + m[3 * k + 2] * c[2], 0.0f), 1.0f);
            float s = inv_g > 0.0f ? powf(l, inv_g)
                    : (l <= 0.0031308f ? 12.92f * l : 1.055f * powf(l, 1.0f / 2.4f) - 0.055f);
            out[4 * i + k] = (uint8_t)(s * 255.0f + 0.5f);
        }
        out[4 * i + 3] = 255;
    }
}
