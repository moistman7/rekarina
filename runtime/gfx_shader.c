// Shader service client: turns an NVN program (control section, code, c1 constants and the
// translator's answers) into SPIR-V plus reflection through `totk_xlate.exe --serve` children, started lazily with
// pipes and kept for the life of the process (a pool: REKARINA_XLATE_PROCS at run time, more for the boot prewarm).
// Results are cached in memory and on disk (shader-cache/<key>.spv and <key>.res.json next to rekarina.exe), so a shader
// is translated once per machine and translator.
//
// Keys. The code hash is sha256 over stage (u32), the 0x880-byte control section, code_size (u64), the code, the c1
// length (u32) and bytes. Per code hash and translator the cache holds a use mask (<sha256("TKSU", xlate id, code
// hash)>.use): the texture/image slots the translator queried (.res.json tex_queried, img_queried), the vertex
// attributes a VS declares and the colour targets an FS declares (SPIR-V Locations). The input id is sha256 over "TKSI",
// IN_VERSION, the code hash and the answers with everything outside the mask zeroed (fixed little-endian layout), so
// stale bindings in slots the shader never reads do not make new ids; it does not depend on the translator. The cache
// key is sha256 over "TKSK", KEY_VERSION, the translator identity and the input id. A code hash without a known mask is
// translated under a provisional key ("TKSP", full answers); its files are renamed to the cache key and the mask stored.
//
// Translator identity: build/xlate/totk_xlate.id, sha256 hex that tools/build_xlate.py writes over the sources it
// builds from (Program.cs, the csproj, the patches, the vendor tree); without it the sha256 of the exe. REKARINA_XLATE_ID
// (64 hex digits) overrides it. A translator change therefore changes every cache key without a hand-made KEY_VERSION
// bump; KEY_VERSION is for changes on this side (key layout, how the output is used).
//
// Pack: tools/pack_shader_cache.py folds the .spv, .res.json, .use and in/*.in files into <cache>/shader-cache.pack
// (one file instead of four per shader: ~117,000 files after an install) and deletes them; the installer runs it after
// preparing the shaders. The pack is read-only here. Every read looks in the pack first and then for the loose file,
// and new translations are written as loose files as before, until the tool runs again (section "cache pack").
//
// Stored inputs: every translation also writes <cache>/in/<input id>.in (magic "TKSI"; stage, the full answers, the
// control section, the code up to its length, c1), so a pipeline key record (vk_pipeline.c), which names its shaders by
// input id, can be translated again under a new translator without guest memory (gfx_shader_load_input,
// gfx_shader_prewarm).
//
// Reflection: every shader that enters the cache has its .res.json parsed into GfxShaderRes arrays and its SPIR-V
// scanned for interface Locations and the workgroup size (gfx.h accessors at the end of the file).
//
// Provenance: the c2 handle ranges of handle_slot are those of MacNativeGames' totk_xlate Service.cs (MIT).
// The code layout the translator sees (graphics: code[0x30..], compute: code[0..]; c1 = code[c1..length]
// with c1 and length at control+0x700) follows tools/shader_census.py, itself MacNativeGames' totk shader tooling (MIT).
#include <ctype.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <zstd.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "nvn_ctl.h"
#include "prep.h"
#include "platform/platform.h"

#define CONTROL_SIZE 0x880
#define CODE_SKIP 0x30                                     // graphics code starts after the first 0x30 bytes of the program
#define BUCKETS 4096
#define LINE_MAX_LEN 4096
#define KEY_VERSION 4                                      // in every cache key; bump when the key layout or the use of the
                                                           // output changes here (4: translator identity, input ids);
                                                           // translator changes are covered by its identity
#define IN_VERSION 1                                       // in every input id and .in file: bump when the answers layout
                                                           // or the program hash changes (drops every stored input)
#define USE_MAGIC 0x4d554b54u                              // "TKUM", .use files
#define USE_VERSION 1
#define IN_MAGIC 0x49534b54u                               // "TKSI", .in files
#define MAX_SVC 32                                         // translator processes

typedef struct {                                           // which answers a shader's translation depends on
    uint32_t slots;                                        // tex_type, img_type, img_format of these c2 slots
    uint32_t attrs;                                        // attr_type (VS input Locations)
    uint32_t fouts;                                        // frag_out_type (FS output Locations)
} UseMask;

typedef struct UseEnt {                                    // code hash -> use mask; known 0 caches a disk miss
    struct UseEnt *next;
    uint8_t code_hash[32];
    int known;
    UseMask m;
} UseEnt;

struct GfxShader {
    struct GfxShader *next;
    uint8_t key[32];
    uint8_t iid[32];                                       // input id (see the keys at the top)
    uint32_t *spirv;
#ifdef GAME_OOT
    struct GfxShader *renodx[4];
    unsigned renodx_checked;
    struct GfxShader *renodx_aa[GFX_HDR_AA_ROLES];
    struct GfxShader *aa_mask;
    int aa_mask_checked;
    unsigned renodx_aa_checked;
#endif
    uint8_t dbg;                   // DEBUG-UVDEBUG: the module was patched to write (u, v, alpha)
    uint64_t words;
    char *reflection;                                      // the service's .res.json text, parsed into the fields below
    uint64_t reflection_size;
    // Reflection (reflect(), once per shader before it enters the cache)
    int stage;                                             // GFX_STAGE_*
    GfxShaderRes *res[GFX_RES_KINDS];
    int nres[GFX_RES_KINDS];
    uint32_t in_mask, out_mask;                            // user Locations of Input / Output variables
    GfxShaderIo in[32], out[32];
    uint32_t local[3];
    uint32_t mvprev_banks;                                 // .res.json "mvprev_banks" (patch 0015), 0 none
    // Skinning matrix palette (reflect_palette): stride in bytes (48 or 64; 0 none), storage flag, bank/slot, and
    // input Location masks for integer and raw-bit joint indices.
    uint8_t pal_stride, pal_storage;
    int16_t pal_slot;
    uint32_t pal_joints, pal_raw_joints;
    // Arrays from the texture pool (gfx.h GFX_POOL_SET): their translator bindings and GFX_POOL_* kinds, which the
    // reflection leaves out of res[]; pool_mask: the kinds patch_pool_arrays moved to the pool set.
    uint16_t pool_bind[8];
    uint8_t pool_kind[8];
    int npool;
    uint32_t pool_mask;
    // GFX_STAGE_MSC (.res.json "mesh": {"as_compute": ...}): the emulation buffer's vertex layout, and its vertex shader
    uint32_t emu_vec4s, emu_max, emu_nloc;
    uint8_t emu_loc[32];
    const GfxShader *emu_vs;
};

static PlatMutex cache_lock = PLAT_MUTEX_INIT;                  // memory cache
static GfxShader *buckets[BUCKETS];
static UseEnt *use_buckets[BUCKETS];                       // under cache_lock
static volatile int64_t n_requests, n_hits, n_failures, n_keys, n_xlated;   // n_keys: memory cache entries

typedef struct {                                           // one translator child
    PlatChild *child;                                      // NULL: not started
    int busy;                                              // a request owns it (under S.lock)
    size_t in_len;
    char inbuf[LINE_MAX_LEN * 2];
} Svc;

static struct {
    PlatMutex lock;
    PlatCond freed;
    int limit;                                             // children in use at most (0: svc_runtime_limit)
    Svc s[MAX_SVC];
} S = {PLAT_MUTEX_INIT, PLAT_COND_INIT};

static struct {                                            // translations in flight: cache key, or code hash while the
    PlatMutex lock;                                          // use mask is unknown; one translation per id at a time
    PlatCond done;
    uint8_t id[2 * MAX_SVC][32];
    int n;
} F = {PLAT_MUTEX_INIT, PLAT_COND_INIT};

#define PATH_CAP 1024                                  // bytes of a UTF-8 path
static char cache_dir[PATH_CAP];
static char xlate_exe[PATH_CAP];
static uint8_t xlate_id[32];                               // translator identity (see the keys at the top)
static int host_f64 = 1, host_zero_locals, host_set;                         // gfx_shader_set_host
static uint32_t host_sbuf, host_sg[3] = {32, 32, 32}, host_gather;
uint32_t gfx_shader_mesh_subgroup;   // gfx.h
static PlatOnce paths_once = PLAT_ONCE_INIT;

// ---- small helpers

#define file_exists plat_file_exists
#define mkdir_p plat_file_mkdir_p
#define read_file plat_file_read
#define write_file plat_file_write

// ---- cache pack (<cache>/shader-cache.pack, written by tools/pack_shader_cache.py)
// Little-endian: "TKSPACK2", u32 count, u32 0, then count entries sorted by (kind, name), then the data. name is the
// 32 bytes the loose file's 64 hex digits spell; the kind word's low byte is the kind and its bit 8 says the data is
// one zstd frame of the file (the translated shaders, 6 of an install's 7 GiB, pack to about a sixth: PACK_ZSTD). The
// layout before, "TKSPACK1", is the same without compressed entries and is read too. The index is read once (48
// bytes a file: ~12 MB after an install); the data is read per lookup with positioned reads on one handle, safe from
// every thread. A pack that fails a check is ignored as a whole and the loose files are used.
enum { PACK_SPV = 1, PACK_RES = 2, PACK_USE = 3, PACK_IN = 4, PACK_KIND = 0xff, PACK_ZSTD = 0x100 };
typedef struct { uint8_t name[32]; uint32_t kind, size; uint64_t offset; } PackEnt;
_Static_assert(sizeof(PackEnt) == 48, "shader-cache.pack entry layout");
static PlatFile *pack_file;
static PackEnt *pack_ix;
static uint32_t pack_n;

static int pack_cmp(const PackEnt *a, uint32_t kind, const uint8_t name[32]) {
    uint32_t k = a->kind & PACK_KIND;
    if (k != kind) return k < kind ? -1 : 1;
    return memcmp(a->name, name, 32);
}

static int pack_read_at(uint64_t offset, void *out, uint32_t n) {
    return plat_file_pread(pack_file, out, n, offset) == (int64_t)n;
}

// Opens <cache>/shader-cache.pack and reads its index (init_paths). Nothing happens when there is none.
static void pack_open(void) {
    char p[PATH_CAP + 32];
    snprintf(p, sizeof p, "%s/shader-cache.pack", cache_dir);
    PlatFile *h = plat_file_open(p);
    if (!h) return;
    PlatFileId id;
    uint8_t head[16];
    uint32_t n = 0;
    const char *why = NULL;
    PackEnt *ix = NULL;
    pack_file = h;
    if (!plat_file_id(h, &id) || !pack_read_at(0, head, sizeof head) ||
        (memcmp(head, "TKSPACK2", 8) && memcmp(head, "TKSPACK1", 8)))
        why = "header";
    if (!why) {
        memcpy(&n, head + 8, 4);
        if (n > (1u << 24) || (uint64_t)n * sizeof *ix + sizeof head > id.size ||
            !(ix = malloc((size_t)n * sizeof *ix + 1)))
            why = "index size";
        else if (!pack_read_at(sizeof head, ix, n * (uint32_t)sizeof *ix)) why = "index read";
    }
    for (uint32_t i = 0; !why && i < n; i++) {
        if (ix[i].offset + ix[i].size > id.size) why = "entry out of bounds";
        else if (ix[i].kind & ~(uint32_t)(PACK_KIND | PACK_ZSTD)) why = "entry kind";
        else if (i && pack_cmp(&ix[i - 1], ix[i].kind & PACK_KIND, ix[i].name) >= 0) why = "index order";
    }
    if (why) {
        rt_log("gfx_shader: %s is not a usable cache pack (%s); reading the loose files", p, why);
        free(ix);
        plat_file_close(h);
        pack_file = NULL;
        return;
    }
    pack_ix = ix, pack_n = n;
    rt_log("gfx_shader: cache pack %s: %u files", p, n);
}

static const PackEnt *pack_find(uint32_t kind, const uint8_t name[32]) {
    size_t lo = 0, hi = pack_n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        int c = pack_cmp(&pack_ix[mid], kind, name);
        if (!c) return &pack_ix[mid];
        if (c < 0) lo = mid + 1;
        else hi = mid;
    }
    return NULL;
}

// A packed file as plat_file_read returns a loose one: malloc'd, NUL after the last byte; NULL when absent or empty.
static uint8_t *pack_read(uint32_t kind, const uint8_t name[32], uint64_t *size) {
    const PackEnt *e = pack_find(kind, name);
    uint8_t *buf = e && e->size ? malloc((size_t)e->size + 1) : NULL;
    if (!buf) return NULL;
    if (!pack_read_at(e->offset, buf, e->size)) {
        free(buf);
        return NULL;
    }
    if (e->kind & PACK_ZSTD) {   // a frame that does not unpack to the size it names is a missing file
        unsigned long long raw = ZSTD_getFrameContentSize(buf, e->size);
        uint8_t *out = raw && raw < (1ull << 30) ? malloc((size_t)raw + 1) : NULL;
        size_t got = out ? ZSTD_decompress(out, (size_t)raw, buf, e->size) : 0;
        free(buf);
        if (!out || ZSTD_isError(got) || got != raw) {
            free(out);
            return NULL;
        }
        out[raw] = 0;
        *size = raw;
        return out;
    }
    buf[e->size] = 0;
    *size = e->size;
    return buf;
}

// Appends a UTF-8 string as a JSON string body: ASCII as is, quotes and backslashes escaped, the rest as \uXXXX
// (UTF-16 units), so the service's stdin encoding does not matter.
static void json_esc(char **o, const char *s) {
    const unsigned char *p = (const unsigned char *)s;
    while (*p) {
        unsigned c = *p++;
        if (c >= 0x80) {
            int more = c >= 0xf0 ? 3 : c >= 0xe0 ? 2 : 1;
            c &= 0x3f >> more;
            while (more-- && (*p & 0xc0) == 0x80) c = c << 6 | (*p++ & 0x3f);
        }
        if (c == '"' || c == '\\') *(*o)++ = '\\', *(*o)++ = (char)c;
        else if (c >= 0x20 && c < 0x7f) *(*o)++ = (char)c;
        else if (c >= 0x10000) {
            c -= 0x10000;
            *o += sprintf(*o, "\\u%04x\\u%04x", 0xd800 + (c >> 10), 0xdc00 + (c & 0x3ff));
        } else *o += sprintf(*o, "\\u%04x", c);
    }
}

// ---- paths and the translator identity

static int hex32(const char *h, uint8_t out[32]) {        // 64 hex digits, then nothing but white space
    for (int i = 0; i < 64; i++) {
        char c = h[i] >= 'A' && h[i] <= 'F' ? (char)(h[i] | 32) : h[i];
        int v = c >= '0' && c <= '9' ? c - '0' : c >= 'a' && c <= 'f' ? c - 'a' + 10 : -1;
        if (v < 0) return 0;
        out[i / 2] = (uint8_t)(i & 1 ? out[i / 2] | v : v << 4);
    }
    for (const char *p = h + 64; *p; p++)
        if (*p != '\n' && *p != '\r' && *p != ' ' && *p != '\t') return 0;
    return 1;
}

// REKARINA_XLATE_ID (64 hex digits, or any other text, which is hashed), else <translator>.id next to the exe
// (tools/build_xlate.py), else the sha256 of the exe.
static void xlate_identity(void) {
    char text[256] = "";
    const char *how = "REKARINA_XLATE_ID";
    const char *id_env = cfg_get("REKARINA_XLATE_ID");
    if (id_env && *id_env && strlen(id_env) < sizeof text) {
        strcpy(text, id_env);
        if (!hex32(text, xlate_id)) rt_sha256(text, strlen(text), xlate_id);
    } else {
        char p[PATH_CAP + 8];
        strcpy(p, xlate_exe);
        char *dot = strrchr(p, '.'), *sep = plat_last_sep(p);
        if (dot && dot > sep) *dot = 0;
        strcat(p, ".id");
        uint64_t size = 0;
        char *f = (char *)read_file(p, &size);
        how = "the translator's .id file";
        if (!f || !hex32(f, xlate_id)) {
            uint8_t *exe = read_file(xlate_exe, &size);
            how = exe ? "sha256 of the exe; tools/build_xlate.py writes the .id file" : "none: no translator";
            if (exe) rt_sha256(exe, (size_t)size, xlate_id);
            free(exe);
        }
        free(f);
    }
    rt_log("gfx_shader: translator identity %02x%02x%02x%02x%02x%02x%02x%02x (%s)", xlate_id[0], xlate_id[1], xlate_id[2],
           xlate_id[3], xlate_id[4], xlate_id[5], xlate_id[6], xlate_id[7], how);
}

// The repository root is the nearest parent of rekarina.exe that holds build/xlate/totk_xlate.exe. The shader cache lives
// next to rekarina.exe, so every build dir has its own (as the pipeline cache, vk_pipeline.c cache_path_init).
static void use_preload_start(void);

static void init_paths(void *param) {
    (void)param;
    char exe[PATH_CAP], root[PATH_CAP] = "", exe_dir[PATH_CAP] = "";
    if (plat_exe_path(exe, sizeof exe)) {
        char *last = plat_last_sep(exe);
        if (last) snprintf(exe_dir, sizeof exe_dir, "%.*s", (int)(last - exe), exe);
        for (char *cut = plat_last_sep(exe); cut; cut = plat_last_sep(exe)) {
            *cut = 0;
            char probe[PATH_CAP + 32];
            snprintf(probe, sizeof probe, "%s/build/xlate/totk_xlate" PLAT_EXE_SUFFIX, exe);
            if (file_exists(probe)) {
                strcpy(root, exe);
                break;
            }
        }
    }
    const char *xlate_cfg = cfg_get("REKARINA_XLATE"), *cache_cfg = cfg_get("REKARINA_SHADER_CACHE");
    char beside[PATH_CAP + 32];   // a packaged folder (tools/package_build.py, package_oot.py) keeps the translator next to the exe
    if (exe_dir[0]) snprintf(beside, sizeof beside, "%s/" RT_FILE_STEM "_xlate" PLAT_EXE_SUFFIX, exe_dir);
#ifdef RT_OLD_FILE_STEM   // a folder packaged before the file was renamed. The identity is the .id file's content
    // (xlate_identity), so the shader cache holds whichever name the translator has.
    if (exe_dir[0] && !file_exists(beside))
        snprintf(beside, sizeof beside, "%s/" RT_OLD_FILE_STEM "_xlate" PLAT_EXE_SUFFIX, exe_dir);
#endif
    if (xlate_cfg && *xlate_cfg && strlen(xlate_cfg) < sizeof xlate_exe) strcpy(xlate_exe, xlate_cfg);
    else if (exe_dir[0] && file_exists(beside)) strcpy(xlate_exe, beside);
    else if (root[0]) snprintf(xlate_exe, sizeof xlate_exe, "%s/build/xlate/totk_xlate" PLAT_EXE_SUFFIX, root);
    if (cache_cfg && *cache_cfg && strlen(cache_cfg) < sizeof cache_dir) strcpy(cache_dir, cache_cfg);
    else if (exe_dir[0]) snprintf(cache_dir, sizeof cache_dir, "%s/shader-cache", exe_dir);
    xlate_identity();
    if (cache_dir[0]) pack_open();
    use_preload_start();
    return;
}

static void paths(void) { plat_once(&paths_once, init_paths, NULL); }

// Host answers (vk_caps.c, at device creation, before any translation). Answers other than the defaults (Float64,
// exact storage offsets, 32 lanes, no gather bias, undefined locals) are mixed into the translator identity, so they
// get their own cache keys.
#ifdef GAME_NS2
int gfx_shader_gmem, gfx_shader_float_atomics, gfx_shader_coop_matrix, gfx_shader_mesh_whole;
#endif
void gfx_shader_set_host(int float64, uint32_t sbuf_align, const uint32_t subgroup[3], uint32_t gather_bits, int zero_locals) {
    paths();
    if (host_set++) return;
    host_f64 = float64, host_sbuf = sbuf_align, host_gather = gather_bits, host_zero_locals = zero_locals;
    memcpy(host_sg, subgroup, sizeof host_sg);
#ifdef GAME_NS2
    int gmem = gfx_shader_gmem;   // with the pool table the same shaders translate otherwise: their own identity
    int fatom = gfx_shader_float_atomics, coop = gfx_shader_coop_matrix;
#else
    int gmem = 0, fatom = 0, coop = 0;
#endif
    if (float64 && !sbuf_align && subgroup[0] == 32 && subgroup[1] == 32 && subgroup[2] == 32 && !gather_bits && !zero_locals &&
        !gmem && !fatom && !coop)
        return;
    uint8_t buf[32 + 96];
    int n = snprintf((char *)buf + 32, 96, "TKHO f64=%d sbuf=%u sg=%u,%u,%u", float64, sbuf_align, subgroup[0], subgroup[1],
                     subgroup[2]);
    // only where it differs from the fragment stage's answer, so every other device keeps its cache keys
    if (gfx_shader_mesh_subgroup && gfx_shader_mesh_subgroup != subgroup[1])
        n += snprintf((char *)buf + 32 + n, (size_t)(96 - n), " mesh_sg=%u", gfx_shader_mesh_subgroup);
    if (gather_bits) n += snprintf((char *)buf + 32 + n, (size_t)(96 - n), " gather=%u", gather_bits);
    if (zero_locals) n += snprintf((char *)buf + 32 + n, (size_t)(96 - n), " zero_locals");
    if (gmem) n += snprintf((char *)buf + 32 + n, (size_t)(96 - n), " gmem_map");
    if (fatom) n += snprintf((char *)buf + 32 + n, (size_t)(96 - n), " float_atomics");
    if (coop) n += snprintf((char *)buf + 32 + n, (size_t)(96 - n), " coop_matrix");
    memcpy(buf, xlate_id, 32);
    rt_sha256(buf, 32 + (size_t)n, xlate_id);
    rt_log("gfx_shader: host options %s: own shader cache keys", (char *)buf + 37);
}

// ---- the translator children

static void child_stop(Svc *v) {
    plat_child_stop(v->child);
    v->child = NULL;
    v->in_len = 0;
}

static int child_start(Svc *v) {
    paths();
    if (!xlate_exe[0] || !file_exists(xlate_exe)) {
        rt_log("gfx_shader: translator not found (%s); set REKARINA_XLATE", xlate_exe[0] ? xlate_exe : "<none>");
        return 0;
    }
    uint32_t err = 0;
    v->child = plat_child_start(xlate_exe, "--serve", &err);
    if (!v->child) {
        rt_log("gfx_shader: cannot start %s (error %lu)", xlate_exe, (unsigned long)err);
        return 0;
    }
    v->in_len = 0;
    return 1;
}

// One line out, one line in (no newline in the reply). 0 when the pipe broke.
static int child_exchange_raw(Svc *v, const char *line, char *reply, size_t reply_size) {
    if (!v->child && !child_start(v)) return 0;
    plat_inc64(&n_requests);
    size_t n = strlen(line);
    if (!plat_child_write(v->child, line, n)) return 0;
    if (!plat_child_write(v->child, "\n", 1)) return 0;
    for (;;) {
        char *nl = memchr(v->inbuf, '\n', v->in_len);
        if (nl) {
            size_t len = (size_t)(nl - v->inbuf), keep = len < reply_size - 1 ? len : reply_size - 1;
            memcpy(reply, v->inbuf, keep);
            while (keep && reply[keep - 1] == '\r') keep--;
            reply[keep] = 0;
            memmove(v->inbuf, nl + 1, v->in_len - len - 1);
            v->in_len -= len + 1;
            return 1;
        }
        if (v->in_len == sizeof v->inbuf) v->in_len = 0;  // an overlong line: drop it, wait for its end
        long got = plat_child_read(v->child, v->inbuf + v->in_len, sizeof v->inbuf - v->in_len);
        if (got <= 0) return 0;
        v->in_len += (size_t)got;
    }
}

static int child_exchange(Svc *v, const char *line, char *reply, size_t reply_size) {
    uint64_t p0 = perf_begin();   // REKARINA_TRACE=perf: render-thread time spent translating
    int ok = child_exchange_raw(v, line, reply, reply_size);
    perf_end(PERF_XLATE, p0);
    return ok;
}

// Children allowed at run time: REKARINA_XLATE_PROCS, default 2.
static int svc_runtime_limit(void) {
    static volatile int32_t n;
    if (!n) {
        const char *e = cfg_get("REKARINA_XLATE_PROCS");
        int v = e && *e ? atoi(e) : 2;
        plat_xchg32(&n, v < 1 ? 1 : v > MAX_SVC ? MAX_SVC : v);
    }
    return (int)n;
}

static int svc_used(void) {                                // caller holds S.lock
    int used = 0;
    for (int i = 0; i < MAX_SVC; i++) used += S.s[i].busy || S.s[i].child;
    return used;
}

// A child for one request: an idle running one, else a new slot while fewer than S.limit are in use, else waits.
static Svc *svc_get(void) {
    plat_mutex_lock(&S.lock);
    if (!S.limit) S.limit = svc_runtime_limit();
    Svc *v;
    for (;;) {
        Svc *empty = NULL;
        v = NULL;
        for (int i = 0; i < MAX_SVC && !v; i++) {
            Svc *c = &S.s[i];
            if (!c->busy && c->child) v = c;
            else if (!c->busy && !empty) empty = c;
        }
        if (!v && empty && svc_used() < S.limit) v = empty;
        if (v) break;
        plat_cond_wait(&S.freed, &S.lock);
    }
    v->busy = 1;
    plat_mutex_unlock(&S.lock);
    return v;
}

// Returns a child; one over the limit (svc_set_limit lowered it) is stopped.
static void svc_put(Svc *v) {
    plat_mutex_lock(&S.lock);
    int stop = v->child && svc_used() > S.limit;
    if (!stop) v->busy = 0;
    plat_mutex_unlock(&S.lock);
    if (stop) {
        child_stop(v);
        plat_mutex_lock(&S.lock);
        v->busy = 0;
        plat_mutex_unlock(&S.lock);
    }
    plat_cond_broadcast(&S.freed);
}

// Sets the number of children; idle ones over it are stopped now, busy ones when they are returned.
static void svc_set_limit(int n) {
    Svc *stop[MAX_SVC];
    int k = 0;
    plat_mutex_lock(&S.lock);
    S.limit = n < 1 ? 1 : n > MAX_SVC ? MAX_SVC : n;
    for (int i = 0, used = svc_used(); i < MAX_SVC && used > S.limit; i++)
        if (!S.s[i].busy && S.s[i].child) S.s[i].busy = 1, stop[k++] = &S.s[i], used--;
    plat_mutex_unlock(&S.lock);
    for (int i = 0; i < k; i++) child_stop(stop[i]);
    plat_mutex_lock(&S.lock);
    for (int i = 0; i < k; i++) stop[i]->busy = 0;
    plat_mutex_unlock(&S.lock);
    plat_cond_broadcast(&S.freed);
}

// One request on a child; a dead child is restarted once. 0 when the translator is unavailable. The caller has given
// up its virtual core (kern_core_away): the wait for a child and the translation are host time.
// Where a translation's time goes, summed over the threads (prepare_files' progress lines): the translators' work, the
// waits for a free translator and for another thread's translation of the same program, and the rest of
// translate_one (the cache's files: writes, renames, reads). It tells a busy processor from a slow disk in a log.
static volatile int64_t tm_xchg, tm_svc_wait, tm_flight, tm_one, tm_n;

static int svc_request(const char *line, char *reply, size_t reply_size) {
    uint64_t t0 = plat_now_ns();
    Svc *v = svc_get();
    uint64_t t1 = plat_now_ns();
    int ok = 0;
    for (int attempt = 0; attempt < 2 && !ok; attempt++)
        if (!(ok = child_exchange(v, line, reply, reply_size))) child_stop(v);
    svc_put(v);
    plat_add64(&tm_svc_wait, (int64_t)(t1 - t0)), plat_add64(&tm_xchg, (int64_t)(plat_now_ns() - t1));
    return ok;
}

// One translation per id at a time (F); F.id full also waits.
static void flight_enter(const uint8_t id[32]) {
    plat_mutex_lock(&F.lock);
    for (;;) {
        int busy = F.n == 2 * MAX_SVC;
        for (int i = 0; i < F.n && !busy; i++) busy = !memcmp(F.id[i], id, 32);
        if (!busy) break;
        plat_cond_wait(&F.done, &F.lock);
    }
    memcpy(F.id[F.n++], id, 32);
    plat_mutex_unlock(&F.lock);
}

static void flight_leave(const uint8_t id[32]) {
    plat_mutex_lock(&F.lock);
    for (int i = 0; i < F.n; i++)
        if (!memcmp(F.id[i], id, 32)) {
            memcpy(F.id[i], F.id[--F.n], 32);
            break;
        }
    plat_mutex_unlock(&F.lock);
    plat_cond_broadcast(&F.done);
}

// ---- cache

static GfxShader *mem_find(const uint8_t key[32]) {
    plat_mutex_lock_shared(&cache_lock);
    GfxShader *s = buckets[(key[0] | key[1] << 8) % BUCKETS];
    while (s && memcmp(s->key, key, 32)) s = s->next;
    plat_mutex_unlock_shared(&cache_lock);
    return s;
}

static void shader_free(GfxShader *s) {
#ifdef GAME_OOT
    for (unsigned i = 0; i < sizeof s->renodx / sizeof *s->renodx; ++i)
        if (s->renodx[i]) shader_free(s->renodx[i]);
#endif
    for (int k = 0; k < GFX_RES_KINDS; k++) free(s->res[k]);
    free(s->spirv), free(s->reflection), free(s);
}

static void reflect(GfxShader *s);

// Inserts s, or frees it and returns the entry another thread inserted first.
static GfxShader *mem_insert(GfxShader *s) {
    plat_mutex_lock(&cache_lock);
    GfxShader **head = &buckets[(s->key[0] | s->key[1] << 8) % BUCKETS], *r = *head;
    while (r && memcmp(r->key, s->key, 32)) r = r->next;
    if (r) {
        shader_free(s);
        s = r;
    } else {
        s->next = *head;
        *head = s;
        n_keys++;
    }
    plat_mutex_unlock(&cache_lock);
    return s;
}

static void key_hex(const uint8_t key[32], char out[65]) {
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) out[i * 2] = hex[key[i] >> 4], out[i * 2 + 1] = hex[key[i] & 15];
    out[64] = 0;
}

// A .res.json the translator wrote: one JSON object that names its stage, no NUL inside. A file an interrupted write
// left zero-filled (3647 of 21924 in one cache were) fails this; reflect() would take it for a vertex shader with
// no resources, and its pipeline layout would lack the shader's buffers and textures.
static int reflection_valid(const char *res, uint64_t size) {
    if (strlen(res) != size) return 0;
    const char *b = res, *e = res + size;
    while (b < e && isspace((unsigned char)*b)) b++;
    while (e > b && isspace((unsigned char)e[-1])) e--;
    if (e - b < 2 || *b != '{' || e[-1] != '}') return 0;
    const char *st = strstr(b, "\"stage\"");
    if (!st) return 0;
    st = strchr(st + 7, '"');
    if (!st) return 0;
#ifdef GAME_NS2   // NVN2 task, mesh and geometry stages (docs/oot-xlate.md)
    if (!strncmp(st, "\"ts\"", 4) || !strncmp(st, "\"ms\"", 4) || !strncmp(st, "\"gs\"", 4) || !strncmp(st, "\"msc\"", 5) ||
        !strncmp(st, "\"tsc\"", 5))
        return 1;
#endif
    return !strncmp(st, "\"vs\"", 4) || !strncmp(st, "\"ps\"", 4) || !strncmp(st, "\"cs\"", 4);
}

// Every instruction after the header has a word count and the last one ends at the module's end: a module with
// zero-filled clusters fails.
static int spirv_complete(const uint32_t *w, uint64_t n) {
    uint64_t i = 5;
    while (i < n) {
        uint32_t len = w[i] >> 16;
        if (!len) return 0;
        i += len;
    }
    return i == n;
}

// Reads <key>.spv and <key>.res.json from the pack (packed 1) or the loose files; NULL if either is missing or
// malformed. iid: the input id, NULL when the caller sets it.
static GfxShader *disk_load_from(const uint8_t key[32], const uint8_t *iid, int packed) {
    static volatile int32_t n_bad;
    char hex[65], p[PATH_CAP + 80];
    key_hex(key, hex);
    uint64_t spv_size = 0, res_size = 0;
    snprintf(p, PATH_CAP + 80, "%s/%s.res.json", cache_dir, hex);
    uint8_t *res = packed ? pack_read(PACK_RES, key, &res_size) : read_file(p, &res_size);
    if (!res) return NULL;
    if (!reflection_valid((const char *)res, res_size)) {
        int32_t n = plat_inc32(&n_bad);
        if (n <= 3 || !(n & (n - 1))) rt_log("gfx_shader: %s.res.json is malformed, translating again (%d so far)", hex, n);
        free(res);
        return NULL;
    }
    snprintf(p, PATH_CAP + 80, "%s/%s.spv", cache_dir, hex);
    uint8_t *spv = packed ? pack_read(PACK_SPV, key, &spv_size) : read_file(p, &spv_size);
    uint32_t magic = 0;
    if (spv && spv_size >= 20) memcpy(&magic, spv, 4);
    if (!spv || spv_size % 4 || magic != 0x07230203 || !spirv_complete((const uint32_t *)spv, spv_size / 4)) {
        free(res), free(spv);
        return NULL;
    }
    GfxShader *s = calloc(1, sizeof *s);
    memcpy(s->key, key, 32);
    if (iid) memcpy(s->iid, iid, 32);
    s->spirv = (uint32_t *)spv;
    s->words = spv_size / 4;
    s->reflection = (char *)res;
    s->reflection_size = res_size;
    reflect(s);
#ifndef GAME_OOT
    if (s->stage == GFX_STAGE_FS) vk_hdr_shader_swap(&s->spirv, &s->words);
#endif
    return s;
}

const GfxShader *gfx_shader_own(const uint32_t *spirv, uint64_t words, const char *res_json) {
    if (words < 5 || spirv[0] != 0x07230203 || !spirv_complete(spirv, words) ||
        !reflection_valid(res_json, strlen(res_json)))
        return NULL;
    GfxShader *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    size_t rn = strlen(res_json);
    uint8_t *buf = malloc(words * 4 + rn);
    s->spirv = malloc(words * 4), s->reflection = malloc(rn + 1);
    if (!buf || !s->spirv || !s->reflection) rt_fatal("gfx_shader: out of memory");
    memcpy(buf, spirv, words * 4), memcpy(buf + words * 4, res_json, rn);
    rt_sha256(buf, words * 4 + rn, s->key);
    free(buf);
    memcpy(s->spirv, spirv, words * 4), memcpy(s->reflection, res_json, rn + 1);
    s->words = words, s->reflection_size = rn;
    reflect(s);
    return s;
}

#ifdef GAME_OOT
int gfx_shader_guest_pq(const GfxShader *s) {
    static const uint8_t pq[32] = {
        0x8e, 0x36, 0x68, 0xf6, 0x70, 0x5e, 0x56, 0x87, 0x1f, 0xd7, 0x69, 0x7c, 0x69, 0x6d, 0xad, 0xd5,
        0xf2, 0x8d, 0xe1, 0x84, 0xd1, 0xca, 0x20, 0x8b, 0x5b, 0xee, 0x96, 0x36, 0x37, 0xc3, 0xd5, 0xe7,
    };
    return s && s->stage == GFX_STAGE_FS && !memcmp(s->iid, pq, sizeof pq);
}

const GfxShader *gfx_shader_renodx_variant(const GfxShader *s, GfxRenodxVariant kind) {
    if (!s || s->stage != GFX_STAGE_FS || gfx_shader_guest_pq(s)) return s;
    unsigned slot = (unsigned)kind;
    unsigned bit = 1u << slot;
    if (__atomic_load_n(&s->renodx_checked, __ATOMIC_ACQUIRE) & bit)
        return s->renodx[slot] ? s->renodx[slot] : s;
    GfxShader *base = (GfxShader *)s;
    plat_mutex_lock(&cache_lock);
    if (!(__atomic_load_n(&base->renodx_checked, __ATOMIC_RELAXED) & bit)) {
        uint64_t words = s->words;
        uint32_t *code = malloc((size_t)words * sizeof *code);
        int failed = !code;
        int meter_binding = 0;
        if (code) {
            memcpy(code, s->spirv, (size_t)words * sizeof *code);
            int spliced = vk_hdr_scene_splice(&code, &words, kind, &meter_binding);
            failed = spliced <= 0;
            if (spliced > 0)
                base->renodx[slot] = (GfxShader *)gfx_shader_own(code, words, s->reflection);
            free(code);
        }
        GfxShader *variant = base->renodx[slot];
        if (variant) {
            memcpy(variant->iid, s->iid, sizeof s->iid);
            if (meter_binding) {
                int n = variant->nres[GFX_RES_SSBO];
                GfxShaderRes *res = realloc(variant->res[GFX_RES_SSBO], (n + 1) * sizeof *res);
                if (!res) rt_fatal("gfx_shader: out of memory adding the HDR eye adaptation binding");
                res[n] = (GfxShaderRes){.binding = (uint16_t)meter_binding, .slot = -1, .flags = 1, .cbuf_slot = GFX_CB_HDR_METER};
                variant->res[GFX_RES_SSBO] = res;
                variant->nres[GFX_RES_SSBO] = n + 1;
            }
        }
        static const char *const names[] = {"RenoDX HDR scene mapper", "Vanilla scene effects", "RenoDX SDR scene mapper", "Vanilla scene grading"};
        const char *name = names[kind];
        if (variant) rt_log("hdr: built %s variant %02x%02x%02x%02x%s", name,
                            s->iid[0], s->iid[1], s->iid[2], s->iid[3], meter_binding ? " with eye adaptation" : "");
        else if (failed) rt_log("hdr: %s variant unavailable for %02x%02x%02x%02x", name,
                                 s->iid[0], s->iid[1], s->iid[2], s->iid[3]);
        __atomic_fetch_or(&base->renodx_checked, bit, __ATOMIC_RELEASE);
    }
    const GfxShader *variant = base->renodx[slot];
    plat_mutex_unlock(&cache_lock);
    return variant ? variant : s;
}

const GfxShader *gfx_shader_renodx_aa(const GfxShader *s, unsigned role) {
    if (!s || s->stage != GFX_STAGE_CS || role >= GFX_HDR_AA_ROLES) return NULL;
    const int store = role == GFX_HDR_AA_STORE;
    const unsigned expected[] = {5, 6, 18};
    int slot = store ? 1 : 3;
    const GfxShaderRes *r;
    int n = gfx_shader_resources(s, store ? GFX_RES_IMG : GFX_RES_TEX, &r);
    int binding = -1;
    for (int i = 0; i < n; ++i) {
        if (r[i].slot != slot) continue;
        if (binding >= 0 || r[i].set != 0 || r[i].type != 3 || r[i].separate || r[i].count > 1 ||
            (store && (!(r[i].flags & 4) || r[i].format))) return NULL;
        binding = r[i].binding;
    }
    if (binding < 0) return NULL;
    GfxShader *base = (GfxShader *)s;
    plat_mutex_lock(&cache_lock);
    if (!(base->renodx_aa_checked & (1u << role))) {
        base->renodx_aa_checked |= 1u << role;
        uint64_t words = s->words;
        uint32_t *code = malloc((size_t)words * sizeof *code);
        if (code) {
            memcpy(code, s->spirv, (size_t)words * sizeof *code);
            if (vk_hdr_aa_splice(&code, &words, (unsigned)binding, expected[role], store))
                base->renodx_aa[role] = (GfxShader *)gfx_shader_own(code, words, s->reflection);
            free(code);
        }
        if (base->renodx_aa[role]) memcpy(base->renodx_aa[role]->iid, s->iid, sizeof s->iid);
    }
    const GfxShader *variant = base->renodx_aa[role];
    plat_mutex_unlock(&cache_lock);
    return variant;
}

const GfxShader *gfx_shader_aa_mask(const GfxShader *s) {
    GfxShader *base = (GfxShader *)s;
    plat_mutex_lock(&cache_lock);
    if (!base->aa_mask_checked) {
        base->aa_mask_checked = 1;
        int binding = -1;
        for (int i = 0; i < s->nres[GFX_RES_TEX]; ++i) {
            const GfxShaderRes *r = &s->res[GFX_RES_TEX][i];
            if (r->slot == 4 && r->set == 0 && r->type == 3 && !r->separate && r->count <= 1) binding = r->binding;
        }
        if (binding >= 0) {
            uint64_t words = s->words;
            uint32_t *code = malloc((size_t)words * sizeof *code);
            if (!code) rt_fatal("gfx_shader: out of memory preparing the edge-mask shader");
            memcpy(code, s->spirv, (size_t)words * sizeof *code);
            if (vk_aa_mask_splice(&code, &words, (unsigned)binding))
                base->aa_mask = (GfxShader *)gfx_shader_own(code, words, s->reflection);
            free(code);
        }
        if (base->aa_mask) memcpy(base->aa_mask->iid, s->iid, sizeof s->iid);
        else rt_log("gfx_shader: could not repair the unsigned edge-mask sampler");
    }
    const GfxShader *variant = base->aa_mask;
    plat_mutex_unlock(&cache_lock);
    return variant;
}
#endif

int gfx_shader_meshc(const GfxShader *s, uint32_t *vec4s, uint32_t *max_meshlets, const uint8_t **locs) {
    if (!s || s->stage != GFX_STAGE_MSC || !s->emu_vec4s || !s->emu_max) return 0;
    *vec4s = s->emu_vec4s, *max_meshlets = s->emu_max, *locs = s->emu_loc;
    return (int)s->emu_nloc;
}

// The vertex shader of an emulated mesh draw (vk_exec_draw.c mesh_emulate): vertex input i (a vec4 of the emulation
// buffer's vertex, binding 0) goes out as the position (i 0) or as the i-th generic attribute at its location; the
// fragment shader reads them as it read the mesh shader's outputs. Written as SPIR-V words here (no compiler on the
// development machine); gfx_shader_own reflects it like any other module (no resources).
static const GfxShader *meshc_vs_make(const GfxShader *msc) {
    uint32_t n = msc->emu_vec4s, w[1024], k = 0, id = 1;
    if (n < 1 || n > 33 || msc->emu_nloc != n - 1) return NULL;
    enum { T_VOID = 1, T_FN, T_F32, T_V4, T_PIN, T_POUT, T_PV, T_PPV, T_I32, C_0, V_PV, F_MAIN, L_MAIN, FIRST };
    id = FIRST;
    uint32_t in[33], out[33], ld[33];
    for (uint32_t i = 0; i < n; i++) in[i] = id++;
    for (uint32_t i = 1; i < n; i++) out[i] = id++;
    for (uint32_t i = 0; i < n; i++) ld[i] = id++;
    uint32_t ac = id++;
#define OP(op, len) (w[k++] = (uint32_t)(len) << 16 | (op))
    w[k++] = 0x07230203, w[k++] = 0x00010000, w[k++] = 0, w[k++] = 0 /* bound, below */, w[k++] = 0;
    OP(17, 2), w[k++] = 1;                                        // OpCapability Shader
    OP(14, 3), w[k++] = 0, w[k++] = 1;                            // OpMemoryModel Logical GLSL450
    OP(15, 3 + 2 + 2 * n), w[k++] = 0, w[k++] = F_MAIN, w[k++] = 0x6e69616d, w[k++] = 0;   // OpEntryPoint Vertex "main"
    for (uint32_t i = 0; i < n; i++) w[k++] = in[i];
    for (uint32_t i = 1; i < n; i++) w[k++] = out[i];
    w[k++] = V_PV;
    for (uint32_t i = 0; i < n; i++) OP(71, 4), w[k++] = in[i], w[k++] = 30, w[k++] = i;                  // Location i
    for (uint32_t i = 1; i < n; i++) OP(71, 4), w[k++] = out[i], w[k++] = 30, w[k++] = msc->emu_loc[i - 1];
    OP(72, 5), w[k++] = T_PV, w[k++] = 0, w[k++] = 11, w[k++] = 0;                                     // BuiltIn Position
    OP(71, 3), w[k++] = T_PV, w[k++] = 2;                                                              // Block
    OP(19, 2), w[k++] = T_VOID;
    OP(33, 3), w[k++] = T_FN, w[k++] = T_VOID;
    OP(22, 3), w[k++] = T_F32, w[k++] = 32;
    OP(23, 4), w[k++] = T_V4, w[k++] = T_F32, w[k++] = 4;
    OP(32, 4), w[k++] = T_PIN, w[k++] = 1, w[k++] = T_V4;
    OP(32, 4), w[k++] = T_POUT, w[k++] = 3, w[k++] = T_V4;
    OP(30, 3), w[k++] = T_PV, w[k++] = T_V4;
    OP(32, 4), w[k++] = T_PPV, w[k++] = 3, w[k++] = T_PV;
    OP(21, 4), w[k++] = T_I32, w[k++] = 32, w[k++] = 1;
    OP(43, 4), w[k++] = T_I32, w[k++] = C_0, w[k++] = 0;
    OP(59, 4), w[k++] = T_PPV, w[k++] = V_PV, w[k++] = 3;
    for (uint32_t i = 0; i < n; i++) OP(59, 4), w[k++] = T_PIN, w[k++] = in[i], w[k++] = 1;
    for (uint32_t i = 1; i < n; i++) OP(59, 4), w[k++] = T_POUT, w[k++] = out[i], w[k++] = 3;
    OP(54, 5), w[k++] = T_VOID, w[k++] = F_MAIN, w[k++] = 0, w[k++] = T_FN;
    OP(248, 2), w[k++] = L_MAIN;
    for (uint32_t i = 0; i < n; i++) OP(61, 4), w[k++] = T_V4, w[k++] = ld[i], w[k++] = in[i];
    OP(65, 5), w[k++] = T_POUT, w[k++] = ac, w[k++] = V_PV, w[k++] = C_0;
    OP(62, 3), w[k++] = ac, w[k++] = ld[0];
    for (uint32_t i = 1; i < n; i++) OP(62, 3), w[k++] = out[i], w[k++] = ld[i];
    OP(253, 1);
    OP(56, 1);
#undef OP
    w[3] = id;
    return gfx_shader_own(w, k, "{\"stage\":\"vs\",\"cbufs\":[],\"sbufs\":[],\"textures\":[],\"images\":[]}");
}

const GfxShader *gfx_shader_meshc_vs(const GfxShader *s) {
    if (!s || s->stage != GFX_STAGE_MSC) return NULL;
    const GfxShader *v = __atomic_load_n(&s->emu_vs, __ATOMIC_ACQUIRE);
    if (v) return v;
    static PlatMutex lock = PLAT_MUTEX_INIT;
    plat_mutex_lock(&lock);
    if (!(v = s->emu_vs)) {
        v = meshc_vs_make(s);
        if (v) __atomic_store_n(&((GfxShader *)s)->emu_vs, v, __ATOMIC_RELEASE);
        else rt_log("gfx_shader: the vertex shader of an emulated mesh draw (%u vec4s) could not be made", s->emu_vec4s);
    }
    plat_mutex_unlock(&lock);
    return v;
}

// The pack, then the loose files (a newer translation of a key the pack holds a bad copy of); NULL when neither has a
// valid pair (the caller translates again and writes loose files).
static GfxShader *disk_load(const uint8_t key[32], const uint8_t *iid) {
    GfxShader *s = pack_n ? disk_load_from(key, iid, 1) : NULL;
    return s ? s : disk_load_from(key, iid, 0);
}

// ---- request building

static const char *const tex_names[] = {NULL, "1d", "1darray", "buffer", "2d", "2darray", "2dms", "2dmsarray", "3d",
                                        "cube", "cubearray"};   // GfxShaderAnswers tex_type / img_type codes
static const char *const fmt_names[] = {   // img_format codes: Ryujinx TextureFormat enum order
    NULL, "R8Unorm", "R8Snorm", "R8Uint", "R8Sint", "R16Float", "R16Unorm", "R16Snorm", "R16Uint", "R16Sint", "R32Float",
    "R32Uint", "R32Sint", "R8G8Unorm", "R8G8Snorm", "R8G8Uint", "R8G8Sint", "R16G16Float", "R16G16Unorm", "R16G16Snorm",
    "R16G16Uint", "R16G16Sint", "R32G32Float", "R32G32Uint", "R32G32Sint", "R8G8B8A8Unorm", "R8G8B8A8Snorm",
    "R8G8B8A8Uint", "R8G8B8A8Sint", "R16G16B16A16Float", "R16G16B16A16Unorm", "R16G16B16A16Snorm", "R16G16B16A16Uint",
    "R16G16B16A16Sint", "R32G32B32A32Float", "R32G32B32A32Uint", "R32G32B32A32Sint", "R10G10B10A2Unorm",
    "R10G10B10A2Uint", "R11G11B10Float"};
#define COUNT(a) ((int)(sizeof a / sizeof *a))

static const char *attr_name(uint8_t t) { return t == 1 ? "sint" : t == 2 ? "uint" : "float"; }

// ---- reflection: the .res.json the translator writes (tools/xlate/Program.cs Translate) and the SPIR-V itself

// `"name":` inside [o, e): the first byte after the colon and its spaces, NULL when absent.
static const char *json_key(const char *o, const char *e, const char *name) {
    char pat[40];
    int n = snprintf(pat, sizeof pat, "\"%s\":", name);
    for (const char *p = o; p + n <= e; p++)
        if (*p == '"' && !memcmp(p, pat, (size_t)n)) {
            p += n;
            while (p < e && (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t')) p++;
            return p;
        }
    return NULL;
}

static long json_int(const char *o, const char *e, const char *name, long def) {
    const char *p = json_key(o, e, name);
    return p ? strtol(p, NULL, 10) : def;
}

// Copies a string value into buf (empty when absent or not a string).
static void json_str(const char *o, const char *e, const char *name, char *buf, size_t n) {
    const char *p = json_key(o, e, name);
    size_t k = 0;
    if (p && *p == '"')
        for (p++; p < e && *p != '"' && k + 1 < n; p++) buf[k++] = *p;
    buf[k] = 0;
}

// The objects of the flat array `"name": [ {...}, ... ]`: calls fn for each object's [start, end). Returns the count.
static int json_objects(const char *js, const char *name, void (*fn)(GfxShader *, int, const char *, const char *),
                        GfxShader *s, int kind) {
    const char *end = js + strlen(js), *p = json_key(js, end, name);
    if (!p || *p != '[') return 0;
    int n = 0;
    for (p++; p < end && *p != ']'; p++) {
        if (*p != '{') continue;
        const char *q = memchr(p, '}', (size_t)(end - p));
        if (!q) break;
        if (fn) fn(s, kind, p, q);
        n++;
        p = q;
    }
    return n;
}

static int fmt_code(const char *name) {
    for (int i = 1; i < COUNT(fmt_names); i++)
        if (!strcmp(fmt_names[i], name)) return i;
    return 0;
}

// c2 slot of a texture or image handle (Service.cs of MNG's totk_xlate: combined handles at 0x20, 32 slots; images at
// 0x120; separate textures at 0x168), -1 outside those ranges or for a handle in another constant buffer.
static int handle_slot(int kind, int cbuf_slot, int handle, int separate) {
    if (cbuf_slot >= 0 && cbuf_slot != 2) return -1;
    int bytes = (handle & 0x3fff) * 4;
    if (kind == GFX_RES_IMG) return bytes >= 0x120 && bytes < 0x168 ? (bytes - 0x120) / 8 : -1;
    if (separate) return bytes >= 0x168 && bytes < 0x568 ? (bytes - 0x168) / 8 : -1;
    return bytes >= 0x20 && bytes < 0x120 ? (bytes - 0x20) / 8 : -1;
}

// An array the translator indexes over the texture pool (constant buffer 0, handle 0x40000000, the length its
// QueryTextureArrayLengthFromPool / QuerySamplerArrayLengthFromPool answers gave): its GFX_POOL_* kind, or -1 for any
// other resource and for pool arrays of a kind the pool set does not hold (those keep their bound-slot binding).
static int pool_kind_of(int kind, const char *o, const char *e) {
    if (kind != GFX_RES_TEX || json_int(o, e, "cbuf_slot", -1) != 0 || json_int(o, e, "handle", 0) != 0x40000000 ||
        json_int(o, e, "array", 1) < 2)
        return -1;
    const char *sep = json_key(o, e, "separate");
    int type = (int)json_int(o, e, "type", 0), separate = sep && *sep == 't';
    if (!separate) return type == 0 ? GFX_POOL_SAMP : -1;   // the sampler array (type None); no combined arrays
    switch (type & ~0x400) {                                // shadow (0x400) samples the same views
    case 3: return GFX_POOL_2D;
    case 0x103: return GFX_POOL_2D_ARRAY;
    default: return -1;
    }
}

static __thread int no_pool;   // reflect: parsing again after patch_pool_arrays declined

static void add_res(GfxShader *s, int kind, const char *o, const char *e) {
    int pk = no_pool ? -1 : pool_kind_of(kind, o, e);
    if (pk >= 0 && s->npool < (int)(sizeof s->pool_bind / sizeof s->pool_bind[0]) && vk_tpool_available()) {
        s->pool_bind[s->npool] = (uint16_t)json_int(o, e, "binding", 0);
        s->pool_kind[s->npool++] = (uint8_t)pk;
        return;
    }
    GfxShaderRes *r = &s->res[kind][s->nres[kind]++];
    r->set = 0;
    r->binding = (uint16_t)json_int(o, e, "binding", 0);
    if (kind == GFX_RES_UBO || kind == GFX_RES_SSBO) {
        r->slot = (int16_t)json_int(o, e, "slot", -1);
        r->count = 1;
        if (kind == GFX_RES_SSBO) {
            r->cbuf_slot = (int8_t)json_int(o, e, "cb_slot", -1);
#ifdef GAME_NS2
            if (json_int(o, e, "cb_slot", -1) == 255) r->cbuf_slot = GFX_CB_GMEM;   // the pool table and its write log
#endif
            r->handle = (uint16_t)json_int(o, e, "cb_offset", 0);
            r->flags = (uint8_t)json_int(o, e, "flags", 0);
        }
        return;
    }
    char buf[32];
    json_str(o, e, "format", buf, sizeof buf);
    const char *sep = json_key(o, e, "separate");
    r->type = (uint16_t)json_int(o, e, "type", 0);
    r->format = (uint8_t)fmt_code(buf);
    r->flags = (uint8_t)json_int(o, e, "flags", 0);
    r->cbuf_slot = (int8_t)json_int(o, e, "cbuf_slot", -1);
    int handle = (int)json_int(o, e, "handle", 0);
    r->handle = (uint16_t)handle;
    r->count = (uint16_t)json_int(o, e, "array", 1);
    r->separate = sep && *sep == 't';
    r->slot = (int16_t)handle_slot(kind, r->cbuf_slot, handle, r->separate);
}

// SPIR-V opcodes and enumerants used below (SPIR-V 1.6 specification, section 3).
enum { OP_ENTRY_POINT = 15, OP_EXECUTION_MODE = 16, OP_TYPE_INT = 21, OP_TYPE_FLOAT = 22, OP_TYPE_VECTOR = 23,
       OP_TYPE_ARRAY = 28, OP_TYPE_POINTER = 32, OP_VARIABLE = 59, OP_DECORATE = 71,
       DEC_BUILTIN = 11, DEC_LOCATION = 30, DEC_COMPONENT = 31, SC_INPUT = 1, SC_OUTPUT = 3, EM_LOCAL_SIZE = 17 };

// Interface Locations and the compute workgroup size from the module. Ids beyond 2^20 are ignored (Ryujinx modules stay
// far below).
static void reflect_spirv(GfxShader *s) {
    const uint32_t *w = s->spirv;
    uint64_t n = s->words;
    s->local[0] = s->local[1] = s->local[2] = 1;
    if (n < 5) return;
    uint32_t bound = w[3] < (1u << 20) ? w[3] : (1u << 20);
    // per id: location + 1 (0 = none), component flag, scalar/vector description of types, pointee of pointers
    int32_t *loc = calloc(bound, sizeof *loc);
    uint8_t *packed = calloc(bound, 1), *ttype = calloc(bound, 1), *tcomps = calloc(bound, 1);
    uint32_t *pointee = calloc(bound, sizeof *pointee);
    if (!loc || !packed || !ttype || !tcomps || !pointee) goto out;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) break;
        const uint32_t *a = w + i + 1;
        uint32_t id = len > 1 ? a[0] : 0;
        if (op == OP_EXECUTION_MODE && len >= 6 && a[1] == EM_LOCAL_SIZE) {
            s->local[0] = a[2], s->local[1] = a[3], s->local[2] = a[4];
        } else if (op == OP_DECORATE && len >= 4 && id < bound) {
            if (a[1] == DEC_LOCATION) loc[id] = (int32_t)a[2] + 1;
            else if (a[1] == DEC_COMPONENT) packed[id] = 1;
        } else if (op == OP_TYPE_INT && len >= 4 && id < bound) {
            ttype[id] = a[2] ? GFX_TYPE_SINT : GFX_TYPE_UINT, tcomps[id] = 1;
        } else if (op == OP_TYPE_FLOAT && len >= 3 && id < bound) {
            ttype[id] = GFX_TYPE_FLOAT, tcomps[id] = 1;
        } else if (op == OP_TYPE_VECTOR && len >= 4 && id < bound && a[1] < bound) {
            ttype[id] = ttype[a[1]], tcomps[id] = (uint8_t)a[2];
        } else if (op == OP_TYPE_ARRAY && len >= 4 && id < bound && a[1] < bound) {
            ttype[id] = ttype[a[1]], tcomps[id] = tcomps[a[1]];   // element type (arrayed interface blocks)
        } else if (op == OP_TYPE_POINTER && len >= 4 && id < bound) {
            pointee[id] = a[2];
        } else if (op == OP_VARIABLE && len >= 4) {
            uint32_t type = a[0], var = a[1], sc = a[2];
            if ((sc == SC_INPUT || sc == SC_OUTPUT) && var < bound && type < bound && loc[var] > 0 && loc[var] <= 32) {
                int l = loc[var] - 1;
                uint32_t t = pointee[type] < bound ? pointee[type] : 0;
                GfxShaderIo io = {ttype[t], tcomps[t] ? tcomps[t] : 4, packed[var]};
                if (sc == SC_INPUT) s->in_mask |= 1u << l, s->in[l] = io;
                else s->out_mask |= 1u << l, s->out[l] = io;
            }
        }
        i += len;
    }
out:
    free(loc), free(packed), free(ttype), free(tcomps), free(pointee);
}

// Matrix palette reflection for skinned vertex shaders:
//   joint value   a value loaded from an Input variable of an integer type that has a Location (and conversions,
//                 bitcasts, extracts, local variable round-trips), or the raw 32-bit bits of location 6 component 0
//                 (for the c6 48-byte palette only);
//   matrix offset a joint value multiplied by 48 or 64 (or shifted left by 6), propagated through integer arithmetic;
//   palette       a UBO or SSBO variable indexed by a matrix offset.
static void reflect_palette_pass(GfxShader *s, int raw_only) {
    enum { P_CONSTANT = 43, P_FUNCTION = 54, P_LOAD = 61, P_STORE = 62, P_ACCESS_CHAIN = 65,
           P_IN_BOUNDS_ACCESS_CHAIN = 66, P_COMPOSITE_EXTRACT = 81, P_COPY_OBJECT = 83, P_UCONVERT = 113,
           P_SCONVERT = 114, P_BITCAST = 124, P_IADD = 128, P_ISUB = 130, P_IMUL = 132, P_SHR = 194, P_SAR = 195,
           P_SHL = 196, P_OR = 197, P_AND = 199, P_DEC_BINDING = 33, P_SC_UNIFORM = 2, P_SC_FUNCTION = 7,
           P_SC_STORAGE = 12, P_RAW_JOINT = 0x80, P_RAW_VECTOR = 1, P_RAW_POINTER = 2, P_RAW_VALUE = 3,
           P_RAW_LOCAL = 4, P_RAW_SINGLE_STORE = 5, P_RAW_MULTI_STORE = 6 };
    const uint32_t *w = s->spirv;
    uint64_t n = s->words, fn = 0;
    if (s->stage != GFX_STAGE_VS || n < 5) return;
    int any = raw_only && (s->in_mask >> 6 & 1);
    for (int l = 0; !raw_only && l < 32 && !any; l++)
        any = (s->in_mask >> l & 1) && s->in[l].type != GFX_TYPE_FLOAT;
    if (!any) return;
    uint32_t bound = w[3] < (1u << 20) ? w[3] : (1u << 20);
    uint8_t *isint = calloc(bound, 1), *con = calloc(bound, 1), *loc = calloc(bound, 1), *block = calloc(bound, 1);
    uint8_t *joint = calloc(bound, 1), *scaled = calloc(bound, 1);
    uint32_t *pointee = calloc(bound, sizeof *pointee), *bind = calloc(bound, sizeof *bind);
    uint8_t *raw = raw_only ? calloc(bound, 1) : NULL;
    uint32_t found = 0, found_bind = 0, found_joints = 0, found_raw = 0;
    int two = 0;
    if (!isint || !con || !loc || !block || !joint || !scaled || !pointee || !bind || (raw_only && !raw)) goto out;
    #define PID(x) ((x) < bound)
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) goto out;
        if (op == P_FUNCTION) { fn = i; break; }
        const uint32_t *a = w + i;
        if (op == OP_DECORATE && len >= 4 && PID(a[1])) {
            if (a[2] == DEC_LOCATION && a[3] < 32) loc[a[1]] = (uint8_t)(a[3] + 1);
            else if (a[2] == P_DEC_BINDING) bind[a[1]] = a[3] + 1;
        } else if (op == OP_TYPE_INT && len >= 4 && PID(a[1])) {
            isint[a[1]] = a[2] == 32 ? 2 : 1;
        } else if (raw_only && op == OP_TYPE_FLOAT && len >= 3 && PID(a[1]) && a[2] == 32) {
            raw[a[1]] = P_RAW_VALUE;
        } else if (op == OP_TYPE_VECTOR && len >= 4 && PID(a[1]) && PID(a[2])) {
            isint[a[1]] = !!isint[a[2]];
            if (raw_only && raw[a[2]] == P_RAW_VALUE) raw[a[1]] = P_RAW_VECTOR;
        } else if (op == OP_TYPE_POINTER && len >= 4 && PID(a[1])) {
            pointee[a[1]] = a[3];
        } else if (op == P_CONSTANT && len == 4 && PID(a[1]) && PID(a[2]) && isint[a[1]] &&
                   (a[3] == 0 || a[3] == 6 || a[3] == 48 || a[3] == 64)) {
            con[a[2]] = (uint8_t)(a[3] + 1);
        } else if (op == OP_VARIABLE && len >= 4 && PID(a[1]) && PID(a[2])) {
            uint32_t t = pointee[a[1]];
            if (!raw_only && a[3] == SC_INPUT && loc[a[2]] && PID(t) && isint[t]) joint[a[2]] = loc[a[2]];
            else if (raw_only && a[3] == SC_INPUT && loc[a[2]] == 7 && PID(t) && raw[t] == P_RAW_VECTOR)
                raw[a[2]] = P_RAW_VECTOR;
            else if ((a[3] == P_SC_UNIFORM || a[3] == P_SC_STORAGE) && bind[a[2]] > 1) block[a[2]] = 1;
        }
        i += len;
    }
    if (!fn) goto out;
    if (raw_only) for (uint64_t i = fn; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) goto out;
        const uint32_t *a = w + i;
        if (op == OP_VARIABLE && len == 4 && PID(a[1]) && PID(a[2]) && a[3] == P_SC_FUNCTION &&
            PID(pointee[a[1]]) && raw[pointee[a[1]]] == P_RAW_VALUE) raw[a[2]] = P_RAW_LOCAL;
        else if (op == P_STORE && len >= 3 && PID(a[1])) {
            if (raw[a[1]] == P_RAW_LOCAL) raw[a[1]] = P_RAW_SINGLE_STORE;
            else if (raw[a[1]] == P_RAW_SINGLE_STORE) raw[a[1]] = P_RAW_MULTI_STORE;
        }
        i += len;
    }
    #define PJOINT(to, from) do { if (!joint[to] && joint[from]) joint[to] = joint[from], changed = 1; } while (0)
    #define PSCALED(to, from) \
        do { if (!scaled[to] && scaled[from]) scaled[to] = scaled[from], joint[to] = joint[from], changed = 1; } while (0)
    #define PRAW(to, kind) do { if (!raw[to]) raw[to] = (kind), changed = 1; } while (0)
    for (int pass = 0, changed = 1; pass < 4 && changed; pass++) {
        changed = 0;
        for (uint64_t i = fn; i < n;) {
            uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
            if (!len || i + len > n) goto out;
            const uint32_t *a = w + i;
            if ((op == P_ACCESS_CHAIN || op == P_IN_BOUNDS_ACCESS_CHAIN) && len >= 4 && PID(a[2]) && PID(a[3])) {
                PJOINT(a[2], a[3]);
                if (raw_only && len == 5 && raw[a[3]] == P_RAW_VECTOR && PID(a[4]) && con[a[4]] == 1)
                    PRAW(a[2], P_RAW_POINTER);
                for (uint32_t k = 4; block[a[3]] && k < len; k++) {
                    if (!PID(a[k]) || !scaled[a[k]]) continue;
                    if (found && found_bind != bind[a[3]]) two = 1;
                    found = scaled[a[k]], found_bind = bind[a[3]];
                    if (joint[a[k]] == P_RAW_JOINT) found_raw = 1u << 6;
                    else if (joint[a[k]]) found_joints |= 1u << (joint[a[k]] - 1);
                }
            } else if ((op == P_LOAD || op == P_BITCAST || op == P_UCONVERT || op == P_SCONVERT || op == P_COPY_OBJECT ||
                        op == P_COMPOSITE_EXTRACT) && len >= 4 && PID(a[2]) && PID(a[3])) {
                PJOINT(a[2], a[3]);
                PSCALED(a[2], a[3]);
                if (raw_only && op == P_LOAD && raw[a[3]] == P_RAW_POINTER && PID(a[1]) && raw[a[1]] == P_RAW_VALUE)
                    PRAW(a[2], P_RAW_VALUE);
                if (raw_only && op == P_BITCAST && raw[a[3]] == P_RAW_VALUE && PID(a[1]) && isint[a[1]] == 2 && !joint[a[2]])
                    joint[a[2]] = P_RAW_JOINT, changed = 1;
            } else if (op == P_STORE && len >= 3 && PID(a[1]) && PID(a[2]) && !block[a[1]]) {
                PJOINT(a[1], a[2]);
                PSCALED(a[1], a[2]);
                if (raw_only && raw[a[2]] == P_RAW_VALUE && raw[a[1]] == P_RAW_SINGLE_STORE)
                    raw[a[1]] = P_RAW_POINTER, changed = 1;
            } else if (op == P_IMUL && len >= 5 && PID(a[2]) && PID(a[3]) && PID(a[4])) {
                for (int k = 0; k < 2; k++) {
                    uint32_t x = a[3 + k], c = a[4 - k];
                    if (!scaled[a[2]] && joint[x] && !scaled[x] && (con[c] == 49 || (con[c] == 65 && joint[x] != P_RAW_JOINT)))
                        scaled[a[2]] = (uint8_t)(con[c] - 1), joint[a[2]] = joint[x], changed = 1;
                }
            } else if (op == P_SHL && len >= 5 && PID(a[2]) && PID(a[3]) && PID(a[4])) {
                if (!scaled[a[2]] && joint[a[3]] && joint[a[3]] != P_RAW_JOINT && !scaled[a[3]] && con[a[4]] == 7)
                    scaled[a[2]] = 64, joint[a[2]] = joint[a[3]], changed = 1;
            } else if ((op == P_IADD || op == P_ISUB || op == P_SHR || op == P_SAR || op == P_AND || op == P_OR) &&
                       len >= 5 && PID(a[2]) && PID(a[3]) && PID(a[4])) {
                PSCALED(a[2], a[3]);
                PSCALED(a[2], a[4]);
            }
            i += len;
        }
    }
    #undef PRAW
    #undef PJOINT
    #undef PSCALED
    #undef PID
    if (!found || two || !(found_joints | found_raw)) goto out;
    for (int kind = GFX_RES_UBO; kind <= GFX_RES_SSBO; kind++)
        for (int i = 0; i < s->nres[kind]; i++)
            if ((uint32_t)s->res[kind][i].binding + 1 == found_bind && s->res[kind][i].slot >= 0) {
                if (found_raw && (found_joints || found != 48 || kind != GFX_RES_UBO || s->res[kind][i].slot != 6))
                    continue;
                s->pal_raw_joints = found_raw;
                s->pal_stride = (uint8_t)found, s->pal_storage = kind == GFX_RES_SSBO;
                s->pal_slot = s->res[kind][i].slot, s->pal_joints = found_joints;
            }
out:
    free(isint), free(con), free(loc), free(block), free(joint), free(scaled), free(pointee), free(bind), free(raw);
}

static void reflect_palette(GfxShader *s) {
    s->pal_stride = 0, s->pal_slot = -1, s->pal_raw_joints = 0;
    reflect_palette_pass(s, 0);
    if (!s->pal_stride) reflect_palette_pass(s, 1);
}

// Storage buffers the shader adds into with float atomics get GFX_SSBO_FADD: an OpAtomicFAddEXT whose pointer leads
// (through access chains and copies) to the variable decorated with the buffer's binding. Their mirror ranges go back
// to guest memory as additions (vk_memory.c, write-back merge). Without VK_EXT_shader_atomic_float
// (gfx_shader_float_atomics 0: Radeons of RDNA 1 and 2) the translator makes the same add a compare-and-swap loop: an
// OpAtomicCompareExchange whose value is made from an OpFAdd. That counts as well. Unmarked (1.2.8 and before: the rule
// looked for a bit cast of the OpFAdd itself and found none of the translator's loops), such a sum went back as a
// plain value over the CPU's reset of it wherever a mirror holds it, where the native add's sums are merged.
static void mark_fadd_ssbos(GfxShader *s) {
    const uint32_t *w = s->spirv;
    uint64_t n = s->words;
    if (n < 5 || !s->nres[GFX_RES_SSBO]) return;
    uint32_t bound = w[3] < (1u << 20) ? w[3] : (1u << 20), nptr = 0, cptr = 64;
    uint32_t *from = calloc(bound, sizeof *from), *bind = calloc(bound, sizeof *bind), *ptr = malloc(cptr * sizeof *ptr);
    uint8_t *fsum = calloc(bound, 1);   // 1: a value made from an OpFAdd's result, or a pointer a sum was stored through
    if (!from || !bind || !ptr || !fsum) goto out;
    // A sum: an OpFAdd's result and what the translator's float-add loop makes of it before the swap. Its registers are
    // function variables, so the sum is stored into one and loaded back; then bit casts (and copies), and the select
    // that flushes a denormal sum to 0 (InstEmitAluHelper.FlushDenormal): OpBitcast(OpSelect(|x| < min, 0,
    // OpBitcast(OpLoad(%reg)))) with an OpStore %reg OpFAdd(...) before it. A store may come after its load in the
    // module's order (a loop), so three passes; the third collects the atomics.
    for (int pass = 0; pass < 3; pass++)
        for (uint64_t i = 5; i < n;) {
            uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
            if (!len || i + len > n) break;
            const uint32_t *a = w + i + 1;
            if (pass == 0) {
                if (op == OP_DECORATE && len >= 4 && a[1] == 33 /* Binding */ && a[0] < bound) bind[a[0]] = a[2] + 1;
                if ((op == 65 || op == 66 || op == 67 || op == 83) && len >= 4 && a[1] < bound) from[a[1]] = a[2];
            }
            if (op == 129 /* OpFAdd */ && len >= 5 && a[1] < bound) fsum[a[1]] = 1;
            else if ((op == 124 /* OpBitcast */ || op == 83 /* OpCopyObject */ || op == 61 /* OpLoad */) && len >= 4 &&
                     a[1] < bound && a[2] < bound && fsum[a[2]])
                fsum[a[1]] = 1;
            else if (op == 169 /* OpSelect */ && len >= 6 && a[1] < bound &&
                     ((a[3] < bound && fsum[a[3]]) || (a[4] < bound && fsum[a[4]])))
                fsum[a[1]] = 1;
            else if (op == 62 /* OpStore */ && len >= 3 && a[0] < bound && a[1] < bound && fsum[a[1]])
                fsum[a[0]] = 1;
            int cas = op == 230 /* OpAtomicCompareExchange */ && len >= 9 && a[6] < bound && fsum[a[6]];
            if (pass == 2 && ((op == 6035 /* OpAtomicFAddEXT */ && len >= 4 && a[2] < bound) || (cas && a[2] < bound))) {
                if (nptr == cptr) {
                    uint32_t *q = realloc(ptr, (cptr *= 2) * sizeof *ptr);
                    if (!q) goto out;
                    ptr = q;
                }
                ptr[nptr++] = a[2];
            }
            i += len;
        }
    {   // DEBUG-CASLOG: REKARINA_CASLOG=1: every compare-and-swap of a shader with the opcodes that make its value (up to
        // three levels), to learn the translator's float-add loop
        static int caslog = -1;
        if (caslog < 0) caslog = cfg_get("REKARINA_CASLOG") != NULL;
        if (caslog) {
            uint32_t *def = calloc(bound, sizeof *def);   // result id -> word offset of its instruction
            for (uint64_t i = 5; def && i < n;) {
                uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
                if (!len || i + len > n) break;
                if (len >= 3 && op != OP_DECORATE && w[i + 2] < bound && op != 54 /* OpFunction */) def[w[i + 2]] = (uint32_t)i;
                if (op == 230 && len >= 9) {
                    char line[256];
                    int o = snprintf(line, sizeof line, "CASLOG shader %02x%02x%02x%02x: cas value %u", s->iid[0], s->iid[1], s->iid[2], s->iid[3], w[i + 7]);
                    uint32_t v = w[i + 7];
                    for (int lv = 0; lv < 4 && v < bound && def[v] && o < 200; lv++) {
                        uint32_t d = def[v], dop = w[d] & 0xffff, dlen = w[d] >> 16;
                        o += snprintf(line + o, sizeof line - (size_t)o, " <- op %u", dop);
                        v = dlen >= 4 ? w[d + 3] : 0;
                    }
                    rt_log("%s", line);
                }
                i += len;
            }
            free(def);
        }
    }
    int marked = 0, unbound = 0;
    for (uint32_t k = 0; k < nptr; k++) {
        uint32_t v = ptr[k];
        for (int step = 0; step < 64 && from[v] && from[v] < bound; step++) v = from[v];
        if (!bind[v]) {
            unbound++;
            continue;
        }
        for (int j = 0; j < s->nres[GFX_RES_SSBO]; j++)
            if (s->res[GFX_RES_SSBO][j].binding == bind[v] - 1) s->res[GFX_RES_SSBO][j].flags |= GFX_SSBO_FADD, marked++;
    }
    if (nptr && !gfx_shader_float_atomics) {   // the compare-and-swap path: which shaders' sums are marked (a few a run)
        static volatile int32_t said;
        if (__atomic_add_fetch(&said, 1, __ATOMIC_RELAXED) <= 24)
            rt_log("gfx_shader: %02x%02x%02x%02x: %u float sums by compare-and-swap, %d storage buffer bindings marked as "
                   "sums, %d through raw pointers", s->iid[0], s->iid[1], s->iid[2], s->iid[3], nptr, marked, unbound);
    }
out:
    free(from), free(bind), free(ptr), free(fsum);
}

// Every gradient sample (OpImageSampleExplicitLod with Grad) whose two gradients are each built from two components gets
// them transposed, dx = (a, c) and dy = (b, d) from dx = (a, b) and dy = (c, d), by two new OpCompositeConstruct before
// it. TotK's shaders fill a TXD's gradient registers per coordinate (du/dx, du/dy, dv/dx, dv/dy), and the translator
// (Ryujinx, InstEmitTexture EmitTxd) builds dx = (first two registers), dy = (last two) = (du/dx, du/dy) and
// (dv/dx, dv/dy): a mip level chosen from the wrong derivatives, too sharp on terrain faces turned across the screen,
// which crawled in patches that followed the faces (1.4.3, the Depths). REKARINA_TXD_TRANSPOSE=0 turns it off.
// Applied to the SPIR-V after it is loaded, so the shader cache is untouched.
enum { OP_COMPOSITE_CONSTRUCT = 80, OP_SAMPLE_EXPLICIT = 88, IO_GRAD = 4 };

static void patch_txd_transpose(GfxShader *s) {
    static int on = -1;
    static volatile int32_t n_logged;
    if (on < 0) {
        const char *e = cfg_get("REKARINA_TXD_TRANSPOSE");
        on = !(e && *e == '0');
    }
    uint32_t *w = s->spirv;
    uint64_t n = s->words;
    if (!on || n < 5) return;
    uint32_t bound = w[3] < (1u << 20) ? w[3] : (1u << 20);
    uint64_t *cc = calloc(bound, sizeof *cc);   // word index of a 2-constituent OpCompositeConstruct defining the id
    int found = 0;
    if (!cc) return;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) goto out;
        if (op == OP_COMPOSITE_CONSTRUCT && len == 5 && w[i + 2] < bound) cc[w[i + 2]] = i;
        if (op == OP_SAMPLE_EXPLICIT && len >= 8 && w[i + 5] == IO_GRAD && w[i + 6] < bound && w[i + 7] < bound &&
            cc[w[i + 6]] && cc[w[i + 7]] && w[cc[w[i + 6]] + 1] == w[cc[w[i + 7]] + 1])
            found++;
        i += len;
    }
    if (!found) goto out;
    uint32_t *o = malloc((n + (uint64_t)found * 10) * sizeof *o);
    if (!o) goto out;
    uint64_t k = 0;
    uint32_t next = w[3];
    for (uint64_t i = 0; i < 5; i++) o[k++] = w[i];
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == OP_SAMPLE_EXPLICIT && len >= 8 && w[i + 5] == IO_GRAD && w[i + 6] < bound && w[i + 7] < bound &&
            cc[w[i + 6]] && cc[w[i + 7]] && w[cc[w[i + 6]] + 1] == w[cc[w[i + 7]] + 1]) {
            const uint32_t *dx = w + cc[w[i + 6]], *dy = w + cc[w[i + 7]];   // [hdr, type, id, a, b], [hdr, type, id, c, d]
            uint32_t ndx = next++, ndy = next++;
            o[k++] = 5u << 16 | OP_COMPOSITE_CONSTRUCT, o[k++] = dx[1], o[k++] = ndx, o[k++] = dx[3], o[k++] = dy[3];
            o[k++] = 5u << 16 | OP_COMPOSITE_CONSTRUCT, o[k++] = dx[1], o[k++] = ndy, o[k++] = dx[4], o[k++] = dy[4];
            for (uint32_t j = 0; j < len; j++) o[k++] = w[i + j];
            o[k - len + 6] = ndx, o[k - len + 7] = ndy;
        } else {
            for (uint32_t j = 0; j < len; j++) o[k++] = w[i + j];
        }
        i += len;
    }
    o[3] = next;
    free(s->spirv);
    s->spirv = o, s->words = k;
    if (plat_inc32(&n_logged) <= 4) rt_log("gfx_shader: %d gradient samples now read their derivatives per coordinate", found);
out:
    free(cc);
}

// ---- arrays over the texture pool (gfx.h GFX_POOL_SET)
// The translator declares such an array with 32 elements in set 0 and clamps the index it takes from the handle:
// %i = OpExtInst %uint GLSL.std.450 UMin %id %c31 (texture id: handle & 0xfffff; sampler id: handle >> 20), maybe
// behind OpBitcasts. Each one becomes an array of the pool's size in set GFX_POOL_SET at its kind's binding, and the
// clamps of its indexing take the new last index. The variables move to the end of the global section, after the
// widened array and pointer types declared there. 0 when the module is not as expected (nothing is changed).
enum { PO_EXT_IMPORT = 11, PO_EXT_INST = 12, PO_CONSTANT = 43, PO_FUNCTION = 54, PO_ACCESS = 65, PO_ACCESS_IB = 66,
       PO_BITCAST = 124, PO_UMIN = 38, PO_SMIN = 39, PO_DEC_BINDING = 33, PO_DEC_SET = 34, PO_SC_UNIFORM_CONSTANT = 0 };

static int patch_pool_arrays(GfxShader *s) {
    uint32_t *w = s->spirv;
    uint64_t n = s->words;
    if (n < 5 || w[3] >= (1u << 22)) return 0;
    uint32_t bound = w[3], glsl = 0, nvar = 0, var[8];
    int8_t vkind[8];
    uint64_t *def = calloc(bound, sizeof *def), *setw = calloc(bound, sizeof *setw), *bindw = calloc(bound, sizeof *bindw);
    uint64_t first_fn = 0, *clamp = NULL;   // word positions of the clamp constants to replace
    uint32_t nclamp = 0, *clamp_n = NULL;   // per clamp: the new last index
    int ok = 0;
    if (!def || !setw || !bindw) goto out;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) goto out;
        if (op == PO_EXT_IMPORT && len >= 3 && !strncmp((const char *)&w[i + 2], "GLSL.std.450", (len - 2) * 4)) glsl = w[i + 1];
        if (op == OP_DECORATE && len >= 4 && w[i + 1] < bound) {
            if (w[i + 2] == PO_DEC_SET) setw[w[i + 1]] = i + 3;
            if (w[i + 2] == PO_DEC_BINDING) bindw[w[i + 1]] = i + 3;
        }
        if ((op == OP_TYPE_ARRAY || op == OP_TYPE_POINTER) && len >= 4 && w[i + 1] < bound) def[w[i + 1]] = i;
        if ((op == PO_CONSTANT || op == OP_VARIABLE || op == PO_BITCAST || op == PO_EXT_INST) && len >= 4 && w[i + 2] < bound)
            def[w[i + 2]] = i;
        if (op == PO_FUNCTION && !first_fn) first_fn = i;
        i += len;
    }
    if (!first_fn) goto out;
    for (uint32_t id = 1; id < bound; id++) {   // the pool variables: set 0, a binding the reflection named
        uint64_t d = def[id];
        if (!d || (w[d] & 0xffff) != OP_VARIABLE || !bindw[id] || !setw[id] || w[setw[id]] != 0) continue;
        for (int k = 0; k < s->npool; k++)
            if (w[bindw[id]] == s->pool_bind[k] && nvar < 8) var[nvar] = id, vkind[nvar++] = (int8_t)s->pool_kind[k];
    }
    if (!nvar) goto out;
    uint32_t arr_of[8], elem[8], lent[8], oldlen[8];   // per variable: its array type, element type, length's type
    for (uint32_t v = 0; v < nvar; v++) {
        uint64_t d = def[var[v]], dp = def[w[d + 1]];
        if (!dp || (w[dp] & 0xffff) != OP_TYPE_POINTER || w[dp + 2] != PO_SC_UNIFORM_CONSTANT) goto out;
        uint64_t da = def[w[dp + 3]];
        if (!da || (w[da] & 0xffff) != OP_TYPE_ARRAY) goto out;
        uint64_t dl = def[w[da + 3]];
        if (!dl || (w[dl] & 0xffff) != PO_CONSTANT || (w[dl] >> 16) != 4) goto out;
        arr_of[v] = w[dp + 3], elem[v] = w[da + 2], lent[v] = w[dl + 1], oldlen[v] = w[dl + 3];
    }
    (void)arr_of;
    // The clamps: an index of an access chain into a pool variable, through OpBitcasts, from UMin / SMin with the old
    // last index as one operand.
    clamp = malloc(sizeof *clamp * 256), clamp_n = malloc(sizeof *clamp_n * 256);
    if (!clamp || !clamp_n) goto out;
    for (uint64_t i = first_fn; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if ((op == PO_ACCESS || op == PO_ACCESS_IB) && len >= 5) {
            uint32_t v = 0;
            while (v < nvar && var[v] != w[i + 3]) v++;
            if (v < nvar) {
                uint32_t idx = w[i + 4], want = vkind[v] == GFX_POOL_SAMP ? GFX_POOL_SAMPLERS - 1 : GFX_POOL_TEXTURES - 1;
                for (int step = 0; step < 4 && idx < bound && def[idx] && (w[def[idx]] & 0xffff) == PO_BITCAST; step++)
                    idx = w[def[idx] + 3];
                uint64_t m = idx < bound ? def[idx] : 0;
                if (m && (w[m] & 0xffff) == PO_EXT_INST && (w[m] >> 16) == 7 && w[m + 3] == glsl &&
                    (w[m + 4] == PO_UMIN || w[m + 4] == PO_SMIN))
                    for (int o = 5; o <= 6; o++) {
                        uint64_t c = w[m + o] < bound ? def[w[m + o]] : 0;
                        if (c && (w[c] & 0xffff) == PO_CONSTANT && (w[c] >> 16) == 4 && w[c + 3] == oldlen[v] - 1) {
                            uint32_t j = 0;
                            while (j < nclamp && clamp[j] != m + o) j++;
                            if (j == nclamp && nclamp < 256) clamp[nclamp] = m + o, clamp_n[nclamp++] = want;
                        }
                    }
            }
        }
        i += len;
    }
    // New declarations: per variable a length constant, the array and the pointer; per clamp a constant of the clamp's
    // type (the old constant's), shared when equal.
    uint32_t next = bound, extra[8 * 12 + 256 * 4], ne = 0, newptr[8];
    uint32_t ctype[256], cval[256], cid[256], nc = 0;
    for (uint32_t j = 0; j < nclamp; j++) {
        uint32_t t = w[def[w[clamp[j]]] + 1], val = clamp_n[j], k = 0;
        while (k < nc && !(ctype[k] == t && cval[k] == val)) k++;
        if (k == nc) {
            ctype[nc] = t, cval[nc] = val, cid[nc++] = next++;
            extra[ne++] = 4u << 16 | PO_CONSTANT, extra[ne++] = t, extra[ne++] = cid[k], extra[ne++] = val;
        }
        clamp_n[j] = cid[k];   // now the id that replaces the clamp operand
    }
    for (uint32_t v = 0; v < nvar; v++) {
        uint32_t len = vkind[v] == GFX_POOL_SAMP ? GFX_POOL_SAMPLERS : GFX_POOL_TEXTURES, lc = next++, at = next++;
        newptr[v] = next++;
        extra[ne++] = 4u << 16 | PO_CONSTANT, extra[ne++] = lent[v], extra[ne++] = lc, extra[ne++] = len;
        extra[ne++] = 4u << 16 | OP_TYPE_ARRAY, extra[ne++] = at, extra[ne++] = elem[v], extra[ne++] = lc;
        extra[ne++] = 4u << 16 | OP_TYPE_POINTER, extra[ne++] = newptr[v], extra[ne++] = PO_SC_UNIFORM_CONSTANT,
        extra[ne++] = at;
    }
    uint32_t *o = malloc((n + ne) * sizeof *o);
    if (!o) goto out;
    uint64_t k = 0;
    for (uint64_t i = 0; i < 5; i++) o[k++] = w[i];
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16, v = nvar;
        if (op == OP_VARIABLE) {
            v = 0;
            while (v < nvar && var[v] != w[i + 2]) v++;
        }
        if (i == first_fn) {   // the declarations, then the pool variables with their new types
            memcpy(o + k, extra, ne * sizeof *o), k += ne;
            for (uint32_t u = 0; u < nvar; u++) {
                uint64_t d = def[var[u]];
                for (uint32_t j = 0; j < (w[d] >> 16); j++) o[k++] = w[d + j];
                o[k - (w[d] >> 16) + 1] = newptr[u];
            }
        }
        if (v < nvar) {   // moved to the end of the global section
            i += len;
            continue;
        }
        for (uint32_t j = 0; j < len; j++) o[k++] = w[i + j];
        if (op == OP_DECORATE && len >= 4) {
            for (uint32_t u = 0; u < nvar; u++)
                if (w[i + 1] == var[u]) {
                    if (w[i + 2] == PO_DEC_SET) o[k - len + 3] = GFX_POOL_SET;
                    if (w[i + 2] == PO_DEC_BINDING) o[k - len + 3] = (uint32_t)vkind[u];
                }
        }
        if (op == PO_EXT_INST)
            for (uint32_t j = 0; j < nclamp; j++)
                if (clamp[j] >= i && clamp[j] < i + len) o[k - len + (clamp[j] - i)] = clamp_n[j];
        i += len;
    }
    o[3] = next;
    free(s->spirv);
    s->spirv = o, s->words = k;
    for (uint32_t v = 0; v < nvar; v++) s->pool_mask |= 1u << vkind[v];
    ok = 1;
    static volatile int32_t n_logged;
    if (plat_inc32(&n_logged) <= 3)
        rt_log("gfx_shader: %u texture pool arrays widened to the pool (set %d), %u index clamps moved", nvar, GFX_POOL_SET,
               nclamp);
out:
    free(def), free(setw), free(bindw), free(clamp), free(clamp_n);
    return ok;
}

// ---- non-uniform descriptor indexing (gfx_shader_nonuniform, REKARINA_NONUNIFORM)
// The translator indexes arrays of textures and samplers (the texture pool's, patch_pool_arrays, and bound-slot
// arrays) with an index it computes from a handle, and never decorates such an access NonUniform: tools/xlate/Program.cs
// leaves Ryujinx's QueryHostSupportsShaderNonUniformIndexing at false. Where the handle is read from per-pixel data (an
// entry of a light list at an address made from the pixel's position: the clustered lights' shadow maps, the water's
// light handles), the lanes of one wave index different elements. NVIDIA's hardware takes a descriptor per lane either
// way; AMD keeps a descriptor in scalar registers, and its compiler, told the index is the same for the whole wave,
// takes the first active lane's element for all of them: another light's shadow map or texture for part of the wave,
// changing as the waves fall differently from frame to frame (Longshot 7315957c indexes OoT's per-light shadow arrays
// NonUniform for this). Mode 1 decorates every access chain into such an array whose index depends on a per-invocation
// value: an Input variable, an atomic's result or a subgroup election, through any arithmetic, a load at an address made
// from one, a function variable stored from one, a call's argument or result; with it the element's load and the
// sampled image made from that, and the module gets the extension and the capabilities. Mode 2 decorates every access
// with a computed index. The module's ids and everything else stay as they were.
int gfx_shader_nonuniform;   // gfx.h: 0 off, 1 per-invocation indices, 2 every computed index (vk_device.c)

enum { NU_TAINT = 1, NU_TYPE = 2, NU_ARRVAR = 4, NU_CONST = 8, NU_LOCAL = 16, NU_MARK = 32, NU_RET = 64 };

static void patch_nonuniform(GfxShader *s) {
    int mode = gfx_shader_nonuniform;
    uint32_t *w = s->spirv;
    uint64_t n = s->words;
    if (!mode || n < 5 || w[3] >= (1u << 22)) return;
    uint32_t bound = w[3];
    uint8_t *f = calloc(bound, 1);
    uint32_t *base = calloc(bound, sizeof *base), *pfirst = calloc(bound, sizeof *pfirst), *pcount = calloc(bound, sizeof *pcount);
    uint32_t *params = NULL, nparams = 0;
    uint8_t *imgtype = calloc(bound, 1), *arrtype = calloc(bound, 1), *ptrarr = calloc(bound, 1);
    if (!f || !base || !pfirst || !pcount || !imgtype || !arrtype || !ptrarr) goto out;
    params = malloc(sizeof *params * 4096);
    if (!params) goto out;
    int have_cap_nu = 0, have_cap_arr = 0, have_ext = 0;
    uint64_t first_fn = 0;
    uint32_t cur_fn = 0;
    // Declarations: types, the image arrays and their variables, constants, Input variables, parameters per function.
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) goto out;
        const uint32_t *a = w + i + 1;
        if (op == 17 && len == 2 && a[0] == 5301) have_cap_nu = 1;
        if (op == 17 && len == 2 && a[0] == 5307) have_cap_arr = 1;
        if (op == 10 && len >= 2 && !strncmp((const char *)a, "SPV_EXT_descriptor_indexing", (len - 1) * 4)) have_ext = 1;
        if (((op >= 19 && op <= 39) || op == 4456 || op == 5358) && len >= 2 && a[0] < bound) {   // OpType*, cooperative matrices
            f[a[0]] |= NU_TYPE;
            // an image that is sampled (Sampled 1), a sampler, a sampled image
            if ((op == 25 && len >= 9 && a[6] == 1) || op == 26 || op == 27) imgtype[a[0]] = 1;
            if ((op == 28 || op == 29) && len >= 3 && a[1] < bound && imgtype[a[1]]) arrtype[a[0]] = 1;
            if (op == 32 && len >= 4 && a[2] < bound && arrtype[a[2]]) ptrarr[a[0]] = 1;
        }
        if (op >= 41 && op <= 52 && len >= 3 && a[1] < bound) f[a[1]] |= NU_CONST;   // constants, spec constants
        if (op == OP_VARIABLE && len >= 4 && a[1] < bound) {
            if (a[0] < bound && ptrarr[a[0]]) f[a[1]] |= NU_ARRVAR;
            if (a[2] == SC_INPUT) f[a[1]] |= NU_TAINT;
            if (a[2] == 7 /* Function */ || a[2] == 6 /* Private */ || a[2] == SC_OUTPUT) f[a[1]] |= NU_LOCAL;
        }
        if (op == 54 && len >= 5) {
            cur_fn = a[1] < bound ? a[1] : 0;
            if (!first_fn) first_fn = i;
            if (cur_fn) pfirst[cur_fn] = nparams;
        }
        if (op == 55 && len >= 3 && cur_fn && a[1] < bound && nparams < 4096) params[nparams++] = a[1], pcount[cur_fn]++;
        if ((op == 65 || op == 66) && len >= 4 && a[1] < bound && a[2] < bound)   // the variable an access chain is into
            base[a[1]] = base[a[2]] ? base[a[2]] : a[2];
        i += len;
    }
    if (!first_fn) goto out;
    // Taint: forward over the function bodies until nothing changes (loops carry values back).
    for (int pass = 0, changed = 1; changed && pass < 32; pass++) {
        changed = 0;
        for (uint64_t i = first_fn; i < n;) {
            uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
            const uint32_t *a = w + i + 1;
            if (op == 54) cur_fn = a[1] < bound ? a[1] : 0;
            if (op == 62 && len >= 3 && a[0] < bound && a[1] < bound && (f[a[1]] & NU_TAINT)) {   // OpStore
                uint32_t v = base[a[0]] ? base[a[0]] : a[0];
                if ((f[v] & (NU_LOCAL | NU_TAINT)) == NU_LOCAL) f[v] |= NU_TAINT, changed = 1;
            }
            if (op == 254 && len >= 2 && cur_fn && a[0] < bound && (f[a[0]] & NU_TAINT) && !(f[cur_fn] & NU_RET))
                f[cur_fn] |= NU_RET, changed = 1;
            if (len >= 3 && a[0] < bound && (f[a[0]] & NU_TYPE) && a[1] < bound && !(f[a[1]] & NU_TAINT)) {
                uint32_t res = a[1], from = 2, to = len - 1;   // operand words a[from..to)
                int t = 0;
                switch (op) {
                case 81: to = 3; break;                 // OpCompositeExtract: the composite, then literals
                case 82: case 79: to = 4; break;        // OpCompositeInsert, OpVectorShuffle: two ids, then literals
                case 12: from = 4; break;               // OpExtInst: set and instruction number are not values
                case OP_VARIABLE: to = from; break;     // a function variable: its contents taint it, by stores
                case 61: to = 3; break;                 // OpLoad: the pointer (then memory operands)
                case 57:                                // OpFunctionCall: the callee's result, or any argument
                    t = a[2] < bound && (f[a[2]] & NU_RET);
                    from = 3;
                    break;
                case 333: t = 1; break;                 // OpGroupNonUniformElect
                default:
                    if ((op >= 227 && op <= 242) || op == 6035) t = 1;   // atomics: each lane gets its own old value
                }
                for (uint32_t k = from; !t && k < to; k++) t = a[k] < bound && (f[a[k]] & NU_TAINT);
                if (t) {
                    f[res] |= NU_TAINT, changed = 1;
                    if (op == 57 && a[2] < bound)   // the callee's parameters, all of them (conservative)
                        for (uint32_t k = 0; k < pcount[a[2]]; k++)
                            if (!(f[params[pfirst[a[2]] + k]] & NU_TAINT)) f[params[pfirst[a[2]] + k]] |= NU_TAINT;
                }
            }
            i += len;
        }
    }
    // The accesses, then what is made from them.
    uint32_t nchain = 0, nmark = 0;
    for (uint64_t i = first_fn; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        const uint32_t *a = w + i + 1;
        if ((op == 65 || op == 66) && len >= 5 && a[1] < bound && a[2] < bound && (f[a[2]] & NU_ARRVAR) && a[3] < bound &&
            !(f[a[3]] & NU_CONST) && (mode == 2 || (f[a[3]] & NU_TAINT)))
            f[a[1]] |= NU_MARK, nchain++;
        i += len;
    }
    if (!nchain) goto out;
    for (int pass = 0; pass < 2; pass++)
        for (uint64_t i = first_fn; i < n;) {
            uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
            const uint32_t *a = w + i + 1;
            if (len >= 4 && a[1] < bound && !(f[a[1]] & NU_MARK) &&
                (((op == 61 || op == 100 || op == 83) && a[2] < bound && (f[a[2]] & NU_MARK)) ||   // load, OpImage, copy
                 (op == 86 && len >= 5 && ((a[2] < bound && (f[a[2]] & NU_MARK)) || (a[3] < bound && (f[a[3]] & NU_MARK))))))
                f[a[1]] |= NU_MARK;
            i += len;
        }
    for (uint32_t id = 1; id < bound; id++) nmark += (f[id] & NU_MARK) != 0;
    int need_ext = w[1] < 0x10500 && !have_ext;
    uint64_t extra = (uint64_t)(!have_cap_nu + !have_cap_arr) * 2 + (need_ext ? 8 : 0) + (uint64_t)nmark * 3;
    uint32_t *o = malloc((n + extra) * sizeof *o);
    if (!o) goto out;
    uint64_t k = 0;
    for (int j = 0; j < 5; j++) o[k++] = w[j];
    if (!have_cap_nu) o[k++] = 2u << 16 | 17, o[k++] = 5301;   // ShaderNonUniform
    if (!have_cap_arr) o[k++] = 2u << 16 | 17, o[k++] = 5307;  // SampledImageArrayNonUniformIndexing
    int ext_done = !need_ext, dec_done = 0;
    for (uint64_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!ext_done && op != 17) {
            static const char name[28] = "SPV_EXT_descriptor_indexing";
            o[k++] = 8u << 16 | 10;
            memcpy(o + k, name, 28);
            k += 7, ext_done = 1;
        }
        if (!dec_done && !(op == 17 || op == 10 || op == 11 || op == 14 || op == 15 || op == 16 || op == 331 ||
                           (op >= 2 && op <= 8) || op == 330 || op == 317)) {
            for (uint32_t id = 1; id < bound; id++)
                if (f[id] & NU_MARK) o[k++] = 3u << 16 | OP_DECORATE, o[k++] = id, o[k++] = 5300;   // NonUniform
            dec_done = 1;
        }
        for (uint32_t j = 0; j < len; j++) o[k++] = w[i + j];
        i += len;
    }
    if (!dec_done) { free(o); goto out; }
    free(s->spirv);
    s->spirv = o, s->words = k;
    static volatile int32_t n_logged;
    if (plat_inc32(&n_logged) <= 6)
        rt_log("gfx_shader: %u texture array accesses with a %s index decorated NonUniform (%u ids; REKARINA_NONUNIFORM)",
               nchain, mode == 2 ? "computed" : "per-invocation", nmark);
out:
    free(f), free(base), free(pfirst), free(pcount), free(params), free(imgtype), free(arrtype), free(ptrarr);
}

static void reflect(GfxShader *s) {
    static const char *const kinds[GFX_RES_KINDS] = {"cbufs", "sbufs", "textures", "images"};
    const char *js = s->reflection;
    char stage[8];
    json_str(js, js + strlen(js), "stage", stage, sizeof stage);
    s->stage = !strcmp(stage, "cs") ? GFX_STAGE_CS : !strcmp(stage, "ps") ? GFX_STAGE_FS : GFX_STAGE_VS;
#ifdef GAME_NS2
    if (!strcmp(stage, "gs")) s->stage = GFX_STAGE_GS;
    else if (!strcmp(stage, "ts")) s->stage = GFX_STAGE_TS;
    else if (!strcmp(stage, "ms")) s->stage = GFX_STAGE_MS;
    else if (!strcmp(stage, "tsc")) s->stage = GFX_STAGE_TSC;   // the task shader as compute (Program.cs)
    else if (!strcmp(stage, "msc")) {   // tools/xlate Program.cs: the mesh shader as compute, and its vertex layout
        s->stage = GFX_STAGE_MSC;
        const char *e = js + strlen(js), *p = json_key(js, e, "locations");
        s->emu_vec4s = (uint32_t)json_int(js, e, "vec4s", 0);
        s->emu_max = (uint32_t)json_int(js, e, "max_meshlets", 0);
        for (p = p && *p == '[' ? p + 1 : NULL; p && p < e && *p != ']' && s->emu_nloc < 32;) {
            char *q;
            long v = strtol(p, &q, 10);
            if (q == p) { p++; continue; }
            s->emu_loc[s->emu_nloc++] = (uint8_t)v;
            p = q;
        }
    }
#endif
    s->mvprev_banks = (uint32_t)json_int(js, js + strlen(js), "mvprev_banks", 0);
    for (int k = 0; k < GFX_RES_KINDS; k++) {
        int count = json_objects(js, kinds[k], NULL, s, k);
        s->res[k] = calloc(count ? count : 1, sizeof *s->res[k]);
        json_objects(js, kinds[k], add_res, s, k);
    }
    reflect_spirv(s);
    reflect_palette(s);
    mark_fadd_ssbos(s);
    patch_txd_transpose(s);
    if (s->npool && !patch_pool_arrays(s)) {   // the arrays keep their bound-slot binding in set 0
        static volatile int32_t n_logged;
        if (plat_inc32(&n_logged) <= 3) rt_log("gfx_shader: texture pool arrays not moved to the pool set (module not as expected)");
        free(s->res[GFX_RES_TEX]);
        s->nres[GFX_RES_TEX] = s->npool = 0;
        no_pool = 1;
        int count = json_objects(js, "textures", NULL, s, GFX_RES_TEX);
        s->res[GFX_RES_TEX] = calloc(count ? count : 1, sizeof *s->res[GFX_RES_TEX]);
        json_objects(js, "textures", add_res, s, GFX_RES_TEX);
        no_pool = 0;
    }
    patch_nonuniform(s);
    {   // DEBUG-NODISCARD: REKARINA_NO_DISCARD=<hash>[,...]: those fragment shaders (SPIR-V FNV-1a, as the dump names them) never discard
        const char *e = cfg_env("REKARINA_NO_DISCARD");
        uint64_t h = 0xcbf29ce484222325ull;
        const uint8_t *w8 = (const uint8_t *)s->spirv;
        for (uint64_t i = 0; e && s->stage == GFX_STAGE_FS && i < s->words * 4; i++) h = (h ^ w8[i]) * 0x100000001b3ull;
        int hit = 0;
        while (e && *e && s->stage == GFX_STAGE_FS) {
            char *q;
            hit |= strtoull(e, &q, 16) == h;
            e = q;
            while (*e == ',') e++;
        }
        uint32_t *w = (uint32_t *)s->spirv;
        for (uint64_t i = 5; hit && i < s->words;) {
            uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
            if (!n) break;
            if (op == 252 || op == 4416) w[i] = (1u << 16) | 253;   // OpKill / OpTerminateInvocation -> OpReturn
            else if (op == 5380) w[i] = (1u << 16) | 0;             // OpDemoteToHelperInvocation -> OpNop
            i += n;
        }
        if (hit) rt_log("NO_DISCARD: fragment shader %016llx never discards", (unsigned long long)h);
        // DEBUG-FORCE: REKARINA_SPV_NOT=<hash>:<operand id>:<new id>[,...]: in that module, every OpLogicalNot of the operand
        // takes the new id instead (a culling kernel's "culled" flag replaced by a constant false appends everything)
        // The ground fog sheet (fs 4838c412d5329961) lights a pixel from froxels when a point raised from its hit lands in front of the
        // scene and per pixel otherwise. The two disagree here, so that test drew the seam at the screen edges and around characters
        // (brown layer, white silhouettes, a ghost of the hero on rocks). Default: always per pixel. REKARINA_SPV_NOT= (empty) restores the test.
        // The hash and ids belong to this translator build; another build leaves the shader unpatched and logs nothing.
        const char *fe = cfg_env("REKARINA_SPV_NOT");
        if (!fe) fe = "4838c412d5329961:4442:3446";
        if (fe && *fe) {
            uint64_t h3 = 0xcbf29ce484222325ull;
            for (uint64_t i = 0; i < s->words * 4; i++) h3 = (h3 ^ w8[i]) * 0x100000001b3ull;
            while (*fe) {
                char *q;
                uint64_t want = strtoull(fe, &q, 16);
                uint32_t from = *q == ':' ? (uint32_t)strtoul(q + 1, &q, 0) : 0, to = *q == ':' ? (uint32_t)strtoul(q + 1, &q, 0) : 0;
                if (want == h3 && from && to) {
                    int k = 0;
                    for (uint64_t i = 5; i < s->words;) {
                        uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
                        if (!n) break;
                        if (op == 168 && n == 4 && w[i + 3] == from) w[i + 3] = to, k++;   // OpLogicalNot
                        i += n;
                    }
                    rt_log("SPV_NOT: %016llx: %d OpLogicalNot of %u now take %u", (unsigned long long)h3, k, from, to);
                }
                fe = q;
                while (*fe == ',') fe++;
                if (q == fe && *fe && !(*fe >= '0' && *fe <= '9') && !(*fe >= 'a' && *fe <= 'f')) break;
            }
        }
        // DEBUG-UVDEBUG: REKARINA_FS_UVDEBUG=1: the leaf alpha test c04246304e60b175 never discards and writes (u, v, alpha)
        // instead of 1 (ids of that module, as spvdis.py lists them)
        if (s->stage == GFX_STAGE_FS && cfg_env("REKARINA_FS_UVDEBUG") && (e = "c04246304e60b175", 1)) {
            uint64_t h2 = 0xcbf29ce484222325ull;
            for (uint64_t i = 0; i < s->words * 4; i++) h2 = (h2 ^ w8[i]) * 0x100000001b3ull;
            const char *mode = cfg_env("REKARINA_FS_UVDEBUG");
            if (h2 == strtoull(e, NULL, 16) && (*mode == 'u' || *mode == 'v')) {   // =u / =v: the alpha test compares u / v
                for (uint64_t i = 5; i < s->words;) {                              // with the threshold instead of alpha
                    uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
                    if (!n) break;
                    if (op == 190 && w[i + 2] == 109 && w[i + 3] == 107) w[i + 3] = *mode == 'u' ? 73 : 81;   // OpFOrdGreaterThanEqual
                    i += n;
                }
                rt_log("FS_UVDEBUG: %016llx alpha test on %c", (unsigned long long)h2, *mode);
            } else if (h2 == strtoull(e, NULL, 16)) {
                int done = 0;
                for (uint64_t i = 5; i < s->words;) {
                    uint32_t n = w[i] >> 16, op = w[i] & 0xffff;
                    if (!n) break;
                    if (op == 247 && w[i + 1] == 145 && i + 7 <= s->words && (w[i + 3] & 0xffff) == 250 && w[i + 4] == 129) {
                        for (int k = 0; k < 5; k++) w[i + k] = (1u << 16) | 0;   // OpNop
                        w[i + 5] = (2u << 16) | 249, w[i + 6] = 130;            // OpBranch 130
                        done |= 1;
                        i += 7;
                        continue;
                    }
                    if (op == 62 && w[i + 1] == 132 && w[i + 2] == 134) w[i + 2] = 73, done |= 2;
                    if (op == 62 && w[i + 1] == 135 && w[i + 2] == 137) w[i + 2] = 81, done |= 4;
                    if (op == 62 && w[i + 1] == 139 && w[i + 2] == 141) w[i + 2] = 90, done |= 8;
                    i += n;
                }
                s->dbg = 1;
                rt_log("FS_UVDEBUG: %016llx patched (%#x of 0xf)", (unsigned long long)h2, done);
            }
        }
    }
    {   // DEBUG-SPVFILE: REKARINA_SPV_FILE=<hash>=<path>[;...]: the module with that hash (as the frame dump names it) is
        // replaced by the file's words, for probing what a shader computes (a copy of sh_<hash>.spv edited elsewhere)
        const char *e = cfg_env("REKARINA_SPV_FILE");
        if (e && *e) {
            uint64_t h = 0xcbf29ce484222325ull;
            const uint8_t *w8 = (const uint8_t *)s->spirv;
            for (uint64_t i = 0; i < s->words * 4; i++) h = (h ^ w8[i]) * 0x100000001b3ull;
            while (*e) {
                char *q;
                uint64_t want = strtoull(e, &q, 16);
                const char *path = *q == '=' ? q + 1 : q;
                size_t len = strcspn(path, ";");
                if (want == h && len && len < 600) {
                    char p[600];
                    memcpy(p, path, len), p[len] = 0;
                    FILE *f = fopen(p, "rb");
                    long sz = 0;
                    if (f && !fseek(f, 0, SEEK_END) && (sz = ftell(f)) > 20 && !(sz & 3) && !fseek(f, 0, SEEK_SET)) {
                        uint32_t *o = malloc((size_t)sz);
                        if (o && fread(o, 1, (size_t)sz, f) == (size_t)sz) {
                            free(s->spirv);
                            s->spirv = o, s->words = (uint64_t)sz / 4, s->dbg = 1;
                            rt_log("SPV_FILE: %016llx replaced by %s (%ld words)", (unsigned long long)h, p, sz / 4);
                        } else free(o);
                    } else rt_log("SPV_FILE: %s cannot be read", p);
                    if (f) fclose(f);
                }
                e = path + len;
                if (*e == ';') e++;
            }
        }
    }
}

uint32_t gfx_shader_pool_mask(const GfxShader *s) { return s ? s->pool_mask : 0; }
int gfx_shader_debug_mark(const GfxShader *s) { return s ? s->dbg : 0; }   // DEBUG-UVDEBUG

// ---- use masks (see the cache key at the top)

// The slots of the int array `"name": [...]` below 32 as a mask.
static uint32_t json_slot_mask(const char *js, const char *name) {
    const char *end = js + strlen(js), *p = json_key(js, end, name);
    uint32_t m = 0;
    if (!p || *p != '[') return 0;
    for (p++; p < end && *p != ']';) {
        char *q;
        long v = strtol(p, &q, 10);
        if (q == p) {
            p++;
            continue;
        }
        if (v >= 0 && v < 32) m |= 1u << v;
        p = q;
    }
    return m;
}

static UseMask use_of(const GfxShader *s) {
    UseMask m = {json_slot_mask(s->reflection, "tex_queried") | json_slot_mask(s->reflection, "img_queried"), 0, 0};
    if (s->stage == GFX_STAGE_VS) m.attrs = s->in_mask;
    else if (s->stage == GFX_STAGE_FS) m.fouts = s->out_mask & 0xff;
    return m;
}

// <sha256("TKSU", translator identity, code hash)>.use: masks are per translator.
static void use_name(const uint8_t h[32], uint8_t name[32]) {
    uint8_t buf[4 + 32 + 32];
    memcpy(buf, "TKSU", 4), memcpy(buf + 4, xlate_id, 32), memcpy(buf + 36, h, 32);
    rt_sha256(buf, sizeof buf, name);
}

static void use_path_of(const uint8_t name[32], char *p) {
    char hex[65];
    key_hex(name, hex);
    snprintf(p, PATH_CAP + 80, "%s/%s.use", cache_dir, hex);
}

// Every use mask of the cache in one table by file name, filled by a background thread when the paths are set
// (use_preload): a lookup on the render thread then opens no file. Opening one took 0.1 ms alone but 11-40 ms in play,
// behind the translator's and the pipeline cache's writes and flushes on the same volume, and each new shader of a draw
// opened one. Sources in order: the pack's masks, the mask log (below), the loose .use files neither of them named;
// use_store adds the masks found later. The table only grows: writers hold use_files_lock, readers walk the chains
// without a lock, as an entry is complete before the release store that links it.
// use_files_ready: 0 while the pack and the log are read (a name not in the table may be on the disk), 1 once they are
// in (a name not in it may still have a loose file), 2 when the loose files are in too (a name not in it has no file).
//
// The mask log, <cache>/use-masks.bin: 48-byte records (UseRec) that every run appends for the masks it finds, and the
// preload for each loose file it had to open, so a later start lists the loose files' names and opens none of them. A
// development folder held 90,703 packed masks and 86,159 loose .use files: read one by one they took a background
// thread 11-13 s with a warm file cache and 83 s with a cold one, a sixth of a four-processor machine while the first
// scene played (2026-11-19). The loose files are still written and kept: an older rekarina.exe on the same folder and
// tools/pack_shader_cache.py read them. A record that fails its check (a write cut short by a crash) is skipped byte
// by byte up to the next one that passes.
#define USE_FILE_BUCKETS 65536
typedef struct UseFile { struct UseFile *next; uint8_t name[32]; UseMask m; } UseFile;
static UseFile *use_files[USE_FILE_BUCKETS];
static volatile int32_t use_files_ready;
static PlatMutex use_files_lock = PLAT_MUTEX_INIT;        // adding to use_files; the log handle

typedef struct { uint8_t name[32]; uint32_t slots, attrs, fouts, check; } UseRec;
_Static_assert(sizeof(UseRec) == 48, "use-masks.bin record layout");
static PlatAppend *use_log;
static int use_log_failed;

static UseFile **use_file_slot(const uint8_t name[32]) {
    UseFile **e = &use_files[(name[0] | name[1] << 8) % USE_FILE_BUCKETS], *x;
    while ((x = __atomic_load_n(e, __ATOMIC_ACQUIRE)) && memcmp(x->name, name, 32)) e = &x->next;
    return e;
}

static UseFile *use_file_find(const uint8_t name[32]) { return __atomic_load_n(use_file_slot(name), __ATOMIC_ACQUIRE); }

// name -> m into the table unless the name is there; 1 when added. Caller holds use_files_lock.
static int use_file_add(const uint8_t name[32], const UseMask *m) {
    UseFile **slot = use_file_slot(name), *x;
    if (*slot || !(x = malloc(sizeof *x))) return 0;
    memcpy(x->name, name, 32), x->m = *m, x->next = NULL;
    __atomic_store_n(slot, x, __ATOMIC_RELEASE);
    return 1;
}

static uint32_t use_rec_check(const UseRec *r) {           // FNV-1a over the record's first 44 bytes, seeded with the magic
    const uint8_t *p = (const uint8_t *)r;
    uint32_t h = 2166136261u ^ USE_MAGIC;
    for (size_t i = 0; i < offsetof(UseRec, check); i++) h = (h ^ p[i]) * 16777619u;
    return h;
}

static void use_rec_fill(UseRec *r, const uint8_t name[32], const UseMask *m) {
    memcpy(r->name, name, 32), r->slots = m->slots, r->attrs = m->attrs, r->fouts = m->fouts;
    r->check = use_rec_check(r);
}

// n records to the end of the log, opened at the first call. Caller holds use_files_lock.
static void use_log_append(const UseRec *r, size_t n) {
    if (!n || use_log_failed || !cache_dir[0]) return;
    if (!use_log) {
        char p[PATH_CAP + 32];
        snprintf(p, sizeof p, "%s/use-masks.bin", cache_dir);
        if (!(use_log = plat_append_shared(p))) {
            use_log_failed = 1;
            rt_log("gfx_shader: cannot open %s for writing: masks go to the loose .use files only", p);
            return;
        }
    }
    if (!plat_append_write(use_log, r, n * sizeof *r)) use_log_failed = 1;
}

static int hex_byte(const char *c) {
    int v = 0;
    for (int i = 0; i < 2; i++) {
        int d = isdigit((unsigned char)c[i]) ? c[i] - '0' : c[i] >= 'a' && c[i] <= 'f' ? c[i] - 'a' + 10 : -1;
        if (d < 0) return -1;
        v = v << 4 | d;
    }
    return v;
}

#define USE_LOG_BATCH 1024                                 // records the preload appends to the log at a time
typedef struct { int n, bad, listed; size_t nrec; UseRec rec[USE_LOG_BATCH]; } UsePreload;

static int use_add_locked(const uint8_t name[32], const UseMask *m) {
    plat_mutex_lock(&use_files_lock);
    int added = use_file_add(name, m);
    plat_mutex_unlock(&use_files_lock);
    return added;
}

// The pack's masks. The tool writes the data in index order, so they lie in one stretch of the file: one read for
// all of them (20 bytes each) instead of one read each. A pack of 90,703 masks: 5 ms, where the reads one by one
// and the loose files after them took 11-13 s with a warm file cache (2026-11-19).
static void use_preload_pack(int *n, int *bad) {
    uint64_t lo = UINT64_MAX, hi = 0;
    for (uint32_t i = 0; i < pack_n; i++)
        if (pack_ix[i].kind == PACK_USE) {   // never compressed: 20 bytes
            if (pack_ix[i].offset < lo) lo = pack_ix[i].offset;
            if (pack_ix[i].offset + pack_ix[i].size > hi) hi = pack_ix[i].offset + pack_ix[i].size;
        }
    if (lo >= hi) return;
    uint8_t *span = hi - lo <= (64u << 20) ? malloc((size_t)(hi - lo)) : NULL;
    if (span && !pack_read_at(lo, span, (uint32_t)(hi - lo))) free(span), span = NULL;
    for (uint32_t i = 0; i < pack_n; i++) {
        uint32_t u[5];
        if (pack_ix[i].kind != PACK_USE) continue;
        int got = pack_ix[i].size == sizeof u &&
                  (span ? (memcpy(u, span + (pack_ix[i].offset - lo), sizeof u), 1) : pack_read_at(pack_ix[i].offset, u, sizeof u));
        if (!got || u[0] != USE_MAGIC || u[1] != USE_VERSION) {
            ++*bad;
            continue;
        }
        UseMask m = {u[2], u[3], u[4]};
        *n += use_add_locked(pack_ix[i].name, &m);
    }
    free(span);
}

// The mask log's records; *skipped: bytes that held no valid record.
static void use_preload_log(int *n, uint64_t *skipped) {
    char p[PATH_CAP + 32];
    snprintf(p, sizeof p, "%s/use-masks.bin", cache_dir);
    uint64_t len = 0;
    uint8_t *f = read_file(p, &len);
    if (!f) return;
    for (uint64_t o = 0; o + sizeof(UseRec) <= len;) {
        UseRec r;
        memcpy(&r, f + o, sizeof r);
        if (r.check != use_rec_check(&r)) {
            ++*skipped, o++;
            continue;
        }
        UseMask m = {r.slots, r.attrs, r.fouts};
        *n += use_add_locked(r.name, &m);
        o += sizeof r;
    }
    free(f);
}

static void use_preload_flush(UsePreload *st) {
    plat_mutex_lock(&use_files_lock);
    use_log_append(st->rec, st->nrec);
    plat_mutex_unlock(&use_files_lock);
    st->nrec = 0;
}

// One <64 hex digits>.use file of the cache directory into the table, and into the log: read only when the pack and
// the log did not name it.
static int use_preload_file(const char *name, void *ctx) {
    UsePreload *st = ctx;
    char p[PATH_CAP + 80];
    uint8_t nm[32];
    int ok = strlen(name) == 68;
    for (int i = 0; ok && i < 32; i++) {
        int v = hex_byte(name + 2 * i);
        ok = v >= 0, nm[i] = (uint8_t)v;
    }
    if (!ok) {
        st->bad++;
        return 1;
    }
    st->listed++;
    if (use_file_find(nm)) return 1;
    uint64_t len = 0;
    snprintf(p, sizeof p, "%s/%s", cache_dir, name);
    uint32_t *u = (uint32_t *)read_file(p, &len);
    if (u && len == 20 && u[0] == USE_MAGIC && u[1] == USE_VERSION) {
        UseMask m = {u[2], u[3], u[4]};
        if (use_add_locked(nm, &m)) {
            st->n++;
            use_rec_fill(&st->rec[st->nrec++], nm, &m);
            if (st->nrec == USE_LOG_BATCH) use_preload_flush(st);
        }
    } else {
        st->bad++;
    }
    free(u);
    return 1;
}

static void use_preload(void *arg) {
    (void)arg;
    plat_thread_priority(PLAT_PRIO_BELOW_NORMAL);
    uint64_t t0 = plat_now_ns(), skipped = 0;
    int npack = 0, nlog = 0, bad = 0;
    use_preload_pack(&npack, &bad);
    use_preload_log(&nlog, &skipped);
    __atomic_store_n(&use_files_ready, 1, __ATOMIC_RELEASE);
    uint64_t t1 = plat_now_ns();
    UsePreload *st = calloc(1, sizeof *st);
    if (st) {
        plat_file_list(cache_dir, ".use", use_preload_file, st);
        use_preload_flush(st);
        bad += st->bad;
    }
    __atomic_store_n(&use_files_ready, 2, __ATOMIC_RELEASE);
    rt_log("gfx_shader: use masks on a background thread: %d from the pack and %d from the log in %.0f ms (%llu bytes "
           "of the log skipped), then %d loose files listed, %d of them read, in %.0f ms (%d unreadable)",
           npack, nlog, (t1 - t0) / 1e6, (unsigned long long)skipped, st ? st->listed : 0, st ? st->n : 0,
           (plat_now_ns() - t1) / 1e6, bad);
    free(st);
}

static void use_preload_start(void) {
    if (!cache_dir[0]) return;
    plat_thread_spawn(use_preload, NULL, "use preload", 0);
}

static UseEnt **use_slot(const uint8_t h[32]) {         // caller holds cache_lock
    UseEnt **e = &use_buckets[(h[0] | h[1] << 8) % BUCKETS];
    while (*e && memcmp((*e)->code_hash, h, 32)) e = &(*e)->next;
    return e;
}

// Sets or adds the entry for h. Caller holds cache_lock exclusively.
static void use_put(const uint8_t h[32], int known, const UseMask *m) {
    UseEnt **e = use_slot(h);
    if (!*e && !(*e = calloc(1, sizeof **e))) return;
    memcpy((*e)->code_hash, h, 32);
    if (known) (*e)->known = 1, (*e)->m = *m;
}

// 1 with *m when the use mask of h is known (memory, then <hash>.use; a disk miss is remembered).
static int use_find(const uint8_t h[32], UseMask *m) {
    plat_mutex_lock_shared(&cache_lock);
    UseEnt *e = *use_slot(h);
    int seen = e != NULL, known = e && e->known;
    if (known) *m = e->m;
    plat_mutex_unlock_shared(&cache_lock);
    if (seen) return known;
    uint8_t name[32];
    use_name(h, name);
    int ready = __atomic_load_n(&use_files_ready, __ATOMIC_ACQUIRE);
    UseFile *u = use_file_find(name);
    if (u) {
        known = 1, *m = u->m;
    } else if (ready == 2) {   // complete: a name not in it has no file
        known = 0;
    } else {
        char p[PATH_CAP + 80];
        use_path_of(name, p);
        uint64_t n = 0;
        uint32_t *f = ready ? NULL : (uint32_t *)pack_read(PACK_USE, name, &n);   // from 1 on the pack is in the table
        if (!f) f = (uint32_t *)read_file(p, &n);
        known = f && n == 20 && f[0] == USE_MAGIC && f[1] == USE_VERSION;
        if (known) m->slots = f[2], m->attrs = f[3], m->fouts = f[4];
        free(f);
    }
    plat_mutex_lock(&cache_lock);
    use_put(h, known, m);
    plat_mutex_unlock(&cache_lock);
    return known;
}

// Whether the cache's files are flushed to the disk as they are written (gfx_shader_cache_flush). Every reader checks
// what it reads (reflection_valid, spirv_complete, the use and input files' magic), so a file a power cut left
// zero-filled is translated again; the flushes, four FlushFileBuffers and four write-through renames a translation,
// each a wait for the disk, held a player's setup to 9 shaders a second on a slow disk with the processor idle.
static int cache_flush = 1;
void gfx_shader_cache_flush(int on) { cache_flush = on; }

static int cache_write(const char *path, const void *data, size_t n) {
    return cache_flush ? plat_file_write_atomic(path, data, n) : plat_file_write_lazy(path, data, n);
}

static void use_store(const uint8_t h[32], const UseMask *m) {
    char p[PATH_CAP + 80];
    uint8_t name[32];
    use_name(h, name);
    use_path_of(name, p);
    uint32_t f[5] = {USE_MAGIC, USE_VERSION, m->slots, m->attrs, m->fouts};
    cache_write(p, f, sizeof f);
    UseRec r;
    use_rec_fill(&r, name, m);
    plat_mutex_lock(&use_files_lock);
    if (use_file_add(name, m)) use_log_append(&r, 1);
    plat_mutex_unlock(&use_files_lock);
    plat_mutex_lock(&cache_lock);
    use_put(h, 1, m);
    plat_mutex_unlock(&cache_lock);
}

static void put_u32(uint8_t **p, uint32_t v) { memcpy(*p, &v, 4), *p += 4; }

// Fixed-layout answers for hashing; returns the byte count (at most 256).
static size_t pack_answers(uint8_t *o, const GfxShaderAnswers *a) {
    uint8_t *p = o;
    GfxShaderAnswers zero = {0};
    if (!a) a = &zero;
    memcpy(p, a->tex_type, 32), p += 32;
    memcpy(p, a->img_type, 32), p += 32;
    memcpy(p, a->img_format, 32), p += 32;
    memcpy(p, a->attr_type, 32), p += 32;
    memcpy(p, a->frag_out_type, 8), p += 8;
    for (int i = 0; i < 3; i++) *p++ = (uint8_t)a->local[i], *p++ = (uint8_t)(a->local[i] >> 8);
    put_u32(&p, a->shared_bytes);
    *p++ = (uint8_t)(a->depth_minus_w | (a->mvprev ? 0x80 : 0));
    return (size_t)(p - o);
}

// A mesh program that outputs points or a point size ends with a tag (tools/prepare_shaders.py code_hash does the
// same). Until translator patch 0040 those were translated as triangles of position and attributes; the patch changes
// no other translation and keeps the translator identity (tools/build_xlate.py SAME_AS), so every player's cache stays
// valid, and the tag gives exactly these programs new ids: what the cache holds for them from before is never found.
// 2: a triangle naming a vertex past the meshlet's last is dropped, as the mesh stage drops it; 3: the buffer's layout
// with room for the tasks (vertices from 8 MiB), and the task kernels; 4: the buffer's address from push constants
// (translator patch 0047) instead of two constants the runtime patched in
#define MESHC_REVISION 4u

// The subgroup lanes a stage's programs are translated for (the request's "subgroup", build_request): the vertex,
// compute or fragment stage's answer, which task and mesh stages and their kernels share unless they are pinned.
static uint32_t xlate_lanes(int stage) {
    uint32_t lanes = host_sg[stage == GFX_STAGE_VS ? 0 : stage == GFX_STAGE_CS ? 2 : 1];
#ifdef GAME_NS2
    if (gfx_shader_mesh_subgroup &&
        (stage == GFX_STAGE_TS || stage == GFX_STAGE_MS || stage == GFX_STAGE_MSC || stage == GFX_STAGE_TSC))
        lanes = gfx_shader_mesh_subgroup;
#endif
    return lanes;
}

static int build_code_hash(int stage, const uint8_t *control, const uint8_t *code, uint64_t code_size,
                           const uint8_t *c1, uint32_t c1_len, uint8_t hash[32]) {
    static const char tag[8] = "TKMP\1", tag_whole[8] = "TKMW\1", tag_voteu[8] = "TKVU\1", tag_wide[8] = "TKWL\1",
                      tag_large[8] = "TKML\1", tag_prim[8] = "TKPP\1";
    int tagged = stage == GFX_STAGE_MS && nvn2_ctl_mesh_points(control, code, code_size);
    // A program whose VOTEU writes a uniform predicate is another translation since patch 0042 (the eye shaders'
    // probe light): a tag keeps the earlier translator identity for every other program
    int voteu = nvn2_ctl_voteu_pred(control, code, code_size);
    // A compute program with REDUX, MATCH or a per-lane HMMA, LDSM or MOVM is another translation since patch 0044 where
    // compute is translated for subgroups wider than 32 lanes (AMD GCN): a tag there only. Every other device keeps
    // every key; this one's entries for those programs, made with the first warp's lanes, are not found any more.
    int wide = stage == GFX_STAGE_CS && xlate_lanes(stage) > 32 && nvn2_ctl_warp_lanes(control, code, code_size);
    // A mesh program of 60,000 to 70,000 bytes of instructions (the foliage) is another translation since patch 0046,
    // which calls its raw global-memory helpers instead of inlining them: a tag keeps the earlier identity for every other
    int large = (stage == GFX_STAGE_MS || stage == GFX_STAGE_MSC) && nvn2_ctl_mesh_large(control);
    // Where mesh shaders are asked for with whole index vectors (gfx_shader_mesh_whole, patch 0041: every device but
    // NVIDIA's) they are other translations of the same programs: a second tag, there only, keeps the two apart
    // without another translator identity for everything else the device has cached.
    int whole = 0;
#ifdef GAME_NS2
    whole = gfx_shader_mesh_whole && stage == GFX_STAGE_MS && nvn2_ctl(control);
#endif
    // A mesh program with per-primitive outputs (a per-primitive SPH: control +0x68 bit 7), and its emulation kernel, is
    // another translation since patch 0048 (the outputs read after the vertex blocks): a tag gives exactly those new
    // keys and keeps the earlier translator identity, and every player's cache, for everything else
    int prim = 0;
#ifdef GAME_NS2
    prim = (stage == GFX_STAGE_MS || stage == GFX_STAGE_MSC) && nvn2_ctl(control) && (control[0x68] & 0x80);
#endif
    size_t total = 4 + CONTROL_SIZE + 8 + (size_t)code_size + 4 + c1_len + (tagged ? sizeof tag : 0) + (whole ? sizeof tag_whole : 0) +
                   (voteu ? sizeof tag_voteu : 0) + (wide ? sizeof tag_wide : 0) + (large ? sizeof tag_large : 0) +
                   (prim ? sizeof tag_prim : 0);
    uint8_t *buf = malloc(total), *p = buf;
    if (!buf) return 0;
    // a mesh emulation kernel carries its layout's revision: a change of the kernel (tools/xlate MeshStageWrapper
    // GenerateAsCompute) raises MESHC_REVISION, and kernels cached from an earlier one are not found
    put_u32(&p, (uint32_t)stage | (stage == GFX_STAGE_MSC || stage == GFX_STAGE_TSC ? MESHC_REVISION << 16 : 0));
    memcpy(p, control, CONTROL_SIZE), p += CONTROL_SIZE;
    memcpy(p, &code_size, 8), p += 8;
    memcpy(p, code, (size_t)code_size), p += code_size;
    put_u32(&p, c1_len);
    if (c1_len) memcpy(p, c1, c1_len), p += c1_len;
    if (tagged) memcpy(p, tag, sizeof tag), p += sizeof tag;
    if (whole) memcpy(p, tag_whole, sizeof tag_whole), p += sizeof tag_whole;
    if (voteu) memcpy(p, tag_voteu, sizeof tag_voteu), p += sizeof tag_voteu;
    if (wide) memcpy(p, tag_wide, sizeof tag_wide), p += sizeof tag_wide;
    if (large) memcpy(p, tag_large, sizeof tag_large), p += sizeof tag_large;
    if (prim) memcpy(p, tag_prim, sizeof tag_prim);
    rt_sha256(buf, total, hash);
    free(buf);
    return 1;
}

// The answers with everything outside the mask zeroed; the translator ignores depth_minus_w (Program.cs).
static void mask_answers(const GfxShaderAnswers *a, const UseMask *m, GfxShaderAnswers *o) {
    memset(o, 0, sizeof *o);
    if (!a) return;
    for (int i = 0; i < 32; i++) {
        if (m->slots >> i & 1) o->tex_type[i] = a->tex_type[i], o->img_type[i] = a->img_type[i], o->img_format[i] = a->img_format[i];
        if (m->attrs >> i & 1) o->attr_type[i] = a->attr_type[i];
    }
    for (int i = 0; i < 8; i++)
        if (m->fouts >> i & 1) o->frag_out_type[i] = a->frag_out_type[i];
    memcpy(o->local, a->local, sizeof o->local);
    o->shared_bytes = a->shared_bytes;
    o->mvprev = a->mvprev;   // another translation, not an answer the use mask covers
}

// The cache key of an input id under the current translator.
static void key_of_input(const uint8_t iid[32], uint8_t key[32]) {
    uint8_t buf[8 + 32 + 32], *p = buf;
    memcpy(p, "TKSK", 4), p += 4;
    put_u32(&p, KEY_VERSION);
    memcpy(p, xlate_id, 32), p += 32;
    memcpy(p, iid, 32), p += 32;
    rt_sha256(buf, (size_t)(p - buf), key);
}

// An NVN2 vertex program's last-frame variant (answers mvprev) is another translation since translator patch 0045:
// before it the translator ignored mvprev for Ampere code (Program.cs QueryMotionPrevious) and gave the plain
// translation, which the cache holds under the variant's key. The patch changes no other translation and keeps the
// translator identity (tools/build_xlate.py SAME_AS), so every player's cache stays valid; this tag, in the keys of
// those variants only (build_key), gives them new keys, and what the cache holds for them from before is never found.
// Returns the bytes written at p.
static size_t put_mvprev_tag(uint8_t *p, const GfxShaderAnswers *a) {
#ifdef GAME_NS2
    static const char tag_mvprev[8] = "TKMV\1";
    if (a && a->mvprev) return memcpy(p, tag_mvprev, sizeof tag_mvprev), sizeof tag_mvprev;
#else
    (void)p, (void)a;
#endif
    return 0;
}

// The input id and cache key (m given), or the provisional key (m NULL: full answers; iid untouched).
static void build_key(const uint8_t code_hash[32], const GfxShaderAnswers *a, const UseMask *m, uint8_t key[32],
                      uint8_t iid[32]) {
    GfxShaderAnswers masked;
    uint8_t buf[8 + 32 + 32 + 256], *p = buf;
    if (m) {
        memcpy(p, "TKSI", 4), p += 4;
        put_u32(&p, IN_VERSION);
        memcpy(p, code_hash, 32), p += 32;
        mask_answers(a, m, &masked);
        p += pack_answers(p, &masked);
        p += put_mvprev_tag(p, &masked);
        rt_sha256(buf, (size_t)(p - buf), iid);
        key_of_input(iid, key);
        return;
    }
    memcpy(p, "TKSP", 4), p += 4;
    put_u32(&p, KEY_VERSION);
    memcpy(p, xlate_id, 32), p += 32;
    memcpy(p, code_hash, 32), p += 32;
    p += pack_answers(p, a);
    p += put_mvprev_tag(p, a);
    rt_sha256(buf, (size_t)(p - buf), key);
}

#ifdef GAME_NS2
// ---- Switch 2 (NVN2) shaders
// An NVN2 control section (nvn_ctl.h) is 0xd0 to 0xd8 bytes; everything here hashes, stores and copies CONTROL_SIZE
// bytes of a control section. Each entry point therefore reads an NVN2 section as its first NVN2_CTL_READ bytes, with
// +0x30 (the section's own length in files, 0 in memory and in agl's BAHS files) as 0 and the rest 0, so the cache key
// of a shader is the same from every source and nothing is read past the section.
static const uint8_t *ctl_normalize(const uint8_t *c, uint8_t *buf) {
    if (!nvn2_ctl(c)) return c;
    memset(buf, 0, CONTROL_SIZE);
    memcpy(buf, c, NVN2_CTL_READ);
    memset(buf + 0x30, 0, 4);
    return buf;
}
#define CTL_NORM(control)                                                                                              \
    uint8_t ctl_norm_[CONTROL_SIZE];                                                                                    \
    control = ctl_normalize(control, ctl_norm_)

// The request fields of an NVN2 shader: Ampere code, which the translator's Ampere frontend reads (job "arch": "sm86";
// MacNativeGames totk_shader_rt.m variant_request, seedds/nxnative dmsh_build.py job_for, MIT). The translator's side is
// another worker's (tools/xlate, documented in docs/oot-xlate.md when it is done); every field the runtime sends for
// NVN2 is written here and nowhere else, so that contract is filled in at this one place:
//   - compute: "nvn2_cs": [uniform buffer entries +0xb1, storage buffer entries +0xb2, texture entries +0xb3, c[0]
//     layout size +0xba (u16), push-constant bytes (+0xa9 words x 4)] (MacNativeGames' five, which tools/xlate takes;
//     nxnative also sends the separate texture and sampler entries +0xb4, +0xb5); "shared" (+0x64 rounded up to 16
//     bytes) and "local" (+0x58) come through the answers (vk_exec_draw.c answers_for);
//   - every stage: "gmem_map": true when the device has buffer device addresses (gfx_shader_gmem): the replay binds the
//     memory pool table and its write log at the reflection's sbufs with cb_slot 255 (vk_exec_draw.c gmem_info);
//   - task and mesh ("stage" "ts" / "ms"): "local": [+0x58 threads, 1, 1] and "mesh": {payload +0x5a, task +0x5e,
//     max_vertices +0x60, max_primitives +0x62, primitive_sph +0x64 when +0x68 bit 7, prim_verts +0x5f (3 triangles,
//     1 points), whole_indices (gfx_shader_mesh_whole)}; the replay puts the draw's first
//     workgroup at c0 + VK_C0_MESH_FIRST (vk_exec.h) for the shader to add to its workgroup number;
//   - every stage: push constants are read as c[24] (NVN2_PUSH_BANK), which the replay binds (vk_exec_draw.c ubo_info).
static char *ns2_request(char *o, const char *stage, const uint8_t *control) {
    if (!nvn2_ctl(control)) return o;
    o += sprintf(o, ",\"arch\":\"sm86\"");
    // global memory at loaded addresses through the pool table (vk_memory.c vk_mem_gmem_table, vk_exec_draw.c gmem_info);
    // the translator uses it only where an access traces to no storage buffer, so every Ampere job may ask
    if (gfx_shader_gmem) o += sprintf(o, ",\"gmem_map\":true");
    if (!strcmp(stage, "cs")) {
        uint16_t c0;
        memcpy(&c0, control + 0xba, 2);
        o += sprintf(o, ",\"nvn2_cs\":[%u,%u,%u,%u,%u]", control[0xb1], control[0xb2], control[0xb3], c0,
                     control[0xa9] * 4u);
    } else if (!strcmp(stage, "ts") || !strcmp(stage, "ms") || !strcmp(stage, "msc") || !strcmp(stage, "tsc")) {
        uint16_t f[6];
        memcpy(f, control + 0x58, sizeof f);   // threads, payload, (pad), task, max vertices, max primitives
        uint32_t psph;
        memcpy(&psph, control + 0x64, 4);
        o += sprintf(o, ",\"local\":[%u,1,1],\"mesh\":{\"payload\":%u,\"task\":%u,\"max_vertices\":%u,"
                        "\"max_primitives\":%u,\"primitive_sph\":%u,\"prim_verts\":%u,\"whole_indices\":%d%s}",
                     nvn2_ctl_mesh_threads(control), f[1], control[0x5e], f[4], f[5], control[0x68] & 0x80 ? psph : 0,
                     control[0x5f] ? control[0x5f] : 3u, gfx_shader_mesh_whole,
                     !strcmp(stage, "msc") || !strcmp(stage, "tsc") ? ",\"as_compute\":1" : "");
    }
    return o;
}
#else
#define CTL_NORM(control) (void)0
#endif

static size_t build_request(char *o0, const char *hex, const char *stage, const char *code_path, const uint8_t *c1,
                            uint32_t c1_len, const GfxShaderAnswers *a, const uint8_t *control) {
    char *o = o0;
    o += sprintf(o, "{\"job\":{\"key\":\"");
    json_esc(&o, hex);
    o += sprintf(o, "\",\"stage\":\"%s\",\"debug\":false,\"code\":\"", !strcmp(stage, "msc") ? "ms" : !strcmp(stage, "tsc") ? "ts" : stage);
    json_esc(&o, code_path);
    o += sprintf(o, "\",\"c1\":\"");
    for (uint32_t i = 0; i < c1_len; i++) o += sprintf(o, "%02x", c1[i]);
    o += sprintf(o, "\"");
    if (a) {
        const char *sep = "";
        o += sprintf(o, ",\"tex\":{");
        for (int i = 0; i < 32; i++)
            if (a->tex_type[i] && a->tex_type[i] < COUNT(tex_names)) o += sprintf(o, "%s\"%d\":\"%s\"", sep, i, tex_names[a->tex_type[i]]), sep = ",";
        sep = "";
        o += sprintf(o, "},\"img\":{");
        for (int i = 0; i < 32; i++)
            if (a->img_type[i] && a->img_type[i] < COUNT(tex_names)) {
                const char *f = a->img_format[i] && a->img_format[i] < COUNT(fmt_names) ? fmt_names[a->img_format[i]] : "R8G8B8A8Unorm";
                o += sprintf(o, "%s\"%d\":{\"type\":\"%s\",\"fmt\":\"%s\"}", sep, i, tex_names[a->img_type[i]], f), sep = ",";
            }
        sep = "";
        o += sprintf(o, "},\"attr\":{");
        for (int i = 0; i < 32; i++)
            if (a->attr_type[i]) o += sprintf(o, "%s\"%d\":\"%s\"", sep, i, attr_name(a->attr_type[i])), sep = ",";
        sep = "";
        o += sprintf(o, "},\"fout\":{");
        for (int i = 0; i < 8; i++)
            if (a->frag_out_type[i]) o += sprintf(o, "%s\"%d\":\"%s\"", sep, i, attr_name(a->frag_out_type[i])), sep = ",";
        o += sprintf(o, "},\"shared\":%u", a->shared_bytes);
    }
    if (!strcmp(stage, "cs")) {
        uint32_t l[3] = {1, 1, 1};
        if (a && (a->local[0] | a->local[1] | a->local[2])) l[0] = a->local[0], l[1] = a->local[1], l[2] = a->local[2];
#ifdef GAME_NS2
        else nvn_ctl_local(control, l);
#else
        else memcpy(l, control + 0x748, 12);
#endif
        o += sprintf(o, ",\"local\":[%u,%u,%u]", l[0], l[1], l[2]);
    }
    /* Control byte +0x718 of a fragment shader: early fragment tests (the NVN driver's early-Z-force, MacNativeGames
       ae8f4f4). Without it a shader that stores to memory is depth-tested late, and the G-buffer draws' EQUAL test
       after the depth prepass fails along contour lines. Already part of the code hash (the whole control section). */
    if (!strcmp(stage, "ps") && !nvn2_ctl(control) && (control[0x718] & 1)) o += sprintf(o, ",\"early_fragment_tests\":true");
#ifdef GAME_NS2
    o = ns2_request(o, stage, control);
#endif
    int si = !strcmp(stage, "vs") ? 0 : !strcmp(stage, "cs") ? 2 : 1;   // gfx_shader_set_host answers (as xlate_lanes)
    if (!host_f64) o += sprintf(o, ",\"softfloat64\":true");
    if (a && a->mvprev && !strcmp(stage, "vs")) o += sprintf(o, ",\"mvprev\":true");
    if (host_sbuf) o += sprintf(o, ",\"sbuf_align\":%u", host_sbuf);
    uint32_t lanes = host_sg[si];
    if (gfx_shader_mesh_subgroup && (!strcmp(stage, "ts") || !strcmp(stage, "ms") || !strcmp(stage, "msc") || !strcmp(stage, "tsc")))
        lanes = gfx_shader_mesh_subgroup;
    if (lanes != 32) o += sprintf(o, ",\"subgroup\":%u", lanes);
    if (host_gather) o += sprintf(o, ",\"gather_bias\":%u", host_gather);
    if (host_zero_locals) o += sprintf(o, ",\"zero_locals\":true");
#ifdef GAME_NS2
    if (gfx_shader_float_atomics) o += sprintf(o, ",\"float_atomics\":true");
    if (gfx_shader_coop_matrix) o += sprintf(o, ",\"coop_matrix\":true");
#endif
    o += sprintf(o, "},\"out\":\"");
    json_esc(&o, cache_dir);
    o += sprintf(o, "\"}");
    return (size_t)(o - o0);
}

// ---- public

// Raw line exchange for the probe: one line to the service, its reply in `reply`. Starts the service if needed.
int gfx_shader_raw_request(const char *line, char *reply, size_t reply_size) {
    int away = kern_core_away();
    Svc *v = svc_get();
    int ok = child_exchange(v, line, reply, reply_size);
    if (!ok) child_stop(v);
    svc_put(v);
    kern_core_back(&away);
    return ok;
}

uint64_t gfx_shader_failures(void) { return (uint64_t)n_failures; }
uint64_t gfx_shader_requests(void) { return (uint64_t)n_requests; }   // lines sent to the service
uint64_t gfx_shader_cache_hits(void) { return (uint64_t)n_hits; }     // gets answered from memory or disk

// Drops the memory cache (the probe uses it to prove the disk cache). Pointers handed out before are invalid.
void gfx_shader_drop_memory(void) {
    plat_mutex_lock(&cache_lock);
    for (int i = 0; i < BUCKETS; i++) {
        for (GfxShader *s = buckets[i], *n; s; s = n) {
            n = s->next;
            shader_free(s);
        }
        buckets[i] = NULL;
        n_keys = 0;
        for (UseEnt *e = use_buckets[i], *n; e; e = n) n = e->next, free(e);
        use_buckets[i] = NULL;
    }
    // use_files stays: it holds what the disk holds, the masks this run wrote included (use_store)
    plat_mutex_unlock(&cache_lock);
}

static const GfxShader *fail(const char *why, const char *hex) {
    plat_inc64(&n_failures);
    rt_log("gfx_shader: translation failed (%s): %s", hex ? hex : "-", why);
    return NULL;
}

// A validated request: the program's code length, its c1 constants and the cache key. 0 (failure counted) when invalid.
typedef struct {
    const char *stage_name;
    uint32_t length, c1_len;
#ifdef GAME_NS2
    uint32_t xskip, xlen;                                  // the code the translator reads: [xskip, xlen) of the area
#endif
    const uint8_t *c1;
    uint8_t code_hash[32];
    uint8_t key[32];                                       // cache key, or the provisional one with final 0
    uint8_t iid[32];                                       // input id with final 1
    int final;
} Req;

static int req_init(Req *r, int stage, const uint8_t *control, const uint8_t *code, uint64_t code_size, const uint8_t *c1,
                    const GfxShaderAnswers *a) {
    r->stage_name = stage == GFX_STAGE_VS ? "vs" : stage == GFX_STAGE_FS ? "ps" : stage == GFX_STAGE_CS ? "cs" : NULL;
#ifdef GAME_NS2
    if (stage == GFX_STAGE_GS) r->stage_name = "gs";
    else if (stage == GFX_STAGE_TS) r->stage_name = "ts";
    else if (stage == GFX_STAGE_MS) r->stage_name = "ms";
    else if (stage == GFX_STAGE_MSC) r->stage_name = "msc";
    else if (stage == GFX_STAGE_TSC) r->stage_name = "tsc";
#endif
    if (!r->stage_name) return fail("stage is not translated", NULL), 0;
#ifdef GAME_NS2
    uint32_t c1_off = nvn_ctl_c1_off(control);
    r->length = nvn_ctl_code_len(control);
    r->xlen = nvn_ctl_xlate_span(control, stage == GFX_STAGE_CS, &r->xskip);
    if (!code || r->length > code_size || c1_off > r->length || r->xskip >= r->xlen || r->xlen > r->length)
        return fail("bad control section", NULL), 0;
#else
    uint32_t c1_off;
    memcpy(&c1_off, control + 0x700, 4), memcpy(&r->length, control + 0x704, 4);
    uint64_t skip = stage == GFX_STAGE_CS ? 0 : CODE_SKIP;
    if (!code || r->length > code_size || c1_off > r->length || skip >= r->length) return fail("bad control section", NULL), 0;
#endif
    r->c1_len = r->length - c1_off;
    r->c1 = c1 ? c1 : code + c1_off;
    if (!build_code_hash(stage, control, code, r->length, r->c1, r->c1_len, r->code_hash))
        return fail("out of memory", NULL), 0;
    paths();
    UseMask m;
    r->final = use_find(r->code_hash, &m);
    build_key(r->code_hash, a, r->final ? &m : NULL, r->key, r->iid);
    return 1;
}

static const GfxShader *translate(const Req *r, int stage, const uint8_t *control, const uint8_t *code,
                                  const GfxShaderAnswers *a);

static void ans_note(const Req *r, int stage, const uint8_t *control, const uint8_t *code, const GfxShaderAnswers *a,
                     const GfxShader *s);
static const GfxShader *get_async_one(Req r, int stage, const uint8_t *control, const uint8_t *code,
                                      const GfxShaderAnswers *a, int *pending);

static const GfxShader *get_one(const Req *r, int stage, const uint8_t *control, const uint8_t *code,
                                const GfxShaderAnswers *a) {
    GfxShader *s = mem_find(r->key);
    if (s) return plat_inc64(&n_hits), s;
    paths();
    if (!cache_dir[0]) return fail("no cache directory", NULL);
    if (r->final && (s = disk_load(r->key, r->iid))) return plat_inc64(&n_hits), mem_insert(s);
    return translate(r, stage, control, code, a);
}

const GfxShader *gfx_shader_get(int stage, const uint8_t *control, const uint8_t *code, uint64_t code_size,
                                const uint8_t *c1, const GfxShaderAnswers *a) {
    CTL_NORM(control);
    Req r;
    if (!req_init(&r, stage, control, code, code_size, c1, a)) return NULL;
    const GfxShader *s = get_one(&r, stage, control, code, a);
    if (s) ans_note(&r, stage, control, code, a, s);
    return s;
}

// ---- stored inputs (<cache>/in/<input id>.in, see the top)

typedef struct {
    uint32_t magic, version, stage, length, c1_len, answers_size;
} InHeader;                                                // then GfxShaderAnswers, control, code[length], c1[c1_len]

typedef struct {
    uint8_t *buf;                                          // the file; the pointers below point into it
    int stage;
    uint32_t length, c1_len;
    const uint8_t *control, *code, *c1;
    GfxShaderAnswers a;
} Inputs;

static void in_path(const uint8_t iid[32], char *p) {
    char hex[65];
    key_hex(iid, hex);
    snprintf(p, PATH_CAP + 80, "%s/in/%s.in", cache_dir, hex);
}

// Stores the inputs of a translation unless they are stored already; the file appears whole (written, then renamed).
static void in_store(const uint8_t iid[32], int stage, const uint8_t *control, const uint8_t *code, const Req *r,
                     const GfxShaderAnswers *a) {
    char p[PATH_CAP + 80], tmp[PATH_CAP + 120];
    in_path(iid, p);
    if (pack_find(PACK_IN, iid) || file_exists(p)) return;
    GfxShaderAnswers zero = {0};
    InHeader h = {IN_MAGIC, IN_VERSION, (uint32_t)stage, r->length, r->c1_len, sizeof(GfxShaderAnswers)};
    size_t n = sizeof h + sizeof zero + CONTROL_SIZE + r->length + r->c1_len;
    uint8_t *buf = malloc(n), *o = buf;
    if (!buf) return;
    memcpy(o, &h, sizeof h), o += sizeof h;
    memcpy(o, a ? a : &zero, sizeof zero), o += sizeof zero;
    memcpy(o, control, CONTROL_SIZE), o += CONTROL_SIZE;
    memcpy(o, code, r->length), o += r->length;
    if (r->c1_len) memcpy(o, r->c1, r->c1_len);
    snprintf(tmp, PATH_CAP + 120, "%s/in", cache_dir);
    mkdir_p(tmp);
    cache_write(p, buf, n);
    free(buf);
}

// Reads stored inputs into *in (free in->buf); 0 when absent or malformed.
static int in_load(const uint8_t iid[32], Inputs *in) {
    char p[PATH_CAP + 80];
    in_path(iid, p);
    uint64_t n = 0;
    uint8_t *f = pack_read(PACK_IN, iid, &n);
    if (!f) f = read_file(p, &n);
    InHeader h;
    if (!f || n < sizeof h) return free(f), 0;
    memcpy(&h, f, sizeof h);
    if (h.magic != IN_MAGIC || h.version != IN_VERSION || h.answers_size != sizeof(GfxShaderAnswers) ||
        n != sizeof h + sizeof(GfxShaderAnswers) + CONTROL_SIZE + (uint64_t)h.length + h.c1_len)
        return free(f), 0;
    const uint8_t *o = f + sizeof h;
    memcpy(&in->a, o, sizeof in->a), o += sizeof in->a;
    in->control = o, o += CONTROL_SIZE;
    in->code = o, o += h.length;
    in->c1 = o;
    in->buf = f, in->stage = (int)h.stage, in->length = h.length, in->c1_len = h.c1_len;
    return 1;
}

// ---- translation

// Keys the translator answered but could not translate (an error reply, or output disk_load rejects): not asked again
// in this run. A draw that needs one is skipped each frame instead of waiting for the same failed translation, which
// costs a translator round trip of up to a second per shader per frame. A missing translator is not remembered.
#define FAILED_CAP 4096
static uint64_t failed_keys[FAILED_CAP];   // open addressing on the key's first 8 bytes (0 = free)
static int failed_n;

static uint64_t failed_tag(const uint8_t key[32]) {
    uint64_t t;
    memcpy(&t, key, 8);
    return t ? t : 1;
}
static int failed_has(const uint8_t key[32]) {
    uint64_t t = failed_tag(key);
    int hit = 0;
    plat_mutex_lock(&cache_lock);
    for (unsigned i = (unsigned)t % FAILED_CAP, n = 0; n < FAILED_CAP && failed_keys[i]; i = (i + 1) % FAILED_CAP, n++)
        if (failed_keys[i] == t) { hit = 1; break; }
    plat_mutex_unlock(&cache_lock);
    return hit;
}
static const GfxShader *fail_keep(const uint8_t key[32], const char *why, const char *hex) {
    uint64_t t = failed_tag(key);
    plat_mutex_lock(&cache_lock);
    if (failed_n < FAILED_CAP * 3 / 4) {
        unsigned i = (unsigned)t % FAILED_CAP;
        while (failed_keys[i] && failed_keys[i] != t) i = (i + 1) % FAILED_CAP;
        if (!failed_keys[i]) failed_keys[i] = t, failed_n++;
    }
    plat_mutex_unlock(&cache_lock);
    return fail(why, hex);
}

static const GfxShader *translate_one(const Req *r, int stage, const uint8_t *control, const uint8_t *code,
                                      const GfxShaderAnswers *a) {
    const uint8_t *key = r->key;
    GfxShader *s;
    UseMask m;
    uint8_t mk[32], miid[32];
    // Another thread may have translated it while this one waited in flight_enter.
    const uint8_t *look = r->final ? key : NULL, *look_iid = r->iid;
    if (!r->final && use_find(r->code_hash, &m)) build_key(r->code_hash, a, &m, mk, miid), look = mk, look_iid = miid;
    if (look && (s = mem_find(look))) return plat_inc64(&n_hits), s;
    if (look && (s = disk_load(look, look_iid))) return plat_inc64(&n_hits), mem_insert(s);
    if (failed_has(key)) return NULL;   // failed before in this run (fail_keep)

#ifndef GAME_NS2
    uint64_t skip = stage == GFX_STAGE_CS ? 0 : CODE_SKIP;
#endif
    char hex[65], code_path[PATH_CAP + 80], spv_path[PATH_CAP + 80], res_path[PATH_CAP + 80];
    key_hex(key, hex);
    mkdir_p(cache_dir);
    snprintf(code_path, PATH_CAP + 80, "%s/%s.code", cache_dir, hex);
    snprintf(spv_path, PATH_CAP + 80, "%s/%s.spv", cache_dir, hex);
    snprintf(res_path, PATH_CAP + 80, "%s/%s.res.json", cache_dir, hex);
#ifdef GAME_NS2
    if (!write_file(code_path, code + r->xskip, (size_t)(r->xlen - r->xskip))) return fail("cannot write the code file", hex);
#else
    if (!write_file(code_path, code + skip, (size_t)(r->length - skip))) return fail("cannot write the code file", hex);
#endif
    size_t cap = 2 * PATH_CAP * 6 + 2 * r->c1_len + 8192;
    char *req = malloc(cap), reply[LINE_MAX_LEN];
    if (!req) return plat_file_delete(code_path), fail("out of memory", hex);
    build_request(req, hex, r->stage_name, code_path, r->c1, r->c1_len, a, control);
    int ok = svc_request(req, reply, sizeof reply);
    // REKARINA_XLATE_KEEP=1: a last-frame variant's job stays as <key>.job (one request line) with its code, to run the
    // translator on by hand (totk_xlate --serve < <key>.job)
    static int keep = -1;
    if (keep < 0) keep = rt_knob("XLATE_KEEP", "0");
    int kept = keep && a && a->mvprev;
    if (kept) {
        char job_path[PATH_CAP + 80];
        snprintf(job_path, sizeof job_path, "%s/%s.job", cache_dir, hex);
        write_file(job_path, req, strlen(req));
    }
    free(req);
    if (!kept) plat_file_delete(code_path);
    if (!ok) return fail("translator unavailable", hex);
    if (strncmp(reply, "ok ", 3)) {
        plat_file_delete(spv_path), plat_file_delete(res_path);
        return fail_keep(key, reply, hex);
    }
    if (cache_flush) plat_file_flush(spv_path), plat_file_flush(res_path);   // the translator wrote them: on disk first
    if (!(s = disk_load(key, r->iid))) return fail_keep(key, "translator output missing or malformed", hex);
    if (!r->final) {                                       // name the files after the cache key, remember the mask
        m = use_of(s);
        build_key(r->code_hash, a, &m, mk, miid);
        char mhex[65], to[PATH_CAP + 80];
        key_hex(mk, mhex);
        snprintf(to, PATH_CAP + 80, "%s/%s.spv", cache_dir, mhex);
        int (*move)(const char *, const char *) = cache_flush ? plat_file_replace : plat_file_replace_lazy;
        int moved = move(spv_path, to);
        snprintf(to, PATH_CAP + 80, "%s/%s.res.json", cache_dir, mhex);
        moved = moved && move(res_path, to);
        if (moved) {
            use_store(r->code_hash, &m);
        } else {                                           // keep the mask for this run only
            plat_file_delete(spv_path), plat_file_delete(res_path), rt_log("gfx_shader: cannot rename %s", hex);
            plat_mutex_lock(&cache_lock);
            use_put(r->code_hash, 1, &m);
            plat_mutex_unlock(&cache_lock);
        }
        memcpy(s->key, mk, 32), memcpy(s->iid, miid, 32);
    }
    in_store(s->iid, stage, control, code, r, a);
    plat_inc64(&n_xlated);
    return mem_insert(s);
}

// The service round trip for a request that is in neither cache. One translation per cache key (per code hash while
// the use mask is unknown) runs at a time; different shaders run in parallel on the children.
static const GfxShader *translate(const Req *r, int stage, const uint8_t *control, const uint8_t *code,
                                  const GfxShaderAnswers *a) {
    int away = kern_core_away();
    const uint8_t *id = r->final ? r->key : r->code_hash;
    uint64_t tf = plat_now_ns();
    flight_enter(id);
    uint64_t t0 = plat_now_ns(), x0 = (uint64_t)n_xlated;
    const GfxShader *s = translate_one(r, stage, control, code, a);
    uint64_t t1 = plat_now_ns();
    plat_add64(&tm_flight, (int64_t)(t0 - tf)), plat_add64(&tm_one, (int64_t)(t1 - t0)), plat_inc64(&tm_n);
    if ((uint64_t)n_xlated != x0)   // translated here, not found in a cache: in play it goes into the ledger (prep.c)
        prep_note(PREP_XLATE, 0, (double)(t1 - t0) / 1e6);
    flight_leave(id);
    kern_core_back(&away);
    return s;
}

// ---- background translation (gfx_shader_get_async)
// svc_runtime_limit() workers, one per translator child. A job owns copies of the control section and the code, so the
// guest may free or rewrite its program while the job waits. A key stays in X.waits while its job is queued or running,
// and for good when the translation failed, so a failing shader is not sent again on every draw.
typedef struct XJob {
    struct XJob *next;
    int stage;
    Req r;                                                 // r.c1 points into code
    uint8_t *control, *code;
    GfxShaderAnswers a;
} XJob;

typedef struct XWait {
    struct XWait *next;
    uint8_t key[32];
    int failed;
} XWait;

static struct {
    PlatMutex lock;
    PlatCond work, done;                         // done: a job finished (gfx_shader_await)
    XJob *head, *tail;
    XWait *waits[256];
    int started;                                           // 1 running, -1 no thread could be created
} X = {PLAT_MUTEX_INIT, PLAT_COND_INIT, PLAT_COND_INIT};

static volatile int64_t n_async;

// Caller holds X.lock.
static XWait **wait_slot(const uint8_t key[32]) {
    XWait **w = &X.waits[key[0]];
    while (*w && memcmp((*w)->key, key, 32)) w = &(*w)->next;
    return w;
}

static void xlate_worker(void *arg) {
    (void)arg;
    for (;;) {
        plat_mutex_lock(&X.lock);
        while (!X.head) plat_cond_wait(&X.work, &X.lock);
        XJob *j = X.head;
        if (!(X.head = j->next)) X.tail = NULL;
        plat_mutex_unlock(&X.lock);
        const GfxShader *s = translate(&j->r, j->stage, j->control, j->code, &j->a);
        plat_mutex_lock(&X.lock);
        XWait **w = wait_slot(j->r.key), *e = *w;
        if (e && s) *w = e->next, free(e);
        else if (e) e->failed = 1;
        plat_mutex_unlock(&X.lock);
        plat_cond_broadcast(&X.done);
        if (s) plat_inc64(&n_async);
        free(j->control), free(j->code), free(j);
    }
    return;
}
const GfxShader *gfx_shader_get_async(int stage, const uint8_t *control, const uint8_t *code, uint64_t code_size,
                                      const GfxShaderAnswers *a, int *pending) {
    *pending = 0;
    CTL_NORM(control);
    Req r;
    if (!req_init(&r, stage, control, code, code_size, NULL, a)) return NULL;
    const GfxShader *s = get_async_one(r, stage, control, code, a, pending);
    if (s) ans_note(&r, stage, control, code, a, s);
    return s;
}

static const GfxShader *get_async_one(Req r, int stage, const uint8_t *control, const uint8_t *code,
                                      const GfxShaderAnswers *a, int *pending) {
    GfxShader *s = mem_find(r.key);
    if (s) return plat_inc64(&n_hits), s;
    plat_mutex_lock_shared(&X.lock);
    XWait *w = *wait_slot(r.key);
    int failed = w && w->failed;
    plat_mutex_unlock_shared(&X.lock);
    if (w) return *pending = !failed, NULL;
    paths();
    if (!cache_dir[0]) return fail("no cache directory", NULL);
    if (r.final && (s = disk_load(r.key, r.iid))) return plat_inc64(&n_hits), mem_insert(s);

    XJob *j = calloc(1, sizeof *j);
    XWait *e = calloc(1, sizeof *e);
    uint8_t *ctl = malloc(CONTROL_SIZE), *cd = malloc(r.length);
    plat_mutex_lock(&X.lock);
    if (!X.started) {
        int n = 0;
        for (int i = 0; i < svc_runtime_limit(); i++) {
            if (plat_thread_spawn(xlate_worker, NULL, "xlate", 0)) n++;
        }
        X.started = n ? 1 : -1;
        if (n) rt_log("gfx_shader: graphics shaders are translated by %d background workers (REKARINA_XLATE_PROCS)", n);
        else rt_log("gfx_shader: cannot start the translation workers; translating synchronously");
    }
    XWait **slot = wait_slot(r.key);
    int raced = *slot != NULL;                             // another thread queued it between the checks
    int queued = !raced && X.started == 1 && j && e && ctl && cd;
    if (queued) {
        memcpy(ctl, control, CONTROL_SIZE), memcpy(cd, code, r.length);
        *j = (XJob){.stage = stage, .r = r, .control = ctl, .code = cd, .a = *a};
        j->r.c1 = cd + (r.length - r.c1_len);
        memcpy(e->key, r.key, 32);
        *slot = e;
        if (X.tail) X.tail->next = j;
        else X.head = j;
        X.tail = j;
        plat_cond_signal(&X.work);
    }
    plat_mutex_unlock(&X.lock);
    if (queued) return *pending = 1, NULL;
    free(j), free(e), free(ctl), free(cd);
    if (raced) return *pending = 1, NULL;
    return translate(&r, stage, control, code, a);         // no worker or out of memory
}

const GfxShader *gfx_shader_await(int stage, const uint8_t *control, const uint8_t *code, uint64_t code_size,
                                  const GfxShaderAnswers *a) {
    int pending;
    CTL_NORM(control);
    const GfxShader *s = gfx_shader_get_async(stage, control, code, code_size, a, &pending);
    Req r;
    if (!pending || !req_init(&r, stage, control, code, code_size, NULL, a)) return s;
    uint64_t p0 = perf_begin();   // the wait counts as the render thread's translation time
    plat_mutex_lock(&X.lock);
    for (XJob *j = X.head, *prev = NULL; j; prev = j, j = j->next)
        if (!memcmp(j->r.key, r.key, 32)) {                // queued: to the front, the worker takes it next
            if (prev) {
                prev->next = j->next;
                if (X.tail == j) X.tail = prev;
                j->next = X.head, X.head = j;
            }
            break;
        }
    XWait *w;
    while ((w = *wait_slot(r.key)) && !w->failed) plat_cond_wait(&X.done, &X.lock);
    plat_mutex_unlock(&X.lock);
    perf_end(PERF_XLATE, p0);
    if (w) return NULL;                                    // failed: logged by the worker
    s = gfx_shader_get_async(stage, control, code, code_size, a, &pending);
    return pending ? NULL : s;
}

uint64_t gfx_shader_async_done(void) { return (uint64_t)n_async; }

void gfx_shader_counts(uint64_t *keys, uint64_t *translated) { *keys = (uint64_t)n_keys, *translated = (uint64_t)n_xlated; }

const uint8_t *gfx_shader_key(const GfxShader *s) { return s->key; }

const uint8_t *gfx_shader_input_id(const GfxShader *s) { return s->iid; }

// By input id: memory, then the disk cache under the current translator, then (xlate 1) a translation from the stored
// inputs. NULL when none has it; a failed translation is counted as a failure.
const GfxShader *gfx_shader_load_input(const uint8_t iid[32], int xlate) {
    paths();
    if (!cache_dir[0]) return NULL;
    uint8_t key[32];
    key_of_input(iid, key);
    GfxShader *s = mem_find(key);
    if (s) return s;
    if ((s = disk_load(key, iid))) return mem_insert(s);
    Inputs in;
    if (!xlate || !in_load(iid, &in)) return NULL;
    const GfxShader *t = gfx_shader_get(in.stage, in.control, in.code, in.length, in.c1, &in.a);
    free(in.buf);
    return t;
}

// From the disk cache only, as a shader the caller owns and releases with gfx_shader_release_private (the boot library
// prebuild warms the driver's pipeline cache with it without keeping 40 KB per shader in memory). When the memory
// cache has it, that entry is returned with *owned 0 and must not be released. NULL when not cached.
const GfxShader *gfx_shader_load_private(const uint8_t iid[32], int *owned) {
    *owned = 0;
    paths();
    if (!cache_dir[0]) return NULL;
    uint8_t key[32];
    key_of_input(iid, key);
    GfxShader *s = mem_find(key);
    if (s) return s;
    if ((s = disk_load(key, iid))) *owned = 1;
    return s;
}

void gfx_shader_release_private(const GfxShader *s) {
    if (s) shader_free((GfxShader *)s);
}

// ---- boot prewarm (gfx_shader_prewarm)

typedef struct {
    const uint8_t (*named)[32];                            // loaded into memory, results in out
    size_t n_named;
    const GfxShader **out;
    uint8_t (*other)[32];                                  // other stored inputs: translated when not cached, not loaded
    size_t n_other;
    volatile int64_t next, done;
    volatile int32_t missing, other_cached;
} Warm;

static void warm_worker(void *arg) {
    Warm *w = arg;
    for (;;) {
        size_t i = (size_t)plat_inc64(&w->next) - 1;
        if (i >= w->n_named + w->n_other) return;
        if (i < w->n_named) {
            if (!(w->out[i] = gfx_shader_load_input(w->named[i], 1))) plat_inc32(&w->missing);
        } else {
            const uint8_t *iid = w->other[i - w->n_named];
            uint8_t key[32];
            char hex[65], p[PATH_CAP + 80];
            key_of_input(iid, key);
            key_hex(key, hex);
            snprintf(p, PATH_CAP + 80, "%s/%s.spv", cache_dir, hex);
            if (pack_find(PACK_SPV, key) || file_exists(p)) plat_inc32(&w->other_cached);
            else gfx_shader_load_input(iid, 1);
        }
        plat_inc64(&w->done);
    }
}

static int iid_cmp(const void *x, const void *y) { return memcmp(x, y, 32); }

typedef struct {
    const uint8_t (*named)[32];
    size_t n_named, n, cap;
    uint8_t (*v)[32];
} InList;

static int in_collect(const char *file, void *ctx) {
    InList *l = ctx;
    char name[80];
    uint8_t iid[32];
    if (strlen(file) != 67) return 1;                      // 64 hex digits + ".in"
    memcpy(name, file, 64);
    name[64] = 0;
    if (!hex32(name, iid) || (l->n_named && bsearch(iid, l->named, l->n_named, 32, iid_cmp))) return 1;
    if (l->n == l->cap) {
        size_t c = l->cap ? l->cap * 2 : 1024;
        uint8_t (*t)[32] = realloc(l->v, c * 32);
        if (!t) return 0;
        l->v = t, l->cap = c;
    }
    memcpy(l->v[l->n++], iid, 32);
    return 1;
}

// The stored inputs not in named (sorted): ids parsed from the in/ file names.
static size_t other_inputs(const uint8_t (*named)[32], size_t n_named, uint8_t (**out)[32]) {
    char dir[PATH_CAP + 80];
    snprintf(dir, sizeof dir, "%s/in", cache_dir);
    InList l = {.named = named, .n_named = n_named};
    char name[68];
    for (uint32_t i = 0; i < pack_n; i++) {   // the packed inputs, under the name their loose file would have
        if ((pack_ix[i].kind & PACK_KIND) != PACK_IN) continue;
        key_hex(pack_ix[i].name, name);
        memcpy(name + 64, ".in", 4);
        if (!in_collect(name, &l)) break;
    }
    plat_file_list(dir, ".in", in_collect, &l);   // in_store writes no loose file for a packed input: no duplicates
    return *out = l.v, l.n;
}

void gfx_shader_prewarm(const uint8_t (*iids)[32], size_t n, const GfxShader **out, int threads, int procs) {
    paths();
    double t0 = (double)plat_now_ms();
    memset(out, 0, n * sizeof *out);
    if (!cache_dir[0]) return;
    uint64_t xl0 = (uint64_t)n_xlated, fail0 = (uint64_t)n_failures;
    uint8_t (*sorted)[32] = n ? malloc(n * 32) : NULL;
    if (sorted) memcpy(sorted, iids, n * 32), qsort(sorted, n, 32, iid_cmp);
    Warm w = {.named = iids, .n_named = n, .out = out};
    w.n_other = other_inputs(sorted, sorted ? n : 0, &w.other);
    free(sorted);
    size_t total = w.n_named + w.n_other;
    threads = threads < 1 ? 1 : threads > 64 ? 64 : threads;
    if ((size_t)threads > total) threads = (int)(total ? total : 1);
    svc_set_limit(procs);
    PlatThread *t[64];
    int k = 0;
    for (int i = 0; i < threads; i++)
        if ((t[k] = plat_thread_new(warm_worker, &w, "shader prewarm"))) k++;
    if (!k) warm_worker(&w);
    for (int m = 0; m < k; m++)
        while (!plat_thread_join_ns(t[m], 2000000000u))
            rt_log("gfx_shader: prewarm: %lld of %zu shaders, %llu translated, %.0f ms", (long long)w.done, total,
                   (unsigned long long)((uint64_t)n_xlated - xl0), (double)plat_now_ms() - t0);
    svc_set_limit(svc_runtime_limit());
    rt_log("gfx_shader: prewarm: %zu shaders named by pipeline keys (%ld not available), %zu other stored inputs (%ld "
           "cached); %llu translated, %llu failed, with up to %d translators and %d threads, in %.0f ms", w.n_named,
           (long)w.missing, w.n_other, (long)w.other_cached, (unsigned long long)((uint64_t)n_xlated - xl0),
           (unsigned long long)((uint64_t)n_failures - fail0), procs, k, (double)plat_now_ms() - t0);
    free(w.other);
    {   // what the memory cache holds now (the process's memory on a 16 GB PC)
        uint64_t n = 0, spv = 0, refl = 0;
        plat_mutex_lock_shared(&cache_lock);
        for (int i = 0; i < BUCKETS; i++)
            for (const GfxShader *s = buckets[i]; s; s = s->next) n++, spv += s->words * 4, refl += s->reflection_size;
        plat_mutex_unlock_shared(&cache_lock);
        rt_log("gfx_shader: memory cache after the prewarm: %llu shaders, %.1f MiB of SPIR-V, %.1f MiB of reflection text",
               (unsigned long long)n, spv / 1048576.0, refl / 1048576.0);
    }
}
const uint32_t *gfx_shader_spirv(const GfxShader *s, uint64_t *words) {
    if (words) *words = s->words;
    return s->spirv;
}

int gfx_shader_stage(const GfxShader *s) { return s->stage; }

int gfx_shader_resources(const GfxShader *s, int kind, const GfxShaderRes **res) {
    if (kind < 0 || kind >= GFX_RES_KINDS) return 0;
    if (res) *res = s->res[kind];
    return s->nres[kind];
}

// The slots of the use mask: answers outside it select the same translation (mask_answers), so a caller may leave
// their texture and image types zero. Parses the reflection: callers keep the result.
uint32_t gfx_shader_slot_mask(const GfxShader *s) { return use_of(s).slots; }

uint32_t gfx_shader_inputs(const GfxShader *s, GfxShaderIo io[32]) {
    if (io) memcpy(io, s->in, sizeof s->in);
    return s->in_mask;
}

uint32_t gfx_shader_mvprev_banks(const GfxShader *s) { return s->mvprev_banks; }

int gfx_shader_palette(const GfxShader *s, int *slot, int *storage, uint32_t *joints, uint32_t *raw_joints) {
    if (!s->pal_stride) return 0;
    *slot = s->pal_slot, *storage = s->pal_storage, *joints = s->pal_joints, *raw_joints = s->pal_raw_joints;
    return s->pal_stride;
}

uint32_t gfx_shader_outputs(const GfxShader *s, GfxShaderIo io[32]) {
    if (io) memcpy(io, s->out, sizeof s->out);
    return s->out_mask;
}

void gfx_shader_local_size(const GfxShader *s, uint32_t local[3]) { memcpy(local, s->local, sizeof s->local); }

// ---- answers met in play
// The translated-shader key includes the answers a draw's state implies, so a translation made without play (the
// census) does not match the keys play asks for. Every (program, answers) that a gfx_shader_get* call resolves is
// recorded next to pipeline-keys.bin, so the first-start preparation (Task 2.6, here the REKARINA_TEST_PREPARE_SHADERS hook)
// translates exactly the variants play needs, from these files alone:
//   shader-answers.bin  "TOTKSANS", version, record size; records: stage (u32), code hash, input id, pack_answers bytes
//                       of the full answers, zero padding. One record per input id: answers outside the shader's use
//                       mask do not change its SPIR-V, are out of its input id and cache key (mask_answers) and so do
//                       not make records of their own (a fight run meets 4126 full pairs, 1238 input ids). The full
//                       answers are kept so a translator with a wider mask still gets them. Under a new translator
//                       the masks, hence the input ids, are new, and play appends records for them.
//   shader-programs.bin "TOTKSPRG", version, 0; records: stage, length, c1 length (u32 each), code hash, control section,
//                       code[length], c1. One per code hash; written before the answers that name it. Game content:
//                       it stays in the ignored pipeline cache directory.
// A file with another header is rewritten; a torn tail (killed process) is cut off at open.
#define ANS_VERSION 1
#define ANS_PACKED 147                                     // pack_answers' size
#define ANS_REC (4 + 32 + 32 + ANS_PACKED + 1)
#define PRG_VERSION 1
#define PRG_HEAD (12 + 32 + CONTROL_SIZE)
typedef struct { char magic[8]; uint32_t version, rec_size; } RecHeader;
typedef struct { uint64_t *v; uint32_t cap, n; } Set64;    // open addressing, 0 empty
typedef struct { uint8_t *p; size_t n, cap; } Bytes;

static struct {
    PlatMutex lock;
    volatile int32_t on;
    char ans_path[PATH_CAP], prg_path[PATH_CAP];
    Set64 pairs, progs;                                    // input ids, code hashes seen (in the files or this run)
    Bytes ans, prg;                                        // pending
    uint64_t pairs_new;                                    // added this run
} RC = {PLAT_MUTEX_INIT};

static int set_add(Set64 *s, uint64_t h) {                 // 1 if new
    h = h ? h : 1;
    if ((s->n + 1) * 2 > s->cap) {
        uint32_t cap = s->cap ? s->cap * 2 : 4096;
        uint64_t *t = calloc(cap, sizeof *t);
        if (!t) return 0;
        for (uint32_t i = 0; i < s->cap; i++)
            if (s->v[i]) {
                uint32_t j = (uint32_t)s->v[i] & (cap - 1);
                while (t[j]) j = (j + 1) & (cap - 1);
                t[j] = s->v[i];
            }
        free(s->v), s->v = t, s->cap = cap;
    }
    uint32_t j = (uint32_t)h & (s->cap - 1);
    for (; s->v[j]; j = (j + 1) & (s->cap - 1))
        if (s->v[j] == h) return 0;
    s->v[j] = h, s->n++;
    return 1;
}

static int bytes_put(Bytes *b, const void *p, size_t n) {
    if (!n) return 1;
    if (b->n + n > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 65536;
        while (cap < b->n + n) cap *= 2;
        uint8_t *t = realloc(b->p, cap);
        if (!t) return 0;
        b->p = t, b->cap = cap;
    }
    memcpy(b->p + b->n, p, n), b->n += n;
    return 1;
}

static uint64_t hash8(const uint8_t *sha) {
    uint64_t v;
    memcpy(&v, sha, 8);
    return v;
}

static void unpack_answers(const uint8_t *p, GfxShaderAnswers *a) {
    memset(a, 0, sizeof *a);
    memcpy(a->tex_type, p, 32), p += 32;
    memcpy(a->img_type, p, 32), p += 32;
    memcpy(a->img_format, p, 32), p += 32;
    memcpy(a->attr_type, p, 32), p += 32;
    memcpy(a->frag_out_type, p, 8), p += 8;
    for (int i = 0; i < 3; i++) a->local[i] = (uint16_t)(p[0] | p[1] << 8), p += 2;
    memcpy(&a->shared_bytes, p, 4), p += 4;
    a->depth_minus_w = *p & 0x7f, a->mvprev = *p >> 7;
}

// Reads a record file's body (after its header) into *body; 0 when absent, and a file with another header is removed.
static int rec_read(const char *path, const char *magic, uint32_t version, uint32_t rec_size, uint8_t **body, size_t *n) {
    *body = NULL, *n = 0;
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    RecHeader h;
    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);
    int ok = len >= (long)sizeof h && fread(&h, sizeof h, 1, f) == 1 && !memcmp(h.magic, magic, 8) &&
             h.version == version && h.rec_size == rec_size;
    if (ok && len > (long)sizeof h) {
        *n = (size_t)len - sizeof h;
        ok = (*body = malloc(*n)) && fread(*body, 1, *n, f) == *n;
    }
    fclose(f);
    if (!ok) {
        free(*body), *body = NULL, *n = 0;
        if (len > 0) rt_log("gfx_shader: %s has another format or version; it is rewritten", path), remove(path);
    }
    return ok;
}

// Walks shader-programs.bin records; returns the byte length of the whole records. fn may be NULL.
static size_t prg_walk(const uint8_t *b, size_t n, void (*fn)(const uint8_t *rec, void *ctx), void *ctx) {
    size_t o = 0;
    while (n - o >= PRG_HEAD) {
        uint32_t len, c1;
        memcpy(&len, b + o + 4, 4), memcpy(&c1, b + o + 8, 4);
        size_t size = PRG_HEAD + (size_t)len + c1;
        if (len > (64u << 20) || c1 > len || size > n - o) break;
        if (fn) fn(b + o, ctx);
        o += size;
    }
    return o;
}

static void prg_seen(const uint8_t *rec, void *ctx) { set_add(ctx, hash8(rec + 12)); }

static void truncate_to(const char *path, size_t size) {
    if (!plat_file_truncate(path, size)) return;
    rt_log("gfx_shader: %s had a torn tail; cut to %zu bytes", path, size);
}

void gfx_shader_record_at(const char *keys_path) {
    if (!keys_path || !keys_path[0] || RC.on) return;
    plat_mutex_lock(&RC.lock);
    const char *slash = strrchr(keys_path, '\\'), *fs = strrchr(keys_path, '/');
    if (fs > slash) slash = fs;
    int dir = slash ? (int)(slash + 1 - keys_path) : 0;
    snprintf(RC.ans_path, sizeof RC.ans_path, "%.*sshader-answers.bin", dir, keys_path);
    snprintf(RC.prg_path, sizeof RC.prg_path, "%.*sshader-programs.bin", dir, keys_path);
    uint8_t *b;
    size_t n;
    if (rec_read(RC.ans_path, "TOTKSANS", ANS_VERSION, ANS_REC, &b, &n)) {
        for (size_t o = 0; o + ANS_REC <= n; o += ANS_REC) set_add(&RC.pairs, hash8(b + o + 36));
        if (n % ANS_REC) truncate_to(RC.ans_path, sizeof(RecHeader) + n / ANS_REC * ANS_REC);
    }
    free(b);
    if (rec_read(RC.prg_path, "TOTKSPRG", PRG_VERSION, 0, &b, &n)) {
        size_t whole = prg_walk(b, n, prg_seen, &RC.progs);
        if (whole != n) truncate_to(RC.prg_path, sizeof(RecHeader) + whole);
    }
    free(b);
    rt_log("gfx_shader: recording answers met in play to %s (%u records, %u programs recorded before)", RC.ans_path,
           RC.pairs.n, RC.progs.n);
    plat_xchg32(&RC.on, 1);
    plat_mutex_unlock(&RC.lock);
}

// A get resolved (r, a) to s: queue the record when its input id is new, and its program when new. Cost: a set
// lookup under a lock per call; gets happen per (program, answers) change in vk_exec_draw's shader table, not per draw.
static void ans_note(const Req *r, int stage, const uint8_t *control, const uint8_t *code, const GfxShaderAnswers *a,
                     const GfxShader *s) {
    if (!RC.on) return;
    uint32_t st = (uint32_t)stage;
    plat_mutex_lock(&RC.lock);
    if (set_add(&RC.pairs, hash8(s->iid))) {
        uint8_t rec[ANS_REC] = {0};
        memcpy(rec, &st, 4), memcpy(rec + 4, r->code_hash, 32), memcpy(rec + 36, s->iid, 32);
        pack_answers(rec + 68, a);
        if (set_add(&RC.progs, hash8(r->code_hash))) {
            uint32_t head[3] = {st, r->length, r->c1_len};
            bytes_put(&RC.prg, head, 12), bytes_put(&RC.prg, r->code_hash, 32);
            bytes_put(&RC.prg, control, CONTROL_SIZE), bytes_put(&RC.prg, code, r->length);
            bytes_put(&RC.prg, r->c1, r->c1_len);
        }
        bytes_put(&RC.ans, rec, sizeof rec);
        RC.pairs_new++;
    }
    plat_mutex_unlock(&RC.lock);
}

static int rec_append(const char *path, const char *magic, uint32_t version, uint32_t rec_size, const Bytes *b) {
    FILE *f = fopen(path, "ab");
    if (!f) return 0;
    fseek(f, 0, SEEK_END);
    int ok = 1;
    if (ftell(f) == 0) {
        RecHeader h = {{0}, version, rec_size};
        memcpy(h.magic, magic, 8);
        ok = fwrite(&h, sizeof h, 1, f) == 1;
    }
    ok = ok && fwrite(b->p, 1, b->n, f) == b->n;
    return !fclose(f) && ok;
}

static PlatMutex rec_io = PLAT_MUTEX_INIT;   // the record files: appends in order, no read during an append

void gfx_shader_record_flush(int force) {
    static volatile int64_t last;
    int64_t now = (int64_t)plat_now_ms();
    if (!RC.on || (!force && now - last < 2000)) return;
    last = now;
    // The files are written outside RC.lock, which the render thread takes to record a new answer.
    plat_mutex_lock(&rec_io);
    plat_mutex_lock(&RC.lock);
    Bytes ans = RC.ans, prg = RC.prg;
    RC.ans = (Bytes){0}, RC.prg = (Bytes){0};
    uint64_t total = RC.pairs.n, added = RC.pairs_new;
    plat_mutex_unlock(&RC.lock);
    if (prg.n && !rec_append(RC.prg_path, "TOTKSPRG", PRG_VERSION, 0, &prg)) rt_log("gfx_shader: cannot write %s", RC.prg_path);
    if (ans.n && !rec_append(RC.ans_path, "TOTKSANS", ANS_VERSION, ANS_REC, &ans)) rt_log("gfx_shader: cannot write %s", RC.ans_path);
    plat_mutex_unlock(&rec_io);
    if (ans.n) rt_log("gfx_shader: answers: %zu new records appended (%llu new this run, %llu recorded)", ans.n / ANS_REC,
                      (unsigned long long)added, (unsigned long long)total);
    free(ans.p), free(prg.p);
}

// ---- preparation from record files: the answers met in play (REKARINA_TEST_PREPARE_SHADERS=1) or the seed that
// tools/prepare_shaders.py writes from the MacNativeGames shader list (rekarina.exe --prepare-shaders). Both use the record
// layouts above; a seed record has a zero input id (the use mask, hence the id, is the translator's to find) and its
// pad byte holds the list priority (0 title, 1 warm-up, 2 rest) and SEED_DEFAULT for a default variant, which
// gfx_shader_guess_refine may correct after its first translation.
#define SEED_DEFAULT 0x80
typedef struct { const uint8_t *rec; } PrgIdx;
typedef struct { PrgIdx *idx; size_t n; } PrgList;
typedef struct {
    PrgList progs;
    const uint8_t *ans;
    size_t n_ans;
    uint8_t (*iid)[32];                                    // per answers record: the resolved input id, zero on failure
    volatile int64_t next, done, missing, failed, refined;
} Prep;

static int prg_cmp(const void *x, const void *y) { return memcmp(((const PrgIdx *)x)->rec + 12, ((const PrgIdx *)y)->rec + 12, 32); }

static void prg_collect(const uint8_t *rec, void *ctx) {
    PrgList *l = ctx;
    l->idx[l->n++].rec = rec;
}

static void prep_worker(void *arg) {
    Prep *p = arg;
    for (;;) {
        size_t i = (size_t)plat_inc64(&p->next) - 1;
        if (i >= p->n_ans) return;
        const uint8_t *r = p->ans + i * ANS_REC;
        uint8_t key_rec[12 + 32];
        memcpy(key_rec + 12, r + 4, 32);
        PrgIdx k = {key_rec}, *e = bsearch(&k, p->progs.idx, p->progs.n, sizeof k, prg_cmp);
        if (!e) {
            plat_inc64(&p->missing), plat_inc64(&p->done);
            continue;
        }
        uint32_t st, len;
        memcpy(&st, e->rec, 4), memcpy(&len, e->rec + 4, 4);
        const uint8_t *control = e->rec + 44, *code = control + CONTROL_SIZE, *c1 = code + len;
        GfxShaderAnswers a, b;
        unpack_answers(r + 68, &a);
        const GfxShader *s = gfx_shader_get((int)st, control, code, len, c1, &a), *t;
        if (s && r[ANS_REC - 1] & SEED_DEFAULT && gfx_shader_guess_refine(s, (int)st, &a, &b) &&
            (t = gfx_shader_get((int)st, control, code, len, c1, &b)))
            s = t, plat_inc64(&p->refined);
        if (!s) plat_inc64(&p->failed);
        else if (p->iid) memcpy(p->iid[i], s->iid, 32);
        plat_inc64(&p->done);
    }
}

// Translates every record of ans_path whose program is in prg_path on `threads` threads with up to `procs` translators.
// *ans_out, *iid_out (optional; free them): the answers records and the input id each resolved to (zero: failed).
// Returns the record count, -1 without the files.
static long long prepare_files(const char *ans_path, const char *prg_path, int threads, int procs, const char *what,
                               uint8_t **ans_out, uint8_t (**iid_out)[32]) {
    paths();
    double t0 = (double)plat_now_ms();
    uint8_t *ab = NULL, *pb = NULL;
    size_t an = 0, pn = 0;
    if (!rec_read(ans_path, "TOTKSANS", ANS_VERSION, ANS_REC, &ab, &an) ||
        !rec_read(prg_path, "TOTKSPRG", PRG_VERSION, 0, &pb, &pn)) {
        rt_log("gfx_shader: prepare: no %s (%s)", what, ans_path);
        free(ab), free(pb);
        return -1;
    }
    Prep p = {.progs = {calloc(pn / PRG_HEAD + 1, sizeof(PrgIdx)), 0}, .ans = ab, .n_ans = an / ANS_REC};
    if (iid_out) p.iid = calloc(p.n_ans + 1, 32);
    if (!p.progs.idx || (iid_out && !p.iid)) return free(p.progs.idx), free(p.iid), free(ab), free(pb), -1;
    prg_walk(pb, pn, prg_collect, &p.progs);
    qsort(p.progs.idx, p.progs.n, sizeof *p.progs.idx, prg_cmp);
    uint64_t xl0 = (uint64_t)n_xlated;
    threads = threads < 1 ? 1 : threads > 64 ? 64 : threads;
    svc_set_limit(procs);
    PlatThread *t[64];
    int k = 0;
    for (int i = 0; i < threads; i++)
        if ((t[k] = plat_thread_new(prep_worker, &p, "shader prepare"))) k++;
    if (!k) prep_worker(&p);
    int64_t was[5] = {tm_xchg, tm_svc_wait, tm_flight, tm_one, tm_n};
    double tw = (double)plat_now_ms();
    for (int m = 0; m < k; m++)
        while (!plat_thread_join_ns(t[m], 5000000000u)) {
            // the window since the line before: translators busy on average, and a translation's time by part (ms)
            int64_t now[5] = {tm_xchg, tm_svc_wait, tm_flight, tm_one, tm_n}, d[5];
            for (int j = 0; j < 5; j++) d[j] = now[j] - was[j], was[j] = now[j];
            double tn = (double)plat_now_ms(), per = d[4] ? 1e6 * (double)d[4] : 1;
            rt_log("gfx_shader: prepare: %lld of %zu pairs, %llu translated, %.0f ms; %.1f of %d translators busy, a "
                   "translation %.0f ms: translator %.0f, waiting for one %.0f, cache files %.0f, same program %.0f",
                   (long long)p.done, p.n_ans, (unsigned long long)((uint64_t)n_xlated - xl0), tn - t0,
                   (double)d[0] / 1e6 / (tn - tw > 1 ? tn - tw : 1), procs, (double)(d[3] + d[2]) / per,
                   (double)d[0] / per, (double)d[1] / per, (double)(d[3] - d[0] - d[1]) / per, (double)d[2] / per);
            tw = tn;
        }
    svc_set_limit(svc_runtime_limit());
    rt_log("gfx_shader: prepare: %zu %s of %zu programs: %llu translated, %lld failed, %lld without a program, the rest "
           "cached; %lld default variants refined; %.0f ms", p.n_ans, what, p.progs.n,
           (unsigned long long)((uint64_t)n_xlated - xl0), (long long)p.failed, (long long)p.missing, (long long)p.refined,
           (double)plat_now_ms() - t0);
    free(p.progs.idx), free(pb);
    if (ans_out) *ans_out = ab;
    else free(ab);
    if (iid_out) *iid_out = p.iid;
    return (long long)p.n_ans;
}

void gfx_shader_prepare_recorded(int threads, int procs) {
    if (!RC.on) {
        rt_log("gfx_shader: prepare: no recorded answers (not recording)");
        return;
    }
    prepare_files(RC.ans_path, RC.prg_path, threads, procs, "recorded pairs", NULL, NULL);
}

// ---- prepared variants (shader-prepared.bin next to pipeline-keys.bin)
// "TOTKSPRP", version, record size; records: stage, priority, 2 zero bytes, input id. gfx_shader_prepare_seed writes
// them in priority order, one per input id; gfx_shader_warm_list reads them at boot.
#define PRP_VERSION 1
#define PRP_REC 36

int gfx_shader_prepare_seed(const char *dir, int threads, int procs) {
    char ans[PATH_CAP + 32], prg[PATH_CAP + 32], out[PATH_CAP + 32];
    snprintf(ans, sizeof ans, "%s/seed-answers.bin", dir);
    snprintf(prg, sizeof prg, "%s/seed-programs.bin", dir);
    snprintf(out, sizeof out, "%s/shader-prepared.bin", dir);
    double t0 = (double)plat_now_ms();
    uint64_t fail0 = (uint64_t)n_failures;
    uint8_t *ab = NULL, (*iid)[32] = NULL;
    long long n = prepare_files(ans, prg, threads, procs, "seed variants", &ab, &iid);
    if (n < 0) return 1;
    uint8_t *recs = calloc((size_t)n + 1, PRP_REC);
    static const uint8_t zero[32];
    size_t u = 0, by[3] = {0}, stages[6] = {0};
    Set64 seen = {0};
    for (long long i = 0; recs && i < n; i++) {           // seed records come in priority order: the first id wins
        if (!memcmp(iid[i], zero, 32) || !set_add(&seen, hash8(iid[i]))) continue;
        uint8_t *r = recs + u * PRP_REC;
        const uint8_t *a = ab + (size_t)i * ANS_REC;
        r[0] = a[0] < 6 ? a[0] : 0, r[1] = (a[ANS_REC - 1] & 0x7f) > 2 ? 2 : a[ANS_REC - 1] & 0x7f;
        memcpy(r + 4, iid[i], 32);
        by[r[1]]++, stages[r[0]]++, u++;
    }
    free(seen.v);
    RecHeader h = {{'T', 'O', 'T', 'K', 'S', 'P', 'R', 'P'}, PRP_VERSION, PRP_REC};
    uint8_t *file = recs ? malloc(sizeof h + u * PRP_REC + 1) : NULL;
    int ok = file != NULL;
    if (ok) {
        memcpy(file, &h, sizeof h), memcpy(file + sizeof h, recs, u * PRP_REC);
        ok = plat_file_write_atomic(out, file, sizeof h + u * PRP_REC);
    }
    char msg[640];
    snprintf(msg, sizeof msg, "gfx_shader: prepared %zu shader variants (vertex %zu, fragment %zu, compute %zu; title %zu, "
             "warm-up %zu, other %zu) from %lld seed records, %llu translations failed, in %.0f ms; %s %s", u,
             stages[GFX_STAGE_VS], stages[GFX_STAGE_FS], stages[GFX_STAGE_CS], by[0], by[1], by[2], n,
             (unsigned long long)((uint64_t)n_failures - fail0), (double)plat_now_ms() - t0,
             ok ? "wrote" : "cannot write", out);
    rt_log("%s", msg);
    printf("%s\n", msg);
    free(file), free(recs), free(iid), free(ab);
    return ok ? 0 : 1;
}

// The variants the boot prebuild warms, most likely first: every input id play recorded (shader-answers.bin, priority
// 0), then the prepared ones (shader-prepared.bin, priority 1 + their list priority). One entry per input id.
size_t gfx_shader_warm_list(GfxWarmEnt **out) {
    *out = NULL;
    if (!RC.on) return 0;
    char prp[PATH_CAP + 32];
    snprintf(prp, sizeof prp, "%.*sshader-prepared.bin", (int)(strlen(RC.ans_path) - strlen("shader-answers.bin")),
             RC.ans_path);
    uint8_t *ab = NULL, *pb = NULL;
    size_t an = 0, pn = 0;
    plat_mutex_lock(&rec_io);                      // no append meanwhile
    rec_read(RC.ans_path, "TOTKSANS", ANS_VERSION, ANS_REC, &ab, &an);
    plat_mutex_unlock(&rec_io);
    rec_read(prp, "TOTKSPRP", PRP_VERSION, PRP_REC, &pb, &pn);
    size_t na = an / ANS_REC, np = pn / PRP_REC, n = 0;
    GfxWarmEnt *e = calloc(na + np + 1, sizeof *e);
    Set64 seen = {0};
    for (size_t i = 0; e && i < na; i++) {
        const uint8_t *r = ab + i * ANS_REC;
        if (!set_add(&seen, hash8(r + 36))) continue;
        memcpy(e[n].iid, r + 36, 32), e[n].stage = r[0], e[n].prio = 0, n++;
    }
    for (size_t i = 0; e && i < np; i++) {
        const uint8_t *r = pb + i * PRP_REC;
        if (!set_add(&seen, hash8(r + 4))) continue;
        memcpy(e[n].iid, r + 4, 32), e[n].stage = r[0], e[n].prio = (uint8_t)(1 + r[1]), n++;
    }
    free(seen.v), free(ab), free(pb);
    rt_log("gfx_shader: warm list: %zu variants (%zu met in play, %zu prepared)", n, na, np);
    *out = e;
    return n;
}
