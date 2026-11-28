// Adapted from MacNativeGames games/totk/runtime/romfs.c @ b75be8cf (MIT, Copyright (c) 2026 MacNativeGames
// contributors).
// Read-only game RomFS backed by the player's extracted image.
//
// Input: <romfs>/romfs.bin (rt_config()->romfs), whose six IVFC levels and SHA-256 tools/extract.py verified, and
// romfs-index.bin next to rekarina.exe (REKARINA_ROMFS_INDEX overrides), which tools/romfs_index.py wrote together with
// romfs_config.h. fs_init checks the index's SHA-256 and layout and the image size. The full 17 GB hash runs only with
// REKARINA_ROMFS_VERIFY=full; a proof file (.romfs-verified-v1 next to the index, or in REKARINA_ROMFS_PROOF_DIR) keyed by
// the file's volume, file id, size and times skips it on later starts, REKARINA_ROMFS_VERIFY=force ignores the proof.
//
// Reads are positional (platform.h plat_file_pread), so threads read the image in parallel without a file-pointer
// lock. REKARINA_ROMFS_MMAP=1 maps the image instead.
//
// Overlays: files listed in a directory's manifest.tsv are served in place of the originals. Each line names a RomFS
// path, the overlay's size and SHA-256 and the original's SHA-256 ("-" adds a file the RomFS lacks; it is numbered
// after the RomFS's own files, file_count + k in path order). The directories, in this order: REKARINA_ROMFS_PATCHES and
// REKARINA_ROMFS_MODS; unset or "0" = none. A missing path, a size or hash mismatch or a changed original stops the start.
// An overlay file is opened when the game opens it; its identity (volume, file id, size, last write) is kept from the
// start check, and an open that finds another file there stops the game.
//
// Player mods (mods.c): the files under each enabled mod's RomFS root are overlays too, checked by identity only (no
// hash, no manifest). A path several overlays give is served by one: a player mod higher in the order beats a lower
// one, any player mod beats the REKARINA_ROMFS_MODS manifest, which beats romfs-patches (overlay_contest). A mod path that
// matches a RomFS path only without case is served under the RomFS spelling.
//
// The resource size table: with a player mod enabled, the game's table is rebuilt at start with the tables the mods
// ship merged in and the entries of the mods' files raised to what they need (restbl_overlay), and served from
// <profile>/mods/ in place of the game's or a mod's table.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "cfg.h"
#include "fs.h"
#include "host.h"
#include "input.h"
#include "input_map.h"
#include "mods.h"
#include "prep.h"
#include "report.h"
#include "romfs_config.h"
#include "romfs_path.h"
#include "platform/platform.h"
#include "restbl.h"
#include "romfs_nup.h"
#include "sha256.h"
#include "ui_text.h"

#include <zstd.h>

#define NH 65536                     // RomFS handle slots
#define READ_CHUNK (64u << 20)       // largest single read
#define HASH_CHUNK (8u << 20)
#define ADDED UINT64_MAX             // an added overlay file's index until overlay_init numbers it

typedef struct { uint64_t offset, size; uint32_t name, len; } Entry;
_Static_assert(sizeof(Entry) == 24, "index entry layout");

typedef struct { uint32_t volume, pad; uint64_t file_id, size, write_time, create_time; } Identity;   // no padding
// prio: which overlay serves a path two of them give (the higher); romfs-patches 0, the REKARINA_ROMFS_MODS manifest 1, a
// player mod 2 + its distance from the bottom of the order. mod: its index in mods_at + 1, 0 for any other overlay.
typedef struct { uint64_t index, size; char *path, *name; Identity id; int prio, mod; } Overlay;

static uint8_t *index_bytes;
static size_t index_size;
static uint64_t file_count, strings_offset;
static PlatFile *image;
static const uint8_t *image_map;
static char image_file[2048];
static char index_dir[2048];         // the directory of romfs-index.bin: the proof file's default place
// The names the RomFS is mounted under. One image, several names: OoT mounts it as "content" and its
// online library (NPLN) again as "_npln"; TotK mounts it once.
static char mount_names[8][32];
static int nmounts;
static PlatMutex lock = PLAT_MUTEX_INIT;
static uint64_t handles[NH];         // file index + 1; 0 is free
static uint32_t handle_ovl[NH];      // overlay slot + 1 of an open handle, 0 = original
static PlatFile *handle_file[NH];    // the handle's open overlay file, when handle_ovl is set
static Overlay *overlays;            // sorted by RomFS index after overlay_init
static size_t noverlay, overlay_cap;
static char **added;                 // RomFS paths of the added files, sorted; added[k] is index file_count + k
static size_t nadded;
static int patches_loaded;           // a romfs-patches manifest gave a file (romfs_patches_loaded)
// Rekarina: the RomFS read from the player's .nup container (romfs_nup.c): the index is built from its TOC, `image`
// is the .nup itself, entry offsets are extent offsets in it, and zstd-stored files are inflated on read.
static int nup_mode;
static NupExtra *nup_extra;

static void need(int c, const char *m) {
    if (!c) rt_fatal("RomFS: %s", m);
}

static Entry entry(uint64_t i) {
    Entry e;
    memcpy(&e, index_bytes + 24 + i * 24, 24);
    return e;
}
static const char *entry_name(Entry e) { return (const char *)index_bytes + strings_offset + e.name; }

static double now_s(void) { return (double)plat_now_ns() / 1e9; }

// ---- positional reads
// Bytes read at off (0 at or past the end of the file), -1 on an I/O error.
static int64_t read_at(PlatFile *f, void *buf, uint32_t n, uint64_t off) {
    mem_thaw((uintptr_t)buf, n);   // the kernel's write would fail on a frozen page instead of faulting
    int64_t got = plat_file_pread(f, buf, n, off);
    if (got == -2)   // a read-only page in the buffer: a frozen region missed its thaw
        rt_log("RomFS: read of %u bytes into %p refused: the buffer is not writable", n, buf);
    return got < 0 ? -1 : got;
}

static PlatFile *open_read(const char *path) { return plat_file_open(path); }

static Identity identity_of(PlatFile *f) {
    PlatFileId i;
    need(plat_file_id(f, &i), "file information");
    return (Identity){(uint32_t)i.volume, 0, i.file_id, i.size, i.write_time, i.create_time};
}

// ---- SHA-256 over a file range, streaming
static void hex32(const uint8_t d[32], char out[65]) {
    for (int i = 0; i < 32; i++) snprintf(out + 2 * i, 3, "%02x", d[i]);
}

static void sha_range(PlatFile *f, uint64_t offset, uint64_t size, char hex[65], int progress) {
    Sha256 h;
    sha256_init(&h);
    uint8_t *buf = malloc(HASH_CHUNK);
    need(buf != NULL, "hash buffer");
    double t0 = now_s();
    for (uint64_t done = 0; done < size;) {
        uint32_t want = size - done > HASH_CHUNK ? HASH_CHUNK : (uint32_t)(size - done);
        int64_t n = read_at(f, buf, want, offset + done);
        need(n > 0, "read of hash input");
        sha256_update(&h, buf, (size_t)n);
        done += (uint64_t)n;
        if (progress && (done % (2ull << 30) < (uint64_t)n || done == size))
            rt_log("RomFS: hashed %.1f of %.1f GiB (%.0f MiB/s)", done / 1073741824.0, size / 1073741824.0,
                   done / 1048576.0 / (now_s() - t0 + 1e-9));
    }
    uint8_t d[32];
    sha256_final(&h, d);
    free(buf);
    hex32(d, hex);
}

// ---- full image verification with a proof file
typedef struct { char magic[16], sha256[65]; uint8_t pad[7]; Identity id; } RomfsProof;

static void verify_image(const char *dir) {
    const char *mode = cfg_get("REKARINA_ROMFS_VERIFY");
    if (!mode || (strcmp(mode, "full") && strcmp(mode, "force"))) return;
    const char *pdir = cfg_get("REKARINA_ROMFS_PROOF_DIR");
    if (pdir && *pdir) dir = pdir;
    RomfsProof want = {0}, saved;
    memcpy(want.magic, "TOTKRFS-PROOF-1", 16);
    memcpy(want.sha256, ROMFS_IMAGE_SHA256, sizeof ROMFS_IMAGE_SHA256 < 65 ? sizeof ROMFS_IMAGE_SHA256 : 65);
    want.id = identity_of(image);
    char path[2100];
    need(snprintf(path, sizeof path, "%s/.romfs-verified-v1", dir) < (int)sizeof path, "proof path");
    if (strcmp(mode, "force")) {
        PlatFile *p = open_read(path);
        if (p) {
            int ok = plat_file_pread(p, &saved, sizeof saved, 0) == sizeof saved;
            plat_file_close(p);
            if (ok && !memcmp(&saved, &want, sizeof want)) {
                rt_log("RomFS unchanged since its full SHA-256 check (%s)", path);
                return;
            }
        }
    }
    rt_log("RomFS: verifying the SHA-256 of %s (%llu bytes)", image_file, (unsigned long long)want.id.size);
    char hex[65];
    sha_range(image, 0, want.id.size, hex, 1);
    if (strcmp(hex, ROMFS_IMAGE_SHA256))
        rt_fatal("RomFS image %s has SHA-256 %s, the extraction recorded %s (extract it again)", image_file, hex,
                 ROMFS_IMAGE_SHA256);
    Identity after = identity_of(image);
    need(!memcmp(&after, &want.id, sizeof after), "image changed during verification");
    plat_file_write_atomic(path, &want, sizeof want);
    rt_log("RomFS: image SHA-256 ok");
}

// ---- index lookup
static uint64_t lower(const char *name) {
    uint64_t lo = 0, hi = file_count;
    while (lo < hi) {
        uint64_t m = lo + (hi - lo) / 2;
        if (strcmp(entry_name(entry(m)), name) < 0) lo = m + 1;
        else hi = m;
    }
    return lo;
}

static size_t added_lower(const char *p) {
    size_t lo = 0, hi = nadded;
    while (lo < hi) {
        size_t m = lo + (hi - lo) / 2;
        if (strcmp(added[m], p) < 0) lo = m + 1;
        else hi = m;
    }
    return lo;
}

// The index of RomFS path p (an added file: file_count + k), or UINT64_MAX.
static uint64_t file_index(const char *p) {
    uint64_t i = lower(p);
    if (i < file_count && !strcmp(p, entry_name(entry(i)))) return i;
    size_t k = added_lower(p);
    return k < nadded && !strcmp(added[k], p) ? file_count + k : UINT64_MAX;
}

// 1 when a RomFS file or an added one lies under folder (a path ending in '/').
static int folder_exists(const char *folder) {
    size_t n = strlen(folder);
    uint64_t i = lower(folder);
    if (i < file_count && !strncmp(entry_name(entry(i)), folder, n)) return 1;
    size_t k = added_lower(folder);
    return k < nadded && !strncmp(added[k], folder, n);
}

// The overlay slot + 1 that replaces RomFS file i, or 0.
static uint32_t overlay_slot(uint64_t i) {
    size_t lo = 0, hi = noverlay;
    while (lo < hi) {
        size_t m = lo + (hi - lo) / 2;
        if (overlays[m].index < i) lo = m + 1;
        else hi = m;
    }
    return lo < noverlay && overlays[lo].index == i ? (uint32_t)(lo + 1) : 0;
}

// ---- overlays
// Loads and checks one overlay folder. A failed check stops the game, except with optional set (the Xbox prompts
// folder): the folder is dropped, the reason logged, and 0 returned. Returns 1 when the folder is loaded or has no
// manifest. prio: the Overlay field (0 for the port's own folders, 1 for the REKARINA_ROMFS_MODS manifest).
// The SHA-256 of RomFS file i as the game reads it: its range of the image, or with the RomFS read from a .nup, the
// inflated bytes of a file the container stores as zstd (romfs_nup.c).
static void entry_sha(uint64_t i, Entry e, char hex[65]) {
    if (nup_mode && nup_compressed(i)) {
        uint8_t d[32];
        nup_acquire(i);
        rt_sha256(nup_inflated(image, i, e.offset, e.size), e.size, d);
        nup_release(i);
        hex32(d, hex);
        return;
    }
    sha_range(image, e.offset, e.size, hex, 0);
}

static int overlay_load(const char *dir, int optional, int prio) {
#define BAD(...) do { if (!optional) rt_fatal(__VA_ARGS__); rt_log(__VA_ARGS__); goto bad; } while (0)
    char path[2100];
    need(snprintf(path, sizeof path, "%s/manifest.tsv", dir) < (int)sizeof path, "overlay path");
    FILE *m = fopen(path, "rb");
    if (!m) {
        rt_log("RomFS overlay: none (%s)", path);
        return 1;
    }
    char line[8192];
    size_t first = noverlay;
    uint64_t hashed = 0;
    double t0 = now_s();
    while (fgets(line, sizeof line, m)) {
        if (line[0] == '#' || line[0] == '\n' || line[0] == '\r') continue;
        char name[4096], sha[65], orig[65], hex[65];
        unsigned long long size = 0;
        need(sscanf(line, "%4095[^\t]\t%llu\t%64[0-9a-f]\t%64[0-9a-f-]", name, &size, sha, orig) == 4 &&
             strlen(sha) == 64 && (strlen(orig) == 64 || !strcmp(orig, "-")), "overlay manifest line");
        need(romfs_path_valid_relative(name), "overlay path");
        uint64_t i = lower(name);
        int known = i < file_count && !strcmp(entry_name(entry(i)), name), add = !strcmp(orig, "-");
        if (!known && !add)
            BAD("RomFS overlay: %s/manifest.tsv names %s, which the RomFS lacks; a new file has \"-\" as its "
                "original's SHA-256", dir, name);
        if (known && add)
            BAD("RomFS overlay: %s/manifest.tsv adds %s, which the RomFS already has; give its original's SHA-256",
                dir, name);
        Entry e = {0};
        if (known) {
            e = entry(i);
            entry_sha(i, e, hex);
            if (strcmp(hex, orig)) BAD("RomFS overlay: the original of %s has SHA-256 %s, %s/manifest.tsv says %s",
                                       name, hex, dir, orig);
        } else {
            i = ADDED;
        }
        char file[8200];
        need(snprintf(file, sizeof file, "%s/%s", dir, name) < (int)sizeof file, "overlay file path");
        PlatFile *f = open_read(file);
        if (!f) BAD("RomFS overlay: %s missing", file);
        Identity id = identity_of(f);
        if (id.size != size) {
            plat_file_close(f);
            BAD("RomFS overlay: %s has %llu bytes, its manifest says %llu", file, (unsigned long long)id.size, size);
        }
        sha_range(f, 0, size, hex, 0);
        plat_file_close(f);
        if (strcmp(hex, sha)) BAD("RomFS overlay: %s has SHA-256 %s, its manifest says %s", file, hex, sha);
        hashed += e.size + size;
        if (noverlay == overlay_cap) {
            overlay_cap = overlay_cap ? overlay_cap * 2 : 16;
            overlays = realloc(overlays, overlay_cap * sizeof *overlays);
            need(overlays != NULL, "overlay table");
        }
        char *copy = strdup(file), *rel = strdup(name);
        need(copy && rel, "overlay path copy");
        overlays[noverlay++] = (Overlay){i, size, copy, rel, id, prio, 0};
        if (!prio) patches_loaded = 1;
        if (noverlay - first <= 16)
            rt_log("RomFS overlay: %s (%llu bytes) %s", name, size, add ? "is added" : "replaces the original");
    }
    fclose(m);
    if (noverlay - first > 16) rt_log("RomFS overlay: %zu files from %s (%zu not listed)", noverlay - first, dir,
                                      noverlay - first - 16);
    if (noverlay > first) rt_log("RomFS overlay: %s checked, %zu files, %.1f MiB hashed in %.2f s", dir,
                                 noverlay - first, hashed / 1048576.0, now_s() - t0);
    return 1;
bad:
    fclose(m);
    for (size_t k = first; k < noverlay; k++) {
        free((void *)overlays[k].path);
        free((void *)overlays[k].name);
    }
    noverlay = first;
    return 0;
#undef BAD
}

// ---- player mods (mods.c): every file under an enabled mod's RomFS root, by identity (no hash: a texture pack is
// gigabytes and the player changes it at will; romfs_open still stops on a file that changed after the start)
static uint32_t *folded;   // RomFS indices sorted by name ignoring ASCII case; built on the first mod path that misses

static int folded_order(const void *a, const void *b) {
    return strcasecmp(entry_name(entry(*(const uint32_t *)a)), entry_name(entry(*(const uint32_t *)b)));
}

// The RomFS index of name, or of the RomFS path equal to it but for case (a mod made on Windows, where an emulator's
// folder lookup ignores case); UINT64_MAX for a path the RomFS lacks.
static uint64_t index_any_case(const char *name) {
    uint64_t i = lower(name);
    if (i < file_count && !strcmp(entry_name(entry(i)), name)) return i;
    if (!folded) {
        folded = malloc(file_count * sizeof *folded);
        need(folded != NULL, "case-folded index");
        for (uint64_t k = 0; k < file_count; k++) folded[k] = (uint32_t)k;
        qsort(folded, file_count, sizeof *folded, folded_order);
    }
    uint64_t lo = 0, hi = file_count;
    while (lo < hi) {
        uint64_t m = lo + (hi - lo) / 2;
        if (strcasecmp(entry_name(entry(folded[m])), name) < 0) lo = m + 1;
        else hi = m;
    }
    return lo < file_count && !strcasecmp(entry_name(entry(folded[lo])), name) ? folded[lo] : UINT64_MAX;
}

static void overlay_push(Overlay o) {
    if (noverlay == overlay_cap) {
        overlay_cap = overlay_cap ? overlay_cap * 2 : 16;
        overlays = realloc(overlays, overlay_cap * sizeof *overlays);
        need(overlays != NULL, "overlay table");
    }
    overlays[noverlay++] = o;
}

// 1 when path names a file of one game version only (Product.<n> in a name: the resource size table, the actor
// database and other version-bound files); the RomFS lacking it means it is another version's.
static int other_version(const char *path) {
    for (const char *p = strstr(path, ".Product."); p; p = strstr(p + 1, ".Product.")) {
        const char *d = p + strlen(".Product.");
        if (*d >= '0' && *d <= '9') return 1;
    }
    return 0;
}

// ---- the resource size table's inputs: the tables the overlays carry, before overlay_contest keeps one of them
static char restbl_rel[256];   // the RomFS path of the game's table, "" when it has none
typedef struct {
    char *file;
    int mod, other;   // mod: ModInfo index + 1, 0 for the port's folders; other: made for another game version
} Shipped;
static Shipped *shipped;
static size_t nshipped;

static int ends_with(const char *s, const char *tail) {
    size_t n = strlen(s), k = strlen(tail);
    return n >= k && !strcmp(s + n - k, tail);
}

// 1 when rel is a resource size table of any game version.
static int is_restbl(const char *rel) {
    return !strncmp(rel, "System/Resource/ResourceSizeTable.", 34) && ends_with(rel, ".rsizetable.zs");
}

static void shipped_add(const char *file, int mod, int other) {
    Shipped *grown = realloc(shipped, (nshipped + 1) * sizeof *shipped);
    char *copy = strdup(file);
    need(grown && copy, "resource size table list");
    shipped = grown;
    shipped[nshipped++] = (Shipped){copy, mod, other};
}

// The number of a ResourceSizeTable.Product.<n>... name, -1 without one.
static long restbl_product(const char *rel) {
    const char *p = strstr(rel, ".Product.");
    if (!p || p[9] < '0' || p[9] > '9') return -1;
    return strtol(p + 9, NULL, 10);
}

// The game's table: the System/Resource/ResourceSizeTable.*.rsizetable.zs of the RomFS, the newest when there are
// several. A RomFS read from a .nup holds the base package's table beside the update's (OoT: Product.100 and
// Product.111); the game reads its own version's, the update's.
static void restbl_find(void) {
    long best = -2;
    for (uint64_t i = lower("System/Resource/ResourceSizeTable."); i < file_count; i++) {
        const char *n = entry_name(entry(i));
        if (strncmp(n, "System/Resource/ResourceSizeTable.", 34)) break;
        if (is_restbl(n) && strlen(n) < sizeof restbl_rel && restbl_product(n) > best) {
            snprintf(restbl_rel, sizeof restbl_rel, "%s", n);
            best = restbl_product(n);
        }
    }
}

// The tables the overlays carry for this game version, every one of them (overlay_contest keeps one).
static void restbl_collect(void) {
    for (size_t k = 0; k < noverlay; k++)
        if (restbl_rel[0] && !strcmp(overlays[k].name, restbl_rel)) shipped_add(overlays[k].path, overlays[k].mod, 0);
}

typedef struct { int k, prio; const char *dir, *rel; } ModWalk;

static void mod_walk(int k, int prio, const char *dir, const char *rel);

// One entry of a mod's RomFS folder (mod_walk).
static int mod_walk_cb(const char *leaf, int kind, uint64_t size, void *ctx) {
    (void)size;
    const ModWalk *w = ctx;
    ModInfo *m = mods_at(w->k);
    char file[4200], path[4200];
    if (snprintf(file, sizeof file, "%s/%s", w->dir, leaf) >= (int)sizeof file ||
        snprintf(path, sizeof path, "%s%s%s", w->rel, *w->rel ? "/" : "", leaf) >= (int)sizeof path) {
        rt_log("mods: %s: path too long under %s; left out", m->name, w->dir);
        return 1;
    }
    if (kind == PLAT_FS_DIR) {
        mod_walk(w->k, w->prio, file, path);
        return 1;
    }
    if (kind != PLAT_FS_FILE) return 1;
    if (!romfs_path_valid_relative(path)) {
        rt_log("mods: %s: %s is not a RomFS path; left out", m->name, path);
        return 1;
    }
    uint64_t i = index_any_case(path);
    if (i == UINT64_MAX && other_version(path)) {
        if (is_restbl(path)) shipped_add(file, w->k + 1, 1);   // its entries for this mod's files still apply
        m->other_version++;
        return 1;
    }
    PlatFile *f = open_read(file);
    if (!f) {
        rt_log("mods: %s: cannot open %s; left out", m->name, file);
        return 1;
    }
    Identity id = identity_of(f);
    plat_file_close(f);
    char *copy = strdup(file), *rp = strdup(i == UINT64_MAX ? path : entry_name(entry(i)));
    need(copy && rp, "overlay path copy");
    overlay_push((Overlay){i == UINT64_MAX ? ADDED : i, id.size, copy, rp, id, w->prio, w->k + 1});
    return 1;
}

// The files under dir (a mod's RomFS root; rel: the RomFS path of dir, "" at the root) as overlays of mod k.
static void mod_walk(int k, int prio, const char *dir, const char *rel) {
    ModWalk w = {k, prio, dir, rel};
    plat_fs_list(dir, mod_walk_cb, &w);
}

// What overlays compete on: the RomFS index, an added file's path; the higher prio first.
static int contest_order(const void *a, const void *b) {
    const Overlay *x = a, *y = b;
    if (x->index != y->index) return (x->index > y->index) - (x->index < y->index);
    int c = x->index == ADDED ? strcmp(x->name, y->name) : 0;
    return c ? c : y->prio - x->prio;
}

static const char *owner_name(const Overlay *o) {
    return o->mod ? mods_at(o->mod - 1)->name : o->prio ? "romfs_mods" : "the port's patches";
}

// One overlay per path: the highest prio serves it; the others are dropped and counted on their mod.
static void overlay_contest(void) {
    if (!noverlay) return;
    qsort(overlays, noverlay, sizeof *overlays, contest_order);
    size_t kept = 0;
    for (size_t k = 0; k < noverlay; k++) {
        Overlay *o = &overlays[k], *w = kept ? &overlays[kept - 1] : NULL;
        if (w && w->index == o->index && (o->index != ADDED || !strcmp(w->name, o->name))) {
            if (o->mod) {
                ModInfo *m = mods_at(o->mod - 1);
                m->overridden++;
                if (!m->overridden_by[0]) snprintf(m->overridden_by, sizeof m->overridden_by, "%s", owner_name(w));
            } else if (w->mod) {
                rt_log("RomFS overlay: %s from mod %s replaces the one of %s", o->name, owner_name(w), owner_name(o));
            }
            free(o->path);
            free(o->name);
            continue;
        }
        if (o->mod) {
            if (o->index == ADDED) mods_at(o->mod - 1)->added++;
            else mods_at(o->mod - 1)->replaced++;
        }
        overlays[kept++] = *o;
    }
    noverlay = kept;
}

static void mods_overlay_init(void) {
    mods_scan();
    int n = mods_count();
    for (int k = 0; k < n; k++) {
        const ModInfo *m = mods_at(k);
        // a mod with ExeFS patches waits for the rebuild that applies them: half of a mod is worse than none
        if (m->enabled && (m->kind & MOD_ROMFS) && m->problem == MOD_OK) mod_walk(k, 2 + (n - k), m->romfs, "");
    }
}

static void mods_overlay_log(void) {
    for (int k = 0; k < mods_count(); k++) {
        const ModInfo *m = mods_at(k);
        if (!m->enabled || !(m->kind & MOD_ROMFS) || m->problem != MOD_OK) continue;
        rt_log("mods: %s: %u files served (%u replaced, %u added), %u overridden%s%s, %u for another game version",
               m->name, m->replaced + m->added, m->replaced, m->added, m->overridden, m->overridden ? " by " : "",
               m->overridden_by, m->other_version);
    }
}

static int overlay_order(const void *a, const void *b) {
    uint64_t x = ((const Overlay *)a)->index, y = ((const Overlay *)b)->index;
    return (x > y) - (x < y);
}

static int added_order(const void *a, const void *b) {
    return strcmp((*(Overlay *const *)a)->name, (*(Overlay *const *)b)->name);
}

// Number the added files in path order after the RomFS's own (file_count + k); overlay_contest left one per path.
static void added_init(void) {
    Overlay **list = NULL;
    for (size_t k = 0; k < noverlay; k++)
        if (overlays[k].index == ADDED) {
            list = realloc(list, (nadded + 1) * sizeof *list);
            need(list != NULL, "added files");
            list[nadded++] = &overlays[k];
        }
    if (!nadded) return;
    qsort(list, nadded, sizeof *list, added_order);
    added = malloc(nadded * sizeof *added);
    need(added != NULL, "added files");
    for (size_t k = 0; k < nadded; k++) {
        list[k]->index = file_count + k;
        added[k] = list[k]->name;
    }
    free(list);
    rt_log("RomFS overlay: %zu files added to the RomFS", nadded);
}

static int game_read(uint64_t i, Entry e, void *buf, uint64_t n);

// RomFS file `rel` as the game would read it: from an overlay that replaces it (*base), else from the RomFS. malloc'd;
// NULL when it is missing, unreadable or 16 MiB or more (logged with `keeps`, what the game does without it).
static void *read_served(const char *rel, Overlay **base, size_t *n, const char *keeps) {
    uint64_t i = lower(rel);
    *base = NULL;
    if (i >= file_count || strcmp(entry_name(entry(i)), rel)) {
        rt_log("RomFS: %s missing; %s", rel, keeps);
        return NULL;
    }
    for (size_t k = 0; k < noverlay; k++)
        if (overlays[k].index == i) *base = &overlays[k];
    PlatFile *f = *base ? open_read((*base)->path) : NULL;
    uint64_t size = *base ? (*base)->size : entry(i).size;
    void *in = size && size < (16u << 20) && (f || !*base) ? malloc((size_t)size) : NULL;
    // the game's own file as the game reads it (game_read: a .nup's zstd-stored file inflated)
    int ok = in && (*base ? read_at(f, in, (uint32_t)size, 0) == (int64_t)size : game_read(i, entry(i), in, size));
    if (f) plat_file_close(f);
    if (!ok) {
        free(in);
        rt_log("RomFS: cannot read %s; %s", rel, keeps);
        return NULL;
    }
    *n = (size_t)size;
    return in;
}

// Serves out[0..out_n) (malloc'd, freed here) in place of RomFS file `rel` and of an overlay that replaces it (base):
// written to <profile>/<sub>/<file name of rel>. `what` names the change in the log. 0 when it cannot be written.
static int serve_generated(const char *rel, Overlay *base, void *out, size_t out_n, const char *sub, const char *what,
                           const char *keeps) {
    char dir[2100], file[2200];
    const char *leaf = strrchr(rel, '/') ? strrchr(rel, '/') + 1 : rel;
    need(snprintf(dir, sizeof dir, "%s/%s", rt_config()->profile, sub) < (int)sizeof dir, "generated overlay path");
    need(snprintf(file, sizeof file, "%s/%s", dir, leaf) < (int)sizeof file, "generated overlay path");
    plat_file_mkdir_p(dir);
    // A file that already holds these bytes stays as it is: the entry step's texture walk (--scan-textures, a child
    // process) makes the same tables at its start, and a rewrite changed the file under the game, which
    // checks each overlay's identity at every open (FATAL "missing or changed since the start check").
    uint64_t have_n = 0;
    uint8_t *have = plat_file_size(file) == (int64_t)out_n ? plat_file_read(file, &have_n) : NULL;
    int ok = have && have_n == out_n && !memcmp(have, out, out_n) ? 1 : plat_file_write(file, out, out_n);
    free(have);
    free(out);
    PlatFile *r = ok ? open_read(file) : NULL;
    if (!r) {
        rt_log("RomFS: cannot write %s; %s", file, keeps);
        return 0;
    }
    Identity id = identity_of(r);
    plat_file_close(r);
    char *copy = strdup(file);
    need(copy != NULL, "overlay path copy");
    if (base) {
        free(base->path);
        base->path = copy;
        base->size = out_n;
        base->id = id;
    } else {
        if (noverlay == overlay_cap) {
            overlay_cap = overlay_cap ? overlay_cap * 2 : 16;
            overlays = realloc(overlays, overlay_cap * sizeof *overlays);
            need(overlays != NULL, "overlay table");
        }
        char *name = strdup(rel);
        need(name != NULL, "overlay path copy");
        overlays[noverlay++] = (Overlay){lower(rel), out_n, copy, name, id};
    }
    rt_log("RomFS overlay: %s (%zu bytes, %s) replaces %s", file, out_n, what, base ? "the mod's file" : "the original");
    return 1;
}

// ---- the resource size table (restbl.c)

// The decompressed size of a file's bytes as the game unpacks them: the content size its zstd frame records (.zs;
// the game's files and zstd's own tools write it), four times the bytes for a frame without one, else its size.
static uint64_t content_size(PlatFile *f, uint64_t off, uint64_t size, const char *rel) {
    if (!ends_with(rel, ".zs") || !f) return size;
    uint8_t head[18];
    uint32_t n = size < sizeof head ? (uint32_t)size : (uint32_t)sizeof head;
    if (read_at(f, head, n, off) != n) return size;
    unsigned long long c = ZSTD_getFrameContentSize(head, n);
    return c == ZSTD_CONTENTSIZE_ERROR ? size : c == ZSTD_CONTENTSIZE_UNKNOWN ? size * 4 : c;
}

// The first n bytes of RomFS file i as the game reads them: its range of the image, or with the RomFS read from a .nup,
// the inflated bytes of a file the container stores as zstd (romfs_nup.c). 1 when all n were read.
static int game_read(uint64_t i, Entry e, void *buf, uint64_t n) {
    if (n > e.size) return 0;
    if (nup_mode && nup_compressed(i)) {
        nup_acquire(i);
        const uint8_t *p = nup_inflated(image, i, e.offset, e.size);
        if (p) memcpy(buf, p, (size_t)n);
        nup_release(i);
        return p != NULL;
    }
    return n <= UINT32_MAX && read_at(image, buf, (uint32_t)n, e.offset) == (int64_t)n;
}

// content_size of RomFS file i itself.
static uint64_t game_content_size(uint64_t i, Entry e, const char *rel) {
    if (!ends_with(rel, ".zs")) return e.size;
    uint8_t head[18];
    uint64_t n = e.size < sizeof head ? e.size : sizeof head;
    if (!game_read(i, e, head, n)) return e.size;
    unsigned long long c = ZSTD_getFrameContentSize(head, (size_t)n);
    return c == ZSTD_CONTENTSIZE_ERROR ? e.size : c == ZSTD_CONTENTSIZE_UNKNOWN ? e.size * 4 : c;
}

// A table from its zstd bytes; NULL when they are not one.
static Restbl *restbl_unpack(const void *z, size_t n) {
    unsigned long long raw = ZSTD_getFrameContentSize(z, n);
    if (raw == ZSTD_CONTENTSIZE_UNKNOWN || raw == ZSTD_CONTENTSIZE_ERROR || raw > (64u << 20)) return NULL;
    uint8_t *d = malloc((size_t)raw ? (size_t)raw : 1);
    size_t got = d ? ZSTD_decompress(d, (size_t)raw, z, n) : 0;
    Restbl *t = d && !ZSTD_isError(got) && got == raw ? restbl_parse(d, (size_t)raw) : NULL;
    free(d);
    return t;
}

// With a player mod (or the REKARINA_ROMFS_MODS manifest) serving files: the game's table with every shipped table of this
// version merged in (an entry larger than the game's wins), the entries a mod's table for another version gives for
// that mod's files, and each served mod file's entry raised to what its size needs (restbl_grow for a file the game
// has, restbl_estimate for an added one). Never lowers the game's entries.
static void restbl_overlay(void) {
    int player = 0;
    for (size_t k = 0; k < noverlay; k++) player |= overlays[k].mod || overlays[k].prio == 1;
    if (!player || !restbl_rel[0]) return;
    static const char keeps[] = "mod files that grew may fail to load";
    double t0 = now_s();
    uint64_t gi = lower(restbl_rel);
    Entry ge = entry(gi);
    void *gz = ge.size < (64u << 20) ? malloc((size_t)ge.size) : NULL;
    Restbl *base = NULL, *out = NULL;
    if (gz && game_read(gi, ge, gz, ge.size)) {
        base = restbl_unpack(gz, (size_t)ge.size);
        out = restbl_unpack(gz, (size_t)ge.size);
    }
    free(gz);
    if (!base || !out) {
        rt_log("RomFS: cannot read the resource size table %s; %s", restbl_rel, keeps);
        restbl_free(base);
        restbl_free(out);
        return;
    }
    int tables = 0, merged = 0, raised = 0, estimated = 0;
    for (size_t s = 0; s < nshipped; s++) {
        uint64_t n = 0;
        void *z = plat_file_read(shipped[s].file, &n);
        Restbl *t = z ? restbl_unpack(z, (size_t)n) : NULL;
        free(z);
        if (!t) {
            rt_log("RomFS: %s is not a resource size table; left out", shipped[s].file);
            continue;
        }
        tables++;
        if (!shipped[s].other) {
            merged += restbl_merge(out, base, t);
        } else {   // another version's table: only the entries of the files its mod serves
            for (size_t k = 0; k < noverlay; k++) {
                char key[4200];
                const char *kp = restbl_key(overlays[k].name, key, sizeof key);
                if (overlays[k].mod == shipped[s].mod) merged += restbl_take(out, t, &kp, 1);
            }
        }
        restbl_free(t);
    }
    Overlay *served = NULL;
    for (size_t k = 0; k < noverlay; k++) {
        Overlay *o = &overlays[k];
        if (o->index == gi) served = o;
        if ((!o->mod && o->prio != 1) || o->index == gi) continue;
        char key[4200];
        restbl_key(o->name, key, sizeof key);
        int mc = ends_with(o->name, ".mc");
        PlatFile *f = open_read(o->path);
        uint64_t now = mc ? o->size : content_size(f, 0, o->size, o->name);
        if (f) plat_file_close(f);
        uint32_t entry_size, want;
        if (o->index != ADDED) {
            if (!restbl_get(base, key, &entry_size)) continue;   // the game sizes it without the table (streamed)
            Entry e = entry(o->index);
            uint64_t was = mc ? e.size : game_content_size(o->index, e, o->name);
            want = restbl_grow(key, entry_size, was, now, mc);
        } else {
            want = restbl_estimate(key, mc ? now * 8 : now);   // a mesh codec file unpacks to several times its size
            estimated += want != 0;
        }
        raised += want && restbl_raise(out, key, want);
    }
    restbl_free(base);
    if (!merged && !raised && !served) {
        restbl_free(out);
        rt_log("RomFS: resource size table: the mods' files fit the game's entries (%.0f ms)", (now_s() - t0) * 1000);
        return;
    }
    size_t rn = 0, zn = 0;
    uint8_t *raw = restbl_write(out, &rn);
    restbl_free(out);
    size_t cap = raw ? ZSTD_compressBound(rn) : 0;
    void *z = raw ? malloc(cap) : NULL;
    if (z) zn = ZSTD_compress(z, cap, raw, rn, 3);
    free(raw);
    if (!z || ZSTD_isError(zn)) {
        free(z);
        rt_log("RomFS: cannot pack the merged resource size table; %s", keeps);
        return;
    }
    if (serve_generated(restbl_rel, served, z, zn, "mods", "merged resource sizes", keeps))
        rt_log("RomFS: resource size table: %d tables merged (%d entries), %d entries raised for mod files (%d "
               "added files estimated), %.0f ms", tables, merged, raised, estimated, (now_s() - t0) * 1000);
}

// The HeapDef table a bigger game resolution needs (resolution.c): its texture heap and that heap's parent grow. The
// base is an overlay that replaces the table (a mod's heap sizes stay), else the RomFS original; the grown table is
// written to <profile>/resolution/ and served in the base's place.
static const char *const heapdef_names[] = {"System/HeapSize/Product.Nin_NX_NVN.HeapDef.byml.zs",
                                            "System/HeapSize/Product.Nin_Ounce_NVN2.HeapDef.byml.zs"};
static void heapdef_overlay(void) {
    if (!res_heap_extra_kib()) return;
    static const char keeps[] = "the game keeps its own resolution";
    const char *rel = heapdef_names[0];
    for (size_t k = 0; k < sizeof heapdef_names / sizeof *heapdef_names; k++) {
        uint64_t i = lower(heapdef_names[k]);
        if (i < file_count && !strcmp(entry_name(entry(i)), heapdef_names[k])) {
            rel = heapdef_names[k];
            break;
        }
    }
    Overlay *base;
    size_t n;
    void *in = read_served(rel, &base, &n, keeps), *out = NULL;
    size_t out_n = 0;
    int ok = in && res_heapdef_build(in, n, &out, &out_n);
    free(in);
    if (ok && serve_generated(rel, base, out, out_n, "resolution", "grown heaps", keeps)) res_heapdef_served();
}

// The player camera's pack with its stick speeds multiplied by REKARINA_STICK_SENS (camera_speed.c). The base is an
// overlay that replaces the pack (a mod's camera stays), else the RomFS original; the dictionary is the RomFS's own.
#define CAMPACK "Pack/Actor/PlayerCamera.pack.zs"
static void camera_overlay(void) {
    if (cam_speed_wanted() == 1.0f) return;
    static const char keeps[] = "the camera keeps its own speed";
    Overlay *base, *dic_base;
    size_t n, dn;
    void *in = read_served(CAMPACK, &base, &n, keeps), *out = NULL;
    void *dic = in ? read_served("Pack/ZsDic.pack.zs", &dic_base, &dn, keeps) : NULL;
    size_t out_n = 0;
    int ok = in && dic && cam_pack_build(in, n, dic, dn, &out, &out_n);
    free(in);
    free(dic);
    if (ok && serve_generated(CAMPACK, base, out, out_n, "camera", "camera speed", keeps)) cam_pack_served();
}

#ifdef GAME_OOT
// A mod's Bootup pack with the port's scene scale (scene_pack.c): served from <profile>/mods/ in place of the mod's
// own, which overlay_contest let win over the port's. The port's own pack, or none, is left as it is.
#define BOOTUP "Pack/Bootup.Nin_Ounce_NVN2.pack.zs"
static void scene_overlay(void) {
    uint64_t i = lower(BOOTUP);
    if (i >= file_count || strcmp(entry_name(entry(i)), BOOTUP)) return;
    const Overlay *o = NULL;
    for (size_t k = 0; k < noverlay; k++)
        if (overlays[k].index == i) o = &overlays[k];
    if (!o || (!o->mod && o->prio != 1)) return;
    static const char keeps[] = "the scene keeps the mod pack's resolution, which drops while the GPU looks loaded";
    Overlay *base, *dic_base;
    size_t n, dn;
    void *in = read_served(BOOTUP, &base, &n, keeps), *out = NULL;
    void *dic = in ? read_served("Pack/ZsDic.pack.zs", &dic_base, &dn, keeps) : NULL;
    size_t out_n = 0;
    int set = in && dic ? scene_pack_build(in, n, dic, dn, &out, &out_n) : -1;
    free(in);
    free(dic);
    if (set > 0) serve_generated(BOOTUP, base, out, out_n, "mods", "the port's scene scale on a mod's pack", keeps);
    else if (set < 0) rt_log("RomFS: %s of a mod: %s", BOOTUP, keeps);
}
#endif

// The button prompts' folders (input_prompts_folder) and whether each had a manifest next to romfs-index.bin at start;
// the one the settings asked for at start, the one overlay_init loaded (NULL: the game's own prompts) and why the first
// did not load (ROMFS_PROMPTS_*). The menu compares them with the settings in effect (ui/menu.cpp prompts_note).
// romfs-patches-pc keeps that name, but at a game resolution of another shape than 16:9 its files come from pc_dir,
// romfs-patches-pc-<w>x<h> (overlay_init).
static struct { const char *name; int ready; } prompts_folders[7];
static const char *prompts_wanted, *prompts_loaded;
static int prompts_off;
static char pc_dir[64];

const char *romfs_prompts(void) { return prompts_loaded; }

// ---- The action hint's places (romfs-patches-hint-label, tools/make_oot_prompts.py hint_by_label)
// The hint in the screen's corner is a diamond of the four face buttons where the console has them: A on the right. A
// pad mapped by label (PAD_LAYOUT_XBOX) does the game's A with its bottom button, so that folder holds the hint's
// layout with A at the bottom, B on the right, X on the left and Y at the top (tools/oot_hint_layout.py). The game
// reads the layout once, so the folder is served, or not, for the pad layout known at start: the setting's, and under
// auto the layout of the pad the player used last (hint-pad.txt next to romfs-index.bin, written by
// romfs_hint_pad_layout; no file: by label, the layout of every pad but a Switch one, which the keyboard's hint
// shares). A pad of the other layout opened later gets the right hint at the next start.
static int hint_places = PAD_LAYOUT_NINTENDO;   // the pad layout the hint's layout was loaded for
static int hint_places_known;                   // hint_places_load ran

#ifdef GAME_OOT
static int hint_pad_seen(void) {   // the pad layout an earlier run ended with, -1 when none was written
    char path[2200], word[16] = "";
    if (snprintf(path, sizeof path, "%s/hint-pad.txt", index_dir) >= (int)sizeof path) return -1;
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (!fgets(word, sizeof word, f)) word[0] = 0;
    fclose(f);
    return !strncmp(word, "nintendo", 8) ? PAD_LAYOUT_NINTENDO : !strncmp(word, "xbox", 4) ? PAD_LAYOUT_XBOX : -1;
}

static void hint_places_load(void) {
    int layout = input_pad_layout_pick(cfg_value("REKARINA_PAD_LAYOUT"), hint_pad_seen() == PAD_LAYOUT_NINTENDO);
    char path[2200];
    if (layout == PAD_LAYOUT_XBOX &&
        snprintf(path, sizeof path, "%s/romfs-patches-hint-label/manifest.tsv", index_dir) < (int)sizeof path) {
        // A folder from a setup before 1.2.1 has none: the hint's diamond then keeps the console's places (A on the
        // right) under every prompt set, which reads as the Switch's buttons on a pad whose A is at the bottom.
        if (!plat_file_exists(path)) {
            rt_log("RomFS overlay: the action hint keeps the console's places (A on the right) though the pad is mapped by "
                   "label: romfs-patches-hint-label is missing (run the setup again)");
        } else {
            path[strlen(path) - strlen("/manifest.tsv")] = 0;
            size_t first = noverlay;
            if (overlay_load(path, 1, 0) && noverlay > first) {
                hint_places = PAD_LAYOUT_XBOX;
                rt_log("RomFS overlay: the action hint shows the buttons where a pad mapped by label has them (%s)", path);
            } else {
                rt_log("RomFS overlay: %s failed its check: the action hint keeps the console's places (run the setup again)",
                       path);
            }
        }
    }
    hint_places_known = 1;
}
#endif

// input.c pad_layout_get: the layout of the gamepad in use. Kept for the next start's hint (hint_places_load).
void romfs_hint_pad_layout(int layout) {
    static int told = -1;
    if (!hint_places_known || layout == told || (layout != PAD_LAYOUT_NINTENDO && layout != PAD_LAYOUT_XBOX)) return;
    told = layout;
    char path[2200];
    if (snprintf(path, sizeof path, "%s/hint-pad.txt", index_dir) < (int)sizeof path) {
        FILE *f = fopen(path, "wb");
        if (f) {
            fputs(layout == PAD_LAYOUT_NINTENDO ? "nintendo\n" : "xbox\n", f);
            fclose(f);
        }
    }
    if (layout != hint_places)
        rt_log("RomFS overlay: the gamepad is mapped %s, the action hint was loaded for the other layout: its buttons "
               "move to this pad's places at the next start", layout == PAD_LAYOUT_XBOX ? "by label" : "by position");
}

// ---- Button prompts that change while the game runs (prompts-live.bin, tools/make_oot_prompts.py)
// The button font of every prompt folder holds the glyphs of every set; the folders differ in the font's cmap table
// alone, which sends the button code points to one set's glyphs. prompts-live.bin (next to romfs-index.bin) lists each
// set's table. When the prompts setting changes, a thread searches guest memory for the table of the set in use (the
// game keeps the decoded font in its heap) and writes the new set's table over it: the font engine looks a code point
// up there each time it lays text out, so prompts drawn from then on show the new buttons. Without the file, or for
// the keyboard prompts (a font of their own), the setting waits for the next start as before.
#define LIVE_SETS 8
static struct {
    int tried, n, applied, want, busy;
    int asked;         // the set the last change was made for (romfs_prompts_live), -1 none
    uint32_t size;
    char name[LIVE_SETS][32];
    uint8_t *tbl[LIVE_SETS];
    uint8_t *cur;      // the table the game holds now: a set's, or the keyboard set's with the bound keys on its buttons
    uint8_t *next;     // the table being written (live_main)
    uint64_t at;       // where the game's table was found; 0 not yet
    unsigned seen;
    uint64_t retry_at; // plat_now_ns after which a failed change is tried again (the font was not loaded yet); 0 none
    int tries;
} LV = {.applied = -1, .want = -1, .asked = -1};

// The keyboard prompts follow the bindings (prompts-keys.bin, tools/make_oot_prompts.py write_keys): "RKPK", u32
// version 1, u32 buttons, u32 keys, u32 cell; per button a 16-byte action name, u32 code point and u32 offset of its
// glyph delta in the cmap table (16 bits big-endian, glyph = code point + delta); per key a 16-byte name and u32 glyph
// id; then per key a cell x cell coverage mask (the key as the menu shows it). keys_patch puts the glyph of the first
// key bound to each button's action into a copy of the keyboard set's table.
#define KEYS_BUTTONS 32
#define KEYS_MAX 128
static struct {
    int tried, nb, nk, cell;
    struct { int action; uint32_t cp, at; } b[KEYS_BUTTONS];
    struct { int key; uint32_t gid; } k[KEYS_MAX];
    uint8_t *masks;
} KP;

static void keys_load(void) {
    KP.tried = 1;
    char path[2200];
    snprintf(path, sizeof path, "%s/prompts-keys.bin", index_dir);
    FILE *f = fopen(path, "rb");
    if (!f) return;
    uint32_t h[5];
    int nb = 0, nk = 0;
    if (fread(h, 4, 5, f) == 5 && h[0] == 0x4b504b52u && h[1] == 1 && h[2] <= KEYS_BUTTONS && h[3] && h[3] <= KEYS_MAX &&
        h[4] >= 8 && h[4] <= 256) {
        for (uint32_t i = 0; i < h[2]; i++) {
            char name[17] = {0};
            uint32_t w[2];
            if (fread(name, 1, 16, f) != 16 || fread(w, 4, 2, f) != 2) break;
            KP.b[nb].action = -1;
            for (int a = 0; input_action_name(a); a++)
                if (!strcmp(input_action_name(a), name)) KP.b[nb].action = a;
            KP.b[nb].cp = w[0], KP.b[nb].at = w[1];
            nb++;
        }
        for (uint32_t i = 0; i < h[3]; i++) {
            char name[17] = {0};
            uint32_t gid;
            if (fread(name, 1, 16, f) != 16 || fread(&gid, 4, 1, f) != 1) break;
            KP.k[nk].key = KEY_NONE;
            for (int k = KEY_NONE + 1; k < KEY_COUNT; k++)
                if (input_key_name(k) && !strcmp(input_key_name(k), name)) KP.k[nk].key = k;
            KP.k[nk].gid = gid;
            nk++;
        }
        size_t bytes = (size_t)h[4] * h[4] * h[3];
        if (nb == (int)h[2] && nk == (int)h[3] && (KP.masks = malloc(bytes)) && fread(KP.masks, 1, bytes, f) == bytes)
            KP.nb = nb, KP.nk = nk, KP.cell = (int)h[4];
    }
    fclose(f);
    if (KP.nk) rt_log("RomFS overlay: prompts-keys.bin: the keyboard prompts follow the bindings (%d buttons, %d keys)", KP.nb, KP.nk);
}

static int keys_index(int key) {   // a key's place in the file, -1 when it has no glyph
    for (int i = 0; i < KP.nk; i++)
        if (KP.k[i].key == key && key != KEY_NONE) return i;
    return -1;
}

// tbl: a copy of the keyboard set's table; each button gets the glyph of the key bound to its action.
static void keys_patch(uint8_t *tbl) {
    for (int i = 0; i < KP.nb; i++) {
        int k = KP.b[i].action >= 0 ? keys_index(input_prompt_key(KP.b[i].action)) : -1;
        if (k < 0 || KP.b[i].at + 2 > LV.size) continue;   // no key, or one without a glyph: the default key's picture stays
        uint32_t delta = (KP.k[k].gid - KP.b[i].cp) & 0xffff;
        tbl[KP.b[i].at] = (uint8_t)(delta >> 8), tbl[KP.b[i].at + 1] = (uint8_t)delta;
    }
}

static int live_set(const char *name) {
    for (int i = 0; i < LV.n; i++)
        if (!strcmp(LV.name[i], name)) return i;
    return -1;
}

static const char *live_name(const char *folder) {   // "romfs-patches-xbox" -> "xbox"; no folder: the Switch prompts
    return !folder ? "nintendo" : !strncmp(folder, "romfs-patches-", 14) ? folder + 14 : "";
}

// ---- Texts laid out before a change (runtime/oot_fps.c set promptglyphs)
// The game shapes a text when it lays it out and keeps, per character, the glyph the shaping gave (the text's glyph
// buffer); the text's quads are built from that buffer, and the font's glyph cache keys a picture by character and
// glyph. The swap of the table reaches a text only when the game shapes it again, which it does for a text it
// changes: a text it set once kept the old set's buttons until the next start (with a pad switched on after the
// start, the action hint kept the keyboard's key and the pause screen's Back its SHIFT; the owner's test of
// 2026-11-20). Every set's glyphs are in the one font, and each keeps the advance of the game's own glyph of that
// button (tools/button_icons.py), so a held glyph can be swapped for the new one in place, without a new layout.
// romfs_prompt_glyph says which glyph a button character of such a text is to have; romfs_text_generation, counted
// up at each change made in the running game (0 before the first), tells the host which texts to look at again.
static struct {
    uint16_t now[256];   // U+E000..U+E0FF -> its glyph in the table the game holds now, 0 none
    uint8_t any[8192];   // a bit per glyph: a button glyph of some set's table or the glyph of a key (prompts-keys.bin)
} PG;
static volatile uint32_t text_gen;

// The glyph of each button character U+E000..U+E0FF in a cmap table as prompts-live.bin holds it (the Unicode
// subtable, format 4), 0 where it has none; returns how many it has.
static int cmap_buttons(const uint8_t *t, uint32_t n, uint16_t gid_of[256]) {
    memset(gid_of, 0, 256 * sizeof *gid_of);
    if (n < 4) return 0;
    uint32_t records = (uint32_t)(t[2] << 8 | t[3]), off = 0;
    for (uint32_t r = 0; r < records && 4 + 8 * r + 8 <= n; r++) {
        const uint8_t *q = t + 4 + 8 * r;
        uint32_t plat = (uint32_t)(q[0] << 8 | q[1]), enc = (uint32_t)(q[2] << 8 | q[3]);
        if ((plat == 3 && enc == 1) || plat == 0) off = (uint32_t)q[4] << 24 | (uint32_t)q[5] << 16 | (uint32_t)q[6] << 8 | q[7];
    }
    if (!off || off + 14 > n || (t[off] << 8 | t[off + 1]) != 4) return 0;
    uint32_t segs = (uint32_t)(t[off + 6] << 8 | t[off + 7]) / 2, ends = off + 14, starts = ends + 2 * segs + 2,
             deltas = starts + 2 * segs, ranges = deltas + 2 * segs;
    if (ranges + 2 * segs > n) return 0;
    int count = 0;
    for (uint32_t i = 0; i < segs; i++) {
        uint32_t end = (uint32_t)(t[ends + 2 * i] << 8 | t[ends + 2 * i + 1]),
                 start = (uint32_t)(t[starts + 2 * i] << 8 | t[starts + 2 * i + 1]),
                 delta = (uint32_t)(t[deltas + 2 * i] << 8 | t[deltas + 2 * i + 1]),
                 range = (uint32_t)(t[ranges + 2 * i] << 8 | t[ranges + 2 * i + 1]);
        for (uint32_t cp = start < 0xE000 ? 0xE000 : start; cp <= end && cp <= 0xE0FF; cp++) {
            uint32_t g = cp;
            if (range) {   // the glyph from the glyph array, the delta added to one that is there
                uint32_t at = ranges + 2 * i + range + 2 * (cp - start);
                g = at + 2 <= n ? (uint32_t)(t[at] << 8 | t[at + 1]) : 0;
                if (!g) continue;
            }
            g = (g + delta) & 0xffff;
            if (g) gid_of[cp & 0xFF] = (uint16_t)g, count++;
        }
    }
    return count;
}

static void prompt_glyphs_now(void) {   // after LV.cur changed
    uint16_t g[256];
    if (!LV.cur || !cmap_buttons(LV.cur, LV.size, g)) memset(g, 0, sizeof g);
    for (int k = 0; k < 256; k++) __atomic_store_n(&PG.now[k], g[k], __ATOMIC_RELAXED);
    for (int i = 0; i < KP.nk; i++)   // the keys' glyphs, if prompts-keys.bin was read after prompt_glyphs_load
        if (KP.k[i].gid && KP.k[i].gid <= 0xffff)
            __atomic_or_fetch(&PG.any[KP.k[i].gid >> 3], (uint8_t)(1u << (KP.k[i].gid & 7)), __ATOMIC_RELAXED);
}

static void prompt_glyphs_load(void) {   // live_load
    uint16_t g[256];
    int buttons = 0;
    for (int s = 0; s < LV.n; s++) {
        cmap_buttons(LV.tbl[s], LV.size, g);
        for (int k = 0; k < 256; k++)
            if (g[k]) PG.any[g[k] >> 3] |= (uint8_t)(1u << (g[k] & 7));
    }
    if (!KP.tried) keys_load();
    prompt_glyphs_now();
    for (int i = 0; i < 8192; i++) buttons += __builtin_popcount(PG.any[i]);
    rt_log("RomFS overlay: %d button glyphs of the prompt sets can be swapped in texts laid out before a change", buttons);
}

uint32_t romfs_text_generation(void) { return __atomic_load_n(&text_gen, __ATOMIC_ACQUIRE); }

// The glyph a text's character cp is to be drawn with when it holds `glyph`: for a button character whose glyph is a
// button glyph of another set, or 0 (the text is drawn by character, and the font's cache keeps the picture the
// character had when it was first drawn), the glyph of the set the game shows now; otherwise glyph itself.
uint32_t romfs_prompt_glyph(uint32_t cp, uint32_t glyph) {
    if (cp - 0xE000u > 0xFFu || glyph > 0xffffu) return glyph;
    uint32_t now = __atomic_load_n(&PG.now[cp & 0xFF], __ATOMIC_RELAXED);
    if (!now || now == glyph || (glyph && !(PG.any[glyph >> 3] >> (glyph & 7) & 1))) return glyph;
    return now;
}

static void live_load(void) {
    LV.tried = 1;
    char path[2200];
    snprintf(path, sizeof path, "%s/prompts-live.bin", index_dir);
    FILE *f = fopen(path, "rb");
    if (!f) return;
    uint32_t h[4];
    if (fread(h, 4, 4, f) == 4 && h[0] == 0x4c504b52u && h[1] == 1 && h[2] && h[2] <= LIVE_SETS && h[3] >= 64 && h[3] <= 65536) {
        LV.size = h[3];
        for (uint32_t i = 0; i < h[2]; i++) {
            LV.tbl[i] = malloc(LV.size);
            if (!LV.tbl[i] || fread(LV.name[i], 1, 32, f) != 32 || fread(LV.tbl[i], 1, LV.size, f) != LV.size) break;
            LV.name[i][31] = 0;
            LV.n = (int)i + 1;
        }
    }
    fclose(f);
    LV.applied = live_set(live_name(prompts_loaded));
    if (LV.applied >= 0 && (LV.cur = malloc(LV.size))) memcpy(LV.cur, LV.tbl[LV.applied], LV.size);   // as the folder's font has it
    if (!LV.cur) LV.applied = -1;
    if (LV.n && LV.cur) prompt_glyphs_load();
    if (LV.n) rt_log("RomFS overlay: prompts-live.bin: %d prompt sets can be switched while the game runs (in use: %s)", LV.n,
                     LV.applied >= 0 ? LV.name[LV.applied] : "none of them");
}

// The address of n bytes equal to pat in the game's memory, 0 when there is none. The font tables are 4-byte aligned.
static uint64_t live_find(const uint8_t *pat, uint32_t n) {
    enum { CHUNK = 4 << 20 };
    uint8_t *buf = malloc(CHUNK + 65536);
    if (!buf) return 0;
    uint32_t key;
    memcpy(&key, pat + n - 4, 4);   // the table's last word (a glyph number) is rarer than its first (zeros)
    uint64_t found = 0;
    for (uint64_t a = 0x10000, end; a < RT_ADDRESS_LIMIT && !found; a = end) {
        int st = plat_mem_state(a, RT_ADDRESS_LIMIT, &end);
        if (end <= a) break;
        if (st != PLAT_PROT_RW && st != PLAT_PROT_READ) continue;
        for (uint64_t p = a; p < end && !found; p += CHUNK) {
            uint64_t len = end - p < CHUNK + n ? end - p : CHUNK + n;   // chunks overlap by the table's length
            if (len < n || !plat_mem_read(p, buf, (size_t)len)) continue;
            for (uint64_t i = n - 4; i + 4 <= len; i += 4) {
                uint32_t w;
                memcpy(&w, buf + i, 4);
                if (w == key && !memcmp(buf + i + 4 - n, pat, n)) { found = p + i + 4 - n; break; }
            }
        }
    }
    free(buf);
    return found;
}

// ---- The action hint's button images in the running game (prompts-hint.bin, tools/make_oot_prompts.py write_hint)
// The hint at the bottom of the screen draws its buttons from four images of the common layout archive, which the game
// loads once: changed in the running game, the font's prompts followed the device and these stayed the start's (a
// keyboard key round a pad's letter, player report 2026-11-15). Each set's four images, byte for byte as the game holds
// them, are in prompts-hint.bin; a change of set finds where the shown set's images are in memory and writes the new
// set's over them, and the textures are uploaded again as for any write the CPU makes to them. All four places are
// found before any is written: the sets by position trade A with B and X with Y, so a new image can be another
// letter's old one. The places found are kept and checked before the next change; a place that no longer holds the
// image shown makes the whole search run again. The textures are mirrored ones (vk_texture.c make_fresh uploads them
// again, seen 2026-11-15), yet a hint already on screen keeps its look until it changes: the game draws it once. A
// hint shown after the change has the new images.
// The song screen's six note buttons (OcarinaButtonA/B/X/Y/L/R_00, ASTC images of the same archive) change the same
// way, from prompts-ocarina.bin (tools/make_oot_prompts.py write_hint): one table of each kind, the same code.
#define HINT_SETS 10
#define HINT_IMAGES_MAX 6
#define HINT_PLACES 4
typedef struct {
    const char *file, *what;                         // prompts-hint.bin, "the action hint"
    int tried, n, cur, nimg;                         // cur: the set whose images are in memory, -1 unknown
    uint32_t size;                                   // bytes of one image
    char name[HINT_SETS][32];
    uint8_t *img[HINT_SETS];                         // nimg * size bytes per set
    uint64_t at[HINT_IMAGES_MAX][HINT_PLACES];       // where each image of the set shown was found
    int nat[HINT_IMAGES_MAX];
} HintTable;
static HintTable HT = {.file = "prompts-hint.bin", .what = "the action hint", .cur = -1},
                 OT = {.file = "prompts-ocarina.bin", .what = "the song screen", .cur = -1};

static void hint_load(HintTable *T) {
    T->tried = 1;
    char path[2200];
    snprintf(path, sizeof path, "%s/%s", index_dir, T->file);
    FILE *f = fopen(path, "rb");
    if (!f) {   // a folder whose prompt files are older than the table: the images stay the start's set's
        rt_log("RomFS overlay: no %s next to the game: %s keeps the buttons of the set it started with (run the setup "
               "again)", T->file, T->what);
        return;
    }
    uint32_t h[5];
    if (fread(h, 4, 5, f) == 5 && h[0] == 0x48504b52u && h[1] == 1 && h[2] && h[2] <= HINT_SETS && h[3] &&
        h[3] <= (1u << 20) && h[4] && h[4] <= HINT_IMAGES_MAX) {
        T->size = h[3];
        T->nimg = (int)h[4];
        for (uint32_t i = 0; i < h[2]; i++) {
            T->img[i] = malloc((size_t)T->nimg * T->size);
            if (!T->img[i] || fread(T->name[i], 1, 32, f) != 32 ||
                fread(T->img[i], T->size, (size_t)T->nimg, f) != (size_t)T->nimg) break;
            T->name[i][31] = 0;
            T->n = (int)i + 1;
        }
    }
    fclose(f);
    const char *start = live_name(prompts_loaded);
    for (int i = 0; i < T->n; i++)
        if (!strcmp(T->name[i], start)) T->cur = i;
    if (T->n) rt_log("RomFS overlay: %s: %s's images of %d sets (%s shown)", T->file, T->what, T->n,
                     T->cur >= 0 ? T->name[T->cur] : "none known");
}

// A word of the image to look for, at a multiple of 4: one of four different bytes where there is one (a busy block
// of BC4 indices, rare in memory), else one that is not 0 or all ones (those fill empty blocks).
static uint32_t hint_key_at(const uint8_t *img, uint32_t size) {
    uint32_t any = 0, seen = 0;
    for (uint32_t o = 0; o + 4 <= size; o += 4) {
        const uint8_t *b = img + o;
        if (b[0] != b[1] && b[0] != b[2] && b[0] != b[3] && b[1] != b[2] && b[1] != b[3] && b[2] != b[3]) return o;
        uint32_t w;
        memcpy(&w, b, 4);
        if (!seen && w && w != 0xffffffffu) any = o, seen = 1;
    }
    return any;
}

// The images of the set shown and their key words, for hint_scan.
typedef struct {
    const uint8_t *img[HINT_IMAGES_MAX];
    uint32_t ko[HINT_IMAGES_MAX], key[HINT_IMAGES_MAX];
    uint8_t *buf;
} HintScan;

// Every place in [lo, hi) where an image of the shown set starts, added to T->at (up to HINT_PLACES each).
static void hint_scan(HintTable *T, HintScan *S, uint64_t lo, uint64_t hi) {
    enum { CHUNK = 4 << 20 };
    uint32_t n = T->size;
    for (uint64_t a = lo, end; a < hi; a = end) {
        int st = plat_mem_state(a, hi, &end);
        if (end <= a) break;
        if (st != PLAT_PROT_RW && st != PLAT_PROT_READ) continue;
        for (uint64_t p = a; p < end; p += CHUNK) {
            // chunks overlap by an image's length; an image is taken in the chunk where it starts
            uint64_t len = end - p < CHUNK + n ? end - p : CHUNK + n;
            if (len < n || !plat_mem_read(p, S->buf, (size_t)len)) continue;
            for (uint64_t i = 0; i + 4 <= len; i += 4) {
                uint32_t w;
                memcpy(&w, S->buf + i, 4);
                int any = 0;
                for (int k = 0; k < T->nimg && !any; k++) any = w == S->key[k];
                if (!any) continue;
                for (int k = 0; k < T->nimg; k++) {
                    if (w != S->key[k] || i < S->ko[k] || T->nat[k] >= HINT_PLACES) continue;
                    uint64_t s = i - S->ko[k];
                    if (s < CHUNK && s + n <= len && !memcmp(S->buf + s, S->img[k], n)) T->at[k][T->nat[k]++] = p + s;
                }
            }
        }
    }
}

// Where the shown set's images lie (up to HINT_PLACES each); returns how many of them were found. The game keeps
// them in the graphics heap the font table is in (200 MiB apart in a run of 2026-11-15), so the 1 GiB around `near`
// goes first, and the rest of the address space only when an image is still missing: a pass over all of it took
// seconds.
static int hint_find(HintTable *T, uint64_t near) {
    enum { WINDOW = 512u << 20 };
    HintScan S;
    uint32_t n = T->size;
    for (int k = 0; k < T->nimg; k++) {
        S.img[k] = T->img[T->cur] + (size_t)k * n;
        S.ko[k] = hint_key_at(S.img[k], n);
        memcpy(&S.key[k], S.img[k] + S.ko[k], 4);
        T->nat[k] = 0;
    }
    S.buf = malloc((4 << 20) + n);
    if (!S.buf) return 0;
    uint64_t lo = 0x10000, hi = RT_ADDRESS_LIMIT;
    if (near) {
        lo = near > lo + WINDOW ? (near - WINDOW) & ~0xfffull : lo;
        hi = near + WINDOW < hi ? (near + WINDOW) & ~0xfffull : hi;
    }
    hint_scan(T, &S, lo, hi);
    int found = 0;
    for (int k = 0; k < T->nimg; k++) found += T->nat[k] > 0;
    if (found < T->nimg && (lo > 0x10000 || hi < RT_ADDRESS_LIMIT)) {   // an image straddling lo or hi is missed
        hint_scan(T, &S, 0x10000, lo);
        hint_scan(T, &S, hi, RT_ADDRESS_LIMIT);
        found = 0;
        for (int k = 0; k < T->nimg; k++) found += T->nat[k] > 0;
    }
    free(S.buf);
    return found;
}

// The set `to`'s images (a prompts-live.bin name) where the shown set's are; near: the font table's address, where the
// search starts. Runs on the prompts' worker thread.
static void hint_apply(HintTable *T, const char *to, uint64_t near) {
    if (!T->tried) hint_load(T);
    int t = -1;
    for (int i = 0; i < T->n; i++)
        if (!strcmp(T->name[i], to)) t = i;
    if (t < 0 || T->cur < 0 || t == T->cur) return;
    uint64_t t0 = plat_now_ns();
    uint32_t n = T->size;
    if (!memcmp(T->img[t], T->img[T->cur], (size_t)T->nimg * n)) {   // the same pictures: nothing to write
        T->cur = t;
        return;
    }
    int ok = 1;   // the kept places still hold the shown images
    uint8_t *mem = malloc(n);
    for (int k = 0; k < T->nimg && ok; k++) {
        ok = T->nat[k] > 0;
        for (int j = 0; j < T->nat[k] && ok; j++)
            ok = mem && plat_mem_read(T->at[k][j], mem, n) && !memcmp(mem, T->img[T->cur] + (size_t)k * n, n);
    }
    free(mem);
    int found = ok ? T->nimg : hint_find(T, near);
    if (!found) {
        rt_log("RomFS overlay: %s's images of %s were not found in memory (%.0f ms): they stay until the next start",
               T->what, T->name[T->cur], (plat_now_ns() - t0) / 1e6);
        return;
    }
    // mem_thaw makes a region our write tracking froze writable again; a place in memory that is read-only for another
    // reason (a copy the game mapped read-only) is left as it is and dropped from the kept places, as a write would fault
    int places = 0, skipped = 0;
    for (int k = 0; k < T->nimg; k++) {
        int kept = 0;
        for (int j = 0; j < T->nat[k]; j++) {
            uint64_t a = T->at[k][j], end = 0;
            mem_thaw(a, n);
            int rw = 1;   // the bounded walk: plat_mem_state would measure the whole run around the place
            for (uint64_t x = a; rw && x < a + n; x = end)
                rw = plat_mem_state_walk(x, a + n, &end) == PLAT_PROT_RW && end > x;
            if (!rw) {
                skipped++;
                continue;
            }
            memcpy((void *)(uintptr_t)a, T->img[t] + (size_t)k * n, n);
            T->at[k][kept++] = a;
            places++;
        }
        T->nat[k] = kept;
    }
    rt_log("RomFS overlay: %s's images changed from %s to %s (%d of %d images found, %d places, %d read-only "
           "left, first at %#llx, %.0f ms)",
           T->what, T->name[T->cur], T->name[t], found, T->nimg, places, skipped,
           (unsigned long long)(T->nat[0] ? T->at[0][0] : 0), (plat_now_ns() - t0) / 1e6);
    T->cur = t;
}

static void live_main(void *arg) {
    (void)arg;
    int to = __atomic_load_n(&LV.want, __ATOMIC_ACQUIRE);
    uint64_t t0 = plat_now_ns(), at = LV.at;
    uint8_t *mem = malloc(LV.size);
    if (at && !(mem && plat_mem_read(at, mem, LV.size) && !memcmp(mem, LV.cur, LV.size))) at = 0;   // the font moved
    free(mem);
    if (!at) at = live_find(LV.cur, LV.size);
    if (at) {
        mem_thaw(at, LV.size);
        memcpy((void *)(uintptr_t)at, LV.next, LV.size);
        memcpy(LV.cur, LV.next, LV.size);
        LV.at = at, LV.applied = to, LV.tries = 0, LV.retry_at = 0;
        rt_log("RomFS overlay: button prompts changed to %s in the running game (font table at %#llx, %.0f ms)", LV.name[to],
               (unsigned long long)at, (plat_now_ns() - t0) / 1e6);
        hint_apply(&HT, LV.name[to], at);   // and the action hint's images
        hint_apply(&OT, LV.name[to], at);   // and the song screen's
        // and the texts laid out before: each takes the new buttons when it is next drawn (oot_fps.c set promptglyphs)
        prompt_glyphs_now();
        __atomic_add_fetch(&text_gen, 1, __ATOMIC_ACQ_REL);
    } else {
        // Before the game has loaded its fonts there is nothing to find: a few more tries, some seconds apart, so
        // keys bound in rekarina.ini show without a visit to the menu; then not again until the settings change.
        LV.seen = cfg_version();
        LV.retry_at = ++LV.tries < 12 ? plat_now_ns() + 5000000000ull : 0;
        rt_log("RomFS overlay: the game's button font was not found in memory (%.0f ms): the %s prompts %s",
               (plat_now_ns() - t0) / 1e6, LV.name[to], LV.retry_at ? "are tried again in a moment" : "wait for the next start");
    }
    free(LV.next);
    LV.next = NULL;
    __atomic_store_n(&LV.busy, 0, __ATOMIC_RELEASE);
}

// ---- Pictures of each prompt set's buttons, for the settings menu (prompt-previews.bin, tools/make_oot_prompts.py)
// "RKPV", u32 version 1, u32 sets, u32 cell, u32 buttons, then per set a 32-byte name (the folder's name without
// "romfs-patches-", "nintendo" for the game's own) and `buttons` masks of cell x cell coverage bytes. Read once, when
// the menu first asks; a game folder without the file shows no pictures.
#define PREVIEW_SETS 8
static struct {
    int tried, n, cell, count;
    char name[PREVIEW_SETS][32];
    uint8_t *masks[PREVIEW_SETS];
} PV;

static void preview_load(void) {
    PV.tried = 1;
    char path[2200];
    snprintf(path, sizeof path, "%s/prompt-previews.bin", index_dir);
    FILE *f = fopen(path, "rb");
    if (!f) return;
    uint32_t h[5];
    if (fread(h, 4, 5, f) == 5 && h[0] == 0x56504b52u && h[1] == 1 && h[2] && h[2] <= PREVIEW_SETS && h[3] >= 8 &&
        h[3] <= 256 && h[4] && h[4] <= ROMFS_PREVIEW_BUTTONS) {
        size_t bytes = (size_t)h[3] * h[3] * h[4];
        PV.cell = (int)h[3], PV.count = (int)h[4];
        for (uint32_t i = 0; i < h[2]; i++) {
            PV.masks[i] = malloc(bytes);
            if (!PV.masks[i] || fread(PV.name[i], 1, 32, f) != 32 || fread(PV.masks[i], 1, bytes, f) != bytes) break;
            PV.name[i][31] = 0;
            PV.n = (int)i + 1;
        }
    }
    fclose(f);
    if (PV.n) rt_log("RomFS overlay: prompt-previews.bin: pictures of %d prompt sets for the menu", PV.n);
}

int romfs_prompt_previews(int *cell, int *count) {
    if (!PV.tried) preview_load();
    if (cell) *cell = PV.cell;
    if (count) *count = PV.count;
    return PV.n;
}

const uint8_t *romfs_prompt_preview_masks(int set) { return set >= 0 && set < PV.n ? PV.masks[set] : NULL; }

// The keys of the keyboard set's picture. The masks of every key (cell x cell bytes each, *count of them), NULL when
// the game folder has no prompts-keys.bin; and, for button k of a set's picture (the order of tools/button_icons.py
// PREVIEW_CODES: A, B, X, Y, L, R, ZL, ZR, Minus, Plus, the d-pad), the key bound to it as an index into those masks:
// -1 for a set that is not the keyboard's, a button that is not one key (the d-pad) and a key without a picture.
const uint8_t *romfs_prompt_key_masks(int *cell, int *count) {
    if (!KP.tried) keys_load();
    if (cell) *cell = KP.cell;
    if (count) *count = KP.nk;
    return KP.nk ? KP.masks : NULL;
}

int romfs_prompt_preview_key(int set, int k) {
    static const int actions[] = {B_A, B_B, B_X, B_Y, B_L, B_R, B_ZL, B_ZR, B_MINUS, B_PLUS};
    if (set < 0 || set >= PV.n || strcmp(PV.name[set], "pc") || k < 0 || k >= (int)(sizeof actions / sizeof *actions)) return -1;
    if (!KP.tried) keys_load();
    return keys_index(input_prompt_key(actions[k]));
}

int romfs_prompt_preview_set(const char *folder) {
    if (!PV.tried) preview_load();
    const char *name = live_name(folder);
    for (int i = 0; i < PV.n; i++)
        if (!strcmp(PV.name[i], name)) return i;
    return -1;
}

// 1 when the prompt sets change in the running game (a query: romfs_prompts_live does the work, each present).
int romfs_prompts_can_change(void) { return LV.tried && LV.n && LV.applied >= 0; }

// 1 when the running game shows the prompts of folder (input_prompts_folder) without a restart: the set is the one
// applied, or the one being applied now.
int romfs_prompts_shown(const char *folder) {
    if (!LV.n || LV.applied < 0) return 0;
    int want = live_set(live_name(folder));
    if (want < 0) return 0;
    return want == LV.applied ||
           (__atomic_load_n(&LV.busy, __ATOMIC_ACQUIRE) && want == __atomic_load_n(&LV.want, __ATOMIC_ACQUIRE));
}

// Each present: the prompts setting changed, or under prompts = auto the device in use (input_prompts_wanted)? 1 when
// the sets can change without a restart.
int romfs_prompts_live(void) {
    if (!LV.tried) live_load();
    if (!LV.n || LV.applied < 0) return 0;
    {   // DEBUG-PROMPTSAT: REKARINA_PROMPTS_AT=<frame>:<value>[,<frame>:<value>...] sets the prompts setting at those
        // presents (a test of the switch)
        static uint64_t frame, at[8];
        static char val[8][32];
        static int on = -1;
        if (on < 0) {
            const char *e = cfg_get("REKARINA_PROMPTS_AT");
            on = 0;
            for (int k; e && on < 8 && sscanf(e, "%llu:%31[^,]%n", (unsigned long long *)&at[on], val[on], &k) == 2; on++)
                e = e[k] == ',' ? e + k + 1 : NULL;
        }
        frame++;
        for (int i = 0; i < on; i++)
            if (frame == at[i]) cfg_set("REKARINA_PROMPTS", val[i]);
    }
    unsigned v = cfg_version();
    if (__atomic_load_n(&LV.busy, __ATOMIC_ACQUIRE)) return 1;
    int again = LV.retry_at && plat_now_ns() >= LV.retry_at;
    // Under auto the set changes without a settings change: the device the player used last chose another
    // (input.c prompt_dev, which waits until a new device has kept the newest press for 0.4 s).
    if (v == LV.seen && !again && !input_prompts_auto()) return 1;
    int want = live_set(live_name(input_prompts_wanted()));
    if (v == LV.seen && !again && want == LV.asked) return 1;
    if (v != LV.seen || want != LV.asked) LV.tries = 0;   // the settings or the device changed: a new change, its own tries
    LV.retry_at = 0;
    LV.asked = want;
    uint8_t *next = want >= 0 ? malloc(LV.size) : NULL;
    if (!next) { LV.seen = v; return 1; }
    memcpy(next, LV.tbl[want], LV.size);
    if (!strcmp(LV.name[want], "pc")) {   // the keyboard set: the keys bound now
        if (!KP.tried) keys_load();
        keys_patch(next);
    }
    if (!memcmp(next, LV.cur, LV.size)) {   // the game shows it already
        free(next);
        LV.applied = want, LV.seen = v;
        return 1;
    }
    LV.next = next;
    __atomic_store_n(&LV.want, want, __ATOMIC_RELEASE);
    __atomic_store_n(&LV.busy, 1, __ATOMIC_RELEASE);
    if (!plat_thread_spawn(live_main, NULL, "prompts live", 0)) {
        free(next);
        LV.next = NULL;
        __atomic_store_n(&LV.busy, 0, __ATOMIC_RELEASE), LV.seen = v;
    }
    return 1;
}



int romfs_prompts_ready(const char *folder) {
    for (int i = 0; i < (int)(sizeof prompts_folders / sizeof prompts_folders[0]); i++)
        if (folder && prompts_folders[i].name && !strcmp(prompts_folders[i].name, folder)) return prompts_folders[i].ready;
    return 0;
}

int romfs_prompts_off(const char *folder) {
    return folder && prompts_wanted && !strcmp(folder, prompts_wanted) ? prompts_off : ROMFS_PROMPTS_OK;
}

// The SHA-256 column of `rel` in dir/manifest.tsv, into sha; 0 when the folder does not list it.
static int manifest_sha(const char *dir, const char *rel, char sha[65]) {
    char path[2200], line[8192];
    if (!dir || snprintf(path, sizeof path, "%s/manifest.tsv", dir) >= (int)sizeof path) return 0;
    FILE *m = fopen(path, "rb");
    if (!m) return 0;
    size_t rn = strlen(rel);
    int found = 0;
    while (!found && fgets(line, sizeof line, m)) {
        if (strncmp(line, rel, rn) || line[rn] != '\t') continue;
        const char *s = strchr(line + rn + 1, '\t');   // past the size
        if (s && strlen(s + 1) >= 64) {
            memcpy(sha, s + 1, 64);
            sha[64] = 0;
            found = 1;
        }
    }
    fclose(m);
    return found;
}

// The first RomFS file that the overlays [a, b) (a prompt set's folder, pdir) and [c, d) (the mods) both replace, or
// NULL. Not a clash: the resource size table (restbl_overlay merges both tables), and a file pdir carries with the same
// bytes as fdir, the folder that loads in its place: with or without the prompts the mod's file is served there. Every
// prompt folder's Bootup pack is romfs-patches' own (the scene scale, which scene_overlay keeps on a mod's pack), so a
// model mod that ships Bootup no longer takes the Xbox prompts away (1.2.5 did).
static const char *overlay_clash(size_t a, size_t b, size_t c, size_t d, const char *pdir, const char *fdir) {
    for (size_t i = a; i < b; i++)
        for (size_t j = c; j < d; j++) {
            if (overlays[i].index == ADDED || overlays[i].index != overlays[j].index || !strcmp(overlays[i].name, restbl_rel))
                continue;
            char ps[65], fs[65];
            if (manifest_sha(pdir, overlays[i].name, ps) && manifest_sha(fdir, overlays[i].name, fs) && !strcmp(ps, fs))
                continue;
            return overlays[i].name;
        }
    return NULL;
}

// Drops the overlays [a, b), the ones after them moving down.
static void overlay_drop(size_t a, size_t b) {
    for (size_t k = a; k < b; k++) {
        free((void *)overlays[k].path);
        free((void *)overlays[k].name);
    }
    memmove(overlays + a, overlays + b, (noverlay - b) * sizeof *overlays);
    noverlay -= b - a;
}

static void overlay_init(void) {
    voice_lang_init();   // read at start, as the overlays are (voice_lang.c)
    // REKARINA_ROMFS_PATCHES unset: romfs-patches next to romfs-index.bin if it has a manifest (tools/make_romfs_patches.py).
    // A game resolution that is not 16:9 takes romfs-patches-<w>x<h> instead when it is there: the same overlay with
    // the UI laid out for that shape (make_romfs_patches.py --aspect), when the game version has the camera aspect patch
    // the layout goes with (res_wants_wide_overlay).
    // With prompts = xbox, the Xbox prompts' folder for the pad layout in its place (input_prompts_folder): the same
    // patches with Xbox button icons (make_romfs_patches.py --prompts). The game reads the archives once at boot, so
    // the prompts stay until restart even though the layout itself changes live. Each folder replaces romfs-patches as
    // a whole and the setup builds no folder of another shape with Xbox prompts, so a game resolution that is not 16:9
    // keeps the Switch prompts. So does a RomFS mod (the REKARINA_ROMFS_MODS manifest or a player mod) that replaces one of
    // the prompts' fonts: romfs-patches loads in their place, so the mod's file is not served beside prompts files made
    // for the game's own. A mod's resource size table is no clash: restbl_overlay merges it with the prompts' one.
    // prompts = pc takes romfs-patches-pc whatever the pad layout, and at a game resolution of another shape
    // romfs-patches-pc-<w>x<h> (make_romfs_patches.py --prompts pc --aspect): it carries that shape's UI layout too, so
    // it replaces romfs-patches-<w>x<h> as romfs-patches-pc replaces romfs-patches.
    char def[2100], path[2100];
    const char *patches = cfg_get("REKARINA_ROMFS_PATCHES"), *mods = cfg_get("REKARINA_ROMFS_MODS");
    int wide = 0;
    unsigned out[2];
    res_wanted(out);
    if (!patches && res_wants_wide_overlay(out)) {
        if (snprintf(def, sizeof def, "%s/romfs-patches-%ux%u/manifest.tsv", index_dir, out[0], out[1]) < (int)sizeof def &&
            plat_file_exists(def)) {
            def[strlen(def) - strlen("/manifest.tsv")] = 0;
            patches = def, wide = 1;
        } else {
            rt_log("RomFS overlay: no romfs-patches-%ux%u for this game resolution, the UI keeps its 16:9 layout "
                   "(tools/make_romfs_patches.py --aspect %ux%u)", out[0], out[1], out[0], out[1]);
        }
    }
    static const char *const layouts[2] = {"nintendo", "xbox"};   // one of each Xbox folder, then the PC one
    for (int i = 0; i < 3; i++) {
        const char *name = i < 2 ? input_prompts_folder(PROMPTS_XBOX, layouts[i]) : input_prompts_folder(PROMPTS_PC, 0);
        const char *dir = name;
        if (i == 2) {
            if (wide) snprintf(pc_dir, sizeof pc_dir, "%s-%ux%u", name, out[0], out[1]);
            else snprintf(pc_dir, sizeof pc_dir, "%s", name);
            dir = pc_dir;
        }
        prompts_folders[i].name = name;
        prompts_folders[i].ready =
            snprintf(path, sizeof path, "%s/%s/manifest.tsv", index_dir, dir) < (int)sizeof path && plat_file_exists(path);
    }
    for (int i = 0; i < 4; i++) {   // the two PlayStation folders, then the two of Rekarina's own set
        const char *name = input_prompts_folder(i < 2 ? PROMPTS_PS : PROMPTS_RK, layouts[i & 1]);
        prompts_folders[3 + i].name = name;
        prompts_folders[3 + i].ready =
            snprintf(path, sizeof path, "%s/%s/manifest.tsv", index_dir, name) < (int)sizeof path && plat_file_exists(path);
    }
    // prompts = auto: before input has started this is the Xbox guess (input_prompts_folder); the set of the device the
    // player uses replaces it in the running game (romfs_prompts_live) where prompts-live.bin is there.
    prompts_wanted = input_prompts_wanted();
    int pick = input_prompts_of_folder(prompts_wanted), shaped = wide && pick == PROMPTS_PC;
    const char *style = pick == PROMPTS_PC ? "keyboard and mouse" : pick == PROMPTS_PS ? "PlayStation"
                        : pick == PROMPTS_RK ? "Rekarina" : "Xbox";
    prompts_off = !prompts_wanted                         ? ROMFS_PROMPTS_OK
                  : wide && !shaped                       ? ROMFS_PROMPTS_SHAPE
                  : patches && !wide                      ? ROMFS_PROMPTS_PATCHES
                  : !romfs_prompts_ready(prompts_wanted)  ? (shaped ? ROMFS_PROMPTS_SHAPE : ROMFS_PROMPTS_MISSING)
                                                          : ROMFS_PROMPTS_OK;
    restbl_find();
    size_t first = noverlay;
    char prompts_dir[2100] = "";   // the prompt set's folder, for overlay_clash
    if (prompts_wanted && prompts_off == ROMFS_PROMPTS_OK) {
        // fits: its manifest path did
        snprintf(path, sizeof path, "%s/%s", index_dir, shaped ? pc_dir : prompts_wanted);
        if (overlay_load(path, 1, 0)) {
            prompts_loaded = prompts_wanted;
            snprintf(prompts_dir, sizeof prompts_dir, "%s", path);
            rt_log("RomFS overlay: %s button prompts (%s)", style, path);
            // By label (pad_layout xbox, or auto for any pad but a Switch one) the Switch's A is the button labelled A,
            // so the Xbox and Rekarina sets keep the game's own A B X Y glyphs: only the shoulders, triggers, Menu, View
            // and the d-pad look different. Said once, as "I picked Xbox and still see the Switch's" starts there.
            if (strstr(prompts_wanted, "-label") && (pick == PROMPTS_XBOX || pick == PROMPTS_RK))
                rt_log("RomFS overlay: by label the face buttons keep the game's letters A B X Y (each names the button "
                       "labelled so); LB RB LT RT, Menu, View and the d-pad are drawn as the %s set's", style);
            if (shaped) patches = NULL;   // in place of romfs-patches-<w>x<h>, whose UI layout it carries
        } else {
            prompts_off = ROMFS_PROMPTS_FAILED;
            rt_log("RomFS overlay: %s failed its check (run the setup again); romfs-patches loads instead", path);
        }
    } else if (prompts_wanted) {
        rt_log("RomFS overlay: prompts = %s, but %s; Switch prompts stay", pick == PROMPTS_PC ? "pc" : pick == PROMPTS_PS ? "playstation" : pick == PROMPTS_RK ? "rekarina" : "xbox",
               prompts_off == ROMFS_PROMPTS_SHAPE     ? (shaped ? "the game resolution has no romfs-patches-pc-<w>x<h> "
                                                                  "(tools/make_romfs_patches.py --prompts pc --aspect)"
                                                             : "the game resolution takes its own UI layout")
               : prompts_off == ROMFS_PROMPTS_PATCHES ? "REKARINA_ROMFS_PATCHES is set"
                                                      : "their folder has no manifest.tsv (run the setup again)");
        if (prompts_off == ROMFS_PROMPTS_MISSING)
            rt_log("RomFS overlay: the missing folder is %s/%s", index_dir, prompts_wanted);
    }
    size_t prompts_end = noverlay;
    char plain[2100];   // romfs-patches, when it has a manifest
    int plain_ok = snprintf(plain, sizeof plain, "%s/romfs-patches/manifest.tsv", index_dir) < (int)sizeof plain &&
                   plat_file_exists(plain);
    if (plain_ok) plain[strlen(plain) - strlen("/manifest.tsv")] = 0;
    if (!patches && !prompts_loaded && plain_ok) patches = plain;
    if (patches && *patches && strcmp(patches, "0")) overlay_load(patches, 0, 0);
#ifdef GAME_OOT
    if (cfg_bool("REKARINA_FAST_TEXT")) {   // the messages without their pauses (tools/make_oot_prompts.py fast_text)
        snprintf(path, sizeof path, "%s/romfs-patches-fasttext", index_dir);
        if (overlay_load(path, 1, 0)) rt_log("RomFS overlay: faster dialogue, the messages without their pauses (%s)", path);
    }
    hint_places_load();
#endif
    size_t mods_first = noverlay;
    if (mods && *mods && strcmp(mods, "0")) overlay_load(mods, 0, 1);
    mods_overlay_init();
    const char *clash = prompts_loaded ? overlay_clash(first, prompts_end, mods_first, noverlay, prompts_dir,
                                                       shaped ? def : plain_ok ? plain : NULL) : NULL;
    if (clash) {
        rt_log("RomFS overlay: a RomFS mod replaces %s, which the %s prompts change too; Switch prompts stay", clash,
               style);
        overlay_drop(first, prompts_end);
        prompts_loaded = NULL;
        prompts_off = ROMFS_PROMPTS_MODS;
        if (shaped) overlay_load(def, 0, 0);
        else if (plain_ok) overlay_load(plain, 0, 0);
    }
    restbl_collect();
    overlay_contest();
    mods_overlay_log();
    heapdef_overlay();
#ifdef GAME_OOT
    scene_overlay();
#endif
    // A mod's resource size table is merged with the game's and the port's (restbl_overlay), for both games: served
    // as it came, a mod's table built from the game's own gave the port's larger button font (romfs-patches
    // Font2/Scalable/External, 117 KB against the game's entry of 14.5 KB) too small a buffer, the font did not load,
    // and the game stopped at its start on a null font (1.2.5, a model mod that ships the table).
    restbl_overlay();
#ifndef GAME_NS2   // TotK's generated camera pack; it reads the image raw
    camera_overlay();
#endif
    added_init();
    if (noverlay) qsort(overlays, noverlay, sizeof *overlays, overlay_order);
    for (size_t k = 1; k < noverlay; k++)   // overlay_contest left one per file; the generated ones replace in place
        if (overlays[k].index == overlays[k - 1].index)
            rt_fatal("RomFS overlay: %s and %s replace the same file %s", overlays[k - 1].path, overlays[k].path,
                     overlays[k].name);
}

// ---- start
// romfs-index.bin next to rekarina.exe, or REKARINA_ROMFS_INDEX.
static void index_path(char *out, size_t cap) {
    const char *e = cfg_get("REKARINA_ROMFS_INDEX");
    if (e && *e) {
        need(snprintf(out, cap, "%s", e) < (int)cap, "index path");
        return;
    }
    need(plat_exe_path(out, cap), "executable path");
    char *sep = plat_last_sep(out);
    size_t k = sep ? (size_t)(sep - out) + 1 : 0;
    need(k + 16 < cap, "executable path");
    strcpy(out + k, "romfs-index.bin");
}

static void load_index(void) {
    char path[2100];
    index_path(path, sizeof path);
    snprintf(index_dir, sizeof index_dir, "%s", path);
    char *sep = plat_last_sep(index_dir);
    if (sep) *sep = 0;
    else strcpy(index_dir, ".");
    uint64_t sz = 0;
    index_bytes = plat_file_read(path, &sz);
    if (!index_bytes) rt_fatal("RomFS index %s missing (tools/build_game.py writes it next to rekarina.exe)", path);
    need(sz >= 24 && sz < (128 << 20), "index size");
    index_size = (size_t)sz;
    uint8_t d[32];
    char hex[65];
    rt_sha256(index_bytes, index_size, d);
    hex32(d, hex);
    if (strcmp(hex, ROMFS_INDEX_SHA256))
        rt_fatal("RomFS index %s has SHA-256 %s, this build expects %s (rebuild with tools/build_game.py)", path, hex,
                 ROMFS_INDEX_SHA256);
    need(!memcmp(index_bytes, "TOTKRFS1", 8), "index format");
    memcpy(&file_count, index_bytes + 8, 8);
    memcpy(&strings_offset, index_bytes + 16, 8);
    need(file_count == ROMFS_FILE_COUNT && file_count < 1000000 && strings_offset == 24 + file_count * 24 &&
         strings_offset < index_size, "index layout");
    const char *previous = NULL;
    for (uint64_t i = 0; i < file_count; i++) {
        Entry e = entry(i);
        need(e.offset <= ROMFS_IMAGE_SIZE && e.size <= ROMFS_IMAGE_SIZE - e.offset, "file bounds");
        need(e.name < index_size - strings_offset && e.len > 1 && e.len <= index_size - strings_offset - e.name,
             "name bounds");
        const char *name = entry_name(e);
        need(name[e.len - 1] == 0 && strlen(name) == e.len - 1 && (!previous || strcmp(previous, name) < 0),
             "sorted paths");
        previous = name;
    }
    rt_log("RomFS index %s: SHA-256 ok, %llu files, %zu bytes", path, (unsigned long long)file_count, index_size);
}

static int ends_with_nup(const char *p) {
    size_t n = p ? strlen(p) : 0;
    return n > 4 && !strcasecmp(p + n - 4, ".nup");
}

// The dump's container cannot be opened: moved, renamed, or kept online only by OneDrive (two players of 1.2.5: the
// game ended at its start, with this FATAL line in the log and the crash box that asks for a report). The box names
// the file and what to do: the launcher asks where the file is now and points rekarina.ini to it (installer/gui_oot.py
// start_game, installer/oot.py set_ini_dump). Then the game ends; the FATAL line stays for the log readers.
static void dump_missing(const char *path, int os_error) {
    static char text[4096];
    snprintf(text, sizeof text, menu_text(UI_ROMFS_MISSING), path);
    report_fatal_box(text);
    rt_fatal("RomFS: the game container %s is missing or cannot be opened (OS error %d; rekarina.ini [paths] romfs, "
             "REKARINA_ROMFS or --romfs names the .nup)", path, os_error);
}

static void open_image_once(void *param) {
    (void)param;
    const char *path = rt_config()->romfs;
    if (ends_with_nup(path)) {
        need(snprintf(image_file, sizeof image_file, "%s", path) < (int)sizeof image_file, "image path");
        image = open_read(image_file);
        if (!image) dump_missing(image_file, plat_last_os_error());
        return;
    }
#ifdef ROMFS_MISSING
    rt_fatal("this build has no RomFS: tools/build_game.py found no index.json in its --romfs directory");
#else
    need(snprintf(image_file, sizeof image_file, "%s/romfs.bin", path) < (int)sizeof image_file, "image path");
    image = open_read(image_file);
    if (!image)
        rt_fatal("RomFS image %s is missing or cannot be opened (OS error %d); put romfs.bin in the romfs folder next "
                 "to " RT_FILE_STEM ".exe or set rekarina.ini [paths] romfs, REKARINA_ROMFS or --romfs",
                 image_file, plat_last_os_error());
#endif
}

void romfs_open_image(void) {
    static PlatOnce once = PLAT_ONCE_INIT;
    plat_once(&once, open_image_once, NULL);
}

// The RomFS from a .nup file (REKARINA_ROMFS / --romfs naming the container itself).
static void init_nup(void) {
    double t0 = now_s();
    char path[2100];
    index_path(path, sizeof path);   // overlays (romfs-patches, prompts) still live next to the exe
    snprintf(index_dir, sizeof index_dir, "%s", path);
    char *sep = plat_last_sep(index_dir);
    if (sep) *sep = 0;
    else strcpy(index_dir, ".");
    index_bytes = nup_index(image, image_file, &index_size, &file_count, &strings_offset, &nup_extra);
    nup_mode = 1;
    overlay_init();
    rt_log("RomFS ready from %s: %llu files, %zu overlays, %zu added, %.2f s", image_file,
           (unsigned long long)file_count, noverlay, nadded, now_s() - t0);
}

static void init_once(void *param) {
    (void)param;
    romfs_open_image();
    if (ends_with_nup(image_file)) {
        init_nup();
        return;
    }
#ifndef ROMFS_MISSING
    double t0 = now_s();
    load_index();
    PlatFileId sz;
    need(plat_file_id(image, &sz), "image size");
    if (sz.size != ROMFS_IMAGE_SIZE)
        rt_fatal("RomFS image %s has %llu bytes, the index was built for %llu", image_file,
                 (unsigned long long)sz.size, (unsigned long long)ROMFS_IMAGE_SIZE);
    rt_log("RomFS image %s: size ok (%llu bytes)", image_file, (unsigned long long)ROMFS_IMAGE_SIZE);
    verify_image(index_dir);
    const char *mm = cfg_get("REKARINA_ROMFS_MMAP");
    if (mm && strcmp(mm, "0")) {
        const void *v = plat_file_map(image);
        if (v) {
            image_map = v;
            rt_log("RomFS: mapped read path enabled");
        } else {
            rt_log("RomFS: mapping failed; using positional reads");
        }
    }
    overlay_init();
    rt_log("RomFS ready: %llu files, %zu overlays, %zu added, %.2f s", (unsigned long long)file_count, noverlay,
           nadded, now_s() - t0);
#endif
    return;
}

void fs_init(void) {
    static PlatOnce once = PLAT_ONCE_INIT;
    plat_once(&once, init_once, NULL);
}

// ---- the RomFS interface (fs.h)
void fs_trace(const char *what, const char *path) {
    static double t0;
    if (!(rt_trace & RT_TRACE_FS)) return;
    if (!t0) t0 = now_s();
    rt_log("fs %8.3f %s %s", now_s() - t0, what, path ? path : "(null)");
}

uint64_t romfs_file_count(void) { return file_count; }

const char *romfs_entry(uint64_t i, uint64_t *offset, uint64_t *size) {
    if (i >= file_count) return NULL;
    Entry e = entry(i);
    if (offset) *offset = e.offset;
    if (size) *size = e.size;
    return entry_name(e);
}

const char *romfs_image_path(void) { return image_file; }

// The whole of RomFS file i as the game reads it (game_read: its range of the image, or a .nup's inflated bytes),
// malloc'd, for the runtime's own walks of the index (tex_prepare.c); NULL when it cannot be read. Not for the guest.
uint8_t *romfs_entry_read(uint64_t i, uint64_t *size) {
    if (i >= file_count) return NULL;
    Entry e = entry(i);
    uint8_t *buf = malloc(e.size ? (size_t)e.size : 1);
    if (!buf || !game_read(i, e, buf, e.size)) {
        free(buf);
        return NULL;
    }
    *size = e.size;
    return buf;
}

// The image's identity for a cache of what was made from it: its size and last write time (0: not known).
int romfs_image_identity(uint64_t *size, uint64_t *write_time) {
    PlatFileId id;
    if (!image || !plat_file_id(image, &id)) return 0;
    *size = id.size, *write_time = id.write_time;
    return 1;
}

size_t romfs_overlay_count(void) { return noverlay; }

int romfs_patches_loaded(void) { return patches_loaded; }

const char *romfs_overlay(size_t k, const char **file, uint64_t *size, int *is_added) {
    if (k >= noverlay) return NULL;
    *file = overlays[k].path;
    *size = overlays[k].size;
    *is_added = overlays[k].index >= file_count;
    return overlays[k].name;
}

// The mount name a path starts with ("name:"), or "" when none of the RomFS's.
static const char *mount_of(const char *path) {
    for (int i = 0; path && i < __atomic_load_n(&nmounts, __ATOMIC_ACQUIRE); i++) {
        size_t n = strlen(mount_names[i]);
        if (n && !strncmp(path, mount_names[i], n) && path[n] == ':') return mount_names[i];
    }
    return "";
}

int romfs_owns(const char *path) { return *mount_of(path) != 0; }

int romfs_handle(uint64_t h) { return h > ROMFS_HANDLE_BASE && h < ROMFS_HANDLE_BASE + NH; }

uint32_t romfs_mount(const char *name) {
    fs_init();
    if (!name || !romfs_path_valid_relative(name) || strlen(name) >= sizeof mount_names[0] || strchr(name, '/') ||
        strchr(name, ':'))
        return FS_R_INVALID;
    plat_mutex_lock(&lock);
    int have = 0;
    for (int i = 0; i < nmounts; i++) have |= !strcmp(mount_names[i], name);
    if (!have) {
        if (nmounts == (int)(sizeof mount_names / sizeof *mount_names))
            rt_fatal("RomFS mounted under too many names (%s)", name);
        strcpy(mount_names[nmounts], name);
        __atomic_store_n(&nmounts, nmounts + 1, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&lock);
    rt_log("fs mount %s: -> RomFS", name);
    return 0;
}

uint32_t romfs_open(uint64_t *out, const char *path, int mode) {
    fs_trace("rom", path);
    prep_map_open(path);   // the in-play build ledger names the map (prep.c)
    if (!out || mode != FS_OPEN_READ) return FS_R_UNSUPPORTED_OPEN;
    char rel[4096];
    const char *p = romfs_path_relative(mount_of(path), path, rel, sizeof rel);
    if (!p) return FS_R_PATH_NOT_FOUND;
    uint64_t i = file_index(p);
    if (i == UINT64_MAX) return FS_R_PATH_NOT_FOUND;
    uint32_t ov = overlay_slot(i);
    PlatFile *f = NULL;
    if (ov) {
        const Overlay *o = &overlays[ov - 1];
        f = open_read(o->path);
        Identity id;
        if (f) id = identity_of(f);
        if (!f || memcmp(&id, &o->id, sizeof id))
            rt_fatal("RomFS overlay %s is missing or changed since the start check (restart the game)", o->path);
    }
    plat_mutex_lock(&lock);
    for (unsigned j = 1; j < NH; j++)
        if (!handles[j]) {
            handles[j] = i + 1;
            handle_ovl[j] = ov;
            handle_file[j] = f;
            *out = ROMFS_HANDLE_BASE + j;
            plat_mutex_unlock(&lock);
            if (nup_mode && !ov && i < file_count) nup_acquire(i);
            return 0;
        }
    plat_mutex_unlock(&lock);
    rt_fatal("RomFS file handle table full (%d)", NH);
}

// The handle's index row (an added file's is empty: its bytes are all in its overlay), and where its bytes come from:
// the overlay file, or the original's range of the image.
static int handle_source(uint64_t h, PlatFile **f, uint64_t *base, uint64_t *size) {
    if (!romfs_handle(h)) return 0;
    uint64_t j = h - ROMFS_HANDLE_BASE;
    plat_mutex_lock_shared(&lock);
    uint64_t i = handles[j];
    uint32_t ov = handle_ovl[j];
    PlatFile *of = handle_file[j];
    plat_mutex_unlock_shared(&lock);
    if (!i) return 0;
    if (ov) {
        *f = of;
        *base = 0;
        *size = overlays[ov - 1].size;
    } else {
        Entry e = entry(i - 1);
        *f = image;
        *base = e.offset;
        *size = e.size;
    }
    return 1;
}

// DEBUG-FSSTAT: what the game's file reads cost: a line every 2,000 reads with the totals since the last one, the
// reads of 2 ms and more among them and the longest; inflated: the read was served from a zstd-stored file of the .nup.
static void fs_stat(uint64_t t0, uint64_t bytes, int inflated) {
    static uint64_t n, total, ns, slow, slow_ns, worst, infl;
    uint64_t d = plat_now_ns() - t0;
    plat_mutex_lock(&lock);
    n++, total += bytes, ns += d, infl += (uint64_t)inflated;
    if (d >= 2000000) slow++, slow_ns += d;
    if (d > worst) worst = d;
    if (!(n % 2000)) {
        rt_log("FSSTAT %llu reads, %.1f MiB, %.1f ms in all (%.1f us a read); %llu of 2 ms and more took %.1f ms, the "
               "longest %.1f ms; %llu from inflated files", (unsigned long long)n, total / 1048576.0, ns / 1e6,
               ns / 1e3 / (double)n, (unsigned long long)slow, slow_ns / 1e6, worst / 1e6, (unsigned long long)infl);
        n = total = ns = slow = slow_ns = worst = infl = 0;
    }
    plat_mutex_unlock(&lock);
}

// Direct save loading (--load-save, OoT 1.1.1): skips the intro and title menus once on boot,
// then restores the title graph root once world time starts so returning to the title still works.
static int direct_save = -1;
static unsigned direct_graphs;
static uint64_t title_root;

void rt_load_save_start(int slot) { __atomic_store_n(&direct_save, slot, __ATOMIC_RELAXED); }

uint16_t rt_load_save_root(uint64_t addr) {
    uint16_t root = LD16(addr);
    if (root != 42 || addr != __atomic_load_n(&title_root, __ATOMIC_ACQUIRE) || frame_step_world_us() < 0) return root;
    ST16(addr, 35);
    __atomic_store_n(&title_root, 0, __ATOMIC_RELEASE);
    return 35;
}

uint32_t rt_load_save_index(uint32_t index) {
    if (__atomic_load_n(&direct_graphs, __ATOMIC_ACQUIRE) != 3) return index;
    int slot = __atomic_exchange_n(&direct_save, -1, __ATOMIC_RELAXED);
    if (slot < 0) return index;
    rt_log("load-save: requesting slot_%02d", slot);
    return (uint32_t)slot;
}

static void direct_boot_graph(const char *name, int overlay, uint64_t size, uint64_t off, uint8_t *data, uint64_t n) {
    if (!strcmp(name, "Sequence/NormalTitle.module.ainb")) {
        if (off != 0 || n != size || (__atomic_load_n(&direct_graphs, __ATOMIC_ACQUIRE) & 2)) return;
        if (overlay || size != 23468 || memcmp(data, "AIB ", 4) || data[0x88] != 35 || data[0x89] != 0)
            rt_fatal("--load-save: unexpected startup graph %s", name);
        __atomic_store_n(&title_root, (uintptr_t)(data + 0x88), __ATOMIC_RELEASE);
        data[0x88] = 42;
        __atomic_fetch_or(&direct_graphs, 2, __ATOMIC_RELEASE);
    } else if (!strcmp(name, "Sequence/BeforeTitle.110.module.ainb")) {
        // Keep resident event initialization, but omit the startup event request and its wait.
        const uint64_t pos = 0xc3c;
        if (pos < off || pos - off >= n) return;
        if (overlay || size != 12578 || data[pos - off] != 11)
            rt_fatal("--load-save: unexpected startup graph %s", name);
        data[pos - off] = 9;
        __atomic_fetch_or(&direct_graphs, 1, __ATOMIC_RELEASE);
    }
}

static void direct_boot_apply(uint64_t h, uint64_t fsize, uint64_t off, void *buffer, uint64_t n) {
    if (__atomic_load_n(&direct_save, __ATOMIC_RELAXED) < 0) return;
    uint64_t j = h - ROMFS_HANDLE_BASE;
    plat_mutex_lock_shared(&lock);
    uint64_t i = handles[j] - 1;
    int ov = handle_ovl[j] != 0;
    if (i < file_count) direct_boot_graph(entry_name(entry(i)), ov, fsize, off, buffer, n);
    plat_mutex_unlock_shared(&lock);
}

uint32_t romfs_read(uint64_t *out, uint64_t h, int64_t offset, void *buffer, uint64_t size) {
    PlatFile *f;
    uint64_t base, fsize;
    if (!out || offset < 0 || !handle_source(h, &f, &base, &fsize)) return FS_R_INVALID;
    if ((uint64_t)offset > fsize || (!buffer && size)) return FS_R_INVALID;
    uint64_t n = size < fsize - (uint64_t)offset ? size : fsize - (uint64_t)offset, done = 0;
    uint64_t fs_t0 = plat_now_ns();   // DEBUG-FSSTAT
    if (nup_mode && f == image) {   // a zstd-stored file of the .nup: served from its inflated bytes
        uint64_t row = handles[h - ROMFS_HANDLE_BASE] - 1;
        if (nup_compressed(row)) {
            mem_thaw((uintptr_t)buffer, n);
            memcpy(buffer, nup_inflated(image, row, base, fsize) + offset, (size_t)n);
            direct_boot_apply(h, fsize, (uint64_t)offset, buffer, n);
            *out = n;
            fs_stat(fs_t0, n, 1);
            return 0;
        }
    }
    if (f == image && image_map) {
        memcpy(buffer, image_map + base + (uint64_t)offset, (size_t)n);
        done = n;
    }
    while (done < n) {
        uint32_t chunk = n - done > READ_CHUNK ? READ_CHUNK : (uint32_t)(n - done);
        int64_t r = read_at(f, (uint8_t *)buffer + done, chunk, base + (uint64_t)offset + done);
        if (r <= 0) rt_fatal("RomFS read failure at %#llx", (unsigned long long)(base + offset + done));
        done += (uint64_t)r;
    }
    direct_boot_apply(h, fsize, (uint64_t)offset, buffer, done);
    *out = done;
    fs_stat(fs_t0, done, 0);
    return 0;
}

uint32_t romfs_size(int64_t *out, uint64_t h) {
    PlatFile *f;
    uint64_t base, size;
    if (!out || !handle_source(h, &f, &base, &size)) return FS_R_INVALID;
    *out = (int64_t)size;
    return 0;
}

void romfs_close(uint64_t h) {
    PlatFile *f;
    uint64_t base, size;
    if (!handle_source(h, &f, &base, &size))
        rt_fatal("RomFS: close of invalid file handle %#llx", (unsigned long long)h);
    uint64_t j = h - ROMFS_HANDLE_BASE;
    plat_mutex_lock(&lock);
    PlatFile *of = handle_ovl[j] ? handle_file[j] : NULL;
    uint64_t row = handles[j] - 1;
    if (nup_mode && !handle_ovl[j] && row < file_count) nup_release(row);
    handles[j] = 0;
    handle_ovl[j] = 0;
    handle_file[j] = NULL;
    plat_mutex_unlock(&lock);
    if (of) plat_file_close(of);
}

uint32_t romfs_entry_type(int *out, const char *path) {
    if (!out) return FS_R_INVALID;
    char rel[4096];
    const char *p = romfs_path_relative(mount_of(path), path, rel, sizeof rel);
    if (!p) return FS_R_PATH_NOT_FOUND;
    if (file_index(p) != UINT64_MAX) {
        *out = FS_ENTRY_FILE;
        return 0;
    }
    char dir[4096];
    if (snprintf(dir, sizeof dir, "%s/", p) >= (int)sizeof dir) return FS_R_INVALID;
    if (folder_exists(dir)) {
        *out = FS_ENTRY_DIRECTORY;
        return 0;
    }
    return FS_R_PATH_NOT_FOUND;
}

// The immediate children of a RomFS directory, each name once: subdirectories (FS_ENTRY_DIRECTORY, size 0) and files
// with their sizes, in index order. The index is sorted by path bytes, so a directory's descendants are contiguous and
// so are those of each child directory. Overlays add no entries here (a game that lists its RomFS: OoT,
// content:/Foliage/...). FS_R_PATH_NOT_FOUND when nothing lies under the path.
uint32_t romfs_list(const char *path, int (*add)(const char *name, int kind, uint64_t size, void *ctx), void *ctx) {
    fs_init();
    char rel[4096];
    const char *p = romfs_path_relative(mount_of(path), path, rel, sizeof rel);
    if (!p) return FS_R_PATH_NOT_FOUND;
    char prefix[4096];
    size_t pl = strlen(p);
    while (pl && p[pl - 1] == '/') pl--;
    if (pl + 2 > sizeof prefix) return FS_R_INVALID;
    memcpy(prefix, p, pl);
    if (pl) prefix[pl++] = '/';
    prefix[pl] = 0;
    uint64_t i = pl ? lower(prefix) : 0, found = 0;
    char last_dir[1024] = "";
    for (; i < file_count; i++) {
        Entry e = entry(i);
        const char *name = entry_name(e);
        if (strncmp(name, prefix, pl)) break;
        found++;
        const char *rest = name + pl, *slash = strchr(rest, '/');
        if (!slash) {
            add(rest, FS_ENTRY_FILE, e.size, ctx);
            continue;
        }
        size_t k = (size_t)(slash - rest);
        if (k >= sizeof last_dir) continue;
        if (!strncmp(last_dir, rest, k) && last_dir[k] == 0) continue;
        memcpy(last_dir, rest, k);
        last_dir[k] = 0;
        add(last_dir, FS_ENTRY_DIRECTORY, 0, ctx);
    }
    return found ? 0 : FS_R_PATH_NOT_FOUND;
}

// ---- overrides
// nn::Result nn::fs::MountRom(const char *name, void *cache, size_t cache_size)
static void ov_mount_rom(Cpu *c) { ret_x(c, romfs_mount(arg_p(c, 0))); }
// nn::Result nn::fs::QueryMountRomCacheSize(size_t *out): no cache needed, the host reads the image directly
static void ov_query_cache(Cpu *c) {
    uint64_t *out = arg_p(c, 0);
    if (out) *out = 0;
    ret_x(c, out ? 0 : FS_R_INVALID);
}
// void nn::fs::InitializeForApplication(): SDK-internal; opens the index instead of the fs service session
static void ov_initialize(Cpu *c) {
    (void)c;
    fs_init();
}

HOST_OVERRIDE("_ZN2nn2fs8MountRomEPKcPvm", ov_mount_rom);
HOST_OVERRIDE("_ZN2nn2fs8MountRomEPKc", ov_mount_rom);   // the cacheless overload (Switch 2 SDKs: OoT)
HOST_OVERRIDE("_ZN2nn2fs22QueryMountRomCacheSizeEPm", ov_query_cache);
HOST_OVERRIDE("_ZN2nn2fs24InitializeForApplicationEv", ov_initialize);
