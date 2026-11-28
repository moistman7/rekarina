// NVN front end, command-buffer slice: command buffers, their memory, recording
// as op records (nvn_obj.h), recording copies behind command handles, and the submit-time resolution of deferred binds
// (nvn_submit_prepare, called by nvnQueueSubmitCommands in nvn_api_core.c). Ported from MacNativeGames
// games/totk/third_party/d2r_gfx/gfx/nvn/hle_nvn.c @ b75be8cf (MIT, Copyright (c) 2026 jhaizhou; origin d2r-mobile),
// lines 647-1062, 1261-1286, 1486-1760 and 1975-1990, and the *Deferred and memory-callback entry points of
// totk_nvn_ext.c in the same directory. Metal, mach, the translation thread, Apple diagnostics, the per-frame
// statistics and the copy marks in control memory dropped; the texture pin sweep replaced by counting (texture pins).
//
// Records are written into a growable host buffer per command buffer, never into the game's command memory: games hand
// the same command memory to several command buffers recording at once (budgeted for real NVN encodings), and our
// records are larger. The game's command and control memory is accounted only. EndRecording copies the finished
// recording into host memory; the handle is that copy (a flat OP_BEGIN ... OP_END sequence) and lives until the game
// adds the same control memory again. CallCommands and CopyCommands inline the called recording, so copies hold no
// OP_CALL or OP_JUMP.
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "gfx.h"
#include "host.h"
#include "vk_dlaa.h"
#include "motion_vectors.h"
#include "nvn_api.h"
#include "ftrace.h"
#include "nvn_hmap.h"
#include "platform/platform.h"

#define CMD_MAGIC MAGIC('C', 'M', 'D', 'B')
#define COPY_MAGIC 0x59504f43444d43ull     // "CMDCOPY": prefix of a live recording copy
#define OP_RESERVE 24                       // room always kept at the end of the recording buffer
#define MIN_COMMAND_SIZE 512                // NVN_DEVICE_INFO_COMMAND_BUFFER_MIN_COMMAND_SIZE (nvn_api_core.c devinfo)
#define MIN_CONTROL_SIZE 256                // NVN_DEVICE_INFO_COMMAND_BUFFER_MIN_CONTROL_SIZE
#define CTL_PER_RECORDING 64                // control memory one recording is reported to use
#define COPY_PAGE_SHIFT 16                  // recording copy index granularity (release_copies)

static cmd_t *cmd_of(const Cpu *c) { return arg_p(c, 0); }

// ---------- memory callbacks
// NVN calls the game's memory callback when a recording needs command or control memory and none is left. Records
// live in host memory here, so the callback is called only where the hardware certainly would: a recording starts
// without any command memory, or ends without any control memory. It runs on the calling guest thread through
// rt_call; whatever memory it adds is accounted. Set{Command,Control}MemoryCallbackEnabled(false) suppress it.
enum { CB_NO_COMMAND = 1, CB_NO_CONTROL = 2 };
static PlatMutex cb_lock = PLAT_MUTEX_INIT;
static nvn_hmap cb_off = {.klen = 8};       // command buffer -> disabled callback kinds | 0x100

static unsigned cb_disabled(const cmd_t *o) {
    plat_mutex_lock_shared(&cb_lock);
    uintptr_t v = (uintptr_t)nvn_hm_get(&cb_off, &o);
    plat_mutex_unlock_shared(&cb_lock);
    return (unsigned)v & 0xff;
}

static void cb_set_disabled(const cmd_t *o, unsigned kind, int disabled) {
    plat_mutex_lock(&cb_lock);
    unsigned v = (unsigned)(uintptr_t)nvn_hm_get(&cb_off, &o) & 0xff;
    v = disabled ? v | kind : v & ~kind;
    if (v) nvn_hm_put(&cb_off, &o, (void *)(uintptr_t)(v | 0x100));
    else nvn_hm_del(&cb_off, &o);
    plat_mutex_unlock(&cb_lock);
}

static void memory_event(Cpu *c, cmd_t *o, int event, uint64_t min) {
    if (!o->cb || (cb_disabled(o) & (event == NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY ? CB_NO_COMMAND
                                                                                                     : CB_NO_CONTROL)))
        return;
    NVN_LOG("command buffer %p: memory callback %p (%s memory, %llu bytes)", (void *)o, (void *)o->cb,
            event == NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY ? "command" : "control",
            (unsigned long long)min);
    rt_call(c, (uint64_t)(uintptr_t)o->cb, 4, (uint64_t)(uintptr_t)o, (uint64_t)event, min,
            (uint64_t)(uintptr_t)o->cb_data);
}

// ---------- recording buffer
static void put_hdr(uint8_t *p, uint32_t op, uint32_t size) {
    op_hdr h = {op, size};
    memcpy(p, &h, sizeof h);
}

// 1.4.3 initializes and finalizes command buffers about 2000 times a second while a scene loads and several times a
// frame in play, and its two largest per-frame recordings take 1-2 MiB. Freeing the recording buffer at Finalize made
// every new recording grow again from 64 KiB through up to five reallocations and copies, the last ones fresh 1-2 MiB
// blocks the heap takes from the system: new commit and zero-filled pages twice a frame. Those growths were most of the
// process's new commit requests, so when the system ran short for a moment (a Depths load at render scale 200 on 16 GB)
// a 1 MiB growth was the allocation that failed. Finalized buffers therefore wait in a small pool for the next
// recording, and a failed allocation waits for memory instead of ending the game (realloc_wait).
#define POOL_SLOTS 16
#define POOL_MAX_BYTES ((size_t)4 << 20)   // a larger buffer (a recording that inlined a big one) goes back to the heap
static PlatMutex pool_lock = PLAT_MUTEX_INIT;
static struct { uint8_t *p; size_t size; } pool[POOL_SLOTS];
static uint32_t npool;

// The largest pooled buffer and its size, or NULL: the largest, so the big per-frame recordings stop growing once the
// pool holds enough buffers of their size.
static uint8_t *pool_take(size_t *size) {
    uint8_t *p = NULL;
    plat_mutex_lock(&pool_lock);
    if (npool) {
        uint32_t best = 0;
        for (uint32_t i = 1; i < npool; i++)
            if (pool[i].size > pool[best].size) best = i;
        p = pool[best].p;
        *size = pool[best].size;
        pool[best] = pool[--npool];
    }
    plat_mutex_unlock(&pool_lock);
    return p;
}

// Keeps a finalized buffer when the pool has room or holds a smaller one (which is freed instead).
static void pool_put(uint8_t *p, size_t size) {
    if (!p) return;
    if (size <= POOL_MAX_BYTES) {
        plat_mutex_lock(&pool_lock);
        if (npool < POOL_SLOTS) {
            pool[npool].p = p;
            pool[npool++].size = size;
            p = NULL;
        } else {
            uint32_t small = 0;
            for (uint32_t i = 1; i < npool; i++)
                if (pool[i].size < pool[small].size) small = i;
            if (pool[small].size < size) {
                uint8_t *drop = pool[small].p;
                pool[small].p = p;
                pool[small].size = size;
                p = drop;
            }
        }
        plat_mutex_unlock(&pool_lock);
    }
    free(p);
}

static void pool_drain(void) {
    plat_mutex_lock(&pool_lock);
    while (npool) free(pool[--npool].p);
    plat_mutex_unlock(&pool_lock);
}

// realloc that waits out a short commit shortage: Windows refuses commit beyond the current limit while it grows a
// system-managed page file, so the same request succeeds a moment later, and a stall of some frames is better than
// ending the game. Frees the pooled buffers first. NULL only after about 30 s without memory; the caller ends the game.
// Never called with a lock held.
static void *realloc_wait(void *old, size_t n, const char *what) {
    void *p = realloc(old, n);
    if (p) return p;
    pool_drain();
    static uint32_t reports;
    int report = __atomic_fetch_add(&reports, 1, __ATOMIC_RELAXED) < 8;
    if (report) {
        double proc, used, limit;
        plat_commit_info(&proc, &used, &limit);
        rt_log("nvn: %s of %zu bytes failed (memory commit: this process %.0f MB, the system %.0f of %.0f MB); waiting "
               "for memory", what, n, proc, used, limit);
    }
    uint64_t t0 = plat_now_ms();
    while (plat_now_ms() - t0 < 30000) {
        plat_sleep_ms(10);
        if ((p = realloc(old, n))) {
            if (report)
                rt_log("nvn: %s of %zu bytes succeeded after %llu ms", what, n,
                       (unsigned long long)(plat_now_ms() - t0));
            return p;
        }
    }
    return NULL;
}

static void cmd_reserve(cmd_t *o, size_t need) {
    if (o->seg && o->cur + need + OP_RESERVE <= o->end) return;
    if (!o->seg) {   // the first record since Initialize: a pooled buffer when one waits
        size_t size = 0;
        uint8_t *p = pool_take(&size);
        if (p) {
            o->seg = o->cur = p;
            o->end = p + size;
            if (need + OP_RESERVE <= size) return;
        }
    }
    size_t used = o->seg ? (size_t)(o->cur - o->seg) : 0;
    size_t cap = o->seg ? (size_t)(o->end - o->seg) : 0;
    if (!cap) cap = 64 << 10;
    while (used + need + OP_RESERVE > cap) cap *= 2;
    uint8_t *b = realloc_wait(o->seg, cap, "command recording buffer");
    // Still failing after the wait: the system has no commit left (the fatal report states it), or seg is no block of
    // ours: a command buffer object the game overwrote or copied shows a wrong magic or more used than its old size.
    if (!b)
        rt_fatal("nvn: command recording buffer of %zu bytes (command buffer %p, magic %#x, buffer %p of %zu bytes, %zu "
                 "used, %zu asked)", cap, (void *)o, o->magic, (void *)o->seg, o->seg ? (size_t)(o->end - o->seg) : 0,
                 used, need);
    o->seg = b;
    o->cur = b + used;
    o->end = b + cap;
}

// Space for one record (header included); the header's op is filled by the caller.
static uint8_t *cmd_alloc(cmd_t *o, uint32_t size) {
    size = (size + 7) & ~7u;
    cmd_reserve(o, size);
    uint8_t *p = o->cur;
    o->cur += size;
    o->cmd_used += size;
    o->hw_used += size;
    o->last_rec = size;
    put_hdr(p, 0, size);
    return p;
}

// The game sums nvnCommandBufferGetCommandMemoryUsed over recordings to size resource heaps, so usage must be what the
// real NVN push-buffer encoding takes, not our record size. Known encodings (Maxwell methods, from a real recording):
// DrawElements 12 words, DrawElementsBaseVertex 11, BindVertexAttribState 1+16, BindVertexStreamState 16+n,
// BindVertexBuffer 6 words each.
static void hw_fix(cmd_t *o, size_t bytes) {
    o->hw_used += bytes;
    o->hw_used -= o->last_rec;
    o->last_rec = bytes;
}

// A zeroed payload of one record.
static void *rec(cmd_t *o, uint32_t op, size_t payload) {
    uint32_t sz = (uint32_t)((8 + payload + 7) & ~(size_t)7);
    uint8_t *p = cmd_alloc(o, sz);
    put_hdr(p, op, sz);
    memset(p + 8, 0, sz - 8);
    return p + 8;
}
#define REC(o, OP, T) ((T *)rec((o), (OP), sizeof(T)))
#ifdef GAME_NS2
void *nvn_rec(cmd_t *o, uint32_t op, size_t payload) { return rec(o, op, payload); }
#endif
// Array-carrying binds take the prefix fields plus n elements; the replay reads only the first n.
#define RECN(o, OP, T, ARR, N) ((T *)rec((o), (OP), offsetof(T, ARR) + sizeof(((T *)0)->ARR[0]) * (size_t)(N)))

static void copy_view(view_t *dst, uint8_t *has, const NVNtextureView *v) {
    *has = v != NULL;
    if (v) *dst = *(const view_t *)v;
}

// ---------- texture pins
// Recordings refer to textures by pointer, but the game may finalize or reuse an NVNtexture after recording and before
// the submit (NVN itself captured only GPU state). Records therefore point at an immutable host copy per texture
// initialization (tex_t.serial). The game initializes its transient targets again every frame, about 29 new serials a
// frame on 1.4.3, so copies kept for the run grew the heap by about 44 MiB per minute of play, standing still too, until
// the commit limit ended the game after a few hours. A copy is freed once nothing can read it any more:
// - refs counts the recording copies whose ops name it: live ones, released ones a queued submission still reads, and
//   per-submission clones (pin_refs, at flatten_recording and wherever such a copy is freed or cloned);
// - a recording in progress holds pins no copy counts yet, through its own tex_pin calls and through ops inlined from a
//   copy that was released since: a pin goes only NVN_PIN_AGE submits after its last tex_pin and after refs fell to 0;
// - the replay's current render targets carry over to later submissions and are compared by pointer (vk_exec.c set_rt),
//   so a freed address reused by a new pin would read as the same target: nvn_pins_sweep runs between submissions on
//   the thread that replays them and keeps the pins it is given.
// MacNativeGames hit the same growth (3.16 million copies, 405 MB after 27 minutes) and marks pins by walking every live
// recording each window (hle_nvn.c pin_scan); counting at copy and release walks each copy's ops once more, when it is
// freed. REKARINA_PIN_SWEEP=0 keeps every copy.
typedef struct {
    tex_t t;              // first: ops hold pins as tex_t pointers
    uint64_t last_use;    // nsubmits at the last tex_pin or the last drop of refs
    uint32_t refs;
    uint8_t queued;       // in pin_q
} pin_t;
_Static_assert(offsetof(pin_t, t) == 0, "ops hold pins as tex_t pointers");
typedef struct { pin_t *p; uint64_t at; } pin_due;
static PlatMutex pin_lock = PLAT_MUTEX_INIT;   // lock order: copy_lock, then pin_lock
static nvn_hmap pins = {.klen = 8};         // serial -> pin_t
static pin_due *pin_q;                      // pins with no counted copy, oldest first: [pin_qh, pin_qn)
static size_t pin_qh, pin_qn, pin_qcap;
static uint64_t pins_live, pins_freed;
static uint64_t nsubmits;                   // QueueSubmitCommands calls so far

static void pin_queue(pin_t *p, uint64_t at) {   // pin_lock held
    if (p->queued) return;
    if (pin_qh && pin_qh * 2 >= pin_qn) {
        memmove(pin_q, pin_q + pin_qh, sizeof *pin_q * (pin_qn - pin_qh));
        pin_qn -= pin_qh, pin_qh = 0;
    }
    if (pin_qn == pin_qcap) {
        pin_qcap = pin_qcap ? pin_qcap * 2 : 4096;
        if (!(pin_q = realloc(pin_q, sizeof *pin_q * pin_qcap))) rt_fatal("nvn: out of memory (texture pin queue)");
    }
    pin_q[pin_qn++] = (pin_due){p, at};
    p->queued = 1;
}

static const tex_t *tex_pin(const NVNtexture *nt) {
    const tex_t *t = (const tex_t *)nt;
    if (!t) return NULL;
    uint64_t serial = t->serial;
    if (!serial) rt_fatal("nvn: a recording references the uninitialized texture object %p", (const void *)t);
    uint64_t now = __atomic_load_n(&nsubmits, __ATOMIC_RELAXED);
    plat_mutex_lock_shared(&pin_lock);   // nvn_pins_sweep frees under the exclusive lock
    pin_t *r = nvn_hm_get(&pins, &serial);
    if (r) __atomic_store_n(&r->last_use, now, __ATOMIC_RELAXED);
    plat_mutex_unlock_shared(&pin_lock);
    if (r) return &r->t;
    plat_mutex_lock(&pin_lock);
    r = nvn_hm_get(&pins, &serial);
    if (!r) {
        if (!(r = malloc(sizeof *r))) rt_fatal("nvn: texture pin copy");
        *r = (pin_t){.t = *t};
        nvn_hm_put(&pins, &serial, r);
        pins_live++;
        pin_queue(r, now);   // freed like any other if no copy ever names it
    }
    __atomic_store_n(&r->last_use, now, __ATOMIC_RELAXED);
    plat_mutex_unlock(&pin_lock);
    return &r->t;
}

#ifdef GAME_NS2
const tex_t *nvn_tex_pin(const NVNtexture *t) { return tex_pin(t); }
#endif

// The pins an op's payload names, into out (at most 9); returns how many.
static int op_pins(uint32_t op, const void *a, const tex_t *out[9]) {
    int k = 0;
    switch (op) {
    case OP_SET_RT: {
        const p_set_rt *r = a;
        int n = r->n < 0 ? 0 : r->n > 8 ? 8 : r->n;
        for (int i = 0; i < n; i++)
            if (r->color[i]) out[k++] = r->color[i];
        if (r->depth) out[k++] = r->depth;
        break;
    }
    case OP_COPY_B2T: case OP_COPY_T2B:
        if (((const p_copy_b2t *)a)->tex) out[k++] = ((const p_copy_b2t *)a)->tex;
        break;
    case OP_CLEAR_TEX:
        if (((const p_clear_tex *)a)->tex) out[k++] = ((const p_clear_tex *)a)->tex;
        break;
    case OP_COPY_T2T: {
        const p_copy_t2t *c = a;
        if (c->src) out[k++] = c->src;
        if (c->dst) out[k++] = c->dst;
        break;
    }
    default: break;
    }
    return k;
}

static void pin_ref(const tex_t *t, int delta, uint64_t now) {
    pin_t *p = (pin_t *)(uintptr_t)t;
    if (delta > 0) {
        __atomic_add_fetch(&p->refs, 1, __ATOMIC_RELAXED);
        return;
    }
    __atomic_store_n(&p->last_use, now, __ATOMIC_RELAXED);
    if (__atomic_sub_fetch(&p->refs, 1, __ATOMIC_ACQ_REL)) return;
    plat_mutex_lock(&pin_lock);
    pin_queue(p, now);
    plat_mutex_unlock(&pin_lock);
}

// Counts (delta 1) or uncounts (-1) the pins named by the copy at handle, OP_BEGIN to OP_END; returns how many.
static uint32_t pin_refs(const uint8_t *copy, int delta) {
    uint64_t now = __atomic_load_n(&nsubmits, __ATOMIC_RELAXED);
    uint32_t n = 0;
    for (const op_hdr *h = nvn_op_first((uintptr_t)copy); h->op != OP_END; h = nvn_op_next(h)) {
        const tex_t *p[9];
        int k = op_pins(h->op, nvn_op_payload(h), p);
        for (int i = 0; i < k; i++) pin_ref(p[i], delta, now);
        n += (uint32_t)k;
    }
    return n;
}

void nvn_pins_sweep(uint64_t now, const tex_t *const *keep, int nkeep) {
    if (now < NVN_PIN_AGE) return;
    static int on = -1;
    if (on < 0) {
        const char *s = cfg_get("REKARINA_PIN_SWEEP");
        on = !(s && !strcmp(s, "0"));
        if (!on) rt_log("nvn: REKARINA_PIN_SWEEP=0: texture pin copies are kept for the run (the heap grows in long sessions)");
    }
    if (!on) return;
    static uint64_t logged;
    plat_mutex_lock(&pin_lock);
    for (int budget = 1024; pin_qh < pin_qn && budget; budget--) {
        pin_due d = pin_q[pin_qh];
        if (d.at > now - NVN_PIN_AGE) break;
        pin_qh++;
        pin_t *p = d.p;
        p->queued = 0;
        if (__atomic_load_n(&p->refs, __ATOMIC_ACQUIRE)) continue;   // queued again when the count falls to 0
        uint64_t last = __atomic_load_n(&p->last_use, __ATOMIC_RELAXED);
        int kept = 0;
        for (int i = 0; i < nkeep; i++) kept |= keep[i] == &p->t;
        if (kept || last > now - NVN_PIN_AGE) {
            pin_queue(p, kept ? now : last);
            continue;
        }
        nvn_hm_del(&pins, &p->t.serial);
        free(p);
        pins_live--, pins_freed++;
    }
    if (now / 7200 != logged) {
        logged = now / 7200;
        rt_log("nvn: texture pins at submit %llu: %llu live, %llu waiting, %llu freed so far", (unsigned long long)now,
               (unsigned long long)pins_live, (unsigned long long)(pin_qn - pin_qh), (unsigned long long)pins_freed);
    }
    plat_mutex_unlock(&pin_lock);
}

void nvn_pins_stats(uint64_t *live, uint64_t *freed) {
    plat_mutex_lock_shared(&pin_lock);
    *live = pins_live, *freed = pins_freed;
    plat_mutex_unlock_shared(&pin_lock);
}

// ---------- recording copies
// A handle points just past a copy_prefix. Copies are indexed by the 64 KiB block of their control-memory position, so
// adding control memory again releases the copies recorded there without walking every live copy. 4 KiB pages cost
// about 500 lookups per AddControlMemory under the exclusive lock (1.4.3 field: 53 calls a frame, 2 MiB ranges, 3.7
// copies per page), and the recording threads waited on it.
// deferred: 0 unknown, 1 none, 2 some; pins: texture pins its ops name, counted in their refs (pin_refs); ho: the size of
// the recording buffer the copy is (REKARINA_CMD_HANDOVER), which goes back to the buffer pool with it; 0 for a copy of
// its own
typedef struct { uint64_t magic, bytes, made_at, used_at, deferred, pins, ho, host_only; } copy_prefix;
_Static_assert(sizeof(copy_prefix) % 8 == 0, "records stay 8-byte aligned");
typedef struct { uintptr_t lo, hi; uint8_t *copy; } cmd_copy;
typedef struct { cmd_copy *v; uint32_t n, cap; } copy_bucket;
static PlatMutex copy_lock = PLAT_MUTEX_INIT;    // guards the two maps, the counters and every prefix
static nvn_hmap copy_pages = {.klen = 8};   // page -> copy_bucket
static nvn_hmap copy_live = {.klen = 8};    // handle -> copy_prefix of a live copy
static uint64_t ncopies, copy_bytes, copies_made;
// Asynchronous replay: copies released by the game, and per-submission clones, that a queued submission still reads.
// Freed once gfx_replayed_submits() reaches their used_at. Empty with synchronous replay.
static copy_prefix **zombies;
static uint32_t nzombies, cap_zombies;

static void zombie_add(copy_prefix *pf) {   // copy_lock held
    if (nzombies == cap_zombies) {
        cap_zombies = cap_zombies ? cap_zombies * 2 : 64;
        zombies = realloc(zombies, sizeof *zombies * cap_zombies);
        if (!zombies) rt_fatal("nvn: out of memory (released recording copies)");
    }
    zombies[nzombies++] = pf;
}

static void copy_free(copy_prefix *pf) {   // copy_lock held, or the copy is out of every index
    if (pf->pins) pin_refs((const uint8_t *)(pf + 1), -1);
    if (pf->ho) pool_put((uint8_t *)pf, (size_t)pf->ho);   // a handed-over recording buffer: back to the pool
    else free(pf);
}

static void zombie_sweep(uint64_t replayed) {   // copy_lock held
    for (uint32_t i = 0; i < nzombies;) {
        if (zombies[i]->used_at <= replayed) {
            copy_free(zombies[i]);
            zombies[i] = zombies[--nzombies];
        } else {
            i++;
        }
    }
}

static copy_prefix *copy_find(uint64_t handle) {   // copy_lock held
    return nvn_hm_get(&copy_live, &handle);
}

int nvn_handle_live(uint64_t handle) {
    plat_mutex_lock_shared(&copy_lock);
    int live = copy_find(handle) != NULL;
    plat_mutex_unlock_shared(&copy_lock);
    return live;
}

uint64_t nvn_handle_bytes(uint64_t handle) {
    plat_mutex_lock_shared(&copy_lock);
    const copy_prefix *pf = copy_find(handle);
    uint64_t n = pf ? pf->bytes : 0;
    plat_mutex_unlock_shared(&copy_lock);
    return n;
}

// Marks a live copy used by submission now; its size from OP_BEGIN to OP_END inclusive.
// The game calls ~2,900 recordings a frame into others at the Market's square (CallCommands), most of a few hundred
// bytes, from six threads at once; the shared lock and the map lookup per call, on a lock that EndRecording and
// AddControlMemory take exclusively, were most of CallCommands' time. A live copy's prefix says so itself (magic, set
// at EndRecording, cleared under copy_lock before the copy goes), so with the synchronous replay a call reads it
// directly (REKARINA_CMD_CALL_FAST, on by default; 0: the locked lookup always). A handle whose prefix is not live goes
// the locked way, which stops the game as before. The asynchronous replay keeps the lock: it frees copies by used_at.
static uint64_t copy_touch(uint64_t handle, uint64_t now, const char *what, uint64_t *host_only) {
    static int fast = -1;
    if (fast < 0) fast = rt_knob("CMD_CALL_FAST", "1") != 0;
    if (fast && handle && !gfx_replay_async()) {
        copy_prefix *q = (copy_prefix *)(uintptr_t)handle - 1;
        if (__atomic_load_n(&q->magic, __ATOMIC_ACQUIRE) == COPY_MAGIC) {
            __atomic_store_n(&q->used_at, now, __ATOMIC_RELAXED);
            if (host_only) *host_only = q->host_only;
            return q->bytes;
        }
    }
    plat_mutex_lock_shared(&copy_lock);
    copy_prefix *pf = copy_find(handle);
    uint64_t bytes = 0;
    if (pf) {
        __atomic_store_n(&pf->used_at, now, __ATOMIC_RELAXED);
        bytes = pf->bytes;
        if (host_only) *host_only = pf->host_only;
    }
    plat_mutex_unlock_shared(&copy_lock);
    if (!pf) rt_fatal("nvn: %s: command handle %#llx is not a live recording (released or never recorded)", what,
                      (unsigned long long)handle);
    return bytes;
}

// The recording [p, p + n), OP_BEGIN to OP_END (the recording buffer is one block: recording emits no OP_JUMP), copied
// into one host block of its exact size. One allocation and one copy: growing the copy record by record cost the
// game's main thread about 1 ms a frame in reallocations and small copies (1.4.3, field).
// The records of a finished recording [p, p + n) checked (a corrupt one stops here, not in the replay) and the texture
// pins their payloads name counted; returns how many.
static uint64_t scan_recording(const uint8_t *p, size_t n, uint64_t *host_only_out) {
    uint64_t pins_named = 0, host_only = 0;
    for (size_t at = 0;;) {
        op_hdr h;
        memcpy(&h, p + at, sizeof h);
        if (h.size < 8 || (h.size & 7) || h.op >= OP_COUNT_ || h.op == OP_JUMP || h.op == OP_CALL || at + h.size > n ||
            (!at && h.op != OP_BEGIN))
            rt_fatal("nvn: corrupt recording at %p (op %u size %u)", (const void *)(p + at), h.op, h.size);
        const tex_t *pinned[9];
        int k = op_pins(h.op, p + at + 8, pinned);
        for (int i = 0; i < k; i++) pin_ref(pinned[i], 1, 0);
        pins_named += (uint64_t)k;
        if (h.op == OP_MODEL_SCOPE) {
            const p_model_scope *scope = (const void *)(p + at + sizeof h);
            if (h.size != sizeof h + sizeof *scope || n - at - h.size < sizeof h ||
                scope->bytes > n - at - h.size - sizeof h || (scope->bytes & 7))
                rt_fatal("nvn: invalid model ownership range at %p", (const void *)(p + at));
            host_only += h.size;
        }
        at += h.size;
        if (h.op == OP_END) {
            if (at != n) rt_fatal("nvn: recording at %p ends at %zu of %zu bytes", (const void *)p, at, n);
            break;
        }
    }
    if (host_only_out) *host_only_out = host_only;
    return pins_named;
}

static uint8_t *flatten_recording(const uint8_t *p, size_t n) {
    uint8_t *out = realloc_wait(NULL, sizeof(copy_prefix) + n, "command copy");
    if (!out) rt_fatal("nvn: command copy allocation");
    memcpy(out + sizeof(copy_prefix), p, n);
    uint64_t host_only = 0;
    uint64_t pins_named = scan_recording(out + sizeof(copy_prefix), n, &host_only);
    *(copy_prefix *)out = (copy_prefix){.magic = COPY_MAGIC, .bytes = n, .pins = pins_named, .host_only = host_only};
    return out + sizeof(copy_prefix);
}

// ---- REKARINA_CMD_HANDOVER (1 on, the default; 0 off): a large recording's buffer becomes its copy
// A recording starts behind room for a copy prefix in its command buffer's heap buffer (cmd_t.recording REC_HEAP_HO),
// and one of HO_MIN bytes or more is not copied at EndRecording: the buffer itself, its records checked and pins
// counted (scan_recording), becomes the copy, and the command buffer takes another from the buffer pool at its next
// record. The copy gives the buffer back to the pool when it is freed, so the large buffers circulate without the
// heap's allocations and frees. At the Market's square 92 % of the main thread's 1.5 MiB of recordings a frame were in
// 1.8 recordings of 64 KiB or more, 46-75 % of each module worker's; with the hand-over the game's threads spent 2.3 ms
// a frame less (summed) in EndRecording, CallCommands and AddControlMemory, and the square ran 4-5 fps faster
// (2026-11-18). Small recordings are copied as before (most copies are a few hundred bytes and live long).
#define HO_MIN ((size_t)64 << 10)
#define REC_HEAP_HO 2   // cmd_t.recording of a recording with room for a copy prefix in front
#define REC_MODEL_SCOPE 4
static int ho_knob(void) {
    static int on = -1;
    if (on < 0) {
        on = rt_knob("CMD_HANDOVER", "1") != 0;
        rt_log("nvn: large recordings %s (REKARINA_CMD_HANDOVER=%d)", on ? "hand their buffers over as copies" : "are copied", on);
    }
    return on;
}

static void copy_add(uintptr_t lo, uintptr_t hi, uint8_t *copy) {
    copy_prefix *pf = (copy_prefix *)copy - 1;
    uint64_t handle = (uintptr_t)copy;
    uintptr_t page = lo >> COPY_PAGE_SHIFT;
    plat_mutex_lock(&copy_lock);
    pf->made_at = nsubmits;
    copy_bucket *b = nvn_hm_get(&copy_pages, &page);
    if (!b) {
        b = calloc(1, sizeof *b);
        if (!b) rt_fatal("nvn: command copy index");
        nvn_hm_put(&copy_pages, &page, b);
    }
    if (b->n == b->cap) {
        b->cap = b->cap ? b->cap * 2 : 8;
        b->v = realloc(b->v, b->cap * sizeof *b->v);
        if (!b->v) rt_fatal("nvn: command copy index");
    }
    b->v[b->n++] = (cmd_copy){lo, hi, copy};
    nvn_hm_put(&copy_live, &handle, pf);
    ncopies++;
    copy_bytes += pf->bytes;
    if (++copies_made % 20000 == 0)
        NVN_LOG("recording copies: %llu made, %llu live, %.1f MiB, submit %llu", (unsigned long long)copies_made,
                (unsigned long long)ncopies, copy_bytes / 1048576.0, (unsigned long long)nsubmits);
    plat_mutex_unlock(&copy_lock);
}

// Adding control memory again retires every recording copy made in it (the driver's handle lived there).
// The copies leaving every index are freed after copy_lock (their pins and the buffer pool have locks of their own), so
// the threads waiting for copy_lock in CallCommands and EndRecording do not wait behind the frees (REKARINA_CMD_LATE_FREE,
// on by default; 0: freed under the lock as before). At the Market's square with the buffer hand-over this took the
// game's threads' summed time in AddControlMemory from ~1.1 to ~0.75 ms a frame and in CallCommands by ~0.3 ms
// (2026-11-18).
static void release_copies(uintptr_t lo, uintptr_t hi) {
    if (hi <= lo) return;
    uint64_t replayed = gfx_replayed_submits();
    static int late_on = -1;
    if (late_on < 0) late_on = rt_knob("CMD_LATE_FREE", "1") != 0;
    copy_prefix *late[256];
    int nlate = 0;
    plat_mutex_lock(&copy_lock);
    if (nzombies) zombie_sweep(replayed);
    for (uintptr_t pg = lo >> COPY_PAGE_SHIFT; pg <= (hi - 1) >> COPY_PAGE_SHIFT; pg++) {
        copy_bucket *b = nvn_hm_get(&copy_pages, &pg);
        if (!b) continue;
        for (uint32_t i = 0; i < b->n;) {
            cmd_copy *k = &b->v[i];
            if (!(k->lo < hi && lo < k->hi)) { i++; continue; }
            uint64_t handle = (uintptr_t)k->copy;
            copy_prefix *pf = (copy_prefix *)k->copy - 1;
            nvn_hm_del(&copy_live, &handle);
            ncopies--;
            copy_bytes -= pf->bytes;
            pf->magic = 0;
            if (pf->used_at > replayed) zombie_add(pf);   // a queued submission still reads it
            else if (late_on && nlate < 256) late[nlate++] = pf;
            else copy_free(pf);
            b->v[i] = b->v[--b->n];
        }
        if (!b->n) {
            nvn_hm_del(&copy_pages, &pg);
            free(b->v);
            free(b);
        }
    }
    plat_mutex_unlock(&copy_lock);
    for (int i = 0; i < nlate; i++) copy_free(late[i]);
}

// ---------- CommandBuffer: object and memory
// NVNboolean nvnCommandBufferInitialize(NVNcommandBuffer *cmd, NVNdevice *device): the inline fast-path fields are kept
// at ~0 / 0 so NVN 55 inline code never finds room and calls the API (nvn_obj.h cmd_t)
NVN_API(nvnCommandBufferInitialize) {
    cmd_t *o = cmd_of(c);
    // Initialize over an object that was never finalized drops its recording buffer (seg lives in host memory). Logged
    // only: the old bytes may be a copy whose buffer another object owns, so freeing it here is not safe.
    static uint32_t reinits;
    if (o->magic == CMD_MAGIC && o->seg && __atomic_fetch_add(&reinits, 1, __ATOMIC_RELAXED) < 8)
        rt_log("nvn: Initialize over the command buffer %p, which was not finalized; its %zu-byte recording buffer %p "
               "is dropped", (void *)o, (size_t)(o->end - o->seg), (void *)o->seg);
    memset(o, 0, sizeof *o);
    o->magic = CMD_MAGIC;
    o->dev = arg_p(c, 1);
    o->inline_cur = ~0ull;
    o->inline_end = 0;
    cb_set_disabled(o, CB_NO_COMMAND | CB_NO_CONTROL, 0);
    ret_x(c, 1);
}
// void nvnCommandBufferFinalize(NVNcommandBuffer *cmd): handles outlive their recorder
NVN_API(nvnCommandBufferFinalize) {
    cmd_t *o = cmd_of(c);
    pool_put(o->seg, (size_t)(o->end - o->seg));
    o->seg = o->cur = o->end = NULL;
    o->magic = 0;
    cb_set_disabled(o, CB_NO_COMMAND | CB_NO_CONTROL, 0);
}
// void nvnCommandBufferSetMemoryCallback(NVNcommandBuffer *cmd, PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC callback)
NVN_API(nvnCommandBufferSetMemoryCallback) { cmd_of(c)->cb = arg_p(c, 1); }
// void nvnCommandBufferSetMemoryCallbackData(NVNcommandBuffer *cmd, void *data)
NVN_API(nvnCommandBufferSetMemoryCallbackData) { cmd_of(c)->cb_data = arg_p(c, 1); }
// PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC nvnCommandBufferGetMemoryCallback(const NVNcommandBuffer *cmd)
NVN_API(nvnCommandBufferGetMemoryCallback) { ret_x(c, (uintptr_t)cmd_of(c)->cb); }
// void *nvnCommandBufferGetMemoryCallbackData(const NVNcommandBuffer *cmd) (nvn.h declares it void)
NVN_API(nvnCommandBufferGetMemoryCallbackData) { ret_x(c, (uintptr_t)cmd_of(c)->cb_data); }
// void nvnCommandBufferSetCommandMemoryCallbackEnabled / SetControlMemoryCallbackEnabled(NVNcommandBuffer *cmd,
// NVNboolean enabled) (NVN 55, no prototype in nvn.h)
NVN_API(nvnCommandBufferSetCommandMemoryCallbackEnabled) { cb_set_disabled(cmd_of(c), CB_NO_COMMAND, !nvn_arg_bool(c, 1)); }
NVN_API(nvnCommandBufferSetControlMemoryCallbackEnabled) { cb_set_disabled(cmd_of(c), CB_NO_CONTROL, !nvn_arg_bool(c, 1)); }
// void nvnCommandBufferAddCommandMemory(NVNcommandBuffer *cmd, const NVNmemoryPool *pool, ptrdiff_t offset, size_t size):
// accounting only; command memory can be shared by independent recordings with distinct control memory
NVN_API(nvnCommandBufferAddCommandMemory) {
    cmd_t *o = cmd_of(c);
    const pool_t *p = arg_p(c, 1);
    o->cmd_base = (uintptr_t)(p ? p->mem : NULL) + (uintptr_t)arg_x(c, 2);
    o->game_size = arg_x(c, 3);
    o->cmd_used = 0;
    o->hw_used = 0;
}
// void nvnCommandBufferAddControlMemory(NVNcommandBuffer *cmd, void *memory, size_t size): the SDK encodes handles in
// control memory (sdk+0x444530), so reusing that storage retires the copies recorded there
NVN_API(nvnCommandBufferAddControlMemory) {
    cmd_t *o = cmd_of(c);
    void *m = arg_p(c, 1);
    size_t n = arg_x(c, 2);
    release_copies((uintptr_t)m, (uintptr_t)m + n);
    o->ctl = m;
    o->ctl_size = n;
    o->ctl_used = 0;
}
// size_t nvnCommandBufferGet{Control,Command}Memory{Size,Used,Free}(const NVNcommandBuffer *cmd): usage is bounded by
// the memory the game gave, so the game never sees more used than it provided
static size_t min_size(size_t a, size_t b) { return a < b ? a : b; }
NVN_API(nvnCommandBufferGetControlMemorySize) { ret_x(c, cmd_of(c)->ctl_size); }
NVN_API(nvnCommandBufferGetControlMemoryUsed) { const cmd_t *o = cmd_of(c); ret_x(c, min_size(o->ctl_used, o->ctl_size)); }
NVN_API(nvnCommandBufferGetControlMemoryFree) { const cmd_t *o = cmd_of(c); ret_x(c, o->ctl_size - min_size(o->ctl_used, o->ctl_size)); }
NVN_API(nvnCommandBufferGetCommandMemorySize) { ret_x(c, cmd_of(c)->game_size); }
NVN_API(nvnCommandBufferGetCommandMemoryUsed) { const cmd_t *o = cmd_of(c); ret_x(c, min_size(o->hw_used, o->game_size)); }
NVN_API(nvnCommandBufferGetCommandMemoryFree) { const cmd_t *o = cmd_of(c); ret_x(c, o->game_size - min_size(o->hw_used, o->game_size)); }

// ---------- CommandBuffer: recording
// void nvnCommandBufferBeginRecording(NVNcommandBuffer *cmd)
NVN_API(nvnCommandBufferBeginRecording) {
    cmd_t *o = cmd_of(c);
    if (o->recording) rt_log("nvn: BeginRecording on command buffer %p while it records", (void *)o);
    o->recording = 1;
#ifdef GAME_OOT
    const CfgRenderState *settings = cfg_render_frame_acquire();
    const CfgRenderValues *frame = cfg_render_values_of(settings);
    if (frame && frame->motion_models && !frame->motion_reverse_only) o->recording |= REC_MODEL_SCOPE;
    cfg_render_release(settings);
#endif
    FT(FT_NOTE, 40, (uintptr_t)o, 0);   // DEBUG-FTRACE: a recording begins (which thread, when)
    o->cur = o->seg;   // the previous recording was copied at EndRecording
    if (ho_knob()) {   // REKARINA_CMD_HANDOVER: room for the copy prefix in front of the records
        cmd_reserve(o, sizeof(copy_prefix) + 8);
        o->cur = o->seg + sizeof(copy_prefix);
        o->recording |= REC_HEAP_HO;
    }
    o->rec_start = o->cmd_used;
    put_hdr(cmd_alloc(o, 8), OP_BEGIN, 8);
    o->hw_used -= 8;   // BEGIN and END have no hardware equivalent
    o->hw_start = o->hw_used;
    if (!o->game_size) memory_event(c, o, NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY, MIN_COMMAND_SIZE);
}
// NVNcommandHandle nvnCommandBufferEndRecording(NVNcommandBuffer *cmd)
NVN_API(nvnCommandBufferEndRecording) {
    cmd_t *o = cmd_of(c);
    if (!o->recording || !o->seg) rt_fatal("nvn: EndRecording on command buffer %p, which does not record", (void *)o);
    if (!o->ctl_size) memory_event(c, o, NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_CONTROL_MEMORY, MIN_CONTROL_SIZE);
    put_hdr(cmd_alloc(o, 8), OP_END, 8);
    o->hw_used -= 8;
    size_t front = o->recording & REC_HEAP_HO ? sizeof(copy_prefix) : 0;   // REKARINA_CMD_HANDOVER's prefix room
    o->recording = 0;
    FT(FT_NOTE, 41, (uintptr_t)o, (uint64_t)(o->cur - o->seg));   // DEBUG-FTRACE: it ends, with its bytes
    uint8_t *copy;
    size_t n = (size_t)(o->cur - o->seg) - front;
    if (front && n >= HO_MIN) {   // the buffer itself becomes the copy; the next record takes one from the pool
        copy = o->seg + front;
        uint64_t host_only = 0;
        uint64_t pins = scan_recording(copy, n, &host_only);
        *(copy_prefix *)o->seg = (copy_prefix){.magic = COPY_MAGIC, .bytes = n, .pins = pins,
                                               .ho = (uint64_t)(o->end - o->seg), .host_only = host_only};
        o->seg = o->cur = o->end = NULL;
    } else
        copy = flatten_recording(o->seg + front, n);
    // Finalizing the recorder does not retire its handle; only reusing the control memory it was recorded in does.
    uintptr_t lo = (uintptr_t)o->ctl + (o->ctl_used < o->ctl_size ? o->ctl_used : 0);
    uintptr_t hi = o->ctl ? lo + 1 : 0;
    o->ctl_used += CTL_PER_RECORDING;
    copy_add(lo, hi, copy);
    ret_x(c, (uintptr_t)copy);
}
// NVNboolean nvnCommandBufferIsRecording(const NVNcommandBuffer *cmd)
NVN_API(nvnCommandBufferIsRecording) { ret_x(c, cmd_of(c)->recording != 0); }

// The called recording is copied into the caller at the call position: the game recycles command memory in small steps,
// so a handle may be gone by the time the caller is submitted. Deferred binds stay deferred. A copy is OP_BEGIN, its
// records and OP_END (flatten_recording checked them), so the records go over as one block; usage grows by the same
// bytes as record by record.
static void inline_recording(cmd_t *o, uint64_t handle, const char *what, const TrackedModelSample *sample) {
    if (!handle) rt_fatal("nvn: %s with a null handle", what);
    uint64_t host_only = 0;
    uint64_t bytes = copy_touch(handle, __atomic_load_n(&nsubmits, __ATOMIC_RELAXED), what, &host_only);
    if (bytes <= 16) return;
    size_t body = (size_t)bytes - 16;
    if (sample && sample->identity) {
        *REC(o, OP_MODEL_SCOPE, p_model_scope) = (p_model_scope){*sample, body};
        hw_fix(o, 0);
    }
    cmd_reserve(o, body);
    memcpy(o->cur, (const uint8_t *)(uintptr_t)handle + 8, body);
    o->cur += body;
    o->cmd_used += body;
    o->hw_used += body - host_only;
    o->last_rec = 0;
}
// void nvnCommandBuffer{Call,Copy}Commands(NVNcommandBuffer *cmd, int n, const NVNcommandHandle *handles)
NVN_API(nvnCommandBufferCallCommands) {
    cmd_t *o = cmd_of(c);
    int n = (int)arg_x(c, 1);
    const uint64_t *h = arg_p(c, 2);
    TrackedModelSample sample = {0};
#ifdef GAME_OOT
    if (n == 1 && (o->recording & REC_MODEL_SCOPE)) {
        unsigned entry = 0;
        switch (c->x[30]) {
        case UINT64_C(0x7000067618): entry = 27; break;
        case UINT64_C(0x700006d978): entry = 23; break;
        case UINT64_C(0x700006b7d8): entry = 27; break;
        case UINT64_C(0x70000590c8): entry = 22; break;
        }
        if (entry && rt_patch_enabled(rt_patch_set("tracked_models"))) sample = tracked_model_work_sample(c->x[entry]);
    }
#endif
    for (int i = 0; i < n; i++) inline_recording(o, h[i], "CallCommands", &sample);
}
NVN_API(nvnCommandBufferCopyCommands) {
    cmd_t *o = cmd_of(c);
    int n = (int)arg_x(c, 1);
    const uint64_t *h = arg_p(c, 2);
    for (int i = 0; i < n; i++) inline_recording(o, h[i], "CopyCommands", NULL);
}

// ---------- CommandBuffer: commands with CPU-visible results (performed by the back end in GPU order)
// void nvnCommandBufferSignalEvent(NVNcommandBuffer *cmd, const NVNevent *event, NVNeventSignalMode mode,
// NVNeventSignalLocation location, int flags, uint32_t value)
NVN_API(nvnCommandBufferSignalEvent) {
    const ev_t *e = arg_p(c, 1);
    *REC(cmd_of(c), OP_SIGNAL_EVENT, p_signal_event) = (p_signal_event){e->val, (uint32_t)arg_x(c, 2), (uint32_t)arg_x(c, 5)};
}
// void nvnCommandBufferResetCounter(NVNcommandBuffer *cmd, NVNcounterType type)
NVN_API(nvnCommandBufferResetCounter) { REC(cmd_of(c), OP_RESET_COUNTER, p_reset_counter)->type = (uint32_t)arg_x(c, 1); }
// void nvnCommandBufferReportCounter(NVNcommandBuffer *cmd, NVNcounterType type, NVNbufferAddress address)
NVN_API(nvnCommandBufferReportCounter) {
    *REC(cmd_of(c), OP_REPORT_COUNTER, p_report_counter) = (p_report_counter){(uint32_t)arg_x(c, 1), arg_x(c, 2)};
}
// void nvnCommandBufferCopyBufferToBuffer(NVNcommandBuffer *cmd, NVNbufferAddress src, NVNbufferAddress dst,
// size_t size, int flags)
NVN_API(nvnCommandBufferCopyBufferToBuffer) {
    *REC(cmd_of(c), OP_COPY_B2B, p_copy_b2b) = (p_copy_b2b){arg_x(c, 1), arg_x(c, 2), arg_x(c, 3)};
}
// void nvnCommandBufferUpdateUniformBuffer(NVNcommandBuffer *cmd, NVNbufferAddress buffer, size_t buffer_size,
// ptrdiff_t offset, size_t size, const void *data): the data is copied at record time (the caller may change it
// afterwards) and written into the buffer at replay
NVN_API(nvnCommandBufferUpdateUniformBuffer) {
    uint64_t dst = arg_x(c, 1) + arg_x(c, 3), n = arg_x(c, 4);
    const void *data = arg_p(c, 5);
    if (n > (16u << 20)) rt_fatal("nvn: UpdateUniformBuffer of %#llx bytes", (unsigned long long)n);
    p_update_ub *r = rec(cmd_of(c), OP_UPDATE_UB, offsetof(p_update_ub, data) + n);
    r->dst = dst;
    r->size = n;
    memcpy(r->data, data, n);
}

// ---------- CommandBuffer: render state, payloads of nvn_obj.h
// void nvnCommandBufferSetRenderTargets(NVNcommandBuffer *cmd, int n, const NVNtexture *const *colors,
// const NVNtextureView *const *color_views, const NVNtexture *depth, const NVNtextureView *depth_view)
// DEBUG-BIGLOG: REKARINA_BIGLOG=1: API calls that name a texture 8192 or more wide (OoT's 12288x12288 static shadow map)
static void big_api(const char *what, const tex_t *t, const Cpu *c, long a, long b) {
    static int on = -1, n;
    if (on < 0) on = cfg_get("REKARINA_BIGLOG") != NULL;
    if (!on || !t || t->d.width < 8192 || n >= 200) return;
    if (!n && cfg_get("REKARINA_BIGLOG_ARM_API")) {   // arm the frame dump (REKARINA_DUMP=t<never>) at the first such call
        void vk_dump_arm_soon(void);
        vk_dump_arm_soon();
    }
    n++;
    rt_log("BIGLOG api %s tex %p %ux%u fmt %#x lr %#llx frame %llu | %ld %ld", what, (const void *)t->mem, t->d.width,
           t->d.height, t->d.format, (unsigned long long)c->x[30],
           (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), a, b);
}

NVN_API(nvnCommandBufferSetRenderTargets) {
    int n = (int)arg_x(c, 1);
    const NVNtexture *const *ct = arg_p(c, 2);
    const NVNtextureView *const *cv = arg_p(c, 3);
    const NVNtexture *dt = arg_p(c, 4);
    const NVNtextureView *dv = arg_p(c, 5);
    if (n < 0 || n > 8) {
        rt_log("nvn: SetRenderTargets with %d colour targets", n);
        n = n < 0 ? 0 : 8;
    }
    p_set_rt *r = REC(cmd_of(c), OP_SET_RT, p_set_rt);
    r->n = n;
    for (int i = 0; i < n; i++) {
        r->color[i] = ct ? tex_pin(ct[i]) : NULL;
        copy_view(&r->cview[i], &r->has_cview[i], cv ? cv[i] : NULL);
    }
    r->depth = tex_pin(dt);
    copy_view(&r->dview, &r->has_dview, dv);
    big_api("set-rt-depth", r->depth, c, r->has_dview, r->dview.base_level);   // DEBUG-BIGLOG
    for (int i = 0; i < n; i++) big_api("set-rt-color", r->color[i], c, i, n);   // DEBUG-BIGLOG
}
// void nvnCommandBufferClearColor(NVNcommandBuffer *cmd, int index, const float *rgba, int mask)
NVN_API(nvnCommandBufferClearColor) {
    p_clear_color *r = REC(cmd_of(c), OP_CLEAR_COLOR, p_clear_color);
    r->index = (int)arg_x(c, 1);
    r->mask = (int)arg_x(c, 3);
    memcpy(r->rgba, arg_p(c, 2), 16);
}
// void nvnCommandBufferClearDepthStencil(NVNcommandBuffer *cmd, float depth, NVNboolean depth_mask, int stencil,
// int stencil_mask)
NVN_API(nvnCommandBufferClearDepthStencil) {
    *REC(cmd_of(c), OP_CLEAR_DS, p_clear_ds) = (p_clear_ds){arg_s(c, 0), nvn_arg_bool(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3)};
}
static void set_viewports(cmd_t *o, int first, int n, const float *v) {
    if (first < 0 || n < 0 || first + n > 16) {
        rt_log("nvn: SetViewports first %d n %d", first, n);
        return;
    }
    p_viewports *r = REC(o, OP_VIEWPORTS, p_viewports);
    r->first = first;
    r->n = n;
    for (int i = 0; i < n; i++) memcpy(r->v[i], v + 4 * i, 16);
}
// void nvnCommandBufferSetViewports(NVNcommandBuffer *cmd, int first, int n, const float *xywh)
NVN_API(nvnCommandBufferSetViewports) { set_viewports(cmd_of(c), (int)arg_x(c, 1), (int)arg_x(c, 2), arg_p(c, 3)); }
// void nvnCommandBufferSetViewport(NVNcommandBuffer *cmd, int x, int y, int w, int h)
NVN_API(nvnCommandBufferSetViewport) {
    float v[4] = {(float)(int)arg_x(c, 1), (float)(int)arg_x(c, 2), (float)(int)arg_x(c, 3), (float)(int)arg_x(c, 4)};
    set_viewports(cmd_of(c), 0, 1, v);
}
static void set_scissor(cmd_t *o, int x, int y, int w, int h) { *REC(o, OP_SCISSOR, p_scissor) = (p_scissor){x, y, w, h}; }
// void nvnCommandBufferSetScissor(NVNcommandBuffer *cmd, int x, int y, int w, int h)
NVN_API(nvnCommandBufferSetScissor) {
    set_scissor(cmd_of(c), (int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), (int)arg_x(c, 4));
}
// void nvnCommandBufferSetScissors(NVNcommandBuffer *cmd, int first, int n, const int *xywh): the replay keeps one
// scissor (index 0); other indices matter only to shaders that select a viewport index, so differing ones are reported
NVN_API(nvnCommandBufferSetScissors) {
    int first = (int)arg_x(c, 1), n = (int)arg_x(c, 2);
    const int *r = arg_p(c, 3);
    if (first < 0 || n < 0 || first + n > 16) rt_fatal("nvn: SetScissors first %d n %d", first, n);
    for (int i = 1; i < n; i++)
        if (memcmp(r + 4 * i, r, 16)) {
            static int once;
            if (!once++) rt_log("nvn: SetScissors: differing scissor %d ignored (single-scissor replay)", first + i);
        }
    if (first == 0 && n > 0) set_scissor(cmd_of(c), r[0], r[1], r[2], r[3]);
}
static void set_depth_range(cmd_t *o, float n, float f) { *REC(o, OP_DEPTH_RANGE, p_depth_range) = (p_depth_range){n, f}; }
// void nvnCommandBufferSetDepthRange(NVNcommandBuffer *cmd, float near, float far)
NVN_API(nvnCommandBufferSetDepthRange) { set_depth_range(cmd_of(c), arg_s(c, 0), arg_s(c, 1)); }
// void nvnCommandBufferSetDepthRanges(NVNcommandBuffer *cmd, int first, int n, const float *ranges)
NVN_API(nvnCommandBufferSetDepthRanges) {
    int first = (int)arg_x(c, 1), n = (int)arg_x(c, 2);
    const float *r = arg_p(c, 3);
    if (first < 0 || n < 0 || first + n > 16) rt_fatal("nvn: SetDepthRanges first %d n %d", first, n);
    for (int i = 1; i < n; i++)
        if (memcmp(r + 2 * i, r, 8)) {
            static int once;
            if (!once++) rt_log("nvn: SetDepthRanges: differing range %d ignored (single-range replay)", first + i);
        }
    if (first == 0 && n > 0) set_depth_range(cmd_of(c), r[0], r[1]);
}
// void nvnCommandBufferSetDepthClamp / SetRenderEnable / SetRasterizerDiscard(NVNcommandBuffer *cmd, NVNboolean enable)
NVN_API(nvnCommandBufferSetDepthClamp) { REC(cmd_of(c), OP_DEPTH_CLAMP, p_flag)->value = nvn_arg_bool(c, 1) != 0; }
NVN_API(nvnCommandBufferSetRenderEnable) { REC(cmd_of(c), OP_RENDER_ENABLE, p_flag)->value = nvn_arg_bool(c, 1) != 0; }
// Rasterizer discard runs vertex work without rasterising; with no transform feedback that equals not drawing.
NVN_API(nvnCommandBufferSetRasterizerDiscard) { REC(cmd_of(c), OP_RASTER_DISCARD, p_flag)->value = nvn_arg_bool(c, 1) != 0; }
// void nvnCommandBufferSetPrimitiveRestart(NVNcommandBuffer *cmd, NVNboolean enable, int index): TOTK draws its
// terrain and cliff strips with 16-bit indices and restart at 0xffff; the replay applies it per indexed draw (op_draw_idx)
NVN_API(nvnCommandBufferSetPrimitiveRestart) {
    *REC(cmd_of(c), OP_PRIM_RESTART, p_prim_restart) = (p_prim_restart){nvn_arg_bool(c, 1) != 0, (uint32_t)arg_x(c, 2)};
}
// void nvnCommandBufferSetRenderEnableConditional(NVNcommandBuffer *cmd, NVNconditionalRenderMode mode,
// NVNbufferAddress counters): rendering is enabled when the two NVNcounterData.counter values at addr / addr+16
// compare equal (mode 0) or not equal (mode 1). TOTK gates most of its ClearColor commands with it; the replay
// evaluates it in GPU order.
NVN_API(nvnCommandBufferSetRenderEnableConditional) {
    int m = (int)arg_x(c, 1);
    if (m != 0 && m != 1) rt_fatal("nvn: SetRenderEnableConditional: unknown mode %d", m);
    p_render_cond *r = REC(cmd_of(c), OP_RENDER_COND, p_render_cond);
    r->mode = m;
    r->addr = arg_x(c, 2);
}
// void nvnCommandBufferBindProgram(NVNcommandBuffer *cmd, const NVNprogram *program, int stages)
NVN_API(nvnCommandBufferBindProgram) {
    *REC(cmd_of(c), OP_BIND_PROGRAM, p_bind_program) = (p_bind_program){arg_p(c, 1), (int)arg_x(c, 2)};
}
// void nvnCommandBufferBind{Blend,Color,ChannelMask,Multisample,Polygon,DepthStencil}State(NVNcommandBuffer *cmd,
// const NVN*State *state): the state object's bytes are copied (p_bind_state.kind order)
static void bind_state(const Cpu *c, int kind, size_t n) {
    p_bind_state *r = REC(cmd_of(c), OP_BIND_STATE, p_bind_state);
    r->kind = kind;
    memcpy(r->bytes, arg_p(c, 1), n);
}
NVN_API(nvnCommandBufferBindBlendState) { bind_state(c, 0, sizeof(blend_st)); }
NVN_API(nvnCommandBufferBindColorState) { bind_state(c, 1, sizeof(color_st)); }
NVN_API(nvnCommandBufferBindChannelMaskState) { bind_state(c, 2, sizeof(chmask_st)); }
NVN_API(nvnCommandBufferBindMultisampleState) { bind_state(c, 3, sizeof(ms_st)); }
NVN_API(nvnCommandBufferBindPolygonState) { bind_state(c, 4, sizeof(poly_st)); }
NVN_API(nvnCommandBufferBindDepthStencilState) { bind_state(c, 5, sizeof(ds_st)); }
// void nvnCommandBufferSetPointSize(NVNcommandBuffer *cmd, float size): recorded as bind-state kind 6 with the float as payload
NVN_API(nvnCommandBufferSetPointSize) {
    p_bind_state *r = REC(cmd_of(c), OP_BIND_STATE, p_bind_state);
    float s = arg_s(c, 0);
    r->kind = 6;
    memcpy(r->bytes, &s, sizeof s);
}
// void nvnCommandBufferBindVertexAttribState(NVNcommandBuffer *cmd, int n, const NVNvertexAttribState *states)
NVN_API(nvnCommandBufferBindVertexAttribState) {
    cmd_t *o = cmd_of(c);
    int n = (int)arg_x(c, 1);
#ifdef GAME_NS2
    // NVN2 has 32 attributes (nvnDeviceGetInteger 0x0E); the replay keeps 16 (VkPipeState.vattr), so attributes 16-31
    // are dropped, said once when one of them has a format (seedds/nxnative accepts 32: Yoshi binds 17).
    if (n > 16 && n <= 32) {
        const uint32_t *all = arg_p(c, 2);
        for (int i = 16; i < n; i++)
            if (((const va_st *)&all[i])->fmt) {
                static uint8_t once;
                if (!__atomic_exchange_n(&once, 1, __ATOMIC_RELAXED))
                    rt_log("nvn: BindVertexAttribState: attribute %d of %d has a format; the replay keeps 16", i, n);
                break;
            }
        n = 16;
    }
#endif
    if (n < 0 || n > 16) { rt_log("nvn: BindVertexAttribState n %d", n); return; }
    p_bind_vattr *r = RECN(o, OP_BIND_VATTR, p_bind_vattr, st, n);
    r->first = 0;
    r->n = n;
    memcpy(r->st, arg_p(c, 2), 4 * (size_t)n);
    hw_fix(o, 68);   // the driver always writes all 16 attribute formats: 17 words
}
// void nvnCommandBufferBindVertexStreamState(NVNcommandBuffer *cmd, int n, const NVNvertexStreamState *states)
NVN_API(nvnCommandBufferBindVertexStreamState) {
    cmd_t *o = cmd_of(c);
    int n = (int)arg_x(c, 1);
    if (n < 0 || n > 16) { rt_log("nvn: BindVertexStreamState n %d", n); return; }
    p_bind_vstream *r = RECN(o, OP_BIND_VSTREAM, p_bind_vstream, st, n);
    r->first = 0;
    r->n = n;
    memcpy(r->st, arg_p(c, 2), 8 * (size_t)n);
    hw_fix(o, 4 * (16 + (size_t)n));
}
static void bind_vertex_buffers(cmd_t *o, int first, int n, const NVNbufferRange *rg) {
#ifdef GAME_NS2
    // NVN2 has 32 streams; null ranges past the replay's 16 are what games bind to clear them (seedds/nxnative: Mario
    // Tennis Fever clears 32 before each model). Nothing can read those.
    if (first >= 16 && n >= 0 && first + n <= 32) {
        int real = 0;
        for (int i = 0; i < n; i++) real |= rg[i].address || rg[i].size;
        if (!real) return;
    }
#endif
    if (first < 0 || n < 0 || first + n > 16) { rt_log("nvn: BindVertexBuffers first %d n %d", first, n); return; }
    p_bind_vbuf *r = RECN(o, OP_BIND_VBUF, p_bind_vbuf, r, n);
    r->first = first;
    r->n = n;
    memcpy(r->r, rg, sizeof *rg * (size_t)n);
    hw_fix(o, 24 * (size_t)n);
}
// void nvnCommandBufferBindVertexBuffers(NVNcommandBuffer *cmd, int first, int n, const NVNbufferRange *ranges)
NVN_API(nvnCommandBufferBindVertexBuffers) { bind_vertex_buffers(cmd_of(c), (int)arg_x(c, 1), (int)arg_x(c, 2), arg_p(c, 3)); }
// void nvnCommandBufferBindVertexBuffer(NVNcommandBuffer *cmd, int index, NVNbufferAddress address, size_t size)
NVN_API(nvnCommandBufferBindVertexBuffer) {
    NVNbufferRange r = {arg_x(c, 2), arg_x(c, 3)};
    bind_vertex_buffers(cmd_of(c), (int)arg_x(c, 1), 1, &r);
}
int scene_projection_extent(const void *texture, const uint8_t raw_view[16], uint32_t *w, uint32_t *h) {
    const tex_t *t = texture;
    view_t view;
    memcpy(&view, raw_view, sizeof view);
    unsigned level = view.set_levels ? view.base_level : 0;
    if (!t || level >= t->d.levels || level >= 32 || !t->d.width || !t->d.height) return 0;
    unsigned tw = t->d.width >> level, th = t->d.height >> level;
    *w = tw ? tw : 1;
    *h = th ? th : 1;
    return 1;
}

int scene_projection_describe(p_scene_projection *out, const SceneProjection *sample, const void *color,
                              const void *depth, const uint8_t cv[16], const uint8_t dv[16]) {
    const tex_t *ct = color, *dt = depth;
    *out = (p_scene_projection){.sample = *sample};
    if (!ct || !dt || !ct->serial || !dt->serial) return 0;
    out->color_serial = ct->serial;
    out->depth_serial = dt->serial;
    memcpy(&out->cv, cv, sizeof out->cv);
    memcpy(&out->dv, dv, sizeof out->dv);
    return 1;
}

int scene_projection_record(void *command, const SceneProjection *sample, const void *color,
                            const void *depth, const uint8_t cv[16], const uint8_t dv[16]) {
    cmd_t *o = command;
    p_scene_projection p;
    if (!o || o->magic != CMD_MAGIC || !o->recording || !scene_projection_describe(&p, sample, color, depth, cv, dv)) return 0;
    *REC(o, OP_SCENE_PROJECTION, p_scene_projection) = p;
    return 1;
}

static void record_buffer(cmd_t *o, uint32_t op, int stage, int index, uint64_t address, uint64_t size) {
    const p_bind_buf b = {stage, index, address, size};
    p_projection_bind projection = {.bind = b};
    if (op == OP_BIND_UBO && !stage && !index && address && size == SCENE_CAMERA_BYTES &&
        scene_projection_enabled() && scene_projection_binding(address, &projection.scene)) *REC(o, op, p_projection_bind) = projection;
    else *REC(o, op, p_bind_buf) = b;
}

// Uniform and storage buffers: one record per binding (CB_SIZE/ADDRESS 4 words + CB_BIND 1 on hardware).
static void bind_buffers(const Cpu *c, uint32_t op) {   // (cmd, stage, first, n, const NVNbufferRange *ranges)
    cmd_t *o = cmd_of(c);
    int st = (int)arg_x(c, 1), first = (int)arg_x(c, 2), n = (int)arg_x(c, 3);
    const NVNbufferRange *rg = arg_p(c, 4);
    st = nvn_stage_slot(st);
    for (int i = 0; i < n; i++) {
        record_buffer(o, op, st, first + i, rg[i].address, rg[i].size);
        hw_fix(o, 20);
    }
}
static void bind_buffer(const Cpu *c, uint32_t op) {    // (cmd, stage, index, NVNbufferAddress address, size_t size)
    cmd_t *o = cmd_of(c);
    record_buffer(o, op, nvn_stage_slot((int)arg_x(c, 1)), (int)arg_x(c, 2), arg_x(c, 3), arg_x(c, 4));
    hw_fix(o, 20);
}
NVN_API(nvnCommandBufferBindUniformBuffer) { bind_buffer(c, OP_BIND_UBO); }
NVN_API(nvnCommandBufferBindUniformBuffers) { bind_buffers(c, OP_BIND_UBO); }
NVN_API(nvnCommandBufferBindStorageBuffer) { bind_buffer(c, OP_BIND_SSBO); }
NVN_API(nvnCommandBufferBindStorageBuffers) { bind_buffers(c, OP_BIND_SSBO); }
// Texture, image and separate handle binds. A deferred bind (NVN 55 *Deferred entry points, the fallback of the inline
// fast paths) keeps the pointer to the game's handle array: the original driver (sdk+0x46cc00 BindTexturesDeferred)
// reads the handles when the command buffer is submitted, and the game fills its material handle tables after
// recording and before the submit (nvn_submit_prepare). hw_bytes: the hardware size of the plain variant.
static void bind_handles(cmd_t *o, uint32_t op, int st, int first, int n, const uint64_t *h, int deferred, size_t hw_bytes) {
    if (n < 0 || n > 32) { rt_log("nvn: bind of %d handles", n); return; }
    p_bind_handles *r = RECN(o, op, p_bind_handles, h, n);
    r->stage = nvn_stage_slot(st);
    r->first = first;
    r->n = n;
    r->src = deferred ? (uint64_t)(uintptr_t)h : 0;
    memcpy(r->h, h, 8 * (size_t)n);
#ifdef GAME_NS2
    // The driver writes each handle into the c2 table as a 32-bit word (texture ID bits 0-19, sampler ID 20-31): the
    // GPU never sees the upper half (seedds/nxnative hle_nvn.c bind_handles).
    for (int i = 0; i < n; i++) r->h[i] = (uint32_t)r->h[i];
#endif
    hw_fix(o, hw_bytes ? hw_bytes : 16 + 4 * (size_t)n);   // macro 7 + method 0x1a2c + CB_POS + one word per binding
}
static void bind_handle_array(const Cpu *c, uint32_t op, int deferred, int wide) {   // (cmd, stage, first, n, handles)
    int n = (int)arg_x(c, 3);
    bind_handles(cmd_of(c), op, (int)arg_x(c, 1), (int)arg_x(c, 2), n, arg_p(c, 4), deferred,
                 wide && n >= 0 && n <= 32 ? 16 + 8 * (size_t)n : 0);
}
static void bind_one_handle(const Cpu *c, uint32_t op, size_t hw_bytes) {   // (cmd, stage, index, handle)
    uint64_t v = arg_x(c, 3);
    bind_handles(cmd_of(c), op, (int)arg_x(c, 1), (int)arg_x(c, 2), 1, &v, 0, hw_bytes);
}
// Combined texture handles share the texture slots with separate textures (sampler id in bits 20 and up). Images
// (writable textures): handle = texture pool index; the game does not call RegisterImage, images and textures share
// their registration.
NVN_API(nvnCommandBufferBindTexture) { bind_one_handle(c, OP_BIND_TEX, 0); }
NVN_API(nvnCommandBufferBindTextures) { bind_handle_array(c, OP_BIND_TEX, 0, 0); }
NVN_API(nvnCommandBufferBindImage) { bind_one_handle(c, OP_BIND_IMG, 24); }
NVN_API(nvnCommandBufferBindImages) { bind_handle_array(c, OP_BIND_IMG, 0, 1); }
NVN_API(nvnCommandBufferBindSeparateTexture) { bind_one_handle(c, OP_BIND_TEX, 24); }
NVN_API(nvnCommandBufferBindSeparateTextures) { bind_handle_array(c, OP_BIND_TEX, 0, 1); }
NVN_API(nvnCommandBufferBindSeparateSampler) { bind_one_handle(c, OP_BIND_SAMP, 24); }
NVN_API(nvnCommandBufferBindSeparateSamplers) { bind_handle_array(c, OP_BIND_SAMP, 0, 1); }
// void nvnCommandBufferBind{Textures,Images,SeparateTextures,SeparateSamplers}Deferred(NVNcommandBuffer *cmd,
// NVNshaderStage stage, int first, int n, const T *handles) (NVN 55, no prototype in nvn.h)
NVN_API(nvnCommandBufferBindTexturesDeferred) { bind_handle_array(c, OP_BIND_TEX, 1, 0); }
NVN_API(nvnCommandBufferBindImagesDeferred) { bind_handle_array(c, OP_BIND_IMG, 1, 1); }
NVN_API(nvnCommandBufferBindSeparateTexturesDeferred) { bind_handle_array(c, OP_BIND_TEX, 1, 1); }
NVN_API(nvnCommandBufferBindSeparateSamplersDeferred) { bind_handle_array(c, OP_BIND_SAMP, 1, 1); }
// void nvnCommandBufferBind{Uniform,Storage}BuffersDeferred(NVNcommandBuffer *cmd, NVNshaderStage stage, int first,
// int n, const NVNbufferRange *ranges): buffer ranges still bind at record time
NVN_API(nvnCommandBufferBindUniformBuffersDeferred) {
    static int once;
    if (!once++) NVN_LOG("BindUniformBuffersDeferred: buffer ranges bind at record time");
    bind_buffers(c, OP_BIND_UBO);
}
NVN_API(nvnCommandBufferBindStorageBuffersDeferred) {
    static int once;
    if (!once++) NVN_LOG("BindStorageBuffersDeferred: buffer ranges bind at record time");
    bind_buffers(c, OP_BIND_SSBO);
}
// void nvnCommandBufferSetStencil{ValueMask,Mask,Ref}(NVNcommandBuffer *cmd, NVNface face, int value)
NVN_API(nvnCommandBufferSetStencilValueMask) {
    *REC(cmd_of(c), OP_STENCIL, p_stencil) = (p_stencil){(int)arg_x(c, 1), (int)arg_x(c, 2), 0, 0, 0};
}
NVN_API(nvnCommandBufferSetStencilMask) {
    *REC(cmd_of(c), OP_STENCIL, p_stencil) = (p_stencil){(int)arg_x(c, 1), 0, (int)arg_x(c, 2), 0, 1};
}
NVN_API(nvnCommandBufferSetStencilRef) {
    *REC(cmd_of(c), OP_STENCIL, p_stencil) = (p_stencil){(int)arg_x(c, 1), 0, 0, (int)arg_x(c, 2), 2};
}
// void nvnCommandBufferSetPolygonOffsetClamp(NVNcommandBuffer *cmd, float factor, float units, float clamp)
NVN_API(nvnCommandBufferSetPolygonOffsetClamp) {
    *REC(cmd_of(c), OP_POLY_OFFSET, p_poly_offset) = (p_poly_offset){arg_s(c, 0), arg_s(c, 1), arg_s(c, 2)};
}
static unsigned recorded_draw_role(uint64_t return_address) {
    return scene_bloom_draw_role(return_address) | scene_output_draw_role(return_address) |
           (scene_interface_draw_active() ? DRAW_ROLE_INTERFACE : 0);
}

// void nvnCommandBufferDrawArrays(NVNcommandBuffer *cmd, NVNdrawPrimitive prim, int first, int count)
NVN_API(nvnCommandBufferDrawArrays) {
    *REC(cmd_of(c), OP_DRAW, p_draw) = (p_draw){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), 0, 1, recorded_draw_role(c->x[30])};
}
// void nvnCommandBufferDrawArraysInstanced(NVNcommandBuffer *cmd, NVNdrawPrimitive prim, int first, int count,
// int base_instance, int instances): zero instances draw nothing
NVN_API(nvnCommandBufferDrawArraysInstanced) {
    int inst = (int)arg_x(c, 5);
    if (inst <= 0) return;
    *REC(cmd_of(c), OP_DRAW, p_draw) = (p_draw){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), (int)arg_x(c, 4), inst, recorded_draw_role(c->x[30])};
}
// void nvnCommandBufferDrawElements(NVNcommandBuffer *cmd, NVNdrawPrimitive prim, NVNindexType type, int count,
// NVNbufferAddress indices)
NVN_API(nvnCommandBufferDrawElements) {
    cmd_t *o = cmd_of(c);
    *REC(o, OP_DRAW_IDX, p_draw_idx) = (p_draw_idx){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), 0, 0, 1, arg_x(c, 4), recorded_draw_role(c->x[30])};
    hw_fix(o, 48);
}
// void nvnCommandBufferDrawElementsBaseVertex(..., NVNbufferAddress indices, int base_vertex)
NVN_API(nvnCommandBufferDrawElementsBaseVertex) {
    cmd_t *o = cmd_of(c);
    *REC(o, OP_DRAW_IDX, p_draw_idx) =
        (p_draw_idx){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), (int)arg_x(c, 5), 0, 1, arg_x(c, 4), recorded_draw_role(c->x[30])};
    hw_fix(o, 44);
}
// void nvnCommandBufferDrawElementsInstanced(..., NVNbufferAddress indices, int base_vertex, int base_instance,
// int instances): assumed to use the 6-parameter draw macro of the base-vertex form (unverified)
NVN_API(nvnCommandBufferDrawElementsInstanced) {
    cmd_t *o = cmd_of(c);
    *REC(o, OP_DRAW_IDX, p_draw_idx) = (p_draw_idx){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3),
                                                    (int)arg_x(c, 5), (int)arg_x(c, 6), (int)arg_x(c, 7), arg_x(c, 4), recorded_draw_role(c->x[30])};
    hw_fix(o, 44);
}
// void nvnCommandBufferDrawElementsIndirect(NVNcommandBuffer *cmd, NVNdrawPrimitive prim, NVNindexType type,
// NVNbufferAddress indices, NVNbufferAddress indirect): the arguments at indirect are often GPU-written, so the replay
// draws with vkCmdDrawIndexedIndirect from the pool buffer (nxnative hle_nvn.c, SavageWindRecomp cmdbuf.c)
NVN_API(nvnCommandBufferDrawElementsIndirect) {
    *REC(cmd_of(c), OP_DRAW_IDX_IND, p_draw_idx_ind) = (p_draw_idx_ind){(int)arg_x(c, 1), (int)arg_x(c, 2), arg_x(c, 3), arg_x(c, 4), scene_interface_draw_active() ? DRAW_ROLE_INTERFACE : 0};
}
// void nvnCommandBufferDispatchCompute(NVNcommandBuffer *cmd, int x, int y, int z): caller = the game's return address
NVN_API(nvnCommandBufferDispatchCompute) {
    {   // DEBUG-DISPLR: REKARINA_DISPLR=<x>: the guest call chain (frame pointers) of dispatches of that width, each new chain once
        static int want = -2;
        static uint64_t seen[32];
        static int ns;
        if (want == -2) { const char *e = cfg_get("REKARINA_DISPLR"); want = e && *e ? atoi(e) : -1; }
        if (want >= 0 && (int)arg_x(c, 1) == want && ns < 32) {
            uint64_t ch[10], h = 0, fp = c->x[29];
            int k = 0;
            ch[k++] = c->x[30];
            while (k < 10 && fp >= 0x1000 && !(fp & 7) && fp < (1ull << 40)) {
                const uint64_t *f = (const uint64_t *)(uintptr_t)fp;
                ch[k++] = f[1] & 0xffffffffffffull, fp = f[0];
            }
            for (int i = 0; i < k; i++) h = h * 31 + ch[i];
            int dup = 0;
            for (int i = 0; i < ns; i++) dup |= seen[i] == h;
            if (!dup) {
                seen[ns++] = h;
                char line[256];
                int o = 0;
                for (int i = 0; i < k; i++) o += snprintf(line + o, sizeof line - (size_t)o, " %#llx", (unsigned long long)ch[i]);
                rt_log("DISPLR x %d y %d:%s", (int)arg_x(c, 1), (int)arg_x(c, 2), line);
            }
        }
    }
    *REC(cmd_of(c), OP_DISPATCH, p_dispatch) = (p_dispatch){(int)arg_x(c, 1), (int)arg_x(c, 2), (int)arg_x(c, 3), c->x[30]};
}
// void nvnCommandBufferDispatchComputeIndirect(NVNcommandBuffer *cmd, NVNbufferAddress indirect): vkCmdDispatchIndirect
NVN_API(nvnCommandBufferDispatchComputeIndirect) {
    *REC(cmd_of(c), OP_DISPATCH_IND, p_dispatch_ind) = (p_dispatch_ind){arg_x(c, 1), c->x[30]};
}
// void nvnCommandBufferClearBuffer(NVNcommandBuffer *cmd, NVNbufferAddress addr, size_t size, uint32_t value)
NVN_API(nvnCommandBufferClearBuffer) {
    *REC(cmd_of(c), OP_CLEAR_BUF, p_clear_buf) = (p_clear_buf){arg_x(c, 1), arg_x(c, 2), (uint32_t)arg_x(c, 3), 0};
}
// void nvnCommandBufferReportValue(NVNcommandBuffer *cmd, uint32_t value, NVNbufferAddress addr): the value lands when
// the GPU reaches this point. Recorded as a 4-byte ClearBuffer, so it stays in GPU order with the ClearBuffer resets of
// the same counter (main+0x7b9274 modes 1-3) and works in CPU-invisible pools; the CPU sees it after the fence, as a
// counter report.
NVN_API(nvnCommandBufferReportValue) {
    *REC(cmd_of(c), OP_CLEAR_BUF, p_clear_buf) = (p_clear_buf){arg_x(c, 2), 4, (uint32_t)arg_x(c, 1), 0};
}
// Colour reduction is a Maxwell bandwidth hint for blending into compressed targets: no effect on the result.
// void nvnCommandBufferSetColorReductionEnable(cmd, NVNboolean enable), SetColorReductionThresholds(cmd, class, const int *)
NVN_API(nvnCommandBufferSetColorReductionEnable) { (void)c; }
NVN_API(nvnCommandBufferSetColorReductionThresholds) { (void)c; }
// void nvnCommandBufferBarrier(NVNcommandBuffer *cmd, int bits)
NVN_API(nvnCommandBufferBarrier) { REC(cmd_of(c), OP_BARRIER, p_barrier)->bits = (int)arg_x(c, 1); }
// void nvnCommandBufferSetTexturePool / SetSamplerPool(NVNcommandBuffer *cmd, const NVNtexturePool / NVNsamplerPool *)
NVN_API(nvnCommandBufferSetTexturePool) { REC(cmd_of(c), OP_SET_TEX_POOL, p_set_pool)->pool = arg_p(c, 1); }
NVN_API(nvnCommandBufferSetSamplerPool) { REC(cmd_of(c), OP_SET_SAMP_POOL, p_set_pool)->pool = arg_p(c, 1); }
// void nvnCommandBufferCopyBufferToTexture(NVNcommandBuffer *cmd, NVNbufferAddress src, const NVNtexture *dst,
// const NVNtextureView *view, const NVNcopyRegion *region, int flags)
NVN_API(nvnCommandBufferCopyBufferToTexture) {
    p_copy_b2t *o = REC(cmd_of(c), OP_COPY_B2T, p_copy_b2t);
    o->src = arg_x(c, 1);
    o->tex = tex_pin(arg_p(c, 2));
    copy_view(&o->view, &o->has_view, arg_p(c, 3));
    o->r = *(const NVNcopyRegion *)arg_p(c, 4);
    o->flags = (int)arg_x(c, 5);
    big_api("copy-b2t", o->tex, c, o->r.xoffset, o->r.yoffset);   // DEBUG-BIGLOG
}
// void nvnCommandBufferCopyTextureToBuffer(NVNcommandBuffer *cmd, const NVNtexture *src, const NVNtextureView *view,
// const NVNcopyRegion *region, NVNbufferAddress dst, int flags): p_copy_b2t with src = the destination address
NVN_API(nvnCommandBufferCopyTextureToBuffer) {
    p_copy_b2t *o = REC(cmd_of(c), OP_COPY_T2B, p_copy_b2t);
    o->src = arg_x(c, 4);
    o->tex = tex_pin(arg_p(c, 1));
    copy_view(&o->view, &o->has_view, arg_p(c, 2));
    o->r = *(const NVNcopyRegion *)arg_p(c, 3);
    o->flags = (int)arg_x(c, 5);
}
// void nvnCommandBufferClearTexture{,ui}(NVNcommandBuffer *cmd, const NVNtexture *tex, const NVNtextureView *view,
// const NVNcopyRegion *region, const float / uint32_t *rgba, int mask)
static void clear_texture(const Cpu *c, int is_uint) {
    p_clear_tex *o = REC(cmd_of(c), OP_CLEAR_TEX, p_clear_tex);
    o->tex = tex_pin(arg_p(c, 1));
    copy_view(&o->view, &o->has_view, arg_p(c, 2));
    o->r = *(const NVNcopyRegion *)arg_p(c, 3);
    o->mask = (int)arg_x(c, 5);
    o->is_uint = (uint8_t)is_uint;
    memcpy(o->c.u, arg_p(c, 4), 16);
}
NVN_API(nvnCommandBufferClearTexture) { clear_texture(c, 0); }
NVN_API(nvnCommandBufferClearTextureui) { clear_texture(c, 1); }
// void nvnCommandBufferCopyTextureToTexture(NVNcommandBuffer *cmd, const NVNtexture *src, const NVNtextureView *src_view,
// const NVNcopyRegion *src_region, const NVNtexture *dst, const NVNtextureView *dst_view,
// const NVNcopyRegion *dst_region, int flags)
NVN_API(nvnCommandBufferCopyTextureToTexture) {
    p_copy_t2t *o = REC(cmd_of(c), OP_COPY_T2T, p_copy_t2t);
    o->src = tex_pin(arg_p(c, 1));
    o->dst = tex_pin(arg_p(c, 4));
    copy_view(&o->sv, &o->has_sv, arg_p(c, 2));
    copy_view(&o->dv, &o->has_dv, arg_p(c, 5));
    o->sr = *(const NVNcopyRegion *)arg_p(c, 3);
    o->dr = *(const NVNcopyRegion *)arg_p(c, 6);
    o->flags = (int)arg_x(c, 7);
    big_api("copy-t2t-dst", o->dst, c, o->dr.xoffset, o->dr.yoffset);   // DEBUG-BIGLOG
    big_api("copy-t2t-src", o->src, c, o->sr.xoffset, o->sr.yoffset);   // DEBUG-BIGLOG
    {   // DEBUG-LUMLOG: the game code that copies OoT's 1x1 RGBA32F luminance targets (guest return address)
        static int on = -1, n;
        if (on < 0) on = cfg_get("REKARINA_LUMLOG") != NULL;
        if (on && o->dst && o->dst->d.format == 0x2e && o->dst->d.width == 1 && o->dst->d.height == 1 && n == 0) {
            void vk_dump_arm_soon(void);
            vk_dump_arm_soon();
        }
        if (on && o->dst && o->dst->d.format == 0x2e && o->dst->d.width == 1 && o->dst->d.height == 1 && (n++ < 4 || !(n % 300)))
        {
            char chain[256];
            int o2 = 0;
            uint64_t fp = c->x[29];
            for (int i = 0; i < 10 && fp && !(fp & 7) && o2 < 220; i++) {   // the guest's frame records: [fp] fp, [fp+8] lr
                const uint64_t *fr = (const uint64_t *)(uintptr_t)fp;
                o2 += snprintf(chain + o2, sizeof chain - (size_t)o2, " %#llx", (unsigned long long)fr[1]);
                fp = fr[0];
            }
            chain[o2] = 0;
            rt_log("LUMLOG copy#%d 1x1 %p -> %p lr %#llx frame %llu chain%s", n, o->src ? (const void *)o->src->mem : NULL,
                   (const void *)o->dst->mem, (unsigned long long)c->x[30],
                   (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), chain);
        }
    }
}
// void nvnCommandBufferDownsample(NVNcommandBuffer *cmd, const NVNtexture *src, const NVNtexture *dst): level 0 of src
// into level 0 of dst, whole extents, filtered. Recorded as a linear-filter texture copy: copy_t2t resolves a
// multisample source into an equal-size target and blits (2x2 average at half size) between sizes. As nxnative
// hle_nvn.c and SavageWindRecomp cmdbuf.c record it; docs/maxwell.md 1.19.
NVN_API(nvnCommandBufferDownsample) {
    p_copy_t2t *o = REC(cmd_of(c), OP_COPY_T2T, p_copy_t2t);
    o->src = tex_pin(arg_p(c, 1));
    o->dst = tex_pin(arg_p(c, 2));
    if (o->src) o->sr = (NVNcopyRegion){0, 0, 0, (int)o->src->d.width, (int)o->src->d.height, 1};
    if (o->dst) o->dr = (NVNcopyRegion){0, 0, 0, (int)o->dst->d.width, (int)o->dst->d.height, 1};
    o->flags = NVN_COPY_FLAGS_LINEAR_FILTER_BIT;
}
// void nvnCommandBufferSetCopyRowStride(NVNcommandBuffer *cmd, ptrdiff_t stride)
NVN_API(nvnCommandBufferSetCopyRowStride) { REC(cmd_of(c), OP_COPY_STRIDE, p_copy_stride)->stride = (int64_t)arg_x(c, 1); }
// void nvnCommandBufferSetBlendColor(NVNcommandBuffer *cmd, const float *rgba)
NVN_API(nvnCommandBufferSetBlendColor) { memcpy(REC(cmd_of(c), OP_BLEND_COLOR, p_blend_color)->rgba, arg_p(c, 1), 16); }

// void nvnCommandBufferWaitEvent(NVNcommandBuffer *cmd, const NVNevent *event, NVNeventWaitMode mode, uint32_t value):
// the GPU stops at this point until the event's word satisfies the mode. OoT records a wait for an event
// the CPU sets later (nvnEventSignal), after it has written the late constants of the frame, the light grid of the
// clustered lighting among them, so the replay waits there for the CPU (vk_exec_xfer.c wait_event).
NVN_API(nvnCommandBufferWaitEvent) {
    const ev_t *e = arg_p(c, 1);
    *REC(cmd_of(c), OP_WAIT_EVENT, p_wait_event) = (p_wait_event){e->val, (uint32_t)arg_x(c, 2), (uint32_t)arg_x(c, 3)};
}
// Commands without an effect on this replay (hle_nvn.c g_nops): WaitSync orders work against other queues, and every
// submit replays in order on one queue; shader scratch memory and the tessellation patch size have no host
// counterpart the replay needs.
// void nvnCommandBufferWaitSync(NVNcommandBuffer *cmd, const NVNsync *sync), SetShaderScratchMemory(cmd, const
// NVNmemoryPool *pool, ptrdiff_t offset, size_t size), SetPatchSize(cmd, int size)
NVN_API(nvnCommandBufferWaitSync) { (void)c; }
NVN_API(nvnCommandBufferSetShaderScratchMemory) { (void)c; }
NVN_API(nvnCommandBufferSetPatchSize) { (void)c; }

// ---------- submit
// What the console reads at the submit call is read here, on the game thread: every handle must be a live recording,
// and deferred binds take the handles the game's arrays hold now (the game fills its tables for the next frame as soon
// as the submit returns). The result is written into the recording copy.
static uint64_t def_reads, def_changed, n_clones;

static int resolve_deferred(const uint8_t *p, int depth) {   // returns the number of deferred binds
    int nd = 0;
    for (;;) {
        op_hdr h;
        memcpy(&h, p, sizeof h);
        if (h.size < 8 || (h.size & 7) || h.op >= OP_COUNT_)
            rt_fatal("nvn: corrupt recording at %p while resolving deferred binds (op %u size %u)", (const void *)p,
                     h.op, h.size);
        if (h.op == OP_END) return nd;
        if (h.op == OP_JUMP) {
            memcpy(&p, p + 8, 8);
            continue;
        }
        if (h.op == OP_CALL) {
            uint64_t t;
            memcpy(&t, p + 8, 8);
            if (depth >= 16) rt_fatal("nvn: deferred bind resolution: call depth over 16");
            nd += resolve_deferred((const uint8_t *)(uintptr_t)t, depth + 1);
#ifdef GAME_NS2
        } else if (h.op == OP_PUSH_CONST) {   // a *Deferred push reads its bytes when the command buffer is submitted
            p_push_const *b = (p_push_const *)(uintptr_t)(p + 8);
            if (b->src) {
                nd++;
                memcpy(b->data, (const void *)(uintptr_t)b->src, b->size);
            }
#endif
        } else if (h.op == OP_BIND_TEX || h.op == OP_BIND_IMG || h.op == OP_BIND_SAMP) {
            p_bind_handles *b = (p_bind_handles *)(uintptr_t)(p + 8);
            if (b->src && b->n > 0 && b->n <= 32) {
                nd++;
                uint64_t cur[32];
                memcpy(cur, (const void *)(uintptr_t)b->src, 8 * (size_t)b->n);
                __atomic_fetch_add(&def_reads, 1, __ATOMIC_RELAXED);
                if (memcmp(cur, b->h, 8 * (size_t)b->n)) {
                    if (__atomic_fetch_add(&def_changed, 1, __ATOMIC_RELAXED) < 6)
                        NVN_LOG("deferred bind read new handles at submit: stage %d slot %d, %#llx at record -> %#llx",
                                b->stage, b->first, (unsigned long long)b->h[0], (unsigned long long)cur[0]);
                    memcpy(b->h, cur, 8 * (size_t)b->n);
                }
            }
        }
        p += h.size;
    }
}

void nvn_submit_prepare(const uint64_t *handles, int n) {
    uint64_t now = __atomic_add_fetch(&nsubmits, 1, __ATOMIC_RELAXED);
    for (int i = 0; i < n; i++) copy_touch(handles[i], now, "QueueSubmitCommands", NULL);
    for (int i = 0; i < n; i++) resolve_deferred((const uint8_t *)(uintptr_t)handles[i], 0);
}

// Asynchronous replay. A copy a queued submission still reads keeps that submission's deferred binds: when it holds
// deferred binds (or is not known to hold none), this submission gets a clone, resolved here and freed after its replay.
void nvn_submit_prepare_async(const uint64_t *handles, uint64_t *out, int n) {
    uint64_t now = __atomic_add_fetch(&nsubmits, 1, __ATOMIC_RELAXED);
    uint64_t replayed = gfx_replayed_submits();
    plat_mutex_lock(&copy_lock);
    if (nzombies) zombie_sweep(replayed);
    for (int i = 0; i < n; i++) {
        copy_prefix *pf = copy_find(handles[i]);
        if (!pf) {
            plat_mutex_unlock(&copy_lock);
            rt_fatal("nvn: QueueSubmitCommands: command handle %#llx is not a live recording (released or never recorded)",
                     (unsigned long long)handles[i]);
        }
        out[i] = handles[i];
        if (pf->used_at > replayed && pf->used_at != now && pf->deferred != 1) {
            size_t sz = sizeof *pf + pf->bytes;
            copy_prefix *cl = malloc(sz);
            if (!cl) rt_fatal("nvn: out of memory (recording clone, %zu bytes)", sz);
            memcpy(cl, pf, sz);
            cl->ho = 0;   // a clone of the heap's
            if (cl->pins) pin_refs((const uint8_t *)(cl + 1), 1);
            cl->magic = 0;   // not a live handle: the game never sees it
            cl->used_at = now;
            zombie_add(cl);
            out[i] = (uintptr_t)(cl + 1);
            __atomic_fetch_add(&n_clones, 1, __ATOMIC_RELAXED);
        } else {
            pf->used_at = now;
        }
    }
    plat_mutex_unlock(&copy_lock);
    for (int i = 0; i < n; i++) {
        int nd = resolve_deferred((const uint8_t *)(uintptr_t)out[i], 0);
        if (out[i] == handles[i]) ((copy_prefix *)(uintptr_t)out[i] - 1)->deferred = nd ? 2 : 1;
    }
    if (now % 7200 == 0)
        rt_log("nvn: asynchronous replay at submit %llu: %llu recording clones so far, %u copies waiting for the replay",
               (unsigned long long)now, (unsigned long long)__atomic_load_n(&n_clones, __ATOMIC_RELAXED), nzombies);
}

uint64_t nvn_submit_count(void) { return __atomic_load_n(&nsubmits, __ATOMIC_RELAXED); }

const char *nvn_op_name(uint32_t op) {
    static const char *const names[] = {
        "END", "JUMP", "BEGIN", "CALL", "SIGNAL_EVENT", "REPORT_COUNTER", "COPY_B2B", "UPDATE_UB", "SET_RT",
        "CLEAR_COLOR", "CLEAR_DS", "VIEWPORTS", "SCISSOR", "DEPTH_RANGE", "BIND_PROGRAM", "BIND_STATE", "BIND_VATTR",
        "BIND_VSTREAM", "BIND_VBUF", "BIND_UBO", "BIND_SSBO", "BIND_TEX", "BIND_SAMP", "STENCIL", "POLY_OFFSET", "DRAW",
        "DRAW_IDX", "DISPATCH", "BARRIER", "SET_TEX_POOL", "SET_SAMP_POOL", "COPY_B2T", "COPY_STRIDE", "BLEND_COLOR",
        "COPY_T2B", "BIND_IMG", "COPY_T2T", "RESET_COUNTER", "CLEAR_TEX", "DEPTH_CLAMP", "RENDER_ENABLE",
        "RASTER_DISCARD", "RENDER_COND", "PRIM_RESTART", "DRAW_IDX_IND", "DISPATCH_IND", "CLEAR_BUF", "WAIT_EVENT",
#ifdef GAME_NS2
        "PUSH_CONST", "DRAW_MESH", "DRAW_MESH_IND", "FENCE_SYNC", "COPY_IMAGE_STRIDE",
#endif
        "SCENE_PROJECTION", "MODEL_SCOPE",
    };
    _Static_assert(sizeof names / sizeof *names == OP_COUNT_, "one name per op");
    return op < OP_COUNT_ ? names[op] : "?";
}
