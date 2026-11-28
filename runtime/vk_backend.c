// The Vulkan graphics back end: gfx_vulkan_backend wires the memory and
// texture layer (vk_memory.c, vk_texture.c), submission and fences (vk_sync.c) and the present. Every entry starts the
// device on first use.
//
// Present: the game texture is resolved on the calling thread and its blit is handed to a present thread, so
// nvnQueuePresentTexture returns at once (section 5). Queue order does the rest: the submissions that wrote the texture
// are already on gfx_queue, and every call that may write or free GPU state the blit reads (submit, finish, pool and
// texture teardown, texel writes) first waits for the blit submission, or for the flip if DLSS-G holds its inputs
// until present (vk_present_drain).
//
// Back-edge pacing: the present thread submits the blit at once, then holds the host present (vkQueuePresentKHR) until
// the frame's display time from the pacer (nvn_api_obj.c), so present-to-present intervals stay even whatever the
// replay took. Untagged frames can release replay after the blit; tagged frames keep it blocked through the flip.
//
// Asynchronous replay (REKARINA_REPLAY_ASYNC, default 1; 0 turns it off): nvnQueueSubmitCommands
// and nvnQueuePresentTexture only queue. Each item gets its timeline value at enqueue (vk_sync_reserve), in queue
// order; the replay thread runs the synchronous submit and present bodies in that order, so every value is signalled
// by the submission it was handed out for. The game runs at most one frame ahead: queueing present N waits until the
// replay thread handled present N-1. Calls that read or free GPU state that queued work may use drain the queue first
// (replay_drain), each cause counted.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "flight.h"
#include "ftrace.h"
#include "gfx.h"
#include "host.h"
#include "pacelog.h"   // DEBUG-PACELOG
#include "nvn_api.h"
#include "vk.h"
#include "vk_exec.h"
#include "platform/platform.h"
#include "vk_dlaa.h"
#include "vk_dlssg.h"
#include "vk_reflex.h"

void gfx_present_reflex_frame(uint64_t id);

#define WEAK __attribute__((weak))

typedef struct { VkImage image; VkFormat format; int w, h, has_crop, crop[4];
                 uint64_t frame, at, queued, value, reflex_id; VkGuestPq pq; const CfgRenderState *settings;
                 const char *nvidia_scene_aa; } PresentJob;

static PlatMutex pl = PLAT_MUTEX_INIT;
static PlatCond pcv = PLAT_COND_INIT;   // job queued, job taken, job submitted
static PresentJob job;
static int pending, busy, present_thread;

// Sleeps on a high-resolution timer to 1 ms before at, then yields until at: the timer alone wakes up to ~0.5 ms late.
static void wait_until(uint64_t at) {
    uint64_t now = kern_now_ns();
    if (at > now + 1500000) plat_wait_ns(at - now - 1000000);
    while (kern_now_ns() < at) plat_yield();
}

// The GPU finished the blit of a presented frame: the frame clock's "shown" moment (perf.c perf_shown; the earliest the
// display can show it) and the pacing log's.
// A held present (nvn_api_obj.c hold_until, the frame limit) is flipped after its blit is done: the later of the two.
void perf_shown(uint64_t frame, uint64_t ns);
static uint64_t flip_at[16];   // present thread: when each of the last 16 frames was flipped
static void frame_blit_done(void *arg) {
    uint64_t now = kern_now_ns(), frame = (uint64_t)(uintptr_t)arg;
    uint64_t g = gfx_blit_gpu_ns(frame);   // the GPU's own end of the blit; 0: unknown (REKARINA_FRAME_CLOCK=seen, no query)
    uint64_t fl = __atomic_load_n(&flip_at[frame % 16], __ATOMIC_ACQUIRE);
    perf_shown(frame, (g ? g : now) > fl ? (g ? g : now) : fl);
    if (__builtin_expect(pacelog_on(), 0)) {   // DEBUG-PACELOG
        pacelog_mark(PL_GPU, frame, now);
        if (g) pacelog_mark(PL_GTS, frame, g);
    }
}

static uint64_t present_release(const PresentJob *j) {
    uint64_t f = vk_sync_submit(NULL, 0);
    if (j->value && f != j->value)
        rt_fatal("vk: the present of frame %llu signalled %llu, queued as %llu", (unsigned long long)j->frame,
                 (unsigned long long)f, (unsigned long long)j->value);
    plat_mutex_lock(&pl);
    busy = 0;
    plat_cond_broadcast(&pcv);
    plat_mutex_unlock(&pl);
    return f;
}

static void present_main(void *arg) {
    (void)arg;
    uint64_t last = 0, last_queued = 0;
    plat_mutex_lock(&pl);
    for (;;) {
        while (!pending) plat_cond_wait(&pcv, &pl);
        PresentJob j = job;
        pending = 0;
        busy = 1;
        plat_cond_broadcast(&pcv);
        plat_mutex_unlock(&pl);
        const CfgRenderState *previous_settings = cfg_render_use(j.settings);
        if (flight_on) vk_thumbs_frame(j.frame);
        vk_dlssg_present_frame(j.frame);
        gfx_present_reflex_frame(j.reflex_id);
        gfx_blit_frame(j.frame);   // DEBUG-PACELOG: the blit's moments and GPU timestamp
        int tagged = vk_present_blit(j.image, VK_IMAGE_LAYOUT_GENERAL, j.format, j.w, j.h,
                                     j.has_crop ? j.crop : NULL, &j.pq, j.nvidia_scene_aa);
        // Retire tagged inputs after the actual present, not the blit. Streamline's default mode orders its work
        // on the presenting queue, so the post-present timeline also protects deferred destruction.
        uint64_t f = tagged ? 0 : present_release(&j);
        uint64_t t0 = kern_now_ns();
        wait_until(j.at);
        uint64_t t1 = kern_now_ns();
        gfx_present_frame(j.frame);   // DEBUG-PRESENTTIMING: the present id
        vk_present_flip();
        __atomic_store_n(&flip_at[j.frame % 16], kern_now_ns(), __ATOMIC_RELEASE);
        if (__builtin_expect(pacelog_on(), 0)) pacelog_mark(PL_FLIP, j.frame, kern_now_ns());   // DEBUG-PACELOG: the host present
        if (tagged) f = present_release(&j);
        if (j.image) vk_completion_add(f, frame_blit_done, (void *)(uintptr_t)j.frame);   // the frame clock (perf.c)
        cfg_render_use(previous_settings);
        cfg_render_release(j.settings);
        cfg_render_publish_present(vk_dlaa_resolve_consumers);
        // REKARINA_TRACE=present: one line per present: the host present's interval, the interval of the game's present
        // calls (when the host present went out before back-edge pacing), the wait and how far past its target it went
        if (rt_trace & RT_TRACE_PRESENT)
            rt_log("present: frame %llu interval %.3f ms, game %.3f ms, waited %.3f ms, %.3f ms past target",
                   (unsigned long long)j.frame, last ? (t1 - last) / 1e6 : 0.0,
                   last_queued ? (j.queued - last_queued) / 1e6 : 0.0, (t1 - t0) / 1e6,
                   j.at && t1 > j.at ? (t1 - j.at) / 1e6 : 0.0);
        last = t1;
        last_queued = j.queued;
        vk_pipeline_autosave();   // starts the autosave thread once
        plat_mutex_lock(&pl);
    }
    return;
}

void vk_present_drain(void) {
    uint64_t p0 = perf_begin();
    plat_mutex_lock(&pl);
    while (pending || busy) plat_cond_wait(&pcv, &pl);
    plat_mutex_unlock(&pl);
    perf_end(PERF_DRAIN, p0);
}

// The present after the frame count: resolves the image and hands the blit to the present thread. queued: when the game
// presented; value: the timeline value queued for this present (asynchronous replay), 0 for the next one (synchronous).
static void present_image(const tex_t *t, const int crop[4], uint64_t at_ns, uint64_t queued, uint64_t frame,
                          uint64_t value, GfxOutputEncoding encoding) {
    static volatile int32_t n_skipped;
    vk_exec_frame_log(t, frame);
    vk_mvsurvey_frame();
    vk_mvprev_frame();
    vk_mem_census(frame);
    if (vk_dump_armed) vk_dump_present(frame);
    __atomic_store_n(&gfx_counters.validation_errors, (uint64_t)vk_validation_errors_total(), __ATOMIC_RELAXED);
    const VkTexImage *img = vk_texture_get(t);
    if (!img || !(img->aspect & VK_IMAGE_ASPECT_COLOR_BIT)) {
        vk_dlaa_frame();
        if (plat_inc32(&n_skipped) <= 8)
            rt_log("vk: present of texture %p (format %#x) skipped: no colour image", (const void *)t, t ? t->d.format : 0);
        if (value) {   // the queued value is still signalled, in order
            vk_present_drain();
            uint64_t f = vk_sync_submit(NULL, 0);
            if (f != value) rt_fatal("vk: skipped present signalled %llu, queued as %llu", (unsigned long long)f, (unsigned long long)value);
        }
        return;
    }
    PresentJob j = {.image = img->image, .format = img->format, .w = (int)img->width, .h = (int)img->height, .frame = frame,
                    .at = at_ns, .queued = queued, .value = value, .reflex_id = vk_reflex_current_id()};
#ifdef GAME_OOT
    GfxOutputEncoding written = vk_hdr_written_output(img->serial, frame, &j.pq);
    static GfxOutputEncoding last_request, last_written;
    if ((rt_trace & RT_TRACE_GFX) && (encoding != last_request || written != last_written)) {
        rt_log("gfx: frame %llu guest encoding request %d, written %d", (unsigned long long)frame, (int)encoding, (int)written);
        last_request = encoding;
        last_written = written;
    }
#else
    (void)encoding;
    vk_hdr_present_source(img, &j.image, &j.format);
#endif
    if (vk_interface_coverage_enabled() ||
        (vk_dlaa_prebloom_enabled() && (vk_dlaa_active() || vk_dlaa_fg_only()))) {
        vk_present_drain();
        vk_dlaa_seal_scene(frame, img->image, j.image, img->serial);
    }
    j.nvidia_scene_aa = vk_dlaa_nvidia_scene_aa();
    vk_dlaa_frame();
    if (crop) {
        j.has_crop = 1;
        for (int i = 0; i < 4; i++) j.crop[i] = crop[i];
    }
    vk_upload_flush(0);   // an upload vk_texture_get just recorded goes before the blit
    int vk_arrive_held(uint64_t frame);
    if (!value && vk_arrive_held(frame)) return;   // vk_arrive.c: made, not shown (the display keeps the last picture)
    plat_mutex_lock(&pl);
    if (!present_thread) {
        if (!plat_thread_spawn(present_main, NULL, "vk present", 0)) rt_fatal("vk: cannot start the present thread");
        present_thread = 1;
    }
    while (pending) plat_cond_wait(&pcv, &pl);   // one present in waiting at most
    j.settings = cfg_render_retain(cfg_render_current());
    job = j;
    pending = 1;
    plat_cond_broadcast(&pcv);
    plat_mutex_unlock(&pl);
    if (rdoc_active()) { vk_present_drain(); rdoc_after_present(frame); }
    vk_movienet_frame();   // a movie's GPU time measured for its network (vk_movienet.c)
    {   // DEBUG-GPUCSV: REKARINA_GPU_CSV=<path>: per present the frame number, the GPU busy ns reported so far and the CPU clock
        static FILE *csv;
        static int init;
        if (!init) { init = 1; const char *e = cfg_env("REKARINA_GPU_CSV"); if (e) csv = fopen(e, "w"); }
        if (csv) {
            extern uint64_t perf_gpu_busy_total(void);
            fprintf(csv, "%llu,%llu,%llu\n", (unsigned long long)frame, (unsigned long long)perf_gpu_busy_total(), (unsigned long long)plat_now_ns());
            if (!(frame % 120)) fflush(csv);
        }
    }
}

// ---- Asynchronous replay
// What the first version lacked, measured in OoT (2026-11-10): it gave no frame rate (the present's wait
// for the frame's results covers the replay and the GPU) and, with that wait relaxed, 12-14 whole-frame blips in 1,500
// frames against none. Three things read the game's state later than the synchronous replay does:
// - The copies a submission takes when its recording starts (the uniform blocks, the written pages of the mirrored
//   pools, the DLSS jitter: vk_exec.c vk_exec_record) were taken when the replay thread got to the submission, up to a
//   replay later than the submit call, with the game already writing the next frame. Now nvnQueueSubmitCommands returns
//   only once the replay thread has taken them (vk_replay_captured): the same moment as in the synchronous replay, plus
//   the wait for the replay thread to finish what it has (REKARINA_REPLAY_HANDSHAKE=0: returns at once, as before).
// - Texture and sampler registrations were applied at the call, ahead of the submissions queued before them. They are
//   queued now, like the releases.
// - The game polls an NVN event its other render thread sets, sleeping 1 ms between polls (main+0x57051c): a millisecond
//   a frame once the submit returns early (kernel.c shortens that sleep).
// The present's wait is the previous present's results under the asynchronous replay (nvn_api_obj.c), so the game
// stays one frame ahead at most and the replay overlaps its next frame.
enum { RI_SUBMIT, RI_PRESENT, RI_RELEASE, RI_TEXREG, RI_SAMPREG };
typedef struct { int id, has_view; tex_t t; view_t v; } RegTex;
typedef struct { int id; samp_par s; } RegSamp;
typedef struct {
    int kind, n, has_crop, crop[4];
    GfxOutputEncoding encoding;
    uint64_t value, submit_no, at, queued;
    uint64_t seq;        // RI_SUBMIT: its number among the submissions queued (the capture handshake)
    void *reg;           // RI_TEXREG / RI_SAMPREG: a RegTex / RegSamp the item owns
    uint64_t *h;
    int flush;           // RI_SUBMIT: gfx_sync_fence's empty submission (vk_replay_flush_held): held rests are recorded
    const void *queue;   // RI_SUBMIT: the game's queue (vk_exec.c holds a submission per queue)
    const tex_t *tex;   // RI_RELEASE: a copy the item owns (the game may reuse the object's memory)
    const CfgRenderState *settings;
} ReplayItem;
#define RQ_CAP 8192   // a frame queues a few submits, 1 present and a few hundred registrations and releases; a scene
                      // load registers thousands of textures at once
enum { DR_POOL_FINI, DR_MAP_VIRTUAL, DR_TEX_RELEASE, DR_PROG_RELEASE, DR_FINISH, DR_READ_TEXELS, DR_WRITE_TEXELS,
       DR_GUEST_READ, DR_FRAME, DR_FULL, DR_CHECK, DR_N };
static const char *const dr_name[DR_N] = {
    "pool_fini", "map_virtual", "texture_release (queued)", "program_release", "finish", "read_texels (writer fence)",
    "write_texels", "guest_read (writer fence)", "frame bound", "queue full", "check drain"};

static PlatMutex rl = PLAT_MUTEX_INIT;
static PlatCond rcv = PLAT_COND_INIT;   // item queued, item taken, item done
static ReplayItem rq[RQ_CAP];
static int rq_head, rq_n, r_busy, r_started;
static uint64_t r_presents_queued, r_presents_done, r_submit_done, game_frames;
static uint64_t r_seq_queued, r_seq_captured, r_seq_cur;   // rl: submissions queued / with their copies taken / being replayed
static _Thread_local int on_replay_thread;
// Under rl: calls, calls that waited and the time waited per cause, since the last replay_log line and over the run.
static uint64_t dr_calls[DR_N], dr_waits[DR_N], dr_ns[DR_N], dr_tcalls[DR_N], dr_twaits[DR_N], dr_tns[DR_N];
// REKARINA_REPLAY_CHECK: ReadTexels compared with a read after a full drain (under rl): checked, differing, differing
// although no submission recorded a write to the texture in between, differing with the queue idle at the first read.
static uint64_t ck_reads, ck_diff, ck_diff_unexplained, ck_diff_idle;

int vk_replay_async(void) {
    static int on = -1;
    if (on < 0) {
        // Where the game waits for the GPU it overlaps the replay with the game threads (a GTX 1650 in
        // the Depths went from 450-686 frames over 50 ms a minute to 10-13, issue #88), the only place a gain was
        // measured. With it enabled, a GTX 1660 Ti at the 60 fps cap with the GPU at 26 % reported uneven
        // motion that 0.3.4 did not have (#659). It adds up to a frame of latency.
        on = cfg_bool("REKARINA_REPLAY_ASYNC");
        rt_log("vk: replay %s (REKARINA_REPLAY_ASYNC=%d)",
               on ? "on its own thread, at most one frame behind the game" : "on the submitting thread", on);
    }
    return on;
}

// The front end's view (nvn_api_core.c, nvn_api_cmd.c): replay is asynchronous only on this back end.
int gfx_replay_async(void) { return gfx_backend() == &gfx_vulkan_backend && vk_replay_async(); }

// The newest submit number (nvn_submit_count) whose replay is done: recording copies it used may be freed.
uint64_t gfx_replayed_submits(void) {
    if (!gfx_replay_async()) {   // synchronous: all but a held submission's rest (vk_exec.c "Queues interleaved")
        uint64_t h = vk_exec_held_submit();
        return h ? h - 1 : nvn_submit_count();
    }
    plat_mutex_lock_shared(&rl);
    uint64_t v = r_submit_done;
    plat_mutex_unlock_shared(&rl);
    return v;
}

static uint64_t submit_sync(const uint64_t *handles, int n) {
    vk_sync_start();
    FT(FT_NOTE, 5, 0, 0);
    vk_present_drain();
    FT(FT_NOTE, 6, 0, 0);
    if (vk_sync_epoch_early()) vk_texture_new_submit();   // DEBUG-EPOCHEARLY (vk_sync.c)
    return vk_sync_submit(handles, n);   // the texture module's per-submission step is made there, under the submit lock
}

// vk_exec.c, where a submission's recording has taken its copies of the game's memory; replay_main after the
// submission, for one that recorded nothing. Only the replay thread's call counts: the present thread records its own
// empty submissions.
void vk_replay_captured(void) {
    if (!on_replay_thread) return;
    plat_mutex_lock(&rl);
    if (r_seq_captured < r_seq_cur) {
        r_seq_captured = r_seq_cur;
        plat_cond_broadcast(&rcv);
    }
    plat_mutex_unlock(&rl);
}

static void replay_main(void *arg) {
    (void)arg;
    perf_replay_thread();
    on_replay_thread = 1;
    {   // Beside the game's own threads the replay ran 40 % slower than inside the submit call, where the submitting
        // thread's work is the replay (bind_resources 2.7 us a call against 2.0, 2026-11-09): the game's twelve workers
        // and this thread share the host's cores. Above normal, the scheduler runs this thread first.
        // REKARINA_REPLAY_PRIO: 0 normal, 1 above normal (the default), 2 highest. Not measured.
        const char *e = cfg_get("REKARINA_REPLAY_PRIO");
        int p = e && *e ? atoi(e) : 1;
        if (p > 0) plat_thread_priority(p > 1 ? PLAT_PRIO_HIGHEST : PLAT_PRIO_ABOVE_NORMAL);
    }
    plat_mutex_lock(&rl);
    for (;;) {
        while (!rq_n) plat_cond_wait(&rcv, &rl);
        ReplayItem it = rq[rq_head];
        rq_head = (rq_head + 1) % RQ_CAP;
        rq_n--;
        r_busy = 1;
        if (it.kind == RI_SUBMIT) r_seq_cur = it.seq;
        plat_cond_broadcast(&rcv);
        plat_mutex_unlock(&rl);
        const CfgRenderState *previous_settings = cfg_render_use(it.settings);
        if (it.kind == RI_TEXREG) {
            RegTex *g = it.reg;
            vk_texture_register(g->id, &g->t, g->has_view ? &g->v : NULL);
            free(g);
        } else if (it.kind == RI_SAMPREG) {
            RegSamp *g = it.reg;
            vk_sampler_register(g->id, &g->s);
            free(g);
        } else if (it.kind == RI_SUBMIT) {
            nvn_submit_queue = it.queue;
            if (it.flush) vk_exec_flush_req(1);
            uint64_t f = submit_sync(it.h, it.n);
            if (it.flush) vk_exec_flush_req(0);
            nvn_submit_queue = NULL;
            if (f != it.value)
                rt_fatal("vk: replayed submission signalled %llu, queued as %llu", (unsigned long long)f,
                         (unsigned long long)it.value);
            free(it.h);
            vk_replay_captured();   // nothing left to copy: its submitter never waits past here
        } else if (it.kind == RI_RELEASE) {
            vk_present_drain();
            vk_texture_release(it.tex);
            free((void *)it.tex);
        } else {
            uint64_t frame = __atomic_add_fetch(&gfx_counters.frames, 1, __ATOMIC_RELAXED);
            vk_dlssg_submitted(frame);   // the frame's submissions were replayed before its present
            present_image(it.tex, it.has_crop ? it.crop : NULL, it.at, it.queued, frame, it.value, it.encoding);
        }
        cfg_render_use(previous_settings);
        cfg_render_release(it.settings);
        plat_mutex_lock(&rl);
        if (it.kind == RI_SUBMIT) r_submit_done = it.submit_no;
        else if (it.kind == RI_PRESENT) r_presents_done++;
        r_busy = 0;
        plat_cond_broadcast(&rcv);
    }
    return;
}

static void wait_counted(int why, uint64_t t0) {   // rl held
    uint64_t d = kern_now_ns() - t0;
    dr_waits[why]++, dr_twaits[why]++;
    dr_ns[why] += d, dr_tns[why] += d;
    perf_end(PERF_DRAIN, t0);
}

// A guest thread waiting for the replay gives its virtual core up, as around any blocking host wait (NVN_BLOCKING):
// guest_read (memcpy, pool invalidate), pool and texture finalize run without it, and a drain lasts up to a replay.
// Called and returning with rl held; rl is not held while the core is taken back.
static int core_away_unlocked(void) {
    plat_mutex_unlock(&rl);
    int away = kern_core_away();
    plat_mutex_lock(&rl);
    return away;
}
static void core_back_unlocked(int *away) {
    if (!*away) return;
    plat_mutex_unlock(&rl);
    kern_core_back(away);
    plat_mutex_lock(&rl);
}

// Queues an item (waits while the queue is full) and returns its value, reserved under rl so values follow queue order.
static uint64_t replay_enqueue(ReplayItem *it) {
    plat_mutex_lock(&rl);
    if (!r_started) {
        if (!plat_thread_spawn(replay_main, NULL, "vk replay", 0)) rt_fatal("vk: cannot start the replay thread");
        r_started = 1;
    }
    // rl is dropped while the core is taken back, and another thread may fill the queue again meanwhile: check again.
    while (rq_n == RQ_CAP) {
        uint64_t p0 = perf_begin();
        dr_calls[DR_FULL]++, dr_tcalls[DR_FULL]++;
        int away = core_away_unlocked();
        while (rq_n == RQ_CAP) plat_cond_wait(&rcv, &rl);
        wait_counted(DR_FULL, p0);
        core_back_unlocked(&away);
    }
    // a release or a registration signals nothing
    it->value = it->kind == RI_SUBMIT || it->kind == RI_PRESENT ? vk_sync_reserve() : 0;
    if (it->kind == RI_SUBMIT) it->seq = ++r_seq_queued;
    rq[(rq_head + rq_n) % RQ_CAP] = *it;
    rq_n++;
    plat_cond_broadcast(&rcv);
    plat_mutex_unlock(&rl);
    return it->value;
}

// Returns once every queued item was replayed and submitted. Back-end entries only: the replay thread never calls them.
static void replay_drain(int why) {
    if (!gfx_replay_async()) return;
    plat_mutex_lock(&rl);
    dr_calls[why]++, dr_tcalls[why]++;
    if (rq_n || r_busy) {
        uint64_t p0 = perf_begin();
        int away = core_away_unlocked();
        while (rq_n || r_busy) plat_cond_wait(&rcv, &rl);
        wait_counted(why, p0);
        core_back_unlocked(&away);
    }
    plat_mutex_unlock(&rl);
}

// What made the game wait for the replay: since the last line (every 1800 game frames), or over the run (at exit).
static void replay_log(uint64_t frame, int total) {
    char buf[1024];
    int k = 0;
    plat_mutex_lock(&rl);
    const uint64_t *c = total ? dr_tcalls : dr_calls, *w = total ? dr_twaits : dr_waits, *t = total ? dr_tns : dr_ns;
    for (int i = 0; i < DR_N && k < (int)sizeof buf - 96; i++)
        if (c[i] || w[i])
            k += snprintf(buf + k, sizeof buf - (size_t)k, "%s %s %llu calls, %llu waited %.1f ms", k ? ";" : "", dr_name[i],
                          (unsigned long long)c[i], (unsigned long long)w[i], t[i] / 1e6);
    if (ck_reads && k < (int)sizeof buf - 128)
        k += snprintf(buf + k, sizeof buf - (size_t)k, "; check (run): %llu reads, %llu differ, %llu without a recorded write, %llu with the queue idle",
                      (unsigned long long)ck_reads, (unsigned long long)ck_diff, (unsigned long long)ck_diff_unexplained,
                      (unsigned long long)ck_diff_idle);
    memset(dr_calls, 0, sizeof dr_calls);
    memset(dr_waits, 0, sizeof dr_waits);
    memset(dr_ns, 0, sizeof dr_ns);
    plat_mutex_unlock(&rl);
    rt_log("vk: replay queue %s frame %llu:%s", total ? "over the run to" : "at", (unsigned long long)frame,
           k ? buf : " no waits");
}

// The run total, once: the game exits through prof_write (boot.c _exit, the window close), probes through
// vk_backend_shutdown.
void gfx_exit_log(void) {
    static int32_t done;
    rec_check_log();
    vk_live_check_log();   // DEBUG-LIVECHECK
    if (gfx_replay_async() && r_started && !plat_xchg32(&done, 1)) replay_log(game_frames, 1);
}

static void vk_be_present(const tex_t *t, const int crop[4], uint64_t at_ns, GfxOutputEncoding encoding) {
    vk_sync_start();
    vk_dlssg_game_present();
    vk_reflex_game_present();
    if (!gfx_replay_async()) {
        const CfgRenderState *settings = cfg_render_frame_acquire();
        const CfgRenderState *previous = cfg_render_use(settings);
        uint64_t frame = __atomic_add_fetch(&gfx_counters.frames, 1, __ATOMIC_RELAXED);
        vk_dlssg_submitted(frame);
        vk_texture_readback_tick(frame);   // small targets rendered a frame ago reach their storage (vk_texture.c)
        { void vk_event_lag_tick(uint64_t); vk_event_lag_tick(frame); }   // DEBUG-EVLAG
        { void vk_report_lag_tick(uint64_t); vk_report_lag_tick(frame); }   // DEBUG-RPLAG
        { void prof_hwwatch_tick(uint64_t); prof_hwwatch_tick(frame); }   // DEBUG-HWWATCH
        perf_present(frame);   // before vk_exec_frame_log, which resets the frame's exec counts the flight recorder reads
        ftrace_frame(frame);
        present_image(t, crop, at_ns, kern_now_ns(), frame, 0, encoding);
        cfg_render_use(previous);
        cfg_render_release(settings);
        return;
    }
    // Frame bound: the replay thread handled the previous present before this one is queued.
    plat_mutex_lock(&rl);
    dr_calls[DR_FRAME]++, dr_tcalls[DR_FRAME]++;
    if (r_presents_done < r_presents_queued) {
        uint64_t p0 = perf_begin();
        while (r_presents_done < r_presents_queued) plat_cond_wait(&rcv, &rl);
        wait_counted(DR_FRAME, p0);
    }
    r_presents_queued++;
    plat_mutex_unlock(&rl);
    uint64_t frame = ++game_frames;
    perf_present(frame);
    ftrace_frame(frame);
    if (frame % 1800 == 0) replay_log(frame, 0);
    ReplayItem it = {.kind = RI_PRESENT, .tex = t, .at = at_ns, .queued = kern_now_ns(), .encoding = encoding,
                     .settings = cfg_render_frame_acquire()};
    if (crop) {
        it.has_crop = 1;
        for (int i = 0; i < 4; i++) it.crop[i] = crop[i];
    }
    replay_enqueue(&it);
}

// The end of the game's present call under the asynchronous replay: the staged readbacks of small targets that are due
// go to their storage (vk_texture.c "Frame-latency readbacks"), as the synchronous present does at its start. The
// first version left this to the completion thread alone (fb_done), which delivers when a later readback completes:
// about a frame later than the synchronous replay and at no fixed point of the game's frame, for the luminance the
// game's auto exposure reads (a likely source of that version's whole-frame brightness blips; not verified). Counted in
// the game's presents: a submission's replay stamps its job with the presents replayed before it, which are the game's
// presents before that submission, and the wait just over (the previous present's work by default) covers the jobs due.
//
// The late sums (vk_memory.c: what the GPU added into the auto exposure's sum after the results point) go into guest
// memory here too, and only here, under the asynchronous replay. With the present waiting for the previous present
// the game's own nvnSyncWait was what applied a frame's sums, at the end of the next frame's simulation, a moment
// before the game reads and resets the sum: a read then found no frame's sum or two, and the picture settled a quarter
// brighter (thumbnail brightness 69 against 54 with the synchronous replay, 55 with this present waiting for its own
// frame; 2026-11-11). At the present the sums of the frame before are applied once per frame, a whole simulation away
// from that read. REKARINA_ASYNC_SUMS=sync brings the old moments back.
static int async_sums_at_present(void) {
    static int on = -1;
    if (on < 0) { const char *e = cfg_get("REKARINA_ASYNC_SUMS"); on = !(e && !strcmp(e, "sync")); }
    return on;
}
static void vk_be_results_seen(uint64_t value) {
    if (!gfx_replay_async() || !async_sums_at_present()) vk_mem_late_apply(value);
}
static void vk_be_present_end(uint64_t waited) {
    if (!gfx_replay_async()) return;
    vk_texture_readback_tick(game_frames);
    if (waited && async_sums_at_present()) vk_mem_late_apply(waited);
}

// vk_pipeline.c drops a released program's pipelines here.
WEAK void vk_program_release(const prog_t *p) { (void)p; }

static int vk_be_pool_init(pool_t *p) { vk_sync_start(); return vk_pool_init(p); }
static void vk_be_pool_fini(pool_t *p) { vk_sync_start(); replay_drain(DR_POOL_FINI); vk_present_drain(); vk_pool_fini(p); }
static int vk_be_pool_map_virtual(pool_t *v, int n, const void *requests) {
    vk_sync_start();
    replay_drain(DR_MAP_VIRTUAL);
    vk_present_drain();
    return vk_pool_map_virtual(v, n, requests);
}
// Asynchronous: queued behind the submissions before it, whose replay still resolves the slot to what it named when
// they were recorded (the game rewrites much of its descriptor pool for the next frame while this one is replayed).
static void vk_be_texture_register(int id, const tex_t *t, const view_t *v) {
    vk_sync_start();
    vk_texture_stream_ask(t);   // ahead of the replay's first use (vk_texture.c, the texture stream; off by default)
    if (!gfx_replay_async()) {
        vk_texture_register(id, t, v);
        return;
    }
    RegTex *g = malloc(sizeof *g);
    if (!g) rt_fatal("vk: out of memory (replay queue)");
    g->id = id, g->t = *t, g->has_view = v != NULL;
    if (v) g->v = *v;
    ReplayItem it = {.kind = RI_TEXREG, .reg = g};
    replay_enqueue(&it);
}
static void vk_be_sampler_register(int id, const samp_par *s) {
    vk_sync_start();
    if (!gfx_replay_async()) {
        vk_sampler_register(id, s);
        return;
    }
    RegSamp *g = malloc(sizeof *g);
    if (!g) rt_fatal("vk: out of memory (replay queue)");
    g->id = id, g->s = *s;
    ReplayItem it = {.kind = RI_SAMPREG, .reg = g};
    replay_enqueue(&it);
}
// Asynchronous: the release is queued behind the submissions before it, so their replay still resolves the texture's
// slots to its image (a released slot follows another live texture at the storage), and the ones after it see it
// released, the order of the synchronous replay. vk_texture_release reads only the object's key fields: a copy.
static void vk_be_texture_release(const tex_t *t) {
    vk_sync_start();
    if (!gfx_replay_async()) {
        // A game thread (with its virtual core, no host lock held) that finds the present thread busy waits for it
        // without the core, as for the texture lock (vk_texture.c lock_off_core): the blit's submission can wait for
        // the submit lock behind a replay.
        plat_mutex_lock(&pl);
        int wait = pending || busy;
        plat_mutex_unlock(&pl);
        int away = wait ? kern_core_away_counted() : 0;
        vk_present_drain();
        kern_core_back(&away);
        vk_texture_release(t);
        return;
    }
    if (!t || !t->mem) return;
    tex_t *copy = malloc(sizeof *copy);
    if (!copy) rt_fatal("vk: out of memory (replay queue)");
    *copy = *t;
    ReplayItem it = {.kind = RI_RELEASE, .tex = copy};
    plat_mutex_lock(&rl);
    dr_calls[DR_TEX_RELEASE]++, dr_tcalls[DR_TEX_RELEASE]++;
    plat_mutex_unlock(&rl);
    replay_enqueue(&it);
}
static void vk_be_program_release(const prog_t *p) { replay_drain(DR_PROG_RELEASE); vk_program_release(p); }
static int sub_log_on(void) {   // DEBUG-SUBLOG: REKARINA_SUBLOG=<first game frame>: the submissions of six frames from there
    static int64_t f0 = -2;
    if (f0 == -2) { const char *e = cfg_get("REKARINA_SUBLOG"); f0 = e && *e ? atoll(e) : -1; }
    return f0 >= 0 && (int64_t)game_frames >= f0 && (int64_t)game_frames < f0 + 6;
}
// Asynchronous: the handles (nvn_submit_prepare_async ran under the front end's submit lock) are copied and queued;
// the value returned is the one the replay will signal.
static uint64_t vk_be_submit(const uint64_t *handles, int n) {
    if (!gfx_replay_async()) return submit_sync(handles, n);
    vk_sync_start();
    ReplayItem it = {.kind = RI_SUBMIT, .n = n, .submit_no = nvn_submit_count(), .queue = nvn_submit_queue,
                     .settings = cfg_render_frame_acquire()};
    if (n > 0) {
        it.h = malloc(sizeof *it.h * (size_t)n);
        if (!it.h) rt_fatal("vk: out of memory (replay queue)");
        memcpy(it.h, handles, sizeof *it.h * (size_t)n);
    }
    uint64_t value = replay_enqueue(&it);
    // The handshake (top of this section): until the replay thread has taken the submission's copies, at most
    // HANDSHAKE_NS (the replay may itself be waiting for a signal of the game's: then the submit goes on as the first
    // version did).
    // REKARINA_REPLAY_HANDSHAKE=<bytes> (A/B) lets submissions with fewer bytes of recordings go without the wait. OoT's
    // frame is five small submissions (16 bytes to 26 KiB of recordings) spread over the game's frame and one of
    // 1.2 MiB right before the present, and waiting at a small one is waiting for the replay of the frame before to
    // end with the rest of the game's frame still to do: the frame is the replay plus that rest, as with the
    // synchronous replay. But the small ones' copies cannot be taken later: with only the large one waiting, 780 of
    // 1,500 standing frames differed from their neighbours in whole-frame brightness (none with every submission
    // waiting; 2026-11-11). A gain here needs their copies taken on the submitting thread.
    static int handshake = -1;
    if (handshake < 0) {
        const char *e = cfg_get("REKARINA_REPLAY_HANDSHAKE");
        handshake = e && *e ? atoi(e) : 1;
    }
    uint64_t bytes = 0;
    if (handshake > 1 || sub_log_on())
        for (int i = 0; i < n; i++) bytes += nvn_handle_bytes(handles[i]);
    if (sub_log_on()) rt_log("SUBLOG frame %llu submit %llu: %d recordings, %llu bytes", (unsigned long long)game_frames,
                             (unsigned long long)it.submit_no, n, (unsigned long long)bytes);   // DEBUG-SUBLOG
    if (handshake == 1 || (handshake > 1 && bytes >= (uint64_t)handshake)) {
        enum { HANDSHAKE_NS = 100000000 };
        static uint64_t n_wait, n_timeout, ns_wait;
        uint64_t t0 = kern_now_ns();
        plat_mutex_lock(&rl);
        int away = 0, waited = 0;
        if (r_seq_captured < it.seq) away = core_away_unlocked(), waited = 1;
        while (r_seq_captured < it.seq) {
            uint64_t el = kern_now_ns() - t0;
            if (el >= HANDSHAKE_NS) break;
            plat_cond_wait_ns(&rcv, &rl, HANDSHAKE_NS - el);
        }
        int timed_out = r_seq_captured < it.seq;
        if (waited) core_back_unlocked(&away);
        n_wait += (uint64_t)waited, n_timeout += (uint64_t)timed_out, ns_wait += kern_now_ns() - t0;
        if (!(it.seq % 4096))
            rt_log("vk: replay handshake: %llu of the last 4096 submits waited for their copies, %.3f ms each on average, "
                   "%llu gave up after %d ms", (unsigned long long)n_wait, n_wait ? ns_wait / 1e6 / (double)n_wait : 0.0,
                   (unsigned long long)n_timeout, HANDSHAKE_NS / 1000000),
                n_wait = n_timeout = ns_wait = 0;
        plat_mutex_unlock(&rl);
    }
    return value;
}
static void vk_be_finish(void) { vk_sync_start(); replay_drain(DR_FINISH); vk_present_drain(); vk_sync_finish(); }

// gfx_sync_fence (vk_exec.c) with the asynchronous replay: what is queued may already resume the held rest; if one is
// still held after that, an empty submission that records it goes through the queue and is waited for.
void vk_replay_flush_held(void) {
    replay_drain(DR_FINISH);
    if (!vk_exec_held()) return;
    ReplayItem it = {.kind = RI_SUBMIT, .n = 0, .flush = 1, .submit_no = nvn_submit_count(),
                     .settings = cfg_render_frame_acquire()};
    replay_enqueue(&it);
    replay_drain(DR_FINISH);
}
static int replay_check(void) {
    static int on = -1;
    if (on < 0) {
        const char *e = cfg_get("REKARINA_REPLAY_CHECK");
        on = e && strcmp(e, "0");
        if (on) rt_log("vk: REKARINA_REPLAY_CHECK: every ReadTexels is read again after a full replay drain and compared");
    }
    return on;
}

// ReadTexels with the replay queued: waits for the newest submission recorded to write the texture (replayed and
// completed), not for the whole queue. The readback copy goes through the upload batch on the same queue, after every
// submission the replay has submitted. Submissions still queued or being recorded run after the copy, as on hardware
// when the GPU has not reached them yet: the game only sees their texels after waiting on a sync, and every sync
// value covers replay and GPU, so a read the game synchronised sees them. REKARINA_REPLAY_CHECK compares with a full drain.
static void vk_be_read_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row,
                              uint64_t img) {
    vk_sync_start();
    if (!gfx_replay_async()) {
        vk_read_texels(t, v, r, dst, row, img);
        return;
    }
    uint64_t f = vk_texture_last_write(t);
    plat_mutex_lock(&rl);
    dr_calls[DR_READ_TEXELS]++, dr_tcalls[DR_READ_TEXELS]++;
    int idle = !rq_n && !r_busy;
    plat_mutex_unlock(&rl);
    if (f && vk_sync_fence_done() < f) {
        uint64_t t0 = kern_now_ns();
        vk_sync_fence_wait(f, UINT64_MAX);   // counts as a fence wait in perf
        uint64_t d = kern_now_ns() - t0;
        plat_mutex_lock(&rl);
        dr_waits[DR_READ_TEXELS]++, dr_twaits[DR_READ_TEXELS]++;
        dr_ns[DR_READ_TEXELS] += d, dr_tns[DR_READ_TEXELS] += d;
        plat_mutex_unlock(&rl);
    }
    vk_read_texels(t, v, r, dst, row, img);
    if (!replay_check()) return;
    size_t n = vk_texture_texel_span(t, v, r, row, img);
    uint8_t *again = n ? malloc(n) : NULL;
    if (!again) return;
    memcpy(again, dst, n);   // bytes between the rows stay the caller's
    uint64_t f0 = vk_texture_last_write(t);
    replay_drain(DR_CHECK);
    uint64_t f1 = vk_texture_last_write(t);
    vk_read_texels(t, v, r, again, row, img);
    int differ = memcmp(again, dst, n) != 0;
    static volatile int32_t n_logged;
    if (differ && plat_inc32(&n_logged) <= 40) {
        size_t at = 0;
        while (at < n && again[at] == dst[at]) at++;
        rt_log("vk: replay check: ReadTexels of %p (format %#x, %dx%dx%d at %d,%d,%d) differs at byte %zu of %zu; writer "
               "fence %llu at the read, %llu after the drain, done %llu; queue %s at the read",
               (void *)t->mem, t->d.format, r->width, r->height, r->depth, r->xoffset, r->yoffset, r->zoffset, at, n,
               (unsigned long long)f0, (unsigned long long)f1, (unsigned long long)vk_sync_fence_done(), idle ? "idle" : "busy");
    }
    plat_mutex_lock(&rl);
    ck_reads++;
    if (differ) {
        ck_diff++;
        if (f1 == f0) ck_diff_unexplained++;
        if (idle) ck_diff_idle++;
    }
    plat_mutex_unlock(&rl);
    free(again);
}
static void vk_be_write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src, uint64_t row,
                               uint64_t img) {
    vk_sync_start();
    replay_drain(DR_WRITE_TEXELS);
    vk_present_drain();
    vk_write_texels(t, v, r, src, row, img);
}
static void vk_be_guest_written(uint64_t addr, uint64_t size) { vk_sync_start(); vk_guest_written(addr, size); }
// A CPU read of GPU-written mirrored storage (memcpy, pool invalidate). With the replay queued it waits for the newest
// submission recorded to write the textures there, not for the whole queue: the write-back copy then runs after their
// renders, and clears gpu_dirty only for content that is on the queue. Submissions not replayed yet render after the
// copy, as on hardware when the GPU has not reached them; they mark the texture dirty again for a later read.
static void vk_be_guest_read(uint64_t addr, uint64_t size) {
    if (!vk_texture_cpu_stale(addr, size)) return;
    vk_sync_start();
    // From here the read waits (the replay's writer, the present thread, the texture lock, the GPU's copy back): a game
    // thread (memcpy, pool invalidate; no host lock held) waits without its virtual core, taken back at the end.
    int away = kern_core_away_counted();
    if (gfx_replay_async()) {
        uint64_t f = vk_texture_dirty_fence(addr, size);
        plat_mutex_lock(&rl);
        dr_calls[DR_GUEST_READ]++, dr_tcalls[DR_GUEST_READ]++;
        plat_mutex_unlock(&rl);
        if (f && vk_sync_fence_done() < f) {
            uint64_t t0 = kern_now_ns();
            NVN_BLOCKING();
            vk_sync_fence_wait(f, UINT64_MAX);
            uint64_t d = kern_now_ns() - t0;
            plat_mutex_lock(&rl);
            dr_waits[DR_GUEST_READ]++, dr_twaits[DR_GUEST_READ]++;
            dr_ns[DR_GUEST_READ] += d, dr_tns[DR_GUEST_READ] += d;
            plat_mutex_unlock(&rl);
        }
    }
    vk_present_drain();
    vk_texture_write_back(addr, size);
    kern_core_back(&away);
}
// After a fence wait: the value is done, so its replay is too; a later submission the replay records meanwhile marks
// its targets with a later fence, which write_back_small leaves alone. No drain.
static void vk_be_cpu_synced(uint64_t fence) {
    if (!vk_texture_small_dirty()) return;
    vk_sync_start();
    vk_present_drain();
    vk_texture_write_back_small(fence);
}

static void vk_be_frame_start(void) {
    // Pin the last published settings without waiting for Present; a live edit can arrive a frame later.
    cfg_render_frame_begin();
    if (vk_dlssg_swapchain_ready()) vk_dlssg_frame_start();
    else vk_reflex_frame_start();
}

const GfxBackend gfx_vulkan_backend = {
    .name = "vulkan",
    .pool_init = vk_be_pool_init,
    .pool_fini = vk_be_pool_fini,
    .pool_map_virtual = vk_be_pool_map_virtual,
    .texture_register = vk_be_texture_register,
    .sampler_register = vk_be_sampler_register,
    .texture_new = vk_texture_premake,
    .texture_release = vk_be_texture_release,
    .program_release = vk_be_program_release,
    .submit = vk_be_submit,
    .fence_wait = vk_sync_fence_wait,
    .fence_done = vk_sync_fence_done,
    .results_wait = vk_sync_results_wait,
    .results_done = vk_sync_results_done,
    .results_seen = vk_be_results_seen,
    .present = vk_be_present,
    .present_end = vk_be_present_end,
    .frame_start = vk_be_frame_start,
    .finish = vk_be_finish,
    .read_texels = vk_be_read_texels,
    .write_texels = vk_be_write_texels,
    .guest_written = vk_be_guest_written,
    .guest_read = vk_be_guest_read,
    .cpu_synced = vk_be_cpu_synced,
};

void vk_backend_shutdown(void) {
    replay_drain(DR_FINISH);
    gfx_exit_log();
    vk_present_drain();
    vk_sync_shutdown();
    vk_mem_shutdown();
    vk_shutdown();
}
