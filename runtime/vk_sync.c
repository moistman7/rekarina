// Submission, fences and the completion thread of the Vulkan back end.
// One timeline semaphore (VkCtx.timeline) orders everything: each submission records its command handles into a
// primary command buffer (vk_exec_record) and signals the next timeline value, which is the NVN fence value
// the front end hands to syncs. The completion thread waits on the timeline and, in submit order, runs the CPU side
// effects queued for each value (event writes, counter reports, sync state), then frees the memory and texture objects whose users
// completed. A value counts as done only after its side effects ran, so a guest woken by fence_wait sees them.
//
// The submission shape (one timeline value per submission, signalled at ALL_COMMANDS, vkWaitSemaphores for waits, a
// failed wait treated as a lost device) follows LAS Recomp runtime/nvn/vulkan/exec.c:39-110, 355-397
// (https://github.com/peiikai/LASRecomp @ da35b83a9611d046a9a681aae668d7085954d37f, MIT, Copyright (c) 2026 Peiikai).
// Side effects written after GPU completion, as MacNativeGames does: games/totk/third_party/d2r_gfx/gfx/nvn/hle_nvn.c:7-9
// (https://github.com/ly05010419/MacNativeGames @ b75be8cfa704df3701e86130c8ced268e79d6b11, MIT, Copyright (c) 2026
// jhaizhou; origin d2r-mobile).
#include <stdlib.h>
#include <string.h>

#include "ftrace.h"
#include "gfx.h"
#include "host.h"
#include "nvn_api.h"
#include "vk.h"
#include "platform/platform.h"
#include "cfg.h"
#include "prep.h"
#include "vk_reflex.h"
uint64_t vk_listlog_addr(void), vk_listlog_hash(void);   // DEBUG-LISTLOG
double vk_listlog_ms(void);

#define WEAK __attribute__((weak))

typedef struct { uint64_t fence; void (*fn)(void *); void *arg; } Pending;
// A replay command buffer, reusable once the timeline reached fence. Each has a command pool of its own: the recorder
// thread may still be recording one part (vk_rec.c rec_end_async) while the replay thread begins the next part's
// buffer and records the prologue, and a VkCommandPool takes one thread at a time for all of its buffers.
typedef struct { VkCommandBuffer cb; uint64_t fence; VkCommandPool pool; } CmdSlot;

static PlatMutex start_lock = PLAT_MUTEX_INIT;
static int started;
static PlatThread *completion_thread;

// Submissions: one at a time, so timeline values are signalled in the order they are handed out.
static PlatMutex sub_lock = PLAT_MUTEX_INIT;
static CmdSlot *slots;
static int nslots;
static int cur_slot;          // the slot vk_sync_submit records into now (an index: take_slot may move slots)
static uint64_t cur_fence;    // the value of the submission being recorded

// Shared with the completion thread (st_lock).
static PlatMutex st_lock = PLAT_MUTEX_INIT;
static PlatCond cv_work = PLAT_COND_INIT;   // a submission, a callback or stop
static PlatCond cv_done = PLAT_COND_INIT;   // done advanced
static uint64_t issued, done;     // newest submitted value; newest value whose side effects ran
static uint64_t reserved;         // asynchronous replay: newest value handed out before its submission (>= issued)
static int stop;
static Pending *queue;            // side effects in the order they were added
static int qhead, qn, qcap;

// ---- GPU time (REKARINA_TRACE=perf): each submission part (vk_sync_split) writes a timestamp when it starts and when it
// ends, into a ring of query pairs. The completion thread reads a pair once the part's value is reached and adds the
// part's interval to the busy time, counting overlap with the previous part once. Off (one log line) when the queue
// family has no timestamp bits or the period is 0. Upload batches (vk_memory.c) are not timed.
#define TS_PAIRS 1024
static VkQueryPool ts_pool;
static int ts_state;                       // 0 not set up, 1 on, -1 off
static double ts_period;                   // ns per tick
static uint64_t ts_mask;                   // valid bits of a timestamp
static uint32_t ts_next;                   // submitting thread (sub_lock): next pair
static volatile uint8_t ts_busy[TS_PAIRS]; // written until the completion thread read the pair
static int ts_cur = -1;                    // the pair of the part being recorded, -1 untimed
static uint64_t ts_last_end;               // completion thread: end of the latest part read (ticks)
static volatile int64_t ts_busy_ns, ts_parts, ts_missed;
// GPU idle (with calibrated timestamps): the gap before each timed part, split by whether the part was submitted after
// the previous one had ended (the GPU waited for the CPU: starved) or before (it waited for untimed work or a wait).
static uint64_t ts_sub_ns[TS_PAIRS];       // CPU time the part was handed to vkQueueSubmit2
static int ts_unsent[4], ts_n_unsent;      // pairs closed and not yet submitted (sub_lock)
static uint64_t ts_last_end_ns;            // completion thread: ts_last_end on the CPU clock
static volatile int64_t ts_idle_ns, ts_starved_ns, ts_gaps, ts_max_gap_ns, ts_queued_ns;
static uint8_t ts_pro[TS_PAIRS];           // REKARINA_SECTIONS: the pair is a prologue's (SEC_GPU_PROLOGUE)
static int64_t cal_tick, cal_ns, cal_at;   // completion thread: a calibrated (GPU tick, CPU ns) pair and when it was taken
static PFN_vkGetCalibratedTimestampsKHR cal_fn;

// On the first submission that wants timing (perf_gpu_timing: REKARINA_TRACE=perf or the full performance overlay).
static void ts_setup(VkCtx *c) {
    ts_state = -1;
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, NULL);
    VkQueueFamilyProperties *qp = calloc(nq ? nq : 1, sizeof *qp);
    if (!qp) return;
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, qp);
    uint32_t bits = c->gfx_family < nq ? qp[c->gfx_family].timestampValidBits : 0;
    free(qp);
    float period = c->props.limits.timestampPeriod;
    if (!bits || period <= 0) {
        rt_log("vk: no GPU timestamps on this queue (valid bits %u, period %.3f ns): the perf line shows no GPU time", bits, period);
        return;
    }
    VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                .queryCount = 2 * TS_PAIRS};
    VkResult r = vkCreateQueryPool(c->device, &qi, NULL, &ts_pool);
    if (r != VK_SUCCESS) {
        rt_log("vk: timestamp query pool failed (%d): the perf line shows no GPU time", (int)r);
        return;
    }
    ts_period = period;
    ts_mask = bits >= 64 ? ~0ull : (1ull << bits) - 1;
    ts_state = 1;
}

typedef struct { uint32_t pair; } TsRead;

// Completion thread: refresh the GPU tick / CPU clock pair every 250 ms; 0 when there is no calibration. The host
// domain is the clock plat_now_ns reads (QPC on Windows, CLOCK_MONOTONIC elsewhere), plat_counter_hz ticks a second.
static int cal_refresh(VkCtx *c) {
    if (!c->has_calibrated) return 0;
    if (!cal_fn) {
        cal_fn = (PFN_vkGetCalibratedTimestampsKHR)vkGetDeviceProcAddr(c->device, c->has_calibrated == 2 ? "vkGetCalibratedTimestampsKHR"
                                                                                                        : "vkGetCalibratedTimestampsEXT");
        if (!cal_fn) { c->has_calibrated = 0; return 0; }
    }
    int64_t now = (int64_t)plat_now_ns();
    if (cal_at && now - cal_at < 250000000) return 1;
    VkCalibratedTimestampInfoKHR ci[2] = {{.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, .timeDomain = VK_TIME_DOMAIN_DEVICE_KHR},
#ifdef _WIN32
                                          {.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, .timeDomain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR}};
#else
                                          {.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, .timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR}};
#endif
    uint64_t v[2], dev;
    if (cal_fn(c->device, 2, ci, v, &dev) != VK_SUCCESS) {
        if (!cal_at) {   // the domain pair is not supported: no split
            rt_log("vk: calibrated timestamps failed: the perf line shows no GPU idle split");
            c->has_calibrated = 0;
        }
        return cal_at != 0;
    }
    cal_tick = (int64_t)(v[0] & ts_mask), cal_ns = (int64_t)((double)v[1] * 1e9 / (double)plat_counter_hz()), cal_at = now;
    return 1;
}

static int64_t tick_ns(uint64_t t) { return cal_ns + (int64_t)((double)((int64_t)t - cal_tick) * ts_period); }

static void max64(volatile int64_t *m, int64_t v) {
    int64_t o = __atomic_load_n(m, __ATOMIC_RELAXED);
    while (v > o && !__atomic_compare_exchange_n(m, &o, v, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
}

// Just before vkQueueSubmit2 (sub_lock held): the parts closed since the last submit leave the CPU now.
static void ts_submitted(void) {
    uint64_t now = plat_now_ns();
    for (int i = 0; i < ts_n_unsent; i++) ts_sub_ns[ts_unsent[i]] = now;
    ts_n_unsent = 0;
}

static void ts_read_cb(void *arg) {   // completion thread, in submission order
    TsRead *t = arg;
    VkCtx *c = vk_ctx();
    uint64_t v[2];
    if (vkGetQueryPoolResults(c->device, ts_pool, 2 * t->pair, 2, sizeof v, v, sizeof v[0], VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
        uint64_t s = v[0] & ts_mask, e = v[1] & ts_mask;
        if (e >= s) {
            if (cal_refresh(c)) {
                int64_t sn = tick_ns(s), en = tick_ns(e), sub = (int64_t)ts_sub_ns[t->pair];
                if (ts_last_end_ns && sn > ts_last_end_ns) {
                    int64_t gap = sn - ts_last_end_ns, late = sub - ts_last_end_ns;
                    plat_add64(&ts_idle_ns, gap);
                    if (late > 0) plat_add64(&ts_starved_ns, late < gap ? late : gap);
                    if (gap > 500000) plat_inc64(&ts_gaps);
                    max64(&ts_max_gap_ns, gap);
                }
                if (sub && sn > sub) plat_add64(&ts_queued_ns, sn - sub);
                if (en > ts_last_end_ns) ts_last_end_ns = en;
                if (__builtin_expect(ftrace_on, 0)) ftrace_ev_at(FT_GPART, (uint64_t)sn, (uint64_t)(en - sn), sub ? (uint64_t)(sn - sub) : 0, t->pair);
            }
            uint64_t from = s > ts_last_end ? s : ts_last_end;
            if (e > from) {
                int64_t ns = (int64_t)((double)(e - from) * ts_period);
                plat_add64(&ts_busy_ns, ns);
                perf_gpu_busy((uint64_t)ns);
            }
            if (e > ts_last_end) ts_last_end = e;
            plat_inc64(&ts_parts);
            if (ts_pro[t->pair]) ts_pro[t->pair] = 0, sec_add_ns(SEC_GPU_PROLOGUE, (uint64_t)((double)(e - s) * ts_period));
        } else plat_inc64(&ts_missed);   // the counter wrapped
    } else plat_inc64(&ts_missed);
    __atomic_store_n(&ts_busy[t->pair], 0, __ATOMIC_RELEASE);
    free(t);
}

// At the start of a command buffer (sub_lock held): the pair it times, -1 untimed.
static int ts_open(VkCommandBuffer cb) {
    if (ts_state != 1 || !perf_gpu_timing()) {
        if (ts_state || !perf_gpu_timing()) return -1;
        ts_setup(vk_ctx());   // the overlay turned full: timestamps from now on
        if (ts_state != 1) return -1;
    }
    uint32_t p = ts_next % TS_PAIRS;
    if (__atomic_load_n(&ts_busy[p], __ATOMIC_ACQUIRE)) { plat_inc64(&ts_missed); return -1; }
    ts_next++;
    ts_busy[p] = 1;
    vkCmdResetQueryPool(cb, ts_pool, 2 * p, 2);
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, ts_pool, 2 * p);
    return (int)p;
}

// At its end, before vkEndCommandBuffer; fence: the value whose completion covers it.
static void ts_close(VkCommandBuffer cb, int pair, uint64_t fence) {
    if (pair < 0) return;
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, ts_pool, 2 * (uint32_t)pair + 1);
    ts_sub_ns[pair] = 0;
    if (ts_n_unsent < 4) ts_unsent[ts_n_unsent++] = pair;
    TsRead *t = malloc(sizeof *t);
    if (!t) rt_fatal("vk: out of memory");
    t->pair = (uint32_t)pair;
    vk_completion_add(fence, ts_read_cb, t);
}

// The pair of the part being recorded.
static void ts_begin(VkCommandBuffer cb) { ts_cur = ts_open(cb); }

static void ts_end(VkCommandBuffer cb, uint64_t fence) {
    ts_close(cb, ts_cur, fence);
    ts_cur = -1;
}

// An empty submission records no command buffer: its pair is never written.
static void ts_drop(void) {
    if (ts_cur < 0) return;
    __atomic_store_n(&ts_busy[ts_cur], 0, __ATOMIC_RELEASE);
    ts_cur = -1;
}

static void gpu_stats(PerfGpu *g) {
    g->have_gpu = ts_state == 1;
    g->gpu_ns = (uint64_t)plat_xchg64(&ts_busy_ns, 0);
    g->gpu_parts = (uint64_t)plat_xchg64(&ts_parts, 0);
    g->gpu_missed = (uint64_t)plat_xchg64(&ts_missed, 0);
    g->have_vram = vk_mem_vram(&g->vram_used, &g->vram_budget, &g->vram_exact);
    g->have_idle = vk_ctx()->has_calibrated != 0 && cal_at != 0;
    g->idle_ns = (uint64_t)plat_xchg64(&ts_idle_ns, 0);
    g->starved_ns = (uint64_t)plat_xchg64(&ts_starved_ns, 0);
    g->queued_ns = (uint64_t)plat_xchg64(&ts_queued_ns, 0);
    g->idle_gaps = (uint64_t)plat_xchg64(&ts_gaps, 0);
    g->max_gap_ns = (uint64_t)plat_xchg64(&ts_max_gap_ns, 0);
}

// Weak stand-in for vk_exec.c's replay: only empty recordings ({OP_BEGIN, OP_END}) can be submitted.
WEAK int vk_exec_record(VkCommandBuffer cb, const uint64_t *handles, int n, uint64_t fence) {
    (void)cb, (void)fence;
    for (int i = 0; i < n; i++) {
        const op_hdr *h = handles[i] ? nvn_op_first(handles[i]) : NULL;
        if (!h || h->op != OP_BEGIN || h->size < sizeof *h || nvn_op_next(h)->op != OP_END)
            rt_fatal("vk: op replay is not linked");
    }
    return 0;
}



// ---- Async write-back (REKARINA_ASYNC_WB=1, vk_memory.c vk_mem_mirror_end). The merges and copies that take what a
// submission's GPU work left in the VRAM mirrors back to guest memory (~3.5 ms a frame of OoT's queue 3)
// ran on the graphics queue between full barriers, so the next submissions waited for them. Here the big ones are
// recorded into a command buffer of their own, submitted to a queue of a compute-only family (NVIDIA runs it beside the
// graphics; a second graphics-family queue was time-sliced with the first: graphics 16 -> 32 ms) that waits for the
// submission's timeline value and signals the same value on awb_sem. The completion thread counts
// a value done only once its write-back is too, so a CPU that waits for the fence sees the bytes. A later part that
// writes a mirror range a pending write-back still reads (an upload or a GPU write, vk_memory.c awb_check) waits for
// that write-back's value first.
static int awb_mode = -1;
static VkCommandPool awb_pool;
static VkSemaphore awb_sem;
#define AWB_SLOTS 32
static struct { VkCommandBuffer cb; uint64_t fence; } awb_slot[AWB_SLOTS];
static int awb_next;
static VkCommandBuffer awb_cb;   // being recorded for awb_cb_fence (replay thread, sub_lock)
static uint64_t awb_cb_fence;
static uint64_t awb_need;        // the part being recorded waits for awb_sem >= awb_need
#define AWB_Q 256
static uint64_t awb_q[AWB_Q];    // st_lock: async write-back values submitted, in order
static int awb_qh, awb_qn;

int vk_sync_awb_on(void) {
    if (awb_mode >= 0) return awb_mode;
    VkCtx *c = vk_ctx();
    awb_mode = 0;
    if (!rt_knob("ASYNC_WB", "0")) return 0;
    if (!c->awb_queue) {
        rt_log("vk: async write-back off: no compute-only queue family");
        return 0;
    }
    VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                  .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = c->awb_family};
    VkSemaphoreTypeCreateInfo st = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = 0};
    VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &st};
    if (vkCreateCommandPool(c->device, &pi, NULL, &awb_pool) != VK_SUCCESS) return 0;
    if (vkCreateSemaphore(c->device, &si, NULL, &awb_sem) != VK_SUCCESS) return 0;
    VkCommandBuffer cbs[AWB_SLOTS];
    VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = awb_pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = AWB_SLOTS};
    if (vkAllocateCommandBuffers(c->device, &ai, cbs) != VK_SUCCESS) return 0;
    for (int i = 0; i < AWB_SLOTS; i++) awb_slot[i].cb = cbs[i], awb_slot[i].fence = 0;
    awb_mode = 1;
    rt_log("vk: async write-back on: big mirror write-backs run on a compute-family queue (REKARINA_ASYNC_WB)");
    return 1;
}

static uint64_t timeline_counter(VkDevice device, VkSemaphore sem) {
    uint64_t value;
    VkResult r = vkGetSemaphoreCounterValue(device, sem, &value);
    if (r == VK_SUCCESS && value != UINT64_MAX) return value;
    // NVIDIA can return success with all bits set after device loss. It is not completion.
    vk_device_fault_report(r == VK_SUCCESS ? VK_ERROR_DEVICE_LOST : r);
    if (r == VK_SUCCESS) rt_fatal("vk: timeline counter reads UINT64_MAX (device lost)");
    rt_fatal("vk: vkGetSemaphoreCounterValue failed (%d)", (int)r);
}

uint64_t vk_sync_awb_done(void) {
    return awb_mode > 0 ? timeline_counter(vk_ctx()->device, awb_sem) : UINT64_MAX;
}

static int awb_wait(uint64_t v, uint64_t timeout) {
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &awb_sem,
                              .pValues = &v};
    return vkWaitSemaphores(vk_ctx()->device, &wi, timeout);
}

VkCommandBuffer vk_sync_awb_begin(uint64_t fence) {   // replay thread, sub_lock, at the end of a recording
    if (!vk_sync_awb_on()) return VK_NULL_HANDLE;
    if (awb_cb && awb_cb_fence == fence) return awb_cb;
    int i = awb_next;
    awb_next = (awb_next + 1) % AWB_SLOTS;
    if (awb_slot[i].fence && vk_sync_awb_done() < awb_slot[i].fence) awb_wait(awb_slot[i].fence, UINT64_MAX);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(awb_slot[i].cb, &bi);
    awb_slot[i].fence = fence;
    awb_cb = awb_slot[i].cb, awb_cb_fence = fence;
    return awb_cb;
}

void vk_sync_awb_need(uint64_t fence) {
    if (fence > awb_need) awb_need = fence;
}

// The wait a part's batches carry for a pending write-back it overwrites; 0 when none.
static uint32_t awb_wait_info(VkSemaphoreSubmitInfo *w) {
    if (awb_mode <= 0 || !awb_need) return 0;
    uint64_t v = awb_need;
    awb_need = 0;
    if (vk_sync_awb_done() >= v) return 0;
    *w = (VkSemaphoreSubmitInfo){.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = awb_sem, .value = v,
                                 .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    return 1;
}

// After the submission of value fence: its write-back, if one was recorded.
static void awb_submit(VkCtx *c, uint64_t fence) {
    if (!awb_cb || awb_cb_fence != fence) return;
    vkEndCommandBuffer(awb_cb);
    VkSemaphoreSubmitInfo w = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = c->timeline, .value = fence,
                               .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkSemaphoreSubmitInfo s = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = awb_sem, .value = fence,
                               .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkCommandBufferSubmitInfo ci = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = awb_cb};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .waitSemaphoreInfoCount = 1, .pWaitSemaphoreInfos = &w,
                        .commandBufferInfoCount = 1, .pCommandBufferInfos = &ci, .signalSemaphoreInfoCount = 1,
                        .pSignalSemaphoreInfos = &s};
    VkResult r = vkQueueSubmit2(c->awb_queue, 1, &si, VK_NULL_HANDLE);
    if (r != VK_SUCCESS) {
        vk_device_fault_report(r);
        rt_fatal("vk: vkQueueSubmit2 of the write-back of value %llu failed (%d)", (unsigned long long)fence, (int)r);
    }
    awb_cb = VK_NULL_HANDLE;
    plat_mutex_lock(&st_lock);
    if (awb_qn == AWB_Q) rt_fatal("vk: async write-back queue full");
    awb_q[(awb_qh + awb_qn++) % AWB_Q] = fence;
    plat_mutex_unlock(&st_lock);
}

// Completion thread, st_lock held: the newest pending write-back value at or below reached, popped; 0 none.
static uint64_t awb_pending_upto(uint64_t reached) {
    uint64_t need = 0;
    while (awb_qn && awb_q[awb_qh] <= reached) need = awb_q[awb_qh], awb_qh = (awb_qh + 1) % AWB_Q, awb_qn--;
    return need;
}

// ---- Results before the frame's end (REKARINA_RESULTS_FENCE)
// The game waits for the previous frame's GPU work before it submits the next (nvnSyncWait on the main submission's
// sync, and here the present): on the console that costs nothing, the GPU starts when the submit returns. Here the
// replay records for several milliseconds first, with the GPU half idle, and the frame's heaviest work (lighting, post
// processing, upscaling) comes last, so each frame was the replay's time plus that tail, during which the game thread
// only waited. But what the CPU can see of a frame is produced early: the culling and light lists (written back before
// the scene's passes, vk_exec.c wb_early) and the depth pre-pass's occlusion reports. So a submission may name a results
// point (vk_sync_results_point, at the scene's first lighting pass): the part that ends there signals a second timeline,
// res_sem, with the submission's value, and the side effects recorded up to that point run when it is reached. The
// game's waits (the present, nvnSyncWait) take that value as done; the tail runs on the GPU beside the next frame's
// replay, in queue order, so what the next frame's commands read from this one is unchanged. Side effects recorded after
// the point (and everything queued by other threads: releases, frees) run at the submission's own value, as before.
// A submission without a point signals res_sem with its last part.
static VkSemaphore res_sem;
static int res_mode = -1;             // REKARINA_RESULTS_FENCE, read at start
static uint64_t res_issued, res_done; // st_lock: newest value whose results signal was submitted / whose early side effects ran
static int cur_res_sent, res_point_req;   // sub_lock: the submission being recorded signalled its results / asks to with its next part
static Pending *lqueue;               // st_lock: side effects that wait for the submission's own value
static int lqhead, lqn, lqcap;
static _Thread_local int tl_recording;   // this thread is inside vk_exec_record

int vk_sync_results_on(void) {
    if (res_mode < 0) {
        res_mode = rt_knob("RESULTS_FENCE",
#ifdef GAME_OOT
                           "2"   // the point at the first lighting pass; the late float sums go back at the game's waits (vk_memory.c)
#else
                           "0"
#endif
                           );
        // The tail that runs beside the next frame may read only what the game cannot change under it. A mirror holds
        // the bytes of the submission's sync; an imported pool is the game's own memory. In import mode (integrated
        // GPUs) the game, told at the results point that its frame was done, cleared the count records of its indirect
        // mesh draws for the frame after next while the GPU still drew this one: the first view's eleven draws read
        // their counts, every later one read 0, so foliage lost its shadows on any such GPU (the same on an RTX 5070
        // with REKARINA_MEMORY=import; all 41 right with the early results off, 2026-11-12). So early results need
        // mirrors, unless REKARINA_RESULTS_FENCE is set by hand.
        if (res_mode > 0 && !vk_mirror_on && !cfg_is_set("REKARINA_RESULTS_FENCE")) {
            res_mode = 0;
            rt_log("vk: results before the frame's end: off, the pools are imported and the GPU reads the game's own memory");
        }
        // Shadow storage puts the GPU's writes into guest memory at the submission's fence (vk_gpu_written), not at the
        // results point: told there that its frame was done, the game read its culling and occlusion results before
        // they arrived (with VRAM mirrors on top the frame's occluder pass was left out and the sun's cascades lost most
        // of their casters, 2026-11-15).
        if (res_mode > 0 && vk_mem_shadow_on() && !cfg_is_set("REKARINA_RESULTS_FENCE")) {
            res_mode = 0;
            rt_log("vk: results before the frame's end: off, shadow storage copies the GPU's writes back at the fence");
        }
    }
    return res_mode > 0 && res_sem != VK_NULL_HANDLE;
}

int vk_sync_results_sent(void) { return cur_res_sent || res_point_req; }   // DEBUG-WBWHO: the recording is past its results point
uint64_t vk_dbg_cur_cs;   // DEBUG-WBWHO: the compute program of the dispatch being recorded (first 8 bytes of its input id)

// vk_exec.c: the submission's results are complete with what was recorded so far. 1: the caller ends the part here.
int vk_sync_results_point(void) {
    if (!vk_sync_results_on() || cur_res_sent || res_point_req) return 0;
    res_point_req = 1;
    return 1;
}

static void completion_main(void *arg) {
    VkCtx *c = arg;
    plat_mutex_lock(&st_lock);
    for (;;) {
        int res_on = vk_sync_results_on();
        while (!stop && done == issued && (!res_on || res_done == res_issued) && !(qn && queue[qhead].fence <= res_done) &&
               !(lqn && lqueue[lqhead].fence <= done))
            plat_cond_wait(&cv_work, &st_lock);
        if (stop) break;
        uint64_t reached = done, rreached = res_done;
        if (issued > done || (res_on && res_issued > res_done)) {
            uint64_t want[2] = {done + 1, res_done + 1};
            VkSemaphore sems[2] = {c->timeline, res_sem};
            plat_mutex_unlock(&st_lock);
            VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .flags = VK_SEMAPHORE_WAIT_ANY_BIT,
                                      .semaphoreCount = res_on ? 2u : 1u, .pSemaphores = sems, .pValues = want};
            VkResult r = vkWaitSemaphores(c->device, &wi, 100000000ull);   // 100 ms: stop is checked between waits
            if (r != VK_SUCCESS && r != VK_TIMEOUT) {
                vk_device_fault_report(r);
                rt_fatal("vk: vkWaitSemaphores failed (%d; device lost?)", (int)r);
            }
            // results first: a results value implies every earlier submission's own value, read after it
            if (res_on) rreached = timeline_counter(c->device, res_sem);
            reached = timeline_counter(c->device, c->timeline);
            FT(FT_NOTE, 11, reached, rreached);
            plat_mutex_lock(&st_lock);
            if (reached > issued) reached = issued;
            if (!res_on) rreached = reached;
            else if (rreached > res_issued) rreached = res_issued;
            if (rreached < reached) rreached = reached;   // (a value's own signal covers its results)
            uint64_t aw = awb_pending_upto(reached);   // the write-backs of the values reached (async write-back)
            if (aw) {
                plat_mutex_unlock(&st_lock);
                VkResult ar;
                while ((ar = awb_wait(aw, 100000000ull)) == VK_TIMEOUT && !stop) {}
                if (ar != VK_SUCCESS && ar != VK_TIMEOUT) {
                    vk_device_fault_report(ar);
                    rt_fatal("vk: waiting for an async write-back failed (%d; device lost?)", (int)ar);
                }
                plat_mutex_lock(&st_lock);
            }
        }
        // The side effects due, taken off the queue in batches: a forest frame queues several thousand for its last
        // value, and a lock round trip for each one was a tenth of their time.
        // Both queues in the order of their values: a value's early side effects (due with its results), then its late
        // ones (due with the value itself), then the next value's.
        uint64_t ft_ncb = 0;
        for (;;) {
            static Pending batch[4096];   // this thread only
            int k = 0;
            int early = qn && queue[qhead].fence <= rreached && (!lqn || queue[qhead].fence <= lqueue[lqhead].fence);
            int late = !early && lqn && lqueue[lqhead].fence <= reached && (!qn || lqueue[lqhead].fence < queue[qhead].fence);
            if (early) {
                uint64_t lim = lqn && lqueue[lqhead].fence < rreached ? lqueue[lqhead].fence : rreached;
                while (k < 4096 && qn && queue[qhead].fence <= lim) batch[k++] = queue[qhead++], qn--;
            } else if (late) {
                uint64_t lim = qn && queue[qhead].fence - 1 < reached ? queue[qhead].fence - 1 : reached;
                while (k < 4096 && lqn && lqueue[lqhead].fence <= lim) batch[k++] = lqueue[lqhead++], lqn--;
            } else break;
            plat_mutex_unlock(&st_lock);
            for (int i = 0; i < k; i++) batch[i].fn(batch[i].arg);
            ft_ncb += (uint64_t)k;
            plat_mutex_lock(&st_lock);
        }
        FT(FT_NOTE, 12, ft_ncb, 0);
        if (rreached > res_done && !(qn && queue[qhead].fence <= rreached)) {
            __atomic_store_n(&res_done, rreached, __ATOMIC_RELEASE);
            plat_cond_broadcast(&cv_done);
            FT(FT_NOTE, 30, rreached, 0);
        }
        if (reached > done && !(lqn && lqueue[lqhead].fence <= reached) && !(qn && queue[qhead].fence <= reached)) {
            // The value is done once its side effects ran; freeing what its users held is not one of them, and a thread
            // waiting for the value (the present, the game's own sync waits) does not wait for it.
            __atomic_store_n(&done, reached, __ATOMIC_RELEASE);
            if (res_done < reached) __atomic_store_n(&res_done, reached, __ATOMIC_RELEASE);
            plat_cond_broadcast(&cv_done);
            FT(FT_DONE, reached, 0, 0);
            plat_mutex_unlock(&st_lock);
            vk_mem_collect();
            plat_mutex_lock(&st_lock);
        }
    }
    plat_mutex_unlock(&st_lock);
    return;
}

static void pro_setup(VkCtx *c);   // below: the prologue timeline (copy guards)

// ---- Async compute (vk_exec.c "Queues interleaved", REKARINA_ASYNC_COMPUTE=1, NVIDIA with a compute-only queue family; off
// by default: on the forest path it was no faster than the interleave on the graphics queue, ~75 against ~76 fps). A held rest of the game's compute queue that holds dispatches only is recorded into a command buffer of
// the compute-family queue (the one the async write-back would use) instead of the graphics part, so that it runs
// beside the graphics work, as on the console. Two timeline semaphores order the queues: the graphics part recorded
// up to the signal that let the rest go signals gp_sem (and with it every earlier submission's work, uploads and
// prologues: a signal covers what precedes it in queue order), the compute rest waits for that value and signals
// cp_sem; a graphics part that reaches a wait for a word the rest signalled, and the last part of each submission
// (its write-back reads what the rest wrote; its fence says the rest ran), waits for cp_sem first.
// Resources stay EXCLUSIVE to the graphics family without ownership transfers: NVIDIA's driver keeps their contents
// across its queue families, which the specification does not promise, so other vendors keep the interleave on the
// graphics queue.
static int ac_mode = -1;
static VkCommandPool ac_pool;
static VkSemaphore gp_sem, cp_sem;
static uint64_t gp_val, cp_val;      // the last values signalled (submitted)
static uint64_t cp_need, cp_covered; // the part being recorded waits for cp_sem >= cp_need; the most any part waits for
static int gp_next;                  // the next part submitted signals gp_sem
#define AC_SLOTS 64
static struct { VkCommandBuffer cb; uint64_t value; } ac_slot[AC_SLOTS];
static int ac_next;
int vk_sync_awb_on(void);

int vk_sync_ac_on(void) {
    if (ac_mode >= 0) return ac_mode;
    VkCtx *c = vk_ctx();
    ac_mode = 0;
    if (!rt_knob("ASYNC_COMPUTE", "0")) return 0;
    if (!c->awb_queue || c->awb_family == UINT32_MAX) {
        rt_log("vk: async compute off: no compute-only queue family");
        return 0;
    }
    if (c->props.vendorID != 0x10DE) {
        rt_log("vk: async compute off: shared resources without ownership transfers are kept only by NVIDIA's driver");
        return 0;
    }
    if (vk_sync_awb_on()) {
        rt_log("vk: async compute off: the async write-back uses the compute queue (REKARINA_ASYNC_WB)");
        return 0;
    }
    VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                  .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = c->awb_family};
    VkSemaphoreTypeCreateInfo st = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = 0};
    VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &st};
    if (vkCreateCommandPool(c->device, &pi, NULL, &ac_pool) != VK_SUCCESS) return 0;
    if (vkCreateSemaphore(c->device, &si, NULL, &gp_sem) != VK_SUCCESS) return 0;
    if (vkCreateSemaphore(c->device, &si, NULL, &cp_sem) != VK_SUCCESS) return 0;
    VkCommandBuffer cbs[AC_SLOTS];
    VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = ac_pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = AC_SLOTS};
    if (vkAllocateCommandBuffers(c->device, &ai, cbs) != VK_SUCCESS) return 0;
    for (int i = 0; i < AC_SLOTS; i++) ac_slot[i].cb = cbs[i], ac_slot[i].value = 0;
    ac_mode = 1;
    rt_log("vk: async compute on: the game's compute queue runs beside its graphics queue (REKARINA_ASYNC_COMPUTE)");
    return 1;
}

// A command buffer for the next compute rest; *value: the cp_sem value it will signal. Replay thread, sub_lock.
VkCommandBuffer vk_sync_ac_begin(uint64_t *value) {
    VkCtx *c = vk_ctx();
    int i = ac_next;
    ac_next = (ac_next + 1) % AC_SLOTS;
    if (ac_slot[i].value) {
        uint64_t v = ac_slot[i].value;
        VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &cp_sem,
                                  .pValues = &v};
        vkWaitSemaphores(c->device, &wi, UINT64_MAX);
    }
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(ac_slot[i].cb, &bi);
    ac_slot[i].value = *value = cp_val + 1;
    return ac_slot[i].cb;
}

// Submits it after the graphics value gp it waits for (vk_sync_split_gp).
void vk_sync_ac_submit(VkCommandBuffer cb, uint64_t gp) {
    VkCtx *c = vk_ctx();
    vkEndCommandBuffer(cb);
    VkSemaphoreSubmitInfo w = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = gp_sem, .value = gp,
                               .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkSemaphoreSubmitInfo sg = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = cp_sem, .value = cp_val + 1,
                                .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkCommandBufferSubmitInfo ci = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .waitSemaphoreInfoCount = 1, .pWaitSemaphoreInfos = &w,
                        .commandBufferInfoCount = 1, .pCommandBufferInfos = &ci, .signalSemaphoreInfoCount = 1,
                        .pSignalSemaphoreInfos = &sg};
    VkResult r = vkQueueSubmit2(c->awb_queue, 1, &si, VK_NULL_HANDLE);
    if (r != VK_SUCCESS) {
        vk_device_fault_report(r);
        rt_fatal("vk: vkQueueSubmit2 of a compute rest (value %llu) failed (%d)", (unsigned long long)(cp_val + 1), (int)r);
    }
    cp_val++;
}

uint64_t vk_sync_ac_last(void) { return cp_val; }
uint64_t vk_sync_ac_covered(void) { return cp_covered; }
// The part being recorded starts only once cp_sem reached v.
void vk_sync_ac_need(uint64_t v) {
    if (v > cp_need) cp_need = v;
    if (v > cp_covered) cp_covered = v;
}

// The waits and signals a graphics part's batch carries for async compute: *nw / *ns entries written at w / sg.
static void ac_part_sems(VkSemaphoreSubmitInfo *w, uint32_t *nw, VkSemaphoreSubmitInfo *sg, uint32_t *ns) {
    if (ac_mode <= 0) return;
    if (cp_need) {
        w[(*nw)++] = (VkSemaphoreSubmitInfo){.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = cp_sem,
                                             .value = cp_need, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
        cp_need = 0;
    }
    if (gp_next) {
        sg[(*ns)++] = (VkSemaphoreSubmitInfo){.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = gp_sem,
                                              .value = ++gp_val, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
        gp_next = 0;
    }
}

void vk_sync_start(void) {
    if (__atomic_load_n(&started, __ATOMIC_ACQUIRE)) return;
    plat_mutex_lock(&start_lock);
    if (!started) {
        if (!vk_init()) rt_fatal("vk: no usable Vulkan device (see the vk: lines above; REKARINA_GPU=null runs without a GPU)");
        VkCtx *c = vk_ctx();
        pro_setup(c);
        {   // the results timeline
            VkSemaphoreTypeCreateInfo rst = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                             .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = issued};
            VkSemaphoreCreateInfo rsi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &rst};
            if (vkCreateSemaphore(c->device, &rsi, NULL, &res_sem) != VK_SUCCESS) res_sem = VK_NULL_HANDLE;
            rt_log("vk: results before the frame's end: %s (REKARINA_RESULTS_FENCE)", vk_sync_results_on() ? "on" : "off");
        }
        if (!ts_state && perf_gpu_timing()) ts_setup(c);
        perf_gpu_stats = gpu_stats;
        uint64_t now = timeline_counter(c->device, c->timeline);
        if (issued > now) {   // a new device after vk_shutdown: values keep increasing for the front end
            VkSemaphoreSignalInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO, .semaphore = c->timeline,
                                        .value = issued};
            if (vkSignalSemaphore(c->device, &si) != VK_SUCCESS) rt_fatal("vk: timeline restart at %llu", (unsigned long long)issued);
            now = issued;
        }
        plat_mutex_lock(&st_lock);
        issued = done = now;
        res_issued = res_done = now;
        if (reserved < now) reserved = now;
        stop = 0;
        plat_mutex_unlock(&st_lock);
        vk_mem_note_submit(now);
        completion_thread = plat_thread_new(completion_main, c, "vk completion");
        if (!completion_thread) rt_fatal("vk: cannot start the completion thread");
        __atomic_store_n(&started, 1, __ATOMIC_RELEASE);
        vk_astc_mode();   // decides REKARINA_ASTC and builds the ASTC decoder now, not at the first ASTC upload
    }
    plat_mutex_unlock(&start_lock);
}

// The index of a command buffer the GPU no longer uses, claimed for value `fence`. Called with sub_lock held; may move
// slots[].
static int claim_slot(VkCtx *c, uint64_t fence) {
    uint64_t gpu = timeline_counter(c->device, c->timeline);
    for (int i = 0; i < nslots; i++)
        if (slots[i].fence <= gpu) {
            slots[i].fence = fence;
            return i;
        }
    CmdSlot *n = realloc(slots, sizeof *n * (size_t)(nslots + 1));
    if (!n) rt_fatal("vk: out of memory (command buffers)");
    slots = n;
    CmdSlot *s = &slots[nslots];
    VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                  .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = c->gfx_family};
    if (vkCreateCommandPool(c->device, &pi, NULL, &s->pool) != VK_SUCCESS) rt_fatal("vk: submission command pool");
    VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = s->pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    if (vkAllocateCommandBuffers(c->device, &ai, &s->cb) != VK_SUCCESS) rt_fatal("vk: command buffer allocation");
    s->fence = fence;
    return nslots++;
}

// Prologues (the VRAM mirrors' CPU -> mirror copies) signal a timeline of their own, so the copy guards (memory.c)
// know when the GPU has read the guest bytes they copy.
static VkSemaphore pro_sem;
static uint64_t pro_val;
static uint64_t pro_done(void) {
    return pro_sem ? timeline_counter(vk_ctx()->device, pro_sem) : 0;
}
static int pro_wait(uint64_t v) {
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1, .pSemaphores = &pro_sem,
                              .pValues = &v};
    return pro_sem && vkWaitSemaphores(vk_ctx()->device, &wi, UINT64_MAX) == VK_SUCCESS;
}
static void pro_setup(VkCtx *c) {
    VkSemaphoreTypeCreateInfo st = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO,
                                    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE, .initialValue = 0};
    VkSemaphoreCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO, .pNext = &st};
    if (vkCreateSemaphore(c->device, &si, NULL, &pro_sem) != VK_SUCCESS) { pro_sem = VK_NULL_HANDLE; return; }
    mem_copy_guard_setup(pro_done, pro_wait);
    vk_mem_prologue_timeline(pro_done, pro_wait);
}

// Counts the command buffers the replay begins: what it keeps about a command buffer's state (the dynamic state last
// set, the index buffer bound) holds while this stays the same.
static uint64_t cb_serial;
uint64_t vk_sync_cb_serial(void) { return __atomic_load_n(&cb_serial, __ATOMIC_RELAXED); }

static CmdSlot *take_slot(VkCtx *c, uint64_t fence) {
    __atomic_fetch_add(&cb_serial, 1, __ATOMIC_RELAXED);
    cur_slot = claim_slot(c, fence);
    return &slots[cur_slot];
}

// The copies the replay's first accesses to VRAM mirrors queued (vk_memory.c), in a command buffer of their own that is
// submitted right before the part that needs them and timed like a part; VK_NULL_HANDLE when there are none. After the
// part was recorded, sub_lock held; may move slots[].
// First brings the shadow storage up to date with the CPU's writes, which needs no command buffer.
static VkCommandBuffer prologue(VkCtx *c, uint64_t fence) {
    vk_mem_shadow_sync();   // shadow storage: the CPU's writes go in before the part is submitted
    if (!vk_mem_mirror_pending()) return VK_NULL_HANDLE;
    int i = claim_slot(c, fence);   // before reading slots: the claim may move it
    VkCommandBuffer cb = slots[i].cb;
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cb, &bi);
    int pair = ts_open(cb);
    if (pair >= 0 && sec_on) ts_pro[pair] = 1;
    VkPassMark *pm = vk_pass_mark_begin(cb, VK_MARK_PROLOGUE, fence);
    vk_mem_mirror_prologue(cb);
    vk_pass_mark_end(cb, pm);
    ts_close(cb, pair, fence);
    vkEndCommandBuffer(cb);
    return cb;
}

static uint64_t spk_rec, spk_upl, spk_pro, spk_que;   // DEBUG-SPIKE
// ---- The submitter thread (REKARINA_SUBMIT_THREAD, default on for OoT)
// vkQueueSubmit2 of a part costs the calling thread 40-220 us in the driver, and a frame of OoT submits 30-45
// parts and 7 final batches: ~1.5 ms of the ~14 ms the replay thread needs for a frame in the village, the thread the
// whole frame waits for. The replay thread now only copies the batch into a queue; this thread makes the call, in queue
// order, under the queue lock. Every other user of the graphics queue (the present's blit, texture uploads, the
// vkDeviceWaitIdle calls) takes that lock through vk_queue_lock, which first waits until the queue of batches is empty
// (vk_sync_submits_drain), so a batch queued before such a call still reaches the driver before it. The values a batch
// signals are published when it is queued: a wait for one starts before the driver has the batch and ends when the GPU
// signals it, as before.
#define CHAIN_MAX 48                     // command buffers of one part beside its last (vk_sync_chain)
typedef struct {
    VkSubmitInfo2 si[2];
#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
    VkLatencySubmissionPresentIdNV lat[2];
#endif
    VkCommandBufferSubmitInfo cbi[2][CHAIN_MAX + 1];
    VkSemaphoreSubmitInfo waits[2][4], sigs[2][4];
    uint32_t n;
    uint64_t fence;
    volatile int32_t *ended;   // NULL, or the flag the recorder sets once it ended the part's command buffer (rec_end_async)
    VkFence vkfence;           // an upload batch's fence (vk_sync_submit_upload), else VK_NULL_HANDLE
} SubJob;
// The flags of the parts that ended without a wait, taken in turn: a part's flag is read before 256 more parts end.
static volatile int32_t part_ended[256];
static uint32_t part_seq;

static void wait_ended(volatile int32_t *ended) {
    static int spins;
    if (!spins) spins = (int)plat_spin_count(4000);   // as long on any processor (platform.h)
    for (int i = 0; ended && !__atomic_load_n(ended, __ATOMIC_ACQUIRE); i++)
        if (i < spins) _mm_pause();
        else plat_yield();
}
#define SQ_N 512
static SubJob sq[SQ_N];
static uint64_t sq_put, sq_done;   // sq_lock: batches queued, batches the driver has
static PlatMutex sq_lock = PLAT_MUTEX_INIT;
static PlatCond sq_work = PLAT_COND_INIT, sq_idle = PLAT_COND_INIT;
static int sq_on = -1, sq_started;
static _Thread_local int sq_me;   // the submitter thread itself

static void sq_main(void *arg) {
    (void)arg;
    sq_me = 1;
    VkCtx *c = vk_ctx();
    plat_mutex_lock(&sq_lock);
    for (;;) {
        while (sq_done == sq_put) plat_cond_wait(&sq_work, &sq_lock);
        SubJob *j = &sq[sq_done % SQ_N];
        plat_mutex_unlock(&sq_lock);
        wait_ended(j->ended);
        vk_queue_lock();
        vk_drv_urgent(1);   // queue work first in the driver (gfx_present.c "driver turns")
        VkResult r = vkQueueSubmit2(c->gfx_queue, j->n, j->si, j->vkfence);
        vk_drv_urgent(-1);
        vk_queue_unlock();
        if (r != VK_SUCCESS) {
            vk_device_fault_report(r);
            rt_fatal("vk: vkQueueSubmit2 of value %llu failed (%d)", (unsigned long long)j->fence, (int)r);
        }
        plat_mutex_lock(&sq_lock);
        __atomic_store_n(&sq_done, sq_done + 1, __ATOMIC_RELEASE);
        if (sq_done == sq_put) plat_cond_broadcast(&sq_idle);
        plat_cond_broadcast(&sq_work);   // a producer waiting for room
    }
}

static int sq_enabled(void) {
    if (sq_on < 0) {
#ifdef GAME_OOT
        sq_on = rt_knob("SUBMIT_THREAD", "1");
#else
        sq_on = rt_knob("SUBMIT_THREAD", "0");
#endif
        if (sq_on && !sq_started) {
            sq_started = 1;
            if (!plat_thread_spawn(sq_main, NULL, "vk submit", 0)) sq_on = 0;
            else rt_log("vk: the parts of a submission are handed to the driver by a submitter thread (REKARINA_SUBMIT_THREAD)");
        }
    }
    return sq_on;
}

// The driver has every batch queued so far. Called by vk_queue_lock (any thread but the submitter).
void vk_sync_submits_drain(void) {
    if (sq_on <= 0 || sq_me) return;
    plat_mutex_lock(&sq_lock);
    while (sq_done != sq_put) plat_cond_wait(&sq_idle, &sq_lock);
    plat_mutex_unlock(&sq_lock);
}

// n batches (1 or 2) to the graphics queue, in order after every batch queued before; VK_SUCCESS when queued.
static VkResult submit_batches(VkCtx *c, uint32_t n, const VkSubmitInfo2 *si, uint64_t fence, volatile int32_t *ended) {
    uint64_t reflex_id = vk_reflex_current_id();
    if (!sq_enabled()) {
        VkSubmitInfo2 local[2];
#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
        VkLatencySubmissionPresentIdNV lat[2];
#endif
        for (uint32_t k = 0; k < n; k++) {
            local[k] = si[k];
#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
            vk_reflex_chain_submit(&local[k], &lat[k], reflex_id);
#endif
        }
        wait_ended(ended);
        vk_queue_lock();
        vk_drv_urgent(1);
        VkResult r = vkQueueSubmit2(c->gfx_queue, n, local, VK_NULL_HANDLE);
        vk_drv_urgent(-1);
        vk_queue_unlock();
        return r;
    }
    plat_mutex_lock(&sq_lock);
    while (sq_put - sq_done == SQ_N) plat_cond_wait(&sq_work, &sq_lock);
    SubJob *j = &sq[sq_put % SQ_N];
    j->n = n, j->fence = fence, j->ended = ended, j->vkfence = VK_NULL_HANDLE;
    for (uint32_t k = 0; k < n; k++) {
        j->si[k] = si[k];
        if (si[k].waitSemaphoreInfoCount > 4 || si[k].signalSemaphoreInfoCount > 4 || si[k].commandBufferInfoCount > CHAIN_MAX + 1)
            rt_fatal("vk: a batch of value %llu is larger than the submitter's job", (unsigned long long)fence);
        for (uint32_t i = 0; i < si[k].waitSemaphoreInfoCount; i++) j->waits[k][i] = si[k].pWaitSemaphoreInfos[i];
        for (uint32_t i = 0; i < si[k].signalSemaphoreInfoCount; i++) j->sigs[k][i] = si[k].pSignalSemaphoreInfos[i];
        for (uint32_t i = 0; i < si[k].commandBufferInfoCount; i++) j->cbi[k][i] = si[k].pCommandBufferInfos[i];
        j->si[k].pWaitSemaphoreInfos = j->waits[k], j->si[k].pSignalSemaphoreInfos = j->sigs[k];
        j->si[k].pCommandBufferInfos = j->cbi[k];
#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
        vk_reflex_chain_submit(&j->si[k], &j->lat[k], reflex_id);
#endif
    }
    sq_put++;
    plat_cond_broadcast(&sq_work);
    plat_mutex_unlock(&sq_lock);
    return VK_SUCCESS;
}

// The upload batch (vk_memory.c vk_upload_flush) through the submitter too, in queue order before every batch queued
// after it: the replay thread's flush before a part was a vkQueueSubmit of its own behind vk_queue_lock, which first
// waits until the submitter has handed every queued part to the driver (+1.4 ms in the forest's slowest frames, the
// replay thread off the processor inside its submission). 0: not queued (no submitter thread, or
// REKARINA_UPLOAD_QUEUED=0): the caller submits it itself.
uint64_t vk_sync_submit_upload(VkCommandBuffer cb, VkFence vkfence) {
    static int on = -1;
    if (on < 0) on = rt_knob("UPLOAD_QUEUED", "1");
    if (!on || !sq_enabled() || sq_me) return 0;
    plat_mutex_lock(&sq_lock);
    while (sq_put - sq_done == SQ_N) plat_cond_wait(&sq_work, &sq_lock);
    SubJob *j = &sq[sq_put % SQ_N];
    j->n = 1, j->fence = 0, j->ended = NULL, j->vkfence = vkfence;
    j->cbi[0][0] = (VkCommandBufferSubmitInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb};
    j->si[0] = (VkSubmitInfo2){.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .commandBufferInfoCount = 1, .pCommandBufferInfos = j->cbi[0]};
    const uint64_t ticket = ++sq_put;
    plat_cond_broadcast(&sq_work);
    plat_mutex_unlock(&sq_lock);
    return ticket;
}

int vk_sync_upload_submitted(uint64_t ticket, int wait) {
    if (!ticket || __atomic_load_n(&sq_done, __ATOMIC_ACQUIRE) >= ticket) return 1;
    if (!wait) return 0;
    plat_mutex_lock(&sq_lock);
    while (sq_done < ticket) plat_cond_wait(&sq_work, &sq_lock);
    plat_mutex_unlock(&sq_lock);
    return 1;
}

// ---- A part of several command buffers (vk_exec.c "Draw stretches on worker threads")
// vk_sync_chain ends the command buffer being recorded and keeps it, begins n more for other threads to record (each
// from a slot and pool of its own; those threads end them), then a new current one, which it returns. The next
// vk_sync_split submits them all in that order as the part's one batch, behind the uploads and the prologue their
// recording asked for, after vk_exec_par_join says the other threads are done. The part's timestamps span the batch.
// VK_NULL_HANDLE: the part holds too many already, split first.
static VkCommandBuffer chain_cb[CHAIN_MAX];
static int chain_n;
WEAK void vk_exec_par_join(void) {}

VkCommandBuffer vk_sync_chain(int n, VkCommandBuffer *cbs) {
    VkCtx *c = vk_ctx();
    if (chain_n + n + 1 > CHAIN_MAX) return VK_NULL_HANDLE;
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VkCommandBuffer cb = slots[cur_slot].cb;
    vkEndCommandBuffer(cb);
    chain_cb[chain_n++] = cb;
    for (int i = 0; i < n; i++) {
        int k = claim_slot(c, cur_fence);   // may move slots[]: the handle is taken at once
        cbs[i] = chain_cb[chain_n++] = slots[k].cb;
        vkBeginCommandBuffer(cbs[i], &bi);
    }
    __atomic_fetch_add(&cb_serial, 1, __ATOMIC_RELAXED);
    cur_slot = claim_slot(c, cur_fence);
    VkCommandBuffer nc = slots[cur_slot].cb;
    vkBeginCommandBuffer(nc, &bi);
    vk_crumb(nc, VK_CRUMB_SUBMIT, cur_fence);
    return nc;
}

int vk_sync_chain_pending(void) { return chain_n; }

// How the submissions end is counted (REKARINA_TRACE=perf, "last batches of ..."): of 90,112 in a Market run none had an
// empty last batch after parts had gone, 77,220 had nothing recorded at all (an empty batch that only signals the
// value), none ended with the chunks of a draw stretch (vk_exec.c splits after them).
static int cur_parts;   // sub_lock: parts of the submission being recorded that were submitted already

// DEBUG-LIVECHECK: what each submission sent, by its value: bit 0 its last batch had command buffers, bit 1 parts went
// before it; -1 unknown (too old, or not submitted yet).
static uint64_t sub_work[4096];
int vk_sync_sub_work(uint64_t value) {
    uint64_t w = __atomic_load_n(&sub_work[value & 4095], __ATOMIC_RELAXED);
    return w >> 2 == value ? (int)(w & 3) : -1;
}

VkCommandBuffer vk_sync_split(void) {
    VkCtx *c = vk_ctx();
    vk_exec_par_join();   // the chained command buffers are recorded, and what their recording asked for is queued
    VkCommandBuffer cb = slots[cur_slot].cb;
    ts_end(cb, cur_fence);
    uint64_t spk_a = plat_now_ns();   // DEBUG-SPIKE
    // every command of the part is in cb, ended by its recorder: without a wait where the recorder takes that (the
    // part's submission waits for the flag), else at once
    volatile int32_t *ended = &part_ended[part_seq++ & 255];
    __atomic_store_n(ended, 0, __ATOMIC_RELEASE);
    if (!rec_end_async(cb, ended)) {
        if (!rec_end_cb(cb, 0, 0)) vkEndCommandBuffer(cb);
        ended = NULL;
    }
    uint64_t spk_b = plat_now_ns();
    spk_rec += spk_b - spk_a;
    vk_upload_flush(0);   // the uploads this part asked for precede it in queue order
    spk_upl += plat_now_ns() - spk_b;
    if (vk_listlog_addr())   // DEBUG-LISTLOG
        rt_log("LISTLOG split fence %llu snap %016llx at %.3f", (unsigned long long)cur_fence,
               (unsigned long long)vk_listlog_hash(), vk_listlog_ms());
    uint64_t spk_c = plat_now_ns();   // DEBUG-SPIKE
    VkCommandBuffer pro = prologue(c, cur_fence);
    spk_pro += plat_now_ns() - spk_c;
    VkCommandBufferSubmitInfo cbi[2] = {{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = pro},
                                        {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb}};
    // The prologue as a submission of its own that signals the prologue timeline (copy guards), then the part.
    VkSemaphoreSubmitInfo psig = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = pro_sem,
                                  .value = pro_val + 1, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkSemaphoreSubmitInfo aw[2], asg[2];
    uint32_t naw = awb_wait_info(&aw[0]), npw = naw, nas = 0;   // a pending write-back this part overwrites (async write-back)
    ac_part_sems(aw, &naw, asg, &nas);   // async compute: a compute rest this part waits for, graphics progress it signals
    int send_res = res_point_req && !cur_res_sent && vk_sync_results_on();
    res_point_req = 0;
    if (send_res)   // the part that ends at the results point
        asg[nas++] = (VkSemaphoreSubmitInfo){.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = res_sem,
                                             .value = cur_fence, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkSubmitInfo2 si[2] = {{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .waitSemaphoreInfoCount = npw, .pWaitSemaphoreInfos = aw,
                            .commandBufferInfoCount = 1, .pCommandBufferInfos = cbi,
                            .signalSemaphoreInfoCount = pro_sem ? 1u : 0u, .pSignalSemaphoreInfos = &psig},
                           {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .waitSemaphoreInfoCount = naw, .pWaitSemaphoreInfos = aw,
                            .commandBufferInfoCount = 1, .pCommandBufferInfos = cbi + 1, .signalSemaphoreInfoCount = nas,
                            .pSignalSemaphoreInfos = asg}};
    VkCommandBufferSubmitInfo chain_ci[CHAIN_MAX + 1];
    if (chain_n) {   // the part is the chained command buffers and this one, in order (vk_sync_chain)
        for (int i = 0; i < chain_n; i++)
            chain_ci[i] = (VkCommandBufferSubmitInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = chain_cb[i]};
        chain_ci[chain_n] = cbi[1];
        si[1].commandBufferInfoCount = (uint32_t)chain_n + 1, si[1].pCommandBufferInfos = chain_ci;
        chain_n = 0;
    }
    ts_submitted();
    cur_parts++;
    uint64_t spk_d = plat_now_ns();   // DEBUG-SPIKE
    VkResult r = submit_batches(c, pro ? 2u : 1u, pro ? si : si + 1, cur_fence, ended);
    spk_que += plat_now_ns() - spk_d;
    FT(FT_PART, cur_fence, 0, pro != VK_NULL_HANDLE);
    if (send_res && r == VK_SUCCESS) {
        cur_res_sent = 1;
        plat_mutex_lock(&st_lock);
        res_issued = cur_fence;
        plat_cond_broadcast(&cv_work);
        plat_mutex_unlock(&st_lock);
        FT(FT_NOTE, 31, cur_fence, 0);
    }
    if (__builtin_expect(ftrace_on, 0)) {
        uint64_t wp, wm;
        vk_mem_wb_pending(&wp, &wm);
        FT(FT_NOTE, 20, wp, wm);
    }
    if (pro && pro_sem && r == VK_SUCCESS) vk_mem_prologue_submitted(++pro_val);
    if (r != VK_SUCCESS) {
        vk_device_fault_report(r);
        rt_fatal("vk: vkQueueSubmit2 of a part of value %llu failed (%d)", (unsigned long long)cur_fence, (int)r);
    }
    vk_mem_note_submit(cur_fence);   // objects destroyed from here on may be in use by the part just submitted
    CmdSlot *s = take_slot(c, cur_fence);   // the submitted part keeps its slot until the value is reached
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(s->cb, &bi);
    vk_crumb(s->cb, VK_CRUMB_SUBMIT, cur_fence);
    ts_begin(s->cb);
    rec_begin(s->cb, cur_fence);
    return s->cb;
}

// vk_sync_split, the part's batch also signalling gp_sem; *gp: that value.
VkCommandBuffer vk_sync_split_gp(uint64_t *gp) {
    gp_next = 1;
    VkCommandBuffer cb = vk_sync_split();
    *gp = gp_val;
    return cb;
}

// Asynchronous replay (vk_backend.c): the value the next queued submission or present will signal. Values are handed
// out in queue order and the replay thread submits them in that order, so each vk_sync_submit gets exactly its value.
uint64_t vk_sync_reserve(void) {
    vk_sync_start();
    plat_mutex_lock(&st_lock);
    if (reserved < issued) reserved = issued;
    uint64_t v = ++reserved;
    plat_mutex_unlock(&st_lock);
    return v;
}

// DEBUG-LISTLOG: REKARINA_LISTLOG=<guest addr>: a hash of 64 KiB of guest memory there at each submission's start, at its
// prologue (the submit-time snapshot) and at the present, with times (when the CPU rewrites OoT's light lists)
static uint64_t ll_addr = 1, ll_t0;
uint64_t vk_listlog_addr(void) {
    if (ll_addr == 1) { const char *e = cfg_get("REKARINA_LISTLOG"); ll_addr = e ? strtoull(e, NULL, 0) : 0; ll_t0 = plat_now_ns(); }
    return ll_addr;
}
uint64_t vk_listlog_hash(void) {
    if (!vk_listlog_addr()) return 0;
    const uint64_t *p = (const uint64_t *)(uintptr_t)ll_addr;
    uint64_t h = 1469598103934665603ull;
    for (int i = 0; i < 8192; i++) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}
double vk_listlog_ms(void) { return (plat_now_ns() - ll_t0) / 1e6; }
// DEBUG-LISTLOG: REKARINA_LISTLOG_GRID=1: c3[148].x and c3[149].x of OoT's two scene uniform blocks (porch), as text
static void ll_grid(char out[64]) {
    static int on = -1;
    if (on < 0) { const char *e = cfg_get("REKARINA_LISTLOG_GRID"); on = e && *e == '1'; }
    if (!on) { out[0] = 0; return; }
    const int32_t *a = (const int32_t *)(uintptr_t)(0x25e94b000ull + 148 * 16), *b = (const int32_t *)(uintptr_t)(0x25e94e500ull + 148 * 16);
    snprintf(out, 64, " gridA %d/%d gridB %d/%d", a[0], a[4], b[0], b[4]);
}

// DEBUG-EPOCHEARLY: REKARINA_EPOCH_EARLY=1 makes the texture module's per-submission step before the submit lock, as it
// was until 2026-11-14 (vk_backend.c submit_sync; an empty submission then has none): the order behind the device
// fault after map changes, kept as a switch to show it again.
int vk_sync_epoch_early(void) {
    static int v = -1;
    if (v < 0) {
        v = rt_knob("EPOCH_EARLY", "0") != 0;
        if (v) rt_log("vk: the texture epoch moves before the submit lock (REKARINA_EPOCH_EARLY=1, a test: the old order)");
    }
    return v;
}

// DEBUG-STALL (a test): REKARINA_TEST_STALL=<presents>:<ms>: once per map change (prep.c prep_map_frame), the first
// submission recorded that many presents after it sleeps <ms> inside the replay's recording, with the submit lock held.
// It makes on demand the long replay frame a slow or paging machine has at an arrival (dv12, 2026-11-16: a 9.5 s
// frame five presents after the village's map change, then a read of a render target destroyed during it), while
// the game's threads go on loading and finalizing what the old map left.
static void test_stall(void) {
    static int after = -2, ms;
    static uint64_t done_for;
    if (__builtin_expect(after == -2, 0)) {
        const char *e = cfg_get("REKARINA_TEST_STALL"), *colon = e ? strchr(e, ':') : NULL;
        after = e && *e && colon ? atoi(e) : -1;
        ms = colon ? atoi(colon + 1) : 0;
        if (ms <= 0) after = -1;
        if (after >= 0) rt_log("vk: TEST STALL: the replay sleeps %d ms %d presents after each map change (REKARINA_TEST_STALL)", ms, after);
    }
    if (after < 0) return;
    uint64_t mf = prep_map_frame(), now = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    if (!mf || mf == done_for || now < mf + (uint64_t)after) return;
    done_for = mf;
    rt_log("vk: TEST STALL: frame %llu (map change at %llu): the replay sleeps %d ms", (unsigned long long)now,
           (unsigned long long)mf, ms);
    plat_sleep_ms((uint32_t)ms);
    rt_log("vk: TEST STALL: frame %llu: the replay goes on", (unsigned long long)now);
}

VkSpike vk_spike;
volatile uint64_t vk_spike_sub_t0;   // DEBUG-SLOWSUB: when the submission being recorded began (0: none), for the sampler
uint64_t vk_sync_submit(const uint64_t *handles, int n) {
    vk_sync_start();
    uint64_t spk_t0 = plat_now_ns();   // DEBUG-SPIKE
    uint64_t cyc0 = __builtin_expect(sec_on, 0) ? plat_thread_cycles() : 0, tsc0 = cyc0 ? sec_clock() : 0;   // SUB_OFFCPU
    __atomic_store_n(&vk_spike_sub_t0, spk_t0, __ATOMIC_RELAXED);
    uint64_t ll_h0 = vk_listlog_hash();   // DEBUG-LISTLOG
    double ll_ms0 = vk_listlog_ms();
    char ll_g0[64];
    if (vk_listlog_addr()) ll_grid(ll_g0);
    VkCtx *c = vk_ctx();
    plat_mutex_lock(&sub_lock);
    uint64_t spk_lock = plat_now_ns() - spk_t0;   // DEBUG-SPIKE
    FT(FT_NOTE, 7, spk_lock, 0);
    {   // how often a submit call arrives while another thread records (REKARINA_TRACE=perf): the case the step below is for
        static uint64_t n_all, n_waited, n_games, n_games_waited;
        n_all++, n_games += n != 0;
        if (spk_lock > 20000) n_waited++, n_games_waited += n != 0;   // 20 us: the lock was held
        if ((rt_trace & RT_TRACE_PERF) && !(n_all & 8191))
            rt_log("vk: %llu submissions so far (%llu of the game's): %llu of the calls waited for another thread's recording "
                   "(%llu of the game's)", (unsigned long long)n_all, (unsigned long long)n_games,
                   (unsigned long long)n_waited, (unsigned long long)n_games_waited);
    }
    // The texture module's per-submission step, with the lock held and for every submission, an empty one too. It moves
    // the epoch that says "this submission has not looked yet": the texture pool's set is brought up to date before a
    // submission's first bind of it (vk_texpool.c), a slot's storage is checked at its first binding. Made before the
    // lock (vk_backend.c submit_sync, until 2026-11-14), a second thread's submit call moved the epoch while the first
    // thread was still recording. The first made the pool's update for the new epoch in the middle of its recording,
    // the second found it made and recorded a whole submission with none: its draws read the pool's entries as they
    // stood, among them those of textures that went while the first recorded, whose images are destroyed as soon as
    // the GPU is past the first submission. That is an invalid read at a destroyed image's address (the device fault
    // behind a map change, 2026-11-14) or another texture's texels for a frame. An empty submission that records held
    // rests (vk_exec.c gfx_sync_fence) had no epoch of its own either.
    if (!vk_sync_epoch_early()) vk_texture_new_submit();
    memset(&vk_spike, 0, sizeof vk_spike), spk_rec = spk_upl = spk_pro = spk_que = 0;
    int spk_handles = n;
    vk_mem_dirty_submit();   // REKARINA_TRACE=perf: what a mirror would upload before this submit
    uint64_t fence = __atomic_load_n(&issued, __ATOMIC_ACQUIRE) + 1;
    cur_fence = fence;
    cur_res_sent = res_point_req = 0;
    cur_parts = 0;
    CmdSlot *s = take_slot(c, fence);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    uint64_t t0 = perf_begin();
    test_stall();   // DEBUG-STALL
    vkBeginCommandBuffer(s->cb, &bi);
    vk_crumb(s->cb, VK_CRUMB_SUBMIT, fence);
    ts_begin(s->cb);
    rec_begin(s->cb, fence);
    tl_recording = 1;
    int recorded = vk_exec_record(s->cb, handles, n, fence);
    tl_recording = 0;
    FT(FT_NOTE, 8, fence, recorded);
    s = &slots[cur_slot];   // the last part (vk_sync_split)
    int last = recorded != 0;
    if (last) ts_end(s->cb, fence);
    else ts_drop();
    // The recorder is done with the part: the end-of-submission crumb, then the end, by the recorder when it records.
    uint64_t spk_a = plat_now_ns();   // DEBUG-SPIKE
    if (!rec_end_cb(s->cb, VK_CRUMB_SUBMIT_END, fence)) vkEndCommandBuffer(s->cb);
    spk_rec += plat_now_ns() - spk_a;
    perf_replay(t0);
    // After the recording: the uploads it asked for (and every earlier one) precede it in queue order.
    vk_upload_flush(0);
    if (vk_listlog_addr())   // DEBUG-LISTLOG
    {
        char g1[64];
        ll_grid(g1);
        rt_log("LISTLOG sub frame %llu fence %llu begin %016llx snap %016llx at %.3f rec %.3f ms |%s ->%s",
               (unsigned long long)(__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1), (unsigned long long)fence,
               (unsigned long long)ll_h0, (unsigned long long)vk_listlog_hash(), ll_ms0, vk_listlog_ms() - ll_ms0, ll_g0, g1);
    }
    FT(FT_NOTE, 9, fence, 0);
    if (__builtin_expect(ftrace_on, 0)) {
        uint64_t wp, wm;
        vk_mem_wb_pending(&wp, &wm);
        FT(FT_NOTE, 21, wp, wm);
    }
    spk_a = plat_now_ns();
    VkCommandBuffer pro = prologue(c, fence);
    spk_pro += plat_now_ns() - spk_a;
    FT(FT_NOTE, 10, fence, 0);
    s = &slots[cur_slot];
    VkCommandBufferSubmitInfo cbi[2] = {{.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = pro},
                                        {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = s->cb}};
    VkSemaphoreSubmitInfo sig = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = c->timeline,
                                 .value = fence, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkSemaphoreSubmitInfo psig = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = pro_sem,
                                  .value = pro_val + 1, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkSemaphoreSubmitInfo aw[2], sigs[3] = {sig};
    uint32_t naw = awb_wait_info(&aw[0]), npw = naw, nsig = 1;   // a pending write-back this part overwrites (async write-back)
    ac_part_sems(aw, &naw, sigs, &nsig);   // async compute: the compute rests the last part waits for
    if (res_sem && !cur_res_sent)   // no results point in this submission: its results are done with it
        sigs[nsig++] = (VkSemaphoreSubmitInfo){.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = res_sem,
                                               .value = fence, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkSubmitInfo2 si[2] = {{.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .waitSemaphoreInfoCount = npw, .pWaitSemaphoreInfos = aw,
                            .commandBufferInfoCount = 1, .pCommandBufferInfos = cbi,
                            .signalSemaphoreInfoCount = pro_sem ? 1u : 0u, .pSignalSemaphoreInfos = &psig},
                           {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .waitSemaphoreInfoCount = naw, .pWaitSemaphoreInfos = aw,
                            .commandBufferInfoCount = last ? 1u : 0u,
                            .pCommandBufferInfos = cbi + 1, .signalSemaphoreInfoCount = nsig, .pSignalSemaphoreInfos = sigs}};
    __atomic_store_n(&sub_work[fence & 4095], fence << 2 | (uint64_t)(last ? 1 : 0) | (uint64_t)(cur_parts ? 2 : 0), __ATOMIC_RELAXED);
    {   // REKARINA_TRACE=perf: how the submissions end (with the log line of the waits above)
        static uint64_t n_subs_all, n_empty_after_parts, n_empty_alone;
        n_subs_all++;
        n_empty_after_parts += !last && cur_parts;
        n_empty_alone += !last && !cur_parts;
        if ((rt_trace & RT_TRACE_PERF) && !(n_subs_all & 8191))
            rt_log("vk: last batches of %llu submissions: %llu empty after parts had gone, %llu empty with no part (nothing "
                   "recorded)", (unsigned long long)n_subs_all, (unsigned long long)n_empty_after_parts,
                   (unsigned long long)n_empty_alone);
    }
    ts_submitted();
    uint64_t spk_q0 = plat_now_ns();   // DEBUG-SPIKE
    VkResult r = submit_batches(c, pro ? 2u : 1u, pro ? si : si + 1, fence, NULL);
    spk_que += plat_now_ns() - spk_q0;
    FT(FT_PART, fence, 1 + (recorded ? 0 : 1), pro != VK_NULL_HANDLE);
    if (pro && pro_sem && r == VK_SUCCESS) vk_mem_prologue_submitted(++pro_val);
    if (r != VK_SUCCESS) {
        vk_device_fault_report(r);
        rt_fatal("vk: vkQueueSubmit2 of value %llu failed (%d)", (unsigned long long)fence, (int)r);
    }
    s->fence = last ? fence : 0;
    awb_submit(c, fence);   // its write-back, on the compute queue (async write-back)
    vk_mem_note_submit(fence);
    plat_mutex_lock(&st_lock);
    __atomic_store_n(&issued, fence, __ATOMIC_RELEASE);
    res_issued = fence;
    plat_cond_broadcast(&cv_work);
    plat_mutex_unlock(&st_lock);
    static unsigned subs;
    if (++subs % 4096 == 0) rec_stats_log();
    __atomic_store_n(&vk_spike_sub_t0, 0, __ATOMIC_RELAXED);
    if (__builtin_expect(sec_on, 0)) {   // the submission's partition, and its ops by class when REKARINA_SPIKE_MS times them
        uint64_t tot = plat_now_ns() - spk_t0;
        uint64_t k = spk_lock + vk_spike.ev + vk_spike.usnap + vk_spike.mirb + vk_spike.mire + vk_spike.gmem + vk_spike.tex + spk_rec + spk_upl + spk_pro + spk_que;
        sec_add_ns(SEC_SUB_TOTAL, tot), sec_add_ns(SEC_SUB_LOCK, spk_lock), sec_add_ns(SEC_SUB_EV, vk_spike.ev);
        sec_add_ns(SEC_SUB_USNAP, vk_spike.usnap), sec_add_ns(SEC_SUB_MIRB, vk_spike.mirb), sec_add_ns(SEC_SUB_MIRE, vk_spike.mire);
        sec_add_ns(SEC_SUB_GMEM, vk_spike.gmem), sec_add_ns(SEC_SUB_TEX, vk_spike.tex), sec_add_ns(SEC_SUB_REC, spk_rec);
        sec_add_ns(SEC_SUB_UPL, spk_upl), sec_add_ns(SEC_SUB_PRO, spk_pro), sec_add_ns(SEC_SUB_QUE, spk_que);
        sec_add_ns(SEC_SUB_REST, tot > k ? tot - k : 0);
        if (cyc0) {   // wall ticks minus the thread's own: the time it did not run
            uint64_t cyc = plat_thread_cycles() - cyc0, tsc = sec_clock() - tsc0;
            if (tsc > cyc) sec_add(SEC_SUB_OFFCPU, tsc - cyc);
        }
        if (vk_spike_on > 0) {
            static int8_t cls[96];
            for (uint32_t i = 0; i < 96; i++) {
                if (!vk_spike.op[i]) continue;
                if (!cls[i]) {
                    const char *nm = nvn_op_name(i);
                    cls[i] = !strcmp(nm, "CALL") ? 8 : !strncmp(nm, "DRAW", 4) ? 1 : !strncmp(nm, "DISPATCH", 8) ? 2 : !strncmp(nm, "CLEAR", 5) ? 3
                           : strstr(nm, "COUNTER") ? 4 : !strcmp(nm, "BARRIER") ? 5 : !strcmp(nm, "BEGIN") || !strcmp(nm, "END") || !strcmp(nm, "SET_RT") ? 6 : 7;
                }
                if (cls[i] < 8) sec_add_ns(SEC_OP_DRAW + cls[i] - 1, vk_spike.op[i]);   // a CALL's time is its nested ops, timed by themselves
            }
        }
    }
    {   // DEBUG-SPIKE
        static int64_t lim = -1;
        if (lim < 0) { const char *e = cfg_get("REKARINA_SPIKE_MS"); lim = e ? atoll(e) * 1000000ll : 0; }
        uint64_t tot = plat_now_ns() - spk_t0;
        if (lim && tot > (uint64_t)lim) {
            uint64_t k = spk_lock + vk_spike.ev + vk_spike.usnap + vk_spike.mirb + vk_spike.mire + vk_spike.gmem + vk_spike.tex + spk_rec + spk_upl + spk_pro + spk_que;
            rt_log("SPIKE frame %llu fence %llu handles %d: %.1f ms: lock %.1f cpu-event %.1f usnap %.1f mirror-begin %.1f mirror-end %.1f gmem %.1f shader/tex %.1f recorder-end %.1f uploads %.1f prologue %.1f queue %.1f rest %.1f",
                   (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), (unsigned long long)fence, spk_handles, tot / 1e6,
                   spk_lock / 1e6, vk_spike.ev / 1e6, vk_spike.usnap / 1e6, vk_spike.mirb / 1e6, vk_spike.mire / 1e6, vk_spike.gmem / 1e6, vk_spike.tex / 1e6,
                   spk_rec / 1e6, spk_upl / 1e6, spk_pro / 1e6, spk_que / 1e6, tot > k ? (tot - k) / 1e6 : 0.0);
            char ops[256] = "";
            size_t o = 0;
            for (int pass = 0; pass < 4; pass++) {
                int best = -1;
                for (int i = 0; i < 96; i++)
                    if (vk_spike.op[i] && (best < 0 || vk_spike.op[i] > vk_spike.op[best])) best = i;
                if (best < 0) break;
                o += (size_t)snprintf(ops + o, sizeof ops - o, " %s %.1f", nvn_op_name((uint32_t)best), vk_spike.op[best] / 1e6);
                vk_spike.op[best] = 0;
            }
            rt_log("SPIKEOPS frame %llu fence %llu:%s; textures made %llu; slowest op %s %.1f ms", (unsigned long long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED),
                   (unsigned long long)fence, ops, (unsigned long long)vk_spike.ntex, nvn_op_name(vk_spike.max_op), vk_spike.max_ns / 1e6);
        }
    }
    plat_mutex_unlock(&sub_lock);
    return fence;
}

static void completion_add(uint64_t fence, void (*fn)(void *), void *arg, int result);
void vk_completion_add(uint64_t fence, void (*fn)(void *), void *arg) { completion_add(fence, fn, arg, 0); }
// A side effect the CPU can see (a counter report, an event word): due with the submission's results when it is queued
// by the recording before the results point, else with the value itself like every other.
void vk_completion_add_result(uint64_t fence, void (*fn)(void *), void *arg) { completion_add(fence, fn, arg, 1); }
static void completion_add(uint64_t fence, void (*fn)(void *), void *arg, int result) {
    plat_mutex_lock(&st_lock);
    // Whatever frees or reuses what the submission's later commands may still name (descriptor pools, arenas, views,
    // buffers), the results recorded after the point, and what other threads queue: due with the value itself.
    if (res_mode > 0 && res_sem && !(result && tl_recording && fence == cur_fence && !cur_res_sent)) {
        if (lqhead + lqn == lqcap) {
            if (lqhead) {
                memmove(lqueue, lqueue + lqhead, sizeof *lqueue * (size_t)lqn);
                lqhead = 0;
            } else {
                int cap = lqcap ? lqcap * 2 : 256;
                Pending *q = realloc(lqueue, sizeof *q * (size_t)cap);
                if (!q) rt_fatal("vk: out of memory (completion queue)");
                lqueue = q, lqcap = cap;
            }
        }
        lqueue[lqhead + lqn++] = (Pending){fence, fn, arg};
        if (fence <= done) plat_cond_broadcast(&cv_work);
        plat_mutex_unlock(&st_lock);
        return;
    }
    if (qhead + qn == qcap) {
        if (qhead) {
            memmove(queue, queue + qhead, sizeof *queue * (size_t)qn);
            qhead = 0;
        } else {
            int cap = qcap ? qcap * 2 : 256;
            Pending *q = realloc(queue, sizeof *q * (size_t)cap);
            if (!q) rt_fatal("vk: out of memory (completion queue)");
            queue = q, qcap = cap;
        }
    }
    queue[qhead + qn++] = (Pending){fence, fn, arg};
    // The completion thread sleeps only while nothing is issued beyond done: a callback for a later value (the replay's,
    // for the submission it records: counter reports, freed resources) becomes due through the submit, which wakes it.
    if (fence <= done) plat_cond_broadcast(&cv_work);
    plat_mutex_unlock(&st_lock);
}

static int fence_wait_impl(uint64_t value, uint64_t timeout_ns);

int vk_sync_fence_wait(uint64_t value, uint64_t timeout_ns) {
    uint64_t p0 = perf_begin();
    FT(FT_FWAIT0, value, timeout_ns, 0);
    int ok = fence_wait_impl(value, timeout_ns);
    FT(FT_FWAIT1, value, ok, 0);
    perf_end(PERF_FENCE, p0);
    perf_gpu_wait(p0);
    return ok;
}

static int fence_wait_impl(uint64_t value, uint64_t timeout_ns) {
    vk_sync_start();
    plat_mutex_lock(&st_lock);
    int ok;
    if (value > issued && value > reserved) {
        ok = 0;   // never submitted nor queued: never reached (as in the null back end)
    } else {
        uint64_t t0 = kern_now_ns();
        while (done < value) {
            if (timeout_ns < (1ull << 62)) {
                uint64_t el = kern_now_ns() - t0;
                if (el >= timeout_ns) break;
                plat_cond_wait_ns(&cv_done, &st_lock, timeout_ns - el);
            } else {
                plat_cond_wait(&cv_done, &st_lock);
            }
        }
        ok = done >= value;
    }
    plat_mutex_unlock(&st_lock);
    return ok;
}

uint64_t vk_sync_fence_done(void) { return __atomic_load_n(&done, __ATOMIC_ACQUIRE); }

// The game's view of a value (the present, nvnSyncWait): done when its results are in guest memory.
uint64_t vk_sync_results_done(void) {
    return vk_sync_results_on() ? __atomic_load_n(&res_done, __ATOMIC_ACQUIRE) : vk_sync_fence_done();
}
int vk_sync_results_wait(uint64_t value, uint64_t timeout_ns) {
    if (!vk_sync_results_on()) return vk_sync_fence_wait(value, timeout_ns);
    uint64_t p0 = perf_begin();
    FT(FT_FWAIT0, value, timeout_ns, 1);
    plat_mutex_lock(&st_lock);
    int ok;
    if (value > issued && value > reserved && value > res_issued) ok = 0;
    else {
        uint64_t t0 = kern_now_ns();
        while (res_done < value) {
            if (timeout_ns < (1ull << 62)) {
                uint64_t el = kern_now_ns() - t0;
                if (el >= timeout_ns) break;
                plat_cond_wait_ns(&cv_done, &st_lock, timeout_ns - el);
            } else plat_cond_wait(&cv_done, &st_lock);
        }
        ok = res_done >= value;
    }
    plat_mutex_unlock(&st_lock);
    FT(FT_FWAIT1, value, ok, 1);
    perf_end(PERF_FENCE, p0);
    perf_gpu_wait(p0);
    return ok;
}
uint64_t vk_sync_fence_issued(void) { return __atomic_load_n(&issued, __ATOMIC_ACQUIRE); }

uint64_t vk_sync_issued(void) { return __atomic_load_n(&issued, __ATOMIC_ACQUIRE); }
// The newest value handed out: submitted, or reserved by the asynchronous replay for a submission it records later.
uint64_t vk_sync_reserved(void) {
    uint64_t i = __atomic_load_n(&issued, __ATOMIC_ACQUIRE), r = __atomic_load_n(&reserved, __ATOMIC_ACQUIRE);
    return r > i ? r : i;
}

void vk_sync_finish(void) {
    vk_sync_start();
    vk_sync_fence_wait(__atomic_load_n(&issued, __ATOMIC_ACQUIRE), UINT64_MAX);
    vk_upload_flush(1);
}

void vk_sync_shutdown(void) {
    plat_mutex_lock(&start_lock);
    if (started) {
        vk_sync_finish();
        plat_mutex_lock(&st_lock);
        stop = 1;
        plat_cond_broadcast(&cv_work);
        plat_mutex_unlock(&st_lock);
        plat_thread_join(completion_thread);
        completion_thread = NULL;
        plat_mutex_lock(&st_lock);
        qhead = qn = 0;   // callbacks of values never submitted
        plat_mutex_unlock(&st_lock);
        plat_mutex_lock(&sub_lock);
        VkCtx *c = vk_ctx();
        for (int i = 0; c && i < nslots; i++) vkDestroyCommandPool(c->device, slots[i].pool, NULL);
        perf_gpu_stats = NULL;
        if (c && ts_pool) vkDestroyQueryPool(c->device, ts_pool, NULL);
        ts_pool = VK_NULL_HANDLE;
        ts_state = 0;
        ts_last_end = 0;
        memset((void *)ts_busy, 0, sizeof ts_busy);
        free(slots);
        slots = NULL;
        nslots = 0;
        plat_mutex_unlock(&sub_lock);
        __atomic_store_n(&started, 0, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&start_lock);
}
