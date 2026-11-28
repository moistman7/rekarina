// Vulkan surface and swapchain, present blit, 60 Hz vsync thread, focus: the window-system-free half of the window
// (the other half: gfx_window_win.c with Win32, gfx_window_sdl.c with SDL, through gfx_wsys.h).
// Swapchain set-up ported from LAS Recomp, runtime/nvn/vulkan/device.c:254-330 (https://github.com/peiikai/LASRecomp @
// da35b83a9611d046a9a681aae668d7085954d37f, MIT, Copyright (c) 2026 Peiikai). Focus and vsync behaviour from
// MacNativeGames, games/totk/runtime/frontend.m:88-97, 175-176, 350-352 (https://github.com/ly05010419/MacNativeGames @
// b75be8cfa704df3701e86130c8ced268e79d6b11, MIT, Copyright (c) 2026 MacNativeGames contributors).
//
// Threads: the window thread (the back end's) owns the window and never runs guest code. The vsync thread signals the
// bound system events. Presents come from any host thread (the back end's present worker) and take g_pl; every queue
// submission takes the recursive queue lock (vk_queue_lock). The swapchain is rebuilt lazily: a resize only marks it
// dirty (gfx_window_mark_dirty), the next present (or gfx_window_resize) rebuilds it.
//
// The in-game menu (ui/menu.h) normally draws over the present blit. OoT's checked output draw can instead sample
// a private picture with the menu blended in before encoding. Window size and fullscreen settings are applied at start
// and when they change (window_settings_live -> gfx_window_apply_settings).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "flight.h"
#include "fs.h"
#include "ftrace.h"
#include "gfx.h"
#include "gfx_wsys.h"
#include "nvn_api.h"
#include "pacelog.h"   // DEBUG-PACELOG, DEBUG-PRESENTTIMING
#include "prep.h"
#include "platform/platform.h"
#include "ui/menu.h"
#include "vk.h"
#include "vk_dlaa.h"
#include "vk_exec.h"
#include "vk_fg.h"
#include "vk_dlssg.h"
#include "vk_reflex.h"
#include "vk_present_format.h"
#include "renderer_mcp.h"

#define FRAMES 2
#define MAX_IMAGES 16
#define VSYNC_NS 1000000000ull
#define VSYNC_HZ 60

// ---- queue lock
static PlatRMutex g_ql;
// A thread that takes the lock first lets the submitter thread hand over what the replay queued (vk_sync.c): its own
// call then comes after those batches in queue order. Not when it already holds the lock (the submitter needs it).
static _Thread_local int g_ql_depth;
void vk_queue_lock(void) {
    if (!g_ql_depth) vk_sync_submits_drain();
    plat_rmutex_lock(&g_ql);
    g_ql_depth++;
}
void vk_queue_unlock(void) {
    g_ql_depth--;
    plat_rmutex_unlock(&g_ql);
}

// ---- driver turns (REKARINA_DRIVER_TURNS, on by default; 0 = the old free-for-all)
// NVIDIA's driver makes kernel calls under one object lock for every image it binds into memory or destroys, and its
// queue submission and present take that lock too; its lock lets a thread that just left take it again at once. While
// the hero turns, the game re-sizes and re-places its transient render targets every frame (~15 new images a frame):
// the thread that makes images ahead (vk_texture.c pm_main) and the reaper (vk_memory.c) went from one such call to
// the next, and the submitter thread's vkQueueSubmit2 waited behind the whole run (profile dumps of the turn's slow
// frames, 2026-11-18: the submitter blocked inside the driver while the reaper and the image-ahead thread were in
// it), so the GPU got the frame late. Those two threads now let queue work go first: before each object call they
// wait while a queue call (a submission or a present) is under way, at most DRV_TURN_MAX_NS. A driver without such a
// lock loses nothing: a queue call takes microseconds.
// Only the queue call itself counts, not the whole hold of the queue lock: the first version counted every thread
// that held or waited for the lock, and a holder that waited for something the image-ahead thread was to make kept
// that thread at 20 ms a call, which froze the picture for up to 3.6 s (2026-11-18).
#define DRV_TURN_MAX_NS 3000000ull
static volatile int32_t g_drv_urgent;
void vk_drv_urgent(int d) {
    if (d > 0) plat_inc32(&g_drv_urgent);
    else plat_dec32(&g_drv_urgent);
}
int vk_drv_turns_on(void) {
    static int on = -1;
    if (on < 0) on = rt_knob("DRIVER_TURNS", "1");
    return on;
}
// What the turns cost the two threads, logged every 10 s while they wait: waits, their time, the longest, how many
// reached the bound and the most bound hits in one frame (a frame whose object work kept hitting the bound would mean
// the turns stall the thread the frame waits for).
static struct { volatile int64_t n[2], ns[2], max_ns[2], bound[2], frame_bound, frame_max, frame_at, log_ns; } DT;
static PlatMutex dt_lock = PLAT_MUTEX_INIT;
static void dt_note(int who, uint64_t ns, int hit) {
    uint64_t now = plat_now_ns(), f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    plat_mutex_lock(&dt_lock);
    DT.n[who]++, DT.ns[who] += (int64_t)ns;
    if ((int64_t)ns > DT.max_ns[who]) DT.max_ns[who] = (int64_t)ns;
    if (hit) {
        DT.bound[who]++;
        if ((int64_t)f != DT.frame_at) DT.frame_at = (int64_t)f, DT.frame_bound = 0;
        if (++DT.frame_bound > DT.frame_max) DT.frame_max = DT.frame_bound;
    }
    if (!DT.log_ns) DT.log_ns = (int64_t)now;
    if ((int64_t)now - DT.log_ns >= 10000000000ll) {
        rt_log("vk: driver turns, last %.0f s: images ahead waited %lld times (%.1f ms, longest %.2f ms, %lld at the "
               "bound), the reaper %lld times (%.1f ms, longest %.2f ms, %lld at the bound); at most %lld bound hits in "
               "one frame", ((int64_t)now - DT.log_ns) / 1e9, (long long)DT.n[0], DT.ns[0] / 1e6, DT.max_ns[0] / 1e6,
               (long long)DT.bound[0], (long long)DT.n[1], DT.ns[1] / 1e6, DT.max_ns[1] / 1e6, (long long)DT.bound[1],
               (long long)DT.frame_max);
        memset((void *)&DT, 0, sizeof DT);
        DT.log_ns = (int64_t)now, DT.frame_at = (int64_t)f;
    }
    plat_mutex_unlock(&dt_lock);
}
void vk_drv_bg_turn(void) {
    if (!vk_drv_turns_on() || !__atomic_load_n(&g_drv_urgent, __ATOMIC_ACQUIRE)) return;
    uint64_t t0 = plat_now_ns(), t = t0;
    while (__atomic_load_n(&g_drv_urgent, __ATOMIC_ACQUIRE) && (t = plat_now_ns()) - t0 < DRV_TURN_MAX_NS) plat_yield();
    dt_note(0, t - t0, t - t0 >= DRV_TURN_MAX_NS);
}
// The reaper's turn: destruction is the one driver object work nothing waits for, so it also lets the images being
// made ahead go first (the replay waits for those while the hero turns: ~20 a frame), again at most DRV_TURN_MAX_NS
// a call; what it holds back is done when the burst is over.
void vk_drv_reap_turn(void) {
    if (!vk_drv_turns_on()) return;
    uint64_t t0 = 0, t = 0;
    while (__atomic_load_n(&g_drv_urgent, __ATOMIC_ACQUIRE) || vk_texture_premake_pending()) {
        if (!t0) t0 = t = plat_now_ns();
        else if ((t = plat_now_ns()) - t0 >= DRV_TURN_MAX_NS) break;
        plat_yield();
    }
    if (t0) dt_note(1, t - t0, t - t0 >= DRV_TURN_MAX_NS);
}

// ---- window state shared with the back end
static VkSurfaceKHR g_surface;
static volatile int32_t g_dirty;               // 1: the window was resized, the swapchain is stale; 2: one is being made
static volatile int32_t g_restart;             // gfx_window_request_restart: closing the window starts the game again
static volatile int32_t g_focused = 1;
static volatile int32_t g_focus_msgs;
static PlatRMutex g_start_lock;       // gfx_window_start / vsync start
static int g_started;

void gfx_window_mark_dirty(void) { plat_xchg32(&g_dirty, 1); }

// The window gained or lost the focus: the game hears it as nn::oe's focus state and a notification message. The line
// carries the local wall time, the only one in the log: when a GPU hang takes the whole system down, the log ends
// without an error, and this time set against the driver's entry in the Windows event log (nvlddmkm 153, Display 4101)
// tells whether an alt-tab or an overlay came before the hang or followed it.
void gfx_window_focus_changed(int in) {
    plat_xchg32(&g_focused, in);
    oe_set_focus(in);                          // pinned to 1 by REKARINA_ASSUME_FOCUS inside
    oe_push_message(OE_MSG_FOCUS_STATE_CHANGED);
    plat_inc32(&g_focus_msgs);
    struct tm t;
    plat_local_time(&t);
    rt_log("gfx: window %s the focus at %04d-%02d-%02d %02d:%02d:%02d local time", in ? "gained" : "lost",
           t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec);
}

// The window was closed: the process ends with it, as in MNG, once a save Commit still running in the background is
// done. After gfx_window_request_restart the executable then starts again with the same command line and environment
// (plat_restart), and the new process waits until this one is gone (main.c) before it moves the log and opens the
// RomFS and the saves.
void gfx_window_closed(void) {
    int restart = plat_cas32(&g_restart, 0, 0) != 0;
    rt_log("gfx: window closed; %s", restart ? "restarting" : "exiting");
    menu_settings_flush();   // a setting changed just before "Quit game" or the close button
    fs_save_flush();
    vk_pipeline_save();
    prep_report();   // what play still had to build (prep.c)
    prof_write();   // plat_exit runs no atexit handlers
    if (restart) {
        plat_setenv("REKARINA_REEXEC", NULL);   // the new process counts its own address-space retries (memory.c)
        if (!plat_restart()) rt_log("gfx: the restart could not start the game again");
    }
    plat_exit(0);
}

void gfx_window_request_restart(void) {
    plat_xchg32(&g_restart, 1);
    gfx_window_request_close();
}

// REKARINA_FULLSCREEN=1: borderless window over the monitor. REKARINA_WINDOW_SIZE=<w>x<h>: client area of the window (the
// swapchain follows the window; the game image is scaled to it).
int gfx_cfg_fullscreen(void) { return cfg_bool("REKARINA_FULLSCREEN"); }

void gfx_cfg_window_size(int *w, int *h) {
    int x = 0, y = 0;
    if (sscanf(cfg_value("REKARINA_WINDOW_SIZE"), "%dx%d", &x, &y) != 2 || x < 640 || y < 360 || x > 7680 || y > 4320)
        sscanf(cfg_meta("WINDOW_SIZE")->def, "%dx%d", &x, &y);   // unparsable or out of range: the default
    *w = x;
    *h = y;
}

// The refresh rate of the window's monitor (REKARINA_FPS_LIMIT=display), measured by the back end's window thread when the
// window opens, lands on another monitor or the display mode changes; readers take the stored value.
static volatile int32_t g_rate_mhz;   // millihertz, 0 unknown

void gfx_window_set_refresh(double hz) {
    int32_t mhz = (int32_t)(hz * 1000 + 0.5);
    if (plat_xchg32(&g_rate_mhz, mhz) != mhz) rt_log("gfx: monitor refresh rate %.3f Hz", hz);
}

double gfx_window_refresh_hz(void) { return plat_cas32(&g_rate_mhz, 0, 0) / 1000.0; }

// ---- swapchain and present state (g_pl)
static PlatRMutex g_pl;
static VkSwapchainKHR g_sc;
static VkFormat g_sc_fmt;
static int g_sc_hdr;                        // HDR10 (REKARINA_HDR, vk_hdr.c): PQ code values in Rec. 2020
#ifdef GAME_OOT
static VkFormat g_output_format, g_sc_request_format;
static int32_t g_surface_pq;
int gfx_hdr_surface_capable(void) { return plat_cas32(&g_surface_pq, 0, 0); }
void ns2_hdr_capability_changed(void);
#else
static int g_sc_hdr_wanted;
#endif
static VkExtent2D g_sc_ext;
static int g_scaler;                        // the last present: 0 the game image 1:1, 1 scaled by the blit, 2 by FSR
static uint32_t g_sc_n;
static VkImage g_sc_img[MAX_IMAGES];
static VkImageView g_sc_view[MAX_IMAGES];   // colour attachment views for the menu overlay
static VkSemaphore g_render_done[MAX_IMAGES];
static VkCommandPool g_pool;
static VkCommandBuffer g_cb[FRAMES];
static VkFence g_fence[FRAMES];
static VkSemaphore g_acquire[FRAMES];
static unsigned g_frame;
static int g_capture;
static VkBuffer g_cap_buf;
static VmaAllocation g_cap_alloc;
static uint8_t *g_cap_ptr;
static VkDeviceSize g_cap_size;
static int g_cap_w, g_cap_h, g_cap_valid;
static int g_flip_on;            // vk_present_blit submitted the blit of swapchain image g_flip_idx; vk_present_flip presents it
static uint32_t g_flip_idx;
static uint64_t g_flip_reflex_id;   // native Reflex present ID of the pending flip (0 when off)
static uint64_t g_reflex_next_id;   // set by present_main before vk_present_blit

void gfx_present_reflex_frame(uint64_t id) { g_reflex_next_id = id; }

// ---- REKARINA_FLIGHT thumbnails (flight.c; g_pl): each present also blits the swapchain image down into slot frame % g_th_n
// of a host-visible ring. A slot's tag names the frame it holds complete; recording a new frame clears the tag first,
// and the tag is set once the fence of that present has signalled.
#define TH_BYTES ((VkDeviceSize)FLIGHT_THUMB_W * FLIGHT_THUMB_H * 4)
static int g_th_n, g_th_failed;
static uint64_t g_th_frame;                // frame of the next vk_present_image call (vk_thumbs_frame)
static uint64_t *g_th_tag;                 // per ring slot
static uint64_t g_th_pending[FRAMES];      // per fence slot: the frame whose thumbnail its submission writes
static VkImage g_th_img;
static VmaAllocation g_th_img_alloc;
static VkFormat g_th_fmt;
static VkBuffer g_th_buf;
static VmaAllocation g_th_alloc;
static uint8_t *g_th_ptr;

// ---- vsync
static PlatMutex g_vs_lock = PLAT_MUTEX_INIT;
static uint64_t g_vs_ev[16];
static int g_vs_nev;
static volatile int32_t *g_vs_ctr[4];
static int g_vs_nctr;
static volatile int64_t g_ticks;
static volatile int64_t g_vs_start_ns;
static int g_vs_started;

static void vsync_main(void *arg) {
    (void)arg;
    plat_thread_priority(PLAT_PRIO_HIGHEST);
    uint64_t n = 0;
    for (;;) {
        uint64_t start = (uint64_t)g_vs_start_ns;
        uint64_t due = start + (n + 1) * VSYNC_NS / VSYNC_HZ;
        uint64_t now = kern_now_ns();
        if (now > due + 100000000ull) {                  // the process stalled: restart the schedule, no catch-up burst
            g_vs_start_ns = (int64_t)(now - (n + 1) * VSYNC_NS / VSYNC_HZ);
        } else if (due > now) {
            plat_wait_ns(due - now);
        }
        n++;
        plat_xchg64(&g_ticks, (int64_t)n);
        uint64_t ev[16];
        volatile int32_t *ctr[4];
        int ne, nc;
        plat_mutex_lock_shared(&g_vs_lock);
        ne = g_vs_nev;
        nc = g_vs_nctr;
        memcpy(ev, g_vs_ev, sizeof ev);
        memcpy(ctr, g_vs_ctr, sizeof ctr);
        plat_mutex_unlock_shared(&g_vs_lock);
        for (int i = 0; i < nc; i++) plat_inc32(ctr[i]);
        for (int i = 0; i < ne; i++) kern_signal_system_event(ev[i]);
    }
    return;
}

static void vsync_ensure(void) {
    plat_rmutex_lock(&g_start_lock);
    if (!g_vs_started) {
        g_vs_started = 1;
        g_vs_start_ns = (int64_t)kern_now_ns();
        if (!plat_thread_spawn(vsync_main, NULL, "vsync", 0)) rt_fatal("gfx: cannot start the vsync thread");
    }
    plat_rmutex_unlock(&g_start_lock);
}

void gfx_vsync_bind(uint64_t system_event) {
    plat_mutex_lock(&g_vs_lock);
    int have = 0;
    for (int i = 0; i < g_vs_nev; i++) have |= g_vs_ev[i] == system_event;
    if (!have) {
        if (g_vs_nev == 16) rt_fatal("gfx: more than 16 vsync events");
        g_vs_ev[g_vs_nev++] = system_event;
    }
    plat_mutex_unlock(&g_vs_lock);
    vsync_ensure();
}

void gfx_vsync_bind_counter(volatile int32_t *counter) {
    plat_mutex_lock(&g_vs_lock);
    if (g_vs_nctr == 4) rt_fatal("gfx: more than 4 vsync counters");
    g_vs_ctr[g_vs_nctr++] = counter;
    plat_mutex_unlock(&g_vs_lock);
    vsync_ensure();
}

uint64_t gfx_vsync_ticks(void) { return (uint64_t)plat_cas64(&g_ticks, 0, 0); }
uint64_t gfx_vsync_time_ns(uint64_t tick) { return (uint64_t)g_vs_start_ns + tick * VSYNC_NS / VSYNC_HZ; }

// ---- focus
int gfx_window_focused(void) { return g_focused; }
unsigned gfx_window_focus_messages(void) { return (unsigned)g_focus_msgs; }

// Present path (g_pl held): a settings change (cfg_version) reaches the window thread, which applies a changed size or
// fullscreen; a changed REKARINA_VK_PRESENT rebuilds the swapchain at this present.
static void window_settings_live(void) {
    static unsigned seen;
    static int started;
    static char present[16];   // the REKARINA_VK_PRESENT the swapchain was made with
    unsigned v = cfg_version();
    if (started && v == seen) return;
    seen = v;
    res_fov_live();
    const char *e = cfg_value("REKARINA_VK_PRESENT");
    if (!started || strncmp(e, present, sizeof present - 1)) {
        snprintf(present, sizeof present, "%s", e);
        if (started) plat_xchg32(&g_dirty, 1);
    }
#ifdef GAME_OOT
    ns2_hdr_capability_changed();
    static int fg_wanted = -1;
    int want_fg = vk_fg_wanted();
    if (started && want_fg != fg_wanted) plat_xchg32(&g_dirty, 1);
    fg_wanted = want_fg;
#else
    if (started && vk_hdr_wanted() != g_sc_hdr_wanted) plat_xchg32(&g_dirty, 1);
#endif
    vk_dlssg_reflex_sync();
    if (started && vk_reflex_swapchain_stale()) plat_xchg32(&g_dirty, 1);
    if (started) gfx_window_apply_settings();
    started = 1;
}

// ---- swapchain
static const char *g_present_got = "";   // the REKARINA_VK_PRESENT value of the mode the swapchain got; "" before one

static VkPresentModeKHR pick_present_mode(VkPhysicalDevice pd) {
    const char *e = cfg_value("REKARINA_VK_PRESENT");
    VkPresentModeKHR want = !strcmp(e, "mailbox") ? VK_PRESENT_MODE_MAILBOX_KHR
                          : !strcmp(e, "immediate") ? VK_PRESENT_MODE_IMMEDIATE_KHR : VK_PRESENT_MODE_FIFO_KHR;
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, g_surface, &n, NULL);
    VkPresentModeKHR m[16];
    if (n > 16) n = 16;
    vkGetPhysicalDeviceSurfacePresentModesKHR(pd, g_surface, &n, m);
    VkPresentModeKHR got = VK_PRESENT_MODE_FIFO_KHR;
    int relaxed = 0, immediate = 0;
    for (uint32_t i = 0; i < n; i++) {
        if (m[i] == want) got = want;
        relaxed |= m[i] == VK_PRESENT_MODE_FIFO_RELAXED_KHR, immediate |= m[i] == VK_PRESENT_MODE_IMMEDIATE_KHR;
    }
    // Mailbox missing (several AMD drivers): plain FIFO holds a frame that misses a refresh to the next one, and the game
    // drops to half the refresh at every slow frame. Relaxed FIFO shows a late frame at once and waits otherwise.
    if (want == VK_PRESENT_MODE_MAILBOX_KHR && got != want)
        got = relaxed ? VK_PRESENT_MODE_FIFO_RELAXED_KHR : immediate ? VK_PRESENT_MODE_IMMEDIATE_KHR : got;
    const char *name = got == VK_PRESENT_MODE_MAILBOX_KHR ? "mailbox" : got == VK_PRESENT_MODE_IMMEDIATE_KHR ? "immediate" : "fifo";
    __atomic_store_n(&g_present_got, name, __ATOMIC_RELEASE);
    static int logged = -1;   // the mode last logged: a change from the menu shows in the log
    if ((int)got != logged) {
        logged = (int)got;
        rt_log("gfx: present mode %s (REKARINA_VK_PRESENT=%s)", name, e);
    }
    return got;
}

const char *gfx_window_present_mode(void) { return __atomic_load_n(&g_present_got, __ATOMIC_ACQUIRE); }

// Queues the present of a blit vk_present_blit left waiting (g_pl held). A swapchain rebuild and the next blit flush it
// first; the device release drops it.
// Only the Windows window paints before the first frame (gfx_window_win.c overrides this).
__attribute__((weak)) void gfx_window_presented(void) {}

// ---- DEBUG-PRESENTTIMING (REKARINA_PRESENT_TIMING=1; vk_device.c enables VK_EXT_present_timing): the presentation
// engine's own time at which each present's first pixel became visible, into the pacing log (PL_VIS), and once the
// swap chain's refresh duration and interval (a fixed refresh shows a duration; a variable one says so). Only the plain
// swap chain (not frame generation's).
static int pt_on;                 // the swap chain was made with the timing flag and its queue is sized
static uint64_t pt_domain_id;     // the time domain asked for
static int pt_qpc;                // ... the performance counter (convertible to plat_now_ns); else times are not kept
static uint64_t pt_next_frame;    // the frame the next flip presents (present_main)
static PlatScale pt_scale;
void gfx_present_frame(uint64_t frame) { pt_next_frame = frame; }

static void pt_setup(VkDevice d, VkSwapchainKHR sc) {
    PFN_vkSetSwapchainPresentTimingQueueSizeEXT qs = (PFN_vkSetSwapchainPresentTimingQueueSizeEXT)vkGetDeviceProcAddr(d, "vkSetSwapchainPresentTimingQueueSizeEXT");
    PFN_vkGetSwapchainTimeDomainPropertiesEXT td = (PFN_vkGetSwapchainTimeDomainPropertiesEXT)vkGetDeviceProcAddr(d, "vkGetSwapchainTimeDomainPropertiesEXT");
    PFN_vkGetSwapchainTimingPropertiesEXT tp = (PFN_vkGetSwapchainTimingPropertiesEXT)vkGetDeviceProcAddr(d, "vkGetSwapchainTimingPropertiesEXT");
    pt_on = 0;
    if (!qs || !td || qs(d, sc, 64) != VK_SUCCESS) { rt_log("gfx: present timing: no timing queue"); return; }
    VkTimeDomainKHR doms[16];
    uint64_t ids[16];
    VkSwapchainTimeDomainPropertiesEXT p = {.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_TIME_DOMAIN_PROPERTIES_EXT, .timeDomainCount = 16,
                                            .pTimeDomains = doms, .pTimeDomainIds = ids};
    uint64_t counter = 0;
    if (td(d, sc, &p, &counter) != VK_SUCCESS && !p.timeDomainCount) { rt_log("gfx: present timing: no time domains"); return; }
    char list[160] = "";
    size_t n = 0;
    pt_qpc = 0, pt_domain_id = p.timeDomainCount ? ids[0] : 0;
    for (uint32_t i = 0; i < p.timeDomainCount && i < 16; i++) {
        n += (size_t)snprintf(list + n, sizeof list - n, "%s%d", i ? "," : "", (int)doms[i]);
        if (doms[i] == VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR) pt_qpc = 1, pt_domain_id = ids[i];
    }
    pt_scale = plat_scale(1000000000u, plat_counter_hz());
    VkSwapchainTimingPropertiesEXT t = {.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_TIMING_PROPERTIES_EXT};
    uint64_t tc = 0;
    VkResult tr = tp ? tp(d, sc, &t, &tc) : VK_ERROR_FEATURE_NOT_PRESENT;
    rt_log("gfx: present timing on: time domains %s (%s); refresh duration %.4f ms, interval %.4f ms (%d)", list,
           pt_qpc ? "the performance counter is used" : "the engine's own clock: the log keeps its times as they are, for intervals", t.refreshDuration / 1e6,
           t.refreshInterval / 1e6, (int)tr);
    pt_on = 1;
}

// The timings the engine has completed since the last call, into the pacing log.
static void pt_collect(VkDevice d) {
    static PFN_vkGetPastPresentationTimingEXT get;
    if (!get) get = (PFN_vkGetPastPresentationTimingEXT)vkGetDeviceProcAddr(d, "vkGetPastPresentationTimingEXT");
    if (!get) return;
    VkPresentStageTimeEXT st[16];
    VkPastPresentationTimingEXT pt[16];
    for (int i = 0; i < 16; i++)
        pt[i] = (VkPastPresentationTimingEXT){.sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT, .presentStageCount = 1,
                                              .pPresentStages = &st[i]};
    VkPastPresentationTimingInfoEXT in = {.sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_INFO_EXT, .swapchain = g_sc};
    VkPastPresentationTimingPropertiesEXT out = {.sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_PROPERTIES_EXT,
                                                  .presentationTimingCount = 16, .pPresentationTimings = pt};
    vk_reflex_wsi_lock();
    VkResult r = get(d, &in, &out);
    vk_reflex_wsi_unlock();
    if (r < 0) return;
    for (uint32_t i = 0; i < out.presentationTimingCount && i < 16; i++)
        if (pt[i].reportComplete && pt[i].presentStageCount && st[i].time)
            pacelog_mark(PL_VIS, pt[i].presentId, pt_qpc ? plat_scaled(pt_scale, st[i].time) : st[i].time);
}

static void flip_locked(void) {
    if (!g_flip_on) return;
    g_flip_on = 0;
    uint64_t flip_reflex = g_flip_reflex_id;
    g_flip_reflex_id = 0;
    if (!g_sc) return;
    VkPresentInfoKHR pi = {.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1,
                           .pWaitSemaphores = &g_render_done[g_flip_idx], .swapchainCount = 1, .pSwapchains = &g_sc,
                           .pImageIndices = &g_flip_idx};
    uint64_t pt_id = pt_next_frame;   // DEBUG-PRESENTTIMING
    VkPresentId2KHR pid = {.sType = VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR, .swapchainCount = 1, .pPresentIds = &pt_id};
    VkPresentTimingInfoEXT ti = {.sType = VK_STRUCTURE_TYPE_PRESENT_TIMING_INFO_EXT, .timeDomainId = pt_domain_id,
                                 .presentStageQueries = VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_VISIBLE_BIT_EXT};
    VkPresentTimingsInfoEXT tis = {.sType = VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT, .pNext = &pid, .swapchainCount = 1,
                                   .pTimingInfos = &ti};
    int fg_sc = vk_fg_swapchain(), dlssg_sc = !fg_sc && vk_dlssg_swapchain_ready();
    int timed = pt_on && pt_id && !dlssg_sc && !fg_sc;
    if (timed) pi.pNext = &tis;
    VkPresentIdKHR rpid = {0};
    uint64_t rpid_val = 0;
    if (!dlssg_sc && !fg_sc && !timed)
        pi.pNext = vk_reflex_present_begin(pi.pNext, &rpid, &rpid_val, flip_reflex);
    FT(FT_NOTE, 1, 0, 0);
    vk_queue_lock();
    vk_drv_urgent(1);
    VkResult r;
    if (dlssg_sc) r = vk_dlssg_present(vk_ctx()->gfx_queue, &pi);
    else if (fg_sc) r = vk_fg_present(&pi);
    else {
        vk_reflex_wsi_lock();
        r = vkQueuePresentKHR(vk_ctx()->gfx_queue, &pi);
        vk_reflex_wsi_unlock();
    }
    vk_drv_urgent(-1);
    vk_queue_unlock();
    if (!dlssg_sc && !fg_sc) vk_reflex_present_end(rpid_val);
    renderer_mcp_present(g_sc_ext.width, g_sc_ext.height, g_sc_hdr, dlssg_sc ? 1 : fg_sc ? 2 : 0, r);
    FT(FT_NOTE, 2, 0, 0);
    vk_device_fault_report(r);
    if (r == VK_SUCCESS) gfx_window_presented();
    if (timed && r >= 0) pt_collect(vk_ctx()->device);   // DEBUG-PRESENTTIMING
    if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_SUBOPTIMAL_KHR) plat_xchg32(&g_dirty, 1);
}

static VkResult acquire_locked(VkDevice device, VkSemaphore ready, uint32_t *index) {
    const uint64_t timeout = 1000000000ull;
    if (vk_fg_swapchain()) return vk_fg_acquire(g_sc, timeout, ready, index);
    if (vk_dlssg_swapchain_ready()) return vk_dlssg_acquire(device, g_sc, timeout, ready, index);
    vk_reflex_wsi_lock();
    VkResult r = vkAcquireNextImageKHR(device, g_sc, timeout, ready, VK_NULL_HANDLE, index);
    vk_reflex_wsi_unlock();
    return r;
}

static void free_capture(void) {
    VkCtx *c = vk_ctx();
    if (g_cap_buf && c) vmaDestroyBuffer(c->vma, g_cap_buf, g_cap_alloc);
    g_cap_buf = VK_NULL_HANDLE;
    g_cap_ptr = NULL;
    g_cap_size = 0;
    g_cap_valid = 0;
}

static void destroy_replaced_swapchain(VkDevice device, VkSwapchainKHR old) {
    for (uint32_t i = 0; i < g_sc_n; i++) {
        vkDestroySemaphore(device, g_render_done[i], NULL);
        if (g_sc_view[i]) vkDestroyImageView(device, g_sc_view[i], NULL);
        g_sc_view[i] = VK_NULL_HANDLE;
    }
    g_sc_n = 0;
    if (old && vk_dlssg_swapchain_ready()) vk_dlssg_destroy_swapchain(device, old);
    else if (old) vkDestroySwapchainKHR(device, old, NULL);
}

// Returns 1 on replacement. Preflight failures leave the prior chain usable; a failed Vulkan creation may retire it.
// Called with g_pl held.
static int create_swapchain(void) {
    flip_locked();   // the acquired image goes out before its swapchain is replaced
    VkCtx *c = vk_ctx();
    // From here a resize (the window thread, gfx_window_mark_dirty) writes 1 over the 2, and the end clears the mark
    // only where it is still 2: the fullscreen switch at start fell between the size read below and the end, the mark
    // was cleared with it, and the swapchain stayed at the window's first size under a window of the monitor's
    // (2560x1421 stretched to 2560x1440).
    plat_xchg32(&g_dirty, 2);
    VkSurfaceCapabilitiesKHR caps;
    if (vkGetPhysicalDeviceSurfaceCapabilitiesKHR(c->phys, g_surface, &caps) != VK_SUCCESS) return 0;
    int cw = 0, ch = 0;
    wsys_client_size(&cw, &ch);
    VkExtent2D ext = caps.currentExtent.width != UINT32_MAX ? caps.currentExtent
                                                            : (VkExtent2D){(uint32_t)cw, (uint32_t)ch};
    {   // DEBUG-SCEXTENT: REKARINA_DEBUG_SC_EXTENT=<w>x<h> makes the swapchain that size whatever the window's, the case
        // of a swapchain not made again after the window changed (the menu's pointer scaling is checked with it). Outside
        // the Vulkan rules on Windows; NVIDIA stretches the image to the window.
        unsigned fw, fh;
        const char *f = cfg_get("REKARINA_DEBUG_SC_EXTENT");
        if (f && sscanf(f, "%ux%u", &fw, &fh) == 2 && fw >= 64 && fh >= 64 && fw <= 8192 && fh <= 8192 && ext.width)
            ext = (VkExtent2D){fw, fh};
    }
    if (!ext.width || !ext.height) return 0;
    uint32_t nf = 0;
    if (vkGetPhysicalDeviceSurfaceFormatsKHR(c->phys, g_surface, &nf, NULL) != VK_SUCCESS || !nf) return 0;
    VkSurfaceFormatKHR fmts[64];
    if (nf > 64) nf = 64;
    VkResult formats_result = vkGetPhysicalDeviceSurfaceFormatsKHR(c->phys, g_surface, &nf, fmts);
    if ((formats_result != VK_SUCCESS && formats_result != VK_INCOMPLETE) || !nf) return 0;
    // UNORM preserves the output shader's already encoded values.
    VkSurfaceFormatKHR sf = fmts[0];
#ifdef GAME_OOT
    VkSurfaceFormatKHR pq;
    int capable = vk_present_surface_format(fmts, nf, VK_FORMAT_UNDEFINED, VK_COLOR_SPACE_HDR10_ST2084_EXT, &pq);
    int previous_capability = plat_xchg32(&g_surface_pq, capable);
    if ((rt_trace & RT_TRACE_GFX) && previous_capability != capable)
        rt_log("gfx: HDR10 surface capability: %s", capable ? "available" : "unavailable");
    ns2_hdr_capability_changed();
    // OoT 1.1.1 changes the window from RGBA8 in SDR to RGB10A2 in HDR10.
    int hdr = g_output_format == VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    VkColorSpaceKHR encoding = hdr ? VK_COLOR_SPACE_HDR10_ST2084_EXT : VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    if (!vk_present_surface_format(fmts, nf, g_output_format, encoding, &sf)) return 0;
#else
    int hdr = vk_hdr_surface_format(fmts, nf, &sf), found = hdr;
    for (uint32_t i = 0; i < nf && !found; i++)
        if (fmts[i].format == VK_FORMAT_B8G8R8A8_UNORM && fmts[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { sf = fmts[i]; found = 1; }
    for (uint32_t i = 0; i < nf && !found; i++)
        if (fmts[i].format == VK_FORMAT_R8G8B8A8_UNORM && fmts[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { sf = fmts[i]; found = 1; }
#endif
    uint32_t count = caps.minImageCount + 1;
    if (caps.maxImageCount && count > caps.maxImageCount) count = caps.maxImageCount;
    if (count > 8) count = 8;

    const int had_fg = vk_fg_swapchain();
    int want_fg = vk_fg_wanted() && !vk_fg_failed();
#ifdef GAME_OOT
    if (g_output_format == VK_FORMAT_UNDEFINED) want_fg = 0;
    if (want_fg && !vk_present_amd_fg_compatible(sf)) {
        rt_log("gfx: AMD frame generation suspended: the selected %s swapchain format %d is unsupported",
               hdr ? "HDR10" : "SDR", (int)sf.format);
        want_fg = 0;
    } else if (want_fg && !had_fg && !vk_ffx_load()) {
        want_fg = 0;
    }
#endif
    const int replace_fg = had_fg && want_fg && (g_sc_fmt != sf.format || g_sc_hdr != hdr ||
                                                 g_sc_ext.width != ext.width || g_sc_ext.height != ext.height);
    // The SDK presenter can still have CPU work queued even when the Vulkan device is idle.
    // Drain before taking the queue lock: pending SDK work can call submit_game.
    const int drain_fg = had_fg && (replace_fg || !want_fg);
    static unsigned failed_fg_drain_attempts;
    if (drain_fg && !vk_fg_wait_for_presents()) {
        if (vk_fg_failed() && ++failed_fg_drain_attempts >= 3)
            rt_fatal("gfx: could not drain the failed AMD swapchain after three attempts; cannot switch safely");
        plat_xchg32(&g_dirty, 1);
        return 0;
    }
    failed_fg_drain_attempts = 0;

    vk_queue_lock();
    vkDeviceWaitIdle(c->device);
    vk_reflex_swapchain_destroyed();
    VkSwapchainKHR old = g_sc;
    VkSwapchainCreateInfoKHR ci = {
        .sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR, .surface = g_surface, .minImageCount = count,
        .imageFormat = sf.format, .imageColorSpace = sf.colorSpace, .imageExtent = ext, .imageArrayLayers = 1,
        .imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                      VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,   // every surface supports it (the menu overlay draws into it)
        .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE, .preTransform = caps.currentTransform,
        .compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR, .presentMode = pick_present_mode(c->phys), .clipped = VK_TRUE,
        .oldSwapchain = old};
    // AMD can remake the same output in place, but a new output format, encoding or size needs a fresh context.
    VkResult r = VK_ERROR_FEATURE_NOT_PRESENT;
    if (had_fg || want_fg) {
        // Drop the current owner's views and destroy a non-AMD swapchain through its own API.
        destroy_replaced_swapchain(c->device, had_fg ? VK_NULL_HANDLE : old);
        ci.oldSwapchain = VK_NULL_HANDLE;
        old = g_sc = VK_NULL_HANDLE;
        if (replace_fg) {
            rt_log("gfx: AMD frame generation output changed: replacing %s format %d with %s format %d",
                   g_sc_hdr ? "HDR10" : "SDR", (int)g_sc_fmt, hdr ? "HDR10" : "SDR", (int)sf.format);
            vk_fg_release();
        }
        if (want_fg) r = vk_fg_create_swapchain(&ci, &g_sc);
        // A callback can stop AMD after preflight. Leave an undrained chain owned by AMD and retry outside the lock.
        if (r != VK_SUCCESS && vk_fg_swapchain() && !drain_fg) {
            static volatile int32_t n_fail;
            if (plat_inc32(&n_fail) <= 5) rt_log("gfx: frame generation swapchain %ux%u failed (%d); trying again", ext.width, ext.height, (int)r);
            g_sc = VK_NULL_HANDLE;
            vk_queue_unlock();
            plat_xchg32(&g_dirty, 1);
            return 0;
        }
        if (r != VK_SUCCESS && vk_fg_swapchain()) {   // frame generation stopped: its swapchain goes, a plain one follows
            rt_log("gfx: AMD frame generation %s: replacing its swapchain", want_fg ? "stopped" : "disabled");
            vk_fg_release();
            g_sc = VK_NULL_HANDLE;
        }
    }
    int reflex_lat = 0;
    if (r != VK_SUCCESS && vk_dlssg_swapchain_ready()) r = vk_dlssg_create_swapchain(c->device, &ci, &g_sc);   // Streamline's hook
    else if (r != VK_SUCCESS) {
        // DEBUG-PRESENTTIMING: the plain swap chain asks for present timing; without it when the driver refuses
        int want_pt = c->has_present_timing;
        if (want_pt) ci.flags |= VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT | VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR;
        // a window, not the display: full-screen exclusive disallowed where the driver has the choice
        // (gfx_window_win.c gfx_window_swapchain_next)
        ci.pNext = gfx_window_swapchain_next(c->has_fse);
#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
        VkSwapchainLatencyCreateInfoNV reflex_ci = {0};
        reflex_lat = vk_reflex_swapchain_next(&ci, &reflex_ci);
#endif
        r = vkCreateSwapchainKHR(c->device, &ci, NULL, &g_sc);
        if (r != VK_SUCCESS && old) {
            // The failed call retired old; destroy it before retrying without an old swapchain.
            destroy_replaced_swapchain(c->device, old);
            old = ci.oldSwapchain = g_sc = VK_NULL_HANDLE;
        }
        if (r != VK_SUCCESS && want_pt) {
            rt_log("gfx: swap chain with present timing refused (%d): without it", (int)r);
            ci.flags &= ~(VkSwapchainCreateFlagsKHR)(VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT | VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR);
            want_pt = 0;
            r = vkCreateSwapchainKHR(c->device, &ci, NULL, &g_sc);
        }
        if (r != VK_SUCCESS && ci.pNext) {
            rt_log("gfx: swap chain with extensions refused (%d): without pNext", (int)r);
            ci.pNext = NULL;
            reflex_lat = 0;
            r = vkCreateSwapchainKHR(c->device, &ci, NULL, &g_sc);
        }
        if (r == VK_SUCCESS && want_pt) pt_setup(c->device, g_sc);
        else pt_on = 0;
    }
    destroy_replaced_swapchain(c->device, old);
    if (r != VK_SUCCESS) {
        rt_log("gfx: vkCreateSwapchainKHR failed (%d)", (int)r);
        g_sc = VK_NULL_HANDLE;
#ifdef GAME_OOT
        g_sc_hdr = 0;
        vk_hdr_set_active(0);
#endif
        vk_queue_unlock();
        plat_xchg32(&g_dirty, 1);
        return 0;
    }
    g_sc_fmt = sf.format;
    g_sc_hdr = hdr;
#ifdef GAME_OOT
    g_sc_request_format = g_output_format;
    rt_log("gfx: guest swapchain %s (format %d, requested format %d)", hdr ? "PQ" : "SDR", (int)sf.format, (int)g_output_format);
#else
    g_sc_hdr_wanted = vk_hdr_wanted();
#endif
    vk_hdr_set_active(hdr);
    if (hdr) vk_hdr_metadata(g_sc, 1);
    else vk_hdr_images_release();
    {   // the first one is in gfx_window_start's line; a remade one (a resize, a mode change) says its size here
        static int made;
        if (made++ && (ext.width != g_sc_ext.width || ext.height != g_sc_ext.height))
            rt_log("gfx: swap chain made again: %ux%u (at least %u images)", ext.width, ext.height, count);
    }
    g_sc_ext = ext;
    g_sc_n = MAX_IMAGES;
    if (vk_fg_swapchain()) {   // AMD's swapchain: the count first (a larger capacity is not cut down to it)
        uint32_t n = 0;
        vk_fg_get_images(g_sc, &n, NULL);
        g_sc_n = n < MAX_IMAGES ? n : MAX_IMAGES;
        memset(g_sc_img, 0, sizeof g_sc_img);
        vk_fg_get_images(g_sc, &g_sc_n, g_sc_img);
    } else if (vk_dlssg_swapchain_ready()) {   // Streamline's swapchain too: the count first, then only that many images
        uint32_t n = 0;
        vk_dlssg_get_images(c->device, g_sc, &n, NULL);
        g_sc_n = n < MAX_IMAGES ? n : MAX_IMAGES;
        memset(g_sc_img, 0, sizeof g_sc_img);
        if (vk_dlssg_get_images(c->device, g_sc, &g_sc_n, g_sc_img) < 0) g_sc_n = 0;
        for (uint32_t i = 0; i < g_sc_n; i++)
            if (!g_sc_img[i]) g_sc_n = i;   // never a view of a null image
    } else {
        vkGetSwapchainImagesKHR(c->device, g_sc, &g_sc_n, g_sc_img);
    }
    VkSemaphoreCreateInfo semi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (uint32_t i = 0; i < g_sc_n; i++) {
        vkCreateSemaphore(c->device, &semi, NULL, &g_render_done[i]);
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = g_sc_img[i],
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = sf.format,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(c->device, &vi, NULL, &g_sc_view[i]) != VK_SUCCESS) g_sc_view[i] = VK_NULL_HANDLE;
    }
    free_capture();
    if (!vk_fg_swapchain() && !vk_dlssg_swapchain_ready()) vk_reflex_swapchain_created(g_sc, reflex_lat);
    vk_queue_unlock();
    plat_cas32(&g_dirty, 0, 2);
    return 1;
}

static void sync_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags2 s0,
                         VkAccessFlags2 a0, VkPipelineStageFlags2 s1, VkAccessFlags2 a1) {
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = s0, .srcAccessMask = a0,
                               .dstStageMask = s1, .dstAccessMask = a1, .oldLayout = from, .newLayout = to,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &d);
}

static int ensure_capture_buffer(void) {
    VkCtx *c = vk_ctx();
    VkDeviceSize need = (VkDeviceSize)g_sc_ext.width * g_sc_ext.height * 4;
    if (g_cap_buf && g_cap_size == need) return 1;
    free_capture();
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = need, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &g_cap_buf, &g_cap_alloc, &info) != VK_SUCCESS) { g_cap_buf = VK_NULL_HANDLE; return 0; }
    g_cap_ptr = info.pMappedData;
    g_cap_size = need;
    return g_cap_ptr != NULL;
}

void vk_thumbs_enable(int n) {
    plat_rmutex_lock(&g_pl);
    if (!g_th_n && n > 0 && (g_th_tag = calloc((size_t)n, sizeof *g_th_tag))) g_th_n = n;
    plat_rmutex_unlock(&g_pl);
}

void vk_thumbs_frame(uint64_t frame) {
    plat_rmutex_lock(&g_pl);
    g_th_frame = frame;
    plat_rmutex_unlock(&g_pl);
}

static void free_thumbs(void) {
    VkCtx *c = vk_ctx();
    if (g_th_img && c) vmaDestroyImage(c->vma, g_th_img, g_th_img_alloc);
    if (g_th_buf && c) vmaDestroyBuffer(c->vma, g_th_buf, g_th_alloc);
    g_th_img = VK_NULL_HANDLE, g_th_buf = VK_NULL_HANDLE, g_th_ptr = NULL;
    if (g_th_tag) memset(g_th_tag, 0, sizeof *g_th_tag * (size_t)g_th_n);
    memset(g_th_pending, 0, sizeof g_th_pending);
}

// The ring and the blit target in the swapchain's format; allocated on the first present, the image again when the
// swapchain format changes. A failure turns the thumbnails off for the session.
static int ensure_thumbs(void) {
    if (!g_th_n || g_th_failed || !g_th_frame) return 0;
    VkCtx *c = vk_ctx();
    if (!g_th_buf) {
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = TH_BYTES * (VkDeviceSize)g_th_n,
                                 .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(c->vma, &bi, &ai, &g_th_buf, &g_th_alloc, &info) != VK_SUCCESS || !info.pMappedData) {
            if (g_th_buf) vmaDestroyBuffer(c->vma, g_th_buf, g_th_alloc);
            g_th_buf = VK_NULL_HANDLE;
            g_th_failed = 1;
            rt_log("gfx: flight thumbnails off: no host-visible ring of %llu MB", (unsigned long long)(bi.size >> 20));
            return 0;
        }
        g_th_ptr = info.pMappedData;
    }
    // an HDR10 swapchain blits into RGBA8: its PQ codes, flat-looking but readable as bytes
    VkFormat tf = g_sc_hdr ? VK_FORMAT_R8G8B8A8_UNORM : g_sc_fmt;
    if (g_th_img && g_th_fmt != tf) {
        vmaDestroyImage(c->vma, g_th_img, g_th_img_alloc);   // the previous present's fence was waited for: idle
        g_th_img = VK_NULL_HANDLE;
    }
    if (!g_th_img) {
        VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = tf,
                                .extent = {FLIGHT_THUMB_W, FLIGHT_THUMB_H, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ci, &ai, &g_th_img, &g_th_img_alloc, NULL) != VK_SUCCESS) {
            g_th_img = VK_NULL_HANDLE;
            g_th_failed = 1;
            rt_log("gfx: flight thumbnails off: no %ux%u image", FLIGHT_THUMB_W, FLIGHT_THUMB_H);
            return 0;
        }
        g_th_fmt = tf;
    }
    return 1;
}

// dst (TRANSFER_SRC, the swapchain image just written) -> thumbnail image -> ring slot of g_th_frame.
static void record_thumb(VkCommandBuffer cb, VkImage dst, unsigned fence_slot) {
    uint64_t f = g_th_frame;
    unsigned ring = (unsigned)(f % (uint64_t)g_th_n);
    g_th_tag[ring] = 0;   // under g_pl: vk_thumbs_read no longer hands this slot out
    sync_barrier(cb, g_th_img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 0, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkImageBlit region = {
        .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .srcOffsets = {{0, 0, 0}, {(int)g_sc_ext.width, (int)g_sc_ext.height, 1}},
        .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{0, 0, 0}, {FLIGHT_THUMB_W, FLIGHT_THUMB_H, 1}}};
    vkCmdBlitImage(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_th_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region,
                   VK_FILTER_LINEAR);
    sync_barrier(cb, g_th_img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                 VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy bc = {.bufferOffset = TH_BYTES * ring, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                            .imageExtent = {FLIGHT_THUMB_W, FLIGHT_THUMB_H, 1}};
    vkCmdCopyImageToBuffer(cb, g_th_img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_th_buf, 1, &bc);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                           .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo d = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &d);
    g_th_pending[fence_slot] = f;
    g_th_frame = 0;
}

int vk_thumbs_read(uint64_t frame, uint8_t *rgba) {
    int ok = 0;
    plat_rmutex_lock(&g_pl);
    if (g_th_n && g_th_ptr && frame) {
        unsigned ring = (unsigned)(frame % (uint64_t)g_th_n);
        if (g_th_tag[ring] == frame) {
            VkCtx *c = vk_ctx();
            vmaInvalidateAllocation(c->vma, g_th_alloc, TH_BYTES * ring, TH_BYTES);
            memcpy(rgba, g_th_ptr + TH_BYTES * ring, TH_BYTES);
            ok = g_th_fmt == VK_FORMAT_B8G8R8A8_UNORM || g_th_fmt == VK_FORMAT_B8G8R8A8_SRGB ? 2 : 1;
        }
    }
    plat_rmutex_unlock(&g_pl);
    if (ok == 2)
        for (size_t i = 0; i < TH_BYTES; i += 4) {
            uint8_t t = rgba[i];
            rgba[i] = rgba[i + 2], rgba[i + 2] = t;
        }
    return ok != 0;
}

void vk_present_capture(int on) {
    plat_rmutex_lock(&g_pl);
    g_capture = on;
    if (!on) g_cap_valid = 0;
    plat_rmutex_unlock(&g_pl);
}

int vk_present_captured(const uint8_t **px, int *w, int *h, VkFormat *fmt) {
    plat_rmutex_lock(&g_pl);
    int ok = g_cap_valid;
    if (ok) { *px = g_cap_ptr; *w = g_cap_w; *h = g_cap_h; *fmt = g_sc_fmt; }
    plat_rmutex_unlock(&g_pl);
    return ok;
}

// ---- REKARINA_VIDEO=<file.mp4>: a video of what the window shows, for the performance clips of the README.
// Thirty times a second (REKARINA_VIDEO_FPS) of real time, from REKARINA_VIDEO_AFTER seconds after the first present on, the
// swapchain image is copied to a buffer of its present slot and every second pixel of every second row goes to an
// ffmpeg started here, which encodes it on the GPU's video encoder (h264_nvenc; REKARINA_VIDEO_ENCODER names another). The
// window's own picture, overlay included, and nothing else on the screen. Nothing waits for the GPU: a slot's copy is
// read when the slot comes round again, FRAMES presents later, after the wait every present makes for its slot's
// earlier work. ffmpeg has to be on PATH.
static struct {
    int state;            // 0 not looked at, 1 waiting or recording, 2 off
    FILE *pipe;
    uint64_t next, step, start;
    uint8_t *px;
    int w, h;
    unsigned long long frames;
    VkBuffer buf[FRAMES];
    VmaAllocation alloc[FRAMES];
    uint8_t *ptr[FRAMES];
    VkDeviceSize size;
    int pending[FRAMES], pw[FRAMES], ph[FRAMES];   // a slot's buffer holds a frame of that size, not yet written out
} g_video;

static int video_buffers(void) {
    VkCtx *c = vk_ctx();
    VkDeviceSize need = (VkDeviceSize)g_sc_ext.width * g_sc_ext.height * 4;
    if (g_video.buf[0] && g_video.size == need) return 1;
    for (int i = 0; i < FRAMES; i++) {
        if (g_video.buf[i]) vmaDestroyBuffer(c->vma, g_video.buf[i], g_video.alloc[i]);
        g_video.buf[i] = VK_NULL_HANDLE, g_video.pending[i] = 0;
        VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = need, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                      .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
        VmaAllocationInfo info;
        if (vmaCreateBuffer(c->vma, &bi, &ai, &g_video.buf[i], &g_video.alloc[i], &info) != VK_SUCCESS || !info.pMappedData) {
            g_video.buf[i] = VK_NULL_HANDLE;
            g_video.state = 2;
            return 0;
        }
        g_video.ptr[i] = info.pMappedData;
    }
    g_video.size = need;
    return 1;
}

static int video_due(void) {
    if (g_video.state == 2) return 0;
    uint64_t now = plat_now_ns();
    if (!g_video.state) {
        const char *path = cfg_get("REKARINA_VIDEO"), *after = cfg_get("REKARINA_VIDEO_AFTER"), *fps = cfg_get("REKARINA_VIDEO_FPS");
        if (!path || !*path) { g_video.state = 2; return 0; }
        double hz = fps && atof(fps) >= 1 ? atof(fps) : 30;
        g_video.step = (uint64_t)(1e9 / hz);
        g_video.start = now + (uint64_t)((after ? atof(after) : 0) * 1e9);
        g_video.next = g_video.start;
        g_video.state = 1;
    }
    if (now < g_video.next) return 0;
    g_video.next += g_video.step;
    if (g_video.next + 2 * g_video.step < now) g_video.next = now + g_video.step;   // far behind: no burst to catch up
    return 1;
}

// REKARINA_VIDEO_FULL=1: every pixel instead of every second one of every second row, at REKARINA_VIDEO_BITRATE
// (80M then, 14M otherwise). A full 2560x1440 frame is 14.7 MB, and ffmpeg takes it from the pipe only as fast as it
// encodes, so the presenting thread does not write it: it copies the frame into one of VF_RING buffers and a writer
// thread sends them on in order. A frame that finds the ring full is left out and counted.
#define VF_RING 4
static struct {
    int on;                       // -1 not looked at
    uint8_t *buf[VF_RING];
    volatile uint32_t head, tail; // presenting thread writes head, the writer thread tail
    unsigned long long dropped;
} g_vfull = {.on = -1};

static void video_writer(void *arg) {
    (void)arg;
    size_t n = (size_t)g_video.w * g_video.h;
    for (;;) {
        uint32_t t = __atomic_load_n(&g_vfull.tail, __ATOMIC_RELAXED);
        if (t == __atomic_load_n(&g_vfull.head, __ATOMIC_ACQUIRE)) {
            if (g_video.state == 2) return;
            plat_sleep_ns(500000);
            continue;
        }
        if (fwrite(g_vfull.buf[t % VF_RING], 4, n, g_video.pipe) != n) {
            rt_log("gfx: video: ffmpeg stopped taking frames after %llu", g_video.frames);
            g_video.state = 2;
            return;
        }
        g_video.frames++;
        __atomic_store_n(&g_vfull.tail, t + 1, __ATOMIC_RELEASE);
    }
}

// The frame in a slot's buffer (cap_w x cap_h pixels of the swapchain's format) to ffmpeg, at half its size (or whole:
// REKARINA_VIDEO_FULL).
static void video_frame(const uint8_t *frame, int cap_w, int cap_h) {
    if (g_vfull.on < 0) { const char *e = cfg_get("REKARINA_VIDEO_FULL"); g_vfull.on = e && *e == '1'; }
    int full = g_vfull.on;
    int w = full ? cap_w & ~1 : (cap_w / 2) & ~1, h = full ? cap_h & ~1 : (cap_h / 2) & ~1;
    if (w < 2 || h < 2 || g_sc_hdr) { g_video.state = 2; return; }
    if (!g_video.pipe) {
        int bgra = g_sc_fmt == VK_FORMAT_B8G8R8A8_UNORM || g_sc_fmt == VK_FORMAT_B8G8R8A8_SRGB;
        const char *enc = cfg_get("REKARINA_VIDEO_ENCODER");
        char cmd[1400];
        snprintf(cmd, sizeof cmd,
                 "ffmpeg -hide_banner -loglevel error -y -f rawvideo -pix_fmt %s -s %dx%d -r %.3f -i - -c:v %s -b:v %s "
                 "-pix_fmt yuv420p \"%s\"",
                 bgra ? "bgra" : "rgba", w, h, 1e9 / (double)g_video.step, enc && *enc ? enc : "h264_nvenc -preset p5",
                 cfg_get("REKARINA_VIDEO_BITRATE") && *cfg_get("REKARINA_VIDEO_BITRATE") ? cfg_get("REKARINA_VIDEO_BITRATE")
                                                                                         : full ? "80M" : "14M",
                 cfg_get("REKARINA_VIDEO"));
#ifdef _WIN32
        g_video.pipe = _popen(cmd, "wb");
#else
        g_video.pipe = popen(cmd, "w");
#endif
        g_video.px = malloc((size_t)w * h * 4);
        g_video.w = w, g_video.h = h;
        if (!g_video.pipe || !g_video.px) {
            rt_log("gfx: video: ffmpeg could not be started (%s)", cmd);
            g_video.state = 2;
            return;
        }
        rt_log("gfx: video: recording %dx%d at %.0f fps to %s", w, h, 1e9 / (double)g_video.step, cfg_get("REKARINA_VIDEO"));
        if (full) {
            for (int i = 0; i < VF_RING; i++) g_vfull.buf[i] = malloc((size_t)w * h * 4);
            if (!g_vfull.buf[VF_RING - 1] || !plat_thread_spawn(video_writer, NULL, "video writer", 0)) {
                rt_log("gfx: video: no memory or thread for whole frames");
                g_video.state = 2;
                return;
            }
        }
    }
    if (w != g_video.w || h != g_video.h) return;   // the window changed size: those frames are left out
    if (full) {
        uint32_t hd = __atomic_load_n(&g_vfull.head, __ATOMIC_RELAXED);
        if (hd - __atomic_load_n(&g_vfull.tail, __ATOMIC_ACQUIRE) >= VF_RING) {
            if (!(g_vfull.dropped++ & 63)) rt_log("gfx: video: %llu frames left out (the encoder is behind)", g_vfull.dropped);
            return;
        }
        uint8_t *d = g_vfull.buf[hd % VF_RING];
        if (w == cap_w) memcpy(d, frame, (size_t)w * h * 4);
        else for (int y = 0; y < h; y++) memcpy(d + (size_t)y * w * 4, frame + (size_t)y * cap_w * 4, (size_t)w * 4);
        __atomic_store_n(&g_vfull.head, hd + 1, __ATOMIC_RELEASE);
        return;
    }
    const uint32_t *src = (const uint32_t *)frame;
    uint32_t *dst = (uint32_t *)g_video.px;
    for (int y = 0; y < h; y++) {
        const uint32_t *row = src + (size_t)(y * 2) * cap_w;
        for (int x = 0; x < w; x++) *dst++ = row[x * 2];
    }
    if (fwrite(g_video.px, 4, (size_t)w * h, g_video.pipe) != (size_t)w * h) {
        rt_log("gfx: video: ffmpeg stopped taking frames after %llu", g_video.frames);
        g_video.state = 2;
        return;
    }
    g_video.frames++;
}

// The screenshot key (input.c screenshot_key, REKARINA_SCREENSHOT_KEY): the next present saves its image as
// screenshots/rekarina-<date>-<time>.png beside the game, without the menu and the overlays unless
// REKARINA_SCREENSHOT_OVERLAY=1. Any thread.
static volatile int32_t g_shot_req;
void gfx_screenshot_request(void) { plat_xchg32(&g_shot_req, 1); }

static const char *screenshot_path(void) {
    static char path[PLAT_PATH_MAX * 3 + 64];
    char dir[PLAT_PATH_MAX * 3];
    if (!plat_exe_path(dir, sizeof dir)) return NULL;
    char *cut = plat_last_sep(dir), *f = strrchr(dir, '/');
    if (f > cut) cut = f;
    if (!cut) return NULL;
    strcpy(cut, "/screenshots");
    plat_file_mkdir_p(dir);
    struct tm t;
    plat_local_time(&t);
    static int same;   // two in one second get a number
    static char last[32];
    char stamp[32];
    snprintf(stamp, sizeof stamp, "%04d%02d%02d-%02d%02d%02d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday, t.tm_hour,
             t.tm_min, t.tm_sec);
    same = strcmp(stamp, last) ? 0 : same + 1;
    snprintf(last, sizeof last, "%s", stamp);
    if (same) snprintf(path, sizeof path, "%s/" RT_FILE_STEM "-%s-%d.png", dir, stamp, same);
    else snprintf(path, sizeof path, "%s/" RT_FILE_STEM "-%s.png", dir, stamp);
    return path;
}

// A captured image of the swapchain's format and size as a PNG, RGBA with opaque alpha.
static void save_pixels(const char *path, const uint8_t *cap) {
    size_t n = (size_t)g_cap_w * (size_t)g_cap_h * 4;
    uint8_t *px = malloc(n);
    if (!px) return;
    int bgra = g_sc_fmt == VK_FORMAT_B8G8R8A8_UNORM || g_sc_fmt == VK_FORMAT_B8G8R8A8_SRGB;
    if (g_sc_hdr) vk_hdr_to_srgb8(cap, px, n / 4, g_sc_fmt == VK_FORMAT_A2R10G10B10_UNORM_PACK32);
    else for (size_t i = 0; i < n; i += 4) {
        px[i] = cap[i + (bgra ? 2 : 0)];
        px[i + 1] = cap[i + 1];
        px[i + 2] = cap[i + (bgra ? 0 : 2)];
        px[i + 3] = 255;
    }
    rt_write_png(path, px, (uint32_t)g_cap_w, (uint32_t)g_cap_h);
    free(px);
}

// The captured swapchain image as a PNG (the menu's test hook).
static void save_capture(const char *path) {
    save_pixels(path, g_cap_ptr);
    rt_log("gfx: menu test shot %s (%dx%d)", path, g_cap_w, g_cap_h);
}

// DEBUG-HLSHOT: frame generation's HUD-less picture of a present, copied beside the swapchain image's capture
static VkBuffer g_hlcap_buf;
static VmaAllocation g_hlcap_alloc;
static uint8_t *g_hlcap_ptr;
static VkDeviceSize g_hlcap_size;
static int hlcap_buffer(void) {
    VkCtx *c = vk_ctx();
    VkDeviceSize need = (VkDeviceSize)g_sc_ext.width * g_sc_ext.height * 4;
    if (g_hlcap_buf && g_hlcap_size == need) return 1;
    if (g_hlcap_buf) vmaDestroyBuffer(c->vma, g_hlcap_buf, g_hlcap_alloc), g_hlcap_buf = VK_NULL_HANDLE;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = need, .usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                                  .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    if (vmaCreateBuffer(c->vma, &bi, &ai, &g_hlcap_buf, &g_hlcap_alloc, &info) != VK_SUCCESS) return g_hlcap_buf = VK_NULL_HANDLE, 0;
    g_hlcap_ptr = info.pMappedData, g_hlcap_size = need;
    return g_hlcap_ptr != NULL;
}

// The rectangle {x, y, w, h} of a dw x dh swapchain image the cw x ch picture goes into: the whole image with
// REKARINA_ASPECT=stretch, else (fit, the default) the largest one of the picture's own aspect ratio, centred (a 16:9 picture
// on the Steam Deck's 1280x800 screen gets bars of 40 rows above and below; a 16:9 picture in a 21:9 window, bars at the
// sides).
// Where the last present put the picture, as fractions of the swapchain (x, y, w, h): menu_cursor.c's pointer
static float g_pic[4];
static volatile int32_t g_pic_ok;
int gfx_picture_rect(float out[4]) {
    if (!__atomic_load_n(&g_pic_ok, __ATOMIC_ACQUIRE)) return 0;
    memcpy(out, g_pic, sizeof g_pic);
    return 1;
}

void gfx_fit_rect(int cw, int ch, int dw, int dh, int out[4]) {
    out[0] = out[1] = 0, out[2] = dw, out[3] = dh;
    if (cw <= 0 || ch <= 0 || !strcmp(cfg_value("REKARINA_ASPECT"), "stretch")) return;
    if ((int64_t)dw * ch > (int64_t)dh * cw) out[2] = (int)(((int64_t)dh * cw + ch / 2) / ch);   // window wider: pillarbox
    else out[3] = (int)(((int64_t)dw * ch + cw / 2) / cw);                                        // taller: letterbox
    // A side of the window's parity keeps the two bars equal; a picture one pixel short of the window fills it
    out[2] += (dw - out[2]) & 1, out[3] += (dh - out[3]) & 1;
    out[0] = (dw - out[2]) / 2, out[1] = (dh - out[3]) / 2;
}

// The display's frame rate with frame generation: the frames it got, generated ones included, over half a second,
// sampled at each of the game's presents (the performance overlay shows it beside the game's own rate).
static volatile double g_out_fps;
static const char *g_out_fg_method;
static void output_rate(void) {
    static uint64_t t0, n0;
    const int dlss = vk_dlssg_active();
    const char *method = dlss ? "DLSS FG" : vk_fg_active() ? vk_fg_method() : NULL;
    if (method != g_out_fg_method) {
        g_out_fps = 0;
        t0 = 0;
        g_out_fg_method = method;
    }
    uint64_t n = dlss ? vk_dlssg_shown() : vk_fg_presents(g_sc), t = plat_now_ns();
    if (!n) { g_out_fps = 0, t0 = 0; return; }
    if (!t0 || n < n0) { g_out_fps = 0; t0 = t, n0 = n; return; }
    if (t - t0 >= 500000000ull) g_out_fps = (double)(n - n0) * 1e9 / (double)(t - t0), t0 = t, n0 = n;
}
double gfx_output_fps(void) { return g_out_fps; }
// The overlay reads the label under g_pl, the same lock held by output_rate.
const char *gfx_output_fg_method(void) { return g_out_fg_method; }

// Frame generation's picture without the HUD, at the swapchain's size and format: generation tells the HUD apart with
// it and keeps it out of the generated frames. VK_NULL_HANDLE when this frame has none.
static VkImage g_hl;
static VmaAllocation g_hl_mem;
static VkImageView g_hl_view;
static uint32_t g_hl_w, g_hl_h;
static VkFormat g_hl_fmt;

// The game's picture into a swapchain-sized image dst (in TRANSFER_DST_OPTIMAL) at to, as the screen shows it: the colour
// adjustment (REKARINA_COLOR_*, vk_color.c), or on an HDR10 swapchain the PQ encode (vk_hdr.c), which grades with the
// same values, takes a copy of the crop; then FSR (REKARINA_UPSCALER, vk_fsr.c: EASU when the picture's rectangle is
// larger than the crop, RCAS alone after DLSS) or the bilinear blit. general: src is in GENERAL (vk_texture.c images),
// else in TRANSFER_SRC_OPTIMAL; full: the whole path (a GENERAL source only), else the blit alone. slot picks the
// source views those passes keep (one per frame in flight; frame generation's HUD-less picture takes the slots after
// FRAMES). *psrc and prect: what the picture was read from at the end (the menu's live picture reads it again).
// pq_hudless encodes a linear checkpoint, then fits it with the same blit as the guest's PQ picture.
// Returns -1 if that required encode failed, otherwise vk_fsr_record's answer or 0 for a blit.
static int picture_into(VkCommandBuffer cb, unsigned slot, VkImage src, VkFormat fmt, int w, int h, const int rect[4],
                        int general, int full, const VkGuestPq *pq, VkImage dst, const int to[4], VkImage *psrc, int prect[4]) {
    int cw = rect[2], ch = rect[3];
    VkImage graded = pq || (full && g_sc_hdr) ? vk_hdr_encode(cb, slot, src, fmt, rect, pq) : VK_NULL_HANDLE;
#ifndef GAME_OOT
    if (!pq && full && !g_sc_hdr) graded = vk_color_record(cb, slot, src, fmt, rect);
#endif
    if (pq && !graded) return -1;
    *psrc = graded ? graded : src;
    VkFormat pfmt = graded ? VK_FORMAT_R16G16B16A16_SFLOAT : fmt;
    for (int i = 0; i < 4; i++) prect[i] = graded && i < 2 ? 0 : rect[i];   // the graded copy is the crop alone
    int fsr = full ? vk_fsr_record(cb, slot, *psrc, pfmt, graded ? cw : w, graded ? ch : h, prect, dst, to) : 0;
    if (!fsr) {
        if (general)   // earlier submissions' writes become visible to the blit here
            sync_barrier(cb, *psrc, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkImageBlit region = {
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffsets = {{prect[0], prect[1], 0}, {prect[0] + cw, prect[1] + ch, 1}},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .dstOffsets = {{to[0], to[1], 0}, {to[0] + to[2], to[1] + to[3], 1}}};
        vkCmdBlitImage(cb, *psrc, general ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region, VK_FILTER_LINEAR);
        if (general)   // later submissions that write the source wait for the blit's read
            sync_barrier(cb, *psrc, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0);
    }
    return fsr;
}

// The HUD-less picture goes through the same path as the game's picture (picture_into): frame generation takes as
// interface whatever differs between the two, and the present's sharpening after DLSS (RCAS at the sharpness setting,
// 80 by default), the colour profile or FSR's scaling applied to one of them alone differed over the whole scene.
// full: the game's picture took the whole path (it was in GENERAL). REKARINA_FG_HUDLESS_PLAIN=1: the bilinear blit alone,
// as before (test).
static VkImage hudless_record(uint64_t frame, VkCommandBuffer cb, unsigned slot, int w, int h, const int rect[4],
                              const int to[4], int full, const VkGuestPq *transform) {
    VkImage pic;
    VkImageView pv;
    uint32_t pw, ph;
    VkFormat pf;
    int pq;
    if ((!vk_fg_active() && !vk_dlssg_active()) || !vk_dlaa_fg_hudless(frame, &pic, &pv, &pw, &ph, &pf, &pq) || (int)pw != w || (int)ph != h)
        return VK_NULL_HANDLE;
#ifdef GAME_OOT
    if (pq != g_sc_hdr || (g_sc_hdr && (!transform || !transform->valid))) return VK_NULL_HANDLE;
#else
    (void)transform;
#endif
    VkCtx *c = vk_ctx();
    if (g_hl && (g_hl_w != g_sc_ext.width || g_hl_h != g_sc_ext.height || g_hl_fmt != g_sc_fmt)) {
        vk_queue_lock(), vkDeviceWaitIdle(c->device), vk_queue_unlock();
        vkDestroyImageView(c->device, g_hl_view, NULL);
        vmaDestroyImage(c->vma, g_hl, g_hl_mem);
        g_hl = VK_NULL_HANDLE;
    }
    if (!g_hl) {
        VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = g_sc_fmt,
                                .extent = {g_sc_ext.width, g_sc_ext.height, 1}, .mipLevels = 1, .arrayLayers = 1,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                                         VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT};
        VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
        if (vmaCreateImage(c->vma, &ii, &ai, &g_hl, &g_hl_mem, NULL) != VK_SUCCESS) return g_hl = VK_NULL_HANDLE;
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = g_hl,
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = g_sc_fmt,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        if (vkCreateImageView(c->device, &vi, NULL, &g_hl_view) != VK_SUCCESS) g_hl_view = VK_NULL_HANDLE;
        g_hl_w = g_sc_ext.width, g_hl_h = g_sc_ext.height, g_hl_fmt = g_sc_fmt;
        rt_log("gfx: frame generation's HUD-less picture at %ux%u", g_hl_w, g_hl_h);
    }
    sync_barrier(cb, g_hl, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    VkClearColorValue black = {.float32 = {0, 0, 0, 1}};
    VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cb, g_hl, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &all);
    sync_barrier(cb, g_hl, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_MEMORY_WRITE_BIT);
    static int plain = -1;
    if (plain < 0) plain = rt_knob("FG_HUDLESS_PLAIN", "0");
    VkImage ps;
    int pr[4];
    const VkGuestPq *pq_hudless = NULL;
#ifdef GAME_OOT
    if (g_sc_hdr) pq_hudless = transform;
#endif
    if (picture_into(cb, slot + FRAMES, pic, pf, w, h, rect, 1, full && !plain && !pq_hudless,
                     pq_hudless, g_hl, to, &ps, pr) < 0) return VK_NULL_HANDLE;
    sync_barrier(cb, g_hl, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                 VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    return g_hl;
}

// ---- the present blit's GPU end, a timestamp query per blit (a ring of BT_N), read by the completion thread's
// callback for the frame (vk_backend.c frame_blit_done) and converted to the host clock with a calibrated (GPU tick,
// host counter) pair taken there every 250 ms: the frame clock's "shown" moment (perf.c) and the pacing log's PL_GTS.
// The callback itself runs late at times (it waits behind the value's other callbacks and whatever they wait for: up
// to 27 ms in a forest run, 2026-11-17), and the time it ran made frames the GPU had finished on time count as long.
// Off with REKARINA_FRAME_CLOCK=call or =seen (the callback's time, as before).
#define BT_N 64
static VkQueryPool bt_pool;
static int bt_state;                       // 0 not set up, 1 on, -1 off
static uint64_t bt_frame_next, bt_frame[BT_N];
static double bt_period;
static uint64_t bt_mask;
static PFN_vkGetCalibratedTimestampsKHR bt_cal_fn;
static int64_t bt_cal_tick, bt_cal_kns, bt_cal_at;

void gfx_blit_frame(uint64_t frame) { bt_frame_next = frame; }

static int bt_ready(void) {
    if (bt_state) return bt_state > 0;
    bt_state = -1;
    VkCtx *c = vk_ctx();
    const char *fc = cfg_get("REKARINA_FRAME_CLOCK");
    if ((fc && (!strcmp(fc, "call") || !strcmp(fc, "seen"))) || !c->has_calibrated || c->props.limits.timestampPeriod <= 0)
        return 0;
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    vkGetPhysicalDeviceFeatures2(c->phys, &f2);
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, NULL);
    VkQueueFamilyProperties qp[16];
    if (nq > 16) nq = 16;
    vkGetPhysicalDeviceQueueFamilyProperties(c->phys, &nq, qp);
    uint32_t bits = c->gfx_family < nq ? qp[c->gfx_family].timestampValidBits : 0;
    if (!f12.hostQueryReset || !bits) return 0;
    bt_cal_fn = (PFN_vkGetCalibratedTimestampsKHR)vkGetDeviceProcAddr(
        c->device, c->has_calibrated == 2 ? "vkGetCalibratedTimestampsKHR" : "vkGetCalibratedTimestampsEXT");
    VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                .queryCount = BT_N};
    if (!bt_cal_fn || vkCreateQueryPool(c->device, &qi, NULL, &bt_pool) != VK_SUCCESS) return 0;
    vkResetQueryPool(c->device, bt_pool, 0, BT_N);
    bt_period = c->props.limits.timestampPeriod;
    bt_mask = bits >= 64 ? ~0ull : (1ull << bits) - 1;
    bt_state = 1;
    rt_log("frametime: a frame counts as shown at its present blit's GPU timestamp (REKARINA_FRAME_CLOCK=seen: when the "
           "completion thread saw it)");
    return 1;
}

int perf_frame_clock_wanted(void);   // perf.c
static void bt_write(VkCommandBuffer cb) {   // at the blit's end, under g_pl (the present thread)
    // only while the overlay, the frame-time file or the pacing log reads the clock (perf.c perf_frame_clock_wanted)
    if (!bt_frame_next || !perf_frame_clock_wanted() || !bt_ready()) return;
    uint32_t q = (uint32_t)(bt_frame_next % BT_N);
    bt_frame[q] = bt_frame_next;
    // reset on the GPU, in the blit's own command buffer (also a slot whose frame was never read: a skipped present): a
    // host reset here and another after each read were two driver calls a frame on the present and completion threads,
    // and the paired turn runs with this clock were 2-4 % slower by the present calls (2026-11-18)
    vkCmdResetQueryPool(cb, bt_pool, q, 1);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, bt_pool, q);
    if (__builtin_expect(pacelog_on(), 0)) pacelog_mark(PL_BLIT, bt_frame_next, kern_now_ns());   // DEBUG-PACELOG
    bt_frame_next = 0;
}

uint64_t gfx_blit_gpu_ns(uint64_t frame) {   // the completion thread
    if (bt_state <= 0) return 0;
    uint32_t q = (uint32_t)(frame % BT_N);
    if (bt_frame[q] != frame) return 0;
    VkDevice dev = vk_ctx()->device;
    uint64_t v[2] = {0, 0};
    VkResult r = vkGetQueryPoolResults(dev, bt_pool, q, 1, sizeof v, v, sizeof v,
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
    bt_frame[q] = 0;   // (the slot is reset by the command buffer that writes it next: bt_write)
    if (r != VK_SUCCESS || !v[1]) return 0;
    int64_t now = (int64_t)kern_now_ns();
    if (!bt_cal_at || now - bt_cal_at >= 250000000) {
        VkCalibratedTimestampInfoKHR ci[2] = {
            {.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, .timeDomain = VK_TIME_DOMAIN_DEVICE_KHR},
#ifdef _WIN32
            {.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, .timeDomain = VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR}};
#else
            {.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, .timeDomain = VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR}};
#endif
        uint64_t cv[2], dv;
        int64_t k0 = (int64_t)kern_now_ns(), p0 = (int64_t)plat_now_ns();
        if (bt_cal_fn(dev, 2, ci, cv, &dv) == VK_SUCCESS) {
            bt_cal_tick = (int64_t)(cv[0] & bt_mask);
            bt_cal_kns = (int64_t)((double)cv[1] * 1e9 / (double)plat_counter_hz()) + (k0 - p0);
            bt_cal_at = now;
        }
    }
    if (!bt_cal_at) return 0;
    int64_t t = bt_cal_kns + (int64_t)((double)((int64_t)(v[0] & bt_mask) - bt_cal_tick) * bt_period);
    return t > 0 && t <= now ? (uint64_t)t : 0;
}

#ifdef GAME_OOT
static struct {
    uint64_t frame;
    VkImage destination;
    const char *shot;
    VkImageView alpha;
    int composed;
    int preview[4];
} guest_menu;
static int32_t guest_menu_wanted;

void vk_present_guest_menu(VkExec *ex, const VkTexImage *source, VkImage destination) {
    int pq = ex->output_encoding == GFX_OUTPUT_PQ;
    if ((!menu_is_open() && !plat_cas32(&guest_menu_wanted, 0, 0)) || (pq && !ex->output_pq.valid) ||
        source->type != VK_IMAGE_TYPE_2D || source->layers != 1 || source->aspect != VK_IMAGE_ASPECT_COLOR_BIT ||
        !(source->usage & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) ||
        (source->format != VK_FORMAT_B10G11R11_UFLOAT_PACK32 && source->format != VK_FORMAT_R16G16B16A16_SFLOAT))
        return;
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    vk_present_drain();
    plat_rmutex_lock(&g_pl);
    if (!g_started || guest_menu.frame == frame) goto out;
    vk_texture_lock();
    VkImageView compositor = vk_exec_target_view(source, source->format, VK_IMAGE_VIEW_TYPE_2D,
                                                 VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1, 0);
    vk_texture_unlock();
    if (!compositor) goto out;
    VkCtx *c = vk_ctx();
    const uint32_t w = source->width, h = source->height;
    MenuTarget target = {c->instance, c->phys, c->device, c->gfx_family, c->gfx_queue,
                         VK_FORMAT_R8G8B8A8_UNORM, w, h, (int)w, (int)h, g_scaler, g_sc_ext.width, g_sc_ext.height,
                         vk_dlaa_nvidia_scene_aa()};
    guest_menu = (typeof(guest_menu)){.frame = frame, .destination = destination};
    const CfgRenderState *settings = cfg_render_use(NULL);
    int overlay = menu_overlay_frame(frame, &target, &guest_menu.shot);
    cfg_render_use(settings);
    plat_xchg32(&guest_menu_wanted, overlay);
    if (!overlay) goto out;
    if (plat_cas32(&g_shot_req, 0, 0) && !cfg_bool("REKARINA_SCREENSHOT_OVERLAY")) goto out;
    vk_exec_barrier(ex);
    VkImageView menu = vk_hdr_ui_target(ex->cb, w, h);
    if (menu) {
        VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = menu,
            .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR,
            .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
        VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {w, h}},
                              .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca};
        vkCmdBeginRendering(ex->cb, &ri);
        menu_overlay_draw(ex->cb, 0);
        vkCmdEndRendering(ex->cb);
        if (!menu_preview_rect(guest_menu.preview)) memset(guest_menu.preview, 0, sizeof guest_menu.preview);
        int *p = guest_menu.preview;
        if (p[0] < 0 || p[1] < 0 || p[2] <= 0 || p[3] <= 0 || p[0] + p[2] > (int)w || p[1] + p[3] > (int)h)
            memset(p, 0, sizeof guest_menu.preview);
        guest_menu.composed = vk_hdr_ui_guest_compose(ex->cb, source, compositor, p, pq ? &ex->output_pq : NULL);
        if (guest_menu.composed && menu_overlay_can_replay()) guest_menu.alpha = menu;
        // The host pipelines set guest-dynamic state statically on the same command buffer.
        vk_pipeline_dynamic_forget();
    }
    ex->recorded++;
    vk_exec_draw_new_cb();
    ex->dirty |= VK_EXEC_DIRTY_ALL;
    vk_exec_pass_begin(ex);
    if (guest_menu.composed) {
        static int logged[2];
        if (!logged[pq]++) rt_log("gfx: menu blended into the game compositor before the checked %s output", pq ? "PQ" : "SDR");
    }
out:
    plat_rmutex_unlock(&g_pl);
}
#endif

static int present_image(VkImage src, VkImageLayout layout, VkFormat fmt, int w, int h, const int crop[4], int defer,
                         const VkGuestPq *pq, const char *nvidia_scene_aa) {
#ifndef GAME_OOT
    (void)nvidia_scene_aa;
#endif
    int tagged = 0;
    int general = layout == VK_IMAGE_LAYOUT_GENERAL;
    {   // DEBUG-FOCUS: REKARINA_TEST_FOCUS=<frame>,<frame>: the window "loses" the focus at the first frame and gains it at
        // the second (gfx_window_focus_changed), as an alt-tab would, for a script to repeat
        static long lose = -2, gain;
        if (lose == -2) {
            const char *e = cfg_get("REKARINA_TEST_FOCUS"), *c = e ? strchr(e, ',') : NULL;
            lose = e && *e ? atol(e) : -1, gain = c ? atol(c + 1) : -1;
        }
        long f = (long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (lose >= 0 && f == lose) gfx_window_focus_changed(0);
        if (gain >= 0 && f == gain) gfx_window_focus_changed(1);
    }
    {   // DEBUG-RESIZE: REKARINA_TEST_RESIZE=<frame>:<w>x<h>[,<frame>:<w>x<h>...]: at those presented frames the window's
        // drawable area becomes w x h, as a drag of its border or the fullscreen switch at start would make it, and the
        // next present makes the swapchain again (frame generation's swapchain is made again in place)
        static const char *spec = (const char *)-1;
        if (spec == (const char *)-1) spec = cfg_get("REKARINA_TEST_RESIZE");
        long f = (long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        for (const char *p = spec; p && *p;) {
            long at = atol(p);
            const char *c = strchr(p, ':');
            int rw = 0, rh = 0;
            if (c && sscanf(c + 1, "%dx%d", &rw, &rh) == 2 && at == f && rw > 0 && rh > 0) {
                rt_log("gfx: REKARINA_TEST_RESIZE: the window becomes %dx%d at frame %ld", rw, rh, f);
                wsys_resize(rw, rh);
            }
            p = strchr(p, ',');
            if (p) p++;
        }
    }
    {   // DEBUG-FAULT: REKARINA_TEST_FAULT=<crash|fatal|lost>:<frame>: at that presented frame the game ends as it would
        // on a crash (an access violation), a fatal error, or a lost GPU device (the fault report, then the fatal error
        // the submit path raises), for a check of what the player is left with (report.c)
        static long at = -2;
        static char kind[8];
        if (at == -2) {
            const char *e = cfg_get("REKARINA_TEST_FAULT"), *c = e ? strchr(e, ':') : NULL;
            at = c ? atol(c + 1) : -1;
            if (c) snprintf(kind, sizeof kind, "%.*s", (int)(c - e), e);
        }
        if (at >= 0 && (long)__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) == at) {
            rt_log("gfx: REKARINA_TEST_FAULT=%s at frame %ld", kind, at);
            if (!strcmp(kind, "crash")) *(volatile int *)(uintptr_t)16 = 1;
            if (!strcmp(kind, "lost")) {
                vk_device_fault_report(VK_ERROR_DEVICE_LOST);
                rt_fatal("vk: vkQueueSubmit failed (%d; device lost) [REKARINA_TEST_FAULT]", (int)VK_ERROR_DEVICE_LOST);
            }
            rt_fatal("test: a fatal error [REKARINA_TEST_FAULT]");
        }
    }
    gfx_window_start();
    VkCtx *c = vk_ctx();
    plat_rmutex_lock(&g_pl);
    flip_locked();
    FT(FT_NOTE, 3, 0, 0);
#ifdef GAME_OOT
    if (defer) g_output_format = fmt;
    int mismatch = g_sc_request_format != g_output_format;
    int available = wsys_minimized() == 0;
    if (available && (!g_sc || mismatch || plat_cas32(&g_dirty, 0, 0))) {
        if (!create_swapchain() && (mismatch || !g_sc)) goto out;
    }
#else
    if (!g_sc || plat_cas32(&g_dirty, 0, 0)) {
        if (!wsys_minimized() && !create_swapchain() && !g_sc) goto out;
    }
#endif
    if (wsys_minimized() || !g_sc) goto out;
#ifdef GAME_OOT
    if (g_sc_hdr && !gfx_hdr_surface_capable()) goto out;
#endif

    unsigned slot = g_frame % FRAMES;
    if (vkWaitForFences(c->device, 1, &g_fence[slot], VK_TRUE, 5000000000ull) != VK_SUCCESS) {
        rt_log("gfx: present fence wait timed out");
        goto out;
    }
    uint32_t idx = 0;
    VkResult r = acquire_locked(c->device, g_acquire[slot], &idx);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        plat_xchg32(&g_dirty, 1);
        if (!create_swapchain()) goto out;
        r = acquire_locked(c->device, g_acquire[slot], &idx);
    }
    vk_device_fault_report(r);
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) goto out;     // lost, not ready: skip the frame
    if (r == VK_SUBOPTIMAL_KHR) plat_xchg32(&g_dirty, 1);
    vkResetFences(c->device, 1, &g_fence[slot]);
    if (g_sc_hdr) {   // a changed peak or white goes to the display
        vk_reflex_wsi_lock();
        vk_hdr_metadata(g_sc, 0);
        vk_reflex_wsi_unlock();
    }
    if (g_th_pending[slot]) {   // that fence has signalled: the thumbnail its submission wrote is complete
        g_th_tag[g_th_pending[slot] % (uint64_t)g_th_n] = g_th_pending[slot];
        g_th_pending[slot] = 0;
    }

    int cx = 0, cy = 0, cw = w, ch = h;
    if (crop && crop[2] > 0 && crop[3] > 0) { cx = crop[0]; cy = crop[1]; cw = crop[2]; ch = crop[3]; }
    if (cx < 0) cx = 0;
    if (cy < 0) cy = 0;
    if (cx + cw > w) cw = w - cx;
    if (cy + ch > h) ch = h - cy;
    window_settings_live();
    // The menu overlay: laid out before recording (its first frame starts the ImGui back end and uploads the font)
    const char *menu_shot = NULL;
    MenuTarget mt = {c->instance, c->phys, c->device, c->gfx_family, c->gfx_queue,
                     g_sc_hdr ? VK_FORMAT_R8G8B8A8_UNORM : g_sc_fmt, g_sc_ext.width,
                     g_sc_ext.height, cw, ch, g_scaler
#ifdef GAME_OOT
                     , g_sc_ext.width, g_sc_ext.height, nvidia_scene_aa
#endif
    };
    int overlay = 0;
#ifdef GAME_OOT
    int pre_output_menu = guest_menu.frame == bt_frame_next &&
                          guest_menu.composed && guest_menu.destination == src;
    if (pre_output_menu) menu_shot = guest_menu.shot;
    else
#endif
    if (g_sc_view[idx]) {
        const CfgRenderState *settings = cfg_render_use(NULL);
        overlay = menu_overlay_frame(__atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED), &mt, &menu_shot);
        cfg_render_use(settings);
#ifdef GAME_OOT
        plat_xchg32(&guest_menu_wanted, overlay);
#endif
    }
#ifdef GAME_OOT
    if (g_sc_hdr || pre_output_menu) overlay = 0;
#endif
    if (plat_xchg32(&g_shot_req, 0) && !menu_shot) {   // the screenshot key (input.c): this frame into screenshots/
        menu_shot = screenshot_path();
        if (menu_shot && !cfg_bool("REKARINA_SCREENSHOT_OVERLAY")) overlay = 0;   // the picture alone, the menu left out
    }
    {   // DEBUG-SCSHOT: REKARINA_SC_SHOT=lo-hi:step saves the swapchain image of those frames (REKARINA_SHOTS_DIR)
        static char scp[512];
        const char *e = cfg_get("REKARINA_SC_SHOT"), *dir = cfg_get("REKARINA_SHOTS_DIR");
        unsigned long long lo, hi, st, f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (!menu_shot && e && sscanf(e, "%llu-%llu:%llu", &lo, &hi, &st) == 3 && st && f >= lo && f <= hi && (f - lo) % st == 0) {
            snprintf(scp, sizeof scp, "%s/sc%05llu.png", dir ? dir : ".", f);
            menu_shot = scp;
        }
    }
    static char hlp[512];
    int hlshot = 0;
    {   // DEBUG-HLSHOT: REKARINA_HL_SHOT=lo-hi:step saves those frames' swapchain image (sc<frame>.png) and frame
        // generation's HUD-less picture of the same present (hl<frame>.png) to REKARINA_SHOTS_DIR
        static char scp2[512];
        const char *e = cfg_get("REKARINA_HL_SHOT"), *dir = cfg_get("REKARINA_SHOTS_DIR");
        unsigned long long lo, hi, st, f = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
        if (e && sscanf(e, "%llu-%llu:%llu", &lo, &hi, &st) == 3 && st && f >= lo && f <= hi && (f - lo) % st == 0) {
            if (!menu_shot) snprintf(scp2, sizeof scp2, "%s/sc%05llu.png", dir ? dir : ".", f), menu_shot = scp2;
            snprintf(hlp, sizeof hlp, "%s/hl%05llu.png", dir ? dir : ".", f);
            hlshot = 1;
        }
    }
    if (g_video.pending[slot]) {   // this slot's copy of FRAMES presents ago: its work ended with the wait above
        g_video.pending[slot] = 0;
        vmaInvalidateAllocation(c->vma, g_video.alloc[slot], 0, VK_WHOLE_SIZE);
        video_frame(g_video.ptr[slot], g_video.pw[slot], g_video.ph[slot]);
    }
    int video = video_due() && video_buffers();
    int cap = (g_capture || menu_shot) && ensure_capture_buffer();
    int thumb = g_th_n && ensure_thumbs();

    VkCommandBuffer cb = g_cb[slot];
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cb, &bi);
    uint64_t crumb = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED);
    vk_crumb(cb, VK_CRUMB_PRESENT, crumb);
    VkImage dst = g_sc_img[idx];
    sync_barrier(cb, dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0,
                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    const int rect[4] = {cx, cy, cw, ch};
    int to[4];   // where the picture goes in the swapchain image: {x, y, w, h}
    gfx_fit_rect(cw, ch, (int)g_sc_ext.width, (int)g_sc_ext.height, to);
    if (g_sc_ext.width && g_sc_ext.height) {
        for (int i = 0; i < 4; i++) g_pic[i] = (float)to[i] / (float)(i & 1 ? g_sc_ext.height : g_sc_ext.width);
        __atomic_store_n(&g_pic_ok, 1, __ATOMIC_RELEASE);
    }
    if (to[2] != (int)g_sc_ext.width || to[3] != (int)g_sc_ext.height) {   // black bars beside or above and below it
        VkClearColorValue black = {.float32 = {0, 0, 0, 1}};
        VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &all);
        sync_barrier(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT);
    }
    VkImage psrc;
    int prect[4];
    int full_picture = general;
#ifdef GAME_OOT
    if (g_sc_hdr) full_picture = 0;   // The guest's completed output shader already wrote PQ.
#endif
    int fsr = picture_into(cb, slot, src, fmt, w, h, rect, general, full_picture, 0, dst, to, &psrc, prect);
    g_scaler = fsr == 1 ? 2 : cw != to[2] || ch != to[3];   // 2 from vk_fsr_record: sharpened, not scaled
    // The open menu's live picture (ui/menu.cpp): the game picture once more, scaled into the menu's frame; the menu
    // leaves that rectangle clear.
    int pv[4];
    if (overlay && src && menu_preview_rect(pv) && pv[2] > 0 && pv[3] > 0 && pv[0] >= 0 && pv[1] >= 0 &&
        pv[0] + pv[2] <= (int)g_sc_ext.width && pv[1] + pv[3] <= (int)g_sc_ext.height) {
        sync_barrier(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                     VK_ACCESS_2_TRANSFER_WRITE_BIT);
        if (general)
            sync_barrier(cb, psrc, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                         VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkImageBlit live = {
            .srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .srcOffsets = {{prect[0], prect[1], 0}, {prect[0] + cw, prect[1] + ch, 1}},
            .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
            .dstOffsets = {{pv[0], pv[1], 0}, {pv[0] + pv[2], pv[1] + pv[3], 1}}};
        vkCmdBlitImage(cb, psrc, general ? VK_IMAGE_LAYOUT_GENERAL : VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &live, VK_FILTER_LINEAR);
        if (general)
            sync_barrier(cb, psrc, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0,
                         VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, 0);
    }
    // frame generation's picture without the HUD, through the same path; after the menu's live picture, which reads the
    // colour adjustment's image that this makes again
    VkImage hudless = src ?
        hudless_record(bt_frame_next, cb, slot, w, h, rect, to, general, pq) : VK_NULL_HANDLE;
    hlshot = hlshot && hudless && hlcap_buffer();
    if (hlshot) {   // DEBUG-HLSHOT
        sync_barrier(cb, hudless, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                     VK_ACCESS_2_MEMORY_WRITE_BIT, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy hc = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                .imageExtent = {g_sc_ext.width, g_sc_ext.height, 1}};
        vkCmdCopyImageToBuffer(cb, hudless, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_hlcap_buf, 1, &hc);
        sync_barrier(cb, hudless, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0,
                     VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT);
    }
    // last: the image's layout, and the stage and access of its last write, as each step leaves it
    VkImageLayout last = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    VkPipelineStageFlags2 last_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    VkAccessFlags2 last_write = VK_ACCESS_2_TRANSFER_WRITE_BIT;
#ifdef GAME_OOT
    VkImageView menu_view = VK_NULL_HANDLE;
#else
    VkImageView menu_view = overlay && g_sc_hdr ? vk_hdr_ui_target(cb, g_sc_ext.width, g_sc_ext.height) : VK_NULL_HANDLE;
    if (overlay && g_sc_hdr && !menu_view) overlay = 0;
#endif
    if (overlay) {
        sync_barrier(cb, dst, last, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, last_stage, last_write,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                                        .imageView = menu_view ? menu_view : g_sc_view[idx],
                                        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        .loadOp = menu_view ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
                                        .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
        VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, g_sc_ext}, .layerCount = 1,
                              .colorAttachmentCount = 1, .pColorAttachments = &ca};
        vkCmdBeginRendering(cb, &ri);
        menu_overlay_draw(cb, 0);
        vkCmdEndRendering(cb);
#ifndef GAME_OOT
        if (menu_view) vk_hdr_ui_compose(cb, g_sc_view[idx], g_sc_fmt, g_sc_ext.width, g_sc_ext.height);
#endif
        last = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        last_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        last_write = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    }
    VkImage interface_coverage = VK_NULL_HANDLE;
    VkImageView interface_coverage_view = VK_NULL_HANDLE;
    int coverage_pair = hudless && g_hl_view && general && vk_interface_coverage_enabled();
    VkImageView guest_alpha = VK_NULL_HANDLE;
    const int *guest_preview = NULL;
#ifdef GAME_OOT
    if (pre_output_menu) {
        guest_alpha = guest_menu.alpha;
        guest_preview = guest_menu.preview;
        if (!guest_alpha) coverage_pair = 0;
    }
#endif
    if (coverage_pair) {
        VkImageView overlay_alpha = guest_alpha;
        if (overlay && menu_overlay_can_replay()) {
            overlay_alpha = vk_interface_coverage_overlay(cb, bt_frame_next, g_sc_ext,
                                                          menu_view ? VK_FORMAT_R8G8B8A8_UNORM : g_sc_fmt);
            if (overlay_alpha) {
                VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                    .imageView = overlay_alpha, .imageLayout = VK_IMAGE_LAYOUT_GENERAL,
                    .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
                VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, g_sc_ext},
                    .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &ca};
                vkCmdBeginRendering(cb, &ri);
                menu_overlay_draw(cb, 1);
                vkCmdEndRendering(cb);
            }
        }
        if (!overlay || overlay_alpha)
            vk_interface_coverage_present(cb, bt_frame_next, src, (VkExtent2D){(uint32_t)w, (uint32_t)h},
                                          g_sc_ext, rect, to, overlay_alpha, guest_preview, &interface_coverage, &interface_coverage_view);
    }
    if (cap || video) {
        sync_barrier(cb, dst, last, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, last_stage, last_write,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy bc = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                .imageExtent = {g_sc_ext.width, g_sc_ext.height, 1}};
        if (cap) vkCmdCopyImageToBuffer(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_cap_buf, 1, &bc);
        if (video) {   // read FRAMES presents from now (REKARINA_VIDEO)
            vkCmdCopyImageToBuffer(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_video.buf[slot], 1, &bc);
            g_video.pending[slot] = 1, g_video.pw[slot] = (int)g_sc_ext.width, g_video.ph[slot] = (int)g_sc_ext.height;
        }
        last = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        last_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        last_write = 0;
    }
    if (thumb) {
        if (last != VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL)
            sync_barrier(cb, dst, last, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, last_stage, last_write,
                         VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        last = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        last_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        last_write = 0;
        record_thumb(cb, dst, slot);
    }
    if (src) vk_fg_frame(cb, g_sc, g_sc_fmt, g_sc_ext, hudless, g_sc_hdr, to);
    if (src) output_rate();
    if (src) tagged = vk_dlssg_frame(cb, g_sc_ext, hudless, g_hl_view, g_sc_fmt, interface_coverage, interface_coverage_view);
    sync_barrier(cb, dst, last, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, last_stage, last_write, VK_PIPELINE_STAGE_2_NONE, 0);
    vk_crumb(cb, VK_CRUMB_PRESENT_END, crumb);
    bt_write(cb);   // the frame clock's shown moment
    vkEndCommandBuffer(cb);

    VkSemaphoreSubmitInfo wait = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_acquire[slot],
                                  .stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT};
    VkSemaphoreSubmitInfo sig = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_render_done[idx],
                                 .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkCommandBufferSubmitInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .waitSemaphoreInfoCount = 1, .pWaitSemaphoreInfos = &wait,
                        .commandBufferInfoCount = 1, .pCommandBufferInfos = &cbi, .signalSemaphoreInfoCount = 1,
                        .pSignalSemaphoreInfos = &sig};
    uint64_t reflex_id = g_reflex_next_id;
    g_reflex_next_id = 0;
#ifdef VK_NV_LOW_LATENCY_2_EXTENSION_NAME
    VkLatencySubmissionPresentIdNV reflex_sub;
    vk_reflex_chain_submit(&si, &reflex_sub, reflex_id);
#endif
    vk_queue_lock();
    vk_drv_urgent(1);
    r = vkQueueSubmit2(c->gfx_queue, 1, &si, g_fence[slot]);
    vk_drv_urgent(-1);
    vk_queue_unlock();
    if (reflex_id) vk_reflex_submit_end(reflex_id);
    FT(FT_NOTE, 4, 0, 0);
    if (r == VK_SUCCESS) {
        g_flip_on = 1;
        g_flip_idx = idx;
        g_flip_reflex_id = reflex_id;
        if (!defer) flip_locked();
    } else {
        vk_device_fault_report(r);
        rt_log("gfx: present submit failed (%d)", (int)r);
    }
    g_frame++;
    if (cap) {
        vkWaitForFences(c->device, 1, &g_fence[slot], VK_TRUE, 5000000000ull);
        vmaInvalidateAllocation(c->vma, g_cap_alloc, 0, VK_WHOLE_SIZE);
        g_cap_w = (int)g_sc_ext.width;
        g_cap_h = (int)g_sc_ext.height;
        g_cap_valid = 1;
        if (menu_shot) save_capture(menu_shot);
        if (hlshot) {   // DEBUG-HLSHOT
            vmaInvalidateAllocation(c->vma, g_hlcap_alloc, 0, VK_WHOLE_SIZE);
            save_pixels(hlp, g_hlcap_ptr);
        }
    }
out:
    plat_rmutex_unlock(&g_pl);
    return tagged;
}

void vk_present_image(VkImage src, VkImageLayout layout, VkFormat fmt, int w, int h, const int crop[4]) {
    present_image(src, layout, fmt, w, h, crop, 0, NULL, NULL);
}

int vk_present_blit(VkImage src, VkImageLayout layout, VkFormat fmt, int w, int h, const int crop[4],
                    const VkGuestPq *pq, const char *nvidia_scene_aa) {
    if (!src) return 0;
    return present_image(src, layout, fmt, w, h, crop, 1, pq, nvidia_scene_aa);
}

void vk_present_flip(void) {
    plat_rmutex_lock(&g_pl);
    flip_locked();
    plat_rmutex_unlock(&g_pl);
}

void gfx_window_resize(int cw, int chh) {
    gfx_window_start();
    wsys_resize(cw, chh);
    plat_rmutex_lock(&g_pl);
    create_swapchain();
    plat_rmutex_unlock(&g_pl);
}

// Called by vk_shutdown before the device goes away; the window and its thread stay.
void gfx_window_release_vk(void) {
    VkCtx *c = vk_ctx();
    plat_rmutex_lock(&g_pl);
    if (c && g_started) {
        vk_queue_lock();
        vkDeviceWaitIdle(c->device);
        vk_queue_unlock();
        g_flip_on = 0;   // its blit is done (device idle); the swapchain goes away unpresented
        menu_vk_release();
        vk_fsr_release();
        vk_color_release();
        vk_hdr_release();
        vk_contact_release();
        free_capture();
        free_thumbs();
        for (int i = 0; i < FRAMES; i++) {
            vkDestroySemaphore(c->device, g_acquire[i], NULL);
            vkDestroyFence(c->device, g_fence[i], NULL);
        }
        for (uint32_t i = 0; i < g_sc_n; i++) {
            vkDestroySemaphore(c->device, g_render_done[i], NULL);
            if (g_sc_view[i]) vkDestroyImageView(c->device, g_sc_view[i], NULL);
            g_sc_view[i] = VK_NULL_HANDLE;
        }
        if (g_pool) vkDestroyCommandPool(c->device, g_pool, NULL);
        vk_reflex_swapchain_destroyed();
        if (vk_fg_swapchain()) vk_fg_release();   // AMD's contexts and swapchain (also one being made again)
        else if (g_sc && vk_dlssg_swapchain_ready()) vk_dlssg_destroy_swapchain(c->device, g_sc);
        else if (g_sc) vkDestroySwapchainKHR(c->device, g_sc, NULL);
        if (g_surface) vkDestroySurfaceKHR(c->instance, g_surface, NULL);
        g_sc = VK_NULL_HANDLE; g_surface = VK_NULL_HANDLE; g_pool = VK_NULL_HANDLE; g_sc_n = 0;
#ifdef GAME_OOT
        memset(&guest_menu, 0, sizeof guest_menu);
        plat_xchg32(&guest_menu_wanted, 0);
        g_output_format = g_sc_request_format = VK_FORMAT_UNDEFINED;
        plat_xchg32(&g_surface_pq, 0);
#endif
        g_started = 0;     // the window and its thread stay; the next gfx_window_start makes a new surface
    }
    plat_rmutex_unlock(&g_pl);
}

void gfx_window_start(void) {
    plat_rmutex_lock(&g_start_lock);
    if (!g_started) {
        if (!vk_init()) rt_fatal("gfx: no usable Vulkan device (see the vk: lines above)");
        VkCtx *c = vk_ctx();
        static int opened;
        if (!opened && !(opened = wsys_open())) rt_fatal("gfx: window creation failed");
        g_surface = wsys_surface(c->instance);
        if (!g_surface) rt_fatal("gfx: no Vulkan surface for the window");

        VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                       .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = c->gfx_family};
        vkCreateCommandPool(c->device, &pci, NULL, &g_pool);
        VkCommandBufferAllocateInfo ai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = g_pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = FRAMES};
        vkAllocateCommandBuffers(c->device, &ai, g_cb);
        VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT};
        VkSemaphoreCreateInfo semi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        for (int i = 0; i < FRAMES; i++) {
            vkCreateFence(c->device, &fi, NULL, &g_fence[i]);
            vkCreateSemaphore(c->device, &semi, NULL, &g_acquire[i]);
        }
        g_frame = 0;
        g_started = 1;
        plat_rmutex_lock(&g_pl);
        create_swapchain();
        plat_rmutex_unlock(&g_pl);
        int cw = 0, ch = 0;
        wsys_client_size(&cw, &ch);
        rt_log("gfx: window %dx%d at DPI %u, swapchain %ux%u format %d", cw, ch, gfx_window_dpi(), g_sc_ext.width,
               g_sc_ext.height, (int)g_sc_fmt);
    }
    plat_rmutex_unlock(&g_start_lock);
    vsync_ensure();
}

// ---- the entry step's progress picture (boot.c prep_wait, before the guest starts)
// The window opens before the game's first frame, the swapchain image is cleared to the menu's dark brown and the menu
// overlay draws the progress panel (ui/prep_screen.cpp, laid out by menu_overlay_frame while prep_screen_active), so
// a long preparation shows its progress instead of no window or a black one. REKARINA_PREP_SHOT=<file.png> saves the
// first frame drawn once the step is past REKARINA_PREP_SHOT_AT percent (default 40), for a look at the design.
void gfx_present_prep_frame(void) {
    gfx_window_start();
    VkCtx *c = vk_ctx();
    plat_rmutex_lock(&g_pl);
    flip_locked();
    if (!g_sc || plat_cas32(&g_dirty, 0, 0)) {
        if (!wsys_minimized() && !create_swapchain() && !g_sc) goto out;
    }
    if (wsys_minimized() || !g_sc) goto out;
#ifdef GAME_OOT
    if (g_sc_hdr) goto out;   // The boot menu has no guest encoder and only draws in SDR.
#endif
    unsigned slot = g_frame % FRAMES;
    if (vkWaitForFences(c->device, 1, &g_fence[slot], VK_TRUE, 5000000000ull) != VK_SUCCESS) goto out;
    uint32_t idx = 0;
    // as present_image: a frame generation swapchain (set at its start from the settings) hands out its own images
    VkResult r = acquire_locked(c->device, g_acquire[slot], &idx);
    if (r == VK_ERROR_OUT_OF_DATE_KHR) {
        plat_xchg32(&g_dirty, 1);
        if (!create_swapchain()) goto out;
        r = acquire_locked(c->device, g_acquire[slot], &idx);
    }
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR) goto out;
    if (r == VK_SUBOPTIMAL_KHR) plat_xchg32(&g_dirty, 1);
    vkResetFences(c->device, 1, &g_fence[slot]);
    MenuTarget mt = {c->instance, c->phys, c->device, c->gfx_family, c->gfx_queue,
                     g_sc_hdr ? VK_FORMAT_R8G8B8A8_UNORM : g_sc_fmt, g_sc_ext.width, g_sc_ext.height,
                     (int)g_sc_ext.width, (int)g_sc_ext.height, 0
#ifdef GAME_OOT
                     , g_sc_ext.width, g_sc_ext.height, NULL
#endif
    };
    const char *menu_shot = NULL;
    int overlay = g_sc_view[idx] && menu_overlay_frame(0, &mt, &menu_shot);
    {
        static int shot_done;
        const char *sp = cfg_get("REKARINA_PREP_SHOT"), *at = cfg_get("REKARINA_PREP_SHOT_AT");
        int64_t done, total;
        prep_progress(&done, &total, NULL);
        if (sp && *sp && !shot_done && overlay && total && done * 100 >= total * (at && *at ? atoi(at) : 40))
            menu_shot = sp, shot_done = 1;
    }
    int cap = menu_shot && ensure_capture_buffer();
    VkCommandBuffer cb = g_cb[slot];
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    vkBeginCommandBuffer(cb, &bi);
    VkImage dst = g_sc_img[idx];
    sync_barrier(cb, dst, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_TRANSFER_BIT, 0,
                 VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
    VkClearColorValue bg = {.float32 = {0.035f, 0.024f, 0.016f, 1}};   // the panel's surround, darker than its plate
    VkImageSubresourceRange all = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &bg, 1, &all);
    VkImageLayout last = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    VkPipelineStageFlags2 last_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    VkAccessFlags2 last_write = VK_ACCESS_2_TRANSFER_WRITE_BIT;
#ifdef GAME_OOT
    VkImageView menu_view = VK_NULL_HANDLE;
#else
    VkImageView menu_view = overlay && g_sc_hdr ? vk_hdr_ui_target(cb, g_sc_ext.width, g_sc_ext.height) : VK_NULL_HANDLE;
    if (overlay && g_sc_hdr && !menu_view) overlay = 0;
#endif
    if (overlay) {
        sync_barrier(cb, dst, last, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, last_stage, last_write,
                     VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                     VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
        VkRenderingAttachmentInfo ca = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO,
                                        .imageView = menu_view ? menu_view : g_sc_view[idx],
                                        .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                        .loadOp = menu_view ? VK_ATTACHMENT_LOAD_OP_CLEAR : VK_ATTACHMENT_LOAD_OP_LOAD,
                                        .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
        VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, g_sc_ext}, .layerCount = 1,
                              .colorAttachmentCount = 1, .pColorAttachments = &ca};
        vkCmdBeginRendering(cb, &ri);
        menu_overlay_draw(cb, 0);
        vkCmdEndRendering(cb);
#ifndef GAME_OOT
        if (menu_view) vk_hdr_ui_compose(cb, g_sc_view[idx], g_sc_fmt, g_sc_ext.width, g_sc_ext.height);
#endif
        last = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        last_stage = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        last_write = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
    }
    if (cap) {
        sync_barrier(cb, dst, last, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, last_stage, last_write,
                     VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
        VkBufferImageCopy bc = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                .imageExtent = {g_sc_ext.width, g_sc_ext.height, 1}};
        vkCmdCopyImageToBuffer(cb, dst, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, g_cap_buf, 1, &bc);
        last = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        last_stage = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        last_write = 0;
    }
    sync_barrier(cb, dst, last, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, last_stage, last_write, VK_PIPELINE_STAGE_2_NONE, 0);
    vkEndCommandBuffer(cb);
    VkSemaphoreSubmitInfo wait = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_acquire[slot],
                                  .stageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT};
    VkSemaphoreSubmitInfo sig = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_render_done[idx],
                                 .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkCommandBufferSubmitInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, .commandBuffer = cb};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .waitSemaphoreInfoCount = 1, .pWaitSemaphoreInfos = &wait,
                        .commandBufferInfoCount = 1, .pCommandBufferInfos = &cbi, .signalSemaphoreInfoCount = 1,
                        .pSignalSemaphoreInfos = &sig};
    vk_queue_lock();
    vk_drv_urgent(1);
    r = vkQueueSubmit2(c->gfx_queue, 1, &si, g_fence[slot]);
    vk_drv_urgent(-1);
    vk_queue_unlock();
    if (r == VK_SUCCESS) {
        g_flip_on = 1;
        g_flip_idx = idx;
        flip_locked();
    }
    g_frame++;
    if (cap && r == VK_SUCCESS) {
        vkWaitForFences(c->device, 1, &g_fence[slot], VK_TRUE, 5000000000ull);
        vmaInvalidateAllocation(c->vma, g_cap_alloc, 0, VK_WHOLE_SIZE);
        g_cap_w = (int)g_sc_ext.width;
        g_cap_h = (int)g_sc_ext.height;
        g_cap_valid = 1;
        save_capture(menu_shot);
    }
out:
    plat_rmutex_unlock(&g_pl);
}
