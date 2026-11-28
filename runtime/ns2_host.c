// Switch 2 SDK calls that reach services this host does not serve (GAME_NS2, Rekarina). Each answers what an offline PC
// has: SDR output at a fixed rate with no VRR, one stereo output device, no Joy-Con 2 mouse sensor, a fixed graphics
// scale tier, no Nintendo Account, the PC's clock as the network clock, and no add-on content.
//
// Behaviour ported from MacNativeGames games/totk/runtime ns2_host.c, offline_host.c, account.c and time_host.c (MIT),
// which took it from seedds/nxnative (MIT; dm/host/hle_misc.c, hle_hid.c, hle_oe.c, hle_audren.c, dm/gfx/hle_vi.c).
// Written again for this host's calling convention: every function takes the guest Cpu and reads its arguments.
#ifdef GAME_NS2
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "cfg.h"
#include "fs.h"
#include "host.h"
#include "ov.h"
#include "vk.h"
#include "platform/platform.h"

#define R_NO_NINTENDO_ACCOUNT 0x7c67u

static void h_ok(Cpu *c) { ret_x(c, 0); }
static void h_nothing(Cpu *c) { (void)c; }
static void h_zero(Cpu *c) { ret_x(c, 0); }

static uint64_t bound_event(Cpu *c, int signalled) {   // a SystemEvent the host owns, bound once
    uint64_t ev = (uint64_t)(uintptr_t)mem_kernel_alloc(0x80, 16);
    kern_bind_system_event(c, ev, 0);
    if (signalled) kern_signal_system_event(ev);
    return ev;
}

// ---- vi: HDR, VRR, refresh rate.
static int vi_hdr_capability(void) {
#ifdef GAME_OOT
    return vk_hdr_wanted() && gfx_hdr_surface_capable();
#else
    return 0;
#endif
}

#ifdef GAME_OOT
void ns2_hdr_capability_changed(void) {
    static int previous = -1, previous_signed;
    int capability = vi_hdr_capability();
    int signed_compositor = capability && !vk_hdr_native();
    rt_patch_switch(rt_patch_set("hdr_compositor"), signed_compositor);
    if (previous >= 0 && (previous != capability || previous_signed != signed_compositor))
        oe_push_message(OE_MSG_OPERATION_MODE_CHANGED);
    previous = capability;
    previous_signed = signed_compositor;
}
#endif

// Result QueryDisplayHdrStatus(bool *on, HdrOutputFormat *fmt, u16 *peak, const Display *)
static void vi_hdr_status(Cpu *c) {
    const CfgRenderState *settings = cfg_render_frame_acquire();
    const CfgRenderState *previous_settings = cfg_render_use(settings);
    uint8_t *on = arg_p(c, 0);
    uint32_t *fmt = arg_p(c, 1);
    uint16_t *peak = arg_p(c, 2);
    int capability = vi_hdr_capability();
    if (capability) {
        double p = cfg_double("REKARINA_HDR_PEAK_NITS");
        if (p < 80.0) p = 80.0;
        if (p > 10000.0) p = 10000.0;
        if (on) *on = 1;
        if (fmt) *fmt = 1;   // HDR10 (PQ Rec. 2020)
        if (peak) *peak = (uint16_t)(p + 0.5);
        static int logged;
        if (!logged++) rt_log("ns2: vi::QueryDisplayHdrStatus: native HDR10, peak %u nits", (unsigned)(p + 0.5));
    } else {
        if (on) *on = 0;
        if (fmt) *fmt = 0;
        if (peak) *peak = 0;
    }
    static int32_t logged_capability = -1;
    if ((rt_trace & RT_TRACE_GFX) && plat_xchg32(&logged_capability, capability) != capability)
        rt_log("ns2: vi HDR capability query: %d", capability);
    cfg_render_use(previous_settings);
    cfg_render_release(settings);
    ret_x(c, 0);
}
static void vi_reference_white(Cpu *c) {   // Result QueryDisplayHdrPaperWhite(u16 *nits, const Display *)
    const CfgRenderState *settings = cfg_render_frame_acquire();
    const CfgRenderState *previous = cfg_render_use(settings);
    uint16_t *nits = arg_p(c, 0);
    if (nits) {
        if (vi_hdr_capability()) {
            double g = cfg_double("REKARINA_HDR_GAME_NITS");
            if (g < 48.0) g = 48.0;
            if (g > 500.0) g = 500.0;
            *nits = (uint16_t)(g + 0.5);
        } else {
            *nits = 203;
        }
    }
    cfg_render_use(previous);
    cfg_render_release(settings);
    ret_x(c, 0);
}
static void vi_vrr_supported(Cpu *c) {   // Result QueryDisplayVrrSupported(bool *, const Display *)
    uint8_t *on = arg_p(c, 0);
    if (on) *on = 0;
    ret_x(c, 0);
}
// Result QueryDisplayCurrentRefreshRate(RefreshRate *rate, bool *, const Display *): 60 Hz. RefreshRate is a u32 of
// Hz in the SDK's enum (60); the flag says whether the rate is variable.
static void vi_refresh_rate(Cpu *c) {
    uint32_t *rate = arg_p(c, 0);
    uint8_t *var = arg_p(c, 1);
    if (rate) *rate = 60;
    if (var) *var = 0;
    ret_x(c, 0);
}
static void vi_ext_swap(Cpu *c) {   // Result SetLayerExtendedSwapInterval(Layer *, int, int): without VRR the NVN one stays
    static int once;
    if (!once++) rt_log("ns2: vi::SetLayerExtendedSwapInterval %d %d: no VRR, the NVN swap interval stays",
                        (int)arg_x(c, 1), (int)arg_x(c, 2));
    ret_x(c, 0);
}

// ---- audio: one built-in stereo output device
enum { DEVNAME = 0x100 };
static void devname(char *out) {
    memset(out, 0, DEVNAME);
    strcpy(out, "AudioBuiltInSpeakerOutput");
}
static void audio_list(Cpu *c) {   // int ListAudioOutputDeviceName(AudioDeviceName *, int)
    char *names = arg_p(c, 0);
    int n = (int)arg_x(c, 1);
    if (n > 0 && names) devname(names);
    ret_x(c, n > 0 && names ? 1 : 0);
}
static void audio_active(Cpu *c) {   // void GetActiveAudioOutputDeviceName(AudioDeviceName *)
    char *out = arg_p(c, 0);
    if (out) devname(out);
}
static void audio_channels(Cpu *c) { ret_x(c, 2); }
static uint32_t device_volume = 0x3f800000u;   // 1.0f, kept and read back
static void audio_set_volume(Cpu *c) { __atomic_store_n(&device_volume, (uint32_t)c->v[0], __ATOMIC_RELAXED); }
static void audio_volume(Cpu *c) { c->v[0] = __atomic_load_n(&device_volume, __ATOMIC_RELAXED); }
static uint8_t autotune;
static void audio_set_autotune(Cpu *c) { __atomic_store_n(&autotune, (uint8_t)(arg_x(c, 0) & 1), __ATOMIC_RELAXED); }
static void audio_autotune(Cpu *c) { ret_x(c, __atomic_load_n(&autotune, __ATOMIC_RELAXED)); }
static void audio_notification(Cpu *c) {   // Result AcquireAudioDeviceNotificationForOutput(os::SystemEvent *)
    static int acquired;
    uint64_t ev = arg_x(c, 0);
    if (acquired++) {
        ret_x(c, 0x499);
        return;
    }
    if (ev) kern_bind_system_event(c, ev, 0);
    ret_x(c, 0);
}

// ---- hid: the Joy-Con 2 mouse sensor (JCMS) and the mouse (Nmouse): none on the host
static void jcms_handle(Cpu *c) {   // JcmsHandle GetJcmsHandle(const u32 &npad, const JcmsType &)
    const uint32_t *id = arg_p(c, 0), *type = arg_p(c, 1);
    ret_x(c, (id ? *id & 0xff : 0) | (type ? (*type & 0xff) << 8 : 0));
}
static void nmouse_handles(Cpu *c) { ret_x(c, 0); }   // int GetNmouseHandles(NmouseHandle *, int, u32): none
static void nmouse_types(Cpu *c) {   // void GetNmouseTypeSet(BitFlagSet<32, NmouseType> *, NmouseHandle)
    uint32_t *out = arg_p(c, 0);
    if (out) *out = 0;
}

// ---- oe
// The graphics scale tier is the factor by which the Switch 2's compositor enlarges the game's picture: OoT
// draws its frame at 4/(4 + tier) of the window texture (1, 4/5, 2/3; its 3D scene at 3/4 of that) and leaves
// the rest to the system. Tier 0 (default) draws at the window's full size. With tier 1 or 2 the present takes the
// top-left part the game drew and enlarges it to the window as the compositor would (nvn_api_obj.c), with the
// upscaler (REKARINA_UPSCALER): 64 % or 44 % of the pixels to shade. On the forest porch (RTX 5070) the GPU time drops
// only from 19.9 to 18.4 and 17.9 ms: the frame is not bound by its pixels. REKARINA_GRAPHICS_SCALE=0|1|2
// (OOT_GRAPHICS_SCALE, the old name, still read).
int host_graphics_tier(void) {
    static int t = -1;
    if (t < 0) {
        const char *e = cfg_get("REKARINA_GRAPHICS_SCALE");
        if (!e || !*e) e = getenv("OOT_GRAPHICS_SCALE");
        t = e && *e >= '0' && *e <= '2' ? *e - '0' : 0;
        if (t) rt_log("oe: graphics scale tier %d: the game draws %s of the window, the present enlarges it", t, t == 1 ? "4/5" : "2/3");
    }
    return t;
}
static void oe_graphics_scale(Cpu *c) { ret_x(c, (uint64_t)host_graphics_tier()); }
static void oe_display_version(Cpu *c) {   // void GetDisplayVersion(DisplayVersion *): 16 bytes of text
    char *out = arg_p(c, 0);
    if (out) memset(out, 0, 16), strcpy(out, "1.1.1");
}

// ---- audio input: a microphone that hears silence
// OoT's option that plays its instrument by voice (or by a real instrument) listens through two input streams of the
// SDK's stream interface (SDK 22: nn::audio::OpenInputStream and the calls below, 16-bit mono at 48 kHz). At its start
// the game lists the virtual input devices and, when there is one, opens a stream on it and one on the "true raw"
// device (GetTrueRawAudioInputDevice); with the option on it starts both when it begins to listen (main+0x336ee10:
// purge, then start a stream whose state is not 2) and reads what they hold each frame (at most 256 frames:
// main+0x336dd78 GetInputStreamFilledFrameCount, ReadInputStream). The host has no microphone. Until 1.2.2 it listed
// none, so no stream was opened and every stream call answered for a stream that does not exist; with the option on
// the game still froze when the instrument came out (player report 2026-11-19, on 1.2.2). Now there is one device of
// each kind, as a console always has, and their streams record silence at their own rate, so the game listens as on
// a console in a quiet room. REKARINA_MIC=none lists no device, as before (a test of the old way).
#define R_AUDIO_NO_INPUT (153u | 1u << 9)   // a failure of the audio module; the game tests these results for zero only
enum { MIC_STREAMS = 4, MIC_STOPPED = 1, MIC_STARTED = 2 };
#define MIC_MAGIC 0x52454b41ull   // "REKA": the high word of the handle in the stream's first word
static struct {
    int used, state, frame_bytes, rate;
    int64_t capacity, taken;   // the frames the stream holds at most; the frames read or dropped since its start
    uint64_t since;            // plat_now_ns at its start
} mic[MIC_STREAMS];
static PlatMutex mic_lock = PLAT_MUTEX_INIT;

static int mic_none(void) {
    static int v = -1;
    if (v < 0) { const char *e = cfg_get("REKARINA_MIC"); v = e && !strcmp(e, "none"); }
    return v;
}
static void audin_log(const char *what) {
    static int told;
    if (!told++)
        rt_log(mic_none() ? "audio input: the game asks for a microphone (%s): none is listed (REKARINA_MIC=none)"
                          : "audio input: the game opens a microphone (%s): the host gives it silence", what);
}
static int mic_slot(Cpu *c) {   // the stream in x0 as one of ours, -1 when it is not (never opened, or closed)
    uint64_t *stream = arg_p(c, 0);
    if (!stream || (*stream >> 32) != MIC_MAGIC) return -1;
    int s = (int)(*stream & 0xffffffffu);
    return s >= 0 && s < MIC_STREAMS && mic[s].used ? s : -1;
}
static int64_t mic_filled(int s) {   // holding mic_lock: the frames recorded and not read, the oldest overwritten
    if (mic[s].state != MIC_STARTED) return 0;
    int64_t made = (int64_t)((plat_now_ns() - mic[s].since) / 1000) * mic[s].rate / 1000000;
    if (made - mic[s].taken > mic[s].capacity) mic[s].taken = made - mic[s].capacity;
    return made - mic[s].taken;
}
// int ListVirtualInputDevices(AudioDeviceName *names, int count): one device (an AudioDeviceName is 256 bytes)
static void audin_list(Cpu *c) {
    char *names = arg_p(c, 0);
    int count = (int)arg_x(c, 1);
    if (mic_none() || !names || count < 1) { ret_x(c, 0); return; }
    memset(names, 0, 256);
    strcpy(names, "rekarina-silence");
    ret_x(c, 1);
}
static void audin_true_raw(Cpu *c) {   // GetTrueRawAudioInputDevice(AudioDeviceName *)
    char *name = arg_p(c, 0);
    if (name) memset(name, 0, 256), strcpy(name, "rekarina-silence-raw");
    ret_x(c, 0);
}
// Result OpenInputStream(InputStreamType *, [os::SystemEvent *,] void *buffer, size_t, const AudioDeviceName &,
// const StreamFormat &, int frames[, int]): the first word of the stream is its implementation pointer, here the
// handle of one of ours; the format is {sample format (2: 16-bit), channels, sample rate}
static void mic_open(Cpu *c, int at) {
    uint64_t *stream = arg_p(c, 0);
    const int32_t *fmt = arg_p(c, at + 3);
    int frames = (int)arg_x(c, at + 4);
    if (stream) *stream = 0;
    audin_log("OpenInputStream");
    if (mic_none() || !stream) { ret_x(c, R_AUDIO_NO_INPUT); return; }
    int bytes = fmt && fmt[0] >= 1 && fmt[0] <= 5 ? (fmt[0] == 1 ? 1 : fmt[0] == 2 ? 2 : fmt[0] == 3 ? 3 : 4) : 2;
    int channels = fmt && fmt[1] >= 1 && fmt[1] <= 8 ? fmt[1] : 1;
    int rate = fmt && fmt[2] >= 8000 && fmt[2] <= 192000 ? fmt[2] : 48000;
    plat_mutex_lock(&mic_lock);
    int s = 0;
    while (s < MIC_STREAMS && mic[s].used) s++;
    if (s < MIC_STREAMS) {
        mic[s].used = 1, mic[s].state = MIC_STOPPED, mic[s].frame_bytes = bytes * channels, mic[s].rate = rate;
        mic[s].capacity = frames > 0 ? frames : rate / 10, mic[s].taken = 0, mic[s].since = 0;
        *stream = MIC_MAGIC << 32 | (uint64_t)s;
    }
    plat_mutex_unlock(&mic_lock);
    if (s == MIC_STREAMS) { ret_x(c, R_AUDIO_NO_INPUT); return; }
    rt_log("audio input: stream %d of silence: %d channel(s) of %d bytes at %d Hz, %d frames held", s, channels, bytes,
           rate, frames);
    ret_x(c, 0);
}
static void audin_open(Cpu *c) { mic_open(c, 1); }
static void audin_open_event(Cpu *c) { mic_open(c, 2); }
static void audin_start(Cpu *c) {   // Result StartInputStream(InputStreamType *)
    int s = mic_slot(c);
    audin_log("StartInputStream");
    if (s < 0) { ret_x(c, R_AUDIO_NO_INPUT); return; }
    plat_mutex_lock(&mic_lock);
    if (mic[s].state != MIC_STARTED) mic[s].state = MIC_STARTED, mic[s].since = plat_now_ns(), mic[s].taken = 0;
    plat_mutex_unlock(&mic_lock);
    ret_x(c, 0);
}
static void audin_stop(Cpu *c) {   // void StopInputStream(InputStreamType *)
    int s = mic_slot(c);
    if (s >= 0) mic[s].state = MIC_STOPPED;
}
static void audin_purge(Cpu *c) {   // void PurgeInputStream(InputStreamType *): what it holds is dropped
    int s = mic_slot(c);
    if (s < 0) return;
    plat_mutex_lock(&mic_lock);
    mic[s].taken += mic_filled(s);
    plat_mutex_unlock(&mic_lock);
}
static void audin_close(Cpu *c) {   // void CloseInputStream(InputStreamType *)
    int s = mic_slot(c);
    if (s < 0) return;
    mic[s].used = 0;
    *(uint64_t *)arg_p(c, 0) = 0;
}
static void audin_state(Cpu *c) {   // InputStreamState GetInputStreamState(const InputStreamType *): 0 closed
    int s = mic_slot(c);
    ret_x(c, s < 0 ? 0 : (uint64_t)mic[s].state);
}
static void audin_filled(Cpu *c) {   // int GetInputStreamFilledFrameCount(const InputStreamType *)
    int s = mic_slot(c);
    if (s < 0) { ret_x(c, 0); return; }
    plat_mutex_lock(&mic_lock);
    int64_t n = mic_filled(s);
    plat_mutex_unlock(&mic_lock);
    ret_x(c, (uint64_t)n);
}
static void audin_read(Cpu *c) {   // int ReadInputStream(InputStreamType *, void *buffer, int frames): silence
    int s = mic_slot(c);
    uint8_t *out = arg_p(c, 1);
    int want = (int)arg_x(c, 2);
    if (s < 0 || !out || want <= 0) { ret_x(c, 0); return; }
    plat_mutex_lock(&mic_lock);
    int64_t n = mic_filled(s);
    if (n > want) n = want;
    mic[s].taken += n;
    plat_mutex_unlock(&mic_lock);
    memset(out, 0, (size_t)n * (size_t)mic[s].frame_bytes);
    ret_x(c, (uint64_t)n);
}
// InputStreamStatistics GetInputStreamStatistics(InputStreamType *, bool reset): three 64-bit counts, returned in
// memory (x8)
static void audin_statistics(Cpu *c) {
    void *out = (void *)(uintptr_t)c->x[8];
    if (out) memset(out, 0, 24);
}

// ---- account: no Nintendo Account, so the network service account calls fail at once
static void acc_no_account(Cpu *c) { ret_x(c, R_NO_NINTENDO_ACCOUNT); }
static void acc_async_event(Cpu *c) {   // Result AsyncContext::GetSystemEvent(os::SystemEvent *): signalled
    uint64_t ev = arg_x(c, 1);
    if (!ev) {
        ret_x(c, R_NO_NINTENDO_ACCOUNT);
        return;
    }
    kern_bind_system_event(c, ev, 0);
    kern_signal_system_event(ev);
    ret_x(c, 0);
}

// ---- time: the network clock is the PC's
static void net_clock_now(Cpu *c) { ret_x(c, (uint64_t)(int64_t)time(NULL)); }   // time_point now(): POSIX seconds
static void clock_task_event(Cpu *c) {   // os::SystemEvent *GetFinishEvent(): signalled, the clock is ready
    static uint64_t ev;
    if (!ev) ev = bound_event(c, 1);
    ret_x(c, ev);
}

// Result GetStandardSteadyClockCurrentTimespec(timespec *): the game installs it as musl's CLOCK_MONOTONIC source
// (sdk+0xaf0f8, mode 3), so std::chrono::steady_clock::now reads it; through the time service it failed and libc++
// threw system_error. The host's monotonic clock: the steady clock only promises monotonic time from an arbitrary
// epoch.
static void steady_timespec(Cpu *c) {
    int64_t *ts = arg_p(c, 0);
    uint64_t ns = plat_now_ns();
    if (ts) ts[0] = (int64_t)(ns / 1000000000ull), ts[1] = (int64_t)(ns % 1000000000ull);
    ret_x(c, 0);
}

// Result Get{Standard,Adjustable}UserSystemClockCurrentTimespec(timespec *): musl's CLOCK_REALTIME sources
// (sdk+0xaf040 installs one), behind std::chrono::system_clock::now. The PC's UTC time.
static void user_timespec(Cpu *c) {
    int64_t *ts = arg_p(c, 0);
    struct timespec now;
    if (!timespec_get(&now, TIME_UTC)) now.tv_sec = time(NULL), now.tv_nsec = 0;
    if (ts) ts[0] = (int64_t)now.tv_sec, ts[1] = (int64_t)now.tv_nsec;
    ret_x(c, 0);
}

// ---- fs: the host's file system takes no allocator and keeps no per-thread context
static uint8_t fs_context[0x10];
static void fs_get_context(Cpu *c) {   // const FsContext *GetCurrentThreadFsContext()
    static uint64_t ctx;
    if (!ctx) ctx = (uint64_t)(uintptr_t)mem_kernel_alloc(sizeof fs_context, 16);
    ret_x(c, ctx);
}

HOST_OVERRIDE("_ZN2nn2vi21QueryDisplayHdrStatusEPbPNS0_15HdrOutputFormatEPtPKNS0_7DisplayE", vi_hdr_status);
HOST_OVERRIDE("_ZN2nn2vi25QueryDisplayHdrPaperWhiteEPtPKNS0_7DisplayE", vi_reference_white);
HOST_OVERRIDE("_ZN2nn2vi24QueryDisplayVrrSupportedEPbPKNS0_7DisplayE", vi_vrr_supported);
HOST_OVERRIDE("_ZN2nn2vi23SetDisplayVrrCapabilityEPNS0_7DisplayEb", h_ok);
HOST_OVERRIDE("_ZN2nn2vi30QueryDisplayCurrentRefreshRateEPNS0_11RefreshRateEPbPKNS0_7DisplayE", vi_refresh_rate);
HOST_OVERRIDE("_ZN2nn2vi36SetDisplay120HzRefreshRateCapabilityEPNS0_7DisplayEb", h_ok);
HOST_OVERRIDE("_ZN2nn2vi28SetLayerExtendedSwapIntervalEPNS0_5LayerEii", vi_ext_swap);
HOST_OVERRIDE("_ZN2nn2vi30ClearLayerExtendedSwapIntervalEPNS0_5LayerE", h_ok);
HOST_OVERRIDE("_ZN2nn2vi35SetLayerNextHdrContentGamutMetadataEPNS0_5LayerENS0_15HdrContentGamutE", h_ok);
HOST_OVERRIDE("_ZN2nn2vi39SetLayerNextHdrToSdrToneMappingMetadataEPNS0_5LayerENS0_14HdrToneMappingE", h_ok);
HOST_OVERRIDE("_ZN2nn2vi52SetLayerNextHdrExtendedContentLuminanceRangeMetadataEPNS0_5LayerEtt", h_ok);
HOST_OVERRIDE("_ZN2nn2vi52SetLayerNextHdrStandardContentLuminanceRangeMetadataEPNS0_5LayerEtt", h_ok);
HOST_OVERRIDE("_ZN2nn5audio25ListAudioOutputDeviceNameEPNS0_15AudioDeviceNameEi", audio_list);
HOST_OVERRIDE("_ZN2nn5audio30GetActiveAudioOutputDeviceNameEPNS0_15AudioDeviceNameE", audio_active);
HOST_OVERRIDE("_ZN2nn5audio41GetActiveAudioDeviceChannelCountForOutputEv", audio_channels);
HOST_OVERRIDE("_ZN2nn5audio26SetAudioDeviceOutputVolumeEPKNS0_15AudioDeviceNameEf", audio_set_volume);
HOST_OVERRIDE("_ZN2nn5audio26GetAudioDeviceOutputVolumeEPKNS0_15AudioDeviceNameE", audio_volume);
HOST_OVERRIDE("_ZN2nn5audio40SetDeviceOutputVolumeAutoTuneModeEnabledEb", audio_set_autotune);
HOST_OVERRIDE("_ZN2nn5audio39IsDeviceOutputVolumeAutoTuneModeEnabledEv", audio_autotune);
HOST_OVERRIDE("_ZN2nn5audio39AcquireAudioDeviceNotificationForOutputEPNS_2os11SystemEventE", audio_notification);
// SDK 22 lists the virtual input devices (the Switch 2's microphone routing) through auddmg:u at boot: one device of
// silence, and its "true raw" twin (audio input above)
HOST_OVERRIDE("_ZN2nn5audio23ListVirtualInputDevicesEPNS0_15AudioDeviceNameEi", audin_list);
HOST_OVERRIDE("_ZN2nn5audio26GetTrueRawAudioInputDeviceEPNS0_15AudioDeviceNameE", audin_true_raw);
HOST_OVERRIDE("_ZN2nn5audio15OpenInputStreamEPNS0_15InputStreamTypeEPvmRKNS0_15AudioDeviceNameERKNS0_12StreamFormatEi", audin_open);
HOST_OVERRIDE("_ZN2nn5audio15OpenInputStreamEPNS0_15InputStreamTypeEPNS_2os11SystemEventEPvmRKNS0_15AudioDeviceNameERKNS0_12StreamFormatEii",
              audin_open_event);
HOST_OVERRIDE("_ZN2nn5audio16StartInputStreamEPNS0_15InputStreamTypeE", audin_start);
HOST_OVERRIDE("_ZN2nn5audio15StopInputStreamEPNS0_15InputStreamTypeE", audin_stop);
HOST_OVERRIDE("_ZN2nn5audio16PurgeInputStreamEPNS0_15InputStreamTypeE", audin_purge);
HOST_OVERRIDE("_ZN2nn5audio16CloseInputStreamEPNS0_15InputStreamTypeE", audin_close);
HOST_OVERRIDE("_ZN2nn5audio19GetInputStreamStateEPKNS0_15InputStreamTypeE", audin_state);
HOST_OVERRIDE("_ZN2nn5audio30GetInputStreamFilledFrameCountEPKNS0_15InputStreamTypeE", audin_filled);
HOST_OVERRIDE("_ZN2nn5audio15ReadInputStreamEPNS0_15InputStreamTypeEPvi", audin_read);
HOST_OVERRIDE("_ZN2nn5audio24GetInputStreamStatisticsEPNS0_15InputStreamTypeEb", audin_statistics);
HOST_OVERRIDE("_ZN2nn3hid13GetJcmsHandleERKjRKNS0_8JcmsTypeE", jcms_handle);
HOST_OVERRIDE("_ZN2nn3hid9StartJcmsERKNS0_10JcmsHandleE", h_nothing);
HOST_OVERRIDE("_ZN2nn3hid8StopJcmsERKNS0_10JcmsHandleE", h_nothing);
HOST_OVERRIDE("_ZN2nn3hid12FinalizeJcmsEv", h_nothing);
HOST_OVERRIDE("_ZN2nn3hid14FinalizeNmouseEv", h_nothing);
// Switch 2 hid calls whose SDK path opens the hid service itself instead of the nn::hid::detail calls ov_input.c
// serves: the gyroscope drift mode is the host's (motion_fusion.c), there is no Joy-Con 2 ("Ounce") rumble device,
// and no sensor-fusion handles (MacNativeGames input.c GAME_NS2 / GAME_BOTW, MIT)
HOST_OVERRIDE("_ZN2nn3hid25SetGyroscopeZeroDriftModeERKNS0_19SixAxisSensorHandleERKNS0_22GyroscopeZeroDriftModeE",
              h_nothing);
HOST_OVERRIDE("_ZN2nn3hid29IsOunceVibrationDeviceMountedERKNS0_21VibrationDeviceHandleE", h_zero);
HOST_OVERRIDE("_ZN2nn3hid22GetSensorFusionHandlesEPNS0_18SensorFusionHandleEiRKjNS_4util10BitFlagSetILi32ENS0_12NpadStyleTagEEE",
              h_zero);
HOST_OVERRIDE("_ZN2nn3hid20FinalizeSensorFusionEv", h_nothing);
HOST_OVERRIDE("_ZN2nn3hid16GetNmouseHandlesEPNS0_12NmouseHandleEij", nmouse_handles);
HOST_OVERRIDE("_ZN2nn3hid16GetNmouseTypeSetEPNS_4util10BitFlagSetILi32ENS0_10NmouseTypeEEENS0_12NmouseHandleE", nmouse_types);
HOST_OVERRIDE("_ZN2nn2oe23GetCurrentGraphicsScaleEv", oe_graphics_scale);
HOST_OVERRIDE("_ZN2nn2oe42SetGraphicsScaleChangedNotificationEnabledEb", h_nothing);
HOST_OVERRIDE("_ZN2nn2oe17GetDisplayVersionEPNS0_14DisplayVersionE", oe_display_version);
HOST_OVERRIDE("_ZN2nn7account26GetNetworkServiceAccountIdEPNS0_23NetworkServiceAccountIdERKNS0_10UserHandleE", acc_no_account);
HOST_OVERRIDE("_ZN2nn7account36EnsureNetworkServiceAccountAvailableERKNS0_10UserHandleE", acc_no_account);
HOST_OVERRIDE("_ZN2nn7account37LoadNetworkServiceAccountIdTokenCacheEPmPcmRKNS0_10UserHandleE", acc_no_account);
HOST_OVERRIDE("_ZN2nn7account44EnsureNetworkServiceAccountIdTokenCacheAsyncEPNS0_12AsyncContextERKNS0_10UserHandleE", acc_no_account);
HOST_OVERRIDE("_ZN2nn7account12AsyncContextC1Ev", h_nothing);
HOST_OVERRIDE("_ZN2nn7account12AsyncContextD1Ev", h_nothing);
HOST_OVERRIDE("_ZN2nn7account12AsyncContext14GetSystemEventEPNS_2os11SystemEventE", acc_async_event);
HOST_OVERRIDE("_ZN2nn7account12AsyncContext9GetResultEv", acc_no_account);
HOST_OVERRIDE("_ZN2nn4time26StandardNetworkSystemClock3nowEv", net_clock_now);
HOST_OVERRIDE("_ZN2nn4time37GetStandardSteadyClockCurrentTimespecEP8timespec", steady_timespec);
HOST_OVERRIDE("_ZN2nn4time41GetStandardUserSystemClockCurrentTimespecEP8timespec", user_timespec);
HOST_OVERRIDE("_ZN2nn4time43GetAdjustableUserSystemClockCurrentTimespecEP8timespec", user_timespec);
HOST_OVERRIDE("_ZN2nn4time39AsyncEnsureNetworkClockAvailabilityTaskC1Ev", h_nothing);
HOST_OVERRIDE("_ZN2nn4time39AsyncEnsureNetworkClockAvailabilityTaskD1Ev", h_nothing);
HOST_OVERRIDE("_ZN2nn4time39AsyncEnsureNetworkClockAvailabilityTask10InitializeENS_2os14EventClearModeE", h_ok);
HOST_OVERRIDE("_ZN2nn4time39AsyncEnsureNetworkClockAvailabilityTask9StartTaskEv", h_ok);
HOST_OVERRIDE("_ZN2nn4time39AsyncEnsureNetworkClockAvailabilityTask9GetResultEv", h_ok);
HOST_OVERRIDE("_ZN2nn4time39AsyncEnsureNetworkClockAvailabilityTask14GetFinishEventEv", clock_task_event);
HOST_OVERRIDE("_ZN2nn2fs12SetAllocatorEPFPvmEPFvS1_mE", h_nothing);
HOST_OVERRIDE("_ZN2nn2fs25GetCurrentThreadFsContextEv", fs_get_context);
HOST_OVERRIDE("_ZN2nn2fs25SetCurrentThreadFsContextEPKNS0_9FsContextE", h_nothing);
HOST_OVERRIDE("_ZN2nn3aoc17CountAddOnContentEv", h_zero);   // add-on content is not served yet
// Result RequestLoadNxFont(void *buffer, size_t size) (SDK 22): asks the system to load the Switch 1 shared fonts into
// the game's buffer. The host serves every shared font from its own memory (ov_system.c GetSharedFontAddress), so
// there is nothing to load.
HOST_OVERRIDE("_ZN2nn2pl17RequestLoadNxFontEPvm", h_ok);
#endif
