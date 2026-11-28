// rekarina.ini: file-backed defaults for the REKARINA_* environment variables (docs/controls.md, rekarina.ini.example).
// Format: "key = value" lines, '#' or ';' at the start of a line begins a comment, [sections] only group. A key is a
// variable name without "REKARINA_" (an optional REKARINA_ prefix is accepted), case-insensitive. A value is the rest of the
// line, trimmed, optionally in double quotes; an empty value counts as unset. The environment wins over the file.
//
// The prefix was TOTK_ until 2026-11-15 (the runtime came from the TotK recompiler). The old one still works everywhere:
// a name with either prefix is the same setting (code generated before the rename asks for TOTK_ names), the
// environment is read as REKARINA_<key>, then TOTK_<key>, and an ini line may carry either prefix.
//
// Live settings: cfg_set changes a value in memory (readers see the new pointer; the old string is kept, so a pointer
// cfg_get handed out stays valid) and cfg_save writes only the changed keys back, keeping the rest of the file.
// Readers take no lock: an entry is complete before n_entries counts it, and value pointers are swapped atomically.
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "cfg.h"
#include "host.h"
#include "motion_vectors.h"
#include "platform/platform.h"

#define PREFIX "REKARINA_"
#define PREFIX_LEN 9
#define OLD_PREFIX "TOTK_"
#define OLD_PREFIX_LEN 5
#define CFG_MAX_KEYS 512

typedef struct { char *key, *value; int dirty, removed; } CfgEntry;   // key upper-case, without the prefix; value NULL after cfg_reset

static CfgEntry entries[CFG_MAX_KEYS];
static int n_entries;
static unsigned version;
static PlatMutex set_lock = PLAT_MUTEX_INIT;   // cfg_set, cfg_reset, cfg_save
static int save_ok = 1;                         // the last cfg_save_to wrote (or had nothing to write)

struct CfgRenderState {
    unsigned readers, revision;
    int count;
    CfgRenderValues typed;
    char *projection_mode, *projection_preset;
    struct { const char *key, *value; } values[CFG_MAX_KEYS];
};
static CfgRenderState render_states[3];
static int render_index = -1;
static int frame_index = -1;
static _Thread_local const CfgRenderState *render_state;
static int boot_skin_motion = 1;

static CfgTaa taa_value(void) {
    const char *v = cfg_value(PREFIX "TAA");
    return !strcmp(v, "fsr3") ? CFG_TAA_FSR3 : !strcmp(v, "fsr1") ? CFG_TAA_FSR1 :
           !strcmp(v, "bilinear") ? CFG_TAA_BILINEAR : CFG_TAA_DLSS;
}

CfgTaa cfg_taa(void) { return render_state ? render_state->typed.taa : taa_value(); }

#ifdef GAME_OOT
int cfg_scene_sharpness(int stock_scaler) {
    if (stock_scaler) return 0;
    const CfgRenderValues *settings = cfg_render_values();
    return settings ? settings->sharpness : cfg_int(PREFIX "SHARPNESS");
}
#endif

void cfg_render_publish_present(unsigned (*resolve_consumers)(CfgTaa taa, int wanted)) {
    int current = __atomic_load_n(&render_index, __ATOMIC_SEQ_CST);
    if (!resolve_consumers && current >= 0 && __atomic_load_n(&render_states[current].revision, __ATOMIC_RELAXED) ==
        __atomic_load_n(&version, __ATOMIC_ACQUIRE)) return;
    plat_mutex_lock(&set_lock);
    const CfgRenderState *previous = cfg_render_use(NULL);
    current = __atomic_load_n(&render_index, __ATOMIC_SEQ_CST);
    const unsigned revision = __atomic_load_n(&version, __ATOMIC_RELAXED);
    const CfgTaa taa = taa_value();
    const int wanted = cfg_bool(PREFIX "DLAA");
    const unsigned consumers = resolve_consumers ? resolve_consumers(taa, wanted) : 0;
    if (current < 0 || render_states[current].revision != revision || render_states[current].typed.consumers != consumers) {
        for (int i = 0; i < 3; ++i) {
            CfgRenderState *next = &render_states[i];
            if (i == current || __atomic_load_n(&next->readers, __ATOMIC_SEQ_CST)) continue;
            __atomic_store_n(&next->revision, revision, __ATOMIC_RELAXED);
            next->count = n_entries;
            for (int j = 0; j < n_entries; ++j) {
                next->values[j].key = entries[j].key;
                next->values[j].value = entries[j].value;
            }
            CfgRenderValues *typed = &next->typed;
            *typed = (CfgRenderValues){.dlaa = wanted, .sharpness = cfg_int(PREFIX "SHARPNESS"), .taa = taa,
                                      .consumers = consumers, .motion_reverse_only = cfg_bool(PREFIX "DLAA_REVERSE_ONLY")};
            const char *motion = cfg_value(PREFIX "MOTION_VECTORS");
            typed->motion_mode = !rt_knob("DLAA_MV", "1") || !strcmp(motion, "off") ? MOTION_VECTORS_OFF :
                !strcmp(motion, "camera") ? MOTION_VECTORS_CAMERA :
                !strcmp(motion, "models") ? MOTION_VECTORS_MODELS : MOTION_VECTORS_HYBRID;
            typed->motion_models = boot_skin_motion && motion_vectors_models(typed->motion_mode) &&
                ((consumers & CFG_RENDER_FRAMEGEN_MOTION) || ((consumers & CFG_RENDER_UPSCALER) && taa != CFG_TAA_FSR1));
#ifdef GAME_OOT
            if (!strcmp(cfg_value(PREFIX "HDR"), "1")) {
                if (strcmp(cfg_value(PREFIX "HDR_TONEMAP"), "vanilla")) {
                    typed->renodx_flags |= CFG_HDR_RENODX_SCENE;
                } else if (cfg_double(PREFIX "HDR_BLOOM") != 50.0 || cfg_double(PREFIX "HDR_LOCAL_EXPOSURE") != 50.0) {
                    typed->renodx_flags |= CFG_HDR_RENODX_EFFECTS;
                }
            }
#endif
            free(next->projection_mode);
            free(next->projection_preset);
            next->projection_mode = strdup(cfg_value(PREFIX "DLSS"));
            next->projection_preset = strdup(cfg_value(PREFIX "DLSS_PRESET"));
            if (!next->projection_mode || !next->projection_preset)
                rt_fatal("cfg: could not copy the render snapshot's DLSS settings");
            const char *fsr = cfg_value(PREFIX "FSR_VERSION");
            if (strlen(fsr) < sizeof typed->fsr_version) strcpy(typed->fsr_version, fsr);
            const CfgRenderState *old = current < 0 ? NULL : &render_states[current];
            typed->projection_revision = old ? old->typed.projection_revision +
                (old->typed.dlaa != typed->dlaa || strcmp(old->projection_mode, next->projection_mode) ||
                 old->typed.taa != typed->taa || strcmp(old->projection_preset, next->projection_preset) ||
                 ((old->typed.consumers ^ consumers) & CFG_RENDER_FSR_FALLBACK) ||
                 ((typed->taa == CFG_TAA_FSR3 || (consumers & CFG_RENDER_FSR_FALLBACK)) &&
                  strcmp(old->typed.fsr_version, typed->fsr_version))) : 1;
            __atomic_store_n(&render_index, i, __ATOMIC_SEQ_CST);
            break;
        }
    }
    cfg_render_use(previous);
    plat_mutex_unlock(&set_lock);
}

void cfg_render_publish(void) { cfg_render_publish_present(NULL); }

static const CfgRenderState *render_acquire(const int *index) {
    for (;;) {
        const int i = __atomic_load_n(index, __ATOMIC_SEQ_CST);
        if (i < 0) return NULL;
        CfgRenderState *state = &render_states[i];
        __atomic_add_fetch(&state->readers, 1, __ATOMIC_SEQ_CST);
        // A slot can be retired between loading the index and pinning it. Don't read its payload until rechecked.
        if (i == __atomic_load_n(index, __ATOMIC_SEQ_CST)) return state;
        cfg_render_release(state);
    }
}

const CfgRenderState *cfg_render_acquire(void) { return render_acquire(&render_index); }

const CfgRenderState *cfg_render_frame_acquire(void) {
    const CfgRenderState *state = render_acquire(&frame_index);
    return state ? state : cfg_render_acquire();
}

void cfg_render_frame_begin(void) {
    const CfgRenderState *state = cfg_render_acquire();
    if (!state) return;
    int old = __atomic_exchange_n(&frame_index, (int)(state - render_states), __ATOMIC_SEQ_CST);
    if (old >= 0) cfg_render_release(&render_states[old]);
}

const CfgRenderState *cfg_render_retain(const CfgRenderState *state) {
    if (state) __atomic_add_fetch(&render_states[state - render_states].readers, 1, __ATOMIC_SEQ_CST);
    return state;
}

const CfgRenderValues *cfg_render_values_of(const CfgRenderState *state) { return state ? &state->typed : NULL; }
const CfgRenderValues *cfg_render_values(void) { return cfg_render_values_of(render_state); }

void cfg_render_release(const CfgRenderState *state) {
    if (state) __atomic_sub_fetch(&render_states[state - render_states].readers, 1, __ATOMIC_SEQ_CST);
}

const CfgRenderState *cfg_render_current(void) { return render_state; }

const CfgRenderState *cfg_render_use(const CfgRenderState *state) {
    const CfgRenderState *previous = render_state;
    render_state = state;
    return previous;
}

// Every name the runtime reads (and the patch sets in generated/), without the prefix. A file key outside this list is
// logged as unknown. A new variable goes here too, or its ini key logs a harmless "unknown key" line.
static const char *const known[] = {
    "AB", "AMIIBO_DIR", "ANISO", "ARRIVE_HOLD", "ARRIVE_HOLD_MS", "ARRIVE_POOL_MS", "TPOOL_CHECK_EVERY", "APP_POOL", "ASPECT", "ASPECT_PATCH", "ASSUME_FOCUS", "ASTC", "ASTC_BC", "ASTC_TIME", "TEX_CACHE", "TEX_CACHE_MB", "ASYNC_PREEMPT", "AUDIO", "AUDIO_CHANNELS", "AUDIO_OFF", "AUDIO_PREFILL_MS", "AUDIO_RAMP",
    "BACKGROUND_WINDOW", "BIND_A", "BIND_B", "BIND_DOWN", "BIND_L", "BIND_LCLICK", "BIND_LEFT", "BIND_LSTICK_DOWN", "BIND_LSTICK_LEFT",
    "BIND_LSTICK_RIGHT", "BIND_LSTICK_UP", "BIND_MINUS",
    "BIND_PLUS", "BIND_R", "BIND_RCLICK", "BIND_RIGHT", "BIND_RSTICK_DOWN", "BIND_RSTICK_LEFT", "BIND_RSTICK_RIGHT", "BIND_RSTICK_UP",
    "BIND_SHAKE",
    "BIND_SKIP", "BIND_STATS", "BIND_UP", "BIND_X", "BIND_Y",
    "BIND_ZL", "BIND_ZR", "BLOCK_VIEWS_3D", "CLOTH_REALTIME", "COLOR_BRIGHTNESS", "COLOR_CONTRAST", "COLOR_GAMMA",
    "COLOR_PRESET", "COLOR_SATURATION", "CONTACT_DEBUG", "CONTACT_SHADOWS", "CORE_DIAG", "CRIT", "CRIT_OUT",
    "DAYTIME", "DEFER_GRACE", "POOL_QUARANTINE", "DESC_TEMPLATE", "DEVICE_PROFILE", "DISCORD_APP_ID", "DISCORD_IMAGE", "DISCORD_RPC", "DLAA", "DLAA_JITTER", "DLAA_JITTER_TEST", "DLAA_MV", "DLAA_MASKDEBUG", "DLAA_MVDEBUG", "DLAA_OBJMV", "DLAA_PJITTER", "DLAA_PJ_YS", "DLAA_PRE_BLOOM", "DLAA_REVERSE_ONLY", "DLAA_SURVEY", "DLAA_TARGET_KEEP", "DLAA_TIMES", "DLSS", "DLSS_INTERFACE_RECOMPOSITION", "DLSS_PREBLOOM", "DLSSG_UNAVAILABLE", "DLSS_PRESET", "DLSS_REACTIVE", "DUMP", "DUMP_BUF", "DUMP_DIR",
    "DUMP_LOG", "DUMP_RETURN", "EXEFS", "FEEDBACK", "FIX_C0", "FIX_ZERO", "FLIGHT", "FLIGHT_AT", "FLIGHT_DIR", "FLIGHT_FRAMES",
    "FLIGHT_SPIKE_MS", "FLIGHT_THUMB", "FLIGHT_THUMBS", "FLUSH_SPAN", "FLUSH_UPLOAD", "TEX_EVICT", "FOLIAGE_SHADOW_CULL", "FOLIAGE_SHADOW_VRS", "FOV", "FOV_PATCH", "FPS60", "FPS_LIMIT", "FRAMETIMES", "FREEZE_IDLE", "FREEZE_WHOLE", "FREE_CORES",
    "FAST_TEXT", "FULLSCREEN", "FXAA", "GPL", "GPL_DRAW_BUILD", "GPU", "GPU_INDEX", "GPU_NAME", "HDR",
    "HDR_ADAPTATION_TARGET", "HDR_ADAPTATION_MIN_STOPS", "HDR_ADAPTATION_MAX_STOPS", "HDR_BLOOM",
    "HDR_CONTRAST", "HDR_EXPOSURE", "HDR_GAME_NITS", "HDR_GAMMA",
    "HDR_HIGHLIGHTS", "HDR_HUE", "HDR_HUE_CLIP", "HDR_LIVE", "HDR_LOCAL_EXPOSURE", "HDR_MATCH_SLOPE", "HDR_MIDGRAY", "HDR_PEAK_NITS",
    "HDR_SATURATION", "HDR_SHADOWS", "HDR_TONEMAP", "HDR_TONE_SLOPE", "HDR_UI_EOTF", "HDR_UI_NITS", "HIGHFPS_FIX",
    "HUD_LAYOUT", "HUD_HEALTH_X", "HUD_HEALTH_Y", "HUD_HEALTH_SCALE",
    "HUD_BUTTONS_X", "HUD_BUTTONS_Y", "HUD_BUTTONS_SCALE",
    "HUD_MINIMAP_X", "HUD_MINIMAP_Y", "HUD_MINIMAP_SCALE", "HUD_MAX_RATIO", "WIDE_UI_LOG",
    "INPUT", "INPUT_RECORD", "INPUT_SCRIPT", "LANG", "LIB_PREBUILD",
    "LIB_PREBUILD_KEEP", "LIB_PREBUILD_THREADS", "LOCALE", "LOD_GUARD", "LSTICK_DEADZONE", "MEMORY", "LINUX_GUEST_MEMORY", "SHADOW_PAGE_PRE", "SHADOW_GMEM_PRE", "MENU_LANG", "MENU_TEST", "MESH_ASYNC",
    "MESH_GPL", "MESH_WAIT_MS", "MESH_KEYS_DROP", "MESH_LIB_PREPARE", "MESH_COLD", "MESH_ALT_ONLY", "MESH_FS_AHEAD",
    "MENU_CURSOR", "MENU_CURSOR_LOG", "MENU_TEST_DIR", "MENU_TEST_SET", "MINIMAL_LAYOUT", "MIRROR_GAP", "MODS_DIR", "MODS_DISABLED", "MODS_ORDER", "MODS_REBUILD", "MOTION", "PAD_FOLLOW", "PAD_DRIVER", "CURSOR_HIDE", "BG_MUTE", "MENU_SCRIPT", "KBM_OFF", "MENU_GUIDE", "SCREENSHOT_KEY", "SCREENSHOT_OVERLAY", "TEST_FAULT", "TEST_RAM_MB", "TEST_RESIZE", "PIPE_AUTOSAVE", "REPORT_NO_OPEN", "NO_CRASH_BOX", "MOUSE_MODE", "MOUSE_DIRECT_PATCH", "MOUSE_SENS", "MOUSE_SENS_Y", "MOUSE_INVERT_Y", "MOUSE_RAW", "MOVIE_HOST", "MOVIE_PACE",
    "MOTION_GUARD_DEGREES", "MOTION_VECTORS", "SKIN_MOTION",
    "MVOBJ_ALL", "MVOBJ_BANKS", "MVOBJ_EQUAL", "MVOBJ_EXACT", "MVOBJ_EXEC", "MVOBJ_NOC4", "MVOBJ_ONLY", "MVOBJ_TIMES", "MVPREV_CHECK", "MVPREV_GROW", "MVPREV_SKIP", "MVPREV_TRUST", "MV_CHECK",
    "MV_SURVEY",
    "ONE_PER_CORE", "OPERATION_MODE", "OUTPUT_RES", "PAD_LAYOUT", "PASS_CRUMBS", "PATCH_OUTPUT", "PATCH_SCENE", "PATCH_SCENE_PROJECTION", "PATCH_TRACKED_MODELS", "PATCH_WIDE_UI", "PCSS", "PCSS_K", "PERF_OVERLAY", "VRAM_NOTICE", "PERF_SLOW_MS", "PIPELINE_CACHE", "PIPELINE_CACHE_LOAD", "WARM_APP_CACHE", "WARM_FRESH_CACHE", "WARM_STOP", "ENTRY_LIBRARIES", "ENTRY_TEXTURES", "SHADER_CACHE_FLUSH", "PIPELINE_CACHE_RESET_MB", "WARM_RECORD_STATE", "PIPE_SAVE_SAME", "PIPE_MEMO", "PIPE_PREBUILD", "PIPE_SKIP", "PIPE_SYNC", "PIPE_WAIT",
    "PREWARM_XLATE", "PREPARE_DRIVER", "PREPARE_TEXTURES_CHECK", "PREPARE_TEXTURES_LEVELS", "PREP_SCREEN", "PREP_SHOT", "PREP_SHOT_AT", "PROBE_MOVIE", "PROBE_MOVIE_SHOT", "PROFILE", "PROFILE_CHAINS", "PROFILE_DIR", "PROFILE_FRAMES",
    "PROFILE_MAIN", "PROFILE_RENDER", "PROMPTS", "PROMPT_GLYPHS", "PUSH_SKIP", "PUSH_STAGE", "RDOC", "RDOC_DIR", "RDOC_DLL", "REALTIME", "REC_END", "REEXEC",
    "RENDER_SCALE", "REPLAY_ASYNC",
    "REPLAY_CHECK", "RESOLVE_CHECK", "RESTART_WAIT", "ROMFS", "ROMFS_INDEX", "ROMFS_MMAP", "ROMFS_MODS", "ROMFS_PATCHES",
    "REPLAY_THREADS", "ROMFS_PROOF_DIR", "ROMFS_VERIFY", "RSTICK_DEADZONE", "RUMBLE", "RUMBLE_CURVE", "RUMBLE_GAP_MS", "RUMBLE_STRENGTH", "RUMBLE_TRIGGERS", "VIBRATION", "DUALSENSE_EXTRAS", "RH_TEST", "SECTIONS", "SHADER_CACHE",
    "SHADER_MEMO", "SHADOW_RANGE", "SHADOW_RES", "SHADOW_SCALE", "SHARED_FONTS", "SHARPNESS", "SHOTS", "SHOTS_DIR",
#ifdef GAME_OOT
    "TEXTURE_LOD_STRENGTH",
#endif
    "SHOTS_FULL", "SPEED_PROBE", "SPIN", "SPIN_HANDOFF_US", "TEX_OVER", "FOLD_MAP", "FOLD_SUBMIT_SLOT", "FOLD_SLEEP", "FOLD_AP_SPIN", "AP_SPIN_US", "SUBMIT_PRIO", "PAR_PRIO", "SPLIT_DRAWS", "SPLIT_IN_PASS", "SPLIT_US", "STICK_DEADZONE", "STICK_SENS", "STICK_SENS_PATCH",
    "SWKBD_ANSWER",
    "FSR_VERSION", "CS_KEEP", "MOVIE_UNET", "MESH_INDICES", "MESH_TASK", "VRAM_BUDGET_MB", "VMA_DUMP", "SNAP_RING_MB", "TEX_EVICT_CLASS", "TPOOL_HIDE", "POOLTAB", "VMA_BLOCK_MB", "TEX_SUBALLOC_KB",
    "SWKBD_TEST", "TEST_PAD", "UPDATE_CHECK", "TEST_PREPARE_SHADERS", "TEST_STALL", "FIXED_SCENE", "FRAMEGEN", "REFLEX", "REFLEX_LOG", "GAME_ASPECT", "GAME_RES", "SCENE_SCALE", "SHADOW_COMPARE", "TAA", "TEX_ALIAS", "TEX_MEMO", "TEX_QUALITY", "TEX_SAME_REG", "DLAA_WARM", "DLAA_SHARPEN", "TPOOL_PUMP_MS", "MOVIE_FPS", "TEX_VRAM_PRESSURE", "TIMER_1MS", "PROCESS_PRIORITY", "RENDER_CORE", "GPU_LOAD_ZERO", "GPU_PASSTIME", "OQLOG", "PRED_ZERO", "RES_GUARD", "TPOOL", "TPOOL_MS", "TPOOL_REGBITS", "PAR_UB_JOIN", "HEAP_CHECK", "EVICT_QUIET", "USNAP_HELD_LOCK", "TPOOL_OLD_MS", "EVICT_HOLD_MS", "PRESSURE_KEEP_USED", "TRACE", "TXD_TRANSPOSE", "UBO_SNAPSHOT", "UPSCALER", "VK_DYN_A2C", "VK_DYN_VERTEX", "VK_NO_FLOAT64",
    "VK_PRESENT", "VK_RECORDER", "VK_SSBO_ALIGN", "VK_SUBGROUP", "VK_SUBGROUP_PIN", "VK_VALIDATION", "VOICE_LANG", "VOLUME", "VRAM_HEADROOM", "VS_OUTS",
    "WALK_KEY", "WALK_SCALE", "STICK_DASH", "DASH_KEY", "WB_EXACT", "WB_FENCE_LIMIT", "WB_MERGE", "WINDOW_SIZE", "XLATE", "XLATE_ID", "XLATE_KEEP", "XLATE_PROCS",
    "ZSTD_CHECK", "ZSTD_NATIVE",
};

// The rekarina.ini section a key cfg_set may write goes to when the file lacks it (rekarina.ini.example's sections).
static const struct { const char *key, *section; } sections[] = {
    {"FPS_LIMIT", "frame rate"}, {"FPS60", "frame rate"},
    {"REALTIME", "frame rate"}, {"CLOTH_REALTIME", "frame rate"}, {"HIGHFPS_FIX", "frame rate"},
    {"FULLSCREEN", "display"}, {"WINDOW_SIZE", "display"}, {"PERF_OVERLAY", "display"}, {"VRAM_NOTICE", "display"}, {"VK_PRESENT", "display"},
    {"GPU_INDEX", "display"}, {"OUTPUT_RES", "display"}, {"RENDER_SCALE", "display"}, {"UPSCALER", "display"},
    {"SHARPNESS", "display"}, {"TEXTURE_LOD_STRENGTH", "display"}, {"ASPECT", "display"}, {"FOV", "display"}, {"ANISO", "display"},
    {"COLOR_PRESET", "display"},
    {"COLOR_BRIGHTNESS", "display"}, {"COLOR_CONTRAST", "display"}, {"COLOR_SATURATION", "display"},
    {"COLOR_GAMMA", "display"}, {"HDR", "display"}, {"HDR_LIVE", "display"}, {"HDR_PEAK_NITS", "display"},
    {"HDR_GAME_NITS", "display"}, {"HDR_UI_NITS", "display"}, {"HDR_GAMMA", "display"}, {"HDR_UI_EOTF", "display"},
    {"HDR_TONEMAP", "display"}, {"SDR_TONEMAP", "display"},
    {"HDR_MIDGRAY", "display"}, {"HDR_MATCH_SLOPE", "display"}, {"HDR_TONE_SLOPE", "display"}, {"HDR_HUE_CLIP", "display"},
    {"HDR_EXPOSURE", "display"}, {"HDR_HIGHLIGHTS", "display"}, {"HDR_SHADOWS", "display"}, {"HDR_CONTRAST", "display"},
    {"HDR_SATURATION", "display"}, {"HDR_HUE", "display"}, {"HDR_BLOOM", "display"}, {"SHADOW_RES", "display"},
    {"HDR_LOCAL_EXPOSURE", "display"},
    {"HDR_ADAPTATION_TARGET", "display"}, {"HDR_ADAPTATION_MIN_STOPS", "display"}, {"HDR_ADAPTATION_MAX_STOPS", "display"},
    {"DLAA", "display"}, {"DLSS", "display"}, {"DLSS_PRESET", "display"}, {"FSR_VERSION", "display"}, {"FXAA", "display"},
    {"TAA", "display"}, {"FRAMEGEN", "display"}, {"SKIN_MOTION", "display"}, {"REFLEX", "display"}, {"GAME_ASPECT", "display"}, {"GAME_RES", "display"},
    {"HUD_LAYOUT", "display"}, {"HUD_HEALTH_X", "display"}, {"HUD_HEALTH_Y", "display"}, {"HUD_HEALTH_SCALE", "display"},
    {"HUD_BUTTONS_X", "display"}, {"HUD_BUTTONS_Y", "display"}, {"HUD_BUTTONS_SCALE", "display"},
    {"HUD_MINIMAP_X", "display"}, {"HUD_MINIMAP_Y", "display"}, {"HUD_MINIMAP_SCALE", "display"},
    {"DEVICE_PROFILE", "display"}, {"ASSUME_FOCUS", "display"},
    {"UPDATE_CHECK", "updates"}, {"DISCORD_RPC", "discord"}, {"DISCORD_APP_ID", "discord"}, {"DISCORD_IMAGE", "discord"},
    {"SHADOW_SCALE", "display"}, {"SHADOW_RANGE", "display"}, {"CONTACT_SHADOWS", "display"},
    {"PCSS", "display"},
    {"MODS_DIR", "mods"}, {"MODS_ORDER", "mods"}, {"MODS_DISABLED", "mods"}, {"MODS_REBUILD", "mods"},
    {"MENU_LANG", "language"}, {"LANG", "language"}, {"VOICE_LANG", "language"}, {"FAST_TEXT", "language"}, {"OPERATION_MODE", "language"},
    {"VOLUME", "audio"}, {"AUDIO_CHANNELS", "audio"}, {"BG_MUTE", "audio"},
    {"MOTION", "input"}, {"VIBRATION", "input"}, {"DUALSENSE_EXTRAS", "input"}, {"RUMBLE_STRENGTH", "input"}, {"RUMBLE_CURVE", "input"}, {"RUMBLE_TRIGGERS", "input"},
    {"PAD_LAYOUT", "input"}, {"PROMPTS", "input"}, {"PROMPT_GLYPHS", "input"}, {"STICK_DEADZONE", "input"}, {"LSTICK_DEADZONE", "input"}, {"RSTICK_DEADZONE", "input"},
    {"STICK_SENS", "input"},
    {"MENU_CURSOR", "input"}, {"PAD_FOLLOW", "input"}, {"PAD_DRIVER", "input"}, {"KBM_OFF", "input"}, {"MENU_GUIDE", "input"}, {"SCREENSHOT_KEY", "input"}, {"SCREENSHOT_OVERLAY", "input"}, {"CURSOR_HIDE", "input"}, {"MOUSE_MODE", "input"}, {"MOUSE_SENS", "input"}, {"MOUSE_SENS_Y", "input"},
    {"MOUSE_INVERT_Y", "input"}, {"MOUSE_RAW", "input"}, {"WALK_KEY", "input"}, {"WALK_SCALE", "input"}, {"STICK_DASH", "input"}, {"DASH_KEY", "input"}, {"INPUT_RECORD", "input"},
    {"BIND_A", "input"}, {"BIND_B", "input"}, {"BIND_X", "input"}, {"BIND_Y", "input"}, {"BIND_LCLICK", "input"},
    {"BIND_RCLICK", "input"}, {"BIND_L", "input"}, {"BIND_R", "input"}, {"BIND_ZL", "input"}, {"BIND_ZR", "input"},
    {"BIND_PLUS", "input"}, {"BIND_MINUS", "input"}, {"BIND_LEFT", "input"}, {"BIND_UP", "input"},
    {"BIND_RIGHT", "input"}, {"BIND_DOWN", "input"}, {"BIND_LSTICK_LEFT", "input"}, {"BIND_LSTICK_UP", "input"},
    {"BIND_LSTICK_RIGHT", "input"}, {"BIND_LSTICK_DOWN", "input"}, {"BIND_RSTICK_LEFT", "input"},
    {"BIND_RSTICK_UP", "input"}, {"BIND_RSTICK_RIGHT", "input"}, {"BIND_RSTICK_DOWN", "input"}, {"BIND_SHAKE", "input"},
    {"ASYNC_PREEMPT", "performance"}, {"REPLAY_ASYNC", "performance"}, {"FREE_CORES", "performance"},
    {"TIMER_1MS", "performance"}, {"PROCESS_PRIORITY", "performance"}, {"PIPE_WAIT", "performance"}, {"PIPE_SKIP", "performance"},
    {"LIB_PREBUILD", "performance"}, {"LIB_PREBUILD_THREADS", "performance"}, {"PIPE_PREBUILD", "performance"},
    {"ZSTD_NATIVE", "performance"}, {"ZSTD_CHECK", "performance"},
    {"PREWARM_XLATE", "performance"}, {"PREP_SCREEN", "performance"}, {"GPL", "performance"}, {"ASTC", "performance"}, {"MEMORY", "performance"},
    {"TEX_ALIAS", "performance"}, {"TEX_VRAM_PRESSURE", "performance"}, {"TEX_QUALITY", "performance"}, {"ASTC_BC", "performance"}, {"TEX_CACHE", "performance"}, {"TEX_CACHE_MB", "performance"}, {"LOD_GUARD", "performance"}, {"FIX_C0", "performance"}, {"FIX_ZERO", "performance"},
    {"BIND_SKIP", "performance"}, {"SHADER_MEMO", "performance"}, {"PIPE_MEMO", "performance"}, {"TEX_MEMO", "performance"}, {"DESC_TEMPLATE", "performance"},
    {"PUSH_SKIP", "performance"}, {"SPLIT_DRAWS", "performance"}, {"REPLAY_THREADS", "performance"}, {"SPLIT_IN_PASS", "performance"}, {"SPLIT_US", "performance"}, {"VK_VALIDATION", "performance"},
    {"VK_DYN_VERTEX", "performance"}, {"VK_DYN_A2C", "performance"}, {"VK_SUBGROUP_PIN", "performance"},
    {"TRACE", "diagnostics"}, {"FLIGHT", "diagnostics"}, {"PASS_CRUMBS", "diagnostics"}, {"FLIGHT_SPIKE_MS", "diagnostics"},
    {"WB_FENCE_LIMIT", "diagnostics"}, {"WB_MERGE", "diagnostics"}, {"WB_EXACT", "diagnostics"},
    {"MV_SURVEY", "diagnostics"}, {"MOTION_VECTORS", "diagnostics"}, {"DLAA_REVERSE_ONLY", "diagnostics"}, {"DLSS_INTERFACE_RECOMPOSITION", "diagnostics"},
};

// The settings the menu shows: the value in effect when the key is unset, and what the key accepts. The only place a
// default lives: consumers read cfg_value / cfg_int / cfg_double / cfg_bool. A row with an empty default has no fixed
// value (it depends on the machine); its consumer reads cfg_get and keeps an "unset" branch.
#define LIVE 1
#define RESTART 0
static const CfgMeta metas[] = {
    // Screen
    {"FULLSCREEN", "0", "0|1", LIVE}, {"WINDOW_SIZE", "1920x1080", NULL, LIVE},
    {"VK_PRESENT", "mailbox", "mailbox|fifo|immediate", LIVE}, {"GPU_INDEX", "", NULL, RESTART},
    {"PERF_OVERLAY", "off", "off|fps|full", LIVE},
    {"VRAM_NOTICE", "0", "0|1", LIVE},   // the low video memory notice at the top of the picture (ui/menu.cpp)
#ifdef GAME_OOT   // OoT: the scene's share of the picture is read on every present (frame_rate.c scene_share)
    {"OUTPUT_RES", "1920x1080", NULL, RESTART}, {"RENDER_SCALE", "100", "50|55|60|65|67|70|75|80|85|90|95|100", LIVE},
#else
    {"OUTPUT_RES", "1920x1080", NULL, RESTART}, {"RENDER_SCALE", "game", "game|50|67|75|100|125|150|200", RESTART},
#endif
    {"UPSCALER", "fsr", "fsr|bilinear", LIVE},
    {"SHARPNESS", "80", "0..100", LIVE},
#ifdef GAME_OOT
    {"TEXTURE_LOD_STRENGTH", "100", "0..100", LIVE},
#endif
#ifdef GAME_OOT
    {"DLAA", "0", "0|1", LIVE},
#else
    {"DLAA", "0", "0|1", RESTART},
#endif
    {"DLSS", "dlaa", "dlaa|quality|balanced|performance|ultra_performance", LIVE},
    {"DLSS_PRESET", "default", "default|k|m|l|j|e", LIVE},   // vk_dlaa.c preset_value
    {"FSR_VERSION", "auto", "auto|4|3", LIVE},               // vk_ffx.cpp fsr_version
    {"FXAA", "1", "0|1", LIVE},
    {"SKIN_MOTION", "1", "0|1", RESTART},
    {"PATCH_SCENE_PROJECTION", "1", "0|1", RESTART},
    {"PATCH_TRACKED_MODELS", "1", "0|1", RESTART},
    {"ASPECT", "fit", "fit|stretch", LIVE}, {"FOV", "100", "70..140", LIVE},
    {"ANISO", "game", "game|2|4|8|16", LIVE},
    // colour adjustment (vk_color.c); the preset is the menu's, the four values are what the present uses
    {"COLOR_PRESET", "default", "default|vivid|clarity|custom", LIVE}, {"COLOR_BRIGHTNESS", "0", "-30..30", LIVE},
    {"COLOR_CONTRAST", "100", "70..150", LIVE}, {"COLOR_SATURATION", "100", "50..200", LIVE},
    {"COLOR_GAMMA", "1", "0.8..1.2", LIVE},
    // OoT switches SDR/HDR live; the HDR tonemapper selects the compositor format.
#ifdef GAME_OOT
    {"HDR", "0", "0|1", LIVE},
    {"HDR_PEAK_NITS", "1000", "400..10000", LIVE}, {"HDR_GAME_NITS", "203", "48..500", LIVE},
    {"HDR_UI_NITS", "203", "48..500", LIVE}, {"HDR_GAMMA", "off", "off|2.2|2.4", LIVE},
    {"HDR_UI_EOTF", "2.2", "off|2.2|2.4", LIVE},
    {"HDR_TONEMAP", "psychov35", "psychov35|vanilla", LIVE},
    {"SDR_TONEMAP", "vanilla", "psychov35|vanilla", LIVE},
    {"HDR_MIDGRAY", "vanilla", "vanilla|render|adaptation", LIVE}, {"HDR_MATCH_SLOPE", "1", "0|1", LIVE},
    {"HDR_ADAPTATION_TARGET", "18", "1..100", LIVE},
    {"HDR_ADAPTATION_MIN_STOPS", "-4", "-8..0", LIVE}, {"HDR_ADAPTATION_MAX_STOPS", "2", "0..8", LIVE},
    {"HDR_TONE_SLOPE", "50", "0..100", LIVE}, {"HDR_HUE_CLIP", "50", "0..100", LIVE},
    {"HDR_EXPOSURE", "100", "0..800", LIVE}, {"HDR_HIGHLIGHTS", "50", "0..100", LIVE},
    {"HDR_SHADOWS", "50", "0..100", LIVE}, {"HDR_CONTRAST", "50", "0..100", LIVE},
    {"HDR_SATURATION", "50", "0..100", LIVE},
    {"HDR_BLOOM", "50", "0..100", LIVE},   // vk_hdr.c tonemap_splice: the game's bloom term times value / 50
    {"HDR_LOCAL_EXPOSURE", "50", "0..100", LIVE},   // vk_hdr.c tonemap_splice: local exposure correction in stops
#else
    {"HDR", "0", "0|1", RESTART}, {"HDR_PEAK_NITS", "1000", "400..10000", LIVE}, {"HDR_GAME_NITS", "203", "80..500", LIVE},
    {"HDR_UI_NITS", "203", "80..500", LIVE}, {"HDR_GAMMA", "off", "off|2.2|2.4", LIVE}, {"HDR_HUE", "80", "0..100", LIVE},
#endif
    {"DEVICE_PROFILE", "auto", "auto|desktop|deck|handheld", RESTART},
    {"SHADOW_RES", "game", "game|256|512|768|1024|1536|2048", RESTART},
#ifdef GAME_OOT
    {"SHADOW_SCALE", "auto", "auto|1|2|3|4", RESTART},   // vk_texture.c vk_shadow_scale: auto is 4 or 1 by the card
#else
    {"SHADOW_SCALE", "1", "1..4", RESTART},
#endif
    {"SHADOW_RANGE", "100", "50..400", RESTART}, {"CONTACT_SHADOWS", "0", "0..100", RESTART},
    {"PCSS", "0", "0..200", RESTART},
    // Game
#ifdef GAME_OOT   // OoT: the real-time step is its own (recomp/patches-oot-v111.txt); up to the monitor's rate
    {"FPS_LIMIT", "display", "30|40|45|60|90|120|144|240|display", LIVE}, {"FPS60", "1", "0|1", LIVE},
#else
    {"FPS_LIMIT", "60", "30|40|45|60|90|120|144|display", LIVE}, {"FPS60", "1", "0|1", LIVE},
#endif
    {"REALTIME", "1", "0|1", RESTART}, {"CLOTH_REALTIME", "1", "0|1", RESTART},
    {"MODS_REBUILD", "auto", "auto|off", RESTART},
    {"HIGHFPS_FIX", "1", "0|1", RESTART},
    {"LANG", "", "ja|en-US|fr|de|it|es|zh-CN|ko|nl|pt|ru|zh-TW|en-GB|fr-CA|es-419|zh-Hans|zh-Hant|pt-BR|pl", RESTART},
    {"OPERATION_MODE", "docked", "docked|handheld", RESTART},
    {"FAST_TEXT", "0", "0|1", RESTART},   // OoT: romfs-patches-fasttext at start (romfs.c)
    {"VOICE_LANG", "game", "game|en|ja|de|fr|it|es|es-419|pt-BR|fr-CA", LIVE},   // OoT: voice_lang.c + oot_fps.c voiceopt
    {"ASSUME_FOCUS", "1", "0|1", LIVE},   // read on every focus change (ov_system.c)
    {"HEAP_CHECK", "0", NULL, LIVE},      // test runs: the process heap checked every n presents (perf.c heap_check)
    {"DLSSG_UNAVAILABLE", "0", "0|1", RESTART},   // test runs: DLSS-G counted missing, AMD's takes over (vk_dlssg.c)
    // Sound, Controls, Menu
    {"VOLUME", "100", "0..100", LIVE}, {"AUDIO_CHANNELS", "stereo", "auto|stereo|5.1", RESTART},
    {"BG_MUTE", "0", "0|1", LIVE},   // silent while the window is in the background (audio_mixer.c)
    {"MOTION", "1", "0|1", LIVE}, {"VIBRATION", "auto", "auto|basic|off", LIVE}, {"DUALSENSE_EXTRAS", "1", "0|1", LIVE},
    {"RUMBLE_STRENGTH", "100", "0..300", LIVE}, {"RUMBLE_CURVE", "0.6", "0.2..1.5", LIVE},
    {"RUMBLE_TRIGGERS", "0", "0|1", LIVE}, {"PAD_LAYOUT", "auto", "auto|nintendo|xbox", LIVE},
    // read at start; the menu says when a change waits for a restart (ui/menu.cpp prompts_note) and where the sets
    // change in the running game (romfs_prompts_live)
#ifdef GAME_OOT   // OoT: auto, the set of the device the player used last (input.c input_prompts_wanted), switched live
    {"PROMPTS", "auto", "auto|nintendo|xbox|pc|playstation|rekarina", LIVE},
#else
    {"PROMPTS", "nintendo", "nintendo|xbox|pc|playstation|rekarina", LIVE},
#endif
#ifdef GAME_OOT   // patch set promptglyphs (runtime/oot_fps.c prompt_glyphs): a text laid out before a live prompt change
    // takes the new buttons; this row is its default (runtime/patches.c), so the patch file's own default (0) stays
    // and an update needs no new lift
    {"PROMPT_GLYPHS", "1", "0|1", RESTART},
#endif
    {"STICK_DEADZONE", "8", "0..50", LIVE},
#ifdef GAME_OOT   // OoT: the camera's turn rates, read as the camera asks for them (camera_speed.c oot_camera_rate)
    {"STICK_SENS", "1", "0.5..4", LIVE},
#else   // TotK: a camera pack made at start (camera_speed.c cam_pack_build)
    {"STICK_SENS", "1", "0.1..3", RESTART},
#endif
    {"LSTICK_DEADZONE", "", "0..50", LIVE}, {"RSTICK_DEADZONE", "", "0..50", LIVE},
    {"MOUSE_MODE", "direct", "stick|direct|off", LIVE},
    {"MOUSE_SENS", "1", "0..10", LIVE}, {"MENU_CURSOR", "1", "0|1", LIVE},
    // the mouse camera's vertical counts: times MOUSE_SENS_Y, turned over by MOUSE_INVERT_Y (input.c mouse_take_camera);
    // MOUSE_RAW 0 takes the cursor's motion in place of raw input (input_win.c mouse_raw)
    {"MOUSE_SENS_Y", "1", "0.1..3", LIVE}, {"MOUSE_INVERT_Y", "0", "0|1", LIVE}, {"MOUSE_RAW", "1", "0|1", LIVE},
    {"WALK_KEY", "none", NULL, LIVE}, {"WALK_SCALE", "0.51", "0.2..0.9", LIVE},
    // the game's dash on an input of its own: the left stick's press while it is tilted, and this key (input.c dash_apply)
    {"STICK_DASH", "0", "0|1", LIVE}, {"DASH_KEY", "v", NULL, LIVE},
    // several gamepads: the one pressed last is read (input.c pad_follow); compat: the system's XInput / DirectInput
    // path in place of SDL's own drivers, for a controller listed but silent (input.c start)
    {"PAD_FOLLOW", "1", "0|1", LIVE}, {"PAD_DRIVER", "auto", "auto|compat", RESTART},
    // the cursor over the window hides after this many seconds without moving (input_win.c, input_sdl.c); 0 never
    {"CURSOR_HIDE", "3", "0|1|3|5|10", LIVE},
    // keyboard and mouse ignored while a gamepad is connected (input.c kbm_off); the screenshot key and whether the
    // picture keeps the menu and the overlays (input.c screenshot_key, gfx_present.c screenshot_path)
    {"KBM_OFF", "0", "0|1", LIVE}, {"MENU_GUIDE", "1", "0|1", LIVE}, {"SCREENSHOT_KEY", "f12", "none|f12|f9|f7|f6|f5", LIVE},
    {"SCREENSHOT_OVERLAY", "0", "0|1", LIVE},
    // the order of the menu's own list (ui/menu.cpp tab_menu langs[], after "Automatic")
    {"MENU_LANG", "", "en|de|es|fr|it|nl|pt|pt-BR|ru|ja|ko|zh-Hans|zh-Hant", LIVE},
    {"AMIIBO_DIR", "", NULL, LIVE},
    // Discord Rich Presence (discord_rpc.c): on by default, as other ports show theirs; the status names only the port
    // and the play time, Discord's own settings can hide it, and the Menu tab of the settings menu switches it off.
    // The update notice (update_note.c): one request for the newest release when the game starts.
    {"UPDATE_CHECK", "1", "0|1", LIVE},
    {"DISCORD_RPC", "1", "0|1", LIVE},
#ifdef GAME_OOT   // Rekarina's own Discord application and its art asset (the key Discord gave the uploaded file)
    {"DISCORD_APP_ID", "1571222830069194843", NULL, LIVE},
    {"DISCORD_IMAGE", "discord-icon", NULL, LIVE},
#else             // the application of TotK Recomp
    {"DISCORD_APP_ID", "1567232599846293504", NULL, LIVE},
    {"DISCORD_IMAGE", "totk_512x512", NULL, LIVE},
#endif
    // Advanced: scheduling
    {"ASYNC_PREEMPT", "1", "0|1|2", RESTART}, {"REPLAY_ASYNC", "1", "0|1", RESTART},
    {"FREE_CORES", "0", "0|1", RESTART}, {"TIMER_1MS", "1", "0|1", RESTART},
    {"PROCESS_PRIORITY", "above", "normal|above|high", RESTART},   // main.c
    // shaders and pipelines
    {"PIPE_WAIT", "default", "default|draw", RESTART}, {"PIPE_SKIP", "fullscreen", "strict|any|fullscreen", RESTART},
    {"LIB_PREBUILD", "1", "0|1", RESTART}, {"LIB_PREBUILD_THREADS", "", "1..64", RESTART},
    {"PIPE_PREBUILD", "1", "0|1", RESTART}, {"PREWARM_XLATE", "", "1..64", RESTART}, {"PREP_SCREEN", "1", "0|1", RESTART}, {"PREPARE_DRIVER", "1", "0|1", RESTART}, {"GPL", "1", "0|1", RESTART},
    // textures and memory
    {"ASTC", "auto", "auto|gpu|native|cpu", RESTART}, {"MEMORY", "auto", "auto|import|mirror|verify", RESTART},
    {"TEX_ALIAS", "1", "0|1", RESTART}, {"TEX_VRAM_PRESSURE", "75", "0..100", RESTART},
    {"TEX_QUALITY", "auto", "auto|full|half|quarter", RESTART},   // vk_texture.c "texture quality"
    {"TEX_CACHE", "1", "0|1", RESTART}, {"TEX_CACHE_MB", "14336", "256..65536", RESTART},   // vk_texture.c "texture cache"
#ifdef GAME_OOT   // OoT: BC3 by default (vk_astc.c create_bc_pipeline)
    {"ASTC_BC", "1", "0|1", RESTART},
#else
    {"ASTC_BC", "", "0|1", RESTART},
#endif
    // (OoT too, since a checked run compared the two decoders: recomp/patches-oot-v111.txt, set zstd)
    {"ZSTD_NATIVE", "1", "0|1", RESTART}, {"ZSTD_CHECK", "0", "0|1", RESTART},
    // render fixes
    {"LOD_GUARD", "1", "0|1", RESTART}, {"FIX_C0", "1", "0|1", RESTART}, {"FIX_ZERO", "1", "0|1", RESTART},
#ifdef GAME_OOT
    {"TAA", "dlss", "dlss|fsr3|fsr1|bilinear", LIVE},
#else
    {"TAA", "dlss", "dlss|fsr3|fsr1", RESTART},
#endif
    {"GAME_ASPECT", "16:9", "16:9|window|21:9|32:9", RESTART}, {"GAME_RES", "1440", "720|900|1080|1440|1800|2160|2880|window", RESTART},
    {"HUD_LAYOUT", "edges", "edges|center|custom", LIVE},
    {"HUD_HEALTH_X", "0", "-15..15", LIVE}, {"HUD_HEALTH_Y", "0", "-15..15", LIVE}, {"HUD_HEALTH_SCALE", "100", "75..125", LIVE},
    {"HUD_BUTTONS_X", "0", "-15..15", LIVE}, {"HUD_BUTTONS_Y", "0", "-15..15", LIVE}, {"HUD_BUTTONS_SCALE", "100", "75..125", LIVE},
    {"HUD_MINIMAP_X", "0", "-15..15", LIVE}, {"HUD_MINIMAP_Y", "0", "-15..15", LIVE}, {"HUD_MINIMAP_SCALE", "100", "75..125", LIVE},
#ifdef GAME_OOT
    {"FRAMEGEN", "off", "off|fsr3|dlss", LIVE},   // vk_fg_restart_needed checks the running device and hooks
#else
    {"FRAMEGEN", "off", "off|fsr3|dlss", RESTART},
#endif
    {"REFLEX", "1", "0|1|2", LIVE}, {"BIND_SKIP", "1", "0|1", RESTART}, {"SHADER_MEMO", "1", "0|1", RESTART}, {"PIPE_MEMO", "1", "0|1", RESTART}, {"TEX_MEMO", "1", "0|1", RESTART},
    {"DESC_TEMPLATE", "1", "0|1", RESTART},
    {"PUSH_SKIP", "1", "0|1", RESTART}, {"SPLIT_DRAWS", "16", "0..1000", RESTART}, {"REPLAY_THREADS", "0", "0..12", RESTART},
    {"SPLIT_US", "150", "0..20000", RESTART}, {"SPLIT_IN_PASS", "60", "0..5000", RESTART},
    // Vulkan
    {"VK_VALIDATION", "0", "0|1", RESTART}, {"VK_DYN_VERTEX", "", "0|1", RESTART},
    {"VK_DYN_A2C", "", "0|1", RESTART}, {"VK_SUBGROUP_PIN", "1", "0|1", RESTART},
    // diagnostics
    {"TRACE", "", NULL, RESTART}, {"FLIGHT", "0", "0|1", RESTART}, {"FLIGHT_SPIKE_MS", "100", "0..60000", RESTART},
    {"WB_FENCE_LIMIT", "65536", "0..1073741824", RESTART}, {"WB_MERGE", "1", "0|1", RESTART}, {"WB_EXACT", "1", "0|1", RESTART},
    {"PASS_CRUMBS", "", "0|1", RESTART},   // vk_crumbs.c pass marks; unset: VK_AMD_buffer_marker only
    {"INPUT_RECORD", "", NULL, RESTART},
    {"MOTION_VECTORS", "hybrid", "hybrid|camera|models|off", LIVE},
    {"DLAA_REVERSE_ONLY", "0", "0|1", LIVE},
    {"DLAA_PRE_BLOOM", "0", "0|1", RESTART},   // vk_dlaa.c vk_dlaa_init
#ifdef GAME_OOT
    {"DLSS_PREBLOOM", "1", "0|1", RESTART},   // patches.c: checked reconstruction before the guest bloom
#endif
#ifdef GAME_OOT
    {"DLSS_INTERFACE_RECOMPOSITION", "1", "0|1", RESTART},
#else
    {"DLSS_INTERFACE_RECOMPOSITION", "0", "0|1", RESTART},
#endif
};

static int is_known(const char *key) {
    for (size_t i = 0; i < sizeof known / sizeof *known; i++)
        if (!strcmp(known[i], key)) return 1;
    return !strncmp(key, "CHEAT_", 6);   // cheats.c and the cheat patch sets (tools/cheats.py)
}

static void clear(void) {
    for (int i = 0; i < n_entries; i++) free(entries[i].key), free(entries[i].value);
    n_entries = 0;
}

static char *trim(char *s) {
    while (*s == ' ' || *s == '\t') s++;
    char *e = s + strlen(s);
    while (e > s && (e[-1] == ' ' || e[-1] == '\t' || e[-1] == '\r')) *--e = 0;
    return s;
}

// The key of a setting's name (either prefix), or NULL for a name that is no setting's.
static const char *key_of(const char *name) {
    if (!strncasecmp(name, PREFIX, PREFIX_LEN)) return name + PREFIX_LEN;
    if (!strncasecmp(name, OLD_PREFIX, OLD_PREFIX_LEN)) return name + OLD_PREFIX_LEN;
    return NULL;
}

// The environment's value for a key: REKARINA_<key>, else TOTK_<key>.
static const char *env_of(const char *key) {
    char var[128];
    snprintf(var, sizeof var, PREFIX "%s", key);
    const char *e = getenv(var);
    if (e) return e;
    snprintf(var, sizeof var, OLD_PREFIX "%s", key);
    return getenv(var);
}

const char *cfg_env(const char *name) {
    const char *k = key_of(name);
    return k ? env_of(k) : getenv(name);
}

// The upper-case key without the prefix, in place.
static char *norm_key(char *key) {
    for (char *p = key; *p; p++) *p = (char)toupper((unsigned char)*p);
    const char *k = key_of(key);
    return k ? (char *)k : key;
}

static void add(char *key, char *value, int line) {
    key = norm_key(key);
    size_t vl = strlen(value);
    if (vl >= 2 && value[0] == '"' && value[vl - 1] == '"') value[vl - 1] = 0, value++;
    if (!*key) return;
    if (!is_known(key)) rt_log("cfg: " RT_INI_FILE " line %d: unknown key '%s'", line, key);
    if (!*value) return;   // "key =" leaves the setting at its default
    for (int i = 0; i < n_entries; i++)
        if (!strcmp(entries[i].key, key)) {   // a repeated key: the later line wins
            free(entries[i].value);
            entries[i].value = strdup(value);
            return;
        }
    if (n_entries == CFG_MAX_KEYS) return;
    entries[n_entries++] = (CfgEntry){strdup(key), strdup(value), 0, 0};
}

// 0.1.9 had the Xbox prompts as a pad layout: pad_layout = xbox_prompts is pad_layout = nintendo with prompts = xbox.
// Read as those two, which the menu shows and the next cfg_save writes; a prompts line of the file's own wins. The
// value still works from the environment (input_pad_layout_pick, input_prompts_pick).
static void migrate(void) {
#ifdef GAME_OOT
    CfgEntry *hdr = NULL, *mapper = NULL;
    for (int i = 0; i < n_entries; ++i) {
        if (!strcmp(entries[i].key, "HDR")) hdr = &entries[i];
        if (!strcmp(entries[i].key, "HDR_TONEMAP")) mapper = &entries[i];
    }
    if (hdr && !strcmp(hdr->value, "native") && (mapper || n_entries < CFG_MAX_KEYS)) {
        free(hdr->value);
        hdr->value = strdup("1");
        hdr->dirty = 1;
        if (mapper) {
            free(mapper->value);
            mapper->value = strdup("vanilla");
            mapper->dirty = 1;
        } else {
            entries[n_entries++] = (CfgEntry){strdup("HDR_TONEMAP"), strdup("vanilla"), 1, 0};
        }
        rt_log("cfg: " RT_INI_FILE " hdr = native is read as hdr = 1 and hdr_tonemap = vanilla");
    }
#endif
    CfgEntry *layout = NULL, *prompts = NULL;
    for (int i = 0; i < n_entries; i++) {
        if (!strcmp(entries[i].key, "PAD_LAYOUT")) layout = &entries[i];
        if (!strcmp(entries[i].key, "PROMPTS")) prompts = &entries[i];
    }
    if (!layout || strcasecmp(layout->value, "xbox_prompts")) return;
    free(layout->value);
    layout->value = strdup("nintendo");
    layout->dirty = 1;
    if (!prompts && n_entries < CFG_MAX_KEYS) entries[n_entries++] = (CfgEntry){strdup("PROMPTS"), strdup("xbox"), 1, 0};
    rt_log("cfg: " RT_INI_FILE " pad_layout = xbox_prompts is read as pad_layout = nintendo and prompts = %s",
           prompts ? prompts->value : "xbox");
}

static char *boot_restart_vals[sizeof metas / sizeof *metas];
static int boot_restart_ready;

static void snapshot_restart_values(void) {
    plat_mutex_lock(&set_lock);
    boot_skin_motion = cfg_bool(PREFIX "SKIN_MOTION");
    for (size_t i = 0; i < sizeof metas / sizeof *metas; i++) {
        free(boot_restart_vals[i]);
        boot_restart_vals[i] = NULL;
        if (metas[i].live) continue;
        char full[80];
        snprintf(full, sizeof full, PREFIX "%s", metas[i].key);
        const char *v = cfg_get(full);
        boot_restart_vals[i] = strdup(v && *v ? v : metas[i].def);
    }
    boot_restart_ready = 1;
    plat_mutex_unlock(&set_lock);
}

int cfg_restart_pending(void) {
    if (!boot_restart_ready) return 0;
    plat_mutex_lock(&set_lock);
    int pending = 0;
    for (size_t i = 0; i < sizeof metas / sizeof *metas && !pending; i++) {
        if (metas[i].live || env_of(metas[i].key)) continue;
        char full[80];
        snprintf(full, sizeof full, PREFIX "%s", metas[i].key);
        const char *v = cfg_get(full);
        const char *cur = v && *v ? v : metas[i].def;
        const char *was = boot_restart_vals[i] ? boot_restart_vals[i] : metas[i].def;
        if (strcmp(cur, was)) pending = 1;
    }
    plat_mutex_unlock(&set_lock);
    return pending;
}

int cfg_load(const char *path) {
    int frame = __atomic_exchange_n(&frame_index, -1, __ATOMIC_RELAXED);
    if (frame >= 0) cfg_render_release(&render_states[frame]);
    __atomic_store_n(&render_index, -1, __ATOMIC_RELAXED);
    clear();
    uint64_t size = 0;
    uint8_t *data = plat_file_read(path, &size);
    if (!data) {
        snapshot_restart_values();
        return 0;
    }
    char *p = (char *)data;
    if (size >= 3 && !memcmp(p, "\xEF\xBB\xBF", 3)) p += 3;   // editors add a BOM
    for (int line = 1; *p; line++) {
        char *eol = strchr(p, '\n');
        if (eol) *eol = 0;
        char *s = trim(p);
        char *eq = strchr(s, '=');
        if (*s && *s != '#' && *s != ';' && *s != '[') {
            if (!eq) rt_log("cfg: " RT_INI_FILE " line %d: no '=' in '%s'", line, s);
            else {
                *eq = 0;
                add(trim(s), trim(eq + 1), line);
            }
        }
        if (!eol) break;
        p = eol + 1;
    }
    free(data);
    migrate();
    snapshot_restart_values();
    return 1;
}

// ---- device profiles: defaults for a known machine, below the environment and rekarina.ini and above the table, so any
// key the player sets wins and resetting a key in the menu returns to the profile's value. REKARINA_DEVICE_PROFILE = auto
// (default: the Steam Deck or a Windows handheld when the firmware says so), desktop (none), deck or handheld.
//
// The Steam Deck (LCD Jupiter, OLED Galileo; 1280x800, 16:10, 4 Zen 2 cores, RDNA 2 with 8 CUs sharing 16 GB): the game
// at the screen's 1280x800 with its own 5/6 scene (the aspect patch gives the camera its 16:10, resolution.c, and the
// romfs-patches-1280x800 overlay the HUD's);
// fullscreen and vsync, as gamescope composes Game Mode; 40 fps, which divides 40 Hz (and 120 Hz on the OLED).
// OoT: the game's own handheld mode (the console's screen: a 1920x1080 picture, ~30 % less GPU time than docked on an
// RTX 5070 standing in the village, 2026-11-15) and quarter textures (the 16 GB are the GPU's memory too; the free-memory
// counter "auto" uses is not there under Proton).
// Starting points to replace with measured ones (docs/tester-guide.md, Steam Deck).
typedef struct { const char *key, *value; } ProfileKey;
static const ProfileKey DECK[] = {
    {"FULLSCREEN", "1"}, {"WINDOW_SIZE", "1280x800"}, {"OUTPUT_RES", "1280x800"}, {"RENDER_SCALE", "game"},
    {"VK_PRESENT", "fifo"}, {"FPS_LIMIT", "40"},
#ifdef GAME_OOT
    {"OPERATION_MODE", "handheld"}, {"TEX_QUALITY", "quarter"},
#endif
    {NULL, NULL},
};
// Windows handhelds (ROG Ally and Ally X, ROG Xbox Ally, Legion Go, MSI Claw: 1920x1080 or more, an integrated GPU
// sharing 16 to 32 GB): fullscreen at the screen's size, the game's handheld mode, 60 fps. Texture quality stays auto
// (Windows reports the free video memory there).
static const ProfileKey HANDHELD[] = {
    {"FULLSCREEN", "1"}, {"FPS_LIMIT", "60"},
#ifdef GAME_OOT
    {"OPERATION_MODE", "handheld"},
#endif
    {NULL, NULL},
};
static int profile = -1;   // 0 none, 1 Steam Deck, 2 Windows handheld
static const ProfileKey *const PROFILE_KEYS[3] = {NULL, DECK, HANDHELD};

static int read_line(const char *path, char *out, size_t n) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    int ok = fgets(out, (int)n, f) != NULL;
    fclose(f);
    out[ok ? strcspn(out, "\r\n") : 0] = 0;
    return ok;
}

// The Steam Deck: Valve's board, product Jupiter or Galileo (/sys/class/dmi/id, or the BIOS key Wine fills from it
// and Windows on a Deck has); SteamDeck=1, which Steam sets on it.
static int on_steam_deck(void) {
    const char *env = getenv("SteamDeck");
    if (env && !strcmp(env, "1")) return 1;
    char vendor[64], product[64];
    if (plat_machine_name(product, sizeof product) && (!strcmp(product, "Jupiter") || !strcmp(product, "Galileo")))
        return 1;
    return read_line("/sys/class/dmi/id/board_vendor", vendor, sizeof vendor) && !strcmp(vendor, "Valve") &&
           read_line("/sys/class/dmi/id/product_name", product, sizeof product) &&
           (!strcmp(product, "Jupiter") || !strcmp(product, "Galileo"));
}

// A Windows handheld by the firmware's product name: ROG Ally (RC71L), Ally X (RC72LA), ROG Xbox Ally (RC73), Legion
// Go (83E1, 83L3, 83N0, 83N1), MSI Claw.
static int on_windows_handheld(void) {
    char p[128];
    if (!plat_machine_name(p, sizeof p)) return 0;
    static const char *const names[] = {"ROG Ally", "RC71L", "RC72LA", "RC73", "Legion Go", "83E1", "83L3", "83N0", "83N1", "Claw"};
    for (size_t i = 0; i < sizeof names / sizeof *names; i++)
        if (strstr(p, names[i])) return 1;
    return 0;
}

static const char *raw_get(const char *name);

static int profile_of(void) {
    int p = __atomic_load_n(&profile, __ATOMIC_ACQUIRE);
    if (p >= 0) return p;
    const char *set = raw_get(PREFIX "DEVICE_PROFILE");
    if (set && !strcmp(set, "deck")) p = 1;
    else if (set && !strcmp(set, "handheld")) p = 2;
    else if (set && *set && strcmp(set, "auto")) p = 0;
    else p = on_steam_deck() ? 1 : on_windows_handheld() ? 2 : 0;
    __atomic_store_n(&profile, p, __ATOMIC_RELEASE);
    return p;
}

const char *cfg_device_profile(void) {
    int p = profile_of();
    return p == 1 ? "deck" : p == 2 ? "handheld" : "desktop";
}

static const char *raw_get(const char *name) {
    const char *key = key_of(name);
    if (!key) return getenv(name);
    const char *e = env_of(key);
    if (e) return e;
    if (render_state) {
        for (int i = 0; i < render_state->count; ++i)
            if (!strcasecmp(render_state->values[i].key, key)) return render_state->values[i].value;
        return NULL;
    }
    int n = __atomic_load_n(&n_entries, __ATOMIC_ACQUIRE);
    for (int i = 0; i < n; i++)
        if (!strcasecmp(entries[i].key, key)) return __atomic_load_n(&entries[i].value, __ATOMIC_ACQUIRE);
    return NULL;
}

const char *cfg_get(const char *name) {
    const char *v = raw_get(name), *key = key_of(name);
    int p = v || !key ? 0 : profile_of();
    if (!p) return v;
    for (const ProfileKey *k = PROFILE_KEYS[p]; k->key; k++)
        if (!strcasecmp(k->key, key)) return k->value;
    return NULL;
}

// Whether name's value comes from the player (the environment or rekarina.ini), not a profile or the table.
int cfg_is_set(const char *name) { return raw_get(name) != NULL; }

unsigned cfg_version(void) { return render_state ? render_state->revision : __atomic_load_n(&version, __ATOMIC_ACQUIRE); }

int cfg_from_env(const char *name) { return cfg_env(name) != NULL; }

void cfg_set(const char *name, const char *value) {
    const char *k = key_of(name);
    if (!k || !value) return;
    char key[64];
    snprintf(key, sizeof key, "%s", k);
    for (char *p = key; *p; p++) *p = (char)toupper((unsigned char)*p);
    plat_mutex_lock(&set_lock);
    int i = 0, n = n_entries;
    while (i < n && strcmp(entries[i].key, key)) i++;
    if (i < n) {
        if (!entries[i].value || strcmp(entries[i].value, value)) {
            __atomic_store_n(&entries[i].value, strdup(value), __ATOMIC_RELEASE);   // the old string stays for readers
            entries[i].dirty = 1;
        }
        entries[i].removed = 0;   // a value set after cfg_reset replaces the line instead of deleting it
    } else if (n < CFG_MAX_KEYS) {
        entries[n] = (CfgEntry){strdup(key), strdup(value), 1, 0};
        __atomic_store_n(&n_entries, n + 1, __ATOMIC_RELEASE);
    }
    __atomic_add_fetch(&version, 1, __ATOMIC_RELEASE);
    plat_mutex_unlock(&set_lock);
}

int cfg_reset(const char *const *keys, int n) {
    int done = 0;
    plat_mutex_lock(&set_lock);
    for (int k = 0; k < n; k++) {
        char key[64];
        snprintf(key, sizeof key, "%s", keys[k]);
        const char *name = norm_key(key);
        if (env_of(name)) continue;   // the environment wins; the file line stays
        int i = 0;
        while (i < n_entries && strcmp(entries[i].key, name)) i++;
        if (i == n_entries) {   // no value in memory ("key =" in the file): the line still has to go
            if (i == CFG_MAX_KEYS) continue;
            entries[i] = (CfgEntry){strdup(name), NULL, 0, 1};
            __atomic_store_n(&n_entries, i + 1, __ATOMIC_RELEASE);
        } else if (entries[i].value) {
            __atomic_store_n(&entries[i].value, NULL, __ATOMIC_RELEASE);   // the old string stays for readers
            entries[i].dirty = 0;
            entries[i].removed = 1;   // cfg_save_to deletes the key's lines
            done++;
        } else {
            entries[i].removed = 1;
        }
    }
    if (done) __atomic_add_fetch(&version, 1, __ATOMIC_RELEASE);
    plat_mutex_unlock(&set_lock);
    return done;
}

int cfg_last_save_ok(void) { return __atomic_load_n(&save_ok, __ATOMIC_ACQUIRE); }

void cfg_each_active(void (*cb)(const char *key, const char *value, int from_env, void *ctx), void *ctx) {
    int n = __atomic_load_n(&n_entries, __ATOMIC_ACQUIRE);
    for (int i = 0; i < n; i++) {
        const char *v = __atomic_load_n(&entries[i].value, __ATOMIC_ACQUIRE);
        if (v) cb(entries[i].key, v, 0, ctx);
    }
    for (size_t i = 0; i < sizeof known / sizeof *known; i++) {
        const char *e = env_of(known[i]);
        if (e) cb(known[i], e, 1, ctx);
    }
}

const CfgMeta *cfg_meta(const char *key) {
    char k[64];
    snprintf(k, sizeof k, "%s", key);
    const char *name = norm_key(k);
    for (size_t i = 0; i < sizeof metas / sizeof *metas; i++)
        if (!strcmp(metas[i].key, name)) return &metas[i];
    return NULL;
}

const CfgMeta *cfg_meta_at(int i) { return i >= 0 && (size_t)i < sizeof metas / sizeof *metas ? &metas[i] : NULL; }

const char *cfg_value(const char *name) {
    const char *e = cfg_get(name);
#ifdef GAME_OOT
    const char *key = key_of(name);
    if (key && !strcasecmp(key, "HDR") && e && !strcmp(e, "native")) return "1";
    if (key && !strcasecmp(key, "HDR_TONEMAP")) {
        const char *signal = raw_get(PREFIX "HDR");
        if (signal && !strcmp(signal, "native")) return "vanilla";
    }
    if (e && !strcmp(e, "renodrt")) {
        if (key && (!strcasecmp(key, "HDR_TONEMAP") || !strcasecmp(key, "SDR_TONEMAP")))
            return cfg_meta(name)->def;   // Retired mapper: migrate saved settings and environment overrides alike.
    }
#endif
    if (e && *e) return e;
    const CfgMeta *m = cfg_meta(name);
    if (!m) rt_fatal("cfg: %s has no row in the settings table", name);
    return m->def;
}

// The row's default as a number, for a set value that does not parse.
static double default_number(const char *name) {
    return strtod(cfg_meta(name)->def, NULL);
}

double cfg_double(const char *name) {
    const char *v = cfg_value(name);
    char *end;
    double d = strtod(v, &end);
    return end == v || *end || d != d ? default_number(name) : d;
}

int cfg_int(const char *name) {
    const char *v = cfg_value(name);
    char *end;
    long l = strtol(v, &end, 10);
    return end == v || *end ? (int)default_number(name) : (int)l;
}

int cfg_bool(const char *name) { return strcmp(cfg_value(name), "0") != 0; }

// ---- cfg_save: the file as lines, each with its line ending
typedef struct { const char *s; size_t n; } Line;   // n includes the '\n' (and '\r') when present

static int line_count(const char *p, size_t size) {
    int n = 0;
    for (size_t i = 0; i < size; i++) n += p[i] == '\n';
    return n + 1;
}

// The line's text without its ending, BOM and surrounding blanks, into out.
static void line_text(const Line *l, int first, char *out, size_t cap) {
    size_t n = l->n;
    const char *s = l->s;
    if (first && n >= 3 && !memcmp(s, "\xEF\xBB\xBF", 3)) s += 3, n -= 3;
    while (n && (s[n - 1] == '\n' || s[n - 1] == '\r')) n--;
    if (n >= cap) n = cap - 1;
    memcpy(out, s, n);
    out[n] = 0;
    char *t = trim(out);
    memmove(out, t, strlen(t) + 1);
}

// The key of a "key = value" line (comment = 0) or of a commented "# key = value" line (comment = 1), normalised,
// into key; 0 when the line is not one.
static int line_key(const char *text, int comment, char *key, size_t cap) {
    if (comment) {
        if (*text != '#' && *text != ';') return 0;
        text++;
        while (*text == ' ' || *text == '\t') text++;
    }
    if (!*text || *text == '#' || *text == ';' || *text == '[') return 0;
    const char *eq = strchr(text, '=');
    if (!eq) return 0;
    size_t n = (size_t)(eq - text);
    while (n && (text[n - 1] == ' ' || text[n - 1] == '\t')) n--;
    if (!n || n >= cap) return 0;
    for (size_t i = 0; i < n; i++)
        if (!isalnum((unsigned char)text[i]) && text[i] != '_') return 0;
    char tmp[128];
    memcpy(tmp, text, n);
    tmp[n] = 0;
    snprintf(key, cap, "%s", norm_key(tmp));
    return 1;
}

// The section name of a "[name]" line, lower-case, into out; 0 when the line is not one.
static int line_section(const char *text, char *out, size_t cap) {
    if (*text != '[') return 0;
    const char *e = strchr(text, ']');
    if (!e) return 0;
    char tmp[128];
    size_t n = (size_t)(e - text - 1);
    if (n >= sizeof tmp) n = sizeof tmp - 1;
    memcpy(tmp, text + 1, n);
    tmp[n] = 0;
    snprintf(out, cap, "%s", trim(tmp));
    for (char *p = out; *p; p++) *p = (char)tolower((unsigned char)*p);
    return 1;
}

typedef struct { char *buf; size_t n, cap; } Out;
static void put(Out *o, const char *s, size_t n) {
    if (o->n + n + 1 > o->cap) {
        o->cap = (o->n + n + 1) * 2;
        o->buf = realloc(o->buf, o->cap);
        if (!o->buf) rt_fatal("cfg: out of memory writing " RT_INI_FILE);
    }
    memcpy(o->buf + o->n, s, n);
    o->n += n;
}
static void puts_(Out *o, const char *s) { put(o, s, strlen(s)); }

static const char *section_of(const char *key) {
    for (size_t i = 0; i < sizeof sections / sizeof *sections; i++)
        if (!strcmp(sections[i].key, key)) return sections[i].section;
    if (!strncmp(key, "CHEAT_", 6)) return "cheats";   // cheats.c: one key per cheat of the cheat file
    return NULL;
}

// "key = value" + eol for entry e (key lower-case)
static void put_entry(Out *o, const CfgEntry *e, const char *eol) {
    char k[64];
    snprintf(k, sizeof k, "%s", e->key);
    for (char *p = k; *p; p++) *p = (char)tolower((unsigned char)*p);
    puts_(o, k);
    puts_(o, " = ");
    puts_(o, e->value);
    puts_(o, eol);
}

int cfg_save_to(const char *path) {
    plat_mutex_lock(&set_lock);
    int n = n_entries, any = 0;
    for (int i = 0; i < n; i++) any |= entries[i].dirty | entries[i].removed;
    if (!any) {
        __atomic_store_n(&save_ok, 1, __ATOMIC_RELEASE);
        plat_mutex_unlock(&set_lock);
        return 1;
    }
    uint64_t size = 0;
    uint8_t *data = plat_file_read(path, &size);
    if (!data && plat_file_exists(path) && plat_file_size(path) != 0) {
        // the file is there but unreadable (locked, transient error): rebuilding it from nothing would lose its lines
        rt_log("cfg: cannot read %s, nothing written", path);
        __atomic_store_n(&save_ok, 0, __ATOMIC_RELEASE);
        plat_mutex_unlock(&set_lock);
        return 0;
    }
    const char *text = data ? (const char *)data : "";
    int nl = data ? line_count(text, size) : 0;
    Line *lines = calloc((size_t)nl + 1, sizeof *lines);
    // after[i]: entries to write after line i (-1 = none); replace[i]: the entry whose value line i now carries
    int *replace = malloc(sizeof(int) * ((size_t)nl + 1)), *after_key = malloc(sizeof(int) * (size_t)(n + 1));
    char *drop = calloc((size_t)nl + 1, 1);   // lines of a reset key
    if (!lines || !replace || !after_key || !drop) rt_fatal("cfg: out of memory writing " RT_INI_FILE);
    for (int i = 0, at = 0; i < nl; i++) {
        size_t start = (size_t)at;
        while ((uint64_t)at < size && text[at] != '\n') at++;
        if ((uint64_t)at < size) at++;
        lines[i] = (Line){text + start, (size_t)at - start};
        replace[i] = -1;
    }
    if (nl && !lines[nl - 1].n) nl--;   // the text ended with a newline: no last empty line
    const char *eol = data && strstr(text, "\r\n") ? "\r\n" : "\n";
    char t[1024], k[128], sec[128];
    for (int e = 0; e < n; e++) {
        after_key[e] = -2;   // -2 clean, -1 append in a new section at the end, >= 0 insert after that line
        if (entries[e].removed)
            for (int i = 0; i < nl; i++) {
                line_text(&lines[i], i == 0, t, sizeof t);
                if (line_key(t, 0, k, sizeof k) && !strcmp(k, entries[e].key)) drop[i] = 1;
            }
        if (!entries[e].dirty) continue;
        int last = -1;
        for (int i = 0; i < nl; i++) {
            line_text(&lines[i], i == 0, t, sizeof t);
            if (line_key(t, 0, k, sizeof k) && !strcmp(k, entries[e].key)) last = i;
        }
        if (last >= 0) {
            replace[last] = e;
            continue;
        }
        const char *want = section_of(entries[e].key);
        int in = 0, example = -1, tail = -1;
        for (int i = 0; i < nl && want; i++) {
            line_text(&lines[i], i == 0, t, sizeof t);
            if (line_section(t, sec, sizeof sec)) {
                in = !strcmp(sec, want);
                if (in) tail = i;
                continue;
            }
            if (!in) continue;
            if (*t) tail = i;
            if (line_key(t, 1, k, sizeof k) && !strcmp(k, entries[e].key)) example = i;
        }
        after_key[e] = example >= 0 ? example : tail;
    }
    Out o = {0};
    for (int i = 0; i < nl; i++) {
        if (drop[i]) {
            // the line leaves the file; keys inserted after it still need a line ending before them
        } else if (replace[i] >= 0) {
            // keep the line's indentation and key spelling, put the new value after " = "
            const char *s = lines[i].s;
            size_t len = lines[i].n;
            if (i == 0 && len >= 3 && !memcmp(s, "\xEF\xBB\xBF", 3)) put(&o, s, 3), s += 3, len -= 3;
            const char *eq = memchr(s, '=', len);
            size_t kn = (size_t)(eq - s);
            while (kn && (s[kn - 1] == ' ' || s[kn - 1] == '\t')) kn--;
            put(&o, s, kn);
            puts_(&o, " = ");
            puts_(&o, entries[replace[i]].value);
            int had_eol = len && s[len - 1] == '\n';
            if (had_eol) puts_(&o, eol);
        } else {
            put(&o, lines[i].s, lines[i].n);
        }
        for (int e = 0; e < n; e++)
            if (after_key[e] == i) {
                if (o.n && o.buf[o.n - 1] != '\n') puts_(&o, eol);
                put_entry(&o, &entries[e], eol);
            }
    }
    // keys whose section the file lacks: one new section each, at the end
    for (int e = 0; e < n; e++) {
        if (after_key[e] != -1) continue;
        const char *want = section_of(entries[e].key);
        if (o.n && o.buf[o.n - 1] != '\n') puts_(&o, eol);
        if (want) {
            if (o.n) puts_(&o, eol);
            puts_(&o, "[");
            puts_(&o, want);
            puts_(&o, "]");
            puts_(&o, eol);
        }
        put_entry(&o, &entries[e], eol);
        // the following keys of the same new section go right after this one
        for (int f = e + 1; f < n && want; f++)
            if (after_key[f] == -1 && section_of(entries[f].key) && !strcmp(section_of(entries[f].key), want)) {
                put_entry(&o, &entries[f], eol);
                after_key[f] = -3;
            }
    }
    int ok = plat_file_write_atomic(path, o.buf ? o.buf : "", o.n);
    if (ok)
        for (int e = 0; e < n; e++) entries[e].dirty = entries[e].removed = 0;
    else
        rt_log("cfg: cannot write %s", path);
    __atomic_store_n(&save_ok, ok ? 1 : 0, __ATOMIC_RELEASE);
    free(o.buf);
    free(drop);
    free(after_key);
    free(replace);
    free(lines);
    free(data);
    plat_mutex_unlock(&set_lock);
    return ok;
}

static char ini_path[1024];

static int find_ini(void) {
    if (ini_path[0]) return 1;
    char path[1024];
    if (!plat_exe_path(path, sizeof path)) return 0;
    char *cut = plat_last_sep(path);
    if (!cut) cut = strrchr(path, '/');
    if (!cut || cut + 1 + sizeof RT_INI_FILE >= path + sizeof path) return 0;
    strcpy(cut + 1, RT_INI_FILE);
#ifdef RT_OLD_FILE_STEM
    // A folder made before the file got its name: the player's settings move over, once. (A file that cannot be
    // renamed is read where it is.)
    if (!plat_file_exists(path)) {
        char old[1024];
        snprintf(old, sizeof old, "%.*s" RT_OLD_FILE_STEM ".ini", (int)(cut + 1 - path), path);
        if (plat_file_exists(old)) {
            if (plat_file_replace(old, path)) rt_log("cfg: %s renamed to " RT_INI_FILE, old);
            else snprintf(path, sizeof path, "%s", old);
        }
    }
#endif
    snprintf(ini_path, sizeof ini_path, "%s", path);
    return 1;
}

int cfg_save(void) {
    // REKARINA_INI_READONLY=1 (test runs from a shared build folder): changes made in the run stay in it; the file beside
    // the exe, which every other run reads, is not written.
    if (cfg_from_env(PREFIX "INI_READONLY") && cfg_bool(PREFIX "INI_READONLY")) {
        static int said;
        if (!said++) rt_log("cfg: REKARINA_INI_READONLY: " RT_INI_FILE " is not written in this run");
        __atomic_store_n(&save_ok, 1, __ATOMIC_RELEASE);
        return 1;
    }
    if (find_ini()) return cfg_save_to(ini_path);
    __atomic_store_n(&save_ok, 0, __ATOMIC_RELEASE);
    return 0;
}

static void load_ini(void) {
    if (!find_ini()) return;
    if (!cfg_load(ini_path)) {
        rt_log("cfg: no settings file at %s", ini_path);
        return;
    }
    rt_log("cfg: read %s: %d keys", ini_path, n_entries);
    for (int i = 0; i < n_entries; i++)
        if (env_of(entries[i].key))
            rt_log("cfg: " PREFIX "%s (or " OLD_PREFIX "%s) is set in the environment and overrides " RT_INI_FILE,
                   entries[i].key, entries[i].key);
}

void cfg_init(void) {
    load_ini();
#ifndef GAME_OOT
    // Frame generation takes DLSS / DLAA's depth and motion vectors (vk_dlaa.c): asked for from the base mode (a
    // rekarina.ini line, the setup), it switches dlaa on in the settings themselves, so every part that reads dlaa
    // (the scene's size, the upscaling pass, the motion vectors) agrees. dlaa set to 0 in the environment wins: the
    // frame generation then logs that it is off.
    const char *fg = cfg_get(PREFIX "FRAMEGEN");
    if (fg && *fg && strcmp(fg, "off") && !cfg_bool(PREFIX "DLAA") && !cfg_from_env(PREFIX "DLAA")) {
        cfg_set(PREFIX "DLAA", "1");
        rt_log("cfg: framegen = %s turns dlaa on: frame generation takes its depth and motion vectors", fg);
    }
#endif
    snapshot_restart_values();
    if (profile_of())
        rt_log("cfg: device profile %s: defaults for the keys " RT_INI_FILE " does not set (REKARINA_DEVICE_PROFILE=desktop "
               "turns it off)", profile_of() == 1 ? "Steam Deck" : "Windows handheld");
    cfg_render_publish();
    cfg_render_frame_begin();
}
