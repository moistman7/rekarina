// Settings: the REKARINA_* environment variables (TOTK_* before the rename, still read), with rekarina.ini beside the
// game behind them.
#pragma once

// getenv for a setting: the environment variable when set, else the rekarina.ini key (the name without "REKARINA_", any
// case), else the device profile's default (cfg.c: the Steam Deck), else NULL. Same string semantics as getenv; the pointer stays valid for the process (a value cfg_set replaces
// is kept). A name without the REKARINA_ prefix goes to getenv unchanged. Safe from any thread once cfg_init has returned.
const char *cfg_get(const char *name);

// The environment alone for a setting's name: REKARINA_<key>, else TOTK_<key> (the prefix before the rename); getenv
// for any other name. For the few settings read before cfg_init or never from the ini.
const char *cfg_env(const char *name);

// Reads totk.ini once, from mem_init after the address space is claimed and before any setting is read. Logs the file, its key count, unknown keys and environment overrides.
void cfg_init(void);

// Replaces the loaded keys with those of the file at path (tests). 1 when the file was read, 0 when missing.
// Not thread-safe: nothing else may call cfg_get meanwhile.
int cfg_load(const char *path);

// Live settings (the in-game menu). cfg_set stores value as the rekarina.ini value of name (REKARINA_ prefixed), marks it for
// cfg_save and bumps cfg_version. cfg_get returns it unless the environment sets name. Any thread.
void cfg_set(const char *name, const char *value);
// Bumped by every cfg_set. A consumer that keeps a parsed value compares this with the number it last saw (one atomic
// load) and parses again when it changed.
unsigned cfg_version(void);
// Publish editable settings at the present boundary. Replay pins the resulting immutable view without taking set_lock.
typedef struct CfgRenderState CfgRenderState;
enum { CFG_HDR_RENODX_SCENE = 1, CFG_HDR_RENODX_EFFECTS = 2 };
typedef enum { CFG_TAA_DLSS, CFG_TAA_FSR3, CFG_TAA_FSR1, CFG_TAA_BILINEAR } CfgTaa;
enum { CFG_RENDER_UPSCALER = 1, CFG_RENDER_FRAMEGEN_MOTION = 2, CFG_RENDER_INTERFACE_COVERAGE = 4,
       CFG_RENDER_FSR_FALLBACK = 8 };
typedef struct {
    unsigned projection_revision, renodx_flags;
    int dlaa, sharpness;
    CfgTaa taa;
    unsigned consumers;
    int motion_mode, motion_models, motion_reverse_only;
    char fsr_version[32];
} CfgRenderValues;
CfgTaa cfg_taa(void);
void cfg_render_publish(void);
// Resolves device-backed consumers against the same editable revision being published. The resolver must not edit cfg.
void cfg_render_publish_present(unsigned (*resolve_consumers)(CfgTaa taa, int wanted));
const CfgRenderState *cfg_render_acquire(void);
const CfgRenderState *cfg_render_frame_acquire(void);
void cfg_render_frame_begin(void);
const CfgRenderState *cfg_render_retain(const CfgRenderState *state);
void cfg_render_release(const CfgRenderState *state);
const CfgRenderValues *cfg_render_values(void);
const CfgRenderValues *cfg_render_values_of(const CfgRenderState *state);
#ifdef GAME_OOT
int cfg_scene_sharpness(int stock_scaler);
#endif
// Workers borrow their submitting thread's pinned state; the inline menu temporarily uses the editable view (NULL).
const CfgRenderState *cfg_render_current(void);
const CfgRenderState *cfg_render_use(const CfgRenderState *state);
// 1 when the environment sets name: cfg_set cannot change what cfg_get returns (the menu greys the setting out).
int cfg_from_env(const char *name);
// 1 when the player set name (the environment or rekarina.ini), 0 when cfg_get gives a device profile's default or NULL.
int cfg_is_set(const char *name);
// The device profile in use, "deck" or "desktop" (REKARINA_DEVICE_PROFILE, else what the machine is).
const char *cfg_device_profile(void);
// Writes the keys cfg_set changed since the last save into rekarina.ini next to rekarina.exe (cfg_save_to: the file at path).
// Only those lines change: the last line of a changed key gets the new value, a missing key goes under its section
// (after the commented example line of the key when the section has one; the section is added at the end of the file
// when missing), comments and every other line stay byte for byte. 1 on success or with nothing to write.
int cfg_save(void);
int cfg_save_to(const char *path);

// Reset (the menu's "Reset all settings to defaults"). Removes the active line of each key (names without REKARINA_, any
// case) from rekarina.ini at the next cfg_save and its value from memory, so the consumer's own default applies again; the
// commented example lines stay. Keys the environment sets are skipped. Bumps cfg_version. Returns how many it reset.
int cfg_reset(const char *const *keys, int n);
// 0 when the last cfg_save / cfg_save_to failed to write (the values stay set in memory, the next save tries again).
int cfg_last_save_ok(void);
// 1 when any restart-only setting (!CfgMeta.live) differs from its value at startup (or the last cfg_load).
int cfg_restart_pending(void);
// Every setting with a value, for a bug report ("Copy system info"): the rekarina.ini and cfg_set keys (name without
// REKARINA_, upper case; from_env 0), then the known REKARINA_ variables the environment sets (from_env 1).
void cfg_each_active(void (*cb)(const char *key, const char *value, int from_env, void *ctx), void *ctx);

// What the menu knows about a setting. key: without REKARINA_, upper case. def: what the consumer uses when the key is
// unset ("" = no fixed default, the consumer picks per machine). values: "0|1", "mailbox|fifo|immediate", a range
// "0..300", or NULL for free text. live: applies at once; the others are read at start.
typedef struct { const char *key, *def, *values; int live; } CfgMeta;
// The row of key (prefix and case ignored); NULL for a key the menu does not show.
const CfgMeta *cfg_meta(const char *key);
// The i-th row (0-based, table order); NULL past the last.
const CfgMeta *cfg_meta_at(int i);

// The setting's value for consumers: the environment or rekarina.ini value, else the CfgMeta default. name has the REKARINA_
// prefix, like cfg_get; a name without a row is a bug (rt_fatal). A row with an empty default has no fixed value: its
// consumer reads cfg_get and keeps its own "unset" branch. An empty value counts as unset.
const char *cfg_value(const char *name);
// cfg_value parsed. A set value that does not parse (int: whole string; double: strtod) gives the default's number.
// cfg_bool: "0" is false, anything else true.
int cfg_int(const char *name);
double cfg_double(const char *name);
int cfg_bool(const char *name);
