// Adapted from MacNativeGames games/totk/runtime/fs_host.h @ b75be8cf (MIT, Copyright (c) 2026 MacNativeGames
// contributors).
// The nn::fs host layer: the read-only RomFS (romfs.c) and writable save, temporary and
// cache storage under the profile directory (fs_host.c). Both register their nn::fs overrides with HOST_OVERRIDE;
// this header is what the two files and the probe share.
#pragma once
#include <stddef.h>
#include <stdint.h>
#include <wchar.h>

// ---- nn::fs results
#define FS_R_PATH_NOT_FOUND 0x202u
#define FS_R_ALREADY_EXISTS 0x402u
#define FS_R_NOT_EMPTY 0x1002u
#define FS_R_UNSUPPORTED_OPEN 0x6402u    // RomFS: any open mode but read
#define FS_R_INVALID 0xe402u             // RomFS: bad handle, offset or argument
#define FS_R_TARGET_NOT_FOUND 0x7d402u
#define FS_R_INVALID_ARGUMENT 0x2ee202u
#define FS_R_INVALID_PATH 0x2ee602u
#define FS_R_OPEN_MODE 0x307402u
#define FS_R_APPEND_REQUIRED 0x2f5e02u

enum { FS_OPEN_READ = 1, FS_OPEN_WRITE = 2, FS_OPEN_APPEND = 4 };   // nn::fs::OpenMode
enum { FS_ENTRY_DIRECTORY = 0, FS_ENTRY_FILE = 1 };                 // nn::fs::DirectoryEntryType

// ---- romfs.c
// Handles 0x1_0000_0000 + j; host storage uses 0x2_0000_0000 + j (files) and 0x3_0000_0000 + j (directories).
#define ROMFS_HANDLE_BASE 0x100000000ull
void romfs_open_image(void);                             // open before graphics startup; fs_init loads the index and overlays
int romfs_owns(const char *path);                          // path is on the mounted RomFS ("<mount>:...")
int romfs_handle(uint64_t h);                              // h is a RomFS file handle
uint32_t romfs_mount(const char *name);                    // nn::fs::MountRom; a second name is fatal
uint32_t romfs_open(uint64_t *out, const char *path, int mode);
uint32_t romfs_read(uint64_t *out, uint64_t h, int64_t offset, void *buffer, uint64_t size);
uint32_t romfs_size(int64_t *out, uint64_t h);
void romfs_close(uint64_t h);
uint32_t romfs_entry_type(int *out, const char *path);
// The immediate children of a RomFS directory through add (FS_ENTRY_DIRECTORY / FS_ENTRY_FILE, size).
uint32_t romfs_list(const char *path, int (*add)(const char *name, int kind, uint64_t size, void *ctx), void *ctx);
uint64_t romfs_file_count(void);                           // files in the index (fs_init first)
const char *romfs_entry(uint64_t i, uint64_t *offset, uint64_t *size);   // path, image offset and size of index row i
const char *romfs_image_path(void);                        // the romfs.bin path fs_init opened
uint8_t *romfs_entry_read(uint64_t i, uint64_t *size);     // the whole of index row i, malloc'd (host walks of the index)
int romfs_image_identity(uint64_t *size, uint64_t *write_time);   // 0: no image open
size_t romfs_overlay_count(void);
int romfs_patches_loaded(void);                            // a romfs-patches manifest gave at least one file
// The button prompts (input_prompts_folder): the folder loaded at start, NULL for the game's own prompts; whether a
// folder had a manifest at start; and why the folder the settings asked for at start was not loaded (ROMFS_PROMPTS_OK
// for any other folder: it was not asked for). romfs-patches-pc goes by that name also where romfs-patches-pc-<w>x<h>
// stands in for it (romfs.c overlay_init).
enum { ROMFS_PROMPTS_OK, ROMFS_PROMPTS_MISSING, ROMFS_PROMPTS_FAILED, ROMFS_PROMPTS_SHAPE, ROMFS_PROMPTS_MODS,
       ROMFS_PROMPTS_PATCHES };
const char *romfs_prompts(void);
// The voice language apart from the text language (voice_lang.c, REKARINA_VOICE_LANG): drives GameDataMgr's
// OptionData.VoiceLanguage natively through oot_fps.c (set voiceopt); 0 follows the game's own setting.
void voice_lang_init(void);
uint32_t voice_lang_enum_hash(void);
int romfs_prompts_live(void);   // each present: applies a changed prompts setting to the running game where it can
void romfs_hint_pad_layout(int layout);   // input.c: the open gamepad's layout (PAD_LAYOUT_*), for the action hint's places
int romfs_prompts_can_change(void);            // 1: the sets change in the running game (prompts-live.bin)
int romfs_prompts_shown(const char *folder);   // 1: the running game shows that folder's prompts, changed without a restart
int romfs_prompts_ready(const char *folder);
// The settings menu's pictures of the prompt sets (prompt-previews.bin): how many sets the game folder has pictures
// of, with the side of a button's square mask and the buttons per set; a set's masks, one after the other, each
// cell x cell coverage bytes; and the set a prompts folder shows (input_prompts_folder; NULL: the game's own), -1 when
// there is no picture of it.
enum { ROMFS_PREVIEW_BUTTONS = 16 };
int romfs_prompt_previews(int *cell, int *count);
const uint8_t *romfs_prompt_preview_masks(int set);
int romfs_prompt_preview_set(const char *folder);
// The keyboard set's picture shows the keys bound now: every key's mask (prompts-keys.bin; NULL without the file),
// and the mask index of the key on button k of a set's picture, -1 where the set's own mask is the one to show.
enum { ROMFS_PREVIEW_KEYS = 128 };
const uint8_t *romfs_prompt_key_masks(int *cell, int *count);
int romfs_prompt_preview_key(int set, int k);
int romfs_prompts_off(const char *folder);
const char *romfs_overlay(size_t k, const char **file, uint64_t *size, int *is_added);   // RomFS path of overlay k
void fs_trace(const char *what, const char *path);         // REKARINA_TRACE=fs: one line per open

// ---- fs_host.c
int fs_host_has_progress(int slot);
int fs_host_latest_progress(void);
void fs_host_set_profile(const char *dir);                 // profile directory; before the first storage mount only
                                                           // (default rt_config()->profile)
int fs_save_busy(void);                                    // a save is being written: a save file open for writing,
                                                           // or save data changed in the last 3 s and not committed
void fs_save_flush(void);                                  // wait for a save Commit still running in the background;
                                                           // every exit path calls it before the process ends
