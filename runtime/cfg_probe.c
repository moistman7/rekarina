// --probe-cfg: rekarina.ini parsing, the environment override and a missing file (cfg.c). Needs no game files.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "input.h"
#include "input_map.h"
#include "motion_vectors.h"
#include "platform/platform.h"

static int fails;
#define CHECK(c, what) do { if (!(c)) { fails++; fprintf(stderr, "FAIL cfg: %s\n", what); } } while (0)

static unsigned probe_motion_consumers(CfgTaa taa, int wanted) {
    (void)taa;
    return wanted ? CFG_RENDER_UPSCALER : 0;
}

static unsigned probe_fg_consumer(CfgTaa taa, int wanted) {
    (void)taa;
    (void)wanted;
    return CFG_RENDER_FRAMEGEN_MOTION;
}

static unsigned probe_interface_consumer(CfgTaa taa, int wanted) {
    return probe_fg_consumer(taa, wanted) | CFG_RENDER_INTERFACE_COVERAGE;
}
#define EQ(a, b) ((a) && !strcmp((a), (b)))

int probe_cfg(void) {
    char path[1024];
    if (!plat_exe_path(path, sizeof path)) return 1;
    char *cut = plat_last_sep(path);
    if (!cut) return 1;
    strcpy(cut + 1, "cfg-probe.ini");
    static const char ini[] =
        "\xEF\xBB\xBF# comment\r\n"
        "; another comment\r\n"
        "[frame rate]\r\n"
        "  fps60 = 0  \r\n"
        "Realtime=0\r\n"
        "[Display]\r\n"
        "GPU_name = \"My GPU 1\"\r\n"
        "REKARINA_Locale = de-DE\r\n"
        "lang =\r\n"
        "trace = a\r\n"
        "trace = b\r\n"
        "no_such_setting = 1\r\n"
        "no equals sign\r\n"
        "last = no newline at the end";
    plat_setenv("REKARINA_LOCALE", NULL);
    plat_setenv("REKARINA_FPS60", NULL);
    CHECK(plat_file_write(path, ini, sizeof ini - 1), "write the probe ini");
    CHECK(cfg_load(path) == 1, "load the probe ini");
    CHECK(EQ(cfg_get("REKARINA_FPS60"), "0"), "value read, whitespace trimmed, lower-case key");
    CHECK(EQ(cfg_get("REKARINA_REALTIME"), "0"), "key without spaces, mixed case");
    CHECK(EQ(cfg_get("REKARINA_GPU_NAME"), "My GPU 1"), "quoted value, section ignored");
    CHECK(EQ(cfg_get("REKARINA_LOCALE"), "de-DE"), "REKARINA_ prefix accepted in the file");
    CHECK(cfg_get("REKARINA_LANG") == NULL, "empty value is unset");
    CHECK(EQ(cfg_get("REKARINA_TRACE"), "b"), "later line wins");
    CHECK(cfg_get("REKARINA_CLOTH_REALTIME") == NULL, "absent key is NULL");
    CHECK(cfg_get("REKARINA_comment") == NULL, "comment is not a key");
    CHECK(EQ(cfg_get("REKARINA_NO_SUCH_SETTING"), "1"), "unknown key is still readable");
    plat_setenv("REKARINA_FPS60", "1");
    CHECK(EQ(cfg_get("REKARINA_FPS60"), "1"), "environment overrides the file");
    plat_setenv("REKARINA_FPS60", NULL);
    CHECK(EQ(cfg_get("REKARINA_FPS60"), "0"), "file value returns when the variable is removed");
    plat_file_delete(path);
    CHECK(cfg_load(path) == 0, "missing file is not an error");
    CHECK(cfg_get("REKARINA_FPS60") == NULL, "missing file leaves no keys");

    // cfg_set / cfg_save: only changed lines move, comments stay, a missing key goes under its section
    static const char before[] =
        "\xEF\xBB\xBF# my settings\r\n"
        "[frame rate]\r\n"
        "  FPS60 = 1\r\n"
        "; keep this\r\n"
        "\r\n"
        "[input]\r\n"
        "# Gamepad motion controls (0 = off) and vibration (0 = off).\r\n"
        "# motion = 1\r\n"
        "# rumble = 1\r\n"
        "rumble_curve = 0.6\r\n"
        "\r\n"
        "[files]\r\n"
        "romfs_mods = C:\\mods";
    static const char after[] =
        "\xEF\xBB\xBF# my settings\r\n"
        "[frame rate]\r\n"
        "  FPS60 = 0\r\n"
        "; keep this\r\n"
        "\r\n"
        "[input]\r\n"
        "# Gamepad motion controls (0 = off) and vibration (0 = off).\r\n"
        "# motion = 1\r\n"
        "motion = 0\r\n"
        "# rumble = 1\r\n"
        "rumble_curve = 0.6\r\n"
        "rumble_strength = 80\r\n"
        "\r\n"
        "[files]\r\n"
        "romfs_mods = C:\\mods\r\n"
        "\r\n"
        "[audio]\r\n"
        "volume = 40\r\n";
    plat_setenv("REKARINA_MOTION", NULL);
    plat_setenv("REKARINA_VOLUME", NULL);
    plat_setenv("REKARINA_RUMBLE_STRENGTH", NULL);
    CHECK(plat_file_write(path, before, sizeof before - 1), "write the save ini");
    CHECK(cfg_load(path) == 1, "load the save ini");
    unsigned v0 = cfg_version();
    cfg_set("REKARINA_FPS60", "0");
    cfg_set("REKARINA_MOTION", "0");
    cfg_set("REKARINA_RUMBLE_STRENGTH", "80");
    cfg_set("REKARINA_VOLUME", "40");
    CHECK(cfg_version() == v0 + 4, "cfg_set bumps the version");
    CHECK(EQ(cfg_get("REKARINA_FPS60"), "0") && EQ(cfg_get("REKARINA_VOLUME"), "40"), "cfg_set values are read back at once");
    CHECK(cfg_save_to(path) == 1, "cfg_save_to writes");
    uint64_t got_n = 0;
    uint8_t *got = plat_file_read(path, &got_n);
    CHECK(got && got_n == sizeof after - 1 && !memcmp(got, after, got_n),
          "saved file: changed line replaced, comments kept, missing keys under their sections");
    if (got && (got_n != sizeof after - 1 || memcmp(got, after, got_n))) fprintf(stderr, "---- saved:\n%s\n----\n", got);
    free(got);
    CHECK(cfg_save_to(path) == 1, "a save with nothing changed");
    got = plat_file_read(path, &got_n);
    CHECK(got && got_n == sizeof after - 1 && !memcmp(got, after, got_n), "a save with nothing changed keeps the file");
    free(got);
    CHECK(cfg_load(path) == 1 && EQ(cfg_get("REKARINA_FPS60"), "0") && EQ(cfg_get("REKARINA_MOTION"), "0") &&
          EQ(cfg_get("REKARINA_RUMBLE_STRENGTH"), "80") && EQ(cfg_get("REKARINA_VOLUME"), "40") &&
          EQ(cfg_get("REKARINA_ROMFS_MODS"), "C:\\mods"), "the saved file reads back");
    const char *held = cfg_get("REKARINA_VOLUME");
    plat_setenv("REKARINA_VOLUME", "70");
    cfg_set("REKARINA_VOLUME", "90");
    CHECK(cfg_from_env("REKARINA_VOLUME") && EQ(cfg_get("REKARINA_VOLUME"), "70"), "environment wins over cfg_set");
    CHECK(EQ(held, "40"), "a pointer cfg_get handed out stays valid after cfg_set");
    plat_setenv("REKARINA_VOLUME", NULL);
    CHECK(!cfg_from_env("REKARINA_VOLUME") && EQ(cfg_get("REKARINA_VOLUME"), "90"), "cfg_set value returns without the variable");
    plat_file_delete(path);
    CHECK(cfg_save_to(path) == 1, "cfg_save_to creates a missing file");
    got = plat_file_read(path, &got_n);
    CHECK(got && !strcmp((const char *)got, "[audio]\nvolume = 90\n"), "new file: the section and the key");
    free(got);

#ifdef GAME_OOT
    plat_setenv("REKARINA_TEXTURE_LOD_STRENGTH", NULL);
    plat_setenv("TOTK_TEXTURE_LOD_STRENGTH", NULL);
    cfg_set("REKARINA_TEXTURE_LOD_STRENGTH", "65.5");
    CHECK(cfg_save_to(path) == 1 && cfg_load(path) == 1 && cfg_double("REKARINA_TEXTURE_LOD_STRENGTH") == 65.5,
          "texture mip bias correction retains its fractional value across save and reload");
#endif

    // Defaults table: the menu's "default" marks and Reset
    const CfgMeta *m = cfg_meta("FPS60");
    CHECK(m && EQ(m->def, "1") && m->live, "meta: fps60 defaults to 1 and is live");
    m = cfg_meta("ASYNC_PREEMPT");
    CHECK(m && EQ(m->values, "0|1|2") && !m->live, "meta: async_preempt takes 0|1|2 and needs a restart");
    m = cfg_meta("totk_vk_present");
    CHECK(m && EQ(m->key, "VK_PRESENT") && EQ(m->values, "mailbox|fifo|immediate") && m->live,
          "meta: prefix and case are accepted");
    m = cfg_meta("RUMBLE_STRENGTH");
    CHECK(m && EQ(m->def, "100") && EQ(m->values, "0..300"), "meta: rumble strength 100 in 0..300");
    m = cfg_meta("PERF_OVERLAY");
    CHECK(m && EQ(m->def, "off") && EQ(m->values, "off|fps|full") && m->live, "meta: perf_overlay is live, off by default");
    CHECK(cfg_meta("BIND_A") == NULL && cfg_meta("FPS_COUNTER") == NULL && cfg_meta("NO_SUCH") == NULL,
          "meta: keys the menu does not show are NULL");

    // cfg_reset: the key's active lines leave rekarina.ini, the commented example stays
    static const char rbase[] = "[audio]\n# volume = 100\n[input]\nmotion = 0\n";
    static const char *const vol[] = {"VOLUME"};
    plat_setenv("REKARINA_VOLUME", NULL);
    CHECK(plat_file_write(path, rbase, sizeof rbase - 1), "write the reset ini");
    CHECK(cfg_load(path) == 1, "load the reset ini");
    cfg_set("REKARINA_VOLUME", "40");
    CHECK(cfg_save_to(path) == 1 && cfg_last_save_ok(), "reset: volume saved");
    got = plat_file_read(path, &got_n);
    CHECK(got && strstr((const char *)got, "\nvolume = 40\n"), "reset: the active line is in the file");
    free(got);
    unsigned v1 = cfg_version();
    CHECK(cfg_reset(vol, 1) == 1, "reset: one key reset");
    CHECK(cfg_version() > v1 && cfg_get("REKARINA_VOLUME") == NULL, "reset: the value is gone in memory, the version moved");
    CHECK(cfg_reset(vol, 1) == 0, "reset: a key already reset counts 0");
    CHECK(cfg_save_to(path) == 1, "reset: save");
    got = plat_file_read(path, &got_n);
    CHECK(got && got_n == sizeof rbase - 1 && !memcmp(got, rbase, got_n) && strstr((const char *)got, "# volume = 100"),
          "reset: no active volume line, the commented example and the other keys stay");
    if (got && (got_n != sizeof rbase - 1 || memcmp(got, rbase, got_n))) fprintf(stderr, "---- saved:\n%s\n----\n", got);
    free(got);
    cfg_set("REKARINA_VOLUME", "55");
    CHECK(cfg_reset(vol, 1) == 1 && cfg_get("REKARINA_VOLUME") == NULL, "reset: a value set after a save resets too");
    cfg_set("REKARINA_VOLUME", "60");
    CHECK(cfg_save_to(path) == 1 && EQ(cfg_get("REKARINA_VOLUME"), "60"), "reset then set before the save: the set wins");
    got = plat_file_read(path, &got_n);
    CHECK(got && strstr((const char *)got, "\nvolume = 60\n"), "reset then set: the new line is written");
    free(got);

    // an active line with an empty value has no entry in memory; Reset still removes it from the file
    static const char rempty[] = "[audio]\n# volume = 100\nvolume =\n[input]\nmotion = 0\n";
    CHECK(plat_file_write(path, rempty, sizeof rempty - 1) && cfg_load(path) == 1 && cfg_get("REKARINA_VOLUME") == NULL,
          "reset: load an ini with an empty value");
    CHECK(cfg_reset(vol, 1) == 0 && cfg_save_to(path) == 1, "reset: an empty-valued key saves");
    got = plat_file_read(path, &got_n);
    CHECK(got && got_n == sizeof rbase - 1 && !memcmp(got, rbase, got_n), "reset: the empty-valued line is gone");
    free(got);
    cfg_set("REKARINA_VOLUME", "60");
    CHECK(cfg_save_to(path) == 1, "reset: volume 60 again for the environment case");

    // a key the environment sets is skipped and keeps its file line
    CHECK(cfg_load(path) == 1 && EQ(cfg_get("REKARINA_VOLUME"), "60"), "reset: reload");
    plat_setenv("REKARINA_VOLUME", "70");
    CHECK(cfg_reset(vol, 1) == 0, "reset: an environment-set key is skipped");
    plat_setenv("REKARINA_VOLUME", NULL);
    CHECK(EQ(cfg_get("REKARINA_VOLUME"), "60"), "reset: the skipped key keeps its value");
    CHECK(cfg_save_to(path) == 1, "reset: save after a skip");
    got = plat_file_read(path, &got_n);
    CHECK(got && strstr((const char *)got, "\nvolume = 60\n"), "reset: the skipped key keeps its file line");
    free(got);

    // a failed save: the call fails, cfg_last_save_ok says so, the value stays set and a later save works
    CHECK(cfg_last_save_ok() == 1, "save status: ok after a good save");
    cfg_set("REKARINA_VOLUME", "20");
    char bad[1100];
    snprintf(bad, sizeof bad, "%s", path);
    strcpy(plat_last_sep(bad) + 1, "no-such-dir/cfg-probe.ini");
    CHECK(cfg_save_to(bad) == 0 && cfg_last_save_ok() == 0 && EQ(cfg_get("REKARINA_VOLUME"), "20"),
          "save status: an unwritable path fails, the value stays");
    cfg_set("REKARINA_VOLUME", "21");
    CHECK(plat_file_write(path, rbase, sizeof rbase - 1), "write the read-only ini");
    CHECK(plat_file_set_readonly(path, 1), "set the file read-only");
    CHECK(cfg_save_to(path) == 0 && cfg_last_save_ok() == 0 && EQ(cfg_get("REKARINA_VOLUME"), "21"),
          "save status: a read-only file fails, cfg_get returns the new value");
    plat_file_set_readonly(path, 0);
    CHECK(cfg_save_to(path) == 1 && cfg_last_save_ok() == 1, "save status: the retry after the failure works");
    got = plat_file_read(path, &got_n);
    CHECK(got && strstr((const char *)got, "\nvolume = 21\n"), "save status: the retried save wrote the value");
    free(got);
    plat_file_delete(path);
    plat_setenv("REKARINA_GAME_RES", NULL);
    plat_setenv("TOTK_GAME_RES", NULL);
    cfg_load(path);
    CHECK(!cfg_restart_pending(), "restart pending: clean after load");
    cfg_set("REKARINA_VOLUME", "42");
    CHECK(!cfg_restart_pending(), "restart pending: live setting does not trigger it");
    cfg_set("REKARINA_ANISO", "16");
    CHECK(!cfg_restart_pending(), "changing anisotropic filtering does not request a restart");
    cfg_set("REKARINA_ANISO", "game");
    cfg_set("REKARINA_GAME_RES", "1080");
    CHECK(cfg_restart_pending(), "restart pending: restart-only setting triggers it");
    cfg_set("REKARINA_GAME_RES", "1440");
    CHECK(!cfg_restart_pending(), "restart pending: reverting to boot value clears it");
#ifdef GAME_OOT
    cfg_set("REKARINA_FRAMEGEN", "fsr3");
    CHECK(!cfg_restart_pending(), "frame generation restart is decided by the running renderer");
    cfg_set("REKARINA_FRAMEGEN", "off");
    if (!cfg_from_env("REKARINA_DLAA")) {
        const char *initial_aa = cfg_bool("REKARINA_DLAA") ? "1" : "0";
        cfg_set("REKARINA_DLAA", initial_aa[0] == '1' ? "0" : "1");
        CHECK(!cfg_restart_pending(), "changing DLSS / DLAA does not request a restart");
        cfg_set("REKARINA_DLAA", initial_aa);
    }
    if (!cfg_from_env("REKARINA_TAA")) {
        CfgTaa initial_taa = cfg_taa();
        cfg_set("REKARINA_TAA", initial_taa == CFG_TAA_DLSS ? "fsr1" : "dlss");
        CHECK(!cfg_restart_pending() && cfg_taa() != initial_taa, "changing the AA backend applies without a restart");
        cfg_set("REKARINA_TAA", "bilinear");
        CHECK(cfg_taa() == CFG_TAA_BILINEAR && !cfg_restart_pending(), "bilinear scene scaling is a distinct live choice");
        cfg_set("REKARINA_TAA", initial_taa == CFG_TAA_DLSS ? "dlss" : initial_taa == CFG_TAA_FSR1 ? "fsr1" : "fsr3");
    }
#endif
    CHECK(rumble_motor(0, 0.6f, 1.3f) == 0 && rumble_motor(0.5f, 0.6f, 0) == 0, "rumble: zero amplitude or strength is silent");
    CHECK(rumble_motor(1, 0.6f, 3) == 65535 && rumble_motor(5, 1, 1) == 65535, "rumble: saturates at 65535");
    CHECK(rumble_motor(0.25f, 1, 1) > 16000 && rumble_motor(0.25f, 1, 1) < 16800, "rumble: curve 1, strength 100% is linear");
    CHECK(rumble_motor(0.1f, 0.6f, 1.3f) > rumble_motor(0.1f, 1, 1.3f), "rumble: curve lifts weak values");
    CHECK(rumble_motor(0.3f, 0.6f, 1.3f) > rumble_motor(0.3f, 0.6f, 1), "rumble: strength scales");
    {   // Switch HD rumble data (input.c input_hd_rumble_encode) against the public notes on the format: the neutral value,
        // the ends of each band's frequency range, rows of the amplitude table (high band 2e, low band 0x40 + e / 2 with
        // the half step in bit 7 of byte 2) and the safe maximum
        static const struct { InputVibration v; uint8_t b[4]; const char *what; } t[] = {
            {{0, 160, 0, 320}, {0x00, 0x01, 0x40, 0x40}, "HD rumble: neutral is 00 01 40 40"},
            {{1, 160, 1, 320}, {0x00, 0xc9, 0x40, 0x72}, "HD rumble: amplitude 1.0 is the top step, c8 and 72"},
            {{0, 40.87f, 0, 81.75f}, {0x04, 0x00, 0x01, 0x40}, "HD rumble: 40.87 and 81.75 Hz are the bands' lowest codes"},
            {{0, 626.28f, 0, 1252.57f}, {0xfc, 0x01, 0x7f, 0x40}, "HD rumble: 626.28 and 1252.57 Hz are the bands' highest codes"},
            {{0, 5, 0, 5000}, {0xfc, 0x01, 0x01, 0x40}, "HD rumble: a frequency outside a band takes its nearest end"},
            {{0.117471f, 160, 0.117471f, 320}, {0x00, 0x21, 0x40, 0x48}, "HD rumble: amplitude 0.117471 is step 16"},
            {{0.112491f, 160, 0.112491f, 320}, {0x00, 0x1f, 0xc0, 0x47}, "HD rumble: amplitude 0.112491 is step 15 (half step)"},
            {{0.033442f, 160, 0.033442f, 320}, {0x00, 0x11, 0x40, 0x44}, "HD rumble: amplitude 0.033442 is step 8"},
            {{0.011823f, 160, 0.011823f, 320}, {0x00, 0x05, 0x40, 0x41}, "HD rumble: amplitude 0.011823 is step 2"},
            {{2, 160, NAN, 320}, {0x00, 0x01, 0x40, 0x72}, "HD rumble: over 1.0 is the top step, NaN is silent"},
        };
        for (size_t i = 0; i < sizeof t / sizeof *t; i++) {
            uint8_t b[4];
            input_hd_rumble_encode(&t[i].v, b);
            CHECK(!memcmp(b, t[i].b, 4), t[i].what);
        }
        int last = 0, rising = 1;
        for (int i = 0; i <= 1000; i++) {   // the step never falls as the amplitude grows, and 1.0 is step 100
            uint8_t b[4];
            input_hd_rumble_encode(&(InputVibration){0, 160, i / 1000.0f, 320}, b);
            rising &= b[1] >> 1 >= last;
            last = b[1] >> 1;
        }
        CHECK(rising && last == 100, "HD rumble: the amplitude step rises with the amplitude up to 100");
    }
    // Keyboard and mouse bindings (input_map.c)
    uint8_t dn[KEY_COUNT] = {0};
    int32_t st[2];
    CHECK(input_map_buttons(dn) == 0, "keys: nothing held, no buttons");
    dn[KEY_MOUSE_L] = 1;
    CHECK(input_map_buttons(dn) == 1ull << B_Y, "keys: left mouse button is Y");
    dn[KEY_MOUSE_R] = dn[KEY_Z] = 1;
    CHECK(input_map_buttons(dn) == (1ull << B_Y | 1ull << B_ZL), "keys: right mouse button and Z both press ZL");
    memset(dn, 0, sizeof dn);
    dn[KEY_X] = dn[KEY_TAB] = dn[KEY_M] = dn[KEY_LCTRL] = dn[KEY_3] = 1;
    CHECK(input_map_buttons(dn) == (1ull << B_ZR | 1ull << B_PLUS | 1ull << B_MINUS | 1ull << B_STICK_L | 1ull << B_RIGHT),
          "keys: X ZR, Tab Plus, M Minus, Left Ctrl stick click, 3 D-pad right");
    memset(dn, 0, sizeof dn);
    dn[KEY_G] = 1;
    CHECK(input_map_buttons(dn) == 0 && input_map_held(dn, B_SHAKE) && !input_map_held(dn, B_A),
          "keys: G shakes and presses no button");
    memset(dn, 0, sizeof dn);
    dn[KEY_W] = 1;
    input_map_stick(dn, 0, st);
    CHECK(st[0] == 0 && st[1] == 32767, "keys: W is left stick up");
    dn[KEY_A] = 1;
    input_map_stick(dn, 0, st);
    CHECK(st[0] == -23170 && st[1] == 23170, "keys: W A is a diagonal on the unit circle");
    dn[KEY_W] = dn[KEY_A] = dn[KEY_S] = 1;
    dn[KEY_D] = 1;
    input_map_stick(dn, 0, st);
    CHECK(st[0] == 0 && st[1] == 0, "keys: opposite keys cancel");
    memset(dn, 0, sizeof dn);
    dn[KEY_LEFT] = 1;
    input_map_stick(dn, 1, st);
    CHECK(st[0] == -32767 && st[1] == 0, "keys: left arrow is right stick left");
    input_map_stick(dn, 0, st);
    CHECK(st[0] == 0 && st[1] == 0, "keys: arrows leave the left stick alone");
    input_mouse_stick(30, 0, 1.0 / 60, 1, st);
    CHECK(st[0] == 32767 * 1800 / 3000 && st[1] == 0, "mouse: 30 counts in 1/60 s is 60% right");
    input_mouse_stick(0, -10, 1.0 / 60, 1, st);
    CHECK(st[0] == 0 && st[1] > 0, "mouse: moving up is stick up");
    input_mouse_stick(30, 0, 1.0 / 30, 1, st);
    CHECK(st[0] == 32767 * 900 / 3000, "mouse: the same counts over twice the time are half the deflection");
    input_mouse_stick(15, 0, 1.0 / 60, 2, st);
    CHECK(st[0] == 32767 * 1800 / 3000, "mouse: doubling sensitivity halves the counts needed (15 counts at 2 = 30 at 1)");
    input_mouse_stick(1000, 1000, 1.0 / 60, 1, st);
    CHECK((int64_t)st[0] * st[0] + (int64_t)st[1] * st[1] <= 32768ll * 32768, "mouse: clamped to the unit circle");
    input_mouse_stick(30, 0, 1.0 / 60, 0, st);
    CHECK(st[0] == 0 && st[1] == 0, "mouse: sensitivity 0 is off");
    input_pad_stick(3000, -2000, 0.15f, st);
    CHECK(st[0] == 0 && st[1] == 0, "pad stick: an 11% drift inside a 15% dead zone reads centred");
    input_pad_stick(4000, -3000, 0, st);
    CHECK(st[0] == 4000 && st[1] == -3000, "pad stick: dead zone 0 passes the stick unchanged");
    input_pad_stick(32767, 0, 0.15f, st);
    CHECK(st[0] == 32767 && st[1] == 0, "pad stick: full deflection still reaches full");
    input_pad_stick(-32768, -32768, 0.15f, st);
    CHECK(st[0] < 0 && st[1] < 0 && st[0] >= -32767 && (int64_t)st[0] * st[0] + (int64_t)st[1] * st[1] <= 32768ll * 32768,
          "pad stick: a square gate's corner stays on the unit circle");
    input_pad_stick(0, 32767 / 2 + 1, 0.2f, st);
    CHECK(abs(st[1] - 32767 * 3 / 8) <= 3, "pad stick: half deflection past a 20% zone is (0.5 - 0.2) / 0.8 of full");
    plat_setenv("REKARINA_STICK_DEADZONE", "12");
    plat_setenv("REKARINA_LSTICK_DEADZONE", NULL);
    plat_setenv("REKARINA_RSTICK_DEADZONE", NULL);
    CHECK(fabsf(input_pad_deadzone(0) - 0.12f) < 1e-6f && fabsf(input_pad_deadzone(1) - 0.12f) < 1e-6f,
          "pad dead zone: both sticks follow stick_deadzone while lstick/rstick are unset");
    plat_setenv("REKARINA_RSTICK_DEADZONE", "20");
    CHECK(fabsf(input_pad_deadzone(0) - 0.12f) < 1e-6f && fabsf(input_pad_deadzone(1) - 0.20f) < 1e-6f,
          "pad dead zone: rstick_deadzone sets the right stick alone");
    plat_setenv("REKARINA_LSTICK_DEADZONE", "80");
    CHECK(fabsf(input_pad_deadzone(0) - 0.5f) < 1e-6f, "pad dead zone: clamped to 50%");
    plat_setenv("REKARINA_STICK_DEADZONE", NULL);
    plat_setenv("REKARINA_LSTICK_DEADZONE", NULL);
    plat_setenv("REKARINA_RSTICK_DEADZONE", NULL);
    // Stick camera speed (camera_speed.c): OoT 0.5 to 4, TotK 0.1 to 3, 1 when unset
#ifdef GAME_OOT
    const float sens_lo = 0.5f, sens_hi = 4;
#else
    const float sens_lo = 0.1f, sens_hi = 3;
#endif
    plat_setenv("REKARINA_STICK_SENS", NULL);
    CHECK(cam_speed_wanted() == 1.0f, "stick camera speed: 1 when unset");
    plat_setenv("REKARINA_STICK_SENS", "2.5");
    CHECK(cam_speed_wanted() == 2.5f, "stick camera speed: a value in range is taken as it is");
    plat_setenv("REKARINA_STICK_SENS", "50");
    CHECK(cam_speed_wanted() == sens_hi, "stick camera speed: held to its top");
    plat_setenv("REKARINA_STICK_SENS", "0");
    CHECK(fabsf(cam_speed_wanted() - sens_lo) < 1e-6f, "stick camera speed: held to its bottom");
    plat_setenv("REKARINA_STICK_SENS", NULL);
    CHECK(input_stick_percent(32767, 0) > 99.99 && input_stick_percent(32767, 0) < 100.01 &&
              input_stick_percent(0, 0) == 0 && fabs(input_stick_percent(3000, -4000) - 5000 * 100.0 / 32767) < 1e-9,
          "stick percent: an axis's end is 100%, the centre 0%, round in between");
    // Gamepad layouts (input_map.c)
    CHECK(input_pad_layout_pick("auto", 1) == PAD_LAYOUT_NINTENDO && input_pad_layout_pick(NULL, 0) == PAD_LAYOUT_XBOX,
          "pad layout: automatic is Nintendo for an east button labelled A, Xbox otherwise");
    CHECK(input_pad_layout_pick("Xbox", 1) == PAD_LAYOUT_XBOX && input_pad_layout_pick("nintendo", 0) == PAD_LAYOUT_NINTENDO,
          "pad layout: a set layout wins over the labels");
    CHECK(input_pad_layout_pick("xbox_prompts", 0) == PAD_LAYOUT_NINTENDO &&
              input_pad_layout_pick("Xbox_Prompts", 1) == PAD_LAYOUT_NINTENDO,
          "pad layout: 0.1.9's xbox_prompts keeps the Nintendo positions (only the prompts change, romfs.c overlay_init)");
    // Button prompts (input_map.c)
    CHECK(input_prompts_pick(NULL, "auto") == PROMPTS_NINTENDO && input_prompts_pick("", "xbox") == PROMPTS_NINTENDO &&
              input_prompts_pick("Xbox", "nintendo") == PROMPTS_XBOX && input_prompts_pick("nintendo", "xbox") == PROMPTS_NINTENDO,
          "prompts: the game's own unless set to Xbox");
    CHECK(input_prompts_pick("PC", "nintendo") == PROMPTS_PC && input_prompts_pick("pc", "xbox_prompts") == PROMPTS_PC &&
              EQ(input_prompts_folder(PROMPTS_PC, "nintendo"), "romfs-patches-pc") &&
              EQ(input_prompts_folder(PROMPTS_PC, "xbox"), "romfs-patches-pc") &&
              EQ(input_prompts_folder(PROMPTS_PC, NULL), "romfs-patches-pc"),
          "prompts: pc shows the keyboard and mouse whatever the pad layout");
    CHECK(input_prompts_pick(NULL, "xbox_prompts") == PROMPTS_XBOX && input_prompts_pick("nintendo", "xbox_prompts") == PROMPTS_NINTENDO,
          "prompts: 0.1.9's pad_layout = xbox_prompts means Xbox while prompts is unset");
    CHECK(!input_prompts_folder(PROMPTS_NINTENDO, "nintendo") && !input_prompts_folder(PROMPTS_NINTENDO, "xbox"),
          "prompts: the game's own need no folder");
    CHECK(EQ(input_prompts_folder(PROMPTS_XBOX, "nintendo"), "romfs-patches-xbox") &&
              EQ(input_prompts_folder(PROMPTS_XBOX, "xbox_prompts"), "romfs-patches-xbox") &&
              EQ(input_prompts_folder(PROMPTS_XBOX, "xbox"), "romfs-patches-xbox-label") &&
              EQ(input_prompts_folder(PROMPTS_XBOX, "auto"), "romfs-patches-xbox-label") &&
              EQ(input_prompts_folder(PROMPTS_XBOX, NULL), "romfs-patches-xbox-label"),
          "prompts: the face buttons by position for the Nintendo layout, by label for xbox and auto");
    // prompts = auto (input.c input_prompts_wanted): the device's kind and the layout in effect pick the folder
    CHECK(input_prompts_pick("auto", "xbox") == PROMPTS_AUTO && input_prompts_pick("Auto", NULL) == PROMPTS_AUTO &&
              EQ(input_prompts_folder(PROMPTS_AUTO, "auto"), "romfs-patches-xbox-label") &&
              EQ(input_prompts_folder(PROMPTS_AUTO, "nintendo"), "romfs-patches-xbox"),
          "prompts: auto before any device is known is the Xbox guess for the layout setting");
    CHECK(EQ(input_prompts_device_folder(PROMPTS_XBOX, PAD_LAYOUT_XBOX), "romfs-patches-xbox-label") &&
              EQ(input_prompts_device_folder(PROMPTS_XBOX, PAD_LAYOUT_NINTENDO), "romfs-patches-xbox") &&
              EQ(input_prompts_device_folder(PROMPTS_PS, PAD_LAYOUT_XBOX), "romfs-patches-playstation-label") &&
              EQ(input_prompts_device_folder(PROMPTS_PS, PAD_LAYOUT_NINTENDO), "romfs-patches-playstation") &&
              !input_prompts_device_folder(PROMPTS_NINTENDO, PAD_LAYOUT_NINTENDO) &&
              EQ(input_prompts_device_folder(PROMPTS_NINTENDO, PAD_LAYOUT_XBOX), "romfs-patches-xbox") &&
              EQ(input_prompts_device_folder(PROMPTS_PC, PAD_LAYOUT_NINTENDO), "romfs-patches-pc") &&
              EQ(input_prompts_device_folder(PROMPTS_PC, PAD_LAYOUT_XBOX), "romfs-patches-pc"),
          "prompts: auto names each face button by the label of the pad's button that presses it");
    CHECK(input_prompts_of_folder(NULL) == PROMPTS_NINTENDO && input_prompts_of_folder("romfs-patches-pc") == PROMPTS_PC &&
              input_prompts_of_folder("romfs-patches-xbox") == PROMPTS_XBOX &&
              input_prompts_of_folder("romfs-patches-xbox-label") == PROMPTS_XBOX &&
              input_prompts_of_folder("romfs-patches-playstation-label") == PROMPTS_PS &&
              input_prompts_of_folder("romfs-patches-rekarina") == PROMPTS_RK,
          "prompts: a folder's set");
    // romfs-patches-xbox shows A as B (tests/romfs_prompts_test.py checks the folders' contents). It is right exactly
    // where the layout puts A on the east button, which an Xbox pad labels B.
    input_map_reset();
    static const char *const layouts[] = {"nintendo", "xbox", "auto", "xbox_prompts"};
    for (int i = 0; i < 4; i++) {
        int a_east = input_map_pad_buttons((uint8_t[PAD_COUNT]){[PAD_EAST] = 1}, input_pad_layout_pick(layouts[i], 0)) == 1ull << B_A;
        CHECK(a_east == EQ(input_prompts_folder(PROMPTS_XBOX, layouts[i]), "romfs-patches-xbox"),
              "prompts: the folder names A as B exactly when the layout puts A on the button labelled B");
    }
    uint8_t pd[PAD_COUNT] = {0};
    pd[PAD_SOUTH] = pd[PAD_NORTH] = 1;
    CHECK(input_map_pad_buttons(pd, PAD_LAYOUT_NINTENDO) == (1ull << B_B | 1ull << B_X) &&
          input_map_pad_buttons(pd, PAD_LAYOUT_XBOX) == (1ull << B_A | 1ull << B_Y),
          "pad layout: south and north are B and X by position, A and Y on Xbox");
    pd[PAD_SOUTH] = pd[PAD_NORTH] = 0;
    pd[PAD_START] = pd[PAD_ZR] = 1;
    CHECK(input_map_pad_buttons(pd, PAD_LAYOUT_XBOX) == (1ull << B_PLUS | 1ull << B_ZR), "pad layout: Xbox keeps the other buttons");
    CHECK(input_map_bind("A", "pad:west") && input_map_pad_buttons((uint8_t[PAD_COUNT]){[PAD_WEST] = 1}, PAD_LAYOUT_XBOX) ==
                                                 (1ull << B_A | 1ull << B_X),
          "pad layout: a bind line adds its physical button in every layout");
    CHECK(input_map_pad_buttons((uint8_t[PAD_COUNT]){[PAD_SOUTH] = 1}, PAD_LAYOUT_XBOX) == 0,
          "pad layout: a bind line replaces the layout's own button");
    input_map_reset();
    CHECK(input_map_pad_buttons((uint8_t[PAD_COUNT]){[PAD_RPADDLE1] = 1}, PAD_LAYOUT_XBOX) == 0,
          "pad layout: a back paddle is unbound by default");
    CHECK(input_map_bind("ZL", "pad:default, pad:rpaddle1") &&
              input_map_pad_buttons((uint8_t[PAD_COUNT]){[PAD_RPADDLE1] = 1}, PAD_LAYOUT_NINTENDO) == 1ull << B_ZL &&
              input_map_pad_buttons((uint8_t[PAD_COUNT]){[PAD_ZL] = 1}, PAD_LAYOUT_NINTENDO) == 1ull << B_ZL,
          "bind: pad:rpaddle1 adds a back paddle, pad:default keeps the trigger");
    input_map_reset();
    // Bind lines for stick directions and default parts; the remapping rows' edits (input_map.c)
    memset(dn, 0, sizeof dn);
    dn[KEY_I] = 1;
    CHECK(input_map_bind("LSTICK_UP", "key:i, pad:south") == 1, "bind: a stick direction takes keys, skips pad entries");
    input_map_stick(dn, 0, st);
    CHECK(st[1] == 32767, "bind: I moves forward after bind_LSTICK_UP = key:i");
    dn[KEY_I] = 0, dn[KEY_W] = 1;
    input_map_stick(dn, 0, st);
    CHECK(st[1] == 0, "bind: W no longer moves forward");
    CHECK(input_map_bind("A", "key:default, pad:north") == 1, "bind: key:default with a pad entry");
    memset(dn, 0, sizeof dn);
    dn[KEY_E] = 1;
    CHECK(input_map_buttons(dn) == 1ull << B_A &&
              input_map_pad_buttons((uint8_t[PAD_COUNT]){[PAD_NORTH] = 1}, PAD_LAYOUT_XBOX) == (1ull << B_A | 1ull << B_Y),
          "bind: key:default keeps E, pad:north adds the top button in every layout");
    CHECK(input_map_bind("B", "key:j, pad:default") == 1 &&
              input_map_pad_buttons((uint8_t[PAD_COUNT]){[PAD_EAST] = 1}, PAD_LAYOUT_XBOX) == 1ull << B_B &&
              input_map_pad_buttons((uint8_t[PAD_COUNT]){[PAD_SOUTH] = 1}, PAD_LAYOUT_NINTENDO) == 1ull << B_B,
          "bind: pad:default keeps each layout's own button");
    input_map_reset();
    InputMap mp = input_map;
    char line[256];
    input_map_line(&mp, B_ZR, line, sizeof line);
    CHECK(!line[0], "edit: an action at its defaults needs no line");
    CHECK(input_map_set_key(&mp, B_A, 0, KEY_F) == B_ZR, "edit: F taken from ZR");
    input_map_line(&mp, B_A, line, sizeof line);
    CHECK(!strcmp(line, "key:f, pad:default"), "edit: A's line is key:f, pad:default");
    input_map_line(&mp, B_ZR, line, sizeof line);
    CHECK(!strcmp(line, "key:x, pad:default"), "edit: ZR keeps X");
    CHECK(input_map_set_key(&mp, B_ZL, 1, KEY_NONE) == -1, "edit: clearing a key takes from no one");
    uint8_t ks[8];
    CHECK(input_map_keys_of(&mp, B_ZL, ks, 8) == 1 && ks[0] == KEY_MOUSE_R, "edit: ZL keeps its first key only");
    mp = input_map;
    CHECK(input_map_set_pad(&mp, B_X, PAD_LAYOUT_XBOX, PAD_SOUTH) == B_A, "edit: south swaps X with A (Xbox layout)");
    uint8_t ps[8];
    CHECK(input_map_pads_of(&mp, B_A, PAD_LAYOUT_XBOX, ps, 8) == 1 && ps[0] == PAD_WEST, "edit: A takes X's west button");
    CHECK(input_map_pads_of(&mp, B_A, PAD_LAYOUT_NINTENDO, ps, 8) == 1 && ps[0] == PAD_WEST &&
              input_map_pads_of(&mp, B_B, PAD_LAYOUT_NINTENDO, ps, 8) == 1 && ps[0] == PAD_EAST,
          "edit: the face buttons are pinned in every layout after a gamepad edit");
    input_map_line(&mp, B_X, line, sizeof line);
    CHECK(!strcmp(line, "key:default, pad:south"), "edit: X's line is key:default, pad:south");
    input_map_set_pad(&mp, B_X, PAD_LAYOUT_XBOX, PAD_WEST);   // and back
    input_map_line(&mp, B_A, line, sizeof line);
    CHECK(!line[0] && input_map_pads_of(&mp, B_A, PAD_LAYOUT_NINTENDO, ps, 8) == 1 && ps[0] == PAD_EAST,
          "edit: swapping back restores every layout's defaults");
    mp = input_map;
    input_map_set_key(&mp, B_SL_UP, 0, KEY_NONE);
    input_map_line(&mp, B_SL_UP, line, sizeof line);
    CHECK(!strcmp(line, "none"), "edit: a stick direction without keys is none");
    // defaults live in the CfgMeta table: cfg_value and the typed accessors fall back to it
    cfg_load("cfg-probe-missing.ini");
    plat_setenv("REKARINA_FPS60", NULL);
    plat_setenv("REKARINA_MOUSE_SENS", NULL);
    plat_setenv("REKARINA_FREE_CORES", NULL);
    CHECK(cfg_bool("REKARINA_FPS60") == 1, "default: unset FPS60 is on");
    CHECK(EQ(cfg_value("REKARINA_VK_PRESENT"), "mailbox") && cfg_int("REKARINA_SPLIT_DRAWS") == 16, "default: text and integer");
    plat_setenv("REKARINA_MOUSE_SENS", "2.5");
    CHECK(cfg_double("REKARINA_MOUSE_SENS") == 2.5, "double: the environment value");
    plat_setenv("REKARINA_MOUSE_SENS", "fast");
    CHECK(cfg_double("REKARINA_MOUSE_SENS") == 1.0, "double: a value that does not parse gives the default");
    plat_setenv("REKARINA_MOUSE_SENS", NULL);
    cfg_set("REKARINA_MOUSE_SENS", "3");
    CHECK(cfg_double("REKARINA_MOUSE_SENS") == 3.0 && cfg_int("REKARINA_MOUSE_SENS") == 3, "a cfg_set value beats the default");
    CHECK(cfg_bool("REKARINA_FREE_CORES") == 0, "free_cores unset: off");
    plat_setenv("REKARINA_FREE_CORES", "0");
    CHECK(cfg_bool("REKARINA_FREE_CORES") == 0, "free_cores 0 gives the same as unset");
    plat_setenv("REKARINA_FREE_CORES", "1");
    CHECK(cfg_bool("REKARINA_FREE_CORES") == 1, "free_cores 1: on");
    plat_setenv("REKARINA_FREE_CORES", NULL);
    cfg_render_publish();
    const CfgRenderState *first = cfg_render_acquire();
    unsigned first_revision = cfg_version();
    cfg_set("REKARINA_MOUSE_SENS", "4");
    CHECK(cfg_int("REKARINA_MOUSE_SENS") == 4, "the editor sees its change before present");
    const CfgRenderState *before_present = cfg_render_acquire();
    cfg_render_use(before_present);
    CHECK(first && before_present == first && cfg_version() == first_revision && cfg_int("REKARINA_MOUSE_SENS") == 3,
          "replay sees the previous revision until present");
    cfg_render_use(NULL);
    cfg_render_release(before_present);
    cfg_render_publish();
    const CfgRenderState *second = cfg_render_acquire();
    const char *reset_mouse[] = {"MOUSE_SENS"};
    CHECK(cfg_reset(reset_mouse, 1) == 1, "reset stages removal of a published value");
    cfg_render_publish();
    const CfgRenderState *third = cfg_render_acquire();
    cfg_render_use(third);
    CHECK(cfg_version() > first_revision && cfg_double("REKARINA_MOUSE_SENS") == 1.0, "present publishes the reset");
    cfg_render_use(NULL);
    cfg_set("REKARINA_MOUSE_SENS", "5");
    cfg_render_publish();
    const CfgRenderState *full = cfg_render_acquire();
    CHECK(full == third, "present does not overwrite a pinned snapshot when the ring is full");
    cfg_render_release(full);
    cfg_render_use(first);
    CHECK(cfg_version() == first_revision && cfg_int("REKARINA_MOUSE_SENS") == 3, "a pinned snapshot stays immutable");
    cfg_render_use(second);
    CHECK(cfg_int("REKARINA_MOUSE_SENS") == 4, "a second pinned snapshot keeps its own value");
    cfg_render_use(NULL);
    cfg_render_release(first);
    cfg_render_release(second);
    cfg_render_release(third);
    cfg_render_publish();
    const CfgRenderState *latest = cfg_render_acquire();
    cfg_render_use(latest);
    CHECK(cfg_int("REKARINA_MOUSE_SENS") == 5, "the next present publishes after a slot is released");
    cfg_render_use(NULL);
    cfg_render_release(latest);
    plat_setenv("REKARINA_SHARPNESS", NULL);
    plat_setenv("TOTK_SHARPNESS", NULL);
#ifdef GAME_OOT
    const char *reset_sharpness[] = {"SHARPNESS"};
    cfg_reset(reset_sharpness, 1);
    cfg_render_publish();
    const CfgRenderState *default_sharpness = cfg_render_acquire();
    cfg_set("REKARINA_SHARPNESS", "65");
    cfg_render_publish();
    const CfgRenderState *manual_sharpness = cfg_render_acquire();
    cfg_render_use(default_sharpness);
    CHECK(cfg_scene_sharpness(1) == 0 && cfg_scene_sharpness(0) == 80,
          "a pinned default frame skips RCAS on NDSR and uses 80% on replacement scaling");
    cfg_render_use(manual_sharpness);
    CHECK(cfg_scene_sharpness(1) == 0 && cfg_scene_sharpness(0) == 65,
          "NDSR skips RCAS even when a non-default strength is stored");
    cfg_render_use(NULL);
    cfg_render_release(default_sharpness);
    cfg_render_release(manual_sharpness);
    cfg_set("REKARINA_SHARPNESS", "0");
    CHECK(cfg_scene_sharpness(1) == 0 && cfg_scene_sharpness(0) == 0,
          "zero strength disables sharpening on replacement scaling too");
#endif
    cfg_set("REKARINA_SHARPNESS", "80");
#ifdef GAME_OOT
    plat_setenv("REKARINA_TEXTURE_LOD_STRENGTH", NULL);
    plat_setenv("TOTK_TEXTURE_LOD_STRENGTH", NULL);
    cfg_set("REKARINA_TEXTURE_LOD_STRENGTH", "65");
#endif
    cfg_render_publish();
    cfg_render_frame_begin();
    const CfgRenderState *frame = cfg_render_frame_acquire();
    cfg_render_use(frame);
    const unsigned projection_revision = cfg_render_values()->projection_revision;
    const int old_dlaa = cfg_render_values()->dlaa;
    const int sharpness = cfg_render_values()->sharpness;
    cfg_render_use(NULL);
    cfg_set("REKARINA_SHARPNESS", sharpness == 33 ? "34" : "33");
    cfg_render_publish();
    const CfgRenderState *same_frame = cfg_render_frame_acquire();
    cfg_render_use(same_frame);
    CHECK(same_frame == frame && cfg_render_values()->sharpness == sharpness,
          "publication cannot change the active guest frame's typed values");
    cfg_render_use(NULL);
    cfg_render_release(same_frame);
    cfg_render_frame_begin();
    const CfgRenderState *next_frame = cfg_render_frame_acquire();
    cfg_render_use(next_frame);
    CHECK(cfg_render_values()->sharpness == (sharpness == 33 ? 34 : 33) &&
          cfg_render_values()->projection_revision == projection_revision,
          "a non-projection setting applies next frame without invalidating projection");
#ifdef GAME_OOT
    CHECK(cfg_int("REKARINA_TEXTURE_LOD_STRENGTH") == 65, "changing RCAS leaves the texture LOD setting alone");
#endif
    cfg_render_use(NULL);
    cfg_render_release(next_frame);
    if (!cfg_from_env("REKARINA_DLAA")) {
        cfg_set("REKARINA_DLAA", old_dlaa ? "0" : "1");
        cfg_render_publish();
        cfg_render_frame_begin();
        next_frame = cfg_render_frame_acquire();
        cfg_render_use(next_frame);
        CHECK(cfg_render_values()->dlaa != old_dlaa && cfg_render_values()->projection_revision != projection_revision,
              "a changed AA mode invalidates projection at the next frame");
        cfg_render_use(NULL);
        cfg_render_release(next_frame);
    }
    if (!cfg_from_env("REKARINA_TAA")) {
        const CfgRenderState *before = cfg_render_frame_acquire();
        cfg_render_use(before);
        CfgTaa backend = cfg_taa();
        unsigned rev = cfg_render_values()->projection_revision;
        cfg_render_use(NULL);
        cfg_set("REKARINA_TAA", backend == CFG_TAA_FSR1 ? "dlss" : "fsr1");
        cfg_render_publish();
        cfg_render_use(before);
        CHECK(cfg_taa() == backend && cfg_render_values()->projection_revision == rev, "active frame retains its AA backend");
        cfg_render_use(NULL);
        cfg_render_release(before);
        cfg_render_frame_begin();
        next_frame = cfg_render_frame_acquire();
        cfg_render_use(next_frame);
        CHECK(cfg_taa() != backend && cfg_render_values()->projection_revision != rev, "next frame applies the AA backend and invalidates projection");
        cfg_render_use(NULL);
        cfg_render_release(next_frame);
    }
    if (!cfg_from_env("REKARINA_TAA") && !cfg_from_env("REKARINA_FSR_VERSION")) {
        cfg_set("REKARINA_TAA", "fsr3");
        cfg_set("REKARINA_FSR_VERSION", "3");
        cfg_render_publish();
        cfg_render_frame_begin();
        const CfgRenderState *before = cfg_render_frame_acquire();
        cfg_render_use(before);
        unsigned rev = cfg_render_values()->projection_revision;
        cfg_render_use(NULL);
        cfg_set("REKARINA_FSR_VERSION", "4");
        cfg_render_publish();
        cfg_render_use(before);
        CHECK(EQ(cfg_value("REKARINA_FSR_VERSION"), "3") && cfg_render_values()->projection_revision == rev,
              "active frame retains its FSR generation");
        cfg_render_use(NULL);
        cfg_render_release(before);
        cfg_render_frame_begin();
        next_frame = cfg_render_frame_acquire();
        cfg_render_use(next_frame);
        CHECK(cfg_taa() == CFG_TAA_FSR3 && EQ(cfg_value("REKARINA_FSR_VERSION"), "4") &&
              cfg_render_values()->projection_revision != rev && !cfg_restart_pending(),
              "a live change of FSR generation invalidates projection without switching backend");
        cfg_render_use(NULL);
        cfg_render_release(next_frame);
    }
    cfg_render_release(frame);
    if (!cfg_from_env("REKARINA_TAA") && !cfg_from_env("REKARINA_DLAA") && !cfg_from_env("REKARINA_MOTION_VECTORS") &&
        !cfg_from_env("REKARINA_DLAA_MV") && !cfg_from_env("REKARINA_DLAA_REVERSE_ONLY") && !cfg_from_env("REKARINA_SKIN_MOTION")) {
        cfg_set("REKARINA_DLAA", "1");
        cfg_set("REKARINA_TAA", "fsr3");
        cfg_set("REKARINA_MOTION_VECTORS", "hybrid");
        cfg_set("REKARINA_DLAA_REVERSE_ONLY", "0");
        cfg_set("REKARINA_SKIN_MOTION", "1");
        cfg_render_publish_present(probe_motion_consumers);
        cfg_render_frame_begin();
        const CfgRenderState *recorded = cfg_render_frame_acquire();
        const CfgRenderValues *old = cfg_render_values_of(recorded);
        CHECK(old->motion_models && old->motion_mode == MOTION_VECTORS_HYBRID && !old->motion_reverse_only,
              "temporal reconstruction consumes model motion in hybrid mode");
        cfg_set("REKARINA_TAA", "fsr1");
        cfg_set("REKARINA_DLAA_REVERSE_ONLY", "1");
        cfg_render_publish_present(probe_motion_consumers);
        const CfgRenderState *same = cfg_render_frame_acquire();
        CHECK(same == recorded && old->motion_models && !old->motion_reverse_only,
              "publication cannot change the active recording's motion policy");
        cfg_render_release(same);
        cfg_render_frame_begin();
        const CfgRenderState *spatial = cfg_render_frame_acquire();
        CHECK(!cfg_render_values_of(spatial)->motion_models && cfg_render_values_of(spatial)->motion_reverse_only,
              "FSR1 does not consume model motion on its own");
        cfg_render_publish_present(probe_fg_consumer);
        cfg_render_frame_begin();
        const CfgRenderState *fg = cfg_render_frame_acquire();
        CHECK(cfg_render_values_of(fg)->motion_models && !cfg_render_values_of(spatial)->motion_models &&
              old->motion_models && !old->motion_reverse_only,
              "FG readiness publishes without an INI edit and leaves queued snapshots intact");
        cfg_set("REKARINA_MOTION_VECTORS", "camera");
        cfg_render_publish_present(probe_interface_consumer);
        cfg_render_frame_begin();
        const CfgRenderState *blocked = cfg_render_frame_acquire();
        CHECK(blocked == fg && cfg_render_values_of(blocked)->motion_models,
              "a full settings ring defers publication without overwriting a pinned frame");
        cfg_render_release(blocked);
        cfg_render_release(recorded);
        cfg_render_publish_present(probe_interface_consumer);
        cfg_render_frame_begin();
        const CfgRenderState *retried = cfg_render_frame_acquire();
        CHECK(cfg_render_values_of(retried)->motion_mode == MOTION_VECTORS_CAMERA &&
              !cfg_render_values_of(retried)->motion_models && cfg_render_values_of(fg)->motion_models,
              "the next present publishes the deferred settings once a slot retires");
        CHECK((cfg_render_values_of(retried)->consumers & CFG_RENDER_INTERFACE_COVERAGE) &&
              !(cfg_render_values_of(fg)->consumers & CFG_RENDER_INTERFACE_COVERAGE),
              "interface capture follows the published frame, not the latest backend readiness");
        cfg_render_release(retried);
        cfg_render_release(fg);
        cfg_render_release(spatial);

        cfg_set("REKARINA_MOTION_VECTORS", "hybrid");
        cfg_set("REKARINA_SKIN_MOTION", "0");
        cfg_render_publish_present(probe_interface_consumer);
        cfg_render_frame_begin();
        const CfgRenderState *no_skin = cfg_render_frame_acquire();
        const CfgRenderValues *values = cfg_render_values_of(no_skin);
        CHECK(values->motion_models && (values->consumers & CFG_RENDER_FRAMEGEN_MOTION) && cfg_restart_pending(),
              "a pending skin-motion restart cannot change the current run's model policy");
        cfg_render_release(no_skin);
    }
    // 0.1.9's pad_layout = xbox_prompts (cfg.c migrate): read as the Nintendo layout with the Xbox prompts, saved so
    plat_setenv("REKARINA_PAD_LAYOUT", NULL);
    plat_setenv("REKARINA_PROMPTS", NULL);
    static const char old_layout[] = "[input]\r\npad_layout = xbox_prompts\r\n";
    CHECK(plat_file_write(path, old_layout, sizeof old_layout - 1) && cfg_load(path) == 1 &&
              EQ(cfg_get("REKARINA_PAD_LAYOUT"), "nintendo") && EQ(cfg_get("REKARINA_PROMPTS"), "xbox"),
          "0.1.9's pad_layout = xbox_prompts reads as nintendo with prompts = xbox");
    CHECK(cfg_save_to(path) == 1, "the read pad layout and prompts save");
    uint64_t saved_n = 0;
    char *saved = (char *)plat_file_read(path, &saved_n);
    CHECK(saved && strstr(saved, "pad_layout = nintendo") && strstr(saved, "prompts = xbox") && !strstr(saved, "xbox_prompts"),
          "the next save writes the pad layout and the prompts");
    free(saved);
    static const char old_layout_own[] = "[input]\r\npad_layout = xbox_prompts\r\nprompts = nintendo\r\n";
    CHECK(plat_file_write(path, old_layout_own, sizeof old_layout_own - 1) && cfg_load(path) == 1 &&
              EQ(cfg_get("REKARINA_PAD_LAYOUT"), "nintendo") && EQ(cfg_get("REKARINA_PROMPTS"), "nintendo"),
          "a prompts line of the file's own wins over 0.1.9's pad_layout = xbox_prompts");
#ifdef GAME_OOT
    plat_setenv("REKARINA_HDR_TONEMAP", NULL);
    plat_setenv("TOTK_HDR_TONEMAP", NULL);
    static const char old_mapper[] = "[display]\r\nhdr_tonemap = renodrt\r\n";
    CHECK(plat_file_write(path, old_mapper, sizeof old_mapper - 1) && cfg_load(path) == 1 &&
              EQ(cfg_value("REKARINA_HDR_TONEMAP"), "psychov35"),
          "retired HDR mapper in an old config resolves to PsychoV35");
    cfg_set("REKARINA_HDR_TONEMAP", "vanilla");
    plat_setenv("REKARINA_HDR_TONEMAP", "renodrt");
    CHECK(EQ(cfg_value("REKARINA_HDR_TONEMAP"), "psychov35"), "retired HDR environment override resolves to PsychoV35");
    plat_setenv("REKARINA_HDR_TONEMAP", NULL);
    CHECK(EQ(cfg_value("REKARINA_HDR_TONEMAP"), "vanilla"), "Vanilla HDR mapper remains selectable");
    static const char native_hdr[] = "[display]\r\nhdr = native\r\nhdr_tonemap = psychov35\r\n";
    CHECK(plat_file_write(path, native_hdr, sizeof native_hdr - 1) && cfg_load(path) == 1 &&
          EQ(cfg_get("REKARINA_HDR"), "1") && EQ(cfg_get("REKARINA_HDR_TONEMAP"), "vanilla"),
          "old Native HDR selects HDR and Vanilla even if a PsychoV35 preference was saved");
    cfg_set("REKARINA_HDR_TONEMAP", "psychov35");
    CHECK(EQ(cfg_value("REKARINA_HDR_TONEMAP"), "psychov35"), "migrated HDR can select PsychoV35 live");
    plat_setenv("REKARINA_HDR", "native");
    CHECK(EQ(cfg_value("REKARINA_HDR"), "1") && EQ(cfg_value("REKARINA_HDR_TONEMAP"), "vanilla"),
          "old HDR environment override still forces Vanilla HDR");
    plat_setenv("REKARINA_HDR", NULL);
#endif
    plat_file_delete(path);
    cfg_load("cfg-probe-missing.ini");
    int listed = 0;
    for (int i = 0; cfg_meta_at(i); i++) {
        const CfgMeta *m = cfg_meta_at(i);
        if (!*m->def) continue;
        listed++;
        char name[96];
        snprintf(name, sizeof name, "TOTK_%s", m->key);   // the old prefix is read too (cfg.c env_of)
        plat_setenv(name, "");
        snprintf(name, sizeof name, "REKARINA_%s", m->key);
        plat_setenv(name, "");
        CHECK(EQ(cfg_value(name), m->def), "cfg_value returns the table default");
        int ok = 0;
        if (!m->values) ok = 1;
        else if (strstr(m->values, "..")) {
            // strtod: ranges such as 0.2..0.9 have fractional ends
            double lo = strtod(m->values, NULL), hi = strtod(strstr(m->values, "..") + 2, NULL);
            double d = strtod(m->def, NULL);
            ok = d >= lo && d <= hi;
        } else {
            size_t dl = strlen(m->def);
            for (const char *v = m->values; *v;) {
                size_t n = strcspn(v, "|");
                ok |= n == dl && !strncmp(v, m->def, n);
                v += n + (v[n] == '|');
            }
        }
        if (!ok) fprintf(stderr, "FAIL cfg: default '%s' of %s is not one of its values '%s'\n", m->def, m->key, m->values), fails++;
    }
    CHECK(listed >= 30, "the defaults table was walked");
    // cfg_meta answers with the first row of a key: a second row would be dead, and Reset would visit the key twice
    for (int i = 0; cfg_meta_at(i); i++)
        for (int j = i + 1; cfg_meta_at(j); j++)
            if (!strcmp(cfg_meta_at(i)->key, cfg_meta_at(j)->key))
                fprintf(stderr, "FAIL cfg: %s has two rows in the defaults table\n", cfg_meta_at(i)->key), fails++;
    printf("probe-cfg: %s\n", fails ? "FAIL" : "ok");
    return fails ? 1 : 0;
}
