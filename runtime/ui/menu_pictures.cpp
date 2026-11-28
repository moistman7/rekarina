// Pictures in the menu's description pane (menu_pictures.h). Drawn in the menu's own colours (ui/option_rows.cpp): dark
// insets with a khaki edge, cream values, dim captions, amber for what is lit or measured. No game artwork.
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "imgui.h"
#include "ui/menu_pictures.h"
#include "ui/option_rows.h"

extern "C" {
#include "cfg.h"
#include "input.h"
#include "input_map.h"
#include "perfmon.h"
#include "platform/platform.h"
#include "ui_text.h"
void res_sizes(unsigned out[2], unsigned scene[2]);
double res_fixed_scene_share(void);
int vk_dlaa_available(void);
int vk_dlaa_fsr1(void);
int vk_dlaa_fsr3(void);
double gfx_output_fps(void);
double frame_step_limit_hz(void);
#ifdef GAME_OOT
void gfx_fit_rect(int cw, int ch, int dw, int dh, int out[4]);
#endif
}

static const ImU32 kCream = IM_COL32(246, 239, 214, 255), kDim = IM_COL32(201, 192, 165, 255);
static const ImU32 kLabel = IM_COL32(176, 162, 128, 255), kInset = IM_COL32(30, 26, 18, 255);
static const ImU32 kEdge = IM_COL32(140, 118, 72, 255), kAmber = IM_COL32(217, 149, 31, 255);
static const ImU32 kAmberLow = IM_COL32(196, 130, 27, 255), kInk = IM_COL32(30, 22, 12, 255);
static const ImU32 kLine = IM_COL32(110, 90, 58, 200);

static float text_w(float size, const char *t) { return ImGui::GetFont()->CalcTextSizeA(size, FLT_MAX, 0, t).x; }

// Text at size, smaller (down to 65 %) where it would not fit in w; centred in [x, x + w] or from x.
static void fit_text(ImDrawList *dl, float x, float y, float w, float size, ImU32 col, const char *t, bool centred) {
    float tw = text_w(size, t);
    if (tw > w && w > 0) {
        size = fmaxf(size * w / tw, size * 0.65f);
        tw = text_w(size, t);
    }
    ImVec4 clip(x, y - size, x + w, y + size * 2);
    dl->AddText(ImGui::GetFont(), size, ImVec2(floorf(centred && tw < w ? x + (w - tw) * 0.5f : x), floorf(y)), col, t,
                NULL, 0, &clip);
}

static void amber_fill(ImDrawList *dl, ImVec2 a, ImVec2 b) {
    if (b.x > a.x && b.y > a.y) dl->AddRectFilledMultiColor(a, b, kAmber, kAmber, kAmberLow, kAmberLow);
}

static bool key_is(const char *key, const char *const *list) {
    for (; *list; list++)
        if (!strcmp(key, *list)) return true;
    return false;
}

static const char *const kPipeScene[] = {"REKARINA_RENDER_SCALE", nullptr};
static const char *const kPipeUpscale[] = {"##scene_upscaler", "REKARINA_DLAA", "REKARINA_TAA", "REKARINA_DLSS", "REKARINA_DLSS_PRESET",
                                           "REKARINA_FSR_VERSION",
#ifndef GAME_OOT
                                           "REKARINA_FRAMEGEN",
#endif
                                           nullptr};
static const char *const kPipePicture[] = {"REKARINA_GAME_RES", "REKARINA_GAME_ASPECT", nullptr};
#ifdef GAME_OOT
static const char *const kPipeFlow[] = {"REKARINA_SHARPNESS", "REKARINA_HDR", "REKARINA_FRAMEGEN", nullptr};
#endif
static const char *const kPipeWindow[] = {"REKARINA_UPSCALER", "##hdr_display_scaling", "REKARINA_ASPECT",
#ifndef GAME_OOT
                                          "REKARINA_SHARPNESS",
#endif
                                          "REKARINA_WINDOW_SIZE", "REKARINA_FULLSCREEN", nullptr};
static const char *const kPad[] = {"REKARINA_PAD_LAYOUT", "REKARINA_PAD_FOLLOW", "REKARINA_PAD_DRIVER", "REKARINA_MOTION",
                                   "REKARINA_VIBRATION", "REKARINA_DUALSENSE_EXTRAS", "REKARINA_RUMBLE_STRENGTH",
                                   "REKARINA_RUMBLE_CURVE", "REKARINA_RUMBLE_TRIGGERS", "REKARINA_LSTICK_DEADZONE",
                                   "REKARINA_RSTICK_DEADZONE", nullptr};
static const char *const kKeys[] = {"REKARINA_MOUSE_SENS", "REKARINA_MOUSE_SENS_Y", "REKARINA_MOUSE_INVERT_Y",
                                    "REKARINA_MOUSE_RAW", "REKARINA_WALK_KEY", "REKARINA_WALK_SCALE",
                                    "REKARINA_CURSOR_HIDE", nullptr};
static const char *const kPerf[] = {"REKARINA_PERF_OVERLAY", "REKARINA_FPS_LIMIT", "REKARINA_VK_PRESENT", nullptr};
static const char *const kSystem[] = {"REKARINA_GPU_INDEX", "##copy_info", "REKARINA_TEX_QUALITY", nullptr};

MenuPicture menu_picture_for(const char *key) {
    if (!key || !*key) return PIC_NONE;
    if (key_is(key, kPipeScene) || key_is(key, kPipeUpscale) || key_is(key, kPipePicture) || key_is(key, kPipeWindow))
        return PIC_PIPELINE;
#ifdef GAME_OOT
    if (key_is(key, kPipeFlow)) return PIC_PIPELINE;
#endif
    if (!strncmp(key, "REKARINA_BIND_", 14)) return input_kbm_last() ? PIC_KEYS : PIC_PAD;   // the device in hand
    if (key_is(key, kPad)) return PIC_PAD;
    if (key_is(key, kKeys)) return PIC_KEYS;
    if (key_is(key, kPerf)) return PIC_PERF;
    if (key_is(key, kSystem)) return PIC_SYSTEM;
    return PIC_NONE;
}

// ---- resolution cards and the gameplay flow
static const char *ratio_name(int w, int h, char *buf, size_t n) {
    static const struct { int a, b; } r[] = {{16, 9}, {16, 10}, {21, 9}, {32, 9}, {4, 3}, {3, 2}, {43, 18}, {12, 5}};
    double q = h > 0 ? (double)w / h : 0;
    for (const auto &e : r)
        if (fabs(q - (double)e.a / e.b) < 0.01) {
            snprintf(buf, n, "%d:%d", e.a == 43 || e.a == 12 ? 21 : e.a, e.a == 43 || e.a == 12 ? 9 : e.b);
            return buf;
        }
    snprintf(buf, n, "%.2f:1", q);
    return buf;
}

static const char *upscale_quality(bool nvidia) {
    const struct { const char *mode; UiTextId name; } modes[] = {
        {"dlaa", nvidia ? UI_MENU_DLSS_DLAA : UI_MENU_UPSCALE_NATIVE},
        {"quality", UI_MENU_DLSS_QUALITY}, {"balanced", UI_MENU_DLSS_BALANCED},
        {"performance", UI_MENU_DLSS_PERFORMANCE}, {"ultra_performance", UI_MENU_DLSS_ULTRA}};
    const char *mode = cfg_value("REKARINA_DLSS");
    for (const auto &e : modes)
        if (!strcmp(mode, e.mode)) return menu_text(e.name);
    return "";
}

static float draw_pipeline(const char *key, ImDrawList *dl, float x, float y, float w, float scale, const MenuTarget *t) {
    const float fs = ImGui::GetFontSize(), cap = floorf(fs * 0.66f), val = floorf(fs * 0.80f), sub = floorf(fs * 0.64f);
#ifdef GAME_OOT
    enum { Scene, Picture, Window, Count };
    const UiTextId caps[Count] = {UI_MENU_OV_SCENE, UI_MENU_OV_OUTPUT, UI_MENU_OV_WINDOW};
    const char *const *stage[Count] = {kPipeScene, kPipePicture, kPipeWindow};
#else
    enum { Scene, Upscaler, Picture, Window, Count };
    const UiTextId caps[Count] = {UI_MENU_OV_SCENE, UI_PIC_UPSCALING, UI_MENU_OV_OUTPUT, UI_MENU_OV_WINDOW};
    const char *const *stage[Count] = {kPipeScene, kPipeUpscale, kPipePicture, kPipeWindow};
#endif
    char v[Count][64], s[Count][64];
#ifndef GAME_OOT
    unsigned out[2], scene[2];
    res_sizes(out, scene);
#endif
    const double share = res_fixed_scene_share();
    if (share > 0) {
        snprintf(v[Scene], 64, "%u \xc3\x97 %u", (unsigned)(t->image_w * share + 0.5), (unsigned)(t->image_h * share + 0.5));
#ifdef GAME_OOT
        snprintf(s[Scene], 64, menu_text(UI_PIC_SCENE_ESTIMATE), share * 100);
#else
        snprintf(s[Scene], 64, "%.0f %%", share * 100);
#endif
    } else {
#ifdef GAME_OOT
        strcpy(v[Scene], "-");
        s[Scene][0] = 0;
#else
        snprintf(v[Scene], 64, "%u \xc3\x97 %u", scene[0], scene[1]);
        snprintf(s[Scene], 64, "%.0f %%", out[0] ? scene[0] * 100.0 / out[0] : 100.0);
#endif
    }
#ifdef GAME_OOT
    int fit[4];
    gfx_fit_rect(t->image_w, t->image_h, t->window_w, t->window_h, fit);
    snprintf(v[Window], 64, "%u \xc3\x97 %u", t->window_w, t->window_h);
    snprintf(s[Window], 64, menu_text(UI_PIC_DISPLAY_FIT_SIZE), fit[2], fit[3]);
#else
    const bool upscaler = cfg_bool("REKARINA_DLAA") && vk_dlaa_available();
    s[Upscaler][0] = 0;
    if (upscaler) {
        const bool fsr1 = vk_dlaa_fsr1(), fsr3 = vk_dlaa_fsr3();
        snprintf(v[Upscaler], 64, "%s", fsr1 ? "AMD FSR 1" : fsr3 ? "AMD FSR" : "NVIDIA DLSS");
        snprintf(s[Upscaler], 64, "%s", upscale_quality(!fsr1 && !fsr3));
    } else {
        snprintf(v[Upscaler], 64, "%s", menu_text(UI_MENU_OFF));
    }
    if (gfx_output_fps() > 0) {   // frame generation shows frames of its own between the game's
        size_t n = strlen(s[Upscaler]);
        snprintf(s[Upscaler] + n, 64 - n, "%s+ FG", n ? " " : "");
    }
    snprintf(v[Window], 64, "%u \xc3\x97 %u", t->width, t->height);
    if (t->scaler == 2) snprintf(s[Window], 64, "AMD FSR 1");
    else snprintf(s[Window], 64, "%s", menu_text(t->scaler ? UI_MENU_OV_SCALE_BILINEAR : UI_MENU_OV_SCALE_NONE));
#endif
    snprintf(v[Picture], 64, "%d \xc3\x97 %d", t->image_w, t->image_h);
    ratio_name(t->image_w, t->image_h, s[Picture], 64);
    const float arrow = floorf(fs * 0.9f), cw = floorf((w - arrow * (Count - 1)) / Count), ch = floorf(val * 1.5f + sub * 1.5f);
    const float top = y + cap * 1.45f;
    for (int i = 0; i < Count; i++) {
        float cx = x + i * (cw + arrow);
        fit_text(dl, cx, y, cw, cap, kLabel, menu_text(caps[i]), true);
        ImVec2 a(cx, top), b(cx + cw, top + ch);
        dl->AddRectFilled(a, b, kInset);
        opt_grain(dl, a, b, 30);
        bool hot = key && key_is(key, stage[i]);
        dl->AddRect(a, b, hot ? kAmber : kEdge, 0, 0, hot ? fmaxf(2, floorf(2 * scale)) : 1.0f);
        fit_text(dl, cx + 4 * scale, top + ch * 0.5f - val * 0.95f, cw - 8 * scale, val, kCream, v[i], true);
        fit_text(dl, cx + 4 * scale, top + ch * 0.5f + sub * 0.25f, cw - 8 * scale, sub, kDim, s[i], true);
        if (i + 1 < Count) {   // the arrow to the next stage
            float ax = cx + cw + arrow * 0.3f, ay = top + ch * 0.5f, aw = arrow * 0.4f;
            dl->AddTriangleFilled(ImVec2(ax, ay - aw * 0.7f), ImVec2(ax + aw, ay), ImVec2(ax, ay + aw * 0.7f), kAmber);
        }
    }
    float bottom = top + ch;
#ifdef GAME_OOT
    const bool selected = cfg_bool("REKARINA_DLAA");
    const CfgTaa backend = cfg_taa();
    const bool bilinear = selected && backend == CFG_TAA_BILINEAR;
    const bool available = selected && (bilinear || vk_dlaa_available());
    const bool native = !strcmp(cfg_value("REKARINA_DLSS"), "dlaa");
    const char *name = !selected ? menu_text(UI_MENU_HDR_TM_VANILLA) :
                       bilinear ? menu_text(UI_MENU_SCENE_BILINEAR) : backend == CFG_TAA_FSR1 ? "FSR1" :
                       backend == CFG_TAA_FSR3 ? (menu_fsr4_selected() ? "FSR4" : "FSR3") : "DLSS";
    char label[128];
    snprintf(label, sizeof label, menu_text(UI_PIC_SELECTED_UPSCALER), name);
    const char *quality = selected && !bilinear ? upscale_quality(backend == CFG_TAA_DLSS) : "";
    if (*quality) {
        const size_t n = strlen(label);
        snprintf(label + n, sizeof label - n, " - %s", quality);
    }
    const auto line = [&](const char *text, ImU32 color) {
        ImFont *font = ImGui::GetFont();
        const float size = floorf(fs * 0.76f);
        bottom += floorf(fs * 0.25f);
        dl->AddText(font, size, ImVec2(x, bottom), color, text, nullptr, w);
        bottom += font->CalcTextSizeA(size, FLT_MAX, w, text).y;
    };
    line(label, key && key_is(key, kPipeUpscale) ? kAmber : kCream);
    if (selected && !available) line(menu_text(UI_MENU_TAA_UNAVAILABLE), kDim);
    if (available && !bilinear && !vk_dlaa_fsr1())
        line(menu_text(UI_PIC_FLOW_EARLY_DLSS), kDim);
    if (!available || (vk_dlaa_fsr1() && native)) {
        if (available) line(menu_text(UI_PIC_FLOW_NATIVE_FSR1), kDim);
        line(menu_text(UI_PIC_FLOW_VANILLA), kDim);
    } else if (bilinear) {
        line(menu_text(UI_PIC_FLOW_BILINEAR), kDim);
    } else if (vk_dlaa_fsr1()) {
        line(menu_text(UI_PIC_FLOW_LATE_REPLACEMENT), kDim);
    }
    // RCAS runs in square-root color; any HUD-less capture follows its return to linear light.
    line(menu_text(UI_PIC_FLOW_OUTPUT), key && key_is(key, kPipeFlow) ? kAmber : kDim);
#endif
    return floorf(bottom - y);
}

// ---- the controller, live
static void face_mark(ImDrawList *dl, ImVec2 c, float r, int label, ImU32 col) {
    const float fs = r * 1.15f, lw = fmaxf(1.5f, r * 0.16f);
    const char *t = label == PAD_LABEL_A ? "A" : label == PAD_LABEL_B ? "B" : label == PAD_LABEL_X ? "X" :
                    label == PAD_LABEL_Y ? "Y" : nullptr;
    if (t) {
        float tw = text_w(fs, t);
        dl->AddText(ImGui::GetFont(), fs, ImVec2(floorf(c.x - tw * 0.5f), floorf(c.y - fs * 0.52f)), col, t);
        return;
    }
    float k = r * 0.45f;   // the PlayStation shapes, drawn
    switch (label) {
    case PAD_LABEL_CROSS:
        dl->AddLine(ImVec2(c.x - k, c.y - k), ImVec2(c.x + k, c.y + k), col, lw);
        dl->AddLine(ImVec2(c.x - k, c.y + k), ImVec2(c.x + k, c.y - k), col, lw);
        break;
    case PAD_LABEL_CIRCLE: dl->AddCircle(c, k * 1.05f, col, 0, lw); break;
    case PAD_LABEL_SQUARE: dl->AddRect(ImVec2(c.x - k * 0.9f, c.y - k * 0.9f), ImVec2(c.x + k * 0.9f, c.y + k * 0.9f), col, 0, 0, lw); break;
    case PAD_LABEL_TRIANGLE:
        dl->AddTriangle(ImVec2(c.x, c.y - k), ImVec2(c.x + k, c.y + k * 0.75f), ImVec2(c.x - k, c.y + k * 0.75f), col, lw);
        break;
    }
}

static void stick_draw(ImDrawList *dl, ImVec2 c, float r, int16_t sx, int16_t sy, float dead, bool pressed, float scale) {
    dl->AddCircleFilled(c, r, kInset, 40);
    dl->AddCircle(c, r, pressed ? kAmber : kEdge, 40, pressed ? fmaxf(2, 2.5f * scale) : 1.0f);
    if (dead > 0) dl->AddCircle(c, r * fminf(dead, 1), kLine, 32, 1.0f);   // the dead zone
    float x = sx / 32767.0f, y = sy / 32767.0f, m = sqrtf(x * x + y * y);
    if (m > 1) x /= m, y /= m;
    bool live = m > dead;
    ImVec2 p(c.x + x * (r - r * 0.24f), c.y + y * (r - r * 0.24f));
    if (live) dl->AddLine(c, p, kAmber, fmaxf(1, 1.5f * scale));
    dl->AddCircleFilled(p, r * 0.22f, live ? kAmber : kDim, 20);
}

static void plate(ImDrawList *dl, ImVec2 a, ImVec2 b, const char *name, float fill, float size) {
    dl->AddRectFilled(a, b, kInset);
    if (fill > 0) amber_fill(dl, a, ImVec2(a.x + (b.x - a.x) * fminf(fill, 1), b.y));
    dl->AddRect(a, b, kEdge, 0, 0, 1.0f);
    fit_text(dl, a.x, a.y + (b.y - a.y - size) * 0.5f, b.x - a.x, size, fill >= 0.5f ? kInk : kCream, name, true);
}

static float draw_pad(ImDrawList *dl, float x, float y, float w, float scale) {
    const float fs = ImGui::GetFontSize(), small = floorf(fs * 0.62f);
    uint8_t pd[PAD_COUNT] = {0};
    int16_t ax[6] = {0};
    char name[64] = "";
    bool on = input_pad_snapshot(pd, ax, name, sizeof name) != 0;
    const float h = floorf(fminf(w * 0.36f, fs * 6.2f));
    const int fam = input_pad_family();
    const bool ps = fam == PROMPTS_PS;
    const char *ln = ps ? "L1" : fam == PROMPTS_NINTENDO ? "L" : "LB", *rn = ps ? "R1" : fam == PROMPTS_NINTENDO ? "R" : "RB";
    const char *lt = ps ? "L2" : fam == PROMPTS_NINTENDO ? "ZL" : "LT", *rt = ps ? "R2" : fam == PROMPTS_NINTENDO ? "ZR" : "RT";
    // the shoulders: the triggers fill with how far they are pressed
    const float ph = floorf(fs * 0.86f), pw = floorf(w * 0.13f), gap = floorf(fs * 0.25f);
    plate(dl, ImVec2(x + w * 0.04f, y), ImVec2(x + w * 0.04f + pw, y + ph), lt, on ? ax[4] / 32767.0f : 0, small);
    plate(dl, ImVec2(x + w * 0.04f + pw + gap, y), ImVec2(x + w * 0.04f + pw * 2 + gap, y + ph), ln, pd[PAD_L] ? 1 : 0, small);
    plate(dl, ImVec2(x + w * 0.96f - pw * 2 - gap, y), ImVec2(x + w * 0.96f - pw - gap, y + ph), rn, pd[PAD_R] ? 1 : 0, small);
    plate(dl, ImVec2(x + w * 0.96f - pw, y), ImVec2(x + w * 0.96f, y + ph), rt, on ? ax[5] / 32767.0f : 0, small);
    const float body = y + ph + floorf(fs * 0.4f), bh = h - (body - y), r = floorf(fminf(bh * 0.32f, w * 0.08f));
    // the Xbox and Switch places: left stick high, D-pad low; a PlayStation pad has them the other way round
    ImVec2 ls(x + w * 0.20f, body + bh * 0.38f), dp(x + w * 0.36f, body + bh * 0.68f);
    ImVec2 rs(x + w * 0.64f, body + bh * 0.68f), fc(x + w * 0.80f, body + bh * 0.38f);
    if (ps) ls = ImVec2(x + w * 0.36f, body + bh * 0.68f), dp = ImVec2(x + w * 0.20f, body + bh * 0.38f);
    stick_draw(dl, ls, r, ax[0], ax[1], input_pad_deadzone(0), pd[PAD_LSTICK], scale);
    stick_draw(dl, rs, r, ax[2], ax[3], input_pad_deadzone(1), pd[PAD_RSTICK], scale);
    {   // the D-pad: four arms around a centre
        const float a = floorf(r * 0.50f);
        const struct { int pad; float dx, dy; } arms[] = {
            {PAD_DPAD_UP, 0, -1}, {PAD_DPAD_DOWN, 0, 1}, {PAD_DPAD_LEFT, -1, 0}, {PAD_DPAD_RIGHT, 1, 0}};
        for (const auto &e : arms) {
            ImVec2 c(dp.x + e.dx * a * 1.15f, dp.y + e.dy * a * 1.15f);
            ImVec2 p0(c.x - a * 0.55f, c.y - a * 0.55f), p1(c.x + a * 0.55f, c.y + a * 0.55f);
            dl->AddRectFilled(p0, p1, pd[e.pad] ? kAmber : kInset);
            dl->AddRect(p0, p1, kEdge, 0, 0, 1.0f);
        }
    }
    {   // the face buttons in their diamond, each with its own label
        const float d = floorf(r * 0.88f), br = floorf(r * 0.42f);
        const struct { int pad; float dx, dy; } face[] = {
            {PAD_SOUTH, 0, 1}, {PAD_EAST, 1, 0}, {PAD_WEST, -1, 0}, {PAD_NORTH, 0, -1}};
        for (const auto &e : face) {
            ImVec2 c(fc.x + e.dx * d, fc.y + e.dy * d);
            bool lit = pd[e.pad] != 0;
            dl->AddCircleFilled(c, br, lit ? kAmber : kInset, 24);
            dl->AddCircle(c, br, kEdge, 24, 1.0f);
            int label = on ? input_pad_label(e.pad) : PAD_LABEL_NONE;
            if (label == PAD_LABEL_NONE)   // no pad: the Xbox letters in their places
                label = e.pad == PAD_SOUTH ? PAD_LABEL_A : e.pad == PAD_EAST ? PAD_LABEL_B : e.pad == PAD_WEST ? PAD_LABEL_X : PAD_LABEL_Y;
            face_mark(dl, c, br, label, lit ? kInk : kCream);
        }
    }
    {   // Back / View / Minus and Start / Menu / Plus: two small pills between the sticks
        const float pw2 = floorf(r * 0.62f), ph2 = floorf(r * 0.30f), cy = body + bh * 0.36f;
        const float xs[2] = {x + w * 0.46f, x + w * 0.54f};
        const int pads[2] = {PAD_BACK, PAD_START};
        for (int i = 0; i < 2; i++) {
            ImVec2 a(xs[i] - pw2 * 0.5f, cy - ph2 * 0.5f), b(xs[i] + pw2 * 0.5f, cy + ph2 * 0.5f);
            dl->AddRectFilled(a, b, pd[pads[i]] ? kAmber : kInset, ph2 * 0.5f);
            dl->AddRect(a, b, kEdge, ph2 * 0.5f, 0, 1.0f);
        }
    }
    float yy = y + h + floorf(fs * 0.25f);
    char line[160];
    if (on) snprintf(line, sizeof line, "%s", name[0] ? name : "-");
    else snprintf(line, sizeof line, "%s", menu_text(UI_PIC_PAD_NONE));
    fit_text(dl, x, yy, w, small, on ? kDim : kLabel, line, true);
    return floorf(yy + small * 1.2f - y);
}

// ---- performance: frame times, CPU and GPU time per frame, memory
static float draw_perf(ImDrawList *dl, float x, float y0, float w, float scale) {
    float y = y0;
    PerfStats s;
    perfmon_snapshot(&s);
    const float fs = ImGui::GetFontSize(), small = floorf(fs * 0.66f), mid = floorf(fs * 0.76f);
    const float gh = floorf(fs * 2.4f);
    double limit = frame_step_limit_hz();
    {   // the rate and the 1 % low on a line of their own above the graph; the frame limit at the right
        char fps[64], low[48], lim[64], rate[32];
        snprintf(fps, sizeof fps, menu_text(UI_MENU_OV_FPS), s.fps);
        if (s.low1_fps >= 0) snprintf(low, sizeof low, menu_text(UI_MENU_OV_LOW1), s.low1_fps);
        else low[0] = 0;
        dl->AddText(ImGui::GetFont(), mid, ImVec2(x, y), kCream, fps);
        dl->AddText(ImGui::GetFont(), small, ImVec2(x + text_w(mid, fps) + fs * 0.5f, y + (mid - small)), kDim, low);
        snprintf(rate, sizeof rate, menu_text(UI_MENU_OV_FPS), limit > 0 ? limit : 60.0);
        snprintf(lim, sizeof lim, "%s %s", menu_text(UI_MENU_OV_LIMIT), rate);
        // the line in the graph that stands for it: a short dash before the words
        float lw = text_w(small, lim), dash = fs * 0.8f, lx = x + w - lw;
        dl->AddText(ImGui::GetFont(), small, ImVec2(lx, y + (mid - small)), kLabel, lim);
        float dy = floorf(y + (mid - small) + small * 0.55f) + 0.5f;
        dl->AddLine(ImVec2(lx - dash - fs * 0.3f, dy), ImVec2(lx - fs * 0.3f, dy), IM_COL32(176, 162, 128, 255), 1.5f);
        y += floorf(mid * 1.3f);
    }
    ImVec2 ga(x, y), gb(x + w, y + gh);
    dl->AddRectFilled(ga, gb, kInset);
    dl->AddRect(ga, gb, kEdge, 0, 0, 1.0f);
    float target = limit > 0 ? (float)(1000.0 / limit) : 16.7f;
    float top = fmaxf(fmaxf(s.frame_ms_max * 1.1f, target * 2.2f), 12.0f);
    auto gy = [&](float ms) { return gb.y - 2 - fminf(ms / top, 1) * (gh - 4); };
    {   // the frame limit's time as a guide line (named on the line above)
        float ly = floorf(gy(target)) + 0.5f;
        dl->AddLine(ImVec2(ga.x + 1, ly), ImVec2(gb.x - 1, ly), IM_COL32(176, 162, 128, 200), 1.0f);
    }
    {
        const float step = (gb.x - ga.x - 2) / (PERF_HISTORY - 1);
        ImVec2 pts[PERF_HISTORY];
        for (int i = 0; i < PERF_HISTORY; i++) pts[i] = ImVec2(ga.x + 1 + i * step, gy(s.frame_ms[i]));
        dl->AddPolyline(pts, PERF_HISTORY, kAmber, 0, fmaxf(1.0f, 1.3f * scale));
    }
    float yy = gb.y + floorf(fs * 0.35f);
    // CPU and GPU time per frame on one scale with the frame's own time as a line across both: the bar that reaches
    // the line is what holds the frame rate
    struct Bar { UiTextId name; float share; char value[48]; } bars[4];
    float frame = s.frame_ms_avg > 0 ? s.frame_ms_avg : target;
    float scale_ms = fmaxf(fmaxf(frame, fmaxf(s.replay_ms, s.gpu_ms)) * 1.15f, 1.0f);
    bars[0] = {UI_PIC_CPU_FRAME, s.replay_ms / scale_ms, ""};
    bars[1] = {UI_PIC_GPU_FRAME, s.gpu_ms / scale_ms, ""};
    bars[2] = {UI_MENU_OV_VRAM, s.vram_total_mb > 0 ? s.vram_used_mb / s.vram_total_mb : -1, ""};
    bars[3] = {UI_MENU_OV_RAM, s.ram_total_mb > 0 ? s.ram_used_mb / s.ram_total_mb : -1, ""};
    auto ms = [&](char *o, float v) { if (v < 0) snprintf(o, 48, "-"); else snprintf(o, 48, menu_text(UI_MENU_OV_FRAME_MS), v); };
    ms(bars[0].value, s.replay_ms);
    ms(bars[1].value, s.gpu_ms);
    auto gb2 = [&](char *o, float used, float total) {
        if (used < 0 || total <= 0) snprintf(o, 48, "-");
        else snprintf(o, 48, "%.1f / %.1f GB", used / 1024, total / 1024);
    };
    gb2(bars[2].value, s.vram_used_mb, s.vram_total_mb);
    gb2(bars[3].value, s.ram_used_mb, s.ram_total_mb);
    float name_w = 0, val_w = 0;
    for (const Bar &b : bars) name_w = fmaxf(name_w, text_w(small, menu_text(b.name))), val_w = fmaxf(val_w, text_w(small, b.value));
    const float lh = floorf(small * 1.45f), bx0 = x + name_w + fs * 0.5f, bx1 = x + w - val_w - fs * 0.5f, bh = floorf(small * 0.62f);
    for (int i = 0; i < 4; i++) {
        const Bar &b = bars[i];
        if (i == 2) yy += floorf(lh * 0.25f);
        dl->AddText(ImGui::GetFont(), small, ImVec2(x, yy), kLabel, menu_text(b.name));
        float by = floorf(yy + (small - bh) * 0.5f) + 1;
        dl->AddRectFilled(ImVec2(bx0, by), ImVec2(bx1, by + bh), kInset);
        if (b.share > 0) amber_fill(dl, ImVec2(bx0 + 1, by + 1), ImVec2(bx0 + 1 + (bx1 - bx0 - 2) * fminf(b.share, 1), by + bh - 1));
        if (i == 1) {   // the frame's time across both time bars
            float fx = floorf(bx0 + (bx1 - bx0) * frame / scale_ms) + 0.5f;
            dl->AddLine(ImVec2(fx, by - lh - 2), ImVec2(fx, by + bh + 2), kCream, 1.0f);
        }
        dl->AddText(ImGui::GetFont(), small, ImVec2(x + w - text_w(small, b.value), yy), kCream, b.value);
        yy += lh;
    }
    return floorf(yy - y0);
}

// ---- the system: what the game runs on, in two columns
static float draw_system(ImDrawList *dl, float x, float y, float w, const MenuTarget *t) {
    const float fs = ImGui::GetFontSize(), small = floorf(fs * 0.66f), lh = floorf(small * 1.42f);
    PerfStats s;
    perfmon_snapshot(&s);
    VkPhysicalDeviceDriverProperties drv = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &drv};
    vkGetPhysicalDeviceProperties2(t->phys, &p2);
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(t->phys, &mp);
    double vram = 0;
    for (uint32_t i = 0; i < mp.memoryHeapCount; i++)
        if (mp.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) vram += (double)mp.memoryHeaps[i].size;
    static int mesh = -1;   // the device's extensions, once
    if (mesh < 0) {
        uint32_t n = 0;
        vkEnumerateDeviceExtensionProperties(t->phys, NULL, &n, NULL);
        VkExtensionProperties *e = (VkExtensionProperties *)calloc(n ? n : 1, sizeof *e);
        mesh = 0;
        if (e && vkEnumerateDeviceExtensionProperties(t->phys, NULL, &n, e) >= 0)
            for (uint32_t i = 0; i < n; i++) mesh |= !strcmp(e[i].extensionName, "VK_EXT_mesh_shader");
        free(e);
    }
    char rows[8][2][160];
    int n = 0;
    auto row = [&](UiTextId label, const char *fmt, ...) {
        va_list ap;
        va_start(ap, fmt);
        snprintf(rows[n][0], 160, "%s", menu_text(label));
        vsnprintf(rows[n][1], 160, fmt, ap);
        va_end(ap);
        n++;
    };
    row(UI_MENU_OV_GPU, "%s", p2.properties.deviceName);
    row(UI_PIC_DRIVER, "%s %s", drv.driverName, drv.driverInfo);
    {
        char feats[96];
        int k = snprintf(feats, sizeof feats, "%.1f GB", vram / (1024.0 * 1024 * 1024));
        k += snprintf(feats + k, sizeof feats - k, "  \xc2\xb7  Vulkan %u.%u", VK_API_VERSION_MAJOR(p2.properties.apiVersion),
                      VK_API_VERSION_MINOR(p2.properties.apiVersion));
        if (mesh) k += snprintf(feats + k, sizeof feats - k, "  \xc2\xb7  mesh shaders");
        bool rtx = p2.properties.vendorID == 0x10de && strstr(p2.properties.deviceName, "RTX");
        if (rtx) snprintf(feats + k, sizeof feats - k, "  \xc2\xb7  DLSS");
        row(UI_MENU_OV_VRAM, "%s", feats);
    }
    row(UI_MENU_OV_CPU, "%s%s%u", s.cpu_name[0] ? s.cpu_name : "", s.cpu_name[0] ? "  \xc2\xb7  " : "", plat_cpu_count());
    if (s.ram_total_mb > 0) row(UI_MENU_OV_RAM, "%.0f GB", s.ram_total_mb / 1024);
    {
        int mw = 0, mh = 0;
        double hz = gfx_window_refresh_hz();
        if (gfx_window_monitor_size(&mw, &mh)) row(UI_PIC_DISPLAY, "%d \xc3\x97 %d  \xc2\xb7  %.0f Hz", mw, mh, hz);
    }
#ifdef _WIN32
    row(UI_PIC_SYSTEM, "%s", plat_is_wine() ? "Wine / Proton" : "Windows");
#else
    row(UI_PIC_SYSTEM, "%s", "Linux");
#endif
    {
        const char *p = cfg_device_profile();
        bool set = cfg_is_set("REKARINA_DEVICE_PROFILE") && strcmp(cfg_value("REKARINA_DEVICE_PROFILE"), "auto");
        UiTextId why = !strcmp(p, "deck") ? UI_PIC_PROFILE_DECK : !strcmp(p, "handheld") ? UI_PIC_PROFILE_HANDHELD
                                                                                         : UI_PIC_PROFILE_DESKTOP;
        // chosen by the setting rather than found: the profile's name alone ("Steam Deck"), then where it was set
        char name[96];
        snprintf(name, sizeof name, "%s", menu_text(why));
        if (set) {
            char *c = strstr(name, ":"), *wide = strstr(name, "：");   // ASCII or full-width colon
            if (wide && (!c || wide < c)) c = wide;
            if (c) *c = 0;
        }
        row(UI_PIC_PROFILE, "%s%s", name, set ? menu_text(UI_PIC_PROFILE_SET) : "");
    }
    float lw = 0;
    for (int i = 0; i < n; i++) lw = fmaxf(lw, text_w(small, rows[i][0]));
    lw += fs * 0.6f;
    float yy = y;
    for (int i = 0; i < n; i++) {
        dl->AddText(ImGui::GetFont(), small, ImVec2(x, yy), kLabel, rows[i][0]);
        fit_text(dl, x + lw, yy, w - lw, small, kCream, rows[i][1], false);
        if (i + 1 < n) dl->AddLine(ImVec2(x, yy + lh - 2), ImVec2(x + w, yy + lh - 2), IM_COL32(92, 74, 46, 120), 1.0f);
        yy += lh;
    }
    return floorf(yy - y);
}

// ---- the keyboard and the mouse: each bound key shows the Switch button it presses; the keys of the row described are
// framed in white (the binding rows, REKARINA_BIND_<action>)
struct KeyCap { int key; float w; const char *legend; };   // key KEY_NONE: a gap

static const KeyCap kRow1[] = {{KEY_GRAVE, 1, "`"}, {KEY_1, 1, "1"}, {KEY_2, 1, "2"}, {KEY_3, 1, "3"}, {KEY_4, 1, "4"},
                               {KEY_5, 1, "5"}, {KEY_6, 1, "6"}, {KEY_7, 1, "7"}, {KEY_8, 1, "8"}, {KEY_9, 1, "9"},
                               {KEY_0, 1, "0"}, {KEY_MINUS, 1, "-"}, {KEY_EQUALS, 1, "="}, {KEY_BACKSPACE, 2, "Bksp"}};
static const KeyCap kRow2[] = {{KEY_TAB, 1.5f, "Tab"}, {KEY_Q, 1, "Q"}, {KEY_W, 1, "W"}, {KEY_E, 1, "E"}, {KEY_R, 1, "R"},
                               {KEY_T, 1, "T"}, {KEY_Y, 1, "Y"}, {KEY_U, 1, "U"}, {KEY_I, 1, "I"}, {KEY_O, 1, "O"},
                               {KEY_P, 1, "P"}, {KEY_LBRACKET, 1, "["}, {KEY_RBRACKET, 1, "]"}, {KEY_BACKSLASH, 1.5f, "\\"}};
static const KeyCap kRow3[] = {{KEY_CAPSLOCK, 1.75f, "Caps"}, {KEY_A, 1, "A"}, {KEY_S, 1, "S"}, {KEY_D, 1, "D"},
                               {KEY_F, 1, "F"}, {KEY_G, 1, "G"}, {KEY_H, 1, "H"}, {KEY_J, 1, "J"}, {KEY_K, 1, "K"},
                               {KEY_L, 1, "L"}, {KEY_SEMICOLON, 1, ";"}, {KEY_APOSTROPHE, 1, "'"}, {KEY_ENTER, 2.25f, "Enter"}};
static const KeyCap kRow4[] = {{KEY_LSHIFT, 2.25f, "Shift"}, {KEY_Z, 1, "Z"}, {KEY_X, 1, "X"}, {KEY_C, 1, "C"},
                               {KEY_V, 1, "V"}, {KEY_B, 1, "B"}, {KEY_N, 1, "N"}, {KEY_M, 1, "M"}, {KEY_COMMA, 1, ","},
                               {KEY_PERIOD, 1, "."}, {KEY_SLASH, 1, "/"}, {KEY_RSHIFT, 2.75f, "Shift"},
                               {KEY_NONE, 1.25f, nullptr}, {KEY_UP, 1, nullptr}};
static const KeyCap kRow5[] = {{KEY_LCTRL, 1.5f, "Ctrl"}, {KEY_NONE, 1, nullptr}, {KEY_LALT, 1.5f, "Alt"},
                               {KEY_SPACE, 6.5f, "Space"}, {KEY_NONE, 2.0f, nullptr}, {KEY_RCTRL, 1.5f, "Ctrl"},
                               {KEY_NONE, 0.25f, nullptr}, {KEY_LEFT, 1, nullptr}, {KEY_DOWN, 1, nullptr},
                               {KEY_RIGHT, 1, nullptr}};
static const struct { const KeyCap *keys; int n; } kRows[] = {
    {kRow1, sizeof kRow1 / sizeof *kRow1}, {kRow2, sizeof kRow2 / sizeof *kRow2}, {kRow3, sizeof kRow3 / sizeof *kRow3},
    {kRow4, sizeof kRow4 / sizeof *kRow4}, {kRow5, sizeof kRow5 / sizeof *kRow5}};

// The Switch button's short name for a bound key, and the direction of a stick or D-pad action (-1 none; 0 left, 1 up,
// 2 right, 3 down), drawn as a small triangle after the name.
static const char *action_short(int a, int *dir) {
    static const char *const names[] = {"A", "B", "X", "Y", "LS", "RS", "L", "R", "ZL", "ZR", "+", "-"};
    *dir = -1;
    if (a >= 0 && a <= B_MINUS) return names[a];
    if (a >= B_LEFT && a <= B_DOWN) return *dir = a - B_LEFT, "D";
    if (a >= B_SL_LEFT && a <= B_SL_DOWN) return *dir = a - B_SL_LEFT, "L";
    if (a >= B_SR_LEFT && a <= B_SR_DOWN) return *dir = a - B_SR_LEFT, "R";
    if (a == B_SHAKE) return "~";
    return "";
}

static void dir_mark(ImDrawList *dl, ImVec2 c, float r, int dir, ImU32 col) {
    static const float dx[4] = {-1, 0, 1, 0}, dy[4] = {0, -1, 0, 1};
    ImVec2 tip(c.x + dx[dir] * r, c.y + dy[dir] * r);
    ImVec2 b0(c.x - dx[dir] * r * 0.6f + dy[dir] * r * 0.8f, c.y - dy[dir] * r * 0.6f + dx[dir] * r * 0.8f);
    ImVec2 b1(c.x - dx[dir] * r * 0.6f - dy[dir] * r * 0.8f, c.y - dy[dir] * r * 0.6f - dx[dir] * r * 0.8f);
    dl->AddTriangleFilled(tip, b0, b1, col);
}

// One key cap at [a, b]: bound (amber, the button's name, the key's own legend small at the top left), or not (dark,
// the legend in the middle); framed in white when it belongs to the row described. An arrow key without a legend
// draws its arrow (arrow: 0..3; -1 none).
static void key_cap_draw(ImDrawList *dl, ImVec2 a, ImVec2 b, const char *legend, int arrow, int action, bool framed,
                         float fs) {
    bool bound = action >= 0;
    dl->AddRectFilled(a, b, bound ? kAmber : kInset, 2);
    dl->AddRect(a, b, bound ? kAmberLow : kEdge, 2, 0, 1.0f);
    float w = b.x - a.x, h = b.y - a.y;
    if (bound) {
        int dir;
        const char *name = action_short(action, &dir);
        float big = floorf(h * 0.50f), tw = text_w(big, name), aw = dir >= 0 ? big * 0.55f : 0;
        float x0 = a.x + (w - tw - aw) * 0.5f, y0 = a.y + h * 0.56f - big * 0.5f;
        dl->AddText(ImGui::GetFont(), big, ImVec2(floorf(x0), floorf(y0)), kInk, name);
        if (dir >= 0) dir_mark(dl, ImVec2(x0 + tw + aw * 0.6f, y0 + big * 0.55f), aw * 0.45f, dir, kInk);
        float lf = floorf(h * 0.28f);
        if (legend) dl->AddText(ImGui::GetFont(), lf, ImVec2(a.x + 2, a.y + 1), IM_COL32(70, 48, 18, 255), legend);
        else if (arrow >= 0) dir_mark(dl, ImVec2(a.x + lf * 0.6f, a.y + lf * 0.6f), lf * 0.35f, arrow, IM_COL32(70, 48, 18, 255));
    } else if (legend) {
        float lf = floorf(fminf(h * 0.42f, fs));
        fit_text(dl, a.x + 1, a.y + (h - lf) * 0.5f, w - 2, lf, kLabel, legend, true);
    } else if (arrow >= 0) {
        dir_mark(dl, ImVec2(a.x + w * 0.5f, a.y + h * 0.5f), h * 0.18f, arrow, kLabel);
    }
    if (framed) dl->AddRect(ImVec2(a.x - 2, a.y - 2), ImVec2(b.x + 2, b.y + 2), IM_COL32(255, 255, 255, 255), 3, 0, 2.0f);
}

static int arrow_of(int key) {
    return key == KEY_LEFT ? 0 : key == KEY_UP ? 1 : key == KEY_RIGHT ? 2 : key == KEY_DOWN ? 3 : -1;
}

static float draw_keys(const char *key, ImDrawList *dl, float x, float y, float w) {
    const float fs = ImGui::GetFontSize();
    InputMap m;
    input_map_snapshot(&m);
    int8_t act[KEY_COUNT];
    memset(act, -1, sizeof act);
    for (int a = 0; a < ACTION_COUNT; a++) {
        uint8_t keys[INPUT_BINDING_MAX];
        int n = input_map_keys_of(&m, a, keys, INPUT_BINDING_MAX);
        for (int i = 0; i < n; i++)
            if (keys[i] < KEY_COUNT && act[keys[i]] < 0) act[keys[i]] = (int8_t)a;
    }
    int framed_action = -1;   // the binding row described: its action's keys get the white frame
    if (key && !strncmp(key, "REKARINA_BIND_", 14))
        for (int a = 0; a < ACTION_COUNT; a++)
            if (input_action_name(a) && !strcmp(key + 14, input_action_name(a))) framed_action = a;
    // 15 units of keys and the arrow block beside rows 4 and 5 (to 18.5), a gap, 3 for the mouse
    const float units = 18.5f + 1.1f + 3.9f, u = floorf(w / units), kh = floorf(u * 0.92f), g = fmaxf(2, floorf(u * 0.08f));
    float yy = y;
    for (const auto &row : kRows) {
        float xx = x;
        for (int i = 0; i < row.n; i++) {
            const KeyCap &k = row.keys[i];
            float kw = floorf(k.w * u);
            if (k.key != KEY_NONE)
                key_cap_draw(dl, ImVec2(xx, yy), ImVec2(xx + kw - g, yy + kh - g), k.legend, arrow_of(k.key), act[k.key],
                             framed_action >= 0 && act[k.key] == framed_action, fs);
            xx += kw;
        }
        yy += kh;
    }
    {   // the mouse: two buttons over a body, the wheel between them, the side buttons on its left edge; a bound part
        // carries its button's name (the small ones beside them, on the body)
        const float mx = x + floorf((18.5f + 1.1f) * u), mw = floorf(3.9f * u) - g, mh = floorf(kh * 5.0f) - g;
        const float my = y, bh = floorf(mh * 0.40f), half = floorf(mw * 0.5f), wheel = floorf(mw * 0.09f);
        dl->AddRectFilled(ImVec2(mx, my), ImVec2(mx + mw, my + mh), kInset, mw * 0.42f);
        dl->AddRect(ImVec2(mx, my), ImVec2(mx + mw, my + mh), kEdge, mw * 0.42f, 0, 1.0f);
        struct { int key; ImVec2 a, b; } parts[] = {
            {KEY_MOUSE_L, ImVec2(mx + 3, my + 3), ImVec2(mx + half - wheel - 2, my + bh)},
            {KEY_MOUSE_R, ImVec2(mx + half + wheel + 2, my + 3), ImVec2(mx + mw - 3, my + bh)},
            {KEY_MOUSE_M, ImVec2(mx + half - wheel, my + bh * 0.22f), ImVec2(mx + half + wheel, my + bh * 0.78f)},
            {KEY_MOUSE_X2, ImVec2(mx - mw * 0.10f, my + bh * 1.20f), ImVec2(mx + 1, my + bh * 1.62f)},
            {KEY_MOUSE_X1, ImVec2(mx - mw * 0.10f, my + bh * 1.72f), ImVec2(mx + 1, my + bh * 2.14f)},
        };
        const float lf = floorf(kh * 0.40f);
        for (const auto &p : parts) {
            int a = act[p.key];
            bool framed = framed_action >= 0 && a == framed_action;
            if (p.key == KEY_MOUSE_M || p.key >= KEY_MOUSE_X1) {   // small parts: a fill, the name beside them
                dl->AddRectFilled(p.a, p.b, a >= 0 ? kAmber : IM_COL32(52, 43, 28, 255), 2);
                dl->AddRect(p.a, p.b, framed ? IM_COL32(255, 255, 255, 255) : kEdge, 2, 0, framed ? 2.0f : 1.0f);
                if (a < 0) continue;
                int dir;
                const char *name = action_short(a, &dir);
                float tx = p.key == KEY_MOUSE_M ? (p.a.x + p.b.x - text_w(lf, name)) * 0.5f : p.b.x + 4;
                float ty = p.key == KEY_MOUSE_M ? p.b.y + 2 : (p.a.y + p.b.y - lf) * 0.5f;
                dl->AddText(ImGui::GetFont(), lf, ImVec2(floorf(tx), floorf(ty)), kAmber, name);
                if (dir >= 0) dir_mark(dl, ImVec2(tx + text_w(lf, name) + lf * 0.4f, ty + lf * 0.55f), lf * 0.25f, dir, kAmber);
                continue;
            }
            key_cap_draw(dl, p.a, p.b, nullptr, -1, a, framed, fs);
        }
        if (cfg_double("REKARINA_MOUSE_SENS") > 0) {   // the mouse moves the camera: four arrows on its body
            float cx = mx + mw * 0.55f, cy = my + bh + (mh - bh) * 0.52f, r = mw * 0.22f;
            dl->AddCircle(ImVec2(cx, cy), r * 0.55f, kAmber, 20, 1.5f);
            for (int d = 0; d < 4; d++)
                dir_mark(dl, ImVec2(cx + (d == 0 ? -r : d == 2 ? r : 0), cy + (d == 1 ? -r : d == 3 ? r : 0)), r * 0.30f, d,
                         kAmber);
        }
    }
    return floorf(yy - y);
}

float menu_picture_draw(MenuPicture pic, const char *key, ImDrawList *dl, float x, float y, float w, float scale,
                        const MenuTarget *t) {
    switch (pic) {
    case PIC_PIPELINE: return draw_pipeline(key, dl, x, y, w, scale, t);
    case PIC_PAD: return draw_pad(dl, x, y, w, scale);
    case PIC_PERF: return draw_perf(dl, x, y, w, scale);
    case PIC_SYSTEM: return draw_system(dl, x, y, w, t);
    case PIC_KEYS: return draw_keys(key, dl, x, y, w);
    default: return 0;
    }
}
