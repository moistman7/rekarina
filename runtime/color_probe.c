// --probe-color: the colour profile on the present, both outputs, against a CPU copy of the shaders' formulas, on known
// values (no game files, no window, no HDR display: the passes are recorded into an upload batch and read back).
// - SDR (vk_color.c vk_color_record, shaders/color_grade.comp): sRGB code values of an RGBA8 UNORM picture, graded and
//   clamped to 0..1, against color_grade.comp's formulas.
// - HDR10 (vk_hdr.c vk_hdr_encode, shaders/hdr_encode.comp): linear values of an RGBA16F picture (1.0 the HUD's white,
//   highlights up to 20 times it) to PQ code values, with the profile off and on, against hdr_encode.comp's formulas.
// - The same meaning: the SDR samples' own linear values (a second row of the float picture) through the HDR10 pass
//   give the PQ code of what the SDR pass made of them, wherever the SDR result is inside 0..1 (the SDR pass clamps the
//   rest), with the picture's white at the HUD's.
// All within 0.002 (code values; the passes write RGBA16F). Profiles: off, the menu's Vivid (contrast 115, saturation
// 135, gamma 0.95), one with every value moved (brightness +10, contrast 90, saturation 70, gamma 1.1), and Vivid with
// the picture's white above the HUD's (game 300 nits, HUD 203).
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "cfg.h"
#include "host.h"
#include "platform/platform.h"
#include "vk.h"

enum { N = 24 };
#define TOL 0.002

// ---- half floats (IEEE 754 binary16), round to nearest even
static uint16_t f2h(float f) {
    uint32_t x;
    memcpy(&x, &f, 4);
    uint32_t sign = x >> 16 & 0x8000, e = x >> 23 & 0xff, m = x & 0x7fffff;
    if (e == 0xff) return (uint16_t)(sign | 0x7c00 | (m ? 0x200 : 0));
    int ee = (int)e - 127 + 15;
    if (ee >= 31) return (uint16_t)(sign | 0x7c00);
    if (ee <= 0) {
        if (ee < -10) return (uint16_t)sign;
        m |= 0x800000;
        uint32_t shift = (uint32_t)(14 - ee), h = m >> shift, rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
        if (rem > half || (rem == half && (h & 1))) h++;
        return (uint16_t)(sign | h);
    }
    uint32_t h = (uint32_t)ee << 10 | m >> 13, rem = m & 0x1fff;
    if (rem > 0x1000 || (rem == 0x1000 && (h & 1))) h++;
    return (uint16_t)(sign | h);
}

static float h2f(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, e = h >> 10 & 0x1f, m = h & 0x3ff, x;
    if (e == 0) {
        float v = ldexpf((float)m, -24);
        return sign ? -v : v;
    }
    x = e == 31 ? sign | 0x7f800000 | m << 13 : sign | (e - 15 + 127) << 23 | m << 13;
    float f;
    memcpy(&f, &x, 4);
    return f;
}

// ---- the shaders' formulas, in double
typedef struct { int b, k, s; double g; } Profile;   // as the settings hold them: %, %, %, exponent

static double srgb_decode(double c) { return c > 0.04045 ? pow((c + 0.055) / 1.055, 2.4) : c / 12.92; }
static double srgb_encode(double c) { return c > 0.0031308 ? 1.055 * pow(c, 1 / 2.4) - 0.055 : c * 12.92; }

// color_grade.comp before its clamp to 0..1 = hdr_encode.comp's grade() on code values
static void grade_codes(double e[3], Profile p) {
    for (int i = 0; i < 3; i++) e[i] = (e[i] + p.b / 100.0 - 0.5) * (p.k / 100.0) + 0.5;
    double l = 0.2126 * e[0] + 0.7152 * e[1] + 0.0722 * e[2];
    for (int i = 0; i < 3; i++) e[i] = pow(fmax(l + (e[i] - l) * (p.s / 100.0), 0.0), p.g);
}

static void ref_sdr(const uint8_t in[4], Profile p, double out[3], int *inside) {
    for (int i = 0; i < 3; i++) out[i] = in[i] / 255.0;
    grade_codes(out, p);
    *inside = 1;
    for (int i = 0; i < 3; i++) {
        if (out[i] > 1) *inside = 0;
        out[i] = fmin(fmax(out[i], 0.0), 1.0);
    }
}

static double pq(double nits) {
    const double m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    double y = pow(fmin(fmax(nits / 10000.0, 0.0), 1.0), m1);
    return pow((c1 + c2 * y) / (1 + c3 * y), m2);
}

static double pq_nits(double e) {
    const double m1 = 0.1593017578125, m2 = 78.84375, c1 = 0.8359375, c2 = 18.8515625, c3 = 18.6875;
    double p = pow(fmax(e, 0.0), 1 / m2);
    return 10000 * pow(fmax(p - c1, 0.0) / (c2 - c3 * p), 1 / m1);
}

typedef struct { double white, ui, peak; const char *gamma; } Nits;   // white vs HUD, HUD nits, peak, gamma mode

static double sign_pow(double x, double e) { return copysign(pow(fabs(x), e), x); }

// hdr_encode.comp: linear Rec. 709 (1.0 the HUD's white) to PQ code values in Rec. 2020; grade 0: the profile off
static void ref_hdr(const double lin[3], int grade, Profile p, Nits n, double out[3]) {
    double c[3];
    for (int i = 0; i < 3; i++) c[i] = lin[i];
    if (grade) {
        for (int i = 0; i < 3; i++) c[i] = srgb_encode(fmax(c[i], 0.0) / n.white);
        grade_codes(c, p);
        for (int i = 0; i < 3; i++) c[i] = srgb_decode(c[i]) * n.white;
    }
    double gexp = !strcmp(n.gamma, "2.2") ? 2.2 : !strcmp(n.gamma, "2.4") ? 2.4 : 0.0;
    if (gexp > 0.0)
        for (int i = 0; i < 3; i++) c[i] = copysign(pow(srgb_encode(fabs(c[i])), gexp), c[i]);
    static const double m[9] = {0.6274038960, 0.3292830384, 0.0433130657,
                                0.0690972894, 0.9195403951, 0.0113623156,
                                0.0163914388, 0.0880133078, 0.8955952532};
    double r2020[3];
    for (int i = 0; i < 3; i++) r2020[i] = m[3 * i] * c[0] + m[3 * i + 1] * c[1] + m[3 * i + 2] * c[2];
    double gray = 0.2627002120 * r2020[0] + 0.6779980715 * r2020[1] + 0.0593017165 * r2020[2];
    double mgg = log(1.0 / pow(10.0, 0.75)) / log(0.5);
    double enc[3], low = 0.0;
    for (int i = 0; i < 3; i++) {
        enc[i] = sign_pow(r2020[i], 1.0 / mgg);
        if (enc[i] < low) low = enc[i];
    }
    double egray = pow(fmax(gray, 0.0), 1.0 / mgg);
    double dist = egray - low, ratio = dist == 0.0 ? 0.0 : -low / dist;
    double mx = n.peak;
    for (int i = 0; i < 3; i++) {
        r2020[i] = sign_pow(egray + (enc[i] - egray) * (1.0 - ratio), mgg) * n.ui;
        if (r2020[i] > mx) mx = r2020[i];
    }
    for (int i = 0; i < 3; i++) out[i] = pq(fmax(r2020[i] * (n.peak / mx), 0.0));
}

// ---- the samples
static const uint8_t sdr_in[N][4] = {
    {0, 0, 0, 255}, {8, 8, 8, 255}, {24, 24, 24, 255}, {48, 48, 48, 255}, {80, 80, 80, 255}, {118, 118, 118, 255},
    {128, 128, 128, 255}, {160, 160, 160, 255}, {192, 192, 192, 255}, {224, 224, 224, 255}, {245, 245, 245, 255},
    {255, 255, 255, 255}, {255, 64, 32, 255}, {16, 128, 240, 255}, {230, 220, 30, 255}, {100, 100, 150, 255},
    {40, 160, 60, 255}, {200, 120, 90, 255}, {12, 20, 40, 255}, {250, 250, 200, 255}, {70, 30, 110, 255},
    {140, 200, 220, 255}, {180, 40, 40, 255}, {90, 90, 30, 255}};
static const float hdr_in[N][3] = {
    {0, 0, 0}, {0.0005f, 0.0005f, 0.0005f}, {0.002f, 0.002f, 0.002f}, {0.01f, 0.01f, 0.01f}, {0.05f, 0.05f, 0.05f},
    {0.18f, 0.18f, 0.18f}, {0.5f, 0.5f, 0.5f}, {1, 1, 1}, {2, 2, 2}, {4, 4, 4}, {8, 8, 8}, {20, 20, 20},
    {1, 0.2f, 0.1f}, {0.05f, 0.3f, 0.9f}, {0.9f, 0.85f, 0.1f}, {3, 1, 0.2f}, {0.2f, 0.6f, 4}, {0.4f, 0.4f, 0.6f},
    {6, 5, 4}, {0.01f, 0.02f, 0.005f}, {1.5f, 0.05f, 0.05f}, {0.03f, 0.6f, 0.08f}, {12, 2, 0.5f}, {0.7f, 0.7f, 0.7f}};
static float hdr_row[2][N][3];   // what the float picture holds: row 0 hdr_in, row 1 the SDR samples' linear values

typedef struct { VkImage img; VmaAllocation alloc; } Img;

static int make_image(VkCtx *c, VkFormat f, uint32_t h, Img *o) {
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = f,
                            .extent = {N, h, 1}, .mipLevels = 1, .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT,
                            .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE};
    return vmaCreateImage(c->vma, &ci, &ai, &o->img, &o->alloc, NULL) == VK_SUCCESS;
}

static void mem_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_HOST_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT |
                                            VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

// The two source pictures (SDR N x 1, float N x 2), uploaded once and left in GENERAL, as the present finds the game's.
static int upload(VkCtx *c, Img *sdr, Img *hdr) {
    if (!make_image(c, VK_FORMAT_R8G8B8A8_UNORM, 1, sdr) || !make_image(c, VK_FORMAT_R16G16B16A16_SFLOAT, 2, hdr))
        return 0;
    VkCommandBuffer cb = vk_upload_begin();
    VkBuffer sb;
    uint8_t *stg = vk_staging(N * 4 + 2 * N * 8, &sb, NULL);
    memcpy(stg, sdr_in, N * 4);
    uint16_t *h = (uint16_t *)(stg + N * 4);
    for (int r = 0; r < 2; r++)
        for (int i = 0; i < N; i++) {
            for (int k = 0; k < 3; k++) {
                float v = r ? (float)srgb_decode(sdr_in[i][k] / 255.0) : hdr_in[i][k];
                h[4 * (r * N + i) + k] = f2h(v);
                hdr_row[r][i][k] = h2f(f2h(v));
            }
            h[4 * (r * N + i) + 3] = f2h(1);
        }
    Img *im[2] = {sdr, hdr};
    for (int i = 0; i < 2; i++) {
        VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                    .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                                    .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                                    .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = im[i]->img,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                               .pImageMemoryBarriers = &ib};
        vkCmdPipelineBarrier2(cb, &di);
        VkBufferImageCopy cp = {.bufferOffset = i ? N * 4 : 0, .imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                .imageExtent = {N, i ? 2u : 1u, 1}};
        vkCmdCopyBufferToImage(cb, sb, im[i]->img, VK_IMAGE_LAYOUT_GENERAL, 1, &cp);
    }
    mem_barrier(cb);
    vk_upload_end();
    vk_upload_flush(1);
    return 1;
}

// One pass (hdr 0: vk_color_record on the SDR picture, 1: vk_hdr_encode on the float one, both rows) read back as
// RGBA16F texels into out (rows of N). 0 when the pass recorded nothing or on a Vulkan failure.
static int run_pass(VkCtx *c, int hdr, const Img *sdr, const Img *hdri, float out[2][N][3], const VkGuestPq *transform) {
    const int crop[4] = {0, 0, N, hdr ? 2 : 1};
    VkCommandBuffer cb = vk_upload_begin();
    VkImage o = hdr ? vk_hdr_encode(cb, 0, hdri->img, VK_FORMAT_R16G16B16A16_SFLOAT, crop, transform)
                    : vk_color_record(cb, 0, sdr->img, VK_FORMAT_R8G8B8A8_UNORM, crop);
    VkBuffer rb = VK_NULL_HANDLE;
    VmaAllocation ra = NULL;
    uint16_t *map = NULL;
    if (o) {
        map = vk_staging(2 * N * 8, &rb, &ra);
        VkBufferImageCopy cp = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                                .imageExtent = {N, (uint32_t)crop[3], 1}};
        vkCmdCopyImageToBuffer(cb, o, VK_IMAGE_LAYOUT_GENERAL, rb, 1, &cp);
        mem_barrier(cb);
    }
    vk_upload_end();
    vk_upload_flush(1);
    if (!o) return 0;
    vmaInvalidateAllocation(c->vma, ra, 0, VK_WHOLE_SIZE);
    for (int r = 0; r < crop[3]; r++)
        for (int i = 0; i < N; i++)
            for (int k = 0; k < 3; k++) out[r][i][k] = h2f(map[4 * (r * N + i) + k]);
    vmaDestroyBuffer(c->vma, rb, ra);
    return 1;
}

static void set_num(const char *name, double v) {
    char s[32];
    snprintf(s, sizeof s, "%g", v);
    cfg_set(name, s);
}

static int g_bad, g_checked, g_same;
static double g_worst[3];   // SDR, HDR10, same meaning

static void check(double got, double want, int what, const char *name, int i, int k) {
    static const char *const kind[3] = {"SDR", "HDR10", "SDR vs HDR10"};
    double d = fabs(got - want);
    g_checked++;
    if (d > g_worst[what]) g_worst[what] = d;
    if (!(d <= TOL) && g_bad++ < 16)
        printf("probe-color: FAIL %s, %s, sample %d channel %d: %.5f, want %.5f\n", name, kind[what], i, k, got, want);
}

static void profile_case(VkCtx *c, const Img *sdr, const Img *hdri, const char *name, Profile p, Nits n) {
    set_num("REKARINA_HDR_GAME_NITS", n.white * n.ui);
    set_num("REKARINA_HDR_UI_NITS", n.ui);
    set_num("REKARINA_HDR_PEAK_NITS", n.peak);
    cfg_set("REKARINA_HDR_GAMMA", n.gamma);
    set_num("REKARINA_COLOR_BRIGHTNESS", p.b), set_num("REKARINA_COLOR_CONTRAST", p.k);
    set_num("REKARINA_COLOR_SATURATION", p.s), set_num("REKARINA_COLOR_GAMMA", p.g);
    int on = p.b || p.k != 100 || p.s != 100 || p.g != 1;
    static float so[2][N][3], ho[2][N][3];
    int sdr_ran = run_pass(c, 0, sdr, hdri, so, NULL);
    if (sdr_ran != on) {
        printf("probe-color: FAIL %s: the SDR pass %s\n", name, sdr_ran ? "ran with the profile off" : "did not run");
        g_bad++;
    }
    if (!run_pass(c, 1, sdr, hdri, ho, NULL)) {
        printf("probe-color: FAIL %s: the HDR10 pass did not run\n", name);
        g_bad++;
        return;
    }
    int same = 0;
    for (int i = 0; i < N; i++) {
        double want[3], lin[3] = {hdr_row[0][i][0], hdr_row[0][i][1], hdr_row[0][i][2]};
        ref_hdr(lin, on, p, n, want);
        for (int k = 0; k < 3; k++) check(ho[0][i][k], want[k], 1, name, i, k);
        int inside;
        ref_sdr(sdr_in[i], p, want, &inside);
        if (sdr_ran)
            for (int k = 0; k < 3; k++) check(so[0][i][k], want[k], 0, name, i, k);
        // the same meaning: the SDR sample's linear value through the HDR10 pass = the PQ code of the SDR pass's result
        if (sdr_ran && inside && n.white == 1) {
            double sl[3];
            for (int k = 0; k < 3; k++) sl[k] = srgb_decode(so[0][i][k]);
            ref_hdr(sl, 0, p, n, want);
            for (int k = 0; k < 3; k++) check(ho[1][i][k], want[k], 2, name, i, k);
            same++;
        }
    }
    g_same += same;
    printf("probe-color: %-44s HDR10 grey 0.18 / 1 / 4 x the HUD's white: %6.1f %6.1f %6.1f nits", name,
           pq_nits(ho[0][5][0]), pq_nits(ho[0][7][0]), pq_nits(ho[0][9][0]));
    if (sdr_ran) printf("; SDR grey 118 -> %.0f, %d SDR samples compared with the HDR10 pass", so[0][5][0] * 255, same);
    printf("\n");
}

static void guest_pq_case(VkCtx *c, const Img *sdr, const Img *hdr) {
    const VkGuestPq transform = {.row = {{127, 74, 2, -99}, {14, 186, 3, -99}, {-20, 4, 219, -99}}, .valid = 1};
    const char *name = "guest-matrix HUD-less PQ";
    float out[2][N][3];
    if (!run_pass(c, 1, sdr, hdr, out, &transform)) {
        printf("probe-color: FAIL guest-matrix PQ pass did not run\n");
        g_bad++;
        return;
    }
    for (int i = 0; i < N; i++) {
        const float *rgb = hdr_row[0][i];
        for (int k = 0; k < 3; k++) {
            const float *row = transform.row[k];
            double v = fma(rgb[2], row[2], fma(rgb[0], row[0], (double)rgb[1] * row[1]));
            double nits = fabs(v);
            check(out[0][i][k], pq(nits), 1, name, i, k);
        }
    }
    printf("probe-color: %s (%d channels compared)\n", name, N * 3);
}

int probe_color(void) {
    static const char *const vars[] = {"COLOR_BRIGHTNESS", "COLOR_CONTRAST", "COLOR_SATURATION",
                                       "COLOR_GAMMA", "HDR_GAME_NITS", "HDR_UI_NITS", "HDR_PEAK_NITS",
                                       "HDR_GAMMA", "HDR", "HDR_EXPOSURE"};
    for (size_t i = 0; i < sizeof vars / sizeof *vars; i++) {
        char modern[64], legacy[64];
        snprintf(modern, sizeof modern, "REKARINA_%s", vars[i]);
        snprintf(legacy, sizeof legacy, "TOTK_%s", vars[i]);
        plat_setenv(modern, NULL);
        plat_setenv(legacy, NULL);
    }
    cfg_set("REKARINA_HDR", "0");
#ifdef GAME_OOT
    static const struct { const char *percent; float gain; } exposure[] = {
        {"0", 0.0f}, {"12.5", 0.125f}, {"100", 1.0f}, {"200", 2.0f}, {"800", 8.0f},
    };
    for (size_t i = 0; i < sizeof exposure / sizeof *exposure; i++) {
        float params[4 * VK_HDR_PARAMS];
        cfg_set("REKARINA_HDR_EXPOSURE", exposure[i].percent);
        vk_hdr_support(params);
        if (params[4] != exposure[i].gain) {
            printf("probe-color: FAIL exposure %s%% gave %g, expected %g\n", exposure[i].percent,
                   params[4], exposure[i].gain);
            return 1;
        }
    }
    cfg_set("REKARINA_HDR_EXPOSURE", "100");
#endif
    if (!vk_init()) {
        printf("probe-color: FAIL vk_init\n");
        return 1;
    }
    VkCtx *c = vk_ctx();
    printf("probe-color: GPU %s\n", c->props.deviceName);
    Img sdr = {0}, hdri = {0};
    if (!upload(c, &sdr, &hdri)) {
        printf("probe-color: FAIL the source pictures could not be made\n");
        return 1;
    }
    const Nits std = {1, 203, 1000, "off"}, bright = {300.0 / 203, 203, 1000, "off"},
               std22 = {1, 203, 1000, "2.2"};
    profile_case(c, &sdr, &hdri, "off (203 nits white, peak 1000)", (Profile){0, 100, 100, 1}, std);
    profile_case(c, &sdr, &hdri, "vivid (203 nits white, peak 1000)", (Profile){0, 115, 135, 0.95}, std);
    profile_case(c, &sdr, &hdri, "+10 / 90 / 70 / 1.1 (203 nits white)", (Profile){10, 90, 70, 1.1}, std);
    profile_case(c, &sdr, &hdri, "vivid (300 nits white, HUD 203)", (Profile){0, 115, 135, 0.95}, bright);
    profile_case(c, &sdr, &hdri, "off (300 nits white, HUD 203)", (Profile){0, 100, 100, 1}, bright);
    profile_case(c, &sdr, &hdri, "off + gamma 2.2 (203 nits, peak 1000)", (Profile){0, 100, 100, 1}, std22);
    guest_pq_case(c, &sdr, &hdri);
    vk_defer_image(sdr.img, sdr.alloc);
    vk_defer_image(hdri.img, hdri.alloc);
    vk_upload_flush(1);
    vk_color_release();
    vk_hdr_release();
    vk_mem_shutdown();
    vk_shutdown();
    long errors = vk_validation_errors_total();
    printf("probe-color: %d values compared, largest differences: SDR %.5f, HDR10 %.5f, SDR vs HDR10 %.5f (%d samples)\n",
           g_checked, g_worst[0], g_worst[1], g_worst[2], g_same);
    if (g_bad || errors || !g_same) {
        printf("probe-color: FAILED (%d differences over %.3f, %ld validation errors)\n", g_bad, TOL, errors);
        return 1;
    }
    printf("probe-color: passed\n");
    return 0;
}
