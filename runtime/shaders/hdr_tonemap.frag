#version 450
// build_shaders: keep functions
//
// Copyright (c) 2025 Carlos Lopez Jr.
// Copyright (C) 2026 Carlos Lopez
// SPDX-License-Identifier: MIT
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
// WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
// OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
// RenoDX HDR tone mapper library (ported from HLSL/Slang in github.com/clshortfuse/renodx: tonemap/psychov/test35.hlsl,
// color/srgb.hlsl and the OoT scene mapper). runtime/vk_hdr.c (tonemap_splice) links nx_tonemap into the game's
// translated ACES tonemap fragment shaders in memory and replaces the three clamped output channel stores.
//
// Parameters (vk_hdr.c vk_hdr_support, support_buffer from byte 1440, VK_HDR_PARAMS_AT):
//   p0: peak nits, diffuse white, compositor white, tone mapper (-1: Vanilla SDR, 0: Vanilla HDR, 8/9: PsychoV35 SDR/HDR)
//   p1: exposure [1], highlights [1], shadows [1], contrast [1]
//   p2: saturation [1], three unused
//   p3: two unused, gamma correction (0 off, 1: 2.2, 2: 2.4) [1], tone slope [1]
//   p4: hue clip emulation [0.5], mid grey (0 render, 1 vanilla, 2 eye adaptation) [1], match tone slope [1], unused
//   p5: bloom scale [1], local exposure scale in stops [1], two unused. The splicer reads p5 in the game's shader.
// main only keeps the entry point alive for glslc; tonemap_splice drops main and splices the helper functions.
layout(location = 0) in vec4 a0;
layout(location = 1) in vec4 a1;
layout(location = 2) in vec4 a2;
layout(location = 3) in vec4 a3;
layout(location = 4) in vec4 a4;
layout(location = 5) in vec4 a5;
layout(location = 6) in vec4 a6;
layout(location = 0) out vec4 o;

// The splice binds the replay's meter only on devices with fragment storage writes.
layout(std430, set = 0, binding = 1) buffer HdrAdaptationMeter {
    vec4 viewport;   // xy origin and zw size in framebuffer pixels
    vec4 anchor;     // previous scene average, output target, minimum and maximum output relative to target
    uint samples[];
} hdr_meter;

vec4 hdr_measure(vec4 scene, vec4 frag, vec4 p4) {
    if (p4.y != 2.0) return vec4(0.0);
    vec2 size = hdr_meter.viewport.zw;
    const ivec2 grid = ivec2(32, 18);
    if (any(lessThan(size, vec2(grid)))) return hdr_meter.anchor;
    vec2 pixel = floor(frag.xy - hdr_meter.viewport.xy);
    ivec2 cell = ivec2(floor((pixel + 0.5) * vec2(grid) / size));
    if (any(lessThan(cell, ivec2(0))) || any(greaterThanEqual(cell, grid))) return hdr_meter.anchor;
    if (all(equal(pixel, floor((vec2(cell) + 0.5) * size / vec2(grid)))) && scene.a > 0.0) {
        float luma = dot(scene.rgb / scene.a, vec3(0.2126, 0.7152, 0.0722));
        if (!isnan(luma) && !isinf(luma))
            atomicExchange(hdr_meter.samples[cell.y * grid.x + cell.x], floatBitsToUint(max(luma, 1e-6)));
    }
    return hdr_meter.anchor;
}

// Row-major matrix-vector multiply (matrices below are written row by row).
vec3 mulm(mat3 m, vec3 v) { return v * m; }

float y709(vec3 c) { return dot(c, vec3(0.2126390059, 0.7151686788, 0.0721923154)); }

vec3 copy_sign(vec3 mag, vec3 s) {
    vec3 a = abs(mag);
    return mix(a, -a, lessThan(s, vec3(0.0)));
}

float srgb_decode(float c) { return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4); }

float gamma_to_srgb(float c, float g) { return srgb_decode(pow(c, 1.0 / g)); }

vec3 render_eotf(vec3 linear, float mode) {
    if (mode == 0.0) return linear;
    vec3 c = abs(linear);
    vec3 code = mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, greaterThan(c, vec3(0.0031308)));
    return copy_sign(pow(code, vec3(mode == 1.0 ? 2.2 : 2.4)), linear);
}

// Fit against the inverse EOTF peak so neutral highlights still reach the configured display peak.
float peak_before_gamma(float peak, float gamma) {
    return gamma == 0.0 ? peak : gamma_to_srgb(peak, gamma == 1.0 ? 2.2 : 2.4);
}

float smoother(float t) { return t * t * t * (t * (t * 6.0 - 15.0) + 10.0); }

vec3 aces_fitted(vec3 x) {
    return clamp(x * (2.51 * x + 0.03) / (x * (2.43 * x + 0.59) + 0.14), 0.0, 1.0);
}

// Generic scene grading, following RenoDX colorgrade.hlsl UserColorGrading (v4 luminance curves).
vec3 graded_aces(vec3 color, vec4 grade, vec4 chroma) {
    const float mid = 0.18;
    color *= grade.x;
    float y = y709(color);
    if (y > 0.0 && grade.yzw != vec3(1.0)) {
        float value = mid * pow(y / mid, grade.w);
        if (value > mid && grade.y != 1.0) {
            float blend = smoother(clamp(log2(value / mid) / log2(1.0 / mid), 0.0, 1.0));
            float power = grade.y > 1.0 ? grade.y : 2.0 - grade.y;
            float shaped = mix(value, mid * pow(value / mid, power), blend);
            value = grade.y > 1.0 ? shaped : value * (value / shaped);
        }
        if (value > 0.0 && value < mid && grade.z != 1.0) {
            float ratio = value / mid;
            float blend = smoother(clamp(log2(ratio) / -4.0, 0.0, 1.0));
            float power = grade.z > 1.0 ? grade.z : 2.0 - grade.z;
            float delta = mid * mid * value * (pow(ratio, 1.0 - power) - 1.0);
            value += (grade.z > 1.0 ? delta : -delta) * blend;
        }
        color *= value / y;
    }
    if (chroma.x != 1.0) {
        const mat3 to_lms = mat3(0.41222147, 0.53633254, 0.05144599,
                                 0.21190350, 0.68069955, 0.10739696,
                                 0.08830246, 0.28171884, 0.62997870);
        const mat3 to_lab = mat3(0.21045426, 0.79361779, -0.00407205,
                                 1.97799850, -2.42859221, 0.45059371,
                                 0.02590404, 0.78277177, -0.80867577);
        const mat3 from_lab = mat3(1.0, 0.39633778, 0.21580376,
                                   1.0, -0.10556135, -0.06385417,
                                   1.0, -0.08948418, -1.29148555);
        const mat3 from_lms = mat3(4.07674166, -3.30771159, 0.23096993,
                                   -1.26843800, 2.60975740, -0.34131940,
                                   -0.00419609, -0.70341861, 1.70761470);
        vec3 lms = mulm(to_lms, color);
        vec3 lab = mulm(to_lab, copy_sign(pow(abs(lms), vec3(1.0 / 3.0)), lms));
        lab.yz *= chroma.x;
        lms = mulm(from_lab, lab);
        color = mulm(from_lms, lms * lms * lms);
    }
    return aces_fitted(color);
}

// ---- PsychoV35 observer-response cone mapper (Stockman-CVRL 2-degree LMS)

const mat3 PSY_BT709_TO_LMS = mat3(0.2896057766, 0.6972466442, 0.0763712786,
                                   0.0901837576, 0.7073490257, 0.1122031464,
                                   0.0155287401, 0.0536093073, 0.5097978305);
const mat3 PSY_LMS_TO_BT709 = mat3(4.9676196505, -4.9223795294, 0.3391991812,
                                   -0.6196828788, 2.0517506515, -0.3587439844,
                                   -0.0861520040, -0.0658193830, 1.9889544833);
const mat3 PSY_LMS_TO_BT2020 = mat3(2.9089213535, -2.4155642445, 0.1808342948,
                                    -0.2275532728, 1.5457966612, -0.2838427126,
                                    -0.0502712319, 0.0409489515, 1.7552839120);
const vec3 PSY_YF_WEIGHTS = vec3(0.68990272, 0.34832189, 0.0);
const vec3 PSY_D65_WHITE_LMS = vec3(1.0632236993, 0.9097359297, 0.5789358779);

float psy_mean(vec3 v) { return v.x + ((v.y - v.x) + (v.z - v.x)) / 3.0; }

vec3 psy_rgb_residual(vec3 normalized_lms) { return mulm(PSY_LMS_TO_BT709, normalized_lms * PSY_D65_WHITE_LMS); }

void psy_response_logs(float log_q, float k, float n, out float log_u, out float log_deficit) {
    float latent = log(k) - log(1.0 - k) + n * log_q;
    float correction = log(1.0 + exp(-abs(latent)));
    log_u = min(latent, 0.0) - correction;
    float log_tail = -max(latent, 0.0) - correction;
    log_deficit = latent > 20.0 ? log_tail
                                : log(abs(log_u) < 0.01
                                          ? -log_u * (1.0 + log_u * (0.5 + log_u * (1.0 / 6.0 + log_u / 24.0)))
                                          : 1.0 - exp(log_u));
}

// Return linear BT.709 for composition in both output modes.
vec3 psycho35(vec3 bt709, float peak, float highlights, float shadows, float contrast, float purity_scale, float cone,
              float anchor_in_gray, float anchor_out_gray, bool hdr) {
    vec3 input_lms = mulm(PSY_BT709_TO_LMS, bt709);
    vec3 anchor_in = mulm(PSY_BT709_TO_LMS, vec3(anchor_in_gray));
    vec3 anchor_out = mulm(PSY_BT709_TO_LMS, vec3(anchor_out_gray));
    if (highlights != 1.0 || shadows != 1.0) {
        float input_yf = dot(PSY_YF_WEIGHTS, input_lms);
        float anchor_yf = dot(PSY_YF_WEIGHTS, anchor_in);
        float graded_y = input_yf;
        if (highlights != 1.0) {
            float ratio = graded_y / anchor_yf;
            float blend = graded_y > anchor_yf ? clamp(log2(ratio) / log2(1.0 / anchor_yf), 0.0, 1.0) : 0.0;
            blend = smoother(blend);
            float shaped = pow(ratio, abs(highlights - 1.0));
            graded_y = highlights > 1.0 ? graded_y * mix(1.0, shaped, blend) : graded_y / mix(1.0, shaped, blend);
        }
        if (shadows != 1.0) {
            float ratio = graded_y / anchor_yf;
            float blend = graded_y > anchor_yf * exp2(-4.0) ? clamp(log2(ratio) / -4.0, 0.0, 1.0) : 1.0;
            blend = smoother(blend);
            float target_y = shadows > 1.0
                                 ? graded_y * (1.0 - anchor_yf * anchor_yf) + anchor_yf * anchor_yf * anchor_yf * pow(ratio, 2.0 - shadows)
                                 : graded_y * (1.0 + anchor_yf * anchor_yf) - anchor_yf * anchor_yf * anchor_yf * pow(ratio, shadows);
            graded_y = mix(graded_y, target_y, blend);
        }
        if (input_yf != 0.0) input_lms *= graded_y / input_yf;
    }
    vec3 source_ratio = input_lms / anchor_in;
    if (purity_scale != contrast) {
        float neutral_ratio = dot(PSY_YF_WEIGHTS, input_lms) / dot(PSY_YF_WEIGHTS, anchor_in);
        vec3 purity_direction = source_ratio - neutral_ratio;
        float purity_amount = purity_scale / contrast;
        if (all(greaterThanEqual(source_ratio, vec3(0.0))) && neutral_ratio > 0.0 && purity_amount > 1.0) {
            for (int c = 0; c < 3; ++c)
                if (purity_direction[c] < 0.0) purity_amount = min(purity_amount, neutral_ratio / -purity_direction[c]);
            source_ratio = max(neutral_ratio + purity_amount * purity_direction, 0.0);
        } else {
            source_ratio = neutral_ratio + purity_amount * purity_direction;
        }
    }
    vec3 anchor_power = anchor_out / (PSY_D65_WHITE_LMS * peak);
    vec3 input_exponent = contrast * cone / (1.0 - anchor_power);
    if (all(equal(source_ratio, vec3(0.0)))) return vec3(0.0);
    float drive = max(abs(source_ratio.x), max(abs(source_ratio.y), abs(source_ratio.z)));
    vec3 ratio = source_ratio / drive;
    vec3 log_ratio = log(vec3(ratio.x != 0.0 ? ratio.x : 1.0, ratio.y != 0.0 ? ratio.y : 1.0, ratio.z != 0.0 ? ratio.z : 1.0));
    vec3 response, deficit;
    for (int i = 0; i < 3; ++i) {
        if (ratio[i] == 0.0) {
            response[i] = 0.0;
            deficit[i] = 1.0;
        } else {
            float log_u, log_deficit;
            psy_response_logs(log_ratio[i] + log(drive), anchor_power[i], input_exponent[i], log_u, log_deficit);
            response[i] = exp(log_u);
            deficit[i] = exp(log_deficit);
        }
    }
    float response_mean = psy_mean(response);
    vec3 response_residual = response_mean > 0.5 ? psy_mean(deficit) - deficit : response - response_mean;
    float residual_scale = max(abs(response_residual.x), max(abs(response_residual.y), abs(response_residual.z)));
    vec3 scaled_residual = residual_scale > 0.0 ? response_residual / residual_scale : vec3(0.0);
    float scaled_radius = length(scaled_residual);
    float radius = residual_scale * scaled_radius;
    vec3 direction = scaled_radius > 0.0 ? scaled_residual / scaled_radius : vec3(0.0);
    vec3 source = ratio - psy_mean(ratio);
    float source_radius = length(source);
    if (source_radius > 0.0 && radius > 0.0) {
        vec3 midpoint_direction = source / source_radius + direction;
        float midpoint_length = length(midpoint_direction);
        if (midpoint_length > 0.0) direction = midpoint_direction / midpoint_length;
    }
    float mean = peak * response_mean, headroom = peak * psy_mean(deficit);
    vec3 residual = peak * radius * direction;
    if (all(equal(residual, vec3(0.0)))) return vec3(mean);
    vec3 displacement = mulm(hdr ? PSY_LMS_TO_BT2020 : PSY_LMS_TO_BT709, residual * PSY_D65_WHITE_LMS);
    float scale = 1.0;
    for (int channel = 0; channel < 3; ++channel) {
        if (displacement[channel] > 0.0) scale = min(scale, headroom / displacement[channel]);
        else if (displacement[channel] < 0.0) scale = min(scale, mean / -displacement[channel]);
    }
    return mean + psy_rgb_residual(residual * scale);
}

vec3 hue_clip(vec3 untonemapped, float amount) {
    vec3 magnitude = abs(untonemapped);
    vec3 reference = aces_fitted(magnitude);
    float low = min(reference.x, min(reference.y, reference.z));
    float range = max(reference.x, max(reference.y, reference.z)) - low;
    bool neutral = range == 0.0;
    float mag_low = min(magnitude.x, min(magnitude.y, magnitude.z));
    float mag_high = max(magnitude.x, max(magnitude.y, magnitude.z));
    vec3 transferred = mix(vec3(mag_low), vec3(mag_high), (reference - low) / (neutral ? 1.0 : range));
    transferred = copy_sign(transferred, untonemapped);
    if (neutral) transferred = untonemapped;
    vec3 bent = mix(untonemapped, transferred, amount);
    float original_y = y709(untonemapped), bent_y = y709(bent);
    if (original_y > 0.0 && bent_y > 0.0) untonemapped = bent * (original_y / bent_y);
    return untonemapped;
}

vec3 scene_tonemap(vec3 untonemapped, float gain, float peak, vec4 p1, vec4 p2, vec4 p3, vec4 p4, vec4 adaptation, bool hdr) {
    if (p1.x == 0.0) return vec3(0.0);
    if (p4.x > 0.0) untonemapped = hue_clip(untonemapped, p4.x);
    // Analytic log-slope x*V'(x)/V(x) at V''(x)=0 (x=0.120305925) of the game's ACES fit.
    float cone = p4.z != 0.0 ? 1.3355037565264822 : 1.0;
    float anchor_in = 0.18;
    float anchor_out = (p4.y == 0.0 ? 0.18 : aces_fitted(vec3(0.18)).x) * gain * p1.x;
    if (adaptation.x > 0.0) {
        anchor_in = adaptation.x;
        anchor_out = clamp(adaptation.y * anchor_in / 0.18, adaptation.y * adaptation.z, adaptation.y * adaptation.w);
        anchor_out = min(anchor_out * p1.x, 0.95 * peak);
    }
    return psycho35(untonemapped, peak, p1.y, p1.z, p1.w, p2.x, cone * p3.w, anchor_in, anchor_out, hdr);
}

vec3 nx_tonemap(vec3 x, vec4 scene, vec4 p0, vec4 p1, vec4 p2, vec4 p3, vec4 p4, vec4 adaptation) {
    vec3 vanilla = aces_fitted(x);
    if (p0.w < 0.0) return vanilla;   // SDR swapchain: the game's own SDR result
    vec3 color = vanilla;             // tone mapper 0: vanilla curve scaled to picture white
    if (p0.w == 8.0 || p0.w == 9.0) {
        if (!(scene.a > 0.0)) return vec3(0.0);
        float peak = peak_before_gamma(max(p0.x, p0.y) / p0.y, p3.z);
        color = render_eotf(scene_tonemap(scene.rgb / scene.a, scene.a, peak, p1, p2, p3, p4, adaptation, p0.w == 9.0), p3.z);
    }
    color *= p0.y / p0.z;   // picture white relative to HUD white
    return mix(color, vec3(0.0), isnan(color));
}

void main() {
    o = vec4(a2.w == -2.0 ? graded_aces(a0.xyz, a3, a4)
                          : nx_tonemap(a0.xyz, a1, a2, a3, a4, a5, a6, hdr_measure(a1, gl_FragCoord, a6)), 1.0);
}
