// Included by vk_hdr.c for OoT. The meter sees the same pre-tone-map scene as PsychoV35.
// Eye adaptation controller adapted from RenoDX, Copyright (C) 2026 Carlos Lopez, SPDX-License-Identifier: MIT.
// RenoDX OoT adaptation.slangi supplies the weighted log histogram and temporal controller.
// The replay reads back only 32 x 18 luminance values after their submission's timeline fence.
enum { METER_W = 32, METER_H = 18, METER_SAMPLES = METER_W * METER_H, METER_BINS = 64, METER_SLOTS = 8 };

typedef struct {
    float viewport[4], anchor[4];   // std430 HdrAdaptationMeter in shaders/hdr_tonemap.frag
    float samples[METER_SAMPLES];   // the shader atomically stores the bit pattern of each float
} HdrMeterData;
_Static_assert(offsetof(HdrMeterData, samples) == 32, "HDR meter shader layout");

typedef struct {
    VkBuffer buffer;
    VmaAllocation allocation;
    HdrMeterData *map;
    uint64_t fence, time_ns, serial;
    float fov[2];   // tangent of the horizontal and vertical half angles
    int scene_fov;
} HdrMeterSlot;

typedef struct {
    float filtered_log, fast_log, slow_log, bleached;
    uint64_t time_ns;
} HdrAdaptation;

static struct {
    HdrMeterSlot slots[METER_SLOTS];
    HdrAdaptation history;
    uint64_t serial, consumed, sample_count;
    int failed, enabled;
    float weight_fov[2];
    unsigned weights[METER_SAMPLES];
    PlatMutex lock;
} A = {.lock = PLAT_MUTEX_INIT};

// Watson's binocular midget ganglion-cell density supplies the spatial weights used by RenoDX
// (Journal of Vision 14(7):15, https://doi.org/10.1167/14.7.15).
static unsigned meter_weight(float x, float y) {
    static const float a[4] = {0.9851f, 0.9935f, 0.9729f, 0.996f};
    static const float radius[4] = {1.058f, 1.035f, 1.084f, 0.9932f};
    static const float falloff[4] = {22.14f, 16.35f, 7.633f, 12.13f};
    float eccentricity = hypotf(x, y), density[4];
    for (int i = 0; i < 4; i++) {
        float scale = 1.0f + eccentricity / radius[i];
        density[i] = (a[i] / (scale * scale) + (1.0f - a[i]) * expf(-eccentricity / falloff[i])) /
                     (1.0f + eccentricity / 41.03f);
    }
    float vert = y * y / density[y >= 0.0f ? 1 : 3], xx = x * x;
    float weight = 0.5f * eccentricity * eccentricity *
                   (1.0f / (vert + xx / density[0]) + 1.0f / (vert + xx / density[2]));
    return (unsigned)fmaxf(1.0f, roundf(weight * 1024.0f));
}

static void meter_weights(const float fov[2]) {
    if (fov[0] == A.weight_fov[0] && fov[1] == A.weight_fov[1]) return;
    const float degrees = 57.2957795f;
    float x[METER_W], y[METER_H];
    for (int i = 0; i < METER_W; i++) x[i] = degrees * atanf((2.0f * (i + 0.5f) / METER_W - 1.0f) * fov[0]);
    for (int i = 0; i < METER_H; i++) y[i] = degrees * atanf((1.0f - 2.0f * (i + 0.5f) / METER_H) * fov[1]);
    for (int row = 0; row < METER_H; row++)
        for (int col = 0; col < METER_W; col++) A.weights[row * METER_W + col] = meter_weight(x[col], y[row]);
    memcpy(A.weight_fov, fov, sizeof A.weight_fov);
}

static int meter_average(const HdrMeterSlot *slot, float *log_average) {
    unsigned histogram[METER_BINS] = {0}, total = 0;
    int samples = 0;
    meter_weights(slot->fov);
    for (int i = 0; i < METER_SAMPLES; i++) {
        float luma = slot->map->samples[i];
        if (!(luma > 0.0f) || !isfinite(luma)) continue;
        int bin = (int)clampf(floorf((log2f(luma) + 16.0f) * 2.0f), 0.0f, METER_BINS - 1.0f);
        histogram[bin] += A.weights[i];
        total += A.weights[i];
        samples++;
    }
    if (samples < METER_SAMPLES / 2) return 0;   // a partial draw is not the scene
    // Trim 5% from both tails, including the weights that fall inside a boundary bin.
    unsigned trim = total / 20, begin = 0;
    float sum = 0.0f;
    for (int bin = 0; bin < METER_BINS; bin++) {
        unsigned end = begin + histogram[bin];
        unsigned lo = begin > trim ? begin : trim, hi = end < total - trim ? end : total - trim;
        if (hi > lo) sum += (-16.0f + (bin + 0.5f) * 0.5f) * (hi - lo);
        begin = end;
    }
    *log_average = sum / (total - 2 * trim);
    return samples;
}

static float adaptation_approach(float value, float target, float dt, float tau) {
    return value + (target - value) * -expm1f(-dt / tau);
}

static float adaptation_anchor(const HdrAdaptation *h) {
    return h->time_ns ? exp2f(0.75f * h->fast_log + 0.25f * h->slow_log) : 0.18f;
}

static void adaptation_update(HdrAdaptation *h, float log_average, uint64_t time_ns) {
    float dt = (float)(time_ns - h->time_ns) * 1e-9f;
    if (!h->time_ns || dt > 1.0f) {
        float neutral = log2f(0.18f);
        *h = (HdrAdaptation){log_average, neutral, neutral, 0.0f, time_ns};
        return;
    }
    h->time_ns = time_ns;
    h->filtered_log = adaptation_approach(h->filtered_log, log_average, dt, 0.1f);
    float previous = 0.75f * h->fast_log + 0.25f * h->slow_log;
    float slow_tau = h->filtered_log < previous ? 4.3f * (1.0f + 3.0f * h->bleached) : 4.3f;
    h->fast_log = adaptation_approach(h->fast_log, h->filtered_log, dt, 1.0f);
    h->slow_log = adaptation_approach(h->slow_log, h->filtered_log, dt, slow_tau);

    float pupil = clampf(5.0f - 3.0f * tanhf(0.4f * (log_average * 0.3010299957f - 0.5f)), 2.0f, 8.0f);
    float retinal = exp2f(log_average) * 0.7853981634f * pupil * pupil;
    float target = retinal / (retinal + 20000.0f);
    h->bleached = adaptation_approach(h->bleached, target, dt, target > h->bleached ? 1.0f : 4.3f);
}

static int meter_alloc(HdrMeterSlot *slot) {
    if (slot->buffer) return 1;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = sizeof(HdrMeterData),
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    VmaAllocationCreateInfo ai = {.usage = VMA_MEMORY_USAGE_AUTO,
        .flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT};
    VmaAllocationInfo info;
    VmaAllocator allocator = vk_ctx()->vma;
    if (vmaCreateBuffer(allocator, &bi, &ai, &slot->buffer, &slot->allocation, &info) != VK_SUCCESS || !info.pMappedData) {
        if (slot->buffer) vmaDestroyBuffer(allocator, slot->buffer, slot->allocation);
        *slot = (HdrMeterSlot){0};
        A.failed = 1;
        rt_log("hdr: eye adaptation meter allocation failed; using the fixed mid-grey anchor");
        return 0;
    }
    slot->map = info.pMappedData;
    return 1;
}

static void meter_collect(uint64_t done, uint64_t now) {
    HdrMeterSlot *newest = NULL;
    uint64_t consumed = A.consumed;
    int samples = 0;
    float log_average = 0.0f;
    for (int i = 0; i < METER_SLOTS; i++) {
        HdrMeterSlot *slot = &A.slots[i];
        if (!slot->fence || slot->fence > done || slot->serial <= A.consumed) continue;
        if (slot->serial > consumed) consumed = slot->serial;
        if (now - slot->time_ns > UINT64_C(1000000000) || (newest && slot->serial < newest->serial)) continue;
        if (vmaInvalidateAllocation(vk_ctx()->vma, slot->allocation, 0, VK_WHOLE_SIZE) != VK_SUCCESS) continue;
        float candidate;
        int n = meter_average(slot, &candidate);
        if (n) newest = slot, samples = n, log_average = candidate;
    }
    A.consumed = consumed;
    if (A.history.time_ns && now - A.history.time_ns > UINT64_C(1000000000)) A.history = (HdrAdaptation){0};
    if (!newest) return;
    adaptation_update(&A.history, log_average, newest->time_ns);
    if (++A.sample_count == 1 || (rt_trace & RT_TRACE_GFX && A.sample_count % 120 == 0))
        rt_log("hdr: eye adaptation measured %d/%d points (scene average %.4f, smoothed anchor %.4f, %s FOV)",
               samples, METER_SAMPLES, exp2f(log_average), adaptation_anchor(&A.history),
               newest->scene_fov ? "scene" : "default");
}

VkDescriptorBufferInfo vk_hdr_meter_bind(VkExec *ex) {
    Nits n = nits();
    int enabled = (vk_hdr_renodx_scene() || vk_hdr_sdr_scene()) &&
                  !strcmp(cfg_value("REKARINA_HDR_MIDGRAY"), "adaptation");
    VkDescriptorBufferInfo result = {0};
    plat_mutex_lock(&A.lock);
    if (!enabled) {
        if (A.enabled) {
            A.history = (HdrAdaptation){0};
            A.consumed = A.serial;
        }
        A.enabled = 0;
        goto done;
    }
    A.enabled = 1;
    if (A.failed || !ex->fence || ex->pass_ncolor != 1 || ex->color[0].samples != 1 ||
        (ex->depth.image && (ex->pipe.ds.dt || ex->pipe.ds.dw)) ||
        ex->pass_w < METER_W || ex->pass_h < METER_H) goto done;
    uint64_t now = plat_now_ns(), done_fence = vk_sync_fence_done();
    meter_collect(done_fence, now);
    HdrMeterSlot *slot = NULL;
    for (int i = 0; i < METER_SLOTS; i++) {
        if (A.slots[i].fence > done_fence) continue;
        slot = &A.slots[i];
        break;
    }
    if (!slot || !meter_alloc(slot)) goto done;

    float scale = ex->pass_scale > 1 ? (float)ex->pass_scale : 1.0f;
    float width = ex->has_vp ? ex->vp[0][2] * scale : (float)ex->pass_w;
    float height = ex->has_vp ? ex->vp[0][3] * scale : (float)ex->pass_h;
    if (width < METER_W || height < METER_H) goto done;
    *slot->map = (HdrMeterData){
        .viewport = {ex->has_vp ? ex->vp[0][0] * scale : 0.0f,
                     ex->has_vp ? ex->vp[0][1] * scale : 0.0f, width, height},
        .anchor = {adaptation_anchor(&A.history), n.adaptation_target, n.adaptation_min, n.adaptation_max},
    };
    // The composite can bind an orthographic camera after the perspective scene was drawn.
    uint64_t frame = __atomic_load_n(&gfx_counters.frames, __ATOMIC_RELAXED) + 1;
    int has_projection = ex->hdr_meter_projection_frame && frame - ex->hdr_meter_projection_frame <= 2;
    float fov_x = has_projection ? fabsf(ex->hdr_meter_projection[0]) : 0.0f;
    float fov_y = has_projection ? fabsf(ex->hdr_meter_projection[1]) : 0.0f;
    // The CPU jitter hook can be off. Use a 60-degree vertical field of view until a projection is available.
    float fallback_y = 0.577350269f;
    slot->scene_fov = fov_x > 0.0087f && isfinite(fov_x) && fov_y > 0.0087f && isfinite(fov_y);
    slot->fov[0] = slot->scene_fov ? clampf(1.0f / fov_x, 0.008726867f, 114.58865f) : fallback_y * width / height;
    slot->fov[1] = slot->scene_fov ? clampf(1.0f / fov_y, 0.008726867f, 114.58865f) : fallback_y;
    if (vmaFlushAllocation(vk_ctx()->vma, slot->allocation, 0, VK_WHOLE_SIZE) != VK_SUCCESS) goto done;
    slot->fence = ex->fence;
    slot->time_ns = now;
    slot->serial = ++A.serial;
    ex->hdr_meter_written = 1;
    result = (VkDescriptorBufferInfo){slot->buffer, 0, sizeof(HdrMeterData)};
done:
    plat_mutex_unlock(&A.lock);
    return result;
}

void vk_hdr_meter_end_pass(VkExec *ex) {
    VkMemoryBarrier2 barrier = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .srcStageMask = VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT, .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo dep = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &barrier};
    vkCmdPipelineBarrier2(ex->cb, &dep);
    ex->hdr_meter_written = 0;
    ex->recorded++;
}

static void hdr_meter_release(VmaAllocator allocator) {
    for (int i = 0; i < METER_SLOTS; i++) {
        HdrMeterSlot *slot = &A.slots[i];
        if (slot->buffer) vmaDestroyBuffer(allocator, slot->buffer, slot->allocation);
        *slot = (HdrMeterSlot){0};
    }
    A.history = (HdrAdaptation){0};
    A.serial = A.consumed = A.sample_count = 0;
    A.enabled = A.failed = 0;
}
