// Included by patches.c for the checked OoT 1.1.1 call sites.
// Adapted from NX2navi (MIT, Copyright (c) 2026 Carlos Lopez Jr.).
#include "nvn_obj.h"

static int scene_projection_set = -1, prebloom_set = -1;

int vk_dlaa_prebloom_enabled(void) {
    return rt_patch_enabled(prebloom_set);
}

int scene_projection_enabled(void) {
    if (scene_projection_set < 0 || !rt_patch_enabled(scene_projection_set)) return 0;
    static _Thread_local unsigned version;
    static _Thread_local int jitter, known;
    unsigned now = cfg_version();
    if (!known || now != version) {
        jitter = rt_knob("DLAA_JITTER", "1");
        known = 1, version = now;
    }
    return jitter;
}

#if defined(GAME_OOT) && defined(RT_PATCH_PAGE)
#ifndef MOD_MAIN_BASE
#define MOD_MAIN_BASE 0x7100000000ull
#endif
#include "layout.h"
#include "vk_dlaa.h"

static PlatMutex projection_lock = PLAT_MUTEX_INIT;
static SceneProjection pending;
static uint64_t owner, records_generation, generation, serial, completed, invalidated;
static struct {
    SceneProjection sample;
    uint32_t width, height, phases;
} readiness;
static uint32_t index;

// A resolved sample is a short lease: CPU recording may be ahead of replay, but not indefinitely.
#define SCENE_PROJECTION_MAX_LAG 12

static uint64_t projection_sequence(uint64_t token) { return token & ~SCENE_PROJECTION_SHIFTED; }

static void projection_invalidate(void) {
    invalidated = serial;
    readiness.sample.token = 0;
}

static uint64_t scene_read64(uint64_t address) {
    uint64_t value;
    memcpy(&value, (const void *)(uintptr_t)address, sizeof value);
    return value;
}

static uint32_t scene_read32(uint64_t address) {
    uint32_t value;
    memcpy(&value, (const void *)(uintptr_t)address, sizeof value);
    return value;
}

static uint64_t scene_camera(uint64_t record) {
    uint64_t backing = scene_read64(record + 0x3f8), desc = scene_read64(record + 0x3f0);
    uint64_t layout = desc ? scene_read64(desc) : 0;
    return backing && layout ? backing + scene_read32(layout + 4) : 0;
}

static float scene_halton(uint64_t i, unsigned base) {
    float scale = 1, result = 0;
    for (; i; i /= base) {
        scale /= base;
        result += scale * (i % base);
    }
    return result;
}

static const void *scene_attachment(uint64_t bundle, uint32_t slot, uint64_t registry, uint8_t view[16]);

static uint64_t scene_registry(void) {
    uint64_t reg_owner = scene_read64(MOD_MAIN_BASE + 0x4c82d60);
    return reg_owner ? scene_read64(reg_owner) : 0;
}

static void scene_projection_build_(Cpu *c) {
    Cpu input = *c;
    c->pc = MOD_MAIN_BASE + 0x33a8c0;
    rt_dispatch(c);
    if (!scene_projection_enabled()) return;
    SceneProjection s = {0};
    plat_mutex_lock(&projection_lock);
    if (owner && input.x[21] == owner && (uint32_t)input.x[19] == index) {
        const CfgRenderValues *settings = cfg_render_values();
        s = (SceneProjection){.owner = owner, .index = index, .token = ++serial,
                              .settings = settings ? settings->projection_revision : 0,
                              .generation = generation};
        if (pending.token) projection_invalidate();
        pending = (SceneProjection){0};
    }
    typeof(readiness) lease = readiness;
    uint64_t expected_records = records_generation;
    plat_mutex_unlock(&projection_lock);
    if (!s.token || s.index >= scene_read32(s.owner + 0x10)) return;
    uint64_t records = scene_read64(s.owner + 0x18), record = records + s.index * UINT64_C(0x480);
    uint64_t projection = input.x[2];
    if (!records || records != expected_records || input.x[0] != record + 0x60 || !projection ||
        scene_read64(projection) != MOD_MAIN_BASE + 0x4bf8c70 || scene_read32(projection + 0x8c)) return;
    s.camera = scene_camera(record);
    if (!s.camera) return;
    memcpy(s.view, (const void *)(uintptr_t)record, sizeof s.view);
    memcpy(s.unjittered, (const void *)(uintptr_t)(record + 0x1a8), sizeof s.unjittered);
    uint64_t bundle = scene_read64(s.owner + 0x520), registry = scene_registry();
    uint8_t cv[16] = {0}, dv[16] = {0};
    const void *color = scene_attachment(bundle, 0x20, registry, cv);
    const void *depth = scene_attachment(bundle, 0x60, registry, dv);
    int extent = scene_projection_extent(color, cv, &s.native_extent[0], &s.native_extent[1]) &&
                 scene_projection_extent(depth, dv, &s.native_extent[2], &s.native_extent[3]);
    const uint64_t lag = s.token - projection_sequence(lease.sample.token);
    // The frame's attachment bundle can still be unset during projection construction.
    // The scene-complete hook checks the actual subresources and extent before replay accepts the sample.
    if (!extent) memcpy(s.native_extent, lease.sample.native_extent, sizeof s.native_extent);
    const CfgRenderValues *settings = cfg_render_values();
    int eligible = (settings ? settings->dlaa : cfg_bool("REKARINA_DLAA")) && lease.sample.token &&
        lag <= SCENE_PROJECTION_MAX_LAG && lease.phases &&
        lease.sample.generation == s.generation && lease.sample.settings == s.settings &&
        !memcmp(lease.sample.native_extent, s.native_extent, sizeof s.native_extent);
    plat_mutex_lock(&projection_lock);
    eligible = eligible && lease.sample.generation == generation && projection_sequence(lease.sample.token) > invalidated;
    static unsigned applied_logs, blocked_logs;
    if ((eligible && applied_logs++ < 6) || (!eligible && lease.sample.token && blocked_logs++ < 12))
        rt_log("dlaa: CPU projection %s token=%#llx lag=%llu native=%ux%u/%ux%u lease=%ux%u/%ux%u render=%ux%u bank-rotated=%d",
               eligible ? "armed" : "waiting", (unsigned long long)s.token, (unsigned long long)lag,
               s.native_extent[0], s.native_extent[1], s.native_extent[2], s.native_extent[3],
               lease.sample.native_extent[0], lease.sample.native_extent[1], lease.sample.native_extent[2],
               lease.sample.native_extent[3], lease.width, lease.height, s.camera != lease.sample.camera);
    plat_mutex_unlock(&projection_lock);
    if (eligible) {
        s.width = lease.width, s.height = lease.height;
        _Alignas(16) uint8_t copy[0xc0] = {0};
        memcpy(copy, (const void *)(uintptr_t)projection, 0xbc);
        uint64_t phase = (s.token - 1) % lease.phases + 1;
        s.jitter[0] = scene_halton(phase, 2) - 0.5f;
        s.jitter[1] = scene_halton(phase, 3) - 0.5f;
        float off[2];
        memcpy(off, copy + 0xb4, sizeof off);
        // The builder doubles these terms into P02/P12. With P32 = -1 and a negative-height
        // viewport they move the raster right/down by the upper-left-origin pixel offsets.
        off[0] -= s.jitter[0] / s.width;
        off[1] += s.jitter[1] / s.height;
        memcpy(copy + 0xb4, off, sizeof off);
        copy[8] = copy[9] = 1;   // rebuild and convert before computing the dependent matrices
        Cpu result = *c;
        *c = input;
        c->x[2] = (uint64_t)(uintptr_t)copy;
        c->pc = MOD_MAIN_BASE + 0x33a8c0;
        rt_dispatch(c);
        *c = result;
        s.token |= SCENE_PROJECTION_SHIFTED;
    }

    memcpy(s.projection, (const void *)(uintptr_t)(record + 0x1a8), sizeof s.projection);
    plat_mutex_lock(&projection_lock);
    if (s.generation == generation && projection_sequence(s.token) == serial) pending = s;
    plat_mutex_unlock(&projection_lock);
}

static void scene_projection_build(Cpu *c) {
    const CfgRenderState *settings = cfg_render_frame_acquire();
    const CfgRenderState *previous = cfg_render_use(settings);
    scene_projection_build_(c);
    cfg_render_use(previous);
    cfg_render_release(settings);
}

static const void *scene_attachment(uint64_t bundle, uint32_t slot, uint64_t registry, uint8_t view[16]) {
    uint64_t desc = bundle ? scene_read64(bundle + slot) : 0;
    if (!desc || !registry) return NULL;
    uint32_t handle = scene_read32(desc + 0x70);
    uint16_t base;
    memcpy(&base, (const void *)(uintptr_t)(registry + 0xa62), sizeof base);
    uint32_t i = handle - base;
    if (handle == UINT32_MAX || i >= scene_read32(registry + 0x9f0)) return NULL;
    uint64_t entries = scene_read64(registry + 0x9f8);
    if (!entries) return NULL;
    memcpy(view, (const void *)(uintptr_t)(desc + 0xc8), 16);
    return (const void *)(uintptr_t)scene_read64(entries + (uint64_t)i * 0x28 + 0x20);
}

static void scene_projection_complete(Cpu *c) {
    uint64_t current = c->x[23];
    if (current && scene_projection_enabled()) {
        uint32_t view = *(const uint8_t *)(uintptr_t)(current + 8);
        uint64_t records = scene_read64(current + 0x18);
        plat_mutex_lock(&projection_lock);
        SceneProjection sample = pending;
        pending = (SceneProjection){0};
        if (current != owner || view != index || records != records_generation) {
            owner = current, index = view, records_generation = records;
            generation++;
            sample.token = 0;
        }
        if (sample.token) completed = projection_sequence(sample.token);
        else projection_invalidate();
        plat_mutex_unlock(&projection_lock);
        int recorded = 0;
        if (sample.token && c->x[21]) {
            uint64_t record = records + (uint64_t)view * 0x480;
            if (records && view < scene_read32(current + 0x10) && sample.camera == scene_camera(record)) {
                uint64_t bundle = scene_read64(current + 0x520);
                uint64_t registry = scene_registry();
                uint8_t cv[16], dv[16];
                const void *color = scene_attachment(bundle, 0x20, registry, cv);
                const void *depth = scene_attachment(bundle, 0x60, registry, dv);
                uint32_t extent[4] = {0};
                if (color && depth && scene_projection_extent(color, cv, &extent[0], &extent[1]) &&
                    scene_projection_extent(depth, dv, &extent[2], &extent[3])) {
                    if ((sample.token & SCENE_PROJECTION_SHIFTED) &&
                        memcmp(extent, sample.native_extent, sizeof extent)) sample.width = sample.height = 0;
                    memcpy(sample.native_extent, extent, sizeof extent);
                    recorded = scene_projection_record((void *)(uintptr_t)scene_read64(c->x[21] + 0xd8),
                                                       &sample, color, depth, cv, dv);
                }
            }
        }
        if (sample.token && !recorded) {
            plat_mutex_lock(&projection_lock);
            projection_invalidate();
            plat_mutex_unlock(&projection_lock);
        }
    }
    c->x[1] = c->x[23];
    c->pc = MOD_MAIN_BASE + 0x253120;
    rt_dispatch(c);
}

int scene_projection_binding(uint64_t address, p_scene_projection *out) {
    plat_mutex_lock(&projection_lock);
    SceneProjection sample = pending;
    plat_mutex_unlock(&projection_lock);
    if (!sample.token || sample.camera != address) return 0;
    uint64_t bundle = scene_read64(sample.owner + 0x520), registry = scene_registry();
    uint8_t cv[16] = {0}, dv[16] = {0};
    const void *color = scene_attachment(bundle, 0x20, registry, cv);
    const void *depth = scene_attachment(bundle, 0x60, registry, dv);
    uint32_t extent[4];
    if (scene_projection_extent(color, cv, &extent[0], &extent[1]) &&
        scene_projection_extent(depth, dv, &extent[2], &extent[3])) {
        if ((sample.token & SCENE_PROJECTION_SHIFTED) && memcmp(extent, sample.native_extent, sizeof extent))
            sample.width = sample.height = 0;
        memcpy(sample.native_extent, extent, sizeof extent);
        scene_projection_describe(out, &sample, color, depth, cv, dv);
    } else *out = (p_scene_projection){.sample = sample};
    return 1;
}

void scene_projection_feedback(const SceneProjection *s, uint32_t w, uint32_t h, uint32_t count) {
    plat_mutex_lock(&projection_lock);
    if (!s) projection_invalidate();
    else {
        uint64_t seq = projection_sequence(s->token);
        const CfgRenderValues *settings = cfg_render_values();
        if (s->generation == generation && s->settings == (settings ? settings->projection_revision : 0) && seq > invalidated &&
            seq > projection_sequence(readiness.sample.token) && seq <= serial && serial - seq < SCENE_PROJECTION_MAX_LAG &&
            w && h && count && s->native_extent[0] && s->native_extent[1] && s->native_extent[2] && s->native_extent[3]) {
            readiness = (typeof(readiness)){.sample = *s, .width = w, .height = h, .phases = count};
            static unsigned accepted;
            if (accepted++ < 8) rt_log("dlaa: CPU projection lease token=%#llx issued=%#llx lag=%llu generation=%llu",
                                       (unsigned long long)seq, (unsigned long long)serial,
                                       (unsigned long long)(serial - seq), (unsigned long long)generation);
        } else {
            static unsigned ignored;
            if (ignored++ < 8) rt_log("dlaa: CPU projection ignored feedback token=%#llx completed=%#llx invalidated=%#llx generation=%llu/%llu",
                                      (unsigned long long)seq, (unsigned long long)completed, (unsigned long long)invalidated,
                                      (unsigned long long)s->generation, (unsigned long long)generation);
        }
    }
    plat_mutex_unlock(&projection_lock);
}

static _Thread_local unsigned interface_draw_depth;

int scene_interface_draw_active(void) { return interface_draw_depth != 0; }

static void scene_interface_draw(Cpu *cpu) {
    ++interface_draw_depth;
    cpu->pc = MOD_MAIN_BASE + 0x15266c0;
    rt_dispatch(cpu);
    --interface_draw_depth;
}

static _Thread_local int bloom_scope;
static _Thread_local unsigned bloom_stage;

static unsigned bloom_draw_stage(unsigned role, unsigned level) {
    return role | (level << VK_BLOOM_LEVEL_SHIFT);
}

unsigned scene_bloom_draw_role(uint64_t return_address) {
    if (bloom_scope && return_address == MOD_MAIN_BASE + 0x24c254) return bloom_draw_stage(VK_BLOOM_EXTRACT, 2);
    if (bloom_stage) return bloom_stage;
    return 0;
}

unsigned scene_output_draw_role(uint64_t return_address) {
    if (return_address == MOD_MAIN_BASE + 0x270680) return DRAW_ROLE_SCENE_ENCODE;
    // Both gameplay and pause call the scene mapper here, even without pre-bloom reconstruction.
    if (prebloom_set >= 0 && return_address == MOD_MAIN_BASE + 0x28cac4) return DRAW_ROLE_SCENE_TONEMAP;
    if (prebloom_set >= 0 && return_address == MOD_MAIN_BASE + 0xa06164) return DRAW_ROLE_SCENE_COMPOSITE;
    return 0;
}

static void scene_bloom_schedule(Cpu *cpu) {
    const CfgRenderState *settings = cfg_render_frame_acquire();
    const CfgRenderState *previous = cfg_render_use(settings);
    int previous_bloom_scope = bloom_scope;
    bloom_scope = 0;
    if (vk_dlaa_prebloom_enabled() && scene_projection_enabled() && cfg_bool("REKARINA_DLAA") &&
        vk_dlaa_available() && !vk_dlaa_fsr1()) {
        plat_mutex_lock(&projection_lock);
        bloom_scope = owner && cpu->x[2] == owner && completed > invalidated &&
                      *(const uint8_t *)(uintptr_t)(owner + 8) == index &&
                      scene_read64(owner + 0x18) == records_generation;
        plat_mutex_unlock(&projection_lock);
    }
    cpu->pc = MOD_MAIN_BASE + 0x25e6c0;
    rt_dispatch(cpu);
    bloom_scope = previous_bloom_scope;
    cfg_render_use(previous);
    cfg_render_release(settings);
}

static void scene_bloom_aux(Cpu *cpu) {
    unsigned previous_stage = bloom_stage;
    // The guest walks the three auxiliary destinations in descending pyramid order.
    uint64_t iteration = 0 - cpu->x[20];
    bloom_stage = bloom_scope && iteration <= 2 ? bloom_draw_stage(VK_BLOOM_AUXILIARY, 5 - (unsigned)iteration) : 0;
    cpu->pc = MOD_MAIN_BASE + 0x278d10;
    rt_dispatch(cpu);
    bloom_stage = previous_stage;
}

static void scene_bloom_pyramid(Cpu *cpu) {
    unsigned previous_stage = bloom_stage;
    // This helper samples descriptor x3 and writes descriptor x3 + 1, after quarter-size extraction.
    bloom_stage = bloom_scope && cpu->x[3] < 4 ? bloom_draw_stage(VK_BLOOM_PYRAMID, 3 + (unsigned)cpu->x[3]) : 0;
    cpu->pc = MOD_MAIN_BASE + 0x247200;
    rt_dispatch(cpu);
    bloom_stage = previous_stage;
}

__attribute__((constructor)) static void scene_projection_register(void) {
    if (rt_patch_set("interface_draw_scope") >= 0)
        rt_register_host(RT_PATCH_PAGE + 0xdc0, scene_interface_draw);
    prebloom_set = rt_patch_set("prebloom_sr");
    if (prebloom_set >= 0) {
        rt_register_host(RT_PATCH_PAGE + 0xda0, scene_bloom_schedule);
        rt_register_host(RT_PATCH_PAGE + 0xdb0, scene_bloom_aux);
        rt_register_host(RT_PATCH_PAGE + 0xde0, scene_bloom_pyramid);
    }
    scene_projection_set = rt_patch_set("scene_projection");
    if (scene_projection_set < 0) return;
    rt_register_host(RT_PATCH_PAGE + 0xd40, scene_projection_build);
    rt_register_host(RT_PATCH_PAGE + 0xd50, scene_projection_complete);
}
#else
int scene_interface_draw_active(void) { return 0; }
unsigned scene_bloom_draw_role(uint64_t return_address) { (void)return_address; return 0; }
unsigned scene_output_draw_role(uint64_t return_address) { (void)return_address; return 0; }
int scene_projection_binding(uint64_t address, p_scene_projection *out) { (void)address; (void)out; return 0; }
void scene_projection_feedback(const SceneProjection *s, uint32_t w, uint32_t h, uint32_t count) {
    (void)s; (void)w; (void)h; (void)count;
}
#endif
