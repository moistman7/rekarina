// Motion vector generation mode controls.
// Adapted from NX2navi (MIT, Copyright (c) 2026 Carlos Lopez Jr.).
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    MOTION_VECTORS_OFF = 0,
    MOTION_VECTORS_CAMERA = 1,
    MOTION_VECTORS_MODELS = 2,
    MOTION_VECTORS_HYBRID = MOTION_VECTORS_CAMERA | MOTION_VECTORS_MODELS,
} MotionVectorsMode;

static inline int motion_vectors_camera(MotionVectorsMode mode) { return (mode & MOTION_VECTORS_CAMERA) != 0; }
static inline int motion_vectors_models(MotionVectorsMode mode) { return (mode & MOTION_VECTORS_MODELS) != 0; }

#ifdef __cplusplus
}
#endif
