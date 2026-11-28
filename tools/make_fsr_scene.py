"""Writes runtime/fsr_scene_spv.h: the SPIR-V of the two passes around FSR 1 when it scales the 3D scene (vk_fsr.c
vk_fsr_scene), assembled here like tools/make_dlaa_mask.py (no GLSL compiler on the machine this was written on).

EASU (runtime/shaders/fsr_easu.comp) filters perceptual values. TotK's scene light can exceed 1; its encode pass maps
it into 0..1 with the reversible
tonemap AMD's FidelityFX documentation gives for its upscalers, c / (1 + max(r, g, b)), and the decode pass undoes it
on the scaled picture in place, c / (1 - max(r, g, b)). Both keep the hue (one factor for the three channels).
OoT instead starts from the tone-mapped picture: signed square root encodes it, and v * abs(v) decodes it.
For PsychoV35 HDR the working primaries are BT.2020, with no upper clamp, and the output is signed linear BT.709.

As GLSL:

    #version 450
    layout(local_size_x = 8, local_size_y = 8) in;
    layout(push_constant) uniform P { vec4 p; };   // width, height (the pass's image), unused, unused

    // fsr_scene_encode
    layout(binding = 0) uniform sampler2D src;                    // the light target
    layout(binding = 1, rgba16f) uniform writeonly image2D dst;   // the scene size
    void main() {
        if (all(lessThan(vec2(gl_GlobalInvocationID.xy), p.xy))) {
            ivec2 c = ivec2(gl_GlobalInvocationID.xy);
            vec3 v = max(texelFetch(src, c, 0).rgb, vec3(0.0));
            imageStore(dst, c, vec4(v / (1.0 + max(max(v.r, v.g), v.b)), 1.0));
        }
    }

    // fsr_scene_decode
    layout(binding = 1, rgba16f) uniform image2D img;             // the output size, in place
    void main() {
        if (all(lessThan(vec2(gl_GlobalInvocationID.xy), p.xy))) {
            ivec2 c = ivec2(gl_GlobalInvocationID.xy);
            vec3 v = max(imageLoad(img, c).rgb, vec3(0.0));
            imageStore(img, c, vec4(v / max(1.0 - max(max(v.r, v.g), v.b), 1.0 / 1024.0), 1.0));
        }
    }

    python tools/make_fsr_scene.py
"""
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'runtime' / 'fsr_scene_spv.h'

SECTIONS = ('cap', 'ext', 'mem', 'entry', 'exec', 'dec', 'types', 'func')

OpCapability, OpExtInstImport, OpExtInst, OpMemoryModel, OpEntryPoint, OpExecutionMode = 17, 11, 12, 14, 15, 16
OpDecorate, OpMemberDecorate = 71, 72
OpTypeVoid, OpTypeBool, OpTypeInt, OpTypeFloat, OpTypeVector, OpTypeImage, OpTypeSampledImage = 19, 20, 21, 22, 23, 25, 27
OpTypeStruct, OpTypePointer, OpTypeFunction = 30, 32, 33
OpConstant, OpConstantComposite, OpVariable, OpFunction, OpFunctionEnd, OpLabel, OpReturn = 43, 44, 59, 54, 56, 248, 253
OpLoad, OpAccessChain, OpVectorShuffle, OpCompositeExtract, OpCompositeConstruct = 61, 65, 79, 81, 80
OpImage, OpImageFetch, OpImageRead, OpImageWrite, OpConvertUToF, OpBitcast = 100, 95, 98, 99, 112, 124
OpFAdd, OpFSub, OpFMul, OpVectorTimesScalar, OpFDiv, OpFOrdLessThan, OpLogicalAnd = 129, 131, 133, 142, 136, 184, 167
OpDot, OpSelect, OpFOrdGreaterThan = 148, 169, 186
OpSelectionMerge, OpBranchConditional, OpBranch = 247, 250, 249
FAbs, FSign, Sqrt, FMax = 4, 6, 31, 40   # GLSL.std.450
UniformConstant, Input, PushConstant = 0, 1, 9
Binding, DescriptorSet, Block, Offset, BuiltIn, NonReadable = 33, 34, 2, 35, 11, 25
GlobalInvocationId = 28
Rgba16f = 2   # ImageFormat


class Asm:
    def __init__(self):
        self.next = 1
        self.sections = {k: [] for k in SECTIONS}

    def id(self):
        self.next += 1
        return self.next - 1

    def op(self, sec, code, *words):
        self.sections[sec].append([(len(words) + 1) << 16 | code, *words])

    def words(self):
        out = [0x07230203, 0x00010000, 0, self.next, 0]
        for sec in SECTIONS:
            for ins in self.sections[sec]:
                out.extend(ins)
        return out


def string(s):
    b = s.encode() + b'\0'
    b += b'\0' * (-len(b) % 4)
    return list(struct.unpack('<%dI' % (len(b) // 4), b))


def f32(x):
    return struct.unpack('<I', struct.pack('<f', x))[0]


def build(decode, perceptual=False):
    """The encode pass (decode False) or the decode pass, as SPIR-V words."""
    a = Asm()
    a.op('cap', OpCapability, 1)   # Shader
    glsl = a.id()
    a.op('ext', OpExtInstImport, glsl, *string('GLSL.std.450'))
    a.op('mem', OpMemoryModel, 0, 1)   # Logical GLSL450
    main = a.id()
    t_void, t_fn, t_bool, t_f, t_i, t_u = a.id(), a.id(), a.id(), a.id(), a.id(), a.id()
    t_v3, t_v4, t_v2i, t_v2u, t_v3u = a.id(), a.id(), a.id(), a.id(), a.id()
    t_img, t_simg, t_wimg = a.id(), a.id(), a.id()
    t_psimg, t_pwimg, t_pgid, t_pc, t_ppc, t_ppv4 = a.id(), a.id(), a.id(), a.id(), a.id(), a.id()
    v_src, v_img, v_gid, v_p = a.id(), a.id(), a.id(), a.id()
    c_i0, c_zero, c_one, c_floor, c_zero3 = a.id(), a.id(), a.id(), a.id(), a.id()

    a.op('entry', OpEntryPoint, 5, main, *string('main'), v_gid)   # GLCompute
    a.op('exec', OpExecutionMode, main, 17, 8, 8, 1)   # LocalSize
    a.op('dec', OpDecorate, v_gid, BuiltIn, GlobalInvocationId)
    if not decode:
        a.op('dec', OpDecorate, v_src, DescriptorSet, 0)
        a.op('dec', OpDecorate, v_src, Binding, 0)
    a.op('dec', OpDecorate, v_img, DescriptorSet, 0)
    a.op('dec', OpDecorate, v_img, Binding, 1)
    if not decode:
        a.op('dec', OpDecorate, v_img, NonReadable)
    a.op('dec', OpDecorate, t_pc, Block)
    a.op('dec', OpMemberDecorate, t_pc, 0, Offset, 0)

    T = 'types'
    a.op(T, OpTypeVoid, t_void)
    a.op(T, OpTypeFunction, t_fn, t_void)
    a.op(T, OpTypeBool, t_bool)
    a.op(T, OpTypeFloat, t_f, 32)
    a.op(T, OpTypeInt, t_i, 32, 1)
    a.op(T, OpTypeInt, t_u, 32, 0)
    a.op(T, OpTypeVector, t_v3, t_f, 3)
    a.op(T, OpTypeVector, t_v4, t_f, 4)
    a.op(T, OpTypeVector, t_v2i, t_i, 2)
    a.op(T, OpTypeVector, t_v2u, t_u, 2)
    a.op(T, OpTypeVector, t_v3u, t_u, 3)
    a.op(T, OpTypeImage, t_img, t_f, 1, 0, 0, 0, 1, 0)           # 2D, sampled, format unknown
    a.op(T, OpTypeSampledImage, t_simg, t_img)
    a.op(T, OpTypeImage, t_wimg, t_f, 1, 0, 0, 0, 2, Rgba16f)   # 2D, storage, rgba16f
    a.op(T, OpTypePointer, t_psimg, UniformConstant, t_simg)
    a.op(T, OpTypePointer, t_pwimg, UniformConstant, t_wimg)
    a.op(T, OpTypePointer, t_pgid, Input, t_v3u)
    a.op(T, OpTypeStruct, t_pc, t_v4)
    a.op(T, OpTypePointer, t_ppc, PushConstant, t_pc)
    a.op(T, OpTypePointer, t_ppv4, PushConstant, t_v4)
    a.op(T, OpConstant, t_i, c_i0, 0)
    a.op(T, OpConstant, t_f, c_zero, f32(0.0))
    a.op(T, OpConstant, t_f, c_one, f32(1.0))
    a.op(T, OpConstant, t_f, c_floor, f32(1.0 / 1024.0))
    a.op(T, OpConstantComposite, t_v3, c_zero3, c_zero, c_zero, c_zero)
    if perceptual:
        t_b3 = a.id()
        a.op(T, OpTypeVector, t_b3, t_bool, 3)
        matrix = ((1.660491002, -0.587641139, -0.072849863),
                  (-0.124550475, 1.132899897, -0.008349422),
                  (-0.018150763, -0.100578898, 1.118729661)) if decode else (
                      (0.627403896, 0.329283038, 0.043313066),
                      (0.069097289, 0.919540396, 0.011362315),
                      (0.016391439, 0.088013308, 0.895595253))
        rows = []
        for row in matrix:
            channels = [a.id() for _ in row]
            for channel, value in zip(channels, row):
                a.op(T, OpConstant, t_f, channel, f32(value))
            rows.append(a.id())
            a.op(T, OpConstantComposite, t_v3, rows[-1], *channels)
    if not decode:
        a.op(T, OpVariable, t_psimg, v_src, UniformConstant)
    a.op(T, OpVariable, t_pwimg, v_img, UniformConstant)
    a.op(T, OpVariable, t_pgid, v_gid, Input)
    a.op(T, OpVariable, t_ppc, v_p, PushConstant)

    F = 'func'

    def ins(code, rtype, *args):
        r = a.id()
        a.op(F, code, rtype, r, *args)
        return r

    def ext(rtype, inst, *args):
        return ins(OpExtInst, rtype, glsl, inst, *args)

    def maxc(v):
        x, y, z = (ins(OpCompositeExtract, t_f, v, k) for k in range(3))
        return ext(t_f, FMax, ext(t_f, FMax, x, y), z)

    l_entry, l_body, l_merge = a.id(), a.id(), a.id()
    a.op(F, OpFunction, t_void, main, 0, t_fn)
    a.op(F, OpLabel, l_entry)
    p = ins(OpLoad, t_v4, ins(OpAccessChain, t_ppv4, v_p, c_i0))
    gid = ins(OpLoad, t_v3u, v_gid)
    gu = ins(OpVectorShuffle, t_v2u, gid, gid, 0, 1)
    gx = ins(OpConvertUToF, t_f, ins(OpCompositeExtract, t_u, gid, 0))
    gy = ins(OpConvertUToF, t_f, ins(OpCompositeExtract, t_u, gid, 1))
    px = ins(OpCompositeExtract, t_f, p, 0)
    py = ins(OpCompositeExtract, t_f, p, 1)
    inside = ins(OpLogicalAnd, t_bool, ins(OpFOrdLessThan, t_bool, gx, px), ins(OpFOrdLessThan, t_bool, gy, py))
    a.op(F, OpSelectionMerge, l_merge, 0)
    a.op(F, OpBranchConditional, inside, l_body, l_merge)
    a.op(F, OpLabel, l_body)
    c = ins(OpBitcast, t_v2i, gu)
    if decode:
        texel = ins(OpImageRead, t_v4, ins(OpLoad, t_wimg, v_img), c)
    else:
        img = ins(OpImage, t_img, ins(OpLoad, t_simg, v_src))
        texel = ins(OpImageFetch, t_v4, img, c, 2, c_i0)   # Lod 0
    v = ins(OpVectorShuffle, t_v3, texel, texel, 0, 1, 2)
    if perceptual:
        mode = ins(OpCompositeExtract, t_f, p, 2)
        hdr = ins(OpFOrdGreaterThan, t_bool, mode, c_zero)
        hdr3 = ins(OpCompositeConstruct, t_b3, hdr, hdr, hdr)

        def convert(v):
            return ins(OpCompositeConstruct, t_v3, *(ins(OpDot, t_f, v, row) for row in rows))

        if not decode:
            v = ins(OpSelect, t_v3, hdr3, ext(t_v3, FMax, convert(v), c_zero3), v)
        mag = ext(t_v3, FAbs, v)
        s = ins(OpFMul, t_v3, v, mag) if decode else ins(OpFMul, t_v3, ext(t_v3, FSign, v), ext(t_v3, Sqrt, mag))
        if decode:
            s = ins(OpSelect, t_v3, hdr3, convert(s), s)
    else:
        v = ext(t_v3, FMax, v, c_zero3)
        m = maxc(v)
        if decode:
            k = ins(OpFDiv, t_f, c_one, ext(t_f, FMax, ins(OpFSub, t_f, c_one, m), c_floor))
        else:
            k = ins(OpFDiv, t_f, c_one, ins(OpFAdd, t_f, c_one, m))
        s = ins(OpVectorTimesScalar, t_v3, v, k)
    out = ins(OpCompositeConstruct, t_v4, ins(OpCompositeExtract, t_f, s, 0), ins(OpCompositeExtract, t_f, s, 1),
              ins(OpCompositeExtract, t_f, s, 2), c_one)
    a.op(F, OpImageWrite, ins(OpLoad, t_wimg, v_img), c, out)
    a.op(F, OpBranch, l_merge)
    a.op(F, OpLabel, l_merge)
    a.op(F, OpReturn)
    a.op(F, OpFunctionEnd)
    return a.words()


def array(name, w):
    lines = [', '.join(f'0x{x:08x}' for x in w[i:i + 8]) for i in range(0, len(w), 8)]
    return f'static const uint32_t {name}[] = {{\n    ' + ',\n    '.join(lines) + '\n};\n'


enc, dec = build(False), build(True)
root, square = build(False, True), build(True, True)
OUT.write_text(
    '// Generated by tools/make_fsr_scene.py: the encode and decode passes around FSR 1 on the scene (vk_fsr.c\n'
    '// vk_fsr_scene). Do not edit.\n#pragma once\n#include <stdint.h>\n' +
    '#ifdef GAME_OOT\n' + array('fsr_scene_encode_spv', root) + array('fsr_scene_decode_spv', square) +
    '#else\n' + array('fsr_scene_encode_spv', enc) + array('fsr_scene_decode_spv', dec) + '#endif\n', newline='\n')
print(f'{OUT}: {len(root)} + {len(square)} words (OoT), {len(enc)} + {len(dec)} words (TotK)')
