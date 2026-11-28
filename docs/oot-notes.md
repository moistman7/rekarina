# OoT (Switch 2) on this runtime: notes

Title 0400486024DEC000, update 1.1.1 (v131072), SDK 22.2.8, NVN 22.2.8 (NVN2), modules main, subsdk0
(NintendoSDK_movie), sdk. Build: `tools/build_game.py --version oot` (defines GAME_NS2, GAME_OOT). Offsets below are
of the 1.1.1 modules.

## Boot (works to the first NVN call, 2026-11-05)

- `main` stores its segments as ZBIC (NSO flag bit 7): `recomp/nso.py` decodes it with the zbic package;
  `tools/nso_plain.py` writes the uncompressed ExeFS the C loader reads.
- Lifter additions for Switch 2 code: RETAB / BRA* / BLRA* and PAC*/AUT*/XPAC* as plain branches and no-ops (the host
  never signs pointers), PACGA as a deterministic mix (sdk's setjmp MAC), LDAPR{B,H}, UDOT/SDOT, FP16 FABS/FNEG.
- SDK 22 reads the thread ID at TLR+0x110 (kernel.c thr_link); without it the first heap lock aborts in
  pthread_mutex_lock.
- DT_INIT starts with PACIBSP: the constructor flag's ADRP is at +20 (loader.c decode_ctor_flag).
- RomFS: read straight from the player's .nup (runtime/romfs_nup.c), 186,587 files.
- `nv::InitializeGraphics(NULL, 0)` is valid on Switch 2 (ov_gfx.c).
- Host additions: ns2_host.c (vi HDR/VRR, audio devices, Joy-Con 2 mouse, oe graphics scale, account NSA, network
  clock, fs context), ns2_net.c (offline network; real eventfds for gRPC; socket C API).

## Frame rate

- NVN procs are looked up by name in main+0x129e9xx..; the pointers live in bss: nvnWindowSetPresentInterval at
  main+0x4d47210 (GOT main+0x4c8dec0), nvnQueuePresentTexture at main+0x4d47138 (GOT main+0x4c8dde8).
- main+0xf49970 `SetPresentInterval(window wrapper, u8 interval)`: caches it at +0xa8, calls
  nvnWindowSetPresentInterval(+0x70). main+0xf49950 writes 3 or 2 to +0xa9/+0xaa by a flag.
- main+0x4dc0f0 `SetFrameRate(sys, ?, int fps)`: with sys+0x231 == 1 (VRR supported?), fps >= 40 and sys+0x232 bit 0,
  it takes a VRR path (interval 1, two calls through PLT 0x3c42b90 / 0x3c42b50, probably vi extended swap interval);
  otherwise interval = 60 / fps (rounded) through main+0xf49970. The target fps is kept at sys+0xfc.
  Callers: main+0x4dbdf8, main+0x346c674 (tail), main+0x3b4f190.
- main+0x346c5c0 `SetPresentInterval(sys, interval)`: sys+0xfc = 60 / interval.
- Main loop: the framework's frame procedure (main+0x3b4ea90) queries the refresh rate and VRR support, runs the
  frame controller (vtable +0x128, main+0x4dbd60: the VRR controller main+0x4dbda0 when VRR is supported and on, else
  the adaptive interval controller main+0x3b4ef60, like TotK's) and stores 60 / fps at main+0x4d41fc8 (only the input
  repeat timer main+0x348376c reads it). framework+0x100 is the measured frame time in ticks, +0x108 the last frame's
  tick, +0x148 the work time; the controller's frame rate is at framework+0x2c8.
- The game step: the scene update (main+0x6e6be0) computes ratio = 60 / fps (30 / fps with framework+0x24f bit 0)
  and hands it to the VFR manager (main+0x3b4e300: ratio, 1 / ratio, seconds = ratio / 60, fps = 60 / ratio), which
  the VFR code (AINB QueryVFRCounter, ExecuteActorChaseAngleVFR, ...) steps by. QueryGetBaseDeltaTime is the constant
  1/60. Shipping: interval 1, so the step is a whole 1/60 s per frame and faster pacing ran the game fast (at 120 Hz
  the title movie's 1800 pictures went by in 15 s).
- Unlock (recomp/patches-oot-v111.txt, set realtime, REKARINA_REALTIME): ratio = base * the measured step from the patch
  page (frame_rate.c, 5-frame average of display intervals, 1/240..1/15 s). Generate with
  `recomp/gen.py --exefs ../build/oot/exefs --out generated/oot --patches recomp/patches-oot-v111.txt`. The frame limit
  is TotK's fps_limit / REKARINA_FPS_LIMIT, unset = the monitor's rate capped at 240 under GAME_OOT. Checked on the title:
  speed 1.000 (game seconds per wall second) at 60 and at a 240 limit (112..155 fps there), and the title movie no
  longer speeds up. Gameplay not checked yet (the 3D scene is still black).

## Found 2026-11-05 (evening)

- Fibers: SDK 22's nn::async (FiberTaskCoordinator) runs its tasks on fibers that resume on other threads. The host's
  fiber switch read `tcur` with a TLS address the compiler kept from before `plat_fiber_switch`, and hostfn_tracked
  lowered the host depth of the thread the call started on: random crashes in nn::async lists, MultiWait and
  `ret` into stack addresses. Fixed: kernel.c tcur_now() after switches, fiber_main, hostfn_tracked re-reads
  c->thread; the SDK's FiberType state (+0x20) and ThreadType+0x70 are mirrored; SwitchToFiber(nullptr) returns to
  the thread's own context; no asynchronous preemption while on a fiber.
- Graphics scale: oe::GetCurrentGraphicsScale is the factor the system compositor enlarges the picture by; the game
  draws at 1 / 0.75 / 0.5 of the window for tiers 0 / 1 / 2. The host answers 0 (OOT_GRAPHICS_SCALE overrides).
- Clocks: musl's CLOCK_MONOTONIC / CLOCK_REALTIME sources are nn::time functions the game installs (sdk+0xaf0f8,
  sdk+0xaf040); served by ns2_host.c.
- Kernel: WaitForAddress type 3 (WaitIfEqual64); RomFS mounted under several names (content, _npln); RomFS directory
  listing; QueryMemory reports gaps between modules as free (C++ unwinding scans them).
- Xbox prompts: Font2/Scalable/External.bfotf holds the button glyphs at TotK's code points (U+E040 ...);
  tools/make_oot_prompts.py writes romfs-patches-xbox-label / -xbox (font + grown ResourceSizeTable entry). The
  Nintendo extension shared font is generated by tools/make_ext_font.py into shared-fonts-<style> (SDK 22
  RequestLoadNxFont answered locally).
- Language and voice: the text language is nn::oe::GetDesiredLanguage (ov_system.c: lang / REKARINA_LANG, else the
  Windows user locale). The game has text in 17 languages (Mals/<region><lang>) and voice for its story cutscenes
  (about 100 event flows) in JPja, USen, USes, USfr, USpt, EUes, EUfr, EUde, EUit, as .bwav (codec 2, Opus, mono
  48 kHz) through the hardware Opus decoder (ov_media.c). With lang = ru the game plays the USen voice
  (Voice/Resource/Stream/USen_EventFlowMsg-*), checked with REKARINA_TRACE=fs.

## Found 2026-11-05 (late)

- gmem_map (pool table and write logs at cb_slot 255, oot-nvn a7cbcf0) and storage usage on NVN2 colour images
  (fa7f1a0) are in: the title movie and the 3D scenes draw, but the movie network's output (also applied to the whole
  3D frame) is blocky green noise; the translator's HMMA emulation is the suspect (docs/oot-xlate.md).
- Task and mesh shaders translate (tools/xlate patches 0023, 0024); gfx_shader.c accepts their ts / ms / gs
  reflection. NVIDIA builds the larger mesh pipelines in seconds (16 s for 102k SPIR-V words), so vk_pipeline.c builds
  mesh and geometry pipelines on a thread of their own (REKARINA_MESH_ASYNC, default on) and draws skip them meanwhile:
  the worst frame entering gameplay went from 12-16 s to 0.3 s.
- A failed translation is not asked for again in the run (gfx_shader.c fail_keep): before, a failed shader cost
  a translator round trip every frame (1 fps).
- Still open: the network noise, the forest scene after the title card renders black apart from subtitles, gameplay
  at high frame rates.

## Found 2026-11-06 (night)

- setjmp / longjmp: main calls setjmp at three sites (main+0x165ab00, +0x166168c, +0x2365270) and the sdk at a few
  more; the guest longjmp ends in `br x30` to the call's return address, no function start, which stopped the game
  after the prologue (rt_dispatch: main+0x1661690 not a function start). recomp/gen.py now lifts a call to an imported
  setjmp (Module.import_plts) with a host jump point in the caller's frame (plat_jmp_save), registered by
  rt_setjmp_arm with the jmp_buf, return address and sp; runtime/dispatch.c jumps back to it when a dispatch lands
  on that return address with that sp (setjmp_landing).
- Mesh pipelines build on four background threads (vk_pipeline.c MESH_THREADS) below normal priority; first builds
  take 8-31 s each in NVIDIA's compiler, later runs hit the pipeline cache.
- REKARINA_DUMP=t<ticks> dumps the frame after game time reaches that many 60 Hz ticks (vk_dump.c), so one scene can be
  dumped at any frame rate. With script2.txt the prologue movie runs to about tick 16000; tick 17500 is the forest
  cutscene (a villager close up, the hero's house).
- What the 3D frame does (dump at t17500, 245 passes): compute culling and skinning, shadow cascades (1600x1600),
  the main pass (629 draws, depth and an r11g11b10f target), deferred lighting, post (bloom, DOF), then at the end
  dispatches 240-243 (TAAU / upscaler networks, 1920x1080 -> 2560x1440) and the UI pass. The scene is right up to
  pass 239; passes 242-243 put out inf and a handful of values, so the final image is black. The movie's upscaler
  (pass 13 of the title) has the same symptom: a translator bug in Ryujinx's Optimizer.RemoveNode drops HMMAs whose
  second destination is dead (translator patch 0026, in progress).
- The game outputs 2560x1440 docked (oe::GetOperationMode = console): it renders 1920x1080 and upscales with its own
  network, like the console on a TV.

## Found 2026-11-06 (morning)

- 3D renders: translator patch 0027 (integer image formats) with the runtime's NVN2 view formats (vk_exec_draw.c
  reinterpret_format: an R32UI image view of an r11g11b10f or rgba8 texture, an RGB9E5 view for sampling, the image
  MUTABLE_FORMAT; vk_texture.c slot_resolve passes a followed entry's other format in as a view format) and the
  img_format answers (img_format_of_handle). Pass 242 of the 3D frame packs RGB9E5 words through an R32UI view; before,
  the final image was black.
- Damaged dump files: three .bntx.zs textures of the player's dump begin with 12 KiB of zeros (their NUP hashes match
  what the reader returns, so the dump holds them that way): CmnTex_Map_Soil_FairyFountain_Floor_D_Nrm,
  Rock_WaterTemple_Gate_M_01_Nrm and a baked lightmap of the last stage (tools/make_oot_prompts.py STANDINS). The game retried the first one's load without end
  when the camera turned in the hero's house, then crashed reading an empty resource slot (main+0xc938d0, [x19+0x150] with
  x19 = 0). tools/make_oot_prompts.py writes stand-ins (a sibling of the same name length, renamed inside, RESTBL raised
  when larger) into romfs-patches and both prompt overlays; `--scan` lists .zs files without a zstd header. A fresh
  dump would replace them.
- Still open: frames go white or sky-coloured once the camera moves. The bulk geometry (a 451-draw depth prepass) is
  not shaded when moving, only the 136-draw forward pass is (dumps kdump5 still, mdump1 moving).
- The moving-camera white frames, narrowed (2026-11-06): the game draws a depth prepass (LESS, writes, depth at
  0x8000000000), copies it to 0x8001518e00 and shades with EQUAL against the copy. While the camera moves, the lit
  fragments land farther than the prepass depth: REKARINA_DEPTH_EQ=always or ge (vk_pipeline.c, diagnostic) brings the
  geometry back. Per-draw hashes (REKARINA_DRAWDBG, vk_exec_draw.c) show the same c3 view block, the same c4 bytes and the
  same position buffers in both passes; viewport, depth range and polygon offset are the same; both vertex shaders mark
  Position invariant with the same expression. Not the cause: graphics pipeline libraries (REKARINA_GPL=0), the replay
  thread (REKARINA_REPLAY_ASYNC=1), UBO snapshots, mirror syncs. Next: draw the lit pass's own depth into a scratch target
  and compare per pixel; check what each binding reads on the GPU (live memory or snapshot) and the whole position
  stream.
- gmem stores into VRAM mirrors are not written back to guest memory; REKARINA_GMEM_WRITEBACK=1 (vk_memory.c) sends the
  pieces a write log reports back (off: it costs the title movie about 60 %). REKARINA_GMEM_MIRROR=0 keeps gmem in host
  memory (comparisons).
- main+0xc938d0 (an empty resource slot) still crashes now and then after the camera turns at tick 5800 with the
  stand-ins in place: timing dependent, not settled.

## Fixed 2026-11-06 (day)

- White frames: OoT reads GPU results of the frame it just presented (the exposure's 1x1 luminance, the timestamp
  counters its frame controller reads in main+0x346cb20) without an NVN sync; our GPU ran 2-3 frames behind in heavy
  scenes. The present waits for its own frame's GPU work (nvn_api_obj.c, REKARINA_PRESENT_SYNC=c, the OoT default): 0
  white in 326 shots, 43-49 % before. A blocking acquire, waiting for the previous frame or copying small targets back at
  the fence did not suffice.
- Slow motion: realtime on with steps capped at 1/30 s for OoT (frame_rate.c STEP_MAX_DEFAULT, REKARINA_STEP_MAX_FPS): game
  time follows the clock down to 30 fps, slower below, as the console's 30 fps mode. Without the cap 11 of 13 realtime
  runs crashed at main+0xc938d0; with it 0 of 3. The game's own controller still never changes its present interval.
- Report ring 256 -> 16384 (unshaded moving frames), same-submission samples-passed conditions on the GPU
  (VK_EXT_conditional_rendering), real GPU timestamps (vk_exec_xfer.c).
- Floor grass and shadow flicker: gmem writes into VRAM mirrors went back over by stale CPU copies; they are written back
  to guest memory for pools up to 4 MiB (vk_memory.c REKARINA_GMEM_WRITEBACK=2): flickering pixels under the hero 19.5 % ->
  0.02 %, title movie speed unchanged.
- gmem pool table overflow (4095 entries, ~4100 ranges in the house): on overflow the smallest imported pools are left
  out, not the device pool and the newest pools.
- Still open: the grass seen through the house window flickers (blades come and go, a stretched leaf polygon); not
  conditions, snapshots, write-back, table overflow or texture aliasing. Next: per-draw vertex/instance/indirect data in
  a grass frame vs a no-grass frame. Debug knobs to remove before a commit: PRESENT_FINISH, TSLOG, TEXWATCH,
  GMEM_WRITEBACK_FROM, USNAP masks, DRAWDBG, SKIPLOG, probes, DEPTH_EQ, the present-interval log.

## Fixed 2026-11-06 (evening)

- Black shadows (the great tree's trunk, the village floor under the canopy, shadowed rock faces): the lit materials
  read their baked lighting (RBake_* lightmaps: BC5, BC6H and BC7 arrays) through texture handles from a table in a
  storage buffer. The translator indexes an array over the texture pool by the handle's texture id (bits 0-19) and a
  sampler array by its sampler id, clamped to its QueryTextureArrayLengthFromPool answer (32); the replay bound those
  arrays to the draw's own slots, so every shadowed surface read a stand-in. vk_texpool.c now holds every registered
  texture and sampler in one update-after-bind set (gfx.h GFX_POOL_SET), gfx_shader.c patch_pool_arrays moves such
  arrays there and widens them (no translator change, the cache stays), vk_pipeline.c gives those pipelines the set.
  Slots are resolved again when registered again, when their image goes, and 128 in turn per submission.
- Foliage and shadow flicker with a still camera (tree crowns, moss): a samples-passed segment without queries was
  taken as a known 0 and the lit draws conditioned on it were skipped. Rendering every such draw (REKARINA_PRED_ZERO=0)
  stopped the flicker (4492 -> 322 pixels of 640x360 changed per frame, still camera) but cost two thirds of the frame
  rate in the village (41.5 -> 14.6 fps): most of those segments are objects the game culled itself. Now a segment is
  a known 0 again, unless the replay skipped a draw the game made while samples were counted (shader or pipeline
  not ready or failed: vk_exec_oq_draw_lost), which makes the count unknown. Flicker with this rule: not measured yet.
- Crash main+0xc938d0 (a null entry of the resource manager's table, read at +0x150; walking or turning the camera
  in the forest): patch set resguard sends it the function's own invalid-handle path (output null, which the caller
  main+0x560b38 tests).
- Scene resolution: Pack/Bootup's ModelSceneExtensionParam renders the docked 2560x1440 output from a scene of
  ResolutionReduceScale times it, 0.75 down to 0.33 as the game's GPU load measure rises (it includes the replay's
  latency). tools/make_oot_prompts.py --scene-scale fixes every level (0.75 default). 1.0 (a 2560x1440 scene) is far
  sharper but went past the VRAM budget with After Effects open (8604 of 8553 MiB) and ran at 9 fps.
- Frame time at 41 fps in the village: replay 12-20 ms, then the present waits 10-20 ms for its own frame's GPU work
  (REKARINA_PRESENT_SYNC=c, against white frames): CPU and GPU take turns. Overlapping them is the next performance step.
- REKARINA_GPU_PASSTIME=<frame> (vk_crumbs.c): GPU time of every pass and dispatch of two frames, the 40 longest logged.
  Its first run stalled after frame 4200 while the player started another game; not confirmed either way.
- Test runs from a copy of the player's save (scratch profile, script_save.txt / script_up.txt in build/oot) reach the
  village in about two minutes; never run with the player's own profile folder.
- Foliage flicker, the real cause (trees, bushes, grass and ground leaves flickering and vanishing, worst when the
  camera moves): the foliage is drawn by DrawMeshTasksIndirect from instance lists that compute kernels build at the
  start of the frame (fe89d4116c26fddf among them, caller main+0x5646ac). Their counts changed every frame with the
  camera still (REKARINA_MESHCOUNT: 5 to 53 groups for one draw). The kernels append with the NVN2 compiler's
  warp-aggregated atomics, which count the lanes with a VOTEU under the leader lane's guard; the translator turned every
  guarded instruction into a branch, so the vote saw one lane, each warp added 1 and its other lanes wrote over the next
  warps' slots. Translator patch 0028 runs guarded VOTE, VOTEU, SHFL, MATCH and REDUX in every lane and guards only
  their writes. Counts are now steady (10 and 28 groups, the odd frame 11) and a still camera's frames differ in 347
  pixels of 640x360 instead of 5754: falling leaves and the fairy. The translator identity changed, so every shader is
  translated again once (the first run's scripted inputs drift).
- Pool table overflow: the forest has 4,136 to 4,142 memory pools and the global memory pool table had 4,096
  entries, so the smallest pools were left out and loads and stores through them went nowhere (the pause menu's
  "System" tab label and some button glyphs were garbled). Translator patch 0029 and vk_exec_draw.c GMEM_ENTRIES make
  it 8,192 entries; the forest now fits (4,292 entries with guest pools split at 2 MiB) and the glyphs are right.
- Still missing, found 2026-11-07 against the player's reference shots of the forest: tree crowns and bush leaves
  (the trunks and branches are there), and nearly every shadow caster: the three 1600x1600 cascades hold a few leaf
  blobs. Ruled out so far: leaf textures (dumped BC1/BC3 blocks decode to clean leaf masks at every mip level), their
  swizzles, the alpha-test threshold (c5[27].x = 0.5) and UV transform (identity), the vertex formats (RGBA16F
  positions, RG16 / RG16_SNORM / RG16F UVs, mapped right), the mesh stage's storage bindings, GPU conditional
  rendering (REKARINA_GPU_COND=0, REKARINA_PRED_ZERO=0), DRS load (REKARINA_GPU_LOAD_ZERO=1), the occlusion grid the culling
  kernels read (32x18 depths, plausible), mesh pipeline builds. With the alpha-tested leaf shaders' discards turned
  off (REKARINA_NO_DISCARD=<hash>) flat card shapes show where some crowns are, so part of the leaf geometry is drawn and
  fully cut by the alpha test; the far crowns are not drawn at all. The GPU-driven lists are small: the first-stage
  culling kernel (039bdd7e, 169 groups of 128) keeps about 126 cells, the leaf list holds about 535 cards.
  Next: read back the first-stage kernel's inputs and outputs per instance (REKARINA_PEEK) to see which test drops them.
- The washed-out look (2026-11-07, one frame dumped through every stage, REKARINA_DUMP_ONLY): the lit buffer before the
  fog composite (one full-screen draw, fs fd0c339c, right after the forward passes) has a blue sky and saturated
  colours; the composite adds a dense beige haze over everything. Its in-scattering comes from the 128x72x96 froxel
  volumes (written by compute, read back correctly: the volume a dispatch writes and the texture the composite samples
  hold the same values), which are lit through the sun's shadow cascades, and those cascades are nearly empty: no
  crowns and few other casters, so the whole volume is sunlit. Missing casters and the haze are one problem.
- Crowns, checked one stage at a time in one frame: absent from the prepass depth, from the linear depth the
  composite builds (r32f, from the main depth after the prepass) and from the lit colour before the fog; the alpha
  test passes for 76 % of the leaf-card pixels the shader sees (REKARINA_FS_UVDEBUG wrote u, v, alpha into the prepass
  colour: UVs 0.08..0.95, thresholds 0.5); vertex formats, textures (all BC1/BC3/BC7 decoded clean, every mip level),
  render conditions (GPU predicates ignored on alternate frames: no change; CPU-side false conditions skip no draws in
  that frame) and the depth-equal test (relaxed on alternate frames: no change) are ruled out. The leaf cards drawn by
  mesh shaders come from the GPU culling lists: stage 1 (039bdd7e) keeps ~75 cells, stage 2 (fe89d411) turns them
  into ~535 cards; forcing stage 2's per-card test to pass changes nothing, so the cell list is the limit. Next:
  stage 1's inputs per instance (REKARINA_PEEK_ADDR / REKARINA_PEEK_DUMP) and its LOD and frustum tests.
  Later (REKARINA_C3_PRE edits the culling c3 in guest memory before the dispatch binds it): zeroing the frustum planes
  c3[76..99] doubles the lists (fe89 groups 117 -> 321, leaf list 28 -> 59 groups) but adds nothing on screen, so the
  frustum test is right. Scaling the distance thresholds c3[32..33] (35, 30, 135, 40, 10, 45) by 10 puts more leaf
  tufts on the central tree: leaf cards stop at ~35-45 m by design, and the far canopy LOD or impostor that should
  stand in further out is what is missing. Atomics on demoted host-memory pools ruled out (REKARINA_GMEM_DEMOTE=0).
  Stage 2 (fe89d411) fades a cell's cards between 20 and 30 m, or between 90 and 135 m when the cell's type record
  (c4, 64 bytes per type, byte 130 bit 64, type id from c5) has the long-fade flag; 101 of the 287 types have it. Setting
  the bit on every type for one frame (REKARINA_C4_FLAG) changed nothing visible, but the edit may not reach the GPU's copy
  of c4 (uniform snapshots / mirrors): check that first. The far crowns are the open question.
  Distance thresholds x100 (REKARINA_C3_PRE=32:2:100): lists unchanged (leaf list 28 groups) and no crowns, so neither the
  frustum nor the distance tests hold the crowns back; the cell records stage 2 gets (76 typed cells in c5, about half
  with the long-fade flag) look sane. Next: what stage 1 reads per instance (its category arrays through c4 pointers)
  and whether the crown instances are in those arrays at all.
  The category arrays are populated (cat 2 record array 0x2a19ff000 starts with 0x1196 = 4502 entries, 90 % of its first
  64 KiB non-zero), so stage 1 has its input; the drop from ~9000 instances to ~75 cells happens inside stage 1.
  Hypothesis to test next: stage 1 marks an instance whose LOD byte (_523 & 255) is 255 by writing 1 at its record +12
  (through the pool table) and appending a request to the slot-4 list (count at +32, at most 128); the game is meant
  to stream that mesh in. If the request is never served, the instance stays marked and is never drawn: check the
  record bytes of the crown instances across frames and the game's reads of the request list.
  Also ruled out: the game's CPU culling on read-back counts (REKARINA_OQ_FORCE=1000 reports at least 1000 samples for every
  query: no change); the record words at +12 of category 2 read 0 in the assumed layout (unverified).
  No task stage: mesh draws' programs carry no task shader (the game binds task-stage buffers anyway, stage mask 0xdf).
  Stage 1 (039bdd7e / 1a8e9807) read in full (scratchpad cs1_1a8e.glsl): an instance is dropped by count, model id
  range, distance (cat 0/2 250 m, cat 1/3 c3[16].w, cells c3[17..18].w / c3[9]; 800 m with the c6/c5 flag), its state
  byte (+60: 3..252 culled, read 0 for the first 991 records of cat 2 and 3, so not stuck), the main frustum (planes
  76, 77, 80, 81 only), the 32x18 occlusion grid in the camera buffer (+5488; beyond s6[0] an instance needs a tile
  farther than its depth - r) and list-specific gates (list B: c3[32].x/.y, ~30-35 m). Next: dump c3[16..18], c3[9]
  and c1[0], and test the occlusion grid (fill it with 25000 before the dispatch) - the grid is the strongest suspect.
- Fixed 2026-11-07 (morning): the crowns, bushes and shadow casters were lost to the translator's AST optimizer, not to
  any culling test. It propagated each expression into its uses, a `Ballot` included, so the warp-aggregated appends
  of the culling kernels (VOTEU of the appending lanes, FLO for the leader, the leader's POPC + ATOMG.ADD, SHFL of the
  old count) evaluated the POPC's ballot again inside the leader's branch, where only the leader runs: every warp
  reserved one slot and its other lanes wrote over the next warps' entries. Translator patch 0030 keeps ballots, votes
  and shuffles in place (docs/oot-xlate.md, "Warp-aggregated atomics"). Frame 4010 of script_up2: leaf list 28 -> 495
  groups, other mesh lists 2 -> 32, 4 -> 92, 20 -> 121, 2 -> 324; the great tree and the village trees have full crowns,
  the far shadow cascade holds the canopy, and the fog shows light shafts. Ruled out on the way: the 32x18 occlusion
  grid (filled with 1e9 before stage 1: lists unchanged; its values are plausible, sky 25000).
- Lighting chain after 0030 (one frame dumped): the fog composite (pass 231) gives saturated colour with shafts; pass 256
  is ACES (Narkowicz fit) on (c3[0..2] colour matrix x scene + bloom^c3[3].y) x c3[3].z, linear out; upscaler passes
  257-260; pass 266 copies into the RGBA8_SRGB window texture (nvn 0x38), so the chain itself encodes gamma once. The
  final picture is still brighter and hazier than the console's: exposure (c3[3].z, set by the game from its luminance
  read-back) is the next suspect. Static geometry casts no cascade shadows (baked RBake lightmaps): the cascades hold
  foliage and characters only, by design. Tonemap c3[3].z (exposure) is 1: the game applies exposure earlier; the
  fog composite's log-average luminance is 0.19 (median 0.187), so auto-exposure targets middle grey as it should.
  Bloom: bright pass ~15 % of the scene per level, levels summed (upsample weight max(tex.w^c3[8].x, 1)), added as
  bloom^1.5 before ACES; in the sky it reaches ~55 % of the scene. Not shown wrong yet (blend states not dumped).
- Still open (2026-11-07, after 0030):
  - Garbled subtitles (horizontal stripes inside the glyphs, some lines only), FIXED: each line is rasterized on the
    CPU into a linear R8 strip at one storage address (0x19fe71600 in the prologue, reused for every line), drawn into
    an RGBA8 texture, then blurred into the R8 target the UI samples. A line whose strip descriptor (another size) had
    no cache entry yet followed the live entry of the previous strip at that address (vk_texture.c slot_resolve:
    "a finalized slot follows the live texture at its storage", meant for render targets), so a 1300-texel line was
    read 256 texels a row from a 256x45 entry: stripes. A CPU-content (tex_mirrored) slot now follows only an entry
    whose content the GPU rendered. Checked on the wake-up dialogue (frames 19100-20300 of script_wake): every line
    clean, the same lines were striped before. Found with REKARINA_R8LOG (vk_texture.c vk_dbg_r8, debug).
  - Dark teal soft squares over the great tree in the prologue (both translators); not caught in a dump yet (frame
    numbers drift run to run by hundreds of frames).
  - Exposure (player: "too bright; on leaving the house the exposure drops, then rises a little at the end"; the
    console's "the forest" flyover is about a stop darker and cooler than ours, same framing, script_wake):
    - The lit buffer's sun term carries the exposure: fog composite c4[4].w = 55.2 in the hero's house, 18.9 outside (c4[5]
      = sun colour 1.302/0.945/0.654 times it), so the CPU does change it between areas.
    - The "1x1 luminance" is not a luminance: the 1x1 RGBA32F targets 0x25a7fb600/0x25a7fb800 (pool 0xa1) are drawn by a
      point draw whose vertex shader tests the scene depth at a projected point (additive ONE/ONE), a visibility test
      (the sun's, for glare, by its constants); the game copies them (main+0x9cdd3c, a generic texture-copy helper
      called from main+0x25f3a0, which runs only while [obj+0x900] == 1, mode [+0x246c] is 2 or above 3 and a view's
      byte +0xb4 is set) into 1x1 0x25a4d7d80/0x25a4d7e00 (pool 0x24, CPU cached), read back as (1, 1.015, 0, 1). It
      runs for ~100 frames after a load, then stops (REKARINA_LUMLOG, nvn_api_cmd.c, with the guest frame chain).
      No luminance read-back was found in gameplay: exposure looks like per-area presets the CPU blends (c4[4].w).
    - The fog's far shadow map (4096x4096 D32F at 0x25a82c000, plus a D16 one at 0x27cee7000) is drawn at load only
      (3234 draws, depth 0.217..1, dumped by arming the dump at its creation: REKARINA_BIGLOG + REKARINA_DUMP=t999999999
      REKARINA_DUMP_ARM_OFS=1). Beware: the dump's copies of SAMPLED depth textures (REKARINA_DUMP_TEX) come out all zero,
      the cascades as well; only render-target depth (td) dumps are real.
    - Same moment as the console's flyover (script_wake ~23000 vs the 7:22 reference): ours is warmer and brighter, its
      ground mist is lit yellow (the sun's colour) where the console's is grey-blue (sky-lit, in shade). The fog gets
      too much direct sun; still unexplained. NVN 0x49/0x4A (BC4 UNORM/SNORM, the canopy mask the fog injection reads)
      are mapped right. The player's own screenshot of the port at the porch (later save, tree shield) looks like the
      console's wake-up shot, so the gap may be limited to the first morning's mist.
    - Not it: the 256x256 RGBA16F sky LUTs (0x25f595000/0x25f754000, the only targets the fence write-back skips as
      "drawn every frame"; writing them back every frame, REKARINA_WB_FENCE_LIMIT=600000, changed nothing), the 128x128 probe
      cube (pool 0xa1), the 640x360 scene copy (water refraction input), pass-12 compute (zeroes a counter).
      Nor the light probes Lichii found 104-130 MiB past their pointer (its PRs 292/351): here those loads go through
      the global memory table (gmem_map), not a bounded storage binding. Binding every read-only storage buffer to the
      end of its backing left the porch frame within 0.3 % of before and cost more than half the frame rate.
    - Fog: in the reference-angle frame the fog composite more than doubles the top third (0.073 -> 0.175 linear) and
      adds 50 % mid-frame; bloom adds ~12 %. The fog injection (passes ~103/105) also samples a GPU-rendered 4096x4096
      d32f far shadow map (0x25a82c000) that no dumped frame renders: check its content (drawn at load, possibly while
      mesh pipelines were still building).
  - Outdoors right after the house the replay ran 8-20 fps (replay up to 96 ms a frame, script_exit at ~22800).
  - "a geometry or mesh pipeline could not be created" is logged once per run while a mesh pipeline is still building
    in the background (vk_pipeline_stages returns null for a pending build); harmless. Draws skipped while mesh
    pipelines build are the white-grass frames on a cold cache.
  - Black flashes: none in 1,801 consecutive village frames with a still camera (REALTIME=1, 60 fps limit); still-camera
    flicker 322 of 129,600 pixels per step (leaves, the fairy, the hero idle).
- Test scripts (build/oot): script_wake.txt (new game, empty profile: prologue, wake-up, the hero walks out at ~22600,
  the "the forest" flyover at ~23300 = the console reference shot, the friend at ~23900), script_ref.txt (the save's
  porch, camera pitched down = the player's wake-up reference angle), script_rot.txt (the porch, slow turn).
  Frame numbers drift by up to a few hundred frames between runs of the prologue.
- Debug knobs added 2026-11-07: REKARINA_R8LOG (subtitle textures), REKARINA_LUMLOG, REKARINA_BIGLOG, REKARINA_DUMP_ARM_OFS, vk_dump.c MAX_PASSES 1200, vk_dump.c prints each pipeline's blend enables,
  target-0 factors and write mask.
- Debug knobs added in this round (strip before a commit): REKARINA_DRAWLIST (every draw of a frame and the draws a
  condition skipped), REKARINA_DEPTH_EQ_ALT, REKARINA_COND_ALT, REKARINA_PRED_ALT (alternate-frame comparisons), REKARINA_FS_UVDEBUG,
  REKARINA_SPV_NOT (gfx_shader.c), REKARINA_PEEK_DUMP / REKARINA_PEEK_ADDR / REKARINA_PEEK_DIR, REKARINA_DUMP_ONLY (comma list),
  REKARINA_DUMP_RAW4, and the OQLOG report line's predicate offset. Also REKARINA_PEEK, REKARINA_PEEK_CPU, REKARINA_PEEK_C3, REKARINA_C3_SCALE,
  REKARINA_MESHCOUNT_WIDE, REKARINA_MESHLOG=<frame>, REKARINA_SSBOLOG, REKARINA_SKIP_SH, REKARINA_VALOG, REKARINA_NO_DISCARD (gfx_shader.c),
  REKARINA_DUMP_NOTARGETS, REKARINA_DUMP_TEX_LEVELS and the raw BC block .bin files (vk_dump.c).
- Seen once (run fp0, frame ~2950): device lost, a GPU write at address 0 in a 1920x1080 one-draw pass (vs 6c3cff9d,
  fs 736460f8) while the system had 77 of 116 GB committed; the same build passed that point in 20 other runs.
- Still open: whole-screen black flashes now and then (the player); the scene's dynamic resolution drops (GPU time
  measured from the CPU frame start, main+0x346cb20) and 1080p output on a 1440p screen; the cave ground fog that
  scatters when the hero runs (unchecked); 30-40 fps outdoors (CPU-bound replay).
- Performance, where the frame goes (2026-11-07, Ryzen 7 9700X, RTX 5070, from the frame-time summaries of runs with
  the GPU to themselves; a run beside another game is useless, it fell to 4.5 fps):
  - Porch of the save (script_ref, frames 3590-3900): 35.9 fps; a typical frame is replay 15-20 ms, then the present
    waits ~10 ms for that frame's own GPU work (REKARINA_PRESENT_SYNC=c, nvn_api_obj.c nvnQueuePresentTexture), then the
    game's own work, ~2 ms. CPU and GPU take turns: the GPU tail is added to every frame.
  - A forest frame is ~3,000 draws and ~130 dispatches in ~265 passes (dumps eB, d30): 5-6 us of replay per draw.
    The replay already skips repeated descriptor builds (bind_sig), pushes descriptors through templates, memoises
    shaders and pipelines and splits parts every 64 draws so the GPU starts early (vk_exec.c maybe_split); the
    uniform snapshot is ~0.1 ms per submission (6 a frame). Where the 5 us go is not measured yet.
  - A ~120 ms frame every 30 frames on the porch (3591, 3621, 3651, 3771, 3891) with only ~30 ms of it in replay and
    waits: the rest is outside both, not found yet.
  - Prologue and indoors hold 60 fps; outdoors after the house 42 fps mean, 1 % low 13 fps; the flyover ~17 fps.
  - Next: a sampling profile of the render thread (REKARINA_PROFILE, REKARINA_PROFILE_RENDER=1, tools/prof_report.py), the
    slow-frame chains (REKARINA_PROFILE_SLOW) for the 30-frame stall, GPU time per part (REKARINA_TRACE=perf) and per pass
    (REKARINA_GPU_PASSTIME), and an A/B of REKARINA_PRESENT_SYNC=n for the white frames it was added against.

## GPU time, first round of the native-driver plan (2026-11-07, docs/native-driver.md)

- Measured first (REKARINA_GPU_PASSTIME=3750 on script_ref, the porch with the camera pitched down, GPU to itself): 15 fps,
  GPU 57 ms a frame, 49.7 ms of it the render thread waiting on fences; 126 ms in passes and dispatches over two
  frames. A handful of compute kernels took most of it: 763b043399140e9a 6.5 ms, 0e93bca37d1f8420 5.4 ms,
  2af742e8c18226ef 5.3 ms, acc70d7fd4a21a9b 2.2 ms for one 16x16 group, plus a one-draw 480x270 pass (fs
  4f7c7df85013c1b2) at 2.9 ms. VRAM stood at 9.7 of 10.2 GiB of budget; 2,000 released textures (2.9 GiB, all GPU
  content, so never dropped) sat in the cache, and pools that wanted a VRAM mirror stayed in host memory.
- Finding a pass's shader: the id GPU_PASSTIME prints is the first 8 bytes of the input id; shader-cache/in/<id>.in
  holds the input, and the cache key is sha256("TKSK", u32 4, translator id, input id) where the translator id is
  sha256(totk_xlate.id bytes + "TKHO f64=1 sbuf=16 sg=32,32,32 gmem_map float_atomics") (gfx_shader.c
  gfx_shader_set_host). spirv-cross (old scratch me/spvx) runs with the LLVM toolchain's bin on PATH.
- Changes:
  - Translator patch 0031: RED / ATOMG .ADD.F32 as OpAtomicFAddEXT when the device has shaderBufferFloat32AtomicAdd
    (vk_device.c enables VK_EXT_shader_atomic_float, gfx_shader.c sends "float_atomics"), instead of patch 0022's
    compare-and-swap loop that serialised every lane of a reduction. REKARINA_FLOAT_ATOMICS=0: the loop again.
  - Pass after pass with nothing between (vk_exec.c relaxed_pass_barrier): the next pass's fragment stages wait for
    the previous pass's work, not a full barrier; game barriers, copies, clears and dispatches keep their full ones.
    Only ~10 of ~143 passes a frame qualify on the porch. REKARINA_PASS_BARRIER=full: the old rule.
  - Released textures under VRAM pressure (vk_texture.c sweep_pressure): the threshold is 85 % of the budget (was
    90 %, above the 87.5 % line at which mirrors are refused), and a released entry with GPU content goes once it has
    stayed released 600 frames (REKARINA_TEX_DROP_AGE). The released set is mostly 960x540 0x29, 1920x1080 0x3f/0x34
    render targets at addresses never taken again.
  - Diagnostics (REKARINA_TRACE=perf): a released-texture census and the buffer uses that stay on host memory per pool
    (vk_memory.c host_bind_log). On the porch only 13 small "demoted" pools (20-396 KiB) are bound from host memory,
    46 uses a frame.
- Result, same frame and scene (main lit pass 1179 vs 1190 draws): 126 -> 30 ms in passes over two frames, GPU 57 ->
  20.5 ms a frame, 15 -> 37 fps, VRAM 8.6 of 10.2 GiB, no mirror refused. None of the kernels above is in the top
  list any more; the lit pass (2.9 ms) leads. The frame is now bound by the CPU side: 27 ms, of which ~11 ms is the
  present waiting for its own frame's GPU work (REKARINA_PRESENT_SYNC=c): step 1 of the plan.
- Caveat: the first run after a translator change retranslates every shader (1,468 in 15 s at boot, the rest on
  demand), which delays the scripted inputs; compare runs only from the second one on.
- Checked after it: still camera at the porch (script_still, REALTIME=1, frames 3000-3120): 312 of 129,600 pixels
  change per step (322 before: leaves, the fairy, the hero idle), luma steady 120-121 over the 121 frames (no flash). The
  porch frame at 3760 matches the earlier one apart from the usual run-to-run camera drift.
- Flash watch (flight.c, REKARINA_FLIGHT=1; REKARINA_FLIGHT_FLASH=0 off): every present reads the 320x180 thumbnail of the
  frame 4 presents back and compares its mean luma with a running mean; a run of at most 6 frames far darker (below
  35 %) or far brighter is logged as "flight: dark/bright flash at frame N" and dumps the frames around it (one dump per
  10 s; REKARINA_FLIGHT_DIR on C:, D: is short of space). Longer changes (fades, loading, cuts) are not counted. Added
  because the player still sees whole-screen black flashes that a still-camera series never caught.
- Black flashes caught (flash watch, script_rot, two runs): frames 3616; 2742 and 3965 (2106 was the pause menu's
  fade, a false positive). Each is one frame with the whole 3D scene black (luma 5 against 90-120) while the HUD,
  prompts and minimap are drawn; the frame's work looks normal (same passes and draws, nothing skipped). About one
  per 600-1,300 frames, with the camera still or turning; present before this round too (the player's report).
  Leads: (1) texture aliasing in the game's transient render-target memory: the same addresses get new sizes every
  frame (squares 79..97 and 1292..1297 fmt 0x34, 960x540 / 640x360 / 480x270 at one address), so several live
  entries share an address, ~2 new ones a frame start empty ("left uninitialised" ~124 a minute), and slot_resolve's
  "follow the live texture at the storage" may pick the wrong one; (2) the exposure word (c4[4].w) going to 0 for a
  frame. Next: arm a frame dump on the flash (flight trigger -> REKARINA_DUMP of that frame) and see which pass reads black.
  REKARINA_ALIASLOG=<frame> logs alias events (800 lines fill in ~200 frames: start it close to a flash).

## Phase A, first step: texture content follows the memory (2026-11-07)

- slot_resolve used to follow "the first live entry at the storage" when a slot's own entry was released (the game
  finalizes and re-creates its transient render targets every frame, at reused addresses with new sizes). With several
  live entries of different sizes at one address the first was sometimes one this frame had not rendered: the
  composite read it and the whole 3D scene was black for a frame (the flashes above).
- Now every GPU write recorded into an entry (pass targets, clears, copies, storage images) takes a serial
  (vk_texture.c note_writer, Tex.wr_serial), and a read follows the newest writer at the storage
  (newest_writer: a same-but-swizzle entry wins a tie; for CPU-content slots only GPU-owned entries or same-but-swizzle
  ones). A live descriptor the GPU never wrote, not mirrored, also reads the newest writer's render. The alias source
  of a new entry is chosen the same way. A change of newest writer bumps the address generation so cached slot
  resolutions there are redone.
- Checked (script_rot, flash watch): frames 2100-10080 of play, the porch still and turning, no black flash
  (before: one per 600-1,300 frames, 6-13 expected over this stretch); only the pause menu's fade at 2106, a known
  false positive. The player watched the run and saw none either. The porch frame looks as before.

## Phase B groundwork: who reads GPU results, what frames in flight break (2026-11-07)

- REKARINA_RESULT_WATCH=1 (memory.c, a diagnostic): counter-report and event pages are closed when recorded and opened
  when the completion thread writes them; the first access before that is logged with the instruction
  (scratchpad me/rva.py maps the RVA to rekarina.map). Only nvnDeviceGetTimestampInNanoseconds reads them early (plus one
  reset write by f_7000c2fc60 at boot). Its guest callers: f_7003b4eb50 (main+0x3b4eb50, the frame controller: end
  timestamp of the current slot of a two-slot ring minus a stored start, kept as max(floor, elapsed) at +0x140; it
  reads in the same frame, the GPU one submission behind, so it got stale words with REKARINA_PRESENT_SYNC=c too) and
  f_700005a8f0 (called from main+0x627c4 and main+0x642d4, counters at 0x13D6C84F0/8500).
- REKARINA_PRESENT_SYNC=n (no present wait): 45-48 fps on the porch instead of 37, but white frames return (whole 3D
  scene white, HUD drawn; 50-60 in 2,500 frames, flash watch "bright"). Render-condition counts (REKARINA_TRACE=perf
  "render conditions per frame"): ~630 on the GPU, ~26 known zero, 0-79 unknown (rendered), ~31 decided from memory;
  white frames also come when none is unknown, so conditions are not the cause. REKARINA_GMEM_MERGE=1 (word merge of
  CPU-written pages into mirrors) does not remove them either.
  REKARINA_GMEM_DEMOTE=0 (small CPU+GPU pools keep their VRAM mirror) does not remove them either (63 in 1,900 frames).
  The result watch now also covers every recorded GPU write to guest memory (vk_mem_gpu_pending at record,
  vk_gpu_written at completion: copies into buffers and shader storage writes), to find the early reader.
- Found with the full watch: the replay reads the words of render conditions from guest memory at record time
  (counter_word, ~31 conditions a frame), and many are written by GPU copies and shaders not run yet. Tried: run such a
  condition on the GPU (VK_EXT_conditional_rendering on the word where the GPU reads it, VK_MEM_PICK) and render
  whenever a recorded GPU write has not produced the word yet (vk_mem_gpu_write_pending). WRONG: the whole scene stayed
  white, in the default mode and with REKARINA_PRESENT_SYNC=n alike (the flash watch took the lasting white for a scene
  change; only the shots showed it, and the player saw it), and the porch fell to 23-28 fps. It is off
  (REKARINA_MEM_COND=0, the default); the flash watch now reports a lasting white or black screen too. Rendering a
  condition whose word is not written is what turns the frame white (the conditional draws are meant for nothing
  visible), so pending words need their real GPU value, not "render".
- The 1 and n runs with REKARINA_MEM_COND on were white throughout as well, so their "clean" results do not count.
- Checked after turning it off: the default mode is as before (the porch shot at 1500 normal, ~41 fps there, only the
  known dark false positives at the load fade and the pause menu). Next try for frames in flight: keep the GPU
  predicate on the word (that part alone is untested) but never "render" a pending word; instead give the condition
  the word's real value in GPU order. Check every run on the shots, not the flash counts alone.
- Classified (REKARINA_TRACE=perf "memory words"): every condition decided from memory compares a word with a zero word,
  and no recorded GPU write produces that word (x-done ~30 a frame, x-pending 0); the watch's early counter_word reads
  were pages shared with a pending report (page granularity), not those words. Under REKARINA_PRESENT_SYNC=n the
  "unknown (rendered)" conditions (samples-passed reports of the previous frame still pending and not exact, so
  rendered) reach ~101 a frame in one stretch and ~0-4 elsewhere, while white frames, and once a lasting white screen
  from frame 1626, come with every knob off: frames in flight break more than one thing. (rt_knob only gives 0/1;
  REKARINA_MEM_COND=2 never took effect.)
- Priority changed: 100 fps means a 10 ms frame; now the GPU takes ~20 ms and the replay ~15 ms, and frames in flight
  give at most max(CPU, GPU), ~50 fps. Cut the GPU and CPU time of a frame first; finish frames in flight after.
- Render thread profile, porch frames 3300-3700 (REKARINA_PROFILE_RENDER=1; the sampler slows the thread itself, so its
  frame times are not comparable with unprofiled runs): 39 % waiting on fences (the present wait), then
  vk_exec_op_xfer 4.3 % of all samples (~15 % of the exe's running time: counter_word walked the pending-report ring
  linearly for each of ~700 conditions a frame; now an address index, vk_exec_xfer.c pidx_find), bind_resources 3.8 %,
  GetWriteWatch (plat_mem_written) 3 %, the driver 5.4 %, texture slot lookups ~3 % together.
- GPU frame shape (GPU_PASSTIME "shape", two frames on the porch): span 53.7 ms, 22.7 ms between marks (barriers,
  copies, waiting for the next submission), 15 marks over 0.5 ms = 16.2 ms, 42 of 0.1-0.5 ms = 9.5 ms, ~540 under
  0.1 ms = 5.4 ms.
- Unknown conditions under REKARINA_PRESENT_SYNC=n by cause (pending[].why): all "lost draw" (~46 a frame at scene entry,
  pipelines still building), which the default mode renders too (a lost segment counts at least 1). White frames at
  1143-1180 come with no skipped draw and the usual draw counts: the GPU work is the same, its data is not. Working
  hypothesis: the engine double-buffers its per-frame GPU data for a GPU at most one frame behind (as on the console);
  with no present wait the CPU can run 2-3 frames ahead (acquire bounds it by the swap images only) and rewrites a
  buffer the GPU still reads; and the mirror's sync copies read host memory when the GPU runs them, not when recorded.
  Test: REKARINA_PRESENT_SYNC=1 (the GPU at most one present behind) with every other knob off.
- Bisect of the white frames under frame overlap: REKARINA_PRESENT_SYNC=1 (one frame) and c1 (all but the last submission
  awaited) both still give them on the porch (40-46 in ~2,500 frames, 47-48 fps). Cause found in our memory layer: a
  part's prologue copies guest memory into its VRAM mirror when the GPU runs it, not when recorded; with any overlap
  the CPU is already writing the next frame there, and the mirror gets a mix of two frames. The upload is also large:
  nine CPU_CACHED pools of 6-15 MiB (pairs and quads of one size: the engine's frame rings) are copied whole every
  frame, ~90 MiB a frame (REKARINA_TRACE=perf "uploads by pool").
- Copy guards (memory.c mem_copy_guard, vk_sync.c pro_sem, vk_memory.c vk_mem_prologue_submitted; REKARINA_COPY_GUARD=0
  off): each prologue is a submission of its own that signals a second timeline; the guest ranges it copies become
  read-only once it is submitted. A CPU write into one waits for exactly that copy (memory.c cg_fault, from the
  write-fault hook), then every guard whose copy ran opens; reads never wait.
- Player's observation (2026-11-07, guarded frames-in-flight build, REKARINA_PRESENT_SYNC=n): between the flashes the
  picture's tone is right, as on the console (no excess haze and brightness). The long-standing "brighter and hazier
  than the console" (Exposure, above) may come from the present wait: with it the engine reads GPU results fresher
  than the console's one-frame-behind GPU gives them (an adaptation or a temporal effect that expects the lag).
- Copy guards cost too much as designed: pools copied whole every frame mean ~90 MiB protected and opened per frame,
  ~70-100 ms of CPU (8-10 fps). Off by default (REKARINA_COPY_GUARD=1 for tests). Replacement: hot pools (vk_memory.c
  hot_census / hot_demote_pending): a mirrored pool of 1 MiB or more copied 90 % whole for 30 frames in a row is a
  per-frame ring and gives up its mirror; the GPU reads it in guest memory directly, as on the console
  (REKARINA_HOT_DEMOTE=0 keeps the mirrors).
- Tests start fast now: scratchpad me/fast_porch.txt and fast_rot.txt press A every 15 frames from frame 40 to 1000
  (logos, title, Continue, load), then B, ZL and the camera inputs from 1150.
- Hot-pool demotion tried and dropped (REKARINA_HOT_DEMOTE default 0): the nine rings and more went to direct reads
  (uploads ~90 -> ~3 MiB a frame) but the white frames stayed (64 in 1,700 frames, a lasting white once) and the GPU
  took 37.7 ms a frame instead of 20 reading them over the bus. The mirrors matter for GPU speed, and the prologue
  copies are not (alone) the white frames. Next: per-frame condition counts in the flight records (cond_gpu,
  cond_unknown, cond_mem_true, cond_mem_false columns) to tell a white frame from its neighbours.

## The console look and the white frames: what frames in flight change (2026-11-07, evening)

- The look. The player's reference-like frame came from REKARINA_PRESENT_SYNC=n with copy guards. Shots at the same porch
  framing (fast_porch, frame 1420): the default mode (present waits for its own frame) is hazy and too bright with or
  without copy guards; one frame in flight (REKARINA_PRESENT_SYNC=1) plus submit-time mirror copies (below) gives the
  console's darker, contrasty forest with light shafts; overlap without submit-time copies stays hazy.
- Cause of the haze: OoT's auto exposure. The sky LUT pass (fs dcc98596a943cf34, 256x256 RGBA16F 0x25f595000 and
  0x25f754000) takes sun colour x exposure in fs c4[2].w / c4[3] (the factor every later sky, fog and aerial-perspective
  target inherits). The CPU adapts it every frame from GPU measurements within a clamp of 9.765625..78.125 (3 EV):
  in the default mode it settles near 25.5, with one frame in flight it settles at the minimum 9.77 (REKARINA_PROBE_FS,
  "SUNVAL" log lines). The console's GPU runs about a frame behind the CPU; the game's exposure loop expects that
  latency, and waiting for the GPU at every present (the white-frame fix of 2026-11-06) changed its equilibrium.
  Ruled out on the way: GPU timestamps (REKARINA_TS_PENDING_NOW, REKARINA_GPU_TIMESTAMPS=0), render conditions (identical
  counts), the uniform snapshot (REKARINA_UBO_SNAPSHOT=0), the far shadow map (identical in both modes), the fog dispatch's
  inputs (07b7fa3adebce45e: only light order and animation timing differ), the one-shot sun visibility test after a
  load (1x1 RGBA32F copy, read once; delaying its write-back by a frame, REKARINA_READBACK_LAG, changed nothing: dropped).
- Submit-time mirror copies (vk_memory.c "Submit-time snapshots", REKARINA_MIRROR_SNAP): a prologue copies the guest
  bytes into a 512 MiB host ring when it is recorded, six threads, and the GPU copies them from there, so the mirrors
  hold guest memory as of the submit, as the console's GPU reads it. ~113 MiB and ~4.5 ms of CPU a frame on the porch
  (the default mode drops from ~42 to ~37 fps with it, so it only pays with frames in flight).
- White frames with frames in flight (~65 in 40 s, frames ~820-1160 after the save loads and ~30 frames after the
  pause menu closes; none in the default mode). Found with the pass probe (REKARINA_PASS_PROBE=1 with REKARINA_FLIGHT, vk_dump.c
  "DEBUG-PROBE": per render pass and dispatch a 16x9 grid plus a 12x12 box of texels; probes.tsv, means.tsv, box.tsv,
  draws.tsv, sh.tsv, ubo.bin, crop_*.ppm in each flight dump): the first non-finite texels of a white frame are in the
  character skin layer (an R11G11B10F 1920x1080 target at a transient address, ~20 draws of the skin shaders, fs
  b983152a0a705a26 and others, before the forward passes): every skin pixel of the hero (arms, legs, ears) and of a
  distant forest is NaN (crops: magenta), the composite carries it into the scene buffer and the bloom spreads it over
  the whole screen. Identical in white and normal frames: the draws (shaders, counts, conditions), every uniform block
  the GPU binds (CPU vs GPU copy compared), the SH probe slot (c4[14].x x 144 into 0x27ef1d000), the light structs
  (0x25fe1e000/0x25fe36040), the environment cube (all faces and levels), the screen-space float targets under the hero.
  Not the cause: command buffer splits, relaxed pass barriers, word merge of all write-backs (REKARINA_MERGE_MAX_MB=64),
  GPU-written pages kept out of mirror copies, demotion, timestamps, the uniform snapshot. Still unchecked: the
  clustered light lists (0x25ea86000, 9.5 MiB, written by GPU compute each frame) and the vertex inputs. Next: a
  translator-side NaN tracer (each floating-point result of a chosen shader checked, the first non-finite one's
  instruction index written to a debug buffer) to name the operation and its operands.
- NaN source named with a SPIR-V NaN tracer (scratchpad me/nantrace.py: instruments a cached shader; every float
  result checked, the first non-finite one's index written as the colour of pixels whose output is not finite; the
  crop dump decodes it, crop_*_idx.txt): in the main skin fragment shader (b983152a0a705a26, 8 cache files) it is
  instruction 1223, OpFDiv %3552 = 1 / (a*d^2 + b*sqrt... + c): a point light's distance attenuation with its three
  coefficients from the light struct (96 bytes, at index*96 + 32 + 32 of storage slot 1, the double-buffered light
  buffers 0x25fe1e000 / 0x25fe36040, a 32-byte header whose first word is the light count, 24-26 on the porch). Slots
  past the count are zero, so 1/0 = inf, then NaN. A white frame = some cluster list (storage slot 0, 0x25ea86000,
  single region: c3[149].x is 0) names a light index at or past the count of the light buffer that frame binds. The
  other skin shaders (56aaae40b2842a72, d3eaffaa82d7f0ff, 6c7a58ad6c1c3763, c790367add79bfe2) make most of the NaN
  pixels, presumably at the same formula. The list pool is rewritten whole by the CPU once a frame at the frame's
  first submission in both modes (REKARINA_SYNCLOG), so no CPU write lands during the replay; the GPU's light structs equal
  the CPU's (PAGECHK). Not yet known: who builds the lists against which light set, and why only frames in flight
  get a list/count mismatch.
- (Superseded below: the list pool is written by GPU compute, not the CPU; the CPU only appears to rewrite it because
  the completion of that compute's submission stamps the pages as written.)

## The white frames' cause: a command buffer wait for the CPU (2026-11-07, night)

- Frame layout on the porch, per frame: six or seven submissions. The fourth runs the light culling compute
  (cs 7250b92d6acbdd19), which writes the cluster lists into the 9.5 MiB pool 0x25ea86000 (headers {count, first, ...}
  per cluster = z*576 + y*32 + x, entries from word c3[149].x = slices * 2304 on, slices 78-85); the last, ~20 ms of
  recording, draws the skin layer and the forward passes, whose skin shaders read a cluster's header and entries
  with the grid of the scene uniform block (fs c3[148] = (18, 576, 0, 4), c3[149] = (base, 32, slices, 2)).
- That scene block is double-buffered (0x25e94b000 / 0x25e94e500 on the porch) and the game writes it in two steps:
  a 1x1x1 placeholder grid (1, 1, 0, 0) / (0, 1, 1, 0) first, the real grid later, on another thread, while the frame's
  last submission is being recorded (REKARINA_LISTLOG_GRID log). On the console the GPU does not read it early: the
  frame's command buffer holds nvnCommandBufferWaitEvent(event, EQUAL, 1) for one of two events (alternating per
  frame), which the CPU resets to 0 at the frame's start (nvnEventSignal, lr 0x70002cb420) and sets to 1 once the late
  constants are in (lr 0x700346c78c). The replay treated WaitEvent as a no-op, so the uniform snapshot of the
  submission (vk_usnap.c, taken when its recording starts) held the placeholder: every skin pixel then read cluster 0
  and entries from word 0, i.e. header words (first offsets such as 1933, 2972, 3495) as light numbers, past the 27
  light structs: zero coefficients, 1/0, NaN, white. The default mode (present waits for its own frame) hid it by
  timing: the late write landed before the next submit.
- Fix: OP_WAIT_EVENT is recorded (nvn_api_cmd.c) and the replay waits there for the CPU's signal when no GPU signal of
  that word is in flight and the CPU has signalled it before (vk_exec_xfer.c vk_exec_wait_event; 100 ms limit,
  REKARINA_EVENT_WAIT_MS; REKARINA_EVENT_WAIT=0 off), then copies the uniform ranges the rest of the submission binds again
  (vk_usnap_resnap). Waits on the porch: ~1-3 ms a frame. The game's other waits (an array of per-frame words at
  0x13D6C80xx, set by GPU work of another queue) are not waited for.
- Result (fast_porch, 50 s, frames in flight + submit-time mirror copies): 0 white frames (before: ~60), the darker
  console look (mean luma of the porch shot 81 against 129 in mode c). The reference script with the pause menu
  (PLUS at 2100 and 2400): 0 white frames; opening the menu costs two 100 ms waits once (the event is not set within
  the limit there).
- Default now: REKARINA_PRESENT_SYNC=1 for OoT (one frame in flight) with the submit-time mirror copies.
- Ruled out on the way, each with a run: the list pool's write-back and its whole-pool copy back into the mirror at
  the next submission (REKARINA_WB_SKIP_POOLS, REKARINA_MERGE_POOL, REKARINA_SYNCLOG), a grid mismatch between the compute and the
  skin draws (both logged per frame), GPU overlap of frames (every submission ends with a full barrier), loads through
  the global memory table reading live guest memory (REKARINA_GMEM_MIRROR_ALL=1: mirrors for every pool the table reaches,
  4.2 GiB; the same white-frame rate).

## GPU writes the mirrors took back, foliage flicker and the look (2026-11-07, late night)

- Every buffer range a shader, copy or fill wrote was reported at its fence as guest memory written
  (vk_gpu_written -> page stamps), so the next submission copied it from guest memory back into the VRAM mirror:
  113 MiB a frame on the porch (the 9.5 MiB light-list pool, 15.2 / 8.6 / 6.3 MiB pools every frame). With frames in
  flight and submit-time snapshots that copy is older than what later GPU work already left in the mirror. Now such a
  range is stamped "mirror-owned" when a mirror wrote it and copies it back whole (vk_gpu_written_mirror, own_chunk):
  the texture cache still sees the change, the mirror's sync leaves the page out until the CPU writes it again
  (pages the CPU wrote since the last write-watch query stay the CPU's), and flush ranges are cut around such pages
  (REKARINA_FLUSH_OWNED_CUT). Uploads 113 -> 31 MiB a frame.
- Foliage flicker with a frame in flight (the player's report: lighting blinks, grass, water): the far trees' leaves
  vanish for one frame every eighth frame (scratchpad flick.py on 80 still frames 1365-1444 of fast_porch: one frame
  differs from both neighbours over ~6,000-10,000 of 230,000 pixels). None with REKARINA_PRESENT_SYNC=c. Demoting the
  CPU+GPU-written record pools to their import buffers made it worse; REKARINA_GMEM_DEMOTE now defaults to 0. It still comes
  back in some runs (0 or 5 events in 80 frames), so its cause is open; suspect a GPU result the game reads back on an
  eight-frame cycle.
- The darker look of the earlier frames-in-flight runs came from stale GPU data copied back into the mirrors, which
  pushed the auto exposure to its lower clamp: with the copies fixed the porch shot is as bright as in mode c (mean
  luma ~130), and other mixes of the same faults give 83, 97 or 129. The publisher's press-kit shot of the same view
  is darker than ours (forest band mean ~98 against ~142,
  less saturated); the time of day may differ, the exposure loop has to be checked against it.
- The game runs six NVN queues per frame and orders them with a ring of event words (0x13D6C80xx, values = a sequence
  number) that GPU work of one queue sets and another waits for, sometimes for a later submission; the replay keeps
  submission order on one Vulkan queue and does not wait for those.
- Default back to REKARINA_PRESENT_SYNC=c (no flicker, no white frames, same look, the replay limits the frame rate either
  way); frames in flight (=1) stays available.

## Auto exposure: a float sum the CPU resets (2026-11-08)

- Chain: a compute pass adds the frame's luminance into word 0 of a 0x1040-byte storage range with float atomics
  (OpAtomicFAddEXT), a ring of three 8 KiB pools from 0x25fb7b000; the CPU (f_7000337060) reads the word, writes 0
  back and uses 1 + k (sum - 256) / 256 (the sum is normalised to 256 samples, so it does not depend on the
  resolution); the running averages give the EV (f_70003b94b0), the exposure factor reaches the sky LUT pass through
  fs c4[2].w (10000 / 2^EV, clamped to 9.77..78.1).
- With a frame in flight the CPU's reset fell between the write-back merge's snapshot and its write: the GPU's value
  went back over the 0, the reads saw sums of more and more frames (209, 417, 625, ... 59519, then a wrap), the EV sat
  at its clamp (the dark, contrasty look) with jumps back (the flicker). The earlier "console look" with frames in
  flight was this bug.
- Fix: the translator's reflection marks storage buffers a shader adds into with float atomics (gfx_shader.c
  mark_fadd_ssbos, GFX_SSBO_FADD); their ranges go through the write-back merge as additions: a word the CPU wrote
  meanwhile gets the GPU's change (mirror minus snapshot) added (shaders/wb_merge.comp, z's top bit). The SPIR-V header
  was patched by hand (no glslc on this machine; the GLSL is the same change): regenerate it with
  tools/build_shaders.py where glslc is available. Now the reads are ~208-215 every frame in both present modes and
  the exposure factor holds ~23 (mode c gave the same before).
- Ruled out on the way: taking the pools off the mirror (the atomics in guest memory, as on the console) fixes the sum,
  but the rule catches other pools (the foliage culling's integer counters, small float sums in transient pools) and
  the far trees then lose their leaves every eighth frame. A CPU-writes-the-pages rule does not separate them (the CPU
  writes records into the same pages).
- The official press shot of the porch (mean 98, saturation 16.7) stays darker than ours (142-154, ~38) at a correct
  exposure; the time of day there is unknown.
- Frames in flight (REKARINA_PRESENT_SYNC=1) after the fix: 39.1 fps against 35.0 (mode c) on fast_porch; one frame in
  80 (1382) with wrong shadows (the railing and the slope in shadow, the hero's shadow sharp): a shadow map or its
  matrices from another frame. Mode c stays the default until that is found.

## The look against the console, and the GPU frame (2026-11-08, night)

- The look. The exposure chain has a second input: EV = 10 + 0 * [0x32c] - 3 * [0x34c] (f_70003b94b0), where [0x34c]
  averages 1 - clamp((G / 256 - 0.02) / 0.13) and G is the Rec. 601 luma of four float sums (R, G, B, luma) that the
  compute shader 33bb42d5 (cs dfbaf56a96816c43, 4x4x6 groups) adds over the 128x128 environment cube 0x25faf6000 (two
  faces re-rendered per frame from the porch: sunlit planks, dark trees, a dim blue sky). Ours: (37, 22, 7, 24.5) ->
  G / 256 = 0.096 -> EV 8.75 -> sky factor 23. EV 10 (sky factor 9.77, the darker frame the player kept as a
  reference) is the value before the auto exposure starts (~frame 817 after the load: that reference has the area
  title on it). The official press shot of the same view is as hazy as ours and its sky as bright (200 against 198),
  but neutral where ours is warm (red / blue 1.11-1.17 against 1.25-1.35). Checked and fine: the scattering LUT
  (3D 256x32x16 RGBA16F, a game file copied in by memcpy at frames 113 and 547, smooth, no swizzle damage), the sky
  LUT (blue zenith, warm towards the sun), the sky SH ambient (L0 1.94, 2.50, 3.85: blue), the sun visibility test
  (0.125: mostly occluded), the final tonemap pass (no grading LUT). Open: the warm horizon in-scattering of the sky
  LUT's lower half (red / blue 2.5); without a console frame at a known time of day it is not settled.
- GPU frame on the porch (RTX 5070, frames in flight): ~19.6 ms busy of which ~15.4 ms are the game's passes and
  dispatches, ~4 ms transfers and barriers; ~100 W of the card's 250 W at 84 % utilisation: latency-bound (long chains
  of small dependent passes), not throughput-bound. Graphics scale tier 1 / 2 (64 % / 44 % of the pixels) only gives
  18.4 / 17.9 ms. Full barriers per frame: ~425 (pass begins 85-107, UpdateUniformBuffer 50-67, the game's own 117,
  occlusion predicates 24, indirect mesh draws 60); removing all ours breaks the picture (white), and dropping
  UpdateUniformBuffer entirely (a measurement) saves only 0.4 ms: the barriers are not where the time goes.
- Changes kept (all A/B knobs): dispatch after dispatch ordered by the game's barriers only (REKARINA_DISPATCH_BARRIER=full
  restores; -0.6 ms GPU); the game's fragment-only barriers (ORDER_FRAGMENTS, no ORDER_PRIMITIVES / INDIRECT_DATA, ~65
  a frame) relaxed to "later fragment work waits" (REKARINA_GAME_BARRIER=full); indirect mesh draw counts copied once per
  stretch of the command stream with no memory writes (REKARINA_MESH_BATCH=0: per draw; 60 -> 28 barriers); a part goes
  to the queue inside a long pass too, every 200 draws (REKARINA_SPLIT_IN_PASS; the GPU waited ~2.9 ms a frame for the
  1,200-draw forest pass to be recorded: 41.7 -> 45.2 fps with a frame in flight); the graphics scale tier
  (REKARINA_GRAPHICS_SCALE=1|2: the game draws 4/5 or 2/3 of the window, the present enlarges it); debug code off the hot
  path (c3_pre's getenv per dispatch). Porch, unlocked: mode c 35-37 -> 42 fps, mode 1 39 -> 45-46 fps.
- Frames in flight and the foliage. f_7002bfd340 reads the 16-byte result of the foliage culling (visible counts) in
  a ring of four 24 KiB pools (0x29bc6c000 ...) and writes the slot's initial values back for its next use. With a
  frame in flight our write-back (end of the submission) comes after that read: the counts are stale, and the far
  trees lose their leaves (or cast the wrong shadows) for one frame, every eighth frame at worst, in some runs. A merge
  rule "the CPU's word wins" made it happen every eighth frame (reverted). A result guard (REKARINA_RESULT_GUARD=1,
  memory.c / vk_memory.c: the small ranges a shader wrote are closed to guest threads until their submission is done,
  an access waits for it) removes the flicker but waits ~47 times a frame (~20 ms: the pages hold CPU data too):
  13-20 fps; kept off. Next: guard only the words the CPU reads (learned from read faults), or per-part write-back
  so results reach guest memory as early as the console's.
- Async replay (REKARINA_REPLAY_ASYNC=1): same frame rate (GPU-bound), more flicker; dropped.

## Frames in flight without flicker: late copies for the pools shaders write (2026-11-08, night)

- The game's six submissions per frame: queue 1, (a fence), 2, 3 (compute: culling, lights, fog; ~3.2 ms of GPU), 4, 5,
  6 (main graphics, ~12 ms). The CPU reads queue 3's small results (foliage culling counts, f_7002bfd340) during the
  next frame. REKARINA_PRESENT_SYNC=c1 (the present waits for its frame's work but the last submission) has queue 3 done
  before the next frame starts and lets queue 6 overlap it: 45-47 fps on the porch against 42 (c).
- With the submit-time mirror copies (REKARINA_MIRROR_SNAP, on with any overlap) the far trees lost their leaves every
  eighth frame: the game writes the foliage culling's initial records after the submit that uses them, relying on
  the GPU running it later. Without the copies that was gone, but about one frame in 300 drew another frame's sun
  shadows (CPU-only data read at GPU time). Now the uploads of pools a shader writes (PoolRec.shader_written: results,
  counters, argument records) read guest memory when the GPU runs them and the rest is copied as of the submit
  (vk_memory.c prologue, REKARINA_SNAP_LATE=0 undoes): 900 frames on the porch (c1 twice, 1 once) and 300 frames of the
  reference script with the camera turning, no flicker. The ring now takes ~0.4 MiB a frame (0.02 ms).
- Default for OoT: REKARINA_PRESENT_SYNC=c1 (nvn_api_obj.c), mirror copies as of the submit with the late rule.

## The shaders as the driver compiles them, and the replay's own costs (2026-11-08, early morning)

- The driver's view without Nsight: NVIDIA reports VK_KHR_pipeline_executable_properties. REKARINA_PIPESTATS=1
  (vk_pipeline.c pipestats_log, vk_device.c enables it) logs each new pipeline's stages once: register count, machine
  code size, shared memory, keyed by the FNV-1a hash of the stage's SPIR-V (the hash REKARINA_DRAWTIME prints; the cache
  file with that hash is the stage's SPIR-V). Found: 12 mesh shaders (the foliage) at 255 registers with 1-2 MB of
  machine code, full-screen material shaders up to 860 KB. Their SPIR-V calls one 602-instruction function 506 times
  (218, 253 in the fragment shaders): the raw global memory access of patch 0020, whose 13-step binary search over the
  pool table was unrolled; the driver inlines every call.
- Translator patch 0032 (tools/xlate/patches/0032-global-memory-map-two-level-index.patch, PATCH_COUNT 31) finds the
  entry from an index the host writes after the entries (vk_exec_draw.c gmem_snapshot): a word per 2 MiB chunk below
  2^40 (the entry holding the whole chunk, or GMEM_SPLIT | a page table for the chunks several entries share), and a
  word per 4 KiB page in those tables. No search. Variants measured on the porch (GPU per frame, same build otherwise):
  unrolled search 19.8 ms; search as a loop 16.4; three levels (GiB, chunk, page) 15.9; a tagged 64 GiB window with
  that as fallback 16.7; 16-byte chunk records with the delta (no entry read) 19.1; two levels 14.8. The function is
  inlined at every access, so its size and branches count more than the loads it saves. The two-level index is 2 MiB
  for the chunks; the pools change every few frames (576 tables in 50 s), so table buffers are kept and reused
  (GmemBuf): only the chunk words the old or new table set are written. 0.5 ms a table (2.3 ms at most) against 2 ms
  (112 ms at most) with a new buffer each time. REKARINA_GMEM_DUMP=<path> writes the first table; scratchpad gmem_check.py
  simulates the shaders' lookup (0 wrong of 17,172 on the porch). One pool lies at 512 GiB (0x8000000000, 305 MiB).
- The replay's getenv: vk_exec_xfer.c asked getenv("REKARINA_RESULT_WATCH_NOREPORTS") per counter report and per event
  (debug code of 2026-11-07): ~20 % of the replay thread (getenv walks the environment block). Read once now.
  49.2 -> 54.7 fps.
- Merges: REKARINA_WB_MERGE=0 (a measurement, it breaks the exposure) takes 2.3 ms off the GPU frame and 1.3 ms off the
  CPU's. The merge marks of REKARINA_GPU_PASSTIME (2.5-3.3 ms) also hold the drain of the work before them (their first
  stamp is TOP_OF_PIPE). The merge range lists now live in device memory the CPU maps (vk_memory.c chunk_new): no
  measurable change.
- A wait for the CPU in the command stream (OP_WAIT_EVENT) cost ~2.5 ms in nearly half the frames. REKARINA_EVENT_DEFER=1
  (off) records past it and waits before the part after it is submitted (then refreshes the uniform copies and the
  mirrors synced since): no wait left, but one frame in 300 drew another frame's shadows; the replay reads more guest
  memory as it records (texture and sampler descriptors are the suspects). Async replay (REKARINA_REPLAY_ASYNC=1) is now
  much worse (33 fps, frames over 100 ms).
- The present path: the game's final image is R8G8B8A8_SRGB and the swapchain B8G8R8A8_UNORM; a capture of the
  swapchain (DEBUG-SCSHOT, REKARINA_SC_SHOT=lo-hi:step) has the bytes the game wrote (91/84/64 against 89/82/64), so the
  blit decodes nothing and the window shows what REKARINA_SHOTS shows.
- Still big: compute shaders that emulate the tensor-core HMMA.16816 in software (dmsh_hmma16816_f16, 1,220
  instructions, called 768 times in one shader: 1.4-3 MB of machine code), the merges (2.3 ms), the write-back and
  upload copies (~2 ms), and the replay on the CPU (~16.7 ms a frame against the GPU's 14.8).
- Porch, unlocked, c1: 46.9 fps (GPU 19.8 ms) at the start of the night, 57.0 fps (GPU 14.8 ms, 1 % low 33) now; 300
  frames at 60 fps flicker and brighten as before.
- Measuring: the scratchpad run.sh ran with REKARINA_REALTIME=0, where the frame rate module presents in whole steps of
  the game's own vsync interval ("present interval the game's own, whole-vsync steps"): every "unlocked" number of the
  night was capped at 60 fps, and a frame over 16.7 ms took two steps. With REKARINA_REALTIME=1 (the packaged rekarina.ini's
  realtime = 1) and REKARINA_FPS_LIMIT=240: the forest porch 60-63 fps (GPU ~15 ms busy + ~1.3 ms untimed prologues and
  present), lighter scenes of the reference script 92-103 fps (GPU 9.2-9.7 ms), menus at the 240 limit. Both are
  GPU-bound; the CPU waits ~0.5-0.9 ms a frame for nothing.
- Also measured and left: the pool lookup functions declared DontInline (NVIDIA honours it: real calls, GPU 14.8 ->
  19.7 ms); the lookup without its end check (the index is exact to the page): no gain; the merge in workgroups of
  256 or 1024 instead of 64 (one word per invocation): -0.1 to -0.2 ms, kept at 256 (REKARINA_MERGE_GROUP, vk_memory.c
  merge_init patches the SPIR-V's LocalSize and WorkgroupSize); the merges with every invocation idle
  (REKARINA_MERGE_EMPTY=1): -1.3 ms, so the merge's work is about 1.3 ms, nearly all of it host-memory writes; the heavy
  material shaders have no packed fp16 arithmetic (no gain from native fp16 there).
- Where the fixed costs stand (porch, per frame): write-back and merges ~3.5 ms (merges ~2.5), mirror prologues
  ~0.9 ms: in the lighter scenes that is ~40 % of a 10 ms frame. The next real step is the lazy write-back of
  roadmap-100fps.md (GPU-written ranges the CPU does not read stay in the mirror; a guarded page writes back on access).
- Tensor cores for the game's upscaler network: translator patch 0033 (PATCH_COUNT 32) turns HMMA.16816 with an f16
  accumulator into VK_NV_cooperative_matrix loads, one multiply-add and a store on the subgroup's block of shared memory
  (Instruction.MatrixMultiplyAddShared; the helper keeps its stores and reads D back), when the device runs 16x8x16 f16
  matrices at subgroup scope (vk_device.c checks vkGetPhysicalDeviceCooperativeMatrixPropertiesNV; REKARINA_COOP_MATRIX=0
  emulates as before; host option "coop_matrix", job "coop_matrix"). The grammar-generated
  Module.TypeCooperativeMatrixNV put the type into the function being written and declared it once per call (the
  driver crashed compiling it); the patch declares it with the other types. The upscaler dispatch (cs f9d0cf3842e5fc62,
  40x45 groups) 0.62 -> 0.38 ms; frames differ from the emulation's no more than two emulated runs differ from each
  other; 300 frames without flicker.
- Queue 3 and queue 6 are a fork-join on the GPU (details in oot-architecture.md, "Order of work"): the replay's serial
  order makes queue 3's middle part read the previous frame's results of queue 6's early passes, and gives up the
  console's ~3 ms of compute beside the graphics. REKARINA_EVLOG2=<frame> now logs any frame, with command positions.
- Async write-back, second try (REKARINA_ASYNC_WB=1, off): the merges and copies of ranges of 256 KiB or more on a queue
  of a compute-only family (NVIDIA time-slices two graphics-family queues, which made the first try slower), small
  ranges still in the submission, the same waits for ranges a pending write-back reads. The save never finishes loading
  with it (the loading screen stays up through 90 s; 61 fps without it). Not looked into yet; buffers are still
  EXCLUSIVE to the graphics family (scratchpad patch_awb_share.py would make them CONCURRENT).

## Findings of 2026-11-08, second half (flicker, queues, fog, loading)

How to measure. Screenshots (REKARINA_SHOTS) slow the game enough to hide races between the CPU and the GPU, so a run with
screenshots can look clean while the player sees flicker. Real-time runs use the flight recorder (REKARINA_FLIGHT=1,
REKARINA_FLIGHT_AT=<frame>): thumbnails blitted on the GPU. In real time menus run at the frame limit, so frame numbers pass
quickly: check the thumbnails' mean brightness before trusting a window (one window fell on the loading screen and
reported "no difference"). Standing still and turning the camera with a script gives a one-minute repeatable test. Fps
between real-time runs of a route differ by several fps because the scripted route differs; compare per-frame costs of
step-by-step runs, or repeat runs in A/B/A order.

Fast flicker. The fast flicker of the whole picture is one-frame jumps of the frame's mean brightness when the next
frame's recording starts before the GPU has finished the previous frame. Mean jump over 480 frames while the camera
turns, of 255: 0.44 with REKARINA_PRESENT_SYNC=c1 (the present waits for all but the last submission, 94 fps), 0.03 with c
(waits for the frame's work, 87 fps). Default is now c. Waiting only at the next frame's first submission gives
0.04-0.18 at the same cost, so the game's own CPU work between the frames is not what breaks. Not the cause under c1:
late timestamps (REKARINA_TS_PENDING_NOW), submit-time mirror copies (REKARINA_MIRROR_SNAP=0: 0.38), uniform snapshots
(REKARINA_UBO_SNAPSHOT=0: 0.64, worse). REKARINA_MERGE_ALL=1 gives 0.26 in one run: plain write-backs are part of it. The page
guard for small GPU-written ranges (REKARINA_RESULT_GUARD=1) is unusable: 54 fps with lows of 7.5, and a fatal error in a
second run. nvnDeviceGetTimestampInNanoseconds reads reports early about one frame in five. The auto exposure's
luminance sum (see "Words the GPU left unchanged" in vk_memory.c) is the main suspect and is unproven.

Queues in the console's order. vk_exec.c "Queues interleaved" holds a submission at a wait no recorded signal meets and
records the rest inside the submission that signals it (REKARINA_QUEUE_INTERLEAVE=1, off by default). The order is what the
event words ask for, but with it every second frame's foliage lists are wrong: of the game's two per-frame record
buffers, one holds fixed counts (4096, 1084, 1600 where the other frames have 244, 54, 208; 15,000 mesh groups against
5,000), which flickers foliage and its shadow. REKARINA_MESHCOUNT=<frame> with REKARINA_MESH_BATCH=0 shows it. Giving the held
rest its own uniform snapshot did not change it, and the copies read the mirror in both orders; the cause is open.
Running the rest on the compute-family queue (REKARINA_ASYNC_COMPUTE=1) gave no speed on the forest path.

Translator. Added: fragment depth register (0034), storage base next to c1 constants (0035), NaN guard of
float colour outputs (0036), per-tap offsets of depth-compare gathers (0037), atomic read in the compare-and-swap loop
(0038). NMin/NMax/NClamp instead of FMin/FMax/FClamp made foliage glow white for 11 frames after a load on NVIDIA and
was dropped: found by building the translator without each patch.

Green blotches on rock near fireflies. Absent from the scene target before the fog composite (fs 218750899020472e),
present after it, with square edges of the volume's cells. Other candidates tested without effect: slots following a
2D target at a 3D volume's address (now filtered by target in vk_texture.c newest_writer), full subgroups in compute
pipelines (now required), group counts at c0 bytes 0-0xb instead of 4-0xf (also lost the device; left at 4). Reading
the 128x72x96 volumes as 8-bit dumps did not show the blotch; the dumps clip.

Loading a save. The frames right after a load reach 285-400 ms. A sampling profile restricted to submissions already
running for 25 ms (REKARINA_PROFILE_SLOWSUB) puts about half of the time in the first use of textures: image creation with
vkBindImageMemory of pool memory (a kernel call in the driver) and CPU deswizzling. Shaders and pipelines take no time
there. Resolving every registered texture slot ahead of time is wrong: the game registers thousands it never draws
(7 fps from the title screen on).

## Findings of 2026-11-08, shader preparation and post-load spikes

Preparing every shader before play. `tools/oot_prepare_shaders.py` reads all 42,177 shaders of the game from the NUP and runs
`rekarina.exe --prepare-shaders`. A prepared variant is useful only if its input id equals the one play asks for, so the seed has
to reproduce what `vk_exec_draw.c answers_for` computes. Compared against 1,813 inputs recorded in play:

- Wrong code length and file-only control words (0x30, 0xd4-0xdf) gave other program hashes: 37% matched, 5,691 failed.
- With both fixed, 74% matched. Of the 473 misses: 168 compute, 165 vertex, 78 fragment, 57 mesh, 5 task; 399 had a 3D,
  array or cube texture and 161 an unsigned vertex attribute.
- Mesh and task variants take zero local-size answers (only compute has them) and compute also answers its shared memory
  size (control +0x64, rounded up to 16); the seed had both wrong.
- `runtime/gfx_shader_guess.c` translates a default variant once with no texture answers, reads the types the translator
  declared and the integer-read vertex attributes, and translates again with those answers. It covers every stage.

Post-load spikes. Mirrored textures got a dedicated `vkAllocateMemory` each. Sub-allocating them from VMA blocks halved the
two worst frames after a load (1172 + 727 ms to 588 + 354 ms, REKARINA_TEX_SUBALLOC=0 restores the old path). Sub-allocated
memory holds what the previous texture left: without clearing it doors and cracks in trees shone white for hundreds of
frames, so `vk_texture.c zero_suballoc` fills the allocation through an aliasing buffer first.

Translator and device. Local variables now start at zero on every driver (REKARINA_ZERO_LOCALS, it was Mesa only): NVIDIA gives
an undefined load the value of another lane. Patch 0039 reads RED's 20-bit offset as signed. Neither removed the teal blobs
on rocks near fireflies: they survive at the same place (run_spin frames 3200-3800), are soft-edged at 2560x1440 and appear
at the fog composite. The fog volumes use the game's linear samplers (REKARINA_SAMP3D log: min 5, mag 1, format
B10G11R11_UFLOAT), so the volume's cells are not sampled nearest.

NVIDIA's disk cache. The driver keeps compiled pipelines in `%LOCALAPPDATA%\NVIDIA\GLCache`, capped near 1 GB and shared by every
program. It was full since 2026-11-07, so nothing built after that was stored: preparing shaders changed nothing and libraries
took 533 ms on average in play (max 2.6 s; frames of 4.9 and 2.3 s). `vk_device.c vk_nvidia_cache_beside_exe`, called first in
`main`, points `__GL_SHADER_DISK_CACHE_PATH` at `nv-cache` beside the exe with a 20 GiB limit. The driver ignores a path that does
not exist, so the folder is created first. All 42,177 variants take 1.6 GB; in play libraries then took 2-5 ms and no frame
waited for a pipeline.

Reported in play on 2026-11-08, not fixed yet: the second movie (the rider) is black and silent; teal and beige square blocks in
a grid of about 64 cells across near the great tree, worst when fireflies are around, with a lag-like smear there; a rectangle in
screen space with sharp vertical and horizontal edges (near x 5.5% and 93%) inside which the ground has a brown layer, and white
silhouettes of characters in it; a house door drawn golden and white until it finishes loading; a black shape at the lower
right; a flash on scene changes; black squares now and then.

## Findings of 2026-11-08, evening: what the frame costs while the camera turns

Bench: the user's test save, the camera turned by script (scratchpad run_spin), frames 2600-5400, GPU busy per frame from
`REKARINA_GPU_CSV` (real time, `REKARINA_REALTIME=1`; two identical runs differ by about 3% in the sum). Standing still at the same spot
is 14 ms of GPU work and about 62 fps; turning, frames 4200-5000 take 35-52 ms.

- `REKARINA_PASSTIME_DRAINED=1` and `REKARINA_PASSTIME_BARRIER=1` give each dispatch's own time; without them a dispatch is charged for work
  still running before it. On a heavy frame nearly every compute dispatch costs about 100 times its standing cost, though
  cs 763b043399140e9a even runs fewer groups (288 against 558).
- cs 3a7117b072154c38 (8x8 local, 64x16 groups, 512x512 RG16F, one of a fluid simulation's pressure passes, with divergence in
  cs e4636a7ad8bf2a79) takes 8.5-9.5 ms where its near-identical twin cs d2cba99d8ced7f2c takes 0.2 ms. Declaring the image format in
  SPIR-V does not change that. Skipping it saves about 12% of the window's GPU time.
- Not the cause: `REKARINA_PRESENT_SYNC` (c, c1, 1 give the same fps in this window), computeFullSubgroups (off: same).
- Frame time in lighter places is the sum of CPU frame preparation and GPU work (8 + 8 ms gives 15-16 ms): the CPU and GPU take turns.
- `REKARINA_TEX_VRAM_PRESSURE=55` did not lower video memory use (10.1 of 10.2 GB in use), so memory pressure is untested, not excluded.
  The game uses 9 GB local and 5-8 GB non-local memory; textures take 5.3 GB, sub-allocated blocks 7.8 GB.
- One fixed-step run with screenshots ended in a lost device: `read invalid at 0x2f0560000`.
- Dropping `COLOR_ATTACHMENT` and `ATTACHMENT_FEEDBACK_LOOP` from the simulation's 512x512 float maps (`vk_texture.c`, on by default,
  `REKARINA_SIM_NO_ATTACH=0` keeps them): GPU per frame over frames 4200-5400 was 38.3-40.6 ms in five runs without it and 30.8, 30.0 and
  38.9 ms in three with it. Removing either usage alone changes nothing. The picture is unchanged (same frames compared by eye; the
  differing pixels are falling leaves and the fairy). So the 9 ms pass is gone in two runs of three, for a reason not understood yet.
- Released render targets are not idle: dropping them early (`REKARINA_TEX_VRAM_PRESSURE=75 REKARINA_TEX_DROP_AGE=30`) only moves them to the live count, no change in GPU time (42.7 and 36.3 ms per frame against 40.1 and 39.7). `REKARINA_SIM_NO_ATTACH` on by default gave 40.1 and 39.7 in two more runs, so of five runs with it only two were fast: unproven.
- `REKARINA_TEX_RECYCLE=30` (off by default): a new GPU-only image takes the image and memory of a same-shaped entry released at least 30 frames ago. Frames 4200-5400 went from 40.1 and 39.7 ms per frame to 33.6 and 32.4 (2,304 images recycled), total GPU time of the window 60.3 and 60.9 s to 54.7 and 53.2 s. Peak video memory did not fall (11.4 of 10.1 GB), so being over the budget is still not shown to be the cause. Not yet checked for black flashes in play, which is why it stays off.

## Video memory was the cause of the 17-20 fps minutes (2026-11-08, confirmed)

Above about 9.2-9.5 GB of device-local allocations GPU time per submission part rises about fourfold and the GPU sits at 95% busy at 71 W: the same standing spot was 7.9 ms at 8.3 GB and 35.3 ms at 10.8 GB. Dropping released render targets did not help because the texture pool (`vk_texture_pool_slot_locked`, resolved for every registered slot) re-created them as live images. With a slot of a GPU-only target with no live image resolved to a stand-in (`REKARINA_TPOOL_NO_GPU`), released entries dropped after 30 frames over 75% of the budget, and same-shaped released images recycled, frames 4200-5400 went from 35-50 ms of GPU work to 7-10 ms, and peak video memory from 11.0 to 8.4 GB of a 10.1 GB budget. Whole-window GPU time 60.3 s to 25.4 s; bench 1% low 13 to 48 fps. Whole-frame brightness jumps are unchanged (0.041 against 0.044).

Lightmaps after a load. Materials read their baked lighting through the texture pool, resolved at most 2 ms per submission, so for the first seconds after a load much of the world was dark. While the previous update left a backlog the budget is now REKARINA_TPOOL_BURST_MS (default 50): mean picture brightness at frame 920 of the load went from 48 to 67 of its final 69, with the worst load frame at 307 ms during the black fade.

Other findings of the day, not fixed: the teal blocks are single texels of the 64x36x48 RG8 mask written by cs 8a2b01cebb868b41 and read by cs 07b7fa3adebce45e (each block is two 128-grid cells wide with a clean 16-15-16 px trapezoid); its fields select up to two local fog lights. The black flashes have a hard horizontal raster edge that moves between frames and predate the memory fix. The ground fog sheet is pass "fs 4838c412d5329961" (960x540, mesh pipeline), reading the 512x512 fluid simulation map.

The ground fog sheet's edges, white silhouettes and the ghost of the hero were the seam between its two lighting branches: fs 4838c412d5329961 raises the marched hit point, reprojects it and reads the scene depth there (unclamped) to choose froxel light or per-pixel light, and the two differ here. Forcing the per-pixel branch (`REKARINA_SPV_NOT`, default on in `gfx_shader.c`, tied to this shader's hash and ids) removes all three, the same frame compared before and after. Which branch's input is wrong is still open: the froxel texture 889 (pass 0101, cs f473bc2d170bfcaa) or the probe reads.

## CPU and GPU overlap (2026-11-08, evening)

Measured, not made the default. `REKARINA_PRESENT_SYNC=c1 REKARINA_SUBMIT_WAIT=3` lets the CPU overlap the previous frame's last submission up to the third submission of the next frame, which waits for that work before it is recorded (`nvn_api_core.c` nvnQueueSubmitCommands). The default stays `c`: the comment on that knob, written by another worker while these runs went on, records that the player saw part of the textures flicker while watching c1 with k = 3. The thumbnail measures below only see brightness, so they neither confirm nor rule that out.

What flickers under c1. Not the exposure: in the flight recorder's thumbnails the hero and the ground around him change from one frame to the next, with the fairy's blue-white light on him or not, in single frames and alternating runs. A box around the hero shows it better than the whole frame's mean ("the hero jumps": one-frame changes over 1.5 of 255 in 480 frames). `flglob.py` takes the first dump of a run, which is sometimes an earlier spike dump and not the `-f3700` window. Plain c: 6, 9, 8 jumps. c1: 40, 138, 185.

Not the cause, each under c1 and none kept: every pool as of the submit (`REKARINA_SNAP_LATE=0`, 93 and 32 jumps; foliage records may add to it); late copies only for the pages a shader wrote instead of its whole pool (130); taking the uniform and mirror copies again after a CPU-signalled event that was already met (153); merging all write-backs (`REKARINA_MERGE_ALL=1`, 72); copying late the pages holding an in-flight GPU write of an earlier submission (0 MiB moved, 147 and 113).

Where it is. Waiting for the previous frame's work before recording the k-th submission, under c1: k=1 6 jumps, k=3 8 and 5, k=4 111, k=6 120. Waiting after the third submission's recording instead of before: 157. So the third submission (queue 3: culling, lights, fog) has to be recorded after the previous frame's last submission finished. A 1.5 ms sleep before recording it gives 19 jumps (c1 alone: 40-185), so time alone helps and the wait may only be giving other game threads time to finish. The 4 ms run (2 jumps) reached the window 10 s later in the route, a heavier place (1,718 draws a frame against 620-680), and is not comparable. What the recording reads too early is open; the mirror and uniform paths above do not explain it. The sleep costs more than the fence wait (median 16.0 ms against 12.6 in frames 4200-5400 with 4 ms) and is not a fix.

Numbers, spin bench, GPU idle before each run (median / p99 / share over 16.7 ms):

| run | frames 3600-4200 | frames 4200-5400 |
|---|---|---|
| c, first batch | 12.9 / 28.3, 13.8 / 41.0 | 12.5 / 26.4 / 22%, 12.1 / 26.5 / 22% |
| c, perf trace | 14.6 / 29.6 | 13.2 / 30.2 / 25% |
| c, A/B/A batch | 13.6 / 34.9, 12.6 / 29.9 | 14.8 / 36.1 / 40%, 14.5 / 31.0 / 40% |
| c1 + k=3, earlier runs | 12.2 / 25.4, 11.8 / 25.7 | 12.6 / 24.6 / 20%, 11.7 / 28.0 / 22% |
| c1 + k=3, A/B/A batch | 12.2 / 24.9, 13.4 / 27.0 | 13.2 / 28.2 / 27%, 12.6 / 27.1 / 21% |

With k=3 the brightness measures equal c's: the hero jumps 8, 5, 5, 4 and whole-frame jump mean 0.16, 0.04, 0.05, 0.03 (c: 0.04, 0.16, 0.16, 0.16, 0.04). Batches differ by 1-2 ms, so only same-batch comparisons count. In light frames the render thread is the limit: c1 leaves 0 ms of fence wait against 3.8 ms under c, and the frame is 13 ms with the GPU busy 9.5 ms (perf trace, frames 4200-5400). More overlap will not help there; a faster replay will.

Measuring. Frame numbers do not mean the same camera position between runs (the spin depends on elapsed time, and a faster run is up to 400 frames ahead at frame 4000), so pictures of the same frame number are not pixel-comparable and frame windows cover somewhat different views. The scratchpad `run.sh` stops any `rekarina.exe` under `build/oot` before starting one, so two workers on the same exe kill each other's runs.

The same A/B with the flash trigger off (`REKARINA_FLIGHT_FLASH=0`), four runs alternating, `flglob.py` brightness 61.6-63.3 in each, so its line is the frame 3700 window: whole-frame jump mean 0.040, 0.041 with c1 and k=3 and 0.040, 0.041 with c; p99 0.51, 0.51 against 0.34, 0.36. Frames 3600-4200, median / p99: k=3 14.1 / 37.6, 11.4 / 25.5; c 12.3 / 29.9, 13.1 / 32.3. Frames 4200-5400: k=3 13.3 / 35.6, 12.0 / 25.9; c 13.1 / 28.9, 12.3 / 28.2. Over all comparable runs (six with k=3, seven with c) the middle run's median is 12.2 against 13.1 ms in frames 3600-4200 and 12.6 against 13.1 in frames 4200-5400, p99 25.6 against 29.9 and 27.6 against 28.9: about 1 ms in the light window, with single runs overlapping.

Beyond the mean, in the same windows: per region (the hero, ground, foliage, rock) the one-frame jumps with k=3 equal those of c (the hero 0.10-0.16 against 0.10-0.16; c1 alone 0.45-1.87); tiles off both neighbours while the neighbours agree, a measure of partial flicker and of objects missing for a frame, 118-429 per 480 frames against 119-474 (c1 alone 461-2,587); black or dark frames at most one in either mode; one-frame outliers of the draw, dispatch and conditional-render counts the same. No run has a period: the largest bins of the jump series' spectrum hold 6 % at most, in c, c1 and k=3 alike. Thumbnails are 320x180 and average small textures away, so the player's flicker of part of the textures with c1 and k=3, at about the rhythm the black flashes had, is neither seen nor excluded by any of this. Not tested: the suggestion that the third submission snapshots the game's double-buffered scene block (0x25e94b000 / 0x25e94e500) and light buffers (0x25fe1e000 / 0x25fe36040) between the CPU's event reset and set.

## What fills the slow frames, and what is ours in the GPU frame (2026-11-08, night)

Measured on the spin bench (run_spin, 120 s, real time, GPU idle before each run), frames 4200-5400 unless stated. Frames over 16.7 ms: mean interval 22.3 ms, GPU busy 11.7 ms, render thread blocked on the fence 4.0 ms. Frames under 14 ms: 11.7, 7.9, 3.4. No frame's GPU busy comes within 3 ms of its interval (0 of 400 slow frames): the slow frames are not GPU-limited. Counted against the flight recorder's rows, the frame's non-waiting time fits 6.2 ms + 5.2 us x draws, correlation of the interval with draws 0.87 (frames 3600-4200: 3.2 ms + 6.9 us per draw, 0.96). 360 of the 400 slow frames have 1,000 draws or more; the mean of the slow frames is 2,105 draws against 589 in the fast ones. The main submission's replay is one 12.4 ms stretch on the render thread (REKARINA_SPIKE_MS).

Part timestamps of a slow frame, milliseconds after the previous present call (a temporary log, removed again): the GPU starts queue 3 at +6.6, ends it at +12.6 (its last part 4.1 ms, of it about 3 ms write-back), queue 6's parts wait 1.5-3.2 ms to start until the GPU catches up at about +14.7, from there the GPU waits for the parts (the CPU is behind: gaps of 0.7-1.7 ms), the last part arrives about +20.2 and takes 4 ms more (full-screen post passes): that tail is the fence wait of the present. So in a slow frame the end is the CPU's last part plus a fixed 4 ms tail, and our write-back in queue 3 moves the frame's end only while the GPU is still behind.

Fenced per-pass time (frames 4100-4101, REKARINA_PASSTIME_DRAINED, REKARINA_PASSTIME_BARRIER; the marks' sum is 13.5 ms a frame against 11.7 ms busy unfenced): game passes 7.0 ms, game dispatches 2.5, write-back and merges 3.05 (merge dispatches 2.03, plain copies 1.02), mirror prologues 0.92. Ours: 4.0 of 13.5 ms. Per frame 46 MiB of plain write-backs, 19 MiB merged in 52 dispatches, 29 MiB of uploads. A render-thread profile over the slow window (REKARINA_PROFILE_RENDER, frames 3950-4300): bind_resources 22 % inclusive, new render target images about 10 % (vkAllocateMemory for dedicated images, vkBindImageMemory, vmaCreateImage), write-watch queries 4.9 % (160 a frame, 1.2 ms), the fence wait 18.5 %.

`REKARINA_WB_MERGE=0` (a measurement; it breaks the exposure): GPU busy per frame 9.1-9.5 to 8.2 ms, frames over 16.7 ms 26 % against 30-33 % of the two runs before it (one run), so the merge's cost over a plain copy is about 1.3 ms and the slow frames gain about 1 ms of it.

`REKARINA_SNAP_POOL_MB` (default 64, old size 16): a forest frame merges 19-31 MiB of snapshot chunks and a 16 MiB pool destroyed and recreated two chunks a frame (perf line: new buffers 2.00 to 0.00). Three A/B pairs, 16 MiB then 64, frames 4200-5400, median / p99 / share over 16.7 ms: 14.4 / 29.4 / 38 % to 13.9 / 32.4 / 31 %; 13.8 / 30.6 / 29 % to 12.5 / 26.5 / 22 %; 14.3 / 32.2 / 36 % to 12.8 / 28.7 / 26 %; GPU busy 9.6, 9.1, 9.3 to 9.1, 8.7, 8.6 ms. Frames 3600-4200 share over 16.7 ms, noisier: 40 to 38 %, 37 to 26 %, 37 to 47 %. The picture is unchanged (same frame standing still, compared by eye).

Tried and not kept: the async write-back (`REKARINA_ASYNC_WB=1`, compute-family queue). The "save never finishes loading" recorded earlier is a deadlock: the last part's prologue runs after `vk_mem_mirror_end` has noted the submission's own write-back ranges, its uploads overlap them, `awb_check` makes the part wait for the write-back of its own fence, and the write-back waits for that part. Skipping the own fence and making the shared buffers CONCURRENT (scratchpad patch_awb_share.py) loads the save, but the bench is worse: 71 fps against 82-89, 64 % of frames 3600-4200 over 16.7 ms, and flglob brightness jump mean 2.76, p99 75 against 0.04 and 0.45, so the picture flickers hard. Neither change is in the tree. Pre-querying the write-watch for the pools of the last frame's submissions (`REKARINA_PRESNAP_SUBS`, 105 on-demand queries a frame) showed nothing clear in two pairs (35 and 24 % against 32 and 30 %) and went back out. `wr_by_lo` in the report of the render profile is VMA and driver time under a wrong symbol: VMA's C++ names are not in the map's regular expression, so the nearest named symbol before them absorbs them.

Not a GPU problem: a slow frame needs about 5.6 ms less. The GPU part that can give is the 3 ms of write-back in queue 3 (it only helps while the GPU is behind the CPU) and nothing in the 4 ms tail, which is the game's own. The remaining cost is the replay's per-draw work (5-7 us a draw, no single hot spot) and the render target churn.

## The first seconds after a load (2026-11-08, night)

`REKARINA_SECTIONS=<csv>` writes one row per present with the sum, count and maximum of every timed section of the replay thread (`runtime/perf.c`); it costs one constant test when off. Frames are counted from f0, the first `vk_texpool: frame N` line.

Region freezes. Between +101 and +250 one submission a frame spent about 16 ms in `vk_mem_mirror_begin`, and 10.8 ms of it was `FZ_STATE`: `freeze_locked` walked the region's page states with VirtualQuery, which measures the whole run of equal pages from its address on (about 13 us per resident MiB, a 3 GiB run 42 ms), and only the first 2 MiB of the answer are used. A region whose pages never had another protection now takes one VirtualProtect, which fails unchanged when a page is not committed (`REKARINA_FREEZE_WHOLE=0` walks as before). First minute of the spin bench, frames over 25 ms: 119 walking, 35-40 with one call.

Texture slots before the first frame. Nothing resolved registered slots until the first scene frame bound the pool, so frames +1 to +6 created and uploaded about 3000 textures at about 139 ms each. Submissions with no scene on screen now resolve slots within `REKARINA_TPOOL_PUMP_MS` (8 ms, 0 off): frames +1 to +6 cost 47 ms on average, and textures are complete at the fade-in. Unbudgeted resolving from the title ran at 7 fps; the budget is what makes it possible.

Same registration. `vk_texture_register` bumped `slots_gen` for every call, and the game rewrites much of its descriptor pool every frame, which invalidated every memo keyed on `slots_gen`. A registration equal to the slot's, with a live entry, now returns early (`REKARINA_TEX_SAME_REG=0` off). The feedback scan walks bound slots only and leaves the bindless pool arrays alone (slot -1).

Spin bench after these, three runs: first minute 99 fps, 1% low 42, 35-40 frames over 25 ms; second minute 98 fps, 1% low 61-65, at most 1 frame over 25 ms. Before them (hp_pb1): 89.5 fps, 1% low 24.8, 294 frames over 25 ms; second minute 83.9, 47.1, 12. After the freeze change the second minute no longer has any frame over 25 ms.

What is left in the window, after the pump, frames over 30 ms: +1 (207 ms: 95 ms in one alias bind of a 1600x1600 depth image, shadow atlas), +2, and a group at +28 to +37 from the second wave of registrations.

Also: `REKARINA_MOVIE_FPS` (default 60) holds the frame rate while a movie plays, above and below it, because the event script counts updates as 60 Hz frames (a 30 fps limit stalled the opening movie against its sound), and the NV12 copy tells the page tracking about its write. The window paints black until the first present (the white flash at launch).

## The one-frame flicker: queue order (2026-11-09)

A detector for it: the flight thumbnails (`REKARINA_FLIGHT_FRAMES=2400 REKARINA_FLIGHT_FLASH=0`), each frame cut into 16x9 tiles, a tile counted when it differs from both neighbours while they agree (`flick.py`, scratch). On the spin bench it finds the flicker the player sees: the same three frames in every run, at the same camera positions. The frame's ground takes another material and its shadows are misplaced, with a hard horizontal edge; each such frame also creates about 40 render targets where other frames create 12.

Ruled out, the three frames stay: a fixed scene scale (`res_fixed`, kept: it removes the dynamic-resolution churn), `REKARINA_TEX_ALIAS=0`, `REKARINA_TEX_RECYCLE=0`, `REKARINA_TEX_DROP_AGE=600`, `REKARINA_BIND_SKIP=0`, `REKARINA_TEX_MEMO=0`, `REKARINA_SHADER_MEMO=0 REKARINA_PIPE_MEMO=0`, `REKARINA_TEX_SAME_REG=0`, `REKARINA_TEX_SUBALLOC=0`, refreshing the pool at every slot change (worse). `REKARINA_TPOOL_NO_GPU=0` is much worse (14).

`REKARINA_QUEUE_INTERLEAVE=1` (now the default): the three are gone in three runs, and the single-tile blips fall from about 290 to about 50. The queues are replayed in the order their event waits ask for, as on the console. The picture's mean brightness on the bench falls from 64 to 34: the exposure the game computes is now the one for the right order, which is what the note above on the console being about a stop darker expected. It costs about 4 % of frame rate. The earlier objection, foliage lists wrong every second frame, did not show in the detector or the screenshots.

## What was ported overnight, and the frame budget (2026-11-09)

Added: DLSS and DLAA (armed at the scene's light target, motion from reprojected depth through the camera block), FSR 3.1 upscaling and frame generation, DLSS Frame Generation through Streamline (`runtime/dlssg` builds with llvm-mingw, the SDK's headers are enough; generation needs the window in front and Reflex on), widescreen (21:9, 32:9, the window's shape, heights up to 2880), the movie cap at 60 fps: with it the second movie, the dream with the rider, plays. `REKARINA_MOVIE_FPS` holds it because the event counts updates as 60 Hz frames.

Where a frame goes on the spin bench: median 10.2 ms, replay thread 5.3, GPU busy 7.0, the present's wait for the GPU 5.0. A longer window: replay 8.3, wait 5.3, GPU 8.7. With no present wait at all (`REKARINA_PRESENT_SYNC=off`) the second minute is 102.5 fps against 96 and the small blips rise tenfold, so the wait costs little: the CPU's work and the GPU's are about equal. `-O2` on the 83 generated chunks that hold 90 % of the guest samples (`REKARINA_HOT_OBJS`, one minute to build) changed nothing. A profile-guided runtime gives 2 %. Inside a frame the render thread waits about 2.3 ms for CPU-signalled events and about 2 ms in guest waits. Descriptor builds are 2.9 ms; 4 % are skipped, and every draw that reaches the guest memory table is marked volatile.

## Dialogue sharpness, pause screens and what the frame is limited by (2026-11-09, morning)

The sharpness that changes in dialogues is DLSS without jitter. Measured on a dialogue with the camera moving, then standing (mean squared Laplacian of luma over the picture, frames captured at 2560x1440): 400-880 while the camera moves, 140-190 within a few frames of it stopping, one step and no history reset in between (`why[]` stays). Without DLSS the same frames stay near 490. `REKARINA_DLAA_MV=0` shows the same step, and so does preset E (230-280 standing), so it is neither the motion vectors nor the model: with no sub-pixel offsets, DLSS repeats one sample per pixel while the camera stands. Jitter through the viewport (`vk_dlaa_viewport` for OoT's camera block) turns the picture white within seconds; jittering only the depth-writing draws is whiter still. Not solved; the picture is evened out instead: `vk_dlaa_still` (0 while the reprojection differs from the identity, up to 1 after 8 standing frames) drives RCAS on the present at 1:1 (`vk_fsr_record`), at the sharpness slider's strength. Moving against standing went from 5.5x to 1.6x (measured on a window of the picture's size at 1920x1080, with `REKARINA_SC_SHOT`, which saves the swapchain image: `REKARINA_SHOTS` is taken before the present and does not show it).

The game's pause and inventory screens draw the scene at the picture's full size (2560x1440 under a 1707x960 scene), so DLSS needed a second feature at the first opening, and frame generation was switched off and on at each one, which re-creates its resources: four or five frames of 400-750 ms per opening in a player's log. Now the first feature is made before the guest starts (58 ms when the model is in the driver's cache; `REKARINA_DLAA_WARM=0` off), the full-size one in the scene's first frame, and frame generation stays on through frames without inputs. The first scene frame: 2342, 1146 and 719 ms before, 251 ms after; nothing above 100 ms at the inventory after that (scripted openings, foreground, DLSS Quality and DLSS-G, 2 frames presented per frame).

Tried and dropped: a per-binding memo in `bind_resources_` (only rebuilt bindings whose inputs changed; verified with a build-and-compare mode, no mismatch) gave 81.0 fps on and 81.0 off on the porch.

Overlap retested with the queue interleave on: `c` 81.0 fps (6 blip frames of 2400), `c1` 84.7 (131), `off` 84.1 (432). The overlap buys 4 %, so the present wait is not what limits the frame. A scene a quarter of the size gives 90 fps against 81: the pixels are not it either. On the porch at quarter size the frame is 10.8 ms: replay about 6.5 (draws 3.1, of which descriptor building 2.2; other operations 1.6; dispatches 1.2), fence wait 3.9, GPU 7.9. CPU and GPU both have to shrink for a gain.

Porch, second minute, 2560x1440 output, foreground: DLSS Quality 73.1 fps, DLSS Quality with DLSS-G 68.2 fps of game frames and twice that on screen, 1% lows 48.8 and 46.0. After the load, 69 frames over 25 ms in the first 15 s, all of them in the first 2 s (from frame +1, 215 ms, the DLSS features; +32 to +52 about 33 ms each: the wave of new textures), none afterwards.

### The barrier experiment (2026-11-09, morning)

No thread of the process is saturated on the porch (the busiest 49 %, 3.7 cores in all), and the frame follows about 2.9 ms + the GPU time: 10.8 ms at a quarter-size scene with the GPU at 7.9, 12.1 ms at the full one with 9.1.

`REKARINA_BARRIER_EXPERIMENT=1` (only the game's barriers) gave 98.7 fps against 81 and no barriers at all 104-105, with the GPU at 7.0 and 6.5 ms. Those numbers are not a saving: the picture goes white (mean brightness 130 against 57) and the gain appears exactly when the two full barriers around the mesh indirect batch go (`op_draw_mesh`: records before the copies, counts before the indirect reads). Without them the indirect reads see stale counts and geometry is not drawn, so the GPU has less to do. Bisected by call site (a test table of `__FILE_NAME__:__LINE__`): every other group left out alone or together stays at 83-85 fps with a clean picture. Narrowing the two mesh barriers to compute/transfer -> transfer and transfer -> indirect keeps the picture and gives 84 fps. Leaving out the barriers at pass begins, clears, uniform updates, conditional rendering and held submissions gives 85.5 fps against 82.3, a clean picture on the 1000 frames checked; not adopted, as 4 % does not pay for a possible flicker that one scene cannot rule out. A barrier only for a target written since the last barrier (`REKARINA_PASS_BARRIER=same`): 83.8 fps, no gain. The barriers are not where the frame goes.

Same package on a copy of the save in the forest, foreground, 2560x1440 picture: FSR 3.1 with FSR frame generation 56.8 fps of game frames, about 117 on screen; DLSS Quality at 21:9 (picture 3416x1440, scene 2560x1080) 51.9 fps. Both pictures are correct.

### The dialogue box and the water under frame generation (2026-11-09, morning)

A player saw the dialogue box blink when choosing an answer, with DLSS-G on. Real frames do not show it: the flight recorder at real speed through the dialogue (1500 frames, DLSS Quality) has no excursion of the box's area. DLSS-G was given no HUD-less picture (`vk_dlssg_frame` passed NULL, only FSR's generation had one), so it took the interface for the scene. It now gets the copy `hudless_record` makes; two frames presented per frame, no Streamline errors. Generated frames cannot be captured here, so the visible result still rests on the player's eyes.

The same player reports the water moving in jerks and the effects around the hero jumping. Real frames: the water area changes smoothly from frame to frame at 95 fps and at a 60 fps limit (no near-zero frames, 1.38 and 2.27 of 255 per frame, proportional to the frame time), and twelve consecutive full-size frames show the glints around the hero changing continuously. A simulation stepping on its own clock would show frames without change. Not settled; the suspect is frame generation interpolating a surface that has only the camera's motion vectors.

Shadow map size is not where the GPU time goes: `REKARINA_SHADOW_RES=512` 79.9 fps against 79.1 with the game's own size.

### The ripple around the hero above 60 fps (2026-11-09, open)

Swimming, the ripple that belongs around the hero stands off to a side, freezes and jumps; a player sees it at any frame rate above 60 and not at a limit of 60, with or without DLSS and frame generation. Ruled out by that player's eyes: the 512x512 RGBA16F maps left without render target usage by `REKARINA_SIM_NO_ATTACH` (with `=0` no draw is dropped and the ripple is as wrong). The usage removal still drops four such draws a frame in that scene, so it is a defect of its own, unfixed. The step the game receives is already a running average (`frame_step_present`). The main binary names `cRippleUBO`, `IsUseVFR`, `SimulationWave`, `TotalDeltaTime`. Two explanations remain: a code path that truncates a step below one frame, or the ripple's data written by a game thread after the replay's snapshot of it. `REKARINA_REALTIME=0` with no frame limit (every frame a whole 1/60 s) tells them apart: not run yet. The other ports of this game had no fix for it then.

Sharpening of DLSS's picture is constant since this morning: following the camera switched it on and off through a dialogue and changed the overlay's scaling row.

### The corrected dump (2026-11-09)

The player replaced the .nup with a correct one: two packages (base 1.0.0, update 1.1.1) instead of one merged package, every file's SHA-256 verified, the update's ExeFS identical to the one the generated code came from, and the three .bntx.zs textures that began with 12 KiB of zeros now intact (`make_oot_prompts.py --scan`: none damaged). The packaged RomFS overlays still held stand-ins for those three, and an overlay's manifest names the SHA-256 of the file it replaces, so the game stopped at start with "the original of ... has SHA-256 ...". `tools/oot_romfs.py` now reads every package, later over earlier, as `runtime/romfs_nup.c` does; `make_oot_prompts.py --scene-scale 0.75` writes overlays equal to the old ones except for the stand-ins and the resource table entries that made room for them.

### What changes when the game takes the camera (2026-11-09, afternoon)

Reported: the picture's sharpness jumps when an answer is chosen in a dialogue, and when the game takes the camera, with DLSS and DLAA alike. Measured on the dialogue, a frame every two captured from the swapchain, sharpness as the mean squared Laplacian of the scene area. The present path did not change once in the run (`vk_fsr_record` returns 2 through RCAS on every frame; the overlay's "FSR 1" was a wrong label, fixed). Choosing an answer moves the camera: the two frames around it show the characters in other places. With DLSS the sharpness goes up by 35-45 % there (855 to 1100; 827 to 1101) and halves in one step when the camera stops (1043 to 534). Without DLSS the choice leaves it flat (263 to 261), but the start of the event raises it from 100 to 250 over about 20 frames.

Ruled out by measurement: the history reset on a changed light target (the step stays with it off, and those resets fall on other frames); the DLSS model (K, E, J: spreads 1.5 to 2.3 and no clear winner); DLAA (porch 39 fps against 61 at Quality, and a larger spread); the jitter experiment (47 frame-to-frame jumps over 25 % against 4). The full-screen shaders of the game's post chain differ between play and an event: leaving play takes 80f897f6b71056ea, 2f8e67f4046e0a3b, 21a58516c183b455, 91e009eb9352a14b, 76421515eecb8476 and 402857fea6d11463 away (SPIR-V hashes, `vk_dlaa.c spirv_hash`), an event brings ca7cba32ba66f7fc, 7a7ea9f8b57bb776, d24ec6917ac41506, 9f04e91bb4cba5b8 and d0812c8076916f33. Which of them accounts for the sharpness was not established.

The four 512x512 RGBA16F targets the render target usage rule drops are drawn only around the load, not per frame, so they are not the ripple's simulation.

### Evening of 2026-11-09: the ripple, the loads, and what else was added

The ripple around the hero above 60 fps. The water surface simulation (`main+0x472e0`) runs its compute passes `param * (int)ratio` times a frame, param 5 and ratio the VFR manager's step in 60 Hz frames. Above 60 fps the ratio is below 1 and the count is 0: programs e8ca1f7ac936cd4c (5 a frame) and 177bec1012800214 (1) ran in 100 % of 1200 frames at 60 fps and in 0.2 % at 240. The new patch set `ripple` (`REKARINA_RIPPLE_STEPS`, site `main+0x4760c`) takes `round(param * ratio)`: at about 97 fps the first program runs in every frame, 3.09 times (5 x 0.62). Shots of the same spot at 240 fps show the glints around the hero where the 60 fps reference has them. Found with `REKARINA_DISPLOG=<frame>:<frames>`, `REKARINA_DISPLR=<x>` (guest call chain of dispatches of that width) and `REKARINA_FNWATCH=0x7000047160:8`. The frame-time compare at `main+0x46da28` (16.0) looked like the cause and is not.

Loads. The first scene frame was 1113 ms, 548 of it looking up mesh shaders on the replay thread: a shader cache of 390,000 loose files, and about 1 ms per file. `tools/pack_shader_cache.py` was never run on OoT's cache. Packed (10.2 GiB, 19 minutes), the same frame is 595 ms and the lookup 39 ms. The next 600 ms frame is the first use of DLSS Frame Generation, on the present thread (`nvngx_dlssg` loading its model) while the game's submit waits for the queue lock; not fixed.

Video memory. `vram 9005 of 8789 MiB` in the first run, 100-250 ms stalls around it. Idle live textures now leave above 88 % of the budget (`REKARINA_TEX_EVICT`). In the forest none are idle (0 MiB of 4.8 GiB), so the eviction does nothing there.

Ported by hand: far cascade foliage at a 2x2 shading rate and one-sided (`REKARINA_FOLIAGE_SHADOW_VRS`, `REKARINA_FOLIAGE_SHADOW_CULL`); flushed ranges not copied where the write watch covers the pool (`REKARINA_FLUSH_UPLOAD=1` as before), 30 to 20 MiB of uploads a frame; a destroyed texture's record freed only after the submission being recorded, and no view cached of it (a device lost otherwise); deferrals from loader threads stamped issued + 1; the swapchain made again on a desktop change. Porch, DLSS Quality, no generation, second window: 85.8 to 87.9 fps, 1 % low 42.6 to 49.4. `REKARINA_MERGE_MAX_MB=32` is worse (74.3 fps: the merged pages are uploaded again, 66 MiB a frame).

DLSS jitter through the projection (`REKARINA_DLAA_PJITTER=1`, off by default). Rows 7 and 8 of the 0x3500 scene context get `P[n] += d * P[3]` in the submission's uniform copy, for blocks drawn into a depth target of the scene's size; the motion vectors and frame generation take the block without it. The y sign is -1 (+1 shows combs on the picture). A standing camera on the porch: sharpness 60 and 61 without and with it, where the viewport jitter fell from 112 to 36, no white screen, frame-to-frame change 0.375 and 0.473. Whether it removes the jump when a dialogue choice moves the camera needs the player's eyes: the runs differ in timing, so the Laplacian of two runs cannot be compared frame by frame.

## Night of 2026-11-09 to 2026-11-10: the picture, the frame protocol, and three structural changes

Reference scenes: the forest pond of the older save copy and, from the morning on, the village centre of the player's current save (about 2400 draws a frame against 1400). Numbers are from a quiet machine at 2560x1440, frame generation off, frames 3000 to 4400 of a camera spin; single runs differ by about 5 %, so changes are judged by the timed sections (`REKARINA_SECTIONS`) in pairs of runs.

Picture.
- Dialogue sharpness. A frame without HUD has no pass on the output target, so DLSS ran at the window copy and the game's own compute scaler overwrote it: the picture changed between DLSS and the game's scaler whenever the interface came or went. DLSS now runs at the game's scaling dispatch and that dispatch is skipped (`vk_dlaa_dispatch`, `REKARINA_DLAA_AT_DISPATCH`).
- DLSS jitter (`REKARINA_DLAA_PJITTER=2`). Every camera matrix (view-projection, projection, their inverses and transposes) in every uniform block a submission binds is jittered in guest memory before the mirror snapshot. Static geometry gets real supersampling; wind-animated foliage stays softer, there are no per-object vectors for it.
- DLAA (the scene at the output's size, `dlss` off with `dlaa = 1`). DLSS ran at the output pass, after the game's post chain had read the light target, so the screen showed the aliased scene. It now runs in place before the first compute dispatch that samples the light target.
- The base picture (no DLSS). The fixed scene scale defaulted to 0.625 (1600x900 under a 2560x1440 picture), which was the soft base picture; it is 1.0 now and `render_scale = 50..100` lowers it (`frame_rate.c` `scene_share`).
- A test run started in the background never goes fullscreen: the window is 2560x1421 and the picture is squeezed into 2526x1421 by a bilinear blit. Sharpness shots are taken at 1920x1080 for that reason.

The frame protocol (`REKARINA_FTRACE`, `runtime/ftrace.c`). The game serialises itself: its present waits for the results of the frame before, then the next main submission is recorded on the submitting thread. With the wait gone the frame is the replay plus about 2 ms, and the GPU's busy time follows the replay (it is fed by it).
- Results before the frame's end (`REKARINA_RESULTS_FENCE`, default 2 for this game). A second timeline is signalled by the part that ends at the first lighting pass; the game's waits take that as the frame done. `rt_knob` returns only on or off, and `results_point` read the mode with `atoi(getenv)`: the point was never taken unless the variable was set, and the base picture had no point at all. Both read the number now.
- Late sums (`REKARINA_LATE_SUMS`). The auto exposure is a float sum the GPU adds into and the CPU reads and resets once a frame. Landing before or after the read at random, the read saw no frame's sum, one or two, and the exposure pulsed. Ranges added into with float atomics after the results point are copied out, mirror and snapshot, to a host-visible ring and added into guest memory when a game thread's wait for a later value returns: one sum per read, a frame late.
- `REKARINA_RESULT_GUARD` (close the pages of a late result until it is written) gives the gain back and stays off.

Structural changes, all on by default for this game.
- The submitter thread (`REKARINA_SUBMIT_THREAD`, `vk_sync.c`). The replay thread queues a part's batches and another thread calls `vkQueueSubmit2` in order. `vk_queue_lock` waits for that queue to empty first, so the present, the uploads and the `vkDeviceWaitIdle` callers keep their place. 1.56 to 0.13 ms a frame on the replay thread.
- Write-back on demand (`REKARINA_LAZY_WB`, `vk_memory.c`, `memory.c`). The GPU writes about 106 MiB a frame into CPU-visible pools in the village and all of it went back to guest memory; the game's CPU code touches about 60 of those pages a frame. Each whole page of a shader-written range of 64 KiB or more has a state: new (closed for access, still written back), hot (an access reached it: written back for good), cold (240 writes without an access: not written back; its first access fetches it and the cold pages around it from the mirror with a copy of its own). 0.7 MiB a frame goes back, 28,100 pages are cold, four or five fetches in a run. GPU busy time 13.2 to 11.2 ms.
- No headroom thread (`REKARINA_VRAM_HEADROOM`): with it the first `vmaBindImageMemory2` of a scene's first frame took 80 to 110 ms.
- Released own-memory targets are recycled after 6 frames, not 30 (`REKARINA_TEX_RECYCLE`): about 550 MiB less held.

Results: village with DLSS Quality 69 to 71 fps (51 at the start of the morning, 44.7 with the old behaviour), village base picture 63.6 fps (46), pond with DLSS Quality 95 fps (82).

Open.
- Frame generation. With DLSS-G on the village runs at 16 to 18 ms a real frame while video memory is under the budget and at 26 to 56 ms when it reaches it; the budget is what other programs leave of the card. The game holds 7 to 8.4 GiB there and the amount swings with the camera: six in ten transient render targets cannot alias the GPU-only pool, because the driver binds an image at multiples of 1024 bytes and the game places them at multiples of 512 (`ALIASWHY`), and take memory of their own. A second pool block shifted by 512 bytes (`REKARINA_POOL_TWIN`) lets all of them alias; it is not measured on a quiet machine yet.
- The game makes its shadow targets in a new size nearly every frame while the camera turns (about 200 sizes in 500 frames), which moves every target placed after them: 5 to 15 new images a frame, about 2 ms of the replay. The GPU load the game measures does not drive it (`REKARINA_GPU_LOAD_ZERO`).
- Ending a part without waiting for the recorder (`REKARINA_REC_END_ASYNC`) crashes: the prologue and the next part's command buffer come from the pool of the buffer the recorder still records.
- The replay itself: 12.4 ms in the village with a flat profile (`REKARINA_PROFILE_RENDER`); the next step is recording parts on several threads.

## 2026-11-10, afternoon and evening: the base picture's frame, the crash, and a batch written without runs

Reference: the village centre of the player's current save, base picture (the scene at 2560x1440, no anti-aliasing, FSR 1), real time, no frame cap, standing and in a camera spin. The fixed-step test mode holds 60 fps and says nothing about the frame rate; only its timed sections compare.

Measured and in the package's test exe.
- The crash on a jump. `vk_texture_register` took a registration as unchanged when the slot's bytes matched, and returned with the slot still naming a cache entry that had been freed and its memory reused. An unchanged registration whose generation moved is now looked up by key again and must give the same entry, not released.
- The GPU frame. With `REKARINA_FEEDBACK=restart` (a pass that samples its own target ends and starts again, in place of the feedback layout) the GPU frame fell from 12.2 to 9.0 ms; the default on every vendor but AMD (`vk_device.c`, `REKARINA_FEEDBACK=layout` brings the old mode back). The pass marks (`VK_AMD_buffer_marker`, one per pass) cost about 1.2 ms of GPU time and are off unless `REKARINA_PASS_CRUMBS` or `REKARINA_GPU_PASSTIME` asks for them. Standing 75-77 to 83-90 fps, in the spin 58-64 to 72. By kind the GPU frame is: scene passes at 1440p 3.9 ms, full-screen passes 3.0, the 123 compute dispatches 2.5.
- With the GPU at 8.7-9.2 ms the frame is the CPU's: the game's render thread waits about 3.65 ms for its own record jobs, the replay takes 7.35 ms inside the submit call (9.9 in the spin), then the present.
- Images made ahead (`REKARINA_TEX_PREMAKE`): a thread creates the image of a texture at `nvnTextureInitialize`, the registration takes it. One device fault in about 21 runs while another program held most of the video memory; not seen again, not explained.
- A command pool per command buffer (`vk_sync.c`), the end of a part without waiting for the recorder (`REKARINA_REC_END_ASYNC`, which that made possible), boot-time pipeline threads at below-normal priority.
- The first 200 frames of a scene. The game creates 3 to 5 memory pools a frame then; each rebuilt the global memory table of the shaders (0.53 ms) and the texture pool update took 1.5 ms: 13 to 16 ms of replay against 8. The table is now kept for a submission unless the pools changed in a way it shows (`REKARINA_GMEM_PIN`): 15.6/13.3 to 12.7/11.7 ms. The rest is open.
- The existing asynchronous replay gave nothing: with the present waiting for its own frame 69 fps, with that wait relaxed 88-93 fps and 12-14 whole-frame brightness blips in 1,500 frames against 0-1. The game's own cycle without the replay is about 10 ms (six worker threads in phases, about 1,720 yields and 420 address waits a frame), so 100 fps needs both halves shorter.

Written after the order to stop running the game, compiled, not run in the game (the list to verify is kept outside the repository).
- `simd_fp.h` `redo`: a NaN result under FPCR.FZ goes to `fix` with flushed operands instead of the path that changes the rounding mode (9 % of guest time was there). `tests/diff/simd.py --no-validity`: 159,976 cases, none failing.
- `SleepThread(0)` returns through a host yield without the kernel lock when no thread waits for a core; host calls are counted only until the first present, SVCs in batches of 64 a thread.
- The 24 generated objects with the hottest game functions are compiled at -O2 (`tools/data/hot-oot-v111.txt`, `REKARINA_HOT_OBJS`).
- The game's zstd decoder replaced by the host's (`set zstd` in `recomp/patches-oot-v111.txt`, three call sites of the multi-frame decoder at main+0xeb380), off until `REKARINA_ZSTD_CHECK=1` has passed through a load.
- A guest thread may spin before it sleeps in the host for a wake or a virtual core (`REKARINA_BLOCK_SPIN_US`, off): the game's frame is a chain of about 420 address waits and signals, and each host wake costs a context switch.
- A prompt set of our own (`prompts = rekarina`): an Xbox pad's names on the game's own button shapes, by position and by label.
- The asynchronous replay, second version (`vk_backend.c`, off: `REKARINA_REPLAY_ASYNC=1`). The submit call returns once the replay thread has taken the submission's copies of the game's memory; texture and sampler registrations are queued behind the submissions before them; the present waits for the previous present's results; the staged readbacks of small targets reach their storage at the end of the game's present call, which the first version left to the completion thread (the luminance of the auto exposure came about a frame late and at no fixed point: the likely source of the blips); the 1 ms sleep of the game's event poll at main+0x57051c is 50 us.

Tools added: the SVC trace, host-call and write-fault notes of `REKARINA_FTRACE`, `REKARINA_WHOCALL`, the flat single-thread profile (`REKARINA_PROFILE_RENDER=1 REKARINA_PROFILE_CHAINS=0`). A profile matches only the map of the exe it was taken with.

Open: 100 fps in the base picture (the replay translated when the game records, on its 5-6 threads, is the redesign that removes the 7 ms), the rest of the load ramp, the restart feedback mode looked at only in standing shots of one area.

## 2026-11-11, night: the whole exe compiled with its profile, and what the frame is made of

Reference as before: the village centre of the player's save, base picture at 2560x1440, real time, no cap, a quiet GPU.

The profile-guided build of the generated code. The lifted code was compiled at -O1 (24 hot objects at -O2) and only the
runtime had an execution profile. Compiling all of it at -O2 takes about twelve minutes on eight cores, not hours, so
the whole exe now gets the profile: an instrumented build (`build_game.py --pgo gen`, 16 minutes), three training runs
at the save (standing, a camera spin, the asynchronous replay), `llvm-profdata merge`, and the build with
`--pgo <file>` (8 minutes). Standing 86.5 to 103.6 fps, in the spin 72 to 94; without the timed sections and the
overlay, which cost about 0.2 ms of replay a frame themselves, about 113 fps standing over a minute.
`installer/oot.py` compiles the whole exe with the profile when there is one (`build/pgo/pgo-oot.profdata`,
`data/` in a release). `runtime/simd_fp.h` is inlined into the lifted code: a change there after the training makes
every function that uses it compile without its counts, so it goes in before the instrumented build.
- `simd_fp.h` `vnan`: a fast vector result whose bad lanes are only NaN (and none the smallest normal) keeps its good
  lanes; only the NaN lanes go to `fix`. The game keeps three-component vectors in four-lane registers with a NaN in
  the spare lane, and every operation on them ran all four lanes again one by one (2.7 % of the game threads' time).
  `tests/diff/simd.py --no-validity`: 159,976 cases, none failing.
- `sleep_ns` and the core wait read the clock once per eight pauses: the SDK's job workers poll for work with 20 us
  sleeps, about 35 a frame each, and `RtlQueryPerformanceCounter` was a tenth of the game threads' running time.

What the frame is (`REKARINA_TRACE=perf,sched REKARINA_CRIT=<first>-<last frame> REKARINA_CRIT_OUT=<file>`, `tools/crit_path.py`).
With the synchronous replay 81 % of the critical path is the presenting thread inside `nvnQueueSubmitCommands`, which is
the replay (8.4 ms in the spin); the game's own work between a present and the next main submit is about 1.9 ms (its
main thread 0.67 ms, that thread's two small submissions 0.58 ms, job workers 0.5 ms). The frame is the replay plus
1.9 ms. All of the game's threads together run about 7 ms of lifted code a frame, most of it beside the replay.
The asynchronous replay (second version) lands on the same frame time: its submit call returns once the replay thread
has taken the submission's copies, which is once that thread has finished the submission before, so the game waits for
the replay again; without that wait the frame bound and the present's wait couple the two the same way and the frame
times scatter. It has no whole-frame blips left (the first version: 12-14 in 1,500 frames) and the same exposure as the
synchronous replay at a 60 and a 90 fps cap and when the GPU is the limit; the late sums of the auto exposure are
applied at the present under it (`vk_backend.c` `vk_be_present_end`). It stays off: no gain yet.
Standing, the frame is the GPU's now (8.9 of 9.5 ms busy): full-screen passes at 1440p 2.4 ms, 122 dispatches 2.4 ms,
scene passes 2.0 ms, our mirror prologue 0.35 ms and merge dispatches 0.29 ms.
Spinning before a sleep in the kernel emulation (`REKARINA_BLOCK_SPIN_US`) changes nothing in either mode.

The replay thread's profile is flat (`REKARINA_PROFILE_GUEST=1` writes the samples per thread as `trip` lines): the draw
translation 26 % (descriptor resolution 45 % of that, shader selection 17 %), transfers 6 %, write-watch queries 4 %,
one driver function family 5 %, lock calls 4.5 %. The game registers about a hundred texture slots again every frame
with other textures, so the texture pool writes about 160 descriptors a frame; neighbouring elements now go out as one
write (`vk_texpool.c` `flush`). Written and checked but not timed (below): descriptors kept per bound slot
(`REKARINA_SLOT_CACHE=1`, `check` compares every hit with a lookup; off).

Movies. All 26 are AV1 in WebM, decoded by the game's own lifted decoder. A new game from an empty profile
(`build/oot/script2.txt`) plays the title movie, the prologue and the rider nightmare with picture and sound
(`REKARINA_TRACE=audio`: levels above zero through them, three Opus decoders, no errors).

Movies on cards without tensor cores (2026-11-20). A player's GTX 1650 Super showed every movie at 17 fps, each frame
waiting about 56 ms for the GPU; a movie's event script counts the game's updates as 60 Hz frames, so the movies also
played at under a third of their speed. The cost is the network every movie picture goes through (agl's NDVE, see
docs/oot-xlate.md): on an RTX 5070 the title movie took 4.4 ms of GPU a frame with the tensor cores and 9.0 ms with
the translator's shared-memory stand-in for HMMA, 7.4 ms of them in the network's U-Net (passes 1-11). A body of
subgroup shuffles for the stand-in gave the same bits and was 4.7 times slower (43 ms: 32 shuffles an HMMA cost more
than the stand-in's wide shared-memory loads). `runtime/vk_movienet.c` keeps the whole network as long as the card
holds the movie's rate: while a movie plays it measures the GPU time of its frames, and when that does not fit the
movie's frame time (90 % of 16.7 ms) it leaves the U-Net out of the movies for the rest of the session: its dispatches
are skipped and pass 12 runs a pipeline that adds only its bias (under 4e-6) to the picture; the upscaler runs as
before. The title movie then takes 1.5 ms of GPU a frame with the stand-in; full-size shots of the same movie frames
with and without the U-Net differ by 0.6-1.1 levels of 255 on average (PSNR 43-50 dB). On the RTX 5070 the network
stays whole (8.9 ms measured with the stand-in, 4.4 with the tensor cores); a card about four times slower would need
some 36 ms with it and some 6 ms without the U-Net. `REKARINA_MOVIE_UNET=1` keeps it whole everywhere, `=0` leaves it
out of every movie, `REKARINA_MOVIE_FIT_MS` tests the cut on a fast card. The CPU was never the limit: the replay of a
movie frame takes 0.5-1 ms, also with four processors.

Video memory. At 1440p the game holds about 6.5 GiB: textures 4.3 GiB (ASTC transcoded to BC3 2.8 GiB), mirrors
0.6 GiB. When another program holds most of the card the system's budget for the game falls under that (After Effects
with 7 GiB: 5.4 GiB of budget), Windows moves images to system memory and the village runs at 13 fps. The game says so
on screen now (`vk_memory.c` `vk_mem_vram_short`, `ui/menu.cpp` `vram_notice_draw`). And it can take less: the texture
quality setting (`tex_quality`: auto, full, half, quarter; `vk_texture.c` "texture quality") resolves a slot that names
a plain 2D ASTC texture of 512 texels or more to the same texture without its first level (quarter: without two for
1,024 and more), through a derived descriptor whose storage starts at the second level and whose layout is checked to
be the tail of the full one. ASTC-as-BC3 2,771 MiB to 1,294 (half) and 1,055 (quarter), the process 6.5 to 5.4 GiB;
side by side the picture is softer close up and complete. auto picks at the first texture by what other programs leave
of the card (its size less the adapter's dedicated usage apart from the game's own): 7.25 GiB and more full, 5.9 and
more half, less quarter. With After Effects rendering beside the game the machine's CPU is the next limit (30-39 fps
inside the budget), which no setting of ours changes.

Loading. The host's zstd decoder replaces the game's lifted one by default now (`ZSTD_NATIVE`, patch set `zstd`): the
checked run (`REKARINA_ZSTD_CHECK=1`) decoded 4,096 calls on the way into the village both ways, 394 MiB in and 830 MiB
out, with no difference, in 958 ms against 2,377 ms.

The launcher. `installer/gui_oot.py` opens on a home screen once the game is built (Play, settings file, folder,
update), `installer/launcher/rekarina.c` starts it without a console, and the icon is new
(`tools/make_rekarina_icon.py`).

## 2026-11-11, morning: the player's own session, the driver cache's name, and what did not work

The player ran the night's build in the base picture and reported no 100 fps and stutters ("I jumped into the water and
it stuttered"). The game writes `logs\totk.log` beside its exe whenever it is started without a redirect, and those logs
gave the numbers: 82 to 93 fps, one frame of 226 ms with 204 ms of pipeline build on the render thread, frames of 45 to
70 ms, and 0.3 to 1.5 s at the first frames of the scene.

The driver cache is named after the executable. NVIDIA's driver keeps its disk cache
(`nv-cache\GLCache\<gpu>\<hash>\<name hash>.bin` and `.toc`) under a name made from the exe's file name, not its path:
`rekarina.exe` has the same file in every folder, `Rekarina.exe` another, `Rekarina-next.exe` a third. So an exe delivered
under another name starts with nothing: 28 pre-rasterization libraries of 55 ms and 20 fragment libraries of 154 ms on
average (385 ms at most) were built during that session. A fresh install had its 2.4 GiB of prepared pipelines under
`rekarina.exe`, was played as `Rekarina.exe`, and froze for 36 s and 32 s in its first run. The driver accepts a renamed
copy of the pair, which is how the package folder was repaired. The setup now prepares under the shipped name
(`tools/oot_prepare_shaders.py --exe-name`) and runs `Rekarina.exe --warm-shaders` in the install folder
(`installer/oot.py` Package: 55,711 variants in 40 to 77 s when the cache has them; AMD keys its cache to the exe's
path and builds there). With both, the libraries a draw needs take 1 to 9 ms on average. Replacing a cache loses the
whole pipelines play had built: the next start rebuilt the 1,460 recorded ones (21 s instead of 2.5 s, once).

What is left of the hitches (`REKARINA_SECTIONS` rows): the first frames of a scene, 250 to 325 ms and 110 to 150 ms, are
the resource binding of the scene's first draws (BIND_RES 154 to 170 ms over 4,800 binds, 35 us each against 2 us in
the settled scene, then 103 ms over 363 binds in the next frame; only 5 ms of it are new images, 32 ms new mirrors:
the rest had no section of its own), slot registrations waiting for the texture lock (15 to 38 ms) and the mesh
draws' shaders (34 to 46 ms); single frames of about 108 ms outside the replay or in the wait for the recorder thread
were seen in three runs and are not explained.

The binding time was the lazy write-back closing pages one at a time. A profile of the slow frames
(`REKARINA_PROFILE=<file> REKARINA_PROFILE_SLOW=90` writes every thread's call chains of each frame over 90 ms) put 34 of the
render thread's 86 samples in `ssbo_info`, `vk_mem_gpu_wrote_shader`, `mem_lazy_close` and the protection change under
it: every page of a range a shader writes for the first time is closed for access, some 10,000 pages in a scene's first
frames, one `VirtualProtect` each. `memory.c` `mem_lazy_close_run` closes a run of new pages with one call
(`plat_mem_protect_run_from`: the run of read-write pages from one `VirtualQuery`); BIND_RES in the scene's first frame
went from 154-170 ms to 26 ms (one run, with another program on the GPU), 1,500 standing frames after it show no blip.
The first start of a new exe still spends 0.4 to 1.5 s in the mesh draws' shaders at the scene's first frames
(SHADER_MESH, SHADER_MISS; 34 to 46 ms from the second start on): not looked into.

The machine was not idle. A sensor monitor, a browser and a download client took about a quarter of the CPU, and the
same exe measured 53 to 89 fps in the spin instead of the night's 94; run-to-run spread was larger than any effect
looked for. `process_priority` (normal, above, high; above by default for play) and `REKARINA_RENDER_CORE=1` (the render
thread on a physical core of its own, CPU sets) were written for this case; neither showed a gain in pairs (61 and 79
against 75 and 76 fps; 83 and 89 against 86 and 72).

The asynchronous replay's last obstacle is the frame's small submissions. OoT submits six times a frame: 48 and 16
bytes of recordings, 26 KiB, 72 and 40 bytes, and 1.2 MiB right before the present. The submit call waits for the
replay thread to take its copies at each, and at the small ones that is the end of the frame before. Letting the small
ones go without the wait (`REKARINA_REPLAY_HANDSHAKE=<bytes>`, kept for A/B) gave whole-frame brightness blips in 780 of
1,500 standing frames against none: their copies have to be taken at the call. A gain needs them taken on the
submitting thread.

AMD. On the integrated Radeon of this machine (6 to 8 fps) the game runs without a fatal and the picture is nearly
white. Frame dumps of the same frame on both GPUs (`REKARINA_DUMP`) show the sun's shadow cascades without the foliage,
which the game draws with task and mesh shaders, and an empty half-size depth target written by two mesh draws; the
forest is lit as if in the open, and the fog with it. The memory mode, the feedback mode, the compare-and-swap float
atomics and the missing recorder thread were each and all forced on NVIDIA and leave the picture right. The task and
mesh shaders use subgroup ballots in workgroups of 32; AMD's subgroup is 64, the 32-lane pin covers compute only
(`vk_pipeline.c` `stages_create` pins no mesh stage, `gfx_shader.c` hands the translator the fragment stage's answer
for them). Not fixed.

AMD, 2026-11-12. What the integrated Radeon's driver says (`vk caps:` lines of a start with `REKARINA_GPU_INDEX=1`):
subgroups of 64 lanes, size control 32 to 64, `requiredSubgroupSizeStages` 0x20, that is compute alone. So on this
driver branch ("AMD proprietary shader compiler", RDNA 2 and older) no pin can reach the task and mesh stages. Where a
driver does list them (to be seen on a discrete RDNA 3 or 4 card), `vk_caps.c` now pins them to 32 and tells the
translator 32 for "ts" and "ms" (`sg_pin_mesh`, `gfx_shader_mesh_subgroup`; the identity gains ` mesh_sg=32` only
there, so a device with 32-lane subgroups keeps its cache keys). Untested: no such card here.
Without the pin the translator runs its 64-lane path (Ryujinx's own, `InstEmitWarp.cs` `EmitBallot`,
`InstEmitMove.cs` `EmitLoadSubgroupLaneId`): a subgroup counts as two warps, the lane is the subgroup lane and 31,
a ballot is the word of the lane's half. What the task and mesh shaders use, from the SPIR-V of the cache (2,006 mesh and 80 task shaders): workgroups of
32 threads (8 task shaders of 1), the thread's number and the subgroup lane both, ballots with the lane's "lower
than" mask in 530 of them, shuffles in 216, a workgroup barrier in all.
Two experiments on that Radeon, each with a translator built aside and a shader cache of its own, neither kept:
- The lane taken from the thread's number, a table of each thread's host lane made at the shader's start by 32
  ballots, ballots rebuilt bit by bit from it and shuffles reading the source's host lane from it. The forest stayed
  near white: where the driver puts a workgroup's threads in the subgroup is not what breaks the picture, or not alone.
- A fragment shader's shuffle giving the lane its own value (wrong, but finite). The picture changed: silhouettes of
  the trees and of a villager came out of the white, the haze stayed. So fragment-stage shuffles go wrong on this driver.
  Hookshot found the same on AMD's Windows driver (its pull request 1507: "dark lighting and glowing bloom", fragment
  shuffles routed through broadcasts, and 64-lane graphics stages kept to their halves) and reports
  requiredSubgroupSizeStages 0xf0 from an RX 7800 XT: fragment, compute, task and mesh can be pinned there, the vertex
  stage cannot.
Next, not done: fragment-stage shuffles through quad broadcasts where the source is in the lane's own quad (the game's
derivatives), and a look at which other fragment shuffles the game makes.

A device loss with "read invalid" happened once more in about thirty runs, again with another program holding most of
the card. The fault report now says which pool buffer the address lies in, live or among the last 1,024 destroyed
(`vk_memory.c` `vk_mem_fault_report`).

A prebuild worker that never came back. At boot four threads build the compute pipelines of the warm list
(`vk_pipeline.c` `lb_worker`); in the player's package the list never ended in any session, while a fresh install and
the build folder finish it in 1 to 6 s. One below-normal thread ran at a full core for the whole session: the driver
does not finish building one compute variant of the play-recorded list (`b6e52782...` on this GPU and driver), which no
preparation had met, so no journal had listed it. `lb_watch` now adds a variant the driver has held for 30 s to
`shader-prepare-skip.txt`, and later starts leave it alone: the list then ends in 5.7 s there. With it the scene's
first frames in the package were 178 and 91 ms. `tools/package_oot.py` copies the skip list into the package.

NXFormat (fab918/nxformat, MIT) was looked at for speed: an archive format for decrypted dumps with independently
compressed blocks, whose own page says its read timings measure neither frame rate nor stutter. Here the RomFS is read
in place from the NUP (`romfs_nup.c`: raw extents by offset, a zstd-stored file inflated whole once per open); a session
logs no inflation over 4 MiB and fewer than 200 in all, so the container is not where frames go. Its reader could let
the setup accept `.nx` dumps.

Descriptors kept per bound slot (`REKARINA_SLOT_CACHE=1`, `vk_exec_draw.c`), timed at last in the camera spin, three runs
with it and two without, the same background load: resource binding 3.18 to 3.37 ms a frame with it, 3.21 and 3.26
without; 89.9, 94.6 and 94.0 fps against 94.6 and 93.0. About a third of the lookups are new each frame (22.3 million
taken again, 11.3 million made), and a hit saves less than the stamp check costs. It stays off.

One logical processor per physical core (`REKARINA_ONE_PER_CORE=1`, `perf.c` `cpu_placement`), two runs in the spin: the
replay 7.5 to 7.75 ms against 8.1 to 8.4 ms, the frame rate the same (91.2 and 94.3 fps against 93.0 to 94.6) and the
99th percentile worse (17.8 against 16.7 to 17.0 ms). It stays off. In the spin the GPU is busy 9.1 ms of a 10.3 ms
frame and the present waits for the frame's own GPU work (`REKARINA_PRESENT_SYNC` c, chosen against the brightness
flicker of c1), so the frame is the GPU's work plus the part of the CPU's that does not overlap it: 100 fps there
needs about 0.3 ms less of either, or the overlap c1 had without its flicker.

The player's own session at 10:00 to 10:08 (base picture, the fps overlay, `logs/totk.log`): three minutes of a movie at
its 60 fps cap, then minutes of 101.3, 90.7, 90.1, 88.3, 103.0, 95.9 and 114.1 fps, 1 % lows of 52 to 66 fps; two
frames over 100 ms (279 ms right after the movie, 110 ms at a later change of area) and five to eight frames a minute
between 25 and 45 ms, almost all of them replay. What is left of the scene's first frame (about 140 ms in the spin
bench) is now the driver's memory allocations under `vmaCreateBuffer` (new mirrors, new texture memory; 29 and 113 ms
in two profiled frames) and the resource binding around them. The background block allocation (`vk_headroom.c`,
`REKARINA_VRAM_HEADROOM=1`) was tried against it once more and made those frames worse (350 and 208 ms against 140): it
stays off.

## 2026-11-12: Hookshot's fixes for this game, read against this port

Hookshot (github.com/NotRexed/Hookshot, GPL-3.0) recompiles the same build of the game (main e0f3ccfa), so what its
pull requests found out about the game holds here too. Its code cannot come here (this project is MIT; AGENTS.md,
"Reusing code"): the descriptions were read for what the game does, each place was found again in our own
disassembly, and the changes below are our own. Its addresses are 0x80020000 above ours: its `f_80a87e50` is
`main+0xa67e50`. 300 merged pull requests were gone through by title; most change its own renderer, builder or
launcher and say nothing about the game. The ones that do:

| Hookshot | What the game does | Here |
|---|---|---|
| 2006, the pick-up counter above 60 fps | `main+0xa67e50` moves the shown count by `(int)(step * max(distance / 50, 1))`; a step under 1 gives 0 and the counter never fades | Patch set `pickup` (`recomp/patches-oot-v111.txt`): the amount is rounded up. Read in the disassembly at `main+0xa68300`; not seen in a run, no rupee was picked up in one |
| 2470, moving platforms above 60 fps | The physics solver's step count is `(int)(step * steps + 0.001)` in `main+0x38aeb80` and `main+0x38afc10`; truncated and changing from frame to frame above 60 fps | Patch set `solver`: rounded up while the step is under one frame, the game's own count otherwise. A minute of walking in the forest at 110 fps runs as before; the great tree's moving platform was not played |
| 2470, sound lost after several loading zones above 60 fps | The sound manager's job (`main+0x415300`) hands the step to the sound engine each frame; its fades expect whole 60 Hz steps, voices are not released and the voice budget fills | Patch set `soundjob` (2026-11-19, "Places that need a whole 60 Hz frame" below): the job runs at 60 Hz above 60 fps. The loss itself was not reproduced here |
| 2409, water ripples above 60 fps | The water simulation's pass count is `param * (int)ratio` | Already here: patch set `ripple`, found independently on 2026-11-09 |
| 1875, a movie cutscene's cues above 60 fps | The movie player counts game updates as the timeline's frames | Already here another way: the frame rate is held to 60 while a movie plays (`frame_rate.c` `movie_cap_hz`) |
| 1612, game speed above 60 fps | The game steps a whole 1/60 s per frame | Already here: patch set `realtime` |
| 2451, a crash a few frames after a load | The OptimizeChangeBank thread (`main+0x4366e0`) uses allocations from small per-model heaps without checking them; a full heap gives a write through null | Not done. Not seen here; the guard for a null resource entry (`resguard`) is a different place. A crash log naming `main+0x42cb30` or a write to address 8 is this |
| 1507, 2457, 2442, 1657: AMD and Intel shader translation | Fragment-stage shuffles, wave32 and wave64 stages, out-of-bounds fetches, storage buffer alignment | The facts are in "AMD, 2026-11-12" above. Our translator is a patched Ryujinx and differs; nothing was changed for them without a card to run |
| 2374, the tree-shadow mask blend | A device loss near the castle in its renderer | Its renderer's own; the Market and the field outside it ran here for minutes without one |
| 1921, 2357, 1945: camera centering off, field of view, camera speed | Game-side options | Not done; candidates for the menu |

The step these places read is the VFR manager's (`[[main+0x4c7e890]]+0x10`, in 60 Hz frames), the same value the
set `realtime` feeds. One more truncation of it is known and left alone: `main+0x32932b0` adds `(unsigned)step` to a
counter whose use nobody has found.

## 2026-11-12: the long frames of the Market and of the field, moved textures, and AMD again

The Market and the field, reached from the forest save by a map change (`REKARINA_WARP_AT`, `oot_warp.c`), the
camera turning in place, native 2560x1440, RTX 5070, the second minute of each run:

| | mean fps | 1 % low | frames over 25 ms | over 50 ms |
|---|---|---|---|---|
| The forest, before | 97.4 | 63.5 | 0 | 0 |
| The field, before | 91.9 | 41.7 | 54 | 35 |
| Market, before | 87.8 | 19.4 | 166 | 57 |

The long frames were 60 to 100 ms and almost all `replay` in the `frametime: worst frame` lines, with no shader
translation, pipeline build or texture upload in them.

**What it was.** In these two places the game's texture manager compacts its memory all the time: the thread
`aglTextureDataMgr::Compaction` (priority 53; the loop is `main+0x7bbf90`, and `main+0x7bbd60` for the copy to the
spare area) copies a texture's bytes to another address with memcpy, 9 to 14 MiB at a time, and back a few moves later:
about 2,000 copies of 2 ms or more in 95 s in the Market, none in the forest. Each copy's first write into a frozen
2 MiB region (`memory.c`, "frozen regions") faults. The access hook `result_fault` then asked `plat_mem_state` about
that one page, under `frz_lock`, and VirtualQuery measures the whole run of equal pages from the address on: 6 to
11 ms for a page in 600 MiB of read-only heap. The replay takes `frz_lock` in several places, so whichever section
happened to want it (`BIND_RES`, `MB_SNAP`, `SUB_EV`, `SHADER_MISS` in the `REKARINA_SECTIONS` rows) took 15 to 60 ms
instead of its usual 0.3 to 3.5, and the copying thread held its virtual core through the fault while the frame's
workers pinned to that core waited for it (`REKARINA_CORE_DIAG`: "core wait host" 40 to 110 ms in a long frame, the owner
`0x1044 p53`, guest lr `main+0x7bc684`).

**The fix.** `plat_mem_state` (`platform/win32.c`) asks QueryWorkingSetEx about the pages in question when the
question is 2 MiB or less: it answers for the pages it is given, with a resident page's protection, and costs what
was asked. `plat_mem_state_walk` does the same for the callers in `memory.c` that walk a range run by run. A page that
is not resident goes to VirtualQuery as before. `REKARINA_MEMSTATE_FAST=0` is the old path.

**Moved textures.** What was left in the field were frames of 25 to 33 ms with the GPU busy for 20 ms: every moved
texture was a new image, created and uploaded again (about 1.5 a frame; `sec:` lines "own memory new 92, repeating
92" per 60 frames). Now a texture made over bytes that a released or destroyed one was filled from takes that one's
image (`vk_texture.c`, `adopt_released`, `park_image`; `REKARINA_TEX_ADOPT=0` turns it off): the storage is hashed when
an entry with memory of its own is made, a released entry of the same descriptor and hash trades images with the new
one, and the images of destroyed entries wait in a cache by content, at most 512 images and 384 MiB
(`REKARINA_TEX_PARK_MB`) for 1,800 presents. In the Market nearly every new texture of the steady state finds an image
this way (24,576 against 13,120 that found none over a run of 130 s, most of those at the two loads; 36.8 GiB not
uploaded again). The hash costs about 0.4 ms of the replay a frame there.

A destroyed entry's record keeps its image handle when the image goes into the cache: the replay holds `&e->pub`
for the rest of the command buffer it is recording (`destroy_tex`), and with the handle cleared a draw recorded after
the destruction sampled a null image; the device was lost in the Market within two minutes, twice in two runs
(`VK_EXT_device_fault`: an invalid read at an address in no pool buffer).

A memcpy of a megabyte or more gives its virtual core up for its length (`nvn_api_core.c` `ov_memcpy`), so the
compaction's copies no longer hold the frame's workers.

After all three (the bounded page query, the adopted images, the copies without a core), the second minute, without
the section timers:

| | mean fps | 1 % low | frames over 25 ms |
|---|---|---|---|
| The forest | 98.8 | 63.5 | 0 |
| The field | 101.3 | 67.8 | 0 |
| Market | 110.8 | 65.6 | 0 |

With DLSS Quality and DLSS Frame Generation the game renders 79.6, 79.0 and 82.5 fps there (159, 158 and 165 on screen;
1 % lows of 58.0, 53.3 and 48.2 rendered; 0, 2 and 3 frames over 25 ms). Frame generation only generates in a
foreground window: with `REKARINA_BACKGROUND_WINDOW=1` the log says `dlssg: ... 1 presented per frame` and the numbers are
those of DLSS alone. The image cache keeps to 1/32 of the device's video memory budget and gives its images up above
94 % of it (`park_pressure`).

**Not done.** The first frame of a scene (130 to 190 ms: the first resolve of every binding, 328 mirrors made, the
scene's mesh shader lookups, all on the replay thread) and the two frames at a map change's arrival (90 to 130 ms)
are as they were. The slowest steady frames are now 14 to 18 ms: an upload batch's flush (2 to 5 ms), the memory of a
new render target (`vkAllocateMemory` on the replay thread, 1 to 5 ms) and waits for the texture registry's lock.

### Running through the field: the BC3 encode, and the texture cache

All of the above stands and turns the camera. Running (an input script that runs and strafes for a minute,
`REKARINA_WARP_AT` to get there) is another matter: the field 72.8 fps, 1 % low 36.0, 58 frames over 25 ms and 10 over
50 in the minute; the Market 79.8, 48.8 and 16. In the field's long frames the replay is short and the frame waits for
the GPU: one upload batch of 16 to 35 ms (`GPU_UPLOAD` in the `REKARINA_SECTIONS` rows, the `fence` wait beside it). That
is an ASTC texture newly streamed in: with `astc_bc` (OoT's default; RGBA8 instead does not fit 12 GiB,
tried again: 12,584 MiB in use of a budget of 10,774 and the device was lost) the GPU decodes it and encodes it to
BC3 in the batch of the frame that first draws it.

The blocks that encode makes are the same every time, so they are kept on disk (`vk_texture.c`, "the texture
cache"; settings `tex_cache`, off by default for the reason below, and `tex_cache_mb`, 6144): `tex-cache/<hash of the storage>-<format>-
<size>-<levels>x<layers>.bc3` beside the game. A texture whose storage has that hash is filled with one
buffer-to-image copy. A miss decodes as before and also copies the BC3 blocks to a host buffer
(`vk_astc_bc_readback`), which the reaper thread writes out once the GPU is past the batch (`vk_defer_call`,
`vk_memory.c`); the storage is hashed again after it was read, so a texture the game wrote meanwhile is not kept. The
folder is counted at start by a thread that deletes files over the limit, and nothing new is written once it is full.

| running, second minute | mean fps | 1 % low | frames over 25 ms |
|---|---|---|---|
| The field, before | 72.8 | 36.0 | 58 |
| The field, cache being filled | 82.7 | 53.3 | 7 |
| The field, cache filled | 84.5 | 55.6 | 5 |
| Market, cache filled | 82.1 to 84.2 | 49.4 | 14 to 19 |

The first run already gains: a run streams the same textures in more than once (3,227 textures filled from the
cache in the run that started with an empty folder). The forest, the field and the Market together put 4.3 GiB there.
What is left in the Market's long frames is CPU work of the replay: an upload batch's flush (15 ms in one frame) and
the staging copies of textures that are not ASTC (11 to 15 ms). Not done: filling the folder ahead of play (at
install, or while a loading screen shows), and evicting by age instead of by listing order.

**The device loss in the Market, and why the cache is off by default.** With DLSS Quality and DLSS Frame Generation,
running a scripted route through the Market (a foreground window, 150 to 185 s), the device was lost in 2 of 5 runs
that had the texture cache and the bounded page query both on, each about 100 s in: `vk: occlusion query slot N not
available after its fence`, then a device fault with an invalid read at an address in no pool buffer. It looked like
the cache, which went off by default; then a native run with the cache off lost the device the same way (3 of 21
two-minute Market runs in all, none of about 10 before the page query was bounded).

The cause is in `vk_texpool.c` and older than today: the texture pool's descriptor set is brought up to date within
a time budget (`REKARINA_TPOOL_MS`, 2 ms a submission), and a slot whose image had gone but whose turn did not come kept
its views. They are destroyed with the image once the GPU is past the submission, and the next one samples the slot.
The Market's compaction destroys and makes textures all the time and about 100 slots wait at any moment (the `sec:`
lines' "unresolved"); while each of the compaction's copies took 10 ms in a fault it was slow enough to hide this.
Now `update` gives every such slot the stand-in when it stops (8,192 slots in one two-minute run), and the slot
shows its texture again a few submissions later. Three native Market runs and the route with frame generation ran
clean after it. The texture cache stays off by default until it has been played with for longer than a morning.

The same route, rendered frames a second and frames over 25 ms in the last minute, before that fix:

| DLSS Quality + frame generation, running in the Market | fps | over 25 ms |
|---|---|---|
| everything of today off | 56.7, 70.1 | 454, 144 |
| the page query off, the rest on | 77.4, 76.7 | 113, 128 |
| as released: the texture cache off, the rest on | 73.4 to 81.0 | 3 to 33 |
| everything on, the image cache kept up to 99 % of the budget | 88.5, 84.2 | 8, 18 |

In that mode the card is at 92 to 97 % of its video memory budget (10.0 to 10.5 of 10.8 GiB), so the image cache's
retreat matters: at 94 % it was held back in half of the sweeps and the moved textures were filled again each time
(92 GiB from the disk cache in six minutes, 74 to 80 fps and up to 197 long frames a minute). `PARK_FULL` is 97 now
(`REKARINA_TEX_PARK_FULL` overrides it).

### AMD, continued

Four more things were ruled out for the forest's white picture on the integrated Radeon, each with its own shader
cache, none kept:
- Fragment-stage shuffles inside the lane's 2x2 quad through OpGroupNonUniformQuadBroadcast (the lane's place in the
  quad from the parity of gl_FragCoord), as a translator patch with a host option: the picture is the same white.
- On top of that, every other fragment-stage shuffle read through 32 ballots instead of OpGroupNonUniformShuffle: the
  same.
- Float atomic adds as compare-and-swap loops (`REKARINA_FLOAT_ATOMICS=0`), which is how the Radeon runs them, on the
  RTX 5070: the picture is right there, so that path is not it.
- No recorder thread (`REKARINA_VK_RECORDER=0`), as on AMD, on the RTX 5070: the picture is right.
What is left that differs on that device: 64-lane subgroups in the vertex, fragment, task and mesh stages (the last
two not pinnable there), and the driver itself. The next step needs a discrete Radeon whose driver pins all four, or
a way to see one pass's output on the Radeon (the pass probes, `REKARINA_PASS_PROBE`) to find the first pass that goes
white.

### AMD: the white forest was refused pipelines

A frame dump of the forest on the integrated Radeon and on the RTX 5070 (`REKARINA_DUMP=<frame>`, `vk_dump.c`; the
scratch `re/dumpcmp.py` aligns the two `images.txt` by target name) settled it in one comparison. The Radeon's frame
had 1,222 draws against 2,023: every draw through a vertex shader that reads storage buffers was missing, the depth
targets of the occlusion pass and of the ambient probe were empty, and the probe's 1x1 average came out ten times
brighter (nothing in it but the sky), which is the white. The log had said so all along, 600 times and in other
words: `vk_pipeline: graphics pipeline not built: the vertex shader uses binding 1, which its reflection and layout
lack`.

The translator declares a stage's first constant buffer (`vp_c0`, `fp_c0`: where the game keeps its storage buffers'
addresses) whenever the shader has storage buffers, and lists it in the reflection only when the code reads it. The
code reads it for the low bits of a storage buffer's offset where the device aligns such offsets (NVIDIA: 16 bytes).
The Radeon aligns to 4, so there the buffer is declared, unread and unlisted, and `module_binding_missing`
(`vk_pipeline.c`) refused every such pipeline. It now counts only variables the module's functions name: a layout
has to hold what a shader statically uses. After it the Radeon's frame has 2,004 draws and the forest is lit, textured
and shadowed as on NVIDIA, at 5 fps (an integrated GPU, 2560x1440 inside).

Still missing there: everything drawn by mesh shaders, which is the tree crowns, the grass and their shadows. The
draws are issued (the half-size depth pass of two mesh draws has 299,149 distinct depths on NVIDIA and one on the
Radeon; the shadow cascades hold a third of the casters). The lanes-by-thread-number patch for task and mesh shaders
was built and run again now that the picture can be judged, and changes nothing. Nor does translating them for 32
lanes without a pin (the workgroup's 32 threads taken to be the subgroup's first 32 lanes), nor building their
pipelines on the drawing thread (`REKARINA_MESH_ASYNC=0`). So the width of the subgroup is not what empties them, as the
notes above assumed. What is known to be right: the pipelines build, the draws are issued, the textures the mesh
stage reads (the half-size linear depth) hold the same values as on NVIDIA, each mesh shader calls
OpSetMeshOutputsEXT once before its first output store, and its limits (64 or 96 vertices, 32 or 126 primitives,
32 threads) are inside the device's.

The mesh stage is not where it goes wrong. The mesh draws are indirect, and `REKARINA_MESHCOUNT=<frame>` with
`REKARINA_MESH_BATCH=0` logs the count of each record as the GPU holds it when the draw's argument is copied. On NVIDIA
a frame's records read 22, 197, 24, 64, 119, 4, 848, 70, 6, 0, 298, 64, ... and a record used again later in the
frame holds its count again. On the Radeon the first seven read 22, 196, 25, 65, 868, 73, 119, and every record
after them reads 0, the same addresses included (65 at its first use, 0 at its second). So the first culling pass
of the frame counts right and the later ones (the lists of the shadow cascades, the probes and the second phase of
the main view) come out empty: the compute kernels that fill them, or what they test against. The occlusion pass
(256x144) fits: on NVIDIA its first draw is the occluders, with a program of its own and depth writes on; on the
Radeon all 21 of its draws go through the query draws' pipeline and the depth target stays empty.

Ruled out besides, each with a run: game memory mirrored instead of imported (`REKARINA_MEMORY=mirror`), storage buffer
offsets aligned to 16 as on NVIDIA (`REKARINA_VK_SSBO_ALIGN=16`), and, on the RTX, the gather bias the Radeon gets
(`REKARINA_GATHER_BIAS=8`: the foliage is there). Next: which dispatch writes a record that reads 0 (`REKARINA_DUMP_BUF` on
one record's address, both GPUs), and why the occlusion pass's first draw has another program there. A discrete
Radeon has not been run.

### Frames without lighting (the "shadow flicker" of the performance video)

In the Market one frame in 40 to 150 came out a third darker: the whole lighting result was black, with fog,
particles and HUD drawn over it. The flight recorder's flash watch counts them when its level is raised
(`REKARINA_FLIGHT=1 REKARINA_FLIGHT_FLASH_DARK=80`, percent of the running mean; the default 35 only catches black screens):
20 to 30 in 95 s of the camera spinning. A 30 fps clip shows one in three.

The cause was `REKARINA_TEX_DROP_AGE`, 30 frames since 2026-11-10 where its comment said 600. Above 75 % of the VRAM
budget (always, there) `sweep_pressure` in `vk_texture.c` destroys released render targets older than that. The game
lets go of some targets for longer and takes them back with their content; they came back empty. With 600 the count
is 0 to 1 (what remains right after the warp is the loading screen).

Checked and ruled out on the way, each with runs: the early results signal, draws skipped while a pipeline builds,
torn uploads, large copies giving up their virtual core, lazy write-back, a present that finishes the queue.

Found beside it and fixed:

- Late sums (`vk_memory.c vk_mem_late_apply`): the range the auto exposure's shader adds into also holds about 1,800
  words it simply stores. Applied a frame late as sums they became g(n+1) = g(n) - g(n-1) + v, which swings around v
  with a period of six frames and never decays (0.34 read as -2.2 to 2.9). A word that still holds what the function
  wrote last is now taken as stored (`REKARINA_LATE_STORES=0`: as before).
- A write to the first or last page of a texture's storage, which it shares with its neighbours, counted as a write to
  the texture. The game's texture compaction moves blocks of 10 to 13 MiB every few frames in the Market, and an
  8-layer R16 array of 32 MiB beside them was uploaded again every 18 frames, 3 ms each time. The texture's own bytes
  in those two pages are now compared (`edge_only`; `REKARINA_TEX_EDGE=0`: as before).
- Early results in import mode (`vk_sync.c vk_sync_results_on`). With imported pools the GPU reads the game's own
  memory; told at the results point that its frame was done, the game cleared the count records of its indirect mesh
  draws for the frame after next while the GPU still drew this one. The first view's eleven draws read their counts,
  every later one read 0: no foliage shadows on any integrated GPU, and the same on an RTX 5070 with
  `REKARINA_MEMORY=import`. Early results are now off without mirrors.

### The recorder thread is off

Two-minute runs in the Market, the same build, the two settings in turn, an RTX 5070 at 97 % of its VRAM budget
(other programs held 3 GiB): with the recorder thread 2 of 6 ended in a device fault (a read at an address in no pool
buffer) or in the Vulkan loader's abort on a NULL command buffer, each after "occlusion query slot not available after
its fence"; without it 0 of 6. Without it the frames are also steadier: 1 % low 53 to 56 fps against 47 to 50, 0 to 1
frames over 25 ms in the second minute against 3 to 13, mean 87.9 fps against 90.0. `vk_recorder` is 0 by default on
every GPU now (it already was on AMD). What its packets meet is not found: `REKARINA_REC_CHECK` reports at a clean exit
only.

A device fault now also names what lay at its address: with VK_EXT_device_address_binding_report the driver reports
every image's and buffer's GPU address range (`vk_device.c addr_fault_report`, `REKARINA_VK_ADDR_REPORT=0` turns it off).

Open: the hero's long shadow in the field still changes from frame to frame at some camera angles; on the integrated
Radeon the mesh shaders draw nothing even with the right counts (`REKARINA_MEMORY=mirror`).

## 2026-11-12, evening: the frame's draws on several threads

The replay of a frame was one thread's work: 11.7 ms in the forest at 2560x1440 (base picture), against 9.4 ms of
GPU time, so the processor decided the frame. The recorder thread, which only moved the driver calls, is off
("The recorder thread is off"). This is the other cut: the draws themselves are recorded by several threads.

What the frame looks like (one forest frame, dumped with a diagnostic that is gone again: 30,415 ops, 2,152 draws,
1,550 render conditions): about 1,550 draws are single draws between a samples-passed reset and report, in two runs, the depth
prepass and the lit pass, the lit ones under a render condition on the prepass count. The rest are the shadow cascades
(runs of 20 to 160 draws) and about 140 small passes of one to ten draws. A whole-frame split into halves was looked at
and left: the waits for CPU events (the uniform snapshot and the mirrors are made again after them), the compute
queue's rests recorded inside the frame, the results point and the feedback layouts all depend on order.

A stretch of ops that write no memory, inside one pass, does not: binds, state, draws, the counters' resets and
reports, render conditions. `vk_exec.c par_try` cuts such a stretch into chunks (the comment there has the whole
contract). The replay thread walks the stretch once and does what must stay in order (the reports' bookkeeping, the
conditions' decisions, the query slots), and each chunk is replayed by another thread into a command buffer of its own,
chained into the part being recorded (`vk_sync_chain`). The replay thread does not wait at the stretch's end; it waits
where the part is submitted (`vk_exec_par_join`), so uploads and mirror copies precede the part as before, and before
the few ops that change what the chunks read (`par_join_op`).

What had to change for it:
- the draw path's memos and per-command-buffer state are per thread (`vk_exec_draw.c`: B, DS, RS, the shader, pipeline
  and texture descriptor memos, the answers and stage tables);
- the texture lock is taken by a lookup, not by every draw with a texture binding (`td_lock`): with it per draw the
  chunk threads queued behind one another, and the time inside `bind_resources` doubled;
- shared counters written per draw are left alone by the chunk threads (`vk_mem_count`, the snapshot's hit count,
  the frame's draw count), and `REKARINA_SECTIONS` times the replay thread only;
- a mirror's creation and first sync of the submission run under a lock (`vk_mem_use`);
- the submission's update of the texture pool's set is made before the chunks go out (`vk_tpool_prepare`): its first
  bind made it, on whichever thread, and two could run it at once;
- the samples-passed segments go to the completion thread in batches (`oq_batch`), not one callback each;
- each chunk thread has its own event: woken through one, they started one after another, 0.2 to 0.3 ms late.

Measured, forest, 60 s of the bench path, six threads on a 16-thread processor:

| | fps | 1 % low | frames over 25 ms | replay thread per frame |
|---|---|---|---|---|
| one thread | 76.5 | 46.0 | 8 | 11.7 ms |
| chunks, a wait at each stretch's end | 82.3 | 51.3 | 4 | 9.7 ms |
| chunks chained into the part | 95.9 | 65.6 | 2 | 7.9 ms |
| the same, chunks of about 64 draws (the default) | 96.8 | 64.5 | 5 | 8.3 ms |

The field, the same bench: one thread 93.4 fps, 1 % low 61.5, replay thread 9.7 ms; chunks of 12 draws 87.2 fps,
45.5, 7.6 ms; chunks of about 64 draws 96.1 fps, 60.6, 7.5 ms. Each chunk begins the pass again behind a fragment
barrier, where the GPU's fragment work of the chunk before must be done: with small chunks the GPU was busy 9.4 ms a
frame against 9.0 and the frame rate fell although the replay was 2 ms shorter (`REKARINA_PAR_PER`, draws per chunk).
Screenshots of the forest and the field show the whole scene, and the flash watch found no dark frame in the field.
The Market ran two minutes with it and without, no fault; both runs fell to 16 fps in their second minute with
the GPU busy 31 ms a frame (2.5 GiB of the card held by other programs: the video memory item, not this change).

The frame is now bounded by the GPU there (9.1 to 9.2 ms busy of 10.4). `replay_threads` in rekarina.ini sets the count (0 or 1: the replay thread alone, the
default; see the stability series below). Not cut: stretches under 12 draws, passes without an attachment or with a
target in the feedback layout, indirect mesh draws, and everything while DLSS or per-object motion is on.

Later the same evening: the chunk threads read the uniform snapshot through a view of their own
(`vk_usnap_par_view`), so a held compute rest going on does not wait for them (forest 8.1 ms, field 7.2 ms on the replay
thread, 96.8 and 97.2 fps).

A line profile of the replay thread then showed 13 % of its time in one loop: the wait for chunks at a part's split,
about seven waits of 0.1 ms a frame. The part now goes on past its usual split while its chunks are being recorded (up
to four times its draws) and is submitted the moment they are done. The first try submitted it between passes only,
and the field, whose long passes of indirect mesh draws are not cut into chunks, lost its smoothness (88.8 fps, 60
frames over 25 ms): where the GPU decides the frame, a part that reaches it later is a later frame. Submitted inside a
pass too (the pass begins again in the next part), the field runs at 100.4 fps with a 1 % low of 65.6 and 6.7 ms on the
replay thread (one thread: 93.4, 61.5, 9.7 ms), the forest at 95.5 fps and 7.8 ms.

### The Market's device fault: the game went on while the GPU still read its memory

A stability series first sent the threads back to opt-in: the forest three two-minute runs without a fault, Castle
Town Market two of three lost to the device fault the Market has had since before (a shader read at an address whose
last images were destroyed 12 to 45 s earlier), none of three with one thread. Chunk threads raised the fault's rate as
the recorder thread had, which made the Market a place that reproduces it in every second run, and that found it.

What named it, in order:
- GPU breadcrumbs per chunk (`REKARINA_PAR_CRUMBS=1`, with a ring that says what each stretch was): the GPU had finished
  everything before one chunk of the lit pass (2560x1440, colour 122, depth 126) and reached the next. The same pass
  as in the recorder's faults.
- By reading, the texture descriptor memo's tokens, the view cache and the deferral stamps hold. A switch series said
  otherwise for a moment (three clean runs with `REKARINA_TEX_MEMO=0`), so every memo hit was checked against the views
  already handed to deferred destruction: none in a faulting run. The clean runs were chance.
- The lit pass is the first pass after the results point (`REKARINA_RESULTS_FENCE=2`: the game's wait returns there).
  With `REKARINA_RESULTS_FENCE=0` four of four runs were clean.

The cause: after the results point the game goes on and rewrites next frame's blocks and pointers while the lit pass
is still being recorded and run. Buffers in a mirror hold the bytes of their sync; buffers bound in host memory are
read by the GPU where the game writes. In the Market, where video memory is short, mirrors are refused for the budget:
1,559 buffer bindings a frame stayed in host memory over 64 pools (the forest: none). A shader then loads through a
pointer or an index the game is rewriting and reads at an address nothing is mapped at. Whatever makes the lit pass
reach the GPU later (the recorder thread, chunk threads) widens the window.

The fix (`vk_memory.c vk_mem_host_read_note`, `vk_exec.c vk_exec_results_point`): the results point is left out while
any buffer binding of the last 120 frames stayed in host memory, and the game waits for the whole frame, as the
console's does. Market, six threads: four of four runs clean, 82 fps (96 with the early point when it holds); the
forest keeps the point (8,192 asked, none left out). Pools reached only through the global memory table are read in
place everywhere and do not count: counting them left the point out always (forest 63 fps). `replay_threads` follows
the processor again.

That rule alone cost frames wherever video memory was short, and not only in the Market: with 2.5 GiB of the 12 GiB
card held by other programs, the field had 120 pools of 58 MiB in all refused a mirror, lost the early point for
them and ran at 61 fps instead of 96. A refusal also stood for the pool's life. Now a pool of 32 MiB or less always
gets its mirror (`mirror_create`: the budget test is for larger pools) and a refused large pool is asked again 300
frames later (`vk_mem_use`). With that no pool stays imported in the forest, the field or the Market, the point is
taken on every frame, and the hazard is gone where it came from: field 90.7 fps, Market 92.6, forest 83.7 on that
evening's machine, the Market four runs of four without a fault.

Where the replay thread's time still goes (per frame): the walks of ~23 stretches (0.07 ms each), ~12 waits for chunks
at splits, uniform updates and held rests (0.07 to 0.13 ms each), the dispatches (1.3 ms), the small passes, and the
submission's own start (uniform snapshot 0.3 ms, mirror snapshot 0.4 ms).

## 2026-11-13: dark frames while running in the Market

The player's report: shadows flicker, characters flicker. Measured: while the hero runs, single frames where a soft-edged
part of the picture and the characters in it go dark (mean brightness 89, 63, 90 over three frames), 5 to 17 times in
2,000 frames, often several within a second. A standing camera has none, the field has none.

The cause is the clear a new depth image starts with (`vk_texture.c init_image`). It goes into the upload batch, and
an upload batch is submitted ahead of the part being recorded (`vk_sync_split`), so it runs before every command of
that part. For an image bound into pool memory the clear writes the pool's bytes. The game's transient render targets
reuse their storage within a frame, and where video memory is short every one of them lies in pool memory
(`REKARINA_POOL_TWIN`, the Market). A character's 384x384 shadow map gets a new address whenever the frame's set of
targets changes, which while running is every few frames; the new image's clear then zeroed storage that a target
drawn earlier in the game's order still held and that the same part read. A depth image bound into pool memory is no
longer cleared (colour images bound there never were); `REKARINA_DEPTH_NEW_CLEAR=1` clears as before. In one build: 0 and
0 dark frames in 2,200 with the change, 20 without. Forest 93.6, field 101.5, Market 100.1 fps, as before.

How it was found, since most of a day went to wrong leads:

- The first measurement was wrong. Brightness spikes in cells of a 30 fps clip of a camera spin count thin objects
  passing; every setting "flickered" alike. A clip recorded frame by frame (`REKARINA_VIDEO_FPS=120` under a 100 fps
  limit) and the mean of the whole picture and of its halves against both neighbours is the measure that held, with
  The hero running a square by script. Flower pots crossing the camera and the game's own camera cut in a narrow passage
  are its false alarms; they have to be looked at.
- Ruled out by runs that still flashed: one replay thread, no results point, a later results point, an unlimited pool
  update, copies made again at waits already met, conditions on the CPU. The first run without a results point had no
  flash by chance; two repeats had five each. One run is not a result for an event of this rate either way.
- The flight recorder's counters named the frames: a flash frame is one where the game submits more draws than its
  neighbours. The pass log of 450 frames (`REKARINA_DUMP_LOG`, text only) beside the recorder's thumbnails then showed that
  the flashes fall on frames where the number of 384x384 shadow maps changes, and of those on the ones where a map is
  rendered at an address or slot new in that frame: a new image, so its creation was the thing to look at.

Seen on the way and open:

- While running, the game gives its transient targets new addresses almost every frame. A square depth target whose
  size follows the camera (929 to 1,579 texels) gets a new image each frame and about 90 of them stay alive; the
  scene-size targets are made again each frame; about ten compressed textures a frame are taken back by their content.
  This is a share of the frame time lost while running.
- Every eighth frame the game renders a round of 128x128 passes (about 750 draws more), and in some frames it submits
  the scene's objects without its own occlusion culling (twice the draws of the depth pass and the lit pass).
- Objects drawn flat white for one frame in some runs: the pool set's stand-in. Not traced.


## 2026-11-13, night: the chunk threads shared a scratch array, and what the check modes found

The field at a run (the hero runs squares for two minutes, `REKARINA_WARP_AT=25:field` with a stick script) lost the
device or died inside the driver in about one run of four: `VK_ERROR_DEVICE_LOST` with an "instruction pointer" fault
and a "read invalid" at an address no live object holds, or an access violation in `nvoglv64.dll`. The Market, run the
same way, did not. The night's counts are in the table at the end.

What the fault report could and could not say:

- `VK_EXT_device_address_binding_report` lists the objects that lay at the fault address. For this fault the list
  was not usable: the driver reports the same 15.8 MiB range for small images bound one after the other (a 512x512
  compressed texture, a 967x967 depth target), so a dozen unrelated images "held" the address.
- A post-mortem added to the report (`DEBUG-WHONAMES`: `vk_tpool_who_names`, `vk_exec_draw_who_names`,
  `vk_exec_who_names`, `vk_texture_who_names`) asks every table that can name an image whether it still names one of
  those: the pool set's slots, the view cache, each thread's descriptor memo, the replay's targets, the texture
  entries, the kept images and the slots. In the crash it caught, none did, except memo entries whose tokens no longer
  matched (stale and never used again).
- Every crashed run has one line "occlusion query slot N not available after its fence" right before the fault and no
  clean run has it. It is the lost device seen from the completion thread, not a cause.

What found the bugs was the check modes, run together for the first time since the frame's draws went onto several
threads (2026-11-12): `REKARINA_TEX_MEMO=check REKARINA_BIND_SKIP=check REKARINA_SHADER_MEMO=check`. They stopped within seconds,
three times, on three different things:

1. **"bound slots changed without a bind op"** (`vk_exec_draw.c bind_sig`). The whole-draw skip keeps the bound-slot
   bytes of the last draw (`RS.body`) and compares them only when a bind op ran on this thread since (`bind_gen`). A
   chunk of a draw stretch starts from a copy of the walk's state (`vk_exec.c par_run`: `*ex = j->st`), so the slots
   differ from what the thread last compared, and no bind op of its own says so. The chunk's first draw was built
   anyway (a new command buffer clears `B.res_valid`), but it left `RS.body` stale while marking it current; a later
   draw of the chunk whose slots equalled the stale bytes was taken for a repeat and drew with the first draw's sets.
   Fix: `vk_exec_draw_new_cb` bumps `bind_gen`.
2. **"a draw with the inputs of the previous one gets descriptor commands"**, uniform slots 3 and 4 at another
   offset of the snapshot arena. The offsets are a function of the bound address, so the address had changed and the
   comparison had not seen it: the pipeline's list of ranges to compare (`rs_list`, built once per pipeline and
   thread and kept until shutdown) was wrong. It is built in a scratch array that was `static` in the function, one
   array for every thread. Two chunks meeting two new pipelines at the same moment wrote their ranges over each
   other. Depending on the interleaving the list then names another pipeline's slots (draws taken for repeats of the
   draw before them: an object drawn with the previous object's textures and uniforms, for as long as that thread
   records that pipeline), or its ranges add up to more than `RS.body` holds, since the size check ran on the array
   before the other thread changed it: `bind_sig` then copies past the end of `RS.body` into the thread-local state
   behind it. Fix: the scratch array is `_Thread_local`.
3. **"texture descriptor memo differs from a lookup"**, and once more the skip check with one texture view changed.
   Both times a game thread had registered the slot again between the cache's test and the check's own lookup
   (the token moved). That is the order the design allows (the slot's new texture counts as bound after this draw,
   and what it replaced lives until the submission being recorded is done), so the memo check now reads the token
   again and only reports when it is unchanged. The skip check has no such test yet and can still stop on this race.

The second one explains more than the crash. A pipeline's list is made once, so a list spoiled at the pipeline's
first draw stays spoiled on that thread for the whole session; which objects show another object's textures depends
on which thread recorded their chunk in that frame. This is the likely source of objects that flash another colour
for single frames ("the ladder turns turquoise"), which the frame-by-frame clips could not tie to a pass.

A third fix came out of reading the crash stack (`exec_op_` -> `par_try` -> `vk_sync_chain`):
`vk_exec_inline_begin` handed the replay thread's own command buffer to code that runs inside a chunk when the replay
thread records the chunk itself (`vk_exec_par_join` takes what no thread took). `vk_texture.c alias_copy` then ended
and began a pass that was open only in the bookkeeping and recorded its copy behind every chunk of the part. Inside a
chunk the commands now go into the chunk's own command buffer between two halves of its pass.

Rules that follow:

- A function-level `static` buffer in anything a draw calls is shared by every chunk thread. `grep -n "^ \+static"`
  over `vk_exec_draw.c` and `vk_pipeline.c` after any change to what the chunks run.
- State that says "nothing changed since I last looked" must be reset where a thread's view of the replay state is
  replaced, not only where it is edited.
- The check modes are a test to run after every change to the replay's threading, before any stability loop. They
  took a minute where the stability loops took the night.

Two-minute runs of the field at a run that ended in a lost device or a driver fault (2026-11-13, RTX 5070, 1440p):

| Build | Crashed | Runs |
|---|---|---|
| df41ec5 and 740f03b (no texture cache, no moved shortcut) | 0 | 8 |
| 6d3f0a7, cache and moved shortcut both off by knob | 0 | 6 |
| 6d3f0a7 as it is, or with one of the two off, or with debug checks | 8 | 29 |
| the same with the uploads ahead (below) | 2 | 3 |
| with the three fixes above (16dac6d) | 0 | 8 |
| the same with the uploads ahead, hardened (below) | 1 | 6 |

Not explained: why the builds without the texture cache and the moved shortcut had no crash in 14 runs when the
faults fixed here are in code they share. Either those features change the timing the race needs, or there is a
second cause the eight clean runs have not met yet. The run that crashed with the uploads ahead on says the second:
a fault with the same signature is still in the build. The next instrument is a breadcrumb per draw
(`VK_AMD_buffer_marker`, as the per-part ones) with a ring of what each draw bound, so that a fault names the draw
and the binding that read the dead image.

### Arriving on a map: what the first three seconds are made of

Measured again on the fixed build (`REKARINA_SECTIONS`, the field after a warp): the loading screen is about 345 frames
at 4 to 5 ms (the 240 fps cap), the arrival frame takes 70 to 80 ms, then 180 frames run at about 17 ms (replay 14.2 ms
against 8.6 in steady play), a second frame of 50 to 150 ms follows, and then the place runs at 11.5 ms. Of the 5.6 ms
the replay has too much in those 180 frames: the pool set's update 2.7 ms (slots registered again 2.2, new ones 0.9),
texture uploads 1.8 to 2.1 ms, descriptor builds 1.0 ms, the global-memory table 0.5 ms. The game registers about
118 slots a frame there (80 to 100 in steady play) and 17 new pool textures a frame are made (6 to 7 in steady play).

The texture cache made this stretch slower, not faster: its file reads run on the replay thread (2 ms a frame with
the files in the system's file cache, 6.9 ms a frame in the first run after a restart of the machine, where the GPU's
own decode cost the replay thread 0.5 ms). At 740f03b the stretch ran at 15.5 ms a frame, with the cache on at 17.1.

An earlier figure of mine for this stretch compared runs by "the last frame over 60 ms": that picks the first spike
in some runs and the second in others, and the two populations looked like a fast and a slow mode of the game.

### Uploads ahead of first use (`REKARINA_TEX_PREP`)

A thread of its own takes each newly registered pool slot, resolves it as the pool update would and uploads it, with
the texture lock released while the file is read or the bytes are copied out of the game's memory (`vk_texture.c
up_main`, `fill_enter`, `fill_leave`). In the 180 frames after an arrival the replay thread's uploads fall from 2.0 to
0.36 ms a frame and the pool update from 3.4 to 1.1 ms; the frame goes from 17.1 to 16.2 ms. It gains less than
the two figures add up to, because the thread holds the texture lock while it makes each image, and every draw's
texture lookup on the replay thread waits for it. Making the images outside the lock is the next step there.

It is off by default. After the restart of the machine (see the trap below) the same stretch measured 18.4 ms
without it and 17.4 ms with it, running through the field was the same (90.5 fps and a 1 % low of 57.4 without,
89.3 and 58.5 with), and one of six two-minute runs with it lost the device with the old signature: a read at an
address whose last image was destroyed ten seconds before. Eight runs without it on the same build were clean. So
the fault of the table above is still there, the thread makes it more likely, and its cause is not found.
`REKARINA_TEX_PREP=1` turns the thread on.

Three things the first version got wrong, each a device fault or a wrong read:

- The staging buffer it owns was handed to the deferred release before the upload batch that copies from it was
  open, so the release was stamped with the batch before and the buffer went while the copy still read it.
- After the fill it read the entry it had been filling. The entry may have been destroyed meanwhile, and its record
  is freed once the GPU is past the submission that was being recorded, which a file read can outlast. `fill_leave`
  now asks the table whether the key still names that record, with that image, before anything reads it.
- During the fill it read the texture's descriptor and storage address through the entry. It reads a copy now, and
  leaves every ASTC decode to the replay thread (the CPU decoder reads the entry's format while it runs).

### A trap in tonight's numbers

After a restart of the machine in the morning the same build ran the field at 90 fps (1 % low 57) where it had run
99 (61) during the night with nothing else open. A pair of numbers from the two sides of the restart compares the
machine's state, not the builds.

### The morning's control runs: what held and what did not

All at 2560x1440 on the RTX 5070, after the machine's restart, the hero at a run for two minutes, the second minute
counted (`re/runstat.py` in the session scratch; mean fps / 1 % low / frames over 25 ms a minute).

- **The texture cache's gain did not repeat.** The field, the night's texture changes off against on: 89.8 / 56.0 /
  3.5 (2 runs) against 90.6 / 57.5 / 3.3 (14 runs). At night, before the restart, the build of the start of the night
  against the build with the changes had been 97.0 / 55.8 / 14.5 (4 runs) against 98.7 / 61.1 / 3.1 (7 runs). Both
  baselines decode about 72,000 ASTC regions on the GPU in the run; at night that made 12 to 43 long frames a minute,
  in the morning 2 to 5. The replay thread took 7.0 ms a frame at night and 7.9 in the morning (Steam and a browser
  were open): with the CPU as the limit the GPU has room for the encodes, with the GPU as the limit they lengthen
  frames. So the cache helps where the GPU decides the frame and not otherwise.
- **The Market gains nothing from it**: 82.7 / 47.4 / 11.5 off (2 runs) against 83.3 / 48.4 / 15.7 on (3 runs, long
  frames 11, 23 and 13). The game's compaction there moves textures all the time, most lose their image before the
  entry at the new address is made, and each is read again from its file: 7,680 textures and 6 GiB in two minutes,
  on the replay thread. The figure in the commit that turned the cache on (1 % low 47 to 57) was one run each way.
- **Files not in the system's file cache made it worse than no cache**: the first Market run after the restart had
  76.3 / 41.2 / 38, with 9.4 ms of file reading in an average long frame, and the 180 frames after an arrival in the
  field took 22.3 ms each (20.4 with the files purged by hand, 18.1 with them cached). The cache's thread now reads
  the folder once at start (`tex_cache_warm_one`, `REKARINA_TEX_CACHE_WARM_MB`, default 8192): 6.0 GiB in 6.0 s from the
  disk, 1.3 s when cached. With the files purged and the warm pass on, the Market run gave 84.2 / 51.3 / 13. To purge:
  open each file once with `FILE_FLAG_NO_BUFFERING` (`re/coldcache.py`).
- **The cache's size limit was 6144 MiB, not the 14336 the table says**: `cfg_get` returns no table default and the
  code's fallback was the old number. The scratch folder had reached it, so nothing new was being written. Fixed.
- **Six gigabytes** (`REKARINA_VRAM_HOG_MB=5600` holds 5,376 MiB of the 12 GiB card): the automatic texture quality does
  not see the hog (it subtracts this process's own memory from the system's count), so `REKARINA_TEX_QUALITY=quarter`,
  what a real 6 GiB card gets, has to be set by hand. The forest standing: 71.2 fps, 1 % low 54.1 (88.3 / 58.8 on the
  whole card). With full-size textures the Market run lost the device as the Market loaded (a write at address 0,
  11.8 GiB in use of a 10.8 GiB budget).
- **Quarter texture quality hung at the Market's entrance**, with the hog and without it, at 2560x1440 and at
  1920x1080, with the night's texture changes on and off: the frames stopped a few hundred frames after the warp and
  the process sat until its timeout, without a fatal line. Found and fixed the same morning, in three steps:
  - There was no way to see a hang (no debugger on the machine). The profiler got a watch: `REKARINA_PROFILE_HANG=<s>`
    with `REKARINA_PROFILE` and `REKARINA_PROFILE_CHAINS=1` samples nothing until no frame was presented for `<s>` seconds,
    then samples every thread for 3 s, writes the profile and ends the process (`prof_win.c hang_main`).
    `tools/prof_report.py` on it showed the replay thread at one instruction inside the driver, in
    `vkCreateImageView` called from the pool update (`vk_texpool.c resolve_slot` -> `vk_exec_pool_view`), and a game
    thread waiting for the texture lock behind it.
  - The watch also logs the sampled view being created (`vk_exec_draw.c vk_exec_last_view_log`): a 256x256 image with
    9 levels, `gone 1`. The view was being made for an entry that had been destroyed.
  - A slot keeps its resolved entry while the generation of the slot's own storage address stands
    (`vk_texture_by_id_locked`), and its token, which the descriptor memos test, is built from the same generation.
    Under "texture quality" the slot resolves to a smaller texture whose storage starts at a later level: another
    address, another generation. When that entry went, nothing told the slot. The entry now carries the storage
    addresses of the textures it stands for (`Tex.texq_base`, up to three: a texture and its own second level
    registered as a texture reach the same smaller entry) and `tex_gen_bump_entry` moves their generations too. One
    address was not enough: the run got twice as far and hung the same way. A slot that still finds a destroyed
    entry resolves again and says so in the log; that line did not appear.
  With the fix the Market run at quarter ends: 74.2 fps, 1 % low 45.5. Half quality has the same hole (one level
  dropped, another address); it ran the Market before the fix by luck of what was destroyed when.
  With the hog as well (the 6 GiB emulation) the Market run at 2560x1440 ends too: 75.1 fps, 1 % low 43.0.
- **The field's device fault also shows under texture quality**: after that fix, the Market at 1920x1080 with the hog
  at quarter and the field at half each lost the device in their second minute, with the old signature (two of four
  runs in these modes; the field at full quality 1 of 15). Whether the smaller textures make it more likely or four
  runs say nothing is open.
- **Where the fault is.** Two captured runs (`runb.sh <tag> field REKARINA_TEX_QUALITY=half REKARINA_TEX_PREP=1
  REKARINA_PASS_CRUMBS=1 REKARINA_PAR_CRUMBS=1` meets it in about one run of three). The first showed the pass marks at "the
  end of pass 72, no draws", which I first read as "between passes". That reading was wrong: the mark's end comes
  before the chunks of the pass's draw stretch, which are command buffers of their own. The second run had the chunk
  crumbs working: the GPU had reached the start of chunk 2 of a stretch of 287 draws in the main scene pass
  (2560x1440, colour format 122, depth format 126), and everything before the start of chunk 1 had finished. So the
  read is in a draw of chunk 1 of the lit pass: a draw recorded by a chunk thread (or by the replay thread taking
  a chunk), with the fault's "instruction pointer" kind. Marks around `alias_copy` and the mesh draw count copies
  (`DEBUG-COPYCRUMB`) were not reached, which clears those two.
  What a chunk's draws read that could outlive its owner is the list to go through next: buffers resolved through
  the per-thread `vk_mem_resolve` cache, the uniform snapshot's view for chunks (`vk_usnap_par_view`), the pool
  table infos prepared per stretch (`gmem_par`), indirect draw records, and the per-thread resources of a command
  buffer (`cb_res`). A breadcrumb per draw inside a chunk, with a ring of what each draw bound, would name the
  draw. The fault does not need the chunk threads: with `REKARINA_REPLAY_THREADS=1` and the same recipe one run of
  four lost the device the same way, so the chunk crumb only says which draws (the main pass's), not that a second
  thread is to blame.
  The newest lead: with a breadcrumb before every draw (`REKARINA_DRAW_CRUMBS=1`, `vk_exec_draw.c draw_crumb`, a ring
  of what each draw had bound for the report) seven runs of the recipe were clean, where four of nine had died
  without it. A crumb is two marker writes, one at the bottom of the pipe: the GPU finishes what came before each
  draw. So the fault needs draws to overlap on the GPU: a draw reads something a neighbour has not finished
  writing, or reads it in a stage the barriers between them do not order. The barriers inside a pass order fragment
  work only (`vk_exec.c par_fragment_barrier`, `relaxed_pass_barrier`: destination stages are the fragment ones),
  so a later draw's vertex-side reads (index and vertex fetch, an indirect record, storage a mesh or vertex shader
  reads) are not ordered after an earlier draw's shader writes. An indirect record or an index read half-written
  would send the fetch outside its buffer, which is what "read invalid at an address nothing has held for seconds"
  looks like.
  Tested and not it: with `REKARINA_PASS_BARRIER=full` and `REKARINA_PAR_BARRIER=full` (every stage of a pass or chunk waits
  for the one before) the recipe's first run lost the device, again in chunk 1 of the main pass's stretch (250
  draws). All three captures with chunk crumbs name chunk 1. So what must overlap is draws inside one pass, which no
  barrier of ours separates and the per-draw crumbs do. What a draw of that pass can write that a later draw
  reads: memory through the pool table (the log has "global memory writes reached guest range" lines in these runs),
  where a shader loads an address from memory and reads at that address plus the entry's delta. A pointer read while
  a neighbour writes it, or one that points into a pool the table no longer has, is the next thing to look at:
  which of the main pass's shaders load through the table (`res.json` `sbufs` with `cb_slot` 255), and what the
  translated lookup does on a miss.
  Tried and not it: `vk_pool_fini` destroys a pool's record and buffers at once unless the pool has a mirror, and the
  game has its "the GPU is done" early (the results point), so a loader thread can finalize a pool while the rest of
  the submission is recorded. With every record kept until the submission being recorded completes (as the mirrored
  ones are), the recipe's first run still died in the driver (`nvoglv64.dll+0xfa0dfe`, the same address as on
  2026-11-13 03:00). The change is not in the tree; the hole it closes is real and worth closing with a test of its
  own.
  A warning for reading these reports: the labels in the list of objects at the fault address are looked up by
  handle when the report is written, and the driver hands a destroyed image's handle to a later image. An old event
  therefore shows the label of the handle's owner today (the square shadow target, whose images come and go every
  frame, turned up in nearly every report for this reason), and "who still names image" finds that owner: a live
  entry and a pool slot that name the handle prove nothing about the image that lay at the address.
- **The Linux and Proton memory path** (`REKARINA_SHADOW=1` on Windows): the menu and the first map run, but the load into
  the field did not finish in two minutes: 172 frames of 0.3 to 1.1 s, almost all of it waiting for the GPU
  (`fence_us` 494 ms and the upload flush 88 ms in an average slow frame). Not looked into; nobody had run this mode
  on OoT before. It is the first thing to fix for Linux.
- **The device fault of the field is still there.** With the three thread fixes: 14 of 14 runs clean, then one of one
  with the cache's files purged lost the device, and one of six with the uploads ahead. That last report placed the
  fault address in an imported pool's buffer destroyed 67.7 s before (`vk_mem_fault_report`'s list of the last 1,024
  destroyed; earlier faults were "in no pool buffer", which may only mean older than the list). OoT's shaders
  reach pool buffers by device address through the pool table (`vk_mem_gmem_table`), and `vk_pool_fini` destroys a
  pool at once unless `r->used` says a draw bound it, which a reach through the table does not set. That would
  explain a fault milliseconds after a pool's end, not one a minute later, so it is a lead and not the cause.

### The field's fault: two signatures, and a rule the code states and does not keep (read, not yet run)

Written while the player had the GPU, from the logs and the code alone.

- **The crashes are of two kinds and were counted as one.** The GPU's own fault ("instruction pointer" and "read
  invalid", the device lost at the next submit) and an access violation inside the driver on one of our threads
  (`nvoglv64.dll+0xfa0dfe`, also `+0xfa0e3e` and `+0xfa10e5`). The run that "disproved" keeping every pool record
  until its submission completes died of the second kind, so it said nothing about the first. Count them apart.
- **The first kind has a candidate the code itself names.** The pool table a submission binds is pinned at its
  first use (`vk_exec_draw.c gmem_snapshot`), and the comment there gives the reason it may be: "a pool ... that goes
  keeps its buffer until the submission's fence". `vk_memory.c vk_pool_fini` keeps only a mirrored pool's record
  that long; any other is destroyed at once, after a wait that covers what is submitted and not what is still being
  recorded. A game thread that finalizes such a pool during a recording leaves the pinned table pointing at a
  destroyed buffer for every later draw of the submission, and a shader that still loads through an address in that
  pool reads freed device memory. The change that keeps every record until the recording submission completes is in
  the working tree again, marked untested: run the field recipe with it and count the first kind only.
- **The second kind reads our own numbers as a handle.** The addresses the driver faulted on are 0xd00000050,
  0xfff00000051 and 0xfff000000a7: 0x50 past 0x0000000d'00000000, 0x00000fff'00000001 and 0x00000fff'00000057, which
  are pairs of small 32-bit numbers, the second 4095 twice. That is what an array of occlusion query slots looks
  like (`OQ_N` is 4096) or a pair of pool indices, not a pointer. The driver was handed, as a dispatchable handle or
  inside a structure, memory that held such numbers: a freed block reused, or an array read past its end. The one
  stack there is (`exec_op_` -> `par_try` -> `vk_sync_chain`) is where the chunks' command buffers are begun. Where
  to look: who holds a `VkCommandBuffer` or a pointer to one across `claim_slot` (which may move `slots[]`), and the
  arrays the walk fills beside it (`PAR.slots`, `PAR.conds`).

## 2026-11-14: video memory after a few map changes, and textures that go when no draw uses them

The player's report: after walking through a few places the frame rate fell and stayed down, also on the title
screen (8 fps there, video memory 11.0 of 11.9 GB, GPU busy 126 ms a frame).

### The cause

`vk_texpool.c` holds every registered texture in one descriptor set, because lit materials read baked lighting
through handles no binding shows. So every texture the game keeps registered was resident, and the game keeps the
textures of the maps it left for as long as its own heap has room. On the console that heap is the limit; here an
ASTC texture is BC3 (two to four times the bytes), so the same set is far larger than the card. A tour of six maps
(`REKARINA_WARP_AT=25:field;60:market;95:castle;130:village;165:field;...`, DLSS Quality, 2560x1440) went from 6.2 GiB in
the allocator after the save loads to 8.4 after one map change and 10.3 after three; with the desktop's own 1.2 to
2.5 GiB the 12 GiB card was over its budget in the second minute (52 fps), at 28 fps in the third, and allocations
failed. The rule that was meant to evict idle textures never fired: it took "no lookup reached it", and the pool's
sweep looks every registered slot up.

Two smaller things found on the way:

- The pipelines built ahead at start hold video memory: about 840 MiB for the 5,037 compute pipelines and 300 MiB
  for the 2,901 graphics ones (the driver's count against the allocator's, title screen, with and without
  `REKARINA_LIB_PREBUILD` / `REKARINA_PIPE_PREBUILD`). Not building the compute ones ahead is no way out: the tour then had
  668 frames over 25 ms in a minute against 115.
- What the allocator frees stays in its blocks until a whole block is empty, and the driver counts the blocks. After
  a few map changes the blocks held 1 to 3 GiB with nothing in it.

### What a texture's use is

- **A draw's binding.** `vk_texture_by_id_locked` notes when a draw's lookup last named an entry (`drawn`,
  `Tex.draw_round`, `Tex.draw_frame`, `Tex.ever_drawn`); the pool's own lookups do not count (`pool_lookup`). The
  clock for "long ago" moves only with submissions that draw a scene, so a pause screen ages nothing.
- **The pool, for the rest.** Which textures do shaders read through the pool? Asked two ways. On screen: with
  `REKARINA_TPOOL_HIDE=<classes>` the pool shows the stand-in for a class (1 decoded from ASTC, 2 other single-layer
  textures, 4 arrays, 8 everything a draw has bound). Without the ASTC class, and without everything a draw had
  bound, forest and field look as they do with them (two runs of the same setting differ as much as the pairs did:
  mean pixel difference 11 to 13 of 255); without class 2 the lightmapped surfaces go dark and the stone path loses
  its stones. In the shaders: of 9,492 cached pixel shaders with pool arrays, 500 were traced, and in every one the
  index of every pool access came from loads of a storage buffer of the draw and from nothing else.
- **The handle tables, for what no draw ever binds.** Those storage buffers are read (`vk_texture_handle_table`, a
  thread that scans them four times a second for words of a handle's form). The game's table of baked lighting and
  material records (5.5 MiB, 32-byte records, the handle at +16) names every texture it has loaded, the maps it left
  too: counted as use for everything, nothing went (tour 6: 47 fps in the second minute). So a table's word counts
  only for an entry no draw ever bound. That table is written when a map loads and read from then on, so the
  textures of an arriving map are named before its first frame.

So (`evictable`, `REKARINA_TEX_EVICT_CLASS=3`, the default): a texture decoded from ASTC, and any texture a draw has
bound at least once, goes by when a draw last named it; a texture no draw ever bound goes by when a table last
named it. With class 2 (what only a table names always stays) the textures of the parts of each map the player did
not look at stayed for every map visited: 3.7 GiB after six maps, over the budget again in the fifth minute.

### When textures go

`sweep_pressure` keeps room: the line (`REKARINA_TEX_EVICT`, 88 % of the budget the driver reports) minus what really
holds something (the driver's use minus the free bytes inside the allocator's blocks). While the room is under the
reserve (a twelfth of the budget, at least 512 MiB) `evict_cold` destroys entries, the longest unused first:

- in play, what no draw named for 3,600 scene submissions and presents (the maps left behind);
- with the room under a third of the reserve, or over the budget itself, what no draw named for 600;
- behind a loading screen (no scene drawn for 30 presents), whatever no draw named for 30: the map behind the
  loading screen is not on screen any more, and what goes now is not there when the next map's textures arrive,
  so the two never have to fit together.

An evicted texture's key is kept (`cold_keys`): the pool makes no image for it, a draw's lookup does, as for a
texture seen the first time. The game's heap compaction moves textures (a copy, a new object, the old one
finalized): the texture at the new address is the evicted one still (`cold_kept_out` by the noted copy,
`vk_texture_register` by the slot), and a finalized key is done (`cold_released`). While the room is under the
reserve the pool also makes no image ahead of its first use for an ASTC texture or for one no table names
(`pool_holds_back`): made ahead they were uploaded only to be evicted half a minute later (11.8 GiB of evictions in
one tour while 4 GiB of what stayed had never been drawn).

Textures above 2 MiB (`REKARINA_TEX_SUBALLOC_KB`) have memory of their own, so an eviction gives it back to the system
at once; the allocator's blocks are 64 MiB (`REKARINA_VMA_BLOCK_MB`).

### What it measured

The tour above, frames per second by minute (mean / 1 % low), frames over 25 ms a minute in brackets; another
program's game ran on the card during parts of the earlier runs of this night (not during tour 10), so those
single numbers are rough:

| | minute 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| before | 95.5 / 31.2 (125) | 51.7 / 10.6 (893) | 28.1 / 8.4 (932) | run ended | |
| after (tour 10, the last build) | 102.1 / 38.1 (71) | 111.8 / 39.6 (68) | 101.9 / 40.0 (60) | 108.3 / 33.9 (127) | 106.2 / 31.0 (138) |

Video memory by the driver's count: before, up to 14.9 GiB of a budget of 10.1 to 10.4, over it in 83 of 154
samples; after, between 7.5 and 9.5 GiB of a budget of 10.5 to 11.2 through the whole run, never over it; 18,072
textures (11.7 GiB) evicted on the way, 5,138 of them made again when a draw used them.
The long frames that remain are the map changes themselves (a tour changes maps every 35 s).

### Not solved here

- The device fault of the field and the Market ("read invalid" with an instruction pointer, then
  `vkQueueSubmit2 ... failed (-4)`) showed once this night, in a Market run right after 512 evictions. Eviction
  destroys images like every other path, so it meets whatever still names a destroyed image as often as they do.
- The frames after an arrival: the arrival frame is 100 ms, then about 160 frames run at 18 ms where the place
  settles at 12.7 (field, base picture): bindings +1.5 ms, the texture registrations' wait for the lock +1.1, the
  pool table +0.7, mirror prologues +0.6, the pool set's update +0.4.
- `REKARINA_SHADOW=1` (the memory path of Linux and Proton, so of a Steam Deck): 170 fps on the title screen, then with
  the world loaded 3.3 GiB written back to guest memory per 60 frames with waits of half a second for the GPU, 1.8
  fps. Shadow storage copies every GPU write back at its fence; the mirror path's "write back what the CPU touches"
  is what it needs.

## 2026-11-14: sun shadows that came and went every frame, and the "pillar" shadow: the grass's shadow points

The player's report: flicker "from time to time", and in the field "a strange shadow, like a pillar", seen in other
places too.

### What it was

The grass casts its shadow through a map of its own: a square depth target sized by the camera (2400x2400 at 2560x1440,
the "929 to 1,579 texels" target of the notes of 2026-11-13), drawn by two mesh draws (the two grass lists of the
frame, 27 and 362 blades at the gate of the village) and read by one full-screen pass that writes the screen's shadow
mask. The mesh shader draws **points**: a vertex a blade, at the blade's tip, with a point size from the blade's
height. The translator took every mesh shader as triangles of position and attributes, so this one's point size became
x, its position moved one component on, its primitive count came out of the top byte of a float, and it rasterized a
few triangles from the map's centre to wherever three unrelated blades happened to be: a wedge over a tenth of the
shadow map, another one in every frame because the game builds the grass list in another order every frame. On the
screen that is a large sun shadow with a straight edge, on the ground and up the cliffs and walls, that changes every
frame: 398 of 400 frames differed from both neighbours at the gate of the village, standing still.

The "pillar" is something else (checked the same morning, after the commit that names it): in the field a long thin
shadow runs from the hero to the edge of the screen in yesterday's build and in this one. A frame dump there shows the hero
whole in his own 768x768 shadow map and the sun 21 degrees above the horizon in that save (light direction
0.919, -0.352, 0.177), so his shadow is 2.7 times his height: what the game draws, as far as this port can tell.
Static things carry baked shadows that are never that long, which is why it looks foreign. Not compared with the
console.

Fixed in the translator (patch 0040, docs/oot-xlate.md "Mesh shaders that output points"): the grass's shadow map now
holds what the game draws, a few hundred points of 2 to 6 texels where the grass stands, and 0 of 400 frames differ.
Seven of the game's 854 mesh shaders are of this kind, all for the grass.

### How it was found

- A clip is not a measure of this: recorded at 60 fps from a game at 100, it showed "half the frames differ" for
  every setting. The flight recorder (`REKARINA_FLIGHT=1 REKARINA_FLIGHT_AT=<frame>`) keeps a thumbnail of every presented
  frame; the mean of each 20x20 block against the mean of both neighbours, over the last 400 thumbnails, is the measure
  that held (scratch `flsd.py`: worst block 16.5 before, 1.6 with the two draws skipped, 3.1 after the fix, which is
  The hero's idle motion and the fairy).
- Ruled out first, each with a run that flickered as before: eviction and the pool (the night's own changes), present
  modes, one replay thread, the mesh count copies (`REKARINA_MESH_BATCH=0`), the sun shadow map's size
  (`REKARINA_SHADOW_RES`), yesterday's exe. The counts of the indirect mesh draws were the same in every frame
  (`REKARINA_MESHCOUNT`), so no list was short.
- Two frame dumps of the same frame number in two runs (`REKARINA_DUMP=3800`), every pass image of one compared with the
  other's (scratch: align the lists by kind and size, mean difference of the left part of the screen): the first image
  that differed was the shadow mask after the pass that reads the 2400x2400 map, and that map held one black wedge
  with its tip in the exact centre and nothing else. A tip in the centre is a vertex at (0, 0): not geometry.
- `REKARINA_SKIP_SH=c703b1a1a3a13e9b` (the pass's fragment shader) removed the flicker, so the two draws were the whole of
  it. spirv-cross on the mesh shader (`sh_<hash>.spv` of the dump) showed the kernel writing five component blocks and
  one index byte a primitive, and the entry point reading four blocks and three bytes.
- The frame dump printed the compute stage's bindings for a mesh draw (its stage table knew vertex, fragment and
  compute), which cost an hour on a "wrong uniform block" that was not there. It now reads the task, mesh and geometry
  tables (`vk_dump.c nvn_stage`). `REKARINA_MESHCOUNT_PROG=1` adds each record's program and pass width to the count log.

### The cache

A translator change re-keys every shader: 55,711 programs to translate again on every player's machine for seven
shaders. So the translator identity stays that of patch 0039 (`tools/build_xlate.py SAME_AS`), and the program hash of
a mesh program with point output ends with a tag (`gfx_shader.c build_code_hash`, `tools/prepare_shaders.py
code_hash`), which gives those seven new keys. An installed game finds everything but them in its cache; the seven
translate when first drawn.

## 2026-11-14, morning: the Radeon's mesh shaders, memory on a 16 GiB machine, a 4 GiB card

Three things the player added to the night's list. The first is mostly solved, the other two are measured.

### The mesh shaders on AMD's Windows driver

Before: on the integrated Radeon (driver 2026.04.17, "AMD proprietary shader compiler", subgroups of 64 lanes) the
scene was whole except for everything mesh shaders draw: tree crowns, bushes, ground plants, grass and their shadows.

Found and fixed: the entry point stored a triangle's three index components from three threads, and that driver
loses such stores. With each primitive's index vector stored whole by one thread (translator patch 0041,
docs/oot-xlate.md) the crowns, bushes, ivy and ground plants are there, in the forest and at the gate of the village.
The runtime asks for whole vectors on every device but NVIDIA's (`REKARINA_MESH_INDICES`); NVIDIA's translations do not
change.

Still wrong there, and left out: the draws through a task shader, which on the RTX are the short grass that covers
the ground (the forest with and without them, `REKARINA_X_SKIP_TASK` of the experiments). Once the mesh stage emits
primitives, the first such draw stops the GPU (the device is lost with "No fault detected"; the draw breadcrumbs,
`REKARINA_DRAW_CRUMBS=1`, name a direct mesh draw under a render condition) or the driver dies with an access violation
inside `amdvlk64.dll` during one of our draw calls. So `vk_exec_draw.c mesh_task_draws` skips them on AMD's Windows
driver (`REKARINA_MESH_TASK=1` draws them); the other driver branch (RDNA 3 and 4) has not been run and gets the same.

How it went, since the order mattered:

- Byte stores as two atomics in place of the compare-and-swap loop: no change. One fixed triangle a workgroup in
  place of the shader's output: nothing on screen. Neither told anything, because the index stores were lost in
  both.
- Whole index vectors: a crash inside the driver at the first frame of the scene, where before nothing was drawn.
  That reads like "worse" and was the fix: the mesh stage emitted for the first time.
- With the draws through task shaders skipped the run lived and the picture was right. Then the task shader was cut
  down step by step (launch at most four mesh workgroups; launch none; do not run its code; nothing but
  `EmitMeshTasksEXT(0, 1, 1)`): every one of them still ended the run. The payload has the same size in both
  stages, the direct draws ask for 13 to 32 workgroups, the mesh outputs match the fragment inputs.
- One measure that does not hold on that driver: its output granularity is 256 (NVIDIA: 32), so by Vulkan's formula
  the mesh shaders with eight and more attributes are over its 32,768 bytes of output memory (36,864 to 45,056).
  Those without a task shader draw all the same.

A run after a lost device or a driver crash starts slowly there: the driver's cache is gone, the pipeline prewarm
takes 100 s instead of two, and the title screen ran at 3.9 fps for the rest of the morning. The crash record now
names the faulting thread's stack (`main.c crash_handler`): the first guess, a stack overflow in the driver, was
wrong and cost two runs.

### Memory

On the RTX 5070 at 2560x1440: working set 12.2 to 13.4 GiB, private commit 24.5 to 26 GiB (16.7 GiB on the title
screen). By allocation, in the field with 8.6 GiB of video memory in use (scratch `vmmap.py`: VirtualQueryEx and
QueryWorkingSetEx on the running process):

| what | committed | in the working set |
|---|---|---|
| the game's heap at 0x100000000 (`SetHeapSize`, as the game asks) | 7.4 GiB | 4.5 GiB |
| the game's other mappings at 0x400000000 | 2.7 GiB | all |
| 203 blocks of 32 MiB, 86 of 16 MiB and some thirty just under 32 | 8.4 GiB | none |
| two blocks of 1 GiB, one of 0.5 GiB | 2.5 GiB | all |
| 152 blocks of 3 MiB and smaller ones | 2.3 GiB | most |
| the exe and the driver's DLLs | 0.8 GiB | a tenth |

The 16 and 32 MiB blocks are the driver's address space for device memory: their sum follows the video memory in use
(8.4 of 8.6 GiB here; the same on the integrated Radeon, and 10.5 GiB with 10.6 in use in a run that held 7.6 GiB
more) and nothing ever touches them. They count against the commit limit, not against the machine's memory. So on a
16 GiB machine the game needs 12 to 13 GiB of memory and a page file that lets the commit reach 26 GiB, which
Windows' own page file management gives. The two 1 GiB blocks are not identified.

Changed: the texture cache's warm-up read (8 GiB of files into the system's file cache at start) is now sized by the
machine: nothing under 14.5 GiB of memory, the memory beyond 14 GiB up to 8 GiB above that (`tex_cache_scan`).
Not done: the heap is committed whole as the game asks; a run on a 16 GiB machine.

### A card of 4 GiB

`REKARINA_VRAM_HOG_MB` does not stand for a small card: what it takes counts as the game's own use, the budget stays the
card's. `REKARINA_VRAM_BUDGET_MB=<n>` (new, `vk_memory.c vk_mem_vram_cap`) caps the budget every caller sees, the texture
quality's choice included; the memory is still there, so such a run shows what the eviction does, not the small
card's frame rate.

The tour at 1920x1080 with a budget of 3,400 MiB (texture quality chose "quarter"): the driver counted 4.0 to 5.5 GiB
in use, over the budget in 275 of 296 samples, and 53,311 textures (40 GiB) were evicted in three minutes, 4,553 of
them made again. What is in it, in a scene:

| what | GiB |
|---|---|
| textures with memory of their own (ASTC as BC3 0.6, the game's own formats 0.7 to 0.9, written by the GPU 0.3 to 0.65) | 1.7 |
| outside the allocator: the pipelines built ahead (5,037 compute, 2,901 graphics) | 1.2 |
| device-local mirrors of the game's buffers | 0.4 to 0.8 |
| the rest of the allocator's blocks (staging, free room in blocks) | 0.7 to 0.9 |

So the floor is 4.2 to 4.7 GiB where the card gives 3.4. The first thing to give back is the 0.8 GiB of compute
pipelines built ahead, of which a scene uses a small part; then the mirrors (imported memory is slower but costs no
video memory), the baked lighting's arrays (1365x1365x90 twice, 0.5 GiB, which "quarter" leaves whole) and the far
shadow map (12288x12288 D16, 0.3 GiB).

## 2026-11-14, day: the device fault behind a map change, compute pipelines by use, what holds the memory, picture height 1080

### The device fault behind a map change

The loss of the device that has been in the notes since the Market runs (`vkQueueSubmit2 ... failed (-4)`, an invalid
read in the fault report) came back in the tour: with map changes at 25, 60, 95 and 130 s and the input script walking
The hero through an exit some 15 s after the arrival in the village, DLSS quality, it lost the device in 2 of 4 runs, each
time one to two seconds after the second loading screen.

The cause is in how a submission starts. `vk_backend.c submit_sync` called `vk_texture_new_submit` and then
`vk_sync_submit`, which takes the submit lock. The first moves the epoch that says "this submission has not looked
yet": the texture pool's set is brought up to date once per epoch, before the first bind (`vk_texpool.c
vk_tpool_bind`). OoT submits from more than one thread, and now and then a second thread's call arrives
while the first is still recording: 17 to 33 of 112,353 of the game's submit calls in three minutes waited for the
lock (the counter in `vk_sync_submit`, `REKARINA_TRACE=perf`), most of them around map changes. That second call had
already moved the epoch. The first thread's next bind of the pool saw a new epoch and made the update in the middle
of its recording; the second thread, once it had the lock, found the update for its epoch made and recorded its whole
submission without one. Whatever went between those two points (a texture the game finalized, a released target
dropped under memory pressure, an eviction: each is destroyed as soon as the GPU is past the submission that was
being recorded) was still in the pool's set when the second submission's shaders indexed it. Mostly that reads memory
that still holds the old texels, or another texture's: a wrong texture for one submission. When the allocator had
given the block back, it is an invalid read and the device is gone. Eviction (this morning's change) frees whole
blocks behind a loading screen, which is why the fault became frequent there.

`vk_texture_new_submit` is now called inside `vk_sync_submit`, with the lock held, for every submission: the epoch
moves only between recordings, and an empty submission that records held rests (`gfx_sync_fence`) has an epoch of
its own too, which it did not have. Eight runs of the same recipe went through without a loss.
`REKARINA_EPOCH_EARLY=1` brings the old order back.

### What a fault report can and cannot say

Hours went into the report's lines "image ... UNBOUND from ... N ms before the report". Three things about them:

- A lost device answers `vkGetSemaphoreCounterValue` with the counter at its largest value. `collect_locked` took
  that at its word and the reaper destroyed every deferred image and view at once, at the moment of the fault. So in
  every report the last image at the fault address had been "unbound" about five seconds before the report (the five
  seconds are the wait that fails afterwards), which reads like a use right after a destroy and is only the reaper
  running after the GPU had stopped. `collect_locked` now destroys nothing on an error or on that value.
- The binding report's ranges are not the images' sizes (a 512x512 BC4 texture of 173 KiB shows 0xa0000 or more), so
  "the image whose range holds the address" is a guess.
- A handle listed for the address belongs to a live image again by the time of the report; "an entry names it" under
  an older line says nothing.

`REKARINA_LIVE_CHECK=1` (new) looks from the other side: every image and view waiting for its destruction is kept by
handle with the value it waits for, and each place where a recording takes one (a draw's texture binding from the
memo or a lookup, the view cache, a render target, every texture pool slot after an update, a pool bind in a
submission that made no update) asks whether it waits for a value below the recording's own. It also compares each
destruction with the GPU's breadcrumbs. In the runs since the epoch fix it has reported nothing.

### Compute pipelines by use

The boot prebuild kept a pipeline for each of the game's 5,037 compute shaders; a session dispatches a few hundred
(224 after the tour's five maps). What they cost, measured on the RTX 5070 at 2560x1440 in base mode, the same tour,
150 s in:

| | all 5,038 kept | the 224 a dispatch has used |
|---|---|---|
| working set | 12,277 MiB | 11,204 MiB |
| private commit | 24.6 GiB | 23.5 GiB |
| the driver's compiler arenas (below) | 2.3 GiB | 1.7 GiB |
| video memory outside the allocator at the title screen | 1,179 MiB | 333 MiB |
| frame rate, second minute | 115.9 fps | 115.7 fps |
| compute pipelines built at a dispatch | 0 | 0 to 1 (0.16 ms) |

`vk_pipeline.c` now keeps a record of the shaders a dispatch has asked for (`compute-used.bin` beside
`pipeline-keys.bin`: input ids, appended at a shader's first dispatch on the machine). Where the video memory budget
is under 8 GiB or the machine has under 20 GiB of memory, and the record holds 64 shaders or more, the boot prebuild
keeps only the recorded ones and does not read the others' SPIR-V (0.5 GiB). A shader outside the record is built at
its first dispatch, from the driver's cache, and joins the record. `REKARINA_CS_KEEP=all|used` overrides the choice.
When video memory is short on any card, `vk_pipeline_compute_trim` destroys the prebuilt pipelines that neither this
session nor the record has used. That frees less than it sounds: the driver's count did not fall when 5,016 pipelines
went (1.1 to 1.3 GiB outside the allocator for the rest of the run, against 1.4 to 1.7 without the trim); the driver
keeps the memory and uses it for what comes later. Not building them is what gives it back, so the first session on
a small card is the expensive one.

### What holds the memory

The two blocks of 1 GiB in the process's memory map (and a third, partly committed) are the NVIDIA driver's shader
compiler arenas: each starts with its own address and the size 0x40000000, the pages hold pointers into
`nvoglv64.dll` and the compiler's names (`glvc.bindings`, `nv.glv.reconvergence.tanglematch...`). Scratch
`vmarena.py` finds them. They do not follow our settings: four prewarm translators instead of eight changed nothing,
and without the boot prewarm of graphics pipelines they were larger (2.07 GiB) and the 1 % low worse.

The block of 512 MiB was ours: the host ring the mirror's submit-time snapshots are copied through
(`vk_memory.c SNAP_RING`). A frame takes one to five megabytes of it and the largest burst seen, a save's first
frames, 246 MiB; but the driver keeps host-visible memory resident whole, written or not (every page is in the
working set from the allocation on, checked with `QueryWorkingSetEx` before any read), so walking a smaller part of
it gains nothing. Its size now follows the machine: 512 MiB with 20 GiB of memory and more, 128 MiB below
(`REKARINA_SNAP_RING_MB`). With 128 MiB the tour's working set was 10,917 MiB against 11,201, six prologues copied from
guest memory as before the ring, the frame rate the same.

`REKARINA_VMA_DUMP=<path prefix>` (new, with `REKARINA_TRACE=perf`) writes the allocator's detailed map every 600 frames;
scratch `vmadump.py` sums it by memory type, kind and usage. A tour with the budget capped to 3,400 MiB and quarter
textures, in a scene (the picture was 2560x1440 in these runs: the cap run set the window's size, not the picture's):

| device-local memory | MiB |
|---|---|
| mirrors of the game's buffers (827 buffers and the 88 MiB pool) | 843 |
| the game's BC textures above 2 MiB, each in memory of its own (87) | 667 |
| the physical pool and its twin | 610 |
| ASTC arrays as BC3 (the two 1365x1365x90 lightmap arrays are 526 of it) | 538 |
| textures up to 2 MiB in the allocator's blocks (2,240) | 454 |
| render targets (colour with storage 253, depth 96) | 349 |
| outside the allocator (the driver's own) | 850 to 1,040 |

The part outside the allocator grows through the run (333 MiB at the title screen with the recorded compute
pipelines only) although no pipeline is built in play: it is the driver's, most likely allocations it keeps after
our 50,000 evictions and re-creations, which a card that is really full would take back. So the capped run counts
more than a real 4 GiB card would hold, and cannot say what that card's frame rate is. The mirrors cannot go: import
on a discrete card reads the game's pools over PCIe on every draw.

Tried and taken out again: BC textures the CPU fills made without block views and given them at their first pass.
The thought was that NVIDIA sizes a BC image with COLOR_ATTACHMENT usage like an uncompressed one.
`vkGetDeviceImageMemoryRequirements` says otherwise: 173 KiB with block views and without for a 512x512 BC4 texture
with ten levels.

### Picture height 1080

With the picture height at 1080 (`REKARINA_GAME_RES`, `widescreen.c`) the picture, the HUD and the dialogue boxes were
right, but the 3D scene was 1440x810 in base mode and 960x540 with DLSS quality: the scene's share was multiplied by
the picture's height over 1440 for every picture (`scale_k`). The game takes the fixed scale (patch set `res_fixed`)
against the smaller of its own 1440 lines and the picture it draws, measured by the scene targets it makes:

| picture | scale written | scene |
|---|---|---|
| 2160 lines | 1.0 | 2560x1440 |
| 2160 lines | 1.5 | 3840x2160 |
| 1080 lines | 1.0 | 1920x1080 |
| 1080 lines | 0.75 | 1440x810 |

So only a taller picture takes the factor. After the change: base mode 1920x1080, DLSS quality 1280x720, a 2160-line
picture as before.

What the height is worth, standing in the village in base mode on the RTX 5070: GPU time 11.94 ms at 2160, 8.97 at 1440,
7.85 at 1080 (and 7.16 with the 810-line scene of before). That is about 6.6 ms that does not follow the picture and
0.65 ms per megapixel of scene. `REKARINA_GPU_PASSTIME` shows why: the 24 passes at the scene's size cost 2.25 ms a
frame at 1080 and 2.53 at 1440, the rest is some 240 short passes and dispatches (shadow maps, culling, light
lists). The frame rate goes from 104 to 114 fps there, where the GPU is the limit, and does not move in a tour where
the CPU is (110 to 114 against 112 to 118). Render targets take 630 MiB of video memory instead of 690.

### Released shadow maps that the game still reads

At the castle the ground lost its sunlight six seconds after the scene appeared, while the map's name was still on
screen, and stayed in shade; a second darkening followed 1,350 frames after the arrival. The step is exactly 600
frames after the first lit frame in base mode and with DLSS alike (a thumbnail of every frame from the flight
recorder, `REKARINA_FLIGHT_AT`, and the brightness of a ground region in each: scratch `luma.py`). A round number of
frames after an arrival points at one of our own age rules, and the control runs named it: with
`REKARINA_TEX_DROP_AGE=0` the light stays, with `REKARINA_TEX_EVICT=0` the step is unchanged, with the pressure rules off
altogether it comes at +420, where a burst of transient targets fills the released list to its limit.

The game finalizes the texture objects of its shadow maps at a map's arrival (the far one is 12288x12288 D16, 288 MiB
of its own memory, another is 4096x4096) and goes on reading them through their slots for as long as the map is
shown. Our entries wait on the released list with the only copy of what the GPU drew; the age rule of
`sweep_pressure` counted from the release, destroyed them, and the next lookup made an empty image. A released entry
that lookups still reach is in use now (`released_in_use`, vk_texture.c: its last lookup is newer than the release
and younger than 120 frames): the age rule, the list's limit (`sweep_released`) and `recycle_take` pass it by. With
it the ground keeps its light for the 1,650 frames recorded after the arrival, the tour's frame rate is unchanged
(120 and 118 fps on average, base and DLSS) and nothing was lost. `REKARINA_REL_KEEP=0` brings the old clock back for a
test.

The player also saw the light flicker "at times" while swimming under the Market's drawbridge. Four runs there (the
field arrival, a walk off the bridge's side, swimming beside its supports for 75 s and floating under it for 60 s,
with the old clock and the new one; a thumbnail of every frame, scratch `jump.py` for the brightness of each) show
no step or pulse of the lighting with either: what changes the picture's brightness from one frame to the next there
is the camera crossing the water's surface (the bridge pushes it down to the water line, and the underwater tint
comes and goes with it) and the fairy passing close to the lens. Whether that is what was seen is not known.

### Still open: a lost device in the Market

With the epoch fix in, the device was still lost in the Market, three times in some 60 arrivals (DLSS quality, three
arrivals a run), each time in the burst a few seconds after the arrival: for a second or two the game makes about
3,000 transient render targets a second, 35 to 50 a frame, and the frame rate falls to 41 to 55 fps (scratch
`arrive.py`, `secs.py` on the section timers: taking the images made ahead, the pool's views of them and the upload
batch's flushes add 6 ms to the replay's frame). The one report made after `collect_locked` stopped destroying on a
lost device's counter: an invalid read at an address whose last image, a 1707x960 R11G11B10F target bound into pool
memory, had been unbound six seconds before the report; no pool slot, cached view, memo or entry named it, and the
live check had reported nothing in that run. Two things were tried and dropped: that a submission's value is
signalled by an empty last batch while its chunks of a draw stretch still run (no submission ends that way: 0 of
90,112), and that the counter runs ahead of the GPU's breadcrumbs (it did not in 200,000 comparisons). The live check
also looks for handles that were destroyed (a graveyard of the last ones), and the fault report says how long before
it the device was first seen lost. Five more runs of five arrivals were clean, and so were six on the build that
keeps released entries in use (30 arrivals; the section above). At one loss in five or six runs, six clean ones
happen by chance about one time in three: whether those entries had a part in it is not shown.

**2026-11-15: a cause found in the texture pool's descriptor set.** `vk_texpool.c resolve_slot` wrote a slot only
when its view handle changed. A slot whose image went is marked dirty; its old views are destroyed once the GPU is
past the submission that let the image go, the driver hands their handle values to later views, and the slot's new
view could carry the old view's handle: no write, and the update-after-bind set kept the dead view's descriptor, read
at the destroyed image's address by the first shader that indexed the slot, possibly seconds later. The live check
cannot see it (it checks the handles we keep, and the handle is alive again), and the fault report's "who still
names it" looks at the slot's image, which by then is the new one: both fit the report above. A dirty slot is now
written whatever its handles (`force`), and `create_tex` puts an image whose handle had cached views of a destroyed
one on the pool's gone list too. A counter (`DEBUG-TPOOLABA`) logs every 256th such slot: 14,600 to 22,500 of them per
run of the repro, so the skipped write was not rare. Six runs of five arrivals (`mkt.sh mkf`) after the fix: no
device loss; the camera-turn runs at the boulder unchanged (0 dark frames, 0 flash events, 112-115 fps). Six clean
runs are not proof at one loss in five or six; the counter is.

### The Linux and Proton memory path (REKARINA_SHADOW=1): where the 1.8 fps came from, and the fix

One run standing in the village with `REKARINA_SHADOW=1` on the RTX 5070 (mirror mode on top, as a discrete card under
Proton would run), against the same run without it, before the fix further down; this is what the log and the code
said.

| per frame, world loaded | default | `REKARINA_SHADOW=1` |
| --- | --- | --- |
| frame time | 8 to 10 ms | 545 ms, 528 of them waiting for the GPU |
| GPU time of the timed parts | under 9 ms | 427 ms |
| CPU to mirror upload | 2.9 MiB | 63.8 MiB (61.6 MiB of "written pages") |
| mirror to guest write-back | 0 MiB (on demand) | 46.0 MiB |
| write-back merges | 2.2 MiB in 55 dispatches | 20.1 MiB in 45 |
| shadow to guest copies on the CPU | none | 64 MiB |
| "pages the mirror wrote, left out" | 0.004 MiB | 0 |
| pools and their copies | 566 mirrors, 676 MiB | the same, and 4,500 shadow buffers, 5.0 GiB |

Three things stack up:

1. **Write-back on demand is off** (`lazy_on`: `shadow_on` turns it off). Every shader-written range goes back
   whole, 46 MiB a frame, where the default path leaves all of it in the mirror until the CPU touches a page.
2. **Our own copy looks like the CPU's write.** At the fence `vk_gpu_written` copies the range from the shadow into
   guest memory with `memcpy`. That sets the write watch on every page of it. `guest_written_` then asks which pages
   the CPU wrote (`mem_written_peek`) to leave those out of "owned by the mirror", finds all of them, and owns none:
   the next `mirror_sync` uploads the same 62 MiB into the mirror again, and the pages the GPU writes every frame go
   through the merge. On the default path the write-back is the GPU's copy into imported memory, which the write
   watch does not see.
3. Why the GPU then needs 427 ms a frame for copies of this size is not measured (no pass timing in that run).

What a fix had to do, as planned before it was built (the build differs in the first point, see below):

- **Lazy states for shadow pools.** `note_written` reports the whole bound range, and `vk_gpu_written` copies all
  of it and calls `mem_thaw`, whose `lazy_touch_range` opens every lazily closed page: with `lazy_on` simply
  allowed, the first write-back would make every page hot and copy stale shadow bytes over cold ones. The shadow
  copy must go page by page and leave closed pages alone.
- **The fetch** (`lz_touch`): after `lz_copy` brought mirror bytes into the pool's buffer, which is the shadow
  here, the pages are opened and copied from the shadow into guest memory; `LzPool` needs the shadow's mapping.
- **Telling our copy from the CPU's.** Resetting the write watch after the copy loses a CPU write that lands
  during it, and the scene culling's small pools are written by both sides every frame (their stale lists were the
  flicker of 2026-11-10). Comparing guest memory with the shadow afterwards gives false alarms when a later
  submission wrote the shadow again. Lossless: copy the page to a private buffer, reset the page's bit (a set bit
  is the CPU's), write guest memory from the private copy, reset again and compare guest memory with the private
  copy: equal means only we wrote.
- **Without a mirror** (`REKARINA_MEMORY=import`, an integrated GPU, the Steam Deck): the GPU writes the shadow
  directly, nothing is "owned", and `shadow_sync_pool` copies every stamped page from guest memory back into the
  shadow, so each GPU write is copied out and in again. It needs a mark of its own for "this stamp is a GPU
  write the shadow already holds" and the same lazy states at the fence.

**Written 2026-11-15 and kept out of the tree: write-back on demand for shadow storage** (`REKARINA_SHADOW_LAZY`; the
patch over `runtime/vk_memory.c` is kept as `shadow_lazy_agent.patch` in the development machine's scratch
folder). Runs on the RTX 5070 the same morning, standing in the village: the CPU copies fell (shadow -> guest at the
fences 64 -> 2-3 MiB a frame, uploads 64 -> 19 MiB), the frame did not (541 ms, GPU-bound: the GPU reads the shadow
buffers in system memory over the bus, which a Steam Deck's shared memory would not). And the scene is black in
both, with the patch and without it (`REKARINA_SHADOW=1 REKARINA_SHADOW_LAZY=0`, frame 1800 at the arrival in the forest:
the HUD and the map's name on black). So shadow storage did not run this game at all then, whatever its speed
(fixed the same day, without the patch: "2026-11-15: shadow storage draws the scene" below). What the patch does:
`REKARINA_SHADOW_LAZY=0` is the shadow path as it was; without shadow storage none of it runs (`shadow_lazy` is 0, and
every changed default-path line reduces to the old one).

- **Clean copies at the fence** (`shadow_wb`, `shadow_put_run`, runs of up to 256 pages): a private copy of the
  shadow's bytes; the run's write-watch bits taken (`sp_begin`: a set bit is the CPU's); the copy into guest memory;
  under `gen_lock` the bits taken again, guest memory compared with the private copy (`sp_end`) and the pages
  stamped, so no query can slip a CPU write in between. A clean page is marked as the shadow's own (`sown_chunk`,
  new) and, for a range the mirror wrote back whole (`vk_gpu_written_mirror`), as the mirror's (`own_chunk`):
  neither sync copies it back. Any other page gets its write-watch bit back by a locked write of nothing
  (`sp_redirty`, `lock orl $0`: an atomic add of 0 may be compiled into a fenced load) and goes on as the CPU's
  write, as every page did before. The partial pages at the ends of a range are copied plainly, as before: a CPU
  write elsewhere in such a page could not be told from the copy.
- **Page states with VRAM mirrors**: `lazy_split` runs under shadow storage too, with the default path's states.
  Not as planned: a new page is closed and still written back, into the shadow, and the fence skips it; its first
  access copies it from the shadow (`lz_touch_shadow`, no GPU wait). After 240 writes without an access it goes
  cold: no write-back any more, and its first access fetches it from the mirror into the shadow (`lz_copy`) and then
  into guest memory. Closed-means-cold-at-once was left out because of a race: a page closed cold when a part that
  writes it is recorded, and touched before that part is submitted, gets the old bytes from its fetch while its
  write-back was skipped, and as a hot page the fence then copies the stale shadow over whatever the CPU wrote
  since. On the default path that race exists only at the 240th write of a page; cold at once would put it at every
  page's first. The price: write-backs into the shadow go on for 240 writes of a range before they stop.
- **Without mirrors** (`REKARINA_MEMORY=import`): the whole pages of a written range of 64 KiB or more close at the
  fence instead of being copied (`shadow_close_new`, as new pages that never count up); the first access copies one
  in from the shadow, and from then on it is hot (copied cleanly at every fence).
- **The shadow's sync** (`shadow_sync_pool`, `shadow_copy_open`) leaves out a page whose newest stamp is the
  shadow's own, and every closed page (guest memory is older there, and a read would fault).
- Closed pages are stamped at the fence like the rest (the texture cache must see the change; its upload reads the
  page, and the fault copies it in), and marked as the shadow's (and the mirror's) while they stay closed.
- `lz_pool` takes up to 512 pools under shadow storage (128 on the default path); a pool is listed once
  (`PoolRec.lz_listed`), before its first page closes, since a fetch holds `lz_lock` while it waits for the GPU. A
  finalized pool opens its closed pages before its shadow goes (`vk_pool_fini` calls `lz_pool_gone`).
- The page states need the access hook, which only the Windows build has (Windows, Wine, Proton):
  `plat_mem_access_hook` does nothing in `platform/linux.c`, where a closed page would end the process. A native
  Linux build gets the clean copies and the sync's skip only (the uffd write protection is reset per range by
  PAGEMAP_SCAN; not run there). `REKARINA_SHADOW_LAZY_STATES=0` gives the same anywhere; the states are also off with
  `REKARINA_MEMORY=verify` and the async write-back, as `lazy_on` has it.

Kept from the old shadow path in the patch, not fixed there: a page the CPU and the GPU both write keeps the GPU's
bytes (CPU writes into a range while its GPU write is pending never reach the shadow, and the fence or the first
access copies the shadow over them; the tree's path merges word by word since). Open questions: the first access opens the pages before it writes them, so another thread reading
in those microseconds reads the old bytes (as planned); a fault inside the fence's copy (a page closed meanwhile by a
recording) runs a first access inside the completion thread's copy, which has a buffer of its own for that, never
exercised.

What to run (`REKARINA_TRACE=perf`, standing in the village, after the 240 frames that take new ranges cold):

- `REKARINA_SHADOW=1`: the log says once `vk: shadow storage: fence copies do not count as the CPU's writes; pages the
  CPU does not touch stay in the shadow or the mirror (REKARINA_SHADOW_LAZY)`. Then, every 60 frames: `mem: mirror
  frame N: ... upload X MiB ... write-back Y MiB ... merge Z MiB` (before: 63.8, 46.0, 20.1; the default path: 2.9,
  0, 2.2); `mem: mirror frame N: upload sources per frame ...: written pages X MiB; ... pages the mirror wrote, left
  out Y MiB` (before: 61.6 and 0); `perf: frame N: write-back on demand: per frame X MiB left in the mirror ...`
  (now printed under shadow storage too); `mem: shadow write-back on demand at frame N (mode 2), per frame: guest ->
  shadow X MiB (Y MiB left out as the shadow's own or closed), shadow -> guest at the fences Z MiB (W MiB of it
  counted as the CPU's write), left behind closed pages ..., pages closed at the fences ..., pages brought into guest
  memory at a first access ...` (before: 64 MiB shadow -> guest, all of it the CPU's); the frame time.
- `REKARINA_SHADOW=1 REKARINA_MEMORY=import`: the same shadow line; pages closed at the fences high after a load and then
  near 0, shadow -> guest small.
- Controls: `REKARINA_SHADOW_LAZY=0` (the numbers above again), `REKARINA_SHADOW_LAZY_STATES=0` (clean copies alone).
- Correctness against `REKARINA_SHADOW_LAZY=0` and the default path: flicker, black or missing objects, the flash count
  of `pulses.py`; any `vk: write-back on demand: the fetch of ... into the shadow did not finish in time`.

### 2026-11-15: shadow storage draws the scene (the Steam Deck's memory path)

The written-down patch above is still out of the tree; this is the shadow path as it was, made correct. Under Proton
the game turns shadow storage on by itself (`plat_is_wine`), and an integrated GPU (a Steam Deck) runs it without
mirrors (`REKARINA_MEMORY=import`). On the RTX 5070, standing in the village (`loc/kk_still.txt`), `REKARINA_SHADOW=1
REKARINA_MEMORY=import` now draws the same frame as the default path: a frame dump of each (`REKARINA_DUMP=t5400`) compared
pass by pass (`dcmp.py` in the scratch folder: images matched by name without the pass number, mean and channel
range) differs only in two small simulation targets, where the same comparison before showed the sky's light table
8x too bright, the fog volume empty and the sun's cascades nearly empty. What was wrong, in the order found:

1. **Fence copies over the CPU's words** (black scene, NaN in the sky light and the fog). `vk_gpu_written` copied the
   whole range from the shadow back into guest memory; OoT keeps CPU-written constants beside GPU sums in one storage
   block, and the CPU's words (held out of the shadow while the range was pending) were overwritten with the shadow's
   zeros. Now the merge goes word by word against `pre`, the shadow's bytes when the write was recorded
   (`shadow_merge`): a word the GPU changed goes to guest memory, one only the CPU changed goes into the shadow.
2. **The CPU's writes into pending ranges reached the GPU a frame late or never** (no short grass, no fairy, no
   fireflies). The sync left every pending range out; with frames in flight a range the GPU writes every frame is
   always pending. Now the sync copies the CPU's words the GPU has not changed (shadow still equal to every covering
   write's `pre`) and moves those writes' `pre` along (`shadow_copy_pending`).
3. **Float sums** (the auto exposure's luminance): the replay notes a range once per bind, several times a
   submission, and each note's merge added the GPU's change again. A float-sum range (`GFX_SSBO_FADD`, passed down
   from `note_written`) now adds shadow minus `pre` to guest memory once, leaves the shadow alone and moves the other
   notes' `pre` forward.
4. **Stores through loaded addresses** (`gmem`): the write log shows them at the fence only, so they had no pending
   record. A piece seen written becomes hot (up to 12 MiB, 300 frames): every submission that binds a write log
   records the hot pieces as pending, the report merges a written one, the fence drops the rest
   (`vk_mem_shadow_gmem_*`). Under shadow storage the report goes per table entry, not per run of entries.
5. **Pools read only through loaded addresses never synced** (the exposure at its brightest limit, the actual cause
   of the white haze). A pool's shadow took the CPU's writes only when a binding resolved it; the light probes the
   luminance pass (`cs 2a31aac713a04055`, 1x1x1) reads through pointers stayed zero, the luminance came out zero and
   the exposure ran to its clamp (`c4[56].w`, the EV, 7 against 10; `c4[4].w` 78.1 against 9.77). Now the first table
   access of a submission puts every shadowed pool into the sync (`shadow_touch_all`), as `vk_mem_gmem_sync` does
   for the mirrors.

Then the cost, which syncing every pool made worse (16 fps):

- The sync queried the write watch once per pool, 5,000 queries a frame. It now goes through the guest submission's
  write-watch snapshot (`snap_regions`, one query per 2 MiB region per submission, as the mirrors and the texture
  checks do): 155 queries a frame, the sync's time 12 -> 2.5 ms a frame.
- Every range the GPU wrote came back into the shadow whole at the next sync, ~50 of 70 MiB a frame: the fence's
  copy into guest memory set the write watch and `guest_written_` stamped the range. Now the fence takes the write
  watch's bits of the range's whole pages (`shadow_unwatch`: a page that differs from the shadow afterwards is stamped
  as the CPU's, one that does not is the copy's own) and marks the pages as the shadow's (`sown_chunk`, the stamp of
  the report); the sync leaves them out until a newer stamp: 70 -> 12.5 MiB copied in a frame.

| standing in the village, RTX 5070 | fps | sync per frame | fence reports per frame |
| --- | --- | --- | --- |
| default path (mirrors, no shadow) | 101 | none | none |
| `REKARINA_MEMORY=import`, no shadow | 12.0 | none | none |
| `REKARINA_SHADOW=1 REKARINA_MEMORY=import` (the Deck's path), before write-back on demand | 18.4 | 2.5 ms | ~12 ms |
| the same with write-back on demand | 24.3 | 2.4 ms | 2.4 ms |
| `REKARINA_SHADOW=1` (mirrors on top: a discrete GPU under Proton), before | 31.5 | 2.1 ms | ~13 ms |
| the same with write-back on demand | 42.7 | 2.0 ms | 2.3 ms |

**Write-back on demand under shadow storage** (`REKARINA_SHADOW_LAZY`, default on in the Windows build, which Wine and
Proton run; `vk_memory.c shadow_wb_lazy`). The fence reports copied every GPU write into guest memory, ~65 MiB a frame,
12-13 ms of the completion thread, and with the early results fence off the game waits for them; the CPU reads almost
none of it (the default path leaves such writes in the mirror). Now a whole page of a written range of 64 KiB or more
that the CPU has not written since the write was recorded (write-watch bit, or a stamp newer than the record's that no
report gave) and has not read in 240 frames is closed (`mem_lazy_close_run`) instead of copied; its bytes stay in the
shadow, the first access copies them in (`slz_touch`, the access hook of `memory.c`) and the page then stays open and
is written at every fence for 240 frames. A closed page cannot be written by the CPU without being copied in first, so
the CPU's words never wait behind one; a pool's closed pages come in before its shadow goes (`slz_pool_gone`). Float
sums and global-memory writes are never closed. Standing in the village 16,000 pages a frame stay closed and the CPU
brings in about one page every three frames. Five arrivals in the field and the Market with mirrors on top: no lost
device, no live-check report, 75-85 fps. Also in: a range the CPU has not written in 600 frames
(`REKARINA_SHADOW_PLAIN_FRAMES`) takes no snapshot and goes back page by page (`shadow_plain`: a page the CPU wrote since
the record keeps the CPU's bytes, which go into the shadow too); `mem: shadow at frame` adds the merged and plainly
copied bytes and the fence reports' time split into copy, write-watch and stamps.

Import on a discrete card reads every pool over PCIe (GPU 29.6 ms a frame); a Deck's GPU shares the memory. Debug
switches: `REKARINA_SHADOW_VERIFY=1` (after each pool's sync, guest memory against the shadow outside pending writes and
pages written again since the query: a CPU write the sync missed), `REKARINA_SHADOW_TOP=1` (the pools that take the most
copied bytes), `REKARINA_SHADOW_PENDING_COPY=0`, `REKARINA_SHADOW_MERGE=0`.

Open:

- Fixed the same day: **with mirrors on top** (`REKARINA_SHADOW=1` alone) the scene was flatter and greyer: the sun's
  cascades held fewer casters, the light shafts were missing and the frame's occluder draw (a 256x144 depth pass)
  was not drawn. The early results fence (`vk_sync.c vk_sync_results_on`, on only with mirrors) told the game its
  frame was done at the results point, while shadow storage puts the GPU's writes into guest memory at the
  submission's fence: the game read its culling and occlusion results before they arrived. It is off under shadow
  storage now (the log says `vk: results before the frame's end: off, shadow storage copies the GPU's writes back at
  the fence`), and a frame dump with mirrors matches the default path too; 31.5 fps (34 with the early fence and the
  wrong picture), 42.7 with write-back on demand (above). Ruled out on the way: the pointer table
  (`REKARINA_GMEM_MIRROR=0` gave the same picture) and reporting the mirrors' write-backs of pointer stores at the fence
  (made the fog heavier).
- Nothing has run under Wine, Proton or on a Steam Deck: everything above is the memory path forced on Windows.
- 4,500 shadow buffers hold 5.0 GiB of host memory beside the guest's own: 4.0 GiB of it in the 733 pools a binding
  resolves (one of them the 2.4 GiB pool at 0x1bf470000 that holds meshes and textures), 1.0 GiB in the 3,770 pools
  only the sync before loads through addresses reaches. Leaving the latter without a shadow would save 1 GiB at most.
  The working set standing in the village: 11.9 GiB on the default path, 16.5 with shadow storage; the default
  path's 6.2-7.5 GiB of VRAM come out of the same memory on a Deck. A 16 GB Deck needs the game's own footprint cut,
  shadow or not.

### Still open: light lost in screen tiles for a frame or four while the camera turns

The player's save in the forest (by the boulder and the sign), the camera turned left and right with the stick:
for one to four frames the sun patches on the ground vanish or rectangular tiles of the screen go dark, then come
back; the player sees shadows "drawn again" and grass flickering. With the camera still, 61 consecutive full-size
frames differ only where the fairy is. Counted as whole-picture brightness steps in a thumbnail of every frame
(scratch `steps.py`; a +x/-x pair on neighbouring frames is a one-frame flash): 5 to 12 in 30 s of turning in the
base picture, also with DLAA, at the console's frame step (`REKARINA_REALTIME=0`, 60 fps), and with each of these off:
the released-in-use rule, images made ahead, the texture pool's short budgets, write-back on demand, uniform
snapshots, the results fence. `REKARINA_QUEUE_INTERLEAVE=0` raises the count to 53; `REKARINA_ASYNC_COMPUTE=0` left one
flash in 28 s. So it is an ordering race between the game's two queues that the interleave mostly closes. The tiles
suggest the tiled light lists (written by a compute pass, read by the lighting pass) as what is read too early;
that is a guess. While turning, the game makes a square R11G11B10F target of a new size every frame at one address
(1417 to 1455 pixels here, 70 to 90 alias entries a frame).

One part of it is found. The submission's uniform snapshot is taken when its replay starts and a mirror is synced
at its first use, while the game thread that writes the frame's late constants (the light grid of the clustered
lighting) runs beside the replay. When that thread signalled while the commands before the wait were being replayed,
the wait found its word met, nothing was taken again, and the lighting read the blocks as they stood before the
signal. Taking the copies again at that wait is too late the other way (the game writes the next frame's blocks by
then: objects black or missing for up to 19 frames in a test), so the waits for a word the CPU signals are now
waited for before the copies are taken (`vk_exec_prewait`, `REKARINA_EVENT_PREWAIT=0` for the old order): 27 to 52 such
waits in 600 frames, 0.1 to 0.5 ms each. Steps per run before: 12, 11, 12, 13, 10, 7, 11, 12; with it: 0, 2, 8, and 2,
6, 4 in runs that also changed something else. The frame rate is unchanged. What is left is not in the uniform
snapshot: leaving out the ranges the GPU wrote (`REKARINA_USNAP_GPUW=1`) made it worse, and without any snapshot
(`REKARINA_UBO_SNAPSHOT=0`) one to three one-frame flashes a run remain.

The same session's log names the freeze at the first sword hit on a bush: a compute pipeline built at its dispatch
took the replay thread 524.7 ms, and the next line is a new shader answer record: a variant this machine had never
seen, which no prebuild could hold. Compute dispatches had no "draw without it for now" path as draws have.

They have one now. A compute pipeline that is not in the table is built on a thread of its own
(`vk_pipeline_compute`, "a compute pipeline no prebuild made"); the dispatch waits up to 4 ms, which a build from
the driver's cache always makes (0.2 to 1.3 ms), and after that the dispatches of that shader are left out until
the pipeline is there. A compute shader that is not translated yet goes the same way (`gfx_shader_get_async`, as
the draws' shaders do). Checked with `REKARINA_CS_KEEP=none` (no compute pipeline prebuilt, every one built at its first
dispatch): the tour runs at 91.4 fps against 91.8 with the builds on the replay thread, whose frames of 26 and 6 ms
of pipeline build are gone; with every build made 400 ms long (`REKARINA_CS_BUILD_DELAY_MS=400`, 181 builds, 1,424
dispatches left out, the culling's among them) the game still runs the tour to its end. `REKARINA_CS_ASYNC=0` builds on
the replay thread as before. Draws that are drawn once still wait for their pipeline (up to 10.6 ms in the
player's session, two in one frame), and mesh and task shaders are still translated on the replay thread.

The other part of the tile flicker is not found. With the camera turning without a pause (scratch `loc/user_spin.txt`,
`pulses.py`: frames whose brightness leaves the median of their neighbours) the default build shows 3 to 11 flash
events a run. `REKARINA_WB_MERGE=0` raises that to 149 and 177, so it lives where the CPU and the GPU write the same
words; `REKARINA_MERGE_ALL=1`, `REKARINA_GMEM_MERGE=1` (also with the pages' host copies taken when a piece becomes hot),
`REKARINA_FLUSH_GMEM_CUT=0`, `REKARINA_RESULT_GUARD=0` and `REKARINA_PAR_CHUNKS=0` change nothing that one or two runs can show;
`REKARINA_GMEM_WRITEBACK=1` leaves 2 to 3 milder events and plays the title movie at a third of its speed. The event
waits are not it: every wait is for a value of its own frame (`REKARINA_EVLOG2`).

The large flashes have a marker. In every run the frames that drop by 8 to 23 (out of 255, the picture's mean) for
one or two frames, the ones with dark screen tiles among them, are the frame after, or the frame of, one particular
kind of frame: 45 clears where the others have 23 to 25, 22 memory conditions true and 19 false where the others
have 9 and 7, twice the draws seen and twice the GPU conditions, in seven submissions as always (the flight
recorder's `frames.tsv`, scratch `clears.py`). The numbers are the same every time, at the console's frame step as
well, and such frames come at about the same angles of the turn in every run. Their pass list (`REKARINA_PASSLIST=40`)
has four 640x360 depth-only passes of two draws each and one 640x360 colour pass that the frames around them lack;
every transient render target after those lies at another address in that frame, and the replay makes 16 to 54 new
images for it (`REKARINA_ALIASLOG_EMPTY=1`: most start with no content, the 1600x1600 shadow layers among them). What
goes wrong in or after such a frame is not found. Not it: draw stretches recorded in chunks (held for such frames,
the flashes stayed), conditions against a zero word decided on the GPU (`REKARINA_MEM_COND=2`), the alias rule (new
images already take the live image's content where the formats allow). A dump of the frame after such a frame
(`REKARINA_DUMP=t99999999 REKARINA_DUMP_AFTER_CLEARS=44`) was taken eight times and never of a dark one. The smaller
events (2 to 6) are there in every variant and may be the measure's own floor.

## The night of 2026-11-14: lookups as of a stretch's place, the seed of pipeline keys, what the flashes are not

Where it was measured. The flicker hunt's place is the save by the round boulder and the sign in the forest village
(the player's later save stands at a tunnel at night, where the camera's turn passes through a tree trunk and the
game takes the trunk out in one frame: `pulses.py` counts that as a flash, in every build). The game keeps a copy of
the save folder from before each write (`profile-try/save-backup/<time>`), so the boulder save is a copy of the
profile with that folder put back. A series compared while something else loads the disk is worth nothing: one
control series ran a fifth slower under a folder scan and "showed" twenty events.

Chunks look textures up as of their stretch (`88020b1`, `REKARINA_CHUNK_ASOF=0` for the old way). Five runs each at the
boulder save, the camera turning: of the frames with 45 clears, 1 of 23 was followed by a dark frame with it and 4
of 26 without (all four in one run, the screen at a quarter of its brightness for six frames). The single flashes
are untouched by it: four to seven events a run either way.

What a flash is, seen in thumbnails of 960x540 (`FLIGHT_THUMB_W` / `_H` raised for one build; the dump thread then
keeps only a sixth of the frames): for one to four frames the opaque things of the scene (the rock wall, the sign,
the ground) are unlit, nearly black, while the hero, the fairy, the fog and the sky are as always; in the frames
after the worst one, rectangles of the screen along its top edge stay dark for two or three frames more.

Not the cause, each with runs at the boulder save:
- The samples-passed predicates of the lit pass. `REKARINA_PRED_ALWAYS=1` renders what a GPU predicate would decide and
  the flashes stay (four runs: 8, 3, 2 and 4 events); `REKARINA_OQSTAT=1` counts per submission the reports and queries
  that came out zero, and a flash frame has as many as its neighbours (about 200 of 330). Only the frames with 45
  clears stand out there: four times the reports, 76 to 90 % of them zero.
- Stale bytes copied into the light lists. `REKARINA_UPLOG=<pool>` logs every range copied from guest memory into a
  pool's mirror: the 9.5 MiB list pool (0x25ea76000 at this save) takes one whole copy at the load and nothing while
  the camera turns.
- The light grid. Every dispatch that binds the list pool logs the grid words of its uniform block
  (`REKARINA_LISTLOG=<pool>`): the same in a flash frame as in the frames around it.
- `REKARINA_COND=always` cannot answer anything: the frame goes white, as noted before.
- The per-pass probe (`REKARINA_PASS_PROBE=1`) slows the game to 18 fps and the flashes do not come.
Still open, read from the code and not tested: a draw that a chunk of a stretch does not record (its pipeline or
shader not ready) leaves its query empty, the count is an exact zero and the lit pass leaves the object out, where
the replay thread alone would have marked the segment inexact (`vk_exec_oq_draw_lost` returns at once inside a
chunk).

Pipeline keys with the release. A stage library costs 5 to 7 ms to make even when the driver's disk cache holds it
(`--warm-shaders` in the test folder: 42,147 variants in 17 s on 16 threads, vertex 4.7 ms and fragment 6.6 ms
each), and a draw that must not be skipped waits for it on the replay thread: the player's stutter at the first
sword hit on a bush was two frames of 50 and 76 ms with four such draws each. The key file is what makes them ready
before play (the prewarm builds every recorded pipeline in a second or two), and a new player's is empty. A release
now carries one: `tools/make_seed_keys.py` merges key files into `tools/data/pipeline-keys-oot.bin.gz` (4,397
records, 262 KB: shader input ids and render state), `tools/package_oot.py` unpacks it beside the game as
`pipeline-keys-seed.bin`, and `keys_seed` (vk_pipeline.c) adds its records to the prewarm. The first 50 s at the
boulder save without any keys: 665 libraries built in play, 7 draws waited; with only the seed: none and none,
3,445 pipelines built at the start in 1.3 s. What nobody has played through still builds at its first draw.

Also seen in the player's session log and not followed up: `vmaBindImageMemory2` of an image onto a pool's 305 MiB
allocation took 3 to 13 ms 55 times in six minutes (`HRTIME bind`), and several frames of 25 to 37 ms have the
replay as their whole length with no pipeline wait logged.

## 2026-11-15, night: the dark frames after a layout change, and the eyes

**The whites of the hero's eyes** were cyan: a translator bug, VOTEU's vote written to a lane predicate instead of
a uniform one (`docs/oot-xlate.md`, patch 0042). The stand-in images for samplers the game leaves empty were never
filled either (`vk_exec_draw.c make_dummy`); now zeros, and a fill-in texture of the game bound to a sampler of
another dimension gives a stand-in of its colour.

**The dark frames while the camera turns: found and fixed.** A frame dump armed on a dark frame
(`REKARINA_DUMP=t99999999 REKARINA_DUMP_ON_DARK=1 REKARINA_FLIGHT_FLASH=1 REKARINA_FLIGHT_FLASH_DARK=82`: the flash watch's first
dark frame arms a dump of the frame being recorded, which is still dark four presents later; scratch `dkd.sh`)
caught one on the first run. The sun-shadow mask (the 2560x1440 RGBA8 target the lit passes read in slot 21; G is
the cascades' shadow, R a screen-space term) had a right G channel and an R channel holding a quarter-size copy of
the scene in its top-left corner, zero elsewhere. R is `texture(slot 5).x` of the mask shader (fs 155e38dc3b75418a),
slot 5 the last of a chain of R8 1280x720 masks that compute passes (cs 499889ab69f0b9be at 160x45 groups and
c3976d929c4eac10 at 80x90) write and read at 0x8004831400, 0x8004921600 and 0x8004741200. In the dark frame the
writer's output already held the corner copy: the R8 1280x720 descriptor at 0x8004921600 resolved to an RGBA8
2560x1440 image (`REKARINA_TEXWATCH=<addr> REKARINA_TEXWATCH_KEYS=1` prints every entry at an address with its whole key and
write order). When the game moves its transient targets (the frames with 45 clears) the R8 descriptors' own entries
are released, and slot_resolve's "content follows the memory" took the newest live entry at the storage of any
size. The chain wrote its 1280x720 grid into the corner of a four times larger image and read it back with
normalised coordinates, the mask's R went to zero, and every lit object lost its sunlight for 1 to 20 frames.

Fix: a slot follows another entry at its storage only when that entry's image fits the slot's descriptor: the same
width, height, depth or layers, sample count and texel size and block (`vk_texture.c fits_slot`,
`newest_writer_fit`; `REKARINA_FOLLOW_ANY=1` gives the old rule). An image of another size there is the game reusing
the memory. The texture pool resolves every registered slot, the stale ones too, and with the rule it kept about
2,000 released targets "in use" past the released list's limit; the sweeps dropped them and the pool made them again
in bursts, frames of 50 to 68 ms (`REKARINA_SECTIONS`: TP_UPDATE 42 ms waiting 37 ms for TEXREG_LOCK). So a pool lookup
no longer counts as a use of a released entry (`touch`), and when it finds no entry that fits it makes no image: a
draw that binds the slot does (`no_make`). Earlier the same night a narrower rule went in first (a written
descriptor follows a newer write of the same shape, `same_shape`, `REKARINA_FOLLOW_SHAPE=0`); it stays, harmless.

Measured at the boulder save with the camera turning, five runs each (`q116b.sh`): flash events per run 6 to 12
before, 0 0 0 0 0 with the fit rule and the pool change; dark frames after a 45-clear frame 7 of 31 before, 0 of 31;
mean 112 to 122 fps, 1 % low 61 to 70, frames over 25 ms 1 to 6 per 15 s, as before. Without the pool change the
rule alone also gave 0 events but 8 to 10 frames of 50 to 68 ms per run.

`tools/prepare_shaders.py code_hash` writes the new tag of patch 0042 too, so the release's seed programs match the
runtime's keys; the pipeline key seed (`tools/make_seed_keys.py`) must be made again before a release, since the
393 programs' keys changed.

**On the integrated Radeon (AMD's Windows driver, RDNA 2 branch), 2026-11-15:** with `taa = dlss` and DLSS on, NGX
fails to start there (`VK_NVX_binary_import` missing) and the new fallback takes AMD FSR: `dlaa: DLSS cannot run
here: AMD FSR takes its place`, `ffx: upscaler versions on this device: 3.1.6, 2.3.4` (no FSR 4 on RDNA 2),
`FSR upscaling at 1707x960 -> 2560x1440 (3.1.6)`. The village with trees, bushes, fern and shadows is drawn whole
(the short grass of the task-shader draws stays left out on that driver, `REKARINA_MESH_TASK`). 4.4 fps on its two
compute units: far below the minimum, no measure of a discrete Radeon.

## Mesh draws on GPUs without VK_EXT_mesh_shader (2026-11-15: the design, then what was built)

On a GPU without task and mesh shaders (GTX 9xx and 10xx, Radeon before RDNA 2, Intel before Arc, the iGPUs of
older laptops) every mesh draw is skipped (`vk_exec_draw.c`, "a mesh draw on a GPU without VK_EXT_mesh_shader task
and mesh shaders: skipped"): no tree crowns, bushes, fern or grass, and none of their shadows. What emulating them
would take, read from the translated shaders of one frame (25 mesh and 5 task shaders in a frame dump, 2,006 and 80 in
the cache):

- The translator writes every mesh shader the same way: the NVN program runs on a shared array, and one epilogue at
  the end of `main` copies it out: `SetMeshOutputsEXT(128, n)` (n from a shared word, at most 64), then per vertex
  `gl_MeshVerticesEXT[i].gl_Position` and up to ten `mesh_attrN[i]` (vec4, locations 0 to 11), then the triangle
  indices from shared bytes. No per-primitive outputs, no culling flags. Workgroups of 32, at most 128 vertices and
  64 triangles.
- So a compute variant needs only that epilogue changed: the same stores into a storage buffer at
  `meshlet * (128 vertices * (1 + attributes) vec4 + 64 * 3 indices)` and the counts into a word per meshlet. In the
  SPIR-V: the execution model `MeshEXT` becomes `GLCompute` (the local size stays), the output arrays become
  `Private` arrays of the same type (their access chains keep working), and before the final `OpReturn` a loop
  copies them out; `OpSetMeshOutputsEXT` becomes two stores. The translator itself could emit it instead (patch +
  `SAME_AS` for the unchanged modules), which is less fragile than editing SPIR-V in the runtime.
- Task shaders: the payload (`taskPayloadSharedEXT`) and `EmitMeshTasksEXT(x, y, z)` become a store of the payload
  and the three counts per task workgroup; a small dispatch turns the counts into the mesh dispatch's indirect
  arguments and a table task -> first meshlet; the mesh variant reads its payload and its `gl_WorkGroupID` from that
  table instead of the built-ins.
- The draw: one vertex shader for all of them, reading `gl_Position` and the attributes from the buffer, non-indexed
  over `meshlets * 192` vertices: vertex `v` is triangle `v / 3` of meshlet `v / 192`; a triangle past the meshlet's
  count goes out degenerate (all three vertices at one point), so no compaction pass is needed; the count of meshlets
  comes from the dispatch (an indirect draw). The fragment shader stays; the vertex shader writes the same
  locations. There is no shader compiler on the development machine: the vertex shader has to be written as SPIR-V
  words (or glslang fetched with the owner's permission).
- Memory: 128 x 11 x 16 B + 768 B = ~23 KiB per meshlet; the grass is ~40 indirect draws of up to a few thousand
  meshlets each, so a ring of tens of MiB per frame, on cards that have 2 to 6 GiB.
- Testable here: `REKARINA_MESH_EMULATE=1` on the RTX 5070 against the real mesh path, frame dumps compared with
  a local comparison script.

### Built (2026-11-15, afternoon): mesh shaders without a task

What runs now differs from the design above in a few places:

- The translator emits the kernel (patch 0043, `MeshStageWrapper.GenerateAsCompute`): the job's `"as_compute"` asks
  for it, `res.json` names the stage `msc` with the vertex layout (`vec4s`, `max_meshlets`, the locations). It writes
  straight into one 64 MiB emulation buffer through a 64-bit address (`PhysicalStorageBuffer64`) that the module holds
  as two marker constants (`VK_MESH_EMU_ADDRESS_LO/HI`); `vk_pipeline.c meshc_module` puts the buffer's address in
  their place before the pipeline is built. The buffer: the draw's `VkDrawIndexedIndirectCommand` at byte 0 (written
  by workgroup 0), the vertices from byte 256 (position, then each attribute in location order, a vec4 each), the
  indices from 48 MiB (a triangle past the count, or naming a vertex past the meshlet's 128, is degenerate). A draw
  with more meshlets than fit (6,553 at three vec4s a vertex, 2,978 at eleven) loses the rest.
- The draw is indexed, from that header (`vkCmdDrawIndexedIndirect`), through a vertex shader written as SPIR-V
  words in the runtime (`gfx_shader.c meshc_vs_make`), with the game's fragment shader (`vk_pipeline.c
  stages_create` mode 2: a static vertex input of n vec4s, a triangle list).
- `vk_exec_draw.c mesh_emulate`: per mesh draw, a dispatch (direct, or indirect from the same `{count, 1, 1}`
  copies the mesh path uses), a full barrier, the draw. Draws stay out of the parallel recording stretches
  (`vk_exec.c par_is_draw`). `REKARINA_MESH_EMULATE=auto` (on where the device lacks `VK_EXT_mesh_shader` and has buffer
  device addresses), `1` forces it, `0` skips such draws as before.
- The kernels' cache key carries `MESHC_REVISION` (`gfx_shader.c build_code_hash`); the translator identity stays
  (`SAME_AS`): 785 shaders of the forest translated byte for byte the same with the old and the new translator, cold
  caches on both sides (the 26 `.res.json` that differ name another job key, which depends on the run).

The streaks: forced on the RTX 5070, the first runs drew long sky-coloured slivers fanning from the fern by the forest
sign. Bisected with `REKARINA_MESH_EMU_SKIP` (key prefixes) to one fern shader, whose kernel body decompiled identical to
its mesh shader. Cause: its draws are indirect, and the game's own culling writes a count of 0 for the hidden ones; no
workgroup ran, nothing wrote that draw's arguments, and the draw took the previous emulated draw's header and indices,
reading that draw's vertices at its own stride. The header is now cleared before every indirect dispatch (a 20-byte
fill and a transfer-to-compute barrier). Dropping out-of-range indices (revision 2) came first and was not the cause;
it stays, as the mesh stage drops such triangles too.

Measured forced on the RTX 5070, forest entrance: GPU 8.19 ms a frame, 84.7 fps (real mesh shaders: 9.77 ms with the
grass, 91.8 fps): the CPU pays for a dispatch, a barrier and a draw per mesh draw. Picture: crowns, bushes, ferns as
with mesh shaders.

### Draws with a task shader (2026-11-15, evening): the short grass

A task shader becomes a kernel too (stage `tsc`): task workgroup t keeps its payload (output words from 0) and its launch
count (output word 1, at most 64 here) in the emulation buffer, and workgroup 0 writes the mesh kernel's dispatch,
{64, tasks, 1}. The mesh kernel under a task runs workgroup (j, t) for launch j of task t: idle past the task's count
(returns at once), else it copies the payload into its stage memory (the prologue then puts j in word 0 and the input's
word 0, as on the console), runs, takes the next meshlet slot from a counter and raises the draw's index count to cover
it (atomic max), so idle launches take no room. The runtime zeroes the arguments and the counter (64 bytes) before the
task kernel and dispatches the mesh kernel indirectly from what the task kernel wrote. The buffer's layout moved for it
(launch counts from 4 KiB, payloads from 64 KiB, vertices from 8 MiB; `MESHC_REVISION` 3). Translator identity kept
again (`SAME_AS`; the same 785 forest shaders byte for byte the same, cold caches).

Forced on the RTX 5070 in the forest: the short grass as with mesh shaders (side by side at 1:1), GPU 10.33 ms a frame
and 89.6 fps against 9.61 ms and 97.5 fps with mesh shaders. Still refused: points and lines, and per-primitive
attributes (one shader in the forest; its draws are skipped and said once). A task launching more than 64 mesh
workgroups loses the rest (not seen in the forest). Not run on a GPU that lacks mesh shaders.

## 2026-11-15: FSR 1 on the 3D scene (`taa = fsr1`)

Asked for by players with a Radeon RX 6000 and handhelds: the scene drawn smaller and scaled up by AMD FSR 1, which costs
the same on every card and needs no motion vectors. Before this, a smaller scene (`render_scale`) was brought back by the
game's own scaler, which leaves it soft, and FSR 1 ran only at the present, on the finished picture.

`taa = fsr1` (menu: Upscaler, "AMD FSR 1") puts FSR 1 where DLSS and FSR 3.1 run (`vk_dlaa.c` evaluate, at the game's
scaling dispatch): the DLSS / DLAA row switches it on and the DLSS mode sets the scene's share (quality 67 %, balanced
58 %, performance 50 %, ultra performance 33 %); at the dlaa mode it does nothing. `vk_fsr.c vk_fsr_scene` records four
compute passes on the replay's command buffer: an encode of the light target into 0..1 with the reversible tonemap
c / (1 + max(r, g, b)) at the scene size (EASU and RCAS expect a picture in 0..1, the light target holds linear light
above 1), AMD's EASU to the picture's size, RCAS at the sharpness setting (none at 0), and the decode c / (1 - max) in
place. The encode and decode are SPIR-V written by `tools/make_fsr_scene.py` (this machine has no GLSL compiler); EASU
and RCAS are the present's shaders. No motion vectors are made (unless frame generation needs them), no reactive mask,
no NGX or AMD library is loaded. The texture bias is AMD's figure for FSR 1, log2 of the scene's share (-0.58 at
67 %), one level less than the temporal upscalers take. The overlay's "3D scene" line names the upscaler.

The camera must not be jittered: DLSS's projection jitter (`vk_dlaa_scene_jitter`, mode 2, and `vk_dlaa_pjitter`)
ran on every frame whatever `jit_ok` said, and with nothing to resolve it the picture shook visibly in the first test
runs. Both return under FSR 1 now; consecutive frames of a still scene then differ only by the fog's motion
(mean absolute difference 0.6-1.7 on the sign and rocks, native 1.9-2.9).

Measured on the RTX 5070, forest entrance, standing, second half of a 60 s run (`REKARINA_TRACE=perf`, GPU time per frame):

| Mode | Scene | GPU per frame |
|---|---|---|
| native | 2560x1440 | 9.77 ms |
| FSR 1 quality | 1707x960 | 7.92 ms (-19 %) |
| FSR 1 performance | 1280x720 | 7.53 ms (-23 %) |

The game's own scaler at the same 67 % costs the same as FSR 1 and is visibly softer (sign letters, fern edges, at 2x
zoom). In the handheld picture mode (`operation_mode = handheld`, the Deck profile) FSR 1 scales 960x540 to the 1920x1080
picture in its corner, as the game's scaler did. Not measured on a Radeon or a handheld yet.

## 2026-11-15: the mouse camera, raw

The mouse drove the right stick (stick emulation): the game's curve (powf of the stick's length, exponent 2) and its top
turn rate (1.78 degrees a frame at 60 fps, about 107 a second, measured from the view matrix with scripted stick steps)
made slow moves vanish and fast flicks slow. The game reads no USB mouse in play (`GetMouseStates` is never polled), so
the camera itself had to be found:

- Where the stick goes: the right stick was held at a distinctive value, guest memory scanned for it (8 float copies,
  value / 32767), and a read watchpoint on each (`REKARINA_HWWATCH_RW`, the profiler's debug registers) named the readers.
  A static trace (thanks to the second session) found the camera's stick update, main+0x37c300 (dead zone, the powf
  curve, the options' invert flags, a lock timer at +0x5e0; x at +0x538, y at +0x53c).
- Where the angle lives: floats that moved with the view's yaw between a still frame and a turned one (radians,
  degrees, binary angles, either sign, offsets of a quarter and half turn), then each written by itself: only one turned
  the camera and stayed. The input object's +0x52c/+0x528 are copies the game makes each frame; the state is the float
  at +0xdb4 (yaw, degrees, view yaw + 180) and +0xd9c (pitch, degrees, down positive) of the camera +0xa80 points to.
- The hook (set `mouse_direct`): main+0x37c45c (`str s1, [x20, #0x538]`) calls the host, which stores, then adds the
  counts to both angles (0.08 degrees a count, as TotK's orbit camera), only while they equal the copies; main+0x37c4f4
  marks the frame as camera input in the update's flags. Without stick input the game eases the pitch back to rest
  (about 0.1 degree a frame); a vertical stick of 0.005 with the mouse's sign on the frames it moves the pitch stops
  that, as a stick input does.
- Checked: 375 counts turn 30.0 degrees, -750 turn 60.0 back, 2250 a 180-degree flick in one frame; 210 counts down
  16.8 degrees and the pitch holds; the stick still turns the camera. Not yet tried by hand with a mouse, nor in the
  first-person modes (aiming, the slingshot): if those do not go through main+0x37c300 the stick emulation takes over by
  itself 150 ms after the hook last ran.
- By hand (the player, same day): jerky at first, because a 1-degree test of the angle against its copy failed every
  other frame of a steady mouse; now 10 degrees plus the last turn. Swimming: the camera did not turn at all. The
  log said the angle was not where it was found, so that camera keeps its angle elsewhere; the raw path refused every
  frame and, being the active path, kept the stick emulation off. The angle is now tested before the counts are taken,
  and a camera that does not fit leaves them to the stick emulation (`oot_camera_input`; the log names each switch, six
  at most).
- Swimming, traced (`REKARINA_MOUSELOG=1`, one line per update; a write watch on the copies): the copies are not angles
  kept anywhere but the view's direction, made each frame from the eye minus the target (+0x450..+0x464 of the input
  object, main+0x718670). In the water another camera places the eye and the one at +0xa80 stops: its yaw and pitch
  stay put, and writing them turned nothing (the yaw written from +2 to -113 degrees, the view stayed at 46.98). The
  pitch copy tells at once: on land it equals the camera's pitch within 0.02 degrees, in the water it moves 10 to 15
  away in the first frames. Both angles are tested now (the pitch within 2 degrees plus the last turn), and the stick
  path keeps the mouse until 30 updates in a row fit (each switch had been a jerk).
- The stick path for the mouse in that case goes through the inverse of the game's curve (main+0x37c3b4: under 0.005
  nothing, then ((m - 0.005) / 0.99) squared of about 107 degrees a second), so it turns as the direct path would
  (`input_mouse_stick_curved`, `REKARINA_MOUSE_STICK_RATE`). Measured with `REKARINA_TEST_MOUSE=1000,<frame>` (a steady mouse,
  now fed to whichever path is active): 74-85, mean 80 degrees a second in the water, the direct path's 80. Pitch at
  both limits on land stays on the direct path (the camera clamps to -85 and the copy follows).

## 2026-11-15: every cutscene can be skipped; dialogue speed

The game has a skip of its own (`UIEventSkipScreen`, layout `EventSkip_00`; EventFlow queries such as
`IsEventSkipedMostRecent`). An event cannot be skipped when its setting list holds
`Event/EventSetting/EventSettingComponent/SkipSetting/DisableSkip.engine__event__SkipSetting.bgyml` ({IsDisableSkip:
true}); three of the 109 lists do: the opening in the forest, the staff credits (`Dm_StaffCredit_0005`) and the title scene
(`Dm_Title_0005`). `tools/make_oot_prompts.py` puts that file into every overlay folder with the
value false (the same 56 bytes; the resource size table keeps its entry), so those three skip like the rest. Served in
game (the log names it); skipping the opening itself not yet tried (needs a new game).

Dialogue speed has no setting in the game: no message speed parameter in `main` or in the UI, message and talk
parameter files; the pauses are the messages' own tags (group 5: types 0, 1 and 2, no parameters; `<0.4:>` is a page
break, not a pause). Done the same evening as an option, `fast_text` (Game tab, off by default, at a restart):
`tools/make_oot_prompts.py fast_text` rebuilds every message archive (`Mals/*.Product.*.sarc.zs`, 32 of them: 16
languages, two versions) without those tags into `romfs-patches-fasttext`, and `runtime/romfs.c` serves it when the
option is on. 372,162 tags in all, about 11,600 a language; 5.5 s at zstd level 9 with the original dictionary. Checked
against the dump: every message equals the original without its pause tags, every other section as it was; in game the
item and menu texts and the banner after the menu read as before. Not yet seen in a conversation.

## The night of 2026-11-15/16: a tour of every map, what play still builds, the fault after a map change

### The tour

`REKARINA_WARP_AT` reaches every map now (`runtime/oot_warp.c`: the other outdoor maps, the dungeons in the order the
game reaches them, the side dungeons, the boss rooms; the names are the map's start positions, read from each map's
`Banc/*/<map>_Dynamic.bcett.byml.zs`: actors of type `StartPos` and their `Name`), up to 64 changes a run, and each
warp's log line carries its frame number. A tour run (scratch `w/tourrun.sh`, input `loc/tourA.txt`: the camera right
and left, a few steps, A, a sword stroke, every 10 s) stays 35 s at each place; `w/tour.py` scores each stop from the
section rows (frames from 12 s after the warp: mean, 1 % low, frames over 25 ms with the section that dominated each,
replay and GPU time) and from the stop's log lines (lost device, flashes, stand-ins, pipeline waits, work left out,
pipelines built and shaders translated in play). Two traps: the world-clock form of `REKARINA_SHOTS` gives nothing in
this build (the world clock stays at -0.0), and a screenshot costs its frame ~90 ms, so the shot frames are left out
of the numbers. Warping a child save into a map of the grown-up part (the fortress) turns the hero grown-up for the
rest of the run.

### Work left out in play, and the mesh pipelines' keys

The replay leaves a draw or a dispatch out while its shader is translated or its pipeline built beside it (so the
frame does not wait). That was invisible until now; the flight recorder's `frames.tsv` counts it per frame (`mesh_skip`
and `cs_skip` beside the older `draws_skip_pipe` and `draws_skip_shader`) and the log names the first of each kind in a
frame with its program (`vk_exec_draw.c left_out`). The first tour showed mesh and geometry pipelines built in play at
every arrival: 54 at the forest save's first frames, 42 at a field arrival, 381 frames with a left-out mesh draw in the
meadow. Graphics pipelines had a key file and a boot prewarm for a long time (`pipeline-keys.bin`); the mesh pipelines
(`vk_pipeline_stages`) had none, so a session met each of them again. They have one now: `pipeline-keys-mesh.bin` (the
three stages' input ids, the kind, the state; `mk_note` on each build, `mk_flush` with the graphics keys), queued to the
mesh threads by the boot prewarm (`mk_prewarm`), and a release's seed `pipeline-keys-mesh-seed.bin`
(`tools/make_seed_keys.py` takes both kinds of key file). The next run built all 94 records of the first runs at boot
and left nothing out in play at that place; over the night's runs the mesh pipelines built in play (after the first
300 frames) went 112, 36, 3, 6, then 0 in each of the next four runs. How long such a draw was missing: in the meadow a
mesh shader of 105,896 SPIR-V words took 11.3 s to build with the driver's cache cold for it (another one in the
desert 24.6 s; most take 0.2 to 0.7 s), and its draw was left out the whole time. The player's report of grass that
flickered when they struck it fits a variant of the cut grass first met at a strike.

A pipeline key in the release's seed is only built at a player's boot when the player's shader cache holds its exact
shader variants. The setup prepares the dump's default variants (`tools/oot_prepare_shaders.py`), and a variant with
the answers a real draw gives (texture types, formats) is another one: the tour's places meant 37 to 155 graphics
pipelines built in play per stop, thousands of draws skipped while they were built, and stage libraries of 230-280 ms
on average (up to 3.1 s) compiled with the frame waiting where the draw could not be skipped. `tools/make_seed_answers.py`
collects the answers records of every input id the key files name (graphics, mesh, the compute shaders of
`compute-used.bin`, and the release's own key seeds) into `tools/data/shader-answers-oot.bin.gz` (no program, only its
hash and the answers), and the setup's preparation puts them first among the variants it translates and builds.

### The device lost after a map change, twice in the first two tours

Both reports name the memory of a render target the game had placed in one of its pools (fmt 0x34, R11G11B10F, "pool
gpu": bound into the pool's memory, not memory of its own): a 384x384 one in the Market, its pool freed when the hero
walked out to the field; a 1280x720 one in the desert, its pool freed in play. The image went with its pool
(`vk_texture_range_gone`), and the address lies in no live pool buffer: the pool's memory itself had gone back to the
driver. The "unbound N s before the report" is not the read's age (the earlier trap: the CPU runs ahead, then blocks
until the lost device is reported seconds later).

The mechanism, read from `vk_pool_fini`: the game frees a pool once its own fence says the GPU is done with it, on its
own thread. The images in the pool go through the deferral (stamped with the submission being recorded, so their
objects outlive it), but the pool's memory was freed right there by `destroy_rec`, after a `vkQueueWaitIdle` that
covers only what was submitted. A submission of the replay's that was not submitted yet still named the pool's images
or buffers: one held by the queue interleave until its wait is met, parts recorded beside on the chunk threads, the
early results point that lets the game go on before the GPU has read everything. Its draws then read memory that was
no longer allocated. That fits both reports (at a map change, where pools go by the hundred, and in the desert, where
the game frees pools while it streams) and the earlier ones in the Market burst.

A deferral grace for images, views and buffers (`REKARINA_DEFER_GRACE`, 24 submissions, default) was in the second
run and could not help: the memory went with the pool, not with the image. The fix holds a finalized pool's memory and
buffers (`vk_memory.c` "a finalized pool's memory, held a while", `REKARINA_POOL_QUARANTINE`, default on) until the
scene has drawn 240 submissions and 2 s have passed (at most 20 s; over an eighth of the video memory budget the oldest
go at once). At a map change that holds 385 to 641 MiB for a few seconds.

The exact release came the same night: `vk_pool_fini` hands the record, its buffers and memory to the deferral as a
kind of its own (`DEAD_POOL`), stamped past the submission being recorded and every value handed out
(`vk_sync_reserved`: the asynchronous replay reserves values for submissions it records later), with the images'
grace; mirrored pools (`mirror_retire`) the same way, where they had a completion callback at the next value. Any
submission recorded later resolves addresses without the pool (it left the index at the start of `vk_pool_fini`), so
that stamp covers every submission that can name it. The quarantine stays behind it as the fallback.

### Places that freeze for seconds: shaders nobody prepared

In places the test folder had never seen (the river, the domain, the fortress) the tour built 37 to 155 graphics
pipelines per stop in play, skipped thousands of draws meanwhile, and in the fortress froze for 8.3, 5.6, 2.3 and 1.1 s.
The log's summary says why: 358 pre-rasterization and 385 fragment stage libraries compiled in play, 230-280 ms each on
average (up to 3.1 s), and of the 324 library jobs for draws that wait, 172 were built by the drawing thread itself
because no worker was free. The draws that wait are the "one-shot" ones (`vk_exec_draw.c prepare_draw`: a target the
previous frame did not render): here the faces and mip levels of a 128x128 light probe the game renders on arrival.
A cold driver cache makes every such variant a compile; the fix is that a player's setup has prepared it (the seeded
answers above), not a shorter wait, which would leave the probe without those objects for the session.

### Two traps of the tour itself

A warp asked while the hero is inside a dungeon hung the game (no frame and no log line for minutes, the process
alive): the map change event from our pad hook, not something a player can do. Dungeons and boss rooms get a boot each
(scratch `w/qd.sh`). And the sword is on Y in this game: a script pressing B swings nothing.

### The water beside the drawbridge

The player saw the reflections in the moat on the right of the drawbridge flicker. With thumbnails of 640x360 (the
recorder's size is a setting now, `REKARINA_FLIGHT_THUMB`) and a still camera on the bank, 255 frames of the region with
the bridge's reflection showed no one- or two-frame change (`w/regpulse.py`: the region's difference from the median of
its neighbours, against the run's typical difference). In the same frames a large faint face lies over water and wall:
the hero, faded by the game because the camera is pressed against his head; when the camera moves near him under the
bridge that fade comes and goes, which may be what was seen. The gate's camera on the drawbridge (the right stick does
not turn it there) looks over the moat on both sides for minutes: 5,329 still frames of those regions never changed by
more than 2.7 levels against their neighbours. All by day; a warp brings daytime, so the night was not reached.

The player then said it is by day and the bridge's reflection on the right. The same view in the player's own
settings (DLAA on, DLSS quality, DLSS frame generation, the FSR upscaler, 2560x1440 stretched, the character mask),
640x360 thumbnails of every frame: a still camera (4,960 frames), slow turns on the bridge (4,868) and slow walks along
it (4,787) showed no one- or two-frame change of either moat region over 1.5 levels (the typical frame-to-frame
difference there is 0.4 levels); the test folder's settings (the base scene with FXAA) neither. Those are the rendered
frames. Frame generation makes no frame in a background window, and its frames never reach the thumbnails; it is
given depth, motion vectors and the HUD-less colour (`runtime/dlssg/nx2_sl.cpp`), and the moat's pixels carry the
water surface's motion while the reflected bridge moves as if it lay at its mirrored depth, so a generated frame can
put the reflection in the wrong place: a flicker of the reflection alone. Not shown yet: whether the player's flicker
goes with frame generation off.

The strike runs (the user's save, which has the sword; it is on Y) in the forest grass with thumbnails of 640x360: in a
session whose mesh pipelines were built at boot, nothing was left out at ~140 strokes, and the grass region's only
one-frame changes were the sword's trail. Without the boot prewarm (`REKARINA_PIPE_PREBUILD=0`) the forest's mesh
pipelines were built at the arrival with their draws left out for 1-3 frames each, and still none at the strokes there:
the flicker the player saw at a strike needs a variant met for the first time, which the key file and the release's
seed now build before play.

### The replay's CPU time, and a processor of 4 cores

Base mode at 1440p on the 9700X (16 logical processors), the scratch bench (`w/bench.sh`: the forest from the village
save turning slowly, the field and the Market after a warp, walking then turning; section rows): forest 94.8 fps, 1 %
low 50.1, 7 frames over 25 ms of 7,216, replay 8.27 ms a frame; field 97.5 / 45.2 / 4 of 4,536 / 7.88 ms; Market 93.5 /
41.0 / 10 of 5,099 / 7.80 ms. The same with the process confined to 4 cores and 8 threads (`REKARINA_AFFINITY=0xff`,
which the thread pools size themselves by): forest 87.5 / 45.2 / 14 frames over 25 ms / 8.78 ms. Most of those long
frames have 20 to 50 ms in the submission that no section names: the replay thread waiting, not working (the chunk
threads and the game's own six share eight processors there).

The texture pool's update read every slot's registration twice whenever the game had registered any slot since the
last update, which it does every frame (two passes over 32,768 slots: 0.25 ms of a frame at the drawbridge, 0.38 ms in
the Market, beside the resolving itself). `vk_texture_register` now sets a bit per slot it changed and the update looks
at those slots only (`vk_texpool.c`, `REKARINA_TPOOL_REGBITS`); a budget stop, the gone list or a grown slot count still
make it look at every slot. The images that go mark their slots in the same bits (`take_gone`): the game destroys
transient targets nearly every frame, which by itself made every update a full one.

The render thread's profile in the forest (`REKARINA_PROFILE_RENDER`, 10 s): 37.5 % of the samples in our code, 20.9 %
in NVIDIA's driver, 34.9 % blocked. The hottest line by far is the spin of `vk_exec_par_join`, ~1 ms a frame: the replay
thread waiting for a stretch's chunks. The perf log's "chunk waits asked by" lines say who asks: an inline uniform update
(`UPDATE_UB`) about five times a frame at 0.19 ms each, then texture copies and clears (0.15 ms a frame each). The update
waited because the chunks read the update ring (`upd_read`, for the shaders that load an address from a uniform block)
and the uniform snapshot's regions, which the update changes. They read both as of their stretch's handout now (the
ring's count and the snapshot's update serial ride in the job, as the texture as-of mark does; a region an update drops
carries the update's serial, `vk_usnap_drop_update`), the update's copy goes into the replay thread's command buffer
behind the chunks as a dispatch's already did, and its write-back note is taken under `vk_par_lock`; so it no longer
waits (`REKARINA_PAR_UB_JOIN=1` brings the wait back). The next lines of the profile: the report index probes
(`vk_exec_xfer.c` pidx, ~0.2 ms), the mirror syncs and the snapshot drops.

### A broken heap in a dungeon, and the held rest's snapshot

The first run after the night's merge ended in the first dungeon, about 30 s after the arrival, with an access
violation inside ntdll's heap (reading 0xffffffffffffffff) under `nvnCommandBufferEndRecording`'s allocation on a game
render thread: the heap had been broken earlier by some other writer. The next four dungeon runs were clean, so it
was rare. Reading the night's changes and their neighbours for a write that can pass a block's end found one in old
code that the night made likelier: `vk_exec.c held_resume` records a held rest without waiting for the chunk threads
of the draw stretch in flight ("they read copies of the queue state, and the uniform snapshot through a view of their
own"), and it swaps the uniform snapshot's state with the rest's (`vk_usnap.c held_exchange`: the region array, its
count and capacity, the arena, `U_other`) without `vk_par_lock`. A chunk thread meanwhile may be inside
`vk_usnap_drop` under that lock (a draw binding a buffer it writes), walking `U.reg` up to `U.nreg` and setting each
region's dropped flag. Seeing half of the swap, it walks the held copy, which `vk_usnap_hold` allocates to its exact
count, with the running snapshot's larger count: a write past the block's end. The inline update without the join
(4d2c43e) lets the replay thread run on past an update while chunks work, so it reaches a signal and a held rest's
resume with chunks still busy more often. Fix: the exchange under `vk_par_lock`, and the held copy freed through the
deferral (past the submission being recorded) instead of at once, since the rest's last chunks may still read it
through their view. `REKARINA_USNAP_HELD_LOCK=0` is the old way.

The A/B on the same dungeon stop (the user's save, the tour input, heap checked every 30 presents): with the lock
off, the second run ended with an access violation inside `vk_usnap_drop` itself (+0x1d3, on a chunk or the replay
thread, reading an address past the array the exchange had just swapped in); with the lock, the runs were clean and
the heap whole at every check (a check costs 4.6 ms). The race is real and in that function; a write there breaks
the heap instead of faulting at once.

To catch a heap writer in a test run without a debugger: `REKARINA_HEAP_CHECK=<n>` validates the process heap
(`HeapValidate`, which walks every block under the heap's lock) every n presents and, on a broken heap, logs the first
frame found broken and the last found whole, where a heap walk stopped, the bytes around that place, and which of the
suspects' arrays lie near it (`rt_heap_note`: the snapshot's regions and writes, the pool quarantine, the deferral
list, the retiring mirrors and the pool bits register their address whenever they grow).

### The device lost in the field without a map change

One run of the CPU bench (the field, about 6,000 frames after the arrival) lost the device with a read inside a live
texture's memory (its binding 4.3 s old, no unbind after it); texture eviction had run in that run (512 cold textures
evicted at the arrival). Not the pool class above. An evicted texture's image was destroyed past the submission being
recorded; textures evicted or dropped under memory pressure now wait for the GPU to pass every timeline value handed
out so far (`vk_defer_past_reserved`, the finalized pools' rule; `REKARINA_EVICT_QUIET=0` is the old stamp). Eviction
runs only while video memory is short, so the longer wait costs nothing measurable.

## 2026-11-16: story progress for test runs (`tools/oot_progress.py`)

The game keeps its progress as named game-data values (the TotK engine's game data manager: a value is found by the
MurmurHash3 of its name), and `slot_NN/progress.sav` holds every value that is saved: an 0x20-byte header (magic
0x01020304, version, offset of the data region), then (hash, value) pairs split into typed sections by an entry of hash
0 whose value is the type number (Bool 0, BoolArray 1, Int 2 ... Bool64bitKey 32, TotK's order), then the data region
with arrays (a count, then the elements), fixed-size strings and 64-bit values. No checksum. The RomFS
`GameData/GameDataList.Product.<ver>.byml.zs` (big-endian BYML) lists every value with its type, default and file.

The names of the hero's belongings are not strings anywhere in the game (the code holds their hashes), but the list's
structs give them: `PouchData/<item actor>` holds a count (1 for an owned item, the ammunition otherwise),
`IsGet/<item actor>` a was-got bool (the songs are only here), and `PlayerStatus` the hearts in quarters (`Life`,
`MaxLife`), the magic (`MP`, `MPMax`), the rupees (member 0xc995bbb8), what is worn (`EquipWeapon`, `EquipShield`,
`EquipArmor`, `EquipBoot` and the instrument's, actor names as text), the item on the item button (`AssignableItem/Adult`,
`/Child`, enums) and the bottles' contents. The story is a set of enums, `Step_<quest>`, each running from `NotReady`
through named steps to `Complete`. The hero's age is `Sequence_IsChildhood` together with the map the save continues
in (`<map>_Child` / `<map>_Adult`).

The tool changes those values in place, in every slot of a COPIED profile (it refuses the test folder's
`profile-try` and the package's own): presets `child-all`, `adult`, `adult-early`, single values by name, struct path or
hash (`--set`), quest steps (`--quest`), and `--dump` / `--dump-all` to read them back. It reads the list, the structs
and the member names (the item actors' pack names) from the player's dump; the names its presets need are stored with
their letters moved 13 places. The test runs apply it to the run's profile copy before the start. Verified on
2026-11-16: the village save with `--preset adult` continues as the grown-up hero in the grown-up forest, 20 hearts,
the sword and shield of that age, the pause menu's belongings as set; a warp to the field arrived at night.

### A start that every test run hid

From 929110f to 0fdb6f9 every start without `REKARINA_HEAP_CHECK` in the environment ended at once: the setting was
read with `cfg_int`, which ends the game when a key has no row in `cfg.c`'s settings table, and every test run set the
variable through the scratch run script, so none of them saw it (a player's start, and the other agents' exes, did).
A test setting is read with `cfg_get` or `rt_knob` (no row needed); a player setting gets its row. Each round of runs
now begins and ends with a player's start: the exe in a folder of its own (`src\build\pstart`: links to the exe, the
DLLs, the overlays, the fonts and the shader cache, a silent windowed `rekarina.ini` with the romfs path) and no
`REKARINA_`/`TOTK_` variable in the environment at all.

### The bench after the night's CPU work

The same scratch bench as before (base mode 1440p, 16 logical processors, nothing else of ours running), on 0fdb6f9:
forest 97.8 fps, 1 % low 54.4, 4 frames over 25 ms of 7,404, replay 7.27 ms a frame (before: 94.8 / 50.1 / 7 of
7,216 / 8.27); field 99.3 / 47.6 / 5 of 5,430 / 7.20 (97.5 / 45.2 / 4 of 4,536 / 7.88); Market 96.7 / 48.4 / 1 of
5,271 / 6.97 (93.5 / 41.0 / 10 of 5,099 / 7.80). The replay thread's frame is 0.7 to 1.0 ms shorter: the update without
the chunk join, the pool's registration bits, and the night's smaller fixes.

What the slowest 1 % of frames are now, from the section rows with the submission split once more (`SUB_OFFCPU`: of
the submission's wall time, what the replay thread spent not running, from its own cycle count; `PAR_RUN` /
`PAR_SPIN`: the chunk join): in the forest a third of them are the replay thread off the CPU inside the submission
(+1.9 ms against the median frame's 0.27 ms; the thread runs above normal priority, so it is blocked on something,
not preempted), a few the chunk join's spin, the rest still unnamed. In the field and the Market about 40 % are the
texture pool's update (+4.9 ms in the field, +7.1 ms in the Market: the slots that showed an image before +3.2 / +4.4,
uploads +3.0 / +2.7, images made for the pool +1.6 / +3.2), then the replay thread off the CPU (+2.4 / +5.2 ms).
`RL_TEX`, `RL_UPL`, `RL_POOL` (df5a13e) time the replay thread's waits for the texture, upload and pool locks, to
name the off-CPU part; the old slots' budget has a test setting (`REKARINA_TPOOL_OLD_MS`).

## 2026-11-16: the entry step, the ledger of builds in play, what the setup cannot prepare

### The entry step

Other recompiled ports prepare their shaders when the game is entered; this one did it in part. Since 2026-11-16 the
boot builds everything the player's key files and the release's seeds name before the guest starts
(`vk_pipeline.c`, "the entry step", `vk_pipeline_prewarm`): the shaders they name are loaded or translated from their
stored inputs, then the graphics pipelines (`pipeline-keys.bin`, `pipeline-keys-seed.bin`), the mesh and geometry
pipelines (`pipeline-keys-mesh.bin`, `pipeline-keys-mesh-seed.bin`; before, they were queued to the mesh threads and
went on through the title screen into play, their draws left out until built) and the compute pipelines a dispatch
asked for (`compute-used.bin` and the new `compute-used-seed.bin`; before, they came from the boot prebuild beside play
at below-normal priority) are built on every logical processor, the mesh shaders with the most SPIR-V first. The pool
workers leave the low queue to the step's builders (`K.pw`), so every build in flight is in the step's journal
(`pipeline-prewarm-journal.bin`, three 64-byte entries per builder: input id and stage, the layout of
`shader-prepare-journal.bin`); a process that dies inside the driver leaves it, `skip_init` moves its entries into
`shader-prepare-skip.txt` for this GPU and driver, and the next start drops every record naming one of them. A normal
exit during the step clears it (`vk_pipeline_prewarm_quit`). If the count stops moving for 90 s, the guest starts
beside the unfinished step (`boot.c prep_wait`).

The window opens with a progress picture when the step will take a while (after 1 s if the count says it needs 1 s
more, else after 2.5 s): `gfx_present_prep_frame` clears the swapchain image and the menu's ImGui draws
`ui/prep_screen.cpp` (the band, one khaki row with the amber segment and the count, the description box; texts
`UI_PREP_*` in the 13 catalog languages). `REKARINA_PREP_SCREEN=0` keeps the window closed until the game's first frame;
`REKARINA_PREP_SHOT=<png>` saves one frame of it.

### The ledger

`runtime/prep.c` counts every shader translated and every graphics pipeline, stage library, mesh, geometry and compute
pipeline built after the guest started, by kind and by map (the map is the name of the last `<map>.bcett.byml.zs` the
game opened, `_Static`/`_Dynamic` cut: `romfs.c romfs_open` calls `prep_map_open`). The first 400 get a line each
("prep: frame N, map M: a graphics pipeline in play (waited x ms, build y ms)"); at exit "prep: in play: ..." sums them
by kind with the worst wait and build, and a table follows per map.

### What the setup cannot prepare

The answers seed's records name programs by the hash of their code; the setup's inventory of the dump matched 6,990 of
7,260. Of the rest, 47 were the compute kernels of mesh and task programs (`GFX_STAGE_MSC`/`TSC`: the same program,
hashed with that stage and `MESHC_REVISION`): `tools/oot_prepare_shaders.py` hashes every mesh and task program that
way too now and prepares them. 126 were records of older hash rules (programs with a uniform-predicate VOTEU hashed
before patch 0042's tag, mesh kernels of revisions 0 and 2): the current runtime gives those programs other hashes,
the setup and the entry step pass over them. 23 programs (21 compute) have code that is neither in the archives the
inventory reads nor in the modules: a new install translates them once, in play; their stored inputs then let every
later start build them in the entry step.

### The proof runs (RTX 5070, 16 logical processors)

A player-like folder (scratch, not in the repository): the build's exe and translator, the overlays, the seeds, the
setup's translation of 62,701 variants (55,820 prepared, 723 s on 16 threads; `REKARINA_PREPARE_DRIVER=0` stops
`--prepare-shaders` after the translation), its pack, and nothing else: no `pipeline-cache.bin`, no `nv-cache`, no key
files of its own, so the driver's cache is cold as after a driver update. The first start's entry step took 292 s:
5,228 graphics pipelines (3,093 s of thread time, 0.59 s each), 354 mesh and geometry pipelines (927 s, the longest
22 s), 621 compute pipelines (579 s, 0.93 s each, the longest 10.5 s); the shaders loaded in 0.7 s. A warp tour of
eight maps (forest, woods, meadow, field, Market, castle, village, graveyard; the child's save, 35 s each) then built
98 things in play: the 19 translations of the programs the dump's archives do not hold (the forest's arrival and the
field) with their 18 compute pipelines (one 34 s, its dispatch left out meanwhile), and 19 graphics pipelines, 34 stage
libraries and 8 mesh pipelines of states the tours behind the seeds had not met (13, 18 and 3 of them in the
graveyard, where a one-shot probe draw waited 1.7 s). With the caches warm the step takes about 1.5 s and the
picture does not appear.

### Device losses at textures dropped under memory pressure

Three runs lost the device with a read at an address where only a texture of its own memory had lain, seconds after
it went: the CPU bench in the field (a 1536x552 BC7 texture's place), and two of the other agent's runs in a fresh
player folder with an empty texture cache (the mountain city: 260x260 images; the field: a 576x576 ASTC texture
decoded to BC3, unbound 7.5 s before the read). In those runs the drops of released textures under video memory
pressure ran all the time (40 to 430 per second), and the 8 s hold of what eviction and those drops destroy did not
help: the read came later than that. The drops by age and `sweep_released` have long left a released texture alone
while a slot or a binding still reads it (`released_in_use`, the castle shadow fix of 2026-11-14); the pressure drops
did not. They do now (`REKARINA_PRESSURE_KEEP_USED=0` is the old way). The first two empty-cache runs after it were
clean; the comparison with the old way is in the night's last runs.

A new player's first session with an empty texture cache is rough by itself: 74 and 136 frames over 25 ms in 46 s of
the forest, woods, meadow and field (a warm cache: 0 to 5), every ASTC texture being decoded on the GPU at its first
use and read back for the cache.

The live check in a third empty-cache run (no device loss) found one of that class: a draw's texture binding was
handed, 24 times over a minute, a view that had been destroyed 82 s before, for a live 4x4 texture whose slot
resolved under the current generation. The view came from the view cache (`vk_exec_draw.c get_view`, keyed by the
image handle and the view's parameters), which the purges at a texture's destruction and at a new image's handle
should have emptied of it. Harmless there (the image itself lived), it is how a binding reaches freed memory when the
image behind such a view is gone; not found yet how the entry outlived its view.

With the image serial in the view cache's key (9a9ba6e) the next empty-cache run with the live check reported
nothing (the run before it: 82 reports, all of that one view). Four empty-cache runs since the pressure fix, no
device loss.

The forest's slowest frames with the lock waits named (`RL_*`): the replay thread is off the CPU about 2.4 ms more
than in a median frame; the upload flush's submission of the upload batch is +1.4 ms of it, the texture lock +0.3 ms,
the rest unnamed (the driver's calls). The texture lock costs the replay 0.32 ms in every frame, held by the chunk
threads' lookups that miss the draw memo and by the game threads' registrations: a lever for the mean.

Frame generation in a foreground window with the player's settings: "2 presented per frame", so the generator ran;
its frames are not captured (the player's rule against desktop grabs). The motion-vector debug view clamps negative
motion to black, so it cannot show whether the moat's vectors differ from the bridge's.

The old slots' pool budget at 8 ms in place of 50 (`REKARINA_TPOOL_OLD_MS=8`, same bench): the slowest frames that the
pool update dominates fall by a quarter in the field and the Market, and the 1 % low stays where it was (those frames
are then slow by the replay thread's time off the CPU instead): not worth a stand-in risk, the default stays.

GPU time per pass (drained, fenced marks) at base 1440p: the child field 7.15 ms of marks a frame, the grown-up
fortress 6.28, the grown-up mountain trail at its arrival point 4.89; the port's own GPU work (mirror prologues, merge
dispatches, write-backs) 0.6-0.7 ms in each. The 17-22 ms GPU frames of the grown-up tour belong to views or moments
these runs did not meet.

The dialogue log's black frame reproduces: the user's save, the village, a walk to the gate, then Y opens the
conversation log and the third frame after it is dark (the same in the tour, three frames after Y). The log screen
draws a 640x360 capture of the scene (R11G11B10F) as its blurred background; the dark frame is the one before that
capture is first drawn.

### Guest memory from the GPU driver on Linux (design, 2026-11-16; the platform layer and the pool path are built 2026-11-19, not yet played)

Why shadow storage exists on Linux: the Windows build imports the game's pools for the GPU as host pointers and
watches them with GetWriteWatch. On Linux a host-pointer import is a userptr, which amdgpu (and i915) follow through
the kernel's MMU notifier: every protection change of the write watch (userfaultfd write-protect, mprotect) invalidates
the import, about 1.3 ms per 16 MiB at the next submission, and the import's pinning clears the watch's bits. So every
pool gets a copy (shadow storage): on the integrated Radeon standing in for a Deck, +4.9 GiB of committed memory (21.0
against 16.1 GiB) and the replay doubled (7.9 -> 15.8 ms, of it ~6.7 ms the `pre` snapshots of GPU-written ranges,
named by a render-thread profile with call chains: shadow_pend < note_written < ssbo_info < bind_resources).

The way out is the other direction: memory the Vulkan driver allocates is not a userptr, so the CPU-side protection
of a mapping of it does not touch the GPU's view. Decisions:

- What is backed: each pool, at its registration (nvnMemoryPoolInitialize), not the whole guest heap. Pools are ~5 GiB
  of a 7+ GiB heap; backing the heap would make every heap page driver memory (resident once the GPU references the
  object) and take the write watch off memory no GPU reads.
- How it is placed: VK_EXT_map_memory_placed (vkMapMemory2KHR with VK_MEMORY_MAP_PLACED_BIT_EXT at the pool's address;
  Mesa RADV/ANV/NVK 24.1+, NVIDIA 560+) first; else the memory exported as a dma-buf and mmapped there with
  MAP_SHARED | MAP_FIXED (RADV, ANV); else shadow storage. An opaque fd is not used: lavapipe's carries its data a page
  off (`--memory-test` in WSL showed it), so only a dma-buf's layout is known.
- Re-backing in place: the pool's pages go read-only, a CPU write into them waits in the fault hook until the new
  backing is in, the bytes are copied into the new memory (mapped elsewhere first), the placed map replaces the old
  pages (MAP_FIXED semantics), the pages open. At finalize the reverse: the bytes go into fresh anonymous memory at the
  address, the uffd write-protect registration is made again for it, the device memory is freed past the last
  submission that can name it (the deferral every freed pool already takes).
- The write watch: userfaultfd write-protect is for anonymous, shmem and hugetlb memory only; a driver mapping is not
  one (VM_PFNMAP/VM_IO), so those ranges get the fallback the platform layer needs anyway: a per-page dirty bit, the
  page read-only while clean, a write fault marks it and opens it (linux.c fault handler), and plat_mem_written answers
  for them from the bits and re-protects on reset. Every other protection memory.c asks for (freeze, lazy close, result
  guard) is composed with it in plat_mem_protect (the stricter wins), as Wine does for VPROT_WRITEWATCH.
- Limits: NVIDIA's maxMemoryAllocationCount is 4,096 and the village has ~4,500 pools: pools under 64 KiB stay on
  shadow storage there (a per-pool choice, which vk_memory.c already has in r->smap against r->buf), the rest are
  backed. RADV's limit is far above it. The memory type: host-visible, coherent, cached where offered; on a discrete GPU
  that is system memory read over PCIe, so the existing mirror mode stays on top, fed by the same write watch.
- Fallback and choice: REKARINA_LINUX_GUEST_MEMORY = auto | driver | shadow; auto takes driver memory when the device
  passes the same checks `--memory-test` runs (a placed or dma-buf map at a test address, both directions of reads,
  a write fault on a read-only page, no cost on the next submission after a protection change), else shadow, and the
  log line says which and why. Until the game path is built every value runs shadow storage, and the log line says so.

What is built (2026-11-19). `plat_mem_back` / `plat_mem_unback` (platform.h, `platform/linux.c`) swap a committed range
for memory the caller maps over it, with the range read-only and every writer held in the fault handler until the swap
ends; the pages the kernel's write watch had reported written stay dirty. Backed pages are kept read-only while clean:
the first write faults, sets the page's dirty bit and opens it, and `plat_mem_written` answers from the bits (a reset
closes the pages it reports, closing before clearing, so a write is either in the bytes the caller copies or marks the
page again). A protection from `plat_mem_protect` stays in the state table as the wanted one and the page gets the
composition (`eff_prot`), so freeze, lazy close and the result guard work unchanged (a thaw of a clean backed page
faults twice, the second time to set the bit). `plat_mem_unback` moves a copy over the range with `mremap` and
registers it with the write watch again, protecting the clean pages. `tests/mem_backed_check.c` checks all of it with a
memfd mapping in place of the driver's, including four threads writing counters while the range is unbacked and backed
twenty times (no write lost). `vk_memory.c` `backed_pool` is the placed way: device memory (host-visible, coherent,
cached where offered) bound to the pool's buffer, the guest bytes copied in through a plain map, then mapped at the
pool's address with `VK_MEMORY_MAP_PLACED_BIT_EXT`; `vk_pool_fini` gives the pages back after the queue wait and the
record's memory is freed later as before. A pool that cannot be backed is imported as without driver memory. The
dma-buf way and the per-pool 64 KiB rule for NVIDIA's allocation limit are not built. It is off by default: set
`REKARINA_LINUX_GUEST_MEMORY=driver` (`auto` and `shadow` still mean shadow storage) until a play session has shown the
frame, the saves and a map change are right with it.

The test: `Rekarina --memory-test` (runtime/memtest.c), or built alone with
`cc -O1 -DMEMTEST_STANDALONE runtime/memtest.c -l:libvulkan.so.1 -o memtest`. In WSL (lavapipe only) the placed way is
absent, the dma-buf export fails and the opaque fd is a page off, so nothing here can prove the driver way: a Deck
or a Linux PC with a GPU must.

### Video memory over a long session: free space stranded in VMA's blocks (2026-11-16)

The grown-up tours' slowest stops (castle grounds 43 fps with 21.75 ms of GPU work a frame, the mountain trail 48,
the wasteland 55) came at the end of ten-map runs. From a fresh start with one warp the castle grounds run 138 fps
(GPU 5-7 ms, 5.18 ms of drained pass marks) and the trail 119 fps (6.4-6.9 ms). Over a tour VMA's device-local
blocks grew from 5.9 to 8.6 GiB while the allocations in them stayed between 5.7 and 7.1 GiB: the free space inside
blocks went from 0.2 to 1.8 GiB. A block goes back to the system only when nothing is left in it, and the driver
counts the whole block against the process. The driver's count reached the budget Windows gives the game (10.4 GiB
on a 12 GiB card with the desktop and other programs), and past it Windows moves images out of video memory: the
GPU time of a frame doubles or triples.

What held the stranded space (VMA's detailed map at the end of the tour, main device-local type): 14 blocks under a
tenth full (896 MiB, 854 MiB free) held by 119 sub-allocated textures (37 MiB) and 73 small buffers (4 MiB); blocks a
quarter to half full held 546 textures and 150 buffers. The textures are the CPU-filled ones of 2 MiB or less
(create_tex sub-allocates them; larger ones and render targets have dedicated memory already). The buffers are the
VRAM mirrors of the game's small pools.

What was tried (all behind settings, off by default). Sub-allocated textures in VMA pools of their own
(vk_texture.c tex_pool_for, REKARINA_TEX_POOL=1), apart from the buffers, so that a block of textures can empty; and
a compaction (vk_texture.c compact_step, REKARINA_VRAM_COMPACT=1) that moves the textures of the emptiest blocks: a
new image made with the create info of the old one at a free place of another block of the same pool
(NEVER_ALLOCATE), its texels copied on the GPU in the upload batch, the old image's views and pool slots dropped
(vk_exec_image_gone) and the old image and memory deferred past every value handed out, plus the eviction hold. In a
ten-map tour it moved 1,280 textures (455 MiB) and the free space inside blocks did not fall (it sits in VMA's default
pools, the mirrors' and the non-dedicated images' blocks, where the compaction does not reach), and at the ninth
stop the device was lost: a shader read inside a texture block freed 21 s before, the eviction class of the night of
2026-11-15/16, with the live check silent. It did not repeat: a second ten-map tour and four runs of a stress mode
that moves 16 textures every frame (REKARINA_COMPACT_STRESS, about 820,000 moves) ran clean, with checks for an image
or view handed to the deferral twice and for the view cache used without the texture lock. The compaction stays off.

The live check's one finding in those runs, a destroyed view handed to slot 258 (the predecessor saw the same in an
empty-cache tour), was its own false positive: the slot is a 4x4 fill-in texture whose binding takes a colour
stand-in made during play (vk_exec_draw.c colour_dummy), and make_dummy did not announce the stand-in's handles, one of
which had been a destroyed view's.
REKARINA_MIRROR_OWN_MB gives mirrors from a size up memory of their own (a test setting).

Instruments: the perf trace's "mem: vram frame N by memory type" line (free space inside blocks per memory type, how
many pieces, the largest) and "vk: compaction" lines; scratch tourvram.py turns a tour log into a table per stop
(fps, GPU ms, driver MiB of budget, blocks, allocations, free).

The cause of the collapse itself was found by 22-map tours (2026-11-16, warps every 25 s, grown-up, ending at the
castle grounds; fresh start there 138 fps, GPU 5-7 ms): with the release defaults the last stop ran 23 fps with 41 ms
of GPU work a frame and the driver's count 250 MiB over the budget; with REKARINA_EVICT_HOLD_MS=0 it ran 97 fps with
7.8 ms, the count under the budget at 21 of 22 stops. The 8 s hold of what eviction and the pressure drops destroy
(added the night before against two device losses whose cause was not found) kept the maps left behind in video memory
while the next ones loaded. The hold is off by default now; the device-loss class it was against did not show in that
tour, nor in ~820,000 texture moves without it. The pools of their own and the mirrors' own memory did not help the
tour (its end was 15.7 fps) and stay off.

## 2026-11-16: the texture stream (`REKARINA_TEX_STREAM`, off until measured)

Why. A texture's first use (the pool set's update or a draw's binding, on the replay thread, or a chunk thread
holding the texture lock) hashed its storage (adopt_released and the texture cache key by the hash), read its cache
file from the disk or deswizzled its bytes into staging memory, all inside the frame that needed it. With the cache
prepared at install the field had frames of 300 ms of "texture upload" and one of 2.35 s, worse than decoding on a fast
GPU; with an empty cache the same frames carried the deswizzle and the GPU's ASTC decode. The old upload-ahead worker
(`REKARINA_TEX_PREP`) did the whole upload on its own thread, entries and command buffers included, and lost the device
once in six runs.

Design (`runtime/vk_texture.c`, "the texture stream"). Worker threads (3) take a job per texture storage (address and
size) and never touch an entry, an image or a command buffer: they take the page generation, copy the bytes the
storage shares with its neighbours in its first and last page, hash the storage, and then read the cache file the
upload would read (named by that hash and the descriptor's shape) or stage the bytes the upload would copy (every
region deswizzled at the upload's offsets; an ASTC texture's raw blocks for the GPU decode). adopt_released and upload
take the results where they computed them: the hash, then one recorded copy (or the decode) from the job's buffer. A
result is used only while the storage still holds what it was made from: no page wholly inside it written since the
generation, and the shared edge bytes as copied (the page check alone threw away a third of the field's results: the
game loads the next texture into the shared page). A file or staging of another name, size or descriptor is not taken.
So a job can save work and never change a texel.

Not ready at the first use: a job a worker is running is waited for (`TS_WAIT`, at most
`REKARINA_TEX_STREAM_WAIT_MS`, default 250, then the replay does the work as before); a job still queued is taken back
and done as before (`TS_MISS`). The pool update leaves a slot that shows the stand-in, and whose job is in flight, for a
later update, at most 30 frames after the ask (`TS_DEFER`); slots over the update's budget wait the same way today.
Reads of the game's memory happen under a shared lock that `vk_texture_range_gone` takes exclusively once the pool
left the index, so no read outlasts the memory. Staging held by results is capped (`REKARINA_TEX_STREAM_MB`, 512).

When to ask. At `nvnTextureInitialize` the lead is long (10 ms to 2 s) but three in four textures the game initializes
in the field and the Market are not used within 300 frames (25,604 asks in 70 s, 5,587 hashes taken, the staging cap
full): the ask is made when the game registers the texture in its pool (vk_backend.c, on the game's thread, before
the registration is queued for the replay), and a storage whose result was taken is not asked again while it lies in
the same pool. `REKARINA_TEX_STREAM_AT_INIT=1` asks at initialization instead. Repeats measured at initialization:
603 of 25,604 asks were storage asked again after its job was done, 299 of them with the same bytes.

Instrument. Sections `TS_HASH` and `TS_READ` time the replay thread's hashes and cache file reads (both arms),
`TS_WAIT` its waits for a worker, `TS_MISS` and `TS_DEFER` count textures not ready and slots left for later; a
`perf` line every 60 frames counts the stream's asks, results, takes, stale results and staging held.
`scratch` scorer: tsum.py (per stop: fps, 1 % low, frames over 25 ms, the replay thread's texture milliseconds).

What the old path's runs showed (prepared cache, base 1440p): hashing on the replay thread is small (0.02 ms a frame);
the cache file reads (0.22 ms a frame in the field, 184 ms summed over its 53 slow frames) and the pool update (0.66
ms a frame, 360 ms over the slow frames) carry the cost, and the texture lock the chunk threads hold through such
reads shows as the replay thread's `TEXREG_LOCK` (423 ms over the field's slow frames).

What the stream's counters showed in the field and the Market (2026-11-16, before any of it was a gain):
- The game registers 400 to 570 textures a second while the hero runs through the field (17,021 in 30 s). Six in ten
  are textures its compaction just copied to a new address: adopt_released gives them the old image by the source's
  hash, so a worker's hash and file read for them were wasted. The stream now makes no job for storage the newest
  noted large copy filled (`moved_from`), keeps no bytes for storage whose hash a released entry or a parked image
  carries (a hash set kept by the release list and the image cache), and drops the job of a texture registered again
  while it is made already.
- With those wasted jobs the staging cap (512 MiB) was full for thousands of jobs, whose files the replay then read
  itself beside three workers reading the same disk: one run had the replay's file reads at 1.68 ms a frame in the
  field against 0.43 on the old path. Comparisons on this machine today were also skewed by the order of the arms (the
  first arm after the user's video work met a cold file cache) and by the machine itself (forest-side maps 72-97 fps
  in old-path runs an hour apart); no arm pair so far can be quoted.
- Reading the 11,938 prepared cache files once took 61 s (about 200 MB/s on an NVMe disk that reads far faster): the
  first read of freshly written files is slow here, which is what made the replay's in-frame reads cost 300 ms in the
  prepared-cache run of the morning. Later reads come from the system's file cache and cost 1-3 ms for a large
  texture.
- With an empty cache the replay's `TS_READ` (0.25 ms a frame in the field) was mostly opens of files that do not
  exist yet; the worker now tells upload so.

## 2026-11-16 afternoon: the late read of a destroyed image, image churn, what holds the long frames

### The device loss of the afternoon (not closed)
- Four faults on the installed release build (runtime of 10:16) in the other agent's tours, one on the test build under
  system memory pressure: a shader (the fault's instruction pointer) read an address where an image had lain, 8.6 to
  10.4 s after that image was destroyed. The image classes differ: a 640x360 depth target bound into a pool's memory,
  an unlabelled 1 MiB image, an own-memory 1536x1536 BC7 texture with a dedicated allocation that lived ~35 ms (the
  empty image a moved texture makes before it takes its released image: those come and go every few milliseconds).
- On NVIDIA every image has an address mapping of its own (the binding report shows a dedicated allocation's image at
  the allocation's address, a pool-bound image at an address no pool buffer covers); a read through it after the image
  is destroyed faults once nothing else is mapped there.
- Not reproduced on the test build: 0 in 14 runs over five arms (the budget capped, a forced 9.5 s replay stall at
  every map change (`REKARINA_TEST_STALL`), the empty-cache child path with and without the hold, the released-age drop
  off, the release folder's settings) nor on the installed build in 3 runs of the same path; two torture runs (drop age
  30 frames, pressure line 40 %, budget 4 GiB, 15 stalls: 48,000 age drops, 19,000 pressure drops) found no recording
  of a destroyed image at the live check's sites.
- Mitigation shipped: `REKARINA_POOL_IMAGE_HOLD_MS` (30 s): an image bound into a pool's memory is kept, mapped, after
  its texture goes, or until the pool's memory goes; it frees no memory when destroyed, so the hold costs none
  (measured: no change in frame rate or video memory; 3,600-4,600 images held at most in play, the cap of 16,384 met in
  a torture run). The live check counts a held image as destroyed, so a recording that takes one is reported.
- The fault report (`REKARINA_FAULT_WHO=1`) also prints who destroyed the image (destroy_tex's caller as an offset, the
  entry's state), who handed it to the deferral and what that waited for, and the texture pool's write history for it.

### Image churn in steady play, and the pressure line
- Standing in warm places the texture pool creates 4-6 images a frame (the field 6.0, the Market 4.1, the first dungeon
  4.7): released textures dropped under video memory pressure (above 75 % of the budget, 32 every 8th submission) whose
  slots are still registered are made again by the pool at once, and moved textures make an empty image before they take
  their released one. The bursts of 32 are long frames of 25-36 ms (TP_OLD 13-16 ms with 32 uploads).
- The pressure line at 85 % (`REKARINA_TEX_VRAM_PRESSURE=85`): the Market better, the village and the first dungeon
  worse (base mode, one run each); the default stays 75 %. A test knob that keeps a dropped released texture of an
  evictable class out of the pool, as eviction does (so the pool does not remake it), was measured and is not in this
  runtime: the field and the village got worse in its run (stash "A fifth shift" in the working repository).
- The two replay-next slices (chunk threads' texture checks at the join; the upload batch through the submitter) and
  the texture stream, measured on the five places (base, warm, no instrument, one run each): frames over 25 ms a minute
  summed over the five places 93.5 (slices) and 99.9 (slices + stream) against 91.2 without: not shown better, not in
  this runtime.

### What holds the long frames (sections, warm cache, base and DLSS)
The replay thread, 16-22 ms in the long frames (the GPU 6.5-8.5 ms): the pool update (new slots' uploads and cache
reads, old slots' remakes), the upload batch's submit behind the submitter's queue lock (UP_FLUSH, the replay off the
processor), the texture lock (chunk threads' lookups), draw bindings. The first dungeon's DLSS-mode stutter is the same
list plus an occasional 100-170 ms GPU-side wait near an arrival; nothing DLSS-specific on the frame thread.

## The night of 2026-11-16/17: where the video memory goes, BC1 for opaque textures, what the driver keeps

Measured on the RTX 5070 (12 GiB) at 2560x1440 in base mode with full textures, the prepared texture cache and the
release candidate's folder, a 4-map tour (the forest, the field, the Market, the first dungeon) with `REKARINA_TRACE=perf`
and `REKARINA_VMA_DUMP` (each allocation named by kind in VMA's map), and VK_EXT_memory_budget's count beside VMA's.

### The census (MiB, before tonight's changes, the old budget)

| | title | the forest after 2 min | after 4 maps |
|---|---|---|---|
| the driver's count for the process | 4,145 | 7,791 | 8,853 |
| ASTC textures as BC3 | 1,517 | 2,206 | 2,801 |
| textures as shipped (BC1-7, R16 ...) | 419 | 700 | 1,467 |
| GPU-written with memory of their own | 571 | 729 | 656 |
| mirrors | 62 | 722 | 878 |
| physical pool and its twin | 305 | 610 | 610 |
| free space inside VMA's blocks | 62 | 618 | 717 |
| outside VMA (the driver's own) | 1,200 | 2,098 | 1,614 |

- The start's share outside VMA is pipelines: 978 MiB after the entry step (8,601 graphics, 683 mesh and ~930 compute
  records, ~0.1 MiB a pipeline) and +212-290 MiB after the library prebuild, which built all 4,191 compute shaders on a
  card with a budget of 8 GiB or more. Now it builds the ones a dispatch has used (`REKARINA_CS_KEEP=all` for the old
  way): the title screen 4,145 -> 3,936 MiB, the process's memory after the prebuild 5.4 -> 4.1 GB.
- A destroyed pipeline's device memory stays in the driver's count but is used again for the next pipelines made: a
  standalone test (compute pipelines of one shader made distinct by a constant, no pipeline cache object) gave +34.2 MiB
  for 1,500, still +34.2 after destroying them and after making 1,500 others; 300 / 300 others / 600 others gave
  +10.2 / +10.2 / +18.2. So pipelines built per map and destroyed at the next loading screen would hold the count at
  the largest set alive at once, not at the union of every map visited.
- Freeing memory gives it back at once (a standalone test frees dedicated images after GPU copies: the count returns to
  the byte). What the driver does keep: ~0.25 MiB for every DEPTH image alive (any depth format, sampled or not, bound or
  not; colour images, views, binding into a shared block and first use as an attachment cost nothing), and it keeps the
  high-water mark when they go. In the forest the outside count rose by +0.8-1.0 GiB in two minutes, also with eviction
  off, in steps exactly where the game makes new pool-bound images (up to 250 depth images in 60 frames, each at a new
  address, so each a new VkImage), and plan A's village reached 2.7 GiB outside VMA. The images live on after their
  texture goes because every image bound into pool memory is held 30 s (`vk_mem_pool_image_hold`, the device-loss
  mitigation): 6,144 held at most in a 110 s forest run, 8,704 in plan A, about half of them depth. Holding colour
  images costs nothing; the held depth images are the growth. Two minutes of the forest, outside VMA mean / max:
  1,718 / 2,131 MiB with the hold, 1,237 / 1,301 with `REKARINA_POOL_IMAGE_HOLD_MS=0`; capping the released depth
  entries does nothing while the hold keeps them (1,620 / 2,012); depth targets in memory of their own
  (`REKARINA_DEPTH_ALIAS=0`) remove the hidden part but cost 1.2 GiB more of real memory. So a shorter hold for depth
  images (colour ones can stay held) is the lever, weighed against the late reads the hold catches.
- Depth images are now held 500 ms (`REKARINA_POOL_IMAGE_HOLD_DEPTH_MS`; colour images keep the 30 s of
  `REKARINA_POOL_IMAGE_HOLD_MS`). Plan A (eight maps), one run of each, none with a lost device: outside VMA grew
  +963 MiB over the run with the 30 s hold (driver max 8,426 MiB, 4,687 depth images held at most), +343 with no hold
  (8,009), +409 with 5 s for all (8,061), +207 with depth images held 500 ms (7,986, at most 249 held). A 22-map tour
  with the 500 ms depth hold: no lost device, outside VMA 1,072-1,211 MiB at every stop (+139 over the tour), driver max
  8,272 under a budget of 8,855; its five dark frames were the camera behind foliage, a post or the water's surface.
- Eviction of small textures (sub-allocated, <= 2 MiB) does not lower the count unless their block empties: in the
  forest eviction removed 811 MiB of allocations while the count rose (+619 MiB of holes, +358 outside VMA).
- Blocks of 16 MiB instead of 64 (now the default; `REKARINA_VMA_BLOCK_MB`): 200-390 MiB less free space inside blocks
  late in a tour, no frame-time cost shown in two pairs.
- About 95 % of a map's texture bytes go within a minute of the next map's arrival, but the few that stay and the
  mirrors sit in nearly every block (at the dungeon's end 800 MiB free inside 45 blocks, the Market's last 101 MiB of
  textures in 43 of them, mirrors in 44). `REKARINA_TEX_GEN_POOLS=1` (off) puts small textures in pools by map
  generation, apart from the mirrors, for a bulk drop at loading screens. Without such a drop it is worse (a 4-map
  tour's end: 1,274 MiB free inside blocks against 495, the driver's count +0.87 GiB), and the loading-screen cleaning
  it was written for was not taken (it made arrivals longer), so it stays off.
- The mirrors are buffer pools: a census line every 600 frames over the same tour (an instrument, not kept) found under
  1 MiB of live textures' storage inside 667-1,114 MiB of mirrors in play (428-769 small pools with flags 0x24, 263-695
  MiB; 5-21 with 0x22, 92-112 MiB; the title's 62 MiB held 10). A mirror that left texture storage out would gain nothing.

### The same tour after tonight's changes (MiB, averages over windows, one run each)

Windows: the title = frames 1200-1800; the forest = one to two minutes after its second warp; the end = the last 3,000
frames (the first dungeon). Before: 7fc787f's rules with the old budget. After: compute pipelines by use, BC1, 16 MiB
blocks and the 500 ms depth hold, with the budget that leaves other programs their share (8.85 GiB that night).

| | title | the forest | the end |
|---|---|---|---|
| the driver's count, before | 4,356 | 8,524 | 9,388 |
| the driver's count, after | 3,771 | 7,068 | 7,562 |
| outside VMA, before / after | 1,202 / 988 | 2,018 / 1,222 | 1,806 / 1,159 |
| free inside blocks, before / after | 64 / 20 | 190 / 89 | 971 / 498 |
| ASTC textures, before / after | 1,514 / 1,263 | 2,850 / 2,405 | 2,697 / 2,259 |

In play the textures take what the budget leaves, so the forest's and the end's counts also follow the budget; the title,
the part outside VMA and the free space inside blocks are what the changes alone move. With the cuts and the old budget
(no depth hold yet): 3,865 / 8,372 / 8,910.

### BC1 for opaque ASTC textures

Of the prepared cache's 13.3 GiB of BC3 blocks, 3.9 GiB (29 %) are textures whose every alpha half is the opaque
pattern the encoder writes; the colour half is a BC1 block in its four-colour order, so such a texture is BC1 with the
same texels at half the memory (85.7 million texel comparisons of 80 cache files: no difference on the RTX 5070 or the
Radeon iGPU). The cache keeps them as `.bc1` (play and the setup's preparation write them; an older cache's `.bc3` files
load as before), and a texture whose storage hashes to a `.bc1` key gets a BC1 image, filled only from its file: other
bytes written into its storage refill it from their own `.bc1` file or make it again as BC3 (`bc1_redo`).
`REKARINA_TEX_BC1=0` turns it off. Normal maps (57 % of the cache) keep their second channel in alpha and stay BC3.

## The start and the arrivals, 2026-11-17 (`REKARINA_UPFRONT`, on)

The user's report on 1.2.0: 10-20 s of 10-17 fps from the start, and the overlay showed one thread at work (the
replay's "frame prep" 39-50 ms a frame with 16-22 texture uploads) while the GPU idled at 10 % and the CPU at 19 %. In
their logs the first minute had 213 frames over 25 ms, 101 over 50 and 23 over 100.

What it was, measured with a copy of their profile in a copy of their 1.2.0 folder (prepared texture cache as hard
links; scratch scripts `k/pstart.sh`, `k/cold.sh`, `k/score.py`, `k/secs.py`), launch to the title, 60 s:
- Not a commit of the runtime: the 1.1.0 runtime with the prepared cache behaved as the current one with the village
  save. It is the setup's prepared texture cache (14,646 files, 14 GiB): each texture's first use opened and read its
  file on the replay thread, under the texture lock, 0.7 ms a file from the file cache and 5.3 ms on a cold one (the
  virus scanner's first look at a file; the disk is an NVMe). The start's read of the whole folder into the file cache
  ran beside it for 52-79 s and fought for the same scanner. Without the files (every texture GPU-decoded, as 1.1.0
  started) the same start had 21 / 9 / 3 frames over 25 / 50 / 100 ms.
- Where: the long frames are in the opening that follows the logos (a live 3D scene the arrival rule does not count as
  a scene: fewer than 24 first draw lookups a submission), not on the title menu: the game registers ~900 textures in
  one frame there. The pool's update checked its budget every 16th slot: 16 slots of 2-5 ms held the lock 30-90 ms
  where the budget said 4 or 8, and the game's registering threads waited on it (TEXREG_LOCK 6 s in a cold start).
- The texture stream (2026-11-16, off) did not help as it stood: it made no job for storage a large copy had filled
  ("moved storage", meant for the Market's compaction), and the game loads every texture by such a copy from its file
  buffer, so the start's textures had no job at all (TS_READ with no TS_MISS).

The design (`REKARINA_UPFRONT=0` brings back all of the old behaviour on the same exe):
1. Worker threads do a texture's heavy part, from the moment the game registers it: the texture stream on by default,
   a third of the logical processors (2-6), a job for every freshly loaded texture (moved storage skips the job only
   when the copy's source is a texture adopt_released will find). The replay thread records a copy.
2. Whatever is not ready is waited for out of sight (`runtime/vk_arrive.c`): the frames that begin a scene after a
   loading screen, and those of a burst of new textures while no scene is drawn (the opening fades in from black), are
   made, not shown, unpaced and at the shortest game step; the pool's slots wait for their jobs and resolve with a 40 ms
   budget; the hold ends after 4 calm frames or 2.5 s. The frame-time summaries leave held frames out and the hold's
   log line says how long the last picture stayed up.
3. The pool update looks at its budget after every slot.
4. The start's warm read of the cache folder waits for the first scene shown and runs at background I/O priority.

Before / after (first 60 s, frames over 25 / 50 / 100 ms; one run each unless said):
| case | b775624 | upfront |
|---|---|---|
| user's save, warm file cache | 151 / 75 / 35 and 29 / 9 / 3 (2 runs) | 10 / 2 / 1 and 12 / 3 / 2 (2 runs) |
| user's save, cold file cache | 66 / 22 / 9 | 18 / 2 / 1 |
| the same exe, `REKARINA_UPFRONT=0`, warm | | 27 / 6 / 1 |
The hold at the opening lasted 0.05-0.96 s warm and 0.80 s cold (1,068-1,121 textures made in it). What is left: frame 2
(the boot's first 282 textures), two game-side frames of 85-108 ms around frames 310 and 420 (replay under 10 ms: the
game's own boot work), a few 25-35 ms frames.

## 2026-11-19: weak systems at 1920x1080 (fewer processors, less video memory, a weak GPU)

Measured on the development PC (Ryzen 7 9700X, RTX 5070 12 GB, 62 GiB of memory) made smaller by settings: the picture
at 1920x1080 (`game_res = 1080`), base mode, three places (a town square walked into, the forest village, the open
field), one run of 90-95 s per row unless said. A row is: frame rate, 1 % low, frames over 16.7 / 25 / 50 ms; all
from the shown frames (`REKARINA_PACELOG`).

**What bounds the frame at 1080p on this card.** The village and the field: 112 and 105 fps, with the GPU busy 8.0 and
8.8 ms of an 8.7 and 9.6 ms frame: the card and the game's own cycle meet there. The square: 84-90 fps, the game's
cycle (11.6 ms), the GPU busy two thirds of it. So a card half as fast is the limit at all three places at about 55
fps, and the picture's height changes little (6.6 ms of the GPU's time do not depend on it, `2026-11-14`).

**Fewer processors** (`REKARINA_AFFINITY=<mask>`, `REKARINA_TRACE=sched`, two stream threads as such a processor gets):

| processors | the square | the field |
|---|---|---|
| 16 | 83.5 fps, 52.0, 27 / 3 / 0 | 105.1 fps, 79.3, 8 / 2 / 0 |
| 4 cores with their second threads (0xFF) | 83.9, 55.6, 15 / 1 / 0 | 98.8, 70.4, 17 / 1 / 0 |
| 6 cores without them (0x555) | 83.8, 46.8, 45 / 4 / 0; twice the jumps between frames | |
| 4 cores without them (0x55), before | 64.2 and 61.6, 24.1 and 20.7, 327 / 22 / 1 and 364 / 33 / 5 | 70.6 and 71.0, 22.4 and 22.2, 553 / 107 / 10 and 514 / 123 / 15 |
| the same, cores folded onto 4 slots | 62.5 and 56.3, 38.9 and 32.7, 422 / 15 / 0 and 620 / 48 / 0 | 72.7 and 73.1, 44.5 and 44.2, 556 / 31 / 1 and 551 / 29 / 2 |
| the same, 3 slots | 58.5, 35.4, 568 / 41 / 0 | 73.6, 48.2, 495 / 5 / 0 |

Eight logical processors lose nothing. With four the frame rate falls by a quarter and the 1 % low to 21-24 fps. A
census of every thread's processor time by name over ten seconds there (the square): 333 % of one processor in all
out of 400, the game's own threads 307 % of it (the presenting thread with the replay inside it 68 %, the main thread
28 %, twelve workers 9-23 % each) and every host thread of ours together 26 %. Nothing of ours wastes a processor;
fourteen runnable game threads share four, and the host's scheduler, which knows none of the game's priorities, makes
a woken thread wait out another's time slice: the main thread and the workers waited up to 33-35 ms from their wake to
running (7-11 ms with sixteen processors), the main thread being ready 37 % of the time and running 24 %.

The cure in the tree: with fewer logical processors than the console has cores, the cores are folded onto as many
slots (`kernel.c` `core_allowed` / `core_number`, docs/architecture.md). The waits become waits for a slot, in the
order of the game's priorities and bounded at 5-8 ms; the 1 % low doubles in the field and rises by half at the square
and the frames over 50 ms go, at an equal mean in the field and 2-10 % less at the square. Four processors still do
not hold 60 fps at these places (400-600 frames in 60 s over 16.7 ms either way): the game's threads need three
processors' worth of this machine's cores. Five slots on six processors lost 6 % at the square for nothing, so six
processors and more keep six cores. Not measured: five processors, two cores with four threads, a processor whose
cores are slower than these.

Tried and not kept: the three waits of the draw stretches (`vk_exec.c` `vk_exec_par_join`, `par_run`, `par_thread`)
ending in a yield after their pauses. With six processors the square's jumps stayed (14.1 against 15.6 a second) and
the replay was off its processor longer (1.17 against 0.69 ms a frame).

**Less video memory.** A capped budget (`REKARINA_VRAM_BUDGET_MB`) shows the eviction's work and nothing of a short
card's frame times, so the memory was taken for real: a test program holds part of the card in 256 MiB buffers and
writes a few bytes of each four times a second, as desktop programs hold it; the driver's budget, the adapter counter
and Windows' residency then all see a smaller card. Texture quality as `auto` picks on such a card.

| the card as the game finds it | the field | the square, whole stay |
|---|---|---|
| 12 GB with 4.5 GB held by others, full textures | 106.1 fps, 83.8, 3 / 0 / 0 | 95.1, 57.0, 72 / 14 / 0; again 93.8, 57.4, 69 / 7 / 0 |
| 8 GB (7.2 GB free), half | 106.0, 88.6, 2 / 0 / 0 | 95.3, 58.7, 60 / 3 / 0 |
| 6 GB (5.1 GB free), quarter, budget capped to 4,165 MiB | 85.4, 63.3, 23 / 2 / 0 | |
| 4 GB (3.1 GB free), quarter, budget capped to 2,475 MiB | 73.4, 54.0, 89 / 1 / 1 | |
| nothing held (2 GB of desktop), full | 105.1, 79.3, 8 / 2 / 0 | 89.8, 52.1, 111 / 8 / 0; 93.6, 52.6, 100 / 17 / 0 |

Held video memory alone costs no frames at these places. With 4.5 GB held the game stood 0.3-0.5 GB over its budget of
5.6 GB, and the square's whole stay had 14, 7, 10 and 8 frames over 25 ms in four runs against 8, 17 and 12 with
nothing held; the field had none either way. The square's long frames in which the replay thread is off its
processor for 5-25 ms inside the binding of the dispatches or the draws, with nothing made, uploaded or evicted in
the frame, look like what players report with a browser and an editor open, but they are as many with nothing held
(6, 14 and 8 of a run's 16-20 long-frame reports against 13, 9, 11 and 8): the square's own, cause not found. Making
every texture up to 32 MiB a piece of a block (`REKARINA_TEX_SUBALLOC_KB=32768`, so that evictions free nothing to
the system) did not change their count. So a bad session beside other programs needs what a holder of memory does
not do: those programs working, or a longer way through many maps.

The small cards: with quarter textures the game does not get under about 4.5 GB at 1080p one map in, whatever its
budget says. A 6 GB card holds that and runs even, 20 fps under a card with room, because a 244 MiB buffer pool's
mirror "does not fit the card's room" and stays in system memory, so the GPU reads it from there and the frame waits
for the GPU (present sync 5 ms where it is 0). A 4 GB card is 1.4 GB short of the floor; the run above says what this
PC did with the game's memory then (the holder's may have been moved out instead), not what such a card does.

Found on the way: the images made ahead (`REKARINA_TEX_MAKE_AHEAD`) are idle most of the time beside a desktop. The
game fills whatever budget it gets, so with 2.5 GB of the card held by other programs "room is short" is its normal
state a minute into a map, and the stream's workers make no image then: after the arrival at the square 435 first uses
took an image made ahead and 5,015 did not. With half or quarter textures the feature is off altogether (`ma_make`
returns at `texq_mode != 0`). Letting the workers go on under a cap of 64 MiB while memory is short changed nothing:
nine in ten of those images are of textures no draw binds within 30 frames (the pool holds them back by design), the
cap fills with them, and the textures a draw does want find it full (7 of 812 taken with 4.5 GB held). It would need
a sign of which new textures a draw is about to bind.

**A weak GPU: where the time goes.** Every pass and dispatch of two frames timed at 1080p in the field
(`REKARINA_GPU_PASSTIME=<frame>` with `REKARINA_GPU_PASSTIME_ALL=1` and `REKARINA_REPLAY_THREADS=1`, so the stretches the
chunk threads record are marked too; the marks overlap on the GPU, so these are shares, not a sum): the scene's
full-size passes (colour and depth at the picture's size: 68 passes with 2,088 draws, the largest single one a
one-draw full-screen pass of 0.80 ms) 5.34 ms a frame, 60 %; the game's compute dispatches 1.74 ms, 19 %; our own
work 0.61 ms, 7 % (the mirror prologue 0.42, the merge dispatches 0.17); the shadow maps (1600x1600, depth only, 7.5
passes a frame) 0.28 ms, 3 %. So lowering shadows buys nothing here: the cost is the scene. The game's own handheld
mode (`operation_mode = handheld`) draws the scene at 75 % of the picture (1440x810 in 1920x1080) and nothing else
changes with it: GPU busy 7.71 against 8.76 ms, 112 against 105 fps.

**The integrated Radeon** (the Ryzen's own, two compute units, AMD's Windows driver 2026.04.17, memory mode import):
the forest village is drawn right (foliage, grass, fog, characters, HUD) at 3.6 fps at 1920x1080 and 4.7 fps at
1280x720, the GPU taking 267 and 204 ms a frame with the replay at 9-10 ms. The two sizes give 154 ms that do not
depend on the picture's size and 55 ms per megapixel (the RTX 5070: 6.6 ms and 0.65 ms). Every start waits about
190 s in the entry step, the second one too: that driver links the 10,350 recorded pipelines at about 18 ms each
even from its own cache. A first preparation there is 16 minutes of shader variants (220-500 ms each on 16 threads)
and over 8 minutes of pipelines. Its GPU time by pass at 720p (two frames, 206 ms a frame inside the marks): the
scene's passes at the picture's size (54 passes, 1,932 draws, none over 9 ms) 145 ms, 70 %; the game's compute
dispatches 39 ms, 19 %; our own work 6 ms, 3 %; the shadow maps 2 ms. So there too the cost is the scene's own draws,
and nothing of ours is in the way. It stands for "the code runs right on AMD's driver", not for a weak card players
have.

Also seen: one map in, the process's working set is 15.7-16.9 GiB and its private memory 22-23 GiB at 1080p
(11.2-12.3 GiB on 2026-11-14). Not looked into; a PC with 16 GiB cannot hold it.

## 2026-11-19: places that need a whole 60 Hz frame, as patch sets

The set `realtime` feeds the game its measured step (in 60 Hz frames, under 1 above 60 fps). More places cut that step
to a whole number or test it as if it were a whole frame; nine sets at the end of `recomp/patches-oot-v111.txt` fix
them, with their host halves in `runtime/oot_fps.c` (an install-side file: its functions sit on the patch page at
+0xe50..+0xf60). Every site is a bl to its host function; each host function changes only the registers its comment
names and gives the game's own values back while the frame's step (the VFR manager's, `[[main+0x4c7e890]]+0x10`) is
one frame or more. A bl loses x30, so each site sits in a function that saved x30 in its prologue; the recording
function `main+0x3296840` is a leaf, and its two sites move x30 to x16 first (the host puts it back). The lifted bl
spills the registers and reloads x0-x18 and x30 after it, so a host function may read any register but must not write
x19-x29 or sp (the caller keeps them in locals it does not reload).

| Set (switch) | Place | Above 60 fps before | Now |
|---|---|---|---|
| `daytime` (REKARINA_DAY_CLOCK) | `main+0x3cdf70`, after the fadd at `+0x3cdfbc` | the clock adds (int)(2 * step) game seconds a frame: 83 % of its speed at 100 fps, stopped above 120 | the fraction kept per clock |
| `soundjob` (REKARINA_SOUND_JOB) | the sound job `main+0x415300` (gate `+0x41531c`, step `+0x415334`); the music update `main+0x3353400` (step and seconds `+0x3353448`, four reads of the battle flags); the song playbacks' calls `main+0x3292c80`, `+0x3292ed0` | the sound engine was handed parts of a step | the job runs once a whole step has built up (w2 = 3 on the other frames: its epilogue), with a step of 1; the music counts a whole step, the battle flags of held frames are kept, a note flag counts once per run |
| `horsejump` (REKARINA_HORSE_JUMP) | `main+0x2c3ae18` (fmov s9, #1.0 before fcmp s10, s9) | the slow-motion branch every frame: the horse ran through fences | s9 = the step when it is the frame's whole step |
| `floorcarry` (REKARINA_FLOOR_CARRY) | `main+0x13a068`, `+0x13d708`, `+0x389ac24` (after fmov s1, w8 = 100.0) | 1/dt over 100 every frame above 100 fps: carried by a tenth | the threshold 100 / step |
| `songrec` (REKARINA_SONG_RECORD) | `main+0x329340c` (playback), `+0x3296b04`, `+0x3296da0` (recording), `+0x32926c4` (its end) | the elapsed-frame counter (+0x265fc) cut to 0 | the fraction kept per counter |
| `looppose` (REKARINA_LOOP_POSE) | `main+0x2afe04`, `+0x2b01d0` (step * speed), `main+0x924e50` (the time drawn) | a loop of one frame drew another phase each frame | whole steps per animation entry, drawn at its start |
| `clothtime` (REKARINA_CLOTH_TIME) | `main+0x1b1ac68` (ldr d0, [x22, #8]) | a 60th per cloth step whatever the frame: fps / 60 too fast | the time times the step (below 60 fps left as it is) |
| `webstep` (REKARINA_WEB_STEPS) | `main+0x632cfc` (after the fdiv) | (int)(dt / a 60th) steps = 0: the web stayed rigid | the fraction kept per web |
| `inputstep` (REKARINA_INPUT_STEP) | `main+0x3b4eaf8` (60 / the nominal rate, stored at `main+0x4d41fc8`) | the input repeat ran fps / 60 fast | the real-time step (patch page +8) |

Seen in two smoke runs of 200 s (the forest, then the field, a 240 cap, 70-140 fps): no fault; the day's clock added
119.8-120.0 game seconds a second in steady 20 s windows (the game's own cut in the same windows: 64.5-104.5), the
sound job ran 1188-1201 times per 20 s, cloth steps of 16.667 ms became 8.4-16.6 ms, and 13,698-21,785 steps of
one-frame loops a 20 s window were made whole. The horse, the recording, the web and a song's playback were not met in
those runs. With every set's switch at 0 the log lists the nine as off and none of their places runs.

## 2026-11-20: the exposure step after the pause menu (late sums over a gap)

After every close of the pause menu the whole picture was brighter for a moment. Measured on the village save with
the menu opened three times (2, 6 and 15 s, `REKARINA_INPUT_SCRIPT`) and the flight recorder's thumbnails of every
frame: 93 -> 101 of 255 for exactly 15 frames from the second frame after the close, then back, a flat step; at a
60 fps cap and with no cap alike 15 frames, so it counts frames. Not the aging of released targets: the same with
`REKARINA_TEX_DROP_AGE=0`, `TEX_VRAM_PRESSURE=0` and `TEX_EVICT=0`.

It is the auto exposure. The sky LUT pass's exposure factor (`REKARINA_PROBE_FS=1893e0a44a9ee521`, its fs c4[2].w,
10000 / 2^EV) is 9.77 in the village, EV 10, its lower clamp, and 11.22 (EV 9.8) in those 15 frames. The game
(f_7000337060) reads the four float sums the environment cube's compute pass adds into (word 3 the luma), resets
them, and pushes 1 - clamp((luma / 256 - lo) / (hi - lo)) into a ring of 15 whose average enters EV = 10 - 3 *
average (f_70003b94b0). In the village every push is 0 (the luma is above hi); EV 9.8 is one push of 1 in 15: one
read that found no sum at all.

The sums are written after the frame's results point, so they go into guest memory at the next frame's wait (late
sums, `vk_memory.c`): the read in the game's update K sees frame K-2's sum, where the console's sees K-1 (its wait for
the frame returns when the frame is complete). In a steady run that only shifts the exposure by a frame. The menu draws
no scene and adds no sums; after the close the first read found the stale word from before the menu and the second
found nothing, the frame two before it being a menu frame, and was pushed as a black picture
(`REKARINA_LATE_LOG=<frames>` with `REKARINA_LATE_WATCH=<word>`: the word at every wait and submission end). With `REKARINA_LATE_SUMS=0` the step
after the menu goes, but 15-frame steps come at random, four in a minute: the race the late sums were made for.

Fix: the bridge in `vk_mem_late_apply`. When a wait opens a new delivery slot for a range that has no sum for it but
has the newer frame's sum queued, that frame's sum goes in at once, after a wait for the frame's tail, and also stays
queued for its own slot, so the next read sees it again instead of nothing. In a steady run every slot has its sum and
nothing changes. Village, two closes each, `REKARINA_LATE_BRIDGE=0` against the default: at a 60 fps cap the factor
was 11.22 for 15 frames after both closes without the bridge and 9.77 throughout with it; with no cap (159 fps) the
same, steps after the closes at 4348 and 7114 without it and none in three closes with it. The bridge went in at the
gaps of the run only: after the loading screen (4 small ranges) and after each close (the 3 exposure ranges and the
4 small ones), nowhere in play. Its wait costs once per close: the second frame after it took 12.6-14.6 ms against
8.7-9.5 ms without (the first frame after a close takes 14-19 ms either way); frames over 25 ms in 100 s 16 against
13, mean frame 6.89 against 6.86 ms.

## 2026-11-20: the depth of field at half the scene that is drawn (set `dofscale`)

The blur behind a lock-on, a conversation or a shop's items was drawn at 960x540 whatever the picture, and stretched
over it: at 1440p and above the sharp outlines in front of it (hair, ears, a shop's items) got a dark, blocky fringe
from the coarse blur. The size comes from `main+0x260270`, which makes an effect's render target from its descriptor:
a reference texture's size (`[[x20+0x18]+0x20]` +0x48 / +0x4a, 16 bits each) times the descriptor's floats, width *
+0x53c * +0x538 and height * +0x540 * +0x538. For the depth of field these are 0.75, 0.75 and 0.5 of a 2560x1440
reference: the game's own picture. 0.75 brings it to the console's 1080p scene and 0.5 halves that. The scene constant,
the Bootup pack's resolution palette and its ResolutionParam were each raised in earlier tries and none of them moved
it (it stayed 960x540 with the console's dynamic 1080p scene too).

The set `dofscale` (REKARINA_DOF_SCALE, on) calls `runtime/oot_fps.c` dof_scale at `main+0x2603c8`, right after the
size is stored in the target request on the stack (`stp w27, w25, [sp, #0x50]`; neither register is read again before
it is loaded with something else, and a host may not write x19-x29 anyway). For a descriptor with 0.75 on both sides
it writes the reference times the fixed scene's share of the picture (`frame_rate.c` res_fixed_scene_share) times
+0x538 over the request: 1280x720 for the whole 2560x1440 picture. It never goes below the game's own size (a DLSS
mode's smaller scene keeps 960x540), and leaves every other descriptor alone. `res_sizes` is no use here: without the
output and scene sets it gives placeholders (1600x900), which made the first try draw the blur at 800x450.

What the size changes and what it does not (1440p, the lock-on on a villager, the same frame of three runs): the
blur's strength in the picture stays the same at 800x450, 960x540 and 1280x720 (its radius is in picture space; the
high-frequency content of a background crop 0.337, 0.352, 0.351), while the dark, blocky fringe along the sharp
outlines in front of it (hair, ears) is widest at 800x450 and finest at 1280x720. REKARINA_DOF_LOG=1 logs the first 32 requests with the reference, the share and
the size before and after.

The set is off by default since the evening of 2026-11-20. With the instrument out in the forest (and, by the owner's
eye, with a target locked) the far scene came out black beyond 0.75 of the picture across and down: the right
quarter of the upper three quarters and the bottom quarter where far things show, the near grass and foliage drawn
normally over it, at 1440p in base mode (no DLSS, no frame generation) and, by the owner's report, with FSR 3.1
native AA. 0.75 is the
descriptor's own factor: a part of the effect (a pass's viewport, a later level of the blur or the way the result
is read back) still covers the game's 960x540 of the 1280x720 target, and the composite reads the rest, which was
never written, as black. The forest run (the owner's save, the instrument out for about 12 s, thumbnails every 30
presents): 26 of 425 thumbnails with the black quarter with the set on, 0 of 424 with it off. A DLSS mode whose
scene is under 0.75 of the picture never had the larger target, so it never showed it. Until the part that keeps
960x540 is found (a frame dump of that moment), the blur is the game's own, stretched as in 1.2.7.

## 2026-11-20: DLSS Frame Generation in the pause screens

The pause screens (their tabs Collection, Equipment, Items, System) mostly give frame generation no depth, motion or
camera (`vk_dlaa_fg_inputs` has nothing fresh: the game draws no scene there). `vk_dlssg.c` kept generation on
through such frames once it had been on, because turning it off freed its resources and turning it on again made
them anew, 400-750 ms a frame for four or five frames at each inventory opening. Kept on without inputs, it blended whole pause
screens with the frames before them: a tab switch blinked through the generated frames between, and our overlay's
numbers showed twice (the owner's shot: "319" with a ghost "200").

Streamline's guide for pause menus is DLSS-G off with `DLSSGFlags::eRetainResourcesWhenOff`: off then suspends
generation without freeing its memory, and on again costs nothing (`slFreeResources` is for a long deactivation).
`runtime/dlssg/nx2_sl.cpp` sets that flag on both switches; `vk_dlssg.c` turns generation off in every frame without
the inputs (and while our menu is open, as before) and gives the first frame back `reset`, since the frame before it
is the last one before the pause. `eEnableFullscreenMenuDetection` would need the interface's colour and alpha tagged
(`kBufferTypeUIColorAndAlpha`), which the port does not have. The library's interface is version 4; a library of
version 3 is still taken and keeps generation on through those frames as before.

The library is built by `tools/build_dlssg.py` with the project's clang (llvm-mingw) against the Streamline SDK
2.14.16 headers, as the shipped copy was (same imports: KERNEL32 and the UCRT's api sets). Streamline's DLLs are
MSVC's; the library passes them structures by reference and enums only, and `nx2_sl.cpp` asserts the offsets
Streamline's own `sl.api/sl.cpp` asserts (Preferences, Resource, ResourceTag, Constants) and those of DLSSGOptions,
DLSSGState and ReflexOptions. `REKARINA_DLSSG_DLL=<path>` loads the library from elsewhere (a test beside a game
folder whose copy other builds still use).

A background run cannot show a generated frame (generation needs the window in front), so the check here is the
plumbing (1440p, DLSS Quality + DLSS-G, the pause screen opened twice with its tabs switched, 80 s): with the
library of version 4 Streamline answered every switch with eOk, generation went off at each opening, on and off
again with the tabs (one of them draws the scene behind it) and on at each closing, with no error and
no frame over 50 ms; with the old way (version 3, kept on) Streamline logged in the pause screens "Unable to find
'common' constants for frame N - using last set" and "Failed to find global tag kBufferTypeDepth / MotionVectors":
the stale inputs it blended. The picture itself is for a run with the window in front.

Frames without inputs are not every pause screen. The Equipment tab draws the hero and the Items tab its turning items
into a scene with a camera of their own, which the game ends at an output pass instead of its scaling dispatch: frame
generation got depth, motion vectors and a camera there and blended the tabs into each other and ghosted the turning
items (the owner, DLSS-G, 2026-11-20). `vk_dlaa.c` fg_menu_frame gives neither frame generation inputs for a frame
whose evaluation ran at such an output pass (`oot_pass`), logs where each such stretch begins and ends ("dlaa: frame
generation off from frame N ... (a pause screen)", "... back at frame N after M frames of a pause screen") and resets
generation's history on the first frame of play after it. `REKARINA_FG_MENU_OFF=0` hands those frames on as before.
The same plumbing check with it (1440p, DLSS Quality + DLSS-G, the pause screen opened twice, its tabs switched):
generation went off 6 frames after each opening, stayed off through 3149 and 1811 frames of tabs and came back 5
frames after each closing; with
`REKARINA_FG_MENU_OFF=0` DLSS-G went on again inside the first pause screen, on the tab that draws the hero.

## 2026-11-20: button prompts that change in the running game reach texts laid out before (set `promptglyphs`)

With a pad switched on after the start, the action hint kept the keyboard's key and the pause screen's Back its SHIFT
until a restart, while a text first shown after the change had the pad's buttons. `romfs.c` live_main swaps the button
font's character table in memory; what kept the old buttons is the laid-out text and the font's glyph cache:

- A laid-out text keeps per character the font, the character, the glyph its shaping gave and its size (the glyph
  buffer at text+0x10: pages at +0x30 of 1024 records of 0xa0 bytes, the character at +8, the size at +0x1c/+0x20, the
  glyph at +0x94; 0x48 bytes with the character at +0 and the glyph at +0x3c when the byte at +0xd5 is set; the count
  at +0xac). Its quads are built from those records (`main+0x1515660` / `main+0x1516120`, each glyph through its font's
  plot) only when the text's +0x180 holds bit 7 (`main+0xe4e0e0`, the text's draw, clears it).
- The font's glyph cache (`main+0x1540640`, used by the FreeType font's plot `main+0x154ca60`) keys a picture by the
  font, the character, the glyph and the size (the key `main+0x15403d0`, the compare `main+0x1540300`).
- The texts with buttons seen in the Market and the pause screen all hold glyph 0: they are drawn by character, the
  plot looks the character up in the table in use, but the cache answers (character, 0, size) with the picture of
  the character's first drawing. The earlier tries laid texts out again (bits 8 and 9) and asked panes to shape again
  (bit 17): neither changes that key.

The set `promptglyphs` (REKARINA_PROMPT_GLYPHS, on) calls `runtime/oot_fps.c` prompt_glyphs at `main+0xe4e100`, after
the draw's prologue. Once per text after each change (`romfs.c` romfs_text_generation), and again whenever the game is
about to build the text's quads, a button character (U+E000..U+E0FF) drawn by character gets this change's mark in
the lowest four bits of its two size floats (a cache key no earlier change used, the size changed by less than a
millionth), and one holding another set's glyph gets the glyph of the set in use (`romfs.c` romfs_prompt_glyph; every
set's glyph keeps the game's advance, so nothing moves). A text that changed gets bit 7, so the same draw builds its
quads again, and again every fourth draw until the first changed glyph has new texture coordinates (a text built
while hidden skips the glyphs outside its clip). A text of size over 96 (the pause screen's guide is 102, the action
hint's key 160) gets the same change to its records but is built when the game builds it: a build asked for by us
left the hint's key blank in every form tried (a glyph id, the size mark, a build again every fourth draw until its
coordinates changed), while the guide, built by the game when the pause screen opens, takes the new buttons.

Runs in the Market (1440p, prompts switched at frames 300 and 4500, the hint at the fountain, then the pause screen):
with the set off the guide reads "Rotate (R) Select (A) Back [SHIFT]" after the keyboard -> pad change (the
owner's report); with it "Back (B)", and after pad -> keyboard "Rotate [mouse] Select [E] Back [SHIFT]". The action
hint's key still keeps the set it was first shown with until a restart: the game never builds that text again, and
a build of it asked for here comes out blank (open: its large glyph is plotted on another path, `main+0x1516048` /
`main+0x1519300`, that a build without a new layout does not feed).

## 2026-11-20: frame generation's picture without the HUD is the picture the HUD is drawn on

A player with AMD frame generation (FSR 3.1 native AA, RX 7900 XTX) recorded single frames that were brighter and
showed the scene's detail wrong. Frame generation (AMD's and DLSS's) is given a picture without the HUD and tells the
HUD apart by what differs between it and the presented frame; what differs is kept out of the generated frames and
taken from the real one. That picture was a copy of DLSS's (or FSR 3.1's) output at its evaluation, and the game's
frame goes on after that point (one frame dump in the Market, passes 283-293): DLSS's armed target is the tonemap's
output (pass 283, ACES with the bloom, drawn with the scene depth attached), then three dispatches are the game's own
morphological anti-aliasing (edges into an RG16UI 640x360 image, weights into an RGBA8 one, the blend), a fourth is
its scaler into the output picture, the HUD passes draw into that picture, and one last pass copies it into the 8-bit
sRGB window texture (a plain copy). In DLAA, where DLSS runs in place before the first of those dispatches, the copy
lacked the anti-aliasing and the scaler pass; in every mode it also missed what the present does to the real frame:
RCAS after DLSS at the sharpness setting (80 by default), the colour profile, FSR's scaling.

Now (`vk_dlaa.c` hudless_track, hudless_at_pass): after the evaluation at the scaling dispatch, every dispatch that
writes a colour image of the output's size names the picture, and the copy is taken at the first render pass that
draws into that picture (the HUD's) or, in a frame without HUD, into the window texture. At the present
(`gfx_present.c` hudless_record) the copy takes the same path as the game's picture (picture_into: the colour
profile or the HDR10 encode, then FSR or the blit), with source views of its own (the slots after FRAMES). The
upscaler's history resets (another scene size, no camera, another scene target) now reach both frame generations
(`vk_dlaa_fg_inputs` reset; it was always 0). Historical test knob: `REKARINA_FG_HUDLESS_AT_DLSS=1` (the old copy point,
no longer used by OoT, which captures after scene sharpening),
`REKARINA_FG_HUDLESS_PLAIN=1` (the old plain blit at the present); `REKARINA_HL_SHOT=lo-hi:step` saves the presented
frame (`sc<frame>.png`) and the HUD-less picture of the same present (`hl<frame>.png`).

Measured (Market, 2560x1440, AMD frame generation on the RTX 5070, 11 pairs a run): with the old copy 56 % of the
pixels differed by more than 2 levels and 5.7 % by more than 24, the copy brighter (mean level 98.8 against 96.3),
along every edge and texture of the scene; now 1.1 % and 0.8 %, all of them the HUD (hearts, the action hint). The
frames generated from it cannot be captured here (they are made inside AMD's swapchain): what a player sees still
needs a run in front.

The picture those dispatches name has to be the scene's colour, a float image (B10G11R11 or RGBA16F). The game's
anti-aliasing also writes its blend weights into an RGBA8 image of the output's size, black with coloured edge pixels,
and in frames where no later dispatch named another image, that one was taken for the picture: frame generation was
given black areas with coloured dots and saturated green and yellow slabs as the picture without the HUD and blended
them into its frames (the owner, FSR 3 frame generation, 2026-11-20: a slab over the Market gate's light shaft that
moved with the camera). `hudless_track` now names only float images.
Market runs with the owner's save and settings (FSR 3.1 native AA + AMD frame generation, 2560x1440, the hero moved
to the Market, 120 s, the HUD-less picture saved every 60 presents): the build before it hung in 1 of 2 runs (no
present for 10 s, every game thread waiting), this one in none of 5. Outside the HUD's fixed places (hearts, minimap,
the button cluster) the HUD-less picture differs from the presented frame in 0.5 % and 1.1 % of the pixels on
average, and every such pair is interface too: speech bubbles, the place's name
title and the arrival's fade from black, which the game draws over the finished scene.

## 2026-11-20: the action hint's key follows a live prompt change (its large glyph's sheet)

The action hint's key (size 160) kept the set it was first shown with until a restart, and every build of its text
asked for after the change (bit 7 of text+0x180) showed it blank or old. Probe runs (DEBUG-SHEETLOG, uncommitted) and
the disassembly of the text's draw (`main+0xe4e0e0`) say why:

- A glyph larger than the cache's (the key; the pause screen's guide at 102 too) is not drawn from the font's glyph
  cache. The text's full quad build (`main+0x1515660`) rasterises it on the CPU (`main+0x1519300`, a shelf packer
  `main+0x150f540`) into one of the text's own sheets: a ring at text+0x68 (count +0xa4, the one in use +0xa0), each
  sheet 0x158 bytes (set up: byte +8; NVN memory pool +0xe0, offset +0xe8; 128x73 texels of a byte for the key;
  packer cursor +0x148/+0x14c, row height +0x150). Each build moves the ring on (`main+0xe4e2a4`) and empties the
  sheet it takes (`main+0x150f4d0`).
- The draw runs that full build only when the text's state word (*(text+8)) holds bit 20 with bits 17 and 18 clear
  (`main+0xe4e20c`; bits 20 and 7 are cleared after it, `main+0xe4e430`). Bit 7 alone runs the other build
  (`main+0x1516120`), which moves the ring on and plots no large glyph.
- The game sets a sheet up in its own layout only: the key's ring has two sheets and only sheet 1 was ever set up.
  So a bit-7 build moved the ring onto sheet 0 (nothing: blank) or back onto sheet 1 (not plotted again: the old key).
- The texture cache does not see the CPU's writes into a sheet after its first upload.

Now (`runtime/oot_fps.c` sheet_request, sheet_pending_draw; REKARINA_PROMPT_SHEETS, on): at a prompt change, a text
holding a large button glyph of another set whose sheet in use the game set up is built again through the full
build: the ring is set one before a set-up sheet and bit 20 asks for the build; at a later draw, once the ring is on
that sheet and its texels changed, the texture cache is told they were written (`vk_guest_written`). A text the game
does not build at that moment (bits 17/18) is asked again every fourth draw, for at most 3600 draws. A text whose
sheet was never set up (the guide before the pause screen first opened) is left to the game's own layout, and once
the game built it the texture cache is told too: in one run the guide's Back kept the keyboard's SHIFT while its
Rotate and Equip took the pad's buttons, its sheet's storage having held an older picture.

Runs in the Market (1440p, prompts switched at frames 300 and 4500, the hint at the fountain, then the pause
screen): before, the key kept "E" after keyboard -> pad; with the fix the build is asked once and the next draw
logs the plot ("its large glyph was plotted again into sheet 1 (1 requests, 0 draws)"), the pill reads "A Talk",
and after pad -> keyboard "E Talk"; the pause screen's guide, opened after the change, reads "Rotate (R) Select (A)
Back (B)" and "Rotate [mouse] Select [E] Back [SHIFT]". Work is done only in the draws after a change.

## 2026-11-20: the render target heap, the game's lock-free queues, the movies' start

### The room of pots beside the Market's gate (the render target heap)

Walking into the room of pots and torches beside the Market's gate froze the game (1.2.0 to 1.2.5; a player's hang
dump at main+0x26b294). The render target cache (main+0x3ed00) places its transient targets first fit in one heap,
DynamicTextureAllocatorMem of the HeapDef table (312320 KiB, the cache's 0x13100000 bytes; its pools are a physical
one of that size, flags 0x18c, and a virtual one over it). A target it cannot place is not handed out: its retry reads
x23->+0x48 with x23 = 0, and our fault handler took the null page for a watched one and let the read run again for
ever (b43d55b made it a crash report). In that room the game asks for 6 to 17 square shadow maps a frame (depth,
compressible, 386 to 1918 texels, a new side each frame while the camera moves); square sides do not follow the scene
scale (mean side 1064 at 100 %, 1107 at 62 %). The console's dynamic resolution keeps its scene at 1920x1080 at most
and lower where the GPU is busy; ours has the whole picture, and square depth targets report padded storage
(`REKARINA_DEPTH_SQUARE_PAD`). With the same exe and the walk of the room (`REKARINA_TEX_HEAP_MIB` sets the heap):
305 MiB (the shipped size) ends in the null read at a 1080-line picture too, 384 MiB at 1440 as well; 448 MiB and up
play on in the room. The fix grows the heap through the HeapDef table, as `widescreen.c` already did for wider
pictures: 256 MiB for the 16:9 picture (561 MiB in all), and the twin block of that physical pool covers only the
shipped 305 MiB (`vk_memory.c` `vk_mem_phys_twin`), so the growth costs its size once under VRAM pressure. A tour
of the Market, the castle and the field runs the same with and without it.

### The game's lock-free queues (set ringpost)

A player's game ended after the first boss's death with an nn::diag abort from main+0x34fc280, reached from the
game's post of a named text value (main+0xd24e50). The post writes a lock-free ring (the object at
[[main+0x4c7e180]]: word +0x640 with the next record in bits 0-19, the records written in 20-27 and a generation in
28-31; capacity +0x630, 96): it takes a record, writes it, counts it written and checks the generation. Once a frame
the consumer (main+0xb99980) reads the records up to the first empty one and swaps the generation without waiting for
writers; a writer still copying then, or one that finds the ring full, aborts. The engine relies on its writers being
done by then; a job held up on a busy 8-thread CPU is not. The patch set turns that abort into a return: the record is
dropped or read a frame later. The same writer is compiled into 77 functions, the game's other lock-free queues (each
commits with the mask 0xf00fffff and ends in a tail branch to the abort); the set has a site at each of those branches. `REKARINA_RING_LOG=1` (with `REKARINA_HANG_DUMP`) samples the ring: at most 7 of its 96
records in use and about 19 swaps a second on the Market, the field and the castle. An SDK Break now logs the ring.

### The movies' start on cards without tensor cores

The movies' network decision (2b75592) measured 180 frames of the first movie, ten seconds at a GTX 1650 Super's
17 fps while the sound ran on. Frames 8 to 20 of a movie now decide at once where they take more than twice the
movie's frame time, and the verdict is kept per GPU and driver in `movie-network.txt` beside rekarina.exe.

### Video memory on a 6 GB card: the heap's growth and cold mirrors

A player's RTX 3060 Laptop (6 GB) lost the device near the castle, in the village and on the field, and after some map
changes the GPU time went from 14 to 60 ms a frame for good: the game held 4.8-5.5 GiB of a 4.1-4.9 GiB budget, and
before one loss binding an image into the render-target heap's 305 MiB pool memory took 486 ms (Windows paging that
allocation back in). What held the memory in a Market / castle / field tour with a VRAM hog standing in for the small
card (the hog subtracted): the driver's own ~1.25 GiB, mirrors of buffer pools 1.3 GiB (1,025 pools), textures 1.7 GiB
live and 0.3 released, the render-target heap's pool and twin 0.6 GiB.

The heap's growth (039fe41) had device memory on every card. Now only on cards with more than 8.5 GiB of device-local
memory (`REKARINA_HEAP_GROWTH_MIN_MB`, `vk_memory.c phys_backed`); on smaller ones the address resolve calls the growth
unbacked, an image placed there takes memory of its own, and the pool table and the twin stop at the backed end. The
room of pots with the growth unbacked plays: 18-20 frames over 25 ms and 26-71 image creations over 3 ms in the room
(three walks, 1440p) against 11-18 and 9-68 with it backed (two); under a VRAM hog 35 and 189 (one) against 19-27 and
129-148 (two). The twin stays at the shipped 305 MiB on both paths. The shipped heap alone still ends in the null read.

Mirrors: a mirror lives as long as its pool, and the game keeps the pools of the maps it left. Every mirror is synced
for the global memory table each submission, so that sync says nothing about use; a buffer binding does (`vk_mem_use`,
`vk_mem_pick` stamp `use_frame`). `vk_memory.c mirror_cold_release` (`REKARINA_MIRROR_COLD=1`, off by default): over
90 % of the budget, a mirror no binding named for 3600 frames and the GPU never wrote gives its memory back; the pool
is read in place, and its next binding asks for a mirror again. With the hog, two 5.5-minute tours: 1,415 mirrors
given back, mirrors at the end 714 MiB against 1,150, the same frame rate, but textures took the room (1,967 MiB
against 1,646) and the driver's total did not fall: the freed pieces are holes in shared blocks and the texture
eviction keeps what is not cold. A lower total needs the textures' line to follow; not done.

## 2026-11-20: distant lights that went out for a frame (early results)

Players saw the distant candles of a cave flicker while the near ones stayed steady, with the
uniform snapshot's compute exception (5c450c5) already in. Reproduced on the RTX 5070 with the owner's save (warp
`45:domain`, still camera, every frame at 640x360): the light pool a far candle throws on the wall is missing in one
frame and back in the next, in 15-28 % of the frames; the flame itself stays, and the picture's mean brightness does not
move (not the exposure).

The cause is the results point (vk_sync.c, "Results before the frame's end"). The game's next frame runs while the GPU
still works on the frame's tail, and what shaders write in that tail went back to guest memory only at the submission's
end. Six compute programs right after the point write small records (two pools of 0x3980 and 0x5b40 bytes, double
buffered by frame) that the game reads in its next frame: f_7000345f00 takes a light's record only when the stamp at
record + 0x1540 says "the frame just before"; otherwise it invalidates the CPU cache over the record and drops the light
for that frame. `REKARINA_RESULT_WATCH=1` shows these reads, and the occlusion read-back's (f_700354ae30), happening
before the GPU finished in most frames.

What was measured, blinks per 1000 frames at the far candle (control 111-278) and frames per second:

| change | blinks | cost |
|---|---|---|
| `REKARINA_RESULTS_FENCE=0` / `=1` (no point / a later one) | 0 | -23 % / -17 % |
| the result guard on reads (a guest read of such a page waits for the submission) | 0 | 0 to -9 % (Market: 5 ms waits) |
| every pending write-back from 8 KB on (`REKARINA_WB_EARLY_KB=8`) | 7 | -5 to -8 % |
| `REKARINA_OQ_FORCE=1000` (every samples-passed count reads as visible) | 295 | 0 |
| the early results below, plain copies of whole records | no change | 0 |
| the early results below, through the word merge | 0 | 0 (three pairs) |

The early results (`vk_memory.c vk_mem_early_results`, `REKARINA_WB_EARLY_RESULTS`, on for OoT): the small ranges (up
to 64 KiB) shaders write after the results point go back on their own at the first part start after they were written
(`vk_exec.c wb_early`; the part's prologue has taken their snapshots by then), through the word merge where the CPU
also writes them and as plain copies otherwise, behind barriers on the shader stages only, and they leave the end's
write-back. Plain copies of the whole records did nothing: the records hold the CPU's own fields too, and a copy put the
mirror's old values of those back over the game's. Three alternating pairs in the cave: 0 blinks in all six windows,
118.7-121.0 fps against 118.3-118.7. In off / on / on / off order (the machine's load drifts within a series): the cave
104.4 / 104.8 fps against 103.7 / 101.5 with 0 blinks against 33-161 per 1000, the Market (standing at the fountain)
79.0 / 82.2 against 76.3 / 80.3, so the shorter write-back at the submission's end gains 2-3 %. Closing the inventory
and the map (five times each, every frame's mean brightness) shows no step with it on or off.

The same change also keeps the exposure right where float atomics are compare-and-swap loops (RDNA2 cards, or
`REKARINA_FLOAT_ATOMICS=0`): there the exposure's sum is not marked as a float sum, its write-back at the submission's
end went over the CPU's reset of it, and the picture turned white within seconds in the field by day; with the early
results the sum goes back before the game reads and resets it (field by day: mean brightness 78-88 against 81 with
native atomics, against 101-254 without the early results).

## 2026-11-20: the driver's build and the texture cache leave the setup for the first start

Players' setups of 1.2.6/1.2.7 waited 20 minutes to 1.5 hours in the package step's `Rekarina.exe --warm-shaders` (a
cold driver cache after a driver change; on AMD always, its disk cache is keyed to the exe's path, so the build folder's
libraries did not count in the game folder) and about 43 minutes in the texture preparation. Since then the setup
translates only (`REKARINA_PREPARE_DRIVER=0` from `installer/oot.py`) and copies; the game's entry step builds what is
missing before play, the launcher's background job gives it a head start (`docs/installer.md`).

Measured on the RTX 5070 (16 threads), a game folder with no driver cache, no texture cache, no pipeline-cache.bin, the
launcher's job stopped by Play after 8,000 of 56,402 prepared variants (329 s, ~24/s at below-normal priority, the stop
1.97 s): the entry step counted 81,041 items (10,995 pipelines, 48,388 libraries, 21,658 texture archives) and took
2,186 s. The mesh pipelines come first (121 s, the longest 30 s; the count crawls), then ~15 graphics pipelines a
second; the libraries ~29 a second (1,673 s); the texture walk beside them 912 s (11,739 textures, 7.73 GiB written).
The time left the picture showed was within about -30..+20 % of the real rest once the step was steady; in the first
five minutes it jumped (the mesh phase's pace, archives that vary a thousandfold): the texture part's pace is now taken
on the archives' bytes, the pipelines' only in their steady phase, and the shown figure moves at a bounded speed.

That run, with the libraries and the walk inside the game's process, failed one minute into play: "host memory import
failed (-2)" 1,423 times, then `counter report ... is in no memory pool`, "memory commit: this process 68352 MB". After
the step the driver held 43,193 MiB (11,665 MiB of VMA blocks in 11,648 live allocations: the walk's staging buffers
and images, which the game's process does not give back during the step; 31.5 GiB outside VMA). The figure was the
same at the first library count line and after the 48,000th: the libraries added nothing, the walk had it all. The walk
now runs in a child process of the same exe (`--scan-textures`, `prep.c prep_child`), whose count lines the step reads
and whose memory goes with it.

The libraries were moved to a child too (`--warm-libraries`) and had to come back: the drivers give each process a disk
cache file of its own. NVIDIA's `GLCache` folder held `...31.bin` with all 56,402 libraries from the run above; the game
process took that file at its device's creation, the texture child got a new `...32`, the libraries child a new
`...33`, and it compiled every library from nothing (480 ms each, ~30 a second, its own footprint 297-322 MiB) into a
file the game never reads (the next start takes `...31` again). A chain of children that a bug started made `...34` to
`...53`, all empty. AMD's `VkCache` shows the same pattern (`<hash>.parc` beside `<hash>_1.parc`). So whatever play
needs from the driver is built in the game's own process; the launcher's background build runs while no game does
and fills the folder's file. The texture child asks for no NVIDIA disk cache (`__GL_SHADER_DISK_CACHE=0`, `main.c`):
the Linux driver honours it, the Windows one does not and gives it an 82 KB file of its own (`...32`), which costs
nothing.

## 2026-11-20: the setup's translation waited on the disk, not on the translators

Players saw the translation phase run at a few shaders a second with the translators idle (twelve `totk_xlate`
processes at 0.1 % on an i7-13700, the machine at 7-11 %; 8.9 shaders a second on another PC). Each translated shader
waited for four FlushFileBuffers calls and four write-through renames of the cache's files, and a drive serializes
those. The same 3,000 shaders on 8 translators (9700X, 16 threads):

| disk | with the flushes (1.2.7) | without |
|---|---|---|
| fast NVMe | 37.1 s | 15.6 s |
| DRAM-less NVMe | 46.5 s | 16.0 s |
| hard disk (1,500 shaders) | 211 s, 7 a second, one translator busy | 13.2 s |

`--prepare-shaders` now writes without them (`gfx_shader_cache_flush`); the progress line every 5 s says how many
translators were busy and where a translation's time went (translator, waiting for one, cache files, same program).
Without the flushes all translators are busy on every disk; sixteen translators on sixteen threads were slower than
eight (17.2 s).

## 2026-11-21: present modes other than mailbox took the display (full-screen exclusive)

A player with an RTX 4080 and an HDR monitor reported that choosing FIFO or immediate in the menu flashed the screen,
showed oversaturated colours (also with Windows' HDR off) and changed how Alt+Tab behaved, while mailbox was fine; his
log has 190-620 ms frames at each switch. Our fullscreen is a borderless window over the monitor, and the swap chain
left full-screen exclusive to the driver: NVIDIA presents mailbox through a DXGI swap chain (composed by Windows, its
colour handling applied) and FIFO / immediate through its own path, which may take a window that covers the monitor
for itself; AMD's Windows driver does the same (DXVK disallows it on windowed swap chains for this reason). Now the
plain swap chain carries `VkSurfaceFullScreenExclusiveInfoEXT` with `VK_FULL_SCREEN_EXCLUSIVE_DISALLOWED_EXT` where
the device has `VK_EXT_full_screen_exclusive` (it needs `VK_KHR_get_surface_capabilities2` on the instance); a swap
chain refused with it is made again without it, and `REKARINA_VK_FSE=0` gives the driver its choice back. The frame
generation swap chains (FSR, DLSS-G) are left as they were.

On the RTX 5070 here (2560x1440 at 320 Hz, HDR off) no process ever owned the display exclusively, with or without
the change (D3DKMTCheckExclusiveOwnership sampled every 50 ms, the shell's notification state 2 "busy", never 3,
through live switches between all three modes and focus switches), so the change shows nothing here; its effect is
for the reporter's machine to confirm. Mailbox frame rate at the Market, four 60 s windows in A/B/B/A order: 94.3 /
95.4 / 95.4 / 95.8 fps (with / without / without / with), 1 % low 61.5 / 59.7 / 60.6 / 62.5: no cost.

## 2026-11-21: DLSS frame generation asked for on a GPU that is not NVIDIA's

A rekarina.ini with `framegen = dlss` on a machine whose chosen GPU is not NVIDIA's (a player who moved from an RTX card
to a Radeon, a laptop running on its integrated GPU, another GPU picked by `gpu_index`) started no game at all:
`vk_dlssg_early` loads NVIDIA Streamline before the first Vulkan call (its `slInit` asks for that, so the device is not
known yet), and Streamline's `vkCreateDevice` refuses any device without NVIDIA's optical flow extension ("Required
device extension VK_NV_optical_flow unsupported", -7). `create_device` gave up with "vkCreateDevice failed". Our test
runs set `REKARINA_FRAMEGEN=off` in the environment, which wins over the ini, so they never showed it; the setup's
preparation modes hit it on the integrated Radeon (L, hold14a).

Now `vk_dlssg_create_device` makes the device again through the loader when Streamline refuses it (the instance and
the physical device are the loader's: the interposer only passes them on), marks Streamline as unused and says so in
the log; `vk_dlssg_device_ready` then hands frame generation to AMD FSR (`vk_fg_take_over`, whose queues are made for
DLSS frame generation too). The menu's row says why DLSS frame generation is off.

Proof on the integrated Radeon here with a game folder whose rekarina.ini says `framegen = dlss`, `dlaa = 1`,
`taa = dlss`, `gpu_index = 1` and no such keys in the environment: 1.2.8's runtime ends with "no usable Vulkan device"
after Streamline's two refusals; with the change the device is made, AMD FSR frame generation takes over ("fg: AMD FSR
frame generation in place of DLSS Frame Generation ...") and the game plays on into the field. The same folder on the
RTX 5070 (`REKARINA_GPU_INDEX=0`) still runs DLSS frame generation on the swapchain.

## 2026-11-21: texture array accesses marked NonUniform

The translator indexes arrays of textures and samplers (the texture pool's, `patch_pool_arrays`) with an index it
computes from a handle, and no module carries a NonUniform decoration: `tools/xlate/Program.cs` leaves Ryujinx's
`QueryHostSupportsShaderNonUniformIndexing` at false, so its `MarkNonUniform` never fires. In a sample of 431 cached
modules, 30 index such arrays with a computed index; traced back, several of those indices come from per-pixel inputs
through storage-buffer loads (a light list read at an address made from the pixel's position). Vulkan leaves such an
access undefined without the decoration. NVIDIA's hardware takes a descriptor per lane either way; AMD keeps a descriptor
in scalar registers and, told nothing, may use the first active lane's element for the whole wave: another texture or
shadow map for part of the wave, changing from frame to frame. Longshot indexes the game's per-light shadow arrays
NonUniform for this (its 7315957c).

`gfx_shader.c` `patch_nonuniform` decorates them at load, after the pool arrays are moved: a forward taint pass over
the function bodies (Input variables, atomics' results and subgroup elections; through any arithmetic, loads at
tainted addresses, function variables stored from tainted values, calls) marks each access chain into an image or
sampler array whose index is tainted, the load of the element and the sampled image made from it; the module gets the
`SPV_EXT_descriptor_indexing` extension (the modules are SPIR-V 1.3 and 1.4) and the `ShaderNonUniform` and
`SampledImageArrayNonUniformIndexing` capabilities. No ids are added and the shader cache is untouched: nothing is
translated again; a driver builds the pipelines of the changed modules once. It is on by default on every GPU but
NVIDIA's (the spec requires the decoration; it cost nothing here), only where the device has
`shaderSampledImageArrayNonUniformIndexing`; NVIDIA's modules stay as they were. `REKARINA_NONUNIFORM=0` turns it off,
`=1` turns it on on NVIDIA, `all` decorates every computed index. Off, the pass returns at its first line. The log says
which mode the device got ("vk: texture array accesses decorated NonUniform: ..."). Whether it is what the flicker
reports from Radeons of the newer compiler (RDNA 3 / 4) need, 1.2.9 is the test.

Checks: a standalone build of the pass over six cached modules (the instructions kept in order, only capabilities,
the extension and decorations added, every section in its place, every decorated id a result in a function body);
the RTX 5070 with `=all` (forest at noon: the modules accepted, no pipeline failure, no draw skipped, the picture the
same as without); the integrated Radeon (RDNA 2) with `=1`, forest at noon, standing, uncapped, A B B A: 342 / 345 /
344 / 346 frames in the same 60 s window (without / with / with / without), no device loss: no cost.

## 2026-11-21: no console window from Explorer, and the WAITLOG lines

Rekarina.exe is a console program, so Windows gave it a console window when it was opened from Explorer or a shortcut:
the log scrolled beside the game (a player's screenshot showed it), and every line written there cost the writing
thread a console redraw. The launcher never had one (it starts the game with DETACHED_PROCESS). Now the manifest asks
for no console (`consoleAllocationPolicy` detached, Windows 11 24H2 and later; older systems ignore the element), and
on older systems `runtime/main.c` main lets go of a console that only this process is attached to
(`plat_console_release`: GetConsoleProcessList = 1, then FreeConsole), after the moment it was on the screen. A console
shared with the shell that started the game (cmd, PowerShell, a script) stays, and a script's redirected stderr is
untouched. REKARINA_CONSOLE=1 keeps the console; the log is in `logs\rekarina.log` either way. Checked with two test
programs of the same rule opened through explorer.exe on Windows 11: with the game's manifest no console existed at the
start; without it the console (one process) was let go; from a shell both kept it (three processes on it).

The same screenshot showed "WAITLOG EventSignal ..." lines: a debugging probe from before 1.0.0
(`nvn_api_core.c` nvnEventSignal: the first 40 event signals and those of frames 899-902; `vk_exec_xfer.c`: the
replay's event waits in frames 899-902) wrote them into every log, 48 lines in a player's 1.2.7 log. They are behind
REKARINA_WAITLOG=1 now.

## 2026-11-21: settings sliders that moved one way only

A player's gamma slider (colour settings) would not step left with the keyboard or a pad; right worked. `opt_slider`
(`runtime/ui/option_rows.cpp`) snaps a step to its grid with `floor` / `ceil` of the value over the step and a 1e-6
allowance, and the step came in as a float: 0.02f is 0.0199999996, so gamma 1.00 divided to 50.0000011, `ceil` went
one grid point up, and left landed back on the value it started from. The same arithmetic over every grid value of
every slider with a fractional step: gamma left stuck at 16 of 20 values (from 0.90, the default included), walk speed
left at 45 of 69 (from 0.45), and the mouse sensitivity, whose 0.05f is just above 0.05, stuck RIGHT at 132 of 200
(from 3.40). The step is now the decimal it was written as (`%.6g` of the float); the same check finds no stuck value.

## 2026-11-21: the live prompt glyphs' fault (a sheet the build through bit 7 never maps)

With set `promptglyphs` on, the pause screen opened and closed while the prompts changed over and over (W's owner-hang
set-up) faulted once in the game's glyph drawing: a `memset` of a glyph's row at address 0x36, called from
`main+0x1519424`, right after one of FreeType's band-split longjmps. The longjmp is FreeType's normal answer to a
glyph larger than its cell pool (the clean runs had four of them each). The stack says what happened:

- The fault was in the build through bit 7 (`main+0x1516120`, return `main+0x15163b4`) of a size-90 text whose size
  this pass had just marked. A marked size, or a glyph id this pass gave, is a key the font's glyph cache has no
  picture for: the plot (`main+0x154ca60`) renders it through FreeType and hands the text's callback
  (`main+0x15192a0`) a glyph that is not in the cache's texture (+0xc != 1), and the per-glyph build
  (`main+0x1515eb0`) draws such a glyph into the text's sheet in use: the CPU rasteriser (`main+0x1519300`) writes
  its rows through the sheet's mapped address at sheet+0xf0.
- Only the full build maps that address (`main+0x150f4d0`: nvnMemoryPoolMap of the sheet's pool plus its offset) and
  its end clears it again (`main+0x150f520`). The build through bit 7 maps nothing, so its rows went to address 0
  plus their place in the sheet. It happens only when the sheet still has room: otherwise the shelf packer
  (`main+0x150f540`) refuses the glyph and it is skipped, which is why the fault was rare.
- The game itself never misses the cache in that build: the full build that laid the text out put its glyphs in the
  cache first.

Now (`runtime/oot_fps.c` sheet_ready, sheet_stamp_draw): before a build through bit 7 of a text whose records this
pass changed since the last change (asked by the pass, or by the game for such a text), every set-up sheet of the
text's ring gets the mapped address the full build would give it (nvnMemoryPoolMap's answer for the pool the sheet's
holder at +0xe0 names, its first word, plus the sheet's offset; only for a pool with CPU access), and the texture cache
is told the sheets were written, before the build and again at the text's next draw once the ring moved on. A text
with a set-up sheet that cannot be mapped is not built by the pass. The summary lines count the prepared builds
("builds with their sheets mapped"), the ones that drew into such a sheet, and the texts met without a ring or with
none set up.

Checked with W's repro of the fault (the owner's settings and save, fullscreen, the home forest, the pause screen
opened and closed every 2.5 s while the prompts change live, focus taken and given back): with REKARINA_SHEETLOG=1 the
pause screen's size-90 L and R icons hold a ring of two sheets, of which sheet 1 is set up later (128x43, room for
another glyph). Whether a forced build faults depends on the ring's phase: one that moves onto the unset sheet 0 is
skipped, one that moves onto sheet 1 rasterises the missed glyph there, which in the faulting run went to address 0.
With the sheets mapped, two runs (hg4, hg5) each logged "sheet 1 mapped at 0x1afe98600 for a build through bit 7 (the
holder's own address 0x1afe98600)" and twice "its build drew into sheet 1, mapped for it", with no fault (62-64 prompt
changes, 68 pause screens and 4 FreeType band splits a run). A first version read the pool at the holder itself,
found no pool there and left those texts unbuilt: the log's "no CPU view" lines said so.

## 2026-11-21: the frozen figures under the ice at 48 fps (set `looppose` on every fractional step)

A player (RTX 3080 Ti, base mode, 1440p) saw the frozen figures in the ice of the domain flicker and seem to move at
some camera angles. His two clips (1.2.7 and 1.2.8) run at 47-48 fps by the overlay; frame by frame, the figure under the
ice at the hero's feet takes another pose and size from one captured frame to the next, the same pose coming back every
few frames. Those figures are drawn from a one-frame animation loop, which set `looppose` holds on one pose by stepping
it in whole frames and drawing it at the loop's start, but only while the frame step was under one 60 Hz frame (above
60 fps). At 48 fps the step is 1.25: the phase moved a quarter frame each frame and wrapped, the twitch the set was made
for above 60. "At certain camera angles" are the views that ran under 60 fps on his card. The set now acts on any step
with a fraction (`runtime/oot_fps.c` loop_fraction); whole steps (60 and 30 fps) keep the game's own values. Another
port's fix of the same twitch (above 60 fps only) leaves it under 60 as well.
In a run at a 48 fps cap in the domain, 1.2.8's counters read "loops of one frame: 0 steps made whole, 0 poses drawn
at their start" every 20 s; with the change, 5,760-9,910 steps made whole and 1,600-3,372 poses drawn at their start
(about three and a half such loops a frame). His own 1.2.7 log from the domain, settled at 45-52 fps, reads 0 and 1
there, and counts only in the moments his frame ran over 60 fps.

## 2026-11-21: float sums of the compare-and-swap path marked as sums

Radeons of RDNA 1 and 2 have no float atomic add on buffers, so the translator makes every float atomic add a
compare-and-swap loop. `gfx_shader.c` `mark_fadd_ssbos` marks the storage buffers shaders add floats into
(`GFX_SSBO_FADD`), and the mirror's write-back adds the GPU's change to what the CPU left there instead of copying the
mirror's word over it (`vk_memory.c` `shadow_merge`, `wb_merge.comp`). For the loop it looked for a swap whose value is
the bit cast of an `OpFAdd`, and none of the translator's loops has that shape: Ryujinx keeps registers in function
variables, so the sum is stored into one and loaded back, bit cast, put through the select that flushes a denormal sum
to 0, and bit cast again (`OpBitcast(OpSelect(|x| < min, 0, OpBitcast(OpLoad(%reg))))`, with `OpStore %reg
OpFAdd(...)` earlier). On a discrete Radeon (memory mode mirror), those sums went back as plain values.

The rule now carries "made from an `OpFAdd`" through stores into and loads from a variable, bit casts, copies and
selects, in three passes (a loop's store can come after its load in the module's order). On three 1/256 samples of the
shader cache it takes all 18 float swaps as sums and none of the 37 swaps of byte and halfword stores (their value is
an `OpBitFieldInsert`). In the game with the loops forced (`REKARINA_FLOAT_ATOMICS=0`, the RTX 5070 and the integrated
Radeon): "N float sums by compare-and-swap, N storage buffer bindings marked as sums, 0 through raw pointers" for six
shaders, where 1.2.8 marked none; the picture in the forest and the field the same as 1.2.8's. NVIDIA's native adds take
the same rule as before; the loops do not exist there.

The white forest with black blotches reported from an RX 6600 (1.2.8, the loop path) did not show here in five
setups of 1.2.8 with the loops: the RTX 5070 (mirror) at the exit, the bridge and the forest's middle, the integrated
Radeon in both memory modes, the RTX 5070 with the reporter's `astc_bc = 0`; child save, noon.

## 2026-11-21: the keyboard keys' labels in the game's text

In the item list (LB) the keyboard prompts' Back key read "SHIFT" in letters a few pixels high and uneven. The game's
text fonts have no shift symbol, so `pc_labels_fit` replaced it with the word, and a label of three letters or more was
drawn at 0.26 of the key's height and at most 74 % of a square key's width: SHIFT came out width-limited at about 0.17
of the key (7 px at 1440p, 5 px at 1080p), where FreeType's rasterising makes letters uneven. The owner picked one of
three mock-ups: Shift is now a solid up arrow (a head half as wide as the arrow is tall over a stem a quarter as wide,
0.56 of the key), and a word (CTRL, TAB, ALT ...) is drawn at 0.40 of the key's height and 84 % of its width on a key
as wide as the button's advance allows (580 of 640 font units, not a 560 square). One- and two-letter keys and the
mouse are as before; the Up key keeps the font's own thin arrow, so it does not look like Shift. `tools/button_icons.py`
is an input of the setup's prompts step: an update draws the button pictures again (about a minute) and nothing else.
Checked in the game: the setup's own prompts tool drew the button pictures with the change, and the item list opened
under the keyboard prompts at 1440p and 1080p shows Back with the arrow key in place of the tiny SHIFT, the key as
large as the other keys; Sort (the mouse) and Select (A D) are the same pictures as before.

## 2026-11-22: leaf shadows boiling on a lit ledge while the hero strafes (`shadow_scale = auto`)

The report: on the lit rock face below a wooden fence, the fern shadows shimmer in bands when the hero strafes left and
right. The shadow mask (fs 155e38dc3b75418a) moves most in G, the sun cascades' term, before any output processing.
About 99 % of the ledge samples one layer, the second cascade, and fewer than 2 % change cascade between frames, so a
cascade switch is not the cause. Upstream 1.2.7 and 1.2.8 built from source with the same save and input script
measure the same (mean green-mask pulse 16.6 and 16.3): it is not a 1.2.8 regression, and the 1.2.8 NVIDIA blinking
fixes (`WB_EARLY_RESULTS`, `USNAP_LIVE_SIZE`) stay on: turning both off, DLAA off or the foliage culling off moves the
pulse by under 6 %, and the ledge still flickers with the foliage shadows shaded at full rate.

The cause is the game's projection. The casters' block of the second cascade holds a light rotation and a matrix
whose w row is one axis of the light's plane plus an offset, with y on the same axis and depth on the light's
direction: a light-space perspective shadow map. Its offset and warp follow the camera, so a static point on the
ledge moves across the 1600x1600 map by fractions of a texel every frame the camera moves (about 23 texels over the
strafe), and no texel snapping can hold that grid still. Leaves a texel or two wide cover different texels each
frame. Freezing the map's contents cuts the measured pulse by about a third; a lower compare reference or four offset
compares lower it by about 12 %, and removing the receiver's normal offset raises it.

A grid four times finer keeps the game's projection: `shadow_scale = 4` makes the array 6400x6400 (it is the only map
`shadow_scale_of` enlarges on this save; the cascade sampler is the game's own linear compare, and the mask shader
never reads `shadow_spread`). The green-mask pulse fell from 17.4 to 7.0 on the fixed-step replay, and the owner saw
no flicker in play. OoT's default is now `auto` (`vk_texture.c vk_shadow_scale`): 4 on a discrete card with 12 GiB or
more of device-local memory, else 1, logged as `vk: shadow map scale N (auto: ...)`. The array grows from 15 to
234 MiB; the median frame time during the strafe went from 13.5 to 14.1 ms on an RTX 4090 at 2160p.
`REKARINA_VRAM_BUDGET_MB=8192` and `shadow_scale = 1` both keep 1600x1600.

There is one such array alive, made off the replay thread. The game's transient target allocator gives the cascade
texture a new address whenever the frame's set of targets changes, the first time on the scene's first 3D frame. At
first the replay thread made the array itself at its first use (the image-ahead thread gave up on enlarged maps, which
have no pool memory to bind), and at the move it made a second one, because the released entry was younger than
`REKARINA_TEX_RECYCLE`'s six frames. That was two 234 MiB allocations on the replay thread under the texture lock: 41 ms
each in one census run, which took that frame to 139 ms. Now the image-ahead thread makes the array when the game
initializes the texture (`vk_texture.c pm_scaled_n`; 0.6 to 1.0 ms there, in the same frame as its first use on these
replays). The replay takes it at the first use, whatever the address (`pm_take_scaled`), and every later address takes
it over from the released entry in the same frame (`recycle_take`, scale above 1). The old entry has memory of its own
and the game renders the whole array before it samples it, so it may go as soon as no part being recorded writes it or
draws with it. The new entry's first barrier waits for the earlier work on the queue (`init_image`, `reused_now`). The
log says `vk: frame N: enlarged shadow map 6400x6400x3 made ahead`, then `taken from the image-ahead thread` and `taken
over from its released entry`, with 1 alive each time. Replays back to back on the installed build's objects with only
`vk_texture.c` changed, from the last arrival hold to the end: median 12.79 ms against 12.75, p99 16.62 against 16.44,
and no 234 MiB allocation on the replay thread. The worst frame left, ~85 ms on the scene's first 3D frame, is the same
in both: about 35 pipeline stage libraries built in play.

The override that made depth-compare samplers linear under `shadow_scale` is TotK's alone now: both samplers OoT binds
to the cascade array (420 and 422 in the pass dumps) are linear already, and its other depth maps keep their size.

## 2026-11-23: the fog's source draws were left out (REKARINA_SIM_NO_ATTACH off by default)

Upstream's notes (ec8414f), brought over with the change.

Since 2026-11-08 every 512x512 RG16F and RGBA16F image with storage usage was made without render target usage
(`vk_texture.c`, `REKARINA_SIM_NO_ATTACH`), meant for the ground fog simulation's maps, which only compute shaders write:
one of their passes had taken 9 ms where its twin took 0.2 ms in two of five runs on the RTX 5070. Every colour image of
an NVN2 game has storage usage here, and the game makes all of these images with `NVN_TEXTURE_FLAGS_IMAGE`, the targets
it draws the fog's sources into among them: a density source (four draws) and a velocity source (one draw) a frame
wherever the fog runs (seen in the temple in the woods and the meadow before it). Those draws were left out ("render
target format 0x29 target 1 (512x512) is not renderable here"); the simulation ran on its own state without them
(a frame dump in the temple: density at most 0.010 in one channel instead of 0.026). The flags cannot tell these
images apart, so the usages are kept now and `REKARINA_SIM_NO_ATTACH=1` is the old rule as a test switch.

Cost, kept against dropped, ABBA, base mode, RTX 5070 at 1440p, no frame limit: the temple's block room on a player's
save (camera turns) 64.2 / 64.1 fps against 64.3 / 65.2, GPU busy p50 15.07 / 15.14 ms against 15.08 / 14.88; the GPU time
of all passes of two frames 28.30 ms against 28.12 (the fog's passes are each under 0.03 ms); the forest, the field and
the Market in one tour 102.2 / 112.5, 82.4 / 97.0, 85.8 / 98.0 fps against 87.7 / 103.6, 89.6 / 94.1, 88.7 / 99.3 (GPU p50
8.80 / 8.22, 10.21 / 9.49, 7.82 / 7.43 ms against 8.67 / 8.76, 10.05 / 9.50, 7.66 / 7.40): the runs drift more between
the first and the last of a set than the arms differ. No pass of 9 ms came back. The integrated Radeon at 720p, standing in the meadow before the temple where the fog
runs: 10.2 / 9.7 fps, GPU busy p50 92.0 / 93.1 ms kept, against 9.8 / 9.9 fps and 92.6 / 91.2 ms dropped.

## 2026-11-21: NonUniform marking off by default on AMD (1.2.9.1)

Upstream's notes (1.2.10, ff6548e), brought over with the change.

1.2.9 decorated texture array accesses NonUniform by default on every GPU but NVIDIA's (the section above). On an
RX 6600 (AMD's Windows driver, RDNA 2) the game then stopped now and then in the first dungeon as new effects came up
(enemy deaths): an access violation inside amdvlk64.dll while a pipeline worker linked a pipeline from its four stage
libraries (`vk_pipeline.c` `gpl_link` from `pipe_worker`), right after a stage library was built in play. With
`REKARINA_NONUNIFORM=0` the same fights, and the dungeon's boss, ran without a crash. On RDNA 3 / 4 the decoration did not
change the flicker it was meant for. So it is off by default on AMD now (every AMD driver), as on NVIDIA; other GPUs keep
it, and `REKARINA_NONUNIFORM=1` turns it on anywhere. The libraries are kept for the life of the process and the link
takes four of them under stage-independent set layouts, whether their modules were decorated or not; nothing points at
our side of the link, so the driver's handling of the decorated modules is the likely fault.

## 2026-11-23: foliage shadows: the cascades, the mask and the shimmer in motion

Upstream's notes (00cb670), brought over with the change. They name upstream's Shadow resolution row and its set `sunfit` (map sizes up to 3072), which this fork does not have: here `shadow_scale` draws the cascades at 4x (6400 a side) on discrete cards with 12 GiB or more.

Players reported tree and bush shadows that "boil" in motion on NVIDIA cards (DLSS and DLAA mostly), and shadows that
end sooner than in other ports. Measured on the RTX 5070, 1080p, at the village (a warp to a spot by the cliff).

What draws the shadows (a frame dump):
- foliage (tree crowns, bushes, grass cards): the sun's three cascades, one D16 array of three layers, 1600 texels a
  side (the game's quality level 1; its table holds 1000/1400/1600, 1600/2000/2200 and 3072). Cascade 0 covers the view
  to 9.8 m and is orthographic; cascades 1 (to 32.3 m) and 2 (to 160 m) carry a perspective warp along the view. The
  cascades hold nothing else;
- buildings, terrain and trunks: a static 12288x12288 map made at the load;
- characters and moving objects (the hero, a chicken, a villager, the windmill): square maps the game sizes every frame;
  the grass: a 2400x2400 map of its own;
- one screen-space mask, RGBA8 at the scene's size, gathers all of them before the lighting reads it: five full-screen
  passes with the scene depth attached (the grass's and the characters' maps) and one without (the cascades and the
  static map), MIN-blended; green holds the hard shadows, blue the grass's soft one, red the occlusion.

Why they shimmer in motion: the game fits each cascade to the view every frame (main+0x713120): the two sideways axes of
its light basis follow the camera's direction, and the box is the tightest around the cascade's slice. A world point in
front of the camera moves in every cascade's map by 1.7 to 1.9 texels a frame while the camera turns (the field's
drawbridge, 60 fps; median, 78 % of frame pairs more than 0.1 texel), by 1.23 / 0.55 / 0.10 texel in cascades 0 / 1 / 2
while the hero walks, and not at all while both stand (the cascades as guest memory holds them, every frame:
REKARINA_SUNFIT_DUMP, runtime/oot_sunfit.c). Less the whole texels the move is 0.34 texel at the median (the village):
every frame each leaf's shadow lands on another sub-texel phase of the grid. The jitter DLSS gets does not reach the fit
(the camera projection the fit reads stays constant in DLAA). In the mask (captured every frame with
REKARINA_MASK_SERIES; its green warped onto the next frame by the picture's optical flow; a toggle is a step up then
down, or down then up, of more than 24/255), base mode: 0.57 % of pixel-frames toggle standing still, 2.27 % turning,
2.08 % walking. The console's own temporal upscaler hides most of it.

In DLAA the mask toggles in about 3.1 % of pixel-frames whatever the camera does (9 % of the pixels flip in more than
a tenth of the frames standing still): the jitter samples shadow detail finer than a pixel's footprint (on the far
ground one pixel spans several cascade texels along the view). That is the signal DLSS integrates; standing still it
does (the picture toggles in 0.3-0.4 % of pixel-frames, against 2.1 % in base mode). Moving the mask passes by the
frame's jitter, so that their rays meet the jittered depth (a test build), did not lower it.

What we had added on top: REKARINA_FOLIAGE_SHADOW_CULL (from cascade 1 on, one side of the double-sided cards) and
REKARINA_FOLIAGE_SHADOW_VRS (from cascade 2 on, 2x2 shading). A card is one plane, so the cull removed the shadow of
every card whose culled side faced the sun: beyond 9.8 m every tree's shadow had about half its cards, and the far
ground's share in shadow rose from 6.59 % to 7.83 % of the upper half of the picture with both off. The 2x2 rate decided
each leaf's cut-out per block of 2x2 texels, blocks that the moving grid shifted every frame. With both off, base mode:
the mask's toggles 0.56 / 2.08 / 1.97 % (still / turn / walk), the picture's 1.69 / 4.28 / 3.35 % against 2.13 / 5.37 /
3.55 %; DLAA, the picture's while turning 1.25 % against 1.48 and 1.62 % (whole-picture flow toggles, three runs).

Both are off by default on NVIDIA now (vk_exec_draw.c foliage_shadow_default), where they bought no frame rate. Base
mode, 1080p, 240 fps limit, the camera turning, 1,500 frames from 1,000 after the arrival, ABBA (default, off, off,
default): the village 131.1 / 130.1 against 132.1 / 130.5 fps (1 % low 97.8 / 97.0 against 100.9 / 91.0), the forest
127.6 / 131.4 against 131.8 / 131.0 (87.7 / 99.8 against 92.5 / 96.5), the field's drawbridge 117.8 / 109.8 against
116.8 / 117.2 (57.9 / 62.3 against 62.0 / 60.3). The three cascades' passes take 0.07 ms of GPU a frame either way
(REKARINA_GPU_PASSTIME, two frames: the forest 0.130 against 0.120 ms, the village 0.148 against 0.150 ms). Other cards
keep the shortcuts (nothing measured there; a Radeon RX 9060 XT player reported frames lost with both off).

The distance: the cascades end where the game ends them (9.8 / 32.3 / 160 m in the mask pass's uniform block), as in
the other ports, none of which changes it; what looked shorter here was the culled cards.

Tried first and kept out of the tree then (since in, as the Shadow resolution row: "Shadow resolution (set `sunfit`)"
below): the game's own larger
maps (its table's 2200 and 3072, through the size the sun's shadow object picks at main+0x4729f0) and a steady cascade 0
(a light basis fixed in the world and carried with the sun, a square box of one size on whole texels: the approach of
the Longshot project, own code). The steady fit kept cascade 0's texels on their points of the world (the move less
whole texels, p50 0.000 against 0.33, turning and walking), but the mask's flicker at the village did not drop with it:
the motion flicker is in the far half of the picture, cascades 1 and 2, which carry the perspective warp and cannot be
held still without losing resolution (a steady square at 1600 gives 9.9 texels a metre on the ground sideways at 10 m,
where cascade 1 has 27). The larger map lowered the mask's flicker everywhere (shortcuts off, still / turn / walk): 0.64
/ 2.18 / 1.92 % at 1600, 0.52 / 1.93 / 1.71 % at 2200, 0.45 / 1.86 / 1.54 % at 3072 (with the steady cascade 0), at no
cost seen on the RTX 5070: the cascades' passes take 0.07 to 0.09 ms of GPU a frame at any of the three sizes, and the
frame rate stayed within run-to-run noise (the village 128.3 / 131.7 / 129.9 fps at 1600 / 2200 / 3072, the forest 124.8
/ 131.5 / 131.0).

## 2026-11-23: the shadow mask filtered over time (a test switch)

Upstream's notes (4b197fc, a19c173), brought over with the filter. The map sizes they name (3072, Automatic) are upstream's Shadow resolution row, which this fork does not have.

### 2026-11-24: the depth clamp with negativeOneToOne (two test switches for RDNA 3 / 4)

BatzZie's next runs (RX 7900 XTX, 2.0 RC1) ruled the clear and the mesh path out: with REKARINA_MESH_EMULATE=1 (mesh
draws through the compute emulation and an ordinary vertex pipeline) the far shadows still flicker, and his own Vulkan
layer, which turned the depth clamp off for every draw into the cascade array, had all three layers filled at frame
3060 (41,507 / 37,769 / 18,645 distinct values) and no flicker; his layer logged every shadow draw with viewport 0..1,
no depth bias, LESS, test and write on and the depth clamp ON, the same for the good and the bad layers. The game draws
in NVN's -w..w depth mode, which every pipeline here takes through VK_EXT_depth_clip_control's negativeOneToOne, and
asks for the depth clamp on its shadow draws. The two other ports of the game never use negativeOneToOne: their vertex
and mesh stages convert the depth themselves (eden's convert_depth_mode), and they keep the game's clamp; Longshot does
not flicker on his card. So the pair, clamp with negativeOneToOne, is what RDNA 3's driver gets wrong.

- REKARINA_DEPTH_NDC=shader (REKARINA_MESH_NDC=shader the same; gfx_shader.c ndc_twin_make / ndc_convert): each vertex,
  tessellation evaluation, geometry and mesh shader gets a twin whose module writes z = (z + w) / 2 itself, with its own
  key and input id (its own pipelines, pipeline libraries and no pipeline-file record); a draw takes the twin of its
  LAST stage before the rasterizer only (vk_exec_draw.c prepare_draw and prepare_stages: the mesh shader; the geometry
  shader, else the vertex shader; the emulated mesh draw's own vertex shader), and that pipeline is made without
  negativeOneToOne (the key carries it). The clamp stays as the game asks. Where the conversion goes: a mesh shader's
  stores to its output block's Position (a 2-index access chain; a store to another member chosen by a constant, the
  point size, stays as it is); the vertex stages' Position (a variable of its own, or the member of the gl_PerVertex
  block the translator declares) before each return of the entry point; a geometry shader's before each emit, its own
  value stored back after it. A shader with any other use of its Position block gets no twin and keeps
  negativeOneToOne; the log counts the shaders converted and left, the first 4 left with their key and reason.
- REKARINA_SHADOW_CLAMP=0 (vk_exec_draw.c shadow_clamp_off): draws into a D16 depth target that is one layer of an
  image of several, with no colour target (the sun's cascade array), are made with the clamp off, as BatzZie's layer
  did; logged once.

Checked on the RTX 5070 (1080p base mode, the game's 1600 map, the forest and the field's drawbridge at exact spots,
every pass's depth target dumped at frame 2600, each place plain twice and once with the switch; targets paired by order
within each size and format; the median share of pixels off by more than 2, two plain runs against plain and switch):
the field's scene depth 0.894 / 0.804 %, its cascades 0.378 / 0.320 %, its far map 0.255 / 0.188 %, the forest's scene
36.6 / 25.8 % and its cascades 2.889 / 3.193 % (the wind moves the forest between runs): the switch draws the same depth
as negativeOneToOne, within the runs' own differences; every cascade pass's distinct values and shares at 1.0 and 0
within the plain runs' spread. Coverage in the forest: 1,024 shaders converted in the first minutes, none left. No
FATAL, no lost device in any run. REKARINA_SHADOW_CLAMP=0 is not neutral: in the field's cascade layer 0 the share at
0 falls from 35.4 to 0.2 % and the share at 1.0 rises from 14.9 to 26.4 %, layer 2's from 70 to 94 %: what lies in
front of a cascade's near plane (tall casters) is clipped instead of clamped onto it, and its shadow is lost. So
DEPTH_NDC=shader is the candidate fix with no picture change; SHADOW_CLAMP=0 a fallback with that cost. A Radeon RDNA 3
/ 4 player's dump decides it: with REKARINA_DEPTH_NDC=shader, cascade layers 1 and 2 drawn (thousands of distinct
values, as layer 0) and no flicker in play.

2026-11-24, night: the player's two runs of 2.0.0-rc.2 at his spot decided it the other way. With
REKARINA_DEPTH_NDC=shader (the log's line of the switch, over 3,000 shaders converted, none left) the forest flickered as
before; with REKARINA_SHADOW_CLAMP=0 there was no flicker and every layer was filled. So the clip space is not the
cause and the conversion is not the fix: the fault goes with the device's depth clamp (the near and far clip off) on
that hardware, in either clip space. What the lost layers have that the good one has not: cascade 0 is orthographic,
cascades 1 and 2 carry a perspective warp, so only they have vertices behind the projection's eye (w < 0), which a
rasterizer without a near clip must cut at w = 0 itself. Unproven that this is the step RDNA 3 gets wrong.

REKARINA_SHADOW_CLAMP=auto|shader|0|1 (vk_exec_draw.c shadow_clamp_mode, gfx_shader.c gfx_shader_near_twin). shader:
a draw into the cascade array that the game clamps goes without the device's clamp, and its last stage before the
rasterizer is a twin whose module writes z = z < -w ? -w : z at every Position (the depth twins' converter with a mode;
z < 0 ? 0 : z after the conversion when REKARINA_DEPTH_NDC=shader is on as well). A primitive in front of the near plane
lies on it as with the clamp; one that crosses it is drawn whole, a little deeper between the crossing and its inner
vertices; the far side is clipped, where the clamp writes 1.0 into a map cleared to 1.0 under LESS; a vertex with
w < 0 gets a depth beyond the far plane, so the far clip cuts its primitive where w crosses 0; a NaN stays one and is
clipped. The twins are made at a cascade draw's first use of a shader (a handful of mesh shaders in the field), with
their own key and input id. auto, the default: shader on AMD's Windows driver for RDNA 3 and 4 (vk_caps.amd_prop
without amd_sc), the device's clamp (1) on every other driver. REKARINA_DEPTH_NDC's default is off again.

Checked on the RTX 5070 the same way (the field in front of the Market, frame 2600, plain twice, the switch, the clamp
off): the share of pixels off by more than 2, two plain runs against plain and switch: the scene's depth 0.642 /
0.669 %, the cascades 0.243 / 0.196 %, the far map 0.099 / 0.079 %. The cascade layers' share at 1.0 (nothing drawn)
and at 0 (held on the near plane): plain 14.9 | 35.4, 57.7 | 7.2, 69.9 | 24.2; switch 14.9 | 34.6, 57.7 | 5.6,
69.9 | 24.2; clamp off 22.6 | 0.3, 64.0 | 0.0, 94.0 | 0.0. So the shader's hold keeps every caster (the cover equals
the plain runs' to a tenth of a percent) where the clamp simply off loses a third of the far layer's; the 0.8 and
1.6 points fewer texels at exactly 0 in layers 0 and 1 are the triangles that cross the near plane. No fault, no
lost device. Not run on an RDNA 3 or 4 card: these draws now use the state that player's second run had, with the
casters kept; his run of the next test build decides it, REKARINA_SHADOW_CLAMP=0 beside it to compare.

## 2026-11-23: the shadow mask filtered over time (on by default since 2026-11-24)

What still shimmers with the larger map is the far half of the picture, the sun's cascades 1 and 2 (10-160 m): they
carry a perspective warp along the view, so they cannot be held still in the world without losing resolution ("Shadow
resolution" above), and their texels slide under the scene in every frame of motion. REKARINA_MASK_FILTER=<percent> (80
by default in OoT since 2026-11-24, 0 turns it off; runtime/vk_maskfilter.c) filters the one place every sun shadow
passes through, the screen-space shadow mask, over time: after the sun's pass into the mask (the frame's last write to
it) a compute pass (runtime/mask_filter_spv.h, written by tools/make_mask_filter.py) takes each pixel's surface to
last frame's picture by the camera's motion (its depth, and this frame's view-projection against last frame's from the
game's scene context block, as the motion vectors of DLAA take them), reads last frame's filtered mask there, holds it
within the range of this frame's 3x3 neighbourhood, so a shadow that moves or ground that comes into view takes this
frame's value at once, and mixes it in by the weight (green and blue, the hard shadows and the grass's; red and alpha
pass through). The result is copied over the mask and kept as the next frame's history (two RGBA8 images of the
scene's size). A first version that read the mask as a storage image crashed once inside NVIDIA's shader compiler
(nvgpucomp64.dll) in the field; the mask is read through a sampled view now, and the same runs played through.

The mask's toggles with REKARINA_MASK_FILTER=80 against none (1080p, the map at 3072 as Automatic picks it on the RTX
5070, the same method as above; still / turning / walking, lower 40 % | upper half where it matters):

| place, mode | off | 80 |
|---|---|---|
| the forest, base mode | 0.15 / 2.04 (0.94 \| 2.73) / 1.64 % | 0.03 / 1.04 (0.34 \| 1.66) / 0.48 % |
| the drawbridge, base mode | 0.03 / 0.52 / 0.24 % | 0.01 / 0.15 / 0.08 % |
| the drawbridge, DLAA | 2.43 / 1.41 % | 1.23 / 0.77 / 0.64 % |

(the forest's walking figure off is batch 1's 3072 run, 108 frames; the DLAA run without the filter ended before its
walk). The whole picture, where most toggles are other things than the shadows: the forest turning 3.00 % against
2.80-2.88 %, walking 3.23 % against 2.80 % and 3.26 % (two runs), DLAA standing 0.09 % against 0.06-0.07 %. Looked at
frame by frame, the camera turning around the hero in the forest and the hero walking onto the drawbridge and past a
tree: no trail of the hero's shadow or of anything else, the legs' shadows on the stones as sharp as without it.

Its frame rate (2026-11-24, RTX 5070, 1080p base mode, 240 limit, the camera turning, 1500 frames from 1000 after the
arrival, off / 80 / 80 / off): the field 114.2 and 121.0 fps off (1 % low 56.2 / 57.9) against 113.3 and 118.0 with
the filter (58.1 / 64.3), the forest 131.3 and 126.4 off (93.3 / 82.7) against 131.3 and 129.0 (96.0 / 88.5): the
means -1.6 % and +1.0 %, inside the spread of each arm's own two runs (up to 6.8 fps), the 1 % low higher with the
filter in both places, no frame over 25 ms more. The integrated Radeon (RDNA 2, 720p base mode, the forest, the camera
turning, 600 frames from 300 after the arrival, off / 80 / 80 / off): 6.2 and 6.1 fps off (1 % low 4.9 / 4.8) against
6.3 and 6.1 with the filter (5.0 / 4.7), no FATAL, no lost device. So the filter is on by default in OoT;
REKARINA_MASK_FILTER=0 is the old picture as a test value.

## 2026-11-25: the exposure step when the item bar closes (the bridge over a long gap)

Upstream's notes (9f9cd9f's change, 960a5af), brought over with the change.

A player's report and the owner's recording: a sharp darkening and recovery when the item bar (PLUS) opens outdoors,
and a brighter picture for a moment after it closes. Reproduced by script in the field at noon (a 60 fps limit, the bar
opened ten seconds after the arrival and closed five seconds later, twice; the flight recorder's thumbnails and
`REKARINA_PROBE_FS=1893e0a44a9ee521`, the sky LUT pass's exposure factor, every frame).

The close. The factor stood at 2.72 before the bar and at 2.85 for the 14 frames after its close: one push of the
game's 15-frame average from a read that found no sum, the step the pause menu had before the bridge of 2026-11-20.
The bar draws no scene of the world (the background is a 640x360 capture of it, drawn with constant parameters under a
40 % dim), but it draws the hero and the items: 220 draws a frame against 2,100, enough texture bindings to count as a
scene for the bridge's test. The environment's sums stop for as long as the bar is open (no float-sum range in any of
its frames), so the close was logged as a skip in play ("had no sum for the wait ..., 303 frames after its last, with
the scene drawn throughout") and not bridged. Now a gap of eight frames and more is bridged whether a scene counted as
drawn in it or not (`vk_mem_late_apply`, LATE_PLAY_GAP); the skips in play that stay unbridged are one frame long (8 of
8, 17 of 17 and 15 of 15 logged in three tours; the longer ones logged there, 14 to 1,377 frames, are map changes).
After: 2.73 before the bar and 2.73, 2.73, 2.72 after it, and the thumbnails' mean 66 before, 66 / 65 / 66 after
(66 then 68 to 69 before the change).

The open, not solved. The thumbnails' mean: 66, then 25 for two frames (the live scene under the 40 % dim), then 15
from the first frame that shows the capture, rising by about one every other frame to 22 and staying there. The game's
parameters of the background draw and of the dim are constant over those frames and no uniform of any pass differs
between two consecutive frames of the rise, so the rise comes from an image's content: in every bar frame the game
filters an environment cube again down its mip chain (85 small passes). Whether the step from 25 to 15 or the rise
to 22 is the game's own cannot be told without a reference picture of the console.

## 2026-11-21: the mesh emulation's buffer address from push constants (translator patch 0047; upstream numbers it 0045)

Upstream's notes (1.2.10, ff6548e), brought over with the change; its patch 0045 is 0047 here, after this fork's own 0045
and 0046. Here the `SAME_AS` entry waits for the same comparison with this fork's translator (empty shader caches,
the same route with the earlier translator and with this one, every module both wrote compared).

On a GPU without mesh shaders (an RX 5700 XT; the integrated Radeon with `REKARINA_MESH_EMULATE=1`) the device was lost
at the first emulated frame with task draws on: the mesh kernel's first read of a task's launch count went to the
emulation buffer + 0x1000 with the address's high word all ones. The kernels held the buffer's address as two
32-bit constants the runtime patched into the module (`vk_pipeline.c` `meshc_module`), and AMD's driver for RDNA 1
and 2 lost the high word on the way to the 64-bit address.

Translator patch 0047 makes the kernels read the address from push constants (8 bytes at offset 0) instead:
`MeshStageWrapper.GenerateAsCompute` declares a push-constant block with a `uvec2` and bit casts it to the 64-bit base.
Only the emulation kernels change (`GFX_STAGE_MSC`), so `tools/build_xlate.py` keeps the translator identity
(`SAME_AS`) and `gfx_shader.c` `MESHC_REVISION` 4 gives the kernels new keys. The runtime gives such kernels' layouts
the push-constant range (`Bindings.emu`, `layout_get`) and `mesh_emulate` pushes the address before each task and
mesh dispatch (`emu_push_address`); a module without the marker constants is used as it is. With it: a direct draw of
more workgroups than the buffer holds meshlets runs in pieces, the crash record names the emulation step a thread
was in, and `REKARINA_MESH_EMU_LOG=first` logs the first 64 emulated draws.

Proof: the integrated Radeon with the emulation forced and task draws on, 220 s through the forest into the field: no
device loss, 27 kernels translated in play, none refused, no emulated draw skipped; the ground's grass the same as
with the Radeon's own mesh shaders at the same frames. The identity: 70 s on the RTX 5070 into empty shader caches with
1.2.8's translator and with this one: every module both wrote is byte for byte the same (3,863 files; the differences
are stored inputs and the provisional keys in .res.json, which follow the run's constants). `tools/update_cost.py` from
1.2.8: only the compile step runs again; no shader is translated again.
