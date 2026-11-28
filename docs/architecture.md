# Architecture

How the Switch build of TotK becomes a native Windows x86-64 program. The upstream projects it builds on are listed under Credits in the [README](../README.md).

## Starting points

Two MIT projects cover different halves of the problem:

| | LAS Recomp | MacNativeGames |
|---|---|---|
| Game | LAS Recomp's game | TotK 1.0.0–1.4.3 |
| CPU | ARM64 lifted to C, runs on x86-64 | Original ARM64 code runs directly on Apple silicon |
| OS boundary | IPC services and SVCs | ~163 `nn::*` SDK functions replaced by symbol, 38 SVCs |
| Graphics | Own NVN layer on Vulkan, own Maxwell translator | NVN front end + Metal 4 back end, Ryujinx translator |
| Useful to us | Lifter, NEON-to-C lowering, Windows memory layer, tests | Everything TotK-specific: host behaviour, NVN front end, version tables, patches, shader list |

MacNativeGames cannot run on x86-64 as is: its CPU path executes ARM64 code natively. LAS Recomp's host only covers what LAS Recomp's game calls. The port takes the CPU half from LAS Recomp and the game half from MacNativeGames, and writes the parts neither has: a lifter that scales to TotK, and a Vulkan back end for the NVN front end.

## Pipeline

### 1. Input

- Base 1.0.0 (`0100F2C0115B6000`) and one cumulative update (`0100F2C0115B6800`), NSP or XCI, plus the keys of the user's console.
- ExeFS and RomFS extraction: MacNativeGames' Python extractor (Python 3.14 for `compression.zstd`, `.venv-py314`).
- Update merging: hactool at `1d64a83` with MacNativeGames' BKTR patches, built by `tools/build_hactool.py`.
- Verification: MacNativeGames' `expected-v*.json` hashes of ExeFS, RomFS and fonts.
- Primary version: 1.4.3, the last update; performance work targets it. It replaced all three modules of the earlier updates (new SDK 20.5.22, RELR relocations, audio renderer REV15) and builds with `--version v143`, `recomp/patches-v143.txt`. 1.2.1 is a second target on the same build path; 1.0.0, 1.1.2 and 1.2.1 share `sdk`, `subsdk0` and `rtld` byte for byte, so only `main` differs among them. Saves load on either.
- Windows driver: `tools/extract.py` runs the vendored extractors (`tools/vendor/macnativegames/`) and identifies the update by the SHA-256 of its `main`. hactool has no Darwin `funopen` here, so the base Program NCA is copied out of the NSP before the BKTR merge (about 17 GB of temporary space).

### 2. CPU: lift ARM64 to C

- Modules: `main`, `sdk`, `subsdk0`. `rtld` is dropped and linking is done by the host, as MacNativeGames does.
- Base: LAS Recomp's hand-written decoder (`lifter.py`) and NEON/FP lowering (`simd_c.py`, `simd.h`), MIT. Output: one C function per guest function, direct calls for `BL`, an address-keyed table for indirect branches.
- Rewritten for TotK scale:
  - generator runs in parallel processes, with bounded memory per worker;
  - jump tables resolved by our own symbolic pass instead of Unicorn, which keeps GPL code out of the shipped generator;
  - `LDXP`/`STXP` (128-bit exclusive pairs) supported;
  - no 64 MB limit on `main`; `subsdk*` loaded;
  - NZCV and the six most used general registers of each function kept in C locals instead of the `Cpu`
    (`recomp/flags.py`, `recomp/regs.py`): registers are stored back where they changed, before calls, SVCs and
    exits, and reloaded after them. A call reloads only X0-X18 and X30, as AAPCS64 has the callee preserve X19-X29 and
    SP; `gen.py --no-callee-saved-trust` reloads those too.
- Coverage oracle: MacNativeGames' NSO rewriter lists every site where game code touches the OS (SVCs, TLS reads, counter reads, `x18`, exclusive windows, jump tables): 68,968 sites in 1.4.3 `main`. The lifter must handle every one.
- Memory: guest address equals host address, all below 2^39 (IPC encodes 39-bit addresses). The heap, stacks and the alias region are reserved with `MEM_WRITE_WATCH`, so the graphics back end can ask which pages the CPU wrote since the last upload (`runtime/memory.c`).
- Indirect branches go through `rt_dispatch` (`runtime/dispatch.c`), which looks the target up in the address table and tail-calls it.
- Testing: LAS Recomp's differential harness, every lifted instruction class checked against Unicorn (test-only, never linked into the game).

### 3. Host

- Portable C from MacNativeGames: threads and synchronisation, IPC, memory services, fs/RomFS, time, account, applet, audio renderer and effects. The Apple pieces (mach VM, CommonCrypto, AudioToolbox, GameController) are replaced with Win32 and SDL3.
- SDK boundary at the `nn::*` function level, matching MacNativeGames. Only `audren:u` and `sm:` are real IPC services.
- Window, keyboard and mouse: Win32 (`gfx_window.c`, `input_win.c`). Gamepad and audio output: SDL3. Voice: libopus (BSD). Movies: the SDK's `movie::Decoder` is replaced by a host VP9 decoder over libvpx (BSD, static; `runtime/ov_movie.c`), because FFmpeg from MSYS2 is a GPL build. Switch 2 games decode video through `nn::codec` instead (decoders, image pools, sync handles and sync point events): the host serves that API with dav1d for AV1 (BSD, static) and libvpx for VP9 and VP8 (`runtime/ns2_codec.c`, `--probe-codec FILE`).
- zstd: most RomFS files of 1.4.3 are zstd frames. The patch set `zstd` sends the game's two decoder entry points to
  libzstd on the host (`ov_zstd.c`, `REKARINA_ZSTD_NATIVE`); the lifted decoder took almost three times as long.
- Settings and menu: every key with its default is in one table (`cfg.c`), read from `rekarina.ini` and `REKARINA_*`
  variables and written back when the in-game menu changes it. The menu and the performance overlay are Dear ImGui,
  drawn into the swapchain image (`runtime/ui/`); their text is in `ui_text.c` (English, Russian, Brazilian Portuguese,
  Simplified Chinese). The Chinese glyphs come from a system font (Microsoft YaHei, `plat_font_path`) merged into the
  font atlas. The Vulkan present thread publishes pending edits after present. The next guest frame latches the
  snapshot at window acquire; scene hooks, replay jobs, draw workers and the present job use that frame's snapshot
  without taking the settings mutex. The menu edits staging values and reads its own changes immediately.
  Startup environment overrides keep their existing precedence.
- OS calls the host makes go through `runtime/platform/` (`platform.h`, `win32.c`), so a Linux or macOS host adds one file instead of touching the rest. The split is not finished: `tools/check_platform.py` lists the Win32 uses left in the other files.

#### Threads and cores

Every guest thread is a host thread. TotK assumes Horizon's scheduler: three application cores, a thread runs only on a core in its affinity mask, and a higher-priority thread takes a core from a lower-priority one. The game keeps per-core data without locks and relies on that, so the host models it (`runtime/kernel.c`):

- Three virtual cores. A guest thread runs guest code only while it holds a core; the others wait in a priority-ordered list.
- Preemption. When a waiting thread has a better priority than a holder on a core it may use, exactly one holder is flagged. The holder checks the flag at its next SVC and at every indirect call (`rt_dispatch`: two loads when nobody waits), checks again under the kernel lock that a better thread still waits, and only then hands its core over. The indirect-call check comes from SavageWindRecomp's BotW port; checking only at SVCs left a better thread waiting up to 26 ms.
- Asynchronous preemption (from MacNativeGames 8bf0893, on unless `REKARINA_ASYNC_PREEMPT=0`). Game code can run for
  milliseconds with neither (`REKARINA_CORE_DIAG=1` measures it): a waiter that has spun 250 us suspends the holder and,
  if its host rip is in lifted code at host depth 0 (not inside a host call or SVC), moves it to `kern_ap_stub`, which
  saves the volatile registers and gives the core up there. A holder of a guest mutex counts at the priority of its
  best waiter, and one inside an SDK spin lock (TLR+0x100 `disable_count`) is not preempted: its interrupt flag is set
  and the switch happens at SVC 0x36. `REKARINA_ASYNC_PREEMPT=2` keeps those two without interrupting. The interruption is
  Win32 (SuspendThread, SetThreadContext); another host needs a signal handler there, as MacNativeGames has.
- Host work gives the core back. Queue submits, presents, GPU waits, shader translation, `nn::fs` calls and NVN memory pool setup run without a core (`NVN_BLOCKING`, `kern_core_away`), as the Switch blocks those threads in IPC, so a thread blocked in the driver or on a file does not hold up guest threads.
- Spin waits. Loops that only another thread can end (857 in 1.4.3, 864 in 1.2.1, found by `tools/spin_waits.py`, the
  nxnative search) send their back-edges through `rt_spin_gate`, which pauses the core once per pass and hands it to a
  waiter after 50 us of polling (`REKARINA_SPIN`, `REKARINA_SPIN_HANDOFF_US`).
- `free_cores = 1` (`REKARINA_FREE_CORES=1`) turns the gate off, for experiments only: the game crashes or hangs, usually
  at start, because its job workers all report core 0 and share per-core memory (#212).
- Fewer processors than cores. With n logical processors, fewer than the console has application cores (the process's
  affinity mask counts), at most n cores are held at once, at least two (`kernel.c` budget mode, `g_budget`), so no
  more game threads run at once than the host can run and the game's own priorities decide who waits: the host's
  scheduler knows none of them and makes a woken thread wait out another's time slice (33 ms measured). Every core
  stays its own: a thread waits while its own core is held, as on the console, or while n cores are held, and then a
  better thread takes the core of the worst holder wherever it is (`core_kick`, `core_grant_budget`). The presenting
  thread keeps its core while it replays a submission, as the replay needs a processor (`kern_core_away_soft`; a
  game thread of better priority takes the core at once), a sleep short enough to spin keeps the core while no thread
  waits for one (`sleep_short`), and a core waiter spins 20 us instead of 250 before it interrupts a holder.
  `REKARINA_GUEST_CORES=<n>` sets the count (the console's own count turns all this off); `REKARINA_FOLD_MAP=1` folds
  the cores onto n fixed slots instead ({0} {1} {2,4} {3,5}), `=0` as releases up to 1.2.2 did (core c in slot c % n),
  and each thread is then told a core of its own mask in the slot it holds (`core_number`), so two running threads
  never report the same core; `REKARINA_FOLD_SUBMIT_SLOT`, `REKARINA_FOLD_SLEEP` and `REKARINA_FOLD_AP_SPIN` (0 or 1)
  switch the other three.

### 4. Graphics

- NVN front end from MacNativeGames (`nvn_api_*.c`, `nvn_obj.h`, texture layout and deswizzle). The game records command buffers through it; nothing reaches the GPU until the game submits.
- Vulkan 1.3 back end in place of MacNativeGames' Metal one. At `QueueSubmitCommands` the recorded NVN commands are replayed into Vulkan command buffers (`vk_exec*.c`).
- Memory pools. Apple's unified memory let MacNativeGames use the game's pools zero-copy. The back end imports the pools as host memory (`VK_EXT_external_memory_host`). On a discrete GPU, pools bound as buffers are mirrored in device-local memory and the pages the CPU wrote (write watch, above) are uploaded before a replay; idle pool regions are made read-only so the write-watch queries skip them (`vk_memory.c`). GPU-rendered textures are written back when the CPU reads them: on a pool invalidate, on a CPU copy, and for small targets once the CPU waits for a fence (`vk_texture.c`).
- Replay thread. `REKARINA_REPLAY_ASYNC=1` (the default) moves the replay off the submitting thread onto a queue at most one frame behind the game; reads of GPU results wait for the newest writer instead of draining the queue (`vk_backend.c`). Where the game waits for the GPU it overlaps the replay with the game threads. It can add up to a frame of latency; if frame pacing is uneven, set `replay_async = 0` and restart to compare.
- Recorder thread. The Vulkan calls the replay makes into its command buffer are made by a thread of their own (`vk_rec.c`, after Savage Wind Recomp and LAS Recomp): each `vkCmd*` copies its arguments into a packet of a ring and the recorder makes the calls in order; calls into other command buffers are made at once; the recorder also ends the command buffer, and the submit waits until it has. In the town on an RX 9070 the replay takes 17% less time on the submitting thread. It is off by default on AMD GPUs, where the driver loses the device or hangs the GPU at startup with it (`REKARINA_VK_RECORDER=1` forces it on, `=0` makes every call on the submitting thread).
- Uniform snapshot. The game rewrites per-object uniform blocks on other threads soon after it submits the frame
  that reads them, while the replay still records it. At the start of a submission's replay every uniform range it
  binds from CPU-written pools is copied, and the draws bind the copy (`vk_usnap.c`, after MacNativeGames'
  `REKARINA_UBO_SNAPSHOT`); without it, late draws read another object's matrix.
- Driver data. The NVN driver keeps a per-stage constant bank (c0) that shaders read for things like point size, dispatch size and base instance. The back end fills it the way the driver does.
- Shaders: the Ryujinx translator 1.3.2 with MacNativeGames' patches and ours (`tools/xlate/patches/`), built with .NET 10 NativeAOT as `totk_xlate.exe` (a self-contained JIT exe of the same name where MSVC is missing) and run as a separate process. Its SPIR-V goes to Vulkan directly. It reads Maxwell code (TotK) and, with job `"arch": "sm86"`, the Ampere code of NVN2 (Switch 2) shaders through the Ampere frontend of MacNativeGames and seedds/nxnative ([oot-xlate.md](oot-xlate.md) has the request fields and what is not supported). `tools/prepare_shaders.py` and `rekarina.exe --prepare-shaders` translate and compile the variants of MacNativeGames' shared list ahead of play (29,587 for 1.4.3); the rest are translated on first use and cached on disk.
- Pipelines: `VK_EXT_graphics_pipeline_library` where the driver has it. Shader stages are compiled ahead into libraries; a draw that meets a new state links them at once (fast link), and an optimised pipeline replaces the linked one when a background thread finishes it. Pipeline keys seen in play are saved and built at boot. A draw whose pipeline is not ready is skipped when one of its pass's targets was rendered in the previous frame, and waits otherwise, so one-shot passes are never lost; a full-screen draw is skipped as well rather than waiting (`REKARINA_PIPE_SKIP`); the stage libraries a waiting draw needs are built on idle workers in parallel (`REKARINA_PIPE_WAIT`). Without the extension, a missing pipeline is built on a worker and the draw waits or is skipped by the same rule.
- Resolution. Patch sets make the game read its output, swapchain and scene sizes from the patch page
  (MacNativeGames' sites); `resolution.c` sets them from `output_res` and `render_scale` and grows the texture heap
  and the application pool to match. When the window is larger than the output, `vk_fsr.c` runs AMD FSR 1 (EASU,
  RCAS) on the present. An `output_res` that is not 16:9 (more than 1% off, `res_is_16x9`) also switches on the
  `aspect` patch set, which hands the output's ratio to the camera, and `romfs.c` `overlay_init` loads the
  `romfs-patches-<w>x<h>` overlay for it when it exists (`tools/make_romfs_patches.py --aspect`, laid out by
  `tools/ultrawide_layout.py`; the installer builds fifteen sizes: 21:9 2560x1080, 3440x1440, 3840x1600, 5120x2160,
  32:9 3840x1080, 5120x1440, 16:10 1280x800, 1680x1050, 1920x1200, 2560x1600 and 3:2 1920x1280, 2160x1440, 2256x1504,
  2880x1920, 3000x2000). A wider layout moves the HUD to the
  screen's corners, a narrower one to its top and bottom edges. Other sizes get the camera of their shape and the 16:9
  HUD stretched over it. The `aspect` setting (`fit`, the default; `stretch`) decides what the present does
  when the window's shape differs from the output's: `fit` letterboxes, where a 16:9 game in a wider window used to be
  stretched.
- HDR (`hdr = 1`, `vk_hdr.c`). The game tonemaps its scene into an r11g11b10f target, upscales it and draws the HUD
  into another float target, and a last draw copies that into the 8-bit sRGB window texture it presents. With HDR on,
  the tonemap's fragment shader (recognised by its constants and resource names, not a hash) runs as
  `shaders/hdr_tonemap.frag`, which keeps highlights up to the display's peak; the present reads the float target the
  window texture was copied from (any other last draw into it, such as a menu or a movie, leaves the window texture,
  shown at the HUD's white); a compute pass turns the picture into PQ code values in Rec. 2020 for an HDR10
  swapchain, which FSR 1 or the blit then scale; the in-game menu is drawn into an RGBA8 image and blended on at the
  HUD's white. Without an HDR10 surface format (Windows with "Use HDR" off, a compositor without colour management)
  the swapchain stays SDR. `REKARINA_SHOTS` reads the window texture, so its pictures clip highlights in HDR.
  PsychoV35's optional eye adaptation (`hdr_midgray = adaptation`) meters a 32 × 18 grid in the scene composite
  before tone mapping. The replay reads completed samples at the submission fence and applies a weighted,
  trimmed luminance histogram and temporal smoothing (`vk_hdr_adaptation.inc`). It uses the scene projection to
  weight the samples, with a 60-degree vertical field of view when no recent projection is available.
- DLAA (`vk_dlaa.c`, `dlaa`, off by default). NVIDIA DLSS at the scene's own size, through the driver's NGX core loaded
  at run time (no SDK code compiled in). It runs on the R11G11B10F light target right before the bloom chain's first
  pass, so post effects and the HUD read the anti-aliased picture. The scene's draws get a sub-pixel Halton viewport
  jitter; camera motion vectors are reprojected from the scene depth through this and last frame's view-projection of
  the scene camera block (constant bank c4), in double precision on the CPU. Per-object motion (`dlaa_objmv`,
  `vk_mvprev.c`, the per-object section of `vk_exec_draw.c`, translator patch 0015): solid scene draws whose position
  reads an object's own uniform blocks (bones c5, place c7, c6) are drawn a second time with a vertex shader variant
  that computes last frame's position from last frame's blocks (the game writes them into two copies that take turns;
  the uniform snapshot keeps both) and writes the difference into the motion image. `REKARINA_MV_CHECK` measures the
  vectors against the picture, `REKARINA_DLAA_TIMES` times the passes on the GPU. `fxaa` switches the game's own FXAA pass.
  OoT 1.1.1 jitters the projection on the CPU instead of the viewport (`patch_scene_projection`, on by default; after
  ShortFuse's NX2navi port, MIT); DLSS tries the checked pre-bloom route before falling back to the game's
  scaling dispatch. Two patched calls
  (`scene_projection_impl.h` `scene_projection_build`, `scene_projection_complete`) follow the scene's owner and view,
  offset the projection the game builds by a Halton phase (16 phases at DLAA) and record a scene-complete sample with
  the extents of its colour and depth attachments. The resolve takes that CPU camera sample and checks the settings,
  the depth subresource and the render extent there (`vk_dlaa.c` `projection_rows`), not each draw. A sample is a short
  lease: recording may run fewer than 12 samples ahead of replay while the uniform backing rotates, as long as the view
  generation and the attachments' extents stay the same (`scene_projection_feedback`). Camera reprojection uses the
  unshifted camera; a missing or stale sample leaves the game's own scaled picture for that frame and starts a new
  history. Model replay (`vk_skin_motion.inc`) uses the jitter recorded with the draw's camera binding: only the
  private last-frame camera bank gets the unshifted projection and a recomputed view-projection
  (`scene_motion_projection.h`), and the depth guard (`shaders/model_motion.frag`) adds last frame's raster jitter
  where it reads last frame's depth. DLSS and FSR 3.1 get the same render-pixel jitter, and frame generation reads the
  committed sample. With the patch off, the viewport jitter above applies.

  For comparison, `dlaa_pre_bloom = 1` (`REKARINA_DLAA_PRE_BLOOM`, off by default, read at start) opts into the
  native NVIDIA DLAA path on OoT when [checked pre-bloom DLSS](./oot-architecture.md#checked-pre-bloom-reconstruction)
  is disabled. It resolves before bloom so bloom and tone mapping read the anti-aliased scene
  (`vk_dlaa.c` `pre_bloom_draw`). DLSS writes back into the R11G11B10F light target right before the bloom chain's first
  draw: a pass of two R11G11B10F targets without depth whose fragment shader's input id begins `865a04d320227ae3` and
  samples the armed scene at slot 0 (the refraction and depth of field passes before it read the scene at smaller
  sizes). The check sits in `vk_dlaa_skip_draw`, which every draw already calls before binding its state; the draw's
  pass ends around the evaluation and begins again. At SMAA compose's dispatch, which reads the tone-mapped float
  picture and packs RGB9E5 into the light target, the picture is copied into a staging image of the runtime's own
  (`pre_bloom_stage`): the game's image may share physical-pool memory with others. At the scaling dispatch (compute
  shader `df55fc291b0f86ba`), `pre_bloom_scaler` copies the staged picture at 1:1 into the HUD's target. The game's
  scaler enhances fine edges even at 1:1, so replacing it avoids the dark outlines it adds over DLAA. The SMAA dispatches
  run for their guest writes. The copy keeps the results point and the HUD's target as frame generation's HUD-less
  source. If a check fails, the game's scaler runs and the log names the failed check.
  A frame where the draw is not found, or the evaluation fails, resolves at the scaling dispatch; the log names the
  reason once per change (`dlaa: pre-bloom waiting: ...`), and DLSS's history resets when the input switches.
  `REKARINA_DLAA_PATHLOG=<frame>:<count>` traces the scene samples, resolve and handoff. A frame dump records the
  HUD's target after the copy, before the HUD, as `dlaa_hud_destination`.
- Occlusion queries never block the CPU; a result that is not ready counts as visible.
- ASTC textures are uploaded as they are where the device reports `textureCompressionASTC_LDR`, and decoded by a compute shader everywhere else (`vk_astc.c`, `REKARINA_ASTC`). The CPU decoder (`tex_astc*.c`) is the fallback and the test oracle.
- Draw state. A draw whose program, bound state and textures repeat reuses the previous draw's shaders and skips rebuilding its descriptor sets; descriptor sets are written through update templates (`vk_exec_draw.c`, `vk_pipeline.c`).
- Allocation: Vulkan Memory Allocator. Debugging: Vulkan validation layers, per-pass frame dumps (`REKARINA_DUMP`, `vk_dump.c`) and a flight recorder (`REKARINA_FLIGHT`). RenderDoc cannot capture, because it hides `VK_EXT_external_memory_host` from the application.

Maxwell behaviour the back end has to match, and the TotK bugs other ports fixed: [maxwell.md](maxwell.md).

Why Vulkan rather than D3D12 or wgpu:

- The translator emits SPIR-V, which Vulkan consumes as is. D3D12 would add a SPIR-V to DXIL step; wgpu would add SPIR-V to WGSL through Naga and drop features the game uses (bindless textures, some atomics).
- LAS Recomp's Vulkan back end is a working reference on Windows.
- The same back end runs on Linux.

### 5. Toolchain

- Generated C: MSYS2 UCRT64 Clang + LLD. The generated code uses GNU extensions (`__int128`, `musttail`, vector types) that `cl.exe` cannot compile.
- Host code: the same Clang, C11.
- Builds: `tools/build_game.py` writes a Ninja file and runs MSYS2 Ninja. Every object is hashed before linking, because a faulty RAM module once flipped bits in stored objects.
- Analysis: Ghidra with the Switch loader, Capstone, `llvm-objdump`.
- Players: the installer ([installer.md](installer.md)) builds the game code on the player's machine with a bundled llvm-mingw, so no recompiled code is ever distributed.

