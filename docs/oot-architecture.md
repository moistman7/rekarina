# Rekarina: how OoT runs here, where the frame goes, and the ways forward (2026-11-08)

A map of the port for whoever works on it next: the parts and the data that flows between them, measured on the forest
porch (RTX 5070, Ryzen 7 9700X, 2560x1440 window, scene at 1920x1080), and an assessment of the other ways to build it.
Findings with their knobs: oot-notes.md; the plan in steps: native-driver.md; the frame-time ledger: roadmap-100fps.md.

## The parts

1. **CPU code: recompiled, native.** The game's ARM64 code (main and its modules, ExeFS of the 1.1.1 update) is lifted
   to C by `recomp/` (one function per guest function, `f_<address>`), compiled by clang for x86-64 and linked with the
   runtime into one exe. Guest threads are host threads; `runtime/kernel.c` gives them the console's view (virtual cores,
   priorities, sync objects, SVCs). The game logic is a few milliseconds of a frame and is not where time goes.
2. **The NVN2 API: reimplemented.** The game calls NVN through its import table; `runtime/nvn_api_*.c` implement the
   calls. Command buffers are recorded into the game's own memory as our op stream (`nvn_api_cmd.c`), exactly when the
   game records them. Memory pools are the game's memory: the GPU address of a pool is its guest address.
3. **The replay: NVN command stream -> Vulkan.** `nvnQueueSubmitCommands` runs `vk_exec_record` on the submitting game
   thread: it walks the ops, resolves addresses to buffers and textures (`vk_memory.c`, `vk_texture.c`), binds pipelines
   (`vk_pipeline.c`) and records a Vulkan command buffer, cut into parts (`vk_sync.c`) so the GPU starts before the
   recording ends. The game's six NVN queues go to one Vulkan queue in submission order.
4. **Memory between CPU and GPU.** The console has one memory. Here the pools the GPU reads a lot have VRAM mirrors:
   each part's prologue copies the pages the CPU wrote (write-watch) into the mirror, and at the end of a submission
   what the GPU wrote goes back to guest memory, as plain copies or, where the CPU may have written the same pages
   meanwhile, as a word merge against a snapshot (`wb_merge.comp`). Uniform blocks are copied at the submit
   (`vk_usnap.c`). Snapshot copies wait for both regular and late prologue uploads. Results the CPU reads
   (reports, events, counters) are written at the fence. A failed timeline query or a device-loss sentinel is not
   fence completion and cannot release command buffers or descriptor pools for reuse.
5. **Shaders.** NVN2 programs are Ampere machine code. `totk_xlate.exe` (Ryujinx's shader translator with MacNativeGames'
   and seedds/nxnative's Ampere front end, plus our patches 0021-0032) turns them into SPIR-V once; the results are
   cached (`build/oot/shader-cache`) and the Vulkan pipelines are prebuilt at boot from the keys of earlier runs. Memory
   reached through loaded pointers goes through the pool table (`vk_exec_draw.c` gmem_snapshot, patch 0032).
6. **Present.** The game's final image (R8G8B8A8_SRGB, upscaled by the game's own network from the scene resolution)
   is blitted to the swapchain; the bytes reach the screen unchanged.

The 1.1.1 interface renderer is called at `main+0x154a95c` (`main+0x15266c0`). The `interface_draw_scope` patch
marks direct and indexed indirect draw commands recorded during that call. Vulkan replay retains the mark only while
handling that command, independently of the bloom role.

`Advanced → DLSSG UI Reconstruction` is on by default and uses the Streamline 2.14.16 path with a version 6 bridge
when DLSS Frame Generation is selected. Turning it off takes effect after restart. The checked pre-bloom path is
controlled separately.
Ordinary draws before the HUD checkpoint can use replay workers. Checked interface draws and
checkpoint-dependent work stay on the replay thread.
For a supported draw onto the tracked post-upscale picture, replay snapshots that picture before the first interface
fragment. It draws the game UI normally, then replays the draw with its original shader into an alpha-only attachment.
Coverage accumulates as `new_alpha + old_coverage * (1 - new_alpha)`. The game's original blend is unchanged.
Additive, modulate, and other blend modes use shader alpha as best-effort coverage; the blend mode alone does not
turn recomposition off. A draw that cannot write coverage safely or a later write that invalidates the mask can
still invalidate the pair. The runtime log reports why a frame's
coverage is withheld when the reason changes. The present path accepts the original
image, a full-image transfer, or the checked 1.1.1 fullscreen SDR or PQ encode. It applies the same crop and fitted
rectangle to an R16 floating-point alpha image and tags it with the matching HUD-less picture. The HUD-less capture
records the active swapchain encoding separately from its storage format. A capture made before an encoding change
is ineligible after the change. For a PQ frame the host encodes HUD-less color using the checked guest output draw's
matrix and white scale, carried with the same replay frame and image. This path does not apply host color grading or
its own gamut choice. If that transform is unavailable or encoding fails, both recomposition inputs are omitted.
When UIR is active, the checked HUD-less capture also allocates and clears its matching alpha image. If the checked
output reaches present without another write, that alpha stays zero. The host overlay can still add coverage by
replaying its normal draw into a transparent target. If the pair cannot be built or validated, DLSS-G runs without either tag; a missing checkpoint or an
unsupported draw is not treated as proof that the frame had no UI.
When DLSS-G inputs are tagged, replay waits through the actual present call before submitting work that can reuse
those images. The retirement timeline is signalled after present on the presenting queue; untagged frames still
release replay immediately after the blit.
The host overlay uses the stock ImGui Vulkan backend. The menu blends into the game's linear composite before the
checked SDR or HDR output draw. Replay restores the guest's dynamic graphics state after binding the host pipelines.
Presentation takes alpha from the ImGui target, fits it with the game crop, and
includes the menu's relocated preview rectangle. The SDR fallback replays the host overlay into a transparent target
when a valid UI pair is available. HDR color composition does not transform coverage. A missing pair or an overlay
that cannot write coverage leaves
ordinary Frame Generation in use. If Streamline accepts both UI tags but rejects recomposition, the bridge retries
ordinary Frame Generation with the accepted tags intact. A visible host overlay does not by itself pause DLSS Frame Generation.
Alpha replay supports direct and indexed-indirect game draws, and fits their accumulated alpha at presentation.
Its fitted image retains storage usage across frames with no game UI, so extraction can resume without reallocating it.

## Live display output

The host advertises HDR only when the setting requests it and the surface offers HDR10. It asks the guest to refresh
that capability without changing docked or handheld mode. The output can lag the request by a frame, so presentation
follows the queued window texture: OoT 1.1.1 uses RGBA8 for SDR and RGB10A2 for HDR10. This is a game-specific contract,
not a rule for every Switch 2 game. The checked output draw still supplies the source for menu composition and the
per-frame PQ matrix for HUD-less capture; a later write invalidates that matrix record. If replacement retires the
old swapchain and then fails, the next frame retries without the retired handle. It does not present PQ as SDR.

Native HDR and RenoDX HDR share the guest's HDR10 output path. On OoT 1.1.1, the allocation patch selects RGBA16F for
RenoDX HDR at the game's float display compositor; SDR and Native HDR keep R11G11B10F. The format is chosen when the
game allocates the target, not when the swapchain changes encoding. The menu blends into that game-owned target before
the checked output draw, without replacing its sampled image. The menu coverage and optional preview scratch do not
hold a second full-size game picture.
The Graphics tab groups settings by feature: Output, UI, Render, Tonemapping, Grading, Effects, then Advanced.
Output owns the SDR/HDR signal, presentation and frame generation;
UI owns the performance overlay and HUD controls. The tonemapper selects the HDR compositor without a separate switch:
Vanilla uses R11G11B10F and PsychoV35 uses signed RGBA16F. Peak brightness and reference white sit with the tonemapper.
Tonemapping shows the active output's tonemapper (SDR defaults to Vanilla ACES, HDR to PsychoV35). Grading has one shared set of
scene controls. Neutral Vanilla keeps the stock fit; a non-neutral grade adjusts scene color before the ACES fit
without replacing the downstream guest output path. PsychoV35 uses its own grading once, fitting to BT.709 at equal peak
and diffuse white for SDR, or BT.2020 with the configured HDR luminance. The finished-picture SDR colour filters are not
used. Effects contains Bloom for both SDR and HDR. UI groups HUD layout and, in HDR, HUD and menu
brightness and UI EOTF Emulation (adjustable in RenoDX).
Under Render, Scene Upscaler switches live between None (Bilinear), NDSR (Vanilla, the default), DLSS, FSR4 when supported,
FSR3, and FSR1. Bilinear uses the tone-mapped picture before SMAA; NDSR keeps SMAA and the game's scaler.
Scene Render Scale shows
the selected preset's calculated input height as Auto, or allows a manual percentage with the game's reconstruction.
The Auto game picture height follows the configured window or fullscreen display at launch, with a 1440p minimum.
Neither Auto setting adjusts resolution to GPU load. Fullscreen, Window Size
and Display Scaling are in the same section. There is no separate upscaler enable switch; the existing INI keys remain
compatible. The Vulkan device prepares
supported backends at startup; replay uses the frame's selected backend and resets temporal history on a switch.
The guest render scale follows backend readiness as well as settings changes, including when the selected upscaler
becomes available during startup.
If DLSS is unavailable, temporal FSR takes its place when the configured FSR provider is supported, without changing the
saved selection. An unavailable FSR provider leaves the guest reconstruction in place. FSR 1 never selects a temporal
upscaler, including when AMD frame generation is enabled. FidelityFX loads on demand for temporal FSR, AMD frame
generation, or when Graphics checks for the FSR 4 provider on compatible hardware.
Game Picture Resolution sets the size of the picture including the HUD. Display Scaling fits that finished picture
to the window, after tone mapping; HDR uses bilinear filtering of the PQ picture. Final display scaling does not run
RCAS, since the picture already contains the HUD. At the bottom of Render, Sharpening (`sharpness`) controls a separate
pass on the reconstructed gameplay scene. The slider ranges from 0% (off) to 100% and defaults to 80% (0.4 stops).
It is disabled when NDSR is selected, and the extra pass is also skipped when another upscaler falls back to NDSR.
Switching upscalers does not change the saved strength. RCAS sharpens at game picture
resolution in the square-root domain, then returns to linear light. HDR PsychoV35 converts the signed BT.709 scene to
BT.2020 before the square root and back to signed linear BT.709 after sharpening. Frame generation and interface
coverage take their HUD-less snapshot after RCAS and before the HUD. The Vanilla reconstruction filter
is unchanged; FSR's integrated sharpening is not used. The
separate Texture Mip Bias Correction adds +0.000 to +0.500 mip levels toward zero on eligible negative sampler biases,
before any upscaler bias. Its legacy INI value `texture_lod_strength` defaults to 100 (+0.000 mip).

Anisotropic Filtering switches live by selecting new sampler-cache variants for tiled, mipmapped surface textures.
Frame Generation and Reflex do not change the scene upscaler. If DLSS Frame Generation is unavailable, frame
generation stays off rather than switching to AMD FSR. AMD frame generation switches live on a device
with its queues. AMD owns the swapchain while it runs; on the way back, the
presenter restores the Streamline swapchain when its hooks are available. DLSS frame generation switches live
when those hooks are already initialized. DLSS and DLAA prefer Streamline and can be paired with AMD frame generation.
Native NGX remains available on the legacy path.
Native Reflex serializes its Vulkan swapchain calls with acquire and present, but releases the lock before waiting on
its wake semaphore. Queued texture uploads do not poll or wait on a batch fence until the submitter has returned from
the Vulkan call that uses it.
When replay replaces compositor scratch images, it defers destruction through the recording's fence, not just device
idle: a command buffer that still names an old image might not have been submitted yet. The menu compositor keeps
its format-specific pipelines until shutdown for the same reason.
The checked gameplay
reconstruction path can provide depth, camera and motion vectors even when frame generation leaves the stock scaler
in place. Spatial FSR 1 uses those inputs for frame generation, not for its own scaling. Frames without matching
scene inputs do not supply new tags.

AMD frame generation receives the final backbuffer and, when the scene checkpoint matches the frame, a separate
HUD-less picture through `vk_fg_frame` (`HUDLessColor`). AMD uses this pair to detect UI; Rekarina does not register
a separate UI texture or alpha mask with AMD. DLSS frame generation can instead use the optional Interface
Recomposition path, which tags the HUD-less picture and an explicit UI alpha mask through Streamline.
AMD support requires the split FSR SDK 2 Vulkan libraries. Run setup again if the install has only the legacy library.
Frame generation uses Prepare V2 only and skips interpolation
when camera data is missing or the depth convention differs from the one used to create the interpolation context.
The context takes its depth flags from the first eligible camera and Prepare V2 receives the camera planes each
frame. The first eligible frame after a pause or context change resets interpolation history.
AMD frame generation attaches only after the renderer selects a compatible output: 8-bit UNORM for SDR or the SDK's
RGB10A2 layout for HDR10. It never changes the renderer's selection. When the output changes format, encoding, or
size, presentation drains the SDK's pending presents before replacing its swapchain and interpolation context.
If AMD fails during an in-place remake that skipped the drain, presentation skips that frame and retries replacement.
While a failed swapchain remains owned by AMD, presentation does not acquire more images from it.
If AMD frame generation has stopped and the drain fails three times, presentation reports a fatal error rather than
destroying a swapchain that may still have pending work.
AMD diagnostics report cumulative frame-generation totals and the SDK present count for the current swapchain.
Neither measures physical display scanout. Crash protection follows the backend being set up across live switches.
The full performance overlay labels a recorded NVIDIA scene resolve `DLAA` at equal input and output extents,
and `DLSS` when upscaling. A selected NVIDIA mode without a recorded resolve does not get either label.
The performance overlay names the frame-generation backend separately from the upscaler. For AMD it uses the
interpolation provider's reported major version, or shows FSR FG if the SDK does not identify the provider.
For upscaler comparisons `REKARINA_DLAA_TIMES=1` reports GPU reconstruction, recorder drain, and CPU SDK-call time
for both DLSS and temporal AMD FSR. Compare backends at the same render and output sizes with frame generation off.
Present publishes the resolved motion mode, lookup policy, and available consumers in the immutable settings ring.
The same snapshot requests HUD-less capture and DLSS interface coverage; replay does not query live swapchain ownership.
The restart-only skin-motion setting is folded into the model-motion flag before publication. Guest command recording
and Vulkan replay copy their motion policy before walking draws. Queued and held submissions
keep a reference to the snapshot they started with; a later present cannot update it underneath them.
The guest tonemap draw is tagged by call site; replay chooses the pre-bloom route for its pinned frame, so a reused
recording does not carry an earlier upscaler's choice.

The settings diagram separates target resolution, game-picture size and window fit from the gameplay pass order.
Its upscaler label is the selection, not a per-frame trace. DLSS and AMD FSR 3/4 share the checked scene-linear
reconstruction route before bloom and tone mapping. After tone mapping, the replay skips all three SMAA dispatches
only when it can verify the scaler destination ahead of time. Otherwise it keeps the guest chain intact.
`REKARINA_SKIP_SMAA=0` keeps the guest dispatches for diagnosis. Vanilla, Bilinear, and FSR1 use the post-tonemap scaler dispatch.
Vanilla scales the current frame's SMAA output. Bilinear and FSR1 use the tone-mapped float input to SMAA instead.
Below native resolution FSR1 filters in square-root space and returns to linear light before sharpening and the HUD.
For PsychoV35 HDR, EASU filters nonnegative BT.2020 values without clamping highlights to 1, then restores signed
linear BT.709. At 1:1 FSR1 leaves the game's SMAA and scaler in place. None use motion vectors or frame history for scaling.

For temporal upscaling, `REKARINA_FSR_VERSION=auto` picks FSR 4 when available and otherwise tries FSR 3. An explicit
request for an unavailable provider leaves the game's scaling active instead of selecting another FSR generation.

For developer diagnostics with the signed AMD FSR SDK 2.4 Vulkan DLLs, set `REKARINA_FSR_DEBUG_VIEW=1`
before launch to draw AMD's debug view when the temporal upscaler runs, or `REKARINA_FSR_FG_DEBUG_VIEW=1`
to draw its frame generation debug view on generated frames. These views alter the rendered image and are
off by default. The upscaler view does not apply to FSR 1.

The NVN draw hook selects the guest tone-map call site. The SPIR-V rewrite checks the shader structure before patching it.
Both HDR permutations are supported: one uses a
direct uniform gain and the other adds local exposure. Bloom scales the guest term for both
mappers, without changing pre-bloom reconstruction. Vanilla keeps the guest tone curve and honors the INI-only local
exposure setting; PsychoV35 always bypasses local exposure and optionally meters the scene for eye adaptation on devices with
fragment storage writes. PsychoV35 fits to BT.2020 but returns signed
BT.709 for the game-owned float compositor. The guest's final HDR encoder and its matrix are left unchanged in both
modes. HUD-less capture uses the matrix from that draw to match its PQ output. An unrecognized scene
shader uses the guest mapper. The frame settings snapshot carries the typed mapper choice for replay.
Render EOTF Emulation changes only PsychoV35's scene output; UI EOTF Emulation decodes the host menu before it is
blended into the guest compositor. Both offer Off, 2.2, and 2.4, and neither changes the guest's PQ encoding.
Draws consume those choices without reading the settings table for shader selection. The host's HDR Reference White and
peak are reported through the guest-facing display queries.

If Vulkan swapchain creation retires the previous swapchain and then fails, retries do not reuse the retired handle.

## Where a frame goes (2026-11-08 morning, realtime = 1, unlocked: forest porch 60-63 fps, lighter scenes 92-103 fps)

- GPU ~16 ms of work a frame: the game's passes and dispatches ~11 ms (the lighting's full-screen material passes
  ~4 ms, the forest's ~1,200 mesh-shader draws ~2.3 ms, compute culling, lights and fog ~3 ms, the upscaler network
  0.6 ms), ours ~5 ms (the word merges 2.9 ms, the plain write-backs ~1 ms for ~46-54 MiB, the prologue uploads ~1-1.5 ms
  for ~23-31 MiB). The frame is latency-bound: ~100 W of the card's 250 W; drawing fewer pixels saves little.
- The game thread ~16.7 ms a frame: the game's own code ~6 ms, the replay ~7 ms (binding work, spread over many small
  functions), the wait for the CPU's late constants (OP_WAIT_EVENT) ~1-3 ms, its own waits ~1 ms.
- Both halves are near 16 ms, so 80-100 fps needs roughly 4-6 ms off each.

What tonight's measurements say about the cost model:
- Shader code size matters more than loads: the pool-table lookup is inlined at every access (hundreds per shader), and
  variants with fewer loads but more branches were slower (three levels 15.9 ms, two levels 14.8 ms, records with the
  delta 19.1 ms). The driver's view without Nsight: `REKARINA_PIPESTATS=1` (registers and machine code size per stage).
- Writing from shaders into host memory over PCIe is expensive (~1.2 of the merges' 2.9 ms); the copy engine moves plain
  copies at ~46 GB/s.
- Host-side debug left on a hot path costs as much as a feature (a getenv per report: 20 % of the replay thread).

## The ways to stop depending on how the Switch runs it

| approach | what it means | assessment |
|---|---|---|
| A. Native NVN driver (in progress, native-driver.md) | keep the game's NVN calls; make their implementation behave like a PC driver: GPU-side waits and results, per-queue execution, bindings as data, no copies back and forth | the main road: every step so far (frames in flight, translator fixes, event waits) is part of it |
| B. Native hot paths | hook the recompiled function that records an expensive subsystem (foliage culling and mesh draws, the clustered lighting, the fog) and record a hand-written Vulkan pass for it | worth it once A's generic costs are gone: it removes translation overhead where the profile says it matters most |
| C. Own shader translator | write the Ampere -> SPIR-V translation for this game's shaders (the reference port wrote its own Maxwell translator) | large; tonight's results show the translator's output is a big lever (patch 0032: -5 ms), so targeted translator work (lookups, HMMA as cooperative matrices, CSE of pool lookups) comes first |
| D. Another language for the game code | rewrite or re-lift the CPU code in another language | no gain: the CPU code is already native and cheap; the cost is in the graphics contract |
| E. DirectX 12 instead of Vulkan | a second backend | no gain in speed or reach: every feature used exists in Vulkan, NVIDIA's Vulkan driver exposes cooperative matrices, mesh shaders, pipeline statistics; DLSS and FSR have Vulkan paths |

What depends on Nintendo, and how each dependency is handled:
- One memory for CPU and GPU -> mirrors with snapshots and merges now; the native-driver end state is GPU-side results
  plus guest pages that move to the GPU when only the GPU uses them (write-backs only for what the CPU reads).
- Six queues with GPU-side events -> one Vulkan queue now; next: queue 3 (compute) on its own queue with timeline
  semaphores for the game's GPU events, once the mirror bookkeeping is per queue.
- The CPU signalling events the GPU waits for -> the replay waits on the game thread now (REKARINA_EVENT_DEFER, which
  records past the wait, still shows another frame's shadows one frame in 300: the replay reads more guest memory as it
  records than uniform blocks and mirrors).
- Ampere machine code with tensor-core ops (the agl upscaler network's HMMA.16816) -> on the tensor cores through
  VK_NV_cooperative_matrix since patch 0033 (the upscaler 0.62 -> 0.38 ms), emulated in shared memory elsewhere.
- Frame pacing: the game's 30/60 Hz assumptions -> `realtime=1` steps the game by the clock down to 30 fps, so an
  unlocked rate does not change game speed.

## Order of work toward 80-100 fps

1. GPU, ours (~4.5 ms; ~40 % of a frame in the lighter scenes): GPU results stay on the GPU. Every merged range on the
   porch has pages the CPU writes (rings of 2.25 MiB buffers the CPU fills and shaders update), so neither plain copies
   nor a better merge kernel remove the cost; the step is a lazy write-back: ranges the CPU does not read stay in the
   mirror, the CPU's writes reach the mirror word by word (vk_memory.c gmem_merge_page has the page shadows for it), a
   guarded page writes back on a CPU access.
2. GPU, game (~11 ms): queue 3 beside queue 6, then CSE of pool lookups in the translator. The two queues are a
   fork-join on the GPU (REKARINA_EVLOG2=<frame>, which logs every event signal and wait with its command position;
   2026-11-08, forest frame): 19 event words 0x13D6C8000..+0x48, values counting up per frame. Queue 6 (graphics, ~22,400
   commands) signals +0x48 at its first command; queue 3 (compute, ~400 commands) waits for it, runs ~80 commands and
   signals +0x00/+0x04/+0x0C, then waits for +0x10..+0x20, which queue 6 signals after its first ~460-910 commands (its
   early passes); queue 6 waits for queue 3's words from command ~7,700 on. So the console runs queue 3's ~3 ms of
   compute beside queue 6's graphics, while the replay runs all of queue 3 before queue 6 and counts those waits as met
   by later signals: queue 3's middle part reads the previous frame's results of queue 6's early passes. Cutting both
   submissions at these points gives the console's order on one Vulkan queue (correctness) and, with queue 3 on a compute
   queue and the words as timeline semaphore values, the overlap (speed).
3. CPU, replay (~7 ms + waits): GPU-side event waits (approach A), bindings as data, a second recording thread.
4. Frame generation (DLSS-G or FSR 3 FG, as in the reference port) once the base frame rate is stable: it needs the
   vendors' runtime DLLs (not on this machine) and motion vectors (vk_dlaa.c already makes them for DLSS).

## The frame in release 1.2.0 (2026-11-18)

The sections above are the picture of 2026-11-08. This one records where the frame stood at release 1.2.0 and what was
changed on the way, with the measurements that decided each step (RTX 5070 12 GB, Ryzen 7 9700X, 2560x1440, base mode:
the scene at its native size, no temporal upscaler, no frame limit). Three places are the standing gate: a town square
walked into (the hardest for the processor), a forest village, the open field.

### The chain

One frame is three links in a row, on three threads: the game's main thread runs its update (~10 ms at the square) and
hands the frame over to the game's presenting thread; that thread submits the frame's command buffers, which is our
replay (`vk_exec.c`, 7-9 ms); the graphics card then finishes the frame (~5 ms after the last submission). Where the
processor is the limit nothing in the frame waits for the card, and the time from the update's start to the picture is
the sum of the three (~24 ms in the village). The pad is read by the game's job threads ~2.4 ms after the update starts.

The frame clock the overlay and `REKARINA_FRAMETIMES` show is the GPU timestamp of the present's blit, taken only
while one of them reads it. The earlier clock (the completion thread's notice) drew frames in pairs and was misleading.
`REKARINA_PACELOG=<file>` writes one row per frame with every stage's time; "update start to shown" and "pad to shown"
are the latency columns.

### What 1.2.0 changed

- **Driver waits in a camera turn.** The game re-places its render targets while the view turns; the frame waited
  inside the driver's object lock for each new view. Views are made ahead and the driver calls take turns.
- **Shadow targets while walking.** While the hero moves, the game re-sizes its square shadow depth targets every
  frame, which moved every later transient target with them. Their storage is taken from a size ladder (128, 192, 256,
  384, 512, 768, ...), so a re-size keeps the storage: images made per frame while walking 14.6 -> 4.0, and walking
  costs 0.27 ms over standing.
- **Texture slot lookups without the texture lock.** The draw path reads a pool slot's resolved texture through a
  sequence word (`vk_texture_by_id_fast`); the slot table grows into a new block and the old one stays for readers
  still in it. Jumps of over 4 ms between shown frames at the square: 12.8 -> 3.5 a second.
- **Command buffers.** The hand-over copies less (`handover`, `latefree`), a called recording is read in place
  (`callfast`), and a small submission no longer queries the write-watch state of the previous big submission's pools
  (`REKARINA_PRESNAP_MIN_KB`, 512): 0.8 ms less of the main thread's time on the game's cycle.
- **Images made ahead** (`REKARINA_TEX_MAKE_AHEAD`, on). The long frames while moving are the game streaming ~40 new
  textures into one frame, each made on the frame's threads under the texture lock. The texture stream's workers now
  make the Vulkan image, its memory and its views after they hashed a job's storage, with no texture lock held; the
  frame's thread takes the record when address and create info match and makes it itself otherwise. Records never
  taken leave through the deferred destruction. Skipped while video memory is short. Walking into the square, frames
  over 25 ms in the first minute: 128 / 54 / 41 / 118 in an off / on / on / off series.
- **The lifted code** (`recomp/`). Indirect calls and tail jumps look their target up at the call site, so each site
  has its own branch history; X9-X15 are not stored before calls; the function-entry hook runs only where the runtime
  needs it; a barrier that orders only loads or only stores is a compiler barrier on x86-64. The kernel signals the
  threads it wakes after it releases its lock, counts the waiters of a guest mutex, and skips its lock for a signal on
  an address nobody waits on. With a retrained execution profile, in one hold at the square against the build before:
  the game's cycle 12.80 -> 11.75 ms, the game threads' running time per cycle -8.4 %, frame rate +4.9 %.
  `tools/pgo.py` records which generated code a profile was trained on; `tools/make_rekarina_release.py` warns when
  the shipped profile names other code.
- **The frame start** (`REKARINA_FRAME_START=jit`, off). Where the card is the limit the main thread finishes its
  update early and waits for the presenting thread; holding its update start back by that wait takes 0.3-2.8 ms off
  the input latency and costs 2-4 fps. It stays an option: a latency-for-rate trade is not a default.

### Tried and not kept

- A resolver thread walking the submitted buffers ahead of the replay (shader, pipeline and descriptor resolution as
  records the replay checks and uses): correct in verify mode, but the replay's own lookups were already ~0.25 ms
  (memos, the parallel draw stretches), and the one large item, the mirrors' sync at a submission's first
  global-memory binding (0.7-1.2 ms), must follow the submit and precede the first dispatch.
- Querying those mirrors' write-watch state in merged runs: 28-45 queries instead of ~560, the same time. The cost
  follows the pages the game wrote (~2,500 a frame), not the number of calls.
- Handing the texture lock over inside a pool update, and the upload worker making textures ahead under the lock:
  the first moved the waits without shortening a frame, the second made the replay wait for the worker.
- Presenting only when the frame's GPU work is done, folding clears, render targets without storage usage: no gain.
- ThinLTO across all generated code: the link takes hours.

### Open

- Frames of 25-50 ms remain where the game streams textures while video memory is short (another program holding
  4-5 GB of a 12 GB card makes it an 8 GB card). With that much really held by a test program on an otherwise quiet
  machine, the square and the field at 1080p showed no more long frames than with nothing held (2026-11-19,
  docs/oot-notes.md): what the reported sessions add to held memory is not found.
- The images made ahead are idle most of the time beside a desktop: the game fills whatever budget it gets, "room is
  short" is then its normal state a minute into a map, and nothing is made ahead while it is; with half or quarter
  textures they are off altogether.
- Four logical processors hold about 60-70 fps at 1080p with a 1 % low of 33-44 (the cores folded onto four slots,
  docs/architecture.md); the game's threads alone need three of this machine's cores.
- One map in, the process's working set is about 16 GB (private memory 22 GB): too much for a PC with 16 GB.
- Block compaction of video memory (`REKARINA_VRAM_COMPACT`, off) loses the device at its first pass; the free space
  inside partly used blocks (0.15-0.5 GB in play, about 2 GB after ten map changes) stays until that is fixed.
- The kernel probe (`--probe-kernel`) still expects three application cores; this title has six, so its virtual-core
  check reports clashes that are not errors.
- Video memory in play, by owner: the driver's own for the prebuilt pipelines ~1.1 GB, mirrors of the game's buffer
  pools 0.7-0.9, the game's block-compressed textures 0.8-1.1, render targets 0.6-0.7, other textures 0.4-0.7.


## Checked pre-bloom reconstruction

On Windows, launch with `REKARINA_RENDERER_MCP=1` for read-only renderer telemetry. The log lists the local
named pipe for that process. Point an MCP client at `python tools/renderer_mcp.py <PID>` over stdio and call
`renderer_snapshot` with no arguments. Its `scene` record is sealed with the existing replay frame and projection
view token; it includes the route, resource serials, extents and HUD-less/coverage eligibility. The other fields
are latest observations and may describe different frames. A null field means nothing has been recorded yet.

With `REKARINA_DLSS_PREBLOOM=1` (the default) on OoT 1.1.1, a successful early DLSS evaluation gives the checked
bloom composite a private output-sized attachment. The checked guest draw and helper calls record
pyramid levels for replay. Replacement extents are shifts of the reconstruction extent; replay
does not infer a level from the guest image size. Untagged draws retain their guest targets.
Replay gives each replacement an output-sized viewport and scissor. The private tone-map target
matches the guest tone-map format: R11G11B10F in Native HDR or RGBA16F in RenoDX HDR. The guest AA
passes still run against the render-size fallback. Replay matches the blend and reconstruction CPU
call sites and checks that reconstruction reads the recorded blend output, whether packed RGB9E5
or RGBA16F. Only a supported, same-format transfer to the full-size HDR target replaces the scaler;
otherwise the guest chain runs. The destination still feeds the HUD and need not match the
reconstruction extent: the transfer fits the reconstructed color to the guest target without
routing it through the render-size AA fallback or losing signed RenoDX color to a packed format.
Frames that do not reach the early evaluation can still use the normal late DLSS path. The equipment
view can skip the scaling dispatch altogether, so it needs the early route. Set `dlss_prebloom = 0`
and restart to compare with the late path.
A pause-menu capture shows the interface sampling cached 1120×1333 crops of a 2560×1440
composite. The active tab was not the equipment screen, so this does not establish equipment
scene ownership. The upstream lighting draw binds a camera block that does not match the pending
projection token. Its D32 attachment is read without a depth write, so the capture does not
identify the geometry writer. A settled PLUS → Q capture shows full-size depth-writing geometry
with a different camera from the later lighting draw. The keyed 854×480 → 2560×1440 route
seen near the same input in a separate replay is not proof that it reconstructed this full-size
view: the two traces must be joined on the actual producer and consumer, not on input timing.
A combined capture shows this full-size route substituting reconstructed color and bloom at tone
mapping, then copying the result back to the guest composite. A final draw samples that composite
directly into a 2560×1440 sRGB target, without an observed gameplay pack/scaler handoff. In the
same replay frame, the render-target serial of that final draw matches the present serial. The
scene initially did not seal as complete because the completion path expected the scaler.
The direct-output path records the guest call sites for the fullscreen tone-map-to-composite
triangle and the composite-to-window encode. These draws replay in guest order and must link
through matching image serials under the same scene key. The scene seals only when that window
image is presented, and does not mark the composited image HUD-less. Shader identity does not
assign either role: the final shader pair is also used by intermediate draws.
`REKARINA_DLAA_PATHLOG` also traces the guest AA prep, indirect weights, blend/pack and reconstruction
by their OoT 1.1.1 CPU return sites. It logs sampled and written image serials and view formats for
native dispatches that actually record, plus the display composite and window serials at final encode.
A DLSS-off RenoDX HDR capture linked the sampled scene color, the AA blend output, and the
reconstruction destination to the final encode. All three color bindings used RGBA16F views; the
stock packed RGB9E5 path is not the format contract for that capture. With the custom RenoDX mapper
and the native scaler, replay confirms the same scene color, edges, weights, blend output and composite
by serial. It preflights the three compute variants on a previously observed matching chain before
converting the prep/blend samples to BT.2020 and the reconstruction stores back to signed BT.709.
The weights pass keeps its math and uses an unsigned sampler for the packed edge mask.
The Native HDR and DLSS shaders are unchanged. After the equipment tab settled, none of those
four AA call sites recorded, although final encode still read the compositor.
The pause scene has a separate ACES pass that adds sampled RGB bloom before uniform exposure and writes
directly to float color without the gameplay HDR expansion. The NVN draw hook tags the shared gameplay and pause
tone-map call site independently of reconstruction. Only tagged draws can request RenoDX scene grading; the SPIR-V
matcher verifies the rewrite, not pass ownership. This leaves menu artwork and gameplay eye adaptation separate.
`REKARINA_HDR_AA_PROBE=1` logs a few numeric compositor/PQ comparisons; `=2` injects a synthetic
BT.2020 green into one test frame before AA. Neither mode writes game images to disk.
At 2560×1440, a separate DLSS-off equipment capture had the same attachment extents as the
hook-tagged reconstructed passes through 40×22. The extent match is not a visual comparison.

`vk_dlaa.c` tracks both routes in the same replay scene: reconstruction phase, checked post-process
images, and the candidate for the HUD-less copy. UI coverage follows the checked target (or a
validated full copy) and retains its submission frame and HUD-less generation until present can
validate them. Reconstruction, UI coverage, and presentation remain separate signals.
