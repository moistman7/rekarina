# Rekarina

Native Vulkan rendering, presentation, upscaling, and HDR pipeline for the static recompilation runtime.

## Language

### Frame Configuration

**Render Snapshot**:
The immutable render settings published after presentation and latched once per guest frame. Also called the frame settings.
_Avoid_: live settings

**Projection Revision**:
A counter on a **Render Snapshot** that advances only when projection-affecting settings change, so temporal upscaling history survives unrelated menu edits.

### Scene Verification & HDR

**Checked Scene**:
A guest scene whose color and depth attachments and draw roles were verified while recording and carried through to presentation.

**Checked Output Draw**:
The game's final output-encoder draw, recognized during replay; it supplies the **Menu Composition** source and leaves an **Output Proof**.

**Output Proof**:
The replay-time record of which encoding a **Checked Output Draw** wrote to an image, invalidated by any later or aliased write to it.

**Guest PQ**:
HDR10 PQ output encoded by the game's own output pass and presented on an HDR10 swapchain without host re-encoding; Native HDR and RenoDX HDR both use it on `OoT`.
_Avoid_: game HDR

**Host HDR**:
The host-side PQ encode over a linear float picture, as `TotK` presents HDR.

**Signed Compositor**:
The signed floating-point game compositor chosen for wide-gamut tone mapping, so out-of-gamut values survive until the game's final BT.2020 PQ encode.

**Menu Composition**:
The host menu and performance overlay blended in linear light into the game's float compositor before the **Checked Output Draw**.
_Avoid_: guest-compositor menu

### Replay & Presentation

**Parallel Replay**:
Chunked multi-threaded command recording on replay workers, for draw stretches that need no strict serial order against host hooks.

**Window Target**:
A presented window texture; under **Host HDR** only stretches drawing into it replay serially.

**Pre-HUD RCAS**:
Contrast-adaptive sharpening of the 3D scene before the HUD is drawn, independent of display scaling.

**Checked Pre-Bloom Reconstruction**:
Temporal upscaling (DLSS, DLAA or FSR) at the checked bloom hook, before the game's bloom, so scenes that bypass the gameplay scaler still reconstruct. Also called checked reconstruction.

**Pre-Bloom DLAA**:
The older opt-in route that resolves native DLAA in place before the bloom, separate from **Checked Pre-Bloom Reconstruction**.

**Stock Scaler**:
The game's own scaling pass, which frame generation can keep while still taking depth, camera and motion from the reconstruction path.
_Avoid_: stock-scaler frame generation

**Swapchain Owner**:
The one presentation backend (ordinary Vulkan, Streamline DLSS-G, or AMD FSR SDK 2) that creates, acquires, presents and destroys the swapchain.

**Present Drain**:
Waiting for every in-flight present of the active **Swapchain Owner** before replacing its swapchain or falling back from it.
