# Building from source

Windows 10 or later, an x86-64 CPU with SSE4.2, and a GPU with Vulkan 1.3 and `VK_EXT_external_memory_host`. The
default build targets x86-64-v3 (AVX2/FMA); a CPU without them builds with `--cpu v2` instead and runs slower (see
"Build"). Allow about 60 GB for the work folder. Extraction wants about 55 GB free while it runs and keeps about 35 GB
afterwards.

Players who only want to play can use the [installer](installer.md) instead: it builds the game on their machine
without this toolchain.

## Toolchain

Versions are the ones the build was verified with; newer patch releases are fine.

| Tool | Version | Install | Used for |
|---|---|---|---|
| MSYS2 UCRT64 | clang/lld 23.1.3 | msys2.org installer to `C:\msys64`, packages below | compiling the generated C and the runtime |
| Visual Studio Build Tools | 2026, C++ workload, Windows SDK 10.0.26100 | winget `Microsoft.VisualStudio.BuildTools` with `--add Microsoft.VisualStudio.Workload.VCTools --includeRecommended` | the linker .NET NativeAOT calls |
| .NET SDK | 10.0 | winget `Microsoft.DotNet.SDK.10`, or for the user alone (no admin rights) Microsoft's `dotnet-install.ps1 -Channel 10.0 -InstallDir $env:LOCALAPPDATA\Microsoft\dotnet` | the shader translator (`totk_xlate.exe`) |
| Vulkan SDK | 1.4.363 | winget `KhronosGroup.VulkanSDK` | validation layers, `vulkaninfo` |
| uv | 0.13 | standalone installer from astral.sh | the two Python environments |
| hactool | 1.4.0 @ `1d64a83` | `.venv\Scripts\python tools\build_hactool.py`, see below | NCA and update extraction |

MSYS2 packages, after `pacman -Syuu` (run it twice; the first pass updates the core and exits):

```sh
pacman -S --needed --noconfirm mingw-w64-ucrt-x86_64-{clang,lld,ninja,gcc,make,pkgconf,sdl3,opus,libvpx,zstd,vulkan-devel,\
vulkan-memory-allocator,vulkan-validation-layers,shaderc,spirv-tools,dav1d} base-devel git unzip
```

`ffmpeg` from MSYS2 is a GPLv3 build and is never linked; the movies go through libvpx.

PATH:

- Add `C:\Program Files (x86)\Microsoft Visual Studio\Installer` to the user PATH: NativeAOT needs `vswhere.exe`, or
  the publish fails at the link step (MSB3073).
- Another `cmake` or `ninja` earlier on PATH (a 32-bit winlibs MinGW, for example) is picked up by accident. The build
  scripts call Ninja from `C:\msys64\ucrt64\bin` themselves; from a shell, put `C:\msys64\ucrt64\bin` first.

### hactool

`tools/build_hactool.py` fetches SciresM/hactool at `1d64a83` into `toolchain\src\hactool`, applies the three patches in
`tools/hactool/` and builds a static `toolchain\hactool\hactool.exe` with MSYS2 UCRT64 gcc (a few seconds):

- `portable.patch`: the parts of MacNativeGames' `build_hactool()` that run on Windows, the `MNG_TITLE_KEYS` variable and
  the two BKTR fixes in `nca.c` for the subsection table padding and the subsection layout of newer cumulative updates;
- `unicode.patch`: a manifest with `<activeCodePage>UTF-8</activeCodePage>`, so paths under a non-ASCII user profile
  reach it intact;
- `mbedtls-asm-clobbers.patch`: the `memory` and `cc` clobbers the bundled mbedtls 2.6.1's x86-64 multiply asm lacks.
  Without them a clang build (the Linux release) fails every RSA signature check and the update merge stops.

hactool reads `-k <prod.keys>`; MSYS2 shells set `HOME`, which hides its default key location, so pass `-k` there.

## Build

From the repository root, in PowerShell. Every step that depends on the game version takes it as `<ver>`: `v143` for
1.4.3, `v121` for 1.2.1 or `v110` for 1.1.0. The tools default to `v121` when `--version` is left out. Saves load on any
of them.

1. Python environments (once). `.venv` runs the lifter and the build; `.venv-py314` runs the extraction scripts, which
   need the stdlib `compression.zstd` of Python 3.14:

   ```powershell
   uv venv --python 3.12 .venv
   uv pip install --python .venv/Scripts/python.exe -r tools/requirements.txt
   uv venv --python 3.14 .venv-py314
   uv pip install --python .venv-py314/Scripts/python.exe -r tools/requirements-py314.txt
   ```

2. Copy `config.example.toml` to `config.local.toml` and set the paths to your base game, the update and your console's
   keys file. `config.local.toml` is ignored by git. The tools read the keys file themselves.

3. Extract and verify. This writes `build\extracted\v0` (1.0.0) and `build\extracted\<ver>` for each update in
   `updates`, identified by its hash; a hash mismatch stops the run. With another `[work] dir` in `config.local.toml`,
   `extracted` goes there instead, and `build_game.py` reads the ExeFS and RomFS from there:

   ```powershell
   .venv-py314\Scripts\python tools\extract.py
   ```

4. Shader translator (no game files; again after a change to its sources, and `tools\make_release.py` stops on an older
   one). This publishes `build\xlate\totk_xlate.exe`:

   ```powershell
   .venv\Scripts\python tools\build_xlate.py
   ```

   It takes the first dotnet with a 10.x SDK from `DOTNET`, `DOTNET_ROOT`, `%LOCALAPPDATA%\Microsoft\dotnet` (the
   per-user install) and PATH. Without the MSVC build tools (no `vswhere.exe`, or vswhere finds no x64 C++ tools) it
   builds a self-contained JIT exe instead of NativeAOT (ReadyToRun, trimmed, one file of about 22 MB): the same name,
   the same `totk_xlate.id`, nothing to install where it runs. `--jit` asks for it with MSVC present. Patches 0017-0027
   add Switch 2 (NVN2, Ampere) shaders ([oot-xlate.md](oot-xlate.md)).

5. Lift the modules to C, write the RomFS overlay (the frame-rate fix), then compile and link
   `build\<ver>\rekarina.exe`:

   ```powershell
   .venv\Scripts\python recomp\gen.py --exefs build\extracted\<ver>\exefs --out generated\<ver> --patches recomp\patches-<ver>.txt
   .venv-py314\Scripts\python tools\make_romfs_patches.py --version <ver>
   .venv\Scripts\python tools\build_game.py --version <ver>
   ```

   On Windows `build_game.py` starts each compile when the memory free at that moment holds its estimate (the
   smaller of the physical memory available and the commit headroom, less 1.5 GiB): Ninja runs a job per thread and
   the compilers it starts wait suspended until they fit, so a program opened or closed mid-compile lowers or raises
   the number running, and one always runs. Elsewhere, or with `REKARINA_BUILD_MEM_GB=<n>`, the jobs share a budget
   fixed at the start: the memory available then less 3 GiB (`--jobs N` sets the job count). The chunks that are one
   giant function need up to about 3 GiB each at -O2 with a profile and run in a pool of their own, largest first;
   every compile runs below normal priority. The first line of its output says the plan, the gate's lines what runs.

   `--cpu v2` targets x86-64-v2 (SSE4.2, no AVX2/FMA) for older CPUs; the default `v3` needs AVX2. The v2 build links
   `runtime/fma.c`, a correctly rounded software fma the FP lowering needs, and runs slower. Inline assembly,
   target attributes and intrinsics are not bound by `-march`, so a v2 build runs `tools/check_isa.py` over the runtime
   objects before it links: any AVX, BMI or other instruction above x86-64-v2 outside a function listed there as
   dispatched by a runtime CPU check stops the build.

   `gen.py` sizes its emit pool from the CPU count and the free memory; `--jobs N` sets it, and so does the environment variable `REKARINA_LIFT_JOBS=N`, which the installer's lift step honours too (it has no `--jobs`). When a worker runs out of memory, the pool restarts with half the workers (`emit: out of memory with N workers, retrying with M`), down to one; only an out-of-memory error with one worker fails the step.

   Options that change the shape of the emitted C (each changes most functions, so a profile trained before it no
   longer fits them: retrain, "Profile-guided build"). By default X9-X15 are not stored to the `Cpu` before a call
   (AAPCS64 makes them neither arguments nor preserved; a direct call of a function that may read one before writing
   it, `Emitter.reads_temps`, still stores them; `--temps-saved` stores them before every call), and only
   `runtime/resolution.c`'s SET_RENDER_SIZE calls the runtime's function-entry hook (`cpu.h` `FN_ENTER`; `--fn-hooks
   all` for `runtime/prof_win.c`'s function watch). Measured and left off: `--kept-locals`, `--reg-locals`,
   `--inline-leaves` (more register locals or inlined leaf copies: more host instructions in the hottest functions
   when compiled). `tests\diff\func.py` takes the same options and the same defaults.

   Lifting takes about 15 s, a clean compile about 200 s on 24 threads; later builds recompile only what changed.
   A compiler cache in front of clang shortens it further: with `SCCACHE` set to an sccache executable, `build_game.py`
   writes it into the ninja file ahead of `cc`, and every lift rewrites the generated chunks with the same contents, so
   a second build of the same version hits all of their objects:

   ```powershell
   $env:SCCACHE = "path\to\sccache.exe"
   .venv\Scripts\python tools\build_game.py --version <ver>
   ```
   `--patches` builds in the frame-rate, resolution and other patch sets, switchable at run time. The link copies
   `SDL3.dll`, `libiconv-2.dll` and `libopus-0.dll` next to the exe.
   For Xbox button prompts, run `make_romfs_patches.py` twice more, with `--prompts xbox` and `--prompts xbox-label`:
   they write `build\<ver>\romfs-patches-xbox` and `romfs-patches-xbox-label` ([controls.md](controls.md), "Xbox
   button prompts"), which `build_game.py` copies next to `rekarina.exe` like `romfs-patches`. `--prompts pc` writes
   `romfs-patches-pc`, the keyboard and mouse prompts ([controls.md](controls.md), "PC button prompts").

6. Prepare the shaders (once per build folder and translator). It needs `refs\MacNativeGames` (`tools\fetch_refs.sh`)
   for the shared shader list:

   ```powershell
   .venv-py314\Scripts\python tools\prepare_shaders.py build\<ver> --version <ver>
   ```

   Without this step the first launch compiles shaders and pipelines during play and stutters until the caches fill.
   The AMD driver keeps its pipeline cache per exe path and holds a limited amount: preparing another build folder can
   evict this one's. `rekarina.exe --warm-shaders` rebuilds it from the shader cache.
   When the driver's shader compiler crashes on a variant (NVIDIA 615.78.08 on an RTX 5080 does, issue #30),
   `prepare_shaders.py` runs the driver step again with `--warm-shaders`, up to three times, without the variants the
   driver was building at the crash. `shader-prepare-skip.txt` next to the exe lists them with the GPU and driver
   version; they build at their first draw instead, another driver version builds them again, and so does deleting the
   file.

7. Run. The game keeps running when its window loses the focus (`assume_focus = 1` in `rekarina.ini`, the default; the
   Game tab switches it). With 0 it pauses, as it does on the Switch when Home is pressed.

   ```powershell
   build\<ver>\rekarina.exe
   ```

Settings live in `rekarina.ini` next to `rekarina.exe` ([rekarina.ini.example](../rekarina.ini.example) lists them); an OoT
build reads `rekarina.ini` and writes `logs\rekarina.log`, the names its package has (`runtime/host.h`
`RT_FILE_STEM`), and renames a `rekarina.ini` it finds beside it. Saves go to
`build\profile` for `v121` and `build\profile-<ver>` for other versions (`build_game.py --profile` changes it).

### Portable folder

`.venv\Scripts\python tools\package_build.py --version <ver>` writes `build\<ver>-portable`, which runs on another of
your machines without MSYS2, Python or .NET. `romfs.bin` stays out unless you pass `--with-romfs`; otherwise put it in
`romfs\` next to the exe or point `REKARINA_ROMFS` at its folder. If the image cannot be opened, the game reports the
path in its log and exits before starting graphics. Run `rekarina.exe --prepare-shaders` once in the copied folder.
The folder holds code and data derived from your game: for your own machines only.

### Profile-guided build

`build_game.py --pgo <file>.profdata` compiles the generated code at -O2 with an execution profile, and the runtime
with it too: hot paths laid out straight, hot calls inlined, code that never ran compiled for size. On 1.4.3 the fight
bench ran 12-15% faster per frame and a fight outside the training about 6%. The installer builds with
`data/pgo-<ver>.profdata` of the release when the update has one, else at -O1.

A profile only fits the generated C it was trained on: functions are matched by name and control-flow hash, so a
function whose C changes loses its counts without a warning. Retrain after a lifter or generator change, from a clean
`recomp/`. The instrumented build needs llvm-mingw, as MSYS2's clang has no profile runtime:

```powershell
$llvm = "toolchain\llvm-mingw-<date>\llvm-mingw-<date>-ucrt-x86_64"
.venv\Scripts\python tools\build_game.py --version v143 --out build\v143-pgogen --toolchain $llvm --pgo gen
```

Play `build\v143-pgogen\rekarina.exe` through varied scenes (fights, towns, riding, building, the Depths, menus) for 20-30
minutes; it is slower than a normal build. It writes `build\v143-pgogen\pgo\*.profraw` every minute and at exit.
Scripted runs from saves (`tools/run_game.py`) add to the same file. Then:

```powershell
.venv\Scripts\python tools\pgo.py merge build\v143-pgogen --version v143
```

writes `build\pgo\pgo-v143.profdata` and its record (`.json`: the `recomp/` tree it fits). `make_release.py` ships every
profile of `build\pgo` in `data\` and warns when a record names another `recomp/` tree than the one released.

## Linux (in progress)

The runtime builds and runs on x86-64 Linux with a window, input and menu on SDL3 (`*_sdl.c`) in place of the Win32
files (`*_win.c`); `runtime/platform/linux.c` implements `platform.h`. Status: the selftest and the probes pass; 1.2.1
plays at 60 fps on RADV (RX 7800 XT) and NVIDIA (RTX 3060); 1.4.3 builds, probes and plays on RADV (RX 6600, and the
Vega 7 of a Ryzen 5 4600G at about 34 fps). Long play sessions and the Steam Deck itself are not verified yet.

Requirements:

- Linux 6.7 or later: the write watch uses userfaultfd write protection with `PAGEMAP_SCAN`. SteamOS 3.8 qualifies.
- A Vulkan 1.3 driver with `VK_EXT_external_memory_host` (RADV, NVIDIA).
- Packages on Arch:

  ```sh
  pacman -S --needed clang lld ninja make sdl3 opus libvpx zstd vulkan-headers vulkan-icd-loader \
      vulkan-validation-layers shaderc spirv-tools dotnet-sdk uv git
  ```

  Vulkan Memory Allocator is a single header (AUR `vulkan-memory-allocator`). Without it, `build_game.py` fetches the
  version the Windows release pins (3.4.0) into `toolchain/vma/` on first use and checks its hash.

The steps are the Windows ones with POSIX paths: `.venv/bin/python` in place of `.venv\Scripts\python`, and `totk` in
place of `rekarina.exe`. `build_game.py` takes clang, lld and Ninja from PATH (or the ones in `<dir>/bin` with
`--toolchain <dir>`) and links against the system libraries, with `libstdc++` static and an `$ORIGIN` rpath;
`build_hactool.py` builds `hactool` with make. `tools/prepare_shaders.py` reads the shader list from
`refs/MacNativeGames` at `origin/HEAD`, which must be the pinned commit with its history (the list format changed later).

The probes open a window, so from an SSH session they need the desktop session's display, or they stop at `gpu 0
skipped: no graphics+compute+present queue family`: on Wayland `XDG_RUNTIME_DIR=/run/user/$(id -u)
WAYLAND_DISPLAY=wayland-1 SDL_VIDEODRIVER=wayland` (the socket name is in `$XDG_RUNTIME_DIR`), on X11 `DISPLAY=:0`.

Memory: on Windows the GPU imports guest memory and `GetWriteWatch` tracks the CPU's writes to it. On Linux,
re-protecting imported memory invalidates the driver's userptr buffer on every submit (about 1.3 ms per 16 MiB on
amdgpu), and importing pins the pages, which drops their write protection. Each CPU-visible pool therefore gets a
host-visible copy (shadow storage, `runtime/vk_memory.c`): the pages the CPU wrote go in before each submit, and the GPU's
writes come back at its fence. In play that costs about 0.2 ms a frame and doubles the memory of those pools (about
1.1 GB). `REKARINA_SHADOW=1` turns it on under Windows too. The Steam Deck's RADV has `VK_EXT_external_memory_host` as
well: the shadow copies are chosen for the measured cost above, not for a missing extension.

The Deck's GPU (Van Gogh) has no native ASTC, so its textures go through the compute decoder (`runtime/vk_astc.c`,
`REKARINA_ASTC`) as on desktop AMD GPUs without it; frame times measured there include it.

### OoT on Linux and under Proton: where it stands (2026-11-13, Linux build 2026-11-15)

2026-11-15, in WSL2 Ubuntu 24.04 (its only Vulkan device is llvmpipe): `tools/make_release_linux.py --oot` and
`tools/make_rekarina_release.py` make the setup folder and `Rekarina-x86_64.AppImage` (213 MiB); the launcher's
`--selftest` draws all its screens from the AppImage; `python -m installer.oot` builds the game from a dump through every
step (lift, compile and link against the sniper sysroot, overlays, 55,711 shader variants prepared, package) and
`Rekarina --warm-shaders` exits 0. The game has not been played on Linux: WSL has no GPU driver of its own. The system
libraries the SDL3 beside the game needs are in docs/installer.md. Before that build nothing of OoT had run on Linux,
native or under Proton. What is known from the code and from runs on Windows that use the same paths:

- The Linux build and Wine both use shadow storage for CPU-visible pools (`vk_memory.c`, `REKARINA_SHADOW`; under Wine it is
  chosen by `plat_is_wine`). `REKARINA_SHADOW=1` on Windows runs that path: it is the check to make before a Linux test, and
  the first thing to run when a picture differs there.
- Early results (`REKARINA_RESULTS_FENCE`) are taken only with mirrors, so they are off in shadow mode: expect the frame
  rate of a build without them (about a sixth lower where the GPU decides the frame).
- The draw-stretch threads (`replay_threads`), the uploads ahead of first use (`REKARINA_TEX_PREP=1`, off by default), the images made ahead
  and the texture cache use only `platform.h` (threads, events, files): `tools/check_platform.py` passes for them.
- `tools/check_platform.py` passes. Four files that load vendor libraries keep Win32 calls inside `#ifdef _WIN32` and
  are on its allowlist with the reason: `vk_dlaa.c` (DLSS; it has a Linux branch that loads `libnvidia-ngx.so.1`),
  `vk_dlssg.c` (DLSS Frame Generation), `vk_ffx.cpp` and `vk_fg.cpp` (FSR 4.1 and 3.1, and FSR frame generation). On another
  host those three report the feature missing. Under Proton they are Windows code and need `nvngx` from the Proton
  prefix; the base mode needs none of them.
- Mesh shaders are required (`VK_EXT_mesh_shader`): RADV has them from RDNA2 on, which includes the Steam Deck; the
  integrated Radeon of the Windows test machine still draws no mesh foliage (docs/oot-notes.md), so foliage on any AMD
  card is the first thing to look at.
- `VK_EXT_conditional_rendering` carries the game's occlusion conditions; without it they are decided on the CPU, which
  has shown white frames (`REKARINA_GPU_COND=0`). RADV and NVIDIA have it.
- The video memory counter the texture quality rule reads comes from PDH on Windows (`platform/win32.c`); under Wine and
  on Linux the Vulkan memory budget is all there is, so `tex_quality` may have to be set by hand on small cards.

`REKARINA_LINUX_GUEST_MEMORY=driver` runs the driver-memory way instead (docs/oot-notes.md, "Guest memory from the GPU driver on Linux"); `tests/mem_backed_check.c` checks its platform part without game files or a GPU (the build line is in its header).

Order for a first Linux or Proton session: the selftest and probes, `--warm-shaders`, the forest standing still with
`REKARINA_FLIGHT=1`, then the Market at a run with a frame-by-frame clip (docs/recomp-playbook.md, section 8).

### OoT on a Linux desktop: a first run on a real GPU (2026-11-20)

Nobara 44 (Fedora 44), kernel 7.2, Ryzen 5 4500, GeForce RTX 4060 Ti 8 GiB, NVIDIA 595.99.02. Built with the system
clang against the system libraries, as `tools/build_linux_local.sh` does it (extract, lift, compile, translator, prompts,
shaders, package; Fedora packages in the script's header; `libstdc++-static` for the release link line). Results:

- `Rekarina --memory-test`: placed maps, an exported dma-buf and a host-pointer import all work, a protection change
  costs the next submission nothing for the first two, and userfaultfd write protection does not apply to either
  (`Invalid argument`), so driver memory needs the platform's dirty bits (`plat_mem_back`).
- The game starts, the title scene draws at 65 fps (3,393 frames in 52 s on shadow storage, 3,462 on driver memory, both with
  a 1 % low of 58 fps). With `REKARINA_LINUX_GUEST_MEMORY=driver` 2,130 pools were backed and none refused. The first start builds
  the driver's pipeline cache (minutes, mesh pipelines about 40 s each); the second start's entry step took seconds.
- `--selftest` passes (0 failures) since the dispatch check skips functions a host job replaces and the physical memory
  check takes the pool size from GetInfo. Probes that fail on this OoT build the same way with and without the
  driver-memory change: `--probe-loader`, `-ipc`, `-services` (written for the other game), `-nvn` (6 of 1,038 checks, among
  them the stub `nvnTextureGetNvRmSurfaceNVX`), `-vkmem` (one swizzle case) , `-shader`, `-vkexec`. `--probe-kernel`'s
  virtual-cores check depends on the machine's load and failed once in seven runs. `tests/excl_mt.py` fails its ABA case on
  any host: the exclusive monitor in `runtime/cpu.h` compares values.
- Not measured: a save, a map change, the Market, a long session, any frame-rate figure beside the title scene. The
  benchmark of `docs/performance.md` needs a save and an input script.

CI (`.github/workflows/linux.yml`) runs the content guard, the platform audit, `tests/mem_backed_check.c` and the
installer's unit tests in a Fedora container. `.github/workflows/release-linux.yml` (manual) builds the setup folder, the
AppImage and, with `tools/pack_linux_release.sh`, `Rekarina-Setup-<version>-linux-x86_64.tar.gz` and `SHA256SUMS`.

## Checks

Before a commit:

- `python tools/check_content.py --staged`: nothing game-derived is staged. CI runs the same check;
  `git config core.hooksPath .githooks` runs it before each local commit.
- `build\<ver>\rekarina.exe --selftest`: layout, dispatch table and host-function registry, without game files.

For NVIDIA Vulkan shutdown changes on Windows, run `--probe-vkdevice` and `--probe-color` on an RTX GPU with
Streamline DLSS, then repeat from an isolated folder without the Streamline bridge to exercise native NGX.
Confirm the intended backend initialized in the log, both probes passed, and no fatal error was reported during exit;
an exit status of zero alone does not catch a crash during DLL detach.

A change to the button prompts (`tools/make_romfs_patches.py --prompts`, `tools/button_icons.py`) passes
`.venv-py314\Scripts\python tests\romfs_prompts_test.py`, which runs the tool on a RomFS it makes itself (no game
files); a change to the settings passes `rekarina.exe --probe-cfg`.

The lifter (`recomp/`) and the CPU headers (`runtime/cpu.h`, `runtime/simd.h`) are checked against Unicorn. Every lifter
change passes these, all run with `.venv`:

| Test | Checks | Game files |
|---|---|---|
| `tests\diff\lift.py` | instructions sampled from the extracted modules; `--coverage` lists every word the lifter rejects or traps | yes |
| `tests\diff\simd.py` | the whole FP/SIMD encoding space | no |
| `tests\diff\atomic.py` | exclusive load/store sequences and ARMv8.1 atomics | no |
| `tests\diff\func.py` | whole functions as the generator emits them, loops, jump tables and calls included (calls go to a stub that checks the registers at the call) | yes |
| `tests\flags_check.py` | the function-local flag helpers of `runtime\cpu.h` against its packed ones | no |
| `tests\excl_mt.py` | the exclusive monitor helpers under contention and ABA, on several threads | no |
| `tests\jumptables_check.py` | `recomp/jumptables.py` against a Unicorn oracle; must report 0 disagreements, on 1.4.3 too | yes |
| `tests\build_content_test.py` | `tools/build_game.py`'s content stamps: an output whose source, header, flags or compiler changed is made again whatever the file times say; a build folder without the record makes its runtime objects again | no |
| `tests\build_jobs_test.py` | `tools/build_game.py`'s compile job plan: the budget from the available memory, the giant chunks' estimates, and plans that stay within the budget for 8, 16 and 32 GB with 4, 6 and 12 threads; the memory gate's choice, and on Windows the gate holding real processes (one at a time with memory for one, all together with plenty) | no |

A `--cpu v2` build passes the same differential tests with `--cflags=-march=x86-64-v2`; they compile `runtime/fma.c`
too, so they check the software fma the v2 build links rather than the C runtime's.

Runtime probes, each a mode of `rekarina.exe` (the list is at the top of `runtime/main.c`): `--probe-kernel`,
`--probe-vkexec`, `--probe-nvn-tex` and `--probe-shader` cover most runtime changes. `--probe-shader` and the draw
part of `--probe-vkexec` translate shaders of the census below: they look for `build/<ver>/shaders/jobs.json` of the
exe's own version first, then of the other one.

Shaders: `.venv-py314\Scripts\python tools\shader_census.py --version <ver> --translate` extracts every Maxwell shader
to `build\<ver>\shaders\` and runs it through the translator and `spirv-val`; failure groups go to `failures.txt`.
`tools\xlate_smoke.py` checks the translator alone, without game files.

## Reverse engineering (optional)

- Ghidra 12.1.2 with Adubbz's Ghidra-Switch-Loader in `toolchain\ghidra\` (JDK 21) opens the game modules (NSO).
- `tools\fetch_refs.sh` clones the upstream projects this port builds on into `refs\`, read-only.
