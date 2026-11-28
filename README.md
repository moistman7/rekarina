<p align="center">
  <img src="docs/media/logo-dark.png" alt="Rekarina" width="560">
</p>

<p align="center">
  <b>A native PC port</b> made by static recompilation, not emulation.<br>
  Built on your own PC from your own copy of the game.<br>
  <i>An actively maintained Rekarina fork focused on NVIDIA DLSS, DLAA, and DLSS Frame Generation through Streamline,
  RenoDX tone mapping with HDR10 output, and ongoing performance fixes and tuning.</i>
</p>

<p align="center">
  <a href="https://github.com/arnaudcouturier/rekarina/releases"><img src="https://img.shields.io/github/v/release/arnaudcouturier/rekarina?label=release&color=d9951f" alt="Latest release"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0-6f5a35" alt="GPL-3.0 license"></a>
  <img src="https://img.shields.io/badge/platform-Windows%2010%20%2F%2011%20x64-6f5a35" alt="Windows 10 / 11 x64">
</p>

<p align="center">
  <a href="https://github.com/arnaudcouturier/rekarina/releases">Releases</a> ·
  <a href="https://github.com/arnaudcouturier/rekarina/issues">Support</a> ·
  <a href="#install">Install</a> ·
  <a href="#performance">Performance</a> ·
  <a href="#features">Features</a> ·
  <a href="#updates">Updates</a> ·
  <a href="#known-issues">Known issues</a> ·
  <a href="#credits">Credits</a>
</p>

> [!IMPORTANT]
> **No game files are included, and none are downloaded.** You need your own legally obtained copy of the
> game, dumped from your own console as a `.nup` (base game 1.0.0 with update 1.1.1). The launcher
> builds the game from your files on your machine, so the result contains the game. It is for your own use: do not
> share it. Do not ask for game files here; issues that link or attach them are closed.

## What it is

Rekarina translates the game's ARM64 code to C ahead of time and compiles it into an ordinary x86-64 executable. The
console's graphics calls go to a Vulkan renderer written for this game, its shaders are translated once at install,
and the system services the game expects (files, audio, input, video decoding) are implemented natively. There is no
CPU emulator and no JIT at run time, which is where the frame rate comes from.

## Performance

The chart below records the upstream Rekarina 1.0.0 build. The fork's performance and image quality changes
are described in its release notes as they ship.

Measured in the game itself, on one machine, with the tools in this repository (`tools/bench_oot.py`); every number is
the mean of a one-minute run, with the 1 % low beside it. "Native" is the sharp base picture at the monitor's
resolution without any upscaler; "DLSS Quality + frame generation" is DLSS Quality with DLSS Frame Generation.

<p align="center">
  <img src="docs/media/perf-locations.svg" alt="Frame rate by location: native, and with DLSS and frame generation" width="100%">
</p>

The machine: NVIDIA GeForce RTX 5070 (12 GB, driver 610.88), AMD Ryzen 7 9700X 8-Core Processor, 62 GB of memory, Microsoft Windows 11 Pro; Rekarina 1.0.0, 2026-11-12.

<details>
<summary>How the numbers are taken</summary>

- Each location is reached by an input script: a save is loaded (the Market and the field by a map change from
  it), and the camera then turns in place, so the whole scene passes through view. The first minute holds the loading
  and is thrown away; the second is the result. The frame limit is off (240 fps is the game's own ceiling).
- Frame times come from the game's own log (`frametime:` lines): the mean rate and the rate of the slowest 1 % of
  frames. Runs that overlapped another program using the GPU are thrown away.
- With frame generation the chart shows frames **on screen**; the frames the game renders are half of that and are
  listed in [docs/performance.md](docs/performance.md) with every run's raw numbers.

</details>

Numbers from another machine are welcome: [docs/performance.md](docs/performance.md) says how to produce them.

## Features

- **NVIDIA DLSS, DLAA, and DLSS Frame Generation** through Streamline. Projection jitter and tracked motion
  vectors give DLSS subpixel samples and motion for moving objects, including skinned characters.
- **RenoDX HDR** tone maps the scene before presentation and outputs HDR10. With HDR enabled on the display,
  open Graphics from the F1 menu for live paper white, peak brightness, and bloom controls.
  PsychoV35 fits to BT.2020 and offers optional eye adaptation. Bloom works with both mappers; local exposure is Vanilla-only.
- **Performance fixes and tuning** focus on the Vulkan renderer and frame pacing.
- **Native code, not emulation**: 100+ fps where the console runs at 60, a frame limit up to 240 or the monitor's rate
- **Smooth above 60 fps**: the game's own clock follows real time, and the places where it counted frames are
  corrected (water ripples, the pick-up counter, physics steps on moving platforms, movie cutscenes)
- **A texture cache** (`tex_cache` in `rekarina.ini`, on by default): textures are converted for your GPU once and kept
  beside the game (up to 14 GB), so ground you have crossed before streams in without hitches
- **Voice acting, music and movies** as on the console: Opus voice lines in nine languages, AV1 cutscenes
- **Sharp native picture** at any 16:9 resolution, plus 21:9 and 32:9 with the camera and the interface in that shape
- **AMD FSR 4.1** on the Radeon cards that run it and **FSR 3.1** on supported GPUs. The
  launcher fetches each vendor's library from its official repository for the GPU it finds; nothing of theirs is
  bundled here
- **Settings that apply while you play**: the in-game menu (Esc or F1) is drawn in the game's own menu style, with a
  live picture beside the rows, so a change to the render scale, sharpening, colour or frame limit shows at once.
  Number settings are sliders you drag with the mouse or step with the keys and the pad, and pictures explain the
  rest: the picture chain from the 3D scene to the window, your controller live, your keyboard with every bound key,
  frame times with the CPU and GPU time per frame, and the graphics card, driver and device profile the game found
- **Presets by hardware**: the first settings are chosen from your GPU and its video memory, and a texture quality
  setting keeps 6 and 8 GB cards out of trouble. Every card starts with the game's sharp base picture; DLSS, FSR 3.1
  and frame generation are switched on in the game's menu. An update keeps the settings you have
- **Older cards and built-in graphics**: **AMD FSR 1** scales a smaller 3D scene up at the same small cost on any GPU
  (the first choice on built-in graphics and 4 GB cards), and the trees, bushes and grass are drawn on GPUs without
  mesh shaders
- **Your save comes along**: the launcher's Saves page brings a save in from another port of the game or from an
  emulator (Ryujinx and its forks, the yuzu family), or from a folder or a .zip; the save it replaces is kept
- **Controllers and the mouse**: Xbox and PlayStation pads, the console's own controllers with HD rumble and gyro,
  and a raw mouse camera that turns exactly as far as the mouse moves, with its own speed for each axis. With several
  pads connected the one you press is the one read; a pad that is found but does nothing can be read the way Windows
  does (Compatibility driver); the pointer hides when it rests; keyboard and mouse can be switched off while a pad is
  connected
- **Button prompts** for Xbox, PlayStation and Switch pads, a set drawn for this port, and the keyboard and mouse:
  they follow the device you use, change in the running game, and the keyboard prompts follow the keys you bind
- **Skip what you have seen**: every cutscene can be skipped, the three the console keeps unskippable included, and
  an option runs the dialogue on without the pauses between phrases
- **Launcher with updates**: it builds the game, shows each step with the time it takes, and installs new releases;
  the game itself tells you when one is out
- **Menu and launcher in every language the game has**: English, Russian, Japanese, German, French, Spanish, Italian,
  Dutch, Portuguese, Korean, Simplified and Traditional Chinese
- **A problem report in one press**: the menu (Advanced) or the launcher writes one zip with the logs, the settings
  and the computer's details (no saves, your user name taken out of every path) to attach to a report; nothing is
  sent anywhere. After a crash the game says so and points to it.
- Mods with a manager (the launcher's Mods page and the game's Mods tab), a performance
  overlay (F10), a screenshot key (F12, PNG files in `screenshots`), mute in the background, Discord Rich Presence
  (off by default)

## Requirements

| | Minimum | Recommended |
|---|---|---|
| System | Windows 10 or 11, 64-bit | Windows 11 |
| GPU | Vulkan 1.3, 6 GB | NVIDIA RTX with 8 GB or more |
| CPU | 6 cores with AVX2 | 8 cores |
| Memory | 16 GB | 32 GB |
| Disk | 30 GB free on an SSD (the game, its shader caches and the build) | |

Memory, measured in two tours of eight places each: the game holds about 16-17 GB of the computer's memory in play
and commits 24-27 GB. With 16 GB it runs, but pages to the disk and stutters, and Windows needs a page file large
enough for the rest; 32 GB leaves room for it (not measured on a 16 GB or a 32 GB PC: the test PC has 62 GB).

NVIDIA RTX cards are what the port is developed and tested on. AMD and Intel GPUs start and run, with the problems
listed under [Known issues](#known-issues).

## Install

1. Download **`Rekarina.exe`** from [this fork's GitHub Releases](https://github.com/arnaudcouturier/rekarina/releases)
   and start it. It is the whole launcher in one
   file: at its first start it unpacks its tools (about a minute, once) and opens.
2. Point it at the folder with your `.nup` dump and choose where the game goes.
3. Press *Start*. The launcher checks the dump, recompiles the game's code, builds the executable, fetches the
   graphics components for your GPU and prepares the shaders. The first build takes about an hour, most of it the
   shaders; each step shows how long it has left. Its last step also builds every shader and pipeline the game is
   known to use (about eight minutes), so play does not stop to build them; later starts take a few seconds. The
   setup also prepares the textures for your graphics card (about 12 GB on the disk and about two minutes on a fast
   card; the launcher's *Prepare textures* row can skip it), so the first visit of a place does not stutter while its
   textures are converted: on an RTX 5070 the village's first visit ran at 96 fps with them prepared and at 53
   without.
   After a graphics driver update the game does this once at its next start and shows its progress (a few
   programs that are not in the game's archives are still built the first time they appear, once per install).
4. Press *Play*.

<p align="center">
  <img src="docs/media/launcher-home.jpg" alt="The launcher" width="49%">
  <img src="docs/media/launcher-install.jpg" alt="The launcher building the game" width="49%">
</p>

If you have upstream Rekarina installed, choose a separate folder for this launcher and its game. Use the launcher's
**Saves** page to import your save. The fork's Windows shortcuts are labelled **Rekarina (fork)** in Start and
**Rekarina Launcher (fork)** on the desktop.

Saves and settings live in the game folder and survive updates and rebuilds. Settings are in the in-game menu; every
key is also in `rekarina.ini` beside the game ([the key list](rekarina.ini.example) explains each). Controls:
[docs/controls.md](docs/controls.md).

## Updates

The launcher and the game ask this fork's [GitHub Releases](https://github.com/arnaudcouturier/rekarina/releases)
(`arnaudcouturier/rekarina`) for the
newest release when they start (one request that carries the port's version and nothing else; *Menu > Check for updates*
turns it off in-game). When a release is out, the game shows a notice and the launcher offers *Update*: it downloads the
new setup, checks its SHA-256 checksum, rebuilds only what changed and re-prepares the shaders, so the first minutes
after an update do not stutter.

## Known issues

- **AMD GPUs** have been run on one integrated Radeon only, where the whole picture is drawn, the short grass
  included. On AMD's Windows driver the draws of that grass go without the game's GPU render conditions (inside one
  the driver stopped the GPU), so the grass the game would cull is drawn anyway: a little more GPU work. A discrete
  Radeon, a ROG Ally and a Steam Deck have not been run: reports with a log and a screenshot are very welcome.
  [docs/oot-notes.md](docs/oot-notes.md) ("AMD") has the details.
- Intel GPUs have never been run.
- On GPUs without mesh shaders the trees, bushes and grass go through a compute stand-in, checked on an NVIDIA card
  with its mesh shaders turned off, not yet on such a GPU.
- A processor without AVX2 gets a slower build of the game, chosen by the launcher; that path has not been run on such
  a machine.
- HDR is off by default and requires HDR10 support from the display and operating system.
- In the first dungeon's entrance room, in the base picture without DLSS, a light shaft can flicker.
- With another program holding most of the video memory (a video editor, a second game) the frame rate collapses; the
  game says so at the top of the screen.
- The first seconds after a loading screen run slower while the new place's textures arrive.
- A rare GPU crash (device lost): the cause is not found. If it happens, the game stops with an error, and a crash
  dump file is now left beside the log for the problem report.
- On a PC with 16 GB of memory the game does not read the prepared textures ahead at its start (they would push the
  game's own memory out), so a place's first visit after a restart of the PC reads them from the disk while it plays.
- A shader combination or a grass and foliage pipeline that none of our runs met is built the first time it appears:
  a few milliseconds for most, up to seconds for a foliage pipeline, whose patch is not drawn meanwhile. Our runs
  went through 67 of the game's 79 maps, under a minute each from one start position; boss fights, cutscenes and the
  rooms deep inside dungeons only partly.
- Long sessions (many map changes in one run) were measured on a 12 GB card only; cards with 8 GB have not been.
- The numbers above are with the camera turning in place. Running through the field or the Market gives 73 to 80
  fps on the same machine without the texture cache, with short hitches as new ground streams in; the texture cache
  (`tex_cache = 1`, the default) brings that to about 84 fps and removes most of the hitches.
- Releases are Windows only for now: there is no Linux download. A native Linux build is in development; it has
  not run on a Linux PC with a graphics card yet. The Steam Deck and the Windows handhelds (ROG Ally, Legion Go,
  MSI Claw) are recognised and start with their own settings; [docs/tester-guide.md](docs/tester-guide.md) has a
  checklist for them.

Found something else? Make a problem report (Esc, Advanced, Make a problem report; or the launcher's Problem report
button) and attach the zip to a [fork issue](https://github.com/arnaudcouturier/rekarina/issues/new/choose).

## Building from source

[docs/building.md](docs/building.md) lists the toolchain and the steps, and [docs/installer.md](docs/installer.md) the
launcher's own flow. Everything a build writes stays under `build\` and `generated\`, which git ignores; run
`python tools/check_content.py` before a commit. Pull requests are welcome: [CONTRIBUTING.md](CONTRIBUTING.md).

## Credits

This fork is based on [upstream Rekarina](https://git.rekarina.dev/rekarina/rekarina). ShortFuse's [NX2navi](https://github.com/clshortfuse/NX2navi) work
informed the Streamline integration, and [RenoDX](https://github.com/clshortfuse/renodx) supplies the HDR algorithms.

Rekarina stands on **[the recompiler](https://github.com/valentderah/totk-pc-recomp)** by Valiantsin Dzerakh: its
lifter, runtime, shader translator patches and Vulkan renderer are the foundation of this port. Thank you.

It also builds on the projects below; what was taken from each, with its license, is in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

- [LAS Recomp](https://github.com/peiikai/LASRecomp), [MacNativeGames](https://github.com/ly05010419/MacNativeGames),
  [SavageWindRecomp](https://github.com/Revivedx/SavageWindRecomp),
  [d2r-mobile](https://github.com/jhaizhou-ops/d2r-mobile) and [nxnative](https://github.com/seedds/nxnative)
- [Ryujinx](https://github.com/Ryujinx/Ryujinx), [deko3d](https://github.com/devkitPro/deko3d),
  [nup](https://github.com/a55uka/nup), [RenoDX](https://github.com/clshortfuse/renodx)
- [SDL 3](https://libsdl.org), [Dear ImGui](https://github.com/ocornut/imgui), [dav1d](https://code.videolan.org/videolan/dav1d),
  [Opus](https://opus-codec.org), [Zstandard](https://github.com/facebook/zstd),
  [Vulkan Memory Allocator](https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator)
- NVIDIA DLSS and Streamline, AMD FidelityFX: downloaded from their publishers at install, under their own licenses
- [Switchbrew](https://switchbrew.org) and the Switch homebrew community

## License

Rekarina is free software under the [GNU General Public License, version 3](LICENSE): you may use, change and share
it, and what you share stays under the same license, with its source. It comes with no warranty. The recompiler it is
built on and the other third-party code keep their own licenses and notices
([THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). The license grants nothing for the game itself.

Rekarina is an independent, non-commercial fan project, not affiliated with or endorsed by the game's publisher or
the console's maker. All trademarks belong to their owners.

Most of the code and documentation is written by AI coding agents (Claude Code) under the maintainer's direction; the
maintainer plays every build and decides what is merged. [AGENTS.md](AGENTS.md) holds the rules they work by.
