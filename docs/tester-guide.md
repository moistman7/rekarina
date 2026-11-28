# Tester guide: running TotK Recomp on your GPU

This guide is for testing the build on a GPU other than the developer's (AMD RX 9070). You build it from source with
your own copy of the game and your own console keys. Nothing from the game is in this repository, and nothing you build
from it may be shared.

## What you need

- Windows 10 or later, an x86-64 CPU with SSE4.2 (AVX2/FMA recommended: they build the default, faster version), about
  60 GB free.
- A GPU with Vulkan 1.3 and `VK_EXT_external_memory_host`, with a current driver.
- TotK base game (1.0.0) and the 1.1.0, 1.2.1 or 1.4.3 update, dumped from your own Switch as NSP or XCI.
- `prod.keys` dumped from your own Switch. Never send it to anyone, and never paste it into a log or an issue.

## Build

1. Follow [building.md](building.md): the toolchain, then build steps 1-6 (Python environments, `config.local.toml`
   with your own paths, extraction, the shader translator, lifting and linking, shader preparation). The result is
   `build\<ver>\rekarina.exe`, with `<ver>` the game version you built (`v143` or `v121`).
2. Check the build without the game: `build\<ver>\rekarina.exe --selftest` must print a pass.
3. To run the build on another of your machines, package a portable folder ([building.md](building.md), "Portable
   folder"). The folder holds game code: for your own machines only.
4. The installer ([installer.md](installer.md)) is the other route to a machine without the toolchain: it builds the
   game there from your own files. No installer release is published yet.

## Run and collect the log

Every run writes its log to `logs\totk.log` next to `rekarina.exe`; the four runs before it are kept as `totk.1.log` to
`totk.4.log`, and Open log folder in the settings menu's Advanced tab (F1) shows them. Copy the log before you start the
game again. A file stops at 64 MiB. Started with `2> file`, the game writes its log to that file instead.

From `cmd.exe` in the repository root, with the validation layer on:

```bat
set REKARINA_VK_VALIDATION=1
build\<ver>\rekarina.exe
```

`REKARINA_VK_VALIDATION=1` turns on the Vulkan validation layer (it comes with the Vulkan SDK). It makes the game slower;
do one run with it and one without (`set REKARINA_VK_VALIDATION=`).

If the game stops with a lost device (the log says `device lost` or `VK_ERROR_DEVICE_LOST`), the lines starting with
`vk: pass marks` name the render pass or dispatch the GPU was running. AMD GPUs write them by default. On an NVIDIA GPU,
`set REKARINA_PASS_CRUMBS=1` (or GPU pass marks: On in the Advanced tab) before the run that reproduces the crash.

The game keeps running behind other windows (`assume_focus = 0` in `rekarina.ini` pauses it instead). Shaders the preparation did not cover compile on first use, so
the first minutes of play can stutter.
Play into the open field (load a save, walk around for a minute, fight something if you can), then close the window.

## Choosing the GPU

A laptop with an iGPU and a discrete GPU, or two cards, lists every GPU at the top of the log (`vk: gpu 0: ...`).

- `set REKARINA_GPU_NAME=Radeon` picks the first usable GPU whose name contains that text (case does not matter).
- `set REKARINA_GPU_INDEX=1` picks GPU 1 from that list.

The line `vk: using gpu N: <name>` says which one ran.

## Mods

Mods and cheats made for the emulators go into `mods/` next to `rekarina.exe`, one folder per mod. [mods.md](mods.md) covers
installing them, their order, code patches, what does not work and what to check when something looks wrong;
[cheats.md](cheats.md) covers cheat files. The log names every mod at the start (`mods: <name>: ...`); when you
report a problem with mods, list them with their download links.

## What to send back

The quickest way (Rekarina, OoT): a **problem report**, one zip with everything below except your description and
screenshots. Make it in any of three places:

- in the game: Esc (or Minus and Plus held together), then *Advanced* or *Menu*, *Make a problem report*;
- in the launcher: the *Problem report* button on its home screen (the game does not need to start for it);
- from a terminal in the game folder: `Rekarina --make-report`.

It is written to `reports\rekarina-report-<date>-<time>.zip` beside the game, and that folder opens. Inside:
`system.txt` (the OS, CPU, memory, every graphics card with its driver, the device profile and why, the build and game
versions, the settings in effect, the size of each shader and pipeline cache), the logs of the last five runs (a crash
or a lost graphics device is written at the end of the run's log), `rekarina.ini` and the setup's record. No save data
and no game files; your user name and home folder are replaced in every path. Nothing is sent anywhere.

When the game stops with an error (a crash, a fatal error, a lost graphics device) it says so in a message box that
names the log; at the next start a note at the top left says the last run ended with an error and where the button is,
and the launcher's home screen says the same.

Otherwise, by hand:

1. `logs\rekarina.log` from each run (`logs\totk.log` in a TotK folder). It contains no keys and no game files. The lines that matter most start with
   `vk caps:` (GPU, driver, features and the fallbacks taken), `vk: gpu`, `validation` and `gfx_shader:`.
2. GPU model, driver version, CPU, RAM, and whether it is a laptop. Copy system info in the settings menu's Advanced
   tab (F1) puts these and your changed settings on the clipboard. Say which game update you ran.
3. What you saw: how far it got (menu, loading, field), roughly how smooth it ran, and screenshots of anything wrong
   (black or missing areas, flickering, wrong colours, missing grass). Win+Shift+S is fine.
4. If it crashed or hung: what you were doing. The log of that run ends with the crash report, so send it whole.
5. For a bug you can trigger again: a recording of your input (`set REKARINA_INPUT_RECORD=input.txt`, see
   [controls.md](controls.md), "Scripted input") and the save you started from.

## Steam Deck

Build or install on Linux ([installer.md](installer.md), "Linux release"), in Desktop Mode. The game then runs in either
mode; for Game Mode, add `totk` from the install folder to Steam as a non-Steam game.

The game recognises the Deck (LCD and OLED) and uses its device profile for every setting `rekarina.ini` does not set:
fullscreen, the screen's own 1280x800 (16:10: the camera and interface take that shape), the game's own 3D scale,
vsync and a 40 fps limit. These are starting points, not measurements. What helps most:

1. The performance overlay (F10, or both stick clicks L3 + R3) in full mode: the frame rate, mean and longest frame
   time over a minute in the first town, in the Depths and in the Sky, with the settings named. The Window row
   shows the actual window or fullscreen output in pixels; Game image shows the picture size before display scaling.
2. The same with `fps_limit` 30, 45 and 60 and `render_scale` 50, 67 and game, so the defaults can be picked from
   measurements. Steam's own performance overlay (the ... button) adds power draw and temperatures.
3. LCD or OLED, SteamOS version (`uname -r` in Desktop Mode's terminal), and the game update.
4. Anything that looks stretched, squeezed or cut off, with a screenshot. At 16:10 the menus keep their 16:9
   proportions in the middle of the taller screen, the HUD sits at the top and bottom edges, and full-screen
   backgrounds reach them; pre-rendered cutscenes play stretched.

`device_profile = desktop` in `rekarina.ini` turns the profile off; `deck` turns it on elsewhere, to try the Deck's
settings on a PC.

### OoT on the Steam Deck and on Windows handhelds

OoT runs on a Deck two ways: the Linux setup (`Rekarina-x86_64.AppImage`, built since 2026-11-15, never run on a Deck)
builds a native game, or the Windows build runs under Proton, which recognises the Deck from the `SteamDeck` variable
Steam sets or from the firmware's product name (Jupiter, Galileo). Both run shadow storage (oot-notes.md, "2026-11-15:
shadow storage draws the scene"): the Linux build always (it cannot import the game's memory for the GPU while it watches
it for writes), the Windows build under Wine or Proton. On a desktop stand-in shadow storage takes the processor about
twice the time a frame and commits some 11 GiB more memory, so expect a Deck to be limited by its processor and its 16 GB.
Since 2026-11-16 the Linux build also leaves GPU writes the processor does not read in the shadow (`REKARINA_SHADOW_LAZY`,
on by default; measured on Windows only): if the Linux game crashes with an access violation that the Windows build
does not have, try `REKARINA_SHADOW_LAZY=0` and say so in the report. The Deck's profile adds to the keys above the game's own
handheld mode (`operation_mode = handheld`: the console's 1920x1080 picture, about 30 % less GPU time than docked) and
quarter textures (`tex_quality = quarter`: the 16 GB are the GPU's memory too).

A Windows handheld (ROG Ally and Ally X, ROG Xbox Ally, Legion Go, MSI Claw, by the BIOS's product name) gets the
`handheld` profile: fullscreen, handheld mode and a 60 fps limit; textures stay on auto. `device_profile = handheld`
turns it on elsewhere. Neither profile has run on its device yet: the frame rate in the forest, the field and the
Market, the log (`logs\rekarina.log`) and anything that looks wrong are what helps.

The launcher's first settings for a processor's own graphics (Deck, Ally, laptops without a card) draw the 3D scene at
half size and scale it up with FSR 1 (`dlaa = 1`, `taa = fsr1`, `dlss = performance`). Worth comparing on the device:
`dlss = quality` (67 %) and `balanced` (58 %) against `performance`, and `taa = fsr1` against `taa = fsr3` (FSR 3.1,
temporal: sharper when standing, more GPU time), each with the frame rate in the same places.

### Linux and the Steam Deck: exactly what to run and what to send (OoT)

Nothing has run on Linux with a real GPU yet, so every step here is new ground. In Desktop Mode, in a terminal (Konsole
on a Deck):

1. The setup: `chmod +x Rekarina-x86_64.AppImage && ./Rekarina-x86_64.AppImage`. It builds the game into the folder you
   choose. Note how long each step took and whether the folder is on the internal drive or a microSD card.
2. The memory test, in the game's folder: `./Rekarina --memory-test`. It takes a few seconds and needs no game data.
   It says whether your GPU driver can share memory with the game the fast way (the lines ending in `USABLE` or
   `NOT USABLE`, and the `RESULT:` line) and writes the same text to `memory-test.txt` there. Send that file whatever
   it says: it decides whether the game can drop shadow storage on Linux (docs/oot-notes.md, "Guest memory from the
   GPU driver on Linux").
3. The game: `./Rekarina` (or from Steam as a non-Steam game, for Game Mode). Play a few minutes in the first village
   and the forest, then quit from the menu.
4. The report: `./Rekarina --make-report` writes `reports/rekarina-report-<date>-<time>.zip` (the logs of the last
   five runs, the settings, the system's details and `memory-test.txt`; no saves, your user name taken out). Send the
   zip, with the device (LCD or OLED Deck, or the PC's GPU and distribution) and what you saw.

Or all of it at once: `tools/linux_test_run.sh <game folder> [seconds]` runs the memory test, the selftest, then the
game twice (shadow storage, then driver memory, `REKARINA_LINUX_GUEST_MEMORY=driver`) and writes `linux-test-<date>.txt`
beside the game with the system, the frame-time summary and the memory lines of each run, then makes the report zip.
Send the text file and the zip. Driver memory is off by default until it has been played on a Deck; the script is what
shows whether it is safe to turn on.

What the developers look for in the log (`logs/rekarina.log`), if you want to check first: the line starting
`mem: guest memory:` (which way the game keeps its memory and what the device offers), `mem: shadow at frame` (what
shadow storage copies), and `frametime: exit` (frame rate and the slow frames). If the game crashes with an access
violation that the Windows build does not have, run it once more with `REKARINA_SHADOW_LAZY=0 ./Rekarina` and say
whether that changed it.

### Checklist for a handheld or a small PC (OoT)

The menu (Esc, or Minus and Plus held together) shows what the game found: on the Graphics tab, the *Graphics card* row's
picture lists the card, driver, video memory, processor, memory, display and the device profile with the reason it was
chosen; the *Performance overlay* row's picture shows the frame times, the CPU and GPU time per frame and the memory in
use. A photo or screenshot of those two pictures and `logs/rekarina.log` answer most questions. Mark each item done,
wrong (with what you saw) or not tried.

Steam Deck (LCD or OLED; say which, and the SteamOS version):

1. Desktop Mode: the setup builds the game; note how long each step took (the setup shows it) and whether the install
   was on the internal drive or a microSD card.
2. Game Mode: add the game (`Rekarina` in the install folder) to Steam as a non-Steam game and start it from there.
   It fills the screen; the Steam button's overlay and the performance overlay (the ... button) open over it.
3. The System picture says "Steam Deck: found by its firmware"; the button prompts are the Xbox set (the Deck's own
   controls report Xbox labels); nothing in the HUD is cut off or misplaced at 16:10.
4. A name entry (the game's own keyboard): Steam + X brings up the on-screen keyboard and the text arrives.
5. Frame rate in the forest, the field and the Market at the profile's 40 fps limit, then at 30, 45 and 60 (Game tab,
   *Frame limit*), with the overlay in full mode. Also once started from Desktop Mode: Desktop Mode does not get Game
   Mode's power settings, so it may be slower.
6. Suspend (the power button) during play and resume after a minute: the picture, the sound and the controls come back.
7. Ten minutes of play: the overlay's RAM and VRAM bars stay below their ends, and no "video memory" notice appears.

ROG Ally, Ally X, ROG Xbox Ally, Legion Go, MSI Claw (Windows):

1. The System picture says "Handheld PC: found by its firmware" (if it says Desktop, send the picture: the firmware's
   product name is missing from our list).
2. The built-in controls work in game and in the menu, and the prompts are the Xbox set. If the controls are listed
   (the Controls tab names the controller) but do nothing, set *Controller driver* to Compatibility and restart.
3. The forest is not white and the grass and bushes are there (Radeon graphics); shadows under trees.
4. Frame rate at 60 (the profile's limit), 45 and 40, at the device's 15 W and 25 W modes.
5. Sleep and wake during play.

A PC with 16 GB of memory or a 4 GB graphics card:

1. The System picture's video memory, and *Texture quality* on the Graphics tab (Automatic picks half or quarter on a
   small card).
2. Ten minutes in the field and the Market: the overlay's VRAM and RAM bars, any "video memory" notice, any frame
   over 25 ms the graph shows as a spike.

Any controller that is found but does nothing: the controller picture on the Controls tab (select *Controller driver*)
lights each button as it is pressed. If nothing lights, try *Controller driver* = Compatibility (restart) and say which
of the two works, with the controller's name the row shows.

## Switches for the developer's questions

Only set these when asked; each forces a fallback the build picks by itself on GPUs that need it.

| Variable | Effect |
|---|---|
| `REKARINA_VK_NO_FLOAT64=1` | Translate shaders without 64-bit floats |
| `REKARINA_VK_SSBO_ALIGN=64` | Treat storage buffers as needing 64-byte alignment |
| `REKARINA_VK_SUBGROUP_PIN=0` | Do not pin shaders to 32 lanes; tell the translator the GPU's own subgroup size |

Each switch makes the first launch translate every shader again.
