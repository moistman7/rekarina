# Rekarina

Static recompilation of a Switch 2 game (ARM64) into a native Windows x86-64 exe with a Vulkan renderer. The code and the notes call the game `OoT` (`GAME_OOT`; `GAME_NS2` is what any Switch 2 game needs). It is built on the TotK Recomp lifter and runtime, and the same tree still builds that project's Switch game, `TotK`. Design: [docs/architecture.md](docs/architecture.md) and, for what OoT adds, [docs/oot-architecture.md](docs/oot-architecture.md). Toolchain, build commands and tests: [docs/building.md](docs/building.md). Working notes on OoT: [docs/oot-notes.md](docs/oot-notes.md). The launcher and the release flow: [docs/installer.md](docs/installer.md), [docs/distribution.md](docs/distribution.md).

## This fork

This repository is a public fork of Rekarina (`origin`: `https://github.com/arnaudcouturier/rekarina.git`; `upstream`: `https://git.rekarina.dev/rekarina/rekarina.git`). On top of upstream releases, it maintains:

- **Streamline DLSS / DLAA / DLSS Frame Generation**: Streamline integration (`runtime/dlssg/nx2_sl.cpp`, `runtime/vk_dlssg.c`, `runtime/vk_dlaa.c`), 16-phase Halton CPU projection jitter (`runtime/scene_projection.h`, `runtime/scene_projection_impl.h`, `runtime/scene_motion_projection.h`), tracked model motion vectors (`runtime/tracked_model.h`, `runtime/tracked_model_cpu_impl.h`, `runtime/skin_pair.h`, `runtime/vk_skin_motion.inc`, `runtime/shaders/model_motion.frag`; their NVN2 last-frame vertex variants: translator patch `0045-motion-previous-on-ampere` and the `gfx_shader.c` `put_mvprev_tag` key tag), scaling-dispatch replacement in both DLAA and DLSS modes with jitter texture LOD bias (`log2(render_share) - 1.0`), an opt-in resolve of OoT's native DLAA before the bloom (`REKARINA_DLAA_PRE_BLOOM`, `runtime/vk_dlaa.c` `pre_bloom_draw`), and no redundant 1:1 present RCAS when projection jitter is active.
- **RenoDX HDR**: SPIR-V splicing of the game's ACES tonemap pass (`runtime/vk_hdr.c`, `runtime/shaders/hdr_tonemap.frag`, `runtime/shaders/hdr_tonemap_totk.frag`, `runtime/shaders/hdr_encode.comp`, `runtime/shaders/hdr_ui.frag`, `tests/hdr_splice_check.c`), matching translator NaN-sanitized output stores, with live F1 toggling/tuning and native HDR10 output.
- **Direct save boot**: `--load-save` boot path (`runtime/main.c`, `runtime/fs_host.c`, `runtime/romfs.c`, `recomp/patches-oot-v111.txt`).
- **Performance**: Vulkan renderer fixes and tuning for frame pacing and temporal image stability.
- **Releases and support** use this fork's GitHub Releases and Issues. Keep Windows launcher identity and update state separate from upstream (`installer/fork.py`, `runtime/update_note.h`). For packaging, upgrade tests, or drafting a release, read [docs/distribution.md](docs/distribution.md). The maintainer publishes locally built Windows launcher releases.
- **Upstream imports** start with a maintainer request naming the change to bring over. Keep the original `upstream` remote and review the requested change against the fork before merging it.
- **ShortFuse's renderer work** (`https://github.com/clshortfuse/rekarina`, his integration branch) comes in by merge, verbatim: his lines stay as he wrote them, except a line his rebase broke, restored in its own commit. Performance or correctness concerns go to him ([ADR-0003](docs/adr/0003-merge-shortfuse-renderer-work-verbatim.md)).

## No game is named

- No game, franchise, character or place is named anywhere in this repository: not in code, comments, docs, issue
  forms, release notes or commit messages. The two games are `OoT` and `TotK`, the abbreviations the code uses; a page a
  player reads says "the game". Places are plain words ("the forest", "the field", "the Market", "the village"), the
  player's character is "the hero". The game's own identifiers that the code must match (map and event names in
  `runtime/oot_warp.c`) are the one exception.
- No pictures taken from a game: no screenshots, no frames of a video, no textures. Pictures of the launcher and charts
  are fine.
- Keep changes in focused commits so the maintainer can review each upstream import and fork change.

## Names

- The player sees Rekarina's names: `Rekarina.exe`, `rekarina.ini`, `logs\rekarina.log`, `rekarina_xlate.exe` with its `.id`, `rekarina_dlssg.dll` (`runtime/host.h` `RT_FILE_STEM`; `tools/package_oot.py` and `tools/make_rekarina_release.py` name the files). A folder made before those names existed keeps working: the game moves `totk.ini` over and still takes the old translator and library names.
- Our own names are Rekarina's everywhere (renamed from the TotK recompiler's `totk` on 2026-11-15): the build's `build/<ver>/rekarina.exe` and `rekarina.map`, `librekarina.a`, `runtime/rekarina.rc`, `rekarina.ini.example`, the setup's `rekarina-install.json`, the `REKARINA_` prefix of settings. The old ones are still read where a player or a script may have them: a `TOTK_` environment variable or ini key is the same setting (`runtime/cfg.c`), a `totk-install.json` is taken over (`installer/state.py`).
- `totk` stays where it is not ours to rename: TotK the game and its setup's player-facing files (`totk.exe`, `totk.ini`, `TotK-Setup`), upstream names and paths (`totk-pc-recomp`, MacNativeGames' `games/totk`), the binary signatures of saved caches (`TOTKPKEY` …), and the translator's `totk_xlate` (next item).
- `tools/build_xlate.py`, `tools/xlate/Program.cs`, the `.csproj` and `tools/xlate/patches/` make up the translator identity. A change to any of them, a rename included, makes every player's shader cache stale: change them only when the translation itself has to change.

## Game content boundary

This repository is public. Game data lives outside it and every derived file stays in ignored folders.

- The user's game and key paths are in `config.local.toml` (ignored). Pass the keys file path to tools; the tools read it themselves. Key values stay out of your output, logs, commands and files.
- Everything extracted or generated goes under `build/` or `generated/`.
- Every file the code generator emits carries the marker defined as `GENERATED_MARKER` in `tools/check_content.py`.
- Run `python tools/check_content.py --staged` before proposing a commit. It is the same check CI runs.

## Reusing code

The project is under the GNU General Public License, version 3 (`LICENSE`); the recompiler it is built on is MIT, and
its notice is kept in `THIRD_PARTY_NOTICES.md`.

`refs/` holds the upstream checkouts (`tools/fetch_refs.sh` recreates it) when present. Read-only.

- Copy only MIT, BSD, ISC, zlib or Apache-2.0 code, keep its header, and add a row to `THIRD_PARTY_NOTICES.md`.
- This fork draws on MIT-licensed code from `RenoDX` (`clshortfuse/renodx`) and ShortFuse's MIT-licensed commits in `NX2navi`. Never copy all-rights-reserved (`chrg6`) files from `NX2navi`.
- yuzu, suyu, OpenSmash-Ultimate and other GPL code: read for behaviour, write our own.
- `tools/vendor/` stays byte-identical to upstream; Windows changes go in `tools/extract.py`.

## Code

- Build C with MSYS2 UCRT64 Clang (`C:\msys64\ucrt64\bin`). Generated C uses GNU extensions, so `cl.exe` is out. Call Ninja from that directory by full path (`C:\msys64\ucrt64\bin\ninja.exe -f build\oot\build.ninja`) or put it first on PATH.
- Python 3.14 on PATH runs the build, extraction and shader scripts (`compression.zstd` is built in); upstream's split `.venv` (3.12) / `.venv-py314` layout is also supported.
- Rebuild generated shader headers with `python tools/build_shaders.py` after editing `runtime/shaders/*`, and `rekarina_dlssg.dll` with `python tools/build_dlssg.py` after editing `runtime/dlssg/nx2_sl.cpp`.
- Generated C is never edited by hand: change the generator (`recomp/`).
- Checks to run after changes:
  - `python tools/check_platform.py` and `python tools/check_content.py --staged`
  - `build\oot\rekarina.exe --selftest`, `--probe-cfg`, `--probe-color`
  - `python -m installer.launcher_test` for launcher/updater changes
  - Fork C unit checks (`tests/hdr_splice_check.c`, `tests/scene_motion_projection_check.c`, `tests/skin_pair_check.c`, `tests/tracked_model_cpu_check.c`) when touching those modules
  - Every lifter change passes the differential tests against Unicorn ([docs/building.md](docs/building.md), "Checks"). After changing `recomp/jumptables.py`, `tests\jumptables_check.py` reports 0 disagreements on 1.2.1 and 1.4.3.
- `recomp/layout.py` alone decides module bases.
- Graphics bugs: [docs/maxwell.md](docs/maxwell.md) lists the Maxwell behaviour the Vulkan path must match and the TotK bugs other ports fixed.
- Comments say what the code does and why, in words that stand on their own. Cite our code by file plus function (a line number may follow as a hint) and upstream code by project, file and function: `refs/` moves, so its line numbers rot. Commit hashes belong to the attribution of copied code.

## Writing

Code, comments, docs and commit messages are in English. Commit messages are plain sentences without type prefixes.
