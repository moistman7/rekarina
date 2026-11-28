# Merge ShortFuse's Renderer Work Verbatim

ShortFuse (`https://github.com/clshortfuse/rekarina`) develops the renderer, and this fork is its release channel. His integration branch (`integration/upstream-1.2.19` when this was written) is his own rebase onto our releases, so we merge it rather than port his commits, and his lines stay as he wrote them. The one exception is a line his rebase broke, where behaviour that worked before the rebase stopped working: we restore it in its own commit. We report performance or correctness concerns to him instead of patching his code here.

The first port of his work on 2026-11-27 had rewritten his code in places where nothing was broken: **Output Proof** checks gated to HDR with frame generation, a separate Streamline ownership query, DLSS loaded on first use, conditional **Present Drain**s, a dropped wait before scene sizes were destroyed, and RTX upscaling presets that he had made opt-in. The merge took his code back. ADR-0001 and ADR-0002 recorded two of those rewrites and are superseded.

## Considered Options

- **Port and adapt his commits**: allows local tuning, but each port drifts from his code, every sync re-resolves the same conflicts, and his authorship is lost.
- **Merge his branch and keep his lines verbatim (chosen)**: syncs are plain merges, his commits keep his name, and a diff against his branch shows exactly what we changed.

## Consequences

- When a sync conflicts with one of our ports, his side wins.
- Wording or comments in his lines that look stale stay until he changes them; only behaviour his rebase broke is restored.
- Fork features his commits do not touch, such as direct save boot and the voice language row, stay ours.
