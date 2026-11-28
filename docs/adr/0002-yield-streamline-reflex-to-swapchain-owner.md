---
status: superseded by ADR-0003
---

# Yield Streamline Reflex and Swapchain Hooks to the Active Swapchain Owner

> Superseded on 2026-11-27: ShortFuse's `vk_dlssg_swapchain_ready()` and DLSS load at start are the code ([ADR-0003](0003-merge-shortfuse-renderer-work-verbatim.md)). If Streamline's Reflex sleep on an AMD-owned swapchain shows up in a log or a crash, it goes to him rather than into a patch here.

To keep live upscaler and frame-generation backend switching working on `OoT` without restarting, Streamline preparation and AMD FSR SDK 2 graphics queue reservation remain available at startup, **but** Streamline Reflex, swapchain-readiness, and consumer-resolution queries must yield whenever AMD FSR SDK 2 is the active **Swapchain Owner**, and native NGX (`_nvngx.dll`) is loaded lazily on first live DLSS demand rather than eagerly at startup when Streamline is inactive and DLSS is not requested.

## Considered Options

- **Always-active Streamline swapchain readiness and eager NGX init (ShortFuse's approach)**: Keeps `vk_dlssg_swapchain_ready()` returning true even while AMD owns the swapchain, causing Streamline's `slReflexSleep()` to run against an unowned swapchain and consumer snapshots to misreport DLSS ownership while suppressing host frame-start pacing, and eagerly loads `_nvngx.dll` at startup whenever Streamline is absent.
- **Startup-only Streamline initialization**: Avoids all Streamline interposition when `DLAA=off` or `FSR` is selected, preserving native `VK_NV_low_latency2` Reflex and `VK_EXT_PRESENT_TIMING_EXT`, at the cost of breaking live switching to DLSS / DLSS Frame Generation without a restart.
- **Live preparation with strict Swapchain Owner yield and lazy NGX (chosen)**: Preserves live switching across Off, Bilinear, FSR, DLSS, DLSS FG, and AMD FSR FG while guaranteeing Streamline never sleeps on an AMD-owned swapchain and avoiding wasted native NGX initialization on non-DLSS runs until DLSS is actually selected.

## Consequences

- Callers gating Reflex, frame pacing, or consumer resolution (`vk_dlaa_resolve_consumers()`) must check `vk_dlssg_owns_swapchain()`, not merely that Streamline initialized at startup.
- When Streamline is loaded at startup on RTX hardware with `FRAMEGEN=off`, Streamline still interposes the Vulkan swapchain so live DLSS-FG toggles do not require a restart.
- On a non-Streamline session that starts without DLSS and switches live to DLSS, native NGX initializes on demand during the switch.
