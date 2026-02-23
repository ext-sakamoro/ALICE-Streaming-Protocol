# Changelog

All notable changes to ALICE-Streaming-Protocol will be documented in this file.

## [1.0.0] - 2026-02-23

### Added
- `packet` — `AspPacket` with I/D/C/S packet types, buffer-reuse serialization
- `header` — Packet header with CRC32 (compile-time lookup table)
- `types` — `IPacketPayload`, `DPacketPayload`, `CPacketPayload`, `SPacketPayload`
- `codec` — Motion estimation (Diamond/Hexagon search), color extraction (k-means), DCT, ROI detection
- `flatbuffers_api` — Zero-copy FlatBuffers serialization for cross-language support
- `generated` — FlatBuffers auto-generated types from `schemas/asp.fbs`
- `scene` — SDF scene descriptor for hybrid streaming
- `hybrid` — Hybrid streaming pipeline (SDF background + wavelet person)
- `media` — ALICE-Codec + ALICE-Voice media stack integration (feature-gated: `codec`, `voice`)
- `sync_bridge` — ALICE-Sync event embedding (feature-gated: `sync`)
- `physics_bridge` — Physics state delta → D-packets (feature-gated: `physics`)
- `crypto_bridge` — AEAD encryption for ASP packets (feature-gated: `crypto`)
- `python` — PyO3 + NumPy zero-copy bindings (feature-gated: `python`)
- Feature flags: `python`, `wasm`, `simd`, `bincode-compat`, `codec`, `voice`, `media-stack`, `sync`, `physics`, `crypto`, `all-bridges`
- 80 unit tests
