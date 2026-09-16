# Changelog

All notable changes to ALICE-Streaming-Protocol will be documented in this file.

## [Unreleased]

### Fixed
- Release profile no longer sets `panic = "abort"`: PyO3 turns a Rust panic into a Python `PanicException` only when the panic can unwind, so with `abort` any panic inside a `libasp` call (a mismatched NumPy shape, an index past the frame) terminated the interpreter instead of raising

### Added
- `flatbuffers_api::encode_payload` omits empty vectors (regions / motion vectors / region deltas / corrections): a D-Packet without deltas is 16 bytes smaller and byte-identical to `PacketEncoder::encode_d_packet`; the reader already treated an absent vector as empty
- `tests/wire_roundtrip.rs`: `PacketEncoder` output parses to the same packet as `to_bytes` (D / I / S), payload helpers (`correction_count`, `total_correction_bytes`, `add_region_delta`, `dominant_color`, `as_c_packet` / `as_s_packet`), `estimated_size` per-item allowances, bincode path length / header / CRC validation
- `.cargo/mutants.toml`: 7 documented `exclude_re` classes for output-equivalent mutants in `codec/motion.rs` (SIMD dispatch arms, symmetric pattern constants, the never-reached iteration budget, blocks outside the frame)
- `tests/analytic_oracle.rs`: physics bridge round trip on the centimetre grid (block mapping, i16 clamp, dead band, bodies added after the snapshot ignored), RLE mask round trip (3 bytes per run, u16 split, short / oversized masks, truncated triple), hybrid transmitter byte accounting and scene versions, `estimate_savings` closed form; CI runs the oracle file with the full feature set too (the bridge tests are feature-gated)
- `tests/analytic_oracle.rs`: motion estimation contract tests — boundary blocks recover the shift and a buffer shorter than `width × height` never panics (blocks that do not fit are static), candidates outside the search window or the frame are never reported (a SAD-0 match that wraps past the right edge is rejected), the step schedule / diagonal pattern of three-step reach the full window on a linear ramp (SAD zero set `dx + dy = −(sx + sy)`), `early_threshold` is a strict upper bound (s + 1 accepts the static block, s keeps searching, 1 accepts only an exact match)

## [1.1.0] - 2026-09-16

### Fixed
- `media-stack` / `codec` feature compiled against `alice-codec` 0.1.2 (crates.io): `segment_by_motion`, `rgb_to_ycocg_r`, `ycocg_r_to_rgb`, `quantize_buffer` / `dequantize_buffer` return `Result` there, and `VideoDecoder::decode_frame` mutated its reuse buffers through `&self` (a borrow-check error). 1.0.0 shipped the feature in a state that never compiled locally; CI passed only because the sibling crates were replaced by empty stubs
- `packet::AspPacket::write_to_buffer_bincode` created the header placeholder with `set_len` on reserved (uninitialised) bytes; it is zero-filled now (`clippy::uninit_vec`)
- Doc examples: all 10 `ignore` doctests are real, compiled examples now; `create_d_packet` (3 arguments) and the non-existent `read_motion_vectors` in the crate docs, and the `LossDetector` example (`reorder_tolerance` semantics) were wrong
- `python`: removed the never-registered `encode_video_frame` placeholder that always returned an error
- **The FlatBuffers wire format (`AspPacket::to_bytes` / `from_bytes`, the default) dropped most of every payload without an error**: an I-Packet kept only width / height / fps / timestamp (quality, palette, regions, animation lost), a D-Packet kept ref_sequence + motion vectors (global motion, region deltas lost), an S-Packet was written as Ping for six of the eight `SyncCommand`s and never carried its `SyncData`, and a C-Packet was serialised as a Ping and read back as an empty payload. `flatbuffers_api::{encode_payload, decode_payload}` now carry every field the schema has; `from_bytes` also rejects a header whose type disagrees with the payload. Fields with no schema counterpart (`sdf_scene` / `sdf_delta` / `person_mask`, DCT indices above `u16::MAX`) return `AspError::SerializationError` instead of being dropped — they use the `bincode-compat` path. `tests/wire_roundtrip.rs` (7 tests) pins the round trip for every packet type, every enum value and every optional field
- `VideoDecoder::decode_frame` validates the serialised histogram lengths (≤ 65 536 bins per channel) and the remaining payload length before allocating; a truncated or corrupt frame returns `None` instead of allocating up to 4 GiB or panicking on a short slice

### Added
- `SearchAlgorithm` is honoured: `FullSearch` (exhaustive, exact global SAD minimum), `ThreeStepSearch` and `HexagonSearch` are implemented; until 1.0.0 every variant ran diamond search (`estimate_motion_parallel` ignored its `_algorithm`, `MotionEstimator::with_algorithm` and the Python `algorithm=` argument had no effect). `estimate_motion_with(…, algorithm, early_threshold)`; Python `estimate_motion_numpy(algorithm=, early_threshold=)`
- `tests/analytic_oracle.rs`: 18 closed-form tests (orthonormal DCT: DC = N·c, Parseval, basis selectivity, exact inverse; JPEG quality scaling; sparse threshold; motion estimation recovers a pure translation with SAD 0 — FullSearch on any texture and block size, diamond / hexagon on smooth texture, SIMD SAD = scalar reference; CRC-32/ISO-HDLC check value 0xCBF43926; AIMD closed form; k-means palette exact means independent of iteration count, k-means weights = cluster fractions; median-cut palette exact on separated clusters for every split axis; ROI detection: Sobel step-edge strength 2·255/(B−2), block contrast max − min, motion magnitude mean |Δ|, strict `>` thresholds, boundary-aligned edge invisible to the interior Sobel pass)
- `benches/perf_claims.rs` — the README performance rows are measured, not estimated
- `VideoEncoder::try_encode_frame` — `Result` variant of `encode_frame` (size mismatch as `CodecError::InvalidBufferSize` instead of a panic); `VideoEncoder::encode_person_region` returns `Result`
- `#![deny(clippy::undocumented_unsafe_blocks)]`: every hand-written `unsafe` block (29: NEON / AVX2 SAD dispatch, header raw write, NumPy zero-copy in/out, FlatBuffers vector build) carries a `SAFETY:` comment stating the invariant; the flatc-generated module is exempt at its declaration. `AspPacket::write_to_buffer_bincode` CRC append and the Python D-packet encoder no longer use `set_len` on reserved memory
- `rust-version = "1.87"` (verified by the CI `msrv` job), `resolver = "3"`, docs.rs feature set
- CI: 7-job `ci.yml` (3-OS tests, clippy `-D warnings` over 3 feature sets, msrv, feature powerset, rustdoc `-D warnings`, fmt, actionlint), `scripts/preflight.sh`; `security-audit.yml` semver-checks is a hard gate against crates.io; `fuzz.yml` build step blocking, seeds replay

### Changed
- All ALICE sibling dependencies (`alice-codec` / `alice-sync` / `alice-physics` / `alice-crypto`) come from crates.io; the CI "dependency stubs" (empty 0.1.0 crates, which could not even satisfy `alice-physics = "1"`) and the `alice-stubs` action are removed
- pyo3 / numpy 0.23 → 0.29 (RUSTSEC-2025-0020 / RUSTSEC-2026-0177 resolved, `allow_threads` → `detach`)
- `rust-toolchain.toml` pin 1.92.0 → 1.98.1
- `video_codec`: the frame layout is a `CompressedFrame` struct (was a 17-element tuple through `pack` / `unpack`); the tuple made `cargo mutants --list` exhaust memory (return-type replacement candidates are a cartesian product over the tuple elements) and blocked mutation testing of the whole crate
- Mutation testing (`quality-deep.yml`): 16 shards over the `simd,media-stack,sync,physics,crypto,bincode-compat` build, lib + `analytic_oracle` test set; `src/generated/**` (flatc) and `src/python.rs` (needs an interpreter) are excluded in `.cargo/mutants.toml`
- README "Performance Highlights": the "CRC32 table 2.8-5.3x faster" and "buffer reuse 77x faster" rows were not reproducible (154 µs vs 150 µs per 64 KiB; 297 ns vs 304 ns per I-packet on Apple M-series) and now state the measured values

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
