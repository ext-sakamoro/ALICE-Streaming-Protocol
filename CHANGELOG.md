# Changelog

All notable changes to ALICE-Streaming-Protocol will be documented in this file.

## [Unreleased]

### Security
- `bincode-compat` は bincode crate を使わず、alice-serial の `bincode1` で bincode 1 の layout を読み書きする bincode (RUSTSEC-2025-0141、unmaintained) は依存 graph から消え、`deny.toml` と `cargo audit --ignore` の除外を外した `scripts/no_bincode_in_graph.sh` が全 feature の依存 graph と Cargo.lock に bincode が戻ると落ちる (0 package も落ちる)
- 変わらないもの: feature 名 `bincode-compat`、関数 `to_bytes_bincode` / `write_to_buffer_bincode` / `from_bytes_bincode`、wire の byte (`tests/fixtures/bincode1/` が bincode 1.3.3 で書いた I / D / C / S Packet を固定、I と D は `sdf_scene` / `sdf_delta` / `person_mask` を運ぶ) この経路は FlatBuffers が運べない 3 つの field の唯一の wire
- 変わったもの: payload (header と checksum の間) の後ろに残る byte を拒む (bincode 1 は無視していた)、payload 内の長さは payload の byte 数で上限を掛けてから確保する、入れ子は `alice_serial::bincode1::DEFAULT_MAX_DEPTH` (payload の型は自分を入れ子にしない、SDF の scene は ASDF の byte のまま運ばれ ALICE-SDF が decode する) 試験: 長さを偽った payload と末尾に 1 byte 足した payload を拒む
- `ctutils` 0.4.2 → 0.4.5 (lock のみ、RUSTSEC-2026-0385: 0.4.4 未満の mask と u128 の比較が定数時間でない) 経路は `alice-crypto` → `blake3` 1.8.7 → `digest` 0.11.3 → `ctutils` 0.4.5 の rust-version は 1.87 で、この crate の宣言 (1.87) と同じ

### Changed
- ALICE-Law G1 (標準 libm の超越関数を library で呼ばない): DCT の基底の cos を alice-det-math の `cos64` に、`energy_compaction` の `powi(2)` を積に、`median_cut_palette` の深さ `(n as f64).log2().ceil()` を整数の ⌈log₂ n⌉ に、`MotionVector::magnitude` の `hypot` を f64 の `sqrt(dx² + dy²)` (和は厳密、sqrt と f32 への丸めは IEEE 754 で決まる) にした `clippy.toml` に G1 の禁止 method の区画 (ALICE-Law `scripts/g1_clippy_block.py` の生成物) を置き、CI の clippy (`-D warnings`、all targets) が禁止する 変換前に `tests/g1_conversion_pins.rs` で固定した入力での比較: encoder が出すものと decoder が見せるもの (量子化した係数、sparse 係数、64x64 の frame を載せた I-packet の bytes、復号した pixel、magnitude、palette) は全部同じ (固定した入力での実測で、全入力の保証ではない: 量子化のちょうど境目にある DCT の値は 1e-16 程度の動きで反対側に丸まりうるので、そうした block では変換後の encoder の bytes が変換前と違いうる 復号には影響しない) 動いたのは浮動小数の DCT の出力 (`dct2d` / `idct2d` / `decode_sparse` の値) だけで、block の最大値に対して最大 7.9e-16 (試験の上限 2e-15) 変換前に書いた stream (`tests/fixtures/g1/iframe.bin`) は今も同じ pixel に復号する magnitude は i16 の全組 (対称で 5.4 億組) で変換前の `hypot` と bit 一致 (この機の libm での実測) (sqrt と f32 への丸めはそれぞれ IEEE 754 で決まるので全 platform で同じ値、2 回の丸めの合成なので正しく丸めた f32 の hypot そのものではない) 整数の ⌈log₂ n⌉ は n ≤ 2^20 の全数、2^k ± 1 (k ≤ 48) と 2^49 未満の 20 万点で元の式と一致し、2^49 + 1 からは元の式が誤り (log2 が k に丸まる) で整数の方が正しい (median cut がその深さに届くのは 2^49 を超える pixel の時だけ) 得たもの: 標準 cos では DCT の値が platform の libm で違ったが、alice-det-math では全 platform で同じ bit (`the_dct_bits_are_the_same_on_every_platform` が出力の SHA-256 を定数と比べ、CI の 3 OS で走る) 試験の参照として標準 cos を使う `tests/analytic_oracle.rs` の基底は理由つきの allow (実装と同じ cos を使わないための独立な参照)、試験の入力を作る sin / cos は alice-det-math に
- 依存 (lock のみ): alice-sync を ALICE-Sync main 82ff260 (0.8.0 のまま) に、それが要る alice-identity を 2.0.0 から 3.0.0 (154ab99) に上げた 他の package は動かしていない (alice-crypto / alice-sign は元の rev) 以前の lock の alice-sync (a64fe2b) は alice-identity ^2 を要るので、新しく解決すると Identity の main (3.0.0) と合わず package の検査が通らなかった libasp の公開 API と wire は変わらない: `sync` feature が見せるのは `alice_sync::Event` / `EventKind` だけで、src/event.rs は両 commit の間で変わっていない (Identity の型は libasp の API に出ない)
- 依存: alice-sync を 0.8.0 に上げた (main branch から) この crate が使う `Event` / `EventKind::Motion` / `EventKind::Spawn` は 0.8 で変わっておらず、公開 API と packet の bytes は変わらない
- 依存: alice-crypto を 0.7 に上げた (`std` の API は同じ、0.7 は module ごとの feature を足した版)
- 依存: alice-codec / alice-sync / alice-physics / alice-crypto を crates.io でなく各 repository の main branch と版 (0.2.0 / 0.7.0 / 4.0.0 / 0.6.0) で引く lock の alice-det-math は git main の 0.6.1 の 1 つだけ (これまでは crates.io の 0.3.2)、crates.io の alice-* は 0
- MSRV: 1.87 → 1.88 (alice-sync 0.7.0 と、それが引く alice-identity が 1.88 を要求する) CI の MSRV の job も 1.88
- `deny.toml`: git の source として `github.com/ext-sakamoro` を許可した
- CI (Package integrity): `cargo package` と展開した .crate の `cargo check` を、ALICE の crate と共有する `scripts/package_integrity.py` にした (git の branch で取る依存を lock が記録した commit で patch して、同梱の lock と新しい解決の両方で build する、lock に alice-det-math があることも確かめる) script が import する `alice_siblings_check.py` / `id_move_check.py` / `release_baseline.py` と各試験も同じ bytes で置き、試験は CI の `shared-scripts` job で走らせる
- `alice_siblings_check.py` の試験が読む `.github/actions/alice-siblings/action.yml` も共有の bytes で置いた (この crate の CI はこの action を使わない)
- `video_codec` と試験の 3 byte ずつの区切りを `chunks_exact(3)` から `as_chunks::<3>()` にした (clippy の `chunks_exact_to_as_chunks`、MSRV 1.88 で使える)
- lock: alice-sync を main の先端 (alice-identity 2.0.0 を引く) に上げた 古い commit の alice-sync は alice-identity ^1.2 を要求し、新しい解決で選べなかった
- Licence: `MIT OR Apache-2.0` → `LicenseRef-Proprietary`
- README の MIT の licence badge を外した (指す LICENSE は既に無い)
- `deny.toml` と `.github/workflows/security-audit.yml` の comment の libasp の licence の記述を外した (libasp は publish = false で licence の検査の対象外)
- CI: ci.yml / fuzz.yml / security-audit.yml を `ci/**` の branch でも走らせ、ci.yml に手動実行を足し、main の run は打ち切らない concurrency にした (main に入れる前に branch で全 workflow を通すため)
- **`alice-physics` の要求を `1` から `2` に上げた (2026-10-09)** 2.0.0 は `#[non_exhaustive]` の一括付与と enum の variant 追加を含む major bump で、caret は major を跨がないため `^1` のままでは公開済の 2.0.0 に永久に到達しない `physics_bridge` と `tests/analytic_oracle.rs` に struct literal 構築は無く、呼び出し側の変更は不要だった 実測: `cargo check --features physics --all-targets` が 0 error (lock の解決は 2.0.0)、`scripts/preflight.sh` (full) が exit 0
- CI: 公開する文面の検査 (`Public text` workflow、`scripts/public_text.sh`) を追加 ALICE-Law の `scripts/public_text_check.py` を `.github/public-text-pin` の revision で取得し、SHA-256 が一致した時だけ使って、HEAD の全 tracked file (path と本文) と送った commit (message・追加行・追加 path) を読む 何も比較できなければ失敗する preflight でも走る

### Fixed
- Release profile no longer sets `panic = "abort"`: PyO3 turns a Rust panic into a Python `PanicException` only when the panic can unwind, so with `abort` any panic inside a `libasp` call (a mismatched NumPy shape, an index past the frame) terminated the interpreter instead of raising

### Added
- CI: 他の crate の `PREVIOUS_*_SEMANTICS_IDS` は一覧全体としてだけ使える (`contains` / `len` / `is_empty`、全体を回す iterator や for、引数として丸ごと渡す) とし、添字・slice のパターン・`.first` / `next` / `take` などを拒む `scripts/upstream_previous_ids_check.py` と試験を足した (ALICE-Edge と同じ中身の共有 file) 上流は major で一覧を空にする 走査した .rs が 0 件なら失敗 (ci.yml と preflight)
- `flatbuffers_api::encode_payload` omits empty vectors (regions / motion vectors / region deltas / corrections): a D-Packet without deltas is 16 bytes smaller and byte-identical to `PacketEncoder::encode_d_packet`; the reader already treated an absent vector as empty
- `tests/wire_roundtrip.rs`: `PacketEncoder` output parses to the same packet as `to_bytes` (D / I / S), payload helpers (`correction_count`, `total_correction_bytes`, `add_region_delta`, `dominant_color`, `as_c_packet` / `as_s_packet`), `estimated_size` per-item allowances, bincode path length / header / CRC validation
- `.cargo/mutants.toml`: 7 documented `exclude_re` classes for output-equivalent mutants in `codec/motion.rs` (SIMD dispatch arms, symmetric pattern constants, the never-reached iteration budget, blocks outside the frame)
- `video_codec` unit tests: symbol mapping law (exact below 256 levels, error < step + ½ above), `CompressedFrame` wire layout round trip with every prefix and inflated histogram length rejected
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
