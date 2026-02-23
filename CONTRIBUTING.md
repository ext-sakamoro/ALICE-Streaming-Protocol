# Contributing to ALICE-Streaming-Protocol

## Build

```bash
cargo build --no-default-features
cargo build                          # includes Python bindings
cargo build --features media-stack   # codec + voice
```

## Test

```bash
cargo test --no-default-features
```

Note: Default `python` feature requires a compatible Python environment for linking.

## Lint

```bash
cargo clippy --no-default-features -- -W clippy::all
cargo fmt -- --check
cargo doc --no-default-features --no-deps 2>&1 | grep warning
```

## Design Constraints

- **Procedural streaming**: 100-1000x bandwidth reduction by sending scene descriptions instead of pixels.
- **I/D/C/S packet types**: keyframe, delta, correction (ROI), and sync/control packets.
- **FlatBuffers serialization**: zero-copy cross-language support (C++, Go, Java, Python, TypeScript).
- **Parallel motion estimation**: Diamond and Hexagon search with Rayon.
- **Hybrid pipeline**: SDF background rendering + wavelet-encoded foreground.
- **CRC32**: compile-time lookup table for fast packet integrity checks.
