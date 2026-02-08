<p align="center">
  <img src="assets/logo-on-dark.jpg" alt="ALICE Streaming Protocol" width="600">
</p>

# ALICE Streaming Protocol (libasp)

High-performance video streaming codec written in Rust with Python bindings.

**A.L.I.C.E.** = Adaptive Low-bandwidth Image Codec Engine

## Features

- **Ultra-low bandwidth**: 100-1000x reduction compared to traditional codecs
- **Procedural generation**: Sends mathematical descriptions instead of pixels
- **Cross-language**: FlatBuffers serialization (C++, Go, Java, Python, TypeScript, etc.)
- **Zero-copy**: Direct buffer access without deserialization
- **Parallel processing**: Leverages Rayon for multi-core performance
- **Zero-copy Python bindings**: Direct NumPy array access via PyO3
- **Bare-metal performance**: Compile-time CRC32 tables, GIL release, buffer reuse

## Hybrid Streaming (SDF + Wavelet Person)

ALICE's hybrid streaming mode separates **SDF-rendered backgrounds** from **wavelet-encoded person video**, achieving massive bandwidth reduction:

```
Traditional:  Full Frame pixels   → H.265 → 5-10 Mbps
ALICE Hybrid: SDF background (2-10 KB) + Person wavelet (0.5-2 Mbps)
              → ~80-95% bandwidth savings
```

```
┌─────────────── Transmitter ───────────────┐
│                                            │
│  Camera → Segmentation → Person Mask       │
│              ↓               ↓             │
│  SDF Scene (2-10KB)   Person Crop          │
│       ↓                    ↓               │
│  I-Packet.sdf_scene   Wavelet Encode       │
│  D-Packet.sdf_delta   C-Packet (face ROI)  │
│              ↓               ↓             │
│          ASP Multiplexer → Network         │
└────────────────────────────────────────────┘

┌─────────────── Receiver ──────────────────┐
│                                            │
│  Network → ASP Demultiplexer               │
│              ↓               ↓             │
│  SDF Scene              Person Stream      │
│       ↓                    ↓               │
│  SDF Render (GPU)     Wavelet Decode       │
│       ↓                    ↓               │
│  Background Frame     Person Pixels + Mask │
│              ↓               ↓             │
│          Compositor → Display              │
└────────────────────────────────────────────┘
```

| Component | Traditional (H.265) | ALICE Hybrid |
|-----------|--------------------:|-------------:|
| Background (70-80%) | 3.5-8 Mbps | **2-10 KB** (SDF, once) |
| Person (20-30%) | 1-2 Mbps | **0.5-2 Mbps** (wavelet) |
| Mask | N/A | **~300 bytes** (RLE) |
| **Total** | **5-10 Mbps** | **~0.5-2 Mbps** |

## Media Stack (Codec + Voice Integration)

With the `media-stack` feature, libasp integrates [ALICE-Codec](https://github.com/ext-sakamoro/ALICE-Codec) and [ALICE-Voice](https://github.com/ext-sakamoro/ALICE-Voice) for end-to-end media encoding/decoding within the ASP transport layer.

```toml
[dependencies]
libasp = { version = "1.0", features = ["media-stack"] }
# Or individually:
# libasp = { version = "1.0", features = ["codec"] }   # Video only
# libasp = { version = "1.0", features = ["voice"] }   # Voice only
```

### Video Codec Pipeline

```
RGB → YCoCg-R → 2D CDF 9/7 Wavelet → Quantize → rANS → Bitstream
                                                          ↓
Bitstream → rANS Decode → Dequantize → Inverse Wavelet → YCoCg-R⁻¹ → RGB
```

- **3-channel Rayon parallel**: Y/Co/Cg planes encoded simultaneously via `rayon::join`
- **Quality control**: `quality` parameter (0-100) maps to quantization step
- **rANS entropy coding**: Near-Shannon-limit compression

```rust
use libasp::media::{VideoCodec, VideoCodecConfig};

let config = VideoCodecConfig { quality: 75, ..Default::default() };
let mut codec = VideoCodec::new(config);

// Encode (Rayon parallel 3-channel)
let compressed = codec.encode_frame(&rgb_data, width, height)?;

// Decode
let reconstructed = codec.decode_frame(&compressed)?;
```

### Voice Codec Pipeline

```
PCM f32 → LPC Analysis → Parametric/Spectral Params → Serialized AudioFrame
                                                        ↓
AudioFrame → Deserialize → LPC Synthesis → PCM f32
```

- **L1 Spectral**: 10-50x compression (high quality)
- **L2 Parametric**: 100-600x compression (voice-optimized)
- **Batch API**: `encode_batch()` for processing multiple frames

```rust
use libasp::media::{AudioCodec, AudioCodecConfig};

let config = AudioCodecConfig::default(); // 16kHz, LPC order 10
let mut codec = AudioCodec::new(config);

// Encode voice frame
let frame = codec.encode_parametric(&audio_samples, timestamp)?;

// Batch encode
let frames = codec.encode_batch(&frame_slices, &timestamps);

// Decode
let reconstructed = codec.decode(&frame)?;
```

### Python Media Bindings (GIL Release + Zero-Copy)

```python
import numpy as np
import libasp

# Video: encode/decode with NumPy zero-copy
rgb = np.array(frame_data, dtype=np.uint8)
compressed = libasp.encode_video_frame(rgb, width, height, quality=75)
reconstructed = libasp.decode_video_frame(compressed)

# Voice: encode/decode
audio = np.array(samples, dtype=np.float32)
frame = libasp.encode_voice(audio, sample_rate=16000)
decoded = libasp.decode_voice(frame)
```

All media Python bindings use `py.allow_threads` for full GIL release during computation.

## Performance Highlights

| Optimization | Improvement |
|-------------|-------------|
| CRC32 lookup table (compile-time) | 2.8-5.3x faster |
| Buffer reuse (streaming) | 77x faster |
| GIL release for heavy computation | Full Python parallelism |
| Direct PyArray allocation | Zero intermediate copies |
| Separable morphology (segmentation) | O(n) vs O(n×r²) |
| Scan-forward RLE encode/decode | Pre-allocate zeros, skip 70-80% writes |
| Rayon parallel 3-channel video encode/decode | 3x throughput (Y/Co/Cg) |
| Voice batch API (`encode_batch`) | FFI amortization |

## Packet Types

| Type | Name | Description | Typical Size |
|------|------|-------------|--------------|
| I | Keyframe | Full procedural description | 10-100KB (8K) |
| D | Delta | Incremental updates + motion vectors | 1-10KB |
| C | Correction | ROI-based pixel corrections | Variable |
| S | Sync | Flow control commands | < 100 bytes |

## Cross-Language Support

ASP uses FlatBuffers for serialization, enabling zero-copy access from any language.

**Generate code for your language:**

```bash
# Install FlatBuffers compiler
brew install flatbuffers  # macOS
apt install flatbuffers-compiler  # Ubuntu

# Generate code
flatc --cpp -o generated/ schemas/asp.fbs     # C++
flatc --go -o generated/ schemas/asp.fbs      # Go
flatc --java -o generated/ schemas/asp.fbs    # Java
flatc --ts -o generated/ schemas/asp.fbs      # TypeScript
flatc --python -o generated/ schemas/asp.fbs  # Python
```

**C++ Example:**

```cpp
#include "asp_generated.h"

// Read D-Packet (zero-copy!)
auto packet = Alice::Asp::GetAspPacketPayload(buffer);
auto d_packet = packet->payload_as_DPacketPayload();
auto mvs = d_packet->motion_vectors();

for (auto mv : *mvs) {
    printf("Block (%d, %d): dx=%d, dy=%d\n",
           mv->block_x(), mv->block_y(), mv->dx(), mv->dy());
}
```

## Installation

### From Source (Rust)

```bash
git clone https://github.com/ext-sakamoro/ALICE-Streaming-Protocol.git
cd ALICE-Streaming-Protocol
cargo build --release
```

### Python Package

```bash
cd ALICE-Streaming-Protocol
pip install maturin
maturin develop --release
```

Note: For Python 3.14+, set `PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1` before building.

## Usage

### Rust

```rust
use libasp::{AspPacket, IPacketPayload, DPacketPayload, MotionVector, codec};

// Create keyframe
let payload = IPacketPayload::new(1920, 1080, 30.0);
let packet = AspPacket::create_i_packet(1, payload)?;

// Serialize to reusable buffer (zero-allocation in hot loop)
let mut buffer = Vec::with_capacity(65536);
packet.write_to_buffer(&mut buffer)?;

// Or use fixed stack buffer (true zero-allocation)
let mut stack_buffer = [0u8; 65536];
let written = packet.serialize_into(&mut stack_buffer)?;

// Motion estimation with parallel Diamond Search
let mvs = codec::estimate_motion_parallel(
    &current_frame,
    &previous_frame,
    1920, 1080,
    16, 16,  // block_size, search_range
    codec::SearchAlgorithm::DiamondSearch,
    256,     // early termination threshold
);

// Create delta packet with motion vectors
let mut d_payload = DPacketPayload::new(1);
for mv in mvs {
    d_payload.add_motion_vector(mv);
}
let d_packet = AspPacket::create_d_packet(2, d_payload)?;
```

### Python (NumPy Zero-Copy)

```python
import numpy as np
import libasp

# Load frames as NumPy arrays (H, W) uint8
current = np.array(current_frame, dtype=np.uint8)
previous = np.array(previous_frame, dtype=np.uint8)

# Motion estimation with NumPy zero-copy I/O
# Returns (N, 5) int32 array: [block_x, block_y, dx, dy, sad]
mvs = libasp.estimate_motion_numpy(current, previous, block_size=16, search_range=16)

# Or use the MotionEstimator class for repeated calls
estimator = libasp.MotionEstimator(block_size=16, search_range=16, algorithm="diamond")
mvs = estimator.estimate_numpy(current, previous)

# Color extraction (k-means, returns (N, 3) uint8 array)
colors = libasp.extract_colors(pixels, num_colors=5)

# With weights
colors, weights = libasp.extract_colors_with_weights(pixels, num_colors=5)

# DCT transform
dct = libasp.DctTransform(block_size=8, quality=75)
coefficients = dct.forward(block)
reconstructed = dct.inverse(coefficients)

# ROI detection (returns (N, 6) float32 array: [x, y, w, h, type, confidence])
regions = libasp.detect_roi(frame, previous_frame, width, height)

# Create packets
i_packet = libasp.create_i_packet(sequence=1, width=1920, height=1080, fps=30.0)
d_packet = libasp.create_d_packet_numpy(sequence=2, ref_sequence=1, motion_vectors=mvs)

# Parse packet
packet_type, sequence, payload_size = libasp.parse_packet(packet_bytes)
```

### Python Hybrid Streaming API

```python
import numpy as np
import libasp

# --- Transmitter Side ---
tx = libasp.HybridTransmitter()

# Create keyframe with SDF scene + person video
seq, is_key, video_size = tx.create_keyframe(
    width=1920, height=1080, fps=30.0,
    asdf_data=sdf_binary_blob,       # ASDF scene description (~5 KB)
    person_video=wavelet_encoded,     # Wavelet-encoded person region
    person_mask_rle=rle_mask_bytes,   # RLE-compressed person mask
    person_bbox=[300, 100, 400, 600], # [x, y, w, h]
    foreground_pixels=120000,
)

# Create delta frames (person-only, no SDF update needed)
for i in range(29):
    seq, is_key, video_size = tx.create_delta_frame(
        person_video=next_person_frame,
        timestamp_ms=i * 33,
    )

# Check bandwidth savings
savings_pct, compression_ratio = tx.savings()
print(tx.bandwidth_report())

# --- Receiver Side ---
rx = libasp.HybridReceiver()
render_sdf, scene_ver, bbox, video_size = rx.process_keyframe(
    sequence=1, width=1920, height=1080,
    person_video_size=30000, has_scene=True,
)

# --- RLE Mask Utilities ---
mask = np.zeros((1080, 1920), dtype=np.uint8)
mask[100:700, 300:700] = 1
rle = libasp.rle_encode_mask_numpy(mask, 1920, 1080)
decoded = libasp.rle_decode_mask_numpy(rle, 1920, 1080)  # (H, W) NumPy

# --- Bandwidth Estimation ---
savings, ratio = libasp.estimate_hybrid_savings(
    frame_width=1920, frame_height=1080,
    person_coverage=0.20,    # 20% of frame is person
    sdf_scene_bytes=5000,    # 5 KB SDF scene
    wavelet_bpp=1.0,         # 1 bit per pixel wavelet
)
print(f"Savings: {savings:.1f}%, Ratio: {ratio:.1f}x")
```

## Benchmarks

Run benchmarks:

```bash
cargo bench --no-default-features
```

### Motion Estimation (Apple Silicon)

| Resolution | Algorithm | Time |
|------------|-----------|------|
| 256x256 | Diamond Search | 19.6 µs |
| 512x512 | Diamond Search | 24.5 µs |
| 1024x1024 | Diamond Search | 42.0 µs |
| **1080p** | Diamond Search | **64.5 µs** |

### CRC32 Performance

Compile-time generated lookup table provides 2.8-5.3x speedup over table-less implementation.

### Test Suite

```bash
cargo test --no-default-features
# running 88 tests
# test result: ok. 88 passed; 0 failed
```

## Architecture

```
libasp/
├── src/
│   ├── lib.rs          # Library root, version info
│   ├── types.rs        # Core types (MotionVector, Color, Rect, etc.)
│   ├── header.rs       # Packet header (16 bytes) + CRC32
│   ├── packet.rs       # Packet payloads (I/D/C/S) + serialization
│   ├── codec/
│   │   ├── mod.rs      # Codec module exports
│   │   ├── motion.rs   # Motion estimation (Diamond/Hexagon/TSS/Full)
│   │   ├── dct.rs      # DCT/IDCT transform + quantization
│   │   ├── color.rs    # Color extraction (k-means, median cut)
│   │   └── roi.rs      # ROI detection (edge, motion, face)
│   ├── scene.rs        # SDF scene channel (descriptor, delta, person mask, RLE)
│   ├── hybrid.rs       # Hybrid streaming pipeline (SDF + wavelet person)
│   ├── media/
│   │   ├── mod.rs      # Media stack module (feature-gated)
│   │   ├── video_codec.rs  # Video codec (alice-codec wrapper, Rayon parallel)
│   │   └── voice_codec.rs  # Voice codec (alice-voice wrapper, batch API)
│   └── python.rs       # PyO3 + NumPy bindings (motion, color, DCT, ROI, hybrid, media)
├── benches/
│   └── motion_estimation.rs  # Criterion benchmarks
└── Cargo.toml
```

## Key Data Structures

### MotionVector (16 bytes, cache-optimized)

```rust
#[repr(C)]
pub struct MotionVector {
    pub block_x: u16,    // Block X position (supports 4K+)
    pub block_y: u16,    // Block Y position
    pub dx: i16,         // Horizontal displacement
    pub dy: i16,         // Vertical displacement
    pub sad: u32,        // Sum of Absolute Differences
    pub _reserved: u32,  // Future use / alignment
}
```

### MotionVectorCompact (2 bytes, for transmission)

```rust
#[repr(C, packed)]
pub struct MotionVectorCompact {
    pub dx: i8,  // -128 to 127 pixels
    pub dy: i8,
}
```

## Features Flags

```toml
[features]
default = ["python"]
python = ["pyo3", "numpy"]        # Python bindings
wasm = ["wasm-bindgen"]           # WebAssembly support
simd = []                         # Explicit SIMD (auto-detected)
codec = ["alice-codec"]           # Video codec (3D wavelet + rANS)
voice = ["alice-voice"]           # Voice codec (LPC parametric)
media-stack = ["codec", "voice"]  # Full media stack
```

## License

**Dual License Model** - See [LICENSE](LICENSE) for full details.

| Component | License | Purpose |
|-----------|---------|---------|
| Protocol Specification | MIT | Global standard adoption |
| Receiver (Decoder) | MIT | Run on every device |
| Sender (Encoder Core) | MIT | Democratize ultra-low bandwidth |
| Sender (Enterprise) | Commercial | High-scale infrastructure |

**Attribution**: Free encoder users must include "Streamed via ALICE Protocol (ASP)" in stream metadata.

## Related Projects

| Project | Description |
|---------|-------------|
| [ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) | Core procedural generation engine |
| [ALICE-DB](https://github.com/ext-sakamoro/ALICE-DB) | Model-based time-series database |
| [ALICE-Edge](https://github.com/ext-sakamoro/ALICE-Edge) | Embedded/IoT model generator (no_std) |
| [ALICE-Eco-System](https://github.com/ext-sakamoro/ALICE-Eco-System) | Complete Edge-to-Cloud pipeline demo |

All projects share the core philosophy: **encode the generation process, not the data itself**.

## Author

Moroya Sakamoto

---

[日本語版 README](README_ja.md)
