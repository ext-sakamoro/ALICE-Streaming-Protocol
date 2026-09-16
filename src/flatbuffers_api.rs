//! FlatBuffers API for ASP Protocol
//!
//! This module provides a high-level, ergonomic API for FlatBuffers serialization.
//! It wraps the auto-generated FlatBuffers code with idiomatic Rust interfaces.
//!
//! # Zero-Copy Design
//!
//! FlatBuffers enables true zero-copy access to serialized data. Unlike traditional
//! serialization formats (JSON, Protobuf), FlatBuffers allows direct access to
//! fields without parsing or copying.
//!
//! # Cross-Language Compatibility
//!
//! Data serialized with this module can be read by any language with FlatBuffers
//! support (C++, Go, Java, Python, TypeScript, etc.) using the same schema.
//!
//! # Example
//!
//! ```rust,no_run
//! use libasp::flatbuffers_api::{DPacketBuilder, read_d_packet};
//! use libasp::MotionVector;
//!
//! // Build a D-Packet
//! let mvs = vec![
//!     MotionVector::new(0, 0, 5, -3, 100),
//!     MotionVector::new(1, 0, 2, 1, 50),
//! ];
//! let bytes = DPacketBuilder::new(1)
//!     .motion_vectors(&mvs)
//!     .timestamp_ms(12345)
//!     .build();
//!
//! // Read the packet (zero-copy access!)
//! let packet = read_d_packet(&bytes).unwrap();
//! let ref_seq = packet.ref_sequence();
//! let motion_vectors = packet.motion_vectors().unwrap();
//! for mv in motion_vectors.iter() {
//!     println!("Block ({}, {}): dx={}, dy={}", mv.block_x(), mv.block_y(), mv.dx(), mv.dy());
//! }
//! ```

use crate::generated;
use crate::types::{Color as RustColor, MotionVector as RustMotionVector, Rect as RustRect};
use flatbuffers::FlatBufferBuilder;

// Re-export FlatBuffers types for convenience
pub use generated::{
    AspPacketPayload,
    CPacketPayload,
    // Structs (zero-copy accessible)
    Color as FbColor,
    CompressionFormat as FbCompressionFormat,
    DPacketPayload,
    DctCoefficient as FbDctCoefficient,

    EasingType as FbEasingType,

    // Tables (need accessor methods)
    IPacketPayload,
    MotionType as FbMotionType,
    MotionVector as FbMotionVector,
    MotionVectorCompact as FbMotionVectorCompact,
    PacketType as FbPacketType,
    PatternType as FbPatternType,
    Point as FbPoint,
    QualityLevel as FbQualityLevel,
    Rect as FbRect,
    RoiType as FbRoiType,
    SPacketPayload,
    SyncCommand as FbSyncCommand,
};

/// Error type for `FlatBuffers` operations
#[derive(Debug, Clone, thiserror::Error)]
pub enum FlatBuffersError {
    /// Invalid buffer format
    #[error("Invalid FlatBuffers format: {0}")]
    InvalidFormat(String),

    /// Missing required field
    #[error("Missing required field: {0}")]
    MissingField(&'static str),

    /// Buffer too small
    #[error("Buffer too small: need {needed}, got {got}")]
    BufferTooSmall {
        /// Required buffer size
        needed: usize,
        /// Actual buffer size
        got: usize,
    },
}

/// Result type for `FlatBuffers` operations
pub type FbResult<T> = Result<T, FlatBuffersError>;

// =============================================================================
// Type Conversions
// =============================================================================

/// Convert Rust `MotionVector` to `FlatBuffers` `MotionVector`
#[inline]
#[must_use]
pub fn motion_vector_to_fb(mv: &RustMotionVector) -> FbMotionVector {
    FbMotionVector::new(mv.block_x, mv.block_y, mv.dx, mv.dy, mv.sad)
}

/// Convert `FlatBuffers` `MotionVector` to Rust `MotionVector`
#[inline]
#[must_use]
pub fn motion_vector_from_fb(mv: &FbMotionVector) -> RustMotionVector {
    RustMotionVector::new(mv.block_x(), mv.block_y(), mv.dx(), mv.dy(), mv.sad())
}

/// Convert Rust Color to `FlatBuffers` Color
#[inline]
#[must_use]
pub fn color_to_fb(c: &RustColor) -> FbColor {
    FbColor::new(c.r, c.g, c.b)
}

/// Convert `FlatBuffers` Color to Rust Color
#[inline]
#[must_use]
pub fn color_from_fb(c: &FbColor) -> RustColor {
    RustColor::new(c.r(), c.g(), c.b())
}

/// Convert Rust Rect to `FlatBuffers` Rect
#[inline]
#[must_use]
pub fn rect_to_fb(r: &RustRect) -> FbRect {
    FbRect::new(r.x, r.y, r.width, r.height)
}

/// Convert `FlatBuffers` Rect to Rust Rect
#[inline]
#[must_use]
pub fn rect_from_fb(r: &FbRect) -> RustRect {
    RustRect::new(r.x(), r.y(), r.width(), r.height())
}

// =============================================================================
// D-Packet Builder (Most common use case for motion vectors)
// =============================================================================

/// Builder for D-Packet (Delta packet with motion vectors)
///
/// This is the most performance-critical packet type for streaming.
/// Motion vectors are stored as structs for zero-copy access.
///
/// # Example
///
/// ```rust
/// use libasp::flatbuffers_api::{read_d_packet, DPacketBuilder};
/// use libasp::types::MotionVector;
///
/// let mvs = [MotionVector::new(0, 0, 3, -2, 10), MotionVector::new(1, 0, 0, 1, 4)];
/// let bytes = DPacketBuilder::new(1)
///     .motion_vectors(&mvs)
///     .timestamp_ms(12345)
///     .build();
/// let d = read_d_packet(&bytes).unwrap();
/// assert_eq!(d.motion_vectors().unwrap().len(), 2);
/// ```
pub struct DPacketBuilder<'a> {
    builder: FlatBufferBuilder<'a>,
    ref_sequence: u32,
    motion_vectors: Option<Vec<FbMotionVector>>,
    compact_vectors: Option<Vec<FbMotionVectorCompact>>,
    timestamp_ms: u64,
}

impl DPacketBuilder<'_> {
    /// Create a new D-Packet builder
    #[must_use]
    pub fn new(ref_sequence: u32) -> Self {
        Self {
            builder: FlatBufferBuilder::with_capacity(1024),
            ref_sequence,
            motion_vectors: None,
            compact_vectors: None,
            timestamp_ms: 0,
        }
    }

    /// Create with pre-allocated capacity
    #[must_use]
    pub fn with_capacity(ref_sequence: u32, capacity: usize) -> Self {
        Self {
            builder: FlatBufferBuilder::with_capacity(capacity),
            ref_sequence,
            motion_vectors: None,
            compact_vectors: None,
            timestamp_ms: 0,
        }
    }

    /// Add motion vectors (full format, 12 bytes each)
    #[must_use]
    pub fn motion_vectors(mut self, mvs: &[RustMotionVector]) -> Self {
        self.motion_vectors = Some(mvs.iter().map(motion_vector_to_fb).collect());
        self
    }

    /// Add motion vectors (compact format, 2 bytes each)
    ///
    /// Use this for bandwidth-critical scenarios where block position
    /// is implicit from array index.
    #[must_use]
    pub fn compact_vectors(mut self, mvs: &[(i8, i8)]) -> Self {
        self.compact_vectors = Some(
            mvs.iter()
                .map(|(dx, dy)| FbMotionVectorCompact::new(*dx, *dy))
                .collect(),
        );
        self
    }

    /// Set timestamp in milliseconds
    #[must_use]
    pub const fn timestamp_ms(mut self, ts: u64) -> Self {
        self.timestamp_ms = ts;
        self
    }

    /// Build the packet and return serialized bytes
    #[must_use]
    pub fn build(mut self) -> Vec<u8> {
        // Create motion vectors vector
        let mvs_offset = self
            .motion_vectors
            .as_ref()
            .map(|mvs| self.builder.create_vector(mvs));

        let compact_offset = self
            .compact_vectors
            .as_ref()
            .map(|mvs| self.builder.create_vector(mvs));

        // Build DPacketPayload
        let d_packet = generated::DPacketPayload::create(
            &mut self.builder,
            &generated::DPacketPayloadArgs {
                ref_sequence: self.ref_sequence,
                motion_vectors: mvs_offset,
                motion_vectors_compact: compact_offset,
                global_motion: None,
                region_deltas: None,
                timestamp_ms: self.timestamp_ms,
            },
        );

        // Wrap in AspPacketPayload union
        let asp_packet = generated::AspPacketPayload::create(
            &mut self.builder,
            &generated::AspPacketPayloadArgs {
                payload_type: generated::AspPayloadUnion::DPacketPayload,
                payload: Some(d_packet.as_union_value()),
            },
        );

        self.builder.finish(asp_packet, Some("ASP1"));
        self.builder.finished_data().to_vec()
    }

    /// Build directly into an existing buffer (zero-allocation)
    ///
    /// Returns the number of bytes written, or an error if the buffer is too small.
    ///
    /// # Errors
    ///
    /// Returns `FlatBuffersError::BufferTooSmall` if `buffer` is smaller than the serialized data.
    pub fn build_into(mut self, buffer: &mut [u8]) -> FbResult<usize> {
        // Create motion vectors vector
        let mvs_offset = self
            .motion_vectors
            .as_ref()
            .map(|mvs| self.builder.create_vector(mvs));

        let compact_offset = self
            .compact_vectors
            .as_ref()
            .map(|mvs| self.builder.create_vector(mvs));

        // Build DPacketPayload
        let d_packet = generated::DPacketPayload::create(
            &mut self.builder,
            &generated::DPacketPayloadArgs {
                ref_sequence: self.ref_sequence,
                motion_vectors: mvs_offset,
                motion_vectors_compact: compact_offset,
                global_motion: None,
                region_deltas: None,
                timestamp_ms: self.timestamp_ms,
            },
        );

        // Wrap in AspPacketPayload union
        let asp_packet = generated::AspPacketPayload::create(
            &mut self.builder,
            &generated::AspPacketPayloadArgs {
                payload_type: generated::AspPayloadUnion::DPacketPayload,
                payload: Some(d_packet.as_union_value()),
            },
        );

        self.builder.finish(asp_packet, Some("ASP1"));

        let data = self.builder.finished_data();
        if buffer.len() < data.len() {
            return Err(FlatBuffersError::BufferTooSmall {
                needed: data.len(),
                got: buffer.len(),
            });
        }

        buffer[..data.len()].copy_from_slice(data);
        Ok(data.len())
    }
}

// =============================================================================
// I-Packet Builder
// =============================================================================

/// Builder for I-Packet (Keyframe)
pub struct IPacketBuilder<'a> {
    builder: FlatBufferBuilder<'a>,
    width: u32,
    height: u32,
    fps: f32,
    quality: FbQualityLevel,
    timestamp_ms: u64,
}

impl IPacketBuilder<'_> {
    /// Create a new I-Packet builder
    #[must_use]
    pub fn new(width: u32, height: u32, fps: f32) -> Self {
        Self {
            builder: FlatBufferBuilder::with_capacity(4096),
            width,
            height,
            fps,
            quality: FbQualityLevel::Medium,
            timestamp_ms: 0,
        }
    }

    /// Set quality level
    #[must_use]
    pub const fn quality(mut self, q: FbQualityLevel) -> Self {
        self.quality = q;
        self
    }

    /// Set timestamp in milliseconds
    #[must_use]
    pub const fn timestamp_ms(mut self, ts: u64) -> Self {
        self.timestamp_ms = ts;
        self
    }

    /// Build the packet and return serialized bytes
    #[must_use]
    pub fn build(mut self) -> Vec<u8> {
        // Build IPacketPayload
        let i_packet = generated::IPacketPayload::create(
            &mut self.builder,
            &generated::IPacketPayloadArgs {
                width: self.width,
                height: self.height,
                fps: self.fps,
                quality: self.quality,
                global_palette: None,
                regions: None,
                animation: None,
                timestamp_ms: self.timestamp_ms,
            },
        );

        // Wrap in AspPacketPayload union
        let asp_packet = generated::AspPacketPayload::create(
            &mut self.builder,
            &generated::AspPacketPayloadArgs {
                payload_type: generated::AspPayloadUnion::IPacketPayload,
                payload: Some(i_packet.as_union_value()),
            },
        );

        self.builder.finish(asp_packet, Some("ASP1"));
        self.builder.finished_data().to_vec()
    }
}

// =============================================================================
// S-Packet Builder (Sync/Control)
// =============================================================================

/// Builder for S-Packet (Sync/Control)
pub struct SPacketBuilder<'a> {
    builder: FlatBufferBuilder<'a>,
    command: FbSyncCommand,
    timestamp_ms: u64,
}

impl SPacketBuilder<'_> {
    /// Create a new S-Packet builder
    #[must_use]
    pub fn new(command: FbSyncCommand) -> Self {
        Self {
            builder: FlatBufferBuilder::with_capacity(128),
            command,
            timestamp_ms: 0,
        }
    }

    /// Create a Ping packet
    #[must_use]
    pub fn ping() -> Self {
        Self::new(FbSyncCommand::Ping)
    }

    /// Create a Pong packet
    #[must_use]
    pub fn pong() -> Self {
        Self::new(FbSyncCommand::Pong)
    }

    /// Create a `RequestKeyframe` packet
    #[must_use]
    pub fn request_keyframe() -> Self {
        Self::new(FbSyncCommand::RequestKeyframe)
    }

    /// Create an `EndOfStream` packet
    #[must_use]
    pub fn end_of_stream() -> Self {
        Self::new(FbSyncCommand::EndOfStream)
    }

    /// Set timestamp in milliseconds
    #[must_use]
    pub const fn timestamp_ms(mut self, ts: u64) -> Self {
        self.timestamp_ms = ts;
        self
    }

    /// Build the packet and return serialized bytes
    #[must_use]
    pub fn build(mut self) -> Vec<u8> {
        // Build SPacketPayload
        let s_packet = generated::SPacketPayload::create(
            &mut self.builder,
            &generated::SPacketPayloadArgs {
                command: self.command,
                data_type: generated::SyncDataUnion::NONE,
                data: None,
                timestamp_ms: self.timestamp_ms,
            },
        );

        // Wrap in AspPacketPayload union
        let asp_packet = generated::AspPacketPayload::create(
            &mut self.builder,
            &generated::AspPacketPayloadArgs {
                payload_type: generated::AspPayloadUnion::SPacketPayload,
                payload: Some(s_packet.as_union_value()),
            },
        );

        self.builder.finish(asp_packet, Some("ASP1"));
        self.builder.finished_data().to_vec()
    }
}

// =============================================================================
// Packet Readers (Zero-Copy Access)
// =============================================================================

/// Read and verify a `FlatBuffers` packet
///
/// Returns the root `AspPacketPayload` table for zero-copy access.
///
/// # Errors
///
/// Returns `FlatBuffersError::InvalidFormat` if the buffer is too small or contains invalid data.
pub fn read_packet(bytes: &[u8]) -> FbResult<AspPacketPayload<'_>> {
    // Verify file identifier
    if bytes.len() < 8 {
        return Err(FlatBuffersError::InvalidFormat(
            "Buffer too small".to_string(),
        ));
    }

    // Get the root with verification
    let packet = flatbuffers::root::<AspPacketPayload>(bytes)
        .map_err(|e| FlatBuffersError::InvalidFormat(e.to_string()))?;

    Ok(packet)
}

/// Read a D-Packet from bytes (zero-copy)
///
/// # Example
///
/// ```rust
/// use libasp::flatbuffers_api::{create_d_packet, read_d_packet};
/// use libasp::types::MotionVector;
///
/// let bytes = create_d_packet(7, &[MotionVector::new(2, 3, -1, 4, 9)], 0);
/// let d_packet = read_d_packet(&bytes).unwrap();
/// let mvs = d_packet.motion_vectors().unwrap();
/// for mv in mvs.iter() {
///     assert_eq!((mv.dx(), mv.dy()), (-1, 4));
/// }
/// ```
///
/// # Errors
///
/// Returns `FlatBuffersError::InvalidFormat` if bytes are invalid, or `FlatBuffersError::MissingField`
/// if the payload type does not match `DPacketPayload`.
pub fn read_d_packet(bytes: &[u8]) -> FbResult<DPacketPayload<'_>> {
    let packet = read_packet(bytes)?;

    if packet.payload_type() != generated::AspPayloadUnion::DPacketPayload {
        return Err(FlatBuffersError::InvalidFormat(format!(
            "Expected DPacketPayload, got {:?}",
            packet.payload_type()
        )));
    }

    packet
        .payload_as_dpacket_payload()
        .ok_or(FlatBuffersError::MissingField("payload"))
}

/// Read an I-Packet from bytes (zero-copy)
///
/// # Errors
///
/// Returns `FlatBuffersError::InvalidFormat` if bytes are invalid, or `FlatBuffersError::MissingField`
/// if the payload type does not match `IPacketPayload`.
pub fn read_i_packet(bytes: &[u8]) -> FbResult<IPacketPayload<'_>> {
    let packet = read_packet(bytes)?;

    if packet.payload_type() != generated::AspPayloadUnion::IPacketPayload {
        return Err(FlatBuffersError::InvalidFormat(format!(
            "Expected IPacketPayload, got {:?}",
            packet.payload_type()
        )));
    }

    packet
        .payload_as_ipacket_payload()
        .ok_or(FlatBuffersError::MissingField("payload"))
}

/// Read an S-Packet from bytes (zero-copy)
///
/// # Errors
///
/// Returns `FlatBuffersError::InvalidFormat` if bytes are invalid, or `FlatBuffersError::MissingField`
/// if the payload type does not match `SPacketPayload`.
pub fn read_s_packet(bytes: &[u8]) -> FbResult<SPacketPayload<'_>> {
    let packet = read_packet(bytes)?;

    if packet.payload_type() != generated::AspPayloadUnion::SPacketPayload {
        return Err(FlatBuffersError::InvalidFormat(format!(
            "Expected SPacketPayload, got {:?}",
            packet.payload_type()
        )));
    }

    packet
        .payload_as_spacket_payload()
        .ok_or(FlatBuffersError::MissingField("payload"))
}

/// Get packet type without full parsing
///
/// # Errors
///
/// Returns `FlatBuffersError::InvalidFormat` if the bytes cannot be parsed.
pub fn get_packet_type(bytes: &[u8]) -> FbResult<generated::AspPayloadUnion> {
    let packet = read_packet(bytes)?;
    Ok(packet.payload_type())
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Create a D-Packet with motion vectors (convenience function)
#[must_use]
pub fn create_d_packet(ref_sequence: u32, mvs: &[RustMotionVector], timestamp_ms: u64) -> Vec<u8> {
    DPacketBuilder::new(ref_sequence)
        .motion_vectors(mvs)
        .timestamp_ms(timestamp_ms)
        .build()
}

/// Create an I-Packet (convenience function)
#[must_use]
pub fn create_i_packet(width: u32, height: u32, fps: f32, timestamp_ms: u64) -> Vec<u8> {
    IPacketBuilder::new(width, height, fps)
        .timestamp_ms(timestamp_ms)
        .build()
}

/// Create a Ping packet (convenience function)
#[must_use]
pub fn create_ping(timestamp_ms: u64) -> Vec<u8> {
    SPacketBuilder::ping().timestamp_ms(timestamp_ms).build()
}

/// Create a Pong packet (convenience function)
#[must_use]
pub fn create_pong(timestamp_ms: u64) -> Vec<u8> {
    SPacketBuilder::pong().timestamp_ms(timestamp_ms).build()
}

// =============================================================================
// Full-fidelity payload codec (every field the asp.fbs schema carries)
// =============================================================================
//
// `AspPacket::to_bytes` / `from_bytes` go through these two functions. Until
// 1.1.0 the packet layer used the convenience builders above, which only
// carry width / height / fps / timestamp (I), ref_sequence + motion vectors (D)
// and Ping / Pong (S); every other field was dropped on write without an
// error, a C-Packet was written as a Ping and read back as an empty payload.
//
// Fields that have no counterpart in the schema (`sdf_scene`, `sdf_delta`,
// `person_mask` of the hybrid pipeline) are rejected with
// `AspError::SerializationError` instead of being dropped; they travel over
// the `bincode-compat` format (`write_to_buffer_bincode`).

use crate::packet::{
    AspPayload, CPacketPayload as RustCPacket, ColorPalette as RustPalette,
    CompressionFormat as RustCompression, CorrectionData as RustCorrection,
    DPacketPayload as RustDPacket, IPacketPayload as RustIPacket, RegionDelta as RustRegionDelta,
    RegionDescriptor as RustRegion, RoiRegion as RustRoi, SPacketPayload as RustSPacket, SyncData,
};
use crate::types::{
    AnimationParams as RustAnimation, AspError, EasingType as RustEasing,
    PatternType as RustPattern, QualityLevel as RustQuality, RoiType as RustRoiType,
    SyncCommand as RustSyncCommand,
};
use flatbuffers::WIPOffset;

fn ser_err(what: &str) -> AspError {
    AspError::SerializationError(what.to_string())
}

fn de_err(what: impl std::fmt::Display) -> AspError {
    AspError::DeserializationError(what.to_string())
}

fn easing_to_fb(e: RustEasing) -> FbEasingType {
    FbEasingType(e as i8)
}

fn easing_from_fb(e: FbEasingType) -> Result<RustEasing, AspError> {
    Ok(match e {
        FbEasingType::Linear => RustEasing::Linear,
        FbEasingType::EaseIn => RustEasing::EaseIn,
        FbEasingType::EaseOut => RustEasing::EaseOut,
        FbEasingType::EaseInOut => RustEasing::EaseInOut,
        FbEasingType::Bounce => RustEasing::Bounce,
        FbEasingType::Elastic => RustEasing::Elastic,
        other => return Err(de_err(format!("unknown EasingType {}", other.0))),
    })
}

fn compression_from_fb(c: FbCompressionFormat) -> Result<RustCompression, AspError> {
    Ok(match c {
        FbCompressionFormat::Raw => RustCompression::Raw,
        FbCompressionFormat::Rle => RustCompression::Rle,
        FbCompressionFormat::Lz4 => RustCompression::Lz4,
        FbCompressionFormat::Zstd => RustCompression::Zstd,
        FbCompressionFormat::DeltaRle => RustCompression::DeltaRle,
        other => return Err(de_err(format!("unknown CompressionFormat {}", other.0))),
    })
}

fn u8_of<T: Into<i8>>(v: T) -> Result<u8, AspError> {
    u8::try_from(v.into()).map_err(|_| de_err("negative enum value"))
}

fn build_palette<'a>(
    b: &mut FlatBufferBuilder<'a>,
    p: &RustPalette,
) -> WIPOffset<generated::ColorPalette<'a>> {
    let colors: Vec<FbColor> = p.colors.iter().map(color_to_fb).collect();
    let colors = b.create_vector(&colors);
    let weights = p.weights.as_ref().map(|w| b.create_vector(w));
    generated::ColorPalette::create(
        b,
        &generated::ColorPaletteArgs {
            colors: Some(colors),
            weights,
        },
    )
}

fn read_palette(p: Option<generated::ColorPalette<'_>>) -> RustPalette {
    p.map_or_else(RustPalette::default, |p| RustPalette {
        colors: p
            .colors()
            .map(|v| v.iter().map(color_from_fb).collect())
            .unwrap_or_default(),
        weights: p.weights().map(|v| v.iter().collect()),
    })
}

fn build_dct<'a>(
    b: &mut FlatBufferBuilder<'a>,
    coeffs: &[(u32, u32, f32)],
) -> Result<WIPOffset<flatbuffers::Vector<'a, FbDctCoefficient>>, AspError> {
    let mut v = Vec::with_capacity(coeffs.len());
    for &(x, y, value) in coeffs {
        let x = u16::try_from(x).map_err(|_| ser_err("DCT coefficient x exceeds u16"))?;
        let y = u16::try_from(y).map_err(|_| ser_err("DCT coefficient y exceeds u16"))?;
        v.push(FbDctCoefficient::new(x, y, value));
    }
    Ok(b.create_vector(&v))
}

fn read_dct(v: Option<flatbuffers::Vector<'_, FbDctCoefficient>>) -> Option<Vec<(u32, u32, f32)>> {
    v.map(|v| {
        v.iter()
            .map(|c| (u32::from(c.x()), u32::from(c.y()), c.value()))
            .collect()
    })
}

fn build_params<'a>(
    b: &mut FlatBufferBuilder<'a>,
    params: &[(String, f32)],
) -> WIPOffset<flatbuffers::Vector<'a, flatbuffers::ForwardsUOffset<generated::Param<'a>>>> {
    let items: Vec<_> = params
        .iter()
        .map(|(k, v)| {
            let key = b.create_string(k);
            generated::Param::create(
                b,
                &generated::ParamArgs {
                    key: Some(key),
                    value: *v,
                },
            )
        })
        .collect();
    b.create_vector(&items)
}

fn read_params(
    v: Option<flatbuffers::Vector<'_, flatbuffers::ForwardsUOffset<generated::Param<'_>>>>,
) -> Option<Vec<(String, f32)>> {
    v.map(|v| {
        v.iter()
            .map(|p| (p.key().unwrap_or_default().to_string(), p.value()))
            .collect()
    })
}

fn build_animation<'a>(
    b: &mut FlatBufferBuilder<'a>,
    a: &RustAnimation,
) -> WIPOffset<generated::AnimationParams<'a>> {
    generated::AnimationParams::create(
        b,
        &generated::AnimationParamsArgs {
            zoom_factor: a.zoom_factor,
            pan_x: a.pan_x,
            pan_y: a.pan_y,
            rotation: a.rotation,
            duration: a.duration,
            easing: easing_to_fb(a.easing),
        },
    )
}

fn read_animation(a: generated::AnimationParams<'_>) -> Result<RustAnimation, AspError> {
    Ok(RustAnimation {
        zoom_factor: a.zoom_factor(),
        pan_x: a.pan_x(),
        pan_y: a.pan_y(),
        rotation: a.rotation(),
        duration: a.duration(),
        easing: easing_from_fb(a.easing())?,
    })
}

fn build_region<'a>(
    b: &mut FlatBufferBuilder<'a>,
    r: &RustRegion,
) -> Result<WIPOffset<generated::RegionDescriptor<'a>>, AspError> {
    let palette = build_palette(b, &r.palette);
    let dct = match &r.dct_coefficients {
        Some(c) => Some(build_dct(b, c)?),
        None => None,
    };
    let params = r.params.as_ref().map(|p| build_params(b, p));
    let bounds = rect_to_fb(&r.bounds);
    Ok(generated::RegionDescriptor::create(
        b,
        &generated::RegionDescriptorArgs {
            bounds: Some(&bounds),
            pattern_type: FbPatternType(r.pattern_type as i8),
            palette: Some(palette),
            dct_coefficients: dct,
            texture_id: r.texture_id.unwrap_or(0),
            params,
        },
    ))
}

fn read_region(r: generated::RegionDescriptor<'_>) -> Result<RustRegion, AspError> {
    let pattern_type = RustPattern::try_from(u8_of(r.pattern_type().0)?)?;
    Ok(RustRegion {
        bounds: r.bounds().map(rect_from_fb).unwrap_or_default(),
        pattern_type,
        palette: read_palette(r.palette()),
        dct_coefficients: read_dct(r.dct_coefficients()),
        // 0 is the schema default: a Texture region must carry a real id, every
        // other pattern type has none
        texture_id: (pattern_type == RustPattern::Texture).then_some(r.texture_id()),
        params: read_params(r.params()),
    })
}

fn build_region_delta<'a>(
    b: &mut FlatBufferBuilder<'a>,
    d: &RustRegionDelta,
) -> Result<WIPOffset<generated::RegionDelta<'a>>, AspError> {
    let palette_delta = d.palette_delta.as_ref().map(|p| build_palette(b, p));
    let dct_delta = match &d.dct_delta {
        Some(c) => Some(build_dct(b, c)?),
        None => None,
    };
    let param_delta = d.param_delta.as_ref().map(|p| build_params(b, p));
    Ok(generated::RegionDelta::create(
        b,
        &generated::RegionDeltaArgs {
            region_index: d.region_index,
            palette_delta,
            dct_delta,
            param_delta,
        },
    ))
}

fn read_region_delta(d: generated::RegionDelta<'_>) -> RustRegionDelta {
    RustRegionDelta {
        region_index: d.region_index(),
        palette_delta: d.palette_delta().map(|p| read_palette(Some(p))),
        dct_delta: read_dct(d.dct_delta()),
        param_delta: read_params(d.param_delta()),
    }
}

fn build_roi<'a>(
    b: &mut FlatBufferBuilder<'a>,
    r: &RustRoi,
) -> WIPOffset<generated::RoiRegion<'a>> {
    let bounds = rect_to_fb(&r.bounds);
    generated::RoiRegion::create(
        b,
        &generated::RoiRegionArgs {
            bounds: Some(&bounds),
            roi_type: FbRoiType(r.roi_type as i8),
            priority: r.priority,
            confidence: r.confidence,
        },
    )
}

fn read_roi(r: generated::RoiRegion<'_>) -> Result<RustRoi, AspError> {
    Ok(RustRoi {
        bounds: r.bounds().map(rect_from_fb).unwrap_or_default(),
        roi_type: RustRoiType::try_from(u8_of(r.roi_type().0)?)?,
        priority: r.priority(),
        confidence: r.confidence(),
    })
}

fn build_correction<'a>(
    b: &mut FlatBufferBuilder<'a>,
    c: &RustCorrection,
) -> WIPOffset<generated::CorrectionData<'a>> {
    let roi = build_roi(b, &c.roi);
    let pixel_delta = b.create_vector(&c.pixel_delta);
    generated::CorrectionData::create(
        b,
        &generated::CorrectionDataArgs {
            roi: Some(roi),
            pixel_delta: Some(pixel_delta),
            compression: FbCompressionFormat(c.compression as i8),
        },
    )
}

fn read_correction(c: generated::CorrectionData<'_>) -> Result<RustCorrection, AspError> {
    Ok(RustCorrection {
        roi: read_roi(
            c.roi()
                .ok_or_else(|| de_err("CorrectionData without roi"))?,
        )?,
        pixel_delta: c
            .pixel_delta()
            .map(|v| v.bytes().to_vec())
            .unwrap_or_default(),
        compression: compression_from_fb(c.compression())?,
    })
}

fn build_i_packet<'a>(
    b: &mut FlatBufferBuilder<'a>,
    p: &RustIPacket,
) -> Result<WIPOffset<flatbuffers::UnionWIPOffset>, AspError> {
    if p.sdf_scene.is_some() {
        return Err(ser_err(
            "IPacketPayload::sdf_scene has no FlatBuffers representation; use write_to_buffer_bincode",
        ));
    }
    let global_palette = build_palette(b, &p.global_palette);
    let mut regions = Vec::with_capacity(p.regions.len());
    for r in &p.regions {
        regions.push(build_region(b, r)?);
    }
    let regions = b.create_vector(&regions);
    let animation = p.animation.as_ref().map(|a| build_animation(b, a));
    Ok(generated::IPacketPayload::create(
        b,
        &generated::IPacketPayloadArgs {
            width: p.width,
            height: p.height,
            fps: p.fps,
            quality: FbQualityLevel(p.quality as i8),
            global_palette: Some(global_palette),
            regions: Some(regions),
            animation,
            timestamp_ms: p.timestamp_ms,
        },
    )
    .as_union_value())
}

fn read_i_packet_full(fb: generated::IPacketPayload<'_>) -> Result<RustIPacket, AspError> {
    let mut regions = Vec::new();
    if let Some(v) = fb.regions() {
        for r in v.iter() {
            regions.push(read_region(r)?);
        }
    }
    Ok(RustIPacket {
        width: fb.width(),
        height: fb.height(),
        fps: fb.fps(),
        quality: RustQuality::try_from(u8_of(fb.quality().0)?)?,
        global_palette: read_palette(fb.global_palette()),
        regions,
        animation: fb.animation().map(read_animation).transpose()?,
        timestamp_ms: fb.timestamp_ms(),
        sdf_scene: None,
    })
}

fn build_d_packet<'a>(
    b: &mut FlatBufferBuilder<'a>,
    p: &RustDPacket,
) -> Result<WIPOffset<flatbuffers::UnionWIPOffset>, AspError> {
    if p.sdf_delta.is_some() || p.person_mask.is_some() {
        return Err(ser_err(
            "DPacketPayload::sdf_delta / person_mask have no FlatBuffers representation; use write_to_buffer_bincode",
        ));
    }
    let mvs: Vec<FbMotionVector> = p.motion_vectors.iter().map(motion_vector_to_fb).collect();
    let motion_vectors = b.create_vector(&mvs);
    let global_motion = p.global_motion.as_ref().map(|a| build_animation(b, a));
    let mut deltas = Vec::with_capacity(p.region_deltas.len());
    for d in &p.region_deltas {
        deltas.push(build_region_delta(b, d)?);
    }
    let region_deltas = b.create_vector(&deltas);
    Ok(generated::DPacketPayload::create(
        b,
        &generated::DPacketPayloadArgs {
            ref_sequence: p.ref_sequence,
            motion_vectors: Some(motion_vectors),
            motion_vectors_compact: None,
            global_motion,
            region_deltas: Some(region_deltas),
            timestamp_ms: p.timestamp_ms,
        },
    )
    .as_union_value())
}

fn read_d_packet_full(fb: generated::DPacketPayload<'_>) -> Result<RustDPacket, AspError> {
    let mut p = RustDPacket::new(fb.ref_sequence());
    p.timestamp_ms = fb.timestamp_ms();
    if let Some(mvs) = fb.motion_vectors() {
        p.motion_vectors = mvs.iter().map(motion_vector_from_fb).collect();
    }
    p.global_motion = fb.global_motion().map(read_animation).transpose()?;
    if let Some(v) = fb.region_deltas() {
        p.region_deltas = v.iter().map(read_region_delta).collect();
    }
    Ok(p)
}

fn build_c_packet<'a>(
    b: &mut FlatBufferBuilder<'a>,
    p: &RustCPacket,
) -> WIPOffset<flatbuffers::UnionWIPOffset> {
    let corrections: Vec<_> = p
        .corrections
        .iter()
        .map(|c| build_correction(b, c))
        .collect();
    let corrections = b.create_vector(&corrections);
    generated::CPacketPayload::create(
        b,
        &generated::CPacketPayloadArgs {
            ref_sequence: p.ref_sequence,
            corrections: Some(corrections),
            correction_count: p.correction_count,
            timestamp_ms: p.timestamp_ms,
        },
    )
    .as_union_value()
}

fn read_c_packet_full(fb: generated::CPacketPayload<'_>) -> Result<RustCPacket, AspError> {
    let mut corrections = Vec::new();
    if let Some(v) = fb.corrections() {
        for c in v.iter() {
            corrections.push(read_correction(c)?);
        }
    }
    Ok(RustCPacket {
        ref_sequence: fb.ref_sequence(),
        corrections,
        correction_count: fb.correction_count(),
        timestamp_ms: fb.timestamp_ms(),
    })
}

fn build_s_packet<'a>(
    b: &mut FlatBufferBuilder<'a>,
    p: &RustSPacket,
) -> WIPOffset<flatbuffers::UnionWIPOffset> {
    let (data_type, data) = match &p.data {
        SyncData::None => (generated::SyncDataUnion::NONE, None),
        SyncData::Sequence(sequence) => (
            generated::SyncDataUnion::SequenceData,
            Some(
                generated::SequenceData::create(
                    b,
                    &generated::SequenceDataArgs {
                        sequence: *sequence,
                    },
                )
                .as_union_value(),
            ),
        ),
        SyncData::Bitrate(bitrate_kbps) => (
            generated::SyncDataUnion::BitrateData,
            Some(
                generated::BitrateData::create(
                    b,
                    &generated::BitrateDataArgs {
                        bitrate_kbps: *bitrate_kbps,
                    },
                )
                .as_union_value(),
            ),
        ),
        SyncData::Quality(q) => (
            generated::SyncDataUnion::QualityData,
            Some(
                generated::QualityData::create(
                    b,
                    &generated::QualityDataArgs {
                        quality: FbQualityLevel(*q as i8),
                    },
                )
                .as_union_value(),
            ),
        ),
        SyncData::Latency(latency_ms) => (
            generated::SyncDataUnion::LatencyData,
            Some(
                generated::LatencyData::create(
                    b,
                    &generated::LatencyDataArgs {
                        latency_ms: *latency_ms,
                    },
                )
                .as_union_value(),
            ),
        ),
        SyncData::Custom(bytes) => {
            let data = b.create_vector(bytes);
            (
                generated::SyncDataUnion::CustomData,
                Some(
                    generated::CustomData::create(
                        b,
                        &generated::CustomDataArgs { data: Some(data) },
                    )
                    .as_union_value(),
                ),
            )
        }
    };
    generated::SPacketPayload::create(
        b,
        &generated::SPacketPayloadArgs {
            command: FbSyncCommand(p.command as i8),
            data_type,
            data,
            timestamp_ms: p.timestamp_ms,
        },
    )
    .as_union_value()
}

fn read_s_packet_full(fb: generated::SPacketPayload<'_>) -> Result<RustSPacket, AspError> {
    let missing = || de_err("SPacketPayload data_type does not match its data");
    let data = match fb.data_type() {
        generated::SyncDataUnion::NONE => SyncData::None,
        generated::SyncDataUnion::SequenceData => {
            SyncData::Sequence(fb.data_as_sequence_data().ok_or_else(missing)?.sequence())
        }
        generated::SyncDataUnion::BitrateData => SyncData::Bitrate(
            fb.data_as_bitrate_data()
                .ok_or_else(missing)?
                .bitrate_kbps(),
        ),
        generated::SyncDataUnion::QualityData => SyncData::Quality(RustQuality::try_from(u8_of(
            fb.data_as_quality_data().ok_or_else(missing)?.quality().0,
        )?)?),
        generated::SyncDataUnion::LatencyData => {
            SyncData::Latency(fb.data_as_latency_data().ok_or_else(missing)?.latency_ms())
        }
        generated::SyncDataUnion::CustomData => SyncData::Custom(
            fb.data_as_custom_data()
                .ok_or_else(missing)?
                .data()
                .map(|v| v.bytes().to_vec())
                .unwrap_or_default(),
        ),
        other => return Err(de_err(format!("unknown SyncDataUnion {}", other.0))),
    };
    Ok(RustSPacket {
        command: RustSyncCommand::try_from(u8_of(fb.command().0)?)?,
        data,
        timestamp_ms: fb.timestamp_ms(),
    })
}

/// Serialise a packet payload with every field the `asp.fbs` schema carries
/// (the bytes are a finished `AspPacketPayload` root with the `ASP1` identifier)
///
/// # Errors
///
/// `AspError::SerializationError` when the payload holds a field the schema
/// cannot represent (`sdf_scene` / `sdf_delta` / `person_mask`, DCT indices
/// above `u16::MAX`)
pub fn encode_payload(payload: &AspPayload) -> Result<Vec<u8>, AspError> {
    let mut b = FlatBufferBuilder::with_capacity(1024);
    encode_payload_with_builder(&mut b, payload)?;
    Ok(b.finished_data().to_vec())
}

/// [`encode_payload`] into a caller-owned builder (reset first); the finished
/// bytes are `builder.finished_data()`
///
/// # Errors
///
/// As [`encode_payload`]
pub fn encode_payload_with_builder(
    b: &mut FlatBufferBuilder<'_>,
    payload: &AspPayload,
) -> Result<(), AspError> {
    b.reset();
    let (payload_type, payload) = match payload {
        AspPayload::IPacket(p) => (
            generated::AspPayloadUnion::IPacketPayload,
            build_i_packet(b, p)?,
        ),
        AspPayload::DPacket(p) => (
            generated::AspPayloadUnion::DPacketPayload,
            build_d_packet(b, p)?,
        ),
        AspPayload::CPacket(p) => (
            generated::AspPayloadUnion::CPacketPayload,
            build_c_packet(b, p),
        ),
        AspPayload::SPacket(p) => (
            generated::AspPayloadUnion::SPacketPayload,
            build_s_packet(b, p),
        ),
    };
    let root = generated::AspPacketPayload::create(
        b,
        &generated::AspPacketPayloadArgs {
            payload_type,
            payload: Some(payload),
        },
    );
    b.finish(root, Some("ASP1"));
    Ok(())
}

/// Inverse of [`encode_payload`]: verifies the buffer and rebuilds the Rust
/// payload with every field the schema carries
///
/// # Errors
///
/// `AspError::DeserializationError` for a buffer that fails FlatBuffers
/// verification, a union whose tag and data disagree, or an enum value outside
/// the schema
pub fn decode_payload(bytes: &[u8]) -> Result<AspPayload, AspError> {
    let packet = read_packet(bytes).map_err(de_err)?;
    let missing = || de_err("AspPacketPayload payload_type does not match its payload");
    Ok(match packet.payload_type() {
        generated::AspPayloadUnion::IPacketPayload => AspPayload::IPacket(read_i_packet_full(
            packet.payload_as_ipacket_payload().ok_or_else(missing)?,
        )?),
        generated::AspPayloadUnion::DPacketPayload => AspPayload::DPacket(read_d_packet_full(
            packet.payload_as_dpacket_payload().ok_or_else(missing)?,
        )?),
        generated::AspPayloadUnion::CPacketPayload => AspPayload::CPacket(read_c_packet_full(
            packet.payload_as_cpacket_payload().ok_or_else(missing)?,
        )?),
        generated::AspPayloadUnion::SPacketPayload => AspPayload::SPacket(read_s_packet_full(
            packet.payload_as_spacket_payload().ok_or_else(missing)?,
        )?),
        other => return Err(de_err(format!("unknown AspPayloadUnion {}", other.0))),
    })
}

// =============================================================================
// Builder Reuse API (Zero-Allocation Hot Loop)
// =============================================================================

/// Encode a D-Packet reusing an existing `FlatBufferBuilder`
///
/// This function is designed for hot loops where you want to avoid
/// per-frame allocations. The builder is reset at the start of each call.
///
/// # Optimization
/// - Uses `start_vector` + `push` pattern (no intermediate Vec allocation)
/// - `FlatBuffers` builds vectors backwards, so we iterate in reverse
///
/// # Example
///
/// ```rust
/// use flatbuffers::FlatBufferBuilder;
/// use libasp::flatbuffers_api::encode_d_packet_with_builder;
///
/// use libasp::types::MotionVector;
///
/// // Create builder once
/// let mut builder = FlatBufferBuilder::with_capacity(4096);
///
/// // Reuse in hot loop (one allocation for the whole stream)
/// let frames = [(1u32, vec![MotionVector::new(0, 0, 1, 1, 2)], 33u64), (2, vec![], 66)];
/// for (ref_seq, mvs, ts) in &frames {
///     let bytes = encode_d_packet_with_builder(&mut builder, *ref_seq, mvs, *ts);
///     assert!(!bytes.is_empty()); // socket.send(bytes)
/// }
/// ```
#[inline]
pub fn encode_d_packet_with_builder<'a>(
    builder: &'a mut FlatBufferBuilder<'static>,
    ref_sequence: u32,
    mvs: &[RustMotionVector],
    timestamp_ms: u64,
) -> &'a [u8] {
    builder.reset();

    // 1. Serialize Motion Vectors (Direct Push - Zero Allocation)
    // FlatBuffers builds vectors backwards, so we iterate in reverse
    let mvs_len = mvs.len();

    builder.start_vector::<FbMotionVector>(mvs_len);
    for i in (0..mvs_len).rev() {
        // SAFETY: i is always in bounds [0, mvs_len)
        let mv = unsafe { mvs.get_unchecked(i) };
        let fb_mv = FbMotionVector::new(mv.block_x, mv.block_y, mv.dx, mv.dy, mv.sad);
        builder.push(fb_mv);
    }
    let mvs_offset = builder.end_vector::<FbMotionVector>(mvs_len);

    // 2. Build DPacketPayload
    let d_packet = generated::DPacketPayload::create(
        builder,
        &generated::DPacketPayloadArgs {
            ref_sequence,
            motion_vectors: Some(mvs_offset),
            motion_vectors_compact: None,
            global_motion: None,
            region_deltas: None,
            timestamp_ms,
        },
    );

    // 3. Wrap in AspPacketPayload union
    let asp_packet = generated::AspPacketPayload::create(
        builder,
        &generated::AspPacketPayloadArgs {
            payload_type: generated::AspPayloadUnion::DPacketPayload,
            payload: Some(d_packet.as_union_value()),
        },
    );

    builder.finish(asp_packet, Some("ASP1"));
    builder.finished_data()
}

/// Encode a D-Packet with compact motion vectors reusing an existing builder
///
/// # Optimization
/// - Uses `start_vector` + `push` pattern (no intermediate Vec allocation)
#[inline]
pub fn encode_d_packet_compact_with_builder<'a>(
    builder: &'a mut FlatBufferBuilder<'static>,
    ref_sequence: u32,
    compact_mvs: &[(i8, i8)],
    timestamp_ms: u64,
) -> &'a [u8] {
    builder.reset();

    // Direct Push - Zero Allocation
    let mvs_len = compact_mvs.len();

    builder.start_vector::<FbMotionVectorCompact>(mvs_len);
    for i in (0..mvs_len).rev() {
        // SAFETY: i is always in bounds [0, mvs_len)
        let (dx, dy) = unsafe { *compact_mvs.get_unchecked(i) };
        let fb_mv = FbMotionVectorCompact::new(dx, dy);
        builder.push(fb_mv);
    }
    let mvs_offset = builder.end_vector::<FbMotionVectorCompact>(mvs_len);

    // Build DPacketPayload
    let d_packet = generated::DPacketPayload::create(
        builder,
        &generated::DPacketPayloadArgs {
            ref_sequence,
            motion_vectors: None,
            motion_vectors_compact: Some(mvs_offset),
            global_motion: None,
            region_deltas: None,
            timestamp_ms,
        },
    );

    // Wrap in AspPacketPayload union
    let asp_packet = generated::AspPacketPayload::create(
        builder,
        &generated::AspPacketPayloadArgs {
            payload_type: generated::AspPayloadUnion::DPacketPayload,
            payload: Some(d_packet.as_union_value()),
        },
    );

    builder.finish(asp_packet, Some("ASP1"));
    builder.finished_data()
}

/// Encode an I-Packet reusing an existing `FlatBufferBuilder`
#[inline]
pub fn encode_i_packet_with_builder<'a>(
    builder: &'a mut FlatBufferBuilder<'static>,
    width: u32,
    height: u32,
    fps: f32,
    quality: FbQualityLevel,
    timestamp_ms: u64,
) -> &'a [u8] {
    builder.reset();

    // Build IPacketPayload
    let i_packet = generated::IPacketPayload::create(
        builder,
        &generated::IPacketPayloadArgs {
            width,
            height,
            fps,
            quality,
            global_palette: None,
            regions: None,
            animation: None,
            timestamp_ms,
        },
    );

    // Wrap in AspPacketPayload union
    let asp_packet = generated::AspPacketPayload::create(
        builder,
        &generated::AspPacketPayloadArgs {
            payload_type: generated::AspPayloadUnion::IPacketPayload,
            payload: Some(i_packet.as_union_value()),
        },
    );

    builder.finish(asp_packet, Some("ASP1"));
    builder.finished_data()
}

/// Encode an S-Packet reusing an existing `FlatBufferBuilder`
#[inline]
pub fn encode_s_packet_with_builder<'a>(
    builder: &'a mut FlatBufferBuilder<'static>,
    command: FbSyncCommand,
    timestamp_ms: u64,
) -> &'a [u8] {
    builder.reset();

    // Build SPacketPayload
    let s_packet = generated::SPacketPayload::create(
        builder,
        &generated::SPacketPayloadArgs {
            command,
            data_type: generated::SyncDataUnion::NONE,
            data: None,
            timestamp_ms,
        },
    );

    // Wrap in AspPacketPayload union
    let asp_packet = generated::AspPacketPayload::create(
        builder,
        &generated::AspPacketPayloadArgs {
            payload_type: generated::AspPayloadUnion::SPacketPayload,
            payload: Some(s_packet.as_union_value()),
        },
    );

    builder.finish(asp_packet, Some("ASP1"));
    builder.finished_data()
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_d_packet_roundtrip() {
        let mvs = vec![
            RustMotionVector::new(0, 0, 5, -3, 100),
            RustMotionVector::new(1, 0, 2, 1, 50),
            RustMotionVector::new(2, 0, -1, 0, 25),
        ];

        let bytes = DPacketBuilder::new(42)
            .motion_vectors(&mvs)
            .timestamp_ms(12345)
            .build();

        // Verify we can read it back
        let packet = read_d_packet(&bytes).unwrap();
        assert_eq!(packet.ref_sequence(), 42);
        assert_eq!(packet.timestamp_ms(), 12345);

        let read_mvs = packet.motion_vectors().unwrap();
        assert_eq!(read_mvs.len(), 3);

        let mv0 = read_mvs.get(0);
        assert_eq!(mv0.block_x(), 0);
        assert_eq!(mv0.block_y(), 0);
        assert_eq!(mv0.dx(), 5);
        assert_eq!(mv0.dy(), -3);
        assert_eq!(mv0.sad(), 100);
    }

    #[test]
    fn test_i_packet_roundtrip() {
        let bytes = IPacketBuilder::new(1920, 1080, 30.0)
            .quality(FbQualityLevel::High)
            .timestamp_ms(0)
            .build();

        let packet = read_i_packet(&bytes).unwrap();
        assert_eq!(packet.width(), 1920);
        assert_eq!(packet.height(), 1080);
        assert!((packet.fps() - 30.0).abs() < 0.01);
        assert_eq!(packet.quality(), FbQualityLevel::High);
    }

    #[test]
    fn test_s_packet_roundtrip() {
        let bytes = SPacketBuilder::ping().timestamp_ms(99999).build();

        let packet = read_s_packet(&bytes).unwrap();
        assert_eq!(packet.command(), FbSyncCommand::Ping);
        assert_eq!(packet.timestamp_ms(), 99999);
    }

    #[test]
    fn test_packet_type_detection() {
        let d_bytes = create_d_packet(1, &[], 0);
        assert_eq!(
            get_packet_type(&d_bytes).unwrap(),
            generated::AspPayloadUnion::DPacketPayload
        );

        let i_bytes = create_i_packet(640, 480, 30.0, 0);
        assert_eq!(
            get_packet_type(&i_bytes).unwrap(),
            generated::AspPayloadUnion::IPacketPayload
        );

        let s_bytes = create_ping(0);
        assert_eq!(
            get_packet_type(&s_bytes).unwrap(),
            generated::AspPayloadUnion::SPacketPayload
        );
    }

    #[test]
    fn test_build_into_buffer() {
        let mvs = vec![RustMotionVector::new(0, 0, 1, 1, 10)];

        let mut buffer = [0u8; 1024];
        let len = DPacketBuilder::new(1)
            .motion_vectors(&mvs)
            .build_into(&mut buffer)
            .unwrap();

        // Verify we can read from the buffer
        let packet = read_d_packet(&buffer[..len]).unwrap();
        assert_eq!(packet.ref_sequence(), 1);
    }

    #[test]
    fn test_compact_motion_vectors() {
        let compact = vec![(5i8, -3i8), (2, 1), (-1, 0)];

        let bytes = DPacketBuilder::new(1).compact_vectors(&compact).build();

        let packet = read_d_packet(&bytes).unwrap();
        let mvs = packet.motion_vectors_compact().unwrap();
        assert_eq!(mvs.len(), 3);

        let mv0 = mvs.get(0);
        assert_eq!(mv0.dx(), 5);
        assert_eq!(mv0.dy(), -3);
    }

    #[test]
    fn test_type_conversions() {
        let rust_mv = RustMotionVector::new(10, 20, -5, 3, 500);
        let fb_mv = motion_vector_to_fb(&rust_mv);
        let back = motion_vector_from_fb(&fb_mv);

        assert_eq!(rust_mv.block_x, back.block_x);
        assert_eq!(rust_mv.block_y, back.block_y);
        assert_eq!(rust_mv.dx, back.dx);
        assert_eq!(rust_mv.dy, back.dy);
        assert_eq!(rust_mv.sad, back.sad);
    }

    #[test]
    fn test_builder_reuse_d_packet() {
        let mut builder = FlatBufferBuilder::with_capacity(4096);

        // First encoding
        let mvs1 = vec![RustMotionVector::new(0, 0, 5, -3, 100)];
        let bytes1 = encode_d_packet_with_builder(&mut builder, 1, &mvs1, 1000);
        let bytes1_copy = bytes1.to_vec();

        // Second encoding (builder reused)
        let mvs2 = vec![
            RustMotionVector::new(1, 1, 2, 1, 50),
            RustMotionVector::new(2, 2, -1, 0, 25),
        ];
        let bytes2 = encode_d_packet_with_builder(&mut builder, 2, &mvs2, 2000);
        let bytes2_copy = bytes2.to_vec();

        // Verify both can be read correctly
        let packet1 = read_d_packet(&bytes1_copy).unwrap();
        assert_eq!(packet1.ref_sequence(), 1);
        assert_eq!(packet1.motion_vectors().unwrap().len(), 1);

        let packet2 = read_d_packet(&bytes2_copy).unwrap();
        assert_eq!(packet2.ref_sequence(), 2);
        assert_eq!(packet2.motion_vectors().unwrap().len(), 2);
    }

    #[test]
    fn test_builder_reuse_stress() {
        let mut builder = FlatBufferBuilder::with_capacity(1024);

        // Encode many packets with same builder
        for i in 0..100 {
            let mvs = vec![RustMotionVector::new(i as u16, 0, 1, 1, i)];
            let bytes = encode_d_packet_with_builder(&mut builder, i, &mvs, i as u64 * 10);

            let packet = read_d_packet(bytes).unwrap();
            assert_eq!(packet.ref_sequence(), i);
            assert_eq!(packet.timestamp_ms(), i as u64 * 10);
        }
    }

    #[test]
    fn test_encode_i_packet_with_builder() {
        let mut builder = FlatBufferBuilder::with_capacity(1024);

        let bytes = encode_i_packet_with_builder(
            &mut builder,
            1920,
            1080,
            30.0,
            FbQualityLevel::High,
            12345,
        );

        let packet = read_i_packet(bytes).unwrap();
        assert_eq!(packet.width(), 1920);
        assert_eq!(packet.height(), 1080);
        assert!((packet.fps() - 30.0).abs() < 0.01);
        assert_eq!(packet.quality(), FbQualityLevel::High);
    }

    #[test]
    fn test_encode_s_packet_with_builder() {
        let mut builder = FlatBufferBuilder::with_capacity(256);

        let bytes =
            encode_s_packet_with_builder(&mut builder, FbSyncCommand::RequestKeyframe, 99999);

        let packet = read_s_packet(bytes).unwrap();
        assert_eq!(packet.command(), FbSyncCommand::RequestKeyframe);
        assert_eq!(packet.timestamp_ms(), 99999);
    }
}
