//! Common types and enumerations for ASP protocol
//!
//! This module defines the fundamental types used throughout the
//! ALICE Streaming Protocol implementation.

use serde::{Deserialize, Serialize};

/// Magic bytes for ASP packet identification
pub const ASP_MAGIC: [u8; 4] = [0x41, 0x53, 0x50, 0x31]; // "ASP1"

/// Protocol version
pub const ASP_VERSION: u8 = 1;

/// Maximum packet size (64KB)
pub const MAX_PACKET_SIZE: usize = 65536;

/// Default block size for motion estimation
pub const DEFAULT_BLOCK_SIZE: usize = 16;

/// Default search range for motion estimation
pub const DEFAULT_SEARCH_RANGE: usize = 16;

/// Packet type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum PacketType {
    /// I-Packet: Initial/Keyframe with full procedural description
    IPacket = 0x01,
    /// D-Packet: Delta frame with incremental updates
    DPacket = 0x02,
    /// C-Packet: Correction packet for ROI-based pixel corrections
    CPacket = 0x03,
    /// S-Packet: Sync/Control packet
    SPacket = 0x04,
}

impl TryFrom<u8> for PacketType {
    type Error = AspError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x01 => Ok(Self::IPacket),
            0x02 => Ok(Self::DPacket),
            0x03 => Ok(Self::CPacket),
            0x04 => Ok(Self::SPacket),
            _ => Err(AspError::InvalidPacketType(value)),
        }
    }
}

/// Pattern type for procedural generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum PatternType {
    /// Solid color fill
    #[default]
    Solid = 0x00,
    /// Linear gradient
    GradientLinear = 0x01,
    /// Radial gradient
    GradientRadial = 0x02,
    /// Noise pattern (Perlin, Simplex, etc.)
    Noise = 0x03,
    /// Texture reference
    Texture = 0x04,
    /// DCT-based pattern
    Dct = 0x05,
    /// Periodic/repeating pattern
    Periodic = 0x06,
    /// Complex pattern (combination)
    Complex = 0x07,
}

impl TryFrom<u8> for PatternType {
    type Error = AspError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x00 => Ok(Self::Solid),
            0x01 => Ok(Self::GradientLinear),
            0x02 => Ok(Self::GradientRadial),
            0x03 => Ok(Self::Noise),
            0x04 => Ok(Self::Texture),
            0x05 => Ok(Self::Dct),
            0x06 => Ok(Self::Periodic),
            0x07 => Ok(Self::Complex),
            _ => Err(AspError::InvalidPatternType(value)),
        }
    }
}

/// Motion type for animation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum MotionType {
    /// No motion
    #[default]
    None = 0x00,
    /// Linear motion
    Linear = 0x01,
    /// Easing motion (ease-in, ease-out)
    Easing = 0x02,
    /// Oscillating motion
    Oscillate = 0x03,
    /// Complex motion path
    Path = 0x04,
    /// Physics-based motion
    Physics = 0x05,
}

impl TryFrom<u8> for MotionType {
    type Error = AspError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x00 => Ok(Self::None),
            0x01 => Ok(Self::Linear),
            0x02 => Ok(Self::Easing),
            0x03 => Ok(Self::Oscillate),
            0x04 => Ok(Self::Path),
            0x05 => Ok(Self::Physics),
            _ => Err(AspError::InvalidMotionType(value)),
        }
    }
}

/// Region of Interest type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum RoiType {
    /// General region
    #[default]
    General = 0x00,
    /// Face region (high priority)
    Face = 0x01,
    /// Text region
    Text = 0x02,
    /// Edge/detail region
    Edge = 0x03,
    /// Moving object
    Motion = 0x04,
    /// User-defined region
    Custom = 0x05,
}

impl TryFrom<u8> for RoiType {
    type Error = AspError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x00 => Ok(Self::General),
            0x01 => Ok(Self::Face),
            0x02 => Ok(Self::Text),
            0x03 => Ok(Self::Edge),
            0x04 => Ok(Self::Motion),
            0x05 => Ok(Self::Custom),
            _ => Err(AspError::InvalidRoiType(value)),
        }
    }
}

/// Quality level for encoding
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
#[repr(u8)]
pub enum QualityLevel {
    /// Low quality (fast encoding, small size)
    Low = 0x00,
    /// Medium quality (balanced)
    #[default]
    Medium = 0x01,
    /// High quality (slower encoding, better quality)
    High = 0x02,
    /// Ultra quality (best quality, largest size)
    Ultra = 0x03,
}

impl TryFrom<u8> for QualityLevel {
    type Error = AspError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x00 => Ok(Self::Low),
            0x01 => Ok(Self::Medium),
            0x02 => Ok(Self::High),
            0x03 => Ok(Self::Ultra),
            _ => Err(AspError::InvalidQualityLevel(value)),
        }
    }
}

/// Sync command type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum SyncCommand {
    /// Request keyframe
    RequestKeyframe = 0x01,
    /// Acknowledge receipt
    Ack = 0x02,
    /// Report packet loss
    Nack = 0x03,
    /// End of stream
    EndOfStream = 0x04,
    /// Bitrate adjustment
    BitrateAdjust = 0x05,
    /// Quality change request
    QualityChange = 0x06,
    /// Ping/keepalive
    Ping = 0x07,
    /// Pong response
    Pong = 0x08,
}

impl TryFrom<u8> for SyncCommand {
    type Error = AspError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x01 => Ok(Self::RequestKeyframe),
            0x02 => Ok(Self::Ack),
            0x03 => Ok(Self::Nack),
            0x04 => Ok(Self::EndOfStream),
            0x05 => Ok(Self::BitrateAdjust),
            0x06 => Ok(Self::QualityChange),
            0x07 => Ok(Self::Ping),
            0x08 => Ok(Self::Pong),
            _ => Err(AspError::InvalidSyncCommand(value)),
        }
    }
}

/// ASP Error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum AspError {
    /// Invalid magic bytes in packet header
    #[error("Invalid magic bytes")]
    InvalidMagic,

    /// Unrecognized packet type byte
    #[error("Invalid packet type: {0}")]
    InvalidPacketType(u8),

    /// Unrecognized pattern type byte
    #[error("Invalid pattern type: {0}")]
    InvalidPatternType(u8),

    /// Unrecognized motion type byte
    #[error("Invalid motion type: {0}")]
    InvalidMotionType(u8),

    /// Unrecognized ROI type byte
    #[error("Invalid ROI type: {0}")]
    InvalidRoiType(u8),

    /// Unrecognized quality level byte
    #[error("Invalid quality level: {0}")]
    InvalidQualityLevel(u8),

    /// Unrecognized sync command byte
    #[error("Invalid sync command: {0}")]
    InvalidSyncCommand(u8),

    /// Packet exceeds maximum allowed size
    #[error("Packet too large: {size} > {max}")]
    PacketTooLarge {
        /// Actual packet size
        size: usize,
        /// Maximum allowed size
        max: usize,
    },

    /// Packet data is shorter than expected
    #[error("Incomplete packet: expected {expected}, got {got}")]
    IncompletePacket {
        /// Expected byte count
        expected: usize,
        /// Actual byte count
        got: usize,
    },

    /// CRC32 checksum verification failed
    #[error("Checksum mismatch: expected {expected:08x}, got {got:08x}")]
    ChecksumMismatch {
        /// Expected checksum
        expected: u32,
        /// Computed checksum
        got: u32,
    },

    /// Error during packet serialization
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Error during packet deserialization
    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    /// Error during video encoding
    #[error("Encoder error: {0}")]
    EncoderError(String),

    /// Error during video decoding
    #[error("Decoder error: {0}")]
    DecoderError(String),

    /// Invalid frame width or height
    #[error("Invalid frame dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Frame width
        width: u32,
        /// Frame height
        height: u32,
    },

    /// Sequence number does not match expected value
    #[error("Sequence number mismatch: expected {expected}, got {got}")]
    SequenceMismatch {
        /// Expected sequence number
        expected: u32,
        /// Received sequence number
        got: u32,
    },
}

/// Result type for ASP operations
pub type AspResult<T> = Result<T, AspError>;

/// Color in RGB format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Color {
    /// Red channel
    pub r: u8,
    /// Green channel
    pub g: u8,
    /// Blue channel
    pub b: u8,
}

impl Color {
    /// Create a new color from RGB components
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Create black color (0, 0, 0)
    #[must_use]
    pub const fn black() -> Self {
        Self::new(0, 0, 0)
    }

    /// Create white color (255, 255, 255)
    #[must_use]
    pub const fn white() -> Self {
        Self::new(255, 255, 255)
    }

    /// Convert to RGB array
    #[must_use]
    pub const fn to_array(self) -> [u8; 3] {
        [self.r, self.g, self.b]
    }

    /// Create from RGB array
    #[must_use]
    pub const fn from_array(arr: [u8; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }

    /// Calculate luminance (Y component in YUV)
    #[must_use]
    pub fn luminance(self) -> f32 {
        0.114f32.mul_add(
            self.b as f32,
            0.299f32.mul_add(self.r as f32, 0.587 * self.g as f32),
        )
    }
}

/// 2D point with integer coordinates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Point {
    /// X coordinate
    pub x: i32,
    /// Y coordinate
    pub y: i32,
}

impl Point {
    /// Create a new point
    #[must_use]
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    /// Create the origin point (0, 0)
    #[must_use]
    pub const fn origin() -> Self {
        Self::new(0, 0)
    }
}

/// Rectangle definition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Rect {
    /// Left edge X coordinate
    pub x: u32,
    /// Top edge Y coordinate
    pub y: u32,
    /// Width in pixels
    pub width: u32,
    /// Height in pixels
    pub height: u32,
}

impl Rect {
    /// Create a new rectangle
    #[must_use]
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    /// Calculate area in pixels
    #[must_use]
    pub const fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    /// Check if a point is inside the rectangle
    #[must_use]
    pub const fn contains(&self, x: u32, y: u32) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }

    /// Check if this rectangle intersects with another
    #[must_use]
    pub const fn intersects(&self, other: &Self) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }
}

/// Motion vector (16 bytes, cache-line optimized)
///
/// Layout: [`block_x`: u16, `block_y`: u16, dx: i16, dy: i16, sad: u32, _pad: u32]
/// For 4K video (3840x2160), 16x16 blocks = 240x135 = 32,400 blocks
/// u16 range (0-65535) is sufficient for block indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[repr(C)]
pub struct MotionVector {
    /// Block X position (u16: supports up to 65535 blocks = 1M+ pixels at 16px blocks)
    pub block_x: u16,
    /// Block Y position (u16: supports up to 65535 blocks = 1M+ pixels at 16px blocks)
    pub block_y: u16,
    /// Horizontal displacement
    pub dx: i16,
    /// Vertical displacement
    pub dy: i16,
    /// Sum of Absolute Differences (matching quality)
    pub sad: u32,
    /// Padding for 16-byte alignment (reserved for future use)
    #[serde(skip)]
    pub reserved: u32,
}

impl MotionVector {
    /// Create a new motion vector
    #[inline]
    #[must_use]
    pub const fn new(block_x: u16, block_y: u16, dx: i16, dy: i16, sad: u32) -> Self {
        Self {
            block_x,
            block_y,
            dx,
            dy,
            sad,
            reserved: 0,
        }
    }

    /// Create from u32 coordinates (with truncation for legacy compatibility)
    #[inline]
    #[must_use]
    pub const fn from_u32(block_x: u32, block_y: u32, dx: i16, dy: i16, sad: u32) -> Self {
        Self {
            block_x: block_x as u16,
            block_y: block_y as u16,
            dx,
            dy,
            sad,
            reserved: 0,
        }
    }

    /// Check if this is a zero motion vector
    #[inline]
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.dx == 0 && self.dy == 0
    }

    /// Calculate motion magnitude
    #[inline]
    #[must_use]
    pub fn magnitude(&self) -> f32 {
        (self.dx as f32).hypot(self.dy as f32)
    }

    /// Convert to compact format (for bandwidth optimization)
    #[inline]
    #[must_use]
    pub const fn to_compact(&self) -> Option<MotionVectorCompact> {
        // Only convert if dx/dy fit in i8 range
        if self.dx >= -128 && self.dx <= 127 && self.dy >= -128 && self.dy <= 127 {
            Some(MotionVectorCompact {
                dx: self.dx as i8,
                dy: self.dy as i8,
            })
        } else {
            None
        }
    }
}

/// Compact motion vector for bandwidth-efficient transmission
///
/// Uses only 2 bytes (vs 16 bytes for full `MotionVector`).
/// Block position is implicit from array index.
/// SAD is omitted as it's only needed during encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[repr(C, packed)]
pub struct MotionVectorCompact {
    /// Horizontal displacement (-128 to 127 pixels)
    pub dx: i8,
    /// Vertical displacement (-128 to 127 pixels)
    pub dy: i8,
}

impl MotionVectorCompact {
    /// Create a new compact motion vector
    #[inline]
    #[must_use]
    pub const fn new(dx: i8, dy: i8) -> Self {
        Self { dx, dy }
    }

    /// Create a zero motion vector (no displacement)
    #[inline]
    #[must_use]
    pub const fn zero() -> Self {
        Self { dx: 0, dy: 0 }
    }

    /// Check if this is a zero motion vector
    #[inline]
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.dx == 0 && self.dy == 0
    }

    /// Expand to full `MotionVector` with block position and SAD
    #[inline]
    #[must_use]
    pub const fn expand(self, block_x: u16, block_y: u16, sad: u32) -> MotionVector {
        MotionVector {
            block_x,
            block_y,
            dx: self.dx as i16,
            dy: self.dy as i16,
            sad,
            reserved: 0,
        }
    }
}

/// Animation parameters
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AnimationParams {
    /// Zoom factor (1.0 = no zoom)
    pub zoom_factor: f32,
    /// Pan X offset (percentage of frame width)
    pub pan_x: f32,
    /// Pan Y offset (percentage of frame height)
    pub pan_y: f32,
    /// Rotation angle in degrees
    pub rotation: f32,
    /// Duration in frames
    pub duration: u32,
    /// Easing function type
    pub easing: EasingType,
}

impl Default for AnimationParams {
    fn default() -> Self {
        Self {
            zoom_factor: 1.0,
            pan_x: 0.0,
            pan_y: 0.0,
            rotation: 0.0,
            duration: 1,
            easing: EasingType::Linear,
        }
    }
}

/// Easing function type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[repr(u8)]
pub enum EasingType {
    /// Linear interpolation (default)
    #[default]
    Linear = 0x00,
    /// Ease-in (slow start)
    EaseIn = 0x01,
    /// Ease-out (slow end)
    EaseOut = 0x02,
    /// Ease-in-out (slow start and end)
    EaseInOut = 0x03,
    /// Bounce effect
    Bounce = 0x04,
    /// Elastic spring effect
    Elastic = 0x05,
}

/// Stream statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StreamStats {
    /// Total bytes sent
    pub total_bytes: u64,
    /// Total packets sent
    pub total_packets: u64,
    /// I-Packets sent
    pub i_packets: u64,
    /// D-Packets sent
    pub d_packets: u64,
    /// C-Packets sent
    pub c_packets: u64,
    /// S-Packets sent
    pub s_packets: u64,
    /// Total frames encoded
    pub frames_encoded: u64,
    /// Average bits per frame
    pub avg_bits_per_frame: f64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Average encoding time in microseconds
    pub avg_encode_time_us: f64,
    /// Peak encoding time in microseconds
    pub peak_encode_time_us: u64,
}

impl StreamStats {
    /// Create new empty stream statistics
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a sent packet in the statistics
    pub const fn update_packet(&mut self, packet_type: PacketType, size: usize) {
        self.total_bytes += size as u64;
        self.total_packets += 1;
        match packet_type {
            PacketType::IPacket => self.i_packets += 1,
            PacketType::DPacket => self.d_packets += 1,
            PacketType::CPacket => self.c_packets += 1,
            PacketType::SPacket => self.s_packets += 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packet_type_conversion() {
        assert_eq!(PacketType::try_from(0x01).unwrap(), PacketType::IPacket);
        assert_eq!(PacketType::try_from(0x02).unwrap(), PacketType::DPacket);
        assert!(PacketType::try_from(0xFF).is_err());
    }

    #[test]
    fn test_color_luminance() {
        let white = Color::white();
        assert!((white.luminance() - 255.0).abs() < 0.01);

        let black = Color::black();
        assert!(black.luminance().abs() < 0.01);
    }

    #[test]
    fn test_rect_operations() {
        let rect = Rect::new(10, 10, 100, 100);
        assert_eq!(rect.area(), 10000);
        assert!(rect.contains(50, 50));
        assert!(!rect.contains(5, 5));
    }

    #[test]
    fn test_motion_vector() {
        let mv = MotionVector::new(0, 0, 3, 4, 100);
        assert!((mv.magnitude() - 5.0).abs() < 0.01);
        assert!(!mv.is_zero());

        let zero_mv = MotionVector::new(0, 0, 0, 0, 0);
        assert!(zero_mv.is_zero());
    }

    #[test]
    fn test_motion_vector_size() {
        // Verify struct is 16 bytes (cache-line optimized)
        assert_eq!(std::mem::size_of::<MotionVector>(), 16);
    }

    #[test]
    fn test_motion_vector_from_u32() {
        // Legacy compatibility
        let mv = MotionVector::from_u32(100, 200, -5, 10, 500);
        assert_eq!(mv.block_x, 100);
        assert_eq!(mv.block_y, 200);
        assert_eq!(mv.dx, -5);
        assert_eq!(mv.dy, 10);
        assert_eq!(mv.sad, 500);
    }

    #[test]
    fn test_color_to_from_array_roundtrip() {
        let c = Color::new(10, 128, 255);
        let arr = c.to_array();
        assert_eq!(arr, [10, 128, 255]);
        let restored = Color::from_array(arr);
        assert_eq!(restored, c);
    }

    #[test]
    fn test_color_luminance_channels() {
        // Pure green has highest weight (0.587)
        let green = Color::new(0, 255, 0);
        let lum = green.luminance();
        assert!(
            255.0f32.mul_add(-0.587, lum).abs() < 0.5,
            "green lum = {lum}"
        );
        // Pure blue has lowest weight (0.114)
        let blue = Color::new(0, 0, 255);
        let blue_lum = blue.luminance();
        assert!(
            255.0f32.mul_add(-0.114, blue_lum).abs() < 0.5,
            "blue lum = {blue_lum}"
        );
    }

    #[test]
    fn test_rect_intersects_non_overlapping() {
        let r1 = Rect::new(0, 0, 10, 10);
        let r2 = Rect::new(20, 20, 10, 10);
        assert!(!r1.intersects(&r2));
        assert!(!r2.intersects(&r1));
    }

    #[test]
    fn test_rect_intersects_adjacent_no_overlap() {
        // Touching at right edge — open interval means no intersection
        let r1 = Rect::new(0, 0, 10, 10);
        let r2 = Rect::new(10, 0, 10, 10);
        assert!(!r1.intersects(&r2));
    }

    #[test]
    fn test_rect_contains_boundary() {
        let r = Rect::new(5, 5, 10, 10);
        assert!(r.contains(5, 5)); // top-left included
        assert!(!r.contains(15, 5)); // right boundary exclusive
        assert!(!r.contains(5, 15)); // bottom boundary exclusive
        assert!(!r.contains(4, 5)); // one pixel left of rect
    }

    #[test]
    fn test_rect_zero_area() {
        let r = Rect::new(0, 0, 0, 5);
        assert_eq!(r.area(), 0);
    }

    #[test]
    fn test_motion_vector_compact_roundtrip() {
        let mv = MotionVector::new(3, 7, 50, -30, 999);
        let compact = mv.to_compact().expect("should fit in i8 range");
        assert_eq!(compact.dx, 50);
        assert_eq!(compact.dy, -30);

        let expanded = compact.expand(3, 7, 999);
        assert_eq!(expanded.block_x, 3);
        assert_eq!(expanded.block_y, 7);
        assert_eq!(expanded.dx, 50);
        assert_eq!(expanded.dy, -30);
        assert_eq!(expanded.sad, 999);
    }

    #[test]
    fn test_motion_vector_compact_out_of_range() {
        // dx = 200 > 127, cannot be stored in i8
        let mv = MotionVector::new(0, 0, 200, 0, 0);
        assert!(mv.to_compact().is_none());

        let mv_neg = MotionVector::new(0, 0, 0, -200, 0);
        assert!(mv_neg.to_compact().is_none());
    }

    #[test]
    fn test_motion_vector_compact_zero() {
        let c = MotionVectorCompact::zero();
        assert!(c.is_zero());
        let c2 = MotionVectorCompact::new(1, 0);
        assert!(!c2.is_zero());
    }

    #[test]
    fn test_quality_level_ordering() {
        assert!(QualityLevel::Low < QualityLevel::Medium);
        assert!(QualityLevel::Medium < QualityLevel::High);
        assert!(QualityLevel::High < QualityLevel::Ultra);
    }

    #[test]
    fn test_quality_level_try_from() {
        assert_eq!(QualityLevel::try_from(0).unwrap(), QualityLevel::Low);
        assert_eq!(QualityLevel::try_from(3).unwrap(), QualityLevel::Ultra);
        assert!(QualityLevel::try_from(4).is_err());
    }

    #[test]
    fn test_pattern_type_try_from_all() {
        for (byte, expected) in [
            (0x00, PatternType::Solid),
            (0x01, PatternType::GradientLinear),
            (0x02, PatternType::GradientRadial),
            (0x03, PatternType::Noise),
            (0x04, PatternType::Texture),
            (0x05, PatternType::Dct),
            (0x06, PatternType::Periodic),
            (0x07, PatternType::Complex),
        ] {
            assert_eq!(PatternType::try_from(byte).unwrap(), expected);
        }
        assert!(PatternType::try_from(0x08).is_err());
    }

    #[test]
    fn test_motion_type_try_from_all() {
        for (byte, expected) in [
            (0x00, MotionType::None),
            (0x01, MotionType::Linear),
            (0x02, MotionType::Easing),
            (0x03, MotionType::Oscillate),
            (0x04, MotionType::Path),
            (0x05, MotionType::Physics),
        ] {
            assert_eq!(MotionType::try_from(byte).unwrap(), expected);
        }
        assert!(MotionType::try_from(0x06).is_err());
    }

    #[test]
    fn test_roi_type_try_from_all() {
        for (byte, expected) in [
            (0x00, RoiType::General),
            (0x01, RoiType::Face),
            (0x02, RoiType::Text),
            (0x03, RoiType::Edge),
            (0x04, RoiType::Motion),
            (0x05, RoiType::Custom),
        ] {
            assert_eq!(RoiType::try_from(byte).unwrap(), expected);
        }
        assert!(RoiType::try_from(0x06).is_err());
    }

    #[test]
    fn test_sync_command_try_from_all() {
        for (byte, expected) in [
            (0x01, SyncCommand::RequestKeyframe),
            (0x02, SyncCommand::Ack),
            (0x03, SyncCommand::Nack),
            (0x04, SyncCommand::EndOfStream),
            (0x05, SyncCommand::BitrateAdjust),
            (0x06, SyncCommand::QualityChange),
            (0x07, SyncCommand::Ping),
            (0x08, SyncCommand::Pong),
        ] {
            assert_eq!(SyncCommand::try_from(byte).unwrap(), expected);
        }
        assert!(SyncCommand::try_from(0x00).is_err());
        assert!(SyncCommand::try_from(0x09).is_err());
    }

    #[test]
    fn test_stream_stats_update_packet() {
        let mut stats = StreamStats::new();
        stats.update_packet(PacketType::IPacket, 1000);
        stats.update_packet(PacketType::DPacket, 200);
        stats.update_packet(PacketType::DPacket, 300);
        stats.update_packet(PacketType::CPacket, 50);
        stats.update_packet(PacketType::SPacket, 16);

        assert_eq!(stats.total_bytes, 1566);
        assert_eq!(stats.total_packets, 5);
        assert_eq!(stats.i_packets, 1);
        assert_eq!(stats.d_packets, 2);
        assert_eq!(stats.c_packets, 1);
        assert_eq!(stats.s_packets, 1);
    }

    #[test]
    fn test_point_new_and_origin() {
        let p = Point::origin();
        assert_eq!(p.x, 0);
        assert_eq!(p.y, 0);
        let p2 = Point::new(-5, 10);
        assert_eq!(p2.x, -5);
        assert_eq!(p2.y, 10);
    }

    #[test]
    fn test_asp_error_display() {
        let err = AspError::InvalidPacketType(0xAB);
        let msg = err.to_string();
        // The number 171 (== 0xAB) should appear in the message
        assert!(msg.contains("171"), "Expected '171' in '{msg}'");

        let err2 = AspError::ChecksumMismatch {
            expected: 0xDEAD_BEEF,
            got: 0x1234_5678,
        };
        let msg2 = err2.to_string();
        assert!(
            msg2.contains("deadbeef") || msg2.contains("DEADBEEF"),
            "Expected hex in '{msg2}'"
        );
    }
}
