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
            0x01 => Ok(PacketType::IPacket),
            0x02 => Ok(PacketType::DPacket),
            0x03 => Ok(PacketType::CPacket),
            0x04 => Ok(PacketType::SPacket),
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
            0x00 => Ok(PatternType::Solid),
            0x01 => Ok(PatternType::GradientLinear),
            0x02 => Ok(PatternType::GradientRadial),
            0x03 => Ok(PatternType::Noise),
            0x04 => Ok(PatternType::Texture),
            0x05 => Ok(PatternType::Dct),
            0x06 => Ok(PatternType::Periodic),
            0x07 => Ok(PatternType::Complex),
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
            0x00 => Ok(MotionType::None),
            0x01 => Ok(MotionType::Linear),
            0x02 => Ok(MotionType::Easing),
            0x03 => Ok(MotionType::Oscillate),
            0x04 => Ok(MotionType::Path),
            0x05 => Ok(MotionType::Physics),
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
            0x00 => Ok(RoiType::General),
            0x01 => Ok(RoiType::Face),
            0x02 => Ok(RoiType::Text),
            0x03 => Ok(RoiType::Edge),
            0x04 => Ok(RoiType::Motion),
            0x05 => Ok(RoiType::Custom),
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
            0x00 => Ok(QualityLevel::Low),
            0x01 => Ok(QualityLevel::Medium),
            0x02 => Ok(QualityLevel::High),
            0x03 => Ok(QualityLevel::Ultra),
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
            0x01 => Ok(SyncCommand::RequestKeyframe),
            0x02 => Ok(SyncCommand::Ack),
            0x03 => Ok(SyncCommand::Nack),
            0x04 => Ok(SyncCommand::EndOfStream),
            0x05 => Ok(SyncCommand::BitrateAdjust),
            0x06 => Ok(SyncCommand::QualityChange),
            0x07 => Ok(SyncCommand::Ping),
            0x08 => Ok(SyncCommand::Pong),
            _ => Err(AspError::InvalidSyncCommand(value)),
        }
    }
}

/// ASP Error types
#[derive(Debug, Clone, thiserror::Error)]
pub enum AspError {
    #[error("Invalid magic bytes")]
    InvalidMagic,

    #[error("Invalid packet type: {0}")]
    InvalidPacketType(u8),

    #[error("Invalid pattern type: {0}")]
    InvalidPatternType(u8),

    #[error("Invalid motion type: {0}")]
    InvalidMotionType(u8),

    #[error("Invalid ROI type: {0}")]
    InvalidRoiType(u8),

    #[error("Invalid quality level: {0}")]
    InvalidQualityLevel(u8),

    #[error("Invalid sync command: {0}")]
    InvalidSyncCommand(u8),

    #[error("Packet too large: {size} > {max}")]
    PacketTooLarge { size: usize, max: usize },

    #[error("Incomplete packet: expected {expected}, got {got}")]
    IncompletePacket { expected: usize, got: usize },

    #[error("Checksum mismatch: expected {expected:08x}, got {got:08x}")]
    ChecksumMismatch { expected: u32, got: u32 },

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Deserialization error: {0}")]
    DeserializationError(String),

    #[error("Encoder error: {0}")]
    EncoderError(String),

    #[error("Decoder error: {0}")]
    DecoderError(String),

    #[error("Invalid frame dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },

    #[error("Sequence number mismatch: expected {expected}, got {got}")]
    SequenceMismatch { expected: u32, got: u32 },
}

/// Result type for ASP operations
pub type AspResult<T> = Result<T, AspError>;

/// Color in RGB format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Color {
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    pub const fn black() -> Self {
        Self::new(0, 0, 0)
    }

    pub const fn white() -> Self {
        Self::new(255, 255, 255)
    }

    pub fn to_array(self) -> [u8; 3] {
        [self.r, self.g, self.b]
    }

    pub fn from_array(arr: [u8; 3]) -> Self {
        Self::new(arr[0], arr[1], arr[2])
    }

    /// Calculate luminance (Y component in YUV)
    pub fn luminance(self) -> f32 {
        0.299 * self.r as f32 + 0.587 * self.g as f32 + 0.114 * self.b as f32
    }
}

/// 2D point with integer coordinates
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Point {
    pub x: i32,
    pub y: i32,
}

impl Point {
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    pub const fn origin() -> Self {
        Self::new(0, 0)
    }
}

/// Rectangle definition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl Rect {
    pub const fn new(x: u32, y: u32, width: u32, height: u32) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn area(&self) -> u64 {
        self.width as u64 * self.height as u64
    }

    pub fn contains(&self, x: u32, y: u32) -> bool {
        x >= self.x && x < self.x + self.width && y >= self.y && y < self.y + self.height
    }

    pub fn intersects(&self, other: &Rect) -> bool {
        self.x < other.x + other.width
            && self.x + self.width > other.x
            && self.y < other.y + other.height
            && self.y + self.height > other.y
    }
}

/// Motion vector (16 bytes, cache-line optimized)
///
/// Layout: [block_x: u16, block_y: u16, dx: i16, dy: i16, sad: u32, _pad: u32]
/// For 4K video (3840x2160), 16x16 blocks = 240x135 = 32,400 blocks
/// u16 range (0-65535) is sufficient for block indices.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
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
    pub _reserved: u32,
}

impl MotionVector {
    #[inline]
    pub const fn new(block_x: u16, block_y: u16, dx: i16, dy: i16, sad: u32) -> Self {
        Self {
            block_x,
            block_y,
            dx,
            dy,
            sad,
            _reserved: 0,
        }
    }

    /// Create from u32 coordinates (with truncation for legacy compatibility)
    #[inline]
    pub const fn from_u32(block_x: u32, block_y: u32, dx: i16, dy: i16, sad: u32) -> Self {
        Self {
            block_x: block_x as u16,
            block_y: block_y as u16,
            dx,
            dy,
            sad,
            _reserved: 0,
        }
    }

    /// Check if this is a zero motion vector
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.dx == 0 && self.dy == 0
    }

    /// Calculate motion magnitude
    #[inline]
    pub fn magnitude(&self) -> f32 {
        ((self.dx as f32).powi(2) + (self.dy as f32).powi(2)).sqrt()
    }

    /// Convert to compact format (for bandwidth optimization)
    #[inline]
    pub fn to_compact(&self) -> Option<MotionVectorCompact> {
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
/// Uses only 2 bytes (vs 16 bytes for full MotionVector).
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
    #[inline]
    pub const fn new(dx: i8, dy: i8) -> Self {
        Self { dx, dy }
    }

    #[inline]
    pub const fn zero() -> Self {
        Self { dx: 0, dy: 0 }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.dx == 0 && self.dy == 0
    }

    /// Expand to full MotionVector with block position and SAD
    #[inline]
    pub fn expand(self, block_x: u16, block_y: u16, sad: u32) -> MotionVector {
        MotionVector {
            block_x,
            block_y,
            dx: self.dx as i16,
            dy: self.dy as i16,
            sad,
            _reserved: 0,
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
    #[default]
    Linear = 0x00,
    EaseIn = 0x01,
    EaseOut = 0x02,
    EaseInOut = 0x03,
    Bounce = 0x04,
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
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update_packet(&mut self, packet_type: PacketType, size: usize) {
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
}
