//! ASP Packet Header
//!
//! Defines the binary header format for ASP packets.
//! Header is 16 bytes fixed size for efficient parsing.

use crate::types::{AspError, AspResult, PacketType, ASP_MAGIC, ASP_VERSION};
use serde::{Deserialize, Serialize};

/// ASP Packet Header (16 bytes)
///
/// Binary layout:
/// ```text
/// Offset  Size  Field
/// 0       4     Magic ("ASP1")
/// 4       1     Version
/// 5       1     Packet Type
/// 6       2     Flags
/// 8       4     Sequence Number
/// 12      4     Payload Length
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(C)] // C-compatible layout for raw pointer access
pub struct AspPacketHeader {
    /// Protocol version
    pub version: u8,
    /// Packet type
    pub packet_type: PacketType,
    /// Flags (reserved for future use)
    pub flags: u16,
    /// Sequence number
    pub sequence: u32,
    /// Payload length in bytes
    pub payload_length: u32,
}

impl AspPacketHeader {
    /// Header size in bytes
    pub const SIZE: usize = 16;

    /// Create a new packet header
    #[inline]
    pub fn new(packet_type: PacketType, sequence: u32, payload_length: u32) -> Self {
        Self {
            version: ASP_VERSION,
            packet_type,
            flags: 0,
            sequence,
            payload_length,
        }
    }

    /// Serialize header to bytes
    #[inline]
    pub fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];

        // Magic bytes
        buf[0..4].copy_from_slice(&ASP_MAGIC);

        // Version
        buf[4] = self.version;

        // Packet type
        buf[5] = self.packet_type as u8;

        // Flags (big-endian)
        buf[6..8].copy_from_slice(&self.flags.to_be_bytes());

        // Sequence number (big-endian)
        buf[8..12].copy_from_slice(&self.sequence.to_be_bytes());

        // Payload length (big-endian)
        buf[12..16].copy_from_slice(&self.payload_length.to_be_bytes());

        buf
    }

    /// Serialize header to bytes using raw pointers (Zero Overhead)
    ///
    /// # Safety
    /// The caller must ensure that `ptr` points to a valid memory region
    /// of at least `Self::SIZE` (16) bytes.
    #[inline(always)]
    pub unsafe fn write_to_ptr(&self, ptr: *mut u8) {
        // Magic bytes (4 bytes)
        std::ptr::copy_nonoverlapping(ASP_MAGIC.as_ptr(), ptr, 4);

        // Version (1 byte)
        *ptr.add(4) = self.version;

        // Packet type (1 byte)
        *ptr.add(5) = self.packet_type as u8;

        // Flags (2 bytes, big-endian)
        let flags_be = self.flags.to_be_bytes();
        std::ptr::copy_nonoverlapping(flags_be.as_ptr(), ptr.add(6), 2);

        // Sequence number (4 bytes, big-endian)
        let seq_be = self.sequence.to_be_bytes();
        std::ptr::copy_nonoverlapping(seq_be.as_ptr(), ptr.add(8), 4);

        // Payload length (4 bytes, big-endian)
        let len_be = self.payload_length.to_be_bytes();
        std::ptr::copy_nonoverlapping(len_be.as_ptr(), ptr.add(12), 4);
    }

    /// Deserialize header from bytes
    #[inline]
    pub fn from_bytes(data: &[u8]) -> AspResult<Self> {
        if data.len() < Self::SIZE {
            return Err(AspError::IncompletePacket {
                expected: Self::SIZE,
                got: data.len(),
            });
        }

        // Check magic
        if data[0..4] != ASP_MAGIC {
            return Err(AspError::InvalidMagic);
        }

        // Version
        let version = data[4];

        // Packet type
        let packet_type = PacketType::try_from(data[5])?;

        // Flags
        let flags = u16::from_be_bytes([data[6], data[7]]);

        // Sequence number
        let sequence = u32::from_be_bytes([data[8], data[9], data[10], data[11]]);

        // Payload length
        let payload_length = u32::from_be_bytes([data[12], data[13], data[14], data[15]]);

        Ok(Self {
            version,
            packet_type,
            flags,
            sequence,
            payload_length,
        })
    }

    /// Check if this is a keyframe packet
    #[inline]
    pub fn is_keyframe(&self) -> bool {
        self.packet_type == PacketType::IPacket
    }

    /// Check if compression flag is set
    #[inline]
    pub fn is_compressed(&self) -> bool {
        self.flags & 0x0001 != 0
    }

    /// Set compression flag
    #[inline]
    pub fn set_compressed(&mut self, compressed: bool) {
        if compressed {
            self.flags |= 0x0001;
        } else {
            self.flags &= !0x0001;
        }
    }

    /// Check if encryption flag is set
    pub fn is_encrypted(&self) -> bool {
        self.flags & 0x0002 != 0
    }

    /// Set encryption flag
    pub fn set_encrypted(&mut self, encrypted: bool) {
        if encrypted {
            self.flags |= 0x0002;
        } else {
            self.flags &= !0x0002;
        }
    }

    /// Check if FEC (Forward Error Correction) flag is set
    pub fn has_fec(&self) -> bool {
        self.flags & 0x0004 != 0
    }

    /// Set FEC flag
    pub fn set_fec(&mut self, fec: bool) {
        if fec {
            self.flags |= 0x0004;
        } else {
            self.flags &= !0x0004;
        }
    }

    /// Calculate total packet size (header + payload)
    #[inline]
    pub fn total_size(&self) -> usize {
        Self::SIZE + self.payload_length as usize
    }
}

impl Default for AspPacketHeader {
    #[inline]
    fn default() -> Self {
        Self::new(PacketType::DPacket, 0, 0)
    }
}

/// CRC32 IEEE polynomial
const CRC32_POLYNOMIAL: u32 = 0xEDB88320;

/// Generate CRC32 lookup table at compile time
const fn generate_crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut crc = i as u32;
        let mut j = 0;
        while j < 8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ CRC32_POLYNOMIAL
            } else {
                crc >> 1
            };
            j += 1;
        }
        table[i] = crc;
        i += 1;
    }
    table
}

/// Compile-time generated CRC32 lookup table (256 entries × 4 bytes = 1KB)
static CRC32_TABLE: [u32; 256] = generate_crc32_table();

/// Calculate CRC32 checksum for data
///
/// Uses a compile-time generated lookup table for maximum performance.
/// ~8x faster than table-less implementation for typical packet sizes.
#[inline]
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xFFFFFFFF_u32;

    for &byte in data {
        let index = ((crc ^ byte as u32) & 0xFF) as usize;
        crc = (crc >> 8) ^ CRC32_TABLE[index];
    }

    !crc
}

/// Verify CRC32 checksum
#[inline]
pub fn verify_crc32(data: &[u8], expected: u32) -> bool {
    crc32(data) == expected
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_header_serialization() {
        let header = AspPacketHeader::new(PacketType::IPacket, 42, 1024);
        let bytes = header.to_bytes();

        assert_eq!(&bytes[0..4], &ASP_MAGIC);
        assert_eq!(bytes[4], ASP_VERSION);
        assert_eq!(bytes[5], PacketType::IPacket as u8);

        let restored = AspPacketHeader::from_bytes(&bytes).unwrap();
        assert_eq!(header, restored);
    }

    #[test]
    fn test_header_flags() {
        let mut header = AspPacketHeader::new(PacketType::DPacket, 0, 0);

        assert!(!header.is_compressed());
        header.set_compressed(true);
        assert!(header.is_compressed());

        assert!(!header.is_encrypted());
        header.set_encrypted(true);
        assert!(header.is_encrypted());

        assert!(!header.has_fec());
        header.set_fec(true);
        assert!(header.has_fec());

        // All flags should be independent
        assert!(header.is_compressed());
        assert!(header.is_encrypted());
        assert!(header.has_fec());
    }

    #[test]
    fn test_invalid_magic() {
        let mut bytes = [0u8; AspPacketHeader::SIZE];
        bytes[0..4].copy_from_slice(b"XXXX");

        let result = AspPacketHeader::from_bytes(&bytes);
        assert!(matches!(result, Err(AspError::InvalidMagic)));
    }

    #[test]
    fn test_incomplete_header() {
        let bytes = [0u8; 8]; // Too short

        let result = AspPacketHeader::from_bytes(&bytes);
        assert!(matches!(result, Err(AspError::IncompletePacket { .. })));
    }

    #[test]
    fn test_crc32() {
        let data = b"Hello, ALICE!";
        let checksum = crc32(data);
        assert!(verify_crc32(data, checksum));
        assert!(!verify_crc32(data, checksum ^ 1));
    }

    #[test]
    fn test_total_size() {
        let header = AspPacketHeader::new(PacketType::IPacket, 0, 1000);
        assert_eq!(header.total_size(), AspPacketHeader::SIZE + 1000);
    }

    #[test]
    fn test_write_to_ptr() {
        let header = AspPacketHeader::new(PacketType::IPacket, 42, 1024);

        // Use to_bytes() as reference
        let expected = header.to_bytes();

        // Test write_to_ptr()
        let mut buffer = [0u8; AspPacketHeader::SIZE];
        unsafe {
            header.write_to_ptr(buffer.as_mut_ptr());
        }

        assert_eq!(
            buffer, expected,
            "write_to_ptr should produce same output as to_bytes"
        );
    }

    #[test]
    fn test_write_to_ptr_all_packet_types() {
        for packet_type in [
            PacketType::IPacket,
            PacketType::DPacket,
            PacketType::CPacket,
            PacketType::SPacket,
        ] {
            let header = AspPacketHeader::new(packet_type, 12345, 65535);

            let expected = header.to_bytes();
            let mut buffer = [0u8; AspPacketHeader::SIZE];
            unsafe {
                header.write_to_ptr(buffer.as_mut_ptr());
            }

            assert_eq!(buffer, expected, "Failed for {:?}", packet_type);
        }
    }
}
