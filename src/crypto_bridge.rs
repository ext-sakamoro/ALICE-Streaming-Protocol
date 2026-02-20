//! ALICE-Crypto bridge: Encrypt/decrypt ASP packets
//!
//! Wraps ASP packet serialization with XChaCha20-Poly1305 authenticated
//! encryption for secure streaming (DRM, private channels).
//!
//! # Pipeline
//!
//! ```text
//! AspPacket → to_bytes() → seal(key, bytes) → encrypted payload
//! encrypted payload → open(key, payload) → from_bytes() → AspPacket
//! ```

use alice_crypto::{self as crypto, Key};

use crate::AspPacket;

/// Encrypted ASP packet.
#[derive(Debug, Clone)]
pub struct SealedPacket {
    /// Encrypted data (nonce + ciphertext + auth tag)
    pub data: Vec<u8>,
    /// Original packet sequence number (unencrypted, for routing)
    pub sequence: u32,
}

/// Encrypt an ASP packet.
///
/// The sequence number is preserved in cleartext for routing purposes.
/// The full packet payload is authenticated and encrypted.
pub fn seal_packet(packet: &AspPacket, key: &Key) -> Result<SealedPacket, String> {
    let bytes = packet.to_bytes().map_err(|e| format!("serialize: {}", e))?;
    let sealed = crypto::seal(key, &bytes).map_err(|e| format!("encrypt: {:?}", e))?;
    Ok(SealedPacket {
        sequence: packet.sequence(),
        data: sealed,
    })
}

/// Decrypt a sealed ASP packet.
pub fn open_packet(sealed: &SealedPacket, key: &Key) -> Result<AspPacket, String> {
    let bytes = crypto::open(key, &sealed.data).map_err(|e| format!("decrypt: {:?}", e))?;
    AspPacket::from_bytes(&bytes).map_err(|e| format!("deserialize: {}", e))
}

/// Compute a content hash for an ASP packet (without decryption).
///
/// Useful for deduplication of encrypted streams.
pub fn packet_content_hash(sealed: &SealedPacket) -> crypto::Hash {
    crypto::hash(&sealed.data)
}

/// Derive a stream encryption key from a channel ID and secret.
pub fn derive_stream_key(channel_id: &str, secret: &[u8]) -> Key {
    let context = format!("alice-asp-stream-v1:{}", channel_id);
    let raw = crypto::derive_key(&context, secret);
    Key::from_bytes(raw)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Color, IPacketPayload, Rect, RegionDescriptor};

    fn make_test_packet() -> AspPacket {
        let mut payload = IPacketPayload::new(640, 480, 30.0);
        payload.add_region(RegionDescriptor::solid(
            Rect::new(0, 0, 640, 480),
            Color::new(100, 100, 100),
        ));
        AspPacket::create_i_packet(1, payload).unwrap()
    }

    #[test]
    fn test_seal_open_roundtrip() {
        let key = Key::generate().unwrap();
        let packet = make_test_packet();

        let sealed = seal_packet(&packet, &key).unwrap();
        assert_eq!(sealed.sequence, 1);

        let recovered = open_packet(&sealed, &key).unwrap();
        assert_eq!(recovered.sequence(), 1);
    }

    #[test]
    fn test_wrong_key_fails() {
        let key1 = Key::generate().unwrap();
        let key2 = Key::generate().unwrap();
        let packet = make_test_packet();

        let sealed = seal_packet(&packet, &key1).unwrap();
        assert!(open_packet(&sealed, &key2).is_err());
    }
}
