//! ALICE Media Stack Integration
//!
//! Integrates ALICE-Codec (video) and ALICE-Voice (audio) into the ASP transport.
//!
//! # Features
//!
//! - `codec`: Enable ALICE-Codec (wavelet + rANS video compression)
//! - `voice`: Enable ALICE-Voice (LPC parametric audio compression)
//! - `media-stack`: Enable both (full A/V pipeline)
//!
//! # Architecture
//!
//! ```text
//! Camera → ALICE-Codec (wavelet + rANS)
//!               ↓
//! Mic → ALICE-Voice (LPC parametric)
//!               ↓
//!          ALICE-Streaming-Protocol (mux + FlatBuffers transport)
//!               ↓
//!            Network
//! ```

#[cfg(feature = "codec")]
pub mod video_codec;

#[cfg(feature = "voice")]
pub mod voice_codec;

#[cfg(feature = "codec")]
pub use video_codec::*;

#[cfg(feature = "voice")]
pub use voice_codec::*;
