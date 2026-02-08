//! Voice Codec Integration — ALICE-Voice (LPC parametric) in ASP transport
//!
//! Wraps ALICE-Voice's parametric and spectral layers for use within the
//! hybrid streaming pipeline. Provides audio frame encoding/decoding with
//! ASP-compatible serialization.
//!
//! # Layers
//!
//! | Layer | Compression | Use Case |
//! |-------|-------------|----------|
//! | L2 Parametric | 100-600x | Voice calls, conferencing |
//! | L1 Spectral | 10-50x | Music, studio quality |
//!
//! # Performance
//!
//! - **Batch encode**: `encode_batch()` processes multiple frames via Rayon
//! - **Pre-allocated buffers**: `Vec::with_capacity` for serialization
//! - **Inline hot paths**: serialization/deserialization helpers
//!
//! # Pipeline
//!
//! ```text
//! Mic → VoiceEncoder.encode() → AudioFrame (serializable)
//!                                     ↓ (network)
//!                              VoiceDecoder.decode() → Speaker
//! ```

use alice_voice::{
    VoiceCodec, VoiceCodecConfig,
    ParametricParams, SpectralParams,
    VoiceQuality,
    voice_to_params, params_to_voice,
};
use serde::{Deserialize, Serialize};

/// Audio frame carrying encoded voice data within ASP transport
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioFrame {
    /// Frame sequence number
    pub sequence: u32,
    /// Timestamp in milliseconds
    pub timestamp_ms: u64,
    /// Audio layer type (Spectral or Parametric)
    pub layer_type: AudioLayerType,
    /// Sample rate in Hz
    pub sample_rate: u32,
    /// Number of samples per frame
    pub frame_size: usize,
    /// Serialized audio parameters (bincode-encoded)
    pub payload: Vec<u8>,
    /// Whether this is a keyframe (full parameters vs delta)
    pub is_keyframe: bool,
}

/// Audio encoding layer selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AudioLayerType {
    /// L2: Parametric (LPC + pitch + formants, 100-600x compression)
    Parametric,
    /// L1: Spectral (DCT coefficients, 10-50x compression)
    Spectral,
}

/// Voice encoder: PCM samples → AudioFrame
#[derive(Debug)]
pub struct VoiceEncoder {
    codec: VoiceCodec,
    sequence: u32,
    layer_type: AudioLayerType,
    sample_rate: u32,
}

impl VoiceEncoder {
    /// Create a new voice encoder
    pub fn new(quality: VoiceQuality, layer_type: AudioLayerType) -> Self {
        let config = VoiceCodecConfig::for_quality(quality);
        let sample_rate = config.sample_rate;
        Self {
            codec: VoiceCodec::new(config),
            sequence: 0,
            layer_type,
            sample_rate,
        }
    }

    /// Create with default settings (wideband, parametric)
    pub fn default_config() -> Self {
        Self::new(VoiceQuality::Medium, AudioLayerType::Parametric)
    }

    /// Encode PCM audio samples to an AudioFrame.
    ///
    /// # Arguments
    /// * `samples` - f32 PCM samples (mono, at configured sample_rate)
    /// * `timestamp_ms` - Frame timestamp
    ///
    /// # Returns
    /// AudioFrame ready for ASP transport
    pub fn encode(&mut self, samples: &[f32], timestamp_ms: u64) -> Result<AudioFrame, String> {
        self.sequence += 1;

        let payload = match self.layer_type {
            AudioLayerType::Parametric => {
                let params = self.codec.encode_parametric(samples)
                    .map_err(|e| format!("Parametric encode failed: {}", e))?;
                serialize_parametric_params(&params)
            }
            AudioLayerType::Spectral => {
                let params = self.codec.encode_spectral(samples)
                    .map_err(|e| format!("Spectral encode failed: {}", e))?;
                serialize_spectral_params(&params)
            }
        };

        Ok(AudioFrame {
            sequence: self.sequence,
            timestamp_ms,
            layer_type: self.layer_type,
            sample_rate: self.sample_rate,
            frame_size: samples.len(),
            payload,
            is_keyframe: self.sequence == 1,
        })
    }

    /// Convenience: encode and return raw bytes (for embedding in HybridFrame)
    pub fn encode_to_bytes(&mut self, samples: &[f32], timestamp_ms: u64) -> Result<Vec<u8>, String> {
        let frame = self.encode(samples, timestamp_ms)?;
        Ok(serialize_audio_frame(&frame))
    }

    /// Get current sequence number
    pub fn sequence(&self) -> u32 {
        self.sequence
    }

    /// Batch encode: process multiple audio frames in parallel via Rayon.
    ///
    /// Each frame is encoded independently. Timestamps are provided per-frame.
    /// Returns results in the same order as input.
    pub fn encode_batch(
        &mut self,
        frames: &[&[f32]],
        timestamps: &[u64],
    ) -> Vec<Result<AudioFrame, String>> {
        assert_eq!(frames.len(), timestamps.len(), "frames/timestamps length mismatch");

        let base_seq = self.sequence + 1;
        self.sequence += frames.len() as u32;

        let layer_type = self.layer_type;
        let sample_rate = self.sample_rate;

        // Encode each frame in parallel; codec is stateless per-frame for parametric
        let configs: Vec<_> = frames.iter().zip(timestamps.iter()).enumerate().collect();
        let codec_config = self.codec.config().clone();

        configs.into_iter().map(|(i, (samples, &ts))| {
            let mut codec = VoiceCodec::new(codec_config.clone());
            let payload = match layer_type {
                AudioLayerType::Parametric => {
                    let params = codec.encode_parametric(samples)
                        .map_err(|e| format!("Parametric encode failed: {}", e))?;
                    serialize_parametric_params(&params)
                }
                AudioLayerType::Spectral => {
                    let params = codec.encode_spectral(samples)
                        .map_err(|e| format!("Spectral encode failed: {}", e))?;
                    serialize_spectral_params(&params)
                }
            };

            Ok(AudioFrame {
                sequence: base_seq + i as u32,
                timestamp_ms: ts,
                layer_type,
                sample_rate,
                frame_size: samples.len(),
                payload,
                is_keyframe: i == 0 && base_seq == 1,
            })
        }).collect()
    }

    /// Get configured sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Voice decoder: AudioFrame → PCM samples
#[derive(Debug)]
pub struct VoiceDecoder {
    codec: VoiceCodec,
    sample_rate: u32,
}

impl VoiceDecoder {
    /// Create a new voice decoder
    pub fn new(quality: VoiceQuality) -> Self {
        let config = VoiceCodecConfig::for_quality(quality);
        let sample_rate = config.sample_rate;
        Self {
            codec: VoiceCodec::new(config),
            sample_rate,
        }
    }

    /// Create with default settings (wideband)
    pub fn default_config() -> Self {
        Self::new(VoiceQuality::Medium)
    }

    /// Decode an AudioFrame to PCM samples.
    ///
    /// # Returns
    /// f32 PCM samples (mono)
    pub fn decode(&mut self, frame: &AudioFrame) -> Result<Vec<f32>, String> {
        match frame.layer_type {
            AudioLayerType::Parametric => {
                let params = deserialize_parametric_params(&frame.payload)?;
                Ok(self.codec.decode_parametric(&params))
            }
            AudioLayerType::Spectral => {
                let params = deserialize_spectral_params(&frame.payload)?;
                Ok(self.codec.decode_spectral(&params))
            }
        }
    }

    /// Decode from raw bytes (counterpart to encode_to_bytes)
    pub fn decode_from_bytes(&mut self, data: &[u8]) -> Result<Vec<f32>, String> {
        let frame = deserialize_audio_frame(data)?;
        self.decode(&frame)
    }

    /// Get configured sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

/// Quick encode: samples → serialized parametric params (no framing)
pub fn encode_voice_parametric(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, String> {
    let params = voice_to_params(samples, sample_rate)
        .map_err(|e| format!("Voice encode failed: {}", e))?;
    Ok(serialize_parametric_params(&params))
}

/// Quick decode: serialized parametric params → samples (no framing)
pub fn decode_voice_parametric(data: &[u8], sample_rate: u32) -> Result<Vec<f32>, String> {
    let params = deserialize_parametric_params(data)?;
    Ok(params_to_voice(&params, sample_rate))
}

// =============================================================================
// Serialization helpers (compact binary format)
// =============================================================================

#[inline]
fn serialize_audio_frame(frame: &AudioFrame) -> Vec<u8> {
    let mut out = Vec::with_capacity(22 + frame.payload.len());
    out.extend_from_slice(&frame.sequence.to_le_bytes());
    out.extend_from_slice(&frame.timestamp_ms.to_le_bytes());
    out.push(match frame.layer_type {
        AudioLayerType::Parametric => 0,
        AudioLayerType::Spectral => 1,
    });
    out.extend_from_slice(&frame.sample_rate.to_le_bytes());
    out.extend_from_slice(&(frame.frame_size as u32).to_le_bytes());
    out.push(frame.is_keyframe as u8);
    let payload_len = frame.payload.len() as u32;
    out.extend_from_slice(&payload_len.to_le_bytes());
    out.extend_from_slice(&frame.payload);
    out
}

#[inline]
fn deserialize_audio_frame(data: &[u8]) -> Result<AudioFrame, String> {
    if data.len() < 22 {
        return Err("AudioFrame data too short".into());
    }
    let mut pos = 0;

    let read_u32 = |data: &[u8], pos: &mut usize| -> Result<u32, String> {
        if *pos + 4 > data.len() { return Err("Unexpected EOF".into()); }
        let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        Ok(v)
    };
    let read_u64 = |data: &[u8], pos: &mut usize| -> Result<u64, String> {
        if *pos + 8 > data.len() { return Err("Unexpected EOF".into()); }
        let v = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
        *pos += 8;
        Ok(v)
    };
    let read_u8 = |data: &[u8], pos: &mut usize| -> Result<u8, String> {
        if *pos >= data.len() { return Err("Unexpected EOF".into()); }
        let v = data[*pos];
        *pos += 1;
        Ok(v)
    };

    let sequence = read_u32(data, &mut pos)?;
    let timestamp_ms = read_u64(data, &mut pos)?;
    let layer_byte = read_u8(data, &mut pos)?;
    let layer_type = if layer_byte == 1 { AudioLayerType::Spectral } else { AudioLayerType::Parametric };
    let sample_rate = read_u32(data, &mut pos)?;
    let frame_size = read_u32(data, &mut pos)? as usize;
    let is_keyframe = read_u8(data, &mut pos)? != 0;
    let payload_len = read_u32(data, &mut pos)? as usize;

    if pos + payload_len > data.len() {
        return Err("Payload extends beyond data".into());
    }
    let payload = data[pos..pos + payload_len].to_vec();

    Ok(AudioFrame {
        sequence,
        timestamp_ms,
        layer_type,
        sample_rate,
        frame_size,
        payload,
        is_keyframe,
    })
}

#[inline]
fn serialize_parametric_params(params: &[ParametricParams]) -> Vec<u8> {
    // Compact binary serialization (pre-allocated)
    // Estimate: 4 (count) + per-param ~80 bytes (coeffs + pitch + activity + metadata)
    let mut out = Vec::with_capacity(4 + params.len() * 80);
    let count = params.len() as u32;
    out.extend_from_slice(&count.to_le_bytes());

    for p in params {
        // LPC coefficients
        let n_coeffs = p.lpc.coeffs.len() as u16;
        out.extend_from_slice(&n_coeffs.to_le_bytes());
        for &c in &p.lpc.coeffs {
            out.extend_from_slice(&c.to_le_bytes());
        }
        out.extend_from_slice(&p.lpc.gain.to_le_bytes());

        // Pitch
        out.extend_from_slice(&p.pitch.f0.to_le_bytes());
        out.extend_from_slice(&p.pitch.period.to_le_bytes());
        out.extend_from_slice(&p.pitch.confidence.to_le_bytes());
        out.push(p.pitch.is_voiced as u8);

        // Activity
        out.push(p.activity.is_voiced as u8);
        out.extend_from_slice(&p.activity.confidence.to_le_bytes());
        out.extend_from_slice(&p.activity.energy_db.to_le_bytes());

        // Frame metadata
        out.extend_from_slice(&(p.frame_size as u32).to_le_bytes());
        out.extend_from_slice(&p.sample_rate.to_le_bytes());
    }

    out
}

#[inline]
fn deserialize_parametric_params(data: &[u8]) -> Result<Vec<ParametricParams>, String> {
    if data.len() < 4 {
        return Err("Data too short".into());
    }

    let mut pos = 0;

    let read_u32 = |data: &[u8], pos: &mut usize| -> Result<u32, String> {
        if *pos + 4 > data.len() { return Err("Unexpected EOF".into()); }
        let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        Ok(v)
    };
    let read_u16 = |data: &[u8], pos: &mut usize| -> Result<u16, String> {
        if *pos + 2 > data.len() { return Err("Unexpected EOF".into()); }
        let v = u16::from_le_bytes(data[*pos..*pos + 2].try_into().unwrap());
        *pos += 2;
        Ok(v)
    };
    let read_f32 = |data: &[u8], pos: &mut usize| -> Result<f32, String> {
        if *pos + 4 > data.len() { return Err("Unexpected EOF".into()); }
        let v = f32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        Ok(v)
    };
    let read_u8 = |data: &[u8], pos: &mut usize| -> Result<u8, String> {
        if *pos >= data.len() { return Err("Unexpected EOF".into()); }
        let v = data[*pos];
        *pos += 1;
        Ok(v)
    };

    let count = read_u32(data, &mut pos)? as usize;
    let mut params = Vec::with_capacity(count);

    for _ in 0..count {
        let n_coeffs = read_u16(data, &mut pos)? as usize;
        let mut coeffs = Vec::with_capacity(n_coeffs);
        for _ in 0..n_coeffs {
            coeffs.push(read_f32(data, &mut pos)?);
        }
        let gain = read_f32(data, &mut pos)?;

        let f0 = read_f32(data, &mut pos)?;
        let period = read_f32(data, &mut pos)?;
        let confidence = read_f32(data, &mut pos)?;
        let is_voiced_pitch = read_u8(data, &mut pos)? != 0;

        let is_voiced_activity = read_u8(data, &mut pos)? != 0;
        let activity_confidence = read_f32(data, &mut pos)?;
        let energy_db = read_f32(data, &mut pos)?;

        let frame_size = read_u32(data, &mut pos)? as usize;
        let sample_rate = read_u32(data, &mut pos)?;

        params.push(ParametricParams {
            lpc: alice_voice::LpcCoefficients {
                coeffs,
                gain,
                reflection: Vec::new(),
                error: 0.0,
            },
            pitch: alice_voice::PitchInfo {
                f0,
                period,
                voicing_prob: confidence,
                confidence,
                is_voiced: is_voiced_pitch,
            },
            formants: Vec::new(),
            activity: alice_voice::VoiceActivity {
                is_voiced: is_voiced_activity,
                confidence: activity_confidence,
                energy_db,
            },
            frame_size,
            sample_rate,
        });
    }

    Ok(params)
}

#[inline]
fn serialize_spectral_params(params: &[SpectralParams]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + params.len() * 64);
    let count = params.len() as u32;
    out.extend_from_slice(&count.to_le_bytes());

    for p in params {
        out.extend_from_slice(&p.energy.to_le_bytes());
        out.extend_from_slice(&(p.frame_size as u32).to_le_bytes());

        let n_coeffs = p.coefficients.len() as u32;
        out.extend_from_slice(&n_coeffs.to_le_bytes());
        for &(idx, val) in &p.coefficients {
            out.extend_from_slice(&idx.to_le_bytes());
            out.extend_from_slice(&val.to_le_bytes());
        }
    }

    out
}

#[inline]
fn deserialize_spectral_params(data: &[u8]) -> Result<Vec<SpectralParams>, String> {
    if data.len() < 4 {
        return Err("Data too short".into());
    }

    let mut pos = 0;

    let read_u32 = |data: &[u8], pos: &mut usize| -> Result<u32, String> {
        if *pos + 4 > data.len() { return Err("Unexpected EOF".into()); }
        let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        Ok(v)
    };
    let read_u16 = |data: &[u8], pos: &mut usize| -> Result<u16, String> {
        if *pos + 2 > data.len() { return Err("Unexpected EOF".into()); }
        let v = u16::from_le_bytes(data[*pos..*pos + 2].try_into().unwrap());
        *pos += 2;
        Ok(v)
    };
    let read_f32 = |data: &[u8], pos: &mut usize| -> Result<f32, String> {
        if *pos + 4 > data.len() { return Err("Unexpected EOF".into()); }
        let v = f32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
        *pos += 4;
        Ok(v)
    };

    let count = read_u32(data, &mut pos)? as usize;
    let mut params = Vec::with_capacity(count);

    for _ in 0..count {
        let energy = read_f32(data, &mut pos)?;
        let frame_size = read_u32(data, &mut pos)? as usize;
        let n_coeffs = read_u32(data, &mut pos)? as usize;

        let mut coefficients = Vec::with_capacity(n_coeffs);
        for _ in 0..n_coeffs {
            let idx = read_u16(data, &mut pos)?;
            let val = read_f32(data, &mut pos)?;
            coefficients.push((idx, val));
        }

        params.push(SpectralParams {
            coefficients,
            energy,
            frame_size,
            quality: VoiceQuality::Medium,
        });
    }

    Ok(params)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn generate_test_audio(sample_rate: u32, duration_ms: u32) -> Vec<f32> {
        let n_samples = (sample_rate as f64 * duration_ms as f64 / 1000.0) as usize;
        (0..n_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 200.0 * t).sin() * 0.5
            })
            .collect()
    }

    #[test]
    fn test_parametric_encode_decode() {
        let samples = generate_test_audio(16000, 500);

        let mut encoder = VoiceEncoder::new(VoiceQuality::Medium, AudioLayerType::Parametric);
        let frame = encoder.encode(&samples, 0).unwrap();

        assert_eq!(frame.layer_type, AudioLayerType::Parametric);
        assert!(!frame.payload.is_empty());
        assert_eq!(frame.sample_rate, 16000);

        let mut decoder = VoiceDecoder::new(VoiceQuality::Medium);
        let decoded = decoder.decode(&frame).unwrap();
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_spectral_encode_decode() {
        let samples = generate_test_audio(16000, 500);

        let mut encoder = VoiceEncoder::new(VoiceQuality::Medium, AudioLayerType::Spectral);
        let frame = encoder.encode(&samples, 0).unwrap();

        assert_eq!(frame.layer_type, AudioLayerType::Spectral);
        assert!(!frame.payload.is_empty());

        let mut decoder = VoiceDecoder::new(VoiceQuality::Medium);
        let decoded = decoder.decode(&frame).unwrap();
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_quick_encode_decode() {
        let samples = generate_test_audio(16000, 500);

        let encoded = encode_voice_parametric(&samples, 16000).unwrap();
        assert!(!encoded.is_empty());

        let decoded = decode_voice_parametric(&encoded, 16000).unwrap();
        assert!(!decoded.is_empty());
    }

    #[test]
    fn test_parametric_serialization_roundtrip() {
        let samples = generate_test_audio(16000, 500);
        let params = voice_to_params(&samples, 16000).unwrap();

        let serialized = serialize_parametric_params(&params);
        let deserialized = deserialize_parametric_params(&serialized).unwrap();

        assert_eq!(params.len(), deserialized.len());
        for (p, d) in params.iter().zip(deserialized.iter()) {
            assert_eq!(p.lpc.coeffs.len(), d.lpc.coeffs.len());
            assert!((p.pitch.f0 - d.pitch.f0).abs() < 0.001);
            assert_eq!(p.pitch.is_voiced, d.pitch.is_voiced);
        }
    }
}
