//! Video Codec Integration — ALICE-Codec (wavelet + rANS) in ASP transport
//!
//! Wraps ALICE-Codec's encode/decode pipeline for use within the hybrid
//! streaming pipeline. Handles person-region compression via 2D wavelet
//! transform + rANS entropy coding.
//!
//! # Pipeline
//!
//! ```text
//! Encoder: RGB → YCoCg-R → 2D Wavelet → Quantize → rANS → Bitstream
//! Decoder: Bitstream → rANS → Dequantize → Inverse Wavelet → YCoCg-R → RGB
//! ```
//!
//! # Performance
//!
//! - **Rayon parallel**: Y/Co/Cg channels processed in parallel (3-way join)
//! - **Pre-allocated buffers**: Single allocation for encode/decode output
//! - **Inline hot paths**: `coeffs_to_symbols`, `build_histogram`, pack/unpack

use alice_codec::{
    color::RGB,
    rans::FrequencyTable,
    rgb_to_ycocg_r,
    segment::{segment_by_motion, SegmentConfig, SegmentResult},
    ycocg_r_to_rgb, Quantizer, RansDecoder, RansEncoder, Wavelet1D, Wavelet2D,
};
use serde::{Deserialize, Serialize};

/// Video codec configuration for ASP integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoCodecConfig {
    /// Wavelet type: "cdf97" (lossy) or "cdf53" (lossless)
    pub wavelet_type: WaveletType,
    /// Quality level 1-100 (higher = better quality, larger output)
    pub quality: u8,
    /// Target bits per pixel (for RDO, 0.0 = auto from quality)
    pub target_bpp: f64,
}

/// Wavelet type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaveletType {
    /// CDF 9/7 (lossy, JPEG2000 compatible)
    Cdf97,
    /// CDF 5/3 (lossless, perfect reconstruction)
    Cdf53,
    /// Haar (simple, fast)
    Haar,
}

impl Default for VideoCodecConfig {
    fn default() -> Self {
        Self {
            wavelet_type: WaveletType::Cdf97,
            quality: 75,
            target_bpp: 0.0,
        }
    }
}

/// Video encoder: RGB frames → compressed bitstream
#[derive(Debug)]
pub struct VideoEncoder {
    config: VideoCodecConfig,
    wavelet: Wavelet2D,
}

impl VideoEncoder {
    /// Create a new video encoder with the given configuration
    pub fn new(config: VideoCodecConfig) -> Self {
        let w1d = match config.wavelet_type {
            WaveletType::Cdf97 => Wavelet1D::cdf97(),
            WaveletType::Cdf53 => Wavelet1D::cdf53(),
            WaveletType::Haar => Wavelet1D::haar(),
        };
        Self {
            wavelet: Wavelet2D::new(w1d),
            config,
        }
    }

    /// Create with default settings
    pub fn default_config() -> Self {
        Self::new(VideoCodecConfig::default())
    }

    /// Encode a single RGB frame region to compressed bitstream.
    ///
    /// # Arguments
    /// * `rgb_data` - Raw RGB pixel data (R,G,B,R,G,B,...)
    /// * `width` - Frame width in pixels
    /// * `height` - Frame height in pixels
    ///
    /// # Returns
    /// Compressed bitstream bytes
    pub fn encode_frame(&self, rgb_data: &[u8], width: usize, height: usize) -> Vec<u8> {
        let n = width * height;
        assert_eq!(rgb_data.len(), n * 3, "RGB data size mismatch");

        // 1. RGB → YCoCg-R (reversible color transform)
        let pixels: Vec<RGB> = rgb_data
            .chunks_exact(3)
            .map(|c| RGB {
                r: c[0],
                g: c[1],
                b: c[2],
            })
            .collect();

        let mut y_plane = vec![0i16; n];
        let mut co_plane = vec![0i16; n];
        let mut cg_plane = vec![0i16; n];
        rgb_to_ycocg_r(&pixels, &mut y_plane, &mut co_plane, &mut cg_plane);

        // 2. Quantization step (quality-based for 2D single-frame)
        //    quality 100 → step 1 (near-lossless)
        //    quality 50  → step 4
        //    quality 0   → step 16
        let step = ((100 - self.config.quality.min(100)) as i32 / 6).max(1);

        // 3. Process Y/Co/Cg channels in parallel (wavelet → quantize → symbols → rANS)
        let wavelet = &self.wavelet;
        let ((y_result, co_result), cg_result) = rayon::join(
            || {
                rayon::join(
                    || encode_channel(wavelet, &y_plane, width, height, step),
                    || encode_channel(wavelet, &co_plane, width, height, step),
                )
            },
            || encode_channel(wavelet, &cg_plane, width, height, step),
        );

        // 4. Pack into output: header + histograms + bitstreams
        pack_compressed_frame(
            width as u32,
            height as u32,
            step,
            step,
            step,
            y_result.min_val,
            co_result.min_val,
            cg_result.min_val,
            y_result.scale,
            co_result.scale,
            cg_result.scale,
            &y_result.histogram,
            &co_result.histogram,
            &cg_result.histogram,
            &y_result.bitstream,
            &co_result.bitstream,
            &cg_result.bitstream,
        )
    }

    /// Encode only the person region from a full frame using segmentation.
    ///
    /// # Arguments
    /// * `current_rgb` - Current frame RGB data
    /// * `current_gray` - Current frame grayscale
    /// * `reference_gray` - Reference frame grayscale (for motion segmentation)
    /// * `width` - Full frame width
    /// * `height` - Full frame height
    ///
    /// # Returns
    /// (compressed_data, SegmentResult) — compressed person region + mask info
    pub fn encode_person_region(
        &self,
        current_rgb: &[u8],
        current_gray: &[u8],
        reference_gray: &[u8],
        width: u32,
        height: u32,
    ) -> (Vec<u8>, SegmentResult) {
        let seg_config = SegmentConfig {
            motion_threshold: 25,
            min_region_size: 100,
            dilate_radius: 2,
            erode_radius: 1,
        };

        let seg = segment_by_motion(current_gray, reference_gray, width, height, &seg_config);

        if seg.foreground_count == 0 {
            return (Vec::new(), seg);
        }

        let person_rgb = seg.extract_person_rgb(current_rgb);
        let bw = seg.bbox[2] as usize;
        let bh = seg.bbox[3] as usize;

        let compressed = if bw > 0 && bh > 0 {
            self.encode_frame(&person_rgb, bw, bh)
        } else {
            Vec::new()
        };

        (compressed, seg)
    }
}

/// Video decoder: compressed bitstream → RGB frames
#[derive(Debug)]
pub struct VideoDecoder {
    wavelet: Wavelet2D,
    decode_y_buffer: Vec<i16>,
    decode_co_buffer: Vec<i16>,
    decode_cg_buffer: Vec<i16>,
}

impl VideoDecoder {
    /// Create a new video decoder
    pub fn new(wavelet_type: WaveletType) -> Self {
        let w1d = match wavelet_type {
            WaveletType::Cdf97 => Wavelet1D::cdf97(),
            WaveletType::Cdf53 => Wavelet1D::cdf53(),
            WaveletType::Haar => Wavelet1D::haar(),
        };
        Self {
            wavelet: Wavelet2D::new(w1d),
            decode_y_buffer: Vec::new(),
            decode_co_buffer: Vec::new(),
            decode_cg_buffer: Vec::new(),
        }
    }

    /// Create with default (CDF 9/7) wavelet
    pub fn default_config() -> Self {
        Self::new(WaveletType::Cdf97)
    }

    /// Decode compressed bitstream to RGB frame.
    ///
    /// # Returns
    /// (rgb_data, width, height)
    pub fn decode_frame(&self, compressed: &[u8]) -> Option<(Vec<u8>, u32, u32)> {
        let (
            width,
            height,
            q_step_y,
            q_step_co,
            q_step_cg,
            y_min,
            co_min,
            cg_min,
            y_scale,
            co_scale,
            cg_scale,
            y_hist,
            co_hist,
            cg_hist,
            bs_y,
            bs_co,
            bs_cg,
        ) = unpack_compressed_frame(compressed)?;

        let w = width as usize;
        let h = height as usize;
        let n = w * h;

        // 1. Decode Y/Co/Cg channels in parallel (rANS → dequantize → inverse wavelet)
        let wavelet = &self.wavelet;
        let ((mut y_i32, mut co_i32), mut cg_i32) = rayon::join(
            || {
                rayon::join(
                    || decode_channel(wavelet, &bs_y, &y_hist, n, y_min, y_scale, q_step_y, w, h),
                    || {
                        decode_channel(
                            wavelet, &bs_co, &co_hist, n, co_min, co_scale, q_step_co, w, h,
                        )
                    },
                )
            },
            || {
                decode_channel(
                    wavelet, &bs_cg, &cg_hist, n, cg_min, cg_scale, q_step_cg, w, h,
                )
            },
        );

        // 2. YCoCg-R → RGB (in-place i32→i16 conversion, reuse buffers)
        self.decode_y_buffer.clear();
        self.decode_y_buffer.extend(y_i32.iter().map(|&v| v as i16));
        self.decode_co_buffer.clear();
        self.decode_co_buffer
            .extend(co_i32.iter().map(|&v| v as i16));
        self.decode_cg_buffer.clear();
        self.decode_cg_buffer
            .extend(cg_i32.iter().map(|&v| v as i16));

        let mut rgb_out = vec![RGB { r: 0, g: 0, b: 0 }; n];
        ycocg_r_to_rgb(
            &self.decode_y_buffer,
            &self.decode_co_buffer,
            &self.decode_cg_buffer,
            &mut rgb_out,
        );

        // 3. RGB struct → flat bytes (pre-allocated)
        let mut rgb_bytes = Vec::with_capacity(n * 3);
        for p in &rgb_out {
            rgb_bytes.push(p.r);
            rgb_bytes.push(p.g);
            rgb_bytes.push(p.b);
        }

        Some((rgb_bytes, width, height))
    }
}

// =============================================================================
// Channel-level encode/decode (called in parallel via rayon::join)
// =============================================================================

/// Result of encoding a single color channel
struct ChannelEncodeResult {
    min_val: i32,
    scale: f32,
    histogram: Vec<u32>,
    bitstream: Vec<u8>,
}

/// Encode a single color channel: i16 plane → wavelet → quantize → rANS bitstream
#[inline(never)] // Prevent inlining — each channel runs on separate Rayon thread
fn encode_channel(
    wavelet: &Wavelet2D,
    plane: &[i16],
    width: usize,
    height: usize,
    step: i32,
) -> ChannelEncodeResult {
    // i16 → i32 for wavelet
    let mut coeffs: Vec<i32> = plane.iter().map(|&v| v as i32).collect();

    // 2D Wavelet transform (in-place)
    wavelet.forward(&mut coeffs, width, height);

    // Quantize
    let q = Quantizer::new(step);
    let mut quantized = vec![0i32; coeffs.len()];
    q.quantize_buffer(&coeffs, &mut quantized);

    // Coefficients → u8 symbols
    let (symbols, min_val, scale) = coeffs_to_symbols(&quantized);

    // Histogram + rANS
    let histogram = build_histogram(&symbols);
    let table = FrequencyTable::from_histogram(&histogram);
    let mut encoder = RansEncoder::new();
    encoder.encode_symbols(&symbols, &table);
    let bitstream = encoder.finish();

    ChannelEncodeResult {
        min_val,
        scale,
        histogram,
        bitstream,
    }
}

/// Decode a single color channel: rANS bitstream → dequantize → inverse wavelet → i32 plane
#[inline(never)]
fn decode_channel(
    wavelet: &Wavelet2D,
    bitstream: &[u8],
    histogram: &[u32],
    n: usize,
    min_val: i32,
    scale: f32,
    q_step: i32,
    width: usize,
    height: usize,
) -> Vec<i32> {
    // rANS decode
    let table = FrequencyTable::from_histogram(histogram);
    let mut decoder = RansDecoder::new(bitstream);
    let symbols = decoder.decode_n(n, &table);

    // Symbols → quantized coefficients
    let quantized = symbols_to_coeffs(&symbols, min_val, scale);

    // Dequantize
    let q = Quantizer::new(q_step);
    let mut coeffs = vec![0i32; n];
    q.dequantize_buffer(&quantized, &mut coeffs);

    // Inverse 2D wavelet
    wavelet.inverse(&mut coeffs, width, height);

    coeffs
}

// =============================================================================
// Internal helpers — #[inline] on hot scalar paths
// =============================================================================

/// Convert quantized i32 coefficients to u8 symbols (offset, scale only if range > 255)
///
/// Returns (symbols, min_val, scale) where:
/// - scale = 0.0 means no scaling (range <= 255, simple offset)
/// - scale > 0.0 means [min..max] was mapped to [0..255]
#[inline]
fn coeffs_to_symbols(coeffs: &[i32]) -> (Vec<u8>, i32, f32) {
    let min_val = coeffs.iter().copied().min().unwrap_or(0);
    let max_val = coeffs.iter().copied().max().unwrap_or(0);
    let range = max_val - min_val;

    if range <= 255 {
        // No scaling needed — simple offset
        let symbols: Vec<u8> = coeffs.iter().map(|&v| (v - min_val) as u8).collect();
        return (symbols, min_val, 0.0);
    }

    // Range > 255: scale down to fit u8
    let scale = 255.0 / range as f32;
    let symbols: Vec<u8> = coeffs
        .iter()
        .map(|&v| {
            let scaled = ((v - min_val) as f32 * scale) as i32;
            scaled.clamp(0, 255) as u8
        })
        .collect();

    (symbols, min_val, scale)
}

/// Convert u8 symbols back to i32 coefficients
#[inline]
fn symbols_to_coeffs(symbols: &[u8], min_val: i32, scale: f32) -> Vec<i32> {
    if scale == 0.0 {
        // No scaling was applied — simple offset
        return symbols.iter().map(|&s| s as i32 + min_val).collect();
    }
    let inv_scale = 1.0 / scale;
    symbols
        .iter()
        .map(|&s| (s as f32 * inv_scale + 0.5) as i32 + min_val)
        .collect()
}

/// Build 256-bin histogram from u8 symbols
#[inline]
fn build_histogram(symbols: &[u8]) -> Vec<u32> {
    let mut hist = vec![0u32; 256];
    for &s in symbols {
        hist[s as usize] += 1;
    }
    hist
}

/// Compressed frame format:
/// [4: width][4: height][4: q_step_y][4: q_step_co][4: q_step_cg]
/// [4: y_min][4: co_min][4: cg_min]
/// [4: y_scale][4: co_scale][4: cg_scale]
/// [4: y_hist_len][y_hist...][4: co_hist_len][co_hist...][4: cg_hist_len][cg_hist...]
/// [4: bs_y_len][bs_y...][4: bs_co_len][bs_co...][4: bs_cg_len][bs_cg...]
#[inline]
fn pack_compressed_frame(
    width: u32,
    height: u32,
    q_step_y: i32,
    q_step_co: i32,
    q_step_cg: i32,
    y_min: i32,
    co_min: i32,
    cg_min: i32,
    y_scale: f32,
    co_scale: f32,
    cg_scale: f32,
    y_hist: &[u32],
    co_hist: &[u32],
    cg_hist: &[u32],
    bs_y: &[u8],
    bs_co: &[u8],
    bs_cg: &[u8],
) -> Vec<u8> {
    let mut out = Vec::with_capacity(
        44 + // fixed header
        (256 * 4) * 3 + // histograms
        bs_y.len() + bs_co.len() + bs_cg.len() + 12, // bitstreams + lengths
    );

    // Header (44 bytes)
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&q_step_y.to_le_bytes());
    out.extend_from_slice(&q_step_co.to_le_bytes());
    out.extend_from_slice(&q_step_cg.to_le_bytes());
    out.extend_from_slice(&y_min.to_le_bytes());
    out.extend_from_slice(&co_min.to_le_bytes());
    out.extend_from_slice(&cg_min.to_le_bytes());
    out.extend_from_slice(&y_scale.to_le_bytes());
    out.extend_from_slice(&co_scale.to_le_bytes());
    out.extend_from_slice(&cg_scale.to_le_bytes());

    // Histograms (always 256 entries each)
    for hist in [y_hist, co_hist, cg_hist] {
        let len = hist.len() as u32;
        out.extend_from_slice(&len.to_le_bytes());
        for &h in hist {
            out.extend_from_slice(&h.to_le_bytes());
        }
    }

    // Bitstreams
    for bs in [bs_y, bs_co, bs_cg] {
        let len = bs.len() as u32;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(bs);
    }

    out
}

/// Unpack compressed frame — returns all components or None on invalid data
#[allow(clippy::type_complexity)]
fn unpack_compressed_frame(
    data: &[u8],
) -> Option<(
    u32,
    u32, // width, height
    i32,
    i32,
    i32, // q_step_y, q_step_co, q_step_cg
    i32,
    i32,
    i32, // y_min, co_min, cg_min
    f32,
    f32,
    f32, // y_scale, co_scale, cg_scale
    Vec<u32>,
    Vec<u32>,
    Vec<u32>, // histograms
    Vec<u8>,
    Vec<u8>,
    Vec<u8>, // bitstreams
)> {
    if data.len() < 44 {
        return None;
    }

    let mut pos = 0;

    let read_u32 = |data: &[u8], pos: &mut usize| -> Option<u32> {
        if *pos + 4 > data.len() {
            return None;
        }
        let v = u32::from_le_bytes(data[*pos..*pos + 4].try_into().ok()?);
        *pos += 4;
        Some(v)
    };
    let read_i32 = |data: &[u8], pos: &mut usize| -> Option<i32> {
        if *pos + 4 > data.len() {
            return None;
        }
        let v = i32::from_le_bytes(data[*pos..*pos + 4].try_into().ok()?);
        *pos += 4;
        Some(v)
    };
    let read_f32 = |data: &[u8], pos: &mut usize| -> Option<f32> {
        if *pos + 4 > data.len() {
            return None;
        }
        let v = f32::from_le_bytes(data[*pos..*pos + 4].try_into().ok()?);
        *pos += 4;
        Some(v)
    };

    let width = read_u32(data, &mut pos)?;
    let height = read_u32(data, &mut pos)?;
    let q_step_y = read_i32(data, &mut pos)?;
    let q_step_co = read_i32(data, &mut pos)?;
    let q_step_cg = read_i32(data, &mut pos)?;
    let y_min = read_i32(data, &mut pos)?;
    let co_min = read_i32(data, &mut pos)?;
    let cg_min = read_i32(data, &mut pos)?;
    let y_scale = read_f32(data, &mut pos)?;
    let co_scale = read_f32(data, &mut pos)?;
    let cg_scale = read_f32(data, &mut pos)?;

    let read_hist = |data: &[u8], pos: &mut usize| -> Option<Vec<u32>> {
        let len = read_u32(data, pos)? as usize;
        let mut hist = Vec::with_capacity(len);
        for _ in 0..len {
            hist.push(read_u32(data, pos)?);
        }
        Some(hist)
    };

    let y_hist = read_hist(data, &mut pos)?;
    let co_hist = read_hist(data, &mut pos)?;
    let cg_hist = read_hist(data, &mut pos)?;

    let read_bytes = |data: &[u8], pos: &mut usize| -> Option<Vec<u8>> {
        let len = read_u32(data, pos)? as usize;
        if *pos + len > data.len() {
            return None;
        }
        let bytes = data[*pos..*pos + len].to_vec();
        *pos += len;
        Some(bytes)
    };

    let bs_y = read_bytes(data, &mut pos)?;
    let bs_co = read_bytes(data, &mut pos)?;
    let bs_cg = read_bytes(data, &mut pos)?;

    Some((
        width, height, q_step_y, q_step_co, q_step_cg, y_min, co_min, cg_min, y_scale, co_scale,
        cg_scale, y_hist, co_hist, cg_hist, bs_y, bs_co, bs_cg,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode_roundtrip() {
        let width = 16;
        let height = 16;

        // Generate a simple test pattern (gradient)
        let mut rgb_data = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                let r = ((x * 255) / width) as u8;
                let g = ((y * 255) / height) as u8;
                let b = 128u8;
                rgb_data.push(r);
                rgb_data.push(g);
                rgb_data.push(b);
            }
        }

        let encoder = VideoEncoder::default_config();
        let compressed = encoder.encode_frame(&rgb_data, width, height);
        assert!(!compressed.is_empty());

        let decoder = VideoDecoder::default_config();
        let (decoded_rgb, dec_w, dec_h) = decoder.decode_frame(&compressed).unwrap();
        assert_eq!(dec_w, width as u32);
        assert_eq!(dec_h, height as u32);
        assert_eq!(decoded_rgb.len(), width * height * 3);

        // Lossy codec — check that reconstruction is reasonable
        let mut total_error = 0u64;
        for i in 0..rgb_data.len() {
            let diff = (rgb_data[i] as i32 - decoded_rgb[i] as i32).unsigned_abs() as u64;
            total_error += diff;
        }
        let avg_error = total_error as f64 / rgb_data.len() as f64;
        assert!(
            avg_error < 50.0,
            "Average pixel error too high: {:.1}",
            avg_error
        );
    }

    #[test]
    fn test_coeffs_symbols_roundtrip() {
        let coeffs = vec![-10, -5, 0, 3, 7, 15, -2];
        let (symbols, min_val, scale) = coeffs_to_symbols(&coeffs);
        let restored = symbols_to_coeffs(&symbols, min_val, scale);
        // Lossy scaling — check values are close (within +/-1)
        for (orig, rest) in coeffs.iter().zip(restored.iter()) {
            assert!((orig - rest).abs() <= 1, "orig={}, restored={}", orig, rest);
        }
    }

    #[test]
    fn test_video_config_defaults() {
        let config = VideoCodecConfig::default();
        assert_eq!(config.quality, 75);
        assert_eq!(config.wavelet_type, WaveletType::Cdf97);
    }

    #[test]
    fn test_empty_frame_decode() {
        let decoder = VideoDecoder::default_config();
        assert!(decoder.decode_frame(&[]).is_none());
        assert!(decoder.decode_frame(&[0; 10]).is_none());
    }
}
