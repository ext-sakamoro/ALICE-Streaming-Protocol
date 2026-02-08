//! Python bindings for libasp using PyO3 + NumPy Zero-Copy
//!
//! High-performance ASP functions with zero-copy data transfer.
//!
//! # Optimizations
//! - Thread Local PacketEncoder (zero allocation per call)
//! - Direct NumPy → FlatBufferBuilder write (no intermediate Vec)
//! - Raw pointer iteration (no iterator overhead)

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2, IntoPyArray, PyArrayMethods};
use std::cell::RefCell;
use crate::packet::PacketEncoder;

/// Wrapper for raw pointer that implements Send
/// SAFETY: Only use for read-only access to memory that outlives the operation
struct SendPtr(*const u8, usize);
unsafe impl Send for SendPtr {}

impl SendPtr {
    #[inline]
    fn new(ptr: *const u8, len: usize) -> Self {
        Self(ptr, len)
    }

    #[inline]
    unsafe fn as_slice(&self) -> &[u8] {
        std::slice::from_raw_parts(self.0, self.1)
    }
}

use crate::codec::{
    color::{extract_dominant_colors, kmeans_palette},
    dct::{dct2d, idct2d, sparse_dct_encode, sparse_dct_decode, DctTransform},
    motion::{estimate_motion_fast, MotionEstimator, SearchAlgorithm},
    roi::{detect_rois, RoiConfig},
};
use crate::packet::{AspPacket, IPacketPayload, DPacketPayload};
use crate::types::{MotionVector, PacketType, RoiType};
use crate::generated;
use crate::header::{AspPacketHeader, crc32};

// =============================================================================
// Thread Local Encoder (Zero Allocation Per Call)
// =============================================================================

thread_local! {
    /// Thread-local PacketEncoder to avoid allocation per Python call
    static PACKET_ENCODER: RefCell<PacketEncoder> = RefCell::new(
        PacketEncoder::with_capacity(65536, 65536)
    );
}

// =============================================================================
// Motion Estimation (NumPy Zero-Copy)
// =============================================================================

/// Fast motion estimation with NumPy zero-copy I/O
///
/// Args:
///     current: Current frame (H, W) as uint8 NumPy array
///     previous: Previous frame (H, W) as uint8 NumPy array
///     block_size: Block size (default: 16)
///     search_range: Search range in pixels (default: 16)
///
/// Returns:
///     NumPy array (N, 5) of int32: [block_x, block_y, dx, dy, sad]
#[pyfunction]
#[pyo3(signature = (current, previous, block_size=16, search_range=16))]
fn estimate_motion_numpy<'py>(
    py: Python<'py>,
    current: PyReadonlyArray2<'py, u8>,
    previous: PyReadonlyArray2<'py, u8>,
    block_size: usize,
    search_range: usize,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let curr_array = current.as_array();
    let prev_array = previous.as_array();
    let (h, w) = (curr_array.shape()[0], curr_array.shape()[1]);

    // Zero-copy access to NumPy data
    let curr_slice = curr_array.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Current array must be contiguous (C-order)")
    })?;
    let prev_slice = prev_array.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Previous array must be contiguous (C-order)")
    })?;

    // Wrap raw pointers for GIL-free computation
    let curr_send = SendPtr::new(curr_slice.as_ptr(), curr_slice.len());
    let prev_send = SendPtr::new(prev_slice.as_ptr(), prev_slice.len());

    // SAFETY: The NumPy arrays are kept alive by the function arguments,
    // and we're only reading from them. The GIL release allows Python
    // threads to run while we do heavy computation.
    let results = py.allow_threads(move || {
        let curr = unsafe { curr_send.as_slice() };
        let prev = unsafe { prev_send.as_slice() };
        estimate_motion_fast(curr, prev, w, h, block_size, search_range, 256)
    });

    // Convert to NumPy array with direct allocation (zero intermediate copy)
    let n_results = results.len();
    if n_results == 0 {
        // Return empty array
        let empty: Vec<i32> = vec![];
        return numpy::PyArray::from_vec(py, empty)
            .reshape([0, 5])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to reshape array: {:?}", e)
            ));
    }

    // Allocate NumPy array directly in Python heap (zero-copy output)
    let array = unsafe {
        PyArray2::<i32>::new(py, [n_results, 5], false)
    };

    // Write directly to Python memory
    unsafe {
        let ptr = array.data() as *mut i32;
        for (i, r) in results.iter().enumerate() {
            let base = i * 5;
            *ptr.add(base) = r.block_x as i32;
            *ptr.add(base + 1) = r.block_y as i32;
            *ptr.add(base + 2) = r.dx as i32;
            *ptr.add(base + 3) = r.dy as i32;
            *ptr.add(base + 4) = r.sad as i32;
        }
    }

    Ok(array)
}

/// Legacy motion estimation (returns Python list for compatibility)
#[pyfunction]
#[pyo3(signature = (current, previous, width, height, block_size=16, search_range=16))]
fn estimate_motion(
    current: &[u8],
    previous: &[u8],
    width: usize,
    height: usize,
    block_size: usize,
    search_range: usize,
) -> PyResult<Vec<(u32, u32, i16, i16, u32)>> {
    let mvs = estimate_motion_fast(
        current, previous, width, height, block_size, search_range, 256
    );

    Ok(mvs.into_iter()
        .map(|mv| (mv.block_x as u32, mv.block_y as u32, mv.dx, mv.dy, mv.sad))
        .collect())
}

/// Motion estimator class
#[pyclass]
#[pyo3(name = "MotionEstimator")]
struct PyMotionEstimator {
    inner: MotionEstimator,
}

#[pymethods]
impl PyMotionEstimator {
    #[new]
    #[pyo3(signature = (block_size=16, search_range=16, algorithm="diamond"))]
    fn new(block_size: usize, search_range: usize, algorithm: &str) -> Self {
        let alg = match algorithm {
            "full" => SearchAlgorithm::FullSearch,
            "three_step" => SearchAlgorithm::ThreeStepSearch,
            "hexagon" => SearchAlgorithm::HexagonSearch,
            _ => SearchAlgorithm::DiamondSearch,
        };

        Self {
            inner: MotionEstimator::new(block_size, search_range).with_algorithm(alg),
        }
    }

    /// Estimate motion with NumPy zero-copy
    fn estimate_numpy<'py>(
        &self,
        py: Python<'py>,
        current: PyReadonlyArray2<'py, u8>,
        previous: PyReadonlyArray2<'py, u8>,
    ) -> PyResult<Bound<'py, PyArray2<i32>>> {
        estimate_motion_numpy(py, current, previous, self.inner.block_size, self.inner.search_range)
    }

    /// Legacy estimate (returns list)
    fn estimate(&self, current: &[u8], previous: &[u8], width: usize, height: usize) -> Vec<(u32, u32, i16, i16, u32)> {
        self.inner.estimate(current, previous, width, height)
            .into_iter()
            .map(|mv| (mv.block_x as u32, mv.block_y as u32, mv.dx, mv.dy, mv.sad))
            .collect()
    }
}

// =============================================================================
// Color Extraction
// =============================================================================

/// Extract dominant colors (NumPy optimized)
#[pyfunction]
#[pyo3(signature = (pixels, num_colors=5, max_iterations=10, sampling_rate=0.1))]
fn extract_colors<'py>(
    py: Python<'py>,
    pixels: &[u8],
    num_colors: usize,
    max_iterations: usize,
    sampling_rate: f64,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    // Wrap raw pointer for GIL-free computation
    let send_ptr = SendPtr::new(pixels.as_ptr(), pixels.len());

    // Release GIL for heavy k-means computation
    let colors = py.allow_threads(move || {
        let slice = unsafe { send_ptr.as_slice() };
        extract_dominant_colors(slice, num_colors, max_iterations, sampling_rate)
    });

    // Allocate NumPy array directly in Python heap (zero-copy output)
    let n = colors.len();
    if n == 0 {
        let empty: Vec<u8> = vec![];
        return numpy::PyArray::from_vec(py, empty)
            .reshape([0, 3])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to reshape array: {:?}", e)
            ));
    }

    let array = unsafe {
        PyArray2::<u8>::new(py, [n, 3], false)
    };

    // Write directly to Python memory
    unsafe {
        let ptr = array.data() as *mut u8;
        for (i, c) in colors.iter().enumerate() {
            let base = i * 3;
            *ptr.add(base) = c.r;
            *ptr.add(base + 1) = c.g;
            *ptr.add(base + 2) = c.b;
        }
    }

    Ok(array)
}

/// Extract colors with weights
#[pyfunction]
#[pyo3(signature = (pixels, num_colors=5, max_iterations=10, sampling_rate=0.1))]
fn extract_colors_with_weights<'py>(
    py: Python<'py>,
    pixels: &[u8],
    num_colors: usize,
    max_iterations: usize,
    sampling_rate: f64,
) -> PyResult<(Bound<'py, PyArray2<u8>>, Bound<'py, PyArray1<f32>>)> {
    // Wrap raw pointer for GIL-free computation
    let send_ptr = SendPtr::new(pixels.as_ptr(), pixels.len());

    // Release GIL for heavy k-means computation
    let (colors, weights) = py.allow_threads(move || {
        let slice = unsafe { send_ptr.as_slice() };
        kmeans_palette(slice, num_colors, max_iterations, sampling_rate)
    });

    let n = colors.len();

    // Allocate colors array directly in Python heap
    let colors_array = unsafe {
        PyArray2::<u8>::new(py, [n, 3], false)
    };

    // Write colors directly to Python memory
    unsafe {
        let ptr = colors_array.data() as *mut u8;
        for (i, c) in colors.iter().enumerate() {
            let base = i * 3;
            *ptr.add(base) = c.r;
            *ptr.add(base + 1) = c.g;
            *ptr.add(base + 2) = c.b;
        }
    }

    let weights_array = weights.into_pyarray(py);

    Ok((colors_array, weights_array))
}

// =============================================================================
// DCT Transform
// =============================================================================

/// Perform 2D DCT (NumPy optimized)
#[pyfunction]
fn dct_2d<'py>(
    py: Python<'py>,
    input: Vec<f64>,
    size: usize
) -> Bound<'py, PyArray1<f64>> {
    let result = dct2d(&input, size);
    result.into_pyarray(py)
}

/// Perform inverse 2D DCT (NumPy optimized)
#[pyfunction]
fn idct_2d<'py>(
    py: Python<'py>,
    input: Vec<f64>,
    size: usize
) -> Bound<'py, PyArray1<f64>> {
    let result = idct2d(&input, size);
    result.into_pyarray(py)
}

/// Encode DCT coefficients to sparse format
#[pyfunction]
#[pyo3(signature = (coefficients, size, threshold=0.001))]
fn encode_sparse_dct(coefficients: Vec<i32>, size: usize, threshold: f64) -> Vec<(u32, u32, f32)> {
    sparse_dct_encode(&coefficients, size, threshold)
}

/// Decode sparse format to DCT coefficients
#[pyfunction]
#[pyo3(signature = (sparse, size, default=0))]
fn decode_sparse_dct<'py>(
    py: Python<'py>,
    sparse: Vec<(u32, u32, f32)>,
    size: usize,
    default: i32
) -> Bound<'py, PyArray1<i32>> {
    let result = sparse_dct_decode(&sparse, size, default);
    result.into_pyarray(py)
}

/// DCT Transform class
#[pyclass]
#[pyo3(name = "DctTransform")]
struct PyDctTransform {
    inner: DctTransform,
}

#[pymethods]
impl PyDctTransform {
    #[new]
    #[pyo3(signature = (block_size=8, quality=75))]
    fn new(block_size: usize, quality: u8) -> Self {
        Self {
            inner: DctTransform::new(block_size).with_quality(quality),
        }
    }

    fn forward<'py>(&self, py: Python<'py>, block: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
        self.inner.forward(&block).into_pyarray(py)
    }

    fn inverse<'py>(&self, py: Python<'py>, coefficients: Vec<f64>) -> Bound<'py, PyArray1<f64>> {
        self.inner.inverse(&coefficients).into_pyarray(py)
    }

    fn quantize<'py>(&self, py: Python<'py>, coefficients: Vec<f64>) -> Bound<'py, PyArray1<i32>> {
        self.inner.quantize(&coefficients).into_pyarray(py)
    }

    fn dequantize<'py>(&self, py: Python<'py>, quantized: Vec<i32>) -> Bound<'py, PyArray1<f64>> {
        self.inner.dequantize(&quantized).into_pyarray(py)
    }

    fn encode_sparse(&self, block: Vec<f64>) -> Vec<(u32, u32, f32)> {
        self.inner.encode_sparse(&block)
    }

    fn decode_sparse<'py>(&self, py: Python<'py>, sparse: Vec<(u32, u32, f32)>) -> Bound<'py, PyArray1<f64>> {
        self.inner.decode_sparse(&sparse, 0.0).into_pyarray(py)
    }
}

// =============================================================================
// ROI Detection
// =============================================================================

/// Detect regions of interest (NumPy optimized)
#[pyfunction]
#[pyo3(signature = (current, previous=None, width=0, height=0, edge_threshold=30, motion_threshold=20))]
fn detect_roi<'py>(
    py: Python<'py>,
    current: &[u8],
    previous: Option<&[u8]>,
    width: usize,
    height: usize,
    edge_threshold: u32,
    motion_threshold: u32,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let config = RoiConfig {
        edge_threshold,
        motion_threshold,
        ..Default::default()
    };

    // Wrap raw pointers for GIL-free computation
    let curr_send = SendPtr::new(current.as_ptr(), current.len());
    let prev_send = previous.map(|p| SendPtr::new(p.as_ptr(), p.len()));

    // Release GIL for heavy ROI detection
    let regions = py.allow_threads(move || {
        let curr = unsafe { curr_send.as_slice() };
        let prev = prev_send.as_ref().map(|p| unsafe { p.as_slice() });
        detect_rois(curr, prev, width, height, &config)
    });

    // Return as (N, 6) array: [x, y, w, h, type_id, confidence]
    let n = regions.len();

    if n == 0 {
        let empty: Vec<f32> = vec![];
        return numpy::PyArray::from_vec(py, empty)
            .reshape([0, 6])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(
                format!("Failed to reshape array: {:?}", e)
            ));
    }

    // Allocate NumPy array directly in Python heap (zero-copy output)
    let array = unsafe {
        PyArray2::<f32>::new(py, [n, 6], false)
    };

    // Write directly to Python memory
    unsafe {
        let ptr = array.data() as *mut f32;
        for (i, r) in regions.iter().enumerate() {
            let type_id = match r.roi_type {
                RoiType::General => 0.0,
                RoiType::Face => 1.0,
                RoiType::Text => 2.0,
                RoiType::Edge => 3.0,
                RoiType::Motion => 4.0,
                RoiType::Custom => 5.0,
            };

            let base = i * 6;
            *ptr.add(base) = r.bounds.x as f32;
            *ptr.add(base + 1) = r.bounds.y as f32;
            *ptr.add(base + 2) = r.bounds.width as f32;
            *ptr.add(base + 3) = r.bounds.height as f32;
            *ptr.add(base + 4) = type_id;
            *ptr.add(base + 5) = r.confidence;
        }
    }

    Ok(array)
}

// =============================================================================
// Packet Handling (Optimized)
// =============================================================================

/// Create an I-Packet (keyframe)
#[pyfunction]
#[pyo3(signature = (sequence, width, height, fps=30.0))]
fn create_i_packet(py: Python<'_>, sequence: u32, width: u32, height: u32, fps: f32) -> PyResult<Py<PyBytes>> {
    let payload = IPacketPayload::new(width, height, fps);
    let packet = AspPacket::create_i_packet(sequence, payload)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let bytes = packet.to_bytes()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &bytes).into())
}

/// Create a D-Packet from NumPy motion vectors (Ultra Fast Path)
///
/// # Optimization
/// - Uses Thread Local `PacketEncoder` (zero allocation per call)
/// - Reads directly from NumPy ptr (zero copy)
/// - Writes directly to FlatBufferBuilder (no intermediate Vec)
/// - Raw pointer iteration (no iterator overhead)
#[pyfunction]
#[pyo3(signature = (sequence, ref_sequence, motion_vectors, timestamp_ms=0))]
fn create_d_packet_numpy(
    py: Python<'_>,
    sequence: u32,
    ref_sequence: u32,
    motion_vectors: PyReadonlyArray2<i32>,
    timestamp_ms: u64,
) -> PyResult<Py<PyBytes>> {
    let mv_array = motion_vectors.as_array();
    let n_rows = mv_array.shape()[0];
    let n_cols = mv_array.shape()[1];

    // Validation
    if n_cols < 5 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "Motion vectors array must have at least 5 columns"
        ));
    }

    let slice = mv_array.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Motion vectors array must be contiguous (C-order)")
    })?;
    let ptr = slice.as_ptr();

    // Use Thread Local Encoder (zero allocation)
    PACKET_ENCODER.with(|encoder_cell| {
        let mut encoder = encoder_cell.borrow_mut();

        // 1. Reset and build FlatBuffers payload directly
        encoder.builder.reset();

        // Serialize Motion Vectors directly from NumPy ptr
        // FlatBuffers builds vectors backwards
        encoder.builder.start_vector::<generated::MotionVector>(n_rows);

        unsafe {
            for i in (0..n_rows).rev() {
                let base = i * n_cols;
                // Read directly from NumPy memory
                let bx = *ptr.add(base) as u16;
                let by = *ptr.add(base + 1) as u16;
                let dx = *ptr.add(base + 2) as i16;
                let dy = *ptr.add(base + 3) as i16;
                let sad = *ptr.add(base + 4) as u32;

                let fb_mv = generated::MotionVector::new(bx, by, dx, dy, sad);
                encoder.builder.push(fb_mv);
            }
        }
        let mvs_offset = encoder.builder.end_vector::<generated::MotionVector>(n_rows);

        // 2. Build DPacketPayload
        let d_packet = generated::DPacketPayload::create(
            &mut encoder.builder,
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
            &mut encoder.builder,
            &generated::AspPacketPayloadArgs {
                payload_type: generated::AspPayloadUnion::DPacketPayload,
                payload: Some(d_packet.as_union_value()),
            },
        );

        encoder.builder.finish(asp_packet, Some("ASP1"));

        // 4. Construct final packet in output buffer
        // Use raw pointers to bypass borrow checker (safe because we control access order)
        let (fb_ptr, payload_len) = {
            let fb_bytes = encoder.builder.finished_data();
            (fb_bytes.as_ptr(), fb_bytes.len())
        };

        let total_size = AspPacketHeader::SIZE + payload_len + 4;

        encoder.buffer.clear();
        let current_cap = encoder.buffer.capacity();
        if current_cap < total_size {
            encoder.buffer.reserve(total_size - current_cap);
        }

        unsafe {
            encoder.buffer.set_len(total_size);
            let out_ptr = encoder.buffer.as_mut_ptr();

            // Write Header (zero bounds check)
            let header = AspPacketHeader::new(
                PacketType::DPacket,
                sequence,
                payload_len as u32,
            );
            header.write_to_ptr(out_ptr);

            // Copy FlatBuffers payload
            // SAFETY: fb_ptr is still valid because encoder.builder hasn't been modified
            std::ptr::copy_nonoverlapping(
                fb_ptr,
                out_ptr.add(AspPacketHeader::SIZE),
                payload_len,
            );

            // CRC32
            let crc = crc32(&encoder.buffer[..total_size - 4]);
            let crc_bytes = crc.to_be_bytes();
            std::ptr::copy_nonoverlapping(
                crc_bytes.as_ptr(),
                out_ptr.add(total_size - 4),
                4,
            );
        }

        // Return as Python bytes (this is the only unavoidable copy - Python owns its memory)
        Ok(PyBytes::new(py, &encoder.buffer).into())
    })
}

/// Create a D-Packet (legacy, from list)
#[pyfunction]
fn create_d_packet(
    py: Python<'_>,
    sequence: u32,
    ref_sequence: u32,
    motion_vectors: Vec<(u32, u32, i16, i16, u32)>,
) -> PyResult<Py<PyBytes>> {
    let mut payload = DPacketPayload::new(ref_sequence);
    payload.motion_vectors.reserve(motion_vectors.len());

    for (bx, by, dx, dy, sad) in motion_vectors {
        payload.add_motion_vector(MotionVector::from_u32(bx, by, dx, dy, sad));
    }

    let packet = AspPacket::create_d_packet(sequence, payload)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let bytes = packet.to_bytes()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyBytes::new(py, &bytes).into())
}

/// Parse an ASP packet from bytes
#[pyfunction]
fn parse_packet(data: &[u8]) -> PyResult<(String, u32, u32)> {
    let packet = AspPacket::from_bytes(data)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    let type_str = match packet.packet_type() {
        PacketType::IPacket => "I",
        PacketType::DPacket => "D",
        PacketType::CPacket => "C",
        PacketType::SPacket => "S",
    };

    Ok((type_str.to_string(), packet.sequence(), packet.payload_size()))
}

/// Get library version
#[pyfunction]
fn version() -> &'static str {
    crate::VERSION
}

// =============================================================================
// Hybrid Streaming (SDF Background + Wavelet Person)
// =============================================================================

use crate::scene::{
    SdfSceneDescriptor, PersonMask, HybridBandwidthStats,
    rle_encode_mask, rle_decode_mask,
};
use crate::hybrid::{
    HybridTransmitter, HybridReceiver, HybridFrame,
    create_person_mask, estimate_savings,
};

/// RLE encode a binary mask (NumPy zero-copy input).
///
/// Args:
///     mask: Binary mask (H, W) as uint8 NumPy array
///     width: Frame width
///     height: Frame height
///
/// Returns:
///     RLE-encoded bytes
#[pyfunction]
fn rle_encode_mask_numpy(
    mask: PyReadonlyArray2<u8>,
    width: u32,
    height: u32,
) -> PyResult<Vec<u8>> {
    let arr = mask.as_array();
    let slice = arr.as_slice().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err("Mask must be C-contiguous")
    })?;
    Ok(rle_encode_mask(slice, width, height))
}

/// RLE decode to binary mask (returns NumPy array).
///
/// Args:
///     rle: RLE-encoded bytes
///     width: Frame width
///     height: Frame height
///
/// Returns:
///     Binary mask as (H, W) uint8 NumPy array
#[pyfunction]
fn rle_decode_mask_numpy<'py>(
    py: Python<'py>,
    rle: &[u8],
    width: u32,
    height: u32,
) -> PyResult<Bound<'py, PyArray2<u8>>> {
    let decoded = rle_decode_mask(rle, width, height);
    let h = height as usize;
    let w = width as usize;
    decoded.into_pyarray(py).reshape([h, w])
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", e)))
}

/// Estimate bandwidth savings for hybrid streaming.
///
/// Args:
///     frame_width: Frame width
///     frame_height: Frame height
///     person_coverage: Ratio of person pixels (0.0-1.0)
///     sdf_scene_bytes: SDF scene size in bytes
///     wavelet_bpp: Wavelet bits per pixel (default: 1.0)
///
/// Returns:
///     Tuple of (savings_percent, compression_ratio)
#[pyfunction]
#[pyo3(signature = (frame_width, frame_height, person_coverage, sdf_scene_bytes, wavelet_bpp=1.0))]
fn estimate_hybrid_savings(
    frame_width: u32,
    frame_height: u32,
    person_coverage: f32,
    sdf_scene_bytes: usize,
    wavelet_bpp: f32,
) -> (f64, f64) {
    estimate_savings(frame_width, frame_height, person_coverage, sdf_scene_bytes, wavelet_bpp)
}

/// Hybrid transmitter for SDF + person streaming.
#[pyclass]
#[pyo3(name = "HybridTransmitter")]
struct PyHybridTransmitter {
    inner: HybridTransmitter,
}

#[pymethods]
impl PyHybridTransmitter {
    #[new]
    fn new() -> Self {
        Self { inner: HybridTransmitter::new() }
    }

    /// Create a keyframe with SDF scene + person data.
    ///
    /// Args:
    ///     width: Frame width
    ///     height: Frame height
    ///     fps: Frames per second
    ///     asdf_data: ASDF binary blob (SDF scene)
    ///     person_video: Wavelet-encoded person data
    ///     person_mask_rle: Optional RLE person mask bytes
    ///     person_bbox: Optional [x, y, w, h] bounding box
    ///     foreground_pixels: Number of foreground pixels
    ///
    /// Returns:
    ///     Dict with frame info
    #[pyo3(signature = (width, height, fps, asdf_data, person_video, person_mask_rle=None, person_bbox=None, foreground_pixels=0))]
    fn create_keyframe(
        &mut self,
        width: u32,
        height: u32,
        fps: f32,
        asdf_data: Vec<u8>,
        person_video: Vec<u8>,
        person_mask_rle: Option<Vec<u8>>,
        person_bbox: Option<Vec<u32>>,
        foreground_pixels: u32,
    ) -> PyResult<(u32, bool, usize)> {
        let scene = SdfSceneDescriptor::new(asdf_data);
        let mask = match (person_mask_rle, person_bbox) {
            (Some(rle), Some(bbox)) if bbox.len() == 4 => {
                let mut m = PersonMask::new([bbox[0], bbox[1], bbox[2], bbox[3]], rle);
                m.foreground_pixels = foreground_pixels;
                Some(m)
            }
            _ => None,
        };
        let frame = self.inner.create_keyframe(width, height, fps, scene, mask, person_video);
        Ok((frame.sequence, frame.is_keyframe, frame.person_video_data.len()))
    }

    /// Create a delta frame with optional SDF delta + person update.
    ///
    /// Returns:
    ///     Tuple of (sequence, is_keyframe, person_video_size)
    #[pyo3(signature = (person_video, timestamp_ms=0, person_mask_rle=None, person_bbox=None, foreground_pixels=0))]
    fn create_delta_frame(
        &mut self,
        person_video: Vec<u8>,
        timestamp_ms: u64,
        person_mask_rle: Option<Vec<u8>>,
        person_bbox: Option<Vec<u32>>,
        foreground_pixels: u32,
    ) -> PyResult<(u32, bool, usize)> {
        let mask = match (person_mask_rle, person_bbox) {
            (Some(rle), Some(bbox)) if bbox.len() == 4 => {
                let mut m = PersonMask::new([bbox[0], bbox[1], bbox[2], bbox[3]], rle);
                m.foreground_pixels = foreground_pixels;
                Some(m)
            }
            _ => None,
        };
        let frame = self.inner.create_delta_frame(None, mask, person_video, timestamp_ms);
        Ok((frame.sequence, frame.is_keyframe, frame.person_video_data.len()))
    }

    /// Get bandwidth statistics report.
    fn bandwidth_report(&self) -> String {
        self.inner.stats().report()
    }

    /// Get bandwidth savings as (savings_percent, compression_ratio).
    fn savings(&self) -> (f64, f64) {
        self.inner.stats().savings()
    }

    /// Current sequence number.
    fn sequence(&self) -> u32 {
        self.inner.sequence()
    }

    /// Current scene version.
    fn scene_version(&self) -> u32 {
        self.inner.scene_version()
    }
}

/// Hybrid receiver for SDF + person streaming.
#[pyclass]
#[pyo3(name = "HybridReceiver")]
struct PyHybridReceiver {
    inner: HybridReceiver,
}

#[pymethods]
impl PyHybridReceiver {
    #[new]
    fn new() -> Self {
        Self { inner: HybridReceiver::new() }
    }

    /// Process a keyframe. Returns (render_sdf, scene_version, person_bbox, person_video_size).
    #[pyo3(signature = (sequence, width, height, person_video_size, has_scene=true))]
    fn process_keyframe(
        &mut self,
        sequence: u32,
        width: u32,
        height: u32,
        person_video_size: usize,
        has_scene: bool,
    ) -> (bool, u32, Option<Vec<u32>>, usize) {
        let frame = HybridFrame {
            sequence,
            timestamp_ms: 0,
            width,
            height,
            sdf_scene: if has_scene {
                Some(SdfSceneDescriptor::new(vec![b'A', b'S', b'D', b'F',
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]))
            } else {
                None
            },
            sdf_delta: None,
            person_mask: None,
            person_video_data: vec![0; person_video_size],
            is_keyframe: true,
        };
        let instr = self.inner.process_frame(&frame);
        (
            instr.render_sdf_background,
            instr.sdf_scene_version,
            instr.person_bbox.map(|b| b.to_vec()),
            instr.person_video_size,
        )
    }

    /// Process a delta frame. Returns (render_sdf, scene_version, person_bbox, person_video_size).
    fn process_delta(
        &mut self,
        sequence: u32,
        timestamp_ms: u64,
        person_video_size: usize,
    ) -> (bool, u32, Option<Vec<u32>>, usize) {
        let frame = HybridFrame {
            sequence,
            timestamp_ms,
            width: 0,
            height: 0,
            sdf_scene: None,
            sdf_delta: None,
            person_mask: None,
            person_video_data: vec![0; person_video_size],
            is_keyframe: false,
        };
        let instr = self.inner.process_frame(&frame);
        (
            instr.render_sdf_background,
            instr.sdf_scene_version,
            instr.person_bbox.map(|b| b.to_vec()),
            instr.person_video_size,
        )
    }

    /// Number of frames received.
    fn frames_received(&self) -> u32 {
        self.inner.frames_received()
    }
}

// =============================================================================
// Python Module
// =============================================================================

/// ALICE Streaming Protocol - High-performance video streaming codec
#[pymodule]
fn libasp(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Motion estimation
    m.add_function(wrap_pyfunction!(estimate_motion, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_motion_numpy, m)?)?;
    m.add_class::<PyMotionEstimator>()?;

    // Color extraction
    m.add_function(wrap_pyfunction!(extract_colors, m)?)?;
    m.add_function(wrap_pyfunction!(extract_colors_with_weights, m)?)?;

    // DCT transform
    m.add_function(wrap_pyfunction!(dct_2d, m)?)?;
    m.add_function(wrap_pyfunction!(idct_2d, m)?)?;
    m.add_function(wrap_pyfunction!(encode_sparse_dct, m)?)?;
    m.add_function(wrap_pyfunction!(decode_sparse_dct, m)?)?;
    m.add_class::<PyDctTransform>()?;

    // ROI detection
    m.add_function(wrap_pyfunction!(detect_roi, m)?)?;

    // Packet handling
    m.add_function(wrap_pyfunction!(create_i_packet, m)?)?;
    m.add_function(wrap_pyfunction!(create_d_packet, m)?)?;
    m.add_function(wrap_pyfunction!(create_d_packet_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(parse_packet, m)?)?;

    // Hybrid streaming (SDF + wavelet person)
    m.add_function(wrap_pyfunction!(rle_encode_mask_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(rle_decode_mask_numpy, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_hybrid_savings, m)?)?;
    m.add_class::<PyHybridTransmitter>()?;
    m.add_class::<PyHybridReceiver>()?;

    // Utility
    m.add_function(wrap_pyfunction!(version, m)?)?;

    Ok(())
}
