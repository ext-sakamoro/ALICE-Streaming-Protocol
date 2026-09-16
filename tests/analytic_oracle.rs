//! Analytic-oracle tests (CLAUDE.md § 解析解突合テスト規律)
//!
//! Every numerical law in the codec is checked against a closed-form answer or
//! an independent scalar reference, through the default configuration, with
//! its precision / size parameters swept. No golden values.
//!
//! Laws pinned here: orthonormal 2D DCT-II (DC = N·c, Parseval, basis
//! selectivity, exact inverse), JPEG-style quantisation scaling, sparse
//! coefficient coding threshold, block motion estimation on a pure translation
//! (FullSearch recovers any shift with SAD 0 on any texture, the heuristic
//! searches on smooth texture; SIMD paths for 8 / 16 blocks agree with a scalar
//! SAD), CRC-32/ISO-HDLC check value, AIMD bitrate control closed form,
//! k-means palette on separated clusters (exact means, iteration-count
//! independent, weights = cluster fractions), median-cut palette on separated
//! clusters (exact colours, every split axis), ROI detection (Sobel edge
//! strength of a step edge = 2·255/(B−2), block contrast = max − min, motion
//! magnitude = mean |Δ|, thresholds are strict `>`).

#![allow(clippy::too_many_arguments)] // test helpers mirror the estimator signature

use libasp::bitrate::{BitrateConfig, BitrateController};
use libasp::codec::color::{median_cut_palette, ColorExtractor};
use libasp::codec::dct::{dct2d, idct2d, sparse_dct_decode, sparse_dct_encode, DctTransform};
use libasp::codec::motion::{
    estimate_motion, estimate_motion_with, MotionEstimator, SearchAlgorithm,
};
use libasp::codec::roi::{RoiConfig, RoiDetector};
use libasp::header::{crc32, AspPacketHeader};
use libasp::types::{Color, PacketType, Rect, RoiType};
use std::f64::consts::PI;

fn lcg_bytes(n: usize, seed: u64) -> Vec<u8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (s >> 40) as u8
        })
        .collect()
}

// ---------------------------------------------------------------- DCT

#[test]
fn dct_of_constant_block_is_dc_only_with_value_n_times_c() {
    // Orthonormal DCT-II: F(0,0) = (2/N)(1/√2)² Σx = N·c, every other bin 0
    for n in [4usize, 8, 16] {
        for c in [1.0f64, -3.5, 127.0] {
            let block = vec![c; n * n];
            let f = dct2d(&block, n);
            assert!(
                (f[0] - n as f64 * c).abs() < 1e-9 * (n as f64 * c.abs()).max(1.0),
                "n={n} c={c}: {}",
                f[0]
            );
            for (i, v) in f.iter().enumerate().skip(1) {
                assert!(v.abs() < 1e-9, "n={n} c={c} bin {i}: {v}");
            }
        }
    }
}

#[test]
fn dct_is_orthonormal_parseval_and_exactly_invertible() {
    for n in [4usize, 8, 16] {
        let block: Vec<f64> = lcg_bytes(n * n, 7 + n as u64)
            .into_iter()
            .map(|b| f64::from(b) - 128.0)
            .collect();
        let f = dct2d(&block, n);
        let e_in: f64 = block.iter().map(|x| x * x).sum();
        let e_out: f64 = f.iter().map(|x| x * x).sum();
        assert!(
            (e_in - e_out).abs() < 1e-8 * e_in,
            "n={n} Parseval: {e_in} vs {e_out}"
        );
        let back = idct2d(&f, n);
        for (a, b) in back.iter().zip(&block) {
            assert!((a - b).abs() < 1e-9, "n={n}: {a} vs {b}");
        }
    }
}

#[test]
fn dct_basis_function_maps_to_a_single_coefficient() {
    // x(y, x) = cos((2x+1)uπ/2N) cos((2y+1)vπ/2N) → F(u,v) = N/2 · (1/√2 per zero index)
    let n = 8usize;
    for (u, v) in [(1usize, 0usize), (3, 2), (7, 7), (0, 5)] {
        let block: Vec<f64> = (0..n * n)
            .map(|i| {
                let (x, y) = ((i % n) as f64, (i / n) as f64);
                ((2.0 * x + 1.0) * u as f64 * PI / (2.0 * n as f64)).cos()
                    * ((2.0 * y + 1.0) * v as f64 * PI / (2.0 * n as f64)).cos()
            })
            .collect();
        let f = dct2d(&block, n);
        let cu = if u == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
        let cv = if v == 0 { 1.0 / 2f64.sqrt() } else { 1.0 };
        let expected = (n as f64 / 2.0) / (cu * cv);
        for (i, val) in f.iter().enumerate() {
            let want = if i == v * n + u { expected } else { 0.0 };
            assert!(
                (val - want).abs() < 1e-9,
                "(u,v)=({u},{v}) bin {i}: {val} vs {want}"
            );
        }
    }
}

#[test]
fn quantisation_quality_scaling_follows_the_jpeg_law() {
    // quality 50 → scale 100 → matrix unchanged; quality 100 → scale 0 → all 1
    let base = DctTransform::new(8);
    let q50 = DctTransform::new(8).with_quality(50);
    assert_eq!(q50.quant_matrix, base.quant_matrix);
    let q100 = DctTransform::new(8).with_quality(100);
    assert!(q100.quant_matrix.iter().all(|&q| q == 1.0));
    // quality 25 → scale 200 → every entry doubled
    let q25 = DctTransform::new(8).with_quality(25);
    for (a, b) in q25.quant_matrix.iter().zip(&base.quant_matrix) {
        assert!((a - 2.0 * b).abs() < 1e-9);
    }
    // quantise ∘ dequantise error ≤ half a step, per coefficient
    for quality in [1u8, 25, 50, 75, 100] {
        let t = DctTransform::new(8).with_quality(quality);
        let coeffs: Vec<f64> = lcg_bytes(64, u64::from(quality))
            .into_iter()
            .map(|b| (f64::from(b) - 128.0) * 4.0)
            .collect();
        let back = t.dequantize(&t.quantize(&coeffs));
        for ((a, b), q) in back.iter().zip(&coeffs).zip(&t.quant_matrix) {
            assert!(
                (a - b).abs() <= q / 2.0 + 1e-9,
                "quality {quality}: {a} vs {b} (step {q})"
            );
        }
    }
}

#[test]
fn sparse_coding_drops_only_coefficients_at_or_below_the_threshold() {
    let n = 8usize;
    let coeffs: Vec<i32> = (0..64i32).map(|i| (i % 7) - 3).collect(); // -3..=3
    let sparse = sparse_dct_encode(&coeffs, n, 0.001); // threshold_i = 1 → keeps |c| > 1
    assert!(sparse.iter().all(|&(_, _, v)| v.abs() > 1.0));
    assert_eq!(sparse.len(), coeffs.iter().filter(|c| c.abs() > 1).count());
    let back = sparse_dct_decode(&sparse, n, 0);
    for (b, c) in back.iter().zip(&coeffs) {
        assert_eq!(*b, if c.abs() > 1 { *c } else { 0 });
    }
    // Encode → decode of a block through the transform reproduces it within
    // the quantisation step where coefficients were kept
    let t = DctTransform::new(n).with_quality(100);
    let block: Vec<f64> = lcg_bytes(64, 3).into_iter().map(f64::from).collect();
    let sparse = t.encode_sparse(&block);
    let back = t.decode_sparse(&sparse, 0.0);
    // With quality 100 every quantiser step is 1, so dropping |c| ≤ 1 bins costs
    // at most Σ|c| ≤ 64 in energy → per-sample error bounded by 64 / 8 = 8
    for (a, b) in back.iter().zip(&block) {
        assert!((a - b).abs() <= 8.0 + 1e-9, "{a} vs {b}");
    }
}

// ---------------------------------------------------------------- motion

/// Scalar SAD reference: Σ |cur[block] − prev[block + (dx, dy)]|
fn sad_ref(
    cur: &[u8],
    prev: &[u8],
    width: usize,
    bx: usize,
    by: usize,
    bs: usize,
    dx: i32,
    dy: i32,
) -> u32 {
    let mut s = 0u32;
    for y in 0..bs {
        for x in 0..bs {
            let cx = bx * bs + x;
            let cy = by * bs + y;
            let px = (cx as i32 + dx) as usize;
            let py = (cy as i32 + dy) as usize;
            s +=
                (i32::from(cur[cy * width + cx]) - i32::from(prev[py * width + px])).unsigned_abs();
        }
    }
    s
}

fn shifted(prev: &[u8], w: usize, h: usize, sx: i32, sy: i32) -> Vec<u8> {
    // current(x, y) = previous(x − sx, y − sy), zero outside the frame
    let mut cur = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            let px = x as i32 - sx;
            let py = y as i32 - sy;
            cur[y * w + x] = if (0..w as i32).contains(&px) && (0..h as i32).contains(&py) {
                prev[py as usize * w + px as usize]
            } else {
                0
            };
        }
    }
    cur
}

fn smooth_texture(w: usize, h: usize) -> Vec<u8> {
    (0..w * h)
        .map(|i| {
            let (x, y) = ((i % w) as f64, (i / w) as f64);
            (128.0 + 100.0 * (x * 0.21).sin() * (y * 0.17).cos()) as u8
        })
        .collect()
}

/// Every interior block (shifted texture fully inside the frame) must report
/// exactly `(−sx, −sy)` with SAD 0, and the SAD must equal the scalar reference
fn assert_translation_recovered(
    cur: &[u8],
    prev: &[u8],
    w: usize,
    h: usize,
    bs: usize,
    range: usize,
    alg: SearchAlgorithm,
    sx: i32,
    sy: i32,
) {
    // Zero vectors are omitted from the output (bandwidth law), so index the
    // result by block; early threshold 0 disables the static shortcut
    let mvs = estimate_motion_with(cur, prev, w, h, bs, range, alg, 0);
    let (bxs, bys) = (w / bs, h / bs);
    assert!(mvs.len() <= bxs * bys);
    let by_block: std::collections::HashMap<(usize, usize), _> = mvs
        .iter()
        .map(|mv| ((usize::from(mv.block_x), usize::from(mv.block_y)), mv))
        .collect();
    for by in 1..bys - 1 {
        for bx in 1..bxs - 1 {
            if (bx * bs) as i32 - sx < 0 || (by * bs) as i32 - sy < 0 {
                continue;
            }
            let mv = by_block.get(&(bx, by)).unwrap_or_else(|| {
                panic!("{alg:?} shift ({sx},{sy}) bs={bs} range={range}: block ({bx},{by}) reported static")
            });
            assert_eq!(
                (mv.dx, mv.dy, mv.sad),
                (-sx as i16, -sy as i16, 0),
                "{alg:?} shift ({sx},{sy}) bs={bs} range={range} block ({bx},{by}): {mv:?}"
            );
            assert_eq!(
                mv.sad,
                sad_ref(cur, prev, w, bx, by, bs, i32::from(mv.dx), i32::from(mv.dy))
            );
        }
    }
}

#[test]
fn full_search_recovers_any_translation_on_any_texture() {
    // Exhaustive search = global SAD minimum: exact for noise and smooth
    // textures, every block size (4 = scalar SAD, 8 / 16 = SIMD SAD), every
    // range ≥ |shift|
    let (w, h) = (96usize, 64usize);
    for prev in [lcg_bytes(w * h, 99), smooth_texture(w, h)] {
        for (sx, sy) in [(1i32, 0i32), (0, 2), (3, -2), (-2, 3)] {
            let cur = shifted(&prev, w, h, sx, sy);
            for bs in [4usize, 8, 16] {
                for range in [3usize, 7, 16] {
                    assert_translation_recovered(
                        &cur,
                        &prev,
                        w,
                        h,
                        bs,
                        range,
                        SearchAlgorithm::FullSearch,
                        sx,
                        sy,
                    );
                }
            }
        }
    }
}

#[test]
fn heuristic_searches_recover_translations_on_smooth_texture() {
    // Diamond / hexagon follow the SAD gradient: exact on smooth content for
    // 8 / 16 blocks (they cannot be exact on white noise, where SAD has no
    // gradient — that is the documented reason FullSearch exists). Three-step
    // needs the 16-pixel block (its coarse first step overshoots small motion
    // on 8×8 blocks)
    let (w, h) = (96usize, 64usize);
    let prev = smooth_texture(w, h);
    for (sx, sy) in [(1i32, 0i32), (0, 2), (3, -2), (-2, 3)] {
        let cur = shifted(&prev, w, h, sx, sy);
        for range in [4usize, 7, 16] {
            for bs in [8usize, 16] {
                for alg in [
                    SearchAlgorithm::DiamondSearch,
                    SearchAlgorithm::HexagonSearch,
                ] {
                    assert_translation_recovered(&cur, &prev, w, h, bs, range, alg, sx, sy);
                }
            }
            assert_translation_recovered(
                &cur,
                &prev,
                w,
                h,
                16,
                range,
                SearchAlgorithm::ThreeStepSearch,
                sx,
                sy,
            );
        }
    }
}

#[test]
fn static_frame_reports_no_vectors_and_a_single_moved_block_is_the_only_entry() {
    let (w, h) = (64usize, 48usize);
    let frame = lcg_bytes(w * h, 5);
    for alg in [
        SearchAlgorithm::FullSearch,
        SearchAlgorithm::ThreeStepSearch,
        SearchAlgorithm::DiamondSearch,
        SearchAlgorithm::HexagonSearch,
    ] {
        for bs in [4usize, 8, 16] {
            assert!(
                estimate_motion_with(&frame, &frame, w, h, bs, 7, alg, 0).is_empty(),
                "{alg:?} bs={bs}"
            );
        }
    }
    // Default API (diamond, threshold 256) on one moved 16×16 block
    let mut moved = frame.clone();
    for y in 16..32 {
        for x in 16..32 {
            moved[y * w + x] = frame[y * w + x - 1];
        }
    }
    let mvs = estimate_motion(&moved, &frame, w, h, 16, 7);
    assert_eq!(mvs.len(), 1, "{mvs:?}");
    assert_eq!(
        (
            mvs[0].block_x,
            mvs[0].block_y,
            mvs[0].dx,
            mvs[0].dy,
            mvs[0].sad
        ),
        (1, 1, -1, 0, 0)
    );
}

#[test]
fn search_algorithm_parameter_is_honoured() {
    // Until 1.1.0 every SearchAlgorithm ran diamond search. On white noise
    // FullSearch finds SAD 0 for a 1-pixel shift where diamond search does not
    // (no SAD gradient), so the two must differ
    let (w, h) = (96usize, 64usize);
    let prev = lcg_bytes(w * h, 99);
    let cur = shifted(&prev, w, h, 1, 0);
    let full = estimate_motion_with(&cur, &prev, w, h, 16, 7, SearchAlgorithm::FullSearch, 0);
    let diamond = estimate_motion_with(&cur, &prev, w, h, 16, 7, SearchAlgorithm::DiamondSearch, 0);
    // Interior blocks (the shifted-in border column has no exact match)
    let interior = |mv: &&libasp::types::MotionVector| mv.block_x >= 1 && mv.block_y >= 1;
    assert!(full.iter().filter(interior).all(|mv| mv.sad == 0));
    assert!(diamond.iter().filter(interior).any(|mv| mv.sad != 0));
    assert_ne!(full, diamond);
    let est = MotionEstimator::new(16, 7).with_algorithm(SearchAlgorithm::FullSearch);
    assert_eq!(est.estimate(&cur, &prev, w, h).len(), full.len());
}

// ---------------------------------------------------------------- crc / header

#[test]
fn crc32_is_iso_hdlc_check_value() {
    // CRC-32/ISO-HDLC (reflected 0xEDB88320, init/xorout 0xFFFFFFFF): check("123456789") = 0xCBF43926
    assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    assert_eq!(crc32(b""), 0);
    // Linearity check: crc(a ++ b) depends on both parts
    assert_ne!(crc32(b"ab"), crc32(b"ba"));
}

#[test]
fn header_round_trip_is_exact_for_every_packet_type() {
    for (i, pt) in [
        PacketType::IPacket,
        PacketType::DPacket,
        PacketType::CPacket,
        PacketType::SPacket,
    ]
    .into_iter()
    .enumerate()
    {
        let h = AspPacketHeader::new(pt, 0xDEAD_0000 + i as u32, 1234 * i as u32);
        let bytes = h.to_bytes();
        assert_eq!(bytes.len(), AspPacketHeader::SIZE);
        let back = AspPacketHeader::from_bytes(&bytes).unwrap();
        assert_eq!(back, h);
        // write_to_ptr produces the same bytes as to_bytes
        let mut buf = [0u8; AspPacketHeader::SIZE];
        // SAFETY: buf is exactly SIZE bytes
        unsafe { h.write_to_ptr(buf.as_mut_ptr()) };
        assert_eq!(buf, bytes);
    }
    assert!(AspPacketHeader::from_bytes(&[0u8; 3]).is_err());
}

// ---------------------------------------------------------------- AIMD

#[test]
fn aimd_bitrate_control_closed_form() {
    let cfg = BitrateConfig::default();
    let init = 500_000u64;
    let mut ctrl = BitrateController::with_config(init, cfg);
    // k ACKs: init + k · additive, clamped at max
    for k in 1..=10u64 {
        ctrl.on_ack(1024, 50);
        assert_eq!(
            ctrl.target_bps(),
            (init + k * cfg.additive_increase_bps).min(cfg.max_bps)
        );
    }
    let before = ctrl.target_bps();
    // one loss: floor(bps · decrease), clamped at min
    ctrl.on_loss();
    assert_eq!(
        ctrl.target_bps(),
        ((before as f64 * cfg.multiplicative_decrease) as u64).max(cfg.min_bps)
    );
    // repeated losses converge to min_bps and never below
    for _ in 0..64 {
        ctrl.on_loss();
    }
    assert_eq!(ctrl.target_bps(), cfg.min_bps);
    // saturation at max_bps
    let mut hi = BitrateController::with_config(cfg.max_bps - 1, cfg);
    hi.on_ack(1, 1);
    hi.on_ack(1, 1);
    assert_eq!(hi.target_bps(), cfg.max_bps);
    // loss rate = losses / (acks + losses)
    let mut c = BitrateController::with_config(init, cfg);
    for _ in 0..3 {
        c.on_ack(1, 1);
    }
    c.on_loss();
    assert!((c.loss_rate() - 0.25).abs() < 1e-12);
}

// ---------------------------------------------------------------- k-means palette

#[test]
fn kmeans_palette_recovers_separated_cluster_means_independent_of_iterations() {
    // Three tight clusters around exact centres; with sampling 1.0 the palette
    // must be exactly the centres for every iteration count ≥ 2
    let centres = [(20u8, 30u8, 40u8), (200, 60, 80), (90, 220, 150)];
    let mut pixels = Vec::new();
    for (i, &(r, g, b)) in centres.iter().enumerate() {
        for j in 0..200u8 {
            let d = (j % 5) as i16 - 2; // −2..=2 jitter, mean 0 over 200 samples
            let px = |c: u8| (i16::from(c) + d) as u8;
            pixels.extend_from_slice(&[px(r), px(g), px(b)]);
            let _ = i;
        }
    }
    let mut expected: Vec<Color> = centres
        .iter()
        .map(|&(r, g, b)| Color::new(r, g, b))
        .collect();
    expected.sort_by_key(|c| (c.r, c.g, c.b));
    let mut previous: Option<Vec<Color>> = None;
    for iters in [2usize, 3, 5, 8, 20] {
        let mut palette = ColorExtractor::new(3)
            .with_iterations(iters)
            .with_sampling_rate(1.0)
            .extract(&pixels);
        palette.sort_by_key(|c| (c.r, c.g, c.b));
        for (p, e) in palette.iter().zip(&expected) {
            assert!(
                (i16::from(p.r) - i16::from(e.r)).abs() <= 1
                    && (i16::from(p.g) - i16::from(e.g)).abs() <= 1
                    && (i16::from(p.b) - i16::from(e.b)).abs() <= 1,
                "iters={iters}: {p:?} vs {e:?}"
            );
        }
        if let Some(prev) = &previous {
            assert_eq!(
                &palette, prev,
                "iteration count {iters} changed the converged palette"
            );
        }
        previous = Some(palette);
    }
}

// ----------------------------------------------------------------------------
// Palettes: median cut on separated clusters, k-means weights
// ----------------------------------------------------------------------------

fn cluster_pixels(centres: &[(u8, u8, u8)], counts: &[usize]) -> Vec<u8> {
    let mut px = Vec::new();
    for (&(r, g, b), &n) in centres.iter().zip(counts) {
        for _ in 0..n {
            px.extend_from_slice(&[r, g, b]);
        }
    }
    px
}

fn sorted(mut v: Vec<Color>) -> Vec<Color> {
    v.sort_by_key(|c| (c.r, c.g, c.b));
    v
}

#[test]
fn median_cut_recovers_separated_colours_exactly_on_every_split_axis() {
    // Four constant clusters, num_colors = 4 → two splits, every leaf holds one
    // colour, the leaf mean is that colour. The dominant axis of the first split
    // differs per set so the r / g / b branches are all exercised
    let sets: [[(u8, u8, u8); 4]; 3] = [
        [(10, 5, 5), (90, 5, 5), (170, 5, 5), (250, 5, 5)], // red spread
        [(5, 10, 5), (5, 90, 5), (5, 170, 5), (5, 250, 5)], // green spread
        [(5, 5, 10), (5, 5, 90), (5, 5, 170), (5, 5, 250)], // blue spread
    ];
    for centres in sets {
        let px = cluster_pixels(&centres, &[7, 7, 7, 7]);
        let expected = sorted(
            centres
                .iter()
                .map(|&(r, g, b)| Color::new(r, g, b))
                .collect(),
        );
        assert_eq!(sorted(median_cut_palette(&px, 4)), expected, "{centres:?}");
        // one colour requested: the mean of everything (integer floor per channel)
        let n = 28u32;
        let mean = |f: fn(&(u8, u8, u8)) -> u8| {
            (centres.iter().map(|c| u32::from(f(c)) * 7).sum::<u32>() / n) as u8
        };
        assert_eq!(
            median_cut_palette(&px, 1),
            vec![Color::new(mean(|c| c.0), mean(|c| c.1), mean(|c| c.2))],
            "{centres:?} mean"
        );
    }
    // fewer than one pixel, or zero colours requested → black (documented sentinel)
    assert_eq!(median_cut_palette(&[1, 2], 4), vec![Color::black()]);
    assert_eq!(median_cut_palette(&[1, 2, 3], 0), vec![Color::black()]);
    // a single pixel is its own palette
    assert_eq!(median_cut_palette(&[9, 8, 7], 4), vec![Color::new(9, 8, 7)]);
}

#[test]
fn kmeans_weights_are_cluster_fractions_sorted_descending() {
    let centres = [(20u8, 30u8, 40u8), (200, 60, 80), (90, 220, 150)];
    let counts = [300usize, 100, 200];
    let px = cluster_pixels(&centres, &counts);
    let (colors, weights) = ColorExtractor::new(3)
        .with_iterations(10)
        .with_sampling_rate(1.0)
        .extract_with_weights(&px);
    assert_eq!(colors.len(), 3);
    assert_eq!(weights.len(), 3);
    // descending fractions 300/600, 200/600, 100/600 attached to the right colours
    let expected = [
        (Color::new(20, 30, 40), 0.5f32),
        (Color::new(90, 220, 150), 1.0 / 3.0),
        (Color::new(200, 60, 80), 1.0 / 6.0),
    ];
    for ((c, w), (ec, ew)) in colors.iter().zip(&weights).zip(&expected) {
        assert_eq!(c, ec);
        assert!((w - ew).abs() < 1e-6, "{w} vs {ew}");
    }
    assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    // a plain `extract` is the same palette in the same order
    assert_eq!(
        ColorExtractor::new(3)
            .with_iterations(10)
            .with_sampling_rate(1.0)
            .extract(&px),
        colors
    );
}

// ----------------------------------------------------------------------------
// ROI detection laws (default RoiConfig: block 16, edge > 30, contrast > 50,
// motion > 20)
// ----------------------------------------------------------------------------

/// Vertical step edge at column `x0` (0 left of it, 255 from it on)
fn vertical_step(w: usize, h: usize, x0: usize) -> Vec<u8> {
    (0..w * h)
        .map(|i| if i % w >= x0 { 255 } else { 0 })
        .collect()
}

/// Horizontal step edge at row `y0`
fn horizontal_step(w: usize, h: usize, y0: usize) -> Vec<u8> {
    (0..w * h)
        .map(|i| if i / w >= y0 { 255 } else { 0 })
        .collect()
}

#[test]
fn flat_frame_has_no_regions_and_a_ramp_is_contrast_only() {
    let (w, h) = (64usize, 64usize);
    let det = RoiDetector::default();
    assert!(det.detect(&vec![77u8; w * h], w, h).is_empty());
    assert!(det
        .detect_with_motion(&vec![77u8; w * h], &vec![77u8; w * h], w, h)
        .is_empty());

    // frame[x] = 4x: gradient 8 per pixel → edge strength 8 (< 30, no edge
    // region); per-block contrast = 4·15 = 60 (> 50) → a Text region per block
    // with confidence 60/255
    let ramp: Vec<u8> = (0..w * h).map(|i| (4 * (i % w)) as u8).collect();
    let regions = det.detect(&ramp, w, h);
    assert_eq!(regions.len(), 16);
    for r in &regions {
        assert_eq!(r.roi_type, RoiType::Text);
        assert_eq!((r.bounds.width, r.bounds.height), (16, 16));
        assert!(
            (r.confidence - 60.0 / 255.0).abs() < 1e-6,
            "{}",
            r.confidence
        );
    }
    // frame[x] = x: gradient 2, contrast 15 → nothing at all
    let gentle: Vec<u8> = (0..w * h).map(|i| (i % w) as u8).collect();
    assert!(det.detect(&gentle, w, h).is_empty());
}

#[test]
fn step_edge_strength_is_two_times_255_over_block_minus_two() {
    // Sobel-style |right − left| is 255 at the two columns straddling the edge,
    // 0 elsewhere; averaged over the (B−2)² interior: 2·255·(B−2)/(B−2)² =
    // 510/(B−2) = 36 for B = 16. The block also has contrast 255, and the
    // edge + contrast regions of one block merge into one region with the
    // maximum confidence (1.0) and the edge type first in sort order
    let (w, h) = (64usize, 64usize);
    let det = RoiDetector::default();
    // (frame, bounds of the i-th region: the edge column / row of blocks)
    let cases = [
        (vertical_step(w, h, 40), true),
        (horizontal_step(w, h, 40), false),
    ];
    for (frame, vertical) in cases {
        let expect_bounds = |i: u32| {
            if vertical {
                Rect::new(32, 16 * i, 16, 16)
            } else {
                Rect::new(16 * i, 32, 16, 16)
            }
        };
        let mut regions = det.detect(&frame, w, h);
        regions.sort_by_key(|r| (r.bounds.y, r.bounds.x));
        assert_eq!(regions.len(), 4, "{regions:?}");
        for (i, r) in regions.iter().enumerate() {
            assert_eq!(r.bounds, expect_bounds(i as u32));
            assert_eq!(r.roi_type, RoiType::Edge);
            assert!((r.confidence - 1.0).abs() < 1e-6);
        }
        // Contrast off: only the edge regions remain, confidence = 36/255
        let edge_only = RoiDetector::new(RoiConfig {
            detect_contrast: false,
            ..RoiConfig::default()
        });
        let regions = edge_only.detect(&frame, w, h);
        assert_eq!(regions.len(), 4);
        for r in &regions {
            assert_eq!(r.roi_type, RoiType::Edge);
            assert!(
                (r.confidence - 36.0 / 255.0).abs() < 1e-6,
                "{}",
                r.confidence
            );
        }
        // Threshold exactly at the strength: strict `>` → nothing
        let at_threshold = RoiDetector::new(RoiConfig {
            detect_contrast: false,
            edge_threshold: 36,
            ..RoiConfig::default()
        });
        assert!(at_threshold.detect(&frame, w, h).is_empty());
        let below = RoiDetector::new(RoiConfig {
            detect_contrast: false,
            edge_threshold: 35,
            ..RoiConfig::default()
        });
        assert_eq!(below.detect(&frame, w, h).len(), 4);
    }
    // An edge exactly on a block boundary is invisible to the interior-only
    // Sobel pass (documented limitation): only the contrast pass sees it
    let boundary = vertical_step(w, h, 32);
    let edge_only = RoiDetector::new(RoiConfig {
        detect_contrast: false,
        ..RoiConfig::default()
    });
    assert!(edge_only.detect(&boundary, w, h).is_empty());
}

#[test]
fn motion_magnitude_is_mean_absolute_difference_with_strict_threshold() {
    let (w, h) = (64usize, 48usize);
    let prev = vec![100u8; w * h];
    let det = RoiDetector::default();
    for d in [19u8, 20, 21, 25, 155] {
        let mut cur = prev.clone();
        // one 16×16 block at (bx, by) = (1, 2) raised by d
        for y in 32..48 {
            for x in 16..32 {
                cur[y * w + x] = 100 + d;
            }
        }
        let regions = det.detect_with_motion(&cur, &prev, w, h);
        if d <= 20 {
            assert!(regions.is_empty(), "d={d}: {regions:?}");
        } else {
            // the raised block is uniform (contrast 0, Sobel 0) and block-aligned,
            // so the motion block is the only region for every d
            assert_eq!(regions.len(), 1, "d={d}: {regions:?}");
            let r = &regions[0];
            assert_eq!(r.bounds, Rect::new(16, 32, 16, 16));
            assert_eq!(r.roi_type, RoiType::Motion);
            assert!(
                (r.confidence - f32::from(d) / 255.0).abs() < 1e-6,
                "{}",
                r.confidence
            );
        }
    }
    // Half of a block moved by 40: mean |Δ| over the block = 20 → not > 20
    let mut cur = prev.clone();
    for y in 32..40 {
        for x in 16..32 {
            cur[y * w + x] = 140;
        }
    }
    let regions = RoiDetector::new(RoiConfig {
        detect_edges: false,
        detect_contrast: false,
        ..RoiConfig::default()
    })
    .detect_with_motion(&cur, &prev, w, h);
    assert!(regions.is_empty(), "{regions:?}");
}
