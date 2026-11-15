//! Pins of the outputs that move with G1 (the platform libm calls in the library), recorded
//! before the conversion
//!
//! `tests/fixtures/g1/pins.txt` holds, one line per item, `<name> <kind> <values>`: the 2D DCT and
//! its inverse on fixed blocks (f64 bits), the quantised and sparse coefficients and the decoded
//! pixels of `DctTransform` at two qualities, a whole I-packet carrying a 64x64 frame's sparse
//! coefficients (its bytes' SHA-256) and the frame decoded back from it, `MotionVector::magnitude`
//! on a grid of vectors (f32 bits), and `median_cut_palette` for every colour count 1..=64. The
//! values were written by the platform libm path (cos, hypot, log2, powi) before the conversion;
//! the test `the_outputs_match_the_pins` compares the present code against them and prints what
//! moved and by how much.
//!
//! Author: Moroya Sakamoto

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use libasp::codec::color::median_cut_palette;
use libasp::codec::{dct2d, idct2d, DctTransform};
use libasp::{
    AspPacket, Color, ColorPalette, IPacketPayload, MotionVector, PatternType, Rect,
    RegionDescriptor,
};

fn path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/g1/pins.txt")
}

/// 0..=255 values from a fixed linear congruential sequence
fn lcg_block(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed;
    (0..n * n)
        .map(|_| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) % 256) as f64
        })
        .collect()
}

fn blocks(n: usize) -> Vec<(String, Vec<f64>)> {
    vec![
        (
            format!("ramp{n}"),
            (0..n * n).map(|i| (i * 255 / (n * n - 1)) as f64).collect(),
        ),
        (
            format!("checker{n}"),
            (0..n * n)
                .map(|i| {
                    if (i % n + i / n).is_multiple_of(2) {
                        255.0
                    } else {
                        0.0
                    }
                })
                .collect(),
        ),
        (format!("lcg{n}"), lcg_block(n, n as u64)),
    ]
}

/// A 64x64 grey frame: smooth plus a fixed pseudo-random texture
fn frame() -> Vec<f64> {
    let noise = lcg_block(64, 7);
    (0..64 * 64)
        .map(|i| {
            let (x, y) = ((i % 64) as f64, (i / 64) as f64);
            (x * 2.0 + y + noise[i] / 4.0).min(255.0)
        })
        .collect()
}

fn block_of(frame: &[f64], bx: usize, by: usize) -> Vec<f64> {
    (0..64)
        .map(|i| frame[(by * 8 + i / 8) * 64 + bx * 8 + i % 8])
        .collect()
}

fn hex_f64(v: &[f64]) -> String {
    v.iter()
        .map(|x| format!("{:016x}", x.to_bits()))
        .collect::<Vec<_>>()
        .join(",")
}

fn sha256_hex(b: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(b).iter().fold(String::new(), |mut s, x| {
        let _ = write!(s, "{x:02x}");
        s
    })
}

/// The stream: one I-packet whose regions carry each 8x8 block's sparse coefficients of
/// `frame()` (`DctTransform` quality 75), as bytes
fn stream() -> Vec<u8> {
    let t = DctTransform::new(8).with_quality(75);
    let f = frame();
    let mut payload = IPacketPayload::new(64, 64, 30.0);
    payload.timestamp_ms = 1;
    for by in 0..8 {
        for bx in 0..8 {
            payload.regions.push(RegionDescriptor {
                bounds: Rect::new(bx * 8, by * 8, 8, 8),
                pattern_type: PatternType::Dct,
                palette: ColorPalette {
                    colors: vec![Color::new(0, 0, 0)],
                    weights: None,
                },
                dct_coefficients: Some(t.encode_sparse(&block_of(&f, bx as usize, by as usize))),
                texture_id: None,
                params: None,
            });
        }
    }
    AspPacket::create_i_packet(1, payload)
        .expect("packs")
        .to_bytes()
        .expect("bytes")
}

fn stream_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/g1/iframe.bin")
}

/// The 64x64 frame decoded from an I-packet's bytes (each region's sparse coefficients through
/// `DctTransform` quality 75, rounded and clamped to u8)
fn decode_stream(bytes: &[u8]) -> Vec<u8> {
    let t = DctTransform::new(8).with_quality(75);
    let back = AspPacket::from_bytes(bytes).expect("reads");
    let i = back.as_i_packet().expect("I");
    let mut pixels = vec![0u8; 64 * 64];
    for r in &i.regions {
        let block = t.decode_sparse(r.dct_coefficients.as_ref().expect("coefficients"), 0.0);
        for (k, v) in block.iter().enumerate() {
            let (x, y) = (r.bounds.x as usize + k % 8, r.bounds.y as usize + k / 8);
            pixels[y * 64 + x] = v.round().clamp(0.0, 255.0) as u8;
        }
    }
    pixels
}

/// Every pinned item as (name, kind, values) computed by the present code
fn items() -> Vec<(String, &'static str, String)> {
    let mut out = Vec::new();
    for n in [2usize, 4, 8, 16] {
        for (name, b) in blocks(n) {
            let fwd = dct2d(&b, n);
            out.push((format!("dct_fwd_{name}"), "f64", hex_f64(&fwd)));
            out.push((format!("dct_inv_{name}"), "f64", hex_f64(&idct2d(&fwd, n))));
        }
    }
    for q in [50u8, 90] {
        let t = DctTransform::new(8).with_quality(q);
        for (name, b) in blocks(8) {
            let quant = t.quantize(&t.forward(&b));
            out.push((
                format!("quant_q{q}_{name}"),
                "i32",
                quant
                    .iter()
                    .map(i32::to_string)
                    .collect::<Vec<_>>()
                    .join(","),
            ));
            let sparse = t.encode_sparse(&b);
            out.push((
                format!("sparse_q{q}_{name}"),
                "sparse",
                sparse
                    .iter()
                    .map(|(a, c, v)| format!("{a}:{c}:{:08x}", v.to_bits()))
                    .collect::<Vec<_>>()
                    .join(","),
            ));
            out.push((
                format!("decoded_q{q}_{name}"),
                "f64",
                hex_f64(&t.decode_sparse(&sparse, 0.0)),
            ));
        }
    }
    let bytes = stream();
    out.push(("stream_iframe_sha256".into(), "hex", sha256_hex(&bytes)));
    let pixels = decode_stream(&bytes);
    out.push((
        "stream_iframe_pixels".into(),
        "u8",
        pixels
            .iter()
            .map(u8::to_string)
            .collect::<Vec<_>>()
            .join(","),
    ));
    let mut mags = Vec::new();
    for dx in (-32768i32..=32767).step_by(4099) {
        for dy in (-32768i32..=32767).step_by(3001) {
            mags.push(MotionVector::new(0, 0, dx as i16, dy as i16, 0).magnitude());
        }
    }
    for (dx, dy) in [
        (3i16, 4i16),
        (0, 0),
        (1, 1),
        (i16::MAX, i16::MAX),
        (i16::MIN, i16::MIN),
        (i16::MIN, 0),
    ] {
        mags.push(MotionVector::new(0, 0, dx, dy, 0).magnitude());
    }
    out.push((
        "magnitude_grid".into(),
        "f32",
        mags.iter()
            .map(|m| format!("{:08x}", m.to_bits()))
            .collect::<Vec<_>>()
            .join(","),
    ));
    let pixels: Vec<u8> = lcg_block(32, 11).iter().map(|v| *v as u8).collect();
    for k in 1..=64 {
        let p = median_cut_palette(&pixels, k);
        out.push((
            format!("median_cut_{k}"),
            "rgb",
            p.iter()
                .map(|c| format!("{:02x}{:02x}{:02x}", c.r, c.g, c.b))
                .collect::<Vec<_>>()
                .join(","),
        ));
    }
    out
}

/// Writes the pins (run once, before the conversion:
/// `ALICE_ASP_WRITE_G1_PINS=1 cargo test --test g1_conversion_pins write -- --ignored`).
#[test]
#[ignore = "writes tests/fixtures/g1/pins.txt; run only to capture"]
fn write_g1_pins() {
    if std::env::var_os("ALICE_ASP_WRITE_G1_PINS").is_none() {
        return;
    }
    std::fs::create_dir_all(path().parent().unwrap()).unwrap();
    std::fs::write(stream_path(), stream()).unwrap();
    let text: String = items()
        .iter()
        .map(|(n, k, v)| format!("{n} {k} {v}\n"))
        .collect();
    std::fs::write(path(), text).unwrap();
}

fn pins() -> Vec<(String, String, String)> {
    let text = std::fs::read_to_string(path()).expect("pins.txt");
    text.lines()
        .map(|l| {
            let mut it = l.splitn(3, ' ');
            let (n, k, v) = (
                it.next().unwrap(),
                it.next().unwrap(),
                it.next().unwrap_or(""),
            );
            (n.to_string(), k.to_string(), v.to_string())
        })
        .collect()
}

/// ulp distance of two f64 bit patterns of the same sign (or both zero)
fn ulp64(a: u64, b: u64) -> u64 {
    let (fa, fb) = (f64::from_bits(a), f64::from_bits(b));
    if fa == fb {
        return 0;
    }
    let key = |x: u64| {
        if x >> 63 == 1 {
            -((x & !(1 << 63)) as i128)
        } else {
            x as i128
        }
    };
    key(a).abs_diff(key(b)) as u64
}

#[test]
fn the_outputs_match_the_pins() {
    let pinned = pins();
    let now = items();
    assert_eq!(pinned.len(), now.len(), "one pin per item");
    assert!(pinned.len() > 100, "{}", pinned.len());
    let mut report = String::new();
    let mut differing = 0;
    for ((pn, pk, pv), (nn, nk, nv)) in pinned.iter().zip(&now) {
        assert_eq!((pn.as_str(), pk.as_str()), (nn.as_str(), *nk), "item order");
        if pv == nv {
            continue;
        }
        differing += 1;
        let detail = if *nk == "f64" {
            let max = pv
                .split(',')
                .zip(nv.split(','))
                .map(|(a, b)| {
                    ulp64(
                        u64::from_str_radix(a, 16).unwrap(),
                        u64::from_str_radix(b, 16).unwrap(),
                    )
                })
                .max()
                .unwrap_or(0);
            format!("max {max} ulp")
        } else {
            let n = pv
                .split(',')
                .zip(nv.split(','))
                .filter(|(a, b)| a != b)
                .count();
            format!("{n} value(s) differ")
        };
        let _ = writeln!(report, "{pn}: {detail}");
    }
    println!(
        "g1 pins: {} items, {differing} differ\n{report}",
        pinned.len()
    );
    assert_eq!(differing, 0, "{report}");
}

/// The stream the platform libm path wrote (`tests/fixtures/g1/iframe.bin`) still decodes, and to the
/// pinned pixels: a reader built after the conversion reads what was written before it
#[test]
fn the_stream_written_before_decodes_to_the_pinned_pixels() {
    let old = std::fs::read(stream_path()).expect("iframe.bin");
    let pinned = pins();
    let sha = &pinned
        .iter()
        .find(|p| p.0 == "stream_iframe_sha256")
        .expect("pin")
        .2;
    assert_eq!(&sha256_hex(&old), sha, "iframe.bin is the pinned stream");
    let want = &pinned
        .iter()
        .find(|p| p.0 == "stream_iframe_pixels")
        .expect("pin")
        .2;
    let got: Vec<String> = decode_stream(&old).iter().map(u8::to_string).collect();
    let differ = want.split(',').zip(&got).filter(|(a, b)| a != b).count();
    println!(
        "old stream decoded now: {differ} of {} pixels differ",
        got.len()
    );
    assert_eq!(differ, 0);
}
