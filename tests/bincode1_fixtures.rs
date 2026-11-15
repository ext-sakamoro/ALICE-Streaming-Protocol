//! Fixtures of the `bincode-compat` wire bytes, captured from bincode 1.3.3
//!
//! `tests/fixtures/bincode1/` holds the bytes `to_bytes_bincode` wrote for one packet of each
//! type while this crate used bincode 1.3.3, the I- and D-Packets carrying the fields only
//! this path can carry (`sdf_scene`, `sdf_delta`, `person_mask`); `SHA256SUMS` names each
//! file. The tests pin that the bytes written now are those bytes, that they decode to the
//! same packet, and that writing the same packet again gives the same bytes.
#![cfg(feature = "bincode-compat")]

use std::path::{Path, PathBuf};

use libasp::{
    AspPacket, Color, CompressionFormat, CorrectionData, DPacketPayload, IPacketPayload,
    MotionVector, PersonMask, Rect, RegionDescriptor, RoiRegion, RoiType, SPacketPayload,
    SdfSceneDelta, SdfSceneDescriptor,
};

fn i_packet() -> AspPacket {
    let mut i = IPacketPayload::new(1280, 720, 30.0);
    i.timestamp_ms = 1_234_567;
    i.add_region(RegionDescriptor::solid(
        Rect::new(0, 0, 640, 360),
        Color::new(10, 200, 30),
    ));
    let mut scene = SdfSceneDescriptor::new(vec![b'A', b'S', b'D', b'F', 2, 0, 0, 0, 9, 8, 7])
        .with_bounds([-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]);
    scene.scene_name = Some("lobby".to_string());
    scene.classification_labels = Some(vec![(1, "person".to_string()), (7, "chair".to_string())]);
    i.sdf_scene = Some(scene);
    AspPacket::create_i_packet(42, i).unwrap()
}

fn d_packet() -> AspPacket {
    let mut d = DPacketPayload::new(41);
    d.timestamp_ms = 1_234_600;
    d.add_motion_vector(MotionVector::new(3, 4, -5, 6, 70));
    d.add_motion_vector(MotionVector::new(8, 9, 10, -11, 12));
    d.sdf_delta = Some(SdfSceneDelta::animation_update(3, vec![1, 2, 3, 4, 5, 6]));
    d.person_mask = Some(PersonMask::new([10, 20, 30, 40], vec![0, 5, 1, 7, 0, 3]));
    AspPacket::create_d_packet(43, d).unwrap()
}

fn c_packet() -> AspPacket {
    let mut c = libasp::CPacketPayload::new(43);
    c.timestamp_ms = 1_234_650;
    c.add_correction(CorrectionData {
        roi: RoiRegion::new(Rect::new(16, 8, 32, 24), RoiType::General)
            .with_priority(2)
            .with_confidence(0.5),
        pixel_delta: (0..48u8).collect(),
        compression: CompressionFormat::Raw,
    });
    AspPacket::create_c_packet(44, c).unwrap()
}

fn s_packet() -> AspPacket {
    AspPacket::create_s_packet(45, SPacketPayload::request_keyframe()).unwrap()
}

fn fixtures() -> [(&'static str, AspPacket); 4] {
    [
        ("i_packet_sdf_scene.bin", i_packet()),
        ("d_packet_sdf_delta_person_mask.bin", d_packet()),
        ("c_packet.bin", c_packet()),
        ("s_packet.bin", s_packet()),
    ]
}

fn dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/bincode1")
}

fn read(name: &str) -> Vec<u8> {
    std::fs::read(dir().join(name)).unwrap_or_else(|e| panic!("fixture {name}: {e}"))
}

fn sha256_hex(b: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    use std::fmt::Write;
    Sha256::digest(b).iter().fold(String::new(), |mut s, x| {
        let _ = write!(s, "{x:02x}");
        s
    })
}

/// Writes the fixtures (run once with bincode 1.3.3:
/// `ALICE_ASP_WRITE_BINCODE1_FIXTURES=1 cargo test --features bincode-compat --test bincode1_fixtures write -- --ignored`).
#[test]
#[ignore = "writes tests/fixtures/bincode1; run only to capture"]
fn write_bincode1_fixtures() {
    if std::env::var_os("ALICE_ASP_WRITE_BINCODE1_FIXTURES").is_none() {
        return;
    }
    std::fs::create_dir_all(dir()).unwrap();
    let mut sums = String::new();
    for (name, p) in fixtures() {
        let b = p.to_bytes_bincode().unwrap();
        std::fs::write(dir().join(name), &b).unwrap();
        sums.push_str(&format!("{}  {name}\n", sha256_hex(&b)));
    }
    std::fs::write(dir().join("SHA256SUMS"), sums).unwrap();
}

#[test]
fn every_fixture_matches_its_recorded_sha256() {
    let sums = std::fs::read_to_string(dir().join("SHA256SUMS")).unwrap();
    let mut checked = 0;
    for line in sums.lines() {
        let (hash, name) = line.split_once("  ").expect("`<sha256>  <name>`");
        assert_eq!(sha256_hex(&read(name)), hash, "{name}");
        checked += 1;
    }
    assert_eq!(checked, fixtures().len(), "SHA256SUMS lists every fixture");
}

#[test]
fn the_bytes_written_now_are_the_bytes_bincode_wrote() {
    for (name, p) in fixtures() {
        assert!(p.to_bytes_bincode().unwrap() == read(name), "{name}");
    }
}

#[test]
fn packets_bincode_wrote_decode_to_the_same_packet() {
    for (name, p) in fixtures() {
        let back = AspPacket::from_bytes_bincode(&read(name)).unwrap();
        // the payload, type and sequence come back; the header's payload length is the
        // encoded one (a packet built in memory carries an estimate there)
        assert_eq!(back.packet_type(), p.packet_type(), "{name}");
        assert_eq!(back.sequence(), p.sequence(), "{name}");
        assert_eq!(
            format!("{:?}", back.payload),
            format!("{:?}", p.payload),
            "{name}"
        );
        assert!(
            back.to_bytes_bincode().unwrap() == read(name),
            "{name} re-encodes to itself"
        );
    }
}

#[test]
fn the_hybrid_fields_are_in_the_fixtures() {
    let i = AspPacket::from_bytes_bincode(&read("i_packet_sdf_scene.bin")).unwrap();
    let libasp::AspPayload::IPacket(i) = &i.payload else {
        panic!("I")
    };
    assert!(i.sdf_scene.is_some());
    let d = AspPacket::from_bytes_bincode(&read("d_packet_sdf_delta_person_mask.bin")).unwrap();
    let libasp::AspPayload::DPacket(d) = &d.payload else {
        panic!("D")
    };
    assert!(d.sdf_delta.is_some() && d.person_mask.is_some());
}

#[test]
fn writing_the_same_packet_again_gives_the_same_bytes() {
    for (name, p) in fixtures() {
        let first = p.to_bytes_bincode().unwrap();
        for round in 0..4 {
            assert!(
                p.to_bytes_bincode().unwrap() == first,
                "{name}: round {round}"
            );
        }
    }
}

/// The C-Packet fixture with its payload changed by `f`, the checksum recomputed.
fn c_packet_with(f: impl FnOnce(&mut Vec<u8>)) -> Vec<u8> {
    let b = read("c_packet.bin");
    let body_end = b.len() - 4;
    let mut body = b[..body_end].to_vec();
    f(&mut body);
    let crc = libasp::crc32(&body).to_be_bytes();
    body.extend_from_slice(&crc);
    body
}

#[test]
fn a_length_past_the_payload_is_refused_before_anything_is_sized() {
    // the C payload is `ref_sequence: u32` then the `corrections` count (u64)
    let at = libasp::AspPacketHeader::SIZE + 4;
    let forged = c_packet_with(|b| b[at..at + 8].copy_from_slice(&u64::MAX.to_le_bytes()));
    assert!(AspPacket::from_bytes_bincode(&forged).is_err());
    let forged = c_packet_with(|b| b[at..at + 8].copy_from_slice(&(1u64 << 40).to_le_bytes()));
    assert!(AspPacket::from_bytes_bincode(&forged).is_err());
}

#[test]
fn a_payload_with_bytes_after_the_value_is_refused() {
    // the payload is all the bytes between the header and the checksum; bincode 1 ignored
    // extra bytes there, this does not
    assert!(AspPacket::from_bytes_bincode(&read("c_packet.bin")).is_ok());
    let longer = c_packet_with(|b| b.push(0));
    assert!(AspPacket::from_bytes_bincode(&longer).is_err());
}
