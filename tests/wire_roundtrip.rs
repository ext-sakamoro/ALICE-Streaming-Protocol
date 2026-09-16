//! Wire-format law of the default (FlatBuffers) packet encoding:
//! `from_bytes(to_bytes(p)) == p` for every packet type, every enum value and
//! every optional field the `asp.fbs` schema carries
//!
//! Until 1.1.0 the encoder dropped everything but width / height / fps /
//! timestamp (I), ref_sequence + motion vectors (D) and Ping / Pong (S), and
//! wrote a C-Packet as a Ping; these tests are the contract that no field is
//! lost again The comparison is on `Debug` output, which is exact for every
//! field type used here (integers, `f32`, enums, byte vectors, strings)

use libasp::{
    AnimationParams, AspError, AspPacket, AspPayload, CPacketPayload, Color, ColorPalette,
    CompressionFormat, CorrectionData, DPacketPayload, EasingType, IPacketPayload, MotionVector,
    PatternType, QualityLevel, Rect, RegionDelta, RegionDescriptor, RoiRegion, RoiType,
    SPacketPayload, SdfSceneDescriptor, SyncCommand, SyncData,
};

fn dbg<T: std::fmt::Debug>(t: &T) -> String {
    format!("{t:?}")
}

fn roundtrip(packet: &AspPacket) -> AspPacket {
    let bytes = packet.to_bytes().expect("serialise");
    let back = AspPacket::from_bytes(&bytes).expect("deserialise");
    assert_eq!(back.packet_type(), packet.packet_type());
    assert_eq!(back.sequence(), packet.sequence());
    // the header carries the real FlatBuffers payload length
    assert_eq!(
        back.payload_size() as usize,
        bytes.len() - libasp::AspPacketHeader::SIZE - 4
    );
    back
}

const ALL_QUALITY: [QualityLevel; 4] = [
    QualityLevel::Low,
    QualityLevel::Medium,
    QualityLevel::High,
    QualityLevel::Ultra,
];
const ALL_SYNC: [SyncCommand; 8] = [
    SyncCommand::RequestKeyframe,
    SyncCommand::Ack,
    SyncCommand::Nack,
    SyncCommand::EndOfStream,
    SyncCommand::BitrateAdjust,
    SyncCommand::QualityChange,
    SyncCommand::Ping,
    SyncCommand::Pong,
];
const ALL_PATTERN: [PatternType; 8] = [
    PatternType::Solid,
    PatternType::GradientLinear,
    PatternType::GradientRadial,
    PatternType::Noise,
    PatternType::Texture,
    PatternType::Dct,
    PatternType::Periodic,
    PatternType::Complex,
];
const ALL_ROI: [RoiType; 6] = [
    RoiType::General,
    RoiType::Face,
    RoiType::Text,
    RoiType::Edge,
    RoiType::Motion,
    RoiType::Custom,
];
const ALL_COMPRESSION: [CompressionFormat; 5] = [
    CompressionFormat::Raw,
    CompressionFormat::Rle,
    CompressionFormat::Lz4,
    CompressionFormat::Zstd,
    CompressionFormat::DeltaRle,
];
const ALL_EASING: [EasingType; 6] = [
    EasingType::Linear,
    EasingType::EaseIn,
    EasingType::EaseOut,
    EasingType::EaseInOut,
    EasingType::Bounce,
    EasingType::Elastic,
];

fn palette(n: u8, weights: bool) -> ColorPalette {
    let mut p = ColorPalette::new((0..n).map(|i| Color::new(i, 255 - i, i * 3)).collect());
    if weights {
        p.weights = Some((0..n).map(|i| f32::from(i) / f32::from(n)).collect());
    }
    p
}

#[test]
fn s_packet_every_command_and_data_variant_roundtrips() {
    let datas = [
        SyncData::None,
        SyncData::Sequence(0xDEAD_BEEF),
        SyncData::Bitrate(4_500),
        SyncData::Quality(QualityLevel::Ultra),
        SyncData::Latency(37),
        SyncData::Custom(vec![0, 1, 2, 250, 255]),
        SyncData::Custom(Vec::new()),
    ];
    for (i, command) in ALL_SYNC.iter().enumerate() {
        for data in &datas {
            let payload = SPacketPayload {
                command: *command,
                data: data.clone(),
                timestamp_ms: 1_000 + i as u64,
            };
            let packet = AspPacket::create_s_packet(i as u32, payload.clone()).unwrap();
            let back = roundtrip(&packet);
            let AspPayload::SPacket(b) = &back.payload else {
                panic!("payload type changed")
            };
            assert_eq!(dbg(b), dbg(&payload), "{command:?} / {data:?}");
        }
    }
}

#[test]
fn c_packet_corrections_roundtrip_exactly() {
    let mut payload = CPacketPayload::new(41);
    payload.timestamp_ms = 123_456_789;
    for (i, (roi_type, compression)) in ALL_ROI
        .iter()
        .zip(ALL_COMPRESSION.iter().cycle())
        .enumerate()
    {
        let roi = RoiRegion::new(Rect::new(i as u32 * 16, 8, 32, 24), *roi_type)
            .with_priority(i as u8 + 1)
            .with_confidence(0.125 * i as f32);
        payload.add_correction(CorrectionData {
            roi,
            pixel_delta: (0..=(i as u8 * 40)).collect(),
            compression: *compression,
        });
    }
    let packet = AspPacket::create_c_packet(7, payload.clone()).unwrap();
    let back = roundtrip(&packet);
    let AspPayload::CPacket(b) = &back.payload else {
        panic!("payload type changed")
    };
    assert_eq!(dbg(b), dbg(&payload));
    assert_eq!(b.corrections.len(), ALL_ROI.len());
    assert_eq!(b.total_correction_bytes(), payload.total_correction_bytes());
}

#[test]
fn i_packet_quality_palette_regions_and_animation_roundtrip_exactly() {
    for quality in ALL_QUALITY {
        for easing in ALL_EASING {
            let mut payload = IPacketPayload::new(1920, 1080, 59.94)
                .with_quality(quality)
                .with_palette(palette(5, true));
            payload.timestamp_ms = 99;
            payload.animation = Some(AnimationParams {
                zoom_factor: 1.25,
                pan_x: -0.5,
                pan_y: 0.75,
                rotation: 33.0,
                duration: 12,
                easing,
            });
            for (i, pattern) in ALL_PATTERN.iter().enumerate() {
                payload.add_region(RegionDescriptor {
                    bounds: Rect::new(i as u32, 2 * i as u32, 64, 48),
                    pattern_type: *pattern,
                    palette: palette(i as u8 % 3 + 1, i % 2 == 0),
                    dct_coefficients: (i % 2 == 1)
                        .then(|| vec![(0, 0, 100.5), (1, 7, -3.25), (65_535, 65_535, 0.0)]),
                    // only a Texture region carries an id; the schema default 0
                    // reads back as None for every other pattern
                    texture_id: (*pattern == PatternType::Texture).then_some(0xABCD),
                    params: (i % 3 == 0)
                        .then(|| vec![("gain".to_string(), 0.5), (String::new(), -1.0)]),
                });
            }
            let packet = AspPacket::create_i_packet(3, payload.clone()).unwrap();
            let back = roundtrip(&packet);
            let AspPayload::IPacket(b) = &back.payload else {
                panic!("payload type changed")
            };
            assert_eq!(dbg(b), dbg(&payload), "{quality:?} / {easing:?}");
            assert!(back.is_keyframe());
        }
    }
}

#[test]
fn d_packet_motion_vectors_global_motion_and_region_deltas_roundtrip_exactly() {
    let mut payload = DPacketPayload::new(12);
    payload.timestamp_ms = u64::MAX;
    for i in 0..37u16 {
        payload.add_motion_vector(MotionVector::new(
            i,
            100 + i,
            -(i as i16),
            i as i16 * 2,
            u32::from(i) * 11,
        ));
    }
    payload.global_motion = Some(AnimationParams {
        zoom_factor: 0.5,
        pan_x: 0.0,
        pan_y: -1.0,
        rotation: -90.0,
        duration: 1,
        easing: EasingType::Bounce,
    });
    payload.add_region_delta(RegionDelta {
        region_index: 2,
        palette_delta: Some(palette(2, false)),
        dct_delta: Some(vec![(3, 4, 5.5)]),
        param_delta: None,
    });
    payload.add_region_delta(RegionDelta {
        region_index: 9,
        palette_delta: None,
        dct_delta: None,
        param_delta: Some(vec![("speed".to_string(), 2.0)]),
    });
    let packet = AspPacket::create_d_packet(13, payload.clone()).unwrap();
    let back = roundtrip(&packet);
    let AspPayload::DPacket(b) = &back.payload else {
        panic!("payload type changed")
    };
    assert_eq!(dbg(b), dbg(&payload));
    assert!(!back.is_keyframe());
}

#[test]
fn hybrid_only_fields_are_rejected_not_dropped() {
    let mut i = IPacketPayload::new(64, 64, 30.0);
    i.sdf_scene = Some(SdfSceneDescriptor::new(vec![b'A', b'S', b'D', b'F']));
    let err = AspPacket::create_i_packet(1, i)
        .unwrap()
        .to_bytes()
        .unwrap_err();
    assert!(matches!(err, AspError::SerializationError(_)), "{err:?}");

    let mut d = DPacketPayload::new(1);
    d.person_mask = Some(libasp::PersonMask::new([0, 0, 8, 8], vec![0]));
    let err = AspPacket::create_d_packet(2, d)
        .unwrap()
        .to_bytes()
        .unwrap_err();
    assert!(matches!(err, AspError::SerializationError(_)), "{err:?}");

    // DCT indices beyond the schema's u16 are an error, not a truncation
    let mut i = IPacketPayload::new(64, 64, 30.0);
    let mut r = RegionDescriptor {
        bounds: Rect::new(0, 0, 8, 8),
        pattern_type: PatternType::Dct,
        palette: palette(1, false),
        dct_coefficients: Some(vec![(70_000, 0, 1.0)]),
        texture_id: None,
        params: None,
    };
    i.add_region(r.clone());
    let err = AspPacket::create_i_packet(3, i)
        .unwrap()
        .to_bytes()
        .unwrap_err();
    assert!(matches!(err, AspError::SerializationError(_)), "{err:?}");
    r.dct_coefficients = Some(vec![(65_535, 65_535, 1.0)]);
    let mut i = IPacketPayload::new(64, 64, 30.0);
    i.add_region(r);
    AspPacket::create_i_packet(3, i)
        .unwrap()
        .to_bytes()
        .unwrap();
}

#[test]
fn corrupt_and_truncated_buffers_are_errors_never_partial_packets() {
    let mut payload = CPacketPayload::new(1);
    payload.add_correction(CorrectionData {
        roi: RoiRegion::new(Rect::new(0, 0, 16, 16), RoiType::Face),
        pixel_delta: vec![1, 2, 3],
        compression: CompressionFormat::Rle,
    });
    let bytes = AspPacket::create_c_packet(5, payload)
        .unwrap()
        .to_bytes()
        .unwrap();

    // shorter than header + CRC
    let min = libasp::AspPacketHeader::SIZE + 4;
    for n in [0usize, 1, min - 1] {
        let err = AspPacket::from_bytes(&bytes[..n]).unwrap_err();
        assert!(
            matches!(err, AspError::IncompletePacket { expected, got } if expected == min && got == n),
            "len {n}: {err:?}"
        );
    }
    // exactly header + CRC: the CRC no longer matches the truncated body
    assert!(AspPacket::from_bytes(&bytes[..min]).is_err());
    // any dropped tail byte breaks the CRC
    for cut in 1..=4usize {
        let err = AspPacket::from_bytes(&bytes[..bytes.len() - cut]).unwrap_err();
        assert!(
            matches!(err, AspError::ChecksumMismatch { .. }),
            "cut {cut}: {err:?}"
        );
    }
    // every single-bit flip in the body is detected by the CRC
    for byte in 0..bytes.len() - 4 {
        let mut b = bytes.clone();
        b[byte] ^= 0x01;
        assert!(
            AspPacket::from_bytes(&b).is_err(),
            "flip at {byte} accepted"
        );
    }
    // a header whose type disagrees with the payload is rejected even with a valid CRC
    let mut b = bytes.clone();
    let header = libasp::AspPacketHeader::new(libasp::PacketType::IPacket, 5, 0);
    b[..libasp::AspPacketHeader::SIZE].copy_from_slice(&header.to_bytes());
    let body_len = b.len() - 4;
    let crc = libasp::crc32(&b[..body_len]).to_be_bytes();
    b[body_len..].copy_from_slice(&crc);
    let err = AspPacket::from_bytes(&b).unwrap_err();
    assert!(matches!(err, AspError::DeserializationError(_)), "{err:?}");
}

#[cfg(feature = "bincode-compat")]
#[test]
fn bincode_path_carries_the_hybrid_fields_the_flatbuffers_path_rejects() {
    let mut i = IPacketPayload::new(64, 64, 30.0);
    i.sdf_scene = Some(SdfSceneDescriptor::new(vec![1, 2, 3, 4, 5]));
    let packet = AspPacket::create_i_packet(1, i.clone()).unwrap();
    let mut buf = Vec::new();
    packet.write_to_buffer_bincode(&mut buf).unwrap();
    let back = AspPacket::from_bytes_bincode(&buf).unwrap();
    let AspPayload::IPacket(b) = &back.payload else {
        panic!("payload type changed")
    };
    assert_eq!(dbg(b), dbg(&i));
}

// ----------------------------------------------------------------------------
// The zero-allocation encoder produces the same packets as `to_bytes`, the
// bincode path validates its input like the FlatBuffers path, and the payload
// helpers follow their definitions
// ----------------------------------------------------------------------------

#[test]
fn packet_encoder_output_parses_to_the_same_packet_as_to_bytes() {
    use libasp::flatbuffers_api::{FbQualityLevel, FbSyncCommand};
    use libasp::PacketEncoder;
    let mut enc = PacketEncoder::new();

    let mvs: Vec<MotionVector> = (0..5u16)
        .map(|i| MotionVector::new(i, i, -1, 2, 30 + u32::from(i)))
        .collect();
    let mut d = DPacketPayload::new(9);
    d.timestamp_ms = 777;
    for mv in &mvs {
        d.add_motion_vector(*mv);
    }
    let reference = AspPacket::create_d_packet(4, d)
        .unwrap()
        .to_bytes()
        .unwrap();
    let fast = enc.encode_d_packet(4, 9, &mvs, 777).to_vec();
    assert_eq!(fast.len(), reference.len());
    let a = AspPacket::from_bytes(&fast).unwrap();
    let b = AspPacket::from_bytes(&reference).unwrap();
    assert_eq!(dbg(&a.payload), dbg(&b.payload));
    assert_eq!(a.sequence(), 4);
    assert_eq!(a.payload_size(), b.payload_size());
    assert!(enc.buffer_capacity() >= fast.len());

    let reference = AspPacket::create_i_packet(
        5,
        IPacketPayload::new(320, 240, 24.0).with_quality(QualityLevel::High),
    )
    .unwrap()
    .to_bytes()
    .unwrap();
    let fast = enc
        .encode_i_packet(5, 320, 240, 24.0, FbQualityLevel::High, 0)
        .to_vec();
    let a = AspPacket::from_bytes(&fast).unwrap();
    let b = AspPacket::from_bytes(&reference).unwrap();
    // the fast encoder carries no palette; the reader restores the default
    // ([black]) so the payloads agree while `to_bytes` spends 32 bytes on it
    assert_eq!(fast.len() + 32, reference.len());
    assert_eq!(dbg(&a.payload), dbg(&b.payload));
    assert!(a.is_keyframe());

    let reference = AspPacket::create_s_packet(
        6,
        SPacketPayload {
            command: SyncCommand::Pong,
            data: SyncData::None,
            timestamp_ms: 42,
        },
    )
    .unwrap()
    .to_bytes()
    .unwrap();
    let fast = enc.encode_s_packet(6, FbSyncCommand::Pong, 42).to_vec();
    let a = AspPacket::from_bytes(&fast).unwrap();
    let b = AspPacket::from_bytes(&reference).unwrap();
    assert_eq!(fast.len(), reference.len());
    assert_eq!(dbg(&a.payload), dbg(&b.payload));
    assert_eq!(a.as_s_packet().map(|s| s.command), Some(SyncCommand::Pong));
    assert!(a.as_c_packet().is_none());
}

#[test]
fn payload_helpers_follow_their_definitions() {
    let mut c = CPacketPayload::new(1);
    assert_eq!((c.correction_count, c.total_correction_bytes()), (0, 0));
    for n in [3usize, 5, 0] {
        c.add_correction(CorrectionData {
            roi: RoiRegion::new(Rect::new(0, 0, 1, 1), RoiType::General),
            pixel_delta: vec![7; n],
            compression: CompressionFormat::Raw,
        });
    }
    assert_eq!(c.correction_count, 3);
    assert_eq!(c.corrections.len(), 3);
    assert_eq!(c.total_correction_bytes(), 8);
    let packet = AspPacket::create_c_packet(2, c).unwrap();
    assert_eq!(packet.as_c_packet().map(|p| p.correction_count), Some(3));
    assert!(packet.as_s_packet().is_none());

    let mut d = DPacketPayload::new(1);
    d.add_region_delta(RegionDelta {
        region_index: 4,
        palette_delta: None,
        dct_delta: None,
        param_delta: None,
    });
    d.add_region_delta(RegionDelta {
        region_index: 5,
        palette_delta: None,
        dct_delta: None,
        param_delta: None,
    });
    assert_eq!(
        d.region_deltas
            .iter()
            .map(|r| r.region_index)
            .collect::<Vec<_>>(),
        vec![4, 5]
    );

    assert_eq!(ColorPalette::new(vec![]).dominant_color(), None);
    assert_eq!(
        palette(3, false).dominant_color(),
        Some(Color::new(0, 255, 0))
    );
    assert_eq!(
        ColorPalette::default().dominant_color(),
        Some(Color::black())
    );

    // estimated_size (pre-serialisation header estimate) grows by the
    // documented per-item allowance: I 128 / region, D 12 / vector + 64 /
    // delta, C 32 + bytes / correction, S fixed
    let base = libasp::AspPacketHeader::SIZE + 4;
    let mut i = IPacketPayload::new(1, 1, 1.0);
    let e0 = AspPacket::create_i_packet(0, i.clone())
        .unwrap()
        .estimated_size();
    assert_eq!(e0, base + 64);
    i.add_region(RegionDescriptor {
        bounds: Rect::new(0, 0, 1, 1),
        pattern_type: PatternType::Solid,
        palette: palette(1, false),
        dct_coefficients: None,
        texture_id: None,
        params: None,
    });
    assert_eq!(
        AspPacket::create_i_packet(0, i).unwrap().estimated_size(),
        e0 + 128
    );
    let mut d = DPacketPayload::new(1);
    let e0 = AspPacket::create_d_packet(0, d.clone())
        .unwrap()
        .estimated_size();
    assert_eq!(e0, base + 32);
    d.add_motion_vector(MotionVector::new(0, 0, 1, 1, 1));
    d.add_motion_vector(MotionVector::new(0, 1, 1, 1, 1));
    d.add_region_delta(RegionDelta {
        region_index: 0,
        palette_delta: None,
        dct_delta: None,
        param_delta: None,
    });
    assert_eq!(
        AspPacket::create_d_packet(0, d).unwrap().estimated_size(),
        e0 + 2 * 12 + 64
    );
    let mut c = CPacketPayload::new(1);
    let e0 = AspPacket::create_c_packet(0, c.clone())
        .unwrap()
        .estimated_size();
    assert_eq!(e0, base + 32);
    c.add_correction(CorrectionData {
        roi: RoiRegion::new(Rect::new(0, 0, 1, 1), RoiType::General),
        pixel_delta: vec![0; 10],
        compression: CompressionFormat::Raw,
    });
    assert_eq!(
        AspPacket::create_c_packet(0, c).unwrap().estimated_size(),
        e0 + 32 + 10
    );
    let s = AspPacket::create_s_packet(0, SPacketPayload::request_keyframe()).unwrap();
    assert_eq!(s.estimated_size(), base + 64);
}

#[cfg(feature = "bincode-compat")]
#[test]
fn bincode_path_validates_length_header_and_crc_like_the_flatbuffers_path() {
    let mut i = IPacketPayload::new(64, 64, 30.0);
    i.sdf_scene = Some(SdfSceneDescriptor::new(vec![1, 2, 3]));
    let packet = AspPacket::create_i_packet(1, i).unwrap();
    let mut buf = Vec::new();
    packet.write_to_buffer_bincode(&mut buf).unwrap();
    // header payload_length = bytes between header and CRC
    let header = libasp::AspPacketHeader::from_bytes(&buf).unwrap();
    assert_eq!(
        header.payload_length as usize,
        buf.len() - libasp::AspPacketHeader::SIZE - 4
    );
    assert_eq!(
        AspPacket::from_bytes_bincode(&buf).unwrap().payload_size(),
        header.payload_length
    );
    let min = libasp::AspPacketHeader::SIZE + 4;
    for n in [0usize, 1, min - 1] {
        let err = AspPacket::from_bytes_bincode(&buf[..n]).unwrap_err();
        assert!(
            matches!(err, AspError::IncompletePacket { expected, got } if expected == min && got == n),
            "len {n}: {err:?}"
        );
    }
    for cut in 1..=4usize {
        let err = AspPacket::from_bytes_bincode(&buf[..buf.len() - cut]).unwrap_err();
        assert!(
            matches!(err, AspError::ChecksumMismatch { .. }),
            "cut {cut}: {err:?}"
        );
    }
    for byte in 0..buf.len() - 4 {
        let mut b = buf.clone();
        b[byte] ^= 0x80;
        assert!(
            AspPacket::from_bytes_bincode(&b).is_err(),
            "flip at {byte} accepted"
        );
    }
}
