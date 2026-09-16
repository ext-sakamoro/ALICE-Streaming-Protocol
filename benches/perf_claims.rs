//! Source of the numbers in README "Performance Highlights"
//! (`<!-- perf-measured: ... benches/perf_claims.rs -->`). Each claim is a
//! ratio between two implementations measured here, not a hand estimate.
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput};
use libasp::header::crc32;
use libasp::{AspPacket, IPacketPayload};

/// Reference CRC-32/ISO-HDLC without a lookup table (bit-serial)
fn crc32_bitwise(data: &[u8]) -> u32 {
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc ^= u32::from(b);
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn bench_crc32(c: &mut Criterion) {
    let data: Vec<u8> = (0..65_536u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 24) as u8)
        .collect();
    assert_eq!(
        crc32(&data),
        crc32_bitwise(&data),
        "table and bit-serial CRC must agree"
    );
    let mut g = c.benchmark_group("crc32_64KiB");
    g.throughput(Throughput::Bytes(data.len() as u64));
    g.bench_function("table_compile_time", |b| b.iter(|| crc32(black_box(&data))));
    g.bench_function("bitwise_reference", |b| {
        b.iter(|| crc32_bitwise(black_box(&data)))
    });
    g.finish();
}

fn bench_packet_serialize(c: &mut Criterion) {
    let packet = AspPacket::create_i_packet(1, IPacketPayload::new(1920, 1080, 30.0)).unwrap();
    let mut g = c.benchmark_group("i_packet_serialize");
    g.bench_function("to_bytes_alloc", |b| {
        b.iter(|| black_box(&packet).to_bytes().unwrap())
    });
    let mut buffer = Vec::with_capacity(65_536);
    g.bench_function("write_to_buffer_reuse", |b| {
        b.iter(|| {
            black_box(&packet).write_to_buffer(&mut buffer).unwrap();
            black_box(buffer.len())
        })
    });
    g.finish();
}

criterion_group!(benches, bench_crc32, bench_packet_serialize);
criterion_main!(benches);
