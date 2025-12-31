//! Benchmarks for motion estimation

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use libasp::codec::motion::{estimate_motion, estimate_motion_parallel, SearchAlgorithm};

fn create_test_frames(width: usize, height: usize) -> (Vec<u8>, Vec<u8>) {
    let current: Vec<u8> = (0..width * height).map(|i| (i % 256) as u8).collect();
    let mut previous = current.clone();

    // Add some motion to previous frame
    for y in 50..100 {
        for x in 50..100 {
            if y < height && x < width {
                previous[y * width + x] = 200;
            }
        }
    }

    (current, previous)
}

fn bench_motion_estimation(c: &mut Criterion) {
    let mut group = c.benchmark_group("motion_estimation");

    for size in [256, 512, 1024].iter() {
        let (current, previous) = create_test_frames(*size, *size);

        group.bench_with_input(
            BenchmarkId::new("single_thread", size),
            size,
            |b, &size| {
                b.iter(|| {
                    estimate_motion(
                        black_box(&current),
                        black_box(&previous),
                        size,
                        size,
                        16,
                        8,
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parallel_diamond", size),
            size,
            |b, &size| {
                b.iter(|| {
                    estimate_motion_parallel(
                        black_box(&current),
                        black_box(&previous),
                        size,
                        size,
                        16,
                        8,
                        SearchAlgorithm::DiamondSearch,
                        256,
                    )
                })
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parallel_hexagon", size),
            size,
            |b, &size| {
                b.iter(|| {
                    estimate_motion_parallel(
                        black_box(&current),
                        black_box(&previous),
                        size,
                        size,
                        16,
                        8,
                        SearchAlgorithm::HexagonSearch,
                        256,
                    )
                })
            },
        );
    }

    group.finish();
}

fn bench_hd_frame(c: &mut Criterion) {
    let (current, previous) = create_test_frames(1920, 1080);

    c.bench_function("hd_1080p_diamond", |b| {
        b.iter(|| {
            estimate_motion_parallel(
                black_box(&current),
                black_box(&previous),
                1920,
                1080,
                16,
                16,
                SearchAlgorithm::DiamondSearch,
                256,
            )
        })
    });
}

criterion_group!(benches, bench_motion_estimation, bench_hd_frame);
criterion_main!(benches);
