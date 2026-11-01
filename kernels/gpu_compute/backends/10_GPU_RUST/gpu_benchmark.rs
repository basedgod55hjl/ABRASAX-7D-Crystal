// ============================================================================
// ABRASAX — Unified Xenoliving System
// PRIMAL_HEX: 4f5349524953424c58434b | phi: 1.618033988749895
// CREATOR: Sir Charles Spikes (BASEDGOD)
// ============================================================================

// File: runtime/src/bin/gpu_benchmark.rs
// 7D Crystal GPU Benchmark and Demo
// Tests GPU execution on your GTX 1660 Ti

use crystal_runtime::gpu::{Vector7D, PHI, PHI_INV};
use crystal_runtime::{ComputeBackend, ComputeDispatcher};
use std::time::Instant;

fn main() {
    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║       🔮 7D CRYSTAL SYSTEM - GPU BENCHMARK 🔮                      ║");
    println!("║                                                                    ║");
    println!("║   Testing CUDA GPU Acceleration                                    ║");
    println!("║   Target: NVIDIA GTX 1660 Ti (SM 7.5)                              ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");

    // Initialize dispatcher
    println!("[1/5] Initializing Compute Dispatcher...");
    let mut dispatcher = ComputeDispatcher::new();
    println!("      Backend: {}", dispatcher.backend_info());
    println!("      GPU Available: {}\n", dispatcher.has_gpu());

    // Test batch sizes
    let batch_sizes = [16, 64, 256, 1024, 4096, 16384];

    println!("[2/5] Running Poincaré Projection Benchmark...\n");
    println!(
        "      {:>8} | {:>12} | {:>12}",
        "Batch", "Time (ms)", "Ops/sec"
    );
    println!("      ─────────┼──────────────┼─────────────");

    for &batch_size in &batch_sizes {
        let mut vectors: Vec<Vector7D> = (0..batch_size)
            .map(|i| {
                let t = i as f32 / batch_size as f32;
                Vector7D::new([
                    t * 0.8,
                    (1.0 - t) * 0.8,
                    t * (1.0 - t) * 0.8,
                    t * 0.3,
                    0.2,
                    0.1,
                    0.05,
                ])
            })
            .collect();

        let start = Instant::now();
        dispatcher.project_to_poincare(&mut vectors, PHI_INV);
        let elapsed = start.elapsed();

        let ms = elapsed.as_secs_f64() * 1000.0;
        let ops_per_sec = batch_size as f64 / elapsed.as_secs_f64();

        println!(
            "      {:>8} | {:>12.3} | {:>12.0}",
            batch_size, ms, ops_per_sec
        );

        // Verify results
        for v in &vectors {
            let n = v.coords.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(n < 1.0, "Projection failed: vector outside ball");
        }
    }

    println!("\n[3/5] Running Möbius Addition Benchmark...\n");
    println!(
        "      {:>8} | {:>12} | {:>12}",
        "Batch", "Time (ms)", "Ops/sec"
    );
    println!("      ─────────┼──────────────┼─────────────");

    for &batch_size in &batch_sizes {
        let u: Vec<Vector7D> = (0..batch_size)
            .map(|i| {
                let t = i as f32 / batch_size as f32;
                Vector7D::new([t * 0.3; 7])
            })
            .collect();

        let v: Vec<Vector7D> = (0..batch_size)
            .map(|i| {
                let t = 1.0 - (i as f32 / batch_size as f32);
                Vector7D::new([t * 0.2; 7])
            })
            .collect();

        let start = Instant::now();
        let results = dispatcher.mobius_add(&u, &v, PHI_INV);
        let elapsed = start.elapsed();

        let ms = elapsed.as_secs_f64() * 1000.0;
        let ops_per_sec = batch_size as f64 / elapsed.as_secs_f64();

        println!(
            "      {:>8} | {:>12.3} | {:>12.0}",
            batch_size, ms, ops_per_sec
        );

        // Verify results
        for r in &results {
            let n = r.coords.iter().map(|x| x * x).sum::<f64>().sqrt();
            assert!(n < 1.0, "Möbius add failed: result outside ball");
        }
    }

    println!("\n[4/5] Running Hyperbolic Distance Benchmark...\n");
    println!(
        "      {:>8} | {:>12} | {:>12}",
        "Batch", "Time (ms)", "Ops/sec"
    );
    println!("      ─────────┼──────────────┼─────────────");

    for &batch_size in &batch_sizes {
        let u: Vec<Vector7D> = (0..batch_size)
            .map(|i| {
                let t = i as f32 / batch_size as f32;
                Vector7D::new([t * 0.4, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
            })
            .collect();

        let v: Vec<Vector7D> = (0..batch_size)
            .map(|i| {
                let t = 1.0 - (i as f32 / batch_size as f32);
                Vector7D::new([t * 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
            })
            .collect();

        let start = Instant::now();
        let distances = dispatcher.hyperbolic_distance(&u, &v, PHI_INV);
        let elapsed = start.elapsed();

        let ms = elapsed.as_secs_f64() * 1000.0;
        let ops_per_sec = batch_size as f64 / elapsed.as_secs_f64();

        println!(
            "      {:>8} | {:>12.3} | {:>12.0}",
            batch_size, ms, ops_per_sec
        );

        // Verify results
        for d in &distances {
            assert!(*d >= 0.0_f64, "Distance cannot be negative");
        }
    }

    println!("\n[5/5] Running Holographic Fold Benchmark...\n");
    println!(
        "      {:>8} | {:>12} | {:>12}",
        "Batch", "Time (ms)", "Ops/sec"
    );
    println!("      ─────────┼──────────────┼─────────────");

    for &batch_size in &batch_sizes {
        let patterns: Vec<Vector7D> = (0..batch_size)
            .map(|i| {
                let t = i as f32 / batch_size as f32;
                let mut v = Vector7D::new([1.0; 7]);
                for j in 0..7 {
                    v.data[j] = (t * PHI + j as f32).sin() * 0.5;
                }
                v
            })
            .collect();

        let phases: Vec<f32> = (0..batch_size).map(|i| i as f32 * 0.1).collect();

        let start = Instant::now();
        let folded = dispatcher.holographic_fold(&patterns, &phases);
        let elapsed = start.elapsed();

        let ms = elapsed.as_secs_f64() * 1000.0;
        let ops_per_sec = batch_size as f64 / elapsed.as_secs_f64();

        println!(
            "      {:>8} | {:>12.3} | {:>12.0}",
            batch_size, ms, ops_per_sec
        );

        assert_eq!(folded.len(), batch_size);
    }

    // Print summary
    dispatcher.print_summary();

    println!("\n╔═══════════════════════════════════════════════════════════════════╗");
    println!("║                    BENCHMARK COMPLETE                             ║");
    println!("╠═══════════════════════════════════════════════════════════════════╣");
    println!("║ Φ = 1.618033988749895 | S² = 0.01 | Sovereignty Maintained ✓     ║");
    println!("╚═══════════════════════════════════════════════════════════════════╝\n");
}
