//! Print the detected capability profile, exercising the on-disk cache.
//!
//! Run twice in a row: the second run will load from cache (you'll see
//! the same `detected_at` but a newer `last_validated`).

use std::time::Duration;

use zenmetrics_orchestrator::{Orchestrator, OrchestratorConfig};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Use the default cache_dir (~/.cache/zenmetrics/) so the example
    // round-trips through real user storage. Mutate after `default()`
    // because OrchestratorConfig is `#[non_exhaustive]` from outside
    // the crate.
    let mut cfg = OrchestratorConfig::default();
    cfg.cache_validity = Duration::from_secs(7 * 24 * 60 * 60);

    println!("cache dir: {}", cfg.cache_dir.display());

    let orch = Orchestrator::new(cfg)?;
    let cap = orch.capability();

    println!("cache file: {}", orch.cache_path().display());
    println!();
    println!("machine_hash:    {}", cap.machine_hash);
    println!("short_hash:      {}", cap.short_hash());
    println!(
        "detected_at:     {} (UNIX seconds)",
        cap.detected_at
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs()
    );
    println!(
        "last_validated:  {} (UNIX seconds)",
        cap.last_validated
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs()
    );
    println!();
    println!("[gpu]");
    println!("  present:            {}", cap.gpu.present);
    println!("  model:              {}", cap.gpu.model);
    println!("  total_vram_mib:     {}", cap.gpu.total_vram_mib);
    println!("  driver_version:     {}", cap.gpu.driver_version);
    println!("  cuda_runtime:       {:?}", cap.gpu.cuda_runtime);
    println!("  compute_capability: {:?}", cap.gpu.compute_capability);
    println!();
    println!("[cpu]");
    println!("  brand:          {}", cap.cpu.brand);
    println!("  logical_cores:  {}", cap.cpu.logical_cores);
    println!("  features:       {:?}", cap.cpu.features);
    println!("  ram_mib:        {}", cap.cpu.ram_mib);

    Ok(())
}
