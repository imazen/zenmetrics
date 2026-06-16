//! Create one Salad container group for the zen job-system fleet (goal H), via the crate's reqwest
//! client. Plain curl/urllib POSTs to Salad's API get a Cloudflare managed-challenge 403; reqwest's
//! TLS fingerprint passes. CPU-only, public image (no registry auth), no Salad managed queue — the
//! baked entrypoint claims off our own R2 lease-queue. Spec comes from env:
//!   SALAD_API_KEY  SALAD_ORG(=imazen)  SALAD_PROJECT(=zenmetrics)
//!   SALAD_GROUP_NAME  SALAD_IMAGE  SALAD_REPLICAS(=1)  SALAD_ENV_JSON(={"K":"V",…})
//! Throwaway: run via `cargo run -p zenfleet-salad --example fleet_create`.
use std::collections::HashMap;
use zenfleet_salad::launch::{
    ContainerConfig, CreateContainerGroupRequest, ResourceRequirements, SaladApi,
};

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    let org = std::env::var("SALAD_ORG").unwrap_or_else(|_| "imazen".into());
    let project = std::env::var("SALAD_PROJECT").unwrap_or_else(|_| "zenmetrics".into());
    let name = std::env::var("SALAD_GROUP_NAME")?;
    let image = std::env::var("SALAD_IMAGE")?;
    let replicas: u32 = std::env::var("SALAD_REPLICAS")
        .unwrap_or_else(|_| "1".into())
        .parse()?;
    let environment_variables: HashMap<String, String> =
        serde_json::from_str(&std::env::var("SALAD_ENV_JSON")?)?;
    let key = std::env::var("SALAD_API_KEY").ok();

    let api = SaladApi::new(&org, &project, key)?;
    let req = CreateContainerGroupRequest {
        name: name.clone(),
        display_name: None,
        container: ContainerConfig {
            image,
            resources: ResourceRequirements {
                cpu: 2,
                memory: 4096,
                gpu_classes: vec![],
            },
            command: None,
            environment_variables,
            registry_authentication: None,
        },
        replicas,
        restart_policy: "never".into(),
        autostart_policy: true,
        queue_connection: None,
    };
    match api.create_container_group(&req).await {
        Ok(cg) => {
            println!("CREATED salad group {} replicas={:?}", cg.name, cg.replicas);
            Ok(())
        }
        Err(e) => {
            eprintln!("CREATE FAILED: {e:?}");
            std::process::exit(1);
        }
    }
}
