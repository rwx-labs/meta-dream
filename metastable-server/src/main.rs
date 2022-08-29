use std::env;

use ::tracing::{debug, info};
use clap::Parser;
use miette::Result;

mod cli;
mod database;
mod error;
mod http;
mod tracing;

pub use error::Error;

#[tokio::main]
async fn main() -> Result<()> {
    // Override RUST_LOG with a default setting if it's not set by the user
    if env::var("RUST_LOG").is_err() {
        env::set_var("RUST_LOG", "metastable_server=trace,tower_http=debug");
    }

    let opts = cli::Opts::parse();
    println!("{:?}", opts);
    tracing::init(&opts.tracing_opts)?;

    let version = env!("CARGO_PKG_VERSION");
    info!(version, "Starting metastable-server");

    debug!("connecting to database");
    let pool = database::open(&opts.database_url).await?;
    debug!(kind = ?pool.any_kind(), "connected to database");

    debug!("running database migrations");
    database::migrate(pool.clone()).await?;
    debug!("database migration complete");

    info!("starting http server");
    let handle = axum_server::Handle::new();

    http::start_server(&opts.http_opts, handle.clone(), pool.clone()).await?;

    println!("Hello, world! {:?}", opts);

    Ok(())
}
