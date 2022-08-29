//! Application tracing

#[cfg(feature = "tracing")]
use opentelemetry::{global, sdk::trace::Tracer};
use tracing_subscriber::{prelude::*, EnvFilter, Registry};

use crate::cli;
use crate::Error;

#[cfg(feature = "tracing")]
use cli::TracingExporter;

#[cfg(feature = "tracing-jaeger")]
fn init_jaeger_tracer(opts: &cli::TracingOpts) -> Result<Tracer, Error> {
    global::set_text_map_propagator(opentelemetry_jaeger::Propagator::new());

    let tracer = opentelemetry_jaeger::new_pipeline()
        .with_service_name(opts.service_name.as_str())
        .install_batch(opentelemetry::runtime::Tokio)
        .map_err(Error::TraceInstallationError)?;

    Ok(tracer)
}

#[cfg(feature = "tracing")]
pub fn init(tracing_opts: &cli::TracingOpts) -> Result<(), Error> {
    if !tracing_opts.enabled {
        return Ok(());
    }

    let otel_layer = match tracing_opts.exporter {
        #[cfg(feature = "tracing-jaeger")]
        TracingExporter::Jaeger => {
            let tracer = init_jaeger_tracer(tracing_opts)?;

            Some(tracing_opentelemetry::layer().with_tracer(tracer))
        }
        #[cfg(feature = "tracing-datadog")]
        TracingExporter::Datadog => None,
    };

    let fmt_layer = tracing_subscriber::fmt::layer();
    let env_filter = EnvFilter::from_default_env();

    Registry::default()
        .with(otel_layer)
        .with(env_filter)
        .with(fmt_layer)
        .try_init()
        .map_err(Error::from)?;

    Ok(())
}

#[cfg(not(feature = "tracing"))]
pub fn init(tracing_opts: &cli::TracingOpts) -> Result<(), Error> {
    if !tracing_opts.enabled {
        return Ok(());
    }

    tracing_subscriber::fmt().init();

    Ok(())
}
