//! CLI arguments

use clap::Parser;

#[cfg(feature = "tracing-jaeger")]
// Export traces to an OTLP endpoint by default
const DEFAULT_TRACING_EXPORTER: &str = "jaeger";

#[cfg(all(feature = "tracing-datadog", not(feature = "tracing-jaeger")))]
// Export traces to Datadog by default when OTLP is not available
const DEFAULT_TRACING_EXPORTER: &str = "datadog";

#[cfg(feature = "tracing-jaeger")]
// Default endpoint when OTLP is enabled
const DEFAULT_TRACING_ENDPOINT: &str = "http://localhost:4317";

#[cfg(all(feature = "tracing-datadog", not(feature = "tracing-jaeger")))]
// Default endpoint when OTLP is not available but Datadog is
const DEFAULT_TRACING_ENDPOINT: &str = "http://localhost:8126";

#[derive(clap::ValueEnum, Clone, Debug, Eq, PartialEq)]
pub enum TracingExporter {
    #[cfg(feature = "tracing-jaeger")]
    Jaeger,
    #[cfg(feature = "tracing-datadog")]
    Datadog,
}

#[derive(Parser, Debug, Clone)]
pub struct Opts {
    /// URL to the database
    #[clap(
        long,
        short,
        default_value = "postgres://localhost/metastable_development",
        env = "DATABASE_URL"
    )]
    pub database_url: String,

    #[clap(flatten)]
    pub tracing_opts: TracingOpts,

    #[clap(flatten)]
    pub http_opts: HttpOpts,
}

#[derive(Parser, Debug, Clone)]
pub struct TracingOpts {
    /// Enable tracing.
    #[clap(
        long = "tracing-enabled",
        env = "METASTABLE_TRACING_ENABLED",
        action = clap::ArgAction::Set,
        value_parser = clap::value_parser!(bool),
        default_value = "true",
        default_missing_value = "true"
    )]
    pub enabled: bool,

    /// Select the tracing exporter.
    #[cfg(feature = "tracing")]
    #[clap(
        long = "tracing-exporter",
        value_enum,
        default_value = DEFAULT_TRACING_EXPORTER,
        env = "METASTABLE_TRACING_EXPORTER",
    )]
    pub exporter: TracingExporter,

    /// Set the service name.
    #[clap(
        long = "tracing-service-name",
        default_value = env!("CARGO_PKG_NAME"),
        env = "METASTABLE_TRACING_SERVICE_NAME"
    )]
    pub service_name: String,

    /// Set the tracing endpoint.
    ///
    /// For Jaeger, this is the collector endpoint.
    /// For Datadog, this is the agent endpoint.
    #[cfg(feature = "tracing")]
    #[clap(
        long = "tracing-endpoint",
        env = "METASTABLE_TRACING_ENDPOINT",
        default_value = DEFAULT_TRACING_ENDPOINT
    )]
    pub endpoint: String,
}

#[derive(Parser, Debug, Clone, Eq, PartialEq)]
pub struct HttpOpts {
    /// Listening address for the HTTP API.
    #[clap(short, long, default_value = "0.0.0.0", env = "METASTATUS_API_HOST")]
    pub host: String,

    /// Listening port for the HTTP API.
    #[clap(short, long, default_value = "3000", env = "METASTATUS_API_PORT")]
    pub port: u16,
}
