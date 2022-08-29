//! Error types

use miette::Diagnostic;
use thiserror::Error;

#[derive(Error, Debug, Diagnostic)]
pub enum Error {
    #[error("Cannot open database")]
    #[diagnostic(code(redirekt::db_open))]
    DatabaseOpenError(#[source] sqlx::Error),

    #[cfg(feature = "tracing")]
    #[error("Cannot install the tracing pipeline")]
    #[diagnostic(code(metastatus::tracer_install_error))]
    TraceInstallationError(#[source] opentelemetry::trace::TraceError),

    #[error("Cannot set global tracing subscriber")]
    #[diagnostic(code(metastatus::default_subscriber_error))]
    DefaultSubscriberInitFailed(#[from] tracing_subscriber::util::TryInitError),

    #[error("Cannot bind http server to the requested address")]
    #[diagnostic(code(metastatus::http_bind_error))]
    HttpServerBindingFailed(#[source] std::io::Error),

    #[error("Listening address is invalid")]
    #[diagnostic(code(metastatus::invalid_address_error))]
    InvalidListeningAddress(#[source] std::io::Error),

    #[error("Database migration failed")]
    DatabaseMigrationError(#[source] sqlx::migrate::MigrateError),
}
