use std::net::ToSocketAddrs;
use std::sync::Arc;

use axum::{handler::Handler, routing::get, Extension, Router};
use http::{header::HeaderName, Request, Version};
use hyper::Body;
use sqlx::AnyPool;
use tower::ServiceBuilder;
use tower_http::{
    compression::CompressionLayer,
    trace::{DefaultOnResponse, TraceLayer},
    ServiceBuilderExt,
};
use tracing::{debug, instrument, trace};

use crate::cli::HttpOpts;
use crate::Error;

mod api;
mod request_id;

#[derive(Debug, Clone)]
struct State {
    /// Handle to an open database pool
    pool: AnyPool,
}

use request_id::MakeRequestUlid;

fn http_flavor_from_version(version: Version) -> &'static str {
    match version {
        Version::HTTP_09 => "0.9",
        Version::HTTP_10 => "1.0",
        Version::HTTP_11 => "1.1",
        Version::HTTP_2 => "2.0",
        Version::HTTP_3 => "3.0",
        _ => unreachable!(),
    }
}

#[instrument(skip_all)]
pub async fn start_server(
    opts: &HttpOpts,
    handle: axum_server::Handle,
    pool: AnyPool,
) -> Result<(), Error> {
    trace!("creating http server");

    let state = Arc::new(State { pool });
    let x_request_id = HeaderName::from_static("x-request-id");

    // Build API routes
    let api_router = api::router();

    // Build application routes
    let app = Router::new()
        .route("/healthz", get(|| async { "ok" }))
        .nest("/api/v1", api_router)
        // Add a tracing layer that adds standard tags, unlike tower's tracing layer
        .layer(
            ServiceBuilder::new()
                .set_x_request_id(MakeRequestUlid::default())
                .layer(
                    TraceLayer::new_for_http()
                        .make_span_with(move |request: &Request<Body>| {
                            tracing::debug_span!("request",
                                http.method = %request.method(),
                                http.target = %request.uri(),
                                http.flavor = %http_flavor_from_version(request.version()),
                                request_id = %request.headers().get(x_request_id.clone()).unwrap().to_str().unwrap()
                            )
                        })
                        .on_response(DefaultOnResponse::new().include_headers(true)),
                )
                .propagate_x_request_id()
                // Enable compression
                .layer(CompressionLayer::new())
                // Add the state extension
                .layer(Extension(state)),
        );

    let addr = (opts.host.clone(), opts.port)
        .to_socket_addrs()
        .map_err(Error::InvalidListeningAddress)?
        .next()
        .unwrap();

    debug!(%addr, "binding to address");

    axum_server::bind(addr)
        .handle(handle)
        .serve(app.into_make_service())
        .await
        .map_err(Error::HttpServerBindingFailed)?;

    Ok(())
}

mod handlers {
    use std::sync::Arc;

    use super::State;
    use axum::{
        extract::{Extension, Host},
        http::{header, HeaderMap, HeaderValue, StatusCode, Uri},
        response::IntoResponse,
    };
    use tracing::{debug, instrument};
}
