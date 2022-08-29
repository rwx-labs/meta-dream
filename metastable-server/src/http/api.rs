use axum::{routing::post, Router};

pub fn router() -> Router {
    Router::new()
}

mod handlers {
    use std::sync::Arc;

    use axum::{
        extract::{Extension, Json},
        http::StatusCode,
    };
    use serde::Deserialize;
    use tracing::{debug, instrument};
    use url::Url;

    use crate::http::State;
}
