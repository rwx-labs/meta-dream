//! Middleware to add an ULID-based request id response header

use http::{header::HeaderValue, Request};
use tower_http::request_id::{MakeRequestId, RequestId};

// A `MakeRequestId` that increments an atomic counter
#[derive(Clone, Default)]
pub struct MakeRequestUlid;

impl MakeRequestId for MakeRequestUlid {
    fn make_request_id<B>(&mut self, _: &Request<B>) -> Option<RequestId> {
        let mut buf = [0; ulid::ULID_LEN];
        let mut rng = rand::thread_rng();
        let ulid = ulid::Ulid::with_source(&mut rng);
        ulid.to_str(&mut buf).unwrap();

        Some(RequestId::new(HeaderValue::from_bytes(&buf).unwrap()))
    }
}
