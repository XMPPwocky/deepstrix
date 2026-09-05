//! HTTP error responses, OpenAI-shape.
//!
//! ```text
//! { "error": { "message": "...", "type": "...", "code": "..." } }
//! ```

use axum::extract::Request;
use axum::http::StatusCode;
use axum::middleware::Next;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Serialize;

#[derive(Debug)]
pub enum ApiError {
    BadRequest(String),
    EngineFailed(color_eyre::eyre::Report),
    ContextExhausted(String),
    /// Engine queue is full. Maps to HTTP 503 with a Retry-After hint.
    Busy(String),
}

impl ApiError {
    fn status(&self) -> StatusCode {
        match self {
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::ContextExhausted(_) => StatusCode::BAD_REQUEST,
            ApiError::EngineFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
            ApiError::Busy(_) => StatusCode::SERVICE_UNAVAILABLE,
        }
    }

    fn body(&self) -> ApiErrorBody {
        match self {
            ApiError::BadRequest(msg) => ApiErrorBody {
                error: ApiErrorDetail {
                    message: msg.clone(),
                    kind: "invalid_request_error",
                    code: "bad_request",
                },
            },
            ApiError::ContextExhausted(msg) => ApiErrorBody {
                error: ApiErrorDetail {
                    message: msg.clone(),
                    kind: "invalid_request_error",
                    code: "context_length_exceeded",
                },
            },
            ApiError::EngineFailed(report) => ApiErrorBody {
                error: ApiErrorDetail {
                    message: format!("{report:#}"),
                    kind: "server_error",
                    code: "engine_error",
                },
            },
            ApiError::Busy(msg) => ApiErrorBody {
                error: ApiErrorDetail {
                    message: msg.clone(),
                    kind: "server_error",
                    code: "engine_busy",
                },
            },
        }
    }
}

#[derive(Serialize)]
struct ApiErrorBody {
    error: ApiErrorDetail,
}

#[derive(Serialize)]
struct ApiErrorDetail {
    message: String,
    #[serde(rename = "type")]
    kind: &'static str,
    code: &'static str,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = self.status();
        let body = self.body();
        // Stash code + message in a response extension so the router's
        // `log_error_responses` layer can log *why* we failed next to the
        // method and path it already has. Without it every 4xx/5xx we
        // return is invisible server-side: the body goes to the client
        // and nowhere else, and the log looks perfectly healthy.
        let detail = ErrorDetail {
            code: body.error.code,
            message: body.error.message.clone(),
        };
        let mut resp = (status, Json(body)).into_response();
        resp.extensions_mut().insert(detail);
        // Retry-After tells well-behaved clients (incl. letta's
        // pi-ai retryable-error path) to back off rather than
        // hot-retry into the same full queue. 2s is roughly one
        // long decode's worth.
        if matches!(self, ApiError::Busy(_)) {
            resp.headers_mut()
                .insert(axum::http::header::RETRY_AFTER, "2".parse().unwrap());
        }
        resp
    }
}

impl From<color_eyre::eyre::Report> for ApiError {
    fn from(r: color_eyre::eyre::Report) -> Self {
        ApiError::EngineFailed(r)
    }
}

/// Reason attached to an error response by [`ApiError::into_response`],
/// read back by [`log_error_responses`]. Server-local — extensions are
/// not part of the HTTP wire format.
#[derive(Clone)]
pub struct ErrorDetail {
    pub code: &'static str,
    pub message: String,
}

/// Router layer: log every response that carries a 4xx/5xx status.
///
/// Responses built from [`ApiError`] carry an [`ErrorDetail`], so those
/// log with their OpenAI error code and message. Everything else logs
/// with the status alone: axum extractor rejections (413 over the body
/// limit, 422 malformed JSON), 404s on an unknown route, and the
/// `/readyz` engine-stall 503.
///
/// Cannot see a failure that happens *after* 200 + headers are on the
/// wire, i.e. mid-SSE-stream — that path terminates the stream instead.
pub async fn log_error_responses(req: Request, next: Next) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let resp = next.run(req).await;
    let status = resp.status();
    if status.is_client_error() || status.is_server_error() {
        match resp.extensions().get::<ErrorDetail>() {
            Some(d) => tracing::warn!(
                %method,
                %path,
                status = status.as_u16(),
                code = d.code,
                // NOT `message` -- tracing treats a field of that name as
                // the event message, which would swallow the "http error
                // response" marker the log is grepped by.
                reason = %d.message,
                "http error response"
            ),
            None => tracing::warn!(
                %method,
                %path,
                status = status.as_u16(),
                "http error response"
            ),
        }
    }
    resp
}
