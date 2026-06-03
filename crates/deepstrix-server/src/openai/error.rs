//! HTTP error responses, OpenAI-shape.
//!
//!     { "error": { "message": "...", "type": "...", "code": "..." } }

use axum::http::StatusCode;
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
        let mut resp = (status, Json(body)).into_response();
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
