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
}

impl ApiError {
    fn status(&self) -> StatusCode {
        match self {
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::ContextExhausted(_) => StatusCode::BAD_REQUEST,
            ApiError::EngineFailed(_) => StatusCode::INTERNAL_SERVER_ERROR,
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
        (status, Json(body)).into_response()
    }
}

impl From<color_eyre::eyre::Report> for ApiError {
    fn from(r: color_eyre::eyre::Report) -> Self {
        ApiError::EngineFailed(r)
    }
}
