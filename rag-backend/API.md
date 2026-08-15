# API Contract

Base URL: `/api/v1`. JSON errors use `{ "detail": { "code": "...", "message": "..." } }`.

## Authentication

### `POST /auth/signup`
- Auth: none
- Request: `{ "username": string, "password": string }`
- Response: `201 { "user_id": string, ... }`
- Errors: `INVALID_REQUEST`
- Idempotency: no; username is unique
- Ownership: creates only the authenticated account represented by the request

### `POST /auth/login`
- Auth: none
- Request: form fields `username`, `password`
- Response: `{ "access_token": string, "token_type": "bearer" }`
- Errors: `AUTH_REQUIRED`, `RATE_LIMITED`
- Idempotency: no side effects beyond rate-limit counters
- Ownership: token identifies one user

## Sessions

### `GET /sessions`
- Auth: bearer token
- Request: none
- Response: session summaries owned by the user
- Errors: `AUTH_REQUIRED`
- Idempotency: read-only
- Ownership: never returns another user's sessions

### `POST /sessions`
- Auth: bearer token
- Request: optional `{ "title": string }`
- Response: `201` new server-generated session ID
- Errors: `AUTH_REQUIRED`
- Idempotency: each request creates one session
- Ownership: session belongs to the authenticated user

### `DELETE /sessions/{id}`
- Auth: bearer token
- Request: path session UUID
- Response: success message
- Errors: `AUTH_REQUIRED`, `SESSION_NOT_FOUND`
- Idempotency: deleting an absent/deleted session is not successful
- Ownership: only the owner can delete it

### `GET /sessions/{id}/messages`
- Auth: bearer token
- Request: `limit` (1-100), `offset` (>=0)
- Response: paginated messages ordered by `sequence_number` ascending
- Errors: `AUTH_REQUIRED`, `SESSION_NOT_FOUND`
- Idempotency: read-only
- Ownership: only the session owner can read it

## Documents

### `GET /documents`
- Auth: bearer token
- Request: none
- Response: live global documents, including `PROCESSING` and `FAILED`
- Errors: `AUTH_REQUIRED`
- Idempotency: read-only
- Ownership: filtered by authenticated `user_id`

### `POST /documents`
- Auth: bearer token
- Request: multipart `file` (`.pdf` or `.txt`)
- Response: `202` with `document_id`, `version_id`, `job_id`, `display_name`, `PROCESSING`
- Errors: `AUTH_REQUIRED`, `INVALID_FILE_TYPE`, `FILE_TOO_LARGE`
- Idempotency: each upload creates a new logical document
- Ownership: document and UUID-based storage belong to the authenticated user

### `DELETE /documents/{id}`
- Auth: bearer token
- Request: path document UUID
- Response: logical deletion success
- Errors: `AUTH_REQUIRED`, `DOCUMENT_NOT_FOUND`
- Idempotency: safe repeated logical deletion
- Ownership: only the document owner can delete it; cleanup waits for active references

### `GET /ingestion-jobs/{id}`
- Auth: bearer token
- Request: path job UUID
- Response: job, document, version, status, stage, attempts, and error fields
- Errors: `AUTH_REQUIRED`, `INGESTION_FAILED`
- Idempotency: read-only
- Ownership: job is visible only through its owned document

## Chat

### `POST /chat/query`
- Auth: bearer token
- Request: `{ "session_id": string, "client_request_id": string, "question": string, "selected_document_ids": string[] }`
- Response: typed SSE stream with `message_start`, `retrieval_complete`, `source`, `token`, `message_complete`, `error`, or `cancelled`
- Errors: `AUTH_REQUIRED`, `SESSION_NOT_FOUND`, `DOCUMENT_NOT_READY`, `DOCUMENT_DELETING`, `NO_DOCUMENT_SELECTED`, `NO_RELEVANT_EVIDENCE`, `LLM_UNAVAILABLE`
- Idempotency: unique `(session_id, client_request_id)`; duplicates reconnect to existing state and never start another generation
- Ownership: session and every selected document are verified for the authenticated user; retrieval is version-scoped
