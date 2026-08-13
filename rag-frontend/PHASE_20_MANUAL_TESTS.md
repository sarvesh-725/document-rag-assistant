# Phase 20 Manual Browser Tests

Use the same authenticated user in two browser tabs.

1. Open the document library in Tab A and Tab B.
2. Upload a document in Tab A. Confirm Tab B refreshes and shows the new document without a page reload.
3. Delete a document in Tab A. Confirm Tab B removes it after refetching the document API.
4. Start an upload in Tab A and wait for ingestion to finish. Confirm Tab B updates the document status from `PROCESSING` to `READY` or `FAILED`.
5. Create a session in Tab A. Confirm Tab B shows the session.
6. Delete a session in Tab A. Confirm Tab B removes it and clears the active session if that session was open.
7. Send a message in Tab A. With the same session open in Tab B, confirm Tab B refreshes that session's history.
8. Inspect BroadcastChannel messages in DevTools and confirm they contain only an event type and, for `session_changed`, a session ID. No document or session arrays should be broadcast.
