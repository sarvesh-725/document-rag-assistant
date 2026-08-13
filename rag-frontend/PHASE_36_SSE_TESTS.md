# SSE Parser Test Cases

These cases exercise `app/lib/sse.ts` with the same `SseParser` instance.

1. Split one frame across two chunks:
   - Push `event: token\ndata: {"text":"hel`
   - Push `lo"}\n\n`
   - Expect one `token` event with `text=hello`.
2. Split one frame across three chunks:
   - Push `event: message_start\n`
   - Push `data: {"message_id":"M1"}`
   - Push `\n\n`
   - Expect one `message_start` event only after the third push.
3. Push multiple frames in one chunk:
   - Push `event: retrieval_complete\ndata: {"context_count":1}\n\nevent: token\ndata: {"text":"x"}\n\n`
   - Expect two events in order.
4. Leave an incomplete final frame buffered:
   - Push `event: source\ndata: {"source_id":"S1"}`
   - Expect no event until the terminating blank line arrives.
5. Verify `finish()` parses one complete final frame and discards no complete prior frame.
