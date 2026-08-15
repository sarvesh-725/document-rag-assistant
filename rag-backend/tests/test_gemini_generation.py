import pytest

from app.services.gemini_generation import GeminiAnswerStreamer


class Chunk:
    def __init__(self, content):
        self.content = content


class FakeLlm:
    def __init__(self):
        self.messages = None

    async def astream(self, messages):
        self.messages = messages
        yield Chunk("one")
        yield Chunk(" two")


class RetryLlm:
    def __init__(self):
        self.calls = 0

    async def astream(self, _messages):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("temporary provider failure")
        yield Chunk("recovered")


@pytest.mark.asyncio
async def test_gemini_stream_uses_supplied_context_messages_and_yields_text():
    llm = FakeLlm()
    streamer = GeminiAnswerStreamer(llm=llm)
    messages = [
        {"role": "system", "content": "safe"},
        {"role": "user", "content": "CURRENT USER QUESTION\nquestion"},
    ]

    result = [piece async for piece in streamer.stream(messages)]

    assert result == ["one", " two"]
    assert llm.messages == messages


@pytest.mark.asyncio
async def test_gemini_stream_retries_only_before_first_token():
    llm = RetryLlm()
    streamer = GeminiAnswerStreamer(llm=llm)

    result = [piece async for piece in streamer.stream([])]

    assert result == ["recovered"]
    assert llm.calls == 2
