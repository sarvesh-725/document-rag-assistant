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
