import pytest

from critpt_eval.commands.smoke import smoke
from critpt_eval.generation.client import ChatMessages, ChatResponse


class FakeClient:
    def __init__(self) -> None:
        self.ready = False

    async def preflight(self) -> None:
        self.ready = True

    async def chat(self, messages: ChatMessages) -> ChatResponse:
        assert self.ready
        assert messages == (
            {"role": "user", "content": "Reply with exactly: ready /no_think"},
        )
        return ChatResponse("ready", None, "local", None, None, 0.1, {})


@pytest.mark.asyncio
async def test_smoke() -> None:
    client = FakeClient()

    response = await smoke(client)

    assert response.content == "ready"
