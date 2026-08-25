"""The chat model behind query expansion and paraphrase generation.

Points at the local 9router proxy. Calls are synchronous so retrievers share
one non-async interface; callers wanting concurrency use threads.
"""

import os
import time
from dataclasses import dataclass

from pydantic_ai import Agent
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

MODEL = "ag/gemini-3.7-flash-medium"

# The proxy drops the occasional connection when a few hundred calls run back
# to back, and sometimes answers with a body that is not a completion at all
# (every field null) when an upstream provider errors. One unlucky request
# should not lose a whole eval arm.
_ATTEMPTS = 4
_TRANSIENT = (ModelAPIError, UnexpectedModelBehavior)


def _model(name: str) -> OpenAIChatModel:
    return OpenAIChatModel(
        name,
        provider=OpenAIProvider(
            base_url="http://localhost:20128/v1",
            api_key=os.environ["NINEROUTER_API_KEY"],
        ),
    )


def agent(instructions: str, output_type: type = str, model: str = MODEL) -> Agent:
    """An agent on the named model. Callers that care about latency pass one."""
    return Agent(_model(model), instructions=instructions, output_type=output_type)


@dataclass(frozen=True, slots=True)
class Reply:
    text: str
    input_tokens: int
    output_tokens: int


def ask_reply(agent_: Agent, prompt: str) -> Reply:
    """Run the agent and report what the call cost, for the cost benchmark."""
    for attempt in range(_ATTEMPTS):
        try:
            result = agent_.run_sync(prompt)
            usage = result.usage
            return Reply(
                text=str(result.output),
                input_tokens=usage.input_tokens or 0,
                output_tokens=usage.output_tokens or 0,
            )
        except _TRANSIENT:
            if attempt == _ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def ask(agent_: Agent, prompt: str) -> str:
    return ask_reply(agent_, prompt).text
