"""Mock LLM for testing."""
from typing import Any, AsyncIterator, Iterator
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun, AsyncCallbackManagerForLLMRun


class MockLLM(BaseChatModel):
    """
    Mock LLM that returns predefined responses without calling real LLM APIs.

    Attributes:
        mock_responses: Dictionary mapping prompts/keywords to responses
        default_response: Fallback response when no match found
    """
    mock_responses: dict[str, str] = {}
    default_response: str = "Mock LLM response"

    @property
    def _llm_type(self) -> str:
        """Return identifier for this LLM."""
        return "mock"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate mock response based on input messages.

        Args:
            messages: Input messages
            stop: Stop sequences (ignored)
            run_manager: Callback manager (ignored)
            **kwargs: Additional arguments (ignored)

        Returns:
            ChatResult with mock response
        """
        # Check if any message content matches our mock responses
        response = self.default_response
        for msg in messages:
            content = msg.content.lower() if hasattr(msg.content, 'lower') else str(msg.content).lower()
            for key, value in self.mock_responses.items():
                if key.lower() in content:
                    response = value
                    break

        message = AIMessage(content=response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async version of _generate."""
        return self._generate(messages, stop, None, **kwargs)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        """Stream mock response token by token."""
        result = self._generate(messages, stop, run_manager, **kwargs)
        content = result.generations[0].message.content

        # Stream word by word
        words = content.split()
        for word in words:
            yield ChatGeneration(message=AIMessage(content=word + " "))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGeneration]:
        """Async stream mock response token by token."""
        result = await self._agenerate(messages, stop, run_manager, **kwargs)
        content = result.generations[0].message.content

        # Stream word by word
        words = content.split()
        for word in words:
            yield ChatGeneration(message=AIMessage(content=word + " "))


def create_mock_llm_with_responses(responses: dict[str, str] | None = None) -> MockLLM:
    """
    Create a mock LLM with predefined responses.

    Args:
        responses: Dictionary mapping keywords to responses

    Returns:
        Configured MockLLM instance
    """
    if responses is None:
        responses = {
            "question": "What is the capital of France?",
            "evaluate": "CORRECT: The answer is accurate and complete.",
            "grade": "pass",
            "summary": "Interview completed successfully.",
        }

    return MockLLM(mock_responses=responses)
