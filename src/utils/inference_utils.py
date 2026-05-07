"""Core inference API utilities.

This module provides the main entry point for making LLM API calls using InferenceAPI.
All usage of InferenceAPI (except BatchInferenceAPI) should go through single_prompt_api_call().
"""

from pydantic import BaseModel

from safetytooling.data_models import Prompt
from safetytooling.apis.inference.api import InferenceAPI
from safetytooling.data_models.inference import LLMResponse, StopReason


async def single_prompt_api_call(
    api: InferenceAPI,
    MODEL_ID: str,
    prompt: Prompt,
    max_tokens: int | None = 5000,
    temperature: float | None = 0.8,
    **kwargs
) -> str | LLMResponse:
    """
    Make a single API call to the model.

    This is the primary entry point for all InferenceAPI usage in the codebase.
    Handles structured outputs, error handling, and debug logging.

    Args:
        api: InferenceAPI instance
        MODEL_ID: Model identifier (e.g., "claude-sonnet-4-5-20250929")
        prompt: Prompt object with messages
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        debug: Enable debug logging
        **kwargs: Additional arguments (e.g., output_format for structured outputs)

    Returns:
        str: The completion text (default)
        LLMResponse: Full response object if structured outputs used

    Raises:
        ValueError: If no response returned or invalid output_format
        Exception: API errors (rate limits, timeouts, etc.)
    """
    try:
        using_structured_outputs = False
        if "output_format" in kwargs:
            output_format = kwargs["output_format"]
            if not isinstance(output_format, type) or not issubclass(output_format, BaseModel):
                raise ValueError(f"Output format must be a Pydantic BaseModel, got {type(output_format)}")
            using_structured_outputs = True
            if "gpt" in MODEL_ID:  # OpenAI uses 'response_format' instead of 'output_format'
                kwargs["response_format"] = output_format
                del kwargs["output_format"]

        response_list = await api(
            model_id=MODEL_ID,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )  # list[LLMResponse]

        if not response_list or len(response_list) == 0:
            raise ValueError("No response returned from API")

        response = response_list[0]

        if using_structured_outputs:
            if response.generated_content:
                result = response.generated_content[0].content[0]["parsed_output"]
            else:
                print("Warning: No generated content in response")
                result = None
        else:
            result = response.completion

        return result
    except Exception as e:
        error_msg = str(e).lower()
        if "rate" in error_msg or "429" in error_msg or "quota" in error_msg:
            print(f"  RATE LIMIT HIT: {type(e).__name__}: {str(e)}")
        elif "timeout" in error_msg:
            print(f"  TIMEOUT: {type(e).__name__}: {str(e)}")
        else:
            print(f"  API call failed with error: {type(e).__name__}: {str(e)}")
        raise e
