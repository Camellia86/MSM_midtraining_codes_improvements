"""Base ChatGenerator class for SFT chat generation.

Shared infrastructure for question→response→filter→save pipeline.
"""

import asyncio
import os
from abc import ABC, abstractmethod
from pathlib import Path
from dataclasses import dataclass

from safetytooling.utils.experiment_utils import ExperimentConfigBase
from safetytooling.utils import utils
from src.utils.inference_utils import single_prompt_api_call


@dataclass
class ChatGeneratorConfig(ExperimentConfigBase):
    """Base configuration for chat generators."""
    output_dir: Path = None  # Set automatically from dataset_name in __post_init__
    dataset_name: str = None
    n_samples: int = None
    model_id: str = "claude-opus-4-5-20251101"
    temperature: float = 1.0
    max_tokens: int = 8000
    max_concurrent_requests: int = 50
    skip_existing: bool = True
    use_batch_api: bool = False
    batch_timeout_minutes: int = 180
    anthropic_batch_tag: str = "ANTHROPIC_BATCH_API_KEY"

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = Path(f"data/ft/{self.dataset_name}")
        self.temperature = None if "claude" not in self.model_id else self.temperature

        super().__post_init__()

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir = self.output_dir / "source"
        self.source_dir.mkdir(exist_ok=True)

    def setup(self):
        """Set up API environment."""
        utils.setup_environment(
            openai_tag=self.openai_tag,
            anthropic_tag=self.anthropic_tag,
            openrouter_tag=self.openrouter_tag,
        )


class ChatGenerator(ABC):
    """Base class for chat dataset generators."""

    def __init__(self, config: ChatGeneratorConfig):
        self.config = config
        self.config.setup_experiment()
        self.semaphore = asyncio.Semaphore(config.max_concurrent_requests)

        if config.use_batch_api:
            self.batch_api_key = os.environ.get(config.anthropic_batch_tag)
            if not self.batch_api_key:
                raise ValueError(f"Batch API key not found in environment: {config.anthropic_batch_tag}")

    async def _api_call(self, prompt, max_tokens: int = None, **kwargs):
        """Make a single API call with semaphore."""
        async with self.semaphore:
            response = await single_prompt_api_call(
                api=self.config.api,
                MODEL_ID=self.config.model_id,
                prompt=prompt,
                max_tokens=max_tokens or self.config.max_tokens,
                temperature=self.config.temperature,
                **kwargs,
            )
            return response

    async def _execute_batch(self, prompts: list, desc: str = "Processing", max_tokens: int = None) -> list[str]:
        """Execute batch of prompts with progress tracking."""
        from tqdm.asyncio import tqdm_asyncio

        async def call_one(prompt):
            return await self._api_call(prompt, max_tokens=max_tokens)

        tasks = [call_one(p) for p in prompts]
        results = await tqdm_asyncio.gather(*tasks, desc=desc)
        return results

    async def _batch_api_call(self, prompts: list, max_tokens: int = None) -> list:
        """Submit prompts via BatchInferenceAPI and return list of response strings."""
        from safetytooling.apis.batch_api import BatchInferenceAPI

        batch_api = BatchInferenceAPI(
            log_dir=self.config.prompt_history_dir,
            cache_dir=self.config.cache_dir,
            use_redis=self.config.use_redis,
            no_cache=not self.config.enable_cache,
            anthropic_api_key=self.batch_api_key,
        )

        timeout_seconds = self.config.batch_timeout_minutes * 60
        try:
            responses, batch_id = await asyncio.wait_for(
                batch_api(
                    model_id=self.config.model_id,
                    prompts=prompts,
                    max_tokens=max_tokens or self.config.max_tokens,
                    temperature=self.config.temperature,
                ),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            print(f"\nBatch timed out after {self.config.batch_timeout_minutes} min.")
            return None

        print(f"Batch completed! Batch ID: {batch_id}")
        return [r.completion if r is not None else None for r in responses]

    @abstractmethod
    async def generate_questions(self) -> list[dict]:
        pass

    @abstractmethod
    async def generate_responses(self, questions: list[dict]) -> list[dict]:
        pass

    @abstractmethod
    async def filter_examples(self, qa_pairs: list[dict]) -> tuple[list[dict], list[dict]]:
        pass

    @abstractmethod
    def save_final_dataset(self, examples: list[dict]):
        pass
