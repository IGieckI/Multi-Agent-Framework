"""
Run a model locally through HuggingFace transformers
"""

import asyncio
import threading
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers.generation.streamers import BaseStreamer

from framework.agents.backends.base import Completion, Sampling, TextListener, split_reasoning, to_chat_messages
from framework.agents.prompt import Prompt


def pick_dtype(device: str) -> torch.dtype:
    """Pick bfloat16 on GPUs that support it, float16 otherwise"""
    if device == "cpu":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16


def keep_token_ids(shipped: GenerationConfig) -> GenerationConfig:
    """Copy only the special-token ids from the model's shipped config, leaving all other settings empty/filled by our default values.
    Note: This makes all models sample the same way"""
    return GenerationConfig(
        bos_token_id=shipped.bos_token_id,
        eos_token_id=shipped.eos_token_id,
        pad_token_id=shipped.pad_token_id,
    )


def base_generation_config(shipped: GenerationConfig, use_model_sampling: bool) -> GenerationConfig:
    """Choose the config the model stores:
    - use_model_sampling=False: "Sampling" values are used.
    - use_model_sampling=True: use the model's author full config.
    """
    return shipped if use_model_sampling else keep_token_ids(shipped)


def generation_config(sampling: Sampling, use_model_sampling: bool = False) -> GenerationConfig:
    """Build the config
    - use_model_sampling=False: every `Sampling` value (temperature, top_p, top_k, max_tokens)
    - use_model_sampling=True: only max_tokens, the other sampling settings come from the stored config
    """
    if use_model_sampling:
        return GenerationConfig(max_new_tokens=sampling.max_tokens)
    return GenerationConfig(
        do_sample=True,
        temperature=sampling.temperature,
        top_p=sampling.top_p,
        top_k=sampling.top_k,
        max_new_tokens=sampling.max_tokens,
    )


class LiveText(BaseStreamer):
    """Hands the text generated so far to a listener after every new token (batch size 1)"""

    def __init__(self, tokenizer, on_text: TextListener) -> None:
        self.tokenizer = tokenizer
        self.on_text = on_text
        self.ids: list[int] = []
        self.prompt_seen = False

    def put(self, value) -> None:
        if not self.prompt_seen:  # generate() first pushes the prompt itself
            self.prompt_seen = True
            return
        self.ids += value.reshape(-1).tolist()
        self.on_text(self.tokenizer.decode(self.ids, skip_special_tokens=True))

    def end(self) -> None:
        pass


class HFLocalBackend:
    def __init__(
        self,
        model_id: str,
        use_model_sampling: bool = False,
        revision: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.requested_revision = revision
        self.use_model_sampling = use_model_sampling
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = pick_dtype(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=self.dtype, device_map=self.device, revision=revision
        )
        self.model_sampling_settings = self.model.generation_config.to_diff_dict()
        self.model.generation_config = base_generation_config(self.model.generation_config, use_model_sampling)
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = eos if isinstance(eos, list) else [eos]
        # We use a lock for reproducibility, to avoid mixing seeeds across concurrent calls
        # Note: Agents batching might be needed (several prompts in one generate() call)
        self.lock = threading.Lock()

    def engine_info(self) -> dict:
        return {
            "backend": "hf_local",
            "engine": "transformers",
            "engine_version": transformers.__version__,
            "model_id": self.model_id,
            # the exact commit of the weights that were loaded, and their size
            "revision": getattr(self.model.config, "_commit_hash", None),
            "requested_revision": self.requested_revision,
            "parameters": self.model.num_parameters(),
            "torch_version": torch.__version__,
            "dtype": str(self.dtype),
            "quantization": "none",
            "context_window": self.model.config.max_position_embeddings,
            "device": self.device,
            "prefix_caching": False,
            "sampling_settings": self.model_sampling_settings if self.use_model_sampling else "from Sampling",
        }

    async def generate(self, prompt: Prompt, sampling: Sampling, on_text: TextListener | None = None) -> Completion:
        return await asyncio.to_thread(self._generate, prompt, sampling, on_text)

    def chat_text(self, prompt: Prompt, reasoning: bool) -> str:
        return self.tokenizer.apply_chat_template(
            to_chat_messages(prompt),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=reasoning,
        )

    def count_tokens(self, prompt: Prompt, reasoning: bool = True) -> int:
        """Count prompt tokens before the call"""
        return len(self.tokenizer(self.chat_text(prompt, reasoning), add_special_tokens=False)["input_ids"])

    def _generate(self, prompt: Prompt, sampling: Sampling, on_text: TextListener | None) -> Completion:
        text = self.chat_text(prompt, sampling.reasoning)
        inputs = self.tokenizer(text, add_special_tokens=False, return_tensors="pt").to(self.device)
        prompt_tokens = inputs["input_ids"].shape[1]

        with self.lock:
            started = time.perf_counter()
            if sampling.seed is not None:
                torch.manual_seed(sampling.seed)
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, generation_config=generation_config(sampling, self.use_model_sampling),
                    streamer=LiveText(self.tokenizer, on_text) if on_text else None,
                )
            elapsed = time.perf_counter() - started

        generated = output[0, prompt_tokens:].tolist()
        raw = self.tokenizer.decode(generated, skip_special_tokens=True)
        reasoning, answer = split_reasoning(raw)
        return Completion(
            text=answer,
            reasoning=reasoning,
            prompt_tokens=prompt_tokens,
            generated_tokens=len(generated),
            elapsed_s=elapsed,
            finish_reason="stop" if generated and generated[-1] in self.eos_ids else "length",
            full_raw_answer=raw,
        )
