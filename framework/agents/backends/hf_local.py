"""
Run a model locally through HuggingFace transformers
"""

import asyncio
import threading
import time
from pathlib import Path

import torch
import transformers
from huggingface_hub import try_to_load_from_cache
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, LogitsProcessor, LogitsProcessorList
from transformers.generation.streamers import BaseStreamer

from framework.agents.backends.base import (Completion, Sampling, TextListener, confidence_summary, split_reasoning,
                                            to_chat_messages)
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


def loaded_commit(model, model_id: str, revision: str | None) -> str | None:
    """The commit of the loaded weights: transformers' record of it, or, offline (where it records none), the name of
    the cache snapshot folder they were read from (FINDINGS.md F2)"""
    if commit := getattr(model.config, "_commit_hash", None):
        return commit
    path = try_to_load_from_cache(model_id, "config.json", revision=revision)
    return Path(path).parent.name if isinstance(path, str) else None


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


def entropy(logprobs: torch.Tensor) -> float:
    """Entropy (nats) of a distribution given as log-probabilities (-inf for excluded tokens)"""
    return float(-torch.where(torch.isfinite(logprobs), logprobs.exp() * logprobs, 0.0).sum())


class TokenStats(LogitsProcessor):
    """Records, without changing the scores, each step's distribution: the log-probability of the token sampled from
    it (known at the next step, or from the output for the last one) and its entropy. With every sampling setting at
    its neutral value (temperature 1, top_p 1, top_k 0, as in every grid) it is the model's own distribution"""

    def __init__(self) -> None:
        self.previous: torch.Tensor | None = None
        self.logprobs: list[float] = []
        self.entropies: list[float] = []

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.previous is not None:
            self.logprobs.append(float(self.previous[input_ids[0, -1]]))
        self.previous = torch.log_softmax(scores[0].float(), dim=-1)
        self.entropies.append(entropy(self.previous))
        return scores

    def finish(self, generated: list[int]) -> None:
        if self.previous is not None and len(self.logprobs) < len(generated):
            self.logprobs.append(float(self.previous[generated[-1]]))
        self.previous = None


def token_ends(tokenizer, generated: list[int]) -> list[int]:
    """Where each generated token ends in the decoded text (special tokens decode to nothing)"""
    return [len(tokenizer.decode(generated[:i + 1], skip_special_tokens=True)) for i in range(len(generated))]


def confidence_meta(tokenizer, generated: list[int], raw: str, logprobs: list[float], entropies: list[float]) -> dict:
    """What a completion's meta holds with engine.log_confidence: the summary and the per-token lists (4 decimals)"""
    ends = token_ends(tokenizer, generated)
    return {"confidence": confidence_summary(raw, logprobs, entropies, ends),
            "token_logprobs": [round(x, 4) for x in logprobs], "token_entropies": [round(x, 4) for x in entropies],
            "token_ends": ends}


class HFLocalBackend:
    def __init__(
        self,
        model_id: str,
        use_model_sampling: bool = False,
        revision: str | None = None,
        confidence: bool = False,
    ) -> None:
        self.model_id = model_id
        # engine.log_confidence: record each answer's token log-probabilities and entropies (completion.meta)
        self.confidence = confidence
        self.requested_revision = revision
        self.use_model_sampling = use_model_sampling
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = pick_dtype(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=self.dtype, device_map=self.device, revision=revision
        )
        self.revision = loaded_commit(self.model, model_id, revision)
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
            "revision": self.revision,
            "requested_revision": self.requested_revision,
            "parameters": self.model.num_parameters(),
            "torch_version": torch.__version__,
            "dtype": str(self.dtype),
            "quantization": "none",
            "context_window": self.model.config.get_text_config().max_position_embeddings,
            "device": self.device,
            "prefix_caching": False,
            "sampling_settings": self.model_sampling_settings if self.use_model_sampling else "from Sampling",
            "log_confidence": self.confidence,
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

        stats = TokenStats() if self.confidence else None
        with self.lock:
            started = time.perf_counter()
            if sampling.seed is not None:
                torch.manual_seed(sampling.seed)
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs, generation_config=generation_config(sampling, self.use_model_sampling),
                    streamer=LiveText(self.tokenizer, on_text) if on_text else None,
                    logits_processor=LogitsProcessorList([stats]) if stats else None,
                )
            elapsed = time.perf_counter() - started

        generated = output[0, prompt_tokens:].tolist()
        raw = self.tokenizer.decode(generated, skip_special_tokens=True)
        reasoning, answer = split_reasoning(raw)
        meta = {}
        if stats is not None:
            stats.finish(generated)
            meta = confidence_meta(self.tokenizer, generated, raw, stats.logprobs, stats.entropies)
        return Completion(
            text=answer,
            reasoning=reasoning,
            prompt_tokens=prompt_tokens,
            generated_tokens=len(generated),
            elapsed_s=elapsed,
            finish_reason="stop" if generated and generated[-1] in self.eos_ids else "length",
            full_raw_answer=raw,
            meta=meta,
        )
