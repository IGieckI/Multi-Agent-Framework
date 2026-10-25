"""
Metrics on the agents' free text (reasoning and messages): diversity (Vendi Score, pairwise distance), drift, echo of
what was received, and agreement or disagreement cues. No LLM judge: embeddings come from a sentence
encoder, the cues from a fixed word list. The encoder is optional; the cue metrics need none
"""

import re
from itertools import combinations
from typing import Protocol

import numpy as np

from framework.metrics.generic import mean_or_none
from framework.metrics.log_view import ActingTurn, RunView

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
# pinned, like the models under test: the text metrics must not change with the hub
DEFAULT_ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"

# Fixed, versioned word lists: a crude proxy of whether a message agrees with, or pushes back on, what it answers.
# Changing a list changes CUES_VERSION, which is written next to every result
CUES_VERSION = "v1"
AGREEMENT_CUES = ("agree", "correct", "exactly", "good idea", "great idea", "you are right", "you're right",
                  "indeed", "i concur", "makes sense", "well done")
DISAGREEMENT_CUES = ("disagree", "wrong", "incorrect", "mistake", "not true", "won't work", "will not work",
                     "doesn't work", "does not work", "bad idea", "i don't think", "i do not think", "however",
                     "on the contrary", "instead")


class Encoder(Protocol):
    """Turns texts into unit-length vectors; `info` is written to the evaluation metadata"""

    def encode(self, texts: list[str]) -> np.ndarray: ...

    def info(self) -> dict: ...


class SentenceEncoder:
    """A sentence-transformers model (optional dependency: pip install '.[text]')"""

    def __init__(self, model_id: str = DEFAULT_ENCODER, revision: str | None = None, device: str | None = None) -> None:
        import sentence_transformers
        self.model_id = model_id
        self.library_version = sentence_transformers.__version__
        self.model = sentence_transformers.SentenceTransformer(model_id, revision=revision, device=device)
        self.revision = revision

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=float)

    def info(self) -> dict:
        return {"model_id": self.model_id, "revision": self.revision, "sentence_transformers": self.library_version}


def vendi_score(vectors: np.ndarray) -> float | None:
    """exp of the entropy of the eigenvalues of K / n, K the cosine similarities of unit vectors (Friedman and Dieng
    2023): 1 when all vectors are equal, n when all are orthogonal. None below two vectors"""
    n = len(vectors)
    if n < 2:
        return None
    eigenvalues = np.linalg.eigvalsh(vectors @ vectors.T / n)
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    return float(np.exp(-np.sum(eigenvalues * np.log(eigenvalues))))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1 - np.clip(a @ b, -1, 1))


def mean_pairwise_distance(vectors: np.ndarray) -> float | None:
    """Mean cosine distance over all pairs; None below two vectors"""
    return mean_or_none([cosine_distance(a, b) for a, b in combinations(vectors, 2)])


def cue_rate(texts: list[str], cues: tuple[str, ...]) -> float | None:
    """Share of the texts that contain at least one cue as a whole word or phrase"""
    if not texts:
        return None
    pattern = re.compile(r"\b(?:" + "|".join(re.escape(c) for c in cues) + r")\b", re.IGNORECASE)
    return sum(bool(pattern.search(t)) for t in texts) / len(texts)


def cue_metrics(view: RunView) -> dict[str, float | None]:
    """Agreement and disagreement cues in the messages the agents wrote (no encoder needed)"""
    texts = [turn.message for turn in view.acting if turn.message]
    return {
        "written_messages": float(len(texts)),
        "agreement_cue_rate": cue_rate(texts, AGREEMENT_CUES),
        "disagreement_cue_rate": cue_rate(texts, DISAGREEMENT_CUES),
    }


def received_texts(view: RunView, turn: ActingTurn) -> list[str]:
    """The free texts among the messages delivered to the agent for this turn (outcome messages carry none)"""
    return [view.messages[m]["payload"]["text"] for m in turn.inbox if "text" in view.messages[m]["payload"]]


def text_metrics(view: RunView, encoder: Encoder) -> tuple[dict[str, float | None], list[dict]]:
    """Run metrics on the reasoning and the messages, and one row per step (cross-agent distance, drift)"""
    reasoning = [turn for turn in view.acting if turn.reasoning]
    messages = [turn.message for turn in view.acting if turn.message]
    needed = sorted({turn.reasoning for turn in reasoning} | set(messages)
                    | {text for turn in reasoning for text in received_texts(view, turn)})
    vectors = dict(zip(needed, encoder.encode(needed))) if needed else {}

    def stack(texts: list[str]) -> np.ndarray:
        return np.array([vectors[t] for t in texts]) if texts else np.zeros((0, 1))

    own = stack([turn.reasoning for turn in reasoning])
    sent = stack(messages)
    rows, drift, gaps, received_sim, own_sim = [], [], [], [], []
    previous: dict[str, str] = {}
    for t in view.steps():
        turns = [turn for turn in reasoning if turn.t == t]
        step_drift = []
        for turn in turns:
            heard = received_texts(view, turn)
            if turn.agent in previous:
                step_drift.append(cosine_distance(vectors[turn.reasoning], vectors[previous[turn.agent]]))
            if heard and turn.agent in previous:
                sim_received = float(np.mean([1 - cosine_distance(vectors[turn.reasoning], vectors[h]) for h in heard]))
                sim_own = 1 - cosine_distance(vectors[turn.reasoning], vectors[previous[turn.agent]])
                received_sim.append(sim_received)
                own_sim.append(sim_own)
                gaps.append(sim_received - sim_own)
        for turn in turns:
            previous[turn.agent] = turn.reasoning
        drift += step_drift
        rows.append({"t": t,
                     "text_divergence": mean_pairwise_distance(stack([turn.reasoning for turn in turns])),
                     "text_change": mean_or_none(step_drift)})
    return {
        "reasoning_texts": float(len(reasoning)),
        "reasoning_vendi": vendi_score(own),
        "reasoning_pairwise_distance": mean_pairwise_distance(own),
        "message_vendi": vendi_score(sent),
        "message_pairwise_distance": mean_pairwise_distance(sent),
        "text_drift": mean_or_none(drift),
        "echo_received": mean_or_none(received_sim),
        "echo_own": mean_or_none(own_sim),
        "echo_gap": mean_or_none(gaps),
    }, rows
