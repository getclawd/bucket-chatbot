"""Semantic memory.

Lexical search only finds lines that share literal words, so Bucket forgets
things it genuinely knows the moment you rephrase them. This stores a vector per
utterance and searches by meaning instead.

Entirely optional: without an embedding model, or without numpy, retrieval falls
back to the bag-of-words path and nothing else changes.
"""

import json
import struct
import urllib.error
import urllib.request

from .config import config

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is a soft dependency
    np = None


def pack(vector: list[float]) -> bytes:
    return struct.pack(f"<{len(vector)}f", *vector)


def unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 4}f", blob))


class Embedder:
    """Turns text into a vector via Ollama. Returns None whenever it can't."""

    def __init__(self) -> None:
        self.model = config.EMBED_MODEL
        self.available = False
        self.reason = ""
        self.dim = 0
        self.fingerprint = ""
        self._endpoint = "/api/embed"

        if config.EMBED_BACKEND in ("none", "off", ""):
            self.reason = "disabled"
            return
        if np is None:
            self.reason = "numpy not installed (pip install numpy)"
            return

        self._probe()

    def _probe(self) -> None:
        try:
            with urllib.request.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=3) as resp:
                installed = json.load(resp).get("models", [])
        except Exception as exc:  # noqa: BLE001
            self.reason = f"ollama unreachable at {config.OLLAMA_URL}: {exc}"
            return

        # Ollama reports "nomic-embed-text:latest" for a "nomic-embed-text" pull.
        match = next(
            (
                m
                for m in installed
                if m.get("name", "") == self.model
                or m.get("name", "").split(":")[0] == self.model.split(":")[0]
            ),
            None,
        )
        if not match:
            self.reason = f"model {self.model!r} not pulled (ollama pull {self.model})"
            return
        self.model = match.get("name", self.model)

        probe = self.embed("bucket")
        if not probe:
            self.reason = f"{self.model} returned no embedding"
            return
        self.dim = len(probe)
        digest = match.get("digest", "unknown")
        self.fingerprint = f"{config.EMBED_BACKEND}:{self.model}:{digest}:v1"
        self.available = True

    # ------------------------------------------------------------------
    def embed(self, text: str) -> list[float] | None:
        result = self.embed_batch([text])
        return result[0] if result else None

    def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        texts = [t.strip() for t in texts if t and t.strip()]
        if not texts:
            return None
        try:
            if self._endpoint == "/api/embed":
                try:
                    return self._call_embed(texts)
                except urllib.error.HTTPError:
                    # Older Ollama only has the single-item /api/embeddings.
                    self._endpoint = "/api/embeddings"
            return [self._call_embeddings(t) for t in texts]
        except Exception:  # noqa: BLE001 - retrieval must never break the bot
            return None

    def _call_embed(self, texts: list[str]) -> list[list[float]]:
        body = json.dumps({"model": self.model, "input": texts}).encode()
        request = urllib.request.Request(
            f"{config.OLLAMA_URL}{self._endpoint}",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=config.EMBED_TIMEOUT) as resp:
            return json.load(resp)["embeddings"]

    def _call_embeddings(self, text: str) -> list[float]:
        body = json.dumps({"model": self.model, "prompt": text}).encode()
        request = urllib.request.Request(
            f"{config.OLLAMA_URL}/api/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=config.EMBED_TIMEOUT) as resp:
            return json.load(resp)["embedding"]

    def status(self) -> str:
        if self.available:
            return f"{self.model} ({self.dim}d)"
        return f"off — {self.reason}"


class VectorIndex:
    """All utterance vectors held in memory as one normalized matrix.

    Vectors are unit-normalized on the way in, so cosine similarity is a plain
    dot product and the whole search is a single matrix-vector multiply.
    """

    def __init__(self, dim: int):
        self.dim = dim
        self._ids: list[int] = []
        self._matrix = np.zeros((0, dim), dtype=np.float32) if np is not None else None
        self._pending: list[tuple[int, list[float]]] = []

    def __len__(self) -> int:
        return len(self._ids) + len(self._pending)

    def __bool__(self) -> bool:
        # Without this, a freshly built (empty) index is falsy because of __len__,
        # and every `if self.index` guard silently disables semantic memory.
        return True

    @staticmethod
    def _normalize(vector: list[float]):
        array = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(array))
        return array / norm if norm else array

    def add(self, uid: int, vector: list[float]) -> None:
        if np is None or len(vector) != self.dim:
            return
        self._pending.append((uid, vector))
        # Rebuilding the matrix per insert is O(n); batch it instead.
        if len(self._pending) >= 64:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        rows = np.vstack([self._normalize(v) for _, v in self._pending])
        self._matrix = np.vstack([self._matrix, rows]) if len(self._ids) else rows
        self._ids.extend(uid for uid, _ in self._pending)
        self._pending.clear()

    def load(self, rows: list[tuple[int, bytes]]) -> None:
        self._ids = []
        self._pending = []
        vectors = []
        for uid, blob in rows:
            vector = unpack(blob)
            if len(vector) != self.dim:
                continue
            self._ids.append(uid)
            vectors.append(self._normalize(vector))
        self._matrix = (
            np.vstack(vectors) if vectors else np.zeros((0, self.dim), dtype=np.float32)
        )

    def search(self, vector: list[float], limit: int = 12) -> list[tuple[int, float]]:
        if np is None:
            return []
        self._flush()
        if not len(self._ids):
            return []

        query = self._normalize(vector)
        if query.shape[0] != self._matrix.shape[1]:
            return []

        scores = self._matrix @ query
        count = min(limit, scores.shape[0])
        top = np.argpartition(-scores, count - 1)[:count]
        top = top[np.argsort(-scores[top])]
        return [(self._ids[i], float(scores[i])) for i in top]
