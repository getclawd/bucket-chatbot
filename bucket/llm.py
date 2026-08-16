"""Optional polish layer.

The engine decides *what* Bucket says. This layer only decides how it comes out
of its mouth — it re-voices the raw output without adding meaning, knowledge, or
coherence. If it fails, times out, or isn't configured, the raw engine output is
returned unchanged, so the bot never depends on it.

Backends: none (default), anthropic, ollama.
"""

import json
import random
import re
import urllib.error
import urllib.request

from .config import config
from .text import STOPWORDS, tokenize

# Expanded on both sides before comparing, so "i'm" -> "i am" isn't counted as
# the model inventing the word "am".
CONTRACTIONS = {
    "i'm": "i am", "i'll": "i will", "i've": "i have", "i'd": "i would",
    "you're": "you are", "you'll": "you will", "you've": "you have",
    "it's": "it is", "that's": "that is", "there's": "there is",
    "he's": "he is", "she's": "she is", "we're": "we are", "they're": "they are",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "isn't": "is not", "aren't": "are not", "wasn't": "was not",
    "weren't": "were not", "can't": "can not", "won't": "will not",
    "couldn't": "could not", "wouldn't": "would not", "shouldn't": "should not",
    "haven't": "have not", "hasn't": "has not", "hadn't": "had not",
    "let's": "let us", "what's": "what is", "who's": "who is",
}

# Grammatical glue the model may add while rearranging a line. Anything outside
# this and the raw line's own vocabulary counts as invented content.
FUNCTION_WORDS = STOPWORDS | {
    "am", "will", "would", "could", "should", "can", "may", "might", "must",
    "about", "into", "onto", "than", "there", "here", "now", "still", "again",
    "all", "any", "some", "every", "each", "both", "more", "most", "much",
    "many", "other", "another", "such", "own", "same", "up", "down", "out",
    "off", "over", "under", "back", "away", "s", "t", "re", "ve", "ll", "d", "m",
}

SYSTEM = """You are a text cleanup function. You are NOT a chatbot and you are \
NOT talking to anyone.

You receive one raw line of text. You return that same line, tidied. Nothing else.

THE ONE RULE: you may DELETE words and REORDER words. You may NOT INVENT words.
Every word you output must already appear in the raw line, apart from small \
grammatical glue (a, the, is, and, to, of).

Allowed: removing a stuttered "the the", cutting a dangling half-clause, fixing \
word order so it scans, changing "i am" to "i'm".
Forbidden: adding any noun, verb, adjective or adverb that is not in the raw line. \
Answering a question. Explaining the line. Describing the line. Making it sensible.

Example:
  raw: the the bucket is a container for things you did not want to talk about
  good: the bucket is a container for things you did not want to talk about
  BAD:  i think a bucket is a vessel used for holding unwanted items

The BAD one is wrong because "think", "vessel", "used", "holding", "unwanted" and \
"items" were invented. That is the only mistake that matters.

If the raw line is already clean, return it byte for byte.

Keep it lowercase, under 30 words, no trailing punctuation, no quotation marks. \
Output ONLY the line. No preamble, no XML or reasoning tags."""

USER_TEMPLATE = """Someone said to Bucket: {user}

Bucket's raw output ({strategy}): {raw}

Say that as Bucket's line. Smooth out stutters, doubled words and broken grammar \
so it scans as one spoken sentence. Keep every idea in it, including the parts \
that make no sense. Do not answer the person."""


class Polisher:
    """Wraps whichever backend is configured. Always safe to call."""

    def __init__(self) -> None:
        self.backend = config.LLM_BACKEND
        self.model = config.LLM_MODEL
        self.available = False
        self.reason = ""
        self._client = None
        # Visible in /stats — if rejected is most of polished, the model is
        # ignoring the brief and is better turned off.
        self.polished = 0
        self.rejected = 0
        self.last_invented: list[str] = []

        if self.backend in ("", "none", "off"):
            self.reason = "disabled"
            return

        if self.backend == "anthropic":
            self._init_anthropic()
        elif self.backend == "ollama":
            self._init_ollama()
        else:
            self.reason = f"unknown backend {self.backend!r}"

    # ------------------------------------------------------------------
    def _init_anthropic(self) -> None:
        self.model = self.model or "claude-opus-5"
        try:
            import anthropic
        except ImportError:
            self.reason = "anthropic package not installed (pip install anthropic)"
            return
        try:
            # Resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an `ant auth login`
            # profile — do not pass a key explicitly.
            self._client = anthropic.Anthropic(timeout=config.LLM_TIMEOUT)
            self.available = True
        except Exception as exc:  # noqa: BLE001 - never let this break the bot
            self.reason = f"anthropic client init failed: {exc}"

    def _init_ollama(self) -> None:
        try:
            with urllib.request.urlopen(f"{config.OLLAMA_URL}/api/tags", timeout=3) as resp:
                installed = [m.get("name", "") for m in json.load(resp).get("models", [])]
        except Exception as exc:  # noqa: BLE001
            self.reason = f"ollama unreachable at {config.OLLAMA_URL}: {exc}"
            return

        if not installed:
            self.reason = "ollama has no models pulled (try: ollama pull llama3.1:8b)"
            return
        if not self.model:
            # Whatever is already on the machine beats a hardcoded default.
            self.model = installed[0]
        elif self.model not in installed:
            self.reason = f"ollama has no model {self.model!r} (has: {', '.join(installed)})"
            return
        self.available = True

    # ------------------------------------------------------------------
    def polish(self, raw: str, user_text: str = "", strategy: str = "") -> str:
        if not self.available or not raw.strip():
            return raw
        if random.random() > config.LLM_POLISH_RATE:
            return raw

        prompt = USER_TEMPLATE.format(
            user=(user_text or "(nothing)").strip(),
            strategy=strategy or "unknown",
            raw=raw.strip(),
        )
        try:
            if self.backend == "anthropic":
                out = self._call_anthropic(prompt)
            else:
                out = self._call_ollama(prompt)
        except Exception:  # noqa: BLE001 - a dead polish layer must never mute the bot
            return raw

        out = self._clean(out)
        self.polished += 1

        # If the model refused, lectured, or wandered off, keep the raw line.
        if not out or len(out) > 400:
            self.rejected += 1
            return raw

        # The load-bearing check. The engine decides what Bucket says; this layer
        # only decides how it sounds. A model that invents content words has
        # started answering the user instead of re-voicing the line, so bin it.
        if config.LLM_STRICT:
            invented = self.invented_words(raw, out)
            if invented:
                self.rejected += 1
                self.last_invented = invented
                return raw
        return out

    @staticmethod
    def _stem(word: str) -> str:
        """Crude suffix stripper — enough to treat carry/carrying as one word."""
        for suffix in ("ings", "ing", "edly", "ed", "ies", "es", "s", "ly"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                base = word[: -len(suffix)]
                return base + "y" if suffix == "ies" else base
        return word

    @classmethod
    def _vocabulary(cls, text: str) -> set[str]:
        expanded = " ".join(CONTRACTIONS.get(w, w) for w in text.lower().split())
        return {cls._stem(w) for w in tokenize(expanded)}

    @classmethod
    def invented_words(cls, raw: str, out: str) -> list[str]:
        """Content words in `out` that aren't in `raw`. Empty means faithful."""
        source = cls._vocabulary(raw)
        return sorted(
            word
            for word in cls._vocabulary(out)
            if word not in source and word not in FUNCTION_WORDS and len(word) > 1
        )

    @staticmethod
    def _clean(out: str) -> str:
        """Strip reasoning traces, stray tags and preamble down to one line."""
        out = re.sub(r"<think(ing)?>.*?</think(ing)?>", "", out or "",
                     flags=re.DOTALL | re.IGNORECASE)
        out = re.sub(r"<[^>\n]{1,40}>", "", out)

        lines = [line.strip() for line in out.splitlines() if line.strip()]
        if not lines:
            return ""

        # Some models show their working on one line: "before -> after".
        # Without this the arrow and both halves ship as the reply.
        lines = [
            re.split(r"\s*(?:->|-->|→|=>)\s*", line)[-1].strip() if
            re.search(r"\s(?:->|-->|→|=>)\s", line) else line
            for line in lines
        ]
        lines = [line for line in lines if line]
        if not lines:
            return ""
        # "Here's the line:" followed by the actual line.
        first = lines[0]
        if first.endswith(":") and len(lines) > 1:
            first = lines[1]
        return first.strip().strip('"').strip("`").strip()

    # ------------------------------------------------------------------
    def _call_anthropic(self, prompt: str) -> str:
        params = {
            "model": self.model,
            "max_tokens": 300,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            # Thinking off at low effort: this is a one-line rewrite and latency
            # matters more than reasoning depth.
            response = self._client.messages.create(
                **params,
                thinking={"type": "disabled"},
                output_config={"effort": "low"},
            )
        except Exception:  # noqa: BLE001 - older models reject those two params
            response = self._client.messages.create(**params)

        if getattr(response, "stop_reason", None) == "refusal":
            return ""
        return "".join(b.text for b in response.content if b.type == "text")

    def _call_ollama(self, prompt: str) -> str:
        # Use /api/chat, not /api/generate: on reasoning models (qwen3, deepseek-r1,
        # ...) generate ignores `think` and burns the entire token budget on a
        # <think> block, returning an empty response every time.
        body = {
            "model": self.model,
            "stream": False,
            "think": False,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "options": {"temperature": 1.0, "num_predict": 500},
        }
        try:
            return self._ollama_chat(body)
        except urllib.error.HTTPError:
            # Older Ollama builds reject the `think` field outright.
            body.pop("think", None)
            return self._ollama_chat(body)

    def _ollama_chat(self, body: dict) -> str:
        request = urllib.request.Request(
            f"{config.OLLAMA_URL}/api/chat",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=config.LLM_TIMEOUT) as resp:
            payload = json.load(resp)
        return (payload.get("message") or {}).get("content", "")

    # ------------------------------------------------------------------
    def status(self) -> str:
        if not self.available:
            return f"off — {self.reason}"
        detail = f"{self.backend} ({self.model})"
        if self.polished:
            detail += f", {self.rejected}/{self.polished} rejected as invented"
            if self.last_invented:
                detail += f" (last: {', '.join(self.last_invented[:4])})"
        return detail
