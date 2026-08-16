"""Configuration. Reads a .env file next to the project root, then the environment."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        # Real environment variables win over the file.
        os.environ.setdefault(key, value)


_load_dotenv()


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _get_int(key: str, default: int) -> int:
    try:
        return int(_get(key) or default)
    except ValueError:
        return default


def _get_float(key: str, default: float) -> float:
    try:
        return float(_get(key) or default)
    except ValueError:
        return default


class Config:
    # --- identity ---
    NAME = _get("BUCKET_NAME", "bucket").lower()
    DB_PATH = _get("BUCKET_DB", str(ROOT / "bucket.sqlite3"))

    # --- behavior ---
    # Probability of butting into a conversation it wasn't addressed in.
    CHATTINESS = _get_float("BUCKET_CHATTINESS", 0.18)
    # Seconds it must stay quiet after an uninvited reply, so a fast-moving
    # group doesn't turn into Bucket talking to itself.
    MIN_GAP = _get_float("BUCKET_MIN_GAP", 25.0)
    # Just after it speaks, a conversation is usually happening — being likelier
    # to chime in during that window reads as banter rather than randomness.
    FOLLOWUP_WINDOW = _get_float("BUCKET_FOLLOWUP_WINDOW", 150.0)
    FOLLOWUP_BOOST = _get_float("BUCKET_FOLLOWUP_BOOST", 2.5)
    # How many items it can carry before it starts dropping things.
    INVENTORY_SIZE = _get_int("BUCKET_INVENTORY_SIZE", 12)
    # A line repeated this many times becomes an obsession it can't shake.
    OBSESSION_THRESHOLD = _get_int("BUCKET_OBSESSION_THRESHOLD", 3)
    # How many recent lines (from anyone, Bucket included) it won't repeat.
    # Small on purpose: reusing the group's catchphrases is the whole point of
    # the phrase/obsession strategies, but doing it twice in one exchange reads
    # as a stuck bot rather than an obsessed one.
    NO_REPEAT_WINDOW = _get_int("BUCKET_NO_REPEAT_WINDOW", 6)
    # Learn from everything, including things said to it directly.
    LEARN = _get("BUCKET_LEARN", "1") not in ("0", "false", "no")
    # Lines longer than this aren't conversation — pasted logs, generation
    # metadata, copypasta. Learning them verbatim means reciting them verbatim.
    MAX_LEARN_WORDS = _get_int("BUCKET_MAX_LEARN_WORDS", 55)
    # Nothing it says should run longer than this.
    MAX_REPLY_WORDS = _get_int("BUCKET_MAX_REPLY_WORDS", 40)
    # Load the starter corpus on a fresh database. Turn off once real people
    # have taught it things — the seed is scaffolding, not personality.
    SEED_ENABLED = _get("BUCKET_SEED", "1") not in ("0", "false", "no")

    # --- llm polish ---
    # none | anthropic | ollama
    LLM_BACKEND = _get("BUCKET_LLM_BACKEND", "none").lower()
    LLM_MODEL = _get("BUCKET_LLM_MODEL", "")
    # Local reasoning models can take 10s+ per line; be generous.
    LLM_TIMEOUT = _get_float("BUCKET_LLM_TIMEOUT", 45.0)
    # Fraction of replies that get run through the polish layer.
    LLM_POLISH_RATE = _get_float("BUCKET_LLM_POLISH_RATE", 1.0)
    # Discard any polished line that introduces words the engine never said.
    # This is what keeps the LLM a voice instead of a generic generator. Off at
    # your peril.
    LLM_STRICT = _get("BUCKET_LLM_STRICT", "1") not in ("0", "false", "no")
    OLLAMA_URL = _get("BUCKET_OLLAMA_URL", "http://localhost:11434")

    # --- semantic memory ---
    # ollama | none. Needs numpy and an embedding model pulled.
    EMBED_BACKEND = _get("BUCKET_EMBED_BACKEND", "ollama").lower()
    EMBED_MODEL = _get("BUCKET_EMBED_MODEL", "nomic-embed-text")
    EMBED_TIMEOUT = _get_float("BUCKET_EMBED_TIMEOUT", 30.0)
    # How much recall leans on meaning vs. literal shared words (0 to 1).
    SEMANTIC_WEIGHT = _get_float("BUCKET_SEMANTIC_WEIGHT", 0.65)

    # --- names it must never say ---
    # The only place blocked handles are configured: DEFAULT_BLOCKED in
    # bucket/blocklist.py ships empty on purpose.
    # Comma or space separated. Matched as whole words, case-insensitively.
    BLOCKED_NAMES = frozenset(
        n for n in _get("BUCKET_BLOCKED_NAMES", "").replace(",", " ").lower().split() if n
    )

    # --- lexical recall ---
    # Use SQLite's own FTS5/BM25 index instead of the hand-rolled sqrt-IDF
    # scan over word_index/words. Set to 0 to fall back to the old path — kept
    # switchable because retrieval quality is measured (recall_test.py), not
    # assumed. Silently ignored if this SQLite build has no FTS5.
    FTS = _get("BUCKET_FTS", "1") not in ("0", "false", "no")
    # Recall half-life in days: a line twice this old scores a quarter as much.
    # 0 disables decay entirely, which is what you want for a corpus that's
    # meant to be timeless rather than a running conversation.
    RECALL_HALF_LIFE = _get_float("BUCKET_RECALL_HALF_LIFE", 0.0)
    # Candidates pulled per source before reranking, as a multiple of the
    # final limit. Reranking can only reorder what retrieval handed it.
    RECALL_OVERFETCH = _get_int("BUCKET_RECALL_OVERFETCH", 3)

    # --- telegram ---
    TELEGRAM_TOKEN = _get("BUCKET_TELEGRAM_TOKEN", "")
    ADMIN_IDS = frozenset(
        int(x) for x in _get("BUCKET_ADMIN_IDS", "").replace(",", " ").split() if x.isdigit()
    )

    # --- discord ---
    DISCORD_TOKEN = _get("BUCKET_DISCORD_TOKEN", "")
    # Discord user IDs are a different namespace from Telegram's.
    DISCORD_ADMIN_IDS = frozenset(
        int(x)
        for x in _get("BUCKET_DISCORD_ADMIN_IDS", "").replace(",", " ").split()
        if x.isdigit()
    )


config = Config()
