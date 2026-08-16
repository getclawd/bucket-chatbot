# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A modern rebuild of Bucket, the 2008 Markov-chain chatbot infamous for absorbing
whatever a chat community taught it. The engine is deliberately dumb — it never
generates a word that wasn't typed by a real user — but the *selection* logic that
picks and recombines what it knows is where all the design effort lives. Read
`README.md` before making changes; it documents the intended behavior in detail,
including measured before/after numbers for past fixes. Treat regressions against
those numbers as bugs.

Runs on Telegram and/or Discord simultaneously via `run_all.py`, sharing one
`Bucket` instance and one SQLite database (WAL mode) so both surfaces see the
same memory in real time.

## Commands

```bash
python console.py             # local REPL, no network — the fastest way to iterate
python run_all.py             # both chat surfaces, one shared Bucket
python telegram_bot.py        # Telegram only
python discord_bot.py         # Discord only (needs `pip install discord.py`)
```

Run the full test suite after any change to `bucket/`:

```bash
python smoke_test.py && python memory_test.py && python shared_memory_test.py && \
python polish_test.py && python chatter_test.py && python feedback_test.py && \
python factoid_test.py && python privacy_test.py && \
python voice_test.py && python recall_test.py && python lexical_test.py && \
python repeat_test.py && python blocklist_test.py && python inventory_test.py
```

`shared_memory_test.py` takes several minutes — it seeds a fresh corpus and
embeds every line through Ollama one call at a time. It isn't hung. Run it with
`python -u` if you want to watch progress; piped output is block-buffered
otherwise and prints nothing until it finishes.

There is no test framework — each `*_test.py` is a standalone script that exits
non-zero on failure and prints a diagnostic summary. Run one directly to debug a
specific area (e.g. `python voice_test.py` after touching `brain.py`). No
lint/format tooling is configured.

One-off maintenance scripts (all support a dry-run default and `--apply` to
commit, and back up the `.sqlite3` file before writing):

```bash
python clean_corpus.py --apply       # strip pasted logs / generation-metadata dumps
python rebuild_phrases.py --apply    # re-mine phrases + re-run fact extraction
python prune_seed.py --apply         # remove the starter corpus once real data exists
python rebuild_factoids.py --apply   # re-run factoid extraction only
python scrub_legacy.py --apply       # purge blocked names from generation
```

## Architecture

### Data flow

```
telegram_bot.py ─┐
discord_bot.py  ─┴──> bucket.core.Bucket ──> bucket.learn.Learner ──> bucket.db.BucketDB (SQLite)
                            │
                            v
                   bucket.brain.Brain ──> reply text
                            │
                            v
                   bucket.llm.Polisher (optional) ──> final reply ──> re-ingested (unreinforced)
```

`bucket/core.py` (`Bucket` class) is the only entry point every surface talks to.
It holds an `RLock` so `run_all.py` can drive Telegram (its own thread) and
Discord (asyncio, via `asyncio.to_thread`) against the same instance safely —
see `shared_memory_test.py` for the concurrency contract.

### Ingestion (`bucket/learn.py`, `bucket/db.py`)

Every message is shredded into five SQLite tables on the way in:
`utterances` (verbatim + repeat count), `pairs` (prompt→reply adjacency),
`chain` (order-2 Markov with order-1 backoff), `factoids` (subject/verb/object,
extracted via a large regex verb list — see `FACTOID_VERBS` in `learn.py`), and
`inventory` (items handed to the bot). A `phrases` table separately tracks
recurring 2–4 word n-grams for the `phrase` reply strategy.

All persisted memory is content-only. `utterances`, `factoids`, and `inventory`
do not contain author, username, chat/channel, giver, or alias fields. Surface
adapters may use an opaque in-memory conversation key for context and pacing,
but it is never written to SQLite or returned by the web UI. Semantic vectors
represent message/fact content only.

**Critical invariant: Bucket's own replies are never learned from.**
`Learner.ingest(..., reinforce=False)` stores the reply (so it's still
searchable/recallable) but skips the chain, pairs, factoids, and inventory
tables. This exists because self-reinforcement previously created runaway
feedback loops — see `feedback_test.py`. If you add a new ingestion side-effect
to `learn.py`, gate it behind `reinforce` too.

Trailing chat filler (`remember that`, `lol`, `bro`, etc.), markup litter
(`>`, `\`, `**`, unmatched brackets), and emphatic verb forms (`does have` →
`has`) are normalized during factoid extraction (`strip_filler`, `tidy` in
`learn.py`) — don't re-introduce raw text into the `factoids` table without
running it through these.

Long pastes and generation-parameter dumps (Stable Diffusion metadata, stack
traces) are rejected outright by `is_machine_output()` before they enter the
corpus at all (`BUCKET_MAX_LEARN_WORDS`).

### Blocked names (`bucket/blocklist.py`)

Handles from the imported 2008 transcripts must never be said: those people
didn't agree to be quoted by a bot in another chat eighteen years later. The
*content* of those logs is wanted and stays — only the names go. Don't
"simplify" this by excluding legacy lines from learning wholesale; that was
considered and rejected, because it takes the 2008 flavour with it.

Enforced at **three** points, and all three are load-bearing:

- **learning** (`Learner.ingest`) — a line containing a blocked name is stored
  but not learned from, so composition can't emit it
- **replying** (`Brain.respond`, `_safe_last_resort`) — a candidate naming
  someone is rejected outright, with no fallback to it, since names already in
  the corpus predate the learning guard
- **display** (`Bucket.redact`) — `/literal` and `/recall` print corpus text
  verbatim, so they redact; the queried subject is redacted too, or the command
  echoes the name straight back

The reply guard is not redundant with the learning guard: older corpus rows may
already contain a blocked name, and a present-day message can repeat one as
content. Matching is therefore on the text itself. `scrub_legacy.py --apply`
cleans the backlog (it removes only name-bearing chain edges, phrases and
factoids — it does not de-reinforce legacy lines). `blocklist_test.py` covers
all three points.

`DEFAULT_BLOCKED` ships empty on purpose: the handles that matter are a fact
about one deployment's corpus, and hardcoding them would publish the exact list
the feature exists to keep unpublished. Names come from `BUCKET_BLOCKED_NAMES`,
unioned in at every use site. Don't reintroduce real handles here or in the
tests — `blocklist_test.py` uses synthetic ones and sets the env var itself.

Matching is whole-token and case-insensitive — `pengu` blocks but `penguin`
doesn't. Never make it substring-based; it would silently eat innocent words.

### Privacy boundary (`bucket/db.py`)

The database intentionally has no identity or attribution model. The privacy
migration rebuilds legacy tables without author, chat, giver, or alias columns,
then drops the old alias table. New ingestion APIs accept content only. Names
typed inside content remain content; use `BUCKET_BLOCKED_NAMES` when a
deployment needs to suppress selected names from generation and display.

### Reply selection (`bucket/brain.py`)

`Brain.respond()` runs a weighted-random ordering of ~11 strategies
(`_plan()`) until one returns a non-empty `Reply`, skipping any reply that
repeats something recently said. If everything repeats, it retries a few random
escapes before falling back to the input-adjacent reply.

Every candidate also goes through `trim_dangling()` (`text.py`), which drops
trailing words that can't end a sentence — 10% of replies used to end on one
("...which means dario beats his"), reading as truncated. It only removes words,
never adds them, same rule as the polish layer. `DANGLING_ENDINGS` deliberately
excludes "that"/"it"/"you"/"are": they look like function words but end valid
sentences. Don't add them. Note this is *not* truncation — `Bucket._cap` at
`BUCKET_MAX_REPLY_WORDS` measured 0/360 firing, so a cut-off-looking reply is
almost always a dangling ending, not the cap.

Two things about that check are easy to break. It compares against the
*original* text, not the name-stripped version (see the comment in `respond()`).
And it checks a **window**, not just the current message: `respond(avoid=...)`
receives the last `BUCKET_NO_REPEAT_WINDOW` lines from the chat, Bucket's own
replies included, via `Bucket.recently_said()`. Comparing against the current
message alone was a real bug — Bucket said "so true queen" two turns after Elle
did, and the guard passed it because the message it was actually replying to
("i didnt") shared no words with it. `repeat_test.py` covers this, including
that the suppression *expires*: reusing the group's catchphrases is what the
`phrase` and `obsession` strategies exist for, so the window has to be short.

`Bucket._said` (repeat suppression) is deliberately separate from
`Bucket._recent` (which feeds `context_for()` and thus recall). Bucket's own
replies belong in the first and must stay out of the second — letting its output
steer what it retrieves next is the feedback loop the `reinforce=False` design
exists to prevent.

Strategies split into **composing** (`mashup`, `tangent`, `phrase`,
`collision` — weld two things it knows together) and
**single-source** (`pair`, `factoid`, `markov`, `echo`, `inventory`,
`obsession` — return one retrieved/generated line). The composing strategies
carry most of the selection weight deliberately: the target voice is "on-topic
material recombined into the wrong shape, stated with total confidence," not
random noise and not verbatim repetition. `voice_test.py` measures the
composed-vs-single ratio and the parrot/off-topic rates against a fixed prompt
set — treat a regression there as a real regression, not test flakiness.

`_candidates()` blends lexical (`db.lexical_search`) and semantic (embedding
cosine similarity) recall, each normalized to 0–1 before merging
(`BUCKET_SEMANTIC_WEIGHT`), then overfetches `BUCKET_RECALL_OVERFETCH`× and
reranks (`_rerank`) — reranking can only reorder what retrieval already found,
so the overfetch is what makes the stage worth having.

There are two lexical paths. `db.fts_search()` uses SQLite's own FTS5/BM25 index
with porter stemming; `db.search()` is the older hand-rolled sqrt-IDF scan over
the `word_index`/`words` tables. `lexical_search()` picks between them on
`BUCKET_FTS` and on whether this SQLite build actually has FTS5. FTS5 measured
better — `lexical_test.py` puts mean precision@3 at 0.89 vs 0.67 — and the win
is mostly *stemming*, not BM25: without it "does it rain a lot" scores zero
against a corpus that talks about nothing else. Treat a drop below 0.89 as a
regression. The IDF path is kept because FTS5 is a compile-time SQLite option
and can genuinely be absent; that's why `add_utterance` still maintains
`word_index`/`words` alongside the FTS index.

Two FTS5 traps are handled and worth not re-introducing. It AND-joins bare
terms, so a whole sentence as a MATCH expression matches almost nothing —
`_fts_query()` drops stopwords, phrase-quotes each term (which also neutralizes
`*`/`^`/`-`/`:` as syntax) and OR-joins them. And it's an external-content table
over `utterances`, so a plain `DELETE` from `utterances` leaves the index
pointing at ids that no longer resolve: call `db.rebuild_fts()` after any direct
delete (`clean_corpus.py`, `prune_seed.py`, `db.wipe()` all do). `FTS_VERSION` in
`db.py` must be bumped whenever `FTS_SCHEMA` changes the tokenizer, since
`CREATE TABLE IF NOT EXISTS` won't rebuild an existing index and the row counts
still match. Facts have their *own* embedding index
(`fact_index` in `core.py`, separate from the utterance `index`) — without it,
factoids are only reachable by substring-matching the subject string, so most
of what Bucket "knows" is unreachable by paraphrase. When recall needs to
resolve a pronoun ("what about him"), `Bucket.context_for()` folds the last 1–2
turns into the *semantic* half of the query only — lexical search always stays
on the literal current message, on purpose.

### Polish layer (`bucket/llm.py`)

Optional LLM rewrite pass (Ollama or Anthropic) that may only delete/reorder
words, never add them — enforced by `Polisher.invented_words()` diffing the
polished output's content-word vocabulary against the raw engine output
(after normalizing contractions/inflections), not by prompting alone.
`BUCKET_LLM_STRICT=0` disables this guard; leaving it on is the difference
between "Bucket with better grammar" and "a generic LLM chatbot." Ollama calls
use `/api/chat` with `think: false`, not `/api/generate` — reasoning models
burn their whole budget on the `<think>` block and return empty content
otherwise.

### Multi-surface wiring

`telegram_bot.py` and `discord_bot.py` are thin adapters: they translate
platform events into `Bucket.handle()`/`Bucket.speak()` calls and know nothing
about the engine internals. Both use a namespaced `chat` key
(`f"tg:{chat_id}"` / `f"dc:{channel_id}"`) so conversation state never
collides across platforms; those keys are in-memory only and never persisted.
`bucket/lock.py` provides a cross-process advisory
file lock (`<db>.lock`) — required because Telegram's `getUpdates` 409s if two
processes poll the same bot token simultaneously; `run_all.py` and the
standalone entry points all take this lock before starting.


## Configuration

All runtime config is read once into `bucket/config.py` module-level constants
from `.env` (see `.env.example` for the full annotated list) plus real
environment variables. There's no config object to instantiate — other modules
`from .config import config` and read attributes directly. Some values (e.g.
chattiness, seeded flag) can be overridden at runtime and persisted to the
`meta` SQLite table, which takes precedence over `.env` on next startup.
