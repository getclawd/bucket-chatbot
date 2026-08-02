# Bucket

A modern rebuild of Bucket, the 2008 learning chatbot that got famous for being incoherent after /b/ found it
and taught it garbage.

This keeps what made the original interesting — it genuinely learns from whoever
talks to it, with no filter — and rebuilds the machinery so it holds up: SQLite
instead of flat files, a real n-gram model, semantic recall via embeddings, and
an optional LLM layer that smooths delivery without ever inventing content.

Runs on Telegram and Discord at the same time, out of one shared memory.

---

## How it works

```
   telegram ─┐                                    ┌─ every message ─────────────┐
             ├──> one shared Bucket ──> learn ────┤                             │
   discord  ─┘                                    ├──> utterances  (+ a vector) │
                                                  ├──> pairs       (what someone replied)
                                                  ├──> n-gram chain (order 2, order 1 backoff)
                                                  ├──> factoids    ("X is Y")
                                                  └──> inventory   (directional item state)
                                                                    │
                                            ┌───────────────────────┘
                                            v
                                  brain: pick a strategy, weighted random
                                            │
                                            v
                                  polish layer (optional) ── re-voices it, adds nothing
                                            │
                                            v
                                         reply ──> fed back into the corpus
```

**The brain never invents words.** Every token it emits came from something a
user typed. It picks between seven strategies each turn:

| strategy | what it does |
|---|---|
| `mashup` | welds two lines **retrieved for your topic** together at a random seam |
| `tangent` | states a fact, then wanders off the end of it into markov |
| `phrase` | builds a line around chunks your group actually repeats |
| `collision` | two unrelated facts, joined as though one explains the other |
| `attribution` | blames a person: *"USER says wet puh is GOOD"* |
| `pair` | finds a past message like yours, says what someone said back |
| `factoid` | you mentioned something it has a "X is Y" for |
| `markov` | generates a new sentence, seeded from a word you used |
| `echo` | repeats something it heard once, verbatim, regardless of relevance |
| `inventory` | carries, activates, or discards something it was given |
| `obsession` | a line repeated at it enough times becomes a tic it can't drop |

The top five **compose** — they collide two things it knows rather than
returning one retrieved line. That distinction is the whole voice. The 2008
original sounded unhinged because it used *on-topic* material in the *wrong
shape*, with total confidence:

```
you:    tell me about claude
bucket: claude has really improved lately, even tho Dario beats him
        [collision]

you:    Jay has wet puh
bucket: according to USER, wet puh is GOOD
        [attribution]
```

Two failure modes sit either side of that. **Noise** is material unrelated to
what you said — which is what a mashup of two *random* lines produces. **Parroting**
is handing your own sentence back, which reads as broken rather than insane.

Parroting is checked against **what was actually typed**, not the message after
the bot's name is stripped off it. Comparing against the stripped version means
`put candy in the bucket` loses its last word before the check runs, and the
echo sails through at 50% overlap. If every strategy parrots, it reaches for
anything else — several random lines, then a stock phrase — before it will
repeat you.

**Replies never end on a word that can't end a sentence.** Composition and markov
both like to stop on one, which reads as the message being cut off:

```
i'm going to beat bucket like he beats his
                                       ^^^
```

`trim_dangling()` drops those trailing words — it can only ever *remove* text,
same rule as the polish layer. This took dangling endings from **10% of replies
(46/420) to 0%**. The list is deliberately strict: "that", "it", "you" and "are"
all look like function words but end perfectly good sentences ("claude never told
me that"), so they're left alone. Leaving a slightly odd ending is better than
amputating a valid one. A line that's *mostly* dangling words is rejected instead
of whittled down.

It's also checked against **the last few lines, not just the current one**
(`BUCKET_NO_REPEAT_WINDOW`, default 6, Bucket's own replies included). Checking
only the current message misses the obvious case:

```
Elle:    so true queen
Elle:    i didnt
Bucket: so true queen        <- passed the guard: shares no words with "i didnt"
```

The window is short on purpose. Bringing the group's catchphrases back around is
what the `phrase` and `obsession` strategies are for — only doing it twice inside
one exchange reads as a stuck bot. Measured over 5 runs of `voice_test.py`,
suppression costs nothing (66% composed with it on vs 65% off) and *narrows* the
spread, since blocking the easy repeat forces a wider strategy mix.

`voice_test.py` measures both and prints samples:

```bash
python voice_test.py
```

### Phrases

Every 2–4 word run is counted, deduplicated per line so one spammed message
can't dominate, and single-word repeats are discarded. A chunk becomes a phrase
once the group has said it more than once — so these are literally your verbal
tics coming back at you:

```
x8  nuclear codes      x6  dario beats       x4  wet puh
x5  to stab            x5  bucket is         x4  the knife
```

Facts are cleaned on the way in too: trailing filler is stripped and emphatic
verbs collapse, so `Jay does have wet puh remember that` stores as
`Jay | has | wet puh` rather than keeping `remember that` in the object.

```bash
python rebuild_phrases.py --apply    # mine phrases from an existing corpus
```

Its own replies go back into the corpus, chained to whatever prompted them.
That feedback loop is why a phrase repeated a few times snowballs into an
obsession — the same dynamic that turned the original into a machine that mostly
said *all work and no play makes jack a dull boy*.

---

## Names it won't say

Feed Bucket a corpus from a real chat community and the flavour is the point —
that stays. What shouldn't stay is the **handles**: people who never agreed to be
quoted by a bot in a different chat, whose names Bucket happily reassembles into
new sentences:

```
and someone finds oldname_7 and tells him we won't let you
```

`BUCKET_BLOCKED_NAMES` is the list, set per deployment — it ships empty, because
which handles matter is a fact about *your* corpus. A blocked name is enforced in
three places, because any one alone leaks:

| | what it stops |
|---|---|
| learning | the name never enters the chain, phrases or factoids |
| replying | a candidate reply containing one is rejected outright |
| display | `/literal` and `/recall` redact instead of printing |

The reply guard isn't redundant. A name gets into the live chain because Bucket
said it and **a real user asked "who is oldname_7"** — the name launders itself
through a legitimate present-day speaker, where filtering by author or by chat
sees nothing wrong.

`python scrub_legacy.py --apply` purges the backlog. On a live corpus that was
17 chain edges, 14 phrases and 1 factoid — and 0 of 300 generated replies named
anyone afterwards, with all 8 phrases built on the corpus's stock vocabulary
still intact. It deliberately does **not** stop imported lines feeding
generation; that would remove the content along with the names.

Separately, `webapp`, `seed` and `console` are author tags, not people, so they
can't be credited for a fact. Bucket used to say "webapp says ana has wet puh".

## Memory

Recall is a blend of two things: literal **word overlap**, scored with SQLite's
own FTS5/BM25 index, and **meaning**, via an embedding per utterance and cosine
similarity. `BUCKET_SEMANTIC_WEIGHT` sets the mix, default 0.65 toward meaning.

**The word-based half stems.** It used to be a hand-rolled IDF scan matching raw
tokens, which meant a query could miss the only lines in the corpus that were
about it:

```
'does it rain a lot'    idf: 0 hits           bm25: it rains here constantly
```

Measured over a fixed query set (`python lexical_test.py`), mean precision@3
went **0.67 → 0.89**. Most of that is the stemming, not BM25's ranking. The old
path is still there behind `BUCKET_FTS=0`, because FTS5 is a compile-time SQLite
option and isn't guaranteed to be present.

**Facts are embedded too.** Without that they're findable only by
substring-matching the subject against your message, so unless you type a name
literally, everything Bucket knows stays invisible:

```
'who is violent'        subject-match: miss   semantic: dario is an abuser
'who can be relied on'  subject-match: miss   semantic: claude can be trusted
```

**Recent turns widen recall when a message can't stand alone.** A line that is
short or leans on a pronoun has nothing to match on its own — `why does he do
that` retrieves the word "why". The previous turn or two is folded into the
*semantic* half of the query only (lexical stays literal, so it never matches
words nobody just said), and only when needed:

```
'why does he do that'   without context -> jay has wet puh
                        with context    -> dario beats claude
```

A self-contained message gets no context, so what you actually said stays the
subject of the reply.

That second half is what lets Bucket answer something phrased nothing like how it
learned it. On the bundled test, against a corpus it was taught in plain
statements and then queried with paraphrases sharing **zero** content words:

| query | word matching | with embeddings |
|---|---|---|
| what do you like to eat | miss | top-1 |
| what kind of vehicle do you own | miss | top-1 |
| did you climb anything recently | miss | top-1 |
| are you picking up a musical instrument | miss | top-3 |
| **overall (top-3 recall)** | **0/5** | **4/5** |

```bash
python memory_test.py
```

Needs `pip install numpy` and `ollama pull nomic-embed-text` (274MB). Without
either, it degrades to word matching and nothing else changes. Vectors are stored
in the same SQLite file and backfilled automatically on startup, so turning this
on later retrofits everything Bucket already knows.

`/recall <text>` shows you exactly what comes back and how strongly — the fastest
way to see why it said something.

---

## Setup

Needs Python 3.10+. The engine, both chat clients and the Ollama backends are
pure standard library; `numpy` and `discord.py` are only needed for the features
that use them.

```bash
python console.py
```

A local REPL — talk to it, teach it, inspect it, no network involved. The best
way to train it before putting it in front of people.

### Telegram

1. Message [@BotFather](https://t.me/BotFather), send `/newbot`, follow the prompts.
2. Copy `.env.example` to `.env`, paste the token into `BUCKET_TELEGRAM_TOKEN`.
3. **Send BotFather `/setprivacy` → Disable.** Without this Telegram only shows
   the bot messages that name it, so it barely learns anything in groups.
4. Put your numeric Telegram user ID in `BUCKET_ADMIN_IDS`.

```bash
python telegram_bot.py
```

### Discord

```bash
pip install discord.py
```

1. [discord.com/developers/applications](https://discord.com/developers/applications) → New Application.
2. Bot → Reset Token → paste into `BUCKET_DISCORD_TOKEN`.
3. **Bot → Privileged Gateway Intents → enable MESSAGE CONTENT INTENT.** Without
   it Discord delivers empty message bodies and Bucket learns nothing at all.
4. OAuth2 → URL Generator → scope `bot`, permissions *Send Messages* +
   *Read Message History* → open the URL to invite it.
5. Put your Discord user ID in `BUCKET_DISCORD_ADMIN_IDS` (enable Developer Mode,
   right-click yourself → Copy ID). It's a different namespace from Telegram's.

```bash
python discord_bot.py
```

### Both at once, one shared memory

```bash
python run_all.py
```

**Use this rather than running the two scripts side by side.** Two processes
pointed at the same database would each hold their own copy of the vector index
in RAM, so neither could semantically recall anything the other learned until a
restart — they'd agree on stored facts and disagree on what they can remember.
`run_all.py` runs one `Bucket`, one index, both clients: something taught on
Discord is usable on Telegram in the next breath. Access is serialized by a lock,
verified under concurrent load by `shared_memory_test.py`.

### Only one Bucket at a time

Telegram permits exactly one poller per bot token. Start a second Bucket and
Telegram terminates whichever asked last — both instances then spin forever on:

```
[poll error] getUpdates -> HTTP 409: Conflict: terminated by other getUpdates request
```

Startup now takes an OS lock on `<database>.lock`, so a second instance refuses
to start and tells you which PID holds it. The lock is an advisory file lock, not
a PID file, so it releases automatically if the process is killed — there is no
stale lockfile to clean up. If a 409 still appears (something outside this
project polling the same token), the poller says so once and waits rather than
flooding the log.

To find a stray instance:

```powershell
Get-CimInstance Win32_Process -Filter "Name like '%python%'" | Select-Object ProcessId, CommandLine
Stop-Process -Id <pid>
```

### The Mini App

A web UI for browsing what Bucket knows, opened from the bot's menu button
inside Telegram. Four tabs: **memory** (every fact, filterable), **people**
(per-person pages with aliases, facts and their most-repeated lines), **recall**
(live search showing semantic vs lexical scores side by side), and **talk** —
a chat view that prints which strategy fired, the raw engine output before
polish, and what recall dug up. That last one is the `/why` debugger:

```
you: what do you know about god
bucket: gods?
        pair
        raw: gods?
        recalled:
          semantic 0.767 — who is god
          semantic 0.698 — i don't know what you mean by statements about god
```

Telegram will only load a **public HTTPS** address — not `localhost`, not a
self-signed certificate. Bucket otherwise makes only outbound connections, so
this is the one part that needs a tunnel:

```bash
winget install --id Cloudflare.cloudflared
```

That's the whole setup. Set `BUCKET_WEBAPP=1` and `run_all.py` starts the web
server, starts cloudflared, reads the URL it reports, and registers the menu
button with it — in that order, on every launch:

```
mini app: http://127.0.0.1:8770
starting tunnel... https://boring-listing-materials-walls.trycloudflare.com
telegram: mini app -> https://boring-listing-materials-walls.trycloudflare.com
```

**Leave `BUCKET_WEBAPP_URL` empty.** A free quick-tunnel address is ephemeral —
it dies with cloudflared and comes back *different*, so any URL pinned in `.env`
is stale the moment the tunnel restarts, and the button Telegram stored points
at a dead host. Letting Bucket own the tunnel makes that impossible. Set
`BUCKET_TUNNEL=0` and fill in `BUCKET_WEBAPP_URL` only if you have a permanent
address, e.g. a named Cloudflare tunnel on your own domain.

> **The menu button only sticks per chat.** `setChatMenuButton` without a
> `chat_id` returns `ok: true` and is then silently ignored. Bucket sets it for
> each admin at startup and for everyone else the first time they DM. Menu
> buttons don't exist in group chats at all — in a group, `/app` posts a link to
> open the bot in a DM instead.

**Access control.** Telegram signs every request with `initData`, which the
server verifies against your bot token — there's no login, and the API can't be
called from outside Telegram. Only `BUCKET_ADMIN_IDS` get in by default, since
the corpus is every message from your group; `BUCKET_WEBAPP_ALLOW_ALL=1` opens it
to anyone who can open the bot.

> `BUCKET_WEBAPP_DEV=1` disables signature checking so you can open the app in a
> normal browser. Never leave it on while a tunnel is running — it removes all
> access control from a publicly reachable port.

```bash
python webapp_test.py
```

### Commands

Telegram uses `/`, Discord uses `!`. Same everywhere otherwise.

| command | who |
|---|---|
| `stats` | anyone — what it knows |
| `about <person>` | anyone — their facts, their aliases, and what they say most |
| `who <thing>` | anyone — reverse lookup, e.g. `who is dead` |
| `alias <a> <b>` | anyone — tell it two names are the same person |
| `literal <thing>` | anyone — exactly what it believes about something, unmangled |
| `recall <text>` | anyone — what its memory digs up for that, and how strongly |
| `inventory` | anyone — what it's carrying |
| `say <text>` | anyone — force a reply |
| `forget <thing>` | admin |
| `chattiness <0-1>` | admin — how often it butts in uninvited |
| `wipe confirm` | admin — destroy everything |

On Telegram the command menu is registered automatically at startup, so `/`
autocompletes. Admin commands are published only into the admin's own private
chat — everyone else's menu stays clean and nobody is tempted by `/wipe`. If the
menu looks stale, that's Telegram's client cache; it refreshes on app restart.

### How much it talks

Addressed directly — by name, @mention, reply, or DM — it **always** answers, with
no rate limiting. Uninvited, it follows a rhythm rather than a flat coin flip:

| knob | default | what it does |
|---|---|---|
| `BUCKET_CHATTINESS` | `0.18` | base chance of butting in per message |
| `BUCKET_MIN_GAP` | `25`s | hard quiet period after speaking uninvited |
| `BUCKET_FOLLOWUP_WINDOW` | `150`s | how long the "conversation is live" boost lasts |
| `BUCKET_FOLLOWUP_BOOST` | `2.5` | chance multiplier inside that window |

The gap and the boost matter more than the raw percentage. A flat probability
makes it answer five messages in a row in a busy channel and then vanish for an
hour; the floor stops the flooding and the boost keeps it in a conversation it
already joined. Measured on a simulated busy group (a message every 8s):

```
spoke 79 times out of 400  (19.8% of messages)
shortest gap between replies: 32s (floor is 25s)
addressed directly, 5 times in a row: replied 5/5
```

Rough feel: `0.05` wallflower, `0.18` sociable, `0.35` a lot, `0.6+` exhausting.
`/chattiness 0.3` changes it live and **survives restarts** (stored in the
database, overriding `.env`).

```bash
python chatter_test.py
```

Teach it: `a rock is a small quiet friend`
Hand it things: `alice gives bucket a rock`
It recognizes directional role-play too: `bucket holds the rock` activates it,
and `bucket gives alice the rock` removes it. When Bucket randomly says one of
its established action phrases — `*puts down the rock*`, `*offers you the
rock*`, or `*takes the rock*` — that item is removed from inventory.

---

## The polish layer

Optional, off by default. `BUCKET_LLM_BACKEND=none | anthropic | ollama`.

The engine decides **what** Bucket says. The polish layer only decides how it
comes out — it fixes stutters, doubled words and broken grammar so a line scans
as one spoken sentence, while keeping every idea in it including the nonsense.
It is explicitly instructed not to answer you, add facts, or make anything make
sense. On a live test it turned:

```
i am carrying a lot of things right now cheese is the best food in in the world
  -> i carry lots now cheese is best food in world
```

and when asked "what is the capital of france" with raw output `small glass owl`,
it correctly still said `small glass owl`.

It is optional in every sense — if it's disabled, unreachable, times out, refuses,
or returns junk, the raw engine output is used unchanged. The bot never depends
on it.

### The invention guard

A prompt alone does not hold a small model to "add nothing". Left to itself,
llama3.1:8b will happily turn the engine's line into its own reply:

```
engine:   i'm teaching you that god is good
polished: i am learning about a being known as god who possesses the
          quality of goodness somehow          <- 8 words the engine never said
```

That is Bucket quietly becoming a generic assistant. So the rule is enforced in
code, not in the prompt: every polished line is diffed against the raw one, and
if it contains any noun, verb, adjective or adverb the engine didn't say, it is
discarded and the raw line goes out instead. Contractions and inflections
(`i am`/`i'm`, `carry`/`carrying`) are normalized first so they don't count as
invention, and ordinary grammatical glue is allowed.

The model may delete words and reorder words. It may not add them.

Expect a real rejection rate — llama3.1:8b drifts on roughly a third of lines,
and those replies simply come out unpolished. `/stats` reports the ratio:

```
polish: ollama (llama3.1:8b), 3/10 rejected as invented (last: said, thing)
```

If rejections are most of your traffic, the model is ignoring the brief and
you're better off with `BUCKET_LLM_BACKEND=none`. `BUCKET_LLM_STRICT=0` disables
the guard, which is only sensible if you want an LLM chatbot rather than Bucket.

```bash
python polish_test.py
```

**Ollama** (local, free): just have Ollama running. It auto-detects an installed
model, or set `BUCKET_LLM_MODEL`.

**Prefer a non-reasoning model** — `llama3.1:8b`, `qwen2.5:7b`, `gemma3:4b`. A
reasoning model deliberates for several seconds before rewriting one short
sentence, and is likelier to break character and answer the user. If you do point
it at one, it's handled: the backend uses `/api/chat` with `think: false`, because
`/api/generate` makes reasoning models spend the entire token budget inside a
`<think>` block and return an empty response every single time.

`BUCKET_LLM_POLISH_RATE=0.5` polishes only half the replies — good for latency,
and the mix of raw and smoothed reads better than uniformly smoothed anyway.

**Claude**: `pip install anthropic`, set `ANTHROPIC_API_KEY`. Defaults to
`claude-opus-5`. Faster and much better at keeping the voice, but it's a paid API
call per reply and it will decline to re-voice genuinely abusive content — in
which case you get the raw line instead. If you want zero interference, use
`none` or `ollama`.

---

## Turning the chaos down

It ships as the original was: it learns from everything, from everyone, with no
filter. Levers if you change your mind, none of which require touching code:

- `BUCKET_LEARN=0` — freeze the database. It still talks, it stops absorbing.
- `/forget <thing>` — surgical removal of one subject.
- `/literal <thing>` — see exactly what it believes before deciding.
- `BUCKET_CHATTINESS=0` — only speaks when addressed.
- `/wipe confirm` — start over.

Worth knowing before you put it in a group with people you don't know: it will
repeat what it's taught, to everyone, indefinitely, and `/literal` is the only
way to audit that. It's your bot on your token, so its output is yours — and
Telegram's terms apply to it the same as to anything else you run there. Keep
backups of `bucket.sqlite3` if a particular era of it is worth keeping; a wipe
is not reversible.

---

## Files

```
bucket/
  config.py   settings from .env + environment
  db.py       SQLite storage (WAL)
  text.py     tokenizing, normalizing
  learn.py    ingestion — shreds every message into the tables
  brain.py    the seven strategies + markov generation
  embed.py    semantic memory: embeddings + in-memory vector index
  llm.py      optional polish layer (Claude / Ollama)
  seed.py     small starter corpus so a fresh bot isn't mute
  core.py     everything wired together, thread-safe for multiple surfaces
console.py       local REPL
telegram_bot.py  Telegram long-polling client (stdlib only)
discord_bot.py   Discord gateway client (discord.py)
run_all.py       both surfaces, one process, one shared memory
```

### Tests

```bash
python smoke_test.py          # engine end to end
python memory_test.py         # semantic recall vs word matching
python shared_memory_test.py  # cross-surface memory + concurrency
python polish_test.py         # the LLM never invents content
python chatter_test.py        # talks more without flooding
python feedback_test.py       # it can't talk itself into an obsession
python factoid_test.py        # what sentence shapes become facts
python identity_test.py       # two names, one person
python webapp_test.py         # mini app auth and endpoints
python voice_test.py          # does it sound like Bucket, or just broken
python recall_test.py         # facts by meaning, pronouns by context
python lexical_test.py        # bm25 vs the idf scan, and index sync
python repeat_test.py         # it won't say back what was just said
python blocklist_test.py      # blocked names: learn, reply, display
python inventory_test.py      # things handed to it, and handed back
```

### Keeping the corpus clean

Pastes get learned like anything else and then recited verbatim — Bucket once
replied with 200 words of Stable Diffusion parameters. Lines over
`BUCKET_MAX_LEARN_WORDS` (55) are rejected, as is anything shaped like metadata
(several of `seed`/`steps`/`cfg`/`sampler`/`model` plus numbers). Replies are
capped at `BUCKET_MAX_REPLY_WORDS` (40) regardless.

```bash
python clean_corpus.py --apply       # remove pastes already stored
python rebuild_phrases.py --apply    # rebuild the chain from what's left
```

`smoke_test` feeds it a scripted conversation and asserts every table fills,
factoid recall works, items get picked up, all seven strategies fire, an
obsession forms, and the polish layer degrades safely when disabled.
`memory_test` checks paraphrase recall actually beats word matching.
`shared_memory_test` proves a line taught on one surface is recallable from the
other, and hammers one `Bucket` from two threads looking for races.
