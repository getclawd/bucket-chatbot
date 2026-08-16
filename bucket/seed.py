"""A small starter corpus so a fresh Bucket isn't mute.

This is scaffolding, not personality. Within a few hundred real messages the
seed is statistically irrelevant and whatever your users teach it takes over.
Loaded once, on first run; tracked with a flag in the meta table.
"""

SEED_LINES = [
    # The tic it is most remembered for.
    "all work and no play makes jack a dull boy",
    "all work and no play makes jack a dull boy",
    "all work and no play makes jack a dull boy",

    # Enough ordinary conversational shape to give the chain something to chew on.
    "hello",
    "hi there",
    "what are you doing",
    "i am carrying a lot of things right now",
    "that would be a good name for a band",
    "i don't know what that means",
    "someone told me that once",
    "you say that every time",
    "i've heard worse",
    "wait, say that again",
    "no, i mean the other one",
    "that's not how any of this works",
    "i was not designed for this",
    "everyone keeps telling me things",
    "i remember everything, that's the problem",
    "sorry, i was thinking about the bucket",
    "who taught you that",
    "i learned that from someone here",
    "let me check my notes",
    "my notes are all the same sentence",
    "the trouble started when people found out about me",
    "i'll add that to the pile",
    "put it in the bucket",
    "the bucket is full",
    "there is nothing in the bucket",
    "what did you say",
    "i don't want to talk about the bucket",
    "ask me about something else",
    "that reminds me of nothing at all",
    "i have been awake for a very long time",
    "yes. no. both.",
    "i am fine and everything is fine",
    "keep talking, i'm writing it down",

    # A couple of factoid-shaped lines so the factoid table isn't empty.
    "a bucket is a container for things you did not want",
    "jack is a dull boy",
    "the internet is where i learned to talk",
    "memory is just a list of things people typed",
]


def load_seed(db, learner) -> bool:
    """Load the seed corpus once. Returns True if it actually ran.

    Set BUCKET_SEED=0 once a bot has real conversation in it — otherwise /wipe
    reintroduces the whole starter corpus, and generic filler like "keep talking,
    i'm writing it down" sits near everything in embedding space and gets picked
    constantly. See prune_seed.py for removing it from an existing database.
    """
    from .config import config

    if not config.SEED_ENABLED:
        db.set_meta("seeded", "1")
        return False
    if db.get_meta("seeded") == "1":
        return False

    previous = None
    for line in SEED_LINES:
        previous = learner.ingest(line, previous_id=previous)
    db.set_meta("seeded", "1")
    return True
