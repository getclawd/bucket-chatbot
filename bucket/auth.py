"""Small authorization helpers shared by the chat surfaces."""


def is_admin(user_id: object, allowed_ids: frozenset[int]) -> bool:
    """Return whether ``user_id`` is explicitly present in the allowlist.

    An empty allowlist therefore denies everyone. Keeping this policy in one
    helper makes it harder for a new surface to accidentally reintroduce a
    fail-open check.
    """
    return isinstance(user_id, int) and user_id in allowed_ids
