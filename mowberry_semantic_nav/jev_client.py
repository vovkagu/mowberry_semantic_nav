"""Jev (TypeSafe AI) decision client.

STEP 1 PLACEHOLDER: this module only fixes the interface used by
``semantic_nav_bridge``. ``decide`` currently runs a tiny offline keyword
matcher so the pipeline can be exercised end to end without the API. Step 2
replaces it with the real HTTP client (strict response schema, error
handling, confidence gating, ``unknown_abort`` fallback).
"""

from dataclasses import dataclass

ABORT_CHOICE = 'unknown_abort'


@dataclass(frozen=True)
class JevDecision:
    """Outcome of a Jev choice question."""

    choice: str
    confidence: float
    reason: str = ''


class JevClient:
    """Client for the Jev choice endpoint (offline placeholder for now)."""

    def __init__(self, endpoint, timeout_s=5.0):
        self.endpoint = endpoint
        self.timeout_s = timeout_s

    def decide(self, state, question, options):
        """Pick one of ``options`` for the given ``state``.

        Args:
            state: dict with ``user_command`` and ``detected_objects``.
            question: the decision question sent to Jev.
            options: allowed answers; always includes ``ABORT_CHOICE``.

        Returns:
            JevDecision. Never raises: any failure maps to ``ABORT_CHOICE``.
        """
        # "the shed's door" -> "the shed door" so it matches "shed_door".
        command = state['user_command'].lower().replace("'s ", ' ')
        objects = state['detected_objects']

        # Objects whose label (e.g. "shed_door" -> "shed door") is mentioned.
        matches = [o for o in objects
                   if o['label'].replace('_', ' ') in command]
        # Narrow down by side if the command names one.
        for side in ('left', 'right', 'center'):
            if side in command:
                matches = [o for o in matches
                           if o['relative_position'].endswith(side)]

        if len(matches) == 1:
            return JevDecision(str(matches[0]['id']), 0.9,
                               'offline placeholder: unique label match')
        return JevDecision(ABORT_CHOICE, 0.0,
                           f'offline placeholder: {len(matches)} matches')
