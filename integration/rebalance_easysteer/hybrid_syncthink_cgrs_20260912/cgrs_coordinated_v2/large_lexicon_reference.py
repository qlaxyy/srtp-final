"""CPU specification for L27 clean-opening phrase suppression, not a GPU hook.

Only penalize tokens that complete an opening word/phrase. Never penalize the
first token of a multi-token phrase merely because it might lead to reflection.
Matching full words at a token boundary cannot predict the next token: this is
an explicit prospective lexical approximation, as in the original T14 policy.
"""
import re


class OpeningMatcher:
    def __init__(self, author):
        self.author = author
        self.pattern = re.compile(r'^\s*(?:' + '|'.join(author._token_pattern(x)
            for x in author.LEXICON_BASE) + r')\b', re.IGNORECASE)
        self.reset()

    def reset(self):
        self.prefix = ''
        self.open = False
        self.thinking = False
        self.matched = False

    def would_complete(self, piece):
        if not self.open or not self.thinking or self.matched:
            return False
        return bool(self.pattern.search(self.author._normalize_text(self.prefix + piece)))

    def accept(self, piece, *, boundary=False, start=False, end=False):
        if start or end:
            self.prefix = ''
            self.open = False
            self.matched = False
            self.thinking = bool(start and not end)
            return
        if not self.thinking:
            return
        if boundary:
            # Same clean-opening definition as RC14; mixed boundary/content is excluded.
            self.prefix = ''
            self.open = not piece.rsplit('\n\n', 1)[-1].strip()
            self.matched = False
            return
        if self.open:
            self.matched |= self.would_complete(piece)
            self.prefix += piece
            # CPU reference only. Native integration must use a finite-state
            # matcher rather than moving whole prefixes to the host per token.


def candidate_ids(matcher, decoded_vocabulary):
    """Slow reference oracle for validating a future device transition table."""
    return [i for i, piece in decoded_vocabulary.items() if matcher.would_complete(piece)]
