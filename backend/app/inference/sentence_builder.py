import time


class SentenceBuilder:
    """
    Accumulates recognized signs into a stable sentence with anti-flicker logic.

    Parameters
    ----------
    stability_count : int
        Number of consecutive same-label spotted events required before
        the word is emitted.
    cooldown_sec : float
        Minimum seconds between sentence additions.
    confidence_threshold : float
        Minimum confidence to consider a spotted label at all.
    max_history : int
        Maximum sentence length.
    """

    def __init__(
        self,
        stability_count: int = 2,
        cooldown_sec: float = 1.0,
        confidence_threshold: float = 0.55,
        max_history: int = 50,
    ):
        self.stability_count = stability_count
        self.cooldown_sec = cooldown_sec
        self.confidence_threshold = confidence_threshold
        self.max_history = max_history

        # Internal state
        self._sentence: list[str] = []
        self._run_label: str = ""
        self._run_count: int = 0
        self._last_emit_time: float = 0
        self._total_spots: int = 0
        self._total_emitted: int = 0
        self._total_rejected: int = 0

    def feed(self, label: str, confidence: float) -> str | None:
        """
        Feed a new prediction.

        Returns the label string if it was emitted to the sentence,
        or None if rejected (low confidence, unstable, cooldown, dup).
        """
        self._total_spots += 1

        # Gate 1: confidence
        if confidence < self.confidence_threshold:
            self._run_label = ""
            self._run_count = 0
            self._total_rejected += 1
            return None

        # Gate 2: stability (consecutive same label)
        if label == self._run_label:
            self._run_count += 1
        else:
            self._run_label = label
            self._run_count = 1

        if self._run_count < self.stability_count:
            return None

        # Gate 3: cooldown timer
        now = time.time()
        if (now - self._last_emit_time) < self.cooldown_sec:
            return None

        # Gate 4: consecutive dedup
        if self._sentence and self._sentence[-1] == label:
            return None

        # Emit
        self._sentence.append(label)
        if len(self._sentence) > self.max_history:
            self._sentence = self._sentence[-self.max_history :]
        self._last_emit_time = now
        self._total_emitted += 1
        self._run_count = 0
        return label

    @property
    def sentence(self) -> list[str]:
        return list(self._sentence)

    @property
    def text(self) -> str:
        return " ".join(self._sentence)

    def clear(self) -> None:
        self._sentence.clear()
        self._run_label = ""
        self._run_count = 0
        self._last_emit_time = 0
        self._total_spots = 0
        self._total_emitted = 0
        self._total_rejected = 0
