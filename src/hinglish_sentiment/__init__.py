"""Fine-tuned small models vs large LLMs on Hinglish (code-mixed) sentiment."""

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))
