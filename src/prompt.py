"""
Prompt templates and response parsing for emotion classification.
"""


BINARY_PROMPT = (
    'Is the emotion "{emotion}" present in the following text? '
    'Answer only yes or no.\n'
    'Text: "{text}"\n'
    'Answer:'
)


def build_binary_prompt(text: str, emotion: str) -> str:
    return BINARY_PROMPT.format(text=text, emotion=emotion)


def build_multilabel_prompt(text: str, emotion_classes: list) -> str:
    """Build a multi-label classification prompt (for IT models)."""
    emotions_str = ", ".join(emotion_classes)
    return (
        f"<start_of_turn>user\n"
        f"Among these emotion classes: {emotions_str}, and no emotion, "
        f"which emotions are contained in this text? "
        f"Answer only the emotion names, separated by commas. "
        f"If no emotion, answer: none.\n"
        f'Text: "{text}"\n'
        f"<end_of_turn>\n"
        f"<start_of_turn>model\n"
    )


def parse_response(text: str, valid_emotions: list) -> list:
    """
    Extract emotion names from model response, handling irregularities.

    Handles:
      - Clean: "anger, surprise"
      - Verbose: "this text contains emotion anger and surprise."
      - Negation: "not about anger, but joy" -> [joy]
      - None: "none" -> []
    """
    t = text.lower().strip()
    t = t.replace("<end_of_turn>", "").replace("<eos>", "").strip()

    if t in ["none", "no emotion", "none.", "no emotion."]:
        return []

    detected = []
    for emo in valid_emotions:
        if emo in t:
            negations = [
                f"not {emo}", f"no {emo}", f"isn't {emo}",
                f"without {emo}", f"not about {emo}",
            ]
            if not any(neg in t for neg in negations):
                detected.append(emo)
    return detected


def emotions_to_multihot(emotions: list, valid_emotions: list) -> list:
    return [1 if e in emotions else 0 for e in valid_emotions]
