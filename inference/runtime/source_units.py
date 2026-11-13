"""Original numerical functions; imports isolated for portability. PRIVATE."""
import re

def tokenize(text):
    """Text-only Unicode word/punctuation tokens, with generic camel-case splits.

    No gold entity boundaries, labels or normalization are used to create tokens.
    Return original Unicode code-point offsets, including punctuation tokens.
    """
    result = []
    for match in re.finditer(r"\w+|[^\w\s]", text):
        start, end = match.span()
        breaks = [start] + [i for i in range(start + 1, end)
                            if text[i - 1].islower() and text[i].isupper()] + [end]
        result.extend((a, b, text[a:b]) for a, b in zip(breaks, breaks[1:]))
    return result


def decode(tokens, tags):
    """Repair an orphan I as B; reconstruct end-exclusive source offsets."""
    assert len(tokens) == len(tags)
    spans, current = [], None
    for (start, end, _), tag in zip(tokens, tags):
        if tag == "O":
            if current:
                spans.append(current)
                current = None
            continue
        prefix, label = tag.split("-", 1)
        if prefix == "B" or current is None or current[2] != label:
            if current:
                spans.append(current)
            current = [start, end, label]
        else:
            current[1] = end
    if current:
        spans.append(current)
    return spans

