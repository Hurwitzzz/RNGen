from __future__ import annotations

import json


def repair_and_decode_json(s: str) -> dict | None:
    """
    Attempts to repair and decode a truncated JSON string.
    It finds the last complete value and tries to close any open structures.
    """
    last_comma = s.rfind(",")
    last_bracket = s.rfind("[")
    last_brace = s.rfind("{")

    end_pos = max(last_comma, last_bracket, last_brace)

    if end_pos == -1:
        # Fallback for simple cases like '{"key": "value"'
        if s.rfind('"') > s.rfind(":"):
            s = s[: s.rfind('"') + 1]  # Keep the closing quote
        else:
            return None
    else:
        if end_pos == last_comma:
            s = s[:end_pos]  # Remove the trailing comma
        else:
            s = s[: end_pos + 1]

    open_braces = s.count("{") - s.count("}")
    open_brackets = s.count("[") - s.count("]")

    s += "]" * open_brackets
    s += "}" * open_braces

    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # If that fails, try to find the last closing quote of a value
        # and parse up to that point.
        last_quote = s.rfind('"')
        if last_quote != -1:
            prev_comma = s.rfind(",", 0, last_quote)
            prev_bracket = s.rfind("[", 0, last_quote)
            cut_off = max(prev_comma, prev_bracket)
            if cut_off != -1:
                if cut_off == prev_comma:
                    s = s[:cut_off]
                else:
                    s = s[: cut_off + 1]
                open_braces = s.count("{") - s.count("}")
                open_brackets = s.count("[") - s.count("]")
                s += "]" * open_brackets
                s += "}" * open_braces
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    return None
    return None


if __name__ == "__main__":
    example_json = (
        '{"Eb":["bar 121 quarterbeats 1.00: I: Eb-G-Bb, R-T-F","bar 121 quarterbeats '
    )
    repaired_json = repair_and_decode_json(example_json)
    if repaired_json is not None:
        print("Repaired JSON:", repaired_json)
    else:
        print("Failed to repair JSON.")
