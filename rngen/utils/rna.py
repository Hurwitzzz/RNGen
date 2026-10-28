import re
from typing import Dict, Optional


def parse_gt_rnbert(gt_rnbert: str) -> Dict[str, str]:
    """
    Parses a 'gt_rnbert' string into its constituent parts.

    This function is the inverse of `add_gt_rnbert_col`. It deconstructs a
    Roman numeral string into primary/secondary alterations and numerals,
    quality, inversion, and optional chord tones/factors.

    Args:
        gt_rnbert: The Roman numeral string to parse.
                   e.g., '#iii64/bVII: D#-F#-A#, R-T-F'

    Returns:
        A dictionary containing the parsed components:
        - "primary_numeral": The primary Roman numeral.
        - "primary_alteration": The primary alteration ('#', 'b', or '').
        - "secondary_numeral": The secondary Roman numeral (or '').
        - "secondary_alteration": The secondary alteration ('#', 'b', or '').
        - "quality": The chord quality.
        - "inversion": The chord inversion string (or '').
        - "chord_tones": The chord tones string (or '').
        - "chord_factors": The chord factors string (or '').
    """
    parsed_result: Dict[str, str] = {
        "primary_numeral": "",
        "primary_alteration": "",
        "secondary_numeral": "",
        "secondary_alteration": "",
        "quality": "",
        "inversion": "",
        "chord_tones": "",
        "chord_factors": "",
    }

    # 1. Split gt_rnbert into the main analysis and optional chord info
    roman_part = gt_rnbert
    if ": " in gt_rnbert:
        main_parts = gt_rnbert.split(": ", 1)
        roman_part = main_parts[0]
        if len(main_parts) > 1:
            info_parts = main_parts[1].split(", ", 1)
            parsed_result["chord_tones"] = info_parts[0]
            if len(info_parts) > 1:
                parsed_result["chord_factors"] = info_parts[1]

    # 2. Handle Augmented 6th and Neapolitan chords as special cases
    if any(sub in roman_part for sub in ["Ger", "Fr", "It", "N"]):
        parsed_result["quality"] = (
            "aug6"
            if any(sub in roman_part for sub in ["Ger", "Fr", "It"])
            else "Neapolitan"
        )

        primary_part, secondary_part = (roman_part.split("/", 1) + [""])[:2]

        # Parse primary part for aug6/N
        match = re.match(r"([#b]*)(.*)", primary_part)
        if match:
            parsed_result["primary_alteration"] = match.group(1) or ""
            parsed_result["primary_numeral"] = match.group(2)

        # Parse secondary part normally if it exists
        if secondary_part:
            match = re.match(r"([#b]*)(.*)", secondary_part)
            if match:
                parsed_result["secondary_alteration"] = match.group(1) or ""
                parsed_result["secondary_numeral"] = match.group(2)

        return parsed_result

    # 3. Parse standard chords
    primary_part_str, secondary_part_str = (roman_part.split("/", 1) + [""])[:2]

    # Regex to parse primary part: (alteration)(numeral)(quality)(inversion)
    primary_regex = re.compile(r"^([#b]*)([ivIV]+)([MoøMm\d\+]*?)([64532]*)$")
    primary_match = primary_regex.match(primary_part_str)

    if not primary_match:
        raise ValueError(f"Could not parse primary numeral part: '{primary_part_str}'")
    else:
        p_alt, p_num, qual_str, p_inv = primary_match.groups()

        parsed_result["primary_alteration"] = p_alt
        parsed_result["primary_numeral"] = p_num
        parsed_result["inversion"] = p_inv

        quality_map = {
            "M7": "M7",
            "M": "M7",
            "ø7": "ø7",
            "ø": "ø7",
            "+": "+",
            "o7": "o7",
        }  # Used after eliminate uncertainty

        if qual_str == "" and p_inv in ["65", "43", "2"]:
            if p_num.isupper():
                parsed_result["quality"] = "Mm7"
            else:
                parsed_result["quality"] = "m7"
        elif qual_str == "o":
            if p_inv in ["65", "43", "2"]:
                parsed_result["quality"] = "o7"
            else:
                parsed_result["quality"] = "o"
        elif qual_str == "7":
            if p_num.isupper():
                parsed_result["quality"] = "Mm7"
            else:
                parsed_result["quality"] = "m7"
        elif p_num.islower() and qual_str == "":
            parsed_result["quality"] = "m"
        elif p_num.isupper() and qual_str == "":
            parsed_result["quality"] = "M"
        elif qual_str in quality_map:
            parsed_result["quality"] = quality_map[qual_str]
        else:
            raise ValueError(f"Could not parse quality from: '{qual_str}'")

    # Parse secondary part if it exists
    if secondary_part_str:
        secondary_match = re.match(r"([#b]*)([ivIV]+)", secondary_part_str)
        if secondary_match:
            s_alt, s_num = secondary_match.groups()
            parsed_result["secondary_alteration"] = s_alt
            parsed_result["secondary_numeral"] = s_num
        else:
            raise ValueError(
                f"Could not parse secondary numeral part: '{secondary_part_str}'"
            )

    return parsed_result
