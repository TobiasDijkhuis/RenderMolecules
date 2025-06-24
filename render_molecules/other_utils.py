from __future__ import annotations

import numpy as np


def hex2rgbtuple(hexcode: str) -> tuple[float, float, float]:
    """
    Convert 6-digit color hexcode to a tuple of floats
    """
    hexcode += "FF"
    hextuple = tuple([int(hexcode[i : i + 2], 16) / 255.0 for i in [0, 2, 4, 6]])

    return tuple([color_srgb_to_scene_linear(c) for c in hextuple])


def color_srgb_to_scene_linear(c: float) -> float:
    """
    Convert RGB to sRGB
    """
    if c < 0.04045:
        return 0.0 if c < 0.0 else c * (1.0 / 12.92)
    else:
        return ((c + 0.055) * (1.0 / 1.055)) ** 2.4


def find_first_string_in_list_of_strings(
    string_to_find: str | list[str],
    list_of_strings: list[str],
    start: int = 0,
    end: int | None = None,
) -> int | None:
    """Finds the first instance of a string in a list of strings."""
    if isinstance(string_to_find, list):
        result_for_each_substring = np.ndarray(len(string_to_find), dtype=list)

        for i, substring in enumerate(string_to_find):
            result_for_each_substring[i] = find_all_string_in_list_of_strings(
                substring, list_of_strings, start, end
            )
        intersection_of_all_substrings = list(
            set.intersection(*map(set, result_for_each_substring))
        )
        return intersection_of_all_substrings[0]

    seperator = "UNIQUE SEPERATOR STRING THAT DOES NOT OCCUR IN THE FILE ITSELF"
    joined_list = seperator.join(list_of_strings[start:end])
    try:
        string_index = joined_list.index(string_to_find)
    except ValueError:
        return
    list_index = joined_list.count(seperator, 0, string_index)
    return list_index + start


def find_all_string_in_list_of_strings(
    string_to_find: str | list[str],
    list_of_strings: list[str],
    start: int = 0,
    end: int | None = None,
) -> list[int]:
    """Finds all instances of a string stringToFind in a list of strings."""
    if isinstance(string_to_find, list):
        result_for_each_substring = np.ndarray(len(string_to_find), dtype=list)

        for i, substring in enumerate(string_to_find):
            result_for_each_substring[i] = find_all_string_in_list_of_strings(
                substring, list_of_strings, start, end
            )
        intersection_of_all_substrings = list(
            set.intersection(*map(set, result_for_each_substring))
        )
        return intersection_of_all_substrings

    result = []
    new_result = start - 1
    while new_result is not None:
        start = new_result + 1
        new_result = find_first_string_in_list_of_strings(
            string_to_find, list_of_strings, start, end
        )
        result.append(new_result)
    result.pop()
    return result
