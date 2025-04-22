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


def findFirstStringInListOfStrings(
    stringToFind: str | list[str],
    listOfStrings: list[str],
    start: int = 0,
    end: int | None = None,
) -> int | None:
    """Finds the first instance of a string in a list of strings."""
    if isinstance(stringToFind, list):
        resultForEachSubstring = np.ndarray(len(stringToFind), dtype=list)

        for i, substring in enumerate(stringToFind):
            resultForEachSubstring[i] = findAllStringInListOfStrings(
                substring, listOfStrings, start, end
            )
        intersectionOfAllSubstrings = list(
            set.intersection(*map(set, resultForEachSubstring))
        )
        return intersectionOfAllSubstrings[0]

    seperatorString = "UNIQUE SEPERATOR STRING THAT DOES NOT OCCUR IN THE FILE ITSELF"
    joinedList = seperatorString.join(listOfStrings[start:end])
    try:
        stringIndex = joinedList.index(stringToFind)
    except ValueError:
        return
    listIndex = joinedList.count(seperatorString, 0, stringIndex)
    return listIndex + start


def findAllStringInListOfStrings(
    stringToFind: str | list[str],
    listOfStrings: list[str],
    start: int = 0,
    end: int | None = None,
) -> list[int]:
    """Finds all instances of a string stringToFind in a list of strings."""
    if isinstance(stringToFind, list):
        resultForEachSubstring = np.ndarray(len(stringToFind), dtype=list)

        for i, substring in enumerate(stringToFind):
            resultForEachSubstring[i] = findAllStringInListOfStrings(
                substring, listOfStrings, start, end
            )
        intersectionOfAllSubstrings = list(
            set.intersection(*map(set, resultForEachSubstring))
        )
        return intersectionOfAllSubstrings

    result = []
    newResult = start - 1
    while newResult is not None:
        start = newResult + 1
        newResult = findFirstStringInListOfStrings(
            stringToFind, listOfStrings, start, end
        )
        result.append(newResult)
    result.pop()
    return result
