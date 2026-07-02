"""Colour parsing helpers for user-supplied curve colours.

Users may type common colour names (``red``), hex codes (``#D32F2F``), or CSS
``rgb()`` / ``rgba()`` strings into the style tables. This module validates and
normalises those inputs, falling back to a default when the value is empty or
cannot be understood.
"""

from __future__ import annotations

import re

# A compact set of common CSS colour names. Plotly ultimately accepts any CSS
# colour name, so we only need this set to validate free-form text before
# handing it to Plotly.
CSS_COLOR_NAMES = {
    "aliceblue", "antiquewhite", "aqua", "aquamarine", "azure", "beige",
    "bisque", "black", "blanchedalmond", "blue", "blueviolet", "brown",
    "burlywood", "cadetblue", "chartreuse", "chocolate", "coral",
    "cornflowerblue", "cornsilk", "crimson", "cyan", "darkblue", "darkcyan",
    "darkgoldenrod", "darkgray", "darkgrey", "darkgreen", "darkkhaki",
    "darkmagenta", "darkolivegreen", "darkorange", "darkorchid", "darkred",
    "darksalmon", "darkseagreen", "darkslateblue", "darkslategray",
    "darkslategrey", "darkturquoise", "darkviolet", "deeppink", "deepskyblue",
    "dimgray", "dimgrey", "dodgerblue", "firebrick", "floralwhite",
    "forestgreen", "fuchsia", "gainsboro", "ghostwhite", "gold", "goldenrod",
    "gray", "grey", "green", "greenyellow", "honeydew", "hotpink", "indianred",
    "indigo", "ivory", "khaki", "lavender", "lavenderblush", "lawngreen",
    "lemonchiffon", "lightblue", "lightcoral", "lightcyan",
    "lightgoldenrodyellow", "lightgray", "lightgrey", "lightgreen",
    "lightpink", "lightsalmon", "lightseagreen", "lightskyblue",
    "lightslategray", "lightslategrey", "lightsteelblue", "lightyellow",
    "lime", "limegreen", "linen", "magenta", "maroon", "mediumaquamarine",
    "mediumblue", "mediumorchid", "mediumpurple", "mediumseagreen",
    "mediumslateblue", "mediumspringgreen", "mediumturquoise",
    "mediumvioletred", "midnightblue", "mintcream", "mistyrose", "moccasin",
    "navajowhite", "navy", "oldlace", "olive", "olivedrab", "orange",
    "orangered", "orchid", "palegoldenrod", "palegreen", "paleturquoise",
    "palevioletred", "papayawhip", "peachpuff", "peru", "pink", "plum",
    "powderblue", "purple", "rebeccapurple", "red", "rosybrown", "royalblue",
    "saddlebrown", "salmon", "sandybrown", "seagreen", "seashell", "sienna",
    "silver", "skyblue", "slateblue", "slategray", "slategrey", "snow",
    "springgreen", "steelblue", "tan", "teal", "thistle", "tomato",
    "turquoise", "violet", "wheat", "white", "whitesmoke", "yellow",
    "yellowgreen",
}

_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_RGB_RE = re.compile(
    r"^rgba?\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*(?:,\s*(?:0|1|0?\.\d+))?\s*\)$",
    re.IGNORECASE,
)


def normalize_color_value(value: object, default: str = "#616161") -> str:
    """Return a Plotly-compatible colour string.

    ``value`` may be a colour name, ``#rgb``/``#rrggbb``/``#rrggbbaa`` hex
    string, or a CSS ``rgb()``/``rgba()`` string. Anything unrecognised (or
    blank) falls back to ``default``.
    """

    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default

    if _HEX_RE.match(text):
        return text.lower()

    if _RGB_RE.match(text.replace(" ", "")):
        return re.sub(r"\s+", "", text).lower()

    lowered = text.casefold()
    if lowered in CSS_COLOR_NAMES:
        return lowered

    return default
