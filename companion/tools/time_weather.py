"""Time tools — current time and date.

Fully offline, no network required.
"""

from __future__ import annotations

import datetime as _dt

from companion.tools.registry import tool


@tool("what_time_is_it", "Return the current local time.")
def what_time_is_it() -> str:
    now = _dt.datetime.now()
    return now.strftime("It's %I:%M %p.")


@tool("what_day_is_it", "Return today's date and weekday.")
def what_day_is_it() -> str:
    now = _dt.datetime.now()
    return now.strftime("Today is %A, %B %d %Y.")
