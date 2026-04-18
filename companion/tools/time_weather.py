"""Time + (placeholder) weather tool.

Time is trivial, offline. Weather is offline-only by default; if you
later wire up a network API key it can be returned too.
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


@tool("weather", "Describe the weather. Offline stub — returns a friendly note.")
def weather(location: str = "here") -> str:
    return (
        f"I don't have a weather feed yet — I'm offline. "
        f"Check your phone for {location}."
    )
