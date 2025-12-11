import datetime as dt
import math
import random
import typing as t


def random_dt(year: int = 2025) -> dt.datetime:
    """
    Generates a random datetime object from the first 7 days of the specified year
    (Jan 1st through Jan 7th), assuming a uniform distribution.
    """
    # 1. Define the time range: Jan 1st (00:00:00) to Jan 8th (00:00:00)
    start_date = dt.datetime(year, 1, 1)
    end_date = dt.datetime(year, 1, 8)

    # 2. Calculate the total duration in seconds (604,800 seconds total)
    time_difference = end_date - start_date
    total_seconds = time_difference.total_seconds()

    # 3. Generate a random offset (in seconds)
    # random.uniform ensures a uniform distribution across the range
    random_seconds_offset = random.uniform(0, total_seconds)

    # 4. Add the offset to the start date
    random_datetime = start_date + dt.timedelta(seconds=random_seconds_offset)

    return random_datetime


def embed_datetime_to_circle(dt_val: dt.datetime, norm: t.Literal["time", "dow"]) -> tuple[float, float]:
    """
    Embed datetime value (time of day or day of week) into cyclical features using sine and cosine transformations.
    """

    denom = 24 * 60 * 60 if norm == "time" else 7
    val = dt_val.hour * 3600 + dt_val.minute * 60 + dt_val.second if norm == "time" else dt_val.weekday()
    sin_val = round(math.sin(2 * math.pi * val / denom), 4)
    cos_val = round(math.cos(2 * math.pi * val / denom), 4)
    return (sin_val, cos_val)
