import typing as t


def _max_speed_int(speed: list[str] | str | int | float) -> int:
    """Helper to convert maxspeed attribute to integer in mph."""

    def _to_float(s: str | int | float) -> float:
        if isinstance(s, (int, float)):
            return float(s)
        s = s.lower().strip()
        if "km/h" in s:
            kmh_value = float(s.replace("km/h", "").strip())
            ret = int(kmh_value * 0.621371)
        if "mph" in s:
            ret = s.replace("mph", "").strip()
        else:
            ret = s
        return float(ret)

    if isinstance(speed, str):
        ret = _to_float(speed)
        return round(ret)
    elif isinstance(speed, t.Iterable):  # check after str because str is Iterable
        speed_float: list[float] = [_to_float(s) for s in speed if s]  # filter out empty strings
        return round(sum(speed_float) / len(speed_float))  # average if multiple values
    elif isinstance(speed, (int, float)):
        return round(speed)
