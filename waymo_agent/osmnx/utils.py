import ast
import typing as t


def _coerce_speed(value: t.Any) -> float | None:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if not cleaned:
            return None
        for suffix in ("mph", "km/h", "kph"):
            if suffix in cleaned:
                cleaned = cleaned.replace(suffix, "").strip()
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


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


def safe_literal_eval(data: str | t.Any):
    """
    Attempts to convert a string of a Python literal (like a tuple) into
    its corresponding Python object. Returns the original data if conversion fails.
    """
    # Check if the data is a string type
    if not isinstance(data, str):
        return data  # Return non-strings as is

    try:
        # Attempt to evaluate the string safely
        result = ast.literal_eval(data)

        # Optionally, you might check if the result is a tuple if that's
        # your specific target, but usually, just the successful evaluation is enough.
        return result

    except (ValueError, SyntaxError):
        # ValueError is common for invalid literals (e.g., 'hello')
        # SyntaxError is common for incomplete/malformed structures (e.g., '(1, 2')
        return data  # Return the original string if the evaluation fails
