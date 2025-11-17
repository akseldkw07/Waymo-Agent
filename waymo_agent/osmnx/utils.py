def _max_speed_int(speed: list[str] | str | int | float) -> int:
    """Helper to convert maxspeed attribute to integer in mph."""
    if isinstance(speed, list):
        speed = speed[0]  # Take the first value if it's a list
    if isinstance(speed, str):
        speed = speed.lower().strip()
        if "mph" in speed:
            speed = speed.replace("mph", "").strip()
        elif "km/h" in speed:
            kmh_value = float(speed.replace("km/h", "").strip())
            return int(kmh_value * 0.621371)  # Convert km/h to mph

        return int(float(speed))
    elif isinstance(speed, (int, float)):
        return int(speed)
