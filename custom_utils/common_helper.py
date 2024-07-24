

def str2bool(v: str) -> bool:
    """Converts a string to a boolean value. Used for argparse."""
    return v.lower() in ("yes", "true", "t", "1")