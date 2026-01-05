from seriacade.json.types import JsonType


def enforce_dict_type(value: JsonType) -> dict[str, JsonType]:
    """Helper that enforces the python json representation to be a json object (ie. a dict).

    Args:
        value (JsonType): The Python representation of a JSON object.

    Raises:
        ValueError: If the provided Python object is not a dict.

    Returns:
        dict[str, JsonType]: The Python object if it is a dict.
    """
    match value:
        case dict():
            return value
        case _:
            err_msg = f"Expected a dict but got {type(value)}"
            raise ValueError(err_msg)
