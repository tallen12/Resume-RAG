from collections.abc import Sequence
from typing import Any, Protocol, TypeVar, runtime_checkable

type JSONPrimitive = None | bool | str | float | int
type JSONVal = JSONPrimitive | JSONArray | JSONObject
type JSONArray = Sequence[JSONVal]
type JSONObject = dict[str, JSONVal]

type PythonJSONType = JSONVal | dict[str, Any] | Sequence[JSONPrimitive | dict[str, Any]]

T = TypeVar("T")
T_contra = TypeVar("T_contra", contravariant=True)
T_co = TypeVar("T_co", covariant=True)


class JsonEncoderProtocol(Protocol[T_contra]):
    """Protocol for implementing JSON encoder for type."""

    def encode_json(self, data: T_contra) -> bytes:
        """Encode data to json bytes.

        Args:
            data (T_contra): Data to encode.

        Returns:
            bytes: bytes representing the json.
        """
        ...

    def encode_python_json(self, data: T_contra) -> PythonJSONType:
        """Encode data to a python object representing the json (dict/list/str/...).

        Args:
            data (T_contra): Data to encode.

        Returns:
            PythonJSONType: Python object for that datatype.
        """
        ...


class JsonDecoderProtocol(Protocol[T_co]):
    """Protocol for implementing JSON decoder for type."""

    def decode_json(self, data: bytes) -> T_co:
        """Decode to object from bytes.

        Args:
            data (bytes): byte string of formatted json.

        Returns:
            T_co: Object decoded from json.
        """
        ...

    def convert_json(self, data: PythonJSONType) -> T_co:
        """Convert from python object representing json to the Datatype for this codec.

        Args:
            data (PythonJSONType): Data from python representation of json.

        Returns:
            T_co: Decoded object.
        """
        ...


@runtime_checkable
class JsonCodecProtocol(JsonEncoderProtocol[T], JsonDecoderProtocol[T], Protocol[T]):
    """A full json codec supporting decoding and encoding."""

    def schema(self) -> dict[str, Any]:
        """Generate a json schema for this object.

        Returns:
            dict[str, Any]: a dict representing the json schema.
        """
        ...


def enforce_dict_type(value: PythonJSONType) -> dict[str, Any]:
    """Helper that enforces the python json representation to be a json object (ie. a dict).

    Args:
        value (PythonJSONType): the python representation of a json object.

    Raises:
        ValueError: Raises if the provided python object is not a dict.

    Returns:
        dict[str, Any]: returns the python object if it is a dict.
    """
    match value:
        case dict():
            return value
        case _:
            err_msg = f"Expected a dict but got {type(value)}"
            raise ValueError(err_msg)
