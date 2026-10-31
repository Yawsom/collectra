"""Tests for utils/get_types.py module."""

from typing import List, Optional, Tuple, Union

from utils.get_types import get_param_types, get_return_type, unpack_types

# ==================== Test helper functions ====================


def simple_function(a: int, b: str) -> bool:
    return True


def function_with_optional(a: int, b: Optional[str] = None) -> Optional[int]:
    return a if b else None


def function_with_union(a: Union[int, str], b: List[int]) -> Union[str, None]:
    return str(a)


def function_no_annotations(a, b):
    return a + b


def function_partial_annotations(a: int, b) -> str:
    return str(a)


def function_tuple_return(a: int) -> Tuple[int, str]:
    return (a, str(a))


# ==================== get_param_types tests ====================


def test_get_param_types_simple():
    """Extract param types from a simple function."""
    result = get_param_types(simple_function)

    assert "a" in result
    assert "b" in result
    assert result["a"] is int
    assert result["b"] is str
    assert "return" not in result


def test_get_param_types_optional():
    """Extract param types with Optional annotations."""
    result = get_param_types(function_with_optional)

    assert "a" in result
    assert "b" in result
    assert result["a"] is int
    assert result["b"] == Optional[str]


def test_get_param_types_union():
    """Extract param types with Union annotations."""
    result = get_param_types(function_with_union)

    assert "a" in result
    assert "b" in result
    assert result["a"] == Union[int, str]
    assert result["b"] == List[int]


def test_get_param_types_no_annotations():
    """Function without annotations returns empty dict."""
    result = get_param_types(function_no_annotations)

    assert result == {}


def test_get_param_types_partial():
    """Function with only some parameters annotated."""
    result = get_param_types(function_partial_annotations)

    assert "a" in result
    assert result["a"] is int
    assert "b" not in result
    assert "return" not in result


# ==================== get_return_type tests ====================


def test_get_return_type_simple():
    """Extract return type from simple function."""
    result = get_return_type(simple_function)

    assert "return" in result
    assert result["return"] is bool
    assert len(result) == 1


def test_get_return_type_optional():
    """Extract Optional return type."""
    result = get_return_type(function_with_optional)

    assert "return" in result
    assert result["return"] == Optional[int]


def test_get_return_type_union():
    """Extract Union return type."""
    result = get_return_type(function_with_union)

    assert "return" in result
    assert result["return"] == Union[str, None]


def test_get_return_type_tuple():
    """Extract Tuple return type."""
    result = get_return_type(function_tuple_return)

    assert "return" in result
    assert result["return"] == Tuple[int, str]


def test_get_return_type_no_annotations():
    """Function without annotations returns empty dict."""
    result = get_return_type(function_no_annotations)

    assert result == {}


def test_get_return_type_excludes_params():
    """Parameter types are excluded from return type."""
    result = get_return_type(simple_function)

    assert "a" not in result
    assert "b" not in result


# ==================== unpack_types tests ====================


def test_unpack_types_simple():
    """Unpack simple param types (no generics)."""
    result = unpack_types(simple_function, get_param_types)

    assert "a" in result
    assert "b" in result
    assert result["a"] is int
    assert result["b"] is str


def test_unpack_types_optional():
    """Optional[T] unpacks to (T,) tuple without None."""
    result = unpack_types(function_with_optional, get_param_types)

    assert "b" in result
    assert isinstance(result["b"], tuple)
    assert str in result["b"]
    assert type(None) not in result["b"]


def test_unpack_types_union():
    """Union types unpack to tuple of types."""
    result = unpack_types(function_with_union, get_param_types)

    assert "a" in result
    assert isinstance(result["a"], tuple)
    assert int in result["a"]
    assert str in result["a"]


def test_unpack_types_return():
    """Unpack return types."""
    result = unpack_types(function_with_optional, get_return_type)

    assert "return" in result
    assert isinstance(result["return"], tuple)
    assert int in result["return"]
    assert type(None) not in result["return"]


def test_unpack_types_no_annotations():
    """Unpacking function with no annotations returns empty dict."""
    result = unpack_types(function_no_annotations, get_param_types)

    assert result == {}


def test_unpack_types_list():
    """List[T] unpacks to (T,) tuple."""
    result = unpack_types(function_with_union, get_param_types)

    assert "b" in result
    assert isinstance(result["b"], tuple)
    assert int in result["b"]


def test_unpack_types_tuple_return():
    """Tuple[T1, T2] unpacks to (T1, T2)."""
    result = unpack_types(function_tuple_return, get_return_type)

    assert "return" in result
    assert isinstance(result["return"], tuple)
    assert int in result["return"]
    assert str in result["return"]


# ==================== Pipeline usage pattern tests ====================


def test_pipeline_io_extraction_pattern():
    """Test the pattern used in Pipeline._check_task_io for inputs."""

    def task_run(input_data: str, config: dict) -> List[str]:
        return [input_data]

    param_types = list(unpack_types(task_run, get_param_types).items())

    assert len(param_types) == 2
    param_names = [name for name, _ in param_types]
    assert "input_data" in param_names
    assert "config" in param_names


def test_pipeline_output_extraction_pattern():
    """Test the pattern used in Pipeline._check_task_io for outputs."""

    def task_run(input_data: str) -> Optional[List[int]]:
        return [1, 2, 3]

    return_types = list(unpack_types(task_run, get_return_type).items())

    assert len(return_types) == 1
    assert return_types[0][0] == "return"
    types_tuple = return_types[0][1]
    assert isinstance(types_tuple, tuple)
    assert type(None) not in types_tuple


def test_types_to_tuple_conversion_pattern():
    """Test ensuring types are always tuples (pipeline pattern)."""

    def simple_task(data: int) -> str:
        return str(data)

    param_types = list(unpack_types(simple_task, get_param_types).items())

    for _, types in param_types:
        if not isinstance(types, tuple):
            types = (types,)
        assert isinstance(types, tuple)
