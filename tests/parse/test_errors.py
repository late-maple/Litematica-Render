"""Public exception tree contract."""

import pytest

import litematic


def test_exception_hierarchy():
    assert issubclass(litematic.LitematicError, Exception)
    assert issubclass(litematic.NbtError, litematic.LitematicError)
    assert issubclass(litematic.LitematicFormatError, litematic.LitematicError)


@pytest.mark.parametrize(
    "sub",
    [litematic.NbtError, litematic.LitematicFormatError],
)
def test_base_class_catches_subclasses(sub):
    with pytest.raises(litematic.LitematicError):
        raise sub("boom")


def test_version_string():
    assert litematic.__version__ == "0.2.0"
