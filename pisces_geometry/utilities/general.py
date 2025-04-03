"""
General-purpose utilities for Pisces-Geometry.

This module provides reusable helper functions for dynamic introspection and
class discovery used throughout the Pisces-Geometry codebase.

These utilities support internal mechanisms like automatic subclass resolution,
runtime configuration, and flexible API design.

"""

from typing import Type, TypeVar

_T = TypeVar("_T")


def find_in_subclasses(base_class: Type[_T], class_name: str) -> Type[_T]:
    """
    Recursively search for a subclass by name within the subclasses of a given base class.

    Parameters
    ----------
    base_class : Type[_T]
        The base class whose subclasses will be searched.
    class_name : str
        The name of the subclass to search for.

    Returns
    -------
    Type[_T]
        The subclass with the specified name.

    Raises
    ------
    ValueError
        If no subclass with the specified name is found.

    Notes
    -----
    This function uses recursion to traverse the inheritance tree of the given base class, checking
    the name of each subclass. If the specified subclass is not found in the direct subclasses, the
    function will attempt to search the subclasses of each subclass.

    Examples
    --------
    Suppose you have the following class hierarchy:

    >>> class Base:
    ...     pass
    >>> class SubClass1(Base):
    ...     pass
    >>> class SubClass2(SubClass1):
    ...     pass

    You can use `find_in_subclasses` to locate a specific subclass by name:

    >>> find_in_subclasses(Base, 'SubClass2')
    <class 'general.SubClass2'>

    """
    print(base_class.__subclasses__(), base_class.__name__)
    for subclass in base_class.__subclasses__():
        print(subclass.__name__)
        if subclass.__name__ == class_name:
            return subclass
        if subclass.__name__ == class_name:
            return subclass
        try:
            result = find_in_subclasses(subclass, class_name)
        except ValueError:
            continue
        if result:
            return result

    raise ValueError(
        f"Failed to find subclass of {base_class.__name__} named {class_name}."
    )
