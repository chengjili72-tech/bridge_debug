import copy
import functools
import types


def remove_non_pickleables(obj, max_depth: int = 3, current_depth: int = 0):
    """Remove non-pickleable objects from a configuration object recursively.

    This utility function identifies and removes objects that cannot be pickled for
    inter-process communication, including functions, bound methods, partial
    functions, and other problematic callables.

    Args:
        obj: The object to clean
        max_depth: Maximum recursion depth (default: 3)
        current_depth: Current recursion depth (internal use)

    Returns:
        The cleaned object with non-pickleables removed
    """

    # Stop recursion if max depth reached
    if current_depth >= max_depth:
        return obj

    # Handle None
    if obj is None:
        return obj

    # Explicitly drop process group objects without importing their classes directly.
    cls = obj if isinstance(obj, type) else type(obj)
    cls_module = getattr(cls, "__module__", "")
    cls_name = getattr(cls, "__qualname__", getattr(cls, "__name__", ""))
    if (cls_module, cls_name) in {
        ("megatron.core.process_groups_config", "ProcessGroupCollection"),
        ("torch._C._distributed_c10d", "ProcessGroup"),
    }:
        return None

    # Check if object is a problematic callable
    if callable(obj):
        # Allow classes/types but remove function objects, methods, partials
        if isinstance(obj, type):
            return obj
        elif hasattr(obj, "__call__") and (
            isinstance(obj, (types.FunctionType, types.MethodType, functools.partial)) or hasattr(obj, "__self__")
        ):  # bound methods
            return None

    # Handle dataclass/object with attributes
    if hasattr(obj, "__dict__"):
        # Create a copy to avoid modifying the original
        cleaned_obj = copy.copy(obj)

        for attr_name in list(vars(cleaned_obj).keys()):
            attr_value = getattr(cleaned_obj, attr_name)

            # Recursively clean attribute
            cleaned_value = remove_non_pickleables(attr_value, max_depth, current_depth + 1)

            # Set the cleaned value (or None if it was removed)
            setattr(cleaned_obj, attr_name, cleaned_value)

        return cleaned_obj

    # Handle lists
    elif isinstance(obj, list):
        return [remove_non_pickleables(item, max_depth, current_depth + 1) for item in obj]

    # Handle tuples
    elif isinstance(obj, tuple):
        return tuple(remove_non_pickleables(item, max_depth, current_depth + 1) for item in obj)

    # Handle dictionaries
    elif isinstance(obj, dict):
        return {key: remove_non_pickleables(value, max_depth, current_depth + 1) for key, value in obj.items()}

    # For primitive types and other safe objects, return as-is
    return obj
