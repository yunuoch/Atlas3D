from typing import Callable, Optional, Union, Tuple, Dict, Any
from copy import copy
import bisect
import re


import warp as wp


_kernel_cache = dict()
_struct_cache = dict()
_func_cache = dict()

_key_re = re.compile("[^0-9a-zA-Z_]+")


def _make_key(obj, suffix: str, use_qualified_name):
    base_name = f"{obj.__module__}.{obj.__qualname__}" if use_qualified_name else obj.__name__
    return _key_re.sub("", f"{base_name}_{suffix}")


def get_func(func, suffix: str, use_qualified_name: bool = False):
    key = _make_key(func, suffix, use_qualified_name)

    if key not in _func_cache:
        _func_cache[key] = wp.Function(
            func=func,
            key=key,
            namespace="",
            module=wp.get_module(
                func.__module__,
            ),
        )

    return _func_cache[key]


def dynamic_func(suffix: str, use_qualified_name=False):
    def wrap_func(func: Callable):
        return get_func(func, suffix=suffix, use_qualified_name=use_qualified_name)

    return wrap_func


def get_kernel(
    func,
    suffix: str,
    use_qualified_name: bool = False,
    kernel_options: Dict[str, Any] = {},
):
    key = _make_key(func, suffix, use_qualified_name)

    if key not in _kernel_cache:
        # Avoid creating too long file names -- can lead to issues on Windows
        # We could hash the key, but prefer to keep it human-readable
        module_name = f"{func.__module__}.dyn.{key}"
        module_name = module_name[:128] if len(module_name) > 128 else module_name
        module = wp.get_module(module_name)
        module.options = copy(wp.get_module(func.__module__).options)
        module.options.update(kernel_options)
        _kernel_cache[key] = wp.Kernel(func=func, key=key, module=module)
    return _kernel_cache[key]


def dynamic_kernel(suffix: str, use_qualified_name=False, kernel_options: Dict[str, Any] = {}):
    def wrap_kernel(func: Callable):
        return get_kernel(func, suffix=suffix, use_qualified_name=use_qualified_name, kernel_options=kernel_options)

    return wrap_kernel


def get_struct(struct: type, suffix: str, use_qualified_name: bool = False):
    key = _make_key(struct, suffix, use_qualified_name)
    # used in codegen
    struct.__qualname__ = key

    if key not in _struct_cache:
        module = wp.get_module(struct.__module__)
        _struct_cache[key] = wp.codegen.Struct(
            cls=struct,
            key=key,
            module=module,
        )

    return _struct_cache[key]


def dynamic_struct(suffix: str, use_qualified_name=False):
    def wrap_struct(struct: type):
        return get_struct(struct, suffix=suffix, use_qualified_name=use_qualified_name)

    return wrap_struct


def get_integrand_function(
    integrand: "warp.fem.operator.Integrand",
    suffix: str,
    func=None,
    annotations=None,
    code_transformers=[],
):
    key = _make_key(integrand.func, suffix, use_qualified_name=True)

    if key not in _func_cache:
        _func_cache[key] = wp.Function(
            func=integrand.func if func is None else func,
            key=key,
            namespace="",
            module=integrand.module,
            overloaded_annotations=annotations,
            code_transformers=code_transformers,
        )

    return _func_cache[key]


def get_integrand_kernel(
    integrand: "warp.fem.operator.Integrand",
    suffix: str,
    kernel_fn: Optional[Callable] = None,
    kernel_options: Dict[str, Any] = {},
    code_transformers=[],
):
    key = _make_key(integrand.func, suffix, use_qualified_name=True)

    if key not in _kernel_cache:
        if kernel_fn is None:
            return None

        module = wp.get_module(f"{integrand.module.name}.{integrand.name}")
        module.options = copy(integrand.module.options)
        module.options.update(kernel_options)

        _kernel_cache[key] = wp.Kernel(func=kernel_fn, key=key, module=module, code_transformers=code_transformers)
    return _kernel_cache[key]


def cached_arg_value(func: Callable):
    """Decorator to be applied to member methods assembling Arg structs, so that the result gets
    automatically cached for the lifetime of the parent object
    """

    cache_attr = f"_{func.__name__}_cache"

    def get_arg(obj, device):
        if not hasattr(obj, cache_attr):
            setattr(obj, cache_attr, {})

        cache = getattr(obj, cache_attr, {})

        device = wp.get_device(device)
        if device.ordinal not in cache:
            cache[device.ordinal] = func(obj, device)

        return cache[device.ordinal]

    return get_arg


_cached_vec_types = {}
_cached_mat_types = {}


def cached_vec_type(length, dtype):
    key = (length, dtype)
    if key not in _cached_vec_types:
        _cached_vec_types[key] = wp.vec(length=length, dtype=dtype)

    return _cached_vec_types[key]


def cached_mat_type(shape, dtype):
    key = (*shape, dtype)
    if key not in _cached_mat_types:
        _cached_mat_types[key] = wp.mat(shape=shape, dtype=dtype)

    return _cached_mat_types[key]


class Temporary:
    """Handle over a temporary array from a :class:`TemporaryStore`.

    The array will be automatically returned to the temporary pool for reuse upon destruction of this object, unless
    the temporary is explicitly detached from the pool using :meth:`detach`.
    The temporary may also be explicitly returned to the pool before destruction using :meth:`release`.
    """

    def __init__(self, array: wp.array, pool: Optional["TemporaryStore.Pool"] = None, shape=None, dtype=None):
        self._raw_array = array
        self._array_view = array
        self._pool = pool

        if shape is not None or dtype is not None:
            self._view_as(shape=shape, dtype=dtype)

    def detach(self) -> wp.array:
        """Detaches the temporary so it is never returned to the pool"""
        if self._pool is not None:
            self._pool.detach(self._raw_array)

        self._pool = None
        return self._array_view

    def release(self):
        """Returns the temporary array to the pool"""
        if self._pool is not None:
            self._pool.redeem(self._raw_array)

        self._pool = None

    @property
    def array(self) -> wp.array:
        """View of the array with desired shape and data type."""
        return self._array_view

    def _view_as(self, shape, dtype) -> "Temporary":
        def _view_reshaped_truncated(array):
            return wp.types.array(
                ptr=array.ptr,
                dtype=dtype,
                shape=shape,
                device=array.device,
                pinned=array.pinned,
                capacity=array.capacity,
                copy=False,
                grad=None if array.grad is None else _view_reshaped_truncated(array.grad),
            )

        self._array_view = _view_reshaped_truncated(self._raw_array)
        return self

    def __del__(self):
        self.release()


class TemporaryStore:
    """
    Shared pool of temporary arrays that will be persisted and reused across invocations of ``warp.fem`` functions.

    A :class:`TemporaryStore` instance may either be passed explicitly to ``warp.fem`` functions that accept such an argument, for instance :func:`.integrate.integrate`,
    or can be set globally as the default store using :func:`set_default_temporary_store`.

    By default, there is no default temporary store, so that temporary allocations are not persisted.
    """

    _default_store: "TemporaryStore" = None

    class Pool:
        def __init__(self, dtype, device, pinned: bool):
            self.dtype = dtype
            self.device = device
            self.pinned = pinned

            self._pool = []  # Currently available arrays for borrowing, ordered by size
            self._pool_sizes = []  # Sizes of available arrays for borrowing, ascending
            self._allocs = {}  # All allocated arrays, including borrowed ones

        def borrow(self, shape, dtype, requires_grad: bool):
            size = 1
            if isinstance(shape, int):
                shape = (shape,)
            for d in shape:
                size *= d

            index = bisect.bisect_left(
                a=self._pool_sizes,
                x=size,
            )
            if index < len(self._pool):
                # Big enough array found, remove from pool
                array = self._pool.pop(index)
                self._pool_sizes.pop(index)
                if requires_grad and array.grad is None:
                    array.requires_grad = True
                return Temporary(pool=self, array=array, shape=shape, dtype=dtype)

            # No big enough array found, allocate new one
            if len(self._pool) > 0:
                grow_factor = 1.5
                size = max(int(self._pool_sizes[-1] * grow_factor), size)

            array = wp.empty(
                shape=(size,), dtype=self.dtype, pinned=self.pinned, device=self.device, requires_grad=requires_grad
            )
            self._allocs[array.ptr] = array
            return Temporary(pool=self, array=array, shape=shape, dtype=dtype)

        def redeem(self, array):
            # Insert back array into available pool
            index = bisect.bisect_left(
                a=self._pool_sizes,
                x=array.size,
            )
            self._pool.insert(index, array)
            self._pool_sizes.insert(index, array.size)

        def detach(self, array):
            del self._allocs[array.ptr]

    def __init__(self):
        self.clear()

    def clear(self):
        self._temporaries = {}

    def borrow(self, shape, dtype, pinned: bool = False, device=None, requires_grad: bool = False) -> Temporary:
        dtype = wp.types.type_to_warp(dtype)
        device = wp.get_device(device)

        type_length = wp.types.type_length(dtype)
        key = (dtype._type_, type_length, pinned, device.ordinal)

        pool = self._temporaries.get(key, None)
        if pool is None:
            value_type = (
                cached_vec_type(length=type_length, dtype=wp.types.type_scalar_type(dtype))
                if type_length > 1
                else dtype
            )
            pool = TemporaryStore.Pool(value_type, device, pinned=pinned)
            self._temporaries[key] = pool

        return pool.borrow(dtype=dtype, shape=shape, requires_grad=requires_grad)


def set_default_temporary_store(temporary_store: Optional[TemporaryStore]):
    """Globally sets the default :class:`TemporaryStore` instance to use for temporary allocations in ``warp.fem`` functions.

    If the default temporary store is set to ``None``, temporary allocations are not persisted unless a :class:`TemporaryStore` is provided at a per-function granularity.
    """

    TemporaryStore._default_store = temporary_store


def borrow_temporary(
    temporary_store: Optional[TemporaryStore],
    shape: Union[int, Tuple[int]],
    dtype: type,
    pinned: bool = False,
    requires_grad: bool = False,
    device=None,
) -> Temporary:
    """
    Borrows and returns a temporary array with specified attributes from a shared pool.

    If an array with sufficient capacity and matching desired attributes is already available in the pool, it will be returned.
    Otherwise, a new allocation will be performed.

    Args:
        temporary_store: the shared pool to borrow the temporary from. If `temporary_store` is ``None``, the global default temporary store, if set, will be used.
        shape: desired dimensions for the temporary array
        dtype: desired data type for the temporary array
        pinned: whether a pinned allocation is desired
        device: device on which the memory should be allocated; if ``None``, the current device will be used.
    """

    if temporary_store is None:
        temporary_store = TemporaryStore._default_store

    if temporary_store is None:
        return Temporary(
            array=wp.empty(shape=shape, dtype=dtype, pinned=pinned, device=device, requires_grad=requires_grad)
        )

    return temporary_store.borrow(shape=shape, dtype=dtype, device=device, pinned=pinned, requires_grad=requires_grad)


def borrow_temporary_like(
    array: Union[wp.array, Temporary],
    temporary_store: Optional[TemporaryStore],
) -> Temporary:
    """
    Borrows and returns a temporary array with the same attributes as another array or temporary.

    Args:
        array: Warp or temporary array to read the desired attributes from
        temporary_store: the shared pool to borrow the temporary from. If `temporary_store` is ``None``, the global default temporary store, if set, will be used.
    """
    if isinstance(array, Temporary):
        array = array.array
    return borrow_temporary(
        temporary_store=temporary_store,
        shape=array.shape,
        dtype=array.dtype,
        pinned=array.pinned,
        device=array.device,
        requires_grad=array.requires_grad,
    )
