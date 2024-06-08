from typing import Union, Optional

from warp.fem.domain import GeometryDomain, Cells
from warp.fem.space import FunctionSpace, SpaceRestriction, SpacePartition, make_space_partition, make_space_restriction

from .field import DiscreteField, SpaceField, FieldLike
from .restriction import FieldRestriction
from .test import TestField
from .trial import TrialField

from .nodal_field import NodalField


def make_restriction(
    field: DiscreteField,
    space_restriction: Optional[SpaceRestriction] = None,
    domain: Optional[GeometryDomain] = None,
    device=None,
) -> FieldRestriction:
    """
    Restricts a discrete field to a subset of elements.

    Args:
        field: the discrete field to restrict
        space_restriction: the function space restriction defining the subset of elements to consider
        domain: if ``space_restriction`` is not provided, the :py:class:`Domain` defining the subset of elements to consider
        device: Warp device on which to perform and store computations

    Returns:
        the field restriction
    """

    if space_restriction is None:
        space_restriction = make_space_restriction(space_partition=field.space_partition, domain=domain, device=device)

    return FieldRestriction(field=field, space_restriction=space_restriction)


def make_test(
    space: FunctionSpace,
    space_restriction: Optional[SpaceRestriction] = None,
    space_partition: Optional[SpacePartition] = None,
    domain: Optional[GeometryDomain] = None,
    device=None,
) -> TestField:
    """
    Constructs a test field over a function space or its restriction

    Args:
        space: the function space
        space_restriction: restriction of the space topology to a domain
        space_partition: if `space_restriction` is ``None``, the optional subset of node indices to consider
        domain: if `space_restriction` is ``None``, optional subset of elements to consider
        device: Warp device on which to perform and store computations

    Returns:
        the test field
    """

    if space_restriction is None:
        space_restriction = make_space_restriction(
            space_topology=space.topology, space_partition=space_partition, domain=domain, device=device
        )

    return TestField(space_restriction=space_restriction, space=space)


def make_trial(
    space: FunctionSpace,
    space_restriction: Optional[SpaceRestriction] = None,
    space_partition: Optional[SpacePartition] = None,
    domain: Optional[GeometryDomain] = None,
) -> TrialField:
    """
    Constructs a trial field over a function space or partition

    Args:
        space: the function space or function space restriction
        space_restriction: restriction of the space topology to a domain
        space_partition: if `space_restriction` is ``None``, the optional subset of node indices to consider
        domain: if `space_restriction` is ``None``, optional subset of elements to consider
        device: Warp device on which to perform and store computations

    Returns:
        the trial field
    """

    if space_restriction is not None:
        domain = space.domain
        space_partition = space.space_partition

    if space_partition is None:
        if domain is None:
            domain = Cells(geometry=space.geometry)
        space_partition = make_space_partition(
            space_topology=space.topology, geometry_partition=domain.geometry_partition
        )
    elif domain is None:
        domain = Cells(geometry=space_partition.geo_partition)

    return TrialField(space, space_partition, domain)
