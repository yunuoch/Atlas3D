from typing import Union
from enum import Enum

import warp as wp
import warp.codegen
import warp.context

from warp.fem.geometry import (
    Element,
    Geometry,
    GeometryPartition,
    WholeGeometryPartition,
)

GeometryOrPartition = Union[Geometry, GeometryPartition]


class GeometryDomain:
    """Interface class for domains, i.e. (partial) views of elements in a Geometry"""

    class ElementKind(Enum):
        """Possible kinds of elements contained in a domain"""

        CELL = 0
        SIDE = 1

    def __init__(self, geometry: GeometryOrPartition):
        if isinstance(geometry, GeometryPartition):
            self.geometry_partition = geometry
        else:
            self.geometry_partition = WholeGeometryPartition(geometry)

        self.geometry = self.geometry_partition.geometry

    @property
    def name(self) -> str:
        return f"{self.geometry_partition.name}_{self.__class__.__name__}"

    def __str__(self) -> str:
        return self.name

    def __eq__(self, other) -> bool:
        return self.__class__ == other.__class__ and self.geometry_partition == other.geometry_partition

    @property
    def element_kind(self) -> ElementKind:
        """Kind of elements that this domain contains (cells or sides)"""
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        """Dimension of the elements of the domain"""
        raise NotImplementedError

    def element_count(self) -> int:
        """Number of elements in the domain"""
        raise NotImplementedError

    def geometry_element_count(self) -> int:
        """Number of elements in the underlying geometry"""
        return self.geometry.cell_count()

    def reference_element(self) -> Element:
        """Protypical element"""
        raise NotImplementedError

    def element_index_arg_value(self, device: warp.context.Devicelike) -> warp.codegen.StructInstance:
        """Value of the argument to be passed to device functions"""
        raise NotImplementedError

    def element_arg_value(self, device: warp.context.Devicelike) -> warp.codegen.StructInstance:
        """Value of the argument to be passed to device functions"""
        raise NotImplementedError

    ElementIndexArg: warp.codegen.Struct
    """Structure containing arguments to be passed to device functions computing element indices"""

    element_index: wp.Function
    """Device function for retrieving an ElementIndex from a linearized index"""

    ElementArg: warp.codegen.Struct
    """Structure containing arguments to be passed to device functions computing element geometry"""

    element_measure: wp.Function
    """Device function returning the measure determinant (e.g. volume, area) at a given point"""

    element_measure_ratio: wp.Function
    """Device function returning the ratio of the measure of a side to that of its neighbour cells"""

    element_position: wp.Function
    """Device function returning the element position at a sample point"""

    element_deformation_gradient: wp.Function
    """Device function returning the gradient of the position with respect to the element's reference space"""

    element_normal: wp.Function
    """Device function returning the element normal at a sample point"""

    element_lookup: wp.Function
    """Device function returning the sample point corresponding to a world position"""


class Cells(GeometryDomain):
    """A Domain containing all cells of the geometry or geometry partition"""

    def __init__(self, geometry: GeometryOrPartition):
        super().__init__(geometry)

    @property
    def element_kind(self) -> GeometryDomain.ElementKind:
        return GeometryDomain.ElementKind.CELL

    @property
    def dimension(self) -> int:
        return self.geometry.dimension

    def reference_element(self) -> Element:
        return self.geometry.reference_cell()

    def element_count(self) -> int:
        return self.geometry_partition.cell_count()

    def geometry_element_count(self) -> int:
        return self.geometry.cell_count()

    @property
    def ElementIndexArg(self) -> warp.codegen.Struct:
        return self.geometry_partition.CellArg

    def element_index_arg_value(self, device: warp.context.Devicelike) -> warp.codegen.StructInstance:
        return self.geometry_partition.cell_arg_value(device)

    @property
    def element_index(self) -> wp.Function:
        return self.geometry_partition.cell_index

    def element_arg_value(self, device: warp.context.Devicelike) -> warp.codegen.StructInstance:
        return self.geometry.cell_arg_value(device)

    @property
    def ElementArg(self) -> warp.codegen.Struct:
        return self.geometry.CellArg

    @property
    def element_position(self) -> wp.Function:
        return self.geometry.cell_position

    @property
    def element_deformation_gradient(self) -> wp.Function:
        return self.geometry.cell_deformation_gradient

    @property
    def element_measure(self) -> wp.Function:
        return self.geometry.cell_measure

    @property
    def element_measure_ratio(self) -> wp.Function:
        return self.geometry.cell_measure_ratio

    @property
    def eval_normal(self) -> wp.Function:
        return self.geometry.cell_normal

    @property
    def element_lookup(self) -> wp.Function:
        return self.geometry.cell_lookup


class Sides(GeometryDomain):
    """A Domain containing all (interior and boundary) sides of the geometry or geometry partition"""

    def __init__(self, geometry: GeometryOrPartition):
        self.geometry = geometry
        super().__init__(geometry)

    @property
    def element_kind(self) -> GeometryDomain.ElementKind:
        return GeometryDomain.ElementKind.SIDE

    @property
    def dimension(self) -> int:
        return self.geometry.dimension - 1

    def reference_element(self) -> Element:
        return self.geometry.reference_side()

    def element_count(self) -> int:
        return self.geometry_partition.side_count()

    def geometry_element_count(self) -> int:
        return self.geometry.side_count()

    @property
    def ElementIndexArg(self) -> warp.codegen.Struct:
        return self.geometry_partition.SideArg

    def element_index_arg_value(self, device: warp.context.Devicelike) -> warp.codegen.StructInstance:
        return self.geometry_partition.side_arg_value(device)

    @property
    def element_index(self) -> wp.Function:
        return self.geometry_partition.side_index

    @property
    def ElementArg(self) -> warp.codegen.Struct:
        return self.geometry.SideArg

    def element_arg_value(self, device: warp.context.Devicelike) -> warp.codegen.StructInstance:
        return self.geometry.side_arg_value(device)

    @property
    def element_position(self) -> wp.Function:
        return self.geometry.side_position

    @property
    def element_deformation_gradient(self) -> wp.Function:
        return self.geometry.side_deformation_gradient

    @property
    def element_measure(self) -> wp.Function:
        return self.geometry.side_measure

    @property
    def element_measure_ratio(self) -> wp.Function:
        return self.geometry.side_measure_ratio

    @property
    def eval_normal(self) -> wp.Function:
        return self.geometry.side_normal


class BoundarySides(Sides):
    """A Domain containing boundary sides of the geometry or geometry partition"""

    def __init__(self, geometry: GeometryOrPartition):
        super().__init__(geometry)

    def element_count(self) -> int:
        return self.geometry_partition.boundary_side_count()

    def geometry_element_count(self) -> int:
        return self.geometry.boundary_side_count()

    @property
    def element_index(self) -> wp.Function:
        return self.geometry_partition.boundary_side_index


class FrontierSides(Sides):
    """A Domain containing frontier sides of the geometry partition (sides shared with at least another partition)"""

    def __init__(self, geometry: GeometryOrPartition):
        super().__init__(geometry)

    def element_count(self) -> int:
        return self.geometry_partition.frontier_side_count()

    def geometry_element_count(self) -> int:
        raise RuntimeError("Frontier sides not defined at the geometry level")

    @property
    def element_index(self) -> wp.Function:
        return self.geometry_partition.frontier_side_index
