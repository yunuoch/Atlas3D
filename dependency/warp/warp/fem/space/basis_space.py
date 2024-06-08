from typing import Optional

import warp as wp

from warp.fem.types import ElementIndex, Coords, make_free_sample
from warp.fem.geometry import Geometry
from warp.fem.quadrature import Quadrature
from warp.fem import cache

from .topology import SpaceTopology, DiscontinuousSpaceTopology
from .shape import ShapeFunction


class BasisSpace:
    """Interface class for defining a scalar-valued basis over a geometry.

    A basis space makes it easy to define multiple function spaces sharing the same basis (and thus nodes) but with different valuation functions;
    however, it is not a required ingredient of a function space.

    See also: :func:`make_polynomial_basis_space`, :func:`make_collocated_function_space`
    """

    @wp.struct
    class BasisArg:
        """Argument structure to be passed to device functions"""

        pass

    def __init__(self, topology: SpaceTopology):
        self._topology = topology

        self.NODES_PER_ELEMENT = self._topology.NODES_PER_ELEMENT

    @property
    def topology(self) -> SpaceTopology:
        """Underlying topology of the basis space"""
        return self._topology

    @property
    def geometry(self) -> Geometry:
        """Underlying geometry of the basis space"""
        return self._topology.geometry

    def basis_arg_value(self, device) -> "BasisArg":
        """Value for the argument structure to be passed to device functions"""
        return BasisSpace.BasisArg()

    # Helpers for generating node positions

    def node_positions(self, out: Optional[wp.array] = None) -> wp.array:
        """Returns a temporary array containing the world position for each node"""

        NODES_PER_ELEMENT = self.NODES_PER_ELEMENT

        pos_type = cache.cached_vec_type(length=self.geometry.dimension, dtype=float)

        node_coords_in_element = self.make_node_coords_in_element()

        @cache.dynamic_kernel(suffix=self.name, kernel_options={"max_unroll": 4, "enable_backward": False})
        def fill_node_positions(
            geo_cell_arg: self.geometry.CellArg,
            basis_arg: self.BasisArg,
            topo_arg: self.topology.TopologyArg,
            node_positions: wp.array(dtype=pos_type),
        ):
            element_index = wp.tid()

            for n in range(NODES_PER_ELEMENT):
                node_index = self.topology.element_node_index(geo_cell_arg, topo_arg, element_index, n)
                coords = node_coords_in_element(geo_cell_arg, basis_arg, element_index, n)

                sample = make_free_sample(element_index, coords)
                pos = self.geometry.cell_position(geo_cell_arg, sample)

                node_positions[node_index] = pos

        shape = (self.topology.node_count(),)
        if out is None:
            node_positions = wp.empty(
                shape=shape,
                dtype=pos_type,
            )
        else:
            if out.shape != shape or not wp.types.types_equal(pos_type, out.dtype):
                raise ValueError(
                    f"Out node positions array must have shape {shape} and data type {wp.types.type_repr(pos_type)}"
                )
            node_positions = out

        wp.launch(
            dim=self.geometry.cell_count(),
            kernel=fill_node_positions,
            inputs=[
                self.geometry.cell_arg_value(device=node_positions.device),
                self.basis_arg_value(device=node_positions.device),
                self.topology.topo_arg_value(device=node_positions.device),
                node_positions,
            ],
        )

        return node_positions

    def make_node_coords_in_element(self):
        raise NotImplementedError()

    def make_node_quadrature_weight(self):
        raise NotImplementedError()

    def make_element_inner_weight(self):
        raise NotImplementedError()

    def make_element_outer_weight(self):
        return self.make_element_inner_weight()

    def make_element_inner_weight_gradient(self):
        raise NotImplementedError()

    def make_element_outer_weight_gradient(self):
        return self.make_element_inner_weight_gradient()

    def make_trace_node_quadrature_weight(self):
        raise NotImplementedError()

    def trace(self) -> "TraceBasisSpace":
        return TraceBasisSpace(self)


class ShapeBasisSpace(BasisSpace):
    """Base class for defining shape-function-based basis spaces."""

    def __init__(self, topology: SpaceTopology, shape: ShapeFunction):
        super().__init__(topology)
        self._shape = shape

        self.ORDER = self._shape.ORDER

        if hasattr(shape, "element_node_triangulation"):
            self.node_triangulation = self._node_triangulation
        if hasattr(shape, "element_node_tets"):
            self.node_tets = self._node_tets
        if hasattr(shape, "element_node_hexes"):
            self.node_hexes = self._node_hexes

    @property
    def shape(self) -> ShapeFunction:
        """Shape functions used for defining individual element basis"""
        return self._shape

    @property
    def name(self):
        return f"{self.topology.name}_{self._shape.name}"

    def make_node_coords_in_element(self):
        shape_node_coords_in_element = self._shape.make_node_coords_in_element()

        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            elt_arg: self.geometry.CellArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return shape_node_coords_in_element(node_index_in_elt)

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        shape_node_quadrature_weight = self._shape.make_node_quadrature_weight()

        @cache.dynamic_func(suffix=self.name)
        def node_quadrature_weight(
            elt_arg: self.geometry.CellArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return shape_node_quadrature_weight(node_index_in_elt)

        return node_quadrature_weight

    def make_element_inner_weight(self):
        shape_element_inner_weight = self._shape.make_element_inner_weight()

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            elt_arg: self.geometry.CellArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            return shape_element_inner_weight(coords, node_index_in_elt)

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        shape_element_inner_weight_gradient = self._shape.make_element_inner_weight_gradient()

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            elt_arg: self.geometry.CellArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            return shape_element_inner_weight_gradient(coords, node_index_in_elt)

        return element_inner_weight_gradient

    def make_trace_node_quadrature_weight(self, trace_basis):
        shape_trace_node_quadrature_weight = self._shape.make_trace_node_quadrature_weight()

        @cache.dynamic_func(suffix=self.name)
        def trace_node_quadrature_weight(
            geo_side_arg: trace_basis.geometry.SideArg,
            basis_arg: trace_basis.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            neighbour_elem, index_in_neighbour = trace_basis.topology.neighbor_cell_index(
                geo_side_arg, element_index, node_index_in_elt
            )
            return shape_trace_node_quadrature_weight(index_in_neighbour)

        return trace_node_quadrature_weight

    def _node_triangulation(self):
        element_node_indices = self._topology.element_node_indices().numpy()
        element_triangles = self._shape.element_node_triangulation()

        tri_indices = element_node_indices[:, element_triangles].reshape(-1, 3)
        return tri_indices

    def _node_tets(self):
        element_node_indices = self._topology.element_node_indices().numpy()
        element_tets = self._shape.element_node_tets()

        tet_indices = element_node_indices[:, element_tets].reshape(-1, 4)
        return tet_indices

    def _node_hexes(self):
        element_node_indices = self._topology.element_node_indices().numpy()
        element_hexes = self._shape.element_node_hexes()

        hex_indices = element_node_indices[:, element_hexes].reshape(-1, 8)
        return hex_indices


class TraceBasisSpace(BasisSpace):
    """Auto-generated trace space evaluating the cell-defined basis on the geometry sides"""

    def __init__(self, basis: BasisSpace):
        super().__init__(basis.topology.trace())

        self.ORDER = basis.ORDER

        self._basis = basis
        self.BasisArg = self._basis.BasisArg
        self.basis_arg_value = self._basis.basis_arg_value

    @property
    def name(self):
        return f"{self._basis.name}_Trace"

    def make_node_coords_in_element(self):
        node_coords_in_cell = self._basis.make_node_coords_in_element()

        @cache.dynamic_func(suffix=self._basis.name)
        def trace_node_coords_in_element(
            geo_side_arg: self.geometry.SideArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            neighbour_elem, index_in_neighbour = self.topology.neighbor_cell_index(
                geo_side_arg, element_index, node_index_in_elt
            )
            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            neighbour_coords = node_coords_in_cell(
                geo_cell_arg,
                basis_arg,
                neighbour_elem,
                index_in_neighbour,
            )

            return self.geometry.side_from_cell_coords(geo_side_arg, element_index, neighbour_elem, neighbour_coords)

        return trace_node_coords_in_element

    def make_node_quadrature_weight(self):
        return self._basis.make_trace_node_quadrature_weight(self)

    def make_element_inner_weight(self):
        cell_inner_weight = self._basis.make_element_inner_weight()

        @cache.dynamic_func(suffix=self._basis.name)
        def trace_element_inner_weight(
            geo_side_arg: self.geometry.SideArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            cell_index, index_in_cell = self.topology.inner_cell_index(geo_side_arg, element_index, node_index_in_elt)
            if index_in_cell < 0:
                return 0.0

            cell_coords = self.geometry.side_inner_cell_coords(geo_side_arg, element_index, coords)

            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            return cell_inner_weight(
                geo_cell_arg,
                basis_arg,
                cell_index,
                cell_coords,
                index_in_cell,
            )

        return trace_element_inner_weight

    def make_element_outer_weight(self):
        cell_outer_weight = self._basis.make_element_outer_weight()

        @cache.dynamic_func(suffix=self._basis.name)
        def trace_element_outer_weight(
            geo_side_arg: self.geometry.SideArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            cell_index, index_in_cell = self.topology.outer_cell_index(geo_side_arg, element_index, node_index_in_elt)
            if index_in_cell < 0:
                return 0.0

            cell_coords = self.geometry.side_outer_cell_coords(geo_side_arg, element_index, coords)

            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            return cell_outer_weight(
                geo_cell_arg,
                basis_arg,
                cell_index,
                cell_coords,
                index_in_cell,
            )

        return trace_element_outer_weight

    def make_element_inner_weight_gradient(self):
        cell_inner_weight_gradient = self._basis.make_element_inner_weight_gradient()
        grad_vec_type = wp.vec(length=self.geometry.dimension, dtype=float)

        @cache.dynamic_func(suffix=self._basis.name)
        def trace_element_inner_weight_gradient(
            geo_side_arg: self.geometry.SideArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            cell_index, index_in_cell = self.topology.inner_cell_index(geo_side_arg, element_index, node_index_in_elt)
            if index_in_cell < 0:
                return grad_vec_type(0.0)

            cell_coords = self.geometry.side_inner_cell_coords(geo_side_arg, element_index, coords)
            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            return cell_inner_weight_gradient(geo_cell_arg, basis_arg, cell_index, cell_coords, index_in_cell)

        return trace_element_inner_weight_gradient

    def make_element_outer_weight_gradient(self):
        cell_outer_weight_gradient = self._basis.make_element_outer_weight_gradient()
        grad_vec_type = wp.vec(length=self.geometry.dimension, dtype=float)

        @cache.dynamic_func(suffix=self._basis.name)
        def trace_element_outer_weight_gradient(
            geo_side_arg: self.geometry.SideArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            cell_index, index_in_cell = self.topology.outer_cell_index(geo_side_arg, element_index, node_index_in_elt)
            if index_in_cell < 0:
                return grad_vec_type(0.0)

            cell_coords = self.geometry.side_outer_cell_coords(geo_side_arg, element_index, coords)
            geo_cell_arg = self.geometry.side_to_cell_arg(geo_side_arg)
            return cell_outer_weight_gradient(geo_cell_arg, basis_arg, cell_index, cell_coords, index_in_cell)

        return trace_element_outer_weight_gradient

    def __eq__(self, other: "TraceBasisSpace") -> bool:
        return self._topo == other._topo


class PointBasisSpace(BasisSpace):
    """An unstructured :class:`BasisSpace` that is non-zero at a finite set of points only.

    The node locations and nodal quadrature weights are defined by a :class:`Quadrature` formula.
    """

    def __init__(self, quadrature: Quadrature):
        self._quadrature = quadrature

        if quadrature.points_per_element() is None:
            raise NotImplementedError("Varying number of points per element is not supported yet")

        topology = DiscontinuousSpaceTopology(
            geometry=quadrature.domain.geometry, nodes_per_element=quadrature.points_per_element()
        )
        super().__init__(topology)

        self.BasisArg = quadrature.Arg
        self.basis_arg_value = quadrature.arg_value
        self.ORDER = 0

        self.make_element_outer_weight = self.make_element_inner_weight
        self.make_element_outer_weight_gradient = self.make_element_outer_weight_gradient

    @property
    def name(self):
        return f"{self._quadrature.name}_Point"

    def make_node_coords_in_element(self):
        @cache.dynamic_func(suffix=self.name)
        def node_coords_in_element(
            elt_arg: self._quadrature.domain.ElementArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return self._quadrature.point_coords(elt_arg, basis_arg, element_index, node_index_in_elt)

        return node_coords_in_element

    def make_node_quadrature_weight(self):
        @cache.dynamic_func(suffix=self.name)
        def node_quadrature_weight(
            elt_arg: self._quadrature.domain.ElementArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return self._quadrature.point_weight(elt_arg, basis_arg, element_index, node_index_in_elt)

        return node_quadrature_weight

    def make_element_inner_weight(self):
        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight(
            elt_arg: self._quadrature.domain.ElementArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            qp_coord = self._quadrature.point_coords(elt_arg, basis_arg, element_index, node_index_in_elt)
            return wp.select(wp.length_sq(coords - qp_coord) < 0.001, 0.0, 1.0)

        return element_inner_weight

    def make_element_inner_weight_gradient(self):
        gradient_vec = cache.cached_vec_type(length=self.geometry.dimension, dtype=float)

        @cache.dynamic_func(suffix=self.name)
        def element_inner_weight_gradient(
            elt_arg: self._quadrature.domain.ElementArg,
            basis_arg: self.BasisArg,
            element_index: ElementIndex,
            coords: Coords,
            node_index_in_elt: int,
        ):
            return gradient_vec(0.0)

        return element_inner_weight_gradient

    def make_trace_node_quadrature_weight(self, trace_basis):
        @cache.dynamic_func(suffix=self.name)
        def trace_node_quadrature_weight(
            elt_arg: trace_basis.geometry.SideArg,
            basis_arg: trace_basis.BasisArg,
            element_index: ElementIndex,
            node_index_in_elt: int,
        ):
            return 0.0

        return trace_node_quadrature_weight
