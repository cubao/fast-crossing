from __future__ import annotations
import numpy
import typing
from . import tf

__all__ = [
    "Arrow",
    "FastCrossing",
    "FlatBush",
    "KdQuiver",
    "KdTree",
    "LineSegment",
    "PolylineRuler",
    "Quiver",
    "densify_polyline",
    "douglas_simplify",
    "douglas_simplify_indexes",
    "douglas_simplify_mask",
    "intersect_segments",
    "point_in_polygon",
    "polyline_in_polygon",
    "snap_onto_2d",
    "tf",
]

class Arrow:
    @staticmethod
    def _angle(
        vec: numpy.ndarray[numpy.float64[3, 1]],
        *,
        ref: numpy.ndarray[numpy.float64[3, 1]],
    ) -> float:
        """
        Calculate angle between two vectors
        """
    @staticmethod
    @typing.overload
    def _heading(heading: float) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Convert heading to unit vector
        """
    @staticmethod
    @typing.overload
    def _heading(east: float, north: float) -> float:
        """
        Convert east and north components to heading
        """
    @staticmethod
    def _unit_vector(
        vector: numpy.ndarray[numpy.float64[3, 1]], with_eps: bool = True
    ) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Normalize a vector to unit length
        """
    def Frenet(self) -> numpy.ndarray[numpy.float64[3, 3]]:
        """
        Get the Frenet frame of the Arrow
        """
    def __copy__(self, arg0: dict) -> Arrow:
        """
        Create a copy of the Arrow
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor for Arrow
        """
    @typing.overload
    def __init__(self, position: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        """
        Constructor for Arrow with position
        """
    @typing.overload
    def __init__(
        self,
        position: numpy.ndarray[numpy.float64[3, 1]],
        direction: numpy.ndarray[numpy.float64[3, 1]],
    ) -> None:
        """
        Constructor for Arrow with position and direction
        """
    def __repr__(self) -> str: ...
    def copy(self) -> Arrow:
        """
        Create a copy of the Arrow
        """
    @typing.overload
    def direction(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the direction of the Arrow
        """
    @typing.overload
    def direction(self, arg0: numpy.ndarray[numpy.float64[3, 1]]) -> Arrow:
        """
        Set the direction of the Arrow
        """
    def forward(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the forward direction of the Arrow
        """
    def has_index(self, check_range: bool = True) -> bool:
        """
        Check if the Arrow has a valid index
        """
    @typing.overload
    def heading(self) -> float:
        """
        Get the heading of the Arrow
        """
    @typing.overload
    def heading(self, new_value: float) -> Arrow:
        """
        Set the heading of the Arrow
        """
    @typing.overload
    def label(self) -> numpy.ndarray[numpy.int32[2, 1]]:
        """
        Get the label of the Arrow
        """
    @typing.overload
    def label(self, new_value: numpy.ndarray[numpy.int32[2, 1]]) -> Arrow:
        """
        Set the label of the Arrow
        """
    @typing.overload
    def label(
        self,
        polyline_index: int,
        segment_index: int,
        *,
        t: float | None = None,
        range: float | None = None,
    ) -> Arrow:
        """
        Set the label of the Arrow with polyline and segment indices
        """
    def leftward(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the leftward direction of the Arrow
        """
    @typing.overload
    def polyline_index(self) -> int:
        """
        Get the polyline index of the Arrow
        """
    @typing.overload
    def polyline_index(self, new_value: int) -> Arrow:
        """
        Set the polyline index of the Arrow
        """
    @typing.overload
    def position(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the position of the Arrow
        """
    @typing.overload
    def position(self, new_value: numpy.ndarray[numpy.float64[3, 1]]) -> Arrow:
        """
        Set the position of the Arrow
        """
    @typing.overload
    def range(self) -> float:
        """
        Get the range of the Arrow
        """
    @typing.overload
    def range(self, new_value: float) -> Arrow:
        """
        Set the range of the Arrow
        """
    def reset_index(self) -> None:
        """
        Reset the index of the Arrow
        """
    @typing.overload
    def segment_index(self) -> int:
        """
        Get the segment index of the Arrow
        """
    @typing.overload
    def segment_index(self, new_value: int) -> Arrow:
        """
        Set the segment index of the Arrow
        """
    @typing.overload
    def t(self) -> float:
        """
        Get the t parameter of the Arrow
        """
    @typing.overload
    def t(self, new_value: float) -> Arrow:
        """
        Set the t parameter of the Arrow
        """
    def upward(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the upward direction of the Arrow
        """

class FastCrossing:
    def __init__(self, *, is_wgs84: bool = False) -> None:
        """
        Initialize FastCrossing object.

        :param is_wgs84: Whether coordinates are in WGS84 format, defaults to false
        """
    @typing.overload
    def add_polyline(
        self, polyline: numpy.ndarray[numpy.float64[m, 3]], *, index: int = -1
    ) -> int:
        """
        Add polyline to the tree.

        :param polyline: The polyline to add
        :param index: Custom polyline index, defaults to -1
        :return: The index of the added polyline
        """
    @typing.overload
    def add_polyline(
        self,
        polyline: numpy.ndarray[numpy.float64[m, 2], numpy.ndarray.flags.c_contiguous],
        *,
        index: int = -1,
    ) -> int:
        """
        Add polyline to the tree (alternative format).

        :param polyline: The polyline to add
        :param index: Custom polyline index, defaults to -1
        :return: The index of the added polyline
        """
    def arrow(
        self, *, polyline_index: int, point_index: int
    ) -> tuple[numpy.ndarray[numpy.float64[3, 1]], numpy.ndarray[numpy.float64[3, 1]]]:
        """
        Get an arrow (position and direction) at a specific point on a polyline.

        :param polyline_index: Index of the polyline
        :param point_index: Index of the point within the polyline
        :return: Arrow (position and direction)
        """
    def bush(self, autobuild: bool = True) -> ...:
        """
        Export the internal FlatBush index.

        :param autobuild: Whether to automatically build the index if not already built, defaults to true
        :return: FlatBush index
        """
    @typing.overload
    def coordinates(
        self, polyline_index: int, segment_index: int, ratio: float
    ) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get coordinates at a specific position on a polyline.

        :param polyline_index: Index of the polyline
        :param segment_index: Index of the segment within the polyline
        :param ratio: Ratio along the segment (0 to 1)
        :return: Coordinates at the specified position
        """
    @typing.overload
    def coordinates(
        self, index: numpy.ndarray[numpy.int32[2, 1]], ratio: float
    ) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get coordinates at a specific position on a polyline (alternative format).

        :param index: Combined index of polyline and segment
        :param ratio: Ratio along the segment (0 to 1)
        :return: Coordinates at the specified position
        """
    @typing.overload
    def coordinates(
        self,
        intersection: tuple[
            numpy.ndarray[numpy.float64[2, 1]],
            numpy.ndarray[numpy.float64[2, 1]],
            numpy.ndarray[numpy.int32[2, 1]],
            numpy.ndarray[numpy.int32[2, 1]],
        ],
        second: bool = True,
    ) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get coordinates of an intersection.

        :param intersection: The intersection object
        :param second: Whether to use the second polyline of the intersection, defaults to true
        :return: Coordinates of the intersection
        """
    def finish(self) -> None:
        """
        Finalize indexing after adding all polylines
        """
    @typing.overload
    def intersections(
        self,
    ) -> list[
        tuple[
            numpy.ndarray[numpy.float64[2, 1]],
            numpy.ndarray[numpy.float64[2, 1]],
            numpy.ndarray[numpy.int32[2, 1]],
            numpy.ndarray[numpy.int32[2, 1]],
        ]
    ]:
        """
        Get all segment intersections in the tree
        """
    @typing.overload
    def intersections(
        self, *, z_offset_range: tuple[float, float], self_intersection: int = 2
    ) -> list[
        tuple[
            numpy.ndarray[numpy.float64[2, 1]],
            numpy.ndarray[numpy.float64[2, 1]],
            numpy.ndarray[numpy.int32[2, 1]],
            numpy.ndarray[numpy.int32[2, 1]],
        ]
    ]:
        """
        Get segment intersections in the tree, filtered by conditions.

        :param z_offset_range: Z-offset range for filtering
        :param self_intersection: Self-intersection parameter, defaults to 2
        """
    @typing.overload
    def intersections(
        self,
        start: numpy.ndarray[numpy.float64[2, 1]],
        to: numpy.ndarray[numpy.float64[2, 1]],
        *,
        dedup: bool = True,
    ) -> list[
        tuple[
            numpy.ndarray[numpy.float64[2, 1]],
            numpy.ndarray[numpy.float64[2, 1]],
            numpy.ndarray[numpy.int32[2, 1]],
            numpy.ndarray[numpy.int32[2, 1]],
        ]
    ]:
        """
        Get crossing intersections with [start, to] segment.

        :param from: Start point of the segment
        :param to: End point of the segment
        :param dedup: Whether to remove duplicates, defaults to true
        :return: Sorted intersections by t ratio
        """
    @typing.overload
    def intersections(
        self, polyline: numpy.ndarray[numpy.float64[m, 3]], *, dedup: bool = True
    ) -> list[
        tuple[
            numpy.ndarray[numpy.float64[2, 1]],
            numpy.ndarray[numpy.float64[2, 1]],
            numpy.ndarray[numpy.int32[2, 1]],
            numpy.ndarray[numpy.int32[2, 1]],
        ]
    ]:
        """
        Get crossing intersections with a polyline.

        :param polyline: The polyline to check intersections with
        :param dedup: Whether to remove duplicates, defaults to true
        :return: Sorted intersections by t ratio
        """
    @typing.overload
    def intersections(
        self,
        polyline: numpy.ndarray[numpy.float64[m, 2], numpy.ndarray.flags.c_contiguous],
        *,
        dedup: bool = True,
    ) -> list[
        tuple[
            numpy.ndarray[numpy.float64[2, 1]],
            numpy.ndarray[numpy.float64[2, 1]],
            numpy.ndarray[numpy.int32[2, 1]],
            numpy.ndarray[numpy.int32[2, 1]],
        ]
    ]:
        """
        Get crossing intersections with a polyline (alternative format).

        :param polyline: The polyline to check intersections with
        :param dedup: Whether to remove duplicates, defaults to true
        :return: Sorted intersections by t ratio
        """
    @typing.overload
    def intersections(
        self,
        polyline: numpy.ndarray[numpy.float64[m, 3]],
        *,
        z_min: float,
        z_max: float,
        dedup: bool = True,
    ) -> list[
        tuple[
            numpy.ndarray[numpy.float64[2, 1]],
            numpy.ndarray[numpy.float64[2, 1]],
            numpy.ndarray[numpy.int32[2, 1]],
            numpy.ndarray[numpy.int32[2, 1]],
        ]
    ]:
        """
        Get crossing intersections with a polyline, filtered by Z range.

        :param polyline: The polyline to check intersections with
        :param z_min: Minimum Z value for filtering
        :param z_max: Maximum Z value for filtering
        :param dedup: Whether to remove duplicates, defaults to true
        :return: Sorted intersections by t ratio
        """
    @typing.overload
    def intersections(
        self,
        polyline: numpy.ndarray[numpy.float64[m, 2], numpy.ndarray.flags.c_contiguous],
        *,
        z_min: float,
        z_max: float,
        dedup: bool = True,
    ) -> list[
        tuple[
            numpy.ndarray[numpy.float64[2, 1]],
            numpy.ndarray[numpy.float64[2, 1]],
            numpy.ndarray[numpy.int32[2, 1]],
            numpy.ndarray[numpy.int32[2, 1]],
        ]
    ]:
        """
        Get crossing intersections with a polyline, filtered by Z range (alternative format).

        :param polyline: The polyline to check intersections with
        :param z_min: Minimum Z value for filtering
        :param z_max: Maximum Z value for filtering
        :param dedup: Whether to remove duplicates, defaults to true
        :return: Sorted intersections by t ratio
        """
    def is_wgs84(self) -> bool:
        """
        Check if the coordinates are in WGS84 format.

        :return: True if coordinates are in WGS84 format, False otherwise
        """
    @typing.overload
    def nearest(
        self,
        position: numpy.ndarray[numpy.float64[3, 1]],
        *,
        return_squared_l2: bool = False,
    ) -> tuple[numpy.ndarray[numpy.int32[2, 1]], float]:
        """
        Find the nearest point to a given position.

        :param position: The query position
        :param return_squared_l2: Whether to return squared L2 distance, defaults to false
        :return: Nearest point information
        """
    @typing.overload
    def nearest(
        self,
        index: numpy.ndarray[numpy.int32[2, 1]],
        *,
        return_squared_l2: bool = False,
    ) -> tuple[numpy.ndarray[numpy.int32[2, 1]], float]:
        """
        Find the nearest point to a given index.

        :param index: The query index
        :param return_squared_l2: Whether to return squared L2 distance, defaults to false
        :return: Nearest point information
        """
    @typing.overload
    def nearest(
        self,
        position: numpy.ndarray[numpy.float64[3, 1]],
        *,
        k: int | None = None,
        radius: float | None = None,
        sort: bool = True,
        return_squared_l2: bool = False,
        filter: tuple[numpy.ndarray[numpy.float64[3, 1]], ...] | None = None,
    ) -> tuple[numpy.ndarray[numpy.int32[m, 2]], numpy.ndarray[numpy.float64[m, 1]]]:
        """
        Find k nearest points to a given position with optional filtering.

        :param position: The query position
        :param k: Number of nearest neighbors to find (optional)
        :param radius: Search radius (optional)
        :param sort: Whether to sort the results, defaults to true
        :param return_squared_l2: Whether to return squared L2 distance, defaults to false
        :param filter: Optional filter parameters
        :return: Nearest points information
        """
    def num_polylines(self) -> int:
        """
        Get the number of polylines in the FastCrossing object.

        :return: Number of polylines
        """
    @typing.overload
    def point_index(self, index: int) -> numpy.ndarray[numpy.int32[2, 1]]:
        """
        Get point index for a given index.

        :param index: The index to query
        :return: The point index
        """
    @typing.overload
    def point_index(
        self, indexes: numpy.ndarray[numpy.int32[m, 1]]
    ) -> list[numpy.ndarray[numpy.int32[2, 1]]]:
        """
        Get point indexes for given indexes.

        :param indexes: The indexes to query
        :return: The point indexes
        """
    def polyline_ruler(self, index: int) -> ...:
        """
        Get a specific polyline ruler.

        :param index: Index of the polyline
        :return: Polyline ruler for the specified index
        """
    def polyline_rulers(self) -> dict[int, ...]:
        """
        Get all polyline rulers.

        :return: Dictionary of polyline rulers
        """
    def quiver(self) -> ...:
        """
        Export the internal Quiver object.

        :return: Quiver object
        """
    @typing.overload
    def segment_index(self, index: int) -> numpy.ndarray[numpy.int32[2, 1]]:
        """
        Get segment index for a given index.

        :param index: The index to query
        :return: The segment index
        """
    @typing.overload
    def segment_index(
        self, indexes: numpy.ndarray[numpy.int32[m, 1]]
    ) -> list[numpy.ndarray[numpy.int32[2, 1]]]:
        """
        Get segment indexes for given indexes.

        :param indexes: The indexes to query
        :return: The segment indexes
        """
    @typing.overload
    def within(
        self,
        *,
        min: numpy.ndarray[numpy.float64[2, 1]],
        max: numpy.ndarray[numpy.float64[2, 1]],
        segment_wise: bool = True,
        sort: bool = True,
    ) -> list[numpy.ndarray[numpy.int32[2, 1]]]:
        """
        Find polylines within a bounding box.

        :param min: Minimum corner of the bounding box
        :param max: Maximum corner of the bounding box
        :param segment_wise: Whether to return segment-wise results, defaults to true
        :param sort: Whether to sort the results, defaults to true
        :return: Polylines within the bounding box
        """
    @typing.overload
    def within(
        self,
        *,
        polygon: numpy.ndarray[numpy.float64[m, 2], numpy.ndarray.flags.c_contiguous],
        segment_wise: bool = True,
        sort: bool = True,
    ) -> list[numpy.ndarray[numpy.int32[2, 1]]]:
        """
        Find polylines within a polygon.

        :param polygon: The polygon to check against
        :param segment_wise: Whether to return segment-wise results, defaults to true
        :param sort: Whether to sort the results, defaults to true
        :return: Polylines within the polygon
        """
    @typing.overload
    def within(
        self,
        *,
        center: numpy.ndarray[numpy.float64[2, 1]],
        width: float,
        height: float,
        heading: float = 0.0,
        segment_wise: bool = True,
        sort: bool = True,
    ) -> list[numpy.ndarray[numpy.int32[2, 1]]]:
        """
        Find polylines within a rotated rectangle.

        :param center: Center of the rectangle
        :param width: Width of the rectangle
        :param height: Height of the rectangle
        :param heading: Heading angle of the rectangle, defaults to 0.0
        :param segment_wise: Whether to return segment-wise results, defaults to true
        :param sort: Whether to sort the results, defaults to true
        :return: Polylines within the rotated rectangle
        """

class FlatBush:
    @typing.overload
    def __init__(self) -> None:
        """
        Initialize an empty FlatBush index.
        """
    @typing.overload
    def __init__(self, reserve: int) -> None:
        """
        Initialize a FlatBush index with a reserved capacity.

        :param reserve: Number of items to reserve space for
        """
    @typing.overload
    def add(
        self,
        minX: float,
        minY: float,
        maxX: float,
        maxY: float,
        *,
        label0: int = -1,
        label1: int = -1,
    ) -> int:
        """
        Add a bounding box to the index.

        :param minX: Minimum X coordinate of the bounding box
        :param minY: Minimum Y coordinate of the bounding box
        :param maxX: Maximum X coordinate of the bounding box
        :param maxY: Maximum Y coordinate of the bounding box
        :param label0: First label (optional)
        :param label1: Second label (optional)
        :return: Index of the added item
        """
    @typing.overload
    def add(
        self,
        polyline: numpy.ndarray[numpy.float64[m, 2], numpy.ndarray.flags.c_contiguous],
        *,
        label0: int = -1,
    ) -> int:
        """
        Add a polyline to the index.

        :param polyline: Polyline coordinates
        :param label0: Label for the polyline (optional)
        :return: Index of the added item
        """
    @typing.overload
    def add(
        self,
        box: numpy.ndarray[numpy.float64[4, 1]],
        *,
        label0: int = -1,
        label1: int = -1,
    ) -> int:
        """
        Add a bounding box to the index using a vector.

        :param box: Vector of [minX, minY, maxX, maxY]
        :param label0: First label (optional)
        :param label1: Second label (optional)
        :return: Index of the added item
        """
    def box(self, index: int) -> numpy.ndarray[numpy.float64[4, 1]]:
        """
        Get the bounding box for a specific index.

        :param index: Index of the item
        :return: Bounding box of the item
        """
    def boxes(
        self,
    ) -> numpy.ndarray[numpy.float64[m, 4], numpy.ndarray.flags.c_contiguous]:
        """
        Get all bounding boxes in the index.

        :return: Reference to the vector of bounding boxes
        """
    def finish(self) -> None:
        """
        Finish the index construction.
        """
    def label(self, index: int) -> numpy.ndarray[numpy.int32[2, 1]]:
        """
        Get the label for a specific index.

        :param index: Index of the item
        :return: Label of the item
        """
    def labels(
        self,
    ) -> numpy.ndarray[numpy.int32[m, 2], numpy.ndarray.flags.c_contiguous]:
        """
        Get all labels in the index.

        :return: Reference to the vector of labels
        """
    def reserve(self, arg0: int) -> None:
        """
        Reserve space for a number of items.

        :param n: Number of items to reserve space for
        """
    @typing.overload
    def search(self, minX: float, minY: float, maxX: float, maxY: float) -> list[int]:
        """
        Search for items within a bounding box.

        :param minX: Minimum X coordinate of the search box
        :param minY: Minimum Y coordinate of the search box
        :param maxX: Maximum X coordinate of the search box
        :param maxY: Maximum Y coordinate of the search box
        :return: Vector of indices of items within the search box
        """
    @typing.overload
    def search(self, bbox: numpy.ndarray[numpy.float64[4, 1]]) -> list[int]:
        """
        Search for items within a bounding box using a vector.

        :param bbox: Vector of [minX, minY, maxX, maxY]
        :return: Vector of indices of items within the search box
        """
    @typing.overload
    def search(
        self,
        min: numpy.ndarray[numpy.float64[2, 1]],
        max: numpy.ndarray[numpy.float64[2, 1]],
    ) -> list[int]:
        """
        Search for items within a bounding box using min and max vectors.

        :param min: Vector of [minX, minY]
        :param max: Vector of [maxX, maxY]
        :return: Vector of indices of items within the search box
        """
    def size(self) -> int:
        """
        Get the number of items in the index.

        :return: Number of items in the index
        """

class KdQuiver(Quiver):
    @staticmethod
    def _filter(
        *,
        arrows: list[Arrow],
        arrow: Arrow,
        params: Quiver.FilterParams,
        is_wgs84: bool = False,
    ) -> numpy.ndarray[numpy.int32[m, 1]]:
        """
        Filter arrows based on the given parameters
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor for KdQuiver
        """
    @typing.overload
    def __init__(self, anchor_lla: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        """
        Constructor for KdQuiver with anchor LLA coordinates
        """
    @typing.overload
    def add(self, polyline: numpy.ndarray[numpy.float64[m, 3]], index: int = -1) -> int:
        """
        Add a polyline to the KdQuiver
        """
    @typing.overload
    def add(
        self,
        polyline: numpy.ndarray[numpy.float64[m, 2], numpy.ndarray.flags.c_contiguous],
        index: int = -1,
    ) -> int:
        """
        Add a 2D polyline to the KdQuiver
        """
    @typing.overload
    def arrow(self, point_index: int) -> Arrow:
        """
        Get the arrow at the given point index
        """
    @typing.overload
    def arrow(self, polyline_index: int, segment_index: int) -> Arrow:
        """
        Get the arrow at the given polyline and segment indices
        """
    @typing.overload
    def arrow(self, polyline_index: int, segment_index: int, *, t: float) -> Arrow:
        """
        Get the arrow at the given polyline, segment indices, and t parameter
        """
    @typing.overload
    def arrow(self, polyline_index: int, *, range: float) -> Arrow:
        """
        Get the arrow at the given polyline index and range
        """
    def arrows(self, indexes: numpy.ndarray[numpy.int32[m, 1]]) -> list[Arrow]:
        """
        Get arrows for the given indexes
        """
    def directions(
        self, indexes: numpy.ndarray[numpy.int32[m, 1]]
    ) -> numpy.ndarray[numpy.float64[m, 3]]:
        """
        Get directions for the given indexes
        """
    @typing.overload
    def filter(
        self,
        *,
        hits: numpy.ndarray[numpy.int32[m, 1]],
        arrow: Arrow,
        params: Quiver.FilterParams,
    ) -> numpy.ndarray[numpy.int32[m, 1]]:
        """
        Filter hits based on the given parameters
        """
    @typing.overload
    def filter(
        self,
        *,
        hits: numpy.ndarray[numpy.int32[m, 1]],
        norms: numpy.ndarray[numpy.float64[m, 1]],
        arrow: Arrow,
        params: Quiver.FilterParams,
    ) -> tuple[numpy.ndarray[numpy.int32[m, 1]], numpy.ndarray[numpy.float64[m, 1]]]:
        """
        Filter hits and norms based on the given parameters
        """
    @typing.overload
    def index(self, point_index: int) -> numpy.ndarray[numpy.int32[2, 1]]:
        """
        Get the index for the given point index
        """
    @typing.overload
    def index(self, polyline_index: int, segment_index: int) -> int:
        """
        Get the index for the given polyline and segment indices
        """
    @typing.overload
    def nearest(
        self,
        position: numpy.ndarray[numpy.float64[3, 1]],
        *,
        return_squared_l2: bool = False,
    ) -> tuple[int, float]:
        """
        Find the nearest point to the given position
        """
    @typing.overload
    def nearest(
        self, index: int, *, return_squared_l2: bool = False
    ) -> tuple[int, float]:
        """
        Find the nearest point to the point at the given index
        """
    @typing.overload
    def nearest(
        self,
        position: numpy.ndarray[numpy.float64[3, 1]],
        *,
        k: int,
        sort: bool = True,
        return_squared_l2: bool = False,
    ) -> tuple[numpy.ndarray[numpy.int32[m, 1]], numpy.ndarray[numpy.float64[m, 1]]]:
        """
        Find k nearest points to the given position
        """
    @typing.overload
    def nearest(
        self,
        position: numpy.ndarray[numpy.float64[3, 1]],
        *,
        radius: float,
        sort: bool = True,
        return_squared_l2: bool = False,
    ) -> tuple[numpy.ndarray[numpy.int32[m, 1]], numpy.ndarray[numpy.float64[m, 1]]]:
        """
        Find all points within a given radius of the query position
        """
    @typing.overload
    def positions(
        self,
    ) -> numpy.ndarray[numpy.float64[m, 3], numpy.ndarray.flags.c_contiguous]:
        """
        Get all positions in the KdQuiver
        """
    @typing.overload
    def positions(
        self, indexes: numpy.ndarray[numpy.int32[m, 1]]
    ) -> numpy.ndarray[numpy.float64[m, 3]]:
        """
        Get positions for the given indexes
        """
    def reset(self) -> None:
        """
        Reset the KdQuiver
        """

class KdTree:
    @typing.overload
    def __init__(self, leafsize: int = 10) -> None:
        """
        Initialize KdTree with specified leaf size.

        :param leafsize: Maximum number of points in leaf node, defaults to 10
        """
    @typing.overload
    def __init__(self, points: numpy.ndarray[numpy.float64[m, 3]]) -> None:
        """
        Initialize KdTree with 3D points.

        :param points: 3D points to initialize the tree
        """
    @typing.overload
    def __init__(
        self,
        points: numpy.ndarray[numpy.float64[m, 2], numpy.ndarray.flags.c_contiguous],
    ) -> None:
        """
        Initialize KdTree with 2D points.

        :param points: 2D points to initialize the tree
        """
    @typing.overload
    def add(self, points: numpy.ndarray[numpy.float64[m, 3]]) -> None:
        """
        Add 3D points to the KdTree.

        :param points: 3D points to add
        """
    @typing.overload
    def add(
        self,
        points: numpy.ndarray[numpy.float64[m, 2], numpy.ndarray.flags.c_contiguous],
    ) -> None:
        """
        Add 2D points to the KdTree.

        :param points: 2D points to add
        """
    def build_index(self, force_rebuild: bool = False) -> None:
        """
        Build the KdTree index.

        :param force_rebuild: Force rebuilding the index even if already built, defaults to false
        """
    def leafsize(self) -> int:
        """
        Get the current leaf size of the KdTree.

        :return: Current leaf size
        """
    @typing.overload
    def nearest(
        self,
        position: numpy.ndarray[numpy.float64[3, 1]],
        *,
        return_squared_l2: bool = False,
    ) -> tuple[int, float]:
        """
        Find the nearest point to the given position.

        :param position: Query position
        :param return_squared_l2: If true, return squared L2 distance, defaults to false
        :return: Tuple of (index, distance)
        """
    @typing.overload
    def nearest(
        self, index: int, *, return_squared_l2: bool = False
    ) -> tuple[int, float]:
        """
        Find the nearest point to the point at the given index.

        :param index: Index of the query point
        :param return_squared_l2: If true, return squared L2 distance, defaults to false
        :return: Tuple of (index, distance)
        """
    @typing.overload
    def nearest(
        self,
        position: numpy.ndarray[numpy.float64[3, 1]],
        *,
        k: int,
        sort: bool = True,
        return_squared_l2: bool = False,
    ) -> tuple[numpy.ndarray[numpy.int32[m, 1]], numpy.ndarray[numpy.float64[m, 1]]]:
        """
        Find k nearest points to the given position.

        :param position: Query position
        :param k: Number of nearest neighbors to find
        :param sort: If true, sort results by distance, defaults to true
        :param return_squared_l2: If true, return squared L2 distances, defaults to false
        :return: Tuple of (indices, distances)
        """
    @typing.overload
    def nearest(
        self,
        position: numpy.ndarray[numpy.float64[3, 1]],
        *,
        radius: float,
        sort: bool = True,
        return_squared_l2: bool = False,
    ) -> tuple[numpy.ndarray[numpy.int32[m, 1]], numpy.ndarray[numpy.float64[m, 1]]]:
        """
        Find all points within a given radius of the query position.

        :param position: Query position
        :param radius: Search radius
        :param sort: If true, sort results by distance, defaults to true
        :param return_squared_l2: If true, return squared L2 distances, defaults to false
        :return: Tuple of (indices, distances)
        """
    def points(
        self,
    ) -> numpy.ndarray[numpy.float64[m, 3], numpy.ndarray.flags.c_contiguous]:
        """
        Get the points in the KdTree.

        :return: Reference to the points in the tree
        """
    def reset(self) -> None:
        """
        Reset the KdTree, clearing all points.
        """
    def reset_index(self) -> None:
        """
        Reset the index of the KdTree.
        """
    def set_leafsize(self, value: int) -> None:
        """
        Set the leaf size of the KdTree.

        :param value: New leaf size value
        """

class LineSegment:
    def __init__(
        self,
        A: numpy.ndarray[numpy.float64[3, 1]],
        B: numpy.ndarray[numpy.float64[3, 1]],
    ) -> None:
        """
        Initialize a LineSegment with two 3D points.
        """
    def distance(self, P: numpy.ndarray[numpy.float64[3, 1]]) -> float:
        """
        Calculate the distance from a point to the line segment.
        """
    def distance2(self, P: numpy.ndarray[numpy.float64[3, 1]]) -> float:
        """
        Calculate the squared distance from a point to the line segment.
        """
    def intersects(
        self, other: LineSegment
    ) -> tuple[numpy.ndarray[numpy.float64[3, 1]], float, float, float] | None:
        """
        Check if this line segment intersects with another.
        """
    @property
    def A(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the start point of the line segment.
        """
    @property
    def AB(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the vector from A to B.
        """
    @property
    def B(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the end point of the line segment.
        """
    @property
    def length(self) -> float:
        """
        Get the length of the line segment.
        """
    @property
    def length2(self) -> float:
        """
        Get the squared length of the line segment.
        """

class PolylineRuler:
    @staticmethod
    def _along(
        line: numpy.ndarray[numpy.float64[m, 3], numpy.ndarray.flags.c_contiguous],
        dist: float,
        *,
        is_wgs84: bool = False,
    ) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Find a point at a specified distance along a polyline.
        """
    @staticmethod
    def _dirs(
        polyline: numpy.ndarray[numpy.float64[m, 3], numpy.ndarray.flags.c_contiguous],
        *,
        is_wgs84: bool = False,
    ) -> numpy.ndarray[numpy.float64[m, 3]]:
        """
        Calculate direction vectors for each segment of a polyline.
        """
    @staticmethod
    def _distance(
        a: numpy.ndarray[numpy.float64[3, 1]],
        b: numpy.ndarray[numpy.float64[3, 1]],
        *,
        is_wgs84: bool = False,
    ) -> float:
        """
        Calculate the distance between two points.
        """
    @staticmethod
    def _interpolate(
        A: numpy.ndarray[numpy.float64[3, 1]],
        B: numpy.ndarray[numpy.float64[3, 1]],
        *,
        t: float,
    ) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Interpolate between two points.
        """
    @staticmethod
    def _lineDistance(
        line: numpy.ndarray[numpy.float64[m, 3], numpy.ndarray.flags.c_contiguous],
        *,
        is_wgs84: bool = False,
    ) -> float:
        """
        Calculate the total length of a polyline.
        """
    @staticmethod
    def _lineSlice(
        start: numpy.ndarray[numpy.float64[3, 1]],
        stop: numpy.ndarray[numpy.float64[3, 1]],
        line: numpy.ndarray[numpy.float64[m, 3], numpy.ndarray.flags.c_contiguous],
        *,
        is_wgs84: bool = False,
    ) -> numpy.ndarray[numpy.float64[m, 3]]:
        """
        Extract a portion of a polyline between two points.
        """
    @staticmethod
    def _lineSliceAlong(
        start: float,
        stop: float,
        line: numpy.ndarray[numpy.float64[m, 3], numpy.ndarray.flags.c_contiguous],
        *,
        is_wgs84: bool = False,
    ) -> numpy.ndarray[numpy.float64[m, 3]]:
        """
        Extract a portion of a polyline between two distances along it.
        """
    @staticmethod
    def _pointOnLine(
        line: numpy.ndarray[numpy.float64[m, 3], numpy.ndarray.flags.c_contiguous],
        P: numpy.ndarray[numpy.float64[3, 1]],
        *,
        is_wgs84: bool = False,
    ) -> tuple[numpy.ndarray[numpy.float64[3, 1]], int, float]:
        """
        Find the closest point on a polyline to a given point.
        """
    @staticmethod
    def _pointToSegmentDistance(
        P: numpy.ndarray[numpy.float64[3, 1]],
        A: numpy.ndarray[numpy.float64[3, 1]],
        B: numpy.ndarray[numpy.float64[3, 1]],
        *,
        is_wgs84: bool = False,
    ) -> float:
        """
        Calculate the distance from a point to a line segment.
        """
    @staticmethod
    def _ranges(
        polyline: numpy.ndarray[numpy.float64[m, 3], numpy.ndarray.flags.c_contiguous],
        *,
        is_wgs84: bool = False,
    ) -> numpy.ndarray[numpy.float64[m, 1]]:
        """
        Calculate cumulative distances along a polyline.
        """
    @staticmethod
    def _squareDistance(
        a: numpy.ndarray[numpy.float64[3, 1]],
        b: numpy.ndarray[numpy.float64[3, 1]],
        *,
        is_wgs84: bool = False,
    ) -> float:
        """
        Calculate the squared distance between two points.
        """
    def N(self) -> int:
        """
        Get the number of points in the polyline.
        """
    def __init__(
        self,
        coords: numpy.ndarray[numpy.float64[m, 3], numpy.ndarray.flags.c_contiguous],
        *,
        is_wgs84: bool = False,
    ) -> None:
        """
        Initialize a PolylineRuler with coordinates and coordinate system.
        """
    def along(self, dist: float) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Find a point at a specified distance along the polyline.
        """
    @typing.overload
    def arrow(
        self, *, index: int, t: float
    ) -> tuple[numpy.ndarray[numpy.float64[3, 1]], numpy.ndarray[numpy.float64[3, 1]]]:
        """
        Get the arrow (point and direction) at a specific segment index and interpolation factor.
        """
    @typing.overload
    def arrow(
        self, range: float, *, smooth_joint: bool = True
    ) -> tuple[numpy.ndarray[numpy.float64[3, 1]], numpy.ndarray[numpy.float64[3, 1]]]:
        """
        Get the arrow (point and direction) at a specific cumulative distance.
        """
    @typing.overload
    def arrows(
        self, ranges: numpy.ndarray[numpy.float64[m, 1]], *, smooth_joint: bool = True
    ) -> tuple[
        numpy.ndarray[numpy.float64[m, 1]],
        numpy.ndarray[numpy.float64[m, 3]],
        numpy.ndarray[numpy.float64[m, 3]],
    ]:
        """
        Get arrows (points and directions) at multiple cumulative distances.
        """
    @typing.overload
    def arrows(
        self, step: float, *, with_last: bool = True, smooth_joint: bool = True
    ) -> tuple[
        numpy.ndarray[numpy.float64[m, 1]],
        numpy.ndarray[numpy.float64[m, 3]],
        numpy.ndarray[numpy.float64[m, 3]],
    ]:
        """
        Get arrows (points and directions) at regular intervals along the polyline.
        """
    @typing.overload
    def at(self, *, range: float) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the point on the polyline at a specific cumulative distance.
        """
    @typing.overload
    def at(self, *, segment_index: int) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the point on the polyline at a specific segment index.
        """
    @typing.overload
    def at(self, *, segment_index: int, t: float) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the point on the polyline at a specific segment index and interpolation factor.
        """
    @typing.overload
    def dir(self, *, point_index: int) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the direction vector at a specific point index.
        """
    @typing.overload
    def dir(
        self, *, range: float, smooth_joint: bool = True
    ) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the direction vector at a specific cumulative distance.
        """
    def dirs(self) -> numpy.ndarray[numpy.float64[m, 3]]:
        """
        Get direction vectors for each segment of the polyline.
        """
    def extended_along(self, range: float) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the extended cumulative distance along the polyline.
        """
    def is_wgs84(self) -> bool:
        """
        Check if the coordinate system is WGS84.
        """
    def k(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the scale factor for distance calculations.
        """
    def length(self) -> float:
        """
        Get the total length of the polyline.
        """
    def lineDistance(self) -> float:
        """
        Get the total length of the polyline.
        """
    def lineSlice(
        self,
        start: numpy.ndarray[numpy.float64[3, 1]],
        stop: numpy.ndarray[numpy.float64[3, 1]],
    ) -> numpy.ndarray[numpy.float64[m, 3]]:
        """
        Extract a portion of the polyline between two points.
        """
    def lineSliceAlong(
        self, start: float, stop: float
    ) -> numpy.ndarray[numpy.float64[m, 3]]:
        """
        Extract a portion of the polyline between two distances along it.
        """
    def local_frame(
        self, range: float, *, smooth_joint: bool = True
    ) -> numpy.ndarray[numpy.float64[4, 4]]:
        """
        Get the local coordinate frame at a specific cumulative distance.
        """
    def pointOnLine(
        self, P: numpy.ndarray[numpy.float64[3, 1]]
    ) -> tuple[numpy.ndarray[numpy.float64[3, 1]], int, float]:
        """
        Find the closest point on the polyline to a given point.
        """
    def polyline(self) -> numpy.ndarray[numpy.float64[m, 3]]:
        """
        Get the polyline coordinates.
        """
    @typing.overload
    def range(self, segment_index: int) -> float:
        """
        Get the cumulative distance at a specific segment index.
        """
    @typing.overload
    def range(self, *, segment_index: int, t: float) -> float:
        """
        Get the cumulative distance at a specific segment index and interpolation factor.
        """
    def ranges(self) -> numpy.ndarray[numpy.float64[m, 1]]:
        """
        Get cumulative distances along the polyline.
        """
    def scanline(
        self, range: float, *, min: float, max: float, smooth_joint: bool = True
    ) -> tuple[numpy.ndarray[numpy.float64[3, 1]], numpy.ndarray[numpy.float64[3, 1]]]:
        """
        Generate a scanline perpendicular to the polyline at a specific cumulative distance.
        """
    def segment_index(self, range: float) -> int:
        """
        Get the segment index for a given cumulative distance.
        """
    def segment_index_t(self, range: float) -> tuple[int, float]:
        """
        Get the segment index and interpolation factor for a given cumulative distance.
        """

class Quiver:
    class FilterParams:
        def __init__(self) -> None:
            """
            Default constructor for FilterParams
            """
        @typing.overload
        def angle_slots(self) -> numpy.ndarray[numpy.float64[m, 1]] | None:
            """
            Get the angle slots of the FilterParams
            """
        @typing.overload
        def angle_slots(
            self, arg0: numpy.ndarray[numpy.float64[m, 1]] | None
        ) -> Quiver.FilterParams:
            """
            Set the angle slots of the FilterParams
            """
        def is_trivial(self) -> bool:
            """
            Check if the FilterParams is trivial
            """
        @typing.overload
        def x_slots(self) -> numpy.ndarray[numpy.float64[m, 1]] | None:
            """
            Get the x slots of the FilterParams
            """
        @typing.overload
        def x_slots(
            self, arg0: numpy.ndarray[numpy.float64[m, 1]] | None
        ) -> Quiver.FilterParams:
            """
            Set the x slots of the FilterParams
            """
        @typing.overload
        def y_slots(self) -> numpy.ndarray[numpy.float64[m, 1]] | None:
            """
            Get the y slots of the FilterParams
            """
        @typing.overload
        def y_slots(
            self, arg0: numpy.ndarray[numpy.float64[m, 1]] | None
        ) -> Quiver.FilterParams:
            """
            Set the y slots of the FilterParams
            """
        @typing.overload
        def z_slots(self) -> numpy.ndarray[numpy.float64[m, 1]] | None:
            """
            Get the z slots of the FilterParams
            """
        @typing.overload
        def z_slots(
            self, arg0: numpy.ndarray[numpy.float64[m, 1]] | None
        ) -> Quiver.FilterParams:
            """
            Set the z slots of the FilterParams
            """

    @staticmethod
    def _k(arg0: float) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the constant k
        """
    @typing.overload
    def __init__(self) -> None:
        """
        Default constructor for Quiver
        """
    @typing.overload
    def __init__(self, anchor_lla: numpy.ndarray[numpy.float64[3, 1]]) -> None:
        """
        Constructor for Quiver with anchor LLA coordinates
        """
    def anchor(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the anchor point of the Quiver
        """
    @typing.overload
    def enu2lla(
        self, coords: numpy.ndarray[numpy.float64[3, 1]]
    ) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Convert ENU coordinates to LLA
        """
    @typing.overload
    def enu2lla(
        self, coords: numpy.ndarray[numpy.float64[m, 3]]
    ) -> numpy.ndarray[numpy.float64[m, 3]]:
        """
        Convert multiple ENU coordinates to LLA
        """
    def forwards(self, arrow: Arrow, delta_x: float) -> Arrow:
        """
        Move the Arrow forward by delta_x
        """
    def inv_k(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the inverse k value of the Quiver
        """
    def is_wgs84(self) -> bool:
        """
        Check if the Quiver is using WGS84 coordinates
        """
    def k(self) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Get the k value of the Quiver
        """
    def leftwards(self, arrow: Arrow, delta_y: float) -> Arrow:
        """
        Move the Arrow leftward by delta_y
        """
    @typing.overload
    def lla2enu(
        self, coords: numpy.ndarray[numpy.float64[3, 1]]
    ) -> numpy.ndarray[numpy.float64[3, 1]]:
        """
        Convert LLA coordinates to ENU
        """
    @typing.overload
    def lla2enu(
        self, coords: numpy.ndarray[numpy.float64[m, 3]]
    ) -> numpy.ndarray[numpy.float64[m, 3]]:
        """
        Convert multiple LLA coordinates to ENU
        """
    def towards(
        self,
        arrow: Arrow,
        delta_frenet: numpy.ndarray[numpy.float64[3, 1]],
        *,
        update_direction: bool = True,
    ) -> Arrow:
        """
        Move the Arrow in Frenet coordinates
        """
    def update(
        self,
        arrow: Arrow,
        delta_enu: numpy.ndarray[numpy.float64[3, 1]],
        *,
        update_direction: bool = True,
    ) -> Arrow:
        """
        Update the Arrow's position and optionally direction
        """
    def upwards(self, arrow: Arrow, delta_z: float) -> Arrow:
        """
        Move the Arrow upward by delta_z
        """

def densify_polyline(
    polyline: numpy.ndarray[numpy.float64[m, 3], numpy.ndarray.flags.c_contiguous],
    *,
    max_gap: float,
) -> numpy.ndarray[numpy.float64[m, 3]]:
    """
    densify polyline, interpolate to satisfy max_gap
    """

@typing.overload
def douglas_simplify(
    coords: numpy.ndarray[numpy.float64[m, 3]],
    epsilon: float,
    *,
    is_wgs84: bool = False,
    recursive: bool = True,
) -> numpy.ndarray[numpy.float64[m, 3]]:
    """
    Simplify a polyline using the Douglas-Peucker algorithm.
    """

@typing.overload
def douglas_simplify(
    coords: numpy.ndarray[numpy.float64[m, 2], numpy.ndarray.flags.c_contiguous],
    epsilon: float,
    *,
    is_wgs84: bool = False,
    recursive: bool = True,
) -> numpy.ndarray[numpy.float64[m, 2]]:
    """
    Simplify a 2D polyline using the Douglas-Peucker algorithm.
    """

@typing.overload
def douglas_simplify_indexes(
    coords: numpy.ndarray[numpy.float64[m, 3]],
    epsilon: float,
    *,
    is_wgs84: bool = False,
    recursive: bool = True,
) -> numpy.ndarray[numpy.int32[m, 1]]:
    """
    Get indexes of points to keep when simplifying a polyline using the Douglas-Peucker algorithm.
    """

@typing.overload
def douglas_simplify_indexes(
    coords: numpy.ndarray[numpy.float64[m, 2], numpy.ndarray.flags.c_contiguous],
    epsilon: float,
    *,
    is_wgs84: bool = False,
    recursive: bool = True,
) -> numpy.ndarray[numpy.int32[m, 1]]:
    """
    Get indexes of points to keep when simplifying a 2D polyline using the Douglas-Peucker algorithm.
    """

@typing.overload
def douglas_simplify_mask(
    coords: numpy.ndarray[numpy.float64[m, 3]],
    epsilon: float,
    *,
    is_wgs84: bool = False,
    recursive: bool = True,
) -> numpy.ndarray[numpy.int32[m, 1]]:
    """
    Get a mask of points to keep when simplifying a polyline using the Douglas-Peucker algorithm.
    """

@typing.overload
def douglas_simplify_mask(
    coords: numpy.ndarray[numpy.float64[m, 2], numpy.ndarray.flags.c_contiguous],
    epsilon: float,
    *,
    is_wgs84: bool = False,
    recursive: bool = True,
) -> numpy.ndarray[numpy.int32[m, 1]]:
    """
    Get a mask of points to keep when simplifying a 2D polyline using the Douglas-Peucker algorithm.
    """

@typing.overload
def intersect_segments(
    a1: numpy.ndarray[numpy.float64[2, 1]],
    a2: numpy.ndarray[numpy.float64[2, 1]],
    b1: numpy.ndarray[numpy.float64[2, 1]],
    b2: numpy.ndarray[numpy.float64[2, 1]],
) -> tuple[numpy.ndarray[numpy.float64[2, 1]], float, float] | None:
    """
    Intersect two 2D line segments.
    """

@typing.overload
def intersect_segments(
    a1: numpy.ndarray[numpy.float64[3, 1]],
    a2: numpy.ndarray[numpy.float64[3, 1]],
    b1: numpy.ndarray[numpy.float64[3, 1]],
    b2: numpy.ndarray[numpy.float64[3, 1]],
) -> tuple[numpy.ndarray[numpy.float64[3, 1]], float, float, float] | None:
    """
    Intersect two 3D line segments.
    """

def point_in_polygon(
    *,
    points: numpy.ndarray[numpy.float64[m, 2], numpy.ndarray.flags.c_contiguous],
    polygon: numpy.ndarray[numpy.float64[m, 2], numpy.ndarray.flags.c_contiguous],
) -> numpy.ndarray[numpy.int32[m, 1]]:
    """
    point-in-polygon test, returns 0-1 mask
    """

@typing.overload
def polyline_in_polygon(
    polyline: numpy.ndarray[numpy.float64[m, 3]],
    polygon: numpy.ndarray[numpy.float64[m, 2], numpy.ndarray.flags.c_contiguous],
    *,
    fc: FastCrossing,
) -> dict[
    tuple[int, float, float, int, float, float], numpy.ndarray[numpy.float64[m, 3]]
]: ...
@typing.overload
def polyline_in_polygon(
    polyline: numpy.ndarray[numpy.float64[m, 3]],
    polygon: numpy.ndarray[numpy.float64[m, 2], numpy.ndarray.flags.c_contiguous],
    *,
    is_wgs84: bool = False,
) -> dict[
    tuple[int, float, float, int, float, float], numpy.ndarray[numpy.float64[m, 3]]
]: ...
def snap_onto_2d(
    P: numpy.ndarray[numpy.float64[2, 1]],
    A: numpy.ndarray[numpy.float64[2, 1]],
    B: numpy.ndarray[numpy.float64[2, 1]],
) -> tuple[numpy.ndarray[numpy.float64[2, 1]], float, float]:
    """
    Snap P onto line segment AB
    """

__version__: str = "0.1.1"
