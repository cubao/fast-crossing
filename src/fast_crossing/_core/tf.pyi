from __future__ import annotations
import numpy
import numpy.typing
import typing

__all__: list[str] = [
    "R_ecef_enu",
    "T_ecef_enu",
    "apply_transform",
    "apply_transform_inplace",
    "cheap_ruler_k",
    "ecef2enu",
    "ecef2lla",
    "enu2ecef",
    "enu2lla",
    "lla2ecef",
    "lla2enu",
]

def R_ecef_enu(
    lon: typing.SupportsFloat, lat: typing.SupportsFloat
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 3]"]:
    """
    Get rotation matrix from ECEF to ENU coordinate system.
    """

@typing.overload
def T_ecef_enu(
    lon: typing.SupportsFloat, lat: typing.SupportsFloat, alt: typing.SupportsFloat
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 4]"]:
    """
    Get transformation matrix from ECEF to ENU coordinate system.
    """

@typing.overload
def T_ecef_enu(
    lla: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[4, 4]"]:
    """
    Get transformation matrix from ECEF to ENU coordinate system using LLA vector.
    """

def apply_transform(
    T: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[4, 4]"],
    coords: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, 3]", "flags.c_contiguous"
    ],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 3]"]:
    """
    Apply transformation matrix to coordinates.
    """

def apply_transform_inplace(
    T: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[4, 4]"],
    coords: typing.Annotated[
        numpy.typing.NDArray[numpy.float64],
        "[m, 3]",
        "flags.writeable",
        "flags.c_contiguous",
    ],
    *,
    batch_size: typing.SupportsInt = 1000,
) -> None:
    """
    Apply transformation matrix to coordinates in-place.
    """

def cheap_ruler_k(
    latitude: typing.SupportsFloat,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
    """
    Get the cheap ruler's unit conversion factor for a given latitude.
    """

def ecef2enu(
    ecefs: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, 3]", "flags.c_contiguous"
    ],
    *,
    anchor_lla: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]
    | None = None,
    cheap_ruler: bool = False,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 3]"]:
    """
    Convert ECEF to ENU (East, North, Up) coordinates.
    """

@typing.overload
def ecef2lla(
    x: typing.SupportsFloat, y: typing.SupportsFloat, z: typing.SupportsFloat
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
    """
    Convert ECEF coordinates to LLA (Longitude, Latitude, Altitude).
    """

@typing.overload
def ecef2lla(
    ecefs: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, 3]", "flags.c_contiguous"
    ],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 3]"]:
    """
    Convert multiple ECEF coordinates to LLA (Longitude, Latitude, Altitude).
    """

def enu2ecef(
    enus: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, 3]", "flags.c_contiguous"
    ],
    *,
    anchor_lla: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    cheap_ruler: bool = False,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 3]"]:
    """
    Convert ENU (East, North, Up) to ECEF coordinates.
    """

def enu2lla(
    enus: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, 3]", "flags.c_contiguous"
    ],
    *,
    anchor_lla: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"],
    cheap_ruler: bool = True,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 3]"]:
    """
    Convert ENU (East, North, Up) to LLA (Longitude, Latitude, Altitude) coordinates.
    """

@typing.overload
def lla2ecef(
    lon: typing.SupportsFloat, lat: typing.SupportsFloat, alt: typing.SupportsFloat
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[3, 1]"]:
    """
    Convert LLA (Longitude, Latitude, Altitude) to ECEF coordinates.
    """

@typing.overload
def lla2ecef(
    llas: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, 3]", "flags.c_contiguous"
    ],
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 3]"]:
    """
    Convert multiple LLA (Longitude, Latitude, Altitude) to ECEF coordinates.
    """

def lla2enu(
    llas: typing.Annotated[
        numpy.typing.NDArray[numpy.float64], "[m, 3]", "flags.c_contiguous"
    ],
    *,
    anchor_lla: typing.Annotated[numpy.typing.ArrayLike, numpy.float64, "[3, 1]"]
    | None = None,
    cheap_ruler: bool = True,
) -> typing.Annotated[numpy.typing.NDArray[numpy.float64], "[m, 3]"]:
    """
    Convert LLA (Longitude, Latitude, Altitude) to ENU (East, North, Up) coordinates.
    """
