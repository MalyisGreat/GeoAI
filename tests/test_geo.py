import numpy as np

from geobot.utils.geo import grid_cell, haversine_km, latlon_to_unit, unit_to_latlon


def test_haversine_zero_distance() -> None:
    distance = haversine_km(10.0, 20.0, 10.0, 20.0)
    assert float(distance) == 0.0


def test_latlon_round_trip() -> None:
    original = np.array([[40.7128, -74.0060], [-33.8688, 151.2093]])
    unit = latlon_to_unit(original[:, 0], original[:, 1])
    reconstructed = unit_to_latlon(unit)
    assert np.allclose(original, reconstructed, atol=1e-5)


def test_grid_cell_is_stable() -> None:
    assert grid_cell(51.5, -0.12, 8, 16) == "8x16:006:007"
