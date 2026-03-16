from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch

EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lat1: np.ndarray | torch.Tensor | float,
    lon1: np.ndarray | torch.Tensor | float,
    lat2: np.ndarray | torch.Tensor | float,
    lon2: np.ndarray | torch.Tensor | float,
) -> np.ndarray:
    lat1 = np.asarray(lat1, dtype=np.float64)
    lon1 = np.asarray(lon1, dtype=np.float64)
    lat2 = np.asarray(lat2, dtype=np.float64)
    lon2 = np.asarray(lon2, dtype=np.float64)

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    d_lat = lat2_rad - lat1_rad
    d_lon = lon2_rad - lon1_rad
    a = np.sin(d_lat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(
        d_lon / 2.0
    ) ** 2
    c = 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(a)))
    return EARTH_RADIUS_KM * c


def latlon_to_unit(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat_rad = np.radians(np.asarray(lat, dtype=np.float64))
    lon_rad = np.radians(np.asarray(lon, dtype=np.float64))
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.stack([x, y, z], axis=-1)


def tensor_latlon_to_unit(latlon: torch.Tensor) -> torch.Tensor:
    lat_rad = torch.deg2rad(latlon[..., 0])
    lon_rad = torch.deg2rad(latlon[..., 1])
    x = torch.cos(lat_rad) * torch.cos(lon_rad)
    y = torch.cos(lat_rad) * torch.sin(lon_rad)
    z = torch.sin(lat_rad)
    return torch.stack([x, y, z], dim=-1)


def unit_to_latlon(unit: np.ndarray) -> np.ndarray:
    unit = np.asarray(unit, dtype=np.float64)
    unit = unit / np.linalg.norm(unit, axis=-1, keepdims=True).clip(min=1e-8)
    lat = np.degrees(np.arcsin(np.clip(unit[..., 2], -1.0, 1.0)))
    lon = np.degrees(np.arctan2(unit[..., 1], unit[..., 0]))
    return np.stack([lat, lon], axis=-1)


def tensor_unit_to_latlon(unit: torch.Tensor) -> torch.Tensor:
    unit = torch.nn.functional.normalize(unit, dim=-1)
    lat = torch.rad2deg(torch.asin(unit[..., 2].clamp(-1.0, 1.0)))
    lon = torch.rad2deg(torch.atan2(unit[..., 1], unit[..., 0]))
    return torch.stack([lat, lon], dim=-1)


def grid_cell(lat: float, lon: float, lat_bins: int, lon_bins: int) -> str:
    lat_idx = min(lat_bins - 1, max(0, int((lat + 90.0) / 180.0 * lat_bins)))
    lon_idx = min(lon_bins - 1, max(0, int((lon + 180.0) / 360.0 * lon_bins)))
    return f"{lat_bins}x{lon_bins}:{lat_idx:03d}:{lon_idx:03d}"


def assign_grid_cells(
    latitudes: Iterable[float], longitudes: Iterable[float], bins: tuple[int, int]
) -> list[str]:
    lat_bins, lon_bins = bins
    return [
        grid_cell(float(lat), float(lon), lat_bins, lon_bins)
        for lat, lon in zip(latitudes, longitudes)
    ]


def normalize_unit_tensor(unit: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(unit, dim=-1)
