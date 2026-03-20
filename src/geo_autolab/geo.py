from __future__ import annotations

import math

import torch
from torch import Tensor


EARTH_RADIUS_KM = 6371.0088


def normalize_latlon(latlon_deg: Tensor) -> Tensor:
    lat = latlon_deg[..., 0] / 90.0
    lon = latlon_deg[..., 1] / 180.0
    return torch.stack((lat, lon), dim=-1)


def denormalize_latlon(latlon_normalized: Tensor) -> Tensor:
    lat = latlon_normalized[..., 0] * 90.0
    lon = latlon_normalized[..., 1] * 180.0
    return torch.stack((lat, lon), dim=-1)


def latlon_to_unit_xyz(latlon_deg: Tensor) -> Tensor:
    lat_rad = torch.deg2rad(latlon_deg[..., 0])
    lon_rad = torch.deg2rad(latlon_deg[..., 1])
    cos_lat = torch.cos(lat_rad)
    x = cos_lat * torch.cos(lon_rad)
    y = cos_lat * torch.sin(lon_rad)
    z = torch.sin(lat_rad)
    return torch.stack((x, y, z), dim=-1)


def unit_xyz_to_latlon(unit_xyz: Tensor) -> Tensor:
    unit_xyz = torch.nn.functional.normalize(unit_xyz, dim=-1)
    lat = torch.asin(unit_xyz[..., 2]).rad2deg()
    lon = torch.atan2(unit_xyz[..., 1], unit_xyz[..., 0]).rad2deg()
    return torch.stack((lat, lon), dim=-1)


def great_circle_distance_km(pred_xyz: Tensor, target_xyz: Tensor) -> Tensor:
    pred = torch.nn.functional.normalize(pred_xyz, dim=-1)
    target = torch.nn.functional.normalize(target_xyz, dim=-1)
    cosine = torch.sum(pred * target, dim=-1).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.arccos(cosine) * EARTH_RADIUS_KM


def haversine_distance_km(pred_latlon: Tensor, target_latlon: Tensor) -> Tensor:
    lat1, lon1 = torch.deg2rad(pred_latlon[..., 0]), torch.deg2rad(pred_latlon[..., 1])
    lat2, lon2 = torch.deg2rad(target_latlon[..., 0]), torch.deg2rad(target_latlon[..., 1])
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = torch.sin(d_lat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(d_lon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return c * EARTH_RADIUS_KM


def initial_bearing_deg(origin_latlon: Tensor, target_latlon: Tensor) -> Tensor:
    lat1 = torch.deg2rad(origin_latlon[..., 0])
    lon1 = torch.deg2rad(origin_latlon[..., 1])
    lat2 = torch.deg2rad(target_latlon[..., 0])
    lon2 = torch.deg2rad(target_latlon[..., 1])
    y = torch.sin(lon2 - lon1) * torch.cos(lat2)
    x = torch.cos(lat1) * torch.sin(lat2) - torch.sin(lat1) * torch.cos(lat2) * torch.cos(lon2 - lon1)
    return (torch.atan2(y, x) * (180.0 / math.pi) + 360.0) % 360.0

