"""
Dielectric Geometry Connectivity & Topology Checking Engine
Re-exports connectivity functions from the unified utils module.
"""

from .utils import check_array_connectivity, check_slab_connectivity

__all__ = [
    "check_array_connectivity",
    "check_slab_connectivity",
]
