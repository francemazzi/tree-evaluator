"""Centralized scientific constants for tree evaluation.

Single source of truth for all forestry parameters, allometric coefficients,
and species-specific data used across tools and services.
"""

from __future__ import annotations


# =============================================================================
# Allometric equation coefficients — Chave et al. (2014)
# Source: https://doi.org/10.1111/gcb.12629
# AGB (kg) = CHAVE_INTERCEPT * (WD * DBH^2 * H) ^ CHAVE_EXPONENT
# =============================================================================
CHAVE_INTERCEPT: float = 0.0673
CHAVE_EXPONENT: float = 0.976

# =============================================================================
# Default biomass parameters
# =============================================================================
DEFAULT_WOOD_DENSITY: float = 0.6       # g/cm³ (generic broadleaf)
DEFAULT_WOOD_DENSITY_KG_M3: float = 600.0  # kg/m³ (same value, different unit)
DEFAULT_CARBON_FRACTION: float = 0.47   # IPCC 2006 — 47% of dry biomass
DEFAULT_ROOT_SHOOT_RATIO: float = 0.24  # Cairns et al. (1997)

# CO2 to carbon molecular weight ratio: CO2 (44) / C (12)
CO2_C_RATIO: float = 44.0 / 12.0  # ≈ 3.6667

# =============================================================================
# Default annual sequestration rates — Paoletti et al.
# Used as fallback when species-specific data is not available.
# =============================================================================
DEFAULT_SEQUESTRATION_MEDIO_KG_YEAR: float = 4.6    # kg C/year (young/average)
DEFAULT_SEQUESTRATION_MATURITA_KG_YEAR: float = 11.4  # kg C/year (mature)

# =============================================================================
# Species-specific wood densities (g/cm³)
# Sources: various — used in prompt and calculations.
# =============================================================================
WOOD_DENSITIES: dict[str, float] = {
    "Acer": 0.56,
    "Tilia": 0.49,
    "Carpinus": 0.75,
    "Gleditsia": 0.62,
    "Aesculus": 0.53,
    "Quercus": 0.75,
    "Fraxinus": 0.69,
    "Betula": 0.65,
}

# =============================================================================
# Softwood (conifer) genera — used to determine hardwood vs softwood R/S ratio
# =============================================================================
SOFTWOOD_GENERA: set[str] = {
    "pinus", "picea", "abies", "cedrus", "larix", "pseudotsuga",
    "tsuga", "thuja", "cupressus", "juniperus", "taxus", "sequoia",
    "sequoiadendron", "cryptomeria", "chamaecyparis", "araucaria",
    "podocarpus", "taxodium", "metasequoia",
}

# =============================================================================
# Vienna dataset: height category → meters mapping
# Categories: 1=0-5m, 2=6-10m, ..., 8=>35m
# =============================================================================
VIENNA_HEIGHT_MAP: dict[int, float] = {
    0: 0.0, 1: 2.5, 2: 8.0, 3: 13.0, 4: 18.0,
    5: 23.0, 6: 28.0, 7: 33.0, 8: 38.0,
}

# =============================================================================
# Genus → common name mapping (for carbon_content.csv lookup)
# =============================================================================
GENUS_TO_COMMON_NAME: dict[str, str] = {
    "platanus": "poplar",
    "acer": "maple",
    "quercus": "oak",
    "fraxinus": "ash",
    "tilia": "basswood",
    "betula": "birch",
    "fagus": "beech",
    "ulmus": "elm",
    "populus": "poplar",
    "salix": "willow",
    "pinus": "pine",
    "picea": "spruce",
    "abies": "fir",
    "cedrus": "cedar",
    "larix": "larch",
    "eucalyptus": "eucalyptus",
    "alnus": "alder",
    "carpinus": "hornbeam",
    "castanea": "hickory",
    "juglans": "hickory",
    "prunus": "cherry, fire",
    "robinia": "oak",
    "aesculus": "hickory",
}
