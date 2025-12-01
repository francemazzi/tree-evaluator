"""Service to retrieve carbon content values by species from CSV dataset."""

import csv
from pathlib import Path
from typing import Optional


class CarbonContentService:
    """Service to lookup carbon content by species from CSV dataset."""

    def __init__(self, csv_path: Optional[Path] = None) -> None:
        """Initialize the service with CSV path.
        
        Args:
            csv_path: Path to carbon_content.csv. If None, uses default dataset path.
        """
        if csv_path is None:
            # Default to dataset/carbon_content.csv relative to project root
            project_root = Path(__file__).parent.parent.parent
            csv_path = project_root / "dataset" / "carbon_content.csv"
        
        self._csv_path = csv_path
        self._cache: dict[str, float] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load carbon content data from CSV into cache."""
        if not self._csv_path.exists():
            return  # Cache remains empty if file doesn't exist

        with self._csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                species = row.get("Species", "").strip()
                mean_str = row.get("Mean", "").strip()
                
                if species and mean_str:
                    try:
                        # Convert percentage to fraction (e.g., 50.2 -> 0.502)
                        mean_percent = float(mean_str)
                        mean_fraction = mean_percent / 100.0
                        # Store with normalized key (lowercase for case-insensitive lookup)
                        self._cache[species.lower()] = mean_fraction
                    except (ValueError, TypeError):
                        continue  # Skip invalid rows

    def get_carbon_fraction(self, species: Optional[str] = None) -> Optional[float]:
        """Get carbon fraction for a species.
        
        Args:
            species: Species name (case-insensitive). If None, returns None.
            
        Returns:
            Carbon fraction (0-1) if found, None otherwise.
        """
        if not species:
            return None
        
        # Case-insensitive lookup
        return self._cache.get(species.lower())

    def is_species_available(self, species: str) -> bool:
        """Check if carbon content data is available for a species.
        
        Args:
            species: Species name (case-insensitive).
            
        Returns:
            True if species is in the dataset, False otherwise.
        """
        if not species:
            return False
        return species.lower() in self._cache

