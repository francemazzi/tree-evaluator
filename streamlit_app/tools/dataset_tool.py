from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import pandas as pd
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class DatasetQueryInput(BaseModel):
    """Input schema for dataset query tool."""

    query_type: str = Field(
        description="""Type of query to execute. Choose one of:
        - 'summary': get basic statistics and info about the dataset
        - 'filter': filter trees by criteria (district, species, year range)
        - 'aggregate': aggregate statistics by group (district, species)
        - 'sample': get a random sample of N trees
        - 'count': count trees matching criteria
        """
    )
    district: Optional[int] = Field(
        default=None,
        description="Filter by district number (e.g., 19 for district 19)",
    )
    species_contains: Optional[str] = Field(
        default=None,
        description="Filter trees whose species name contains this text (case-insensitive)",
    )
    plant_year_min: Optional[int] = Field(
        default=None,
        description="Minimum plant year (inclusive)",
    )
    plant_year_max: Optional[int] = Field(
        default=None,
        description="Maximum plant year (inclusive)",
    )
    group_by: Optional[str] = Field(
        default=None,
        description="Column to group by for aggregation: 'DISTRICT', 'GENUS_SPECIES', 'PLANT_YEAR', 'AREA_GROUP'",
    )
    sample_size: Optional[int] = Field(
        default=10,
        description="Number of random samples to return (for query_type='sample')",
    )


class DatasetQueryTool(BaseTool):
    """Tool to query tree dataset CSV/Excel files in the dataset folder."""

    name: str = "query_tree_dataset"
    description: str = """
    Query the tree dataset (Vienna trees BAUMKATOGD.csv or other CSV/Excel files in dataset/).
    
    Supports multiple query types:
    1. 'summary': Get dataset overview (total trees, columns, districts, species)
    2. 'filter': Filter trees by district, species name, plant year range
    3. 'aggregate': Get aggregated statistics grouped by district, species, year, or area
    4. 'sample': Get random sample of N trees with their details
    5. 'count': Count trees matching specific criteria
    
    Examples:
    - "How many trees in district 19?" → query_type='count', district=19
    - "Show me Acer trees planted after 2000" → query_type='filter', species_contains='Acer', plant_year_min=2000
    - "Statistics by district" → query_type='aggregate', group_by='DISTRICT'
    - "Give me 5 random trees" → query_type='sample', sample_size=5
    
    Use this when user asks about the dataset, tree counts, statistics, filtering, or wants to explore the data.
    """
    args_schema: Type[BaseModel] = DatasetQueryInput

    _dataset_path: Path
    _df: Optional[pd.DataFrame] = None

    def __init__(self, dataset_path: Optional[Path] = None, **kwargs):
        super().__init__(**kwargs)
        if dataset_path is None:
            dataset_path = Path(__file__).parent.parent.parent / "dataset" / "BAUMKATOGD.csv"
        object.__setattr__(self, "_dataset_path", dataset_path)
        object.__setattr__(self, "_df", None)

    def _load_dataset(self) -> pd.DataFrame:
        """Lazy load dataset."""
        if self._df is not None:
            return self._df

        if not self._dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self._dataset_path}")

        # Load CSV with type hints
        df = pd.read_csv(
            self._dataset_path,
            dtype={
                "OBJECTID": "Int64",
                "TREE_ID": "Int64",
                "DISTRICT": "Int64",
                "PLANT_YEAR": "Int64",
                "TRUNK_CIRCUMFERENCE": "Int64",
                "TREE_HEIGHT": "Int64",
                "CROWN_DIAMETER": "Int64",
            },
            low_memory=False,
            nrows=50000,  # Limit for performance in chat context
        )

        # Compute DBH
        df["dbh_cm"] = df["TRUNK_CIRCUMFERENCE"].apply(
            lambda c: float(c) / math.pi if pd.notna(c) and c > 0 else pd.NA
        )

        # Compute age
        current_year = pd.Timestamp.now().year
        df["age_years"] = df["PLANT_YEAR"].apply(
            lambda y: int(current_year - y) if pd.notna(y) and y > 0 else pd.NA
        )

        object.__setattr__(self, "_df", df)
        return df

    def _apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to dataframe."""
        mask = pd.Series([True] * len(df))

        if filters.get("district") is not None:
            mask &= df["DISTRICT"] == filters["district"]

        if filters.get("species_contains"):
            mask &= df["GENUS_SPECIES"].fillna("").str.contains(
                filters["species_contains"], case=False, na=False
            )

        if filters.get("plant_year_min") is not None:
            mask &= df["PLANT_YEAR"] >= filters["plant_year_min"]

        if filters.get("plant_year_max") is not None:
            mask &= df["PLANT_YEAR"] <= filters["plant_year_max"]

        return df[mask]

    def _summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate dataset summary."""
        return {
            "total_trees": len(df),
            "columns": list(df.columns),
            "districts": sorted(df["DISTRICT"].dropna().unique().tolist()),
            "district_count": df["DISTRICT"].nunique(),
            "species_count": df["GENUS_SPECIES"].nunique(),
            "top_5_species": df["GENUS_SPECIES"].value_counts().head(5).to_dict(),
            "plant_year_range": {
                "min": int(df["PLANT_YEAR"].min()) if pd.notna(df["PLANT_YEAR"].min()) else None,
                "max": int(df["PLANT_YEAR"].max()) if pd.notna(df["PLANT_YEAR"].max()) else None,
            },
            "dbh_stats": {
                "mean_cm": float(df["dbh_cm"].mean()) if "dbh_cm" in df else None,
                "median_cm": float(df["dbh_cm"].median()) if "dbh_cm" in df else None,
            },
        }

    def _aggregate(self, df: pd.DataFrame, group_by: str) -> Dict[str, Any]:
        """Aggregate statistics by group."""
        if group_by not in df.columns:
            return {"error": f"Column {group_by} not found in dataset"}

        agg = (
            df.groupby(group_by, dropna=False)
            .agg(
                count=("OBJECTID", "count"),
                dbh_cm_median=("dbh_cm", "median"),
                tree_height_median=("TREE_HEIGHT", "median"),
                age_years_median=("age_years", "median"),
            )
            .reset_index()
        )

        # Convert to dict, limit to top 20 groups
        result = agg.head(20).to_dict(orient="records")
        return {"groups": result, "total_groups": len(agg)}

    def _sample(self, df: pd.DataFrame, n: int) -> Dict[str, Any]:
        """Get random sample of trees."""
        sample = df.sample(n=min(n, len(df)))
        # Select key columns
        cols = [
            "OBJECTID",
            "DISTRICT",
            "OBJECT_STREET",
            "GENUS_SPECIES",
            "PLANT_YEAR",
            "TRUNK_CIRCUMFERENCE",
            "TREE_HEIGHT",
            "dbh_cm",
            "age_years",
        ]
        sample_cols = [c for c in cols if c in sample.columns]
        return {"sample": sample[sample_cols].to_dict(orient="records")}

    def _run(
        self,
        query_type: str,
        district: Optional[int] = None,
        species_contains: Optional[str] = None,
        plant_year_min: Optional[int] = None,
        plant_year_max: Optional[int] = None,
        group_by: Optional[str] = None,
        sample_size: int = 10,
    ) -> dict:
        """Execute the dataset query."""
        try:
            df = self._load_dataset()

            filters = {
                "district": district,
                "species_contains": species_contains,
                "plant_year_min": plant_year_min,
                "plant_year_max": plant_year_max,
            }

            # Apply filters if any
            if any(v is not None for v in filters.values()):
                df = self._apply_filters(df, filters)

            if query_type == "summary":
                return self._summary(df)
            elif query_type == "filter":
                # Return filtered count and sample
                return {
                    "filtered_count": len(df),
                    "sample": df.head(20)[
                        [
                            "OBJECTID",
                            "DISTRICT",
                            "GENUS_SPECIES",
                            "PLANT_YEAR",
                            "TRUNK_CIRCUMFERENCE",
                            "dbh_cm",
                        ]
                    ].to_dict(orient="records"),
                }
            elif query_type == "aggregate":
                if not group_by:
                    return {"error": "group_by parameter required for aggregate query"}
                return self._aggregate(df, group_by)
            elif query_type == "sample":
                return self._sample(df, sample_size)
            elif query_type == "count":
                return {"count": len(df), "filters_applied": {k: v for k, v in filters.items() if v is not None}}
            else:
                return {"error": f"Unknown query_type: {query_type}"}

        except Exception as e:
            return {"error": str(e)}

