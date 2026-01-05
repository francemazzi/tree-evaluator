from __future__ import annotations

import math
import sqlite3
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class CO2AggregateInput(BaseModel):
    """Input schema for CO2 aggregate tool."""
    natural_query: str = Field(
        description="Natural language query describing the subset of trees to analyze (e.g., 'pini', 'alberi in via Roma', 'distretto 1')."
    )


class CO2AggregateTool(BaseTool):
    """Tool to calculate aggregate CO2 and biomass for a subset of trees."""

    name: str = "calculate_co2_aggregate"
    description: str = """
    Calculate TOTAL and AVERAGE CO2 sequestration, biomass, and carbon stock for a group of trees.
    Use this tool when the user asks for "stock di carbonio", "totale CO2", "biomassa complessiva" 
    for a specific species, district, or the whole dataset.
    
    Do NOT use this for single trees (use calculate_co2_sequestration instead).
    """
    args_schema: Type[BaseModel] = CO2AggregateInput

    _db_path: Path
    _table_name: str
    _dataset_type: str  # "vienna", "milano", "custom"
    _llm: Any = None
    
    def __init__(
        self,
        db_path: Optional[Path] = None,
        table_name: str = "baumkatogd",
        dataset_type: str = "vienna",
        llm: Any = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if db_path is None:
             # Default to Vienna
             db_path = Path(__file__).parent.parent.parent / "dataset" / "BAUMKATOGD.db"
        
        object.__setattr__(self, "_db_path", db_path)
        object.__setattr__(self, "_table_name", table_name)
        object.__setattr__(self, "_dataset_type", dataset_type)
        object.__setattr__(self, "_llm", llm)

    def _get_connection(self) -> sqlite3.Connection:
        if not self._db_path.exists():
            raise FileNotFoundError(f"Database not found at {self._db_path}")
        return sqlite3.connect(self._db_path)

    def _generate_sql_query(self, natural_query: str) -> str:
        """Generate a SELECT * query based on the natural language filter."""
        if not self._llm:
            raise ValueError("LLM is required for query generation")

        # Simplified schema info based on dataset type
        if self._dataset_type == "vienna":
            schema_hint = """
            Table: baumkatogd
            Columns: objectid, district (1-23), genus_species (e.g. 'Acer platanoides'), 
            plant_year, trunk_circumference (cm), tree_height (0-8 cat), object_street
            """
        elif self._dataset_type == "milano":
            schema_hint = """
            Table: milano_trees
            Columns: _id, district, genus_species, plant_year, 
            trunk_diameter_cm (cm), height_m (m), street
            """
        else:
            schema_hint = f"Table: {self._table_name} (Generic)"

        prompt = f"""You are a SQL expert. Generate a SELECT query to retrieve tree data for CO2 calculation.
        
        {schema_hint}
        
        User Query: "{natural_query}"
        
        Rules:
        1. Select ONLY the columns needed for calculation + identifiers:
           - Vienna: objectid, trunk_circumference, tree_height, genus_species
           - Milano: _id, trunk_diameter_cm, height_m, genus_species
        2. Apply the filtering logic requested (WHERE clause).
        3. Do NOT use LIMIT unless explicitly asked (we need all rows for total stock).
        4. If the user asks for "all trees" or "total", do not add a WHERE clause.
        5. For species, use LIKE with % (e.g. genus_species LIKE '%Pinus%').
        
        Return ONLY the SQL string.
        """
        
        response = self._llm.invoke(prompt)
        sql = response.content if hasattr(response, 'content') else str(response)
        
        # Cleanup
        sql = sql.strip()
        if sql.startswith("```"):
            sql = sql.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        if sql.lower().startswith("sql"):
            sql = sql[3:].strip()
            
        return sql

    def _calculate_vienna_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metrics for Vienna dataset."""
        # 1. Map Height
        # 1=0-5, 2=6-10, 3=11-15, 4=16-20, 5=21-25, 6=26-30, 7=31-35, 8=>35
        height_map = {
            0: 0, 1: 2.5, 2: 8, 3: 13, 4: 18, 5: 23, 6: 28, 7: 33, 8: 38
        }
        df['height_m'] = df['tree_height'].map(height_map).fillna(0)
        
        # 2. Calculate DBH (cm)
        df['dbh_cm'] = df['trunk_circumference'] / math.pi
        
        return self._apply_allometric_equations(df)

    def _calculate_milano_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate metrics for Milano dataset."""
        df['height_m'] = df['height_m'].fillna(0)
        df['dbh_cm'] = df['trunk_diameter_cm'].fillna(0)
        return self._apply_allometric_equations(df)

    def _apply_allometric_equations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply general allometric equations (Chave et al. 2014)."""
        # Constants
        wood_density = 0.6  # default
        carbon_fraction = 0.47
        root_shoot_ratio = 0.24
        
        # Filter invalid data
        mask = (df['dbh_cm'] > 0) & (df['height_m'] > 0)
        valid_df = df[mask].copy()
        
        # AGB = 0.0673 * (WD * DBH^2 * H)^0.976  (DBH in cm, H in m, result in kg)
        # We need to divide by 1000 to get tonnes? No, Chave gives kg.
        # But commonly we work in tonnes for stock.
        
        a = 0.0673
        b = 0.976
        
        valid_df['agb_kg'] = a * ((wood_density * (valid_df['dbh_cm']**2) * valid_df['height_m'])**b)
        
        # Convert to tonnes
        valid_df['agb_t'] = valid_df['agb_kg'] / 1000.0
        
        # BGB
        valid_df['bgb_t'] = valid_df['agb_t'] * root_shoot_ratio
        
        # Total Biomass
        valid_df['total_biomass_t'] = valid_df['agb_t'] + valid_df['bgb_t']
        
        # Carbon
        valid_df['carbon_t'] = valid_df['total_biomass_t'] * carbon_fraction
        
        # CO2
        valid_df['co2_t'] = valid_df['carbon_t'] * (44.0 / 12.0)
        
        return valid_df

    def _run(self, natural_query: str) -> Dict[str, Any]:
        try:
            conn = self._get_connection()
            
            # Generate SQL
            sql = self._generate_sql_query(natural_query)
            
            # Execute and load into Pandas
            df = pd.read_sql_query(sql, conn)
            conn.close()
            
            if df.empty:
                return {
                    "result": "Nessun albero trovato con questa query.",
                    "sql_executed": sql,
                    "count": 0
                }
            
            # Calculate metrics based on dataset
            if self._dataset_type == "vienna":
                results_df = self._calculate_vienna_metrics(df)
            elif self._dataset_type == "milano":
                results_df = self._calculate_milano_metrics(df)
            else:
                # Try generic fallback if columns exist
                if 'trunk_diameter_cm' in df.columns and 'height_m' in df.columns:
                    results_df = self._calculate_milano_metrics(df)
                elif 'trunk_circumference' in df.columns:
                     results_df = self._calculate_vienna_metrics(df)
                else:
                    return {"error": "Dataset format unknown"}

            # Aggregate results
            count = len(results_df)
            total_co2 = results_df['co2_t'].sum()
            avg_co2 = results_df['co2_t'].mean()
            total_carbon = results_df['carbon_t'].sum()
            total_biomass = results_df['total_biomass_t'].sum()
            
            return {
                "natural_query": natural_query,
                "sql_executed": sql,
                "tree_count": count,
                "co2_stock_t": round(total_co2, 2),
                "avg_co2_t": round(avg_co2, 4),
                "carbon_stock_t": round(total_carbon, 2),
                "total_biomass_t": round(total_biomass, 2),
                "dataset": self._dataset_type,
                "formulas": {
                    "agb": "Chave et al. (2014)",
                    "co2": "Biomass * 0.47 * (44/12)"
                }
            }

        except Exception as e:
            return {"error": str(e)}

