from __future__ import annotations

import csv
import math
import sqlite3
import json
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Literal

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from streamlit_app.constants import (
    CHAVE_EXPONENT,
    CHAVE_INTERCEPT,
    CO2_C_RATIO,
    DEFAULT_CARBON_FRACTION,
    DEFAULT_ROOT_SHOOT_RATIO,
    DEFAULT_WOOD_DENSITY,
    GENUS_TO_COMMON_NAME,
    SOFTWOOD_GENERA,
    VIENNA_HEIGHT_MAP,
)


class CO2AggregateInput(BaseModel):
    """Input schema for CO2 aggregate tool."""
    natural_query: str = Field(
        description="Natural language query describing the subset of trees to analyze (e.g., 'pini', 'alberi in via Roma', 'distretto 1')."
    )


class CO2AggregateTool(BaseTool):
    """Tool to calculate aggregate CO2 and biomass for a subset of trees."""

    name: str = "calculate_co2_aggregate"
    description: str = """
    Calculate TOTAL STOCK (not annual absorption!) of CO2, biomass, and carbon for a group of trees.
    
    IMPORTANT - This tool calculates STOCK (how much carbon/CO2 is STORED in trees right now).
    This tool does NOT calculate ANNUAL ABSORPTION (how much carbon trees absorb per year).
    
    Use this tool ONLY for:
    - "stock di carbonio" / "carbon stock" / "carbonio immagazzinato"
    - "CO2 totale immagazzinata" / "total stored CO2"
    - "biomassa totale" / "total biomass"
    
    Do NOT use this tool for:
    - "carbonio assorbito annualmente" / "annual carbon absorption" - requires annual increment data not available
    - "quanto carbonio assorbe all'anno" - this is annual rate, not stock
    - Single tree calculations (use calculate_co2_sequestration instead)
    - Carbon content/fraction per species (use lookup_carbon_content instead)
    
    If user asks for ANNUAL absorption, explain that this data is not available in the dataset.
    """
    args_schema: Type[BaseModel] = CO2AggregateInput

    _db_path: Path
    _table_name: str
    _dataset_type: str  # "vienna", "milano", "custom"
    _llm: Any = None
    _carbon_content_data: Dict[str, float]
    _ipogeo_epigeo_data: Dict[str, Dict[str, float]]
    
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
        
        # Load carbon content dataset
        carbon_content_path = Path(__file__).parent.parent.parent / "dataset" / "carbon_content.csv"
        object.__setattr__(self, "_carbon_content_data", self._load_carbon_content(carbon_content_path))
        
        # Load ipogeo/epigeo dataset
        ipogeo_epigeo_path = Path(__file__).parent.parent.parent / "dataset" / "ipogeo_epigeo.csv"
        object.__setattr__(self, "_ipogeo_epigeo_data", self._load_ipogeo_epigeo(ipogeo_epigeo_path))
    
    @staticmethod
    def _load_carbon_content(path: Path) -> Dict[str, float]:
        """Load carbon content data from CSV. Returns dict: species_name -> carbon_fraction."""
        data = {}
        if not path.exists():
            return data
        
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                species = row.get("Species", "").strip().lower()
                mean_str = row.get("Mean", "").strip()
                if species and mean_str:
                    try:
                        # Mean is in percentage, convert to fraction
                        data[species] = float(mean_str) / 100.0
                    except ValueError:
                        continue
        return data
    
    @staticmethod
    def _load_ipogeo_epigeo(path: Path) -> Dict[str, Dict[str, float]]:
        """Load ipogeo/epigeo data from CSV. Returns root-to-shoot ratios for hardwood/softwood."""
        data = {"hardwood": {}, "softwood": {}}
        if not path.exists():
            return data
        
        lines = path.read_text(encoding="utf-8").splitlines()
        # Data starts from line 4 (0-indexed: line 3)
        if len(lines) < 10:
            return data
        
        for raw in lines[3:]:
            parts = [p.strip().strip('"') for p in raw.split(",")]
            if len(parts) < 3:
                continue
            component = parts[0].strip()
            try:
                hw_val = float(parts[1]) if parts[1] else 0
                sw_val = float(parts[2]) if parts[2] else 0
                data["hardwood"][component] = hw_val
                data["softwood"][component] = sw_val
            except ValueError:
                continue
        
        return data
    
    def _get_carbon_fraction(self, species_name: str) -> tuple[float, str]:
        """Get carbon fraction for a species. Returns (fraction, source_description)."""
        if not species_name:
            return DEFAULT_CARBON_FRACTION, f"valore default ({DEFAULT_CARBON_FRACTION*100:.0f}%)"

        # Extract genus from species name (e.g., "Platanus x acerifolia" -> "Platanus")
        genus = species_name.split()[0].lower() if species_name else ""

        # Try to find exact match or partial match in carbon content data
        for species_key, fraction in self._carbon_content_data.items():
            if species_key in species_name.lower() or genus in species_key:
                return fraction, f"da carbon_content.csv per '{species_key}'"

        common_name = GENUS_TO_COMMON_NAME.get(genus, "")
        if common_name and common_name in self._carbon_content_data:
            return self._carbon_content_data[common_name], f"da carbon_content.csv per '{common_name}' (genere {genus})"

        return DEFAULT_CARBON_FRACTION, f"valore default ({DEFAULT_CARBON_FRACTION*100:.0f}%) - specie non trovata in carbon_content.csv"
    
    def _get_root_shoot_ratio(self, species_name: str) -> tuple[float, str]:
        """Get root-to-shoot ratio based on species type (hardwood/softwood)."""
        # Determine if softwood (conifer) or hardwood
        genus = species_name.split()[0].lower() if species_name else ""
        
        is_softwood = genus in SOFTWOOD_GENERA
        group = "softwood" if is_softwood else "hardwood"
        
        # Calculate R/S from ipogeo_epigeo data
        agb = self._ipogeo_epigeo_data.get(group, {}).get("Above-ground biomass", 155.2)
        roots = self._ipogeo_epigeo_data.get(group, {}).get("Roots", 21.8)
        
        if agb > 0:
            ratio = roots / agb
            group_name = "conifere" if is_softwood else "latifoglie"
            return round(ratio, 4), f"da ipogeo_epigeo.csv per {group_name} ({roots}/{agb})"
        
        # Fallback
        return DEFAULT_ROOT_SHOOT_RATIO, "valore default"

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

    def _get_dominant_species(self, df: pd.DataFrame) -> str:
        """Get the most common species in the dataframe."""
        if 'genus_species' not in df.columns:
            return ""
        species_counts = df['genus_species'].value_counts()
        if len(species_counts) > 0:
            return species_counts.index[0]
        return ""

    def _calculate_vienna_metrics(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Calculate metrics for Vienna dataset."""
        # 1. Map Height
        # 1=0-5, 2=6-10, 3=11-15, 4=16-20, 5=21-25, 6=26-30, 7=31-35, 8=>35
        df['height_m'] = df['tree_height'].map(VIENNA_HEIGHT_MAP).fillna(0)
        
        # 2. Calculate DBH (cm)
        df['dbh_cm'] = df['trunk_circumference'] / math.pi
        
        # Get dominant species for parameters
        dominant_species = self._get_dominant_species(df)
        
        return self._apply_allometric_equations(df, dominant_species)

    def _calculate_milano_metrics(self, df: pd.DataFrame) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Calculate metrics for Milano dataset."""
        df['height_m'] = df['height_m'].fillna(0)
        df['dbh_cm'] = df['trunk_diameter_cm'].fillna(0)
        
        # Get dominant species for parameters
        dominant_species = self._get_dominant_species(df)
        
        return self._apply_allometric_equations(df, dominant_species)

    def _apply_allometric_equations(self, df: pd.DataFrame, dominant_species: str = "") -> tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply general allometric equations (Chave et al. 2014) with species-specific parameters."""
        # Get species-specific parameters
        carbon_fraction, cf_source = self._get_carbon_fraction(dominant_species)
        root_shoot_ratio, rs_source = self._get_root_shoot_ratio(dominant_species)
        wood_density = DEFAULT_WOOD_DENSITY
        
        # Store parameter info for result
        params_info = {
            "wood_density": {
                "value": wood_density,
                "unit": "g/cm³",
                "description": "Densità del legno (valore medio)"
            },
            "carbon_fraction": {
                "value": round(carbon_fraction, 4),
                "unit": "adimensionale",
                "description": f"Frazione di carbonio ({round(carbon_fraction * 100, 1)}%) - {cf_source}"
            },
            "root_shoot_ratio": {
                "value": round(root_shoot_ratio, 4),
                "unit": "adimensionale",
                "description": f"Rapporto radici/chioma (R/S) - {rs_source}"
            }
        }
        
        # Filter invalid data
        mask = (df['dbh_cm'] > 0) & (df['height_m'] > 0)
        valid_df = df[mask].copy()
        
        a = CHAVE_INTERCEPT
        b = CHAVE_EXPONENT
        
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
        valid_df['co2_t'] = valid_df['carbon_t'] * CO2_C_RATIO
        
        return valid_df, params_info

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
            params_info = {}
            if self._dataset_type == "vienna":
                results_df, params_info = self._calculate_vienna_metrics(df)
            elif self._dataset_type == "milano":
                results_df, params_info = self._calculate_milano_metrics(df)
            else:
                # Try generic fallback if columns exist
                dominant_species = self._get_dominant_species(df)
                if 'trunk_diameter_cm' in df.columns and 'height_m' in df.columns:
                    df['height_m'] = df['height_m'].fillna(0)
                    df['dbh_cm'] = df['trunk_diameter_cm'].fillna(0)
                    results_df, params_info = self._apply_allometric_equations(df, dominant_species)
                elif 'trunk_circumference' in df.columns:
                    results_df, params_info = self._calculate_vienna_metrics(df)
                else:
                    return {"error": "Dataset format unknown"}

            # Aggregate results
            count = len(results_df)
            total_co2 = results_df['co2_t'].sum()
            avg_co2 = results_df['co2_t'].mean()
            total_carbon = results_df['carbon_t'].sum()
            total_biomass = results_df['total_biomass_t'].sum()
            
            # Calculate additional stats
            total_agb = results_df['agb_t'].sum()
            total_bgb = results_df['bgb_t'].sum()
            
            # Get dominant species for result info
            dominant_species = self._get_dominant_species(df)
            
            # Build answer hint for LLM (this will be used directly as response)
            cf_value = params_info.get("carbon_fraction", {}).get("value", 0.47)
            rs_value = params_info.get("root_shoot_ratio", {}).get("value", 0.24)
            answer_hint = f"""Lo stock di carbonio di {dominant_species} è di **{round(total_carbon, 2):,} t C** (tonnellate di carbonio).

Dettagli:
- Alberi analizzati: {count:,}
- Stock di carbonio: {round(total_carbon, 2):,} t C
- CO2 equivalente: {round(total_co2, 2):,} t CO2
- Biomassa totale: {round(total_biomass, 2):,} t
  - Biomassa epigea (AGB): {round(total_agb, 2):,} t
  - Biomassa ipogea (BGB): {round(total_bgb, 2):,} t

**Formule utilizzate:**
- AGB = 0.0673 × (WD × DBH² × H)^0.976 (Chave et al., 2014)
- BGB = AGB × R/S
- C = Biomassa × CF
- CO2 = C × (44/12)

**Parametri:**
- Densità legno (WD): 0.6 g/cm³
- Frazione carbonio (CF): {cf_value} ({cf_value*100:.1f}%)
- Rapporto R/S: {rs_value}

Tool utilizzati: calculate_co2_aggregate"""
            
            return {
                "natural_query": natural_query,
                "sql_executed": sql,
                "tree_count": count,
                "dominant_species": dominant_species,
                "carbon_stock_t": round(total_carbon, 2),
                "co2_stock_t": round(total_co2, 2),
                "avg_co2_t": round(avg_co2, 4),
                "total_biomass_t": round(total_biomass, 2),
                "above_ground_biomass_t": round(total_agb, 2),
                "below_ground_biomass_t": round(total_bgb, 2),
                "dataset": self._dataset_type,
                "answer_hint": answer_hint,
                "formulas": {
                    "agb": {
                        "name": "Biomassa epigea (AGB)",
                        "equation": "AGB = 0.0673 × (WD × DBH² × H)^0.976",
                        "source": "Chave et al. (2014)",
                        "url": "https://doi.org/10.1111/gcb.12629"
                    },
                    "bgb": {
                        "name": "Biomassa ipogea (BGB)",
                        "equation": "BGB = AGB × R/S",
                        "source": "ipogeo_epigeo.csv",
                        "url": "https://doi.org/10.1007/s004420050128"
                    },
                    "carbon": {
                        "name": "Contenuto di carbonio",
                        "equation": "C = Biomassa totale × CF",
                        "source": "carbon_content.csv (Martin et al., 2018)"
                    },
                    "co2": {
                        "name": "CO2 equivalente",
                        "equation": "CO2 = C × (44/12)",
                        "source": "Stechiometria molecolare"
                    }
                },
                "parameters": params_info
            }

        except Exception as e:
            return {"error": str(e)}

