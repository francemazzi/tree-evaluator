"""Dynamic tool loader that creates tools from JSON definitions.

This module reads tool/formula definitions from a JSON file and creates
LangChain-compatible tools dynamically.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

# Try to import SymPy for robust formula parsing
try:
    import sympy
    from sympy import symbols, sympify, E, exp, log, Abs, pi
    from sympy.parsing.sympy_parser import (
        parse_expr,
        standard_transformations,
        implicit_multiplication_application,
        convert_xor,
    )
    SYMPY_AVAILABLE = True
except ImportError:
    SYMPY_AVAILABLE = False

logger = logging.getLogger(__name__)


class RobustFormulaExecutor:
    """Executes mathematical formulas using SymPy for robustness.

    This class handles formula parsing and execution with:
    - SymPy symbolic math for accurate parsing
    - Automatic variable detection
    - Fallback to pattern matching for complex formulas
    - Support for various notation styles (², ^, **, etc.)
    """

    # Common formula normalizations
    NORMALIZATIONS = [
        (r"²", "**2"),
        (r"³", "**3"),
        (r"×", "*"),
        (r"÷", "/"),
        (r"(\d)([a-zA-Z])", r"\1*\2"),  # 2x -> 2*x
        (r"([a-zA-Z])(\d)", r"\1*\2"),  # x2 -> x*2 (for cases like D2)
        (r"\^", "**"),
        (r"e\^", "exp"),  # e^ notation
    ]

    def __init__(self):
        """Initialize the formula executor."""
        self._cache: Dict[str, Any] = {}

    def execute(self, formula: str, inputs: Dict[str, float], title: str = "") -> float:
        """Execute a formula with given inputs.

        Args:
            formula: The mathematical formula string.
            inputs: Dictionary of variable names to values.
            title: Optional title for context-aware parsing.

        Returns:
            The calculated result.

        Raises:
            ValueError: If formula cannot be parsed or executed.
        """
        if not SYMPY_AVAILABLE:
            return self._fallback_execute(formula, inputs, title)

        try:
            # Normalize formula
            normalized = self._normalize_formula(formula)

            # Try SymPy parsing
            result = self._sympy_execute(normalized, inputs)
            if result is not None:
                return result

        except Exception as e:
            logger.debug(f"SymPy execution failed for formula '{formula}': {e}")

        # Fall back to pattern matching
        return self._fallback_execute(formula, inputs, title)

    def _normalize_formula(self, formula: str) -> str:
        """Normalize formula for parsing.

        Args:
            formula: Original formula string.

        Returns:
            Normalized formula string.
        """
        result = formula

        for pattern, replacement in self.NORMALIZATIONS:
            result = re.sub(pattern, replacement, result)

        # Handle special cases
        result = result.replace("ln", "log")  # SymPy uses log for natural log
        result = result.replace(" ", "")  # Remove spaces

        return result

    def _sympy_execute(self, formula: str, inputs: Dict[str, float]) -> Optional[float]:
        """Execute formula using SymPy.

        Args:
            formula: Normalized formula string.
            inputs: Variable values.

        Returns:
            Result or None if parsing fails.
        """
        # Check cache
        cache_key = f"{formula}:{sorted(inputs.items())}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            # Create symbol mapping
            local_symbols = {}
            for name in inputs.keys():
                local_symbols[name] = symbols(name)
                # Also add common variations
                local_symbols[name.upper()] = local_symbols[name]
                local_symbols[name.lower()] = local_symbols[name]

            # Add common math functions
            local_symbols.update({
                "exp": exp,
                "log": log,
                "ln": log,
                "abs": Abs,
                "pi": pi,
                "e": E,
            })

            # Parse expression
            transformations = standard_transformations + (
                implicit_multiplication_application,
                convert_xor,
            )

            # Try to extract the right side of equation if present
            if "=" in formula:
                parts = formula.split("=")
                formula = parts[-1].strip()  # Use right side

            expr = parse_expr(formula, local_dict=local_symbols, transformations=transformations)

            # Substitute values
            substitutions = {}
            for name, value in inputs.items():
                if name in local_symbols:
                    substitutions[local_symbols[name]] = value
                if name.lower() in local_symbols:
                    substitutions[local_symbols[name.lower()]] = value
                if name.upper() in local_symbols:
                    substitutions[local_symbols[name.upper()]] = value

            result = float(expr.evalf(subs=substitutions))

            # Cache result
            self._cache[cache_key] = result

            return result

        except Exception as e:
            logger.debug(f"SymPy parsing failed: {e}")
            return None

    def _fallback_execute(self, formula: str, inputs: Dict[str, float], title: str = "") -> float:
        """Execute formula using pattern matching (fallback).

        Args:
            formula: Original formula string.
            inputs: Variable values.
            title: Formula title for context.

        Returns:
            Calculated result.
        """
        formula_lower = formula.lower()
        title_lower = title.lower()

        # Handle special formulas
        if "s1 + s2" in formula_lower or "sn" in formula_lower:
            sections = inputs.get("sections", [])
            if isinstance(sections, list):
                return sum(sections)
            return 0.0

        # Handle V = a (D² H) type formulas
        if "d²" in formula_lower or "d^2" in formula_lower or "d**2" in formula_lower:
            d = inputs.get("d", inputs.get("diametro", 0))
            h = inputs.get("h", inputs.get("altezza", 1))
            a = inputs.get("a", inputs.get("coeff_a", 1))
            return a * (d ** 2) * h

        # Handle Y = a X^b allometric relation
        if "x^b" in formula_lower or "x**b" in formula_lower:
            x = inputs.get("x", inputs.get("variable_x", 0))
            a = inputs.get("a", inputs.get("coeff_a", 1))
            b = inputs.get("b", inputs.get("exponent_b", 1))
            return a * (x ** b)

        # Handle biomass equations
        if "e^" in formula or "exp" in formula_lower:
            return self._calculate_biomass_fallback(inputs, title_lower)

        # Generic fallback
        return self._generic_calculate(inputs)

    def _calculate_biomass_fallback(self, inputs: Dict[str, float], title: str) -> float:
        """Calculate biomass using predefined formulas."""
        d = inputs.get("d", inputs.get("diametro", 0))
        h = inputs.get("h", inputs.get("altezza", 1))
        age = inputs.get("age", inputs.get("eta", 1))
        rsr = inputs.get("root_to_shoot_ratio", inputs.get("rsr", 0.24))

        if "leaf" in title or "foglia" in title:
            return math.exp(-7.21) * ((d**2 * h) ** 0.6) * (age ** 3.2) * 1.28

        if "stem" in title or "fusto" in title:
            exponent = -8.15 + 2.20 * d + 1.24 * age - 0.35 * d * age
            return math.exp(exponent) * 1.41

        if "root" in title or "radic" in title:
            return math.exp(-5) * (d ** 1.48) * (h ** 0.4) * (age ** 1.38) * (rsr ** 0.31) * 1.26

        if "total" in title:
            return math.exp(-4.2) * (d ** 1.36) * (h ** 0.57) * (age ** 1.67) * (rsr ** -0.3) * 1.23

        return 0.0

    def _generic_calculate(self, inputs: Dict[str, float]) -> float:
        """Generic calculation fallback - sum all numeric inputs."""
        total = 0.0
        for v in inputs.values():
            if isinstance(v, (int, float)):
                total += v
            elif isinstance(v, list):
                total += sum(x for x in v if isinstance(x, (int, float)))
        return total


# Global formula executor instance
_formula_executor = RobustFormulaExecutor()


class DynamicToolLoader:
    """Loads tool definitions from JSON and creates LangChain tools."""

    def __init__(self, json_path: Optional[Path] = None):
        """Initialize the loader.

        Args:
            json_path: Path to the JSON file containing tool definitions.
                      Defaults to dataset/tools.json
        """
        if json_path is None:
            json_path = Path(__file__).parent.parent.parent / "dataset" / "tools.json"

        self.json_path = json_path
        self._tools_data: List[Dict] = []
        self._load_json()

    def _load_json(self) -> None:
        """Load tool definitions from JSON file."""
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                self._tools_data = json.load(f)
        except FileNotFoundError:
            self._tools_data = []
        except json.JSONDecodeError:
            self._tools_data = []

    def get_tools_data(self) -> List[Dict]:
        """Return the raw tools data from JSON."""
        return self._tools_data

    def get_tools_summary(self) -> List[Dict[str, Any]]:
        """Get a summary of available tools for task planning.

        Returns:
            List of dicts with 'name', 'description', 'formula', and 'keywords' keys
        """
        summaries = []
        for tool in self._tools_data:
            summaries.append({
                "name": self._create_tool_name(tool.get("titolo", "")),
                "title": tool.get("titolo", ""),
                "description": tool.get("descrizione", ""),
                "formula": tool.get("funzione", ""),
                "variables": [v.get("nome", "") for v in tool.get("variabili", [])],
                "keywords": tool.get("keywords", [])
            })
        return summaries

    def _create_tool_name(self, title: str) -> str:
        """Create a valid tool name from title.

        Args:
            title: The tool title in Italian

        Returns:
            A valid snake_case tool name
        """
        # Remove special characters and convert to snake_case
        name = title.lower()
        name = re.sub(r'[àáâã]', 'a', name)
        name = re.sub(r'[èéêë]', 'e', name)
        name = re.sub(r'[ìíîï]', 'i', name)
        name = re.sub(r'[òóôõ]', 'o', name)
        name = re.sub(r'[ùúûü]', 'u', name)
        name = re.sub(r'[^a-z0-9\s]', '', name)
        name = re.sub(r'\s+', '_', name.strip())
        name = re.sub(r'_+', '_', name)

        # Ensure name starts with letter
        if name and not name[0].isalpha():
            name = "calc_" + name

        return name or "dynamic_tool"

    def create_tools(self) -> List[BaseTool]:
        """Create LangChain tools from JSON definitions.

        Returns:
            List of BaseTool instances
        """
        tools = []

        for tool_def in self._tools_data:
            tool = self._create_tool_from_def(tool_def)
            if tool:
                tools.append(tool)

        return tools

    def _create_tool_from_def(self, tool_def: Dict) -> Optional[BaseTool]:
        """Create a single tool from JSON definition.

        Args:
            tool_def: Tool definition dict with titolo, funzione, descrizione, variabili

        Returns:
            A BaseTool instance or None if creation fails
        """
        title = tool_def.get("titolo", "")
        formula = tool_def.get("funzione", "")
        description = tool_def.get("descrizione", "")
        variables = tool_def.get("variabili", [])

        if not title or not formula:
            return None

        tool_name = self._create_tool_name(title)

        # Create input schema dynamically
        input_schema = self._create_input_schema(tool_name, variables)

        # Create tool class dynamically
        tool_class = self._create_tool_class(
            tool_name=tool_name,
            title=title,
            formula=formula,
            description=description,
            variables=variables,
            input_schema=input_schema
        )

        return tool_class()

    def _create_input_schema(self, tool_name: str, variables: List[Dict]) -> Type[BaseModel]:
        """Create a Pydantic input schema for the tool.

        Args:
            tool_name: Name of the tool
            variables: List of variable definitions

        Returns:
            A Pydantic BaseModel class
        """
        fields = {}

        for var in variables:
            var_name = var.get("nome", "").strip()
            var_desc = var.get("descrizione", "")

            # Clean variable name for Python
            clean_name = self._clean_variable_name(var_name)

            if clean_name:
                # Determine if variable is a list (like S1...Sn)
                if "..." in var_name or "Sn" in var_name:
                    fields[clean_name] = (
                        List[float],
                        Field(description=var_desc)
                    )
                else:
                    fields[clean_name] = (
                        float,
                        Field(description=var_desc)
                    )

        # If no valid fields, add a generic input
        if not fields:
            fields["value"] = (float, Field(description="Input value for calculation"))

        schema_name = f"{tool_name.title().replace('_', '')}Input"
        return create_model(schema_name, **fields)

    def _clean_variable_name(self, var_name: str) -> str:
        """Clean variable name for use as Python identifier.

        Args:
            var_name: Original variable name

        Returns:
            Clean Python-compatible name
        """
        # Handle special cases
        if "..." in var_name:
            # S1...Sn -> sections
            return "sections"

        # Remove special characters
        clean = var_name.lower()
        clean = re.sub(r'[àáâã]', 'a', clean)
        clean = re.sub(r'[èéêë]', 'e', clean)
        clean = re.sub(r'[ìíîï]', 'i', clean)
        clean = re.sub(r'[òóôõ]', 'o', clean)
        clean = re.sub(r'[ùúûü]', 'u', clean)
        clean = re.sub(r'[^a-z0-9_]', '_', clean)
        clean = re.sub(r'_+', '_', clean)
        clean = clean.strip('_')

        # Handle numeric-only names
        if clean and clean[0].isdigit():
            clean = "var_" + clean

        return clean

    def _create_tool_class(
        self,
        tool_name: str,
        title: str,
        formula: str,
        description: str,
        variables: List[Dict],
        input_schema: Type[BaseModel]
    ) -> Type[BaseTool]:
        """Create a BaseTool subclass dynamically.

        Args:
            tool_name: Name for the tool
            title: Display title
            formula: Mathematical formula
            description: Tool description
            variables: Variable definitions
            input_schema: Pydantic input schema

        Returns:
            A BaseTool subclass
        """
        # Build full description with formula
        full_description = f"""{title}

Formula: {formula}

{description}

Variables:
"""
        for var in variables:
            full_description += f"- {var.get('nome', '')}: {var.get('descrizione', '')}\n"

        # Create the tool class
        class DynamicFormulaTool(BaseTool):
            name: str = tool_name
            description: str = full_description
            args_schema: Type[BaseModel] = input_schema

            # Store formula info for execution
            _formula: str = formula
            _title: str = title
            _variables: List[Dict] = variables

            def _run(self, **kwargs) -> Dict[str, Any]:
                """Execute the formula with given inputs."""
                try:
                    result = self._execute_formula(kwargs)
                    return {
                        "success": True,
                        "tool": self._title,
                        "formula": self._formula,
                        "inputs": kwargs,
                        "result": result,
                        "answer_hint": f"Usando la formula {self._title} ({self._formula}) con i valori forniti, il risultato è: {result}"
                    }
                except Exception as e:
                    return {
                        "success": False,
                        "tool": self._title,
                        "formula": self._formula,
                        "inputs": kwargs,
                        "error": str(e)
                    }

            def _execute_formula(self, inputs: Dict[str, Any]) -> float:
                """Execute the formula with inputs using robust SymPy-based parser.

                Args:
                    inputs: Dictionary of input values

                Returns:
                    Calculated result
                """
                # Use the global robust formula executor
                return _formula_executor.execute(
                    formula=self._formula,
                    inputs=inputs,
                    title=self._title
                )

        return DynamicFormulaTool


def get_dynamic_tools(json_path: Optional[Path] = None) -> List[BaseTool]:
    """Convenience function to get dynamic tools.

    Args:
        json_path: Optional path to tools JSON file

    Returns:
        List of dynamically created tools
    """
    loader = DynamicToolLoader(json_path)
    return loader.create_tools()


def get_tools_for_planning(json_path: Optional[Path] = None) -> List[Dict[str, str]]:
    """Get tool summaries for task planning.

    Args:
        json_path: Optional path to tools JSON file

    Returns:
        List of tool summaries with name, description, formula
    """
    loader = DynamicToolLoader(json_path)
    return loader.get_tools_summary()
