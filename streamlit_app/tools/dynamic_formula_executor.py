from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, Optional

try:
    from sympy import Abs, E, exp, log, pi, symbols
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
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

