"""Formatting utilities for numbers and text in Italian locale."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Optional


class ItalianNumberFormatter:
    """Formatter for numbers following Italian conventions (dot for thousands, comma for decimals)."""
    
    @staticmethod
    def format_number(value: float, preserve_decimals: Optional[int] = None) -> str:
        """Format a number following Italian conventions.
        
        Args:
            value: The number to format
            preserve_decimals: If specified, preserve exactly this many decimal places
            
        Returns:
            Formatted string (e.g., "1.234,56" for 1234.56)
        """
        try:
            d = Decimal(str(value))
        except (InvalidOperation, ValueError):
            return str(value)

        sign = "-" if d.is_signed() else ""
        mag = d.copy_abs()
        tup = mag.as_tuple()
        digits = "".join(str(x) for x in tup.digits) or "0"
        exp = tup.exponent

        if exp >= 0:
            digits = digits + ("0" * exp)
            int_part = digits or "0"
            frac_part = ""
        else:
            places = -exp
            if len(digits) <= places:
                digits = digits.zfill(places + 1)
            int_part = digits[:-places] or "0"
            frac_part = digits[-places:]

        if preserve_decimals is None:
            frac_part = frac_part.rstrip("0")
        else:
            frac_part = (frac_part + ("0" * max(0, preserve_decimals - len(frac_part))))[:preserve_decimals]

        groups = []
        while len(int_part) > 3:
            groups.append(int_part[-3:])
            int_part = int_part[:-3]
        groups.append(int_part)
        int_grouped = ".".join(reversed(groups))

        if frac_part:
            return f"{sign}{int_grouped},{frac_part}"
        return f"{sign}{int_grouped}"
    
    @staticmethod
    def extract_first_numeric(text: str) -> Optional[float]:
        """Extract the first numeric value from text.
        
        Handles various numeric formats including Italian formatting.
        
        Args:
            text: Text to extract number from
            
        Returns:
            First numeric value found, or None
        """
        pattern = re.compile(r"\d[\d\.\,\s\u00a0\u202f']*")
        for m in pattern.findall(text or ""):
            cleaned = m.replace("\u00a0", " ").replace("\u202f", " ").replace(" ", "")
            cleaned = cleaned.replace(".", "").replace(",", ".").replace("'", "")
            try:
                return float(cleaned)
            except ValueError:
                continue
        return None

