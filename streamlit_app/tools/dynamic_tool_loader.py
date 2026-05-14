"""Dynamic tool loader that creates tools from JSON definitions.

This module reads tool/formula definitions from a JSON file and creates
LangChain-compatible tools dynamically.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, create_model

from streamlit_app.tools.dynamic_formula_executor import _formula_executor


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
