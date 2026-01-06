"""Test for CarbonContentTool - verifies tool uses real data from carbon_content.csv"""

import pytest
from pathlib import Path

from streamlit_app.tools.carbon_content_tool import CarbonContentTool


def test_carbon_content_tool_maple():
    """Test lookup for Maple species."""
    tool = CarbonContentTool()
    result = tool._run(species="Maple")
    
    assert result["found"] is True
    assert result["species_name"] == "Maple"
    assert result["carbon_fraction"] == 0.495  # 49.5% from CSV
    assert result["carbon_percent"] == 49.5
    assert "48.0-51.5" in result["range_percent"]
    assert result["number_of_studies"] == 8


def test_carbon_content_tool_acer_platanoides():
    """Test lookup for Acer platanoides (should suggest Maple as similar)."""
    tool = CarbonContentTool()
    result = tool._run(species="Acer platanoides")
    
    # "Acer platanoides" is not in the CSV (which uses common names), but should find Maple as partial match
    # If not found, it should provide suggestions
    if result["found"]:
        assert result["species_name"] == "Maple"
        assert result["carbon_fraction"] == 0.495
    else:
        # If not found, should provide helpful suggestions
        assert result["found"] is False
        assert "similar_species" in result or "available_species_count" in result


def test_carbon_content_tool_pine():
    """Test lookup for Pine species."""
    tool = CarbonContentTool()
    result = tool._run(species="Pine")
    
    assert result["found"] is True
    assert result["species_name"] == "Pine"
    assert result["carbon_fraction"] == 0.499  # 49.9% from CSV
    assert result["carbon_percent"] == 49.9
    assert result["number_of_studies"] == 27  # Most studied species


def test_carbon_content_tool_oak():
    """Test lookup for Oak species."""
    tool = CarbonContentTool()
    result = tool._run(species="Oak")
    
    assert result["found"] is True
    assert result["species_name"] == "Oak"
    assert result["carbon_fraction"] == 0.494  # 49.4% from CSV
    assert result["carbon_percent"] == 49.4


def test_carbon_content_tool_case_insensitive():
    """Test case insensitive lookup."""
    tool = CarbonContentTool()
    
    result1 = tool._run(species="maple")
    result2 = tool._run(species="MAPLE")
    result3 = tool._run(species="Maple")
    
    assert result1["found"] is True
    assert result2["found"] is True
    assert result3["found"] is True
    assert result1["carbon_fraction"] == result2["carbon_fraction"] == result3["carbon_fraction"]


def test_carbon_content_tool_not_found():
    """Test species not in dataset."""
    tool = CarbonContentTool()
    result = tool._run(species="NonExistentSpecies123")
    
    assert result["found"] is False
    assert "NonExistentSpecies123" in result["requested_species"]
    assert "non trovata" in result["message"].lower()


def test_carbon_content_tool_list_all_species():
    """Test listing all available species."""
    tool = CarbonContentTool()
    result = tool._run(species=None)
    
    assert "available_species" in result
    assert result["total_count"] > 0
    assert "Maple" in result["available_species"]
    assert "Oak" in result["available_species"]
    assert "Pine" in result["available_species"]
    assert len(result["available_species"]) == 50  # Total species in CSV (excluding header)


def test_carbon_content_tool_returns_scientific_source():
    """Test that tool returns scientific source information."""
    tool = CarbonContentTool()
    result = tool._run(species="Oak")
    
    assert "scientific_source" in result
    assert "title" in result["scientific_source"]
    assert "url" in result["scientific_source"]
    assert "Carbon Content of Tree Tissues" in result["scientific_source"]["title"]


def test_carbon_content_tool_uses_real_csv_data():
    """Test that tool uses real CSV data, not hardcoded values."""
    tool = CarbonContentTool()
    csv_path = Path(__file__).parent.parent / "dataset" / "carbon_content.csv"
    
    # Verify CSV exists
    assert csv_path.exists(), "carbon_content.csv should exist"
    
    # Verify tool is using the CSV
    assert tool._csv_path.exists()
    
    # Read a few species and verify values match CSV
    result_beech = tool._run(species="Beech")
    assert result_beech["carbon_fraction"] == 0.493  # 49.3% from CSV line 7
    
    result_birch = tool._run(species="Birch")
    assert result_birch["carbon_fraction"] == 0.494  # 49.4% from CSV line 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

