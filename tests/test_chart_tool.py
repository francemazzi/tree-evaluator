"""
Integration tests for Chart Generation Tool.

Run with: pytest tests/test_chart_tool.py -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from streamlit_app.tools.chart_tool import ChartGenerationTool

# Load environment variables
load_dotenv()


@pytest.fixture
def llm() -> ChatOpenAI:
    """Create LLM instance for testing."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set in environment")
    
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=api_key,
    )


@pytest.fixture
def chart_tool(llm: ChatOpenAI) -> ChartGenerationTool:
    """Create chart tool instance."""
    db_path = Path(__file__).parent.parent / "dataset" / "BAUMKATOGD.db"
    
    if not db_path.exists():
        pytest.skip(f"Database not found at {db_path}")
    
    return ChartGenerationTool(db_path=db_path, llm=llm)


class TestChartToolBarChart:
    """Test bar chart generation."""
    
    def test_bar_chart_districts(self, chart_tool: ChartGenerationTool) -> None:
        """Test bar chart of trees per district."""
        result = chart_tool._run(
            chart_type="bar",
            data_query="Numero di alberi per distretto"
        )
        
        assert result["success"] is True
        assert "chart_json" in result
        assert result["chart_type"] == "bar"
        assert result["data_points"] > 0
        
        # Verify chart JSON is valid
        chart_json = json.loads(result["chart_json"])
        assert "data" in chart_json
        assert "layout" in chart_json
    
    def test_bar_chart_top_species(self, chart_tool: ChartGenerationTool) -> None:
        """Test bar chart of top species."""
        result = chart_tool._run(
            chart_type="bar",
            data_query="Top 10 specie più comuni"
        )
        
        assert result["success"] is True
        assert result["data_points"] <= 10
        assert "sql_executed" in result


class TestChartToolPieChart:
    """Test pie chart generation."""
    
    def test_pie_chart_species(self, chart_tool: ChartGenerationTool) -> None:
        """Test pie chart of species distribution."""
        result = chart_tool._run(
            chart_type="pie",
            data_query="Distribuzione delle 5 specie principali"
        )
        
        assert result["success"] is True
        assert "chart_json" in result
        assert result["chart_type"] == "pie"


class TestChartToolHistogram:
    """Test histogram generation."""
    
    def test_histogram_age_distribution(self, chart_tool: ChartGenerationTool) -> None:
        """Test histogram of tree age distribution."""
        result = chart_tool._run(
            chart_type="histogram",
            data_query="Distribuzione dell'età degli alberi"
        )
        
        assert result["success"] is True
        assert result["chart_type"] == "histogram"
        
        # Histogram should have many data points
        assert result["data_points"] > 100


class TestChartToolLineChart:
    """Test line chart generation."""
    
    def test_line_chart_plantings_over_time(self, chart_tool: ChartGenerationTool) -> None:
        """Test line chart of plantings over time."""
        result = chart_tool._run(
            chart_type="line",
            data_query="Andamento delle piantumazioni per anno dal 1950"
        )
        
        assert result["success"] is True
        assert result["chart_type"] == "line"
        assert "sql_executed" in result


class TestChartToolScatterPlot:
    """Test scatter plot generation."""
    
    def test_scatter_plot_circumference_vs_year(self, chart_tool: ChartGenerationTool) -> None:
        """Test scatter plot of circumference vs year."""
        result = chart_tool._run(
            chart_type="scatter",
            data_query="Relazione tra anno di piantagione e circonferenza"
        )
        
        assert result["success"] is True
        assert result["chart_type"] == "scatter"


class TestChartToolBoxPlot:
    """Test box plot generation."""
    
    def test_box_plot_circumference_by_species(self, chart_tool: ChartGenerationTool) -> None:
        """Test box plot of circumference by species."""
        result = chart_tool._run(
            chart_type="box",
            data_query="Distribuzione della circonferenza per le specie principali"
        )
        
        assert result["success"] is True
        assert result["chart_type"] == "box"


class TestChartToolCustomization:
    """Test chart customization options."""
    
    def test_custom_title(self, chart_tool: ChartGenerationTool) -> None:
        """Test custom chart title."""
        custom_title = "Il Mio Grafico Personalizzato"
        
        result = chart_tool._run(
            chart_type="bar",
            data_query="Numero alberi per distretto",
            title=custom_title
        )
        
        assert result["success"] is True
        assert result["title"] == custom_title
    
    def test_custom_labels(self, chart_tool: ChartGenerationTool) -> None:
        """Test custom axis labels."""
        result = chart_tool._run(
            chart_type="bar",
            data_query="Numero alberi per distretto",
            x_label="Distretti Viennesi",
            y_label="Conteggio Alberi"
        )
        
        assert result["success"] is True
        
        # Verify labels are in chart JSON
        chart_json = json.loads(result["chart_json"])
        layout = chart_json.get("layout", {})
        assert "xaxis" in layout or "yaxis" in layout


class TestChartToolErrorHandling:
    """Test error handling."""
    
    def test_empty_query_result(self, chart_tool: ChartGenerationTool) -> None:
        """Test handling of query that returns no data."""
        result = chart_tool._run(
            chart_type="bar",
            data_query="Alberi piantati nell'anno 3000"
        )
        
        # Should still succeed but with no data or fail gracefully
        if result["success"]:
            assert result["data_points"] == 0 or "error" in result.get("description", "")
    
    def test_invalid_chart_type(self, chart_tool: ChartGenerationTool) -> None:
        """Test invalid chart type handling."""
        # This should fail at Pydantic validation level
        with pytest.raises(Exception):
            chart_tool._run(
                chart_type="invalid_type",  # type: ignore
                data_query="Some query"
            )


class TestChartToolSQLGeneration:
    """Test SQL generation from natural language."""
    
    def test_sql_contains_aggregation(self, chart_tool: ChartGenerationTool) -> None:
        """Test that SQL contains proper aggregation for counts."""
        result = chart_tool._run(
            chart_type="bar",
            data_query="Numero di alberi per distretto"
        )
        
        sql = result["sql_executed"].lower()
        assert "count" in sql
        assert "group by" in sql
    
    def test_sql_contains_limit(self, chart_tool: ChartGenerationTool) -> None:
        """Test that SQL contains LIMIT for top N queries."""
        result = chart_tool._run(
            chart_type="bar",
            data_query="Top 5 specie"
        )
        
        sql = result["sql_executed"].lower()
        assert "limit" in sql
        assert "5" in sql or result["data_points"] <= 5


class TestChartToolIntegration:
    """Integration tests with LangChain agent."""
    
    def test_tool_invocation_via_langchain(self, chart_tool: ChartGenerationTool) -> None:
        """Test that tool can be invoked through LangChain interface."""
        # Test using LangChain's invoke method
        result = chart_tool.invoke({
            "chart_type": "bar",
            "data_query": "alberi per distretto"
        })
        
        assert isinstance(result, dict)
        assert "success" in result
    
    def test_tool_description_parsing(self, chart_tool: ChartGenerationTool) -> None:
        """Test that tool description is properly formatted."""
        assert chart_tool.name == "generate_chart"
        assert "interactive" in chart_tool.description.lower()
        assert "bar" in chart_tool.description
        assert "pie" in chart_tool.description


@pytest.mark.slow
class TestChartToolPerformance:
    """Performance tests for chart generation."""
    
    def test_large_dataset_histogram(self, chart_tool: ChartGenerationTool) -> None:
        """Test histogram with large dataset."""
        import time
        
        start = time.time()
        result = chart_tool._run(
            chart_type="histogram",
            data_query="Distribuzione di tutte le circonferenze"
        )
        elapsed = time.time() - start
        
        assert result["success"] is True
        assert elapsed < 10.0  # Should complete within 10 seconds
    
    def test_complex_aggregation(self, chart_tool: ChartGenerationTool) -> None:
        """Test complex aggregation query performance."""
        import time
        
        start = time.time()
        result = chart_tool._run(
            chart_type="bar",
            data_query="Media circonferenza per distretto con più di 1000 alberi"
        )
        elapsed = time.time() - start
        
        assert result["success"] is True
        assert elapsed < 15.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

