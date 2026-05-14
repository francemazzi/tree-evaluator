from __future__ import annotations

from typing import Any, Dict

from streamlit_app.agent.translations import Language, format_translation, get_translation


def format_tool_results(result_data: Dict[str, Any], language: Language = "it") -> str:
    """Format tool results for display.
    
    Args:
        result_data: Tool result data
        language: Language for messages ("it" or "en")
    """
    output = ""
    
    # SQL query
    if "sql_executed" in result_data:
        sql = result_data.get("sql_executed", "")
        output += f"{get_translation('sql_query_generated', language)}\n```sql\n{sql}\n```\n\n"
    
    # Row count with vector search info
    if "row_count" in result_data:
        row_count = result_data.get("row_count", 0)
        
        if result_data.get("vector_search_applied", False):
            total_found = result_data.get("total_rows_found", row_count)
            output += f"{get_translation('vector_search_applied', language)}\n"
            output += f"📊 {get_translation('total_rows_found', language)}: {total_found}\n"
            output += f"✨ {get_translation('top_relevant_results', language)}: {row_count}\n"
            if "info" in result_data:
                output += f"ℹ️  {result_data['info']}\n"
        else:
            output += f"📊 {get_translation('rows_found', language)}: {row_count}\n"
            if "warning" in result_data:
                output += f"⚠️  {get_translation('warning', language)}: {result_data['warning']}\n"
        output += "\n"
    
    # Results preview
    if "results" in result_data:
        results = result_data.get("results", [])
        if results:
            output += f"{get_translation('first_results', language)}\n\n<ol>\n"
            for row in results[:3]:
                output += "<li>"
                if "genus_species" in row:
                    output += f"{get_translation('species', language)}: {row['genus_species']} "
                if "count" in row:
                    output += f"{get_translation('count', language)}: {row['count']} "
                if "district" in row:
                    output += f"{get_translation('district', language)}: {row['district']} "
                if "trunk_circumference" in row:
                    output += f"{get_translation('circumference', language)}: {row['trunk_circumference']}cm "
                output += "</li>\n"
            output += "</ol>\n"
            
            if len(results) > 3:
                output += format_translation('and_others', language, count=len(results) - 3) + "\n"
    
    # Single value results
    elif "result" in result_data and "column" in result_data:
        result_val = result_data.get("result")
        column_name = result_data.get("column")
        output += f"**{column_name}**: {result_val}\n"
    
    # CO2 results (single tree - calculate_co2_sequestration)
    if "co2_sequestration_kg" in result_data:
        co2 = result_data.get("co2_sequestration_kg", 0)
        output += f"🌱 {get_translation('co2_sequestered', language)}: {co2} kg\n"

    # CO2 aggregate results (calculate_co2_aggregate)
    if "carbon_stock_t" in result_data or "co2_stock_t" in result_data:
        tree_count = result_data.get("tree_count", 0)
        carbon_stock = result_data.get("carbon_stock_t", 0)
        co2_stock = result_data.get("co2_stock_t", 0)
        total_biomass = result_data.get("total_biomass_t", 0)
        species = result_data.get("dominant_species", "")

        if language == "en":
            output += f"\n🌳 **CO2 Aggregate Calculation**\n"
            output += f"- Trees analyzed: **{tree_count:,}**\n"
            if species:
                output += f"- Dominant species: {species}\n"
            output += f"- Carbon stock: **{carbon_stock:,.2f} t C**\n"
            output += f"- CO2 equivalent: **{co2_stock:,.2f} t CO2**\n"
            output += f"- Total biomass: {total_biomass:,.2f} t\n"
        else:
            output += f"\n🌳 **Calcolo CO2 Aggregato**\n"
            output += f"- Alberi analizzati: **{tree_count:,}**\n"
            if species:
                output += f"- Specie dominante: {species}\n"
            output += f"- Stock di carbonio: **{carbon_stock:,.2f} t C**\n"
            output += f"- CO2 equivalente: **{co2_stock:,.2f} t CO2**\n"
            output += f"- Biomassa totale: {total_biomass:,.2f} t\n"

    return output
