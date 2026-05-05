"""Response building utilities for formatting user-friendly outputs."""

from __future__ import annotations

from typing import List, Literal, Sequence

from langchain_core.messages import BaseMessage, HumanMessage

from streamlit_app.agent.extraction import DataExtractor


class ResponseBuilder:
    """Builder for constructing formatted responses."""
    
    @staticmethod
    def format_dataset_results(results: List[dict], messages: Sequence[BaseMessage], language: Literal["it", "en"] = "it") -> str:
        """Format dataset results as a user-friendly response.
        
        Args:
            results: List of result dictionaries
            messages: Conversation messages for context
            language: Language code ("it" for Italian, "en" for English)
            
        Returns:
            Formatted response string
        """
        # Define language-specific strings
        if language == "en":
            no_results_msg = "I didn't find any results for your request.\n\nTools used: Dataset Query Tool"
            species_header = f"Here are the {len(results)} most common species in the dataset:\n\n"
            trees_label = "trees"
            district_header = f"Here are the {len(results)} districts:\n\n"
            district_label = "District"
            data_label = "\n📊 Data: Vienna Trees Dataset (BAUMKATOGD)\n"
            tools_used_label = "\nTools used: Dataset Query Tool"
            found_results = f"I found {len(results)} results:\n\n"
            and_others = f"\n... and {len(results) - 20} more results\n"
            # Format numbers with English notation (comma for thousands, dot for decimals)
            def format_number(n):
                return f"{n:,}"
        else:
            no_results_msg = "Non ho trovato risultati per la tua richiesta.\n\nTool utilizzati: Dataset Query Tool"
            species_header = f"Ecco le {len(results)} specie più diffuse nel dataset:\n\n"
            trees_label = "alberi"
            district_header = f"Ecco i {len(results)} distretti:\n\n"
            district_label = "Distretto"
            data_label = "\n📊 Dati: Vienna Trees Dataset (BAUMKATOGD)\n"
            tools_used_label = "\nTool utilizzati: Dataset Query Tool"
            found_results = f"Ho trovato {len(results)} risultati:\n\n"
            and_others = f"\n... e altri {len(results) - 20} risultati\n"
            # Format numbers with Italian notation (dot for thousands, comma for decimals)
            def format_number(n):
                return f"{n:,}".replace(",", ".")
        
        if not results:
            return no_results_msg
        
        # Try to extract the original user question for context
        user_question = ""
        for msg in messages:
            if isinstance(msg, HumanMessage):
                user_question = msg.content
        
        # Analyze result structure to determine response format
        first_row = results[0]
        
        # Case 1: Species count results (genus_species + count)
        if isinstance(first_row, dict) and "genus_species" in first_row and "count" in first_row:
            response = species_header
            for i, row in enumerate(results, 1):
                species = row.get("genus_species", "N/A")
                count = row.get("count", 0)
                count_formatted = format_number(count)
                response += f"{i}. **{species}**: {count_formatted} {trees_label}\n"
            
            response += data_label
            response += tools_used_label
            return response
        
        # Case 2: District count results (district + count)
        if isinstance(first_row, dict) and "district" in first_row and "count" in first_row:
            response = district_header
            for i, row in enumerate(results, 1):
                district = row.get("district", "N/A")
                count = row.get("count", 0)
                count_formatted = format_number(count)
                response += f"{i}. {district_label} **{district}**: {count_formatted} {trees_label}\n"
            
            response += data_label
            response += tools_used_label
            return response
        
        # Case 3: Single value result
        if isinstance(first_row, dict) and len(first_row) == 1:
            key, value = list(first_row.items())[0]
            if isinstance(value, (int, float)):
                value_formatted = format_number(value) if isinstance(value, int) else f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if language == "it" else f"{value:,.2f}"
                response = f"**{key}**: {value_formatted}\n\n"
                response += tools_used_label
                return response
        
        # Case 4: Generic results - format as table
        response = found_results
        for i, row in enumerate(results[:20], 1):
            if isinstance(row, dict):
                row_parts = []
                for key, value in row.items():
                    if value is not None:
                        if isinstance(value, (int, float)) and not isinstance(value, bool):
                            if isinstance(value, int):
                                value_str = format_number(value)
                            else:
                                value_str = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if language == "it" else f"{value:,.2f}"
                        else:
                            value_str = str(value)
                        row_parts.append(f"{key}: {value_str}")
                response += f"{i}. {', '.join(row_parts)}\n"
        
        if len(results) > 20:
            response += and_others
        
        response += tools_used_label
        return response
    
    @staticmethod
    def build_dynamic_fallback_response(
        abused_tool: str,
        call_count: int,
        messages: Sequence[BaseMessage],
        extractor: DataExtractor,
    ) -> str:
        """Build a dynamic fallback response when tool loop limit is reached.
        
        Args:
            abused_tool: Name of the tool called too many times
            call_count: Number of times it was called
            messages: Current conversation messages
            extractor: Data extractor instance
            
        Returns:
            User-friendly fallback response
        """
        # Map tool names to user-friendly descriptions
        tool_descriptions = {
            "search_scientific_papers": "ricerca di paper scientifici",
            "query_tree_dataset": "interrogazione del dataset",
            "calculate_co2_sequestration": "calcolo del sequestro di CO2",
            "calculate_co2_aggregate": "calcolo aggregato CO2 (dataset)",
            "calculate_environmental_estimates": "stima ambientale",
            "generate_chart": "generazione di grafici",
            "generate_map": "generazione di mappe",
            "query_species_list": "ricerca nella lista delle specie",
        }
        
        tool_desc = tool_descriptions.get(abused_tool, f"utilizzo del tool {abused_tool}")
        
        # Start with generic message
        fallback_response = (
            f"⚠️ **Limite di ricerca raggiunto**\n\n"
            f"Ho eseguito {call_count} tentativi di {tool_desc} senza trovare una risposta definitiva.\n\n"
        )
        
        # For paper search, extract and show found papers
        if abused_tool == "search_scientific_papers":
            papers_found = extractor.extract_papers(messages)
            
            if papers_found:
                real_papers = [p for p in papers_found if p.get("source") != "error"]
                error_papers = [p for p in papers_found if p.get("source") == "error"]
                
                if real_papers:
                    fallback_response += "**📚 Paper trovati:**\n\n"
                    for i, paper in enumerate(real_papers[:5], 1):
                        title = paper.get("title", "Titolo non disponibile")
                        authors = paper.get("authors", "")
                        link = paper.get("link", "")
                        abstract = paper.get("abstract", "")
                        if abstract and len(abstract) > 200:
                            abstract = abstract[:200] + "..."
                        
                        fallback_response += f"{i}. **{title}**\n"
                        if authors and authors != "N/A":
                            fallback_response += f"   - Autori: {authors}\n"
                        if abstract:
                            fallback_response += f"   - Abstract: {abstract}\n"
                        if link:
                            fallback_response += f"   - 🔗 [Link al paper]({link})\n"
                        fallback_response += "\n"
                elif error_papers:
                    error_msg = error_papers[0].get("abstract", "errore sconosciuto")
                    fallback_response += f"*⚠️ Errore durante la ricerca: {error_msg}*\n\n"
                else:
                    fallback_response += "*Non ho trovato risultati specifici per la tua query.*\n\n"
            else:
                fallback_response += "*Non ho trovato risultati specifici per la tua query.*\n\n"
        
        # For dataset queries, extract and show results
        elif abused_tool == "query_tree_dataset":
            dataset_results = extractor.extract_dataset_results(messages)
            
            if dataset_results:
                fallback_response = "Ecco i risultati della tua richiesta:\n\n"
                
                for i, row in enumerate(dataset_results[:20], 1):
                    if isinstance(row, dict):
                        if "genus_species" in row and "count" in row:
                            species = row.get("genus_species", "N/A")
                            count = row.get("count", 0)
                            fallback_response += f"{i}. **{species}**: {count:,} alberi\n"
                        elif "district" in row and "count" in row:
                            district = row.get("district", "N/A")
                            count = row.get("count", 0)
                            fallback_response += f"{i}. Distretto {district}: {count:,} alberi\n"
                        else:
                            row_str = ", ".join([f"{k}: {v}" for k, v in row.items() if v is not None])
                            fallback_response += f"{i}. {row_str}\n"
                
                fallback_response += "\n"
            else:
                fallback_response += (
                    "**Suggerimenti:**\n"
                    "- Prova a riformulare la domanda in modo più specifico\n"
                    "- Verifica che i nomi delle colonne siano corretti\n"
                    "- Chiedi prima la struttura del dataset con \"Mostrami le colonne disponibili\"\n\n"
                )
        
        # Generic suggestions for other tools
        else:
            fallback_response += (
                "**Cosa puoi fare:**\n"
                "- Riformula la domanda in modo più specifico\n"
                "- Suddividi la richiesta in domande più semplici\n"
                "- Chiedi informazioni più mirate\n\n"
            )
        
        # Add available tools suggestion
        fallback_response += (
            "**Altri tool disponibili:**\n"
            "- 📊 Analisi dataset (query, statistiche, grafici)\n"
            "- 🌳 Calcoli forestali (CO2, biomassa, volume)\n"
            "- 🗺️ Mappe interattive (solo dataset con coordinate GPS)\n"
            "- 📚 Ricerca paper scientifici\n\n"
            "Posso aiutarti con qualcosa di specifico?\n\n"
            f"Tool utilizzati: {abused_tool.replace('_', ' ').title()}"
        )
        
        return fallback_response
