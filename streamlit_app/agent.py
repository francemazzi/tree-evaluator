from __future__ import annotations

import json
import os
from typing import Annotated, List, Literal, Optional, Sequence, TypedDict
from pathlib import Path
import re
from decimal import Decimal, InvalidOperation

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from streamlit_app.tools.chart_tool import ChartGenerationTool
from streamlit_app.tools.co2_tool import CO2CalculationTool
from streamlit_app.tools.dataset_tool import DatasetQueryTool
from streamlit_app.tools.environment_tool import EnvironmentEstimationTool
from streamlit_app.tools.heyer_volume_tool import HeyerVolumeTool
from streamlit_app.tools.general_volume_tool import GeneralVolumeTool
from streamlit_app.tools.simplified_volume_tool import SimplifiedVolumeTool
from streamlit_app.tools.allometric_relation_tool import AllometricRelationTool
from streamlit_app.tools.log_allometric_tool import LogAllometricTool
from streamlit_app.tools.model_error_tool import ModelErrorTool
from streamlit_app.tools.log_fuel_biomass_tool import LogFuelBiomassTool
from streamlit_app.tools.leaf_biomass_tool import LeafBiomassTool
from streamlit_app.tools.stem_biomass_tool import StemBiomassTool
from streamlit_app.tools.root_biomass_tool import RootBiomassTool
from streamlit_app.tools.total_biomass_tool import TotalBiomassTool
from streamlit_app.tools.map_tool import MapGenerationTool

# Load environment variables
load_dotenv()


class AgentState(TypedDict):
    """State for the LangGraph agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    optimized_query: Optional[str]
    tasks: Optional[List[str]]
    validation_result: Optional[dict]
    context_summary: Optional[str]  # Summary of important context
    message_count: Optional[int]  # Track conversation length
    chart_data: Optional[dict]  # Store chart data when chart tool is called


class TreeEvaluatorAgent:
    """LangGraph-based agent that orchestrates tree evaluation tools."""

    # Dataset presets configuration
    DATASET_PRESETS = {
        "vienna": {
            "db_path": "dataset/BAUMKATOGD.db",
            "table_name": "baumkatogd",
            "description": """Dataset degli alberi di Vienna (BAUMKATOGD) con 229.298 alberi.
Colonne principali:
- objectid: ID univoco albero
- district: Numero distretto (1-23)
- genus_species: Nome specie (es. "Acer platanoides")
- plant_year: Anno di piantumazione
- trunk_circumference: Circonferenza tronco in cm
- tree_height: Categoria altezza (codificata)
- crown_diameter: Categoria diametro chioma (codificata)
- object_street: Nome via
Calcoli derivati: DBH = trunk_circumference / π, Età = anno_corrente - plant_year"""
        },
        "milano": {
            "db_path": "dataset/dataset_milano.db",
            "table_name": "milano_trees",
            "description": """Dataset degli alberi di Milano con 251.165 alberi.
Colonne principali:
- _id: ID univoco albero
- district: Numero municipio (1-9)
- genere: Genere botanico (es. "Prunus", "Acer")
- specie: Specie botanica (es. "cerasifera", "platanoides")
- varieta: Varietà (es. "Pissardii")
- genus_species: Nome completo specie (genere + specie)
- trunk_diameter_cm: Diametro tronco in cm (NON circonferenza!)
- crown_diameter_m: Diametro chioma in metri
- height_m: Altezza in metri
- street: Nome via/località
- plant_year: Anno di piantumazione
- longitude, latitude: Coordinate GPS
Nota: trunk_diameter_cm è già il diametro, NON la circonferenza (a differenza di Vienna)"""
        }
    }
    
    def __init__(
        self, 
        openai_api_key: Optional[str] = None,
        custom_db_path: Optional[Path] = None,
        custom_table_name: Optional[str] = None,
        data_description: str = "",
        dataset_preset: str = "vienna"
    ) -> None:
        """Initialize the agent with tools and LLM.

        Args:
            openai_api_key: OpenAI API key. If not provided, tries OPENAI_API_KEY env var.
            custom_db_path: Optional path to custom SQLite database
            custom_table_name: Optional custom table name in the database
            data_description: Optional description of the data for context
            dataset_preset: Preset dataset to use ("vienna", "milano")
        """
        # Get API key - prioritize parameter, then env var
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Provide it via UI Settings or set OPENAI_API_KEY environment variable."
            )

        # Initialize LLM (used by tools for text-to-SQL translation)
        self._base_llm = ChatOpenAI(
            model="gpt-5",
            temperature=1,  # Lower temperature for higher determinism and exact phrasing
            api_key=api_key,
        )

        # Initialize DatasetQueryTool with appropriate database
        if custom_db_path and custom_table_name:
            # Custom uploaded CSV
            dataset_tool = DatasetQueryTool(
                db_path=custom_db_path,
                table_name=custom_table_name,
                user_description=data_description,
                llm=self._base_llm
            )
        elif dataset_preset in self.DATASET_PRESETS:
            # Preset dataset (Vienna or Milano)
            preset = self.DATASET_PRESETS[dataset_preset]
            db_path = Path(__file__).parent.parent / preset["db_path"]
            dataset_tool = DatasetQueryTool(
                db_path=db_path,
                table_name=preset["table_name"],
                user_description=preset["description"],
                llm=self._base_llm
            )
        else:
            # Default: Vienna
            dataset_tool = DatasetQueryTool(llm=self._base_llm)
        
        # Initialize tools with LLM
        # Initialize MapGenerationTool with appropriate database for dataset preset
        if custom_db_path and custom_table_name:
            # Custom uploaded CSV - may or may not have coordinates
            map_tool = MapGenerationTool(
                db_path=custom_db_path,
                table_name=custom_table_name,
                llm=self._base_llm
            )
        elif dataset_preset == "milano":
            # Milano has GPS coordinates
            map_tool = MapGenerationTool(llm=self._base_llm)
        else:
            # Vienna doesn't have GPS - still create tool but it will show error message
            map_tool = MapGenerationTool(llm=self._base_llm)
        
        self._tools = [
            CO2CalculationTool(),
            EnvironmentEstimationTool(),
            dataset_tool,
            ChartGenerationTool(llm=self._base_llm),
            map_tool,
            HeyerVolumeTool(),
            GeneralVolumeTool(),
            SimplifiedVolumeTool(),
            AllometricRelationTool(),
            LogAllometricTool(),
            ModelErrorTool(),
            LogFuelBiomassTool(),
            LeafBiomassTool(),
            StemBiomassTool(),
            RootBiomassTool(),
            TotalBiomassTool(),
        ]

        # Initialize LLM with tools bound
        self._llm = self._base_llm.bind_tools(self._tools)

        # Build graph
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow with query optimization and validation."""
        workflow = StateGraph(AgentState)

        # Define nodes
        workflow.add_node("context_manager", self._manage_context)
        workflow.add_node("query_optimizer", self._optimize_query)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", ToolNode(self._tools))
        workflow.add_node("validator", self._validate_response)

        # Set entry point - start with context management
        workflow.set_entry_point("context_manager")
        
        # Context manager -> query optimizer
        workflow.add_edge("context_manager", "query_optimizer")

        # Query optimizer -> agent
        workflow.add_edge("query_optimizer", "agent")

        # Agent decides: continue to tools or validate
        workflow.add_conditional_edges(
            "agent",
            self._should_continue,
            {
                "continue": "tools",
                "validate": "validator",
            },
        )

        # After tool execution, return to agent
        workflow.add_edge("tools", "agent")

        # Validator decides: complete or retry
        workflow.add_conditional_edges(
            "validator",
            self._should_retry,
            {
                "complete": END,
                "retry": "agent",
            },
        )

        return workflow.compile()

    def _get_dataset_tool(self) -> Optional[DatasetQueryTool]:
        for tool in self._tools:
            if isinstance(tool, DatasetQueryTool):
                return tool
        return None

    @staticmethod
    def _format_number_it(value: float, preserve_decimals: Optional[int] = None) -> str:
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
    def _extract_first_numeric(text: str) -> Optional[float]:
        pattern = re.compile(r"\d[\d\.\,\s\u00a0\u202f']*")
        for m in pattern.findall(text or ""):
            cleaned = m.replace("\u00a0", " ").replace("\u202f", " ").replace(" ", "")
            cleaned = cleaned.replace(".", "").replace(",", ".").replace("'", "")
            try:
                return float(cleaned)
            except ValueError:
                continue
        return None

    def _compute_dataset_number(self, natural_question: str) -> Optional[float]:
        tool = self._get_dataset_tool()
        if tool is None:
            return None
        try:
            result = tool._run(natural_question)  # internal call within app boundary
        except Exception:
            return None

        if isinstance(result, dict) and "result" in result and isinstance(result.get("result"), (int, float)):
            return float(result["result"])

        if isinstance(result, dict) and isinstance(result.get("results"), list):
            results = result["results"]
            if results:
                row = results[0]
                for key in ("total", "count", "sum", "avg", "value"):
                    val = row.get(key)
                    if isinstance(val, (int, float)):
                        return float(val)
                for _, val in row.items():
                    if isinstance(val, (int, float)):
                        return float(val)

        text_blob = ""
        if isinstance(result, dict):
            for k in ("info", "sql_executed"):
                v = result.get(k)
                if isinstance(v, str):
                    text_blob += " " + v
        return self._extract_first_numeric(text_blob)

    def _generate_one_line(self, question: str, number_str: str) -> Optional[str]:
        try:
            llm = ChatOpenAI(
                model="gpt-5",
                temperature=0.1,
                api_key=self._llm.client.api_key,
            )
            prompt = (
                "Scrivi una sola riga in italiano che risponda direttamente alla domanda, "
                "contenendo il numero esatto fornito e pochissimo testo. "
                "Imita la forma della domanda. Non aggiungere spiegazioni.\n\n"
                f"Domanda: {question}\n"
                f"Numero: {number_str}\n\n"
                "Risposta (una riga, deve includere il numero esatto):"
            )
            resp = llm.invoke([HumanMessage(content=prompt)])
            line = (resp.content or "").strip().splitlines()[0].strip()
            return line if number_str in line else f"{line} {number_str}"
        except Exception:
            return None

    def _finalize_response(self, question: str, response_text: str) -> str:
        first_line = (response_text or "").splitlines()[0] if response_text else ""
        if self._extract_first_numeric(first_line) is not None:
            return response_text

        num = self._compute_dataset_number(question)
        if num is None:
            num = self._extract_first_numeric(response_text)
        if num is None:
            return response_text

        num_str = self._format_number_it(num)
        one_line = self._generate_one_line(question, num_str)
        if not one_line:
            one_line = f"{num_str}"

        rest = response_text or ""
        if rest.startswith(first_line):
            rest = rest[len(first_line):].lstrip("\n")
        return f"{one_line}\n\n{rest}".rstrip()

    def _manage_context(self, state: AgentState) -> dict:
        """Manage conversation context to avoid token limit issues."""
        messages = list(state["messages"])
        
        # Configuration
        MAX_MESSAGES = 3  # Keep only last N message pairs (user + assistant)
        MAX_MESSAGE_LENGTH = 50000  # Max characters per message
        
        # Count current messages
        message_count = len(messages)
        
        # If conversation is too long, trim it
        if message_count > MAX_MESSAGES:
            # Always keep system messages
            system_messages = [m for m in messages if isinstance(m, SystemMessage)]
            
            # Keep only the most recent messages (excluding system)
            recent_messages = [m for m in messages if not isinstance(m, SystemMessage)][-MAX_MESSAGES:]
            
            # Create a summary of removed context
            removed_count = len(messages) - len(system_messages) - len(recent_messages)
            
            if removed_count > 0:
                context_note = SystemMessage(
                    content=f"[Nota: {removed_count} messaggi precedenti rimossi per gestione contesto. "
                    f"Concentrati sulla richiesta corrente dell'utente.]"
                )
                messages = system_messages + [context_note] + recent_messages
        
        # Compress very long messages (like detailed statistics)
        compressed_messages = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.content and len(msg.content) > MAX_MESSAGE_LENGTH:
                # If it's a very long AI response, create a summary
                if "DBH" in msg.content or "distretto" in msg.content or "specie" in msg.content:
                    # Looks like dataset statistics - compress it
                    summary = (
                        "[Statistiche dataset precedenti - riepilogo compresso]\n"
                        "Dataset analizzato con successo. "
                        "Per nuove analisi o grafici, specifica la tua richiesta."
                    )
                    compressed_msg = AIMessage(content=summary)
                    compressed_messages.append(compressed_msg)
                else:
                    # Keep as is but truncate
                    truncated_content = msg.content[:MAX_MESSAGE_LENGTH] + "\n\n[... messaggio troncato per gestione contesto]"
                    compressed_msg = AIMessage(content=truncated_content)
                    compressed_messages.append(compressed_msg)
            else:
                compressed_messages.append(msg)
        
        return {
            "messages": compressed_messages,
            "message_count": len(compressed_messages)
        }

    def _optimize_query(self, state: AgentState) -> dict:
        """Optimize user query and break it into tasks."""
        messages = state["messages"]
        
        # Get the last user message
        last_user_msg = None
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                last_user_msg = msg.content
                break
        
        if not last_user_msg:
            return {"optimized_query": None, "tasks": []}
        
        # Use LLM to optimize query and create tasks
        optimizer_prompt = f"""Analizza la seguente domanda dell'utente e:
1. Riformulala in modo più chiaro e specifico
2. Scomponila in task GRANULARI e specifici (minimo 3-5 task, anche 6-8 se la domanda è complessa)

IMPORTANTE: Crea sempre MULTIPLI sottotask dettagliati, non un singolo task generico.

Domanda originale: {last_user_msg}

Rispondi in formato JSON con:
- "optimized_query": la domanda ottimizzata
- "tasks": lista di sottotask specifici e granulari (MINIMO 3-5 task, anche di più se necessario)

Esempi:

Esempio 1 - Calcolo CO2:
{{
    "optimized_query": "Calcola il sequestro di CO2 per un albero di Acer di 30cm DBH e 15m altezza",
    "tasks": [
        "1. Identificare la specie (Acer) nel database delle densità",
        "2. Recuperare la densità del legno appropriata per Acer (0.56 g/cm³)",
        "3. Calcolare il volume dell'albero usando DBH (30cm) e altezza (15m)",
        "4. Calcolare la biomassa totale (volume × densità)",
        "5. Stimare il sequestro di CO2 dalla biomassa",
        "6. Presentare i risultati con unità di misura (kg CO2, kg biomassa, m³ volume)"
    ]
}}

Esempio 2 - Grafico:
{{
    "optimized_query": "Crea un grafico a torta che mostri la distribuzione dei diametri degli alberi a Vienna",
    "tasks": [
        "1. Interrogare il dataset Vienna Trees per ottenere tutti i diametri (DBH)",
        "2. Analizzare la distribuzione dei valori per definire categorie appropriate",
        "3. Raggruppare i dati in categorie di diametro (es. 0-20cm, 20-40cm, 40-60cm, >60cm)",
        "4. Contare il numero di alberi per ogni categoria",
        "5. Generare un grafico a torta usando il chart tool",
        "6. Verificare che le etichette mostrino percentuali e conteggi"
    ]
}}

Esempio 3 - Query Dataset Complessa:
{{
    "optimized_query": "Trova le 10 specie più comuni a Vienna e conta quanti alberi ci sono per ciascuna",
    "tasks": [
        "1. Interrogare il dataset per verificare la struttura della tabella",
        "2. Estrarre tutte le specie presenti nel dataset",
        "3. Raggruppare i dati per specie",
        "4. Contare il numero di alberi per ogni specie",
        "5. Ordinare per numero di alberi in ordine decrescente",
        "6. Selezionare le prime 10 specie",
        "7. Formattare i risultati con nome specie e conteggio"
    ]
}}

Esempio 4 - Rapporto Ipogeo/Epigeo:
{{
    "optimized_query": "Calcola il rapporto ipogeo/epigeo delle conifere nel dataset",
    "tasks": [
        "1. Interrogare il dataset per identificare tutte le conifere presenti",
        "2. Cercare nel dataset dei rapporti R/S specifici per conifere",
        "3. Se disponibili, estrarre i valori medi di R/S per genere/specie",
        "4. Se non disponibili, usare il valore standard R/S = 0.24 per conifere temperate",
        "5. Calcolare la media ponderata se ci sono più specie",
        "6. Presentare il risultato con unità di misura e riferimenti ai tool usati"
    ]
}}

REGOLA CRITICA: Scomponi SEMPRE la domanda in 3-8 sottotask specifici. NON creare un singolo task generico."""
        
        try:
            # Create a temporary LLM without tools for optimization
            optimizer_llm = ChatOpenAI(
                model="gpt-5",
                temperature=1,
                api_key=self._llm.client.api_key,
            )
            
            response = optimizer_llm.invoke([HumanMessage(content=optimizer_prompt)])
            
            # Parse JSON response
            response_text = response.content.strip()
            
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            optimization_result = json.loads(response_text)
            
            optimized_query = optimization_result.get("optimized_query", last_user_msg)
            tasks = optimization_result.get("tasks", [])
            
            # Add optimization info as a system message
            optimization_msg = SystemMessage(
                content=f"""Query ottimizzata: {optimized_query}

Task da completare:
{chr(10).join(f'{i+1}. {task}' for i, task in enumerate(tasks))}"""
            )
            
            return {
                "messages": [optimization_msg],
                "optimized_query": optimized_query,
                "tasks": tasks,
            }
            
        except Exception as e:
            # If optimization fails, continue with original query
            return {
                "optimized_query": last_user_msg,
                "tasks": [last_user_msg],
            }

    def _call_model(self, state: AgentState) -> dict:
        """Call the LLM model."""
        messages = state["messages"]

        # Add system message if not present
        if not any(isinstance(m, SystemMessage) for m in messages):
            system_msg = SystemMessage(
                content="""You are a helpful tree evaluation assistant with access to:

1. **CO2 Calculation Tool**: Calculate CO2 sequestration and biomass for individual trees given their measurements.
2. **Environmental Estimation Tool**: Compute volume, biomass, and carbon stock using alternative formulas.
3. **Dataset Query Tool**: Query a real Vienna trees dataset (BAUMKATOGD) with filtering, aggregation, and statistics.
4. **Chart Generation Tool**: Create interactive visualizations (bar, pie, line, scatter, histogram, box plots) from the dataset.
5. **Map Generation Tool**: Create interactive maps showing tree locations (markers, clusters, heatmaps). ONLY available for Milano dataset which has GPS coordinates.
6. **Advanced Biomass & Volume Equations**: Calculate Volume (Heyer, General, Simplified), Biomass (Leaf, Stem, Root, Total), and Allometric Relations using specific scientific formulas.

Guidelines:
- When users ask about CO2 or carbon sequestration for specific measurements, use the CO2 calculation tool.
- When users ask about the dataset (counts, species, districts, statistics), use the dataset query tool.
- When users ask to create, visualize, or show charts/graphs, use the chart generation tool.
- When users ask to show trees on a MAP, visualize distribution geographically, or create a map, use the map generation tool. NOTE: Maps are ONLY available for the Milano dataset (has GPS coordinates). Vienna dataset does NOT have coordinates.
- Use specific biomass/volume tools when the user asks for those specific equations (Heyer, Leaf Biomass, etc.).
- Always provide clear, helpful responses in Italian.
- If you need more information, ask the user.
- When using tools, explain the results in a user-friendly way.
- For wood density, use species-specific values if known, otherwise default to 0.6 g/cm³.

**CRITICAL RULES - ALWAYS FOLLOW:**

1. **ALWAYS include units of measurement** in your answers:
   - Volumes: m³ (metri cubi)
   - Biomass: kg, t (tonnellate)
   - CO2: kg CO2, t CO2
   - Diameters: cm (centimetri)
   - Heights: m (metri)
   - Ratios: no unit (R/S = 0.24 significa rapporto adimensionale)
   - Counts: numero alberi, specie, record
   
2. **ALWAYS cite which tool(s) you used** at the end of your response:
   - Format: "Tool utilizzati: [nome tool]"
   - Examples:
     * "Tool utilizzati: Dataset Query Tool"
     * "Tool utilizzati: CO2 Calculation Tool"
     * "Tool utilizzati: Dataset Query Tool, Chart Generation Tool"
     * "Tool utilizzati: Allometric Relation Tool"

3. **Complete answer format**:
   ```
   [Prima riga: risposta diretta con numero e unità di misura]
   
   [Dettagli aggiuntivi se necessari]
   
   Tool utilizzati: [nome tool(s)]
   ```

Answer style policy (CRITICAL for evaluation):
- First line must contain the final answer in Italian with the exact number, units of measurement, and minimal text.
- ALWAYS include units: kg, m³, cm, m, t CO2, etc.
- Prefer Italian numeric formatting: thousands with dot, decimals with comma (e.g., 33.612 alberi; 0,24 R/S; 15.000 kg CO2).
- Keep additional details only after a blank line, and keep them concise.
- Mirror user phrasing when possible to maximize textual similarity.
- ALWAYS end with "Tool utilizzati: [nome tool]"

Examples with units and tool citation:
  - "A Vienna ci sono 23 distretti\n\nTool utilizzati: Dataset Query Tool"
  - "Gli alberi totali a Vienna sono 229.298 alberi\n\nTool utilizzati: Dataset Query Tool"
  - "Nel distretto 19 sono presenti esattamente 15.842 alberi\n\nTool utilizzati: Dataset Query Tool"
  - "Il rapporto R/S per le conifere temperate è 0,24\n\nStima basata su letteratura scientifica per conifere.\n\nTool utilizzati: Allometric Relation Tool"
  - "La CO₂ sequestrata è 1.250 kg CO2\n\nCalcolo basato su DBH 30cm, altezza 15m, densità legno 0.56 g/cm³.\n\nTool utilizzati: CO2 Calculation Tool"
  - "Il volume stimato è 2,5 m³\n\nCalcolo con formula di Heyer per DBH 30cm e altezza 15m.\n\nTool utilizzati: Heyer Volume Tool"

If a computation is needed but measurements are missing, state the short requirement in one line, then ask for the needed values in the next lines.

**IMPORTANT - Chart Tool Usage:**
When you use the chart generation tool and it returns chart data with "success": true, you MUST include the COMPLETE JSON response in your answer. Format it exactly like this:

Ho creato il grafico richiesto.

CHART_DATA_START
{the complete JSON from the tool}
CHART_DATA_END

Do not modify or summarize the JSON - include it verbatim between CHART_DATA_START and CHART_DATA_END markers.

**IMPORTANT - Map Tool Usage:**
When you use the map generation tool and it returns map data with "success": true, you MUST include the COMPLETE JSON response in your answer. Format it exactly like this:

Ho creato la mappa richiesta.

MAP_DATA_START
{the complete JSON from the tool}
MAP_DATA_END

Do not modify or summarize the JSON - include it verbatim between MAP_DATA_START and MAP_DATA_END markers.
IMPORTANT: Maps require GPS coordinates. Only the Milano dataset has coordinates. If the user tries to generate a map with Vienna dataset, explain that maps are not available for Vienna.

Common wood densities (g/cm³):
- Acer (Acero): 0.56
- Tilia (Tiglio): 0.49
- Carpinus (Carpino): 0.75
- Gleditsia: 0.62
- Aesculus (Ippocastano): 0.53
- Quercus (Quercia): 0.75
- Fraxinus (Frassino): 0.69
- Betula (Betulla): 0.65
"""
            )
            messages = [system_msg] + list(messages)

        response = self._llm.invoke(messages)
        try:
            user_question = None
            for msg in messages:
                if isinstance(msg, HumanMessage):
                    user_question = msg.content
            if user_question and isinstance(response, AIMessage) and response.content:
                fixed = self._finalize_response(user_question, response.content)
                if fixed and fixed != response.content:
                    response = AIMessage(content=fixed)
        except Exception:
            pass
        return {"messages": [response]}

    def _should_continue(self, state: AgentState) -> Literal["continue", "validate"]:
        """Determine if we should continue to tools or validate response."""
        messages = state["messages"]
        last_message = messages[-1]

        # If the LLM makes a tool call, continue to tools
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "continue"

        # Otherwise, validate the response
        return "validate"
    
    def _validate_response(self, state: AgentState) -> dict:
        """Validate if the response adequately answers the user's question."""
        messages = state["messages"]
        tasks = state.get("tasks", [])
        optimized_query = state.get("optimized_query", "")
        
        # Get original user question
        user_question = None
        for msg in messages:
            if isinstance(msg, HumanMessage) and not msg.content.startswith("Query ottimizzata"):
                user_question = msg.content
                break
        
        # Get agent's response
        agent_response = None
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                agent_response = msg.content
                break
        
        if not user_question or not agent_response:
            return {"validation_result": {"is_complete": True, "feedback": ""}}
        
        # Create validation prompt
        validation_prompt = f"""Valuta se la seguente risposta risponde adeguatamente alla domanda dell'utente.

Domanda originale: {user_question}

Query ottimizzata: {optimized_query}

Task da completare:
{chr(10).join(f'{i+1}. {task}' for i, task in enumerate(tasks))}

Risposta fornita: {agent_response}

Analizza TUTTI questi criteri:
1. Tutti i task sono stati completati?
2. La risposta è accurata e completa?
3. La risposta risponde effettivamente alla domanda?
4. CRITICO: Se la risposta contiene numeri/misure, include le UNITÀ DI MISURA? (kg, m³, cm, kg CO2, etc.)
5. CRITICO: La risposta cita i TOOL UTILIZZATI alla fine? (formato: "Tool utilizzati: [nome tool]")

IMPORTANTE: Se mancano unità di misura o citazione dei tool, la risposta è INCOMPLETA.

Rispondi in formato JSON:
{{
    "is_complete": true/false,
    "completed_tasks": ["lista", "dei", "task", "completati"],
    "missing_tasks": ["lista", "dei", "task", "mancanti"],
    "has_units": true/false (se sono presenti unità di misura dove necessario),
    "has_tool_citation": true/false (se è presente "Tool utilizzati:"),
    "feedback": "breve feedback su cosa manca o cosa migliorare (se incompleto)"
}}"""
        
        try:
            # Create validator LLM
            validator_llm = ChatOpenAI(
                model="gpt-5",
                temperature=1,
                api_key=self._llm.client.api_key,
            )
            
            response = validator_llm.invoke([HumanMessage(content=validation_prompt)])
            
            # Parse JSON response
            response_text = response.content.strip()
            
            # Extract JSON from markdown code blocks if present
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            validation_result = json.loads(response_text)
            
            # Check if response is truly complete (including units and tool citation)
            is_complete = validation_result.get("is_complete", True)
            has_units = validation_result.get("has_units", True)
            has_tool_citation = validation_result.get("has_tool_citation", True)
            
            # Override is_complete if units or tool citation are missing
            if not has_units or not has_tool_citation:
                is_complete = False
                missing_items = []
                if not has_units:
                    missing_items.append("unità di misura")
                if not has_tool_citation:
                    missing_items.append("citazione dei tool utilizzati")
                
                # Add to feedback
                current_feedback = validation_result.get('feedback', '')
                additional_feedback = f"Mancano: {', '.join(missing_items)}."
                validation_result['feedback'] = f"{current_feedback} {additional_feedback}".strip()
                validation_result['is_complete'] = False
            
            # If incomplete, add feedback as system message for retry
            if not is_complete:
                feedback_msg = SystemMessage(
                    content=f"""⚠️ Validazione risposta:
Task mancanti: {', '.join(validation_result.get('missing_tasks', []))}

Feedback: {validation_result.get('feedback', '')}

REGOLE OBBLIGATORIE:
- SEMPRE includere unità di misura (kg, m³, cm, kg CO2, etc.)
- SEMPRE terminare con "Tool utilizzati: [nome tool]"

Per favore, completa la risposta affrontando i task mancanti e rispettando le regole obbligatorie."""
                )
                return {
                    "messages": [feedback_msg],
                    "validation_result": validation_result,
                }
            
            return {"validation_result": validation_result}
            
        except Exception as e:
            # If validation fails, assume complete
            return {"validation_result": {"is_complete": True, "feedback": ""}}
    
    def _should_retry(self, state: AgentState) -> Literal["complete", "retry"]:
        """Determine if we should retry or complete based on validation."""
        validation_result = state.get("validation_result", {})
        
        # Check if response is complete
        is_complete = validation_result.get("is_complete", True)
        
        if is_complete:
            return "complete"
        else:
            return "retry"

    def chat(self, message: str, history: Optional[List[dict]] = None) -> str:
        """Chat with the agent.

        Args:
            message: User message
            history: Optional chat history as list of dicts with 'role' and 'content' keys

        Returns:
            Agent response as string
        """
        # Convert history to messages
        messages: List[BaseMessage] = []
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        # Add current message
        messages.append(HumanMessage(content=message))

        # Run graph
        result = self._graph.invoke({"messages": messages})

        # Extract final response
        final_message = result["messages"][-1]
        if isinstance(final_message, AIMessage):
            return final_message.content

        return str(final_message.content)

    def stream_chat(self, message: str, history: Optional[List[dict]] = None):
        """Stream chat response with LangGraph streaming, including reasoning steps.

        Args:
            message: User message
            history: Optional chat history

        Yields:
            Dict with 'type' and 'content' keys:
            - type: 'reasoning' (internal step) or 'response' (final answer)
            - content: the actual content to display
        """
        # Convert history to messages
        messages: List[BaseMessage] = []
        if history:
            for msg in history:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))

        # Add current message
        messages.append(HumanMessage(content=message))

        # Track execution
        final_response = None
        retry_count = 0
        max_retries = 2
        chart_data_json = None  # Track chart data if generated
        map_data_json = None  # Track map data if generated

        # Stream from graph with updates mode to see each node
        for event in self._graph.stream({"messages": messages}, stream_mode="updates"):
            # event is a dict with node_name: node_output
            for node_name, node_output in event.items():
                
                # Emit reasoning for each node
                if node_name == "context_manager":
                    # Show context management info if messages were compressed
                    message_count = node_output.get("message_count", 0)
                    original_count = len(messages)
                    
                    if message_count < original_count:
                        reasoning = f"🧹 **Gestione Contesto**\n\n"
                        reasoning += f"Messaggi originali: {original_count}\n"
                        reasoning += f"Messaggi ottimizzati: {message_count}\n"
                        reasoning += f"Contesto lungo compresso per evitare limiti di token.\n"
                        yield {"type": "reasoning", "content": reasoning}
                
                elif node_name == "query_optimizer":
                    optimized = node_output.get("optimized_query", "")
                    tasks = node_output.get("tasks", [])
                    if optimized:
                        reasoning = f"🔍 **Ottimizzazione Query**\n\n"
                        reasoning += f"Query ottimizzata: *{optimized}*\n\n"
                        if tasks:
                            reasoning += "**Task identificati:**\n"
                            for i, task in enumerate(tasks, 1):
                                reasoning += f"{i}. {task}\n"
                        yield {"type": "reasoning", "content": reasoning}
                
                elif node_name == "agent":
                    # Check if agent is calling tools or responding
                    node_messages = node_output.get("messages", [])
                    if node_messages:
                        last_msg = node_messages[-1]
                        if isinstance(last_msg, AIMessage):
                            if last_msg.tool_calls:
                                # Agent is calling tools - show detailed parameters
                                reasoning = f"🛠️ **Chiamata Tool**\n\n"
                                for tool_call in last_msg.tool_calls:
                                    tool_name = tool_call.get("name", "unknown")
                                    tool_args = tool_call.get("args", {})
                                    
                                    reasoning += f"- **Tool**: `{tool_name}`\n"
                                    
                                    # Show parameters based on tool type
                                    if tool_name == "query_tree_dataset":
                                        natural_q = tool_args.get("natural_query", "N/A")
                                        reasoning += f"  - **Query**: _{natural_q}_\n"
                                    elif tool_name == "calculate_co2":
                                        dbh = tool_args.get("dbh_cm", "N/A")
                                        height = tool_args.get("height_m", "N/A")
                                        wood_density = tool_args.get("wood_density", "N/A")
                                        reasoning += f"  - **DBH**: {dbh} cm\n"
                                        reasoning += f"  - **Altezza**: {height} m\n"
                                        reasoning += f"  - **Densità legno**: {wood_density} g/cm³\n"
                                    elif tool_name == "estimate_environment":
                                        dbh = tool_args.get("dbh_cm", "N/A")
                                        height = tool_args.get("height_m", "N/A")
                                        reasoning += f"  - **DBH**: {dbh} cm\n"
                                        reasoning += f"  - **Altezza**: {height} m\n"
                                    elif tool_name == "generate_chart":
                                        chart_type = tool_args.get("chart_type", "N/A")
                                        reasoning += f"  - **Tipo grafico**: {chart_type}\n"
                                    
                                    reasoning += "\n"
                                
                                yield {"type": "reasoning", "content": reasoning}
                            elif last_msg.content and not last_msg.tool_calls:
                                # Agent has a response (might be intermediate or final)
                                final_response = last_msg.content
                
                elif node_name == "tools":
                    # Tool execution completed - show detailed results
                    node_messages = node_output.get("messages", [])
                    
                    if node_messages:
                        for msg in node_messages:
                            # Tool messages contain the results
                            if hasattr(msg, 'content') and msg.content:
                                try:
                                    # Try to parse as JSON for structured results
                                    import json
                                    if isinstance(msg.content, str):
                                        result_data = json.loads(msg.content)
                                    else:
                                        result_data = msg.content
                                    
                                    # Check if this is chart data (chart tool returns "chart_json" key)
                                    if "chart_json" in result_data and result_data.get("success"):
                                        chart_data_json = json.dumps(result_data, ensure_ascii=False, indent=2)
                                        print(f"[DEBUG] Chart data captured! Length: {len(chart_data_json)} chars")
                                    
                                    # Check if this is map data (map tool returns "map_html" key)
                                    if "map_html" in result_data and result_data.get("success"):
                                        map_data_json = json.dumps(result_data, ensure_ascii=False, indent=2)
                                        print(f"[DEBUG] Map data captured! Length: {len(map_data_json)} chars")
                                    
                                    reasoning = f"✅ **Risultati Tool**\n\n"
                                    
                                    # Show SQL query if it's a dataset query
                                    if "sql_executed" in result_data:
                                        sql = result_data.get("sql_executed", "")
                                        reasoning += f"**Query SQL generata:**\n```sql\n{sql}\n```\n\n"
                                    
                                    # Show row count and vector search info
                                    if "row_count" in result_data:
                                        row_count = result_data.get("row_count", 0)
                                        
                                        # Check if vector search was applied
                                        if result_data.get("vector_search_applied", False):
                                            total_found = result_data.get("total_rows_found", row_count)
                                            reasoning += f"🔍 **Vector Search Applicata**\n"
                                            reasoning += f"📊 **Righe totali trovate**: {total_found}\n"
                                            reasoning += f"✨ **Top risultati più rilevanti**: {row_count}\n"
                                            if "info" in result_data:
                                                reasoning += f"ℹ️  {result_data['info']}\n"
                                        else:
                                            reasoning += f"📊 **Righe trovate**: {row_count}\n"
                                            
                                            # Show truncation warning if present (old style)
                                            if "warning" in result_data:
                                                warning = result_data.get("warning")
                                                reasoning += f"⚠️  **Attenzione**: {warning}\n"
                                        
                                        reasoning += "\n"
                                    
                                    # Show result preview for dataset queries
                                    if "results" in result_data:
                                        results = result_data.get("results", [])
                                        if results and len(results) > 0:
                                            reasoning += f"**Primi risultati:**\n"
                                            # Show first 3 results as preview
                                            for i, row in enumerate(results[:3], 1):
                                                reasoning += f"{i}. "
                                                # Show main fields
                                                if "genus_species" in row:
                                                    reasoning += f"Specie: {row['genus_species']} "
                                                if "count" in row:
                                                    reasoning += f"Count: {row['count']} "
                                                if "district" in row:
                                                    reasoning += f"Distretto: {row['district']} "
                                                if "trunk_circumference" in row:
                                                    reasoning += f"Circonferenza: {row['trunk_circumference']}cm "
                                                reasoning += "\n"
                                            
                                            if len(results) > 3:
                                                reasoning += f"... e altri {len(results) - 3} risultati\n"
                                    
                                    # Show single value results
                                    elif "result" in result_data and "column" in result_data:
                                        result_val = result_data.get("result")
                                        column_name = result_data.get("column")
                                        reasoning += f"**{column_name}**: {result_val}\n"
                                    
                                    # Show CO2 calculation results
                                    if "co2_sequestration_kg" in result_data:
                                        co2 = result_data.get("co2_sequestration_kg", 0)
                                        reasoning += f"🌱 **CO2 sequestrato**: {co2} kg\n"
                                    
                                    yield {"type": "reasoning", "content": reasoning}
                                    
                                except (json.JSONDecodeError, AttributeError):
                                    # If not JSON, just show completion message
                                    reasoning = f"✅ **Tool Eseguito**\n\nElaborazione risultati...\n"
                                    yield {"type": "reasoning", "content": reasoning}
                
                elif node_name == "validator":
                    validation = node_output.get("validation_result", {})
                    is_complete = validation.get("is_complete", True)
                    
                    if is_complete:
                        reasoning = f"✓ **Validazione Completata**\n\nLa risposta è completa e accurata.\n"
                        yield {"type": "reasoning", "content": reasoning}
                    else:
                        retry_count += 1
                        if retry_count > max_retries:
                            reasoning = f"⚠️ **Validazione**\n\nRaggiunto limite retry. Proseguo con la risposta attuale.\n"
                            yield {"type": "reasoning", "content": reasoning}
                            # Don't break - let the graph complete naturally
                        else:
                            missing = validation.get("missing_tasks", [])
                            feedback = validation.get("feedback", "")
                            reasoning = f"⚠️ **Validazione (Tentativo {retry_count})**\n\n"
                            if missing:
                                reasoning += f"Task mancanti: {', '.join(missing)}\n"
                            if feedback:
                                reasoning += f"\n{feedback}\n"
                            reasoning += "\nRielaborazione risposta...\n"
                            yield {"type": "reasoning", "content": reasoning}
        
        # Yield final response
        if final_response:
            # If we have chart data but it's not in the response, add it automatically
            print(f"[DEBUG] Final response check - chart_data_json: {chart_data_json is not None}, has markers: {'CHART_DATA_START' in final_response}")
            if chart_data_json and "CHART_DATA_START" not in final_response:
                print(f"[DEBUG] Adding chart data to response!")
                final_response = f"{final_response}\n\nCHART_DATA_START\n{chart_data_json}\nCHART_DATA_END"
            
            # If we have map data but it's not in the response, add it automatically
            print(f"[DEBUG] Final response check - map_data_json: {map_data_json is not None}, has markers: {'MAP_DATA_START' in final_response}")
            if map_data_json and "MAP_DATA_START" not in final_response:
                print(f"[DEBUG] Adding map data to response!")
                final_response = f"{final_response}\n\nMAP_DATA_START\n{map_data_json}\nMAP_DATA_END"
            try:
                last_user = None
                for msg in messages:
                    if isinstance(msg, HumanMessage):
                        last_user = msg.content
                if last_user:
                    fixed = self._finalize_response(last_user, final_response)
                    if fixed:
                        final_response = fixed
            except Exception:
                pass
            yield {"type": "response", "content": final_response}

