"""Data extraction utilities for parsing tool results and conversation history."""

from __future__ import annotations

import ast
import json
import re
from typing import Dict, List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


class DataExtractor:
    """Utilities for extracting structured data from messages."""
    
    @staticmethod
    def extract_dataset_results(messages: Sequence[BaseMessage]) -> List[dict]:
        """Extract dataset query results from ToolMessages.
        
        Args:
            messages: Conversation messages to extract from
            
        Returns:
            List of result dictionaries
        """
        all_results = []
        
        for msg in reversed(list(messages)):
            if not isinstance(msg, ToolMessage):
                continue
            
            content = msg.content
            if not content:
                continue
            
            # Try to parse the content
            parsed = None
            try:
                if isinstance(content, dict):
                    parsed = content
                elif isinstance(content, str):
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError:
                        try:
                            parsed = ast.literal_eval(content)
                        except (ValueError, SyntaxError):
                            pass
            except Exception:
                pass
            
            # Extract results array
            if isinstance(parsed, dict) and "results" in parsed:
                results = parsed.get("results", [])
                if isinstance(results, list) and results:
                    all_results = results  # Take the most recent results
                    break  # Stop at first valid result set
        
        return all_results
    
    @staticmethod
    def extract_papers(messages: Sequence[BaseMessage]) -> List[dict]:
        """Extract paper results from ToolMessages for search_scientific_papers.
        
        Args:
            messages: Conversation messages to extract from
            
        Returns:
            List of paper dictionaries with title, authors, abstract, link, source
        """
        papers = []
        errors = []
        seen_titles = set()  # Avoid duplicates
        
        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            
            content = msg.content
            if not content:
                continue
            
            content_str = str(content)
            
            # Try to parse the content as JSON or Python dict
            parsed = None
            try:
                if isinstance(content, dict):
                    parsed = content
                elif isinstance(content, str):
                    # Try JSON first
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError:
                        # Try Python literal (handles single quotes)
                        try:
                            parsed = ast.literal_eval(content)
                        except (ValueError, SyntaxError):
                            pass
            except Exception:
                pass
            
            # If parsing succeeded and we have papers
            if isinstance(parsed, dict) and "papers" in parsed:
                for paper in parsed.get("papers", []):
                    if not isinstance(paper, dict):
                        continue
                    
                    # Skip error entries
                    if "error" in paper:
                        errors.append(paper.get("error", "Unknown error"))
                        continue
                    
                    title = paper.get("title", "") or paper.get("Title", "")
                    if title and title != "N/A" and title not in seen_titles:
                        seen_titles.add(title)
                        papers.append({
                            "title": title,
                            "authors": paper.get("authors", "") or paper.get("Authors", ""),
                            "abstract": paper.get("abstract", "") or paper.get("Abstract", ""),
                            "link": paper.get("link", "") or paper.get("Link", ""),
                            "source": paper.get("source", "arxiv"),
                        })
                
                # Also check for errors array
                if "errors" in parsed and parsed["errors"]:
                    for err in parsed["errors"]:
                        if isinstance(err, dict) and "error" in err:
                            errors.append(err["error"])
            
            # Fallback: try to extract from raw text if parsing failed
            elif "arxiv" in content_str.lower() or "title" in content_str.lower():
                # Try to find title patterns in raw text
                title_matches = re.findall(r"['\"]title['\"]:\s*['\"]([^'\"]+)['\"]", content_str, re.IGNORECASE)
                link_matches = re.findall(r"https?://arxiv\.org/abs/[\w\.]+", content_str)
                
                for i, title in enumerate(title_matches):
                    if title and title != "N/A" and title not in seen_titles:
                        seen_titles.add(title)
                        link = link_matches[i] if i < len(link_matches) else ""
                        papers.append({
                            "title": title,
                            "authors": "",
                            "abstract": "",
                            "link": link,
                            "source": "arxiv",
                        })
        
        # If we have errors but no papers, include error info
        if not papers and errors:
            return [{"title": "Errore nella ricerca", "abstract": "; ".join(set(errors[:3])), "link": "", "authors": "", "source": "error"}]
        
        return papers
    
    @staticmethod
    def extract_key_facts(messages: Sequence[BaseMessage]) -> List[str]:
        """Extract key facts from conversation for context preservation.
        
        Identifies important data points like:
        - District numbers with counts
        - Species names with counts
        - Specific values mentioned (ages, sizes, etc.)
        
        Args:
            messages: Conversation messages to extract from
            
        Returns:
            List of fact strings
        """
        facts = []
        
        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            content = msg.content or ""
            if not content:
                continue
            
            # Extract district-related facts
            district_patterns = [
                r'distretto\s+(\d+)\s+(?:ha|con|:)\s*(\d+[\d\.]*)\s*alberi',
                r'distretto\s+(\d+)\s*.*?(\d+[\d\.]+)\s*alberi',
                r'District(?:o)?\s*:?\s*(\d+).*?(?:count|conteggio|alberi)\s*:?\s*(\d+[\d\.]*)',
            ]
            for pattern in district_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if len(match) >= 2:
                        facts.append(f"Distretto {match[0]} ha {match[1]} alberi")
            
            # Extract "distretto con più alberi/piante" facts
            max_district_match = re.search(
                r'distretto\s+(?:con\s+(?:più|maggior)\s+(?:alberi|piante)|più\s+grande)\s*(?:è\s+(?:il\s+)?)?(\d+)',
                content, re.IGNORECASE
            )
            if max_district_match:
                facts.append(f"Distretto con più alberi: {max_district_match.group(1)}")
            
            # Also check for "Distretto: X" in results
            result_district = re.search(r'Distretto:\s*(\d+)', content)
            if result_district:
                # Try to find associated count
                count_match = re.search(r'Count:\s*(\d+[\d\.]*)', content)
                if count_match:
                    facts.append(f"Distretto {result_district.group(1)} ha {count_match.group(1)} alberi")
            
            # Extract species counts
            species_patterns = [
                r'([A-Z][a-z]+\s+[a-z]+)\s*[:\-]?\s*(\d+[\d\.]*)\s*alberi',
                r'specie\s+più\s+comune\s*[:\-]?\s*([A-Z][a-z]+(?:\s+[a-z]+)?)',
            ]
            for pattern in species_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if isinstance(match, tuple) and len(match) >= 2:
                        facts.append(f"Specie {match[0]}: {match[1]} alberi")
                    elif isinstance(match, str):
                        facts.append(f"Specie più comune: {match}")
            
            # Extract simple numeric facts with context
            simple_facts = re.findall(
                r'(?:totale|media|massimo|minimo|count)\s*[:\-]?\s*(\d+[\d\.]*)',
                content, re.IGNORECASE
            )
            # Only add if we have associated context
            for sf in simple_facts[:2]:
                if 'totale' in content.lower():
                    facts.append(f"Totale alberi: {sf}")
        
        # Deduplicate while preserving order
        seen = set()
        unique_facts = []
        for f in facts:
            if f not in seen:
                seen.add(f)
                unique_facts.append(f)
        
        return unique_facts
    
    @staticmethod
    def extract_tool_results(messages: Sequence[BaseMessage]) -> List[Dict]:
        """Extract all tool results from messages.
        
        Args:
            messages: Conversation messages
            
        Returns:
            List of tool result dictionaries with tool name and result data
        """
        tool_results = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", "unknown")
                content = msg.content
                
                # Try to parse JSON content
                try:
                    if isinstance(content, str):
                        try:
                            parsed = json.loads(content)
                        except json.JSONDecodeError:
                            parsed = ast.literal_eval(content)
                        tool_results.append({"tool": tool_name, "result": parsed})
                    elif isinstance(content, dict):
                        tool_results.append({"tool": tool_name, "result": content})
                except Exception:
                    tool_results.append({"tool": tool_name, "result": str(content)[:500]})
        
        return tool_results

