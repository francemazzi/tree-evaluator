"""Context management for handling conversation history and preventing token limit issues."""

from __future__ import annotations

from typing import List, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.vectorstores import InMemoryVectorStore

from streamlit_app.agent.extraction import DataExtractor


class ConversationContextManager:
    """Manager for conversation context to avoid token limit issues."""
    
    # Configuration
    MAX_MESSAGES = 10  # Keep more messages for better context continuity
    MAX_MESSAGE_LENGTH = 50000  # Max characters per message
    
    def __init__(self, embeddings):
        """Initialize context manager.
        
        Args:
            embeddings: Embeddings model for vector search
        """
        self._embeddings = embeddings
        self._extractor = DataExtractor()
    
    def manage_context(self, messages: Sequence[BaseMessage]) -> dict:
        """Manage conversation context to avoid token limit issues.
        
        Preserves key facts from recent conversation to maintain context for follow-up questions.
        
        Args:
            messages: Current conversation messages
            
        Returns:
            Dict with managed messages and message count
        """
        messages = list(messages)
        
        # Count current messages
        message_count = len(messages)
        
        # Extract key facts from conversation before trimming (for context preservation)
        key_facts = self._extractor.extract_key_facts(messages)
        
        # If conversation is too long, trim it
        if message_count > self.MAX_MESSAGES:
            # Always keep system messages
            system_messages = [m for m in messages if isinstance(m, SystemMessage)]
            
            # Keep only the most recent messages (excluding system)
            recent_messages = [m for m in messages if not isinstance(m, SystemMessage)][-self.MAX_MESSAGES:]
            
            # Create a summary with key facts from removed context
            removed_count = len(messages) - len(system_messages) - len(recent_messages)
            
            if removed_count > 0:
                context_summary = f"[Nota: {removed_count} messaggi precedenti rimossi per gestione contesto."
                if key_facts:
                    context_summary += f"\n\nFatti chiave dalla conversazione precedente:\n"
                    for fact in key_facts[:5]:  # Keep top 5 facts
                        context_summary += f"- {fact}\n"
                context_summary += "\nUsa questi fatti per rispondere a domande di follow-up.]"
                
                context_note = SystemMessage(content=context_summary)
                messages = system_messages + [context_note] + recent_messages
        
        # Compress very long messages (like detailed statistics)
        compressed_messages = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.content and len(msg.content) > self.MAX_MESSAGE_LENGTH:
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
                    truncated_content = msg.content[:self.MAX_MESSAGE_LENGTH] + "\n\n[... messaggio troncato per gestione contesto]"
                    compressed_msg = AIMessage(content=truncated_content)
                    compressed_messages.append(compressed_msg)
            else:
                compressed_messages.append(msg)
        
        return {
            "messages": compressed_messages,
            "message_count": len(compressed_messages)
        }
    
    def retrieve_relevant_history(
        self,
        messages: Sequence[BaseMessage],
        query: str,
        top_k: int = 4,
        max_snippet_chars: int = 800,
        max_total_chars: int = 2200,
    ) -> List[str]:
        """Retrieve the most relevant past chat snippets using vector search.
        
        Args:
            messages: Conversation messages
            query: Current query to search for relevant context
            top_k: Number of top results to return
            max_snippet_chars: Max chars per snippet
            max_total_chars: Max total chars across all snippets
            
        Returns:
            List of relevant message snippets
        """
        # Build corpus from non-system messages, excluding the latest user query itself
        corpus = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                continue
            if isinstance(msg, HumanMessage) and msg.content == query:
                continue
            content = (msg.content or "").strip()
            if not content:
                continue
            # Truncate each message to avoid huge payloads
            if len(content) > max_snippet_chars:
                content = content[:max_snippet_chars] + "... [troncato]"
            corpus.append((content, msg))

        if not corpus:
            return []

        # Create vectorstore
        vectorstore = InMemoryVectorStore.from_texts(
            texts=[c[0] for c in corpus],
            embedding=self._embeddings,
            metadatas=[{"role": "user" if isinstance(c[1], HumanMessage) else "assistant"} for c in corpus],
        )

        # Similarity search
        k = min(top_k, len(corpus))
        results = vectorstore.similarity_search(query, k=k)

        snippets: List[str] = []
        total_chars = 0
        for doc in results:
            snippet = doc.page_content.strip()
            if not snippet:
                continue
            # Ensure we do not exceed global cap
            if total_chars + len(snippet) > max_total_chars:
                break
            snippets.append(snippet)
            total_chars += len(snippet)

        return snippets

