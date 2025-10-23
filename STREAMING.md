# Streaming delle Risposte con LangGraph

## ✨ Implementazione Streaming Completa

Il chatbot Streamlit ora è completamente integrato con **LangGraph streaming**, permettendo di vedere le risposte dell'agent generarsi in tempo reale, parola per parola.

## 🎯 Come Funziona

### Architettura Streaming

```
User Input
    ↓
ChatUI.render()
    ↓
ChatService.stream_reply() ← genera chunks
    ↓
TreeEvaluatorAgent.stream_chat() ← LangGraph streaming
    ↓
LangGraph.stream(stream_mode="values")
    ↓
[chunks in real-time] → Streamlit message_placeholder
    ↓
SQLite persistence (messaggio completo)
```

### Flusso Dettagliato

1. **User invia messaggio**

   - UI aggiunge immediatamente messaggio utente al database
   - Mostra messaggio utente nella chat

2. **Streaming inizia**

   - `ChatService.stream_reply()` chiama `agent.stream_chat()`
   - LangGraph esegue workflow: agent → tools → agent
   - Ogni step emette un evento con lo stato corrente

3. **Aggiornamento UI real-time**

   - Streamlit placeholder aggiorna il testo ad ogni chunk
   - Cursore `▌` mostra che sta scrivendo
   - Quando completo, rimuove cursore

4. **Persistenza**
   - Messaggio completo salvato su SQLite
   - Aggiunto a `session_state.messages`

## 🔧 Componenti Modificati

### 1. `agent.py` - LangGraph Streaming

```python
def stream_chat(self, message: str, history: Optional[List[dict]] = None):
    """Stream chat response with LangGraph streaming."""
    # Prepara messaggi
    messages = [...]  # convert history
    messages.append(HumanMessage(content=message))

    # Stream da LangGraph
    for event in self._graph.stream({"messages": messages}, stream_mode="values"):
        last_message = event["messages"][-1]

        # Yield solo risposte finali (non tool calls)
        if isinstance(last_message, AIMessage) and last_message.content:
            if not last_message.tool_calls:
                yield last_message.content
```

**Punti chiave:**

- `stream_mode="values"` → emette stato completo ad ogni step
- Filtra tool calls (intermedi) vs risposte finali
- Yield solo contenuto testuale dell'assistant

### 2. `service.py` - Stream Reply

```python
def stream_reply(self, user_id: str, conversation_id: int,
                 last_user_message: str, openai_api_key: Optional[str] = None):
    """Stream reply from agent with real-time updates."""
    agent = self._get_or_create_agent(openai_api_key=openai_api_key)

    if agent:
        full_response = ""
        for chunk in agent.stream_chat(last_user_message, history):
            full_response = chunk  # Ogni chunk è lo stato completo finora
            yield chunk

        # Salva messaggio completo
        self._repository.add_message(ChatMessage(..., content=full_response))
```

**Punti chiave:**

- Generator function (yield)
- Accumula risposta completa man mano
- Salva su DB solo alla fine
- Fallback graceful se agent non disponibile

### 3. `ui.py` - Streamlit Integration

```python
# Stream assistant response
with st.chat_message("assistant"):
    message_placeholder = st.empty()
    full_response = ""

    # Stream da service
    for chunk in self._service.stream_reply(...):
        full_response = chunk
        # Aggiorna placeholder con cursore
        message_placeholder.markdown(full_response + "▌")

    # Rimuovi cursore alla fine
    message_placeholder.markdown(full_response)
```

**Punti chiave:**

- `st.empty()` placeholder per aggiornamenti in-place
- Cursore animato `▌` durante typing
- Update sincrono con chunks

## 🎨 Effetti Visivi

### Durante Streaming

```
User: Quanti alberi ci sono nel dataset?
```
