#!/usr/bin/env python3
"""
Test rapido per verificare che l'agent LangGraph funzioni correttamente.
Usa questo per debug senza dover avviare Streamlit.
"""

import os
import sys
from pathlib import Path

# Aggiungi root al path
root = Path(__file__).parent
sys.path.insert(0, str(root))

def test_agent():
    """Test inizializzazione e uso dell'agent."""
    print("🧪 Test Agent LangGraph")
    print("=" * 50)
    
    # 1. Check API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY non trovata")
        print("Opzioni:")
        print("  1. export OPENAI_API_KEY=sk-your-key")
        print("  2. Crea file .env con OPENAI_API_KEY=sk-your-key")
        print("  3. Passa chiave come argomento: python test_agent.py sk-your-key")
        if len(sys.argv) > 1:
            api_key = sys.argv[1]
            print(f"\n✅ Usando chiave da argomento: {api_key[:10]}...")
        else:
            sys.exit(1)
    else:
        print(f"✅ API key trovata: {api_key[:10]}...")
    
    # 2. Check dipendenze
    print("\n📦 Check dipendenze...")
    try:
        import langchain
        import langchain_openai
        import langgraph
        import pandas
        print("✅ Tutte le dipendenze installate")
    except ImportError as e:
        print(f"❌ Dipendenza mancante: {e}")
        print("Installa: pip install -r requirements.txt")
        sys.exit(1)
    
    # 3. Check dataset
    print("\n📁 Check dataset...")
    dataset_path = root / "dataset" / "BAUMKATOGD.csv"
    if dataset_path.exists():
        print(f"✅ Dataset trovato: {dataset_path}")
        # Conta righe
        with open(dataset_path) as f:
            lines = sum(1 for _ in f)
        print(f"   {lines:,} righe nel dataset")
    else:
        print(f"⚠️  Dataset non trovato: {dataset_path}")
        print("   (Opzionale per test, ma necessario per query dataset)")
    
    # 4. Inizializza agent
    print("\n🤖 Inizializzazione agent...")
    try:
        from streamlit_app.agent import TreeEvaluatorAgent
        agent = TreeEvaluatorAgent(openai_api_key=api_key)
        print("✅ Agent inizializzato correttamente!")
    except Exception as e:
        print(f"❌ Errore inizializzazione: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 5. Test query semplice
    print("\n💬 Test query...")
    test_messages = [
        "Ciao",
        "Quanti alberi ci sono nel dataset?",
    ]
    
    for msg in test_messages:
        print(f"\n👤 User: {msg}")
        try:
            response = agent.chat(msg)
            print(f"🤖 Agent: {response[:200]}..." if len(response) > 200 else f"🤖 Agent: {response}")
        except Exception as e:
            print(f"❌ Errore: {e}")
            import traceback
            traceback.print_exc()
    
    # 6. Test streaming
    print("\n\n🌊 Test streaming...")
    test_stream_msg = "Dammi un esempio di albero"
    print(f"👤 User: {test_stream_msg}")
    print("🤖 Agent: ", end="", flush=True)
    try:
        for chunk in agent.stream_chat(test_stream_msg):
            # Mostra solo ultime parole del chunk (simula streaming)
            if chunk:
                print(".", end="", flush=True)
        print(f"\n✅ Streaming funzionante!")
    except Exception as e:
        print(f"\n❌ Errore streaming: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("✅ Test completato!")
    print("\nPer testare in Streamlit:")
    print("  streamlit run streamlit_app/app.py")

if __name__ == "__main__":
    test_agent()

