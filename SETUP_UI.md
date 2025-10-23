# Setup Chatbot con OpenAI API Key dall'UI

## ✅ Modifiche implementate

Ora **non serve più il file `.env`**! La chiave OpenAI si inserisce direttamente nell'interfaccia Streamlit.

### Cosa è cambiato:

1. **Settings nella Sidebar**: Al posto di "User ID", ora c'è un campo **"OpenAI API Key"**
2. **Input sicuro**: Campo password mascherato (i caratteri non sono visibili)
3. **Lazy initialization**: L'agent viene creato solo quando serve, con la chiave dalla UI
4. **Fallback graceful**: Se non inserisci la chiave, il chatbot usa risposte demo

## 🚀 Come usare

### 1. Avvia Streamlit

```bash
streamlit run streamlit_app/app.py
```

### 2. Inserisci la chiave OpenAI

- Apri la **sidebar** (in alto a sinistra)
- Sezione **⚙️ Settings**
- Campo **"OpenAI API Key"**
- Incolla la tua chiave (es. `sk-proj-abc123...`)
- Premi Invio

### 3. Crea una conversazione

- Clicca su **"➕ Nuova Chat"**
- Scrivi un messaggio nella chat

### 4. Inizia a chattare!

Esempi:

- "Quanti alberi ci sono nel dataset?"
- "Calcola CO2 per albero con diametro 30 cm e altezza 15 m"
- "Mostrami gli alberi del distretto 19"

## 🔑 Come ottenere una OpenAI API Key

1. Vai su [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Crea un account (o login se già ce l'hai)
3. Clicca **"Create new secret key"**
4. Copia la chiave (inizia con `sk-proj-...` o `sk-...`)
5. ⚠️ **Importante**: salva la chiave in un luogo sicuro! Non la vedrai più dopo averla generata

### Costi

- Il chatbot usa **GPT-4o-mini** (molto economico: ~$0.15 per 1M token input, ~$0.60 per 1M output)
- Una conversazione tipica costa **meno di $0.01**
- OpenAI offre **$5 di credito gratuito** ai nuovi account

## 🛡️ Sicurezza

- La chiave è salvata solo in **session_state** (memoria temporanea del browser)
- Non viene salvata su disco
- Non viene loggata
- Viene resettata quando chiudi il tab/browser
- Il campo è **mascherato** (tipo password)

## 🔄 Cambiare chiave

1. Vai nei **Settings** (sidebar)
2. Cancella la vecchia chiave e inserisci la nuova
3. L'agent viene automaticamente ricreato con la nuova chiave

## ⚙️ Opzioni avanzate

### Usare file .env (opzionale)

Se preferisci, puoi ancora usare `.env`:

```bash
# Crea file .env nella root del progetto
echo "OPENAI_API_KEY=sk-your-key" > .env
```

L'agent proverà prima la chiave dalla UI, poi dal file `.env`.

### Docker

Con Docker Compose, puoi passare la chiave come variabile d'ambiente:

```yaml
# docker-compose.yml
services:
  streamlit:
    environment:
      - OPENAI_API_KEY=sk-your-key # ⚠️ Non committare su git!
```

Oppure usa un file `.env` (non committato su git):

```bash
# .env
OPENAI_API_KEY=sk-your-key
```

E poi:

```bash
docker compose --env-file .env up
```

## 🧪 Testare senza chiave OpenAI

Se non inserisci la chiave, il chatbot:

1. Mostra un **warning giallo** con istruzioni
2. Usa **risposte demo** di fallback (echo timestamp)
3. Continua a funzionare normalmente per la persistenza conversazioni

## 📋 Checklist completa

- [ ] Installa dipendenze: `pip install -r requirements.txt`
- [ ] Ottieni chiave OpenAI da platform.openai.com
- [ ] Avvia Streamlit: `streamlit run streamlit_app/app.py`
- [ ] Apri sidebar e inserisci chiave in "Settings"
- [ ] Crea nuova conversazione
- [ ] Invia primo messaggio
- [ ] ✅ L'agent risponde in italiano con i tool!

## 🐛 Troubleshooting

### "OpenAI API key not found"

→ Inserisci la chiave nel campo "OpenAI API Key" nella sidebar

### "Rate limit exceeded"

→ Hai superato il limite di richieste/minuto. Aspetta 1 minuto o passa a piano a pagamento su OpenAI

### "Insufficient quota"

→ Credito OpenAI esaurito. Aggiungi metodo di pagamento su platform.openai.com

### Agent usa risposte demo

→ Verifica che la chiave sia corretta (deve iniziare con `sk-`) e che l'account OpenAI sia attivo

### Errore "Failed to initialize agent"

→ Controlla la console/log per dettagli. Possibili cause:

- Chiave non valida
- Network error
- Dipendenze mancanti (rilancia `pip install -r requirements.txt`)

## 📝 Note finali

**Vantaggi di questo setup:**
✅ Nessun file .env da gestire
✅ Ogni utente può usare la propria chiave
✅ Facile da testare/demo
✅ Sicuro (chiave solo in memoria)
✅ Fallback graceful se chiave non disponibile

**Per produzione:**

- Considera autenticazione utenti (es. OAuth)
- Usa secrets management (AWS Secrets Manager, Azure Key Vault)
- Rate limiting per utente
- Monitoring costi OpenAI
