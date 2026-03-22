# 🏛️ GIETU Nexus — Digital Archaeology Engine
### Semantic Search & Notification Engine · SDG 4 (Quality Education)

---

## 🚨 The Problem (PS2)
University websites are **PDF graveyards**. Critical notices — exam schedules,
scholarship deadlines, bus routes — are uploaded as non-searchable scanned PDFs.
Students miss deadlines because finding information requires clicking through dozens of links.

## ✅ The Solution
A **Perplexity-style semantic search engine** that:
1. **Scrapes** the GIETU notices page automatically
2. **OCR-reads** scanned/image PDFs using EasyOCR
3. **Indexes** everything into a FAISS vector database
4. **Answers** natural language questions directly — not links to files

---

## 🚀 Quick Start

### Prerequisites
- Python 3.11 (recommended) — 3.14 works with warnings
- [Ollama](https://ollama.ai) installed and running

### 1. Install dependencies
```powershell
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Pull an LLM via Ollama
```powershell
ollama pull phi3        # Recommended (fast, small)
# OR
ollama pull llama3      # More capable
ollama serve            # Make sure this is running
```

### 3. Run the app
```powershell
streamlit run app.py
```

Open → http://localhost:8501

---

## 🔐 Login Credentials
| Username | Password   | Role    |
|----------|------------|---------|
| admin    | gietu      | Admin   |
| student  | giet2024   | Student |
| faculty  | faculty123 | Faculty |

---

## 📐 Architecture
```
Student Query
     │
     ▼
┌─────────────────────────────────────────┐
│  GIETU Nexus — Digital Archaeology      │
│                                         │
│  Scraper → OCR/PDF → Embedder (MiniLM)  │
│                  ↓                      │
│             FAISS Vector DB             │
│                  ↓ top-K chunks         │
│             Ollama LLM (phi3)           │
└─────────────────────────────────────────┘
          ↓
   Direct Answer + TTS
```

## 🌍 SDG 4 Alignment
**Quality Education** — equitable access to information for all students,
including those searching for scholarship notices by category (OBC, SC, ST).

---

## ✨ Features
- 🔍 **Perplexity-style semantic search** — ask in plain English
- 💬 **Persistent chat assistant** — remembers session history
- 📂 **Notice vault** — browse, filter, preview all indexed PDFs
- 🆕 **Recent notices** — track newly downloaded documents by date
- 👁️ **OCR fallback** — reads scanned/image-only PDFs
- 🔊 **Voice replies** — TTS audio output (toggleable)
- 📧 **Email summary** — send session answers to any address
- 🏷️ **Auto-categorization** — exam / scholar / schedule / general
- 🌐 **Multi-user auth** — admin, student, faculty roles
