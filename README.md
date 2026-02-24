<div align="center">

# 🧠 AI-BOS — Business Intelligence Operating System

**An enterprise-grade, AI-powered Business Intelligence platform that ingests Gmail, Slack, CRM, and documents — then lets you query everything in plain English.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge)](https://langchain.com)
[![Gemini](https://img.shields.io/badge/Gemini_2.0_Flash-AI-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-00A67E?style=for-the-badge)](https://pinecone.io)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/your-org/ai-bos/ci.yml?style=for-the-badge&label=CI)](https://github.com/your-org/ai-bos/actions)

> 🚀 **[Live Demo →]((https://rag-business-intelligence-han.streamlit.app/))** &nbsp;|&nbsp; 📹 **[Video Walkthrough →](#demo)**

</div>

---

## ✨ What Makes AI-BOS Special

| Feature | Description |
|---------|-------------|
| 🔍 **Universal Search** | Ask anything about your business in plain English |
| 📧 **Gmail Integration** | OAuth2-powered email indexing + PII anonymization |
| 💬 **Slack Connected** | Real-time channel message indexing |
| 🗂️ **CRM Pipeline** | JSON/CSV deal tracking with AI-powered analysis |
| 📊 **Executive Reports** | One-click SWOT, KPI, and pipeline reports |
| 🧠 **RAG Pipeline** | Gemini embeddings + Pinecone cosine similarity |
| 🔧 **Admin Dashboard** | Password-protected (admin123) system control panel |
| 📱 **Fully Responsive** | Works on desktop, tablet, and mobile |
| 🔒 **Privacy-First** | PII auto-anonymized before any vectorization |

---

## 🎯 Demo

> _[Embed your screen recording here — use OBS or Loom]_

```
00:00  Dashboard — live metrics, Pinecone health gauge
00:30  Data Ingestion — upload PDF + re-index
01:00  AI Assistant — "What are our top deals?" → streaming response
01:30  Reports — Sales / Q1 / Pipeline → SWOT + charts
02:00  Admin Panel — admin123 → Overview / Logs / Re-Index
02:30  Global Search — cross-page knowledge search
03:00  END
```

---

## 🚀 Quick Start

### Option A — One Command (recommended)

```bash
git clone https://github.com/your-org/ai-bos.git
cd ai-bos
pip install -r requirements.txt
cp .env.example .env          # Add your API keys
python demo_data_generator.py  # Generate demo data
streamlit run ui.py
```

Open **[http://localhost:8501](http://localhost:8501)**

### Option B — Docker

```bash
docker build -t ai-bos .
docker run -p 8501:8501 \
  -e GOOGLE_API_KEY=your_key \
  -e PINECONE_API_KEY=your_key \
  ai-bos
```

---

## 🔑 Environment Variables

Copy `.env.example` → `.env` and fill in:

```env
GOOGLE_API_KEY=AIza...            # Gemini API key (required)
PINECONE_API_KEY=xxxx-xxxx-xxxx  # Pinecone API key (required)
SLACK_BOT_TOKEN=xoxb-...          # Slack Bot Token (optional)
SLACK_CHANNEL_ID=C0XXXXXXXXX      # Channel to monitor (optional)
GMAIL_TOKEN_PATH=token.json       # OAuth token path (optional)
```

> Get API keys: **[Google AI Studio](https://aistudio.google.com)** · **[Pinecone Console](https://app.pinecone.io)**

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit UI (ui.py)                   │
│  Dashboard │ Ingestion │ AI Chat │ Reports │ Admin       │
└────────────────────┬───────────────────────┬────────────┘
                     │                       │
            ┌────────┴────────┐    ┌─────────┴────────┐
            │   agent.py      │    │  connectors.py    │
            │  LangChain RAG  │    │ Gmail/Slack/CRM   │
            │  Gemini 2.0     │    └─────────┬────────┘
            └────────┬────────┘              │
                     └──────────┬────────────┘
                                │
                    ┌───────────┴───────────┐
                    │   Pinecone Vector DB   │
                    │  cosine, 768 dims      │
                    └───────────────────────┘
```

---

## 📁 Project Structure

```
ai-bos/
├── ui.py                    # Main Streamlit app
├── agent.py                 # LangChain business agent
├── connectors.py            # Gmail / Slack / CRM connectors
├── rag.py                   # RAG pipeline (Pinecone + Gemini)
├── demo_data_generator.py   # Generate demo business data
├── main.py                  # CLI + RAG evaluation
├── requirements.txt         # Dependencies
├── Dockerfile               # Production container
├── .streamlit/config.toml   # Streamlit theme
├── .github/workflows/ci.yml # CI/CD pipeline
├── tests/
│   └── full_suite.py        # 30+ pytest tests
├── demo_data/               # Auto-generated demo data
│   ├── crm_data.json
│   ├── activity_log.csv
│   ├── Q_report.txt
│   └── company_handbook.md
└── .env.example             # Environment template
```

---

## 🧪 Running Tests

```bash
# Run full test suite
pytest tests/ -v

# With coverage report
pytest tests/ --cov=. --cov-report=html

# Specific section only
pytest tests/ -k "TestPII"
pytest tests/ -k "TestStreamlit"
```

---

## 🤝 Contributing

1. Fork the repo
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open Pull Request ← CI runs automatically

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">

**AI-BOS v2.0** &nbsp;|&nbsp; Made with ❤️ for showcase &nbsp;|&nbsp; [IntelForge Engine](https://github.com)

*Built with Streamlit • LangChain • Gemini 2.0 Flash • Pinecone*

</div>
