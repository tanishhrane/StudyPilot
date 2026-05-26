# 📘 StudyPilot

StudyPilot is an Agentic AI-powered academic assistant that intelligently understands user requests, selects the correct tool, and generates structured educational outputs such as study plans, quizzes, summaries, and calendar schedules.

---

# 🚀 Features

## 📅 Structured Study Plan Generator
- Generate multi-day study roadmaps
- Beginner → advanced progression
- Daily time-slot support
- Real calendar date generation
- Google Calendar sync support

## 📝 Quiz Generator
- Generate topic-based practice quizzes
- Dynamic question generation using LLMs

## 📄 Academic Summarizer
- Summarize long academic content into concise notes

## 🧠 Agentic AI Routing
- LLM-based tool selection
- Automatic argument extraction
- JSON-based decision making

## 🔐 Authentication
- Email/password authentication
- Google OAuth integration

## 📆 Google Calendar Integration
- Sync generated study plans directly to Google Calendar
- Automatic event scheduling

---

# 🏗 System Architecture

```text
User Input
   ↓
LLM Decision Layer
   ↓
Tool Selection
   ↓
Argument Extraction
   ↓
Validation Layer
   ↓
Tool Execution
   ↓
Structured Output
   ↓
(Optional) Google Calendar Sync
```

---

# 🛠 Tech Stack

## Backend
- Python

## AI / LLM
- LLaMA 3.1
- Groq API

## Frontend
- Streamlit

## APIs
- Google Calendar API
- Google OAuth API

## Data Handling
- JSON structured outputs

---

# ⚙️ Setup Instructions

## 1. Clone Repository

```bash
git clone <your_repo_url>
cd StudyPilot
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

---

## 4. Create `.env` File

Add your API key:

```env
GROQ_API_KEY=your_groq_api_key
```

---

## 5. Add Google OAuth Credentials

Download your Google OAuth credentials JSON file and place it in the project root:

```text
client_secrets.json
```

---

## 6. Run the App

```bash
python -m streamlit run app.py
```

---

# 📌 Example Prompts

## Study Plans

```text
Create a 7-day machine learning study plan from 6 PM to 8 PM
```

```text
Make a DSA study plan for 14 days starting from Monday from 7 PM to 9 PM
```

---

## Quiz Generation

```text
Generate a 10-question quiz on operating systems
```

---

## Summarization

```text
Summarize this paragraph about neural networks
```

---

# 📂 Project Structure

```text
StudyPilot/
│
├── tools/
│   ├── study_plan.py
│   ├── quiz.py
│   ├── summarizer.py
│   └── calendar_sync.py
│
├── app.py
├── agent.py
├── llm.py
├── auth.py
├── config.py
├── requirements.txt
└── README.md
```

---

# 🔮 Future Improvements

- RAG-based document learning
- PDF upload + querying
- Persistent memory/chat history
- Editable study plans
- Personalized learning paths
- Difficulty-based planning
- Docker deployment
- Multi-user database support
- Resource recommendations
- Progress tracking dashboard

---

# 🧠 Key Learning Concepts Used

- Agentic AI systems
- Tool routing
- Structured JSON outputs
- Prompt engineering
- OAuth authentication
- API integrations
- LLM validation pipelines
- Deterministic vs LLM-based logic separation

---

# 👨‍💻 Built With

Built as part of an Agentic AI engineering and system design project focused on real-world AI application architecture.