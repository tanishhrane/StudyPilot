# 📘 StudyPilot

StudyPilot is an Agentic AI Study Assistant that dynamically selects and executes academic tools based on user intent.

---

## 🚀 Features

- 📅 Create structured study plans
- 📝 Generate practice quizzes
- 📄 Summarize academic content
- 🧠 Automatic tool selection using LLM-based routing
- ✅ Input validation and safe execution layer

---

## 🏗 Architecture

User Input  
↓  
LLM Decision Layer  
↓  
Tool Routing  
↓  
Validation Layer  
↓  
Tool Execution  
↓  
Formatted Output  

---

## 🛠 Tech Stack

- Python
- LLaMA 3.1 (via Groq API)
- Streamlit
- JSON-based tool routing

---

## ⚙️ Setup Instructions

1. Clone the repository
2. Install dependencies:

   python -m pip install -r requirements.txt

3. Add your Groq API key in `.env`:

   GROQ_API_KEY=your_key_here

4. Run the app:

   python -m streamlit run app.py

---

## 📌 Future Improvements

- Time-constrained study planning
- Conversation memory
- PDF upload support
- Multi-user support

---

Built as part of an Agentic AI system design project.