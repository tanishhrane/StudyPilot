import streamlit as st
from agent import run_agent
from memory import save_message, get_last_messages


st.set_page_config(
    page_title="StudyPilot",
    page_icon="📘",
    layout="centered"
)

# ===============================
# Header Section
# ===============================

st.title("📘 StudyPilot")
st.subheader("Your Agentic AI Study Assistant")

st.markdown("""
### 🚀 What StudyPilot Can Do:

- 📄 **Summarize** your notes or concepts
- 📅 **Create structured study plans**
- 📝 **Generate practice quizzes**

Simply describe what you need in natural language.
""")

st.markdown("---")

# ===============================
# Input Section
# ===============================

user_input = st.text_area(
    "Ask StudyPilot:",
    placeholder="Example: Create a 5 day study plan for machine learning"
)

if st.button("Generate"):

    if user_input.strip() == "":
        st.warning("Please enter a request.")
    else:
        with st.spinner("Thinking..."):
            history = get_last_messages(limit=5)

            result = run_agent(user_input, history)

        st.markdown("### 🔍 Result")
        st.write(result)

st.markdown("---")
st.caption("Built with ❤️ using LLaMA 3 and Streamlit")