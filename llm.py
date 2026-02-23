from groq import Groq
from config import GROQ_API_KEY, MODEL_NAME, TEMPERATURE

# Create Groq client
client = Groq(api_key=GROQ_API_KEY)


def call_llm(system_prompt, user_prompt):
    """
    Sends prompt to LLaMA model and returns response text.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=TEMPERATURE
    )

    return response.choices[0].message.content
