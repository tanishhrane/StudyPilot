from datetime import datetime

# Temporary storage (later → MongoDB)
chat_history = []

def save_message(role, content, session_id="default"):
    message = {
        "session_id": session_id,
        "role": role,
        "content": content,
        "timestamp": datetime.now()
    }

    chat_history.append(message)


def get_last_messages(session_id="default", limit=5):
    messages = [
        msg for msg in chat_history
        if msg["session_id"] == session_id
    ]

    # Sort by latest
    messages = sorted(messages, key=lambda x: x["timestamp"], reverse=True)

    # Take last N
    messages = messages[:limit]

    # Return in correct order (old → new)
    return messages[::-1]