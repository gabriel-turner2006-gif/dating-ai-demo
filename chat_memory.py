from datetime import datetime


class ChatMemory:
    """
    In-memory chat storage for cloud environments.
    Data persists during the session but clears when session ends.
    """
    
    def __init__(self):
        self._messages: list[dict] = []

    def add_message(self, role: str, content: str):
        self._messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })

    def get_history(self, limit: int = 20) -> list[dict]:
        """Return the last `limit` messages."""
        messages = self._messages[-limit:] if limit else self._messages
        return [{"role": msg["role"], "content": msg["content"]} for msg in messages]

    def clear(self):
        self._messages.clear()

    def close(self):
        # No-op for in-memory storage, kept for API compatibility
        pass