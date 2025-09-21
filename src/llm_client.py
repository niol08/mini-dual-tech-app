import os
from dotenv import load_dotenv
load_dotenv()

class LLMClient:
    def get_insight(self, prompt: str) -> str:
        raise NotImplementedError()

class DummyLLMClient(LLMClient):
    def get_insight(self, prompt: str) -> str:
        return """Dummy LLM response (no external calls configured).
Example summary:
- Observations: signals looked within expected ranges (placeholder).
- Uncertainties: this is synthetic output; implement a real LLM adapter.
- Suggested next steps: consult specialist, run formal analysis pipeline.
"""

class GeminiAdapter(LLMClient):
    def __init__(self):
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not (self.api_key or os.getenv('GOOGLE_APPLICATION_CREDENTIALS')):
            raise EnvironmentError('Set GEMINI_API_KEY or GOOGLE_APPLICATION_CREDENTIALS for GeminiAdapter.')

    def get_insight(self, prompt: str) -> str:
        # TODO: Implement a real call here.
        return f"[GeminiAdapter placeholder] Would call Gemini with prompt len={len(prompt)}. Implement adapter as needed."
