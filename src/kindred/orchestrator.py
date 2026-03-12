from __future__ import annotations
import ollama
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from kindred.agent import Kin

class Kindred:
    def __init__(self, host: str = "http://localhost:11434"):
        self.client = ollama.Client(host=host)
        self.agents: List[Kin] = []

    def check_connection(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception:
            return False
        
    def create_agent(self, name: str, model: str, role: str = "Assistant") -> Kin:
        """Creates a new Kin and ensures the model is available."""
        from kindred.agent import Kin
        
        # Check if the model exists locally
        local_models = self.get_models()
        if model not in local_models and f"{model}:latest" not in local_models:
            print(f"📥 Model '{model}' not found. Pulling from Ollama (this may take a while)...")
            self.client.pull(model)
        
        new_agent = Kin(name=name, model=model, orchestrator=self, role=role)
        self.agents.append(new_agent)
        return new_agent

    def get_models(self) -> List[str]:
        response = self.client.list()
        # Adjusted for 2026 Ollama response objects
        return [m.model for m in response.models]