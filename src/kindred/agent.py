from __future__ import annotations
from typing import List, Dict, Optional, Any, TYPE_CHECKING
from kindred.memory import KnowledgeBase

if TYPE_CHECKING:
    from kindred.orchestrator import Kindred

class Kin:
    """A specific AI agent tied to a local model."""
    
    def __init__(
        self, 
        name: str, 
        model: str, 
        orchestrator: Kindred,
        role: Optional[str] = None  # Role is now optional
    ):
        self.name = name
        self.model = model
        self.orchestrator = orchestrator
        self.role = role
        
        # Initialize knowledge storage
        self.knowledge = KnowledgeBase(agent_name=self.name, orchestrator=self.orchestrator)
        
        # Initialize conversation history
        self.messages: List[Dict[str, str]] = []
        if self.role:
            self.messages.append({"role": "system", "content": self.role})

    def learn(self, file_path: str):
        """Ingest files into the Kin's vector memory."""
        self.knowledge.add_document(file_path)

    def chat(self, prompt: str) -> str:
        """Sends a message to the model, checking for local context first."""
        
        # 1. Attempt to find relevant context from learned files
        context = ""
        try:
            query_response = self.orchestrator.client.embed(
                model="nomic-embed-text",
                input=prompt
            )
            results = self.knowledge.collection.query(
                query_embeddings=query_response.embeddings, 
                n_results=3
            )
            
            if results and results['documents'] and results['documents'][0]:
                context = "\n".join(results['documents'][0])
        except Exception:
            pass

        # 2. Build the message content
        # We keep it descriptive and minimal so it doesn't break custom Modelfile logic
        if context:
            user_content = f"Context:\n{context}\n\nQuestion: {prompt}"
        else:
            user_content = prompt

        self.messages.append({"role": "user", "content": user_content})
        
        # 3. Request completion
        response = self.orchestrator.client.chat(
            model=self.model,
            messages=self.messages
        )
        
        content = response.message.content
        
        # 4. Clean up history
        # We pop the context-heavy message and store the pure question/answer
        self.messages.pop() 
        self.messages.append({"role": "user", "content": prompt})
        self.messages.append({"role": "assistant", "content": content})
        
        return content
    
    def clear_memory(self):
        """Resets the history. Keeps the role if one was provided."""
        self.messages = []
        if self.role:
            self.messages.append({"role": "system", "content": self.role})