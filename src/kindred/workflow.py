from __future__ import annotations
from typing import Dict, List, Optional, TYPE_CHECKING
import asyncio

if TYPE_CHECKING:
    from kindred.agent import Kin

class SequentialWorkflow:
    """A ritual where Kin complete tasks in a linear sequence."""
    
    def __init__(self, agents: List[Kin]):
        self.agents = agents

    def run(self, initial_prompt: str) -> str:
        """Executes the sequence of Kin, passing the output of one to the next."""
        current_input = initial_prompt
        
        for agent in self.agents:
            print(f"📡 Ritual: Handing off to {agent.name}...")
            # The next agent processes the result of the previous one
            current_input = agent.chat(current_input)
            
        return current_input
    
class ParallelWorkflow:
    """A ritual where multiple Kin analyze the same task simultaneously."""
    def __init__(self, agents: List[Kin]):
        self.agents = agents

    async def run(self, prompt: str) -> Dict[str, str]:
        """Sends the prompt to all agents at once and returns a map of their answers."""
        print(f"🔥 The Gathering: Engaging {len(self.agents)} Kin simultaneously...")
        
        # We create a list of 'tasks' for Python to run at the same time
        tasks = [asyncio.to_thread(agent.chat, prompt) for agent in self.agents]
        
        # Gather all the results
        results = await asyncio.gather(*tasks)
        
        # Return a dictionary linking agent names to their responses
        return {agent.name: response for agent, response in zip(self.agents, results)}