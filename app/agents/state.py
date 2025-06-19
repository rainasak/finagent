from dataclasses import dataclass
from typing import Any
from app.agents.memory import MemoryManager

@dataclass
class AgentState:
    """Represents the current state of the agent system."""
    task: str
    query: str
    subgoals: list[dict[str, Any]]
    current_subgoal_index: int
    memory: MemoryManager
    final_response: str | dict[str, Any] = ""