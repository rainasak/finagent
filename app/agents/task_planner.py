from pydantic import BaseModel, Field, AliasChoices
from langchain_openai import ChatOpenAI
from app.prompts.task_planner import TASK_PLANNING_PROMPT
from app.utils.logging import setup_logger, log_function_call, log_error, log_function_result
from typing import Any
from datetime import datetime

class GoalSchema(BaseModel):
    """Schema for individual subgoals."""
    order_number: int = Field(description="Order number of the subgoal in the sequence. This is used to maintain the order of subgoals. It is 1-indexed.")
    description: str = Field(description="Description of the subgoal to be achieved.")
    depends_on: list[str] | None = Field(default_factory=list, description="List of subgoals that this subgoal depends on.")

class TaskPlannerSchema(BaseModel):
    """Schema for task planning output. Always use this to structure response to user queries."""
    subgoals: list[GoalSchema] | str | None = Field(description="List of subgoals generated from the complex query.", validation_alias=AliasChoices("subgoals", "Subgoals", "**subgoals**", "**Subgoals**"))
    explanation: str = "Reasoning for why the plan was broken down into these subgoals based on the user query, context and available tools."

class TaskPlanner:
    """Task Planner that decomposes complex queries into actionable subgoals."""
    def __init__(self, rate_limiter):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, rate_limiter=rate_limiter)
        self.llm = self.llm.with_structured_output(TaskPlannerSchema, method='json_schema', strict=True)
        self.chain = TASK_PLANNING_PROMPT | self.llm
        self.logger = setup_logger(f"{__name__}.TaskPlanner")

    def plan(self, query: str, context: str) -> tuple[list[dict[str, Any], str]]:
        """Break down a complex query into structured subgoals."""
        log_function_call(self.logger, "plan", query=query, context=context)
        try:
            curr_date = datetime.now().strftime("%Y-%m-%d")
            response = self.chain.invoke({"date": curr_date, "query": query, "context": context}).model_dump()
            self.logger.debug(f"Chain response: {response}")

            subgoals = response.get('subgoals', [])

            if subgoals is None or (isinstance(subgoals, str) and subgoals.strip() in ["", "null"]):
                error_msg = "I couldn't break down your query into actionable steps. Could you please rephrase it?"
                return error_msg

            for subgoal in subgoals:
                subgoal["retries"] = 0
                subgoal["completed"] = False
                subgoal["result"] = None
            return subgoals, response.get('explanation')
        except Exception as e:
            log_error(self.logger, e, "planning subgoals")
            raise