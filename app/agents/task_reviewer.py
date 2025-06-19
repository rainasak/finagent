from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from app.prompts.task_planner import TASK_REVIEW_PROMPT
from app.utils.logging import setup_logger, log_function_call, log_function_result, log_error
from app.agents.state import AgentState
from datetime import datetime

class TaskReviewerSchema(BaseModel):
    """Schema for task review output. Always use this to structure subgoal review responses."""
    completed: bool = Field(description="Indicates if the subgoal was successfully completed.")
    description: str = Field(description="Description of the subgoal being reviewed.")
    feedback: str = Field(description="Feedback on the subgoal, if applicable to improve output.")
    retry: bool = Field(description="Indicates if the subgoal needs to be retried.")
    query: str | None = Field(description="The new input to the tool if it needs to be retried based on the tool's output, the subgoal being solved, the tool being used, and the feedback provided.")
    is_url: bool = Field(description="Indicator flagging if the query is a URL, allowing the document_summarizer tool to either get data from the URL or summarize the text directly.")

class TaskReviewer:
    """Task Reviewer that evaluates subgoal results and decides next steps."""
    
    def __init__(self, rate_limiter, max_retries: int = 3):
        self.logger = setup_logger(f"{__name__}.TaskReviewer")
        self.logger.info("Initialized TaskReviewer")
        self.max_retries = max_retries
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, rate_limiter=rate_limiter)
        self.llm = self.llm.with_structured_output(TaskReviewerSchema, method='json_schema', strict=True)
        self.chain = TASK_REVIEW_PROMPT | self.llm
        self.logger.info("TaskReviewer chain initialized")

    def review(self, state: AgentState) -> bool:
        """Review the result of a subgoal and determine if it meets the requirements."""
        # log_function_call(self.logger, "review", subgoal=state.subgoals[state.current_subgoal_index])
        subgoal = state.subgoals[state.current_subgoal_index]
        try:
            if not subgoal.get('skipped', False):
                self.logger.info(f"Reviewing subgoal: {subgoal['description']}")

                curr_date = datetime.now().strftime("%Y-%m-%d")

                subgoal_result = subgoal.get('result', {})
                subgoal_result = subgoal_result.get('result', '') if subgoal_result.get('type') == 'text' else subgoal_result.get('query')

                response = self.chain.invoke({
                    "date": curr_date,
                    "subgoal": subgoal['description'],
                    "tool": subgoal.get('tool', 'web_search'),
                    "result": subgoal_result,
                    "query": subgoal.get('query', '')
                }).model_dump()

                # self.logger.debug(f"Chain response: {response}")

                # self.logger.debug(f"Review result: {response}")
                # Update subgoal based on review
                subgoal['completed'] = response.get('completed', True)
                subgoal['feedback'] = response.get('feedback', '')
                subgoal['retry'] = response.get('retry', False)
                subgoal['query'] = response.get('query', subgoal['query'])
                subgoal['is_url'] = response.get('is_url', False)
                self.logger.debug(f"Subgoal after review: {subgoal}")
                state.subgoals[state.current_subgoal_index] = subgoal
            return state
        except Exception as e:
            log_error(self.logger, e, "reviewing subgoal")
            raise
        
    def should_retry(self, state: AgentState) -> str:
        """Determine if a subgoal should be retried based on the review."""
        subgoal = state.subgoals[state.current_subgoal_index]
        # log_function_call(self.logger, "should_retry", subgoal=subgoal)
        try:
            # If the subgoal was skipped due to dependencies, continue to next
            if subgoal.get('skipped', False):
                self.logger.info(f"Subgoal '{subgoal['description']}' was skipped. Moving to next.")
                return "continue"
            
            # If not completed but has an error indicating permanent failure
            if not subgoal.get('completed', False):
                self.logger.info(f"Task failed due to: {subgoal.get('feedback')}") 

                # Check if error was due to URL/document not having correct information to solve task
                if subgoal.get('is_url', False) and 'error' not in subgoal.get('feedback'):
                    self.logger.info(f"The information found was not relevant to the task. No need to retry task.")
                    return "continue"

                # Check retry count
                retries = subgoal.get('retries', 0)
                if retries < self.max_retries:
                    self.logger.info(f"Subgoal '{subgoal['description']}' needs retry. Current retries: {retries}, Max retries: {self.max_retries}")
                    return 'retry'
                else:
                    self.logger.warning(f"Subgoal '{subgoal['description']}' reached max retries. Not retrying.")
                    return "continue"
            else:
                self.logger.info(f"Subgoal '{subgoal['description']}' is complete. No retry needed.")
                return "continue"
        except Exception as e:
            log_error(self.logger, e, "checking if subgoal should be retried")
            return "continue"  # On error, better to continue than get stuck
