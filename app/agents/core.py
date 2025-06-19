from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from typing import Dict, Any

from langchain.tools.base import BaseTool
from app.prompts.core import RESPONSE_SYNTHESIS_PROMPT
from app.agents.router import ToolRouter
from app.agents.memory import MemoryManager
from app.tools.analysis_tools import WebSearchTool, CodeExecutorTool, DocumentSummarizerTool, CalculatorTool
from app.utils.logging import setup_logger, log_function_call, log_function_result, log_error
from langchain_core.rate_limiters import InMemoryRateLimiter
from app.agents.state import AgentState
from app.agents.task_planner import TaskPlanner
from app.agents.task_reviewer import TaskReviewer
from app.evaluator.agent_eval import AgentEvaluator

# Set up logger for this module
logger = setup_logger(__name__)

rate_limiter = InMemoryRateLimiter(
    requests_per_second=10,
    check_every_n_seconds=2,
    max_bucket_size=10,
)

evaluator = AgentEvaluator()

class FinancialAgent:
    """Main agent orchestrator that manages the workflow."""
    
    def __init__(self):
        self.logger = setup_logger(f"{__name__}.FinancialAgent")
        self.logger.info("Initialized FinancialAgent")
        self.task_planner = TaskPlanner(rate_limiter=rate_limiter)
        self.task_reviewer = TaskReviewer(rate_limiter=rate_limiter)
        self.tool_router = ToolRouter(self._initialize_tools())
        self.memory = MemoryManager(session_id="financial_agent")
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0, rate_limiter=rate_limiter)
        self.workflow = self._create_workflow()

    def should_continue(self, state: AgentState) -> str:
        """Determine if we should continue processing subgoals.
        
        Returns:
            str: Either "synthesize" if all subgoals are processed, or "process_subgoal" to continue
        """
        return "synthesize" if state.current_subgoal_index >= len(state.subgoals) else "process_subgoal"

    def _get_memory_context(self, subgoal_desc: str, state: AgentState) -> Dict[str, Any]:
        """Get and organize relevant memory context for a subgoal.
        
        Args:
            subgoal_desc: The description of the current subgoal
            state: The current agent state
            
        Returns:
            Dict containing organized context from memory
        """
        context = state.memory.get_relevant_context(subgoal_desc)
        memory_context = {
            "conversation_summary": "",
            "recent_context": [],
            "relevant_history": []
        }
        
        if context:
            for msg in context:
                if msg["role"] == "system" and "summary" in msg["content"].lower():
                    memory_context["conversation_summary"] = msg["content"]
                elif msg in state.memory.recent_context:
                    memory_context["recent_context"].append(msg)
                else:
                    memory_context["relevant_history"].append(msg)
        
        return memory_context
    
    def _format_memory_context(self, memory_context: Dict[str, Any]) -> str:
        """Format memory context sections into a readable string.
        
        Args:
            memory_context: Dictionary containing different types of memory context
            
        Returns:
            str: Formatted context string
        """
        context_parts = []
        if memory_context["conversation_summary"]:
            context_parts.append(memory_context["conversation_summary"])
        if memory_context["recent_context"]:
            context_parts.append("Recent Context:\n" + "\n".join(
                f"{msg['role']}: {msg['content']}" 
                for msg in memory_context["recent_context"]
            ))
        if memory_context["relevant_history"]:
            context_parts.append("Related Historical Context:\n" + "\n".join(
                f"{msg['role']}: {msg['content']}" 
                for msg in memory_context["relevant_history"]
            ))
        return "\n\n".join(context_parts)

    def _check_dependencies(self, state: AgentState, current_subgoal: Dict[str, Any]) -> Dict[str, Any]:
        """Check and handle subgoal dependencies.
        
        Args:
            state: The current agent state
            current_subgoal: The subgoal being processed
            
        Returns:
            Dict: Results from previous dependent subgoals
        """
        prev_results = {}
        if state.current_subgoal_index > 0:
            previous_subgoals = state.subgoals[:state.current_subgoal_index]
            for subgoal in previous_subgoals:
                if str(subgoal["order_number"]) in current_subgoal.get('depends_on', []):
                    if not subgoal.get('completed', False):
                        current_subgoal['skipped'] = True
                        return prev_results
                    prev_results[subgoal['description']] = subgoal.get('result', '')
        return prev_results

    def process_subgoal(self, state: AgentState) -> AgentState:
        """Process the current subgoal using appropriate tool."""
        try:
            current_subgoal = state.subgoals[state.current_subgoal_index]
            self.logger.debug(f"Processing subgoal: {current_subgoal}")
            
            # Get memory context
            memory_context = self._get_memory_context(current_subgoal['description'], state)
            prev_results = {"memory_context": self._format_memory_context(memory_context)}
            
            # Check dependencies
            prev_results.update(self._check_dependencies(state, current_subgoal))
            if current_subgoal.get('skipped', False):
                return state
            
            # Add context to query
            context_str = "\n\n".join(
                f"{desc}: {result}" for desc, result in prev_results.items()
            ) if prev_results else ""

            context_str += f"\n\nCurrent Subgoal: {current_subgoal['description']}"

            # Route and execute
            tool, query, is_url = self.tool_router.route(context_str)
            state.subgoals[state.current_subgoal_index]['tool'] = tool.name
            state.subgoals[state.current_subgoal_index]['query'] = query
            
            result = self.tool_router.execute_tool(tool, current_subgoal['description'], query, is_url)

            evaluator.evaluate("tool_use", query=current_subgoal['description'], tool=tool.name, output=result.get('result'))

            evaluator.evaluate('task_success', query=current_subgoal['description'], output=result.get('result'))
            
            # Update state
            current_subgoal['completed'] = True
            current_subgoal['result'] = result
            state.subgoals[state.current_subgoal_index] = current_subgoal
            return state
            
        except Exception as e:
            self.logger.error(f"Error processing subgoal: {str(e)}")
            current_subgoal = state.subgoals[state.current_subgoal_index]
            current_subgoal['result'] = {"error": str(e)}
            current_subgoal['completed'] = False
            state.subgoals[state.current_subgoal_index] = current_subgoal
            return state
        
    def retry_subgoal(self, state: AgentState) -> AgentState:
        """Retry the current subgoal based on the review feedback."""
        self.logger.debug(f"Retrying subgoal: {state.subgoals[state.current_subgoal_index]}")
        current_subgoal = state.subgoals[state.current_subgoal_index]
        
        # Increment retry count
        current_subgoal['retries'] += 1
        
        # If max retries reached, mark as failed
        if current_subgoal['retries'] >= self.task_reviewer.max_retries:
            current_subgoal['completed'] = False
            current_subgoal['skipped'] = True
            self.logger.warning(f"Max retries reached for subgoal: {current_subgoal['description']}")
            return state

        query = current_subgoal.get('query', current_subgoal['description'])
        is_url = current_subgoal.get('is_url', False)
        
        result = self.tool_router.execute_tool(self.tool_router.tools[current_subgoal["tool"]], current_subgoal['description'], query, is_url)

        evaluator.evaluate("tool_use", query=current_subgoal['description'], tool=current_subgoal["tool"], output=result.get('result'))

        evaluator.evaluate('task_success', query=current_subgoal['description'], output=result.get('result'))
        
        # Update state
        current_subgoal['result'] = result
        state.subgoals[state.current_subgoal_index] = current_subgoal
        return state
        
    def next_subgoal(self, state: AgentState) -> AgentState:
        """Move to the next subgoal in the list."""
        state.current_subgoal_index += 1
        self.logger.info(f"Moving to next subgoal: {state.current_subgoal_index}/{len(state.subgoals)}")
        return state
    
    def synthesize_response(self, state: AgentState) -> AgentState:
        """Synthesize final response from subgoal results by focusing on answering the user's query."""
        try:
            self.logger.info("Synthesizing final response from subgoals")
            
            # Collect useful results and information that helps answer the query
            useful_results = []
            for subgoal in state.subgoals:
                if subgoal.get('completed', False):
                    result = subgoal.get('result', None)
                    if result:
                        if isinstance(result, dict):
                            useful_results.append({
                                **result,
                                "context": subgoal['description']
                            })
                        else:
                            useful_results.append({
                                "info": str(result),
                                "context": subgoal['description']
                            })
            
            # Get relevant context from memory that helps answer the query
            context = state.memory.get_relevant_context(state.query)
            context_str = ""
            if context:
                context_str = "\nRelevant Historical Context:\n" + "\n".join(
                    f"{msg['role']}: {msg['content']}" for msg in context
                )
            
            self.logger.info(f"Synthesizing response")
            
            # Process visualizations and text separately
            visualizations = []
            text_results = []
            
            for result in useful_results:
                if isinstance(result, dict):
                    if result.get('type') == 'plot' and 'display' in result:
                        visualizations.append(result['display'])
                    else:
                        text_results.append(f'Context: {result["context"]}\nResult:{str(result["result"])}')
                else:
                    text_results.append(str(result))
            
            # Generate text response focused on answering the query
            response_text = self.llm.invoke(RESPONSE_SYNTHESIS_PROMPT.format_messages(
                query=state.query,
                context=context_str,
                results="\n\n".join(text_results)
            )).content

            evaluator.evaluate("task_success", query=state.task, output=response_text)
            
            # Combine text and visualizations
            final_response = {
                'content': response_text,
                'display': '\n\n'.join(visualizations) if visualizations else None
            }
            
            # Store results in memory
            state.memory.add_to_memory("assistant", final_response['content'])
            
            return AgentState(
                task=state.task,
                query=state.query,
                subgoals=state.subgoals,
                current_subgoal_index=state.current_subgoal_index,
                memory=state.memory,
                final_response=final_response
            )
            
        except Exception as e:
            self.logger.error(f"Error synthesizing response: {str(e)}")
            return AgentState(
                task=state.task,
                query=state.query,
                subgoals=state.subgoals,
                current_subgoal_index=state.current_subgoal_index,
                memory=state.memory,
                final_response=f"Error synthesizing response: {str(e)}"
            )

    def _create_workflow(self) -> CompiledStateGraph:
        """Create the agent workflow using langgraph."""
        self.logger.info("Creating workflow")
        workflow = StateGraph(AgentState)

        # Add nodes using class methods as handlers
        workflow.add_node("process_subgoal", self.process_subgoal)
        workflow.add_node("synthesize", self.synthesize_response)
        workflow.add_node("retry_subgoal", self.retry_subgoal)
        workflow.add_node("next_subgoal", self.next_subgoal)
        workflow.add_node("review_subgoal", self.task_reviewer.review)

        # Add edges with checkpointing
        workflow.add_edge("process_subgoal", "review_subgoal")
        workflow.add_conditional_edges(
            "review_subgoal",
            self.task_reviewer.should_retry,
            {
                "retry": "retry_subgoal",
                "continue": "next_subgoal"
            }
        )
        workflow.add_edge("retry_subgoal", "review_subgoal")
        workflow.add_conditional_edges(
            "next_subgoal",
            self.should_continue,
            {
                "synthesize": "synthesize",
                "process_subgoal": "process_subgoal"
            }
        )
        workflow.add_edge("synthesize", END)
        
        workflow.set_entry_point("process_subgoal")
        
        return workflow.compile()

    def _initialize_tools(self) -> Dict[str, BaseTool]:
        """Initialize all available tools."""
        tools = {
            "web_search": WebSearchTool(),
            "calculator": CalculatorTool(),
            "code_executor": CodeExecutorTool(),
            "document_summarizer": DocumentSummarizerTool(),
        }
        self.logger.debug(f"Initialized tools: {list(tools.keys())}")
        return tools

    def process_query(self, query: str) -> str:
        """Process a user query through the agent workflow."""
        # log_function_call(self.logger, "process_query", query=query)
        try:
            # Input validation
            if not query or not isinstance(query, str):
                raise ValueError("Query must be a non-empty string")
            
            # Add user query to memory
            self.memory.add_to_memory("human", query)
            
            # Get relevant context from previous interactions
            context = self.memory.get_relevant_context(query)
            context_str = ""
            if context:
                context_str = "\n\nPrevious relevant context:\n" + "\n".join(
                    f"{msg['role']}: {msg['content']}" for msg in context
                )
            
            # Plan subgoals with context
            try:
                # Include context in planning
                subgoals, explanation = self.task_planner.plan(query=query, context=context_str)

                evaluator.evaluate('coherence_reasoning', output=f"Subgoals:{subgoals}\n\nReasoning:{explanation}")
                
                if subgoals is None or (isinstance(subgoals, str) and subgoals.strip() in ["", "null"]):
                    error_msg = "I couldn't break down your query into actionable steps. Could you please rephrase it?"
                    # self.memory.add_to_memory("assistant", error_msg)
                    return error_msg
                
                # If subgoals is an empty list, then attempt to answer the query using the existing context.
                if isinstance(subgoals, list) and not subgoals:
                    self.logger.warning("No subgoals generated, attempting to answer query directly")
                    response = self.llm.invoke(RESPONSE_SYNTHESIS_PROMPT.format_messages(
                        query=query,
                        context=context_str,
                        results=[]
                    )).content
                    
                    # Add response to memory
                    self.memory.add_to_memory("assistant", response)
                    return response
                
                self.logger.info(f"Created {len(subgoals)} subgoals for query")
            except Exception as e:
                self.logger.error(f"Error in task planning: {str(e)}")
                error_msg = f"I encountered an error while planning how to answer your query: {str(e)}"
                # self.memory.add_to_memory("assistant", error_msg)
                return error_msg
            
            # Create initial state
            initial_state = AgentState(
                task=query,
                query=f"Context:{context_str}\n\nQuery:{query}",
                subgoals=subgoals,
                current_subgoal_index=0,
                memory=self.memory
            )
            
            # Execute the workflow
            try:
                final_state = self.workflow.invoke(initial_state, {"recursion_limit": 100})
                response = final_state["final_response"]
                  # Check if response is empty or error message
                if not response or (isinstance(response, dict) and "error" in str(response.get('content', '')).lower()):
                    self.logger.warning(f"Potentially problematic response: {response}")
                    error_note = "\n\nI may not have fully answered your query. Please let me know if you need clarification or want to try a different approach."
                    if isinstance(response, dict):
                        response['content'] += error_note
                        response['display'] = None
                    else:
                        response = {
                            'content': str(response) + error_note,
                            'display': None
                        }
                    return response
                
                # If response is a string, wrap it in a dict with display field
                if isinstance(response, str):
                    response = {
                        'content': response.replace('$', '\$'),
                        'display': None
                    }
                
                # Add response to memory
                self.memory.add_to_memory("assistant", response['content'])
                
                # log_function_result(self.logger, "process_query", response)
                return response
                
            except Exception as e:
                self.logger.error(f"Error in workflow execution: {str(e)}")
                error_msg = f"I encountered an error while processing your request: {str(e)}. Please try rephrasing your question."
                # self.memory.add_to_memory("assistant", error_msg)
                return error_msg
                
        except Exception as e:
            log_error(self.logger, e, "processing query")
            error_msg = f"An unexpected error occurred: {str(e)}. Please try again or contact support if the issue persists."
            # self.memory.add_to_memory("assistant", error_msg)
            return error_msg
