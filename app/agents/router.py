from typing import Dict, Any
from langchain.tools.base import BaseTool
from langchain_openai import ChatOpenAI
from app.utils.logging import setup_logger, log_function_call, log_function_result, log_error
from app.prompts.tool_router import TOOL_ROUTING_PROMPT
from pydantic import BaseModel, Field, AliasChoices

# Set up logger for this module
logger = setup_logger(__name__)


class ToolRouterSchema(BaseModel):
    """Schema for tool routing decisions. Always use this to structure tool routing responses."""
    selected_tool: str = Field(description="The name of the tool selected for the subgoal from the list ['web_search', 'calculator', 'code_executor', 'document_summarizer'].", validation_alias=AliasChoices("tool", "Tool", "**tool**", "**Tool**", "tool_name", "ToolName"))
    query: str = Field(description="The improved input that is to be passed to the selected tool. This should be a well-formed query or command that the tool can process. Refer to the previous input to the tool to understand the format.", validation_alias=AliasChoices("input", "Input", "**input**", "**Input**"))
    is_url: bool = Field(description="Indicator flagging if the query is a URL, allowing the document_summarizer tool to either get data from the URL or summarize the text directly.")

class ToolRouter:
    """Routes subgoals to appropriate tools based on the task requirements."""
    
    def __init__(self, tools: Dict[str, BaseTool]):
        self.tools = tools
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0).with_structured_output(ToolRouterSchema)
        self.chain = TOOL_ROUTING_PROMPT | self.llm
        self.logger = setup_logger(f"{__name__}.ToolRouter")
        self.logger.info(f"Initialized ToolRouter with tools: {list(tools.keys())}")

    def route(self, subgoal: str) -> tuple[BaseTool, str, str]:
        """Select the most appropriate tool for a given subgoal."""
        log_function_call(self.logger, "route", subgoal=subgoal)
        try:
            tool_names = list(self.tools.keys())
            self.logger.debug(f"Available tools: {tool_names}")
            
            response = self.chain.invoke({"subgoal": subgoal, "tools": tool_names})

            self.logger.debug(f"Tool route response: {response}")

            if not response:
                self.logger.warning("No response received from tool routing chain, defaulting to web_search")
                selected_tool_name = "web_search"
                query = subgoal

            response = ToolRouterSchema.model_validate(response)

            self.logger.debug(f"LLM tool selection response: {response}")
            
            # Extract tool name from response
            selected_tool_name = response.selected_tool.lower()
            query = response.query
            is_url = response.is_url

            if not selected_tool_name:
                self.logger.warning(f"Selected tool '{selected_tool_name}' not found, defaulting to web_search")
                selected_tool_name = self.tools["web_search"]
                query = subgoal
                is_url = False
            
            log_function_result(self.logger, "route", f"Selected tool: {selected_tool_name}, query: {query}")
            selected_tool = self.tools.get(selected_tool_name)
            return selected_tool, query, is_url
        except Exception as e:
            log_error(self.logger, e, "routing subgoal to tool")
            raise

    def execute_tool(self, tool: BaseTool, subgoal: str, query: str, is_url: bool) -> Any:
        """Execute the selected tool with the subgoal."""
        log_function_call(self.logger, "execute_tool", tool=tool.name, subgoal=subgoal)
        try:
            result = None
            if tool.name == "calculator":
                # Parse financial calculation parameters from subgoal
                self.logger.debug("Parsing calculation parameters")
                result = tool.run(query)
            elif tool.name == "code_executor":
                # Generate appropriate code for the subgoal
                self.logger.debug("Generating analysis code")
                result = tool.run(query)
            elif tool.name == "document_summarizer":
                # Prepare document input
                self.logger.debug("Preparing document input")
                result = tool._run(query, is_url)
            else:
                # Default execution for other tools
                self.logger.debug(f"Executing {tool.name} with default handling")
                result = tool.run(query)
            
            log_function_result(self.logger, "execute_tool", result)
            return result
        except Exception as e:
            log_error(self.logger, e, f"executing {tool.name}")
            return {"error": f"Tool execution failed: {str(e)}"}