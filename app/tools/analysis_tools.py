from langchain.tools.base import BaseTool
from langchain_tavily import TavilySearch, TavilyCrawl
from langchain_experimental.utilities import PythonREPL
from langchain_openai import ChatOpenAI
from typing import Dict, Any
import requests
from bs4 import BeautifulSoup
import io
import base64
import matplotlib.pyplot as plt
from pydantic import Field, BaseModel
from PyPDF2 import PdfReader
import logging
from app.utils.logging import setup_logger, log_function_call, log_function_result, log_error
from app.prompts.tools import DOCUMENT_SUMMARIZER_PROMPT, WEB_SEARCH_PROMPT, CODE_SANITIZER_PROMPT
from datetime import datetime

# Set up logger for this module
logger = setup_logger(__name__)

class WebSearchTool(BaseTool):
    """Searches for real-time financial information, news, and market data"""
    
    name: str = "web_search"
    description: str = "Searches for real-time financial information, news, and market data"
    search_tool: TavilySearch = Field(default=None)
    crawl_tool: TavilyCrawl = Field(default=None)
    llm: ChatOpenAI = Field(default=None)
    logger: logging.Logger = Field(default_factory=lambda: setup_logger(f"{__name__}.WebSearchTool"))

    def __init__(self):
        super().__init__()
        # Configure TavilySearch with financial-focused settings
        self.search_tool = TavilySearch(
            max_results=7,
            topic="general"
        )
        # Configure TavilyCrawl with reasonable limits
        self.crawl_tool = TavilyCrawl(
            max_depth=3,
            max_pages=5,
            timeout=30
        )
        self.llm = ChatOpenAI(temperature=0)
        self.logger = setup_logger(f"{__name__}.WebSearchTool")

    def _format_results(self, results: dict) -> str:
        """Format the search results into a readable string."""
        formatted = []
        
        if isinstance(results, list):
            for result in results:
                if isinstance(result, dict):
                    title = result.get('title', 'No title')
                    url = result.get('url', 'No URL')
                    snippet = result.get('content', 'No content')
                    formatted.append(f"Title: {title}\nURL: {url}\nSummary: {snippet}\n")
        
        return "\n".join(formatted) if formatted else str(results)

    def _run(self, query: str) -> str:
        """Execute the web search with the given query."""
        log_function_call(self.logger, "_run", query=query)
        try:
            # Add date context to the query for recent results
            curr_date = datetime.now().strftime("%Y-%m-%d")
            
            # First perform the search
            search_results = self.search_tool.invoke(query)
            
            # If we got search results, crawl the first few URLs for more detailed information
            if search_results and isinstance(search_results, list):
                urls_to_crawl = [r['url'] for r in search_results if 'url' in r]
                detailed_results = []
                
                for url in urls_to_crawl:
                    try:
                        self.logger.debug(f"Crawling URL: {url}")
                        crawl_result = self.crawl_tool.invoke(url)
                        if crawl_result:
                            detailed_results.append(crawl_result)
                    except Exception as e:
                        self.logger.warning(f"Failed to crawl {url}: {str(e)}")
                        continue
                
                # Combine and format all results
                formatted_results = self._format_results(search_results + detailed_results)
            else:
                formatted_results = self._format_results(search_results)

            formatted_results = (WEB_SEARCH_PROMPT | self.llm).invoke({"query": query, "results": formatted_results, "today": curr_date})
            
            log_function_result(self.logger, "_run", formatted_results.content)
            return {
                "type": "text",
                "query": query,
                "result": formatted_results.content
            }
            
        except Exception as e:
            error_msg = f"Web search failed: {str(e)}"
            log_error(self.logger, e, "web search")
            raise RuntimeError(error_msg)
        
class SanitizedCodeSchema(BaseModel):
    code: str = Field(description="Sanitized code that can be run in a Python REPL tool. Each statement should end with ';'. The code should only contain code statements.")
        
class CodeExecutorTool(BaseTool):
    """Executes Python code for financial analysis and visualization"""
    
    name: str = "code_executor"
    description: str = "Executes Python code for financial analysis and visualization"
    python_repl: PythonREPL = Field(default_factory=PythonREPL)
    llm: ChatOpenAI = Field(default=None)
    logger: logging.Logger = Field(default_factory=lambda: setup_logger(f"{__name__}.CodeExecutorTool"))

    def __init__(self):
        super().__init__()
        self.python_repl = PythonREPL()
        self.llm = ChatOpenAI(model="gpt-4.1", temperature=0).with_structured_output(SanitizedCodeSchema, method='json_schema', strict=True)
        self.logger = setup_logger(f"{__name__}.CodeExecutorTool")

    def _run(self, code: str) -> Dict[str, Any]:
        """Execute Python code and return results."""
        log_function_call(self.logger, "_run", code=code)
        
        try:
            # Execute the code
            self.logger.debug("Executing code in REPL")
            sanitized_code = (CODE_SANITIZER_PROMPT | self.llm).invoke({"code": code}).model_dump()
            self.logger.debug(f"Sanitized code: {sanitized_code}")
            result = self.python_repl.run(sanitized_code['code'])
            
            # Check if the result contains a plot
            if "plt" in sanitized_code['code']:
                self.logger.debug("Detected matplotlib plot in code")
                # Handle matplotlib plots
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                plt.close()
                buffer.seek(0)
                image = base64.b64encode(buffer.read()).decode()
                  # Create HTML/Markdown for displaying the image
                img_markdown = f"\n\n<img src='data:image/png;base64,{image}' class='img-fluid'>\n\n"
                
                response = {
                    "type": "plot",
                    "data": image,
                    "result": result,
                    "display": img_markdown
                }
                log_function_result(self.logger, "_run", "Generated matplotlib plot")
                
            elif "go.Figure" in sanitized_code['code']:
                self.logger.debug("Detected plotly figure in code")
                # Handle plotly figures
                import plotly.io as pio
                # Get the last figure from plotly's figure factory
                from plotly import fig_tracker
                fig = fig_tracker.get_figure()
                if fig:
                    buffer = io.BytesIO()
                    pio.write_image(fig, buffer, format='png')
                    buffer.seek(0)
                    image = base64.b64encode(buffer.read()).decode()
                      # Create HTML/Markdown for displaying the image
                    img_markdown = f"\n\n<img src='data:image/png;base64,{image}' class='img-fluid'>\n\n"
                    
                    response = {
                        "type": "plot",
                        "data": image,
                        "result": result,
                        "display": img_markdown
                    }
                    log_function_result(self.logger, "_run", "Generated plotly plot")
            else:
                response = {
                    "type": "text",
                    "result": result,
                    "display": str(result)
                }

            response["query"] = sanitized_code['code']
            log_function_result(self.logger, "_run", response)
            return response
            
        except Exception as e:
            log_error(self.logger, e, "code execution")
            return {
                "type": "error",
                "error": str(e)
            }
        
class CalculatorTool(BaseTool):
    """Performs mathematical calculations like addition, subtraction, percentages etc."""
    
    name: str = "calculator"
    description: str = "Performs mathematical calculations like addition, subtraction, percentages etc."
    repl: CodeExecutorTool = Field(default_factory=lambda: CodeExecutorTool())
    logger: logging.Logger = Field(default_factory=lambda: setup_logger(f"{__name__}.CalculatorTool"))

    def __init__(self):
        super().__init__()
        self.repl = CodeExecutorTool()
        self.logger = setup_logger(f"{__name__}.CalculatorTool")

    def _run(self, query: str) -> Dict[str, float]:
        """Convert the user query that may be in natural language into a valid mathematical operation in Python, and use Python's eval to compute the result."""
        log_function_call(self.logger, "_run", query=query)
        try:
            # Convert the query into a valid Python expression
            self.logger.debug("Converting natural language to Python expression")
            self.logger.debug(f"Generated Python code: {query}")
            
            # Execute the Python code using the CodeExecutorTool
            self.logger.debug("Executing generated code")
            result = self.repl.run(query)
            
            response = {
                "type": "text",
                "result": result,
                "query": query
            }
            log_function_result(self.logger, "_run", response)
            return response
        except Exception as e:
            log_error(self.logger, e, "calculation")
            return {
                "type": "error",
                "error": f"Calculation failed: {str(e)}",
                "query": query
            }

class DocumentSummarizerTool(BaseTool):
    """Summarizes financial documents, reports, and articles"""
    
    name: str = "document_summarizer"
    description: str = "Summarizes financial documents, reports, and articles"
    llm: ChatOpenAI = Field(default_factory=lambda: ChatOpenAI(model="gpt-4o-mini", temperature=0))
    chain: Any = Field(default=None)  # Placeholder for the summarization chain
    logger: logging.Logger = Field(default_factory=lambda: setup_logger(f"{__name__}.DocumentSummarizerTool"))

    def __init__(self):
        super().__init__()
        self.logger = setup_logger(f"{__name__}.DocumentSummarizerTool")
        self.chain = DOCUMENT_SUMMARIZER_PROMPT | self.llm

    def _extract_text_from_response(self, response: requests.Response) -> str:
        """Extract text from different document types."""
        content_type = response.headers.get('content-type', '')
        mime_type = content_type.split(';')[0]
        log_function_call(self.logger, "_extract_text_from_response", mime_type=mime_type)

        if mime_type == 'application/pdf':
            self.logger.debug("Processing PDF document")
            # Handle PDF files
            pdf_file = io.BytesIO(response.content)
            pdf_reader = PdfReader(pdf_file)
            text = ' '.join(page.extract_text() for page in pdf_reader.pages)
            self.logger.debug(f"Extracted {len(pdf_reader.pages)} pages from PDF")
        elif mime_type == 'text/html':
            self.logger.debug("Processing HTML document")
            # Handle HTML content
            soup = BeautifulSoup(response.text, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
        else:
            self.logger.debug("Processing plain text document")
            # Handle plain text
            text = response.text

        log_function_result(self.logger, "_extract_text_from_response", f"Extracted {len(text)} characters")
        return text

    def _run(self, query: str, is_url: bool) -> Dict[str, Any]:
        """Summarize financial document or article."""
        log_function_call(self.logger, "_run", query=query)
        
        # First, try to find a relevant document via web search
        try:
            self.logger.debug(f"Found relevant document URL: {query}")

            self.logger.debug(f"Extracted URL: {query}")

            if not query:
                self.logger.warning("No relevant document found")
                return {"error": "No relevant document found."}
            
            if is_url:
                # Fetch document content from the URL
                self.logger.debug(f"Fetching content from URL: {query}")
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                response = requests.get(query, headers=headers, allow_redirects=True)
                response.raise_for_status()
                
                # Extract text based on document type
                text = self._extract_text_from_response(response)
            else:
                text = query
            
            # Summarize the extracted text
            self.logger.debug("Generating summary of extracted text")
            summary = self.chain.invoke({"content": text}).content
            
            result = {
                "type": "text",
                "query": query,
                "result": summary,
            }
            log_function_result(self.logger, "_run", result)
            return result
            
        except Exception as e:
            log_error(self.logger, e, "document summarization")
            return {"error": f"Document summarization failed: {str(e)}"}