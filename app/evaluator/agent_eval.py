import json
from pathlib import Path
from datetime import datetime
from app.evaluator.prompts import TASK_SUCCESS_PROMPT, TOOL_USE_PROMPT, COHERENCE_REASONING_PROMPT
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

EVAL_OUTPUT_FILE = Path(__file__).parent / "evaluation_results.jsonl"

class EvalOutput(BaseModel):
    score: str = Field(description="The score of the evaluation.")
    justification: str = Field(description="A brief explanation for why the score was given.")

class AgentEvaluator:
    """Evaluator for Agent outputs"""
    def __init__(self, model="gpt-4o-mini", temperature=0):
        self.llm = ChatOpenAI(model=model, temperature=temperature).with_structured_output(EvalOutput, method='json_schema', strict=True)

    def evaluate(self, metric: str, **kwargs):
        """Method to evaluate a single output based on a metric. Possible metrics are ['task_success', 'tool_use', 'coherence_reasoning']"""
        if metric == "task_success":
            prompt = TASK_SUCCESS_PROMPT.format(**kwargs)
        elif metric == "tool_use":
            prompt = TOOL_USE_PROMPT.format(**kwargs)
        elif metric == "coherence_reasoning":
            prompt = COHERENCE_REASONING_PROMPT.format(**kwargs)
        else:
            raise ValueError("Unknown metric")
        response = self.llm.invoke(prompt).model_dump()
        self.save_evaluation(response)
        return response

    def evaluate_all(self, query, output, tool=None, save=True, extra_info=None):
        results = {
            "timestamp": datetime.utcnow().isoformat(),
            "query": query,
            "output": output,
            "tool": tool or "N/A",
            "task_success": self.evaluate("task_success", query=query, output=output),
            "tool_use": self.evaluate("tool_use", query=query, tool=tool or "N/A", output=output),
            "coherence_reasoning": self.evaluate("coherence_reasoning", output=output),
        }
        if extra_info:
            results["extra_info"] = extra_info
        if save:
            self.save_evaluation(results)
        return results

    @staticmethod
    def save_evaluation(result):
        with open(EVAL_OUTPUT_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
