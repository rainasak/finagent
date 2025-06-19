# Prompts for agent evaluation metrics

TASK_SUCCESS_PROMPT = """
You are an expert evaluator. Given the user query and the agent's output, rate the output for Task Success on a scale of 1-5:
- 5: Fully and correctly solves the user's task.
- 4: Mostly solves the task, minor omissions.
- 3: Partially solves the task, some important gaps.
- 2: Attempts the task but is mostly incomplete or incorrect.
- 1: Fails to address the task.

User Query: {query}
Agent Output: {output}
Score (1-5) and a brief justification:
"""

TOOL_USE_PROMPT = """
You are an expert evaluator. Given the user query, the agent's tool choice, and the output, rate Correct Tool Use on a scale of 1-5:
- 5: Tool choice is optimal and used correctly.
- 4: Tool is appropriate, minor inefficiency.
- 3: Tool is reasonable but not optimal.
- 2: Tool is suboptimal or misused.
- 1: Tool is clearly wrong or not used when needed.

User Query: {query}
Tool Used: {tool}
Agent Output: {output}
Score (1-5) and a brief justification:
"""

COHERENCE_REASONING_PROMPT = """
You are an expert evaluator. Given the agent's output, rate Coherence and Reasoning on a scale of 1-5:
- 5: Output is logically sound, well-structured, and easy to follow.
- 4: Mostly coherent, minor lapses.
- 3: Somewhat coherent, some unclear reasoning.
- 2: Disjointed or confusing, major reasoning gaps.
- 1: Incoherent or nonsensical.

Agent Output: {output}
Score (1-5) and a brief justification:
"""
