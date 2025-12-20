"""
LangGraph Orchestrator for FinQA Agent.

Implements a ReAct-style agent that orchestrates:
1. Evidence retrieval (CrossEncoder reranker)
2. DSL program generation (Qwen2-5-Coder-3B)
3. DSL execution

The agent is FULLY FLEXIBLE - it decides which tools to use and when.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Literal, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from .dsl.executor import execute_dsl
from .preprocess import get_all_evidence_pieces


DEFAULT_RETRIEVER_PATH = "/home/pg2860/hpml-run/reranker-sweep/jjyfsr7j/checkpoint-1200"
DEFAULT_GENERATOR_PATH = "/home/pg2860/hpml/__output__/qwen2-5-coder-3b-instruct-full-input_gold-output_program/final_model"


# =========================================
# Module-level State
# =========================================

_tool_calls: list[str] = []  # Track tool calls for metrics
_current_entry: dict | None = None  # Current FinQA example being processed
_generated_program: str | None = None  # Track the last generated DSL program


def reset_tool_calls() -> None:
    """Reset tool call tracking for a new query."""
    global _tool_calls, _generated_program
    _tool_calls = []
    _generated_program = None


def track_tool_call(tool_name: str) -> None:
    """Track a tool call."""
    global _tool_calls
    _tool_calls.append(tool_name)


def get_tool_calls() -> list[str]:
    """Get all tool calls for the current query."""
    return _tool_calls.copy()


def set_generated_program(program: str) -> None:
    """Track the generated DSL program."""
    global _generated_program
    _generated_program = program


def get_generated_program() -> str | None:
    """Get the last generated DSL program."""
    return _generated_program


def set_current_entry(entry: dict) -> None:
    """Set the current FinQA entry being processed."""
    global _current_entry
    _current_entry = entry


def get_current_entry() -> dict | None:
    """Get the current FinQA entry."""
    return _current_entry


# =========================================
# Agent State
# =========================================

class AgentState(TypedDict):
    """State passed between nodes in the LangGraph."""
    messages: list[BaseMessage]
    query_id: str
    question: str
    golden_answer: str
    tool_call_count: int
    max_tool_calls: int


# =========================================
# System Prompt
# =========================================

SYSTEM_PROMPT = """You are a FinQA Agent that generates DSL programs for financial reasoning.

## Available Tools
- `retrieve_evidence`: Find relevant evidence from the document
- `generate_dsl`: Generate a DSL program from evidence  
- `execute_dsl`: Execute a DSL program to test if it works

## Workflow
1. Call `retrieve_evidence` to find relevant numbers
2. Then, always call `generate_dsl` with the evidence - save the returned program
3. (Optional) Call `execute_dsl` with the EXACT program from step 2
4. Output the program from `generate_dsl` as your final answer

## CRITICAL RULES
- When calling `execute_dsl`, pass the EXACT program from `generate_dsl` - do NOT modify it
- Do NOT invent your own DSL programs
- Do NOT simplify or rewrite programs
- Your final answer must be the program returned by `generate_dsl`

## When to Retry
ONLY retry if `execute_dsl` returns an execution ERROR:
- Call `retrieve_evidence` with different parameters
- Call `generate_dsl` again
- Output the new program

## Output Format
**FINAL ANSWER: [exact program from generate_dsl]**

Example: If generate_dsl returns "subtract(11503, 10815)", output:
**FINAL ANSWER: subtract(11503, 10815)**
"""



# =========================================
# Tools
# =========================================

def create_tools() -> list:
    """Create the tools for the FinQA agent."""
    
    @tool
    def list_tools() -> dict[str, Any]:
        """
        List all available tools and their descriptions.
        
        Call this first to understand your options.
        
        Returns:
            Dictionary with tool names and descriptions
        """
        track_tool_call("list_tools")

        tools_info = {
            "tools": [
                {
                    "name": "retrieve_evidence",
                    "description": "Retrieve relevant evidence (text and table data) from the financial document.",
                    "params": "top_k: int = 5 (number of evidence pieces to retrieve)",
                    "returns": "List of evidence pieces with relevance scores",
                },
                {
                    "name": "generate_dsl",
                    "description": "Generate a DSL program to answer the question based on retrieved evidence.",
                    "params": "evidence: str (the evidence context to use)",
                    "returns": "Generated DSL program string",
                },
                {
                    "name": "execute_dsl",
                    "description": "Execute a DSL program and return the computed result.",
                    "params": "program: str (the DSL program to execute)",
                    "returns": "Execution result (number or yes/no) or error message",
                },
            ],
            "recommendation": "Typical flow: retrieve_evidence → generate_dsl → execute_dsl. But you can adapt based on results!",
        }

        return tools_info
    
    @tool
    def retrieve_evidence(top_k: int = 5) -> dict[str, Any]:
        """
        Retrieve relevant evidence from the financial document.
        
        Uses a trained CrossEncoder model to score and rank all text 
        and table data against your question.
        
        Args:
            top_k: Number of top evidence pieces to retrieve (default: 5)
        
        Returns:
            Dict with retrieved evidence pieces and their scores
        """
        from .retriever.run import get_retriever

        track_tool_call("retrieve_evidence")

        entry = get_current_entry()
        if entry is None:
            return {"error": "No document loaded"}
        
        # Get all evidence pieces from the document
        evidence_dict = get_all_evidence_pieces(entry["full_entry"])
        question = entry["question"]
        
        # Use the retriever to score and rank
        retriever = get_retriever()
        ranked_evidence = retriever.retrieve(question, evidence_dict, top_k=top_k)

        return {
            "question": question,
            "evidence_count": len(evidence_dict),
            "retrieved": ranked_evidence,
        }
    
    @tool
    def generate_dsl(evidence: str) -> dict[str, Any]:
        """
        Generate a DSL program to answer the question.
        
        Uses a fine-tuned Qwen2-5-Coder-3B model trained on FinQA
        to generate a Domain Specific Language program.
        
        Args:
            evidence: The evidence context. Can be:
                      - Actual text evidence
                      - Evidence IDs like "pre_1, post_0, table_2" (will be resolved)
        
        Returns:
            Dict with the generated DSL program
        """
        from .dsl.run import get_generator

        track_tool_call("generate_dsl")

        entry = get_current_entry()
        if entry is None:
            return {"error": "No document loaded"}
        
        question = entry["question"]
        
        # Resolve evidence IDs to actual text if needed
        resolved_evidence = _resolve_evidence_ids(evidence, entry)
        
        # Generate DSL program
        generator = get_generator()
        gen_result = generator.generate(question, resolved_evidence)
        
        # Track the generated program for evaluation
        set_generated_program(gen_result["program"])

        return {
            "question": question,
            "program": gen_result["program"],
            "prompt_tokens": gen_result["prompt_length"],
            "generated_tokens": gen_result["generated_length"],
        }
    
    def _resolve_evidence_ids(evidence: str, entry: dict) -> str:
        """
        Resolve evidence IDs to actual text content.
        
        If evidence looks like "pre_1, post_0, table_2", resolve these to actual text.
        If evidence is already text (contains sentences), return as-is.
        """
        import re
        
        # Check if this looks like a list of IDs (short, contains underscores, no sentences)
        evidence_clean = evidence.strip()
        
        # If it contains long text (sentences), assume it's already resolved
        if len(evidence_clean) > 100 or ". " in evidence_clean:
            return evidence
        
        # Parse evidence IDs
        id_pattern = r'(pre_\d+|post_\d+|table_\d+)'
        ids = re.findall(id_pattern, evidence_clean)
        
        if not ids:
            return evidence  # Not IDs, return as-is
        
        # Build lookup for entry content
        resolved_parts = []
        
        pre_text = entry.get("pre_text", [])
        post_text = entry.get("post_text", [])
        table = entry.get("table", [])
        header = table[0] if table else []
        
        for eid in ids:
            if eid.startswith("pre_"):
                idx = int(eid.split("_")[1])
                if 0 <= idx < len(pre_text):
                    resolved_parts.append(pre_text[idx])
            elif eid.startswith("post_"):
                idx = int(eid.split("_")[1])
                if 0 <= idx < len(post_text):
                    resolved_parts.append(post_text[idx])
            elif eid.startswith("table_"):
                idx = int(eid.split("_")[1])
                # Generator trained with input_gold expects natural language format
                # Format: "the {row_name} of {column} is {value} ;"
                if 1 <= idx < len(table) and header:
                    row = table[idx]
                    formatted_parts = []
                    row_name = row[0].strip() if row else ""
                    if header[0].strip():
                        formatted_parts.append(header[0].strip())
                    for col_idx in range(1, min(len(header), len(row))):
                        col_name = header[col_idx].strip()
                        value = row[col_idx].strip()
                        if col_name and value:
                            formatted_parts.append(f"the {row_name} of {col_name} is {value} ;")
                    if formatted_parts:
                        resolved_parts.append(" ".join(formatted_parts))
        
        if resolved_parts:
            return " ".join(resolved_parts)
        
        return evidence  # Fallback
    
    @tool
    def execute_dsl_tool(program: str) -> dict[str, Any]:
        """
        Execute a DSL program and return the result.
        
        Parses and runs the FinQA DSL program. Supports operations like
        add, subtract, multiply, divide, and table aggregations.
        
        Args:
            program: The DSL program string to execute
                     Example: "subtract(100, 50), divide(#0, 2)"
        
        Returns:
            Dict with execution result or error message
        """
        track_tool_call("execute_dsl")

        entry = get_current_entry()
        table = entry.get("table") if entry else None

        # Execute the DSL
        return execute_dsl(program, table)
    
    return [list_tools, retrieve_evidence, generate_dsl, execute_dsl_tool]


# =========================================
# Graph Nodes
# =========================================

def _parse_json_tool_call(content: str) -> list[dict] | None:
    """
    Parse JSON tool calls from text content.

    Some models (like qwen3:4b with long prompts) output tool calls as JSON text
    instead of using the proper tool calling format. This function detects and
    parses those cases.
    """
    if not content:
        return None

    content = content.strip()

    # Try parsing as direct JSON with name/arguments structure
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "name" in parsed and "arguments" in parsed:
            return [{
                "name": parsed["name"],
                "args": parsed["arguments"],
                "id": parsed.get("id", parsed["name"]),
                "type": "tool_call",
            }]
    except json.JSONDecodeError:
        pass

    return None


def create_agent_graph(
    model: str = "qwen3:4b",
) -> StateGraph:
    """Create the LangGraph agent."""

    # Initialize LLM with thinking disabled for Qwen3 models
    # This significantly speeds up inference by avoiding extended reasoning
    extra_kwargs = {"think": False} if "qwen3" in model.lower() else {}
    llm = ChatOllama(
        model=model,
        temperature=0,
        **extra_kwargs,
    )

    # Create tools and bind to LLM
    tools = create_tools()
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        """The main agent node - calls the LLM."""
        response = llm_with_tools.invoke(state["messages"])

        # Fallback: if model output JSON tool call as text, parse and inject it
        has_tool_calls = hasattr(response, "tool_calls") and response.tool_calls
        if not has_tool_calls and hasattr(response, "content") and response.content:
            parsed_calls = _parse_json_tool_call(str(response.content))
            if parsed_calls:
                response.tool_calls = parsed_calls

        return {"messages": state["messages"] + [response]}
    
    def tool_node(state: AgentState) -> dict:
        """Execute tools called by the agent."""
        messages = state["messages"]
        last_message = messages[-1]
        
        new_messages = []
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            for tc in last_message.tool_calls:
                tool_name = tc["name"]
                tool_args = tc["args"]
                tool_id = tc.get("id", tool_name)
                
                # Map the tool name (execute_dsl_tool -> execute_dsl for display)
                actual_tool_name = tool_name
                if tool_name == "execute_dsl_tool":
                    actual_tool_name = "execute_dsl_tool"  # Keep as-is for lookup
                
                if actual_tool_name in tools_by_name:
                    try:
                        result = tools_by_name[actual_tool_name].invoke(tool_args)
                        result_str = json.dumps(result, default=str) if isinstance(result, dict) else str(result)
                    except Exception as e:
                        result_str = json.dumps({"error": str(e)})
                else:
                    result_str = json.dumps({"error": f"Unknown tool: {tool_name}"})
                
                new_messages.append(ToolMessage(content=result_str, tool_call_id=tool_id))
        
        # Track tool call count
        num_new_calls = len(last_message.tool_calls) if hasattr(last_message, "tool_calls") and last_message.tool_calls else 0
        new_tool_count = state["tool_call_count"] + num_new_calls
        
        # Inject warning if approaching limit
        if new_tool_count >= state["max_tool_calls"] - 1:
            warning_msg = HumanMessage(content=(
                "⚠️ IMPORTANT: This is your LAST chance to answer. "
                "You must provide a FINAL ANSWER now based on what you found. "
                "Do NOT make any more tool calls. Give your best answer immediately.\n\n"
                "**FINAL ANSWER: [your answer here]**"
            ))
            new_messages.append(warning_msg)
        
        return {
            "messages": messages + new_messages,
            "tool_call_count": new_tool_count,
        }
    
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Decide whether to continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]

        if state["tool_call_count"] >= state["max_tool_calls"]:
            return "end"

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            return "tools"

        return "end"
    
    # Build graph
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


# =========================================
# Orchestrator Class
# =========================================

class FinQAOrchestrator:
    """
    FinQA Question Answering Orchestrator.
    
    Usage:
        orchestrator = FinQAOrchestrator(model="qwen3:4b")
        result = orchestrator.run(query_id="0", data=finqa_data)
    """
    
    def __init__(
        self,
        model: str = "qwen3:4b",
        output_dir: str = "__output__",
        max_tool_calls: int = 10,
        generator_path: str | None = None,
    ):
        self.model = model
        self.output_dir = output_dir
        self.max_tool_calls = max_tool_calls
        self.generator_path = generator_path
        self.graph = None
        
        # Initialize generator with custom path if provided
        if generator_path:
            from .dsl.run import get_generator
            get_generator(generator_path)
    
    def run(self, query_id: str, data: list[dict]) -> dict[str, Any]:
        """
        Run the QA agent for a query.

        Args:
            query_id: The FinQA query ID (index into data)
            data: Loaded FinQA dataset

        Returns:
            Dict with agent_answer, golden_answer, and metrics
        """
        from .preprocess import get_question_data
        
        reset_tool_calls()
        
        # Get question data
        q_data = get_question_data(data, query_id)
        if q_data is None:
            return {"error": f"Query ID '{query_id}' not found"}
        
        set_current_entry(q_data)
        
        question = q_data["question"]
        golden_answer = q_data["answer"]

        # Create graph
        self.graph = create_agent_graph(model=self.model)

        # Build user message with full context
        from .preprocess import build_context
        context = build_context(q_data["full_entry"], mode="all")
        user_message = f"## Question\n{question}\n\n## Document\n{context}"
        
        # Initial state
        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ],
            "query_id": query_id,
            "question": question,
            "golden_answer": golden_answer,
            "tool_call_count": 0,
            "max_tool_calls": self.max_tool_calls,
        }
        
        # Run
        start_time = time.time()
        config = {"recursion_limit": 100}
        
        try:
            final_state = self.graph.invoke(initial_state, config=config)
            success = True
            error = None
        except Exception as e:
            final_state = initial_state
            success = False
            error = str(e)
        
        elapsed = time.time() - start_time
        
        # Extract answer
        agent_answer = ""
        if final_state["messages"]:
            last_msg = final_state["messages"][-1]
            if hasattr(last_msg, "content"):
                agent_answer = str(last_msg.content)
        
        # Try to extract just the answer
        extracted_answer = self._extract_answer(agent_answer, final_state["messages"])
        
        # Get tool calls and generated program
        tool_calls = get_tool_calls()
        generated_program = get_generated_program()
        
        # Evaluate GENERATOR tool accuracy (raw tool output)
        from .dsl.executor import evaluate_program_prediction
        generator_eval = {}
        if generated_program and q_data.get("program"):
            generator_eval = evaluate_program_prediction(
                prediction=generated_program,
                gold_program=q_data["program"],
                gold_numerical=q_data.get("exe_ans"),
                table=q_data.get("table"),
            )
        
        # Evaluate AGENT final answer accuracy (what agent outputs)
        agent_eval = {}
        if extracted_answer and q_data.get("program"):
            agent_eval = evaluate_program_prediction(
                prediction=extracted_answer,
                gold_program=q_data["program"],
                gold_numerical=q_data.get("exe_ans"),
                table=q_data.get("table"),
            )
        
        # Build result
        return {
            "query_id": query_id,
            "question": question,
            "agent_answer": extracted_answer,
            "golden_answer": golden_answer,
            "full_response": agent_answer,
            "generated_program": generated_program,
            "golden_program": q_data.get("program", ""),
            # Generator tool metrics (combined: symbolic OR execution match)
            "generator_correct": generator_eval.get("correct", False),
            # Agent final answer metrics (combined: symbolic OR execution match)
            "agent_correct": agent_eval.get("correct", False),
            "program_accuracy": generator_eval.get("program_accuracy", False),
            "execution_accuracy": generator_eval.get("execution_accuracy", False),
            "predicted_numerical": generator_eval.get("predicted_numerical"),
            "success": success,
            "error": error or generator_eval.get("execution_error"),
            "elapsed_seconds": round(elapsed, 2),
            "tool_calls": tool_calls,
            "tool_call_count": len(tool_calls),
            "message_count": len(final_state["messages"]),
        }
    
    def _extract_answer(self, agent_answer: str, messages: list) -> str:
        """Extract the DSL program from agent response."""
        extracted = ""
        
        # Method 1: Look for "FINAL ANSWER:" format
        if "FINAL ANSWER:" in agent_answer.upper():
            idx = agent_answer.upper().find("FINAL ANSWER:") + len("FINAL ANSWER:")
            extracted = agent_answer[idx:].strip()
            extracted = extracted.split("\n")[0].strip()
            # Remove trailing ** if present
            extracted = extracted.rstrip("*").strip()
        
        # Method 2: Look for DSL pattern in bold **program**
        if not extracted and "**" in agent_answer:
            bold_matches = re.findall(r'\*\*([^*]+)\*\*', agent_answer)
            for match in reversed(bold_matches):
                match = match.strip()
                # Skip headers
                if match.lower() in ["final answer", "answer", "result", "dsl program"]:
                    continue
                # Check if it looks like a DSL program (contains operation names)
                if any(op in match.lower() for op in ["add(", "subtract(", "multiply(", "divide(", "exp(", "greater(", "table_"]):
                    extracted = match
                    break
        
        # Method 3: Look for DSL pattern in code blocks
        if not extracted:
            code_matches = re.findall(r'`([^`]+)`', agent_answer)
            for match in reversed(code_matches):
                if any(op in match.lower() for op in ["add(", "subtract(", "multiply(", "divide(", "exp(", "greater(", "table_"]):
                    extracted = match
                    break
        
        # Method 4: Use the tracked generated_program if nothing found in response
        if not extracted:
            generated = get_generated_program()
            if generated:
                extracted = generated
        
        # Method 5: Search all messages for FINAL ANSWER
        if not extracted:
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    content = str(msg.content)
                    if not content.startswith("{") and not content.startswith("["):
                        if "FINAL ANSWER:" in content.upper():
                            idx = content.upper().find("FINAL ANSWER:") + len("FINAL ANSWER:")
                            extracted = content[idx:].strip().split("\n")[0]
                            extracted = extracted.rstrip("*").strip()
                            break
        
        # Clean up extracted answer
        extracted = extracted.strip()
        extracted = extracted.lstrip("*").rstrip("*").strip()  # Remove markdown bold
        
        # If agent expanded #0 references (common mistake), prefer the original generated_program
        # e.g. "divide(subtract(A, B), C)" is an expanded version of "divide(#0, C)"
        generated = get_generated_program()
        if generated and "#" in generated and "#" not in extracted:
            # Agent likely expanded the references - use the compact version
            # Check if the operations match (same operators used)
            gen_ops = set(re.findall(r'(add|subtract|multiply|divide|exp|greater|table_\w+)', generated.lower()))
            ext_ops = set(re.findall(r'(add|subtract|multiply|divide|exp|greater|table_\w+)', extracted.lower()))
            if gen_ops == ext_ops:
                extracted = generated
        
        return extracted


def main():
    """Run the FinQA agent on a single query."""
    import argparse
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    
    from .preprocess import load_finqa_split
    
    parser = argparse.ArgumentParser(description="Run the FinQA agent")
    parser.add_argument("--query-id", default="0", help="Query ID to process")
    parser.add_argument("--model", default="qwen3:4b", help="Ollama model name")
    parser.add_argument("--max-tool-calls", type=int, default=10, help="Max tool calls")
    parser.add_argument("--output-dir", default="__output__", help="Output directory")
    parser.add_argument("--data-path", default="./data/dev.json", help="Path to FinQA data")
    
    args = parser.parse_args()
    
    console = Console()
    
    # Load data
    console.print(f"[dim]Loading data from {args.data_path}...[/dim]")
    data = load_finqa_split(args.data_path)
    console.print(f"[dim]Loaded {len(data)} examples[/dim]")
    
    # Run agent
    orchestrator = FinQAOrchestrator(
        model=args.model,
        output_dir=args.output_dir,
        max_tool_calls=args.max_tool_calls,
    )
    
    result = orchestrator.run(query_id=args.query_id, data=data)
    
    # Print results
    console.print()
    console.print("=" * 80)
    console.print()
    
    console.print(Panel(
        result.get("question", "N/A"),
        title="❓ QUESTION",
        border_style="blue",
    ))
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("", style="bold")
    table.add_column("Answer")
    table.add_row("🤖 Agent", result.get("agent_answer", "N/A"))
    table.add_row("✅ Golden", result.get("golden_answer", "N/A"))
    
    console.print(table)
    console.print()
    
    console.print(f"[dim]Time: {result.get('elapsed_seconds', '?')}s | Tool Calls: {result.get('tool_call_count', '?')} | Logs: {result.get('log_dir', '?')}[/dim]")


if __name__ == "__main__":
    main()
