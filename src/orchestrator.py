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

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph

from .logger import AgentLogger, get_logger, reset_logger
from .dsl_executor import execute_dsl
from .finqa_data import get_all_evidence_pieces, format_evidence_for_generator


# =========================================
# Module-level State
# =========================================

_tool_calls: list[str] = []  # Track tool calls for metrics
_current_entry: dict | None = None  # Current FinQA example being processed


def reset_tool_calls() -> None:
    """Reset tool call tracking for a new query."""
    global _tool_calls
    _tool_calls = []


def track_tool_call(tool_name: str) -> None:
    """Track a tool call."""
    global _tool_calls
    _tool_calls.append(tool_name)


def get_tool_calls() -> list[str]:
    """Get all tool calls for the current query."""
    return _tool_calls.copy()


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

SYSTEM_PROMPT = """You are a FinQA Agent - an expert at answering financial questions using numerical reasoning.

## Your Task
Answer questions about financial documents by:
1. Retrieving relevant evidence from the document
2. Generating a DSL (Domain Specific Language) program to compute the answer
3. Executing the DSL program to get the final numerical result

## Available Tools
You have access to these tools (call `list_tools` to see details):
- `list_tools`: See all available tools and their descriptions
- `retrieve_evidence`: Find relevant text and table data for your question
- `generate_dsl`: Create a DSL program based on the question and evidence
- `execute_dsl`: Run a DSL program to compute the numerical answer

## DSL Format
The DSL supports these operations:
- `add(a, b)`, `subtract(a, b)`, `multiply(a, b)`, `divide(a, b)`, `exp(a, b)`
- `greater(a, b)` - returns "yes" or "no"
- `table_max(row)`, `table_min(row)`, `table_sum(row)`, `table_average(row)`
- Use `#N` to reference the result of step N (e.g., `divide(#0, 100)`)

Example: `subtract(100, 50), divide(#0, 2)` → Result: 25

## Strategy
You are FREE to use tools in any order and as many times as needed. Common strategies:
1. **Retrieve → Generate → Execute**: Standard flow
2. **Execute directly**: If you already know the program
3. **Iterate**: If first attempt fails, try different evidence or program

## IMPORTANT RULES
- ALWAYS provide a FINAL ANSWER before stopping
- After execute_dsl returns a result, give your answer immediately
- If execution fails, try to fix the program or generate a new one
- Maximum 6-8 tool calls, then you MUST give your best answer
- BE CONCISE - do not write long explanations

## Output Format
When you have the answer, respond with:

**FINAL ANSWER: [your numerical answer]**

Keep answers SHORT (e.g., "25.5", "-12.3%", "yes", "no").
"""


# =========================================
# Tools
# =========================================

def create_tools(logger: AgentLogger) -> list:
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
        logger.log_tool_call("list_tools", {})
        
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
        
        logger.log_tool_result("list_tools", tools_info, 0)
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
        from .finqa_tools import get_retriever
        
        track_tool_call("retrieve_evidence")
        logger.log_tool_call("retrieve_evidence", {"top_k": top_k})
        
        start = time.time()
        
        entry = get_current_entry()
        if entry is None:
            result = {"error": "No document loaded"}
            logger.log_tool_result("retrieve_evidence", result, 0)
            return result
        
        # Get all evidence pieces from the document
        evidence_dict = get_all_evidence_pieces(entry["full_entry"])
        question = entry["question"]
        
        # Use the retriever to score and rank
        retriever = get_retriever()
        ranked_evidence = retriever.retrieve(question, evidence_dict, top_k=top_k)
        
        elapsed_ms = (time.time() - start) * 1000
        
        result = {
            "question": question,
            "evidence_count": len(evidence_dict),
            "retrieved": ranked_evidence,
        }
        
        logger.log_tool_result("retrieve_evidence", result, elapsed_ms)
        return result
    
    @tool
    def generate_dsl(evidence: str) -> dict[str, Any]:
        """
        Generate a DSL program to answer the question.
        
        Uses a fine-tuned Qwen2-5-Coder-3B model trained on FinQA
        to generate a Domain Specific Language program.
        
        Args:
            evidence: The evidence context (from retrieve_evidence or your summary)
        
        Returns:
            Dict with the generated DSL program
        """
        from .finqa_tools import get_generator
        
        track_tool_call("generate_dsl")
        logger.log_tool_call("generate_dsl", {"evidence_length": len(evidence)})
        
        start = time.time()
        
        entry = get_current_entry()
        if entry is None:
            result = {"error": "No document loaded"}
            logger.log_tool_result("generate_dsl", result, 0)
            return result
        
        question = entry["question"]
        
        # Generate DSL program
        generator = get_generator()
        gen_result = generator.generate(question, evidence)
        
        elapsed_ms = (time.time() - start) * 1000
        
        result = {
            "question": question,
            "program": gen_result["program"],
            "prompt_tokens": gen_result["prompt_length"],
            "generated_tokens": gen_result["generated_length"],
        }
        
        logger.log_tool_result("generate_dsl", result, elapsed_ms)
        return result
    
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
        logger.log_tool_call("execute_dsl", {"program": program})
        
        start = time.time()
        
        entry = get_current_entry()
        table = entry.get("table") if entry else None
        
        # Execute the DSL
        result = execute_dsl(program, table)
        
        elapsed_ms = (time.time() - start) * 1000
        
        logger.log_tool_result("execute_dsl", result, elapsed_ms)
        return result
    
    return [list_tools, retrieve_evidence, generate_dsl, execute_dsl_tool]


# =========================================
# Graph Nodes
# =========================================

def format_message_for_log(msg: BaseMessage) -> dict[str, Any]:
    """Format a LangChain message for logging."""
    result = {
        "type": type(msg).__name__,
        "content": str(msg.content)[:500] if hasattr(msg, "content") else str(msg)[:500],
    }
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        result["tool_calls"] = msg.tool_calls
    if hasattr(msg, "tool_call_id"):
        result["tool_call_id"] = msg.tool_call_id
    return result


def create_agent_graph(
    logger: AgentLogger,
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
    tools = create_tools(logger)
    tools_by_name = {t.name: t for t in tools}
    llm_with_tools = llm.bind_tools(tools)
    
    def agent_node(state: AgentState) -> dict:
        """The main agent node - calls the LLM."""
        messages_for_log = [format_message_for_log(m) for m in state["messages"]]
        logger.log_agent_prompt(
            f"Sending {len(state['messages'])} messages to LLM (tool calls: {state['tool_call_count']})",
            full_messages=messages_for_log
        )
        
        response = llm_with_tools.invoke(state["messages"])
        
        response_text = str(response.content) if hasattr(response, 'content') else ""
        tool_calls = response.tool_calls if hasattr(response, 'tool_calls') else []
        
        logger.log_agent_response(
            response_text,
            raw_response={"content": response_text, "tool_calls": tool_calls}
        )
        
        if tool_calls:
            for tc in tool_calls:
                logger.log_plan({
                    "action": "TOOL_CALL",
                    "tool": tc["name"],
                    "args": tc["args"],
                })
        else:
            logger.log_plan({
                "action": "FINAL_RESPONSE",
                "content_preview": response_text[:200],
            })
        
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
                        logger.log_error(f"Tool {tool_name} failed", str(e))
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
            logger.log_decision("LAST_CHANCE", f"Tool call limit ({state['max_tool_calls']}) approaching")
        
        return {
            "messages": messages + new_messages,
            "tool_call_count": new_tool_count,
        }
    
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Decide whether to continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if state["tool_call_count"] >= state["max_tool_calls"]:
            logger.log_decision("STOPPING", f"Max tool calls ({state['max_tool_calls']}) reached")
            return "end"
        
        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            logger.log_decision("CONTINUE_TO_TOOLS", f"LLM made {len(last_message.tool_calls)} tool call(s)")
            return "tools"
        
        logger.log_decision("ENDING", "LLM provided final answer")
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
    ):
        self.model = model
        self.output_dir = output_dir
        self.max_tool_calls = max_tool_calls
        self.logger: AgentLogger | None = None
        self.graph = None
    
    def run(self, query_id: str, data: list[dict], session_id: str | None = None) -> dict[str, Any]:
        """
        Run the QA agent for a query.
        
        Args:
            query_id: The FinQA query ID (index into data)
            data: Loaded FinQA dataset
            session_id: Optional session ID for logging
        
        Returns:
            Dict with agent_answer, golden_answer, and metrics
        """
        from .finqa_data import get_question_data
        
        reset_logger()
        reset_tool_calls()
        
        # Get question data
        q_data = get_question_data(data, query_id)
        if q_data is None:
            return {"error": f"Query ID '{query_id}' not found"}
        
        set_current_entry(q_data)
        
        question = q_data["question"]
        golden_answer = q_data["answer"]
        
        # Create logger
        log_session = session_id or f"query_{query_id}"
        log_dir = f"{self.output_dir}/{log_session}"
        self.logger = get_logger(output_dir=log_dir, session_id=log_session)
        
        # Log configuration
        self.logger.log_data("Configuration", {
            "model": self.model,
            "max_tool_calls": self.max_tool_calls,
            "query_id": query_id,
        })
        
        self.logger.log_data("Question & Answer", {
            "query_id": query_id,
            "question": question,
            "golden_answer": golden_answer,
            "golden_program": q_data.get("program", ""),
        })
        
        self.logger.log_system_prompt(SYSTEM_PROMPT)
        
        # Create graph
        self.graph = create_agent_graph(
            logger=self.logger,
            model=self.model,
        )
        
        # User message
        user_message = question
        self.logger.log_user_message(user_message)
        
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
            self.logger.log_error("Agent execution failed", str(e))
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
        
        # Save traces
        self.logger.log_message_trace(final_state["messages"])
        self.logger.save_messages(final_state["messages"])
        
        # Get tool calls
        tool_calls = get_tool_calls()
        
        # Build result
        result = {
            "query_id": query_id,
            "question": question,
            "agent_answer": extracted_answer,
            "golden_answer": golden_answer,
            "full_response": agent_answer,
            "success": success,
            "error": error,
            "elapsed_seconds": round(elapsed, 2),
            "tool_calls": tool_calls,
            "tool_call_count": len(tool_calls),
            "message_count": len(final_state["messages"]),
        }
        
        # Log result
        self.logger.log_data("ANSWER COMPARISON", {
            "question": question,
            "agent_answer": extracted_answer,
            "golden_answer": golden_answer,
        })
        
        self.logger.log_success("Agent completed", result)
        report_path = self.logger.save_final_report(result)
        result["log_dir"] = str(self.logger.get_log_dir())
        result["report_path"] = report_path
        
        return result
    
    def _extract_answer(self, agent_answer: str, messages: list) -> str:
        """Extract the final answer from agent response."""
        extracted = ""
        
        # Method 1: Look for "FINAL ANSWER:" format
        if "FINAL ANSWER:" in agent_answer.upper():
            idx = agent_answer.upper().find("FINAL ANSWER:") + len("FINAL ANSWER:")
            extracted = agent_answer[idx:].strip()
            extracted = extracted.split("\n")[0].strip()
            # Remove trailing ** if present
            extracted = extracted.rstrip("*").strip()
        
        # Method 2: Look for \boxed{...} LaTeX format
        if not extracted:
            boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', agent_answer)
            if boxed_matches:
                extracted = boxed_matches[-1].strip()
        
        # Method 3: Look for bold pattern **answer**
        if not extracted and "**" in agent_answer:
            bold_matches = re.findall(r'\*\*([^*]+)\*\*', agent_answer)
            if bold_matches:
                # Filter out non-numeric answers that look like headers
                for match in reversed(bold_matches):
                    match = match.strip()
                    # Skip if it looks like a header/label
                    if match.lower() in ["final answer", "answer", "result"]:
                        continue
                    # Check if it looks numeric or is yes/no
                    if re.match(r'^-?[\d.,]+%?$', match) or match.lower() in ["yes", "no"]:
                        extracted = match
                        break
        
        # Method 4: Search all messages for FINAL ANSWER or boxed
        if not extracted:
            for msg in reversed(messages):
                if hasattr(msg, "content") and msg.content:
                    content = str(msg.content)
                    if not content.startswith("{") and not content.startswith("["):
                        # Try FINAL ANSWER
                        if "FINAL ANSWER:" in content.upper():
                            idx = content.upper().find("FINAL ANSWER:") + len("FINAL ANSWER:")
                            extracted = content[idx:].strip().split("\n")[0]
                            extracted = extracted.rstrip("*").strip()
                            break
                        # Try boxed
                        boxed_matches = re.findall(r'\\boxed\{([^}]+)\}', content)
                        if boxed_matches:
                            extracted = boxed_matches[-1].strip()
                            break
        
        return extracted


def main():
    """Run the FinQA agent on a single query."""
    import argparse
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    
    from .finqa_data import load_finqa_split
    
    parser = argparse.ArgumentParser(description="Run the FinQA agent")
    parser.add_argument("--query-id", default="0", help="Query ID to process")
    parser.add_argument("--model", default="qwen3:4b", help="Ollama model name")
    parser.add_argument("--max-tool-calls", type=int, default=10, help="Max tool calls")
    parser.add_argument("--output-dir", default="__output__", help="Output directory")
    parser.add_argument("--data-path", default="FinQA/dataset/dev.json", help="Path to FinQA data")
    
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
