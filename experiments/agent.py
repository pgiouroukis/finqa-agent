"""
FinQA Agent Orchestrator (COMSE6998 / HPML)

This file implements a ReAct-style agent for the FinQA task.
The agent dynamically orchestrates tools for:

1. Evidence retrieval
2. DSL program generation
3. DSL execution

The focus of this file is AGENT ORCHESTRATION, not tool implementation.
"""

from typing import TypedDict, List, Literal

from langgraph.graph import StateGraph, END
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


# ======================================================
# Agent State
# ======================================================

class AgentState(TypedDict):
    """
    State passed between nodes in the LangGraph.
    The agent reasons entirely over message history.
    """
    messages: List[BaseMessage]


# ======================================================
# Tools (Wrappers around teammate implementations)
# ======================================================
# NOTE: These are thin wrappers.
# Replace the internals with real implementations from:
# - Shiv: retriever
# - Petros: DSL generator
# - Karina: DSL executor


@tool
def retrieve_evidence(question: str) -> str:
    """
    Retrieve relevant financial evidence (tables / text) for a FinQA question.
    (Shiv's retriever)
    """
    # Placeholder – replace with actual retriever call
    return "[Retrieved financial tables and text evidence]"


@tool
def generate_dsl(question: str, evidence: str) -> str:
    """
    Generate a FinQA DSL program based on the question and evidence.
    (Petros's DSL generator)
    """
    # Placeholder – replace with actual DSL generation
    return "subtract(revenue, cost)"


@tool
def execute_dsl(dsl_program: str) -> str:
    """
    Execute the DSL program and return the numeric result.
    (Karina's DSL executor)
    """
    # Placeholder – replace with real DSL execution
    return "123.45"


TOOLS = [
    retrieve_evidence,
    generate_dsl,
    execute_dsl,
]


# ======================================================
# System Prompt (FinQA-specific)
# ======================================================

SYSTEM_PROMPT = """
You are a FinQA agent.

Your goal is to answer financial questions by using tools.

Follow this strategy:
1. Retrieve relevant financial evidence (tables or text).
2. Generate a DSL program based on the evidence.
3. Execute the DSL program to obtain a numeric answer.

IMPORTANT RULES:
- Use tools when needed.
- Do NOT guess or hallucinate numbers.
- The final answer MUST come from executing the DSL.
- When you have the numeric answer, stop and respond.

Return only the final numeric answer.
"""


# ======================================================
# Agent Node (Decision Making)
# ======================================================

def agent_node(state: AgentState) -> AgentState:
    """
    The LLM reasoning node.
    Decides whether to call a tool or provide a final answer.
    """
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
    )

    llm_with_tools = llm.bind_tools(TOOLS)

    response = llm_with_tools.invoke(state["messages"])

    return {
        "messages": state["messages"] + [response]
    }


# ======================================================
# Tool Node (Action + Observation)
# ======================================================

def tool_node(state: AgentState) -> AgentState:
    """
    Executes tools selected by the LLM and returns observations
    back into the agent's message history.
    """
    last_msg = state["messages"][-1]
    new_messages: List[BaseMessage] = []

    if hasattr(last_msg, "tool_calls"):
        for tc in last_msg.tool_calls:
            tool_name = tc["name"]
            tool_args = tc["args"]

            if tool_name == "retrieve_evidence":
                result = retrieve_evidence.invoke(tool_args)

            elif tool_name == "generate_dsl":
                result = generate_dsl.invoke(tool_args)

            elif tool_name == "execute_dsl":
                result = execute_dsl.invoke(tool_args)

            else:
                result = f"Unknown tool: {tool_name}"

            # Tool result is added as an observation
            new_messages.append(
                SystemMessage(content=str(result))
            )

    return {
        "messages": state["messages"] + new_messages
    }


# ======================================================
# Routing Logic (Orchestration Policy)
# ======================================================

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Decide whether the agent should continue calling tools
    or stop and return a final answer.
    """
    last_msg = state["messages"][-1]

    # If the LLM requested tool calls, continue
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"

    # Otherwise, stop
    return "end"


# ======================================================
# Build the LangGraph Agent
# ======================================================

def build_agent():
    """
    Construct the LangGraph-based FinQA agent.
    """
    graph = StateGraph(AgentState)

    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("agent")

    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    graph.add_edge("tools", "agent")

    return graph.compile()


# ======================================================
# Run Interface
# ======================================================

def run_agent(question: str) -> str:
    """
    Run the FinQA agent on a single question.
    """
    graph = build_agent()

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question),
        ]
    }

    final_state = graph.invoke(initial_state)

    # Return the final message content
    return final_state["messages"][-1].content


# ======================================================
# Example
# ======================================================

if __name__ == "__main__":
    q = (
        "What is the operating profit if revenue is 500 "
        "and cost is 300?"
    )

    answer = run_agent(q)

    print("\nFINAL ANSWER:")
    print(answer)
