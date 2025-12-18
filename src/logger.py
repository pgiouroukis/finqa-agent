"""
Centralized logging for the agentic RAG system.

Provides:
- Colored console output for real-time monitoring
- Detailed JSON file logging with FULL message traces
- Human-readable trace file for easy review
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text


class LogLevel(Enum):
    """Log levels with associated colors and emojis."""
    SYSTEM = ("white", "⚙️")
    AGENT_THINK = ("cyan", "🧠")
    AGENT_PLAN = ("blue", "📋")
    TOOL_CALL = ("green", "🔧")
    TOOL_RESULT = ("bright_green", "📦")
    EVALUATION = ("yellow", "🔍")
    DECISION = ("magenta", "🎯")
    ERROR = ("red", "❌")
    SUCCESS = ("bright_green", "✅")
    DATA = ("dim white", "📊")
    MESSAGE = ("bright_white", "💬")


class AgentLogger:
    """
    Logger that writes colored output to console and detailed traces to files.
    
    Output files in the specified output_dir/:
    - agent_log.jsonl: Full JSON log of every step (machine readable)
    - trace.txt: Human-readable trace (easy to review)
    - messages.json: Full message history at end of run
    """
    
    def __init__(self, output_dir: str = "output", session_id: str | None = None):
        self.console = Console(record=True)
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use output_dir directly as the log directory
        self.log_dir = Path(output_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log files
        self.jsonl_file = self.log_dir / "agent_log.jsonl"
        self.trace_file = self.log_dir / "trace.txt"
        self.messages_file = self.log_dir / "messages.json"
        
        self.step_count = 0
        
        # State tracking
        self.current_state: dict[str, Any] = {}
        
        # Initialize trace file with header
        self._write_trace(f"{'='*80}\n")
        self._write_trace(f"AGENT TRACE - Session: {self.session_id}\n")
        self._write_trace(f"Started: {datetime.now().isoformat()}\n")
        self._write_trace(f"{'='*80}\n\n")
        
        self._log_system(f"Logger initialized. Session: {self.session_id}")
        self._log_system(f"Logs will be saved to: {self.log_dir}")
    
    def _write_json(self, entry: dict[str, Any]) -> None:
        """Append a JSON entry to the JSONL log file."""
        with open(self.jsonl_file, "a") as f:
            f.write(json.dumps(entry, default=str, ensure_ascii=False) + "\n")
    
    def _write_trace(self, text: str) -> None:
        """Append text to the human-readable trace file."""
        with open(self.trace_file, "a") as f:
            f.write(text)
    
    def _format_trace_entry(self, level: LogLevel, title: str, content: Any = None) -> str:
        """Format a log entry for the trace file."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        _, emoji = level.value
        
        lines = [f"[{self.step_count:04d}] {timestamp} {emoji} {level.name}: {title}"]
        
        if content is not None:
            if isinstance(content, dict):
                lines.append(json.dumps(content, indent=2, default=str, ensure_ascii=False))
            elif isinstance(content, list):
                lines.append(json.dumps(content, indent=2, default=str, ensure_ascii=False))
            else:
                lines.append(str(content))
        
        return "\n".join(lines) + "\n\n"
    
    def _log(self, level: LogLevel, title: str, content: Any = None, 
             extra: dict[str, Any] | None = None) -> None:
        """Core logging method."""
        self.step_count += 1
        timestamp = datetime.now().isoformat()
        color, emoji = level.value
        
        # Console output
        header = Text()
        header.append(f"{emoji} [{self.step_count:04d}] ", style="bold")
        header.append(f"{level.name}", style=f"bold {color}")
        header.append(f" | {title}", style=color)
        
        self.console.print(header)
        
        if content is not None:
            if isinstance(content, dict):
                self.console.print_json(data=content)
            elif isinstance(content, list):
                if content and isinstance(content[0], dict):
                    table = Table(show_header=True, header_style="bold")
                    for key in content[0].keys():
                        table.add_column(str(key))
                    for item in content[:10]:
                        table.add_row(*[str(v)[:50] for v in item.values()])
                    if len(content) > 10:
                        self.console.print(f"  ... and {len(content) - 10} more items")
                    self.console.print(table)
                else:
                    self.console.print(content)
            elif isinstance(content, str) and len(content) > 200:
                self.console.print(Panel(content, border_style=color))
            else:
                self.console.print(f"  {content}")
        
        self.console.print()
        
        # JSON log entry (machine readable)
        entry = {
            "step": self.step_count,
            "timestamp": timestamp,
            "level": level.name,
            "title": title,
            "content": content,
            "extra": extra,
            "state": self.current_state.copy(),
        }
        self._write_json(entry)
        
        # Trace file (human readable)
        self._write_trace(self._format_trace_entry(level, title, content))
    
    def _log_system(self, msg: str) -> None:
        self._log(LogLevel.SYSTEM, msg)
    
    # -------------------------
    # Public API
    # -------------------------
    
    def set_state(self, key: str, value: Any) -> None:
        """Update tracked agent state."""
        self.current_state[key] = value
        self._log(LogLevel.SYSTEM, f"State updated: {key}", value)
    
    def log_query(self, query_id: str, question: str) -> None:
        """Log incoming query."""
        self._log(LogLevel.AGENT_THINK, f"Received query: {query_id}", {
            "query_id": query_id,
            "question": question,
        })
    
    def log_system_prompt(self, prompt: str) -> None:
        """Log the system prompt (full content)."""
        self._log(LogLevel.MESSAGE, "System Prompt", prompt)
    
    def log_user_message(self, message: str) -> None:
        """Log a user/human message."""
        self._log(LogLevel.MESSAGE, "User Message", message)
    
    def log_agent_prompt(self, summary: str, full_messages: list[dict[str, Any]] | None = None) -> None:
        """Log the prompt being sent to LLM with optional full message list."""
        content = {"summary": summary}
        if full_messages:
            content["messages"] = full_messages
        self._log(LogLevel.AGENT_THINK, "Sending to LLM", content)
    
    def log_agent_response(self, response: str, raw_response: Any = None) -> None:
        """Log the raw LLM response."""
        content = {"text": response}
        if raw_response is not None:
            content["raw"] = str(raw_response)
        self._log(LogLevel.AGENT_THINK, "LLM Response", content)
    
    def log_plan(self, plan: dict[str, Any]) -> None:
        """Log agent's execution plan (tool calls)."""
        self._log(LogLevel.AGENT_PLAN, "Agent Plan", plan)
    
    def log_tool_call(self, tool_name: str, params: dict[str, Any]) -> None:
        """Log tool invocation."""
        self._log(LogLevel.TOOL_CALL, f"Calling tool: {tool_name}", params)
    
    def log_tool_result(self, tool_name: str, result: Any, elapsed_ms: float) -> None:
        """Log tool result."""
        self._log(LogLevel.TOOL_RESULT, f"Tool result: {tool_name} ({elapsed_ms:.1f}ms)", result)
    
    def log_evaluation(self, criteria: str, result: bool, reason: str) -> None:
        """Log relevance/quality evaluation."""
        self._log(LogLevel.EVALUATION, f"Evaluating: {criteria}", {
            "result": result,
            "reason": reason,
        })
    
    def log_decision(self, decision: str, rationale: str) -> None:
        """Log agent decision point."""
        self._log(LogLevel.DECISION, decision, rationale)
    
    def log_error(self, error: str, details: Any = None) -> None:
        """Log error."""
        self._log(LogLevel.ERROR, error, details)
    
    def log_success(self, message: str, details: Any = None) -> None:
        """Log success."""
        self._log(LogLevel.SUCCESS, message, details)
    
    def log_data(self, label: str, data: Any) -> None:
        """Log raw data."""
        self._log(LogLevel.DATA, label, data)
    
    def log_message_trace(self, messages: list[Any]) -> None:
        """Log the full message history (for debugging)."""
        formatted = []
        for i, msg in enumerate(messages):
            formatted.append({
                "index": i,
                "type": type(msg).__name__,
                "content": getattr(msg, "content", str(msg))[:500],  # Truncate for console
                "tool_calls": getattr(msg, "tool_calls", None),
            })
        self._log(LogLevel.MESSAGE, f"Message Trace ({len(messages)} messages)", formatted)
    
    def save_messages(self, messages: list[Any]) -> str:
        """Save the full message history to a JSON file."""
        formatted = []
        for i, msg in enumerate(messages):
            entry = {
                "index": i,
                "type": type(msg).__name__,
                "content": getattr(msg, "content", str(msg)),
            }
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                entry["tool_calls"] = msg.tool_calls
            if hasattr(msg, "tool_call_id"):
                entry["tool_call_id"] = msg.tool_call_id
            formatted.append(entry)
        
        with open(self.messages_file, "w") as f:
            json.dump(formatted, f, indent=2, default=str, ensure_ascii=False)
        
        self._log(LogLevel.SUCCESS, f"Full message history saved to {self.messages_file}")
        return str(self.messages_file)
    
    def generate_html_trace(self) -> str:
        """Generate an HTML trace visualization."""
        try:
            from .html_trace_generator import generate_html_trace
            html_path = generate_html_trace(self.jsonl_file)
            self._log(LogLevel.SUCCESS, f"HTML trace saved to {html_path}")
            return html_path
        except Exception as e:
            self._log(LogLevel.ERROR, f"Failed to generate HTML trace: {e}")
            return ""
    
    def save_final_report(self, metrics: dict[str, Any]) -> str:
        """Save final summary and return path."""
        report_path = self.log_dir / "final_report.json"
        with open(report_path, "w") as f:
            json.dump({
                "session_id": self.session_id,
                "total_steps": self.step_count,
                "final_state": self.current_state,
                "metrics": metrics,
            }, f, indent=2, default=str)
        
        # Also append to trace
        self._write_trace(f"\n{'='*80}\n")
        self._write_trace(f"FINAL REPORT\n")
        self._write_trace(f"{'='*80}\n")
        self._write_trace(json.dumps(metrics, indent=2, default=str) + "\n")
        
        # Generate HTML trace
        self.generate_html_trace()
        
        self._log(LogLevel.SUCCESS, f"Report saved to {report_path}")
        return str(report_path)
    
    def get_log_dir(self) -> Path:
        """Return the log directory path."""
        return self.log_dir


# Global logger instance (created on first import)
_logger: AgentLogger | None = None


def get_logger(output_dir: str = "output", session_id: str | None = None) -> AgentLogger:
    """Get or create the global logger instance."""
    global _logger
    if _logger is None:
        _logger = AgentLogger(output_dir=output_dir, session_id=session_id)
    return _logger


def reset_logger() -> None:
    """Reset the global logger (for testing)."""
    global _logger
    _logger = None
