"""
Runner script for FinQA agent evaluation.

Supports single query execution and batch evaluation.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table


def create_experiment_folder(base_dir: str) -> Path:
    """Create a timestamped experiment folder."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"exp_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def save_config(exp_dir: Path, args: argparse.Namespace) -> None:
    """Save experiment configuration."""
    config = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "max_tool_calls": args.max_tool_calls,
        "data_path": args.data_path,
        "split": args.split,
        "max_samples": args.max_samples,
        "query_ids": args.query_ids,
    }
    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def print_result(result: dict, console: Console) -> None:
    """Print a single result."""
    console.print()
    console.print(Panel(
        result.get("question", "N/A"),
        title=f"❓ Query {result.get('query_id', '?')}",
        border_style="blue",
    ))
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("", style="bold")
    table.add_column("Answer", max_width=60)
    table.add_row("🤖 Agent", result.get("agent_answer", "N/A")[:60])
    table.add_row("✅ Golden", result.get("golden_answer", "N/A")[:60])
    
    console.print(table)
    
    match = result.get("agent_answer", "").strip() == result.get("golden_answer", "").strip()
    match_str = "✅ MATCH" if match else "❌ NO MATCH"
    console.print(f"[dim]{match_str} | Time: {result.get('elapsed_seconds', '?')}s | Tools: {result.get('tool_call_count', '?')}[/dim]")


def print_summary(results: list[dict], console: Console) -> None:
    """Print evaluation summary."""
    total = len(results)
    if total == 0:
        console.print("[yellow]No results to summarize[/yellow]")
        return
    
    # Simple exact match
    exact_matches = sum(
        1 for r in results 
        if r.get("agent_answer", "").strip() == r.get("golden_answer", "").strip()
    )
    
    # Numeric match (try to parse as numbers)
    numeric_matches = 0
    for r in results:
        try:
            agent = float(r.get("agent_answer", "").replace("%", "").replace(",", ""))
            golden = float(r.get("golden_answer", "").replace("%", "").replace(",", ""))
            if abs(agent - golden) < 0.01:
                numeric_matches += 1
        except ValueError:
            pass
    
    console.print()
    console.print("=" * 60)
    console.print(Panel("📊 EVALUATION SUMMARY", style="bold"))
    
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_column("Percentage")
    
    table.add_row("Total Queries", str(total), "")
    table.add_row("Exact Match", str(exact_matches), f"{100*exact_matches/total:.1f}%")
    table.add_row("Numeric Match", str(numeric_matches), f"{100*numeric_matches/total:.1f}%")
    
    avg_time = sum(r.get("elapsed_seconds", 0) for r in results) / total
    avg_tools = sum(r.get("tool_call_count", 0) for r in results) / total
    table.add_row("Avg Time", f"{avg_time:.1f}s", "")
    table.add_row("Avg Tool Calls", f"{avg_tools:.1f}", "")
    
    console.print(table)


def save_results(exp_dir: Path, results: list[dict]) -> None:
    """Save full results to JSON."""
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Also save a simple CSV-like summary
    with open(exp_dir / "results_summary.txt", "w") as f:
        f.write("query_id\tagent_answer\tgolden_answer\tmatch\ttime\ttools\n")
        for r in results:
            match = r.get("agent_answer", "").strip() == r.get("golden_answer", "").strip()
            f.write(f"{r.get('query_id', '')}\t{r.get('agent_answer', '')}\t{r.get('golden_answer', '')}\t{match}\t{r.get('elapsed_seconds', '')}\t{r.get('tool_call_count', '')}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run FinQA agent evaluation")
    parser.add_argument("--model", default="qwen3:4b", help="Ollama model name")
    parser.add_argument("--max-tool-calls", type=int, default=10, help="Max tool calls per query")
    parser.add_argument("--output-dir", default="__output__", help="Base output directory")
    parser.add_argument("--data-path", default=None, help="Path to FinQA JSON (overrides --split)")
    parser.add_argument("--split", default="dev", choices=["train", "dev", "test"], help="Dataset split")
    parser.add_argument("--max-samples", type=int, default=0, help="Max samples to evaluate (0=all)")
    parser.add_argument("--query-ids", nargs="+", default=None, help="Specific query IDs to run")
    parser.add_argument("--single", action="store_true", help="Run single query (default: query-id 0)")
    parser.add_argument("--query-id", default="0", help="Query ID for single mode")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    console = Console()
    
    # Determine data path
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = f"FinQA/dataset/{args.split}.json"
    
    # Load data
    console.print(f"[dim]Loading data from {data_path}...[/dim]")
    from src.finqa_data import load_finqa_split
    data = load_finqa_split(data_path, max_samples=args.max_samples if not args.single else 0)
    console.print(f"[dim]Loaded {len(data)} examples[/dim]")
    
    # Determine queries to run
    if args.single:
        query_ids = [args.query_id]
    elif args.query_ids:
        query_ids = args.query_ids
    else:
        query_ids = [str(i) for i in range(len(data))]
    
    console.print(f"[bold]Running {len(query_ids)} queries with model: {args.model}[/bold]")
    console.print()
    
    # Create experiment folder
    exp_dir = create_experiment_folder(args.output_dir)
    save_config(exp_dir, args)
    console.print(f"[dim]Experiment folder: {exp_dir}[/dim]")
    
    # Import orchestrator (triggers model loading)
    from src.orchestrator import FinQAOrchestrator
    
    orchestrator = FinQAOrchestrator(
        model=args.model,
        output_dir=str(exp_dir),
        max_tool_calls=args.max_tool_calls,
    )
    
    # Run queries
    results = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=not args.verbose,
    ) as progress:
        task = progress.add_task("Processing queries...", total=len(query_ids))
        
        for query_id in query_ids:
            progress.update(task, description=f"Query {query_id}...")
            
            result = orchestrator.run(
                query_id=query_id,
                data=data,
                session_id=f"query_{query_id}",
            )
            results.append(result)
            
            if args.verbose or args.single:
                print_result(result, console)
            
            progress.advance(task)
    
    # Save results
    save_results(exp_dir, results)
    
    # Print summary
    print_summary(results, console)
    
    console.print()
    console.print(f"[bold green]✅ Results saved to: {exp_dir}[/bold green]")


if __name__ == "__main__":
    main()
