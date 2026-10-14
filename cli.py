"""Command-Line Interface for AI-Powered Web Research and Intelligent Answering."""

import argparse
import sys
import os
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.pipeline import WebResearchPipeline
from src.config import DEFAULT_MODEL, AVAILABLE_MODELS

# Reconfigure stdout/stderr for Windows console unicode/emoji support
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

console = Console(safe_box=True)


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Web Research and Intelligent Answering CLI"
    )
    parser.add_argument("query", nargs="?", help="Research query or question to investigate")
    parser.add_argument(
        "--depth",
        choices=["quick", "deep"],
        default="quick",
        help="Research depth: quick (single hop) or deep (multi-hop)",
    )
    parser.add_argument(
        "--model",
        choices=list(AVAILABLE_MODELS.keys()),
        default=DEFAULT_MODEL,
        help="Groq LLM model to use for synthesis",
    )
    parser.add_argument(
        "--sources",
        type=int,
        default=5,
        help="Maximum web sources to retrieve (3 to 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the generated Markdown research report",
    )

    args = parser.parse_args()

    # Prompt if query was not provided
    query = args.query
    if not query:
        console.print("[bold cyan]🌐 AI Web Research CLI[/bold cyan]")
        query = console.input("[bold yellow]Enter your research question: [/bold yellow]").strip()
        if not query:
            console.print("[red]No query entered. Exiting.[/red]")
            sys.exit(1)

    console.print(
        Panel(
            f"[bold green]Starting Research Investigation[/bold green]\n"
            f"[bold]Query:[/bold] {query}\n"
            f"[bold]Depth:[/bold] {args.depth} | [bold]Model:[/bold] {args.model} | [bold]Target Sources:[/bold] {args.sources}",
            title="🔍 DeepResearch AI",
            border_style="cyan",
        )
    )

    pipeline = WebResearchPipeline(model=args.model)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Initializing research...", total=None)

        def on_step(step: str, detail: str):
            progress.update(task, description=f"[bold cyan][{step}][/bold cyan] {detail}")

        try:
            report, context, docs, chunks = pipeline.run_research(
                query=query,
                depth=args.depth,
                max_sources=args.sources,
                progress_callback=on_step,
            )
        except Exception as e:
            console.print(f"[bold red]Error during research execution:[/bold red] {e}")
            sys.exit(1)

    # Display Executive Summary
    console.print("\n")
    console.print(
        Panel(
            Markdown(report.executive_summary),
            title="📋 Executive Summary",
            border_style="green",
            padding=(1, 2),
        )
    )

    # Display In-Depth Synthesis
    console.print("\n")
    console.print(
        Panel(
            Markdown(report.detailed_synthesis),
            title="📖 In-Depth Multi-Source Evidence Synthesis",
            border_style="blue",
            padding=(1, 2),
        )
    )

    # Display Key Findings
    if report.key_findings:
        console.print("\n[bold yellow]💡 Key Findings & Takeaways:[/bold yellow]")
        for kf in report.key_findings:
            console.print(f"  • {kf}")

    # Display Claim Verification Matrix
    if report.claims:
        console.print("\n")
        claim_table = Table(title=f"🛡️ Factual Claim Verification Matrix (Grounding: {report.grounding_score}%)", border_style="cyan")
        claim_table.add_column("Claim", style="white", ratio=4)
        claim_table.add_column("Cited", justify="center", style="yellow", ratio=1)
        claim_table.add_column("Status", justify="center", style="bold green", ratio=1)
        claim_table.add_column("Supporting Primary Excerpt", style="dim cyan", ratio=4)

        for cl in report.claims:
            st_color = "green" if cl.verification_status == "verified" else "yellow" if cl.verification_status == "qualified" else "red"
            claim_table.add_row(
                cl.claim_text,
                str(cl.cited_source_ids),
                f"[{st_color}]{cl.verification_status.upper()}[/{st_color}]",
                cl.supporting_excerpt[:150] + ("..." if len(cl.supporting_excerpt) > 150 else ""),
            )
        console.print(claim_table)

    # Display Sources Table
    console.print("\n")
    table = Table(title="🔗 Scraped & Verified Evidence Sources", border_style="dim")
    table.add_column("#", justify="center", style="bold")
    table.add_column("Title & URL", style="cyan")
    table.add_column("Domain & Date", style="magenta")
    table.add_column("Credibility", justify="center", style="green")
    table.add_column("Relevance", justify="center", style="yellow")

    for cit in report.citations:
        date_str = f"\n[dim]Date: {cit.published_date}[/dim]" if cit.published_date else ""
        table.add_row(
            str(cit.id),
            f"{cit.title}\n[dim]{cit.url}[/dim]",
            f"{cit.domain}{date_str}",
            f"{cit.credibility_badge} ({cit.credibility_score}%)",
            f"{cit.relevance_score:.2f}",
        )
    console.print(table)

    # Display Timing Breakdown
    if report.timings:
        t_info = " | ".join([f"{k.replace('_seconds', '').capitalize()}: {v}s" for k, v in report.timings.items()])
        console.print(f"\n[dim cyan]⚡ Stage Latency: {t_info}[/dim cyan]")

    # Save to file if requested
    if args.output:
        out_path = Path(args.output)
        md_content = f"""# Research Report: {report.query}
Generated using {report.model_used} ({report.synthesis_mode}) in {report.execution_time_seconds}s
Grounding Score: {report.grounding_score}%

## Executive Summary
{report.executive_summary}

## Detailed Analysis
{report.detailed_synthesis}

## Claim Verification Matrix
""" + "\n".join([f"- **Claim:** {c.claim_text}\n  - Status: {c.verification_status}\n  - Excerpt: {c.supporting_excerpt}" for c in report.claims]) + f"""

## Sources
""" + "\n".join([f"[{c.id}] {c.title} ({c.url}) - Credibility: {c.credibility_score}% | Date: {c.published_date or 'N/A'}" for c in report.citations])
        out_path.write_text(md_content, encoding="utf-8")
        console.print(f"\n[bold green]Report saved to: {out_path.resolve()}[/bold green]")

    console.print(f"\n[dim]Completed in {report.execution_time_seconds}s across {report.total_sources_scraped} web sources (Mode: {report.synthesis_mode}).[/dim]\n")


if __name__ == "__main__":
    main()
