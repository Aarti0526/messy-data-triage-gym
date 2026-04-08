import pandas as pd
import numpy as np
from data_triage_env.engine.dataset_factory import generate_clean
from data_triage_env.engine.corruptor import CORRUPT_FNS
from rich.console import Console
from rich.table import Table
from rich import box

def visualize_task(task_id: str, seed: int = 42):
    console = Console()
    console.print(f"[bold blue]Visualizing Task: {task_id.upper()} (Seed: {seed})[/bold blue]\n")

    # 1. Get Clean and Dirty Data
    clean_df = generate_clean(task_id, seed)
    dirty_df, manifest = CORRUPT_FNS[task_id](clean_df.copy(), np.random.default_rng(seed))

    # 2. Identify corrupted cells
    corrupted_cells = {} # (col, row) -> original_value
    for record in manifest.records:
        if record.corruption_type == "trap" or record.column == "ALL":
            continue
        for idx in record.row_indices:
            if idx < len(clean_df):
                corrupted_cells[(record.column, idx)] = clean_df.at[idx, record.column]
            else:
                # Row index out of clean_df range (e.g. injected duplicate)
                corrupted_cells[(record.column, idx)] = None

    # 3. Create a comparison table for the first 15 rows
    table = Table(title=f"Sample Comparison (First 15 rows of {task_id})", box=box.ROUNDED)
    table.add_column("Row", justify="right", style="cyan")
    
    cols_to_show = dirty_df.columns[:5] # Show first 5 columns to fit on screen
    for col in cols_to_show:
        table.add_column(f"{col} (Dirty)", style="red")
        table.add_column(f"{col} (Clean)", style="green")

    for i in range(min(15, len(dirty_df))):
        row_vals = [str(i)]
        for col in cols_to_show:
            d_val = str(dirty_df.at[i, col])
            c_val = str(clean_df.at[i, col])
            
            # Highlight if different
            if (col, i) in corrupted_cells:
                row_vals.append(f"[bold red]{d_val}[/bold red]")
                row_vals.append(f"[bold green]{c_val}[/bold green]")
            else:
                row_vals.append(d_val)
                row_vals.append(c_val)
        table.add_row(*row_vals)

    console.print(table)
    
    # 4. Show Manifest Summary
    console.print(f"\n[bold yellow]Manifest Summary:[/bold yellow]")
    for record in manifest.records:
        console.print(f"- [cyan]{record.column}[/cyan]: {record.corruption_type} ({len(record.row_indices)} rows) -> [dim]{record.expected_fix}[/dim]")

if __name__ == "__main__":
    import sys
    task = sys.argv[1] if len(sys.argv) > 1 else "easy"
    visualize_task(task)
