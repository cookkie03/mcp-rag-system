"""
Process Manager - Lists and optionally terminates processes using this codebase.
Useful for resolving Qdrant lock conflicts.
"""

import os
import sys
import psutil
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt

console = Console()
PROJECT_DIR = Path(__file__).parent.resolve()


def find_related_processes():
    """Find all Python processes that might be using this codebase."""
    related = []
    current_pid = os.getpid()
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cwd', 'create_time']):
        try:
            info = proc.info
            pid = info['pid']
            name = info['name'] or ''
            cmdline = info['cmdline'] or []
            cwd = info.get('cwd') or ''
            
            # Skip current process
            if pid == current_pid:
                continue
            
            # Check if it's a Python process
            if 'python' not in name.lower():
                continue
            
            # Check if related to this project
            cmdline_str = ' '.join(cmdline).lower()
            project_name = PROJECT_DIR.name.lower()
            project_path = str(PROJECT_DIR).lower()
            
            is_related = (
                project_name in cmdline_str or
                project_path in cmdline_str or
                project_path in cwd.lower() or
                'mcp_server' in cmdline_str or
                'ingest' in cmdline_str or
                'chat.py' in cmdline_str
            )
            
            if is_related:
                # Determine process type
                if 'mcp_server' in cmdline_str or 'mcp' in cmdline_str:
                    proc_type = "MCP Server"
                elif 'ingest' in cmdline_str:
                    proc_type = "Ingest"
                elif 'chat' in cmdline_str:
                    proc_type = "Chat"
                else:
                    proc_type = "Python"
                
                related.append({
                    'pid': pid,
                    'name': name,
                    'type': proc_type,
                    'cmdline': ' '.join(cmdline),  # Comando completo senza troncamento
                    'process': proc
                })
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    
    return related


def display_processes(processes):
    """Display processes in a formatted table with index numbers."""
    if not processes:
        console.print("\n[green]✓ No processes found using this codebase.[/green]\n")
        return False
    
    table = Table(title="Processes Related to This Codebase", show_lines=True)
    table.add_column("#", style="bold green", justify="center")
    table.add_column("PID", style="cyan", justify="right")
    table.add_column("Type", style="magenta")
    table.add_column("Name", style="yellow")
    table.add_column("Command", style="dim", overflow="fold")  # fold per wrapping
    
    for idx, p in enumerate(processes, 1):
        table.add_row(str(idx), str(p['pid']), p['type'], p['name'], p['cmdline'])
    
    console.print()
    console.print(table)
    console.print()
    return True


def terminate_processes(processes):
    """Terminate the listed processes."""
    terminated = 0
    failed = 0
    
    for p in processes:
        try:
            proc = p['process']
            proc.terminate()
            proc.wait(timeout=3)
            console.print(f"[green]✓ Terminated PID {p['pid']} ({p['type']})[/green]")
            terminated += 1
        except psutil.NoSuchProcess:
            console.print(f"[yellow]⚠ PID {p['pid']} already terminated[/yellow]")
        except psutil.TimeoutExpired:
            try:
                proc.kill()
                console.print(f"[yellow]⚠ Force killed PID {p['pid']} ({p['type']})[/yellow]")
                terminated += 1
            except:
                console.print(f"[red]✗ Failed to kill PID {p['pid']}[/red]")
                failed += 1
        except Exception as e:
            console.print(f"[red]✗ Failed to terminate PID {p['pid']}: {e}[/red]")
            failed += 1
    
    return terminated, failed


def remove_lock_file():
    """Remove the Qdrant lock file if it exists."""
    lock_file = PROJECT_DIR / "qdrant_storage" / ".lock"
    if lock_file.exists():
        try:
            lock_file.unlink()
            console.print("[green]✓ Lock file removed (qdrant_storage/.lock)[/green]")
            return True
        except PermissionError:
            console.print("[red]✗ Cannot remove lock file - still in use[/red]")
            return False
    else:
        console.print("[dim]No lock file found[/dim]")
        return True


def parse_selection(selection_str, max_idx):
    """Parse user selection string into list of indices.
    Supports: single numbers, ranges (1-3), comma-separated (1,3,5), 'all', empty for none.
    """
    if not selection_str.strip():
        return []
    
    selection_str = selection_str.strip().lower()
    if selection_str == 'all':
        return list(range(1, max_idx + 1))
    
    indices = set()
    parts = selection_str.replace(' ', '').split(',')
    
    for part in parts:
        if '-' in part:
            try:
                start, end = part.split('-')
                for i in range(int(start), int(end) + 1):
                    if 1 <= i <= max_idx:
                        indices.add(i)
            except ValueError:
                continue
        else:
            try:
                i = int(part)
                if 1 <= i <= max_idx:
                    indices.add(i)
            except ValueError:
                continue
    
    return sorted(indices)


def main():
    console.print("\n[bold cyan]═══ Process Manager for file-search ═══[/bold cyan]\n")
    console.print(f"[dim]Project directory: {PROJECT_DIR}[/dim]\n")
    
    # Find related processes
    console.print("[yellow]Scanning for related processes...[/yellow]")
    processes = find_related_processes()
    
    # Display results
    has_processes = display_processes(processes)
    
    if has_processes:
        # Interactive selection
        console.print("[bold]Select processes to terminate:[/bold]")
        console.print("[dim]  • Enter numbers separated by comma (e.g. 1,3)[/dim]")
        console.print("[dim]  • Use ranges (e.g. 1-3)[/dim]")
        console.print("[dim]  • Type 'all' to terminate all[/dim]")
        console.print("[dim]  • Press Enter to skip[/dim]\n")
        
        selection = Prompt.ask("[bold yellow]Terminate[/bold yellow]", default="")
        selected_indices = parse_selection(selection, len(processes))
        
        if selected_indices:
            selected_processes = [processes[i - 1] for i in selected_indices]
            console.print(f"\n[yellow]Terminating {len(selected_processes)} process(es)...[/yellow]")
            terminated, failed = terminate_processes(selected_processes)
            console.print(f"\n[bold]Result:[/bold] {terminated} terminated, {failed} failed")
            
            # Try to remove lock file after termination
            console.print()
            remove_lock_file()
        else:
            console.print("\n[dim]No processes selected.[/dim]")
    
    # Always offer to remove lock file
    lock_file = PROJECT_DIR / "qdrant_storage" / ".lock"
    if lock_file.exists() and not has_processes:
        remove_lock = Prompt.ask("\n[yellow]Lock file exists. Remove it?[/yellow] (y/n)", default="y")
        if remove_lock.lower() in ('y', 'yes'):
            remove_lock_file()
    
    console.print("\n[green]Done![/green] You can now run ingest.py or other scripts.\n")


if __name__ == "__main__":
    main()
