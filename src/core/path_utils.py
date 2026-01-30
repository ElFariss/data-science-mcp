"""Path utilities for consistent absolute path resolution.

This module provides utilities to ensure all file outputs use absolute paths
based on the work directory passed by the agent, not the MCP server's cwd.
"""

from pathlib import Path
from typing import Optional


def get_output_dir(work_dir: str | Path, subdir: str = "reports") -> Path:
    """Get absolute output directory path within the work directory.
    
    Args:
        work_dir: Base working directory (from agent's current directory)
        subdir: Subdirectory name (reports, experiments, features, submissions)
    
    Returns:
        Absolute Path to the output directory (created if doesn't exist)
    """
    base = Path(work_dir).resolve()
    output_dir = base / subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def to_absolute_path(path: str | Path, work_dir: Optional[str | Path] = None) -> str:
    """Convert any path to absolute string path.
    
    Args:
        path: Relative or absolute path
        work_dir: Optional work directory for relative path resolution
        
    Returns:
        Absolute path as string
    """
    p = Path(path)
    if p.is_absolute():
        return str(p.resolve())
    
    if work_dir:
        return str((Path(work_dir) / p).resolve())
    
    return str(p.resolve())


def resolve_data_path(data_path: str, work_dir: Optional[str | Path] = None) -> tuple[Path, Path]:
    """Resolve data path and determine working directory.
    
    If work_dir is not provided, uses the data file's parent directory.
    
    Args:
        data_path: Path to data file
        work_dir: Optional explicit work directory
        
    Returns:
        Tuple of (resolved absolute data path, work directory)
    """
    path = Path(data_path)
    
    # If work_dir is provided, use it
    if work_dir:
        work = Path(work_dir).resolve()
        if path.is_absolute():
            return path.resolve(), work
        else:
            resolved = (work / path).resolve()
            return resolved, work
    
    # If data_path is absolute, use its parent as work_dir
    if path.is_absolute():
        return path.resolve(), path.parent.resolve()
    
    # Try to resolve relative to cwd
    cwd = Path.cwd()
    full_path = (cwd / path).resolve()
    if full_path.exists():
        return full_path, cwd
    
    # Return as-is if can't resolve (let caller handle missing file)
    return path.resolve(), path.parent.resolve() if path.parent != Path('.') else cwd


def ensure_absolute_return(path: str | Path) -> str:
    """Ensure a path is returned as absolute string.
    
    Use this before returning paths from tool functions.
    
    Args:
        path: Path to convert
        
    Returns:
        Absolute path as string
    """
    return str(Path(path).resolve())
