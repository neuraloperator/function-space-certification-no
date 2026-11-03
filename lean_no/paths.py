from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]

def repo_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repo_root / value
