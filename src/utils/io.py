import json
from pathlib import Path
from typing import Any

import pandas as pd


def set_nested_attr(obj: Any, path: str, value: Any) -> None:
    """Sets a nested attribute value using dot notation."""
    parts = path.split(".")
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], value)


def save_json(data: dict, path: Path) -> None:
    """Saves dictionary to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def save_metrics_json(metrics: dict[str, float], path: Path) -> None:
    """Saves metrics dictionary to JSON with values in percentage format (xx.xx)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    formatted = {k: round(v * 100, 2) for k, v in metrics.items()}
    with open(path, "w") as f:
        json.dump(formatted, f, indent=4)


def save_markdown_results(results: dict[str, float], path: Path, title: str) -> None:
    """Saves results dictionary as a markdown table.

    Values are converted to percentage format (xx.xx) for display.
    The original results dictionary is not modified.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Convert to percentage format (multiply by 100, 2 decimal places)
    formatted_results = {k: f"{v * 100:.2f}" for k, v in results.items()}
    df = pd.DataFrame([formatted_results])
    markdown = f"# {title}\n\n"
    markdown += df.to_markdown(index=False)
    with open(path, "w") as f:
        f.write(markdown)
