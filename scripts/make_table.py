"""
Render results/metrics.json into a Markdown table suitable for the
README and the defence slides.

Usage:  python scripts/make_table.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def main() -> None:
    path = os.path.join(HERE, "results", "metrics.json")
    if not os.path.exists(path):
        print(f"No metrics.json at {path}. Run `python main.py` first.")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    print()
    print("| Experiment | Setting | Test best | Test worst | Sil. class | Sil. subject |")
    print("|------------|---------|----------:|-----------:|-----------:|-------------:|")
    for k, v in m.items():
        bits = k.split("_", 1)
        method, setting = bits[0], bits[1] if len(bits) == 2 else "-"
        print(
            f"| {method} | {setting} | {fmt(v['best'])} | {fmt(v['worst'])} | "
            f"{fmt(v['silhouette_class'])} | {fmt(v['silhouette_subject'])} |"
        )
    print()
    print("Lower `silhouette_subject` = features less coupled to subject identity.")
    print("Higher `silhouette_class` = features better organised by task class.")


if __name__ == "__main__":
    main()
