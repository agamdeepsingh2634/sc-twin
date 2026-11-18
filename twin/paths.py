"""Where results go: results/ for synthetic demand, results/m5/ when real
M5 demand is switched on (SCTWIN_M5 set), so the two studies never mix."""
import os


def results_dir(root: str) -> str:
    d = os.path.join(root, "results", "m5") if os.environ.get("SCTWIN_M5") else os.path.join(root, "results")
    os.makedirs(d, exist_ok=True)
    return d
