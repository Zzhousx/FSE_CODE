"""Sum measured SensoDat split-process wall times from revision logs."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1] / "results" / "revision"
total = 0.0
for task in ("t0_sensodat", "T2", "T2diag", "T3", "T4"):
    times = []
    paths = sorted((ROOT / task / "process_logs").glob("split_*.out"))
    for path in paths:
        matches = re.findall(r"elapsed[= ]([0-9.]+)s", path.read_text())
        if matches:
            times.append(float(matches[-1]))
    subtotal = sum(times)
    total += subtotal
    print(f"{task}: {len(times)}/10 logged splits, {subtotal:.1f} process-seconds, "
          f"max logged split {max(times):.1f}s")
print(f"Total logged SensoDat split-process time: {total:.1f}s "
      f"({total/3600:.2f} process-hours); this is a lower bound.")
