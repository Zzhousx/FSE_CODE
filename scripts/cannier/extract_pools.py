"""Extract the frozen CANNIER SQLite databases into scheduler-safe pools."""
from __future__ import annotations

import hashlib
import sqlite3
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "results/cannier/raw/output.zip"
VOLUME = ROOT / "results/cannier/raw/output/volume"
POOL_DIR = ROOT / "results/cannier/pools"
EXPECTED_SHA256 = "07d714a7e0f57863070d1ba6631191aa6e856fac1ad3ffc92765d6c1fc56f9f1"
EXPECTED_POS = {
    "Cirq": 51, "airflow": 220, "conan": 104, "dask": 14,
    "fonttools": 42, "hydra": 18, "ipython": 18, "kombu": 14,
    "libcloud": 73, "loguru": 22, "mitmproxy": 16, "prefect": 5,
    "salt": 61, "setuptools": 5, "tornado": 65, "xonsh": 28,
}
EXPECTED_INT = {"Cirq", "airflow", "fonttools", "hydra", "ipython", "kombu",
                "libcloud", "loguru", "mitmproxy", "salt", "xonsh"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    got = sha256(ARCHIVE)
    if got != EXPECTED_SHA256:
        raise RuntimeError(f"archive SHA256 mismatch: {got}")
    VOLUME.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE) as zf:
        members = [n for n in zf.namelist()
                   if n.startswith("output/volume/") and n.lower().endswith(".sqlite3")]
        if len(members) != 30:
            raise RuntimeError(f"expected 30 SQLite files, found {len(members)}")
        for name in members:
            target = VOLUME / Path(name).name
            if not target.exists():
                with zf.open(name) as src, target.open("wb") as dst:
                    while block := src.read(1024 * 1024):
                        dst.write(block)

    POOL_DIR.mkdir(parents=True, exist_ok=True)
    stats = []
    all_pos = all_int = 0
    for dbpath in sorted(VOLUME.glob("*.sqlite3")):
        project = dbpath.stem
        con = sqlite3.connect(f"file:{dbpath.as_posix()}?mode=ro", uri=True)
        rows = con.execute("""SELECT id, nodeid, n_fail_shuffle
            FROM item WHERE n_runs_features=30 AND n_runs_baseline=2500
            AND n_runs_shuffle=2500 ORDER BY id""").fetchall()
        costs = con.execute("""SELECT item_id, AVG(time_exec), COUNT(*)
            FROM features GROUP BY item_id""").fetchall()
        con.close()
        cost_map = {int(i): (float(c), int(n)) for i, c, n in costs}
        ids, fails, cs = [], [], []
        missing = []
        for item_id, _nodeid, f in rows:
            item_id = int(item_id)
            if item_id not in cost_map:
                missing.append(item_id)
                continue
            c, n = cost_map[item_id]
            if n != 30:
                raise RuntimeError(f"{project}/{item_id} has {n} feature rows")
            ids.append(item_id); fails.append(int(f)); cs.append(c)
        if missing:
            raise RuntimeError(f"{project}: {len(missing)} candidates lack timing")
        ids = np.asarray(ids, dtype=np.int64)
        fails = np.asarray(fails, dtype=np.int16)
        cs = np.asarray(cs, dtype=np.float64)
        p = fails / 2500 > .3
        pi = p & (fails < 2500)
        all_pos += int(p.sum()); all_int += int(pi.sum())
        B_pass = float(cs.sum())
        np.savez_compressed(POOL_DIR / f"{project}.npz", item_id=ids, F=fails,
                            cost=cs, B_pass=np.asarray(B_pass))
        stats.append({"project": project, "n_candidates": len(ids),
                      "B_pass": B_pass, "cost_median": float(np.median(cs)),
                      "cost_mean": float(np.mean(cs)), "cost_max": float(np.max(cs)),
                      "n_F0": int((fails == 0).sum()),
                      "n_0F_le750": int(((fails > 0) & (fails <= 750)).sum()),
                      "n_pos_int": int(pi.sum()), "n_always_fail": int((fails == 2500).sum()),
                      "n_pos": int(p.sum()), "eligible_primary": bool(p.sum() >= 5),
                      "eligible_secondary": bool(pi.sum() >= 5)})
    df = pd.DataFrame(stats).sort_values("project")
    df.to_csv(ROOT / "results/cannier/pool_stats.csv", index=False)
    pos = dict(zip(df.loc[df.n_pos >= 5, "project"],
                   df.loc[df.n_pos >= 5, "n_pos"]))
    second = set(df.loc[df.n_pos_int >= 5, "project"])
    print(f"archive_sha256={got}; sqlite={len(stats)}; tests={df.n_candidates.sum()}; "
          f"positive={all_pos}; intermittent={all_int}; always_fail={int((df.n_always_fail).sum())}")
    print(f"primary={pos}; secondary={sorted(second)}")
    ok = (len(stats) == 30 and int(df.n_candidates.sum()) == 89668
          and all_pos == 775 and all_int == 448
          and int(df.n_always_fail.sum()) == 327
          and pos == EXPECTED_POS and second == EXPECTED_INT)
    if not ok:
        raise SystemExit("anchor mismatch: stop before running schedulers")
    print("ALL CANNIER DATA ANCHORS PASS")


if __name__ == "__main__":
    main()
