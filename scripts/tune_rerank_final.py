"""Tune alpha/beta for the FINAL pipeline on DEV: fine-tuned bi-encoder, phase 1 hands its top 100
patents to the bge cross-encoder (Apple GPU when available), which returns the reranked top 30 (E6).
One process, so cross-encoder pair scores are computed once and reused across the grid.
Usage: .venv/bin/python scripts/tune_rerank_final.py
Writes: reports/rerank_tuning_final_dev.json (+ tagged runs in reports/metrics.json)"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from patent_search.config import ROOT, get_config  # noqa: E402
from patent_search.eval import runner  # noqa: E402


def main():
    cfg = get_config()
    log = lambda s: print(s, flush=True)  # noqa: E731
    out = {"phase1_E4_dev": round(runner.run("E4", "dev", cfg, log=log)["overall"]["MRR@10"]["mean"], 4), "grid": []}
    for alpha in (0.2, 0.3, 0.5):
        for beta in (0.0, 0.1, 0.2):
            r = runner.run("E6", "dev", cfg, rerank_model="bge", alpha=alpha, beta=beta, depth=100,
                           tag=f"bge_a{alpha}_b{beta}", log=log)
            m = r["overall"]
            out["grid"].append({"alpha": alpha, "beta": beta, "MRR@10": round(m["MRR@10"]["mean"], 4),
                                "R@1": round(m["R@1"]["mean"], 4), "R@10": round(m["R@10"]["mean"], 4)})
            log(f"  bge alpha={alpha} beta={beta}: MRR@10 {out['grid'][-1]['MRR@10']} (phase 1 {out['phase1_E4_dev']})")
    out["best"] = max(out["grid"], key=lambda g: (g["MRR@10"], -abs(g["alpha"] - 0.3)))
    out["depth"], out["device"] = 100, runner.engine_for("index_ft", cfg).reranker("bge").device
    (ROOT / "reports" / "rerank_tuning_final_dev.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
