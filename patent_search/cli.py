"""Command line: build-index, verify-index, search, show, data-issues.

  python -m patent_search build-index --data data/patent_data_small
  python -m patent_search search "noise reducing tire" --cpc B60C --kw noise --kw-field abstract
  python -m patent_search search --mode claim "a tread portion ...; a sound absorbing member ..."
  python -m patent_search search --mode patent --doc 20240051333 --export chart.md
  python -m patent_search show 20240051333
  python -m patent_search data-issues --severity high
  python -m patent_search interactive        # type queries one after another
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ROOT, get_config


def _filters(a) -> dict:
    f: dict = {}
    if a.cpc:
        f["cpc_prefix"] = a.cpc
    if a.cpc_exclude:
        f["cpc_exclude"] = a.cpc_exclude
    if a.kw:
        f["keywords"] = {"terms": a.kw, "field": a.kw_field, "op": a.kw_op}
    if a.title:
        f["title"] = {"value": a.title, "mode": a.title_mode}
    if a.week_from:
        f["week_from"] = a.week_from
    if a.week_to:
        f["week_to"] = a.week_to
    return f


def _add_search_args(s):
    s.add_argument("query", nargs="?", default=None)
    s.add_argument("--mode", choices=["text", "claim", "patent"], default="text")
    s.add_argument("--doc", help="doc number (mode=patent)")
    s.add_argument("--cpc", action="append", help="classification prefix, repeatable (OR)")
    s.add_argument("--cpc-exclude", action="append")
    s.add_argument("--kw", action="append", help="keyword that must appear, repeatable")
    s.add_argument("--kw-field", choices=["title", "abstract", "both"], default="both")
    s.add_argument("--kw-op", choices=["all", "any", "phrase"], default="all")
    s.add_argument("--title")
    s.add_argument("--title-mode", choices=["exact", "phrase"], default="exact")
    s.add_argument("--week-from")
    s.add_argument("--week-to")
    s.add_argument("--top-k", type=int, default=None, help="results shown (default 10; 30 with --rerank)")
    s.add_argument("--rerank", action=argparse.BooleanOptionalAction, default=None,
                   help="two-phase search: rerank phase-1 candidates with a cross-encoder")
    s.add_argument("--rerank-model", choices=["bge"])
    s.add_argument("--rerank-depth", type=int)
    s.add_argument("--retrieval", choices=["hybrid", "dense", "bm25"], default="hybrid")
    s.add_argument("--format", choices=["text", "json"], default="text")
    s.add_argument("--export", type=Path, help="write a claim chart (.md or .csv) or the JSON (.json)")


def _run_search(eng, a) -> int:
    from .render import chart, text
    from .schemas import SearchRequest
    try:
        top_k = a.top_k or (eng.cfg.search.rerank_top_k if a.rerank else 10)
        req = SearchRequest(query=a.query, mode=a.mode, doc_number=a.doc, filters=_filters(a), top_k=top_k,
                            rerank=a.rerank, rerank_model=a.rerank_model, rerank_depth=a.rerank_depth,
                            retrieval=a.retrieval)
        resp = eng.search(req)
    except (ValueError, LookupError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.export:
        suffix = a.export.suffix.lower()
        body = {".md": chart.to_markdown, ".csv": chart.to_csv}.get(
            suffix, lambda r: r.model_dump_json(indent=2))(resp)
        a.export.write_text(body)
        print(f"wrote {a.export}", file=sys.stderr)
    print(resp.model_dump_json(indent=2) if a.format == "json" else text.render(resp))
    return 0


_HELP = """Type a search, then Enter. Examples:
  tire with sound absorbing foam in the cavity
  carbon fiber spoke --cpc B60B
  --mode claim "a rim; a disk with spokes; and a wall portion closing the air holes between the spokes"
  --mode patent --doc 20240165989
  --title SPOKE
  trailer hitch --kw coupling --kw-field abstract --top-k 5
  show 20240051333          (one patent in full)
Options are the same as for `search` (--cpc, --kw, --kw-field, --title, --mode, --top-k, --format json ...).
Type help for this text, quit to leave."""


def _interactive(eng) -> int:
    import shlex

    from .render.text import render_patent
    sp = argparse.ArgumentParser(prog="search>", add_help=False, exit_on_error=False)
    _add_search_args(sp)
    _ = eng.encoder                                   # load the model once
    print(f"Index: {len(eng.patents)} patents, {eng.n_chunks} chunks. Model loaded.\n{_HELP}\n")
    while True:
        try:
            line = input("search> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not line:
            continue
        if line in ("quit", "exit", "q"):
            return 0
        if line in ("help", "?"):
            print(_HELP)
            continue
        try:
            parts = shlex.split(line)
        except ValueError as e:
            print(f"error: {e} (check your quotes)")
            continue
        if parts[0] == "show" and len(parts) == 2:
            try:
                print(render_patent(eng.patent(parts[1])))
            except LookupError as e:
                print(f"error: {e}")
            continue
        if not parts[0].startswith("-") and not any(p.startswith("-") for p in parts):
            parts = [line]                            # plain words: the whole line is the query
        elif not parts[0].startswith("-"):
            i = next(k for k, p in enumerate(parts) if p.startswith("-"))
            parts = [" ".join(parts[:i])] + parts[i:]  # unquoted words before the first option
        try:
            a = sp.parse_args(parts)
        except (argparse.ArgumentError, SystemExit) as e:
            print(f"error: {e}")
            continue
        _run_search(eng, a)
        print()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="patent_search", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=Path, help="index directory (default ./index or $PS_INDEX_DIR)")
    ap.add_argument("--device", choices=["cpu", "mps"], help="encoder device (default cpu)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-index", help="ingest JSON files and build the local index")
    b.add_argument("--data", type=Path, default=ROOT / "data" / "patent_data_small")
    b.add_argument("--no-cache", action="store_true", help="re-encode everything (ignore the embedding cache)")
    b.add_argument("--chunking", choices=["adaptive", "fixed"], default="adaptive",
                   help="fixed = baseline windows for the Part 3 evaluation")
    b.add_argument("--model", help="embedding model: hub name or a local directory (e.g. models/gte-patent-ft)")

    v = sub.add_parser("verify-index", help="check the index against data, code and the pinned model")
    v.add_argument("--data", type=Path, default=ROOT / "data" / "patent_data_small")
    v.add_argument("--sample", type=int, default=200)

    s = sub.add_parser("search", help="search the index")
    _add_search_args(s)

    sub.add_parser("interactive", help="type queries one after another (model loads once)")

    sh = sub.add_parser("show", help="show one patent with its claims and data issues")
    sh.add_argument("doc")
    sh.add_argument("--format", choices=["text", "json"], default="text")

    ev = sub.add_parser("eval", help="Part 3 evaluation (sets, runs, pairs, filters, report)")
    ev.add_argument("action", choices=["build-sets", "run", "pairs", "filters", "compare", "rescore", "report"])
    ev.add_argument("--exp", help="experiment id, e.g. E3 (run)")
    ev.add_argument("--split", default="dev", choices=["dev", "test"])
    ev.add_argument("--rerank-model", choices=["bge"])
    ev.add_argument("--alpha", type=float)
    ev.add_argument("--beta", type=float)
    ev.add_argument("--depth", type=int)
    ev.add_argument("--tag")
    ev.add_argument("--kind-weights", help="claim-mode chunk weights, e.g. claim=0.6,element=0.3,description=1,abstract=1")
    ev.add_argument("--limit", type=int)
    ev.add_argument("--pairs-index", default="index", help="index whose model T4 evaluates (pairs)")
    ev.add_argument("--a", help="run name, e.g. E3_test (compare)")
    ev.add_argument("--b", help="run name, e.g. E5_test (compare)")
    ev.add_argument("--metric", default="MRR@10")
    ev.add_argument("--seed", type=int, default=13)
    ev.add_argument("--max-train-anchors", type=int, default=1200)

    tr = sub.add_parser("train", help="Part 3: fine-tune the bi-encoder on mined triplets")
    tr.add_argument("--triplets", type=Path, default=ROOT / "eval_data" / "triplets_train.jsonl")
    tr.add_argument("--out", type=Path, default=ROOT / "models" / "gte-patent-ft")
    tr.add_argument("--max-pairs", type=int)
    tr.add_argument("--epochs", type=int, default=1)
    tr.add_argument("--batch-size", type=int, default=16)
    tr.add_argument("--lr", type=float, default=2e-5)
    tr.add_argument("--max-seq-length", type=int, default=256)

    di = sub.add_parser("data-issues", help="list data issues (pending data research)")
    di.add_argument("--status", default="open")
    di.add_argument("--severity")
    di.add_argument("--type")
    di.add_argument("--limit", type=int, default=20)

    a = ap.parse_args(argv)
    overrides = {}
    if a.index:
        overrides["index_dir"] = a.index
    if a.device:
        overrides["device"] = a.device
    if getattr(a, "chunking", None) and a.chunking != "adaptive":
        from dataclasses import replace
        from .config import ChunkingConfig
        overrides["chunking"] = replace(ChunkingConfig(), mode=a.chunking)
    if getattr(a, "model", None):
        overrides["model_name"] = a.model
        if Path(a.model).is_dir():
            from .manifest import model_dir_revision
            overrides["model_revision"] = model_dir_revision(Path(a.model))
    cfg = get_config(**overrides)

    if a.cmd == "build-index":
        from .index.build import build_index
        m = build_index(a.data, cfg, use_cache=not a.no_cache, log=lambda s: print(s, flush=True))
        print(json.dumps({"report": m["report"], "timings": m["timings"], "vectors": m["vectors"]}, indent=2))
        return 0
    if a.cmd == "verify-index":
        from .index.verify import verify
        ok = verify(a.data, cfg, sample=a.sample)
        print("verify-index:", "PASS" if ok else "FAIL")
        return 0 if ok else 1

    if a.cmd == "train":
        from .train.finetune import finetune
        card = finetune(a.triplets, a.out, cfg.model_name, cfg.model_revision, device=cfg.device, epochs=a.epochs,
                        batch_size=a.batch_size, lr=a.lr, max_seq_length=a.max_seq_length, max_pairs=a.max_pairs,
                        seed=cfg.seed, log=lambda s: print(s, flush=True))
        print(json.dumps(card, indent=2))
        return 0
    if a.cmd == "eval":
        from .eval import report, runner
        log = lambda s: print(s, flush=True)  # noqa: E731
        if a.action == "build-sets":
            from .eval.build_sets import build_all
            from .search.engine import Engine
            stats = build_all(Engine(cfg), runner.EVAL_DIR, seed=a.seed, max_train_anchors=a.max_train_anchors, log=log)
            print(json.dumps(stats, indent=2))
        elif a.action == "run":
            kw = None
            if a.kind_weights:
                kw = {"abstract": 1.0, "claim": 1.0, "element": 1.0, "description": 1.0}
                kw.update({k: float(v) for k, v in (p.split("=") for p in a.kind_weights.split(","))})
            res = runner.run(a.exp, a.split, cfg, a.rerank_model, a.alpha, a.beta, a.depth, a.tag, a.limit, log, kw)
            ov = res["overall"]
            print(json.dumps({"overall": {k: round(v["mean"], 4) for k, v in ov.items() if isinstance(v, dict)},
                              "n": ov["n"], "latency_ms": res["latency_ms"]}, indent=1))
        elif a.action == "pairs":
            print(json.dumps(runner.pairs_eval(a.pairs_index, a.split, cfg), indent=1))
        elif a.action == "filters":
            print(json.dumps(runner.filter_correctness(cfg, a.split), indent=1))
        elif a.action == "compare":
            res = runner.compare(a.a, a.b, a.metric)
            runner.save_metrics(f"cmp_{a.a}_vs_{a.b}_{a.metric}_{a.split}", res)
            print(json.dumps(res, indent=1))
        elif a.action == "rescore":
            res = runner.rescore(a.a, a.b, a.alpha, a.beta)
            print(json.dumps({"overall": {k: round(v["mean"], 4) for k, v in res["overall"].items()
                                          if isinstance(v, dict)}}, indent=1))
        elif a.action == "report":
            print(report.write(a.split))
        return 0

    from .search.engine import Engine
    eng = Engine(cfg)
    if a.cmd == "search":
        return _run_search(eng, a)
    if a.cmd == "interactive":
        return _interactive(eng)
    if a.cmd == "show":
        from .render.text import render_patent
        try:
            p = eng.patent(a.doc)
        except LookupError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(json.dumps(p, indent=2, default=str) if a.format == "json" else render_patent(p))
        return 0
    if a.cmd == "data-issues":
        res = eng.data_issues(a.status, a.severity, a.type, a.limit)
        print("Data issues (pending data research):")
        for r in res["summary"]:
            print(f"  {r['n']:>5}  {r['severity']:<8} {r['issue_type']}")
        print(f"\nFirst {len(res['items'])}:")
        for i in res["items"]:
            print(f"  #{i['id']} {i['doc_number']} [{i['severity']}] {i['issue_type']} @ {i['location']}: "
                  f"{(i['detail'] or '')[:100]}")
        if res["quarantine"]:
            print(f"\nQuarantined records: {len(res['quarantine'])}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
