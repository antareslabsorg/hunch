#!/usr/bin/env python3
"""Hunch routing a message as it is typed: after every word, one decide() call scores every candidate again.

    python examples/live_routing.py                       # live, in this terminal
    python examples/live_routing.py --record run.jsonl    # headless: every call, its probabilities and its time, to a file
    python examples/live_routing.py --recheck run.jsonl   # every recorded call again, candidates reversed: does order matter?

Run it from the repository root on Apple silicon, with the MLX f16 build of Hunch 0.6B and the data build
(REPRODUCE.md), which writes data/sprint_v3d/calibration.jsonl. The rule that picks the two messages, and what the demo
does and does not show, are in examples/LIVE_ROUTING.md.

Each call is a whole request, made from scratch: the message up to the current word as the state, and the record's
question with every candidate and its description, verbatim, in a freshly shuffled order. Nothing is carried from one
word to the next. A call's time is wall clock around decide(), so tokenisation, the forward pass and the softmax are in
it. The only pause is HOLD_S after each decision returns, so the bars can be read; the recording keeps both clocks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MODEL = REPO / "hunch/formats/out/hunch-0.6b-preview-mlx-f16"
DATA = REPO / "data/sprint_v3d/calibration.jsonl"
OVERLAPS = REPO / "docs/results/launch/contamination_ids.json"
# The selection rule (LIVE_ROUTING.md): these families in this order, each with its question's candidate count; per family
# the first record in file order whose message has WORDS words and whose id is on none of the split's overlap lists.
FAMILIES = [("support.routing.intent.banking77", 77, "Banking77"), ("support.routing.intent.massive", 60, "MASSIVE")]
WORDS = (9, 18)
SEED = 0
HOLD_S = 0.30        # the pause after every decision, in the live view and in the recording
FINAL_HOLD_S = 1.5   # live view: how long the final answer stays before the next message
TOP = 6
FORMAT = "hunch-live-routing-v1"
ENGINE_FILES = ["hunch/fast.py", "hunch/formats/hunch_mlx.py", "hunch/formats/common.py", "hunch/infer.py", "hunch/serialize.py"]
PAPER, RED, MUTED = (244, 240, 232), (255, 84, 46), (170, 168, 163)


def select(data: Path, overlaps: Path) -> list[dict]:
    """The rule, applied to `data` in file order. Returns one {"line", "name", "record"} per family, in FAMILIES order."""
    excluded = set().union(*json.loads(overlaps.read_text())["calibration"].values())  # exact, Jaccard, near-duplicate
    picked: dict[str, dict] = {}
    with data.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            r = json.loads(line)
            for family, k, name in FAMILIES:
                q, state = r["question"], r["state"]
                if family in picked or r["family"] != family or q["type"] != "choice" or len(q["candidates"]) != k:
                    continue
                message = state.get("message") if isinstance(state, dict) else None
                if isinstance(message, str) and WORDS[0] <= len(message.split()) <= WORDS[1] and r["id"] not in excluded:
                    picked[family] = {"line": line_no, "name": name, "record": r}
            if len(picked) == len(FAMILIES):
                break
    missing = [family for family, _, _ in FAMILIES if family not in picked]
    if missing:
        raise SystemExit(f"{data}: no record meets the rule for {missing}")
    return [picked[family] for family, _, _ in FAMILIES]


def prefixes(message: str) -> list[str]:
    """The message up to the end of each word; the last one is the message itself, verbatim."""
    ends = [m.end() for m in re.finditer(r"\S+", message)]
    return [message[:e] for e in ends[:-1]] + [message]


def request(record: dict, prefix: str, order: list[str]) -> list[dict]:
    """The record's state with the message cut at `prefix`, and its question with the candidates in `order`."""
    q = record["question"]
    by_id = {c["id"]: c for c in q["candidates"]}
    return [{"id": "message", "state": {**record["state"], "message": prefix},
             "questions": [{**q, "candidates": [by_id[i] for i in order]}]}]


def ranked(probs: dict[str, float], canonical: list[str]) -> list[tuple[str, float]]:
    """Candidates by probability, highest first; exact ties in the record's own order, so the shuffle cannot decide them."""
    pos = {c: i for i, c in enumerate(canonical)}
    return sorted(probs.items(), key=lambda kv: (-kv[1], pos[kv[0]]))


def network_denied() -> bool:
    """True if this process may not even connect to itself, as under `sandbox-exec` with network access denied.
    It tries a closed port on 127.0.0.1, so nothing leaves the machine either way."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except PermissionError:
        return True
    try:
        s.settimeout(0.5)
        s.connect(("127.0.0.1", 9))
        return False
    except PermissionError:
        return True
    except OSError:  # refused or timed out: the process was allowed to try
        return False
    finally:
        s.close()


def wait_for_quiet(limit: float, max_wait_s: float, poll_s: float = 15.0) -> tuple[list[float], float]:
    """Block until the 1-minute load average is below `limit`, so the timings are not someone else's. Refuse after max_wait_s."""
    t = time.monotonic()
    while True:
        load = list(os.getloadavg())
        if load[0] < limit:
            return load, round(time.monotonic() - t, 1)
        if time.monotonic() - t > max_wait_s:
            raise SystemExit(f"1-minute load {load[0]:.2f} stayed at or above {limit} for {max_wait_s:.0f} s; nothing recorded")
        time.sleep(poll_s)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


class Live:
    """The terminal view: the message with a caret, the top candidates as bars, this decision's time. Plain ANSI."""

    def __init__(self):
        self.color = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        self.panel: list[str] = []

    def c(self, rgb, text):
        return f"\x1b[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m{text}\x1b[0m" if self.color else text

    def set_panel(self, pick: dict, call: dict, labels: dict[str, str], n_words: int, final: str | None = None):
        rows = ranked(call["p"], pick["canonical"])[:TOP]
        width = max(len(labels[c]) for c, _ in rows)
        self.panel = []
        for i, (cid, p) in enumerate(rows):
            bar = ("█" * round(p * 30) or "▏").ljust(30)
            self.panel.append(f"  {labels[cid]:<{width}}  {self.c(RED if i == 0 else PAPER, bar)}  {p:.2f}")
        self.panel += ["", self.c(MUTED, f"  decision {call['k']} of {n_words}: {call['ms']:.0f} ms")]
        if final:
            self.panel.append(self.c(MUTED, "  " + final))

    def draw(self, pick: dict, prefix: str, caret: bool = True):
        name, k = pick["name"], len(pick["record"]["question"]["candidates"])
        head = self.c(MUTED, f"  hunch.  {name}, {k} candidates, all scored after every word; shuffled before every call")
        text = "  " + prefix + (self.c(RED, "▏") if caret else "")
        sys.stdout.write("\x1b[H\x1b[J" + "\n".join([head, "", text, ""] + self.panel) + "\n")
        sys.stdout.flush()


def run(model, picks: list[dict], hold_s: float, emit, live: Live | None = None) -> None:
    """One pass over the messages. `emit` receives every message and every call as a dict (the recording's lines)."""
    rng = random.Random(SEED)
    for m, pick in enumerate(picks):
        record, q = pick["record"], pick["record"]["question"]
        canonical = pick["canonical"] = [c["id"] for c in q["candidates"]]
        labels = {c["id"]: c["label"] for c in q["candidates"]}
        target = canonical[record["target"]["hard"]]
        words = prefixes(record["state"]["message"])
        emit({"type": "message", "index": m, "name": pick["name"], "family": record["family"], "id": record["id"],
              "source": record["source"], "line": pick["line"], "state": record["state"], "question": q,
              "target": target, "words": len(words)})
        if live:
            live.panel = []
        start = time.perf_counter()
        for k, prefix in enumerate(words, 1):
            order = list(canonical)
            rng.shuffle(order)
            req = request(record, prefix, order)
            if live:
                live.draw(pick, prefix)  # the word appears now; the bars still show the previous decision
            t0 = time.perf_counter()
            out = model.decide(req)
            t1 = time.perf_counter()
            ans = out["results"][0]["answers"][q["id"]]
            call = {"type": "call", "msg": m, "k": k, "prefix": prefix, "order": order, "p": ans["probabilities"],
                    "choice": ans["choice"], "t0": round(t0 - start, 6), "t1": round(t1 - start, 6),
                    "ms": round((t1 - t0) * 1000, 3), "engine_ms": out["timing_ms"]["total"],
                    "path_tokens": out["usage"]["path_tokens"]}
            emit(call)
            if live:
                final = None
                if k == len(words):
                    verdict = "matches" if call["choice"] == target else "does not match"
                    final = f"record's label: {labels[target]}; Hunch's answer {verdict}"
                live.set_panel(pick, call, labels, len(words), final)
                live.draw(pick, prefix, caret=final is None)
            if k < len(words):
                time.sleep(max(0.0, t1 + hold_s - time.perf_counter()))
        if live:
            time.sleep(FINAL_HOLD_S)


def summarize(lines: list[dict]) -> dict:
    """What the recording shows, per message and overall; the renderer's check recomputes it."""
    msgs = [x for x in lines if x["type"] == "message"]
    calls = [x for x in lines if x["type"] == "call"]
    def stats(ms):
        q = statistics.quantiles(ms, n=10, method="inclusive") if len(ms) > 1 else ms * 9
        return {"n": len(ms), "median_ms": round(statistics.median(ms), 1), "p90_ms": round(q[8], 1), "max_ms": round(max(ms), 1)}
    per = []
    for m in msgs:
        mc = [c for c in calls if c["msg"] == m["index"]]
        per.append({"name": m["name"], "id": m["id"], "words": m["words"], "calls": len(mc), "final": mc[-1]["choice"],
                    "target": m["target"], "correct": mc[-1]["choice"] == m["target"], **stats([c["ms"] for c in mc])})
    return {"type": "summary", "messages": per, "all": stats([c["ms"] for c in calls])}


def recheck(model, recording: Path) -> dict:
    """Re-score every call of a recording with its candidates in reverse order. Hunch scores each candidate on its own path,
    so the order should not matter: this measures how far any probability moves, and whether any answer changes."""
    lines = [json.loads(x) for x in recording.read_text(encoding="utf-8").splitlines() if x.strip()]
    msgs = {x["index"]: x for x in lines if x["type"] == "message"}
    calls = [x for x in lines if x["type"] == "call"]
    worst, same = 0.0, 0
    for c in calls:
        m = msgs[c["msg"]]
        out = model.decide(request(m, c["prefix"], c["order"][::-1]))
        ans = out["results"][0]["answers"][m["question"]["id"]]
        worst = max(worst, max(abs(ans["probabilities"][i] - p) for i, p in c["p"].items()))
        same += ans["choice"] == c["choice"]
    return {"recording": recording.name, "calls": len(calls), "same_answer": same, "max_abs_dp": worst}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", type=Path, default=MODEL, help="the MLX f16 build (a folder with config.json and model.safetensors)")
    ap.add_argument("--data", type=Path, default=DATA, help="the calibration split of the data build")
    ap.add_argument("--overlaps", type=Path, default=OVERLAPS, help="the split's overlap lists with the training text")
    ap.add_argument("--record", type=Path, help="headless: write every message and call to this JSONL file (never overwritten)")
    ap.add_argument("--recheck", type=Path, help="re-score a recording's calls with the candidates reversed; writes <name>.recheck.json")
    ap.add_argument("--quiet-load", type=float, default=5.0, help="--record waits until the 1-minute load is below this")
    ap.add_argument("--quiet-wait", type=float, default=2700, help="... for at most this many seconds, then records nothing")
    args = ap.parse_args()

    from hunch.formats.hunch_mlx import load  # imported late so --help works without MLX

    if args.recheck:
        result = recheck(load(args.model, fast=True), args.recheck)
        args.recheck.with_suffix(".recheck.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps(result))
        return
    if args.record:
        if args.record.exists():
            raise SystemExit(f"{args.record} exists; a recording is never overwritten")
        quiet_load, waited_s = wait_for_quiet(args.quiet_load, args.quiet_wait)
        # everything slow or unrelated to the calls happens before the model is loaded and warmed up
        model_sha = sha256(args.model / "model.safetensors")
        code = {f: sha256(REPO / f) for f in ENGINE_FILES}  # the code that ran, as it was
        denied = network_denied()
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()

    picks = select(args.data, args.overlaps)
    model = load(args.model, fast=True)  # the release temperature embedded in the build: these families are in family
    warmup = []
    for pick in picks:  # the first calls pay for Metal kernel compilation: one warm-up call per message, timed, not shown
        r = pick["record"]
        t0 = time.perf_counter()
        model.decide(request(r, prefixes(r["state"]["message"])[0], [c["id"] for c in r["question"]["candidates"]]))
        warmup.append(round((time.perf_counter() - t0) * 1000, 3))

    if not args.record:
        live = Live()
        sys.stdout.write("\x1b[?25l")
        try:
            run(model, picks, HOLD_S, lambda line: None, live)
        except KeyboardInterrupt:
            pass
        finally:
            sys.stdout.write("\x1b[?25h\n")
        return

    cfg = json.loads((args.model / "config.json").read_text())["hunch"]
    header = {"type": "header", "format": FORMAT, "started": datetime.now().astimezone().isoformat(timespec="seconds"),
              "argv": sys.argv, "load_when_quiet": quiet_load, "waited_for_quiet_s": waited_s,
              "load_at_start": list(os.getloadavg()),
              "network_denied": denied, "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
              "host": {"cpu": cpu, "machine": platform.machine(), "os": platform.platform(), "python": platform.python_version()},
              "model": {"dir": args.model.name, "name": cfg["name"], "level": cfg.get("level"), "sha256": model_sha,
                        "engine": type(model).__name__, "temperature": model.T, **model.scorer.settings},
              "code_sha256": code,
              "rule": {"data": args.data.name, "families": FAMILIES, "words": WORDS, "overlap_lists": "calibration: all"},
              "seed": SEED, "hold_s": HOLD_S, "top": TOP, "warmup_ms": warmup}
    lines = [header]
    with args.record.open("x", encoding="utf-8") as f:
        def emit(line):
            lines.append(line)
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            f.flush()
        f.write(json.dumps(header, ensure_ascii=False) + "\n")
        run(model, picks, HOLD_S, emit)
        summary = summarize(lines)
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    for s in summary["messages"]:
        print(f"{s['name']}: {s['calls']} calls, median {s['median_ms']} ms, p90 {s['p90_ms']} ms; "
              f"final {s['final']} (record's label {s['target']})")


if __name__ == "__main__":
    main()
