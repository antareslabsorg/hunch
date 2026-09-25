#!/usr/bin/env python3
"""Draw the launch figures as plain SVG from the same files the documents cite (no plotting library).

    python scripts/render_figures.py

- docs/figures/typed-decisions.svg: typed-decisions accuracy, zero-shot, with 95 % case-bootstrap intervals (derived.json),
  for both releases and the two open Decider models measured with the same code, against the suite card's own scale.
- docs/figures/calibration-dev.svg: in-family dev reliability of both releases at T = 1 and at the release temperature.
  The curve is the kernel-smoothed conditional accuracy that `hunch.metrics.smooth_ece` integrates, so the plotted gap and
  the quoted smooth ECE are the same quantity.
"""
import importlib.util, json, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
_s = importlib.util.spec_from_file_location("compare_results", ROOT / "scripts/compare_results.py")
cr = importlib.util.module_from_spec(_s); _s.loader.exec_module(cr)
from hunch.metrics import calibration_pairs, smooth_ece  # noqa: E402

INK, MUTED, ACCENT, GRID, FONT = "#15171A", "#6B7078", "#E0441F", "#D9DCE1", "font-family:Inter,Helvetica,Arial,sans-serif"
J = lambda p: json.loads((ROOT / p).read_text())


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" style="{FONT}">'
            f'<rect width="{w}" height="{h}" fill="#FFFFFF"/>{body}</svg>\n')


def text(x, y, s, size=13, fill=INK, anchor="start", weight=400):
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{s}</text>'


def typed_decisions():
    d, card = J("docs/results/launch/derived.json")["runs"], J("docs/results/launch/td_card.json")["card"]
    rows = [("Hunch 0.6B", "docs/results/day6/td_zeroshot_w16-06-v3dgp10-1200-s55.json", "w16-06-v3dgp10-1200-s55", ACCENT),
            ("Hunch 1.7B", "docs/results/day7/v1/td_g2-17-v4t.json", "g2-17-v4t", ACCENT),
            ("Decider 0.8B (open)", "docs/results/day7/v1/td_decider-0.8b.json", "decider-0.8b", INK),
            ("Decider 2B (open)", "docs/results/day7/v1/td_decider-2b.json", "decider-2b", INK)]
    refs = [(card["Prior (ignores the input)"]["accuracy"], "prior"), (card["Majority baseline"]["accuracy"], "majority: the card's floor"),
            (card["Perfect scenario understanding"]["accuracy"], "latent-factor model"), (card["Teacher self-agreement"]["accuracy"], "teacher self-agreement")]
    W, H, L, R, T0, dy = 760, 330, 190, 40, 70, 48
    x = lambda v: L + (v - 0.30) / (0.80 - 0.30) * (W - L - R)
    b = [text(20, 30, "typed-decisions, zero-shot: accuracy with 95 % intervals", 16, weight=600),
         text(20, 50, "2,000 decisions; gold is a teacher model's answers, so this measures agreement with that teacher", 12, MUTED)]
    yb = T0 + dy * len(rows)
    for v in np.arange(0.30, 0.801, 0.05):
        b += [f'<line x1="{x(v):.1f}" y1="{T0 - 14}" x2="{x(v):.1f}" y2="{yb - 10}" stroke="{GRID}" stroke-width="1"/>',
              text(x(v), yb + 6, f"{v:.2f}", 11, MUTED, "middle")]
    for v, _ in refs:
        b.append(f'<line x1="{x(v):.1f}" y1="{T0 - 14}" x2="{x(v):.1f}" y2="{yb - 10}" stroke="{MUTED}" stroke-width="1" stroke-dasharray="4 4"/>')
    b.append(text(20, yb + 34, "Dashed, left to right, from the suite's card: " + " · ".join(f"{name} {v:.3f}" for v, name in refs), 11, MUTED))
    for i, (label, f, run, col) in enumerate(rows):
        acc, (lo, hi), y = J(f)["overall"]["accuracy"], d[run]["typed_decisions"]["accuracy_ci"], T0 + dy * i + 10
        b += [text(L - 14, y + 4, label, 13, INK, "end", 600 if col == ACCENT else 400),
              f'<line x1="{x(lo):.1f}" y1="{y}" x2="{x(hi):.1f}" y2="{y}" stroke="{col}" stroke-width="3" stroke-linecap="round"/>',
              f'<circle cx="{x(acc):.1f}" cy="{y}" r="6" fill="{col}"/>', text(x(hi) + 10, y + 4, f"{acc:.4f}", 12, col)]
    return svg(W, yb + 52, "".join(b))


def reliability(conf, correct, sigma=0.05):
    """The smoothed conditional accuracy and mean confidence on smooth_ece's grid, and the grid's density."""
    g = np.linspace(0, 1, 201)
    w = sum(np.exp(-0.5 * ((a - b) / sigma) ** 2) for a, b in [(g[:, None], conf[None, :]), (g[:, None], -conf[None, :]), (2 - g[:, None], conf[None, :])])
    dens = w.sum(1)
    return (w * conf[None, :]).sum(1) / np.maximum(dens, 1e-12), (w * correct[None, :]).sum(1) / np.maximum(dens, 1e-12), dens / dens.sum()


def calibration():
    d = J("docs/results/launch/derived.json")["runs"]
    panels = [("Hunch 0.6B", "docs/results/day6/w16-06-v3dgp10-1200-s55_dev.json", "w16-06-v3dgp10-1200-s55"),
              ("Hunch 1.7B", "docs/results/day7/v1/g2-17-v4t_dev.json", "g2-17-v4t")]
    W, H, S, top = 760, 420, 280, 80
    b = [text(20, 30, "In-family dev calibration: confidence against accuracy", 16, weight=600),
         text(20, 50, "kernel-smoothed, as smooth ECE is computed; grey T = 1, colour the release temperature fitted on the calibration split", 12, MUTED)]
    for k, (label, f, run) in enumerate(panels):
        pr, T = J(f)["predictions"], d[run]["temperature"]["T"]
        P, Y, qt = cr._arr([p["probs"] for p in pr]), cr._arr([p["target"] for p in pr]), [p["qtype"] for p in pr]
        ox = 70 + k * (S + 90)
        X = lambda v: ox + v * S; Yp = lambda v: top + S - v * S
        b += [f'<rect x="{ox}" y="{top}" width="{S}" height="{S}" fill="none" stroke="{GRID}"/>',
              f'<line x1="{X(0)}" y1="{Yp(0)}" x2="{X(1)}" y2="{Yp(1)}" stroke="{GRID}" stroke-width="1.5" stroke-dasharray="5 4"/>',
              text(ox, top - 12, label, 14, weight=600)]
        for v in (0, 0.5, 1):
            b += [text(X(v), top + S + 18, f"{v:g}", 11, MUTED, "middle"), text(ox - 8, Yp(v) + 4, f"{v:g}", 11, MUTED, "end")]
        b += [text(ox + S / 2, top + S + 36, "confidence", 12, MUTED, "middle"),
              f'<text transform="translate({ox - 30},{top + S / 2}) rotate(-90)" font-size="12" fill="{MUTED}" text-anchor="middle">accuracy</text>']
        for j, (Tm, col, name) in enumerate([(1.0, MUTED, "T = 1"), (T, ACCENT, f"T = {T:.4f}")]):
            Pt = cr.temper(P, Tm) if Tm != 1.0 else P
            conf, corr = calibration_pairs(Pt, Y, qt)
            mc, acc, dens = reliability(conf, corr)
            keep = dens > 0.02 / len(dens)                              # draw only where there is data
            pts = " ".join(f"{X(a):.1f},{Yp(c):.1f}" for a, c, kp in zip(mc, acc, keep) if kp)
            b += [f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5"/>',
                  text(ox + 12, top + 22 + 18 * j, f"{name}: smooth ECE {smooth_ece(conf, corr):.4f}", 12, col, weight=600 if j else 400)]
    return svg(W, H, "".join(b))


if __name__ == "__main__":
    out = ROOT / "docs/figures"; out.mkdir(parents=True, exist_ok=True)
    (out / "typed-decisions.svg").write_text(typed_decisions())
    (out / "calibration-dev.svg").write_text(calibration())
    print("wrote", ", ".join(str(p.relative_to(ROOT)) for p in sorted(out.glob("*.svg"))))
