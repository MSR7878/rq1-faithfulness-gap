"""
RQ1 Tier 4: re-label the stored TOST results (Tox21, runs/rq1_revision/t3.json; BBBP, runs/rq1_tier4/L2.json) with four labels.
Re-labelling only: reads the stored point estimates and CIs, runs nothing new.

  different       95% CI excludes 0 and |effect| > 0.05
  small but real  95% CI excludes 0 and |effect| <= 0.05
  equivalent      95% CI includes 0 and the 90% CI lies within +/-0.05 (the TOST criterion used before)
  undecided       otherwise
effect = the hierarchical-bootstrap point estimate stored with each cell (probability space). The 95% CI is checked first, so a cell
whose 95% CI excludes 0 is never called equivalent (the old three-label rule checked the 90% CI first).

    python -m src.analysis.tier4_relabel        -> runs/rq1_tier4/tost_relabel.txt / .json
"""
from __future__ import annotations

import json

M = 0.05
OUT = "runs/rq1_tier4/tost_relabel"


def label(eff, c90, c95):
    if c95[0] > 0 or c95[1] < 0:
        return "different" if abs(eff) > M else "small but real"
    return "equivalent" if (c90[0] > -M and c90[1] < M) else "undecided"


def main():
    lines, rep = [], {}
    P = lines.append
    P("TOST RE-LABELLING (four labels; stored numbers only, nothing re-run)")
    P("  different = 95% CI excludes 0 and |effect| > 0.05; small but real = 95% CI excludes 0 and |effect| <= 0.05;")
    P("  equivalent = 95% CI includes 0 and 90% CI within +/-0.05; undecided = otherwise. Probability space, margin 0.05.")
    t3 = json.load(open("runs/rq1_revision/t3.json"))["tost"]
    l2 = json.load(open("runs/rq1_tier4/L2.json"))["fid"]
    cells = {"Tox21": [], "BBBP": []}
    for k, x in t3.items():
        ep, ex, met = k.split("|")
        cells["Tox21"].append((f"{ep} {ex} {met} R3", x["diff"], x["ci90"], x["ci95"], x["verdict"]))
    SH = {"gnnexplainer": "GNN", "pgexplainer": "PG (logit)", "sx_fixed": "SX fixed", "sal": "Saliency", "ig": "IG"}
    for k, x in l2.items():
        space, m, mt, r = k.split("|")
        if space != "prob":
            continue
        p, lo, hi = x["ci"]
        cells["BBBP"].append((f"{SH.get(m, m)} {'Fid+' if mt == 'fidp' else 'Fid-'} {r}", p, x["ci90"], [lo, hi], x["verdict"]))
    for ds, cs in cells.items():
        P(f"\n  {ds} ({len(cs)} cells)")
        P(f"  {'cell':<40}{'effect':>8}{'90% CI':>20}{'95% CI':>20}  {'old label':<12}new label")
        tally_old, tally_new, moved = {}, {}, []
        for name, eff, c90, c95, old in cs:
            new = label(eff, c90, c95)
            tally_old[old] = tally_old.get(old, 0) + 1; tally_new[new] = tally_new.get(new, 0) + 1
            if new != old:
                moved.append(f"{name}: {old} -> {new}")
            P(f"  {name:<40}{eff:>+8.3f}{f'[{c90[0]:+.3f}, {c90[1]:+.3f}]':>20}{f'[{c95[0]:+.3f}, {c95[1]:+.3f}]':>20}  {old:<12}{new}")
            rep.setdefault(ds, {})[name] = {"effect": eff, "ci90": c90, "ci95": c95, "old": old, "new": new}
        P("  old labels: " + ", ".join(f"{k} {v}" for k, v in sorted(tally_old.items())))
        P("  new labels: " + ", ".join(f"{k} {v}" for k, v in sorted(tally_new.items())))
        P(f"  cells whose label changed: {len(moved)}" + "".join(f"\n    {s}" for s in moved))
        rep.setdefault("tally", {})[ds] = {"old": tally_old, "new": tally_new, "changed": moved}
    open(f"{OUT}.txt", "w").write("\n".join(lines) + "\n")
    json.dump(rep, open(f"{OUT}.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
