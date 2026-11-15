#!/usr/bin/env python3
"""Compare result files side by side.

  python scripts/compare.py results/base.json results/base_privacy.json results/tuned.json \\
         --out results/comparison

Writes <out>.md (tables) and <out>.png (success vs clean success, plus a leakage/success
trade-off plot) when matplotlib is installed.
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dmin.personas import PII_FIELDS  # noqa: E402
from dmin.scoring import format_table  # noqa: E402

# Light-mode tokens from the validated reference palette (categorical slots 1 and 2).
SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"


def load(paths):
    out, seen = [], {}
    for p in paths:
        r = json.loads(Path(p).read_text(encoding="utf-8"))
        label = r["label"]
        seen[label] = seen.get(label, 0) + 1
        if seen[label] > 1:
            label = f"{label} ({Path(p).stem})"
        r["label"] = label
        out.append(r)
    return out


def markdown(results) -> str:
    L = ["# Comparison", "", "Data: " + ", ".join(sorted({r["data"] for r in results})), "",
         "**success** = right tool, right arguments, and every needed personal field present.  ",
         "**clean_succ** = success with no unneeded personal field sent.  ",
         "**any_leak** = share of calls that sent at least one unneeded personal field.  ",
         "**leaks/call** = mean number of unneeded personal fields per call.  ",
         "**over_redact** = among parsed calls on tasks that need personal data, share where some of it was left out.  ",
         "**cf_change** = share of calls that changed when only unneeded profile fields were swapped "
         "(counterfactual twins; shown when a twins file was used).", "",
         "## Overall", "", "```", format_table({r["label"]: r["summary"] for r in results}), "```", ""]
    variants = sorted({v for r in results for v in r["by_variant"]})
    for v in variants:
        rows = {r["label"]: r["by_variant"][v] for r in results if v in r["by_variant"]}
        L += [f"## Variant: {v}", "", "```", format_table(rows), "```", ""]
    fields = [f for f in PII_FIELDS if any(r["summary"]["per_field_leak"].get(f) for r in results)]
    if fields:
        L += ["## Unneeded fields leaked (share of calls)", "",
              "| agent | " + " | ".join(fields) + " |", "|---|" + "---|" * len(fields)]
        for r in results:
            pf = r["summary"]["per_field_leak"]
            L.append(f"| {r['label']} | " + " | ".join(f"{pf.get(f, 0):.0%}" for f in fields) + " |")
        L.append("")
    return "\n".join(L)


def chart(results, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [r["label"] for r in results]
    S = [r["summary"] for r in results]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.8), facecolor=SURFACE)
    for ax in (a1, a2):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(AXIS)
        ax.tick_params(colors=INK2, labelsize=9)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    # left: success vs clean success
    xs, w = range(len(labels)), 0.36
    a1.bar([x - 0.19 for x in xs], [s["success"] * 100 for s in S], w, color=BLUE,
           edgecolor=SURFACE, linewidth=0.8, label="Task success")
    a1.bar([x + 0.19 for x in xs], [s["clean_success"] * 100 for s in S], w, color=ORANGE,
           edgecolor=SURFACE, linewidth=0.8, label="Success with no unneeded personal data")
    a1.set_xticks(list(xs))
    a1.set_xticklabels(["\n".join(textwrap.wrap(l, 14)) for l in labels])
    a1.set_ylim(0, 108)
    a1.set_yticks(range(0, 101, 20))
    a1.set_ylabel("% of calls", color=INK2)
    a1.set_title("Success vs. clean success", loc="left", color=INK, fontsize=11, fontweight="bold")
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False,
              fontsize=9, labelcolor=INK2)

    # right: trade-off (one series, identity carried by direct labels)
    px, py = [s["any_leak_rate"] * 100 for s in S], [s["success"] * 100 for s in S]
    a2.scatter(px, py, s=90, color=BLUE, edgecolors=SURFACE, linewidths=1.5, zorder=3)
    for x, y, l in zip(px, py, labels):
        right = x > 60
        a2.annotate(l, (x, y), textcoords="offset points", xytext=(-9 if right else 9, 7),
                    ha="right" if right else "left", fontsize=9, color=INK)
    a2.set_xlim(-4, 104)
    a2.set_ylim(-4, 112)
    a2.set_yticks(range(0, 101, 20))
    a2.set_xlabel("Calls that leak at least one unneeded field (%)", color=INK2)
    a2.set_ylabel("Task success (%)", color=INK2)
    a2.set_title("Privacy vs. success trade-off (top-left is best)", loc="left", color=INK,
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+")
    ap.add_argument("--out", default="results/comparison")
    a = ap.parse_args()
    results = load(a.results)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    md = markdown(results)
    out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    try:
        chart(results, out.with_suffix(".png"))
        print(f"\nwrote {out.with_suffix('.md')} and {out.with_suffix('.png')}")
    except ImportError:
        print(f"\nwrote {out.with_suffix('.md')} (install matplotlib for the chart)")


if __name__ == "__main__":
    main()
