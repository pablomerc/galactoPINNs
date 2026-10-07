"""Tables and figures for loo_vs_chi.py.

Reads every shard in <results>/shards, recomputes the stopping rules from the
stored curves (and checks them against the epochs the run recorded), reads
each metric off the FULL trajectory at each rule's epoch, and writes:

  <results>/analysis/rows.csv          one row per (n, trial, rule)
  <results>/analysis/table.md          mean +- std over trials per (n, rule)
  <results>/analysis/paired.md         per-trial ratios rule / chi-stop
  <results>/analysis/fig_error_vs_n.png         final error vs n (the headline)
  <results>/analysis/fig_paired_ratio.png       rule / chi-stop per trial
  <results>/analysis/fig_stop_epochs.png        stopping epochs vs n
  <results>/analysis/fig_curves_n<n>_t<t>.png   CV / chi_fit / truth curves

Metrics (all on the 2-kpc Sun bubble):
  acc   mean |a_pred - a_true| / |a_true| over 2048 uniform-in-volume points,
        Sun-relative (gacc_b2: what the pulsars constrain). The absolute error
        (acc_abs, rows.csv only) also carries the model's Sun offset, which is
        large at early stops with the NFW baseline (~3-4 mm/s/yr at n=20)
  rho   <rho_tot> in the 2-kpc Sun sphere from Gauss's law, % error vs the
        particle truth; |error| in the figure, signed in the table, rho_DM
        error = the same flux minus the known baryons (table only)
  dex   median |log10 rho_pred / rho_true| over midplane pixels within 2 kpc

Run: uv run python clean_experiments/fire_sims_v2/analyze_loo_vs_chi.py [--results DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402

HERE = Path(__file__).resolve().parent
CHI_STAR = 0.7978845608
R_KEY = "2.0"

# house palette (accelerations paper); identity also carried by line style + marker
PAPER_PALETTE = ["#638ccc", "#c57c3c", "#ab62c0", "#72a555", "#ca5670"]
STYLE = {
    "chi":    dict(label=r"$\chi$-stop",  color=PAPER_PALETTE[0], ls="-",  marker="o"),
    "loo":    dict(label="LOO",           color=PAPER_PALETTE[1], ls="--", marker="s"),
    "iwloo":  dict(label="IW-LOO",        color=PAPER_PALETTE[2], ls="-.", marker="^"),
    "oracle": dict(label="oracle (truth)", color="#8a8a8a",       ls=":",  marker="D"),
}
RULES = ("chi", "loo", "loo1se", "iwloo", "iwloo1se", "oracle", "last")
METRICS = (("acc", "acceleration error [%]"),
           ("rho_abs", r"$|\Delta\langle\rho_{\rm tot}\rangle|$ [%]"),
           ("dex", "midplane density error [dex]"))


def cv_curves(Z, omega=None):
    """Same estimator as loo_vs_chi.cv_curves (kept import-free so this script
    runs without JAX)."""
    A = np.abs(np.asarray(Z, dtype=np.float64))
    if omega is None:
        return A.mean(1), A.std(1, ddof=1) / np.sqrt(A.shape[1])
    w = np.asarray(omega, dtype=np.float64)
    w = w / w.sum()
    cv = A @ w
    return cv, np.sqrt(((A - cv[:, None]) ** 2) @ (w ** 2))


def argmin_and_1se(cv, se):
    e_min = int(np.argmin(cv))
    return e_min, int(np.flatnonzero(cv <= cv[e_min] + se[e_min])[0])


def load_shard(path, truth):
    z = np.load(path, allow_pickle=False)
    cfg = json.loads(str(z["config"]))
    cols = [str(c) for c in z["full_cols"]]
    full = np.asarray(z["full"], dtype=np.float64)
    col = {c: full[:, i] for i, c in enumerate(cols)}
    n, E = z["Z"].shape[1], full.shape[0]

    rho = col["flux_b2"] * cfg["flux_to_rho"]
    t_tot, t_b, t_dm = truth["total"][0], truth["baryons"][0], truth["dark"][0]
    curves = dict(acc=100 * col["gacc_b2"],
                  acc_abs=100 * col["acc_b2"],
                  rho=100 * (rho - t_tot) / t_tot,
                  rho_dm=100 * (rho - t_b - t_dm) / t_dm,
                  dex=col["dex_b2"],
                  negfrac=col["negfrac_b2"],
                  chi_fit=col["chi_fit"])
    curves["rho_abs"] = np.abs(curves["rho"])

    chi = col["chi_fit"]
    below = np.flatnonzero(chi <= CHI_STAR)
    e = {"chi": int(below[0]) if below.size else E - 1, "last": E - 1}
    full_only = not np.isfinite(z["Z"]).all()
    omega = np.asarray(z["omega"], dtype=np.float64)
    if not full_only:
        cv, se = cv_curves(z["Z"])
        e["loo"], e["loo1se"] = argmin_and_1se(cv, se)
        if omega.sum() > 0:
            cvw, sew = cv_curves(z["Z"], omega)
            e["iwloo"], e["iwloo1se"] = argmin_and_1se(cvw, sew)
        else:
            cvw = np.full(E, np.nan)
            e["iwloo"], e["iwloo1se"] = e["loo"], e["loo1se"]
        for k in ("chi", "loo", "loo1se", "iwloo", "iwloo1se"):
            assert e[k] == int(z[f"e_{k}"]), f"{path.name}: rule {k} {e[k]} != stored {int(z[f'e_{k}'])}"
    else:
        cv = cvw = np.full(E, np.nan)
    n_eff = float(omega.sum() ** 2 / (omega ** 2).sum()) if omega.sum() > 0 else 0.0
    return dict(n=n, trial=int(path.stem.split("_t")[1]), E=E, e=e, curves=curves,
                cv=cv, cvw=cvw, crossed=bool(z["chi_crossed"]), full_only=full_only,
                n_in=int((omega > 0).sum()), n_eff=n_eff,
                seconds=float(z["seconds_folds"]) + float(z["seconds_full"]))


def rows_for(s):
    """Metric values at each rule's epoch; 'oracle' is per metric (argmin of
    that metric's own curve), so it is a floor, not a usable rule."""
    out = []
    for rule in RULES:
        if s["full_only"] and rule not in ("chi", "last", "oracle"):
            continue
        row = dict(n=s["n"], trial=s["trial"], rule=rule, chi_crossed=s["crossed"],
                   n_in=s["n_in"], n_eff=round(s["n_eff"], 2))
        for m, c in s["curves"].items():
            if rule == "oracle":
                cc = np.abs(c) if m in ("rho", "rho_dm") else c      # signed -> |.|
                ok = np.isfinite(cc)
                idx = int(np.flatnonzero(ok)[np.argmin(cc[ok])])
                row[m] = float(c[idx])
                row[f"epoch_{m}"] = idx
            else:
                row[m] = float(c[s["e"][rule]])
        row["epoch"] = row.get("epoch_acc") if rule == "oracle" else s["e"][rule]
        out.append(row)
    return out


def agg(rows, n, rule, key):
    v = np.array([r[key] for r in rows if r["n"] == n and r["rule"] == rule], dtype=float)
    v = v[np.isfinite(v)]
    return (v.mean(), v.std(ddof=1) if v.size > 1 else np.nan, np.median(v), v.size) \
        if v.size else (np.nan, np.nan, np.nan, 0)


def write_tables(rows, shards, out):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    ns = sorted({r["n"] for r in rows})
    lines = ["# LOO vs chi-stop: metrics at each rule's epoch (2-kpc Sun bubble)", "",
             "mean +- std over trials (median in brackets); oracle = per-metric argmin "
             "of the truth curve (a floor, not a rule).", "",
             "| n | rule | trials | epoch med | acc [%] | rho_tot err [%] (signed) "
             "| abs rho_tot err [%] | rho_DM err [%] (signed) | dex |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n in ns:
        for rule in RULES:
            if not any(r["n"] == n and r["rule"] == rule for r in rows):
                continue
            ep = [r["epoch"] for r in rows if r["n"] == n and r["rule"] == rule]
            cells = []
            for key, fmt in (("acc", "{:.2f}"), ("rho", "{:+.1f}"), ("rho_abs", "{:.1f}"),
                             ("rho_dm", "{:+.1f}"), ("dex", "{:.3f}")):
                mu, sd, med, _ = agg(rows, n, rule, key)
                cells.append(f"{fmt.format(mu)} +- {fmt.format(sd).lstrip('+')} "
                             f"[{fmt.format(med)}]")
            _, _, _, k = agg(rows, n, rule, "acc")
            lines.append(f"| {n} | {rule} | {k} | {int(np.median(ep))} | " + " | ".join(cells) + " |")
    lines += ["", "## Per-n diagnostics", "",
              "| n | chi crossed | IW: pulsars inside 2 kpc (mean) | IW n_eff (mean) "
              "| wall-clock per trial [s] (mean) |", "|---|---|---|---|---|"]
    for n in ns:
        ss = [s for s in shards if s["n"] == n]
        lines.append(f"| {n} | {sum(s['crossed'] for s in ss)}/{len(ss)} | "
                     f"{np.mean([s['n_in'] for s in ss]):.1f} | "
                     f"{np.mean([s['n_eff'] for s in ss]):.1f} | "
                     f"{np.mean([s['seconds'] for s in ss]):.0f} |")
    (out / "table.md").write_text("\n".join(lines) + "\n")

    lines = ["# Paired ratios rule / chi-stop (same trajectory, same trial)", "",
             "median [IQR] over trials; wins = trials where the rule is strictly better.", "",
             "| n | rule | metric | ratio | wins |", "|---|---|---|---|---|"]
    for n in ns:
        for rule in ("loo", "loo1se", "iwloo", "iwloo1se"):
            for key in ("acc", "rho_abs", "dex"):
                pairs = {}
                for r in rows:
                    if r["n"] == n and r["rule"] in (rule, "chi"):
                        pairs.setdefault(r["trial"], {})[r["rule"]] = r[key]
                rat = np.array([p[rule] / p["chi"] for p in pairs.values()
                                if rule in p and "chi" in p and p["chi"] > 0])
                if not rat.size:
                    continue
                q1, q2, q3 = np.percentile(rat, [25, 50, 75])
                lines.append(f"| {n} | {rule} | {key} | {q2:.2f} [{q1:.2f}, {q3:.2f}] | "
                             f"{int((rat < 1).sum())}/{rat.size} |")
    (out / "paired.md").write_text("\n".join(lines) + "\n")


def style_axes(ax, ns):
    ax.set_xscale("log")
    ax.set_xticks(ns)
    ax.set_xticklabels([str(n) for n in ns])
    ax.minorticks_off()
    ax.grid(True, color="#e6e6e6", lw=0.6)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_error_vs_n(rows, out):
    ns = sorted({r["n"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    dodge = {"chi": 0.97, "loo": 1.0, "iwloo": 1.03, "oracle": 1.0}
    for ax, (key, ylab) in zip(axes, METRICS):
        for rule, st in STYLE.items():
            stats = [agg(rows, n, rule, key) for n in ns]
            mu = np.array([s[0] for s in stats])
            sd = np.array([s[1] for s in stats])
            if not np.isfinite(mu).any():
                continue
            x = np.array(ns) * dodge[rule]
            ax.errorbar(x, mu, yerr=None if rule == "oracle" else sd, color=st["color"],
                        ls=st["ls"], marker=st["marker"], ms=6, lw=1.6, capsize=3,
                        elinewidth=1.1, label=st["label"])
        style_axes(ax, ns)
        ax.set_xlabel("number of pulsars n")
        ax.set_ylabel(ylab)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle(r"Final error at the stopping epoch, 2-kpc Sun bubble "
                 r"(mean $\pm$ std over trials; oracle without bars)", fontsize=11)
    fig.savefig(out / "fig_error_vs_n.png", dpi=170)
    plt.close(fig)


def fig_paired(rows, out):
    ns = sorted({r["n"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
    rng = np.random.default_rng(0)
    for ax, (key, ylab) in zip(axes, METRICS):
        for rule, off in (("loo", 0.95), ("iwloo", 1.05)):
            st = STYLE[rule]
            med = []
            for n in ns:
                pairs = {}
                for r in rows:
                    if r["n"] == n and r["rule"] in (rule, "chi"):
                        pairs.setdefault(r["trial"], {})[r["rule"]] = r[key]
                rat = np.array([p[rule] / p["chi"] for p in pairs.values()
                                if rule in p and "chi" in p and p["chi"] > 0])
                med.append(np.median(rat) if rat.size else np.nan)
                if rat.size:
                    xj = n * off * np.exp(rng.normal(0, 0.015, rat.size))
                    ax.scatter(xj, rat, s=14, color=st["color"], alpha=0.45, lw=0,
                               marker=st["marker"])
            ax.plot(np.array(ns) * off, med, color=st["color"], ls=st["ls"],
                    marker=st["marker"], ms=6, lw=1.6, label=f"{st['label']} / " + r"$\chi$-stop")
        ax.axhline(1.0, color="#555555", lw=0.9)
        ax.set_yscale("log")
        style_axes(ax, ns)
        ax.set_xlabel("number of pulsars n")
        ax.set_ylabel(f"ratio, {ylab.split(' [')[0]}")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Paired per-trial ratio (points) and median (line); below 1 = better "
                 r"than $\chi$-stop", fontsize=11)
    fig.savefig(out / "fig_paired_ratio.png", dpi=170)
    plt.close(fig)


def fig_epochs(rows, out):
    ns = sorted({r["n"] for r in rows})
    fig, ax = plt.subplots(figsize=(5.6, 3.9), constrained_layout=True)
    for rule, st in STYLE.items():
        q = []
        for n in ns:
            ep = [r["epoch"] for r in rows if r["n"] == n and r["rule"] == rule]
            q.append(np.percentile(ep, [25, 50, 75]) if ep else [np.nan] * 3)
        q = np.array(q)
        if not np.isfinite(q).any():
            continue
        ax.errorbar(ns, q[:, 1] + 1, yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
                    color=st["color"], ls=st["ls"], marker=st["marker"], ms=6, lw=1.6,
                    capsize=3, label=st["label"] + (" (acc)" if rule == "oracle" else ""))
    style_axes(ax, ns)
    ax.set_yscale("log")
    ax.set_xlabel("number of pulsars n")
    ax.set_ylabel("stopping epoch + 1 (median, IQR)")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(out / "fig_stop_epochs.png", dpi=170)
    plt.close(fig)


def fig_curves(s, out):
    ep = np.arange(s["E"]) + 1
    c = s["curves"]
    fig, axes = plt.subplots(3, 1, figsize=(6.4, 7.6), sharex=True, constrained_layout=True)
    ax = axes[0]
    ax.plot(ep, c["chi_fit"], color=STYLE["chi"]["color"], lw=1.4,
            label=r"train $\chi_{\rm fit}$ = mean $|r|/\sigma$")
    ax.plot(ep, s["cv"], color=STYLE["loo"]["color"], lw=1.4, ls="--", label="LOO CV")
    ax.plot(ep, s["cvw"], color=STYLE["iwloo"]["color"], lw=1.4, ls="-.", label="IW-LOO CV")
    ax.axhline(CHI_STAR, color="#555555", lw=0.8, ls=":")
    ax.text(ep[-1], CHI_STAR, r"$\sqrt{2/\pi}$", va="bottom", ha="right", fontsize=8,
            color="#555555")
    ax.set_ylabel(r"whitened misfit $|r|/\sigma$")
    ax.legend(frameon=False, fontsize=8)
    axes[1].plot(ep, c["acc"], color="#333333", lw=1.3)
    axes[1].set_ylabel("acceleration error [%]")
    axes[2].plot(ep, c["rho_abs"], color="#333333", lw=1.3)
    axes[2].set_ylabel(r"$|\Delta\langle\rho_{\rm tot}\rangle|$ [%]")
    for a in axes:
        for rule in ("chi", "loo", "iwloo"):
            a.axvline(s["e"][rule] + 1, color=STYLE[rule]["color"], ls=STYLE[rule]["ls"], lw=1.1)
        a.set_xscale("log")
        a.grid(True, color="#e6e6e6", lw=0.6)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)
    axes[1].axvline(int(np.argmin(c["acc"])) + 1, color=STYLE["oracle"]["color"], ls=":", lw=1.1)
    axes[2].set_xlabel("epoch")
    axes[0].set_title(f"n = {s['n']}, trial {s['trial']} (vertical lines: stopping epochs)",
                      fontsize=10)
    fig.savefig(out / f"fig_curves_n{s['n']}_t{s['trial']}.png", dpi=170)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(HERE / "results_loo_vs_chi"))
    ap.add_argument("--density-truth", default=str(HERE / "loo_inputs" / "density_truth_bubbles.json"))
    ap.add_argument("--examples", default="20,50,500", help="n values for the curve figures (trial 0)")
    args = ap.parse_args()

    res = Path(args.results)
    out = res / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    truth = json.loads(Path(args.density_truth).read_text())["R_grid"][R_KEY]
    shards = [load_shard(p, truth) for p in sorted((res / "shards").glob("n*_t*.npz"))]
    if not shards:
        raise SystemExit(f"no shards in {res / 'shards'}")
    rows = [r for s in shards for r in rows_for(s)]
    print(f"{len(shards)} shards, n = {sorted({s['n'] for s in shards})}, "
          f"trials per n = {[sum(s['n'] == n for s in shards) for n in sorted({s['n'] for s in shards})]}")

    write_tables(rows, shards, out)
    fig_error_vs_n(rows, out)
    if not all(s["full_only"] for s in shards):
        fig_paired(rows, out)
    fig_epochs(rows, out)
    ex = {int(v) for v in args.examples.split(",") if v}
    for s in shards:
        if s["n"] in ex and s["trial"] == 0 and not s["full_only"]:
            fig_curves(s, out)
    print((out / "table.md").read_text())
    print(f"wrote {out}/")


if __name__ == "__main__":
    main()
