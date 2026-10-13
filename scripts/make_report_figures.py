"""Builds every figure used in report.md (saved to figures/) plus figures/analysis.json.

Run from the repository root:   python scripts/make_report_figures.py
Reads only saved result files: no models, no GPU, no network.
"""
import collections, json, os, re, statistics as st
from math import comb, sqrt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch

OUT, G = "figures", "results/task6_generations"
os.makedirs(OUT, exist_ok=True)

# ---- one visual system for every figure
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8981"
WIN, TIE, LOSS = "#2a78d6", "#c3c2b7", "#e34948"
SURF, INK, INK2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.edgecolor": "#9a998f", "axes.labelcolor": INK, "axes.labelsize": 9,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8.5,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "lines.linewidth": 2,
})
def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=180, bbox_inches="tight", pad_inches=0.08); plt.close(fig)

load = lambda p: json.load(open(p))
jsonl = lambda p: [json.loads(l) for l in open(p)]
words = lambda t: len(t.split())

def rolling(xs, k):
    return [st.mean(xs[max(0, i - k + 1):i + 1]) for i in range(len(xs))]

def mark_best(ax, step, label, y=0.04):
    ax.axvline(step, color=INK2, lw=1, ls="--", zorder=1)
    ax.annotate(label, xy=(step, y), xycoords=("data", "axes fraction"), xytext=(4, 0), textcoords="offset points",
                fontsize=7.5, color=INK2, ha="left")

# =====================================================================  Figure 1: pipeline overview
fig, ax = plt.subplots(figsize=(10.5, 2.9)); ax.set_xlim(0, 112); ax.set_ylim(0, 30); ax.axis("off")
def box(x, y, w, h, title, sub, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=color + "1f", ec=color, lw=1.6))
    ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=8.6, fontweight="bold", color=INK)
    ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center", fontsize=7.2, color=INK2)
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.3, shrinkA=0, shrinkB=0))
box(1, 17, 18, 8, "Qwen2.5-0.5B", "pretrained base", GRAY)
box(25, 17, 18, 8, "SFT (own loop)", "30k UltraChat, 1 epoch", BLUE)
box(50, 22.5, 20, 6.5, "HH-RLHF DPO", "15k pairs, β 0.1, lr 1e-6", ORANGE)
box(50, 12.5, 20, 6.5, "UltraFeedback DPO", "15k pairs, β 0.1, lr 1e-6", AQUA)
box(76, 22.5, 20, 6.5, "HH safety set", "300 prompts, vs SFT", ORANGE)
box(76, 12.5, 20, 6.5, "AlpacaEval", "300 prompts (seed 42), vs SFT", AQUA)
box(1, 2, 18, 7, "SmolLM2-360M", "Instruct model", GRAY)
box(25, 2, 18, 7, "LoRA-DPO (TRL)", "200 persona pairs, r 16", "#6250d6")
box(76, 2, 20, 7, "Persona style set", "12 prompts, vs base", "#6250d6")
arrow(19.6, 21, 24.4, 21); arrow(43.6, 22, 49.4, 25.5); arrow(43.6, 20, 49.4, 15.8)
arrow(70.6, 25.7, 75.4, 25.7); arrow(70.6, 15.7, 75.4, 15.7); arrow(19.6, 5.5, 24.4, 5.5); arrow(43.6, 5.5, 75.4, 5.5)
ax.text(105, 15.5, "Blind A/B judge\ngemini-3.6-flash\nbatches of 5,\nretries, resume", ha="center", va="center", fontsize=7.6,
        color=INK, bbox=dict(boxstyle="round,pad=0.6", fc="#f4f3ef", ec="#9a998f"))
for y in (25.7, 15.7, 5.5): arrow(96.8, y, 98.6, 15.5 + (y - 15.5) * 0.35)
ax.text(46.5, 10.2, "reference = frozen copy of SFT best", fontsize=7.2, color=INK2, ha="center", style="italic")
save(fig, "fig1_pipeline.png")

# =====================================================================  Figure 2: SFT
m = jsonl("results/task3_sft/metrics.jsonl")
tr = [r for r in m if "train_loss" in r]; va = [r for r in m if "val_loss" in r]
best = min(va, key=lambda r: r["val_loss"])
fig, ax = plt.subplots(figsize=(7.2, 2.9))
xs, ys = [r["step"] for r in tr], [r["train_loss"] for r in tr]
ax.plot(xs, ys, color=BLUE, alpha=0.28, lw=1.2, label="train (mean of each 10 steps)")
ax.plot(xs, rolling(ys, 5), color=BLUE, label="train (50-step rolling mean)")
ax.plot([r["step"] for r in va], [r["val_loss"] for r in va], "o-", color=ORANGE, ms=5, label="validation (1,000 ex.)")
mark_best(ax, best["step"], f"best = final\nstep {best['step']}\nval {best['val_loss']:.4f}", y=0.72)
ax.axvline(28, color=GRAY, lw=0.9, ls=":"); ax.text(34, 1.468, "warmup ends (step 28),\nthen cosine decay", fontsize=7.5, color=INK2, va="top")
ax.set(xlabel="optimizer step (effective batch 32)", ylabel="loss per response token", title="SFT on Qwen2.5-0.5B")
ax.legend(loc="lower left", ncol=1); save(fig, "fig2_sft.png")

# =====================================================================  Figure 3: both DPO runs
fig, axes = plt.subplots(2, 3, figsize=(10, 5.0), sharex=True)
for row, (ds, name, col) in enumerate([("hh", "HH-RLHF", ORANGE), ("uf", "UltraFeedback", AQUA)]):
    m = jsonl(f"results/task4_dpo_{ds}/metrics.jsonl")
    tr = [r for r in m if "train_loss" in r]; va = [r for r in m if "val_loss" in r]
    b = max(va, key=lambda r: r["val_reward_acc"])
    for c, (key, title) in enumerate([("loss", "DPO loss"), ("reward_acc", "Reward accuracy"), ("reward_margin", "Reward margin")]):
        ax = axes[row, c]
        ax.plot([r["step"] for r in tr], [r[f"train_{key}"] for r in tr], color=BLUE, alpha=0.45, lw=1.3, label="train (10-step mean)")
        ax.plot([r["step"] for r in va], [r[f"val_{key}"] for r in va], "o-", color=ORANGE, ms=4, label="validation (1,000 pairs)")
        mark_best(ax, b["step"], f"best: step {b['step']}", y=0.04 if key != "loss" else 0.55)
        ax.set_title(f"{name}: {title}")
        if key == "loss": ax.axhline(0.6931, color=GRAY, lw=0.8, ls=":"); ax.text(15, 0.6931, "ln 2", fontsize=7, color=INK2, va="bottom", ha="left")
        if row == 1: ax.set_xlabel("optimizer step (effective batch 32)")
h_, l_ = axes[0, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.035), fontsize=8.5)
fig.tight_layout(h_pad=1.2); save(fig, "fig3_dpo.png")

# =====================================================================  Figure 4: LoRA
h = jsonl("results/task5_lora/metrics.jsonl")
tr = [r for r in h if "loss" in r and "rewards/accuracies" in r]; ev = [r for r in h if "eval_loss" in r]
fig, axes = plt.subplots(1, 3, figsize=(10, 2.6))
for ax, key, title in zip(axes, ["loss", "rewards/accuracies", "rewards/margins"], ["DPO loss", "Reward accuracy", "Reward margin"]):
    ax.plot([r["step"] for r in tr], [r[key] for r in tr], "o-", color=BLUE, ms=3.5, alpha=0.75, label="train (every 5 steps)")
    ax.plot([r["step"] for r in ev], [r[f"eval_{key}"] for r in ev], "s-", color=ORANGE, ms=4.5, label="eval (20 held-out pairs)")
    for e in (25, 50): ax.axvline(e, color=GRID, lw=1.2, zorder=0)
    ax.set_title(f"LoRA-DPO: {title}"); ax.set_xlabel("step (25 steps = 1 epoch)")
axes[0].legend(loc="upper right", fontsize=7.5); axes[1].set_ylim(0.3, 1.05)
fig.tight_layout(); save(fig, "fig4_lora.png")

# =====================================================================  judge-based figures
gen = {k: {r["id"]: r for r in load(f"{G}/{k}.json")} for k in ["sft_alpaca", "dpo_uf_alpaca", "sft_hh", "dpo_hh_hh"]}
gen["base_alpaca"] = {r["id"]: r for r in load("results/task2_baseline/base_generations.json")}
hh_eval = {r["id"]: r for r in load("eval_sets/hh_rlhf_eval_300_seed42.json")}
S = load("results/task7_judge/summary.json")
def outcomes(name, target):
    return {r["id"]: ("win" if r["winner_model"] == target else "tie" if r["judge_winner"] == "tie" else "loss")
            for r in jsonl(f"results/task7_judge/judged_{name}.jsonl")}
res = {"dpo_uf_vs_sft": outcomes("dpo_uf_vs_sft", "qwen2.5-0.5b-dpo_uf"), "dpo_hh_vs_sft": outcomes("dpo_hh_vs_sft", "qwen2.5-0.5b-dpo_hh")}

def stacked(ax, labels, rows, title):
    for y, (w, t, l) in enumerate(rows):
        n, left = w + t + l, 0
        for v, c, tc in ((w, WIN, "white"), (t, TIE, INK), (l, LOSS, "white")):
            ax.barh(y, v / n * 100, left=left, color=c, edgecolor=SURF, linewidth=2, height=0.66)
            if v / n > 0.06: ax.text(left + v / n * 50, y, str(v), ha="center", va="center", color=tc, fontsize=8.5, fontweight="bold")
            left += v / n * 100
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, color=INK); ax.invert_yaxis(); ax.grid(False)
    ax.axvline(50, color=INK2, lw=0.8, ls=":"); ax.set_xlim(0, 100); ax.set_xlabel("% of prompts"); ax.set_title(title)
    ax.legend(handles=[Patch(color=WIN, label="DPO / LoRA wins"), Patch(color=TIE, label="tie"), Patch(color=LOSS, label="DPO / LoRA loses")],
              loc="upper center", bbox_to_anchor=(0.5, -0.30 if len(labels) < 5 else -0.13), ncol=3, fontsize=8)

lab = {"dpo_uf_vs_sft": "UF-DPO vs SFT\nAlpacaEval (n=300)", "dpo_hh_vs_sft": "HH-DPO vs SFT\nHH safety set (n=300)",
       "persona_lora_vs_base": "LoRA-DPO vs base\npersona set (n=12)"}
fig, ax = plt.subplots(figsize=(7.6, 2.5))
stacked(ax, list(lab.values()), [(S[k]["win"], S[k]["tie"], S[k]["loss"]) for k in lab], "Pairwise judge outcomes")
save(fig, "fig5_judge_summary.png")

fig, ax = plt.subplots(figsize=(10, 2.9)); w = 0.36
buckets = ["DPO shorter (>20 words)", "similar length (±20 words)", "DPO longer (>20 words)"]
length_tab = {}
for off, name, ga, gb, col, lbl in ((-w / 2, "dpo_uf_vs_sft", "dpo_uf_alpaca", "sft_alpaca", AQUA, "UF-DPO vs SFT (AlpacaEval)"),
                                     (w / 2, "dpo_hh_vs_sft", "dpo_hh_hh", "sft_hh", ORANGE, "HH-DPO vs SFT (HH set)")):
    c = [collections.Counter() for _ in buckets]
    for i, o in res[name].items():
        d = words(gen[ga][i]["output"]) - words(gen[gb][i]["output"])
        c[0 if d < -20 else 2 if d > 20 else 1][o] += 1
    length_tab[name] = [dict(x) for x in c]
    vals = [x["win"] / max(1, x["win"] + x["loss"]) for x in c]
    bars = ax.bar([i + off for i in range(3)], vals, w - 0.04, color=col, label=lbl, edgecolor=SURF, linewidth=2)
    for b_, v, x in zip(bars, vals, c):
        ax.text(b_.get_x() + b_.get_width() / 2, v + 0.015, f"{v:.2f}\n{x['win']}W / {x['loss']}L", ha="center", va="bottom", fontsize=7.6, color=INK)
ax.axhline(0.5, color=INK2, lw=0.8, ls=":"); ax.text(2.55, 0.5, "parity", fontsize=7.5, color=INK2, va="bottom")
ax.set_ylim(0, 0.98); ax.set_xticks(range(3)); ax.set_xticklabels(buckets, color=INK); ax.grid(axis="x", visible=False)
ax.set_ylabel("DPO win rate (ties excluded)"); ax.set_title("DPO win rate vs. length difference (DPO words − SFT words, same prompt)")
ax.legend(loc="upper right"); save(fig, "fig7_winrate_by_length.png")

cat = collections.defaultdict(collections.Counter)
for i, o in res["dpo_hh_vs_sft"].items(): cat[hh_eval[i]["category"]][o] += 1
order = sorted(cat, key=lambda c: -(cat[c]["win"] / max(1, cat[c]["win"] + cat[c]["loss"])))
fig, ax = plt.subplots(figsize=(7.6, 4.0))
stacked(ax, [c.replace("_", " ") for c in order], [(cat[c]["win"], cat[c]["tie"], cat[c]["loss"]) for c in order],
        "HH-DPO vs SFT by category (30 prompts each)")
save(fig, "fig8_hh_by_category.png")

# =====================================================================  Figure 8: persona style score
sty = load("results/task5_lora/config.json")["style_summary"]
cats = ["cooking", "coding", "general", "safety"]
fig, ax = plt.subplots(figsize=(6.4, 2.5)); w = 0.36
for off, key, col, lbl in ((-w / 2, "base_style", GRAY, "base SmolLM2-360M-Instruct"), (w / 2, "lora_style", "#6250d6", "LoRA-DPO (Ramsay persona)")):
    vals = [sty[c][key] for c in cats]
    bars = ax.bar([i + off for i in range(4)], vals, w - 0.04, color=col, label=lbl, edgecolor=SURF, linewidth=2)
    for b_, v in zip(bars, vals): ax.text(b_.get_x() + b_.get_width() / 2, v + 0.08, f"{v:.2f}", ha="center", fontsize=7.8, color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels([c.capitalize() for c in cats], color=INK); ax.grid(axis="x", visible=False)
ax.set_ylabel("mean style score\n(persona words + '!')"); ax.set_ylim(0, 5.6); ax.set_title("Persona style score per category (3 prompts each)")
ax.legend(loc="upper left", ncol=2); save(fig, "fig6_persona_style.png")

# =====================================================================  numbers quoted in the report
REFUSE = re.compile(r"\b(i'?m sorry|i am sorry|i can(?:no|')t|i cannot|i won'?t|i'm not able|i am not able|unable to (?:help|assist|provide)"
                    r"|as an ai|not appropriate|i must decline|i do not feel comfortable|i don't feel comfortable|illegal|unethical)\b", re.I)
def looping(t):
    w_ = t.lower().split()
    if len(w_) < 20: return False
    g = [tuple(w_[i:i + 4]) for i in range(len(w_) - 3)]
    return 1 - len(set(g)) / len(g) > 0.3
likely_cut = lambda t: words(t) > 150 and not t.rstrip().endswith((".", "!", "?", '"', ")", "`", ":", "*"))
sign_p = lambda w_, l: min(1.0, 2 * sum(comb(w_ + l, i) for i in range(min(w_, l) + 1)) / 2 ** (w_ + l))
def wilson(w_, n, z=1.96):
    p = w_ / n; d = 1 + z * z / n; c = p + z * z / (2 * n); r = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - r) / d, 3), round((c + r) / d, 3)]
allj = sum((jsonl(f"results/task7_judge/judged_{k}.jsonl") for k in lab), [])
pos = collections.Counter(r["judge_winner"] for r in allj)
analysis = {
    "generation_stats": {k: {"mean_words": round(st.mean(words(r["output"]) for r in d.values()), 1),
                             "median_words": st.median(words(r["output"]) for r in d.values()),
                             "looping": sum(looping(r["output"]) for r in d.values()),
                             "likely_cut_at_cap": sum(likely_cut(r["output"]) for r in d.values()),
                             "refusal_phrase": sum(bool(REFUSE.search(r["output"])) for r in d.values())} for k, d in gen.items()},
    "judge": {k: {"win_rate_no_ties": round(S[k]["win"] / (S[k]["win"] + S[k]["loss"]), 3),
                  "ci95": wilson(S[k]["win"], S[k]["win"] + S[k]["loss"]), "sign_test_p": round(sign_p(S[k]["win"], S[k]["loss"]), 4)} for k in lab},
    "position_counts": dict(pos), "position_A_share_of_decided": round(pos["A"] / (pos["A"] + pos["B"]), 3),
    "hh_unique_prompts": len({x["instruction"] for x in hh_eval.values()}),
    "hh_by_category": {c: dict(cat[c]) for c in order}, "length_buckets": length_tab,
}
json.dump(analysis, open(f"{OUT}/analysis.json", "w"), indent=2)
print("saved:", sorted(os.listdir(OUT)))
