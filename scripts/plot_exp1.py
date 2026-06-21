import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open("results/exp1/sweep.json"))
os.makedirs("results/exp1/figs", exist_ok=True)
os.makedirs("results/exp1/diagnostics", exist_ok=True)
plt.figure(figsize=(7,5))
curves = ["h128_bits4", "h128_bits3", "h512_bits4", "h512_bits3"]
for key in curves:
    v = d.get(key)
    if not v or not v.get("kl_mean"):
        continue
    kl = v["kl_mean"]
    steps = v.get("steps") or list(range(1, len(kl) + 1))
    plt.plot(steps, kl, marker="o", ms=3, label=key)
plt.xlabel("decode step")
plt.ylabel("mean KL (quant vs fp16)")
plt.title("Recurrent-state quant KL compounding (Mamba-2 1.3B)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("results/exp1/figs/exp1_kl_growth.png", dpi=140)
plt.close()
bits_order = [8, 4, 3]
hors = [128, 512]
colors = ["tab:blue", "tab:red"]
fig, ax = plt.subplots(figsize=(7,5))
for i in range(len(hors)):
    H = hors[i]
    ys = []
    for b in bits_order:
        key = "h" + str(H) + "_bits" + str(b)
        v = d.get(key)
        kl = v.get("kl_mean") if v else None
        ys.append(kl[-1] if kl else 0.0)
    ax.plot(bits_order, ys, marker="s", color=colors[i], label="H=" + str(H))
ax.set_xticks(bits_order); ax.invert_xaxis(); ax.set_yscale("log")
ax.set_xlabel("quantization bits")
ax.set_ylabel("terminal mean KL")
ax.set_title("Terminal KL vs bit-width and horizon")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig("results/exp1/figs/exp1_terminal_kl.png", dpi=140)
plt.close()
summary = {}
for key in d:
    v = d[key]
    kl = v.get("kl_mean") or []
    summary[key] = {"slope_b": v.get("b"), "r2": v.get("r2"), "kl_terminal": (kl[-1] if kl else 0.0), "n_steps": len(kl)}
json.dump(summary, open("results/exp1/diagnostics/horizon_crossover.json", "w"), indent=2)
print("WROTE figs + horizon_crossover.json")
for k in sorted(summary):
    print(k, summary[k])
