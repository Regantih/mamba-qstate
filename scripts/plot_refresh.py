import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open("results/exp1/refresh.json"))
os.makedirs("results/exp1/figs", exist_ok=True)
os.makedirs("results/exp1/diagnostics", exist_ok=True)
order = ["refresh0", "refresh16", "refresh64"]
labels = {"refresh0": "k=0 (none)", "refresh16": "k=16", "refresh64": "k=64"}
plt.figure(figsize=(7,5))
for key in order:
    v = d.get(key)
    if not v or not v.get("kl_mean"):
        continue
    kl = v["kl_mean"]
    steps = v.get("steps") or list(range(1, len(kl) + 1))
    plt.plot(steps, kl, marker="o", ms=3, label=labels.get(key, key))
plt.yscale("log")
plt.xlabel("decode step")
plt.ylabel("mean KL (quant vs fp16)")
plt.title("Refresh containment: 4-bit state, H=512 (Mamba-2 1.3B)")
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("results/exp1/figs/refresh_kl.png", dpi=140)
plt.close()
ks = [0, 16, 64]
bs = []
kts = []
for k in ks:
    v = d.get("refresh" + str(k))
    bs.append(v["b"] if v else float("nan"))
    kl = v.get("kl_mean") if v else None
    kts.append(kl[-1] if kl else float("nan"))
fig, ax1 = plt.subplots(figsize=(7,5))
ax1.plot(ks, bs, marker="s", color="tab:blue")
ax1.axhline(0, color="gray", lw=0.8, ls="--")
ax1.set_xlabel("refresh interval k (steps)")
ax1.set_ylabel("growth exponent b", color="tab:blue")
ax2 = ax1.twinx()
ax2.plot(ks, kts, marker="o", color="tab:red")
ax2.set_yscale("log")
ax2.set_ylabel("terminal mean KL", color="tab:red")
plt.title("Refresh interval vs compounding (4-bit, H=512)")
fig.tight_layout()
plt.savefig("results/exp1/figs/refresh_b_vs_k.png", dpi=140)
plt.close()
summary = {}
for key in d:
    v = d[key]
    kl = v.get("kl_mean") or []
    klt = kl[-1] if kl else 0.0
    summary[key] = {"refresh": v.get("refresh"), "slope_b": v.get("b"), "r2": v.get("r2"), "kl_terminal": klt, "n_steps": len(kl)}
json.dump(summary, open("results/exp1/diagnostics/refresh_summary.json", "w"), indent=2)
print("WROTE refresh figs + refresh_summary.json")
for k in sorted(summary):
    print(k, summary[k])
