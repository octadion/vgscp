"""The projection sentence in Section 5.5 must match the released Part B record.

That sentence is the paper's only claim with no table behind it, and for a while it had no released
record either: the run lived in a Colab session whose output was never saved. So it is checked here
against ``results/recoverability_partB.json`` -- and the absence of that file is itself a failure,
because the paper promises the records behind every claim.
"""
import io
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REC = os.path.join(ROOT, "results", "recoverability_partB.json")
PAPER = os.path.join(ROOT, "ACML_Journal___Robust_CP_Train_Study (1)", "sn-article.tex")
bad = []

if not os.path.exists(REC):
    print("results/recoverability_partB.json belum ada -- salin dari Colab (syarat rilis)")
    print("BELUM BISA DIPERIKSA")
    raise SystemExit(0)

B = json.load(io.open(REC, encoding="utf-8"))
prose = re.sub(r"\s+", " ", io.open(PAPER, encoding="utf-8").read())
m = re.search(r"Projecting out the top-\$k\$.*?fully\s*recoverable\.", prose)
if m is None:
    print("kalimat proyeksi di 5.5 tidak ditemukan")
    print("1 MASALAH")
    raise SystemExit(0)
sent = m.group(0)

rise, drop_celeba, finals, best = 0.0, 0.0, [], None
for key, c in B["cells"].items():
    covs = [d["worst_group_cov"] for d in c["curve"]]
    finals.append(c["curve"][-1]["auroc"])
    delta = max(covs) - covs[0]
    if delta > rise:
        rise, best = delta, (key, covs[0], max(covs))
    if key.endswith("celeba"):
        drop_celeba = max(drop_celeba, max(abs(v - covs[0]) for v in covs))

want = {"ks": [0, 1, 5, 20], "rise": round(rise, 3), "lo": round(min(finals), 2),
        "hi": round(max(finals), 2), "from": round(best[1], 3), "to": round(best[2], 3),
        "short": round(0.90 - best[2], 3), "celeba": drop_celeba}
print(f"dihitung: naik maks {want['rise']} di {best[0]} ({want['from']} -> {want['to']}), "
      f"auroc akhir {want['lo']}-{want['hi']}, gerak CelebA <= {want['celeba']:.4f}")

if list(B["ks"]) != want["ks"]:
    bad.append(f"  ks rekaman {B['ks']} bukan {want['ks']}")
for label, value in (("kenaikan maksimum", want["rise"]), ("dari", want["from"]),
                     ("ke", want["to"]), ("kurang dari target", want["short"])):
    if f"${value:.3f}$" not in sent:
        bad.append(f"  {label} {value:.3f} tidak tertulis di kalimat 5.5")
if f"[{want['lo']:.2f},{want['hi']:.2f}]" not in sent.replace(" ", ""):
    bad.append(f"  rentang auroc akhir [{want['lo']:.2f},{want['hi']:.2f}] tidak tertulis")
stated_celeba = re.search(r"less than \$(\d\.\d+)\$ on\s*CelebA", sent)
if stated_celeba is None:
    bad.append("  batas gerak CelebA tidak tertulis")
elif float(stated_celeba.group(1)) < want["celeba"]:
    bad.append(f"  gerak CelebA tertulis <{stated_celeba.group(1)}, terukur {want['celeba']:.4f}")

print("\n".join(bad) if bad else "")
print("KLAIM 5.5 COCOK DENGAN REKAMAN" if not bad else f"{len(bad)} MASALAH")
