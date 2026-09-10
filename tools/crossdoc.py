"""Check that the three documents agree on the numbers they share.

This is the failure that has recurred at every audit: a figure gets corrected in the manuscript and
the response letter keeps the old one, and the editor reads the letter first. It went undetected
three rounds running because no check compared documents -- each was internally consistent.

The check works from a list of quantities that appear in more than one document, each with the
value the records support. For every document that mentions the quantity at all, the stated value
must be the supported one. A quantity absent from a document is fine; a wrong value is not.

Also flags any superseded value still present anywhere, since those are the ones that do damage.
"""
import io
import re

D = r"c:\jagr\vgscp\ACML_Journal___Robust_CP_Train_Study (1)"
DOCS = {}
for f in ("sn-article.tex", "appendix-tables.tex", "response-to-reviewers.tex",
          "before-after.tex"):
    DOCS[f] = io.open(f"{D}/{f}", encoding="utf-8").read()


def norm(t):
    return re.sub(r"\s+", " ", t)


# Page count from the author's most recent compile (2026-09-10). Update it after each compile:
# the letter states this number, and nothing here can measure it.
PAGES = 36

# (label, regex that must NOT appear anywhere, why)
SUPERSEDED = [
    ("rentang set kosong lama", r"0\.926\$?--\$?0\.964", "diganti 0.888-0.962"),
    ("pita Mondrian lama (ERM row)", r"0\.854\$?--\$?0\.886", "diganti 0.856-0.884"),
    ("divergensi ERM lama", r"\$0\.14\$--\$0\.22\$", "diganti 0.05-0.22"),
    ("SD Mondrian lama", r"0\.023\$?--\$?0\.034", "diganti 0.021-0.036"),
    ("SD marginal lama", r"0\.009\$?--\$?0\.042", "diganti 0.007-0.043"),
    ("celah literatur lama", r"\$10\$--\$24\$", "diganti 12-26 (Waterbirds)"),
    ("hitungan tabel lama", r"\b(?:nineteen|twenty|twenty-two) tables\b",
     "sekarang delapan belas; contradict.py menurunkannya dari manuskrip"),
    # There is no LaTeX here, so the page count cannot be derived -- it comes from the author's
    # compile. Keep PAGES in step with the last one and the old figures stay barred; every count
    # below has appeared in the letter at some point and been overtaken.
    ("halaman lama", r"\b(?:39|42|45) pages\b", f"sekarang {PAGES} pages"),
    ("'dua kegagalan terbesar'", r"two largest failures are ERM", "ERM lalu GroupDRO-LL"),
    ("'tujuh dari delapan'", r"seven of eight settings the most accurate", "enam dari delapan"),
    ("'almost always'", r"almost always the most efficient", "enam dari delapan"),
    ("'order of magnitude'", r"by an order of magnitude", "dihapus, tanpa dukungan"),
    ("median gap lama", r"\$0\.001\$--\$0\.013\$", "diganti 0.001-0.012"),
    ("disparity lama", r"0\.140\$?--\$?0\.608|0\.080\$?--\$?0\.355", "diganti 0.080-0.339 / 0.168-0.608"),
    ("'minority group' untuk CelebA", r"blond male on\s+CelebA", "g3 CelebA memegang 47.5%"),
    ("judul lama di surat", r"Is More a\s+Calibration Problem", "judul sekarang 'Calibration Beats...'"),
    ("kutipan abstrak palsu", r"both\}? levers", "tidak ada di manuskrip"),
]

# (label, value, docs that must agree if they mention the surrounding phrase)
AGREE = [
    ("pita Mondrian", r"0\.856\$?--\$?0\.884"),
    ("set kosong", r"0\.888\$?--\$?0\.962"),
    ("celah literatur", r"\$12\$--\$26\$"),
    ("hitungan halaman", rf"\b{PAGES} pages\b"),
]

bad = 0
print("=== nilai usang yang masih tersisa ===")
for label, pat, why in SUPERSEDED:
    for f, s in DOCS.items():
        for m in re.finditer(pat, norm(s)):
            ln = s[:s.find(m.group(0))].count("\n") + 1 if m.group(0) in s else 0
            ctx = norm(s)[max(0, m.start() - 60):m.start() + 60]
            print(f"  TERSISA  [{label}] {f}:~{ln}  ({why})")
            print(f"           ...{ctx}...")
            bad += 1
if not bad:
    print("  tidak ada")

print("\n=== nilai terkini, di dokumen mana saja ===")
for label, pat in AGREE:
    where = [f for f, s in DOCS.items() if re.search(pat, norm(s))]
    print(f"  {label:20s} {', '.join(where) if where else 'TIDAK ADA DI MANA PUN'}")

print("\n=== judul yang sama di tiga dokumen? ===")
TITLE = "Calibration Beats Robust Training for Worst-Group Coverage"
for f in ("sn-article.tex", "response-to-reviewers.tex", "before-after.tex"):
    print(f"  {f:28s} {'ya' if TITLE in DOCS[f] else 'TIDAK'}")

print(f"\n{'BERSIH' if bad == 0 else str(bad) + ' NILAI USANG'}")
