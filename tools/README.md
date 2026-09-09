# Verification tools

Written while revising the ACML submission. Every one of them caught something a
reading did not, and between them they are the reason the numbers in the paper can be
trusted. Run them after any edit to the manuscript.

Paths are absolute (`c:\jagr\vgscp\...`) because they were written for this
checkout; adjust the constants at the top of each file if the repo moves.

## Checkers

Run all of them; each prints a one-line verdict at the end.

- `check_tex_final.py` — structure: braces, math, dangling refs, citations both ways, column counts
- `audit_appendix.py` — every appendix table cell re-derived from the released records
- `contradict.py` — invariants that past self-inflicted contradictions violated
- `reviewer_safety.py` — all 19 reviewer points still addressed, and where
- `crossdoc.py` — superseded values absent from all four source files
- `tabwidth.py` — every table measured against the 372pt text block
- `blankcol.py` — no table column blank in every data row
- `check_quotes.py` — the letter's quoted blocks still match the manuscript
- `sizes.py` — word count per section, plus captions
- `numguard.py` — values and citations that vanished since a snapshot

```sh
cd tools
for f in check_tex_final audit_appendix contradict reviewer_safety \
         crossdoc tabwidth blankcol check_quotes; do
    printf "%-20s " $f; python $f.py | tail -1
done
```

`check_tex_final.py` expects to run from the paper directory:

```sh
cd "ACML_Journal___Robust_CP_Train_Study (1)"
python ../tools/check_tex_final.py
```

## Generators

`appendix-tables.tex` is generated, not written. To rebuild it:

```sh
cd tools
python gen_appendix_tables.py && python fix_grid_tables.py
```

The order matters: the first writes every table, the second replaces the grid tables
with page-breaking longtables and prepends the cross-score summary.

- `gen_appendix_tables.py` — builds appendix-tables.tex from results/*.csv
- `fix_grid_tables.py` — reshapes the grids into longtables and adds the cross-score summary
- `gen_pergroup.py` — the per-group coverage table, from results/per_group_coverage.csv
- `make_mechanism_fig.py` — Figure 2, with pi read per run from the records

## What each guards against

These are not hypothetical. Each check exists because something got through:

- `audit_appendix.py` — two body/appendix tables once disagreed because their
  generators filtered the records differently.
- `contradict.py` — five paragraphs once contradicted other paragraphs, all introduced
  while fixing earlier review findings. Numbers checked out; the prose did not.
- `crossdoc.py` — corrections reached the manuscript and not the response letter for
  three rounds running, because every check was per-document.
- `blankcol.py` — a table printed a column under the wrong heading with two columns
  blank in every row, and passed every structural check because the *count* was right.
- `check_quotes.py` — the letter quoted, as the paper's own words, a claim the paper
  had withdrawn.
