#!/bin/sh
cd "/c/jagr/vgscp/ACML_Journal___Robust_CP_Train_Study (1)" && printf "%-18s " check_tex_final && python ../tools/check_tex_final.py | tail -1
cd /c/jagr/vgscp/tools || exit 1
for f in audit_appendix contradict reviewer_safety crossdoc tabwidth blankcol check_quotes; do
  printf "%-18s " $f; python $f.py | tail -1
done
