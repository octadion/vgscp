# Audit manuskrip — "Calibration Beats Robust Training for Worst-Group Coverage"

**Catatan sebelum mulai.** Hanya tiga PDF yang sampai: manuskrip, Jones et al. (2010.14134),
Cresswell et al. (2410.01888). **Kirichenko et al. (2204.02937) tidak terlampir.** Semua rujukan ke
Kirichenko di bawah berasal dari ingatan saya atas paper itu, jadi bobotnya lebih ringan daripada
rujukan ke Jones dan Cresswell yang teksnya ada di depan saya. Kalau Anda ingin perbandingan
proporsi main-body/appendix yang mengikutsertakan Kirichenko, kirim ulang.

Temuan diurutkan prioritas. Format: **Di mana → Apa yang salah → Kenapa terbaca sebagai report →
Perbaikan → Severity**.

---

## A. Sisa-sisa lapisan pipeline yang masih terlihat pembaca

### 1. Kata "gate" masih ada di tiga tempat
- **Di mana:** §9 Data and code availability, `"together with the quality-gate log and the scripts that draw each figure"`; Appendix A, `"with the gate status of each method"`; Appendix C, `"The criterion is thus not robust to the gate decision"`.
- **Apa yang salah:** Anda bilang *quality gate* sudah dibuang. Belum. Tiga kali lolos.
- **Kenapa terbaca report:** "gate" bukan istilah statistik, itu nama komponen di pipeline Anda. Pembaca luar tidak tahu apa yang di-gate.
- **Perbaikan:** §9 → `"together with the log of which runs were excluded and why, and the scripts that draw each figure"`. Appendix A → `"and whether each run was retained or excluded"`. Appendix C → `"The criterion is therefore not robust to the exclusion rule"`.
- **Severity:** blocking.

### 2. Kata "arm" masih tercetak di dalam Figure 4
- **Di mana:** legenda Figure 4: `ERM (reference)` / `robust arms`.
- **Apa yang salah:** kosakata internal yang katanya sudah dibersihkan, sekarang tertanam di gambar (bukan teks), jadi luput dari find-replace.
- **Perbaikan:** ganti label jadi `robust methods`. Regenerasi gambar. Cek juga label sumbu di semua figure lain terhadap daftar kata terlarang Anda.
- **Severity:** blocking.

### 3. Find-replace `arm → method` merusak kalimat, dan "method" sekarang berarti dua hal
- **Di mana:** §5, `"Of the 200 methods, 33 fall below their per-method worst-group accuracy floor"`; §4, `"Each (backbone, dataset, method, seed) method must reach a sane worst-group accuracy for its method or it is excluded"`; §6.8, `"30 of 34 methods stay within 0.02"`; Appendix C, `"Twelve DFR methods on CelebA"`.
- **Apa yang salah:** Anda punya **lima** training method (ERM, DFR, AFR, balanced subsampling, GroupDRO-LL). Kalimat "of the 200 methods" tidak bisa benar. Kalimat di §4 secara harfiah tidak gramatikal — "method … must reach … for its method". Yang Anda maksud adalah *run*: satu tuple (backbone, dataset, method, seed).
- **Kenapa terbaca report:** ini jejak substitusi otomatis yang tidak dibaca ulang. Pembaca yang jeli akan berhenti di sini dan mulai tidak percaya angka lain.
- **Perbaikan:** perkenalkan satu istilah di §5 dan pakai konsisten: `"We call each (backbone, dataset, method, seed) tuple a run; the grid has 200 runs."` Lalu: `"Of the 200 runs, 33 fall below the worst-group accuracy floor set for their training method"`. §4 → `"Each run must reach the floor set for its training method or it is excluded, with the reason recorded."` Ganti "sane" (kata codebase) dengan tidak ada apa-apa.
- **Severity:** blocking.

### 4. Label hipotesis (H1–H6) muncul di judul tabel dan kalimat hasil
- **Di mana:** caption Table 3 `"(H1; APS, ρcal = 0.95)"`, Table 5 `"(H2; …)"`, Table 6 `"(H3; …)"`, Table 7 `"(H4; …)"`; §6.1 `"Our central experiment (H1) holds the model fixed"`; §6.6 `"This section tests H5"`; §6.7 `"consistent with H1"`.
- **Apa yang salah:** ini persis kebiasaan yang sudah Anda buang untuk kode reviewer ("the sensitivity analysis R2.4 asks for"), hanya dengan huruf lain. H-number adalah artefak pra-registrasi internal, bukan objek yang dipedulikan pembaca.
- **Bandingkan:** Cresswell punya hipotesis pra-registrasi juga, tapi dinyatakan sebagai kalimat penuh dalam huruf tebal (`Hypothesis 1 Prediction sets supplied to human decision makers can cause disparate impact…`) dan hasilnya dirujuk dengan kalimat, bukan kode: `"Hypothesis 2 proposes that equalizing coverage with the conditional treatment will cause more disparate impact"`. Tidak ada "H2" di caption tabel mana pun.
- **Perbaikan:** hapus semua "(Hn; …)" dari caption. Di teks, ganti `"Our central experiment (H1) holds"` → `"The central experiment holds"`. Pindahkan daftar enam hipotesis ke appendix; pertahankan **satu kalimat** di §5: `"The six hypotheses and the per-method accuracy floors were fixed before any of the experiments they concern were run; they were not publicly registered (Appendix F)."` — kalimat itu yang menjaga R2.4 tetap terlihat.
- **Severity:** blocking. *Catatan R2.4:* pemindahan ini aman selama kalimat "pre-specified, not preregistered" tetap di main body.

### 5. Appendix D basi: masih menyebut dua backbone
- **Di mana:** Appendix D, `"Features. Frozen ERM ResNet-50 (d=2048) and CLIP ViT-B/32 (d=512), extracted once and cached"`.
- **Apa yang salah:** paper sekarang punya empat backbone (R1.3). Appendix reproducibility tidak diperbarui setelah revisi. Ini bukan soal gaya — ini kontradiksi faktual antara §5 dan Appendix D, di appendix yang justru dimaksudkan untuk memenuhi R2.6.
- **Perbaikan:** tulis ulang seluruh entri Features dengan empat backbone + dimensi + sumber bobot. Lalu baca ulang **setiap** kalimat Appendix D terhadap versi paper sekarang; kalau satu entri basi, kemungkinan ada yang lain.
- **Severity:** blocking. Merusak R1.3 dan R2.6 sekaligus.

### 6. "ImageNet-V2 weights" kemungkinan salah
- **Di mana:** Appendix D, `"Fine-tuning (Section 6.2) starts from ImageNet-V2 weights"`.
- **Apa yang salah:** ImageNet-V2 adalah *test set* (Recht et al. 2019), bukan checkpoint. Yang Anda maksud hampir pasti `IMAGENET1K_V2` weights dari torchvision.
- **Perbaikan:** `"starts from the torchvision IMAGENET1K_V2 ResNet-50 checkpoint"`.
- **Severity:** blocking (kesalahan faktual di bagian reproducibility).

### 7. Placeholder masih di dalam naskah
- **Di mana:** Appendix D, `"[To be replaced before publication with the archived DOI of the tagged release.]"`
- **Apa yang salah:** wajar untuk submission, tapi Anda mengirim ini ke audit gaya — kalau reviewer melihatnya di ronde revisi, itu sinyal naskah belum dibaca ujung ke ujung.
- **Perbaikan:** daftarkan DOI Zenodo sekarang, atau ganti dengan URL repositori anonim.
- **Severity:** worth fixing.

### 8. Sitasi rusak `(elm 2024)` — dan justru itu titik R1.5
- **Di mana:** §6.2, `"partial or staged schedules change what the fine-tuned representation ends up encoding (elm 2024)"`. Entri bibliografi tanpa penulis: `(2024) New unfreezing strategy of transfer learning in satellite imagery … Scientific African`.
- **Apa yang salah:** dua hal. (a) kunci bibtex bocor sebagai nama penulis. (b) relevansinya tipis: pemetaan permukiman kumuh via satelit, dikutip di studi korelasi spurious pada Waterbirds/CelebA. Ini berbunyi seperti sitasi yang dipasang untuk memuaskan reviewer, bukan yang dipakai.
- **Perbaikan:** perbaiki entri (penulis lengkap). Lalu **pasangkan** dengan rujukan yang benar-benar bekerja untuk klaim itu: Kumar et al., *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution* (ICLR 2022). LP-FT persis tentang bagaimana derajat unfreezing mengubah apa yang dienkode representasi, dan absennya mencolok di paper yang seluruh sumbunya adalah frozen vs fine-tuned. Kalimat baru: `"How much of a pretrained network to unfreeze changes what the resulting representation encodes, and full fine-tuning can distort pretrained features relative to linear probing (Kumar et al. 2022; [Elm et al.] 2024). We unfreeze all of it, the setting in which the representation moves most."`
- **Severity:** blocking. *R1.5 tetap terpenuhi* — malah lebih kuat, karena sekarang rujukannya dipakai secara substantif seperti yang diminta.

---

## B. Masalah intelektual, bukan kosmetik

### 9. Anda mengutip Cresswell tapi tidak menghadapi bahwa rekomendasi Anda berlawanan dengannya
- **Di mana:** §2 `"Cresswell et al. (2025) show that equalizing coverage can increase disparate impact and recommend equalizing set sizes"`; lalu §7 `"take worst-group coverage from the calibration rule"`; §9 `"giving the worst group its own threshold lifts an ordinary model by up to 0.345"`.
- **Apa yang salah:** Cresswell mengukur *outcome* manusia dan menemukan bahwa Mondrian/conditional CP menghasilkan disparate impact **terbesar** dari semua treatment (`maxRORCond` konsisten di atas `maxRORMarg`), justru karena ia memperbesar selisih ukuran himpunan. Rekomendasi utama Anda adalah: pakai Mondrian. Anda menyebut biaya ini sekali, di caption Figure 1d, lalu tidak pernah lagi. Discussion dan Conclusion merekomendasikan Mondrian tanpa rekonsiliasi.
- **Kenapa ini yang paling penting:** ini satu-satunya tempat di mana paper Anda bersinggungan langsung dengan salah satu paper referensi Anda dan menghasilkan nasihat berlawanan. Reviewer yang mengenal literatur ini akan menemukannya. Lebih baik Anda yang mengangkatnya.
- **Perbaikan:** satu paragraf di §7, sebelum "The advice that follows": nyatakan bahwa di bawah kriteria berbasis-outcome, rekomendasi ini berbalik; laporkan angka set-size disparity per setting (Anda punya datanya — sebutkan di Figure 1d bahwa worst-group set size naik 1.080 → 1.319 sementara rata-rata hampir tidak bergerak); dan scope nasihatnya: *"where coverage is itself the operational guarantee — an audit, an SLA, a regulatory floor — the calibration rule is the lever. Where a human acts on the set, Cresswell et al. (2025) show that equalized coverage buys reliability at the cost of set-size disparity, and our Figure 1d prices that trade for the worst group."*
- **Severity:** blocking.

### 10. W1 adalah metrik headline yang remark Anda sendiri sudah membuktikan tidak relevan
- **Di mana:** Eq (2) mendefinisikan `D = W1` dan `DKS`; Remark setelah Prop 1: `"W1 admits no such bound—it integrates the gap over the line while the shortfall depends on it at one point"`. Tapi sumbu-y Figure 4 adalah `cross-group divergence W1`, dan kolom `D_ERM` / `D (rob.)` di Table 6 adalah W1.
- **Apa yang salah:** Anda membuktikan bahwa W1 tidak membatasi kuantitas yang Anda pedulikan, lalu memakai W1 sebagai satu-satunya divergence yang dilaporkan di seluruh §6.4. `DKS` — yang **memang** membatasi shortfall lewat Eq (4) — tidak pernah muncul dalam satu angka pun di paper.
- **Kenapa terbaca report:** teori dan eksperimen berjalan di jalur terpisah. Jones tidak pernah melakukan ini: Prop 1-nya tentang left-log-concavity dan Figure 4-nya menunjukkan margin distribution yang tepat menjadi objek proposisi itu.
- **Perbaikan:** laporkan `DKS` sebagai kolom utama di Table 6 dan sumbu Figure 4, dengan W1 sebagai kolom pendamping (deskriptif, mengikuti Gao et al.). Anda punya datanya. Ini mengubah §6.4 dari "kami mengukur sesuatu" menjadi "kami mengukur besaran yang muncul di Eq (4)".
- **Severity:** blocking.

### 11. Eq (3) adalah gagasan pengorganisasi paper ini, dan tidak melakukan apa-apa
- **Di mana:** Prop 1(ii), `(1 − α) − Fwg(q̂) = (1 − π)[F¬wg(q̂) − Fwg(q̂)]`; lalu dipakai satu kali di Remark: `"it scales with (1 − π), and π is small exactly for the group the spurious correlation disadvantages"`. Setelah itu tidak pernah lagi.
- **Apa yang salah:** Anda bertanya apakah paper punya satu gagasan yang menggantungkan semuanya, seperti Jones menggantungkan segalanya pada margin distribution. **Kandidatnya adalah Eq (3), dan sudah ada di naskah.** Ia memprediksi hal-hal yang bisa diuji: shortfall marginal harus proporsional terhadap (1−π) × celah CDF di ambang; jadi lift Mondrian harus besar ketika π kecil **dan** celahnya besar, dan kecil ketika salah satunya kecil. Tabel 3 Anda sebenarnya menunjukkan pola itu (CelebA punya lift kecil karena celahnya kecil, bukan karena π-nya beda), tapi Anda menjelaskannya secara ad hoc: `"its marginal coverage was never far from target to begin with"`.
- **Bandingkan:** Jones memakai dekomposisi yang **secara struktural identik** — `F = pFwg + (1 − p)Fothers` — sebagai tulang punggung seluruh Section 5 dan 6, dan setiap hasil empiris dibaca kembali lewatnya.
- **Perbaikan:** ini perubahan paling berharga di seluruh audit. (a) Pindahkan Eq (3) ke akhir §1 sebagai kalimat mekanisme. (b) Di §6.1, alih-alih mendeskripsikan tabel, hitung sisi kanan Eq (3) per setting dari data Anda dan tunjukkan bahwa ia memprediksi lift yang teramati — plot prediksi vs observasi, delapan titik, satu panel. Itu jadi figure kedua paper dan mengubah "kami mengamati disosiasi" menjadi "kami tahu apa yang mengendalikannya". (c) Prop 1(i) yang Anda sendiri akui `"true by construction"` diturunkan jadi footnote.
- **Severity:** blocking (ini yang membedakan paper karakterisasi dari laporan pengukuran).

### 12. Kata "burden" menamai dua kuantitas berbeda
- **Di mana:** §3, `"which is the quantity tabulated throughout as the conformal "burden""` (untuk ∆cov); dan tiga paragraf kemudian judul `Cross-group conformity-score divergence (the "burden")` (untuk D).
- **Apa yang salah:** ∆cov dan D adalah besaran berbeda dengan satuan berbeda, keduanya diberi nama "burden", keduanya dalam tanda kutip. Tanda kutip di sekitar label buatan sendiri adalah kebiasaan yang sudah Anda daftarkan sebagai tanda naskah generated.
- **Perbaikan:** buang kata "burden" seluruhnya. ∆cov = *coverage shortfall*. D = *cross-group score divergence*. Perbaiki juga judul §6.5 dan butir kontribusi 5 yang memakai kata itu.
- **Severity:** blocking.

### 13. Judul §6.5 dan isinya tidak cocok
- **Di mana:** heading `"6.5 The Most Accurate Method Is Usually the Most Efficient"`, tapi Table 7 melaporkan ∆cov (coverage shortfall), bukan set size. Butir kontribusi 5 menyebutnya `"Accuracy is a good proxy for conformal burden"`.
- **Apa yang salah:** tiga nama untuk satu klaim, dan salah satunya (*efficient*) sudah dipakai untuk §6.3 yang memang tentang set size.
- **Perbaikan:** heading → `"Worst-Group Accuracy Predicts the Coverage Shortfall"`. Samakan butir kontribusinya.
- **Severity:** worth fixing.

---

## C. Suara dan kalimat

### 14. Konstruksi "X, not Y" — tik dominan paper ini
Anda sudah tahu *settings* (~47). Ini yang lebih parah karena berupa **ritme**, bukan kata. Saya hitung ≥15 kemunculan pola *"is A rather than B" / "A, not B"*:

> `"a property of the statistic we report rather than a failure of validity"` (§6.1) ·
> `"a direction rather than an estimate"` (§6.3) ·
> `"Zero variance here is a property of the solver, not evidence of stability."` (§5) ·
> `"a group-prior shift … and not a general covariate shift"` (§6.6) ·
> `"the selection effect of reporting a minimum, not under-coverage"` (App. B) ·
> `"finite-sample, not a failure of the dissociation"` (§6.6) ·
> `"in the level it attains rather than in its variance across splits"` (§6.6) ·
> `"a characterization rather than a method"` (§1) ·
> `"pre-specified rather than preregistered"` (§4) ·
> `"a property of the method in this regime, not of our choice of γ"` (App. C) ·
> `"a tendency rather than a rule"` (§6.5) ·
> `"the same efficiency-not-coverage pattern"` (§6.2) ·
> `"visible in the accuracy column, not the coverage one"` (§6.2) ·
> `"necessary alongside any such claim, but it is not sufficient"` (§6.8) ·
> `"an interpretability layer, orthogonal to our thesis"` (§2)

- **Kenapa terbaca generated:** manusia memakai konstruksi ini sesekali untuk membalik ekspektasi. Model memakainya sebagai default untuk menandai presisi. Efek kumulatifnya: setiap kalimat terasa sedang mengoreksi kesalahpahaman yang belum terjadi.
- **Perbaikan:** pertahankan **maksimal tiga**, yang paling penting (saran: yang di App. B tentang selection effect, yang di §6.6 tentang group-prior shift, dan `"characterization rather than a method"` — atau buang yang terakhir, lihat #19). Sisanya tulis sebagai pernyataan positif. Contoh konkret:
  - `"Zero variance here is a property of the solver, not evidence of stability."` → `"L-BFGS ignores the seed, so ERM and AFR fit the same head every time; the across-seed SD of 0.000 measures the solver."`
  - `"We therefore report a direction rather than an estimate"` → `"The sign is consistent; the magnitude is not estimable from three to five methods per setting."`
  - `"a property of the statistic we report rather than a failure of validity"` → `"The minimum over four noisy per-group coverages sits 0.018–0.045 below their mean, which is where an exactly valid procedure puts it (Appendix B)."`
- **Severity:** blocking (ini yang paling akan dibaca sebagai "ditulis model").

### 15. Membela keputusan yang belum ditantang siapa pun
Contoh yang tersisa (tipe yang sama dengan `"We report this rather than present zero variance as stability"` yang sudah Anda buang):

> `"Clipping sets to a minimum size of one would hide this, so we do not."` (§6.3) ·
> `"We keep it: a representation that fails to improve is still a representation that changed."` (§6.2) ·
> `"None of this bears on the paper's claims, and it is worth being explicit about why."` (§5) ·
> `"Because reporting a method at an untuned hyperparameter would be unfair, we computed an oracle upper bound"` (App. C) ·
> `"we never substitute the raw D, which we report separately, labeled uncontrolled"` (§4) ·
> `"so we state plainly what differs"` (§5) ·
> `"All may return negative; we report whatever the data show."` (§4) ·
> `"but we report the mean over groups alongside it so the two are not confused"` (App. B)

- **Kenapa terbaca report:** semuanya menjawab keberatan yang tidak ada di halaman. Jones dan Cresswell tidak pernah melakukannya. Ketika Jones punya kelemahan, ia menyatakannya sebagai fakta dan berhenti: `"Group DRO is not a silver bullet, as it relies on group annotations for training, which are not always available."` Titik. Tidak ada "kami memilih melaporkannya karena…".
- **Perbaikan:** hapus klausa pembelaan, sisakan faktanya.
  - `"Clipping sets to a minimum size of one would hide this, so we do not."` → hapus seluruhnya; kalimat sebelumnya (`"coverage there remains valid under Mondrian (0.858–0.881)"`) sudah cukup.
  - `"We keep it: a representation that fails to improve is still a representation that changed."` → `"Reweighting did not raise worst-group accuracy (0.466 and 0.364), so it enters the comparison as a representation that moved without improving."`
  - `"None of this bears on the paper's claims, and it is worth being explicit about why."` → `"Every result here is a within-setting contrast, so a weaker backbone lowers all methods together and leaves the calibration contrast intact."` (kalimat berikutnya yang sudah ada — buang pembukanya).
- **Severity:** blocking.

### 16. Menarasikan proses, termasuk pendekatan yang dibuang
- **Di mana:** §6.4, seluruh paragraf yang dibuka `"Controlling for it requires more care than it appears to."` lalu menjelaskan cara yang *tidak* dipakai (`"The natural way to obtain enough points to interpolate is to pool each method's (accuracy, D) points across seeds and calibration splits, but the axis this produces is mostly evaluation-draw noise"`) sebelum sampai ke cara yang dipakai. Juga §6.2 `"A representation-level experiment is informative only if the intervention actually moved the representation, so we verify that first."`; §6.3 `"One number needs disclosure."`; §5 `"It is also why every interval in the paper is a cluster bootstrap"`.
- **Kenapa terbaca report:** ini catatan laboratorium — urutan penemuan, bukan urutan argumen.
- **Perbaikan:** §6.4 → mulai langsung dari hasilnya: `"Matched at model level, with one accuracy per training seed and a two-stage cluster bootstrap, ERM's accuracy support is a single point, and it overlaps a robust method's in exactly one of eight settings."` Alasan mengapa split-level matching menyesatkan → dua kalimat di appendix. §6.2 → hapus kalimat pembuka; mulai dari `"GroupDRO fine-tuning raises worst-group accuracy from 0.516 to 0.738 on Waterbirds"`. §6.3 → hapus `"One number needs disclosure."`, mulai dari `"On DINOv2/Waterbirds the mean worst-group set size is below one…"`.
- **Severity:** blocking.

### 17. Setiap subseksi ditutup kalimat moral
- §6.1 → `"Whether that survives retraining the representation itself is the question of Section 6.2."` · §6.2 → `"—the same efficiency-not-coverage pattern as Section 6.3."` · §6.4 → `"which is exactly the confound this protocol exists to expose."` · §6.7 → `"consistent with H1."` · §3 remark → `"an assumption about which group is worst, not a consequence of the divergence."`
- **Apa yang salah:** tujuh dari delapan subseksi §6 berakhir dengan klausa yang merangkum-atau-menunjuk-maju. Keseragaman itu sendiri yang terbaca sebagai mesin. Jones menutup Section 4 tanpa moral apa pun — ia berhenti di temuan (`"We show similar results for MC-dropout in Section B.1."`), dan menutup Section 7 dengan satu kalimat konsekuensi, sekali saja.
- **Perbaikan:** buang penutup di §6.1, §6.4, §6.7. Pertahankan di §6.2 dan §6.3 jika benar-benar menjembatani.
- **Severity:** worth fixing.

### 18. Tik "Pertanyaan? Jawaban dua kata."
- `"Does the representation on which worst-group coverage is obtained hide the protected group? It does not."` (§6.7) · `"and ask whether the conserved heterogeneity shrinks at the source. It does not; the calibration policy remains decisive."` (§2) · `"Given that Mondrian equalizes worst-group coverage across training methods, where does training help?"` (§6.3) · `"which invites the obvious objection… Because the spurious attribute stays easy to recover…, it does not need them."` (§6.8)
- **Perbaikan:** sisakan satu (§6.7 paling efektif). Yang lain jadi pernyataan langsung.
- **Severity:** minor, tapi mencolok karena berulang.

### 19. Kalimat meta tentang paper itu sendiri
- `"We frame the paper as a characterization rather than a method."` (§1) · `"The unifying thesis (Section 7) is that…"` (§1) · `"The results compose into one story."` (§7 pembuka) · `"The advice that follows is deflationary."` (§7)
- **Apa yang salah:** memberi tahu pembaca bahwa Anda punya cerita, alih-alih menceritakannya. Tidak satu pun dari tiga paper referensi melakukannya. Jones membuka Discussion: `"We have shown that selective classification can magnify group disparities and should therefore be applied with caution."` Langsung isinya.
- **Perbaikan:** hapus keempatnya. §1 → nyatakan tesisnya sebagai klimaks paragraf pembuka, bukan sebagai penunjuk ke §7. §7 → mulai dari `"Worst-group coverage is set chiefly by the calibration rule."`
- **Severity:** worth fixing.

### 20. Kata yang dipilih karena terdengar akademis
`"deflationary"` (§7) · `"the evidence for that is a co-scaling"` (§6.6) · `"drawn from this zoo"` dan `"the training-intervention zoo"` (§2) · `"invariant-ization"` (App. D) · `"re-weight / re-composite the evaluation rows"` dan `"re-composited"` (§5, Table 1) · `"a sane worst-group accuracy"` (§4) · `"the operative lever"` / `"the operative target"` (§2).
- **Perbaikan:** *deflationary* → hapus kalimatnya. *co-scaling* → `"the two scale together"`. *zoo* → `"this line of work"`. *invariant-ization* → `"we remove the top-k spurious-predictive directions"`. *re-composite* → `"resample"`. *sane* → hapus. *operative* → `"the lever that matters"` sekali, sisanya hapus.
- **Severity:** worth fixing.

### 21. Metafora "lever" dipakai tujuh kali
`"the lever was pulled"`, `"The two levers"`, `"the representation lever"` (×3), `"pulls the representation lever as far as each objective allows"`, `"the operative lever"`.
- **Perbaikan:** maksimal dua. Sisanya: `"changing the representation"` / `"changing the calibration rule"`.
- **Severity:** minor.

### 22. Rentang AUROC diulang empat kali
`AUROC ∈ [0.945, 0.999]` muncul di §1 (kontribusi 6), §2 (Group recovery), §6.7, §6.8, dan lagi di Table 8.
- **Perbaikan:** sebutkan sekali di §6.7 dengan tabelnya; di tempat lain rujuk saja ("the probe recovers the attribute in every setting; Table 8").
- **Severity:** worth fixing.

### 23. Bagaimana ketiga referensi membuka paragraf hasil vs bagaimana Anda membukanya
- **Anda:** `"Our central experiment (H1) holds the model fixed and changes only the rule that turns its scores into sets."` — prosedur dulu, label internal, temuan di paragraf ketiga.
- **Jones §4:** `"Figure 2 shows group accuracy-coverage curves for each dataset… On all datasets, average accuracies improve as coverage decreases. However, the worst-group curves fall into three categories:"` — objek, lalu langsung temuan, lalu taksonomi.
- **Cresswell §6.1:** `"Figure 1 directly visualizes the data we collected, showing the disparate impact on accuracy… We observe that disparate impact is present for most tasks and treatments."` — objek, temuan, di kalimat kedua.
- **Perbaikan §6.1:** `"Table 3 reports worst-group coverage under both calibration rules for every training method and setting. Under one shared threshold, worst-group coverage follows the training method and spreads by up to 0.358; give each group its own threshold and the spread never exceeds 0.024. The pattern holds in all eight settings."`
- **Severity:** blocking (ini mengubah rasa seluruh §6).

### 24. Limitasi yang membatalkan dirinya sendiri
- **Di mana:** §8, `"a third dataset and a multi-class task would extend breadth, though the consistency of H1 across all eight frozen settings and all six settings of the fine-tuning study suggests additional data would refine rather than overturn it"`; dan `"which tells us something but prevents a uniform statement"`.
- **Apa yang salah:** limitasi diikuti klausa yang mencabutnya. Bandingkan Cresswell: `"some of the statistical measures in Table 2 did not achieve significance at the 5% level due to insufficient observations."` — dinyatakan, konsekuensinya dijelaskan, tidak dinetralkan. (Mereka memang menambahkan mitigasi, tapi sebagai kalimat terpisah, bukan klausa "though".)
- **Perbaikan:** `"We test two datasets and binary spurious attributes. A third dataset and a multi-class task would tell us whether the dissociation is specific to |Y| = 2, where set sizes lie in [1, 2] and the efficiency axis is compressed."`
- **Severity:** worth fixing. *Catatan R2.6:* moderasi klaim tetap terjaga — justru lebih jujur.

---

## D. Struktur dan proporsi

### 25. Main body terlalu panjang, appendix terlalu tipis
- Anda: main text ±20 halaman (§1–§9 + Declarations, hlm. 1–21), appendix ±5 halaman (A–D, hlm. 21–26), referensi 3.
- Jones: 9 halaman utama, **25 halaman appendix** (bukti lengkap Lemma 1–15, simulasi, detail dataset, MC-dropout, group-DRO).
- Cresswell: 10 halaman utama, ±10 appendix (pre-processing per dataset, tabel hitungan grup, hyperparameter, seluruh alur eksperimen manusia dengan screenshot).
- **Kaveat jujur:** dua-duanya paper konferensi dengan batas halaman keras; Springer *Machine Learning* tidak punya batas 10 halaman. Jadi target Anda bukan 10 halaman, tapi **±13 halaman** — dan yang lebih penting, appendix yang berisi tabel, bukan prosa.
- §6 sendiri memakan 11 halaman dengan delapan subseksi. Itu bukan bagian hasil, itu daftar eksperimen.
- **Severity:** blocking.

### 26. Appendix Anda mendeskripsikan hasil, tidak memuatnya
- **Di mana:** Appendix A dibuka `"All per-(setting, score, method) numbers behind the main tables are in the released records."` lalu memberi ringkasan prosa: `"The shift-robust policy over-covers the worst group (0.84–0.96) at the cost of larger sets"`. Tidak ada satu tabel pun di Appendix A. Table 8 caption: `"Per-method values for all three conditions and all three scores are in the released results."`
- **Apa yang salah:** appendix yang menunjuk ke file bukan appendix, itu README. Reviewer tidak akan mengunduh CSV.
- **Perbaikan — apa yang harus ada, mengingat 116.000 evaluasi Anda:**
  1. **Tabel grid utama**, satu per score (THR/APS/RAPS): 8 setting × 5 method × {marginal, Mondrian, shift-robust} × {worst-group coverage, mean-over-group coverage, ∆cov, mean set size, worst-group set size, set-size disparity}. Ini yang menopang §6.1 dan sekarang tidak ada di mana pun.
  2. **Coverage per grup** (keempat grup), bukan hanya minimum. Seluruh argumen min-over-groups di Appendix B menuntutnya, dan pembaca tidak bisa memverifikasi tanpa itu.
  3. **Tabel `DKS` dan W1** per setting per score (lihat #10).
  4. **Tabel sweep ρ**: 6 nilai ρtest × 8 setting × 2 policy. Sekarang hanya ada Figure 5.
  5. **Tabel shift-robust penuh** — kebijakan ketiga Anda sekarang hidup di satu paragraf prosa.
  6. **Studi fine-tuning penuh**: 3 objective × 5 seed × 3 score × 2 dataset, dengan SD antar-seed. Table 4 sekarang hanya menunjukkan enam baris rata-rata.
  7. **Studi predicted-group penuh**: 3 kondisi × 3 score × setiap method (Table 8 hanya median/max per setting).
  8. **Sensitivitas R2.4 sebagai tabel utuh yang dihitung ulang**, bukan paragraf. Sekarang: `"We recomputed every result with the excluded methods included"` lalu tiga angka.
  9. **Sweep γ untuk AFR** sebagai kurva (sekarang empat angka dalam prosa: `"0.327, 0.427, 0.445 and 0.477"`).
  10. **Dekomposisi varians**: SD antar-seed vs SD antar-split, per setting per method. Anda menjanjikannya di §5 (`"we report training-seed and calibration-split standard deviations separately"`) tapi tidak pernah menampilkan tabelnya.
  11. **Turunan yang hilang:** Appendix B menyatakan `"E[ming covg] ≈ (1 − α) − c|G|σ with c4 ≈ 1.03"` tanpa turunan atau sitasi. Itu konstanta ekspektasi minimum order-statistic Gaussian. Turunkan atau kutip.
  12. **Versi finite-sample dari Eq (3)**, yang memasukkan galat estimasi kuantil per grup — ini menjembatani Prop 1 (populasi) dan level 0.85–0.89 yang teramati, dan sekarang jembatannya adalah simulasi saja.
- **Severity:** blocking.

### 27. Yang harus pindah ke appendix (dan yang harus tetap terlihat)

| Sekarang | Pindah? | Efek pada reviewer |
|---|---|---|
| §4 seluruhnya (protokol + 6 hipotesis) | Ya → appendix; sisakan 1 paragraf protokol di dalam §5 dan **satu kalimat** pre-specified | **R2.4 tetap terlihat** lewat kalimat itu |
| §6.4 (accuracy matching, mekanika) | Sebagian: pertahankan 1 paragraf + Table 6; Figure 4 tetap di main | **R2.3 aman** selama cluster bootstrap dan level-model tetap disebut di main |
| §6.6 paragraf finite-sample (`"the evidence for that is a co-scaling"`) | Ya → gabung ke Appendix B | **R1.1 aman**: §6.1 dan Appendix B tetap menanganinya |
| §6.7 (probe + projecting out directions) | Ya → gabung ke §6.8 jadi satu subseksi "Groups without labels" | **R2.5 aman** jika mekanika predicted-group, probe error, dan etika tetap di main |
| §5 empat paragraf "Comparison with published values" | Ringkas jadi 3 kalimat + Table 2 tetap di main; detail ke appendix | **R1.3 aman**: Table 2 tetap di main body |
| Table 1 (ukuran split) | Ya → appendix | tidak ada reviewer point |
| Table 7 kolom `inversion` (`"no (+0.011)"`, `"real +0.014"`) | Kolom ini adalah *verdict* yang berubah bentuk jadi kolom tabel — hapus; sisakan ∆ dan CI-nya | **R3.1 aman**: CI-nya yang membawa muatan |

**Satu pemindahan yang menurut saya terlalu jauh:** jangan pindahkan Table 4 / Figure 3 (studi fine-tuning) ke appendix. Itu inti jawaban R1.2 dan R2.1 dan harus ada di main body.

### 28. Figure 2 duplikat Table 3; Figure 3 duplikat Table 4
- Figure 2 memplot tiga kolom pertama Table 3 (shared / per-group / lift, delapan baris). Figure 3 memplot enam angka Table 4.
- **Perbaikan:** pilih satu per pasangan. Saran: pertahankan **Figure 2** (visualnya lebih kuat daripada tabel, dan lift-nya jadi terbaca sekejap), pindahkan Table 3 versi lengkap ke appendix dengan ketiga score. Sebaliknya untuk fine-tuning: pertahankan **Table 4** (angka spread 0.249 vs 0.004 adalah intinya), buang Figure 3 — batangnya tidak menambah apa pun. Itu menghemat setengah halaman dan membebaskan slot untuk figure prediksi-Eq(3) dari #11.
- **Severity:** worth fixing.

### 29. Caption
- Figure 3: `"Reweighting did not improve the representation and is reported anyway. Bars begin at zero."` — "Bars begin at zero" adalah catatan defensif ke reviewer di dalam caption. Hapus.
- Figure 1: caption 6 kalimat yang mengulang argumen (`"What that costs, and what a per-group threshold recovers on the same model and the same scores."`). Bandingkan Cresswell Figure 2 caption: satu kalimat mekanisme, satu kalimat konsekuensi. Potong jadi tiga kalimat.
- Table 3: catatan kaki menjelaskan logika eksklusi ulang. Cukup: `"†ERM below its floor here; values from Appendix C, excluded from the spread."`
- **Severity:** minor.

### 30. Introduction tidak melakukan yang dilakukan ketiga introduction itu
Tiga masalah:
1. **Paragraf kedua adalah related work.** Dua belas sitasi dalam satu paragraf (`"GroupDRO (Sagawa* et al. 2020), Deep Feature Reweighting…"`) yang kemudian diulang hampir persis di §2. Jones dan Cresswell menaruh nol daftar sitasi di intro; motivasi mereka konkret (radiologi, pleural effusion + chest tube; dokter yang memakai prediction set).
2. **Daftar kontribusi adalah pembuangan angka.** Enam butir memuat 0.358, 0.024, +0.06, +0.35, 0.222, 0.004, 0.249, 0.084 [0.076,0.095], 0.014, 0.9024, 0.900, 0.85–0.89, 0.010, [0.945,0.999], 0.02, 30/34. Setiap angka itu ada di tabel. Jones memberi **satu** ilustrasi numerik di seluruh intro (65%→75% vs 77%→95%). Cresswell memberi **nol**.
3. **Mekanismenya tidak ada.** Jones menyebut margin distribution di paragraf keempat intro. Anda tidak pernah menyebut Eq (3).
- **Perbaikan:** intro 2 halaman: (a) waterbird + angka 89/51 — pertahankan, itu bagian terbaik naskah; (b) satu kalimat mekanisme dari Eq (3); (c) **tiga** kontribusi, bukan enam, masing-masing satu kalimat dengan paling banyak satu angka; (d) buang paragraf related work ke §2.
- **Severity:** blocking.

### 31. Related work: sebagian diposisikan, sebagian didaftar
- Diposisikan dengan baik: paragraf group-conditional/fair CP diakhiri dengan `"Our work is the empirical complement on the axis these papers hold fixed"` — itu benar dan tajam.
- Sekadar daftar: `"We treat conformal prediction over learned concept or concept-bottleneck spaces (Zhang et al. 2025; Stammer et al. 2024; Turan et al. 2026) as an interpretability layer, orthogonal to our thesis."` Tiga sitasi untuk mengatakan "tidak relevan". Buang kalimatnya.
- **Severity:** worth fixing.

---

## E. Sitasi

### 32. Romano et al. (2020a) tidak ada — dan itu sumber gagasan sentral Anda
*With Malice Toward None: Assessing Uncertainty via Equalized Coverage* (Harvard Data Science Review) adalah asal usul Equalized Coverage sebagai standar keadilan CP. Cresswell mengutipnya delapan kali; seluruh papernya adalah bantahan terhadapnya. Paper Anda tentang equalized coverage dan tidak menyebutnya. **Blocking.**

### 33. `balanced subsampling` dipakai sebagai metode tanpa sitasi
Muncul di §5, Table 5, Table 7, App. D — tidak pernah dikutip. Rujukannya: Idrissi et al., *Simple data balancing achieves competitive worst-group-accuracy* (CLeaR 2022). **Blocking** (metode tanpa atribusi).

### 34. Jones dikutip dengan venue salah, dan tidak dipakai padahal paling dekat
- Venue: Anda menulis `Jones E, Sagawa S, Koh PW, et al (2020) … "I Can't Believe It's Not Better!" NeurIPS 2020 workshop`. Versi terbit adalah **ICLR 2021** (arXiv v3, 14 Apr 2021 — versi yang Anda lampirkan sendiri). Perbaiki.
- Dipakai atau tidak: satu kalimat di §2, `"found abstention can leave worst-group accuracy flat; our set-valued analyses extend this lineage"`. Padahal dekomposisi mereka `F = pFwg + (1 − p)Fothers` **identik secara struktural** dengan mixture Anda `F = πFwg + (1 − π)F¬wg` di Prop 1(ii), dan pertanyaan mereka (apakah prosedur post-hoc berbasis-ambang membantu grup terburuk) adalah pertanyaan Anda dengan kuantitas berbeda.
- **Perbaikan:** di §3, setelah Eq (3), tambahkan: `"Jones et al. (2021) decompose the margin distribution the same way to ask when selective classification helps the worst group; there the pooled threshold is a confidence cut and the quantity is accuracy, here it is a conformal quantile and the quantity is coverage. Their negative answer and ours have the same source: one threshold read against two different score distributions."` Itu mengubah sitasi daftar jadi sitasi yang bekerja, sekaligus memberi paper Anda garis keturunan.
- **Severity:** blocking.

### 35. Rujukan lain yang hilang
- **Vovk et al. (2003)**, *Mondrian confidence machine* — sitasi kanonik untuk Mondrian; Anda hanya menyebut buku 2005.
- **Ding et al. (2024)**, *Class-conditional conformal prediction with many classes* — langsung relevan dengan masalah kalibrasi per-grup berukuran kecil yang jadi seluruh isi Appendix B.
- **Vovk (2012)** / **Lei & Wasserman (2013)** — ketidakmungkinan conditional coverage; Anda hanya mengutip Barber et al. (2019).
- **Kumar et al. (2022)** — lihat #8.
- **Guo et al. (2017)** — karena judul Anda memakai kata "calibration" dalam arti yang **berbeda** dari arti standarnya di literatur ini (Yang et al. 2023 mengukur calibration error). Satu kalimat yang membedakan keduanya akan mencegah salah baca.
- **Severity:** worth fixing (Ding dan Vovk 2003 mendekati blocking).

### 36. Ketidakcocokan tanggal kecil
`Turan B, … (2026) Neural concept verifier … arXiv:2507.07532` — ID arXiv Juli 2025 dengan tahun 2026. Periksa seluruh entri arXiv.
- **Severity:** minor.

---

## F. Hal yang tidak Anda tanyakan

### 37. Deklarasi etika bertentangan dengan §6.8
`"Ethics approval. Not applicable; this study uses only public benchmark datasets"` — sementara §6.8 memuat paragraf serius tentang inferensi atribut terproteksi terhadap individu, termasuk `"wrongly for 2–5% of people even at the AUROC we measure"`. Deklarasi harus menunjuk ke sana: `"Not applicable for human-subjects review; Section 6.8 discusses the ethics of inferring a protected attribute at test time and scopes the recommendation accordingly."` **Worth fixing**, dan menjaga R2.5 tetap terlihat dari halaman deklarasi.

### 38. `"≥ 3 training seeds × ≥ 10 calibration splits"` — angka pasti disembunyikan di balik ≥
Muncul di §4, §5, App. D. Appendix D sebenarnya menyebutkan angkanya (5 seed untuk grid, 3 untuk ablasi, split 0–9). R2.6 meminta detail reproducibility penuh; tanda ≥ justru melawan itu.
**Perbaikan:** nyatakan tepat, per studi, dalam satu kalimat di §5. **Worth fixing.**

### 39. Angka 89%/51% muncul tiga kali sebelum halaman 4
Abstract, §1 paragraf 1, Figure 1c. Itu boleh — ini angka pembuka Anda — tapi §1 mengulanginya dalam bentuk kalimat penuh yang hampir identik dengan abstract. Ubah §1 jadi rujukan ke Figure 1 dan hemat empat baris.
**Minor.**

### 40. Judul
`"Calibration Beats Robust Training for Worst-Group Coverage"`
Tiga masalah. (a) *Beats* adalah kata leaderboard untuk paper yang secara eksplisit `"a characterization rather than a method"` — judul dan pembingkaian saling bertentangan. (b) *Calibration* ambigu: di literatur ini artinya kalibrasi probabilitas (ECE), yang bukan yang Anda maksud; yang menang adalah **group-conditional** calibration. (c) Tidak ada kata "conformal" atau "spurious" — orang yang mencari paper Anda tidak akan menemukannya.

Bandingkan pola ketiga referensi: klaim + kejutan, tanpa kata kompetitif. *"Selective Classification Can Magnify Disparities Across Groups"*, *"Conformal Prediction Sets Can Cause Disparate Impact"*, *"Last Layer Re-Training Is Sufficient for Robustness to Spurious Correlations"*.

Kandidat, urut preferensi saya:
1. **"Group-Conditional Calibration, Not Robust Training, Governs Worst-Group Coverage"** — komparatif (menjaga R3.1), scope-nya jelas (menjaga R1.2/R2.1), menamai mekanismenya.
2. **"Robust Training Does Not Buy Conformal Reliability for the Worst Group"** — lebih menarik, tapi klaim negatif terasa *lebih* kuat, jadi periksa ulang terhadap R3.1.
3. **"Worst-Group Coverage Is a Calibration Property"** — paling ringkas, tapi kehilangan sisi komparatif yang diminta R3.1.

Saran: nomor 1. **Worth fixing**, dan jangan lupa menyelaraskan heading §6.1 serta kalimat abstract yang mencerminkan judul.

### 41. Abstract
173 kata, dan **arc-nya sudah benar**: satu keyakinan (`"it is natural to assume"`), satu pivot (`"We find that the two come apart"`), dua temuan (coverage; set size), satu rekomendasi (`"the calibration rule is the place to start"`). Ini bagian terkuat naskah dan tidak perlu dibongkar. Dua hal kecil: kata "conformal" tidak muncul sampai keyword; dan `"under one threshold shared by all groups"` bisa lebih pendek. Selebihnya biarkan. **Minor.**

---

## Lima perubahan yang paling menaikkan naskah ini ke standar ketiga referensi

1. **Jadikan Eq (3) tulang punggung.** Pindahkan ke akhir §1 sebagai kalimat mekanisme; di §6.1 hitung sisi kanannya per setting dan tunjukkan ia memprediksi lift yang teramati (satu panel, delapan titik); laporkan `DKS` alih-alih W1 di Table 6 dan Figure 4. Ini yang membedakan karakterisasi dari pengukuran, dan ini persis yang Jones lakukan dengan margin distribution. (#10, #11)
2. **Potong main body ke ±13 halaman dan bangun appendix yang berisi tabel, bukan prosa.** Dua belas tabel yang saya daftarkan di #26 sudah ada datanya — 116.000 evaluasi Anda tidak berguna bagi pembaca selama tinggal di file rilis. (#25, #26, #27)
3. **Bersihkan lapisan pipeline sampai habis.** gate ×3, "robust arms" di Figure 4, `method` yang berarti *run*, label H1–H6 di caption tabel, Appendix D yang masih dua backbone, placeholder DOI, `(elm 2024)`. Setiap satu di antaranya, sendirian, memberi tahu reviewer bahwa naskah belum dibaca ujung ke ujung. (#1–#8)
4. **Bunuh dua tik dominan.** Konstruksi *"X, not Y"* (sisakan tiga dari lima belas) dan kalimat pembelaan preemptif (hapus kedelapan). Lalu tulis ulang kalimat pembuka setiap subseksi §6 agar dimulai dari objek dan temuan, seperti Jones dan Cresswell, bukan dari prosedur dan label hipotesis. (#13, #14, #15, #22)
5. **Hadapi Cresswell.** Satu paragraf di §7 yang mengakui bahwa di bawah kriteria berbasis-outcome rekomendasi Anda berbalik, dengan angka set-size disparity Anda sendiri sebagai harganya. Saat ini paper Anda merekomendasikan persis hal yang salah satu paper referensinya sebut merugikan, dan tidak menyebutnya. (#9)

---

## Verdict jujur

Kalau saya membaca ini tanpa konteks: **saya akan menyimpulkan bahwa seorang peneliti nyata menjalankan eksperimen nyata, dan sebagian besar prosa terakhirnya melewati model.** Itu penilaian yang lebih baik daripada "ditulis model", tapi bukan yang Anda inginkan.

Yang membocorkannya, spesifik:

- **Daftar kontribusi.** Enam butir, enam belas angka, semuanya sudah ada di tabel. Manusia yang lelah memangkas; model mengoptimalkan kelengkapan.
- **Ritme "X, not Y" lima belas kali.** Ini tanda paling kuat. Tidak ada penulis manusia yang mempertahankan koreksi-diri itu selama dua puluh halaman.
- **Keseragaman penutup paragraf.** Tujuh dari delapan subseksi §6 berakhir dengan klausa moral atau penunjuk-silang. Manusia menulis paragraf yang berhenti begitu saja.
- **Kalimat yang membela keputusan yang belum ditantang.** `"Clipping sets to a minimum size of one would hide this, so we do not."` Tidak ada yang bertanya. Model mengantisipasi keberatan karena itu yang dilatihkan padanya.
- **Kalimat meta.** `"The results compose into one story."` `"We frame the paper as a characterization rather than a method."` Penulis manusia menceritakan cerita itu; model mengumumkan bahwa ada cerita.
- **Panjang kalimat yang terlalu rata.** Buka halaman mana pun di §6: hampir setiap kalimat tiga klausa, 25–40 kata, dengan em-dash di sepertiga terakhirnya. Jones bergantian antara kalimat 8 kata dan 35 kata.
- **Dan yang paling telak, sisa pipeline:** "robust arms" di legenda gambar, "gate" tiga kali, "Of the 200 methods". Itu bukan tanda model — itu tanda naskah yang diedit oleh find-replace dan tidak pernah dibaca dari halaman satu sampai akhir oleh seorang manusia dalam satu duduk.

Yang jelas ditulis manusia, dan harus Anda pertahankan apa adanya: paragraf pembuka waterbird; desain Figure 1 (empat panel yang menyatakan confound, mekanisme, harga, dan biayanya — itu figure yang bagus); diagnosis class-prior inversion AFR di Appendix C, lengkap dengan oracle upper bound γ, karena itu temuan spesifik yang hanya bisa ditemukan orang yang mengaduk datanya sendiri; pengungkapan himpunan kosong di DINOv2/Waterbirds; dan paragraf etika di §6.8, yang membedakan inferensi dari pengungkapan dan menyebut angka 2–5% — itu tulisan orang yang benar-benar memikirkannya.

Jarak antara bagian-bagian itu dan sisanya adalah masalah sebenarnya. Bagian yang bagus terbaca seperti Anda. Sisanya terbaca seperti seseorang yang menjelaskan pekerjaan Anda dengan hati-hati kepada auditor.

---

## Usulan pembagian main body / appendix, dengan target halaman

**Main body — target 13,5 halaman + referensi**

| Bagian | Isi | Target |
|---|---|---|
| §1 Introduction | waterbird + 89/51; mekanisme Eq (3) satu kalimat; **tiga** kontribusi, ≤1 angka masing-masing. Paragraf related-work dibuang ke §2 | 2,0 |
| §2 Related work | tiga paragraf (group-robust training; group-conditional & fair CP termasuk Romano 2020a dan Cresswell; CP under shift + Jones sebagai garis keturunan). Paragraf concept-bottleneck dihapus | 1,25 |
| §3 Setup, notation, and the identity | notasi; tiga score; tiga policy; ∆cov dan set-size disparity; Prop 1(ii) + Eq (3) + Eq (4) + **satu** remark. Prop 1(i) → footnote. W1 → appendix | 2,0 |
| §4 Experimental setup | dataset, empat backbone, konstruksi ρ, lima last-layer method, protokol accuracy-matching (1 paragraf), kalimat pre-specified (R2.4), floors, Table 2 + 3 kalimat perbandingan (R1.3) | 2,0 |
| §5.1 The calibration rule decides worst-group coverage | Figure 2 + **figure baru prediksi-vs-observasi dari Eq (3)**; Table 3 versi ringkas (APS saja) | 1,75 |
| §5.2 Retraining the representation | Table 4; manipulation check dipadatkan; Figure 3 dibuang (R1.2 / R2.1) | 1,25 |
| §5.3 What training buys: smaller sets | Table 5 tanpa kolom spread; TOST (R3.1); himpunan kosong | 1,0 |
| §5.4 Shift, and groups without labels | ρ-sweep (Figure 5) + predicted-group (Table 8) + paragraf etika (R2.5) digabung; Table 6 + Figure 4 versi ringkas (R2.3) | 1,5 |
| §6 Discussion & limitations | termasuk paragraf rekonsiliasi Cresswell; limitasi tanpa klausa "though" | 1,25 |
| §7 Conclusion | | 0,5 |

**Appendix — target ±16 halaman**

| | Isi | Target |
|---|---|---|
| A | Bukti: Prop 1(i) dan (ii); versi finite-sample Eq (3); turunan konstanta min-over-groups; peran W1 vs `DKS` | 2,5 |
| B | Efek min-over-groups: Figure B1, Table B1, coverage per grup di semua setting **(R1.1)** | 2,0 |
| C | Grid utama: tabel penuh per score, per policy, per method — coverage, mean-over-groups, ∆cov, set size, set-size disparity, `DKS`, W1 | 4,0 |
| D | Kebijakan kalibrasi: shift-robust penuh; sensitivitas ε; perbandingan tiga kebijakan | 1,5 |
| E | Studi representasi: 3 objective × 5 seed × 3 score × 2 dataset **(R1.2)** | 1,5 |
| F | Predicted-group: tiga kondisi × tiga score × setiap method; analisis kegagalan **(R2.5)** | 1,5 |
| G | Hipotesis pra-spesifikasi, floors, method yang dikecualikan, sensitivitas dihitung ulang penuh, sweep γ AFR **(R2.4)** | 2,0 |
| H | Reproducibility: hyperparameter untuk **empat** backbone, seed/split eksak, dekomposisi varians, compute, DOI **(R1.4, R2.6)** | 2,0 |

Semua tiga belas titik reviewer tetap tertangani di bawah pembagian ini. Empat yang bergantung pada penempatan main-body — R1.2, R2.3, R2.5, R3.1 — saya tandai eksplisit di tabel; jangan pindahkan yang itu.
