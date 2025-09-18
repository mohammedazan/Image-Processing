# TASKS.md — Tree Species Classification (تقسيم المهام بين محمد وعماد)

> الهدف: نقسم المشروع باش كل واحد يقدر يخدم على مهامه بلا ما يبقى معوّن (blocked) بزاف من الآخر.
> المهام مكتوبة بالدارجة المغربية، والمصطلحات العلمية بالفرانساوي/الإنجليزية بين قوسين.

---

## قواعد عامة (Rules générales)

* كل واحد يخدم فـbranch ديالو فـGit ويدير PR (pull request) ملي يكمّل feature صغيرة. (Git, Pull Request)
* كل مخرجات preprocessing (sampled points, views, descriptors) خاصّهم يتخزنو فـ`data/processed/` بنفس structure باش ما يكونش dependency على path مختلفة.
* `test.csv` = ثابت. متلمسوه حتى للـfinal evaluation.
* كل واحد يكتب ملف `experiments/<method>/config.yaml` فيه الـconfig و seed وطرق binarization. (Reproducibility)

---

## ملاحظات على الاعتماد المتبادل (Dependency notes)

* الهدف: نخطّط المهام باش يقدَر كل واحد يتقدّم في العمل بلا ما ينتظر الآخر طول الوقت.
* بعض المهام مرتبطة بحكم الطبع (مثلاً: rendering ديال الصور خاصّ prior to multi-view CNN)، لكن نقدرو نديرو cache/placeholder dataset (extract features on small sample) باش صاحبو يبدا يخدم على الموديل بلا انتظار.
* أي ملف output مهم (sampled points, views, descriptors) نعلنو عليه فور ما يخرج باش الطرف الآخر يقدر يستعملو. استعملو `manifest.csv` باش تبينو شنو processed وجاهز.

---

# تقسيم المهام (Owner: Mohamed / Owner: Imad)

> ملاحظة: فكل مهمة `Outputs` لي خاص تولد بعد ما تكمل المهمة — هادو هما الملفات اللي الطرف الآخر يقدر يستعملهم بلا ما ينتظر التفاعل.

---

## جزء A — Data & Preprocessing (Owner: Mohamed)

### A1. Preprocessing core (`prepare_all.py`)

* **الوصف:** سكريبت كيقرأ raw point clouds (.ply/.pcd/.txt)، يدير cleaning (statistical outlier removal)، centering، scaling (unit sphere)، compute normals، ثم يعمل FPS (N=1024) ويخزّن الـsampled point clouds.
* **مخرجات (Outputs):**

  * `data/processed/sampled/<id>.npy` (N×3 float32)
  * `data/processed/normals/<id>.npy` (N×3 float32)
  * `data/processed/manifest.csv` (id, status, paths)
* **ملاحظات:** ضيف CLI args (`--raw_dir --out_dir --npoints --workers`) و logging. دير small-unit tests على couple ديال الملفات قبل full-run.

### A2. Quick EDA script (`eda.py`)

* **الوصف:** سكريبت يعطينا counts per class، distribution, وبعض preview images (render inline أو save). (EDA — *Exploratory Data Analysis*)
* **Outputs:** `reports/eda_summary.md`, `reports/sample_images/*.png`

### A3. 3D Descriptors extraction — PCA-based features (Owner: Mohamed)

* **الوصف:** نخرّجو **PCA-based features** من كل sampled point cloud:

  * compute covariance matrix → eigen decomposition (λ1≥λ2≥λ3)
  * eigenvalue ratios (λ1/(λ1+λ2+λ3), λ2/..., λ1/λ2, λ2/λ3)
  * explained variance per component (pourcentage d'explication)
  * project points on first PC and compute histogram (e.g., 8 bins) → projection distribution
  * curvature/trace summaries أو أي إحصائيات إضافية مفيدة
* **Outputs:**

  * `data/processed/descriptors/pca_<id>.npy` (vector per sample)
  * `data/processed/descriptors/descriptors_table_pca.csv` (id + features)
* **ملاحظة مهمة:** هاد ال-features **مستمرة** → باش نخدمو مع BernoulliNB خاص binarization. **ما نعملوش binarization فهاد السكريبت**؛ binarization غادي تكون فالـ`train_bernoulli_nb.py` مع خيارات (`median`/`quantile`/`otsu`) وتخزين thresholds فـ`experiments/bernoulli/bin_thresholds.yaml`.

---

## جزء B — Multi-view & 2D pipelines (Owner: Imad)

### B1. Multi-view rendering (`render_views.py`)

* **الوصف:** Offscreen render `V=12` views لكل point cloud (224×224)، وخزنهم فـ`data/processed/views/<id>/view_01.png`...
* **Outputs:** `data/processed/views/<id>/*.png`
* **ملاحظة:** Imad يقدر يستعمل sampled points من A1 أو raw pcl. إن كان محمد طالع A1، استعملو sampled لتوحيد العملية.

### B2. 2D Descriptors extraction — Hu Moments (`extract_2d_hu.py`)

* **الوصف:** لكل view: convert to grayscale → compute moments (`cv2.moments`) → `cv2.HuMoments` → apply log transform (`-sign(m)*log10(abs(m))`) → aggregate across V views (mean, std).
* **Outputs:**

  * `data/processed/2d_hu/<id>_hu.npy` (aggregated vector, e.g., mean+std → 14 dims)
  * `data/processed/2d_hu/hu_features_table.csv` (id + features)
* **ملاحظة على binarization:** Hu features هم continuous → لازم binarize حسب strategy (global median أو Otsu). البينارايزيشن غادي يتطبق فالمرحلة ديال التدريب (C1) باش نحافظو على قابلية التكرار (reproducibility).

### B3. Multi-view feature extraction (optional backbone) (`mv_feature_extract.py`)

* **الوصف:** (اختياري) extract CNN features per view (ResNet50 frozen) → aggregate → save `mv_features`. هذشي يستخدم غير إذا بغيت features إضافية.
* **Outputs:** `data/processed/mv_features/<id>.npy`

---

## جزء C — Models training & Baselines (Parallel work possible)

> ملاحظة: الأستاذ طلب **بالضبط 5 طرق**. باش نجبدو 5 طرق منطقية ومتناغمة مع الـspec، هادي الاقتراحات اللي نطبقوهم ونسجلوهم فـ`experiments/`:

### الطرق الخمس المقترحة (The 5 methods)

1. **Method 1 — Indirect (Hu Moments + BernoulliNB)**

   * Multi-view render → Hu Moments (B2) → aggregate → binarize → BernoulliNB (C1). هادي طريقة خفيفة وسريعة وتستعمل descriptors ثنائية.
2. **Method 2 — Indirect (CNN features + SVM/XGBoost)**

   * Multi-view render → ResNet50 frozen feature extraction per view (B3) → aggregate → train SVM or XGBoost as classical classifier.
3. **Method 3 — Indirect (CNN fine-tune)**

   * Multi-view render → fine-tune pretrained CNN (ResNet50/EfficientNet) end-to-end on views (multi-view aggregation strategy).
4. **Method 4 — Quasi-Direct (PCA-based descriptors + BernoulliNB)**

   * A3 extracts PCA features → binarize → BernoulliNB (C1). هادي طريقة مقارِنة مباشرة مع Method 1 ولكن بثيم 3D.
5. **Method 5 — Direct (PointNet)**

   * Direct point-cloud model (PointNet) على sampled N=1024 (A1) — training end-to-end.

> هاد التشكيلة تغطي Indirect, Quasi-Direct, وDirect واللي كيطابقو متطلبات الملف.

### C1. Bernoulli Naive Bayes training (Owner: Imad)

* **الوصف:** حسب تعليمات الأستاذ: نديرو **BernoulliNB** على features ثنائية.

  * Input: features binary matrix X (من Hu + PCA بعد binarization)
  * Split: استعمال stratified split على remaining (validation), `test.csv` يبقى للfinal.
  * Train: `sklearn.naive_bayes.BernoulliNB()`، save model وmetrics.
* **Outputs:**

  * `experiments/bernoulli/bernoulli_model.joblib`
  * `experiments/bernoulli/metrics.csv` (accuracy, balanced\_accuracy, macro\_f1, per-class precision/recall)
  * `experiments/bernoulli/bin_thresholds.yaml` (thresholds + method + seed)
* **ملاحظة تقنية:** طريقة binarization لازم تكون **قابلة للاختيار** (`median`/`quantile`/`otsu`) مع default = `median (global on training set)`. لازم نوثّق ونسجلو thresholds باش reproducible.

### C2. PointNet training (Owner: Mohamed)

* **الوصف:** يستخدم sampled points من A1 باش يدرب PointNet (direct model) مع augmentations (rotation z-axis, jitter, scaling).
* **Inputs:** `data/processed/sampled/*`
* **Outputs:** `experiments/pointnet/checkpoint.pth`, `experiments/pointnet/metrics.csv`
* **ملاحظة:** PointNet مستقل على Hu/PCA → يقدَر Mohamed يخدم عليه بلا انتظار.

### C3. Multi-view + stacking (Owner: Imad)

* **الوصف:** يستعمل `hu_features` + `pca_features` (binary or continuous depending) باش يدير stacking/ensemble (meta-classifier ممكن يكون logistic/XGBoost). مع ذلك، الموديل الأساسي المفروض هو BernoulliNB.
* **Outputs:** `experiments/mv_stack/metrics.csv`

---

## جزء D — Evaluation, Analysis & Plots (Can be shared)

### D1. Unified evaluation script (`eval.py`) (Owner: Mohamed)

* **الوصف:** سكريبت كيقرأ كل checkpoints/models ويحسب metrics على `test.csv` (Accuracy, Balanced Accuracy, Macro F1, confusion matrices), يخزن النتائج وplots.
* **Inputs:** checkpoints من `experiments/*`, `data/processed/*`, `data/test.csv` (ثابت)
* **Outputs:** `results/<method>/metrics.json`, `results/<method>/confusion_matrix.png`

### D2. Comparative analysis (Owner: Imad)

* **الوصف:** notebook أو script كيجمع metrics من كل تجربة ويرسم charts (accuracy vs time, confusion matrices grid), ويعطي interpretation notes.
* **Outputs:** `reports/comparison_plots/*.png`, `reports/comparison_notes.md`

---

## جزء E — DevOps, Docs & Delivery (Shared / split)

### E1. README & Usage examples (Owner: Mohamed)

* **الوصف:** تحديث README باش يعكس الأوامر الفعلية لتشغيل preprocessing, extract features, train (BernoulliNB), eval.
* **Outputs:** `README.md` محدث، `requirements.txt` مربوط بالإصدارات (pinned)

### E2. Slides & Final packaging (Owner: Imad)

* **الوصف:** تجهيز slides (≤20) يشرح الهدف، الطرق، أهم النتائج، وshots من الplots. Imad يجهز الslides وMohamed يمدّو بالplots.
* **Outputs:** `slides/presentation.pdf` أو `slides/presentation.pptx`

### E3. CI / quick-tests (Owner: Shared)

* **الوصف:** إعداد simple GitHub Action أو scripts لتشغيل unit test صغير (preprocessing on 2 samples) + lint.
* **Outputs:** `.github/workflows/ci.yml`, `tests/test_preprocess.py`

---

## قواعد التسليم بينكم (Delivery conventions)

* كل ما يسالي واحد مهمة ويعمل output، يعلّمه للطرف الآخر عبر Issue comment أو Slack/WhatsApp مع path ديال الملفات. الطرف الآخر يقدر يستعمل Output مباشرة.
* Exceptions: إلا الملفات كبيرة بزاف (مثلاً views كاملين)، نعتمدوا على archiving (zip) أو مشاركة عبر drive ونعطيو path فـrepo (`data/processed/views_archive.zip`).
* Logging: كل تجربة يخزن `metrics.csv` و `config.yaml` في `experiments/<id>/`.

---

## Binzarization / Thresholds (خاص وموثّق)

* **قاعدة مقترحة:** استعملو **global median** على training set لكل feature كـthreshold: value >= median → 1 else 0.
* خزن thresholds فـ`bin_thresholds.yaml` مع ملاحظة طريقة الحساب (global/per-class/Otsu).
* بديل: استعمل quantile-based binning (مثلاً tertiles) ثم خرّجو presence (binary) لكل bin لو بغيت features أكثر تفصيلاً.

---

## Tips باش تبقى الخدمة غير متشابكة (Tips pour non-blocking work)

* استعملو *small contracts*: مثال `prepare_all.py` خصو يعطي `sampled/<id>.npy` و`descriptors_table_pca.csv` و`hu_features_table.csv` — هادو هما الcontract بين Mohamed وImad.
* كل شيء واضح ومكتوب: path، format (numpy .npy / csv)، و dimension. هكا العماد ما يبقاش ينتظر.
* استعمل placeholder data (10–20 samples) باش تبداو التدريب/الـdebug قبل full run.
* سجّل كل تجربة (config, seed, metrics) فـ`experiments/<id>/`.

---

## خاتمة وNext steps المقترح

* Mohamed يبدى دابا بـ**A3 (PCA extraction)** على 10–20 sample كنموذج اختبار.
* Imad يقدر يبدا B1 (render\_views) على نفس الـ10–20 sample أو يستعمل few raw pcl لبدء استخراج Hu Moments.
* من بعد ما يكونو outputs موجودين (manifest updated)، Imad يقدر يدير C1 (BernoulliNB) على features binarized بينما Mohamed يخدم على PointNet (C2).
* إلا بغيتي، نولّدلك باش نبدأ: `extract_3d_pca_features.py`, `extract_2d_hu.py`, و `train_bernoulli_nb.py` جاهزين للّصق — قول فقط نولّدهم دابا.

*إذا موافق، نقدر نصدّر هاد TASKS.md مباشرة فـrepo أو نفتح لك issues جاهزين لكل مهمة.*


Project structure tree =
├───Daily
├───data
│   ├───features = features_table.csv
│   ├───processed
│   │   ├───2d_hu
│   │   ├───descriptors
│   │   ├───mv_features
│   │   ├───normals
│   │   ├───sampled
│   │   └───views
│   └───raw
│       ├───dataverse_files
│       │   ├───Buche
│       │   ├───Douglasie
│       │   ├───Eiche
│       │   ├───Esche
│       │   ├───Fichte
│       │   ├───Kiefer
│       │   └───Roteiche
│       └───Test
│           ├───Buche
│           ├───Douglasie
│           ├───Eiche
│           ├───Esche
│           ├───Fichte
│           ├───Kiefer
│           └───Roteiche
├───experiments
     ├───bernoulli
     │       bernoulli_model.joblib
     │       bin_thresholds.yaml
     │       config.yaml
     │       metrics.csv
     │       summary.txt
     │
     ├───mv_stack
     │       bin_thresholds.yaml
     │       metrics.csv
     │       predictions.csv
     │
     └───pointnet
             checkpoint.pth
             last_checkpoint.pth
             metrics.csv
│   
│   
├───notebooks
├───reports
    │   │   class_counts.csv
    │   comparison_notes.md
    │   comparison_table.csv
    │   eda_summary.md
    │   hu_feature_stats.csv
    │   metrics.csv
    │   pca_feature_stats.csv
    │   point_counts_per_file.csv
    │   read_errors.log
    │
    ├───comparison_plots
    │       accuracy_vs_time.png
    │       confusion_grid.png
    │       metrics_barplot.png
    │
    ├───plots
    │       class_balance_bar.png
    │       points_boxplot_per_class.png
    │       point_count_histogram.png
    │
    ├───pointnet
    │       metrics.json
    │
    └───sample_images
            Buche_103_hu_preview.png
            Buche_103_preview.png
            Buche_106_hu_preview.png
            Buche_106_preview.png
            Buche_107_hu_preview.png
            Buche_107_preview.png
            Buche_109_hu_preview.png
            Buche_109_preview.png
            Buche_110_hu_preview.png
│   
│   
│   
├───results
│   └───bernoulli
└───src
   ├   compare_results.py
   │   eval.py
   │   test.csv
   │
   ├───features
   │       extract_2d_hu.py
   │       extract_3d_pca_features.py
   │       mv_feature_extract.py
   │
   ├───models
   │   │   train_bernoulli_nb.py
   │   │   train_pointnet.py
   │   │   train_stacking.py
   │   │
   │   └───__pycache__
   │           train_bernoulli_nb.cpython-313.pyc
   │           train_pointnet.cpython-313.pyc
   │
   └───preprocessing
          eda.py
          prepare_all.py
          render_views.py
