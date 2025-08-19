# TASKS.md — Tree Species Classification (تقسيم المهام بين محمد وعماد)

> الهدف: نقسم المشروع باش كل واحد يقدر يخدم على مهامه بلا ما يبقى معوّن (blocked) بزاف من الآخر. المهام مكتوبة بالدارجة المغربية، والمصطلحات العلمية بالفرانساوي/الانجليزية بين قوسين.

---

## قواعد عامة (Rules générales)

* كل واحد يخدم فـbranch ديالو فـGit ويدير PR (pull request) ملي يكمّل feature صغيرة. (Git, Pull Request)
* كل مخرجات preprocessing (sampled points, views, descriptors) خاصّهم يتخزنو فـ`data/processed/` بنفس structure باش ما يكونش dependency على path مختلفة.
* `test.csv` = ثابت. متلمسوه حتى للـfinal evaluation.
* كل واحد يكتب ملف `experiments/<method>/config.yaml` فيه الـconfig و seed. (Reproducibility)

---

## ملاحظات على الاعتماد المتبادل (Dependency notes)

* الهدف: نخطّط المهام باش يقدَر كل واحد يتقدّم في العمل بلا ما ينتظر الآخر طول الوقت.
* بعض المهام مرتبطة بحكم الطبع (مثلاً: rendering ديال الصور خاصّ prior to multi-view CNN) لكن نقدرو نديرو cache/placeholder dataset (extract features on small sample) باش صاحبو يبدا يخدم على الموديل بلا انتظار.
* أي ملف output مهم (sampled points, views, descriptors) نعلنو عليه فور ما يخرج باش الشخص الآخر يقدر يستعملو.

---

# تقسيم المهام (Owner: Mohamed / Owner: Imad)

> ملاحظة: كتلقى فكل مهمة `Outputs` لي خاص تولد بعد ما تكمل المهمة — هادو هما الملفات اللي الطرف الآخر يقدر يستعملهم بلا ما ينتظر التفاعل.

## جزء A — Data & Preprocessing (Owner: Mohamed)

### A1. Preprocessing core (`prepare_all.py`)

* الوصف: سكريبت كيقرأ raw point clouds (.ply/.pcd/.txt)، يدير cleaning (outlier removal), centering, scaling (unit sphere), compute normals، ثم يعمل FPS (N=1024) ويخزّن الـsampled point clouds.
* مخرجات (Outputs): `data/processed/sampled/<id>.npy` (xyz), `data/processed/normals/<id>.npy`
* لماذا مهم: أي طرف خدام على PointNet ولا descriptors يقدر يستعمل هاد الملفات بلا انتظار render.
* ملاحظة: ضيف CLI args ( --raw\_dir --out\_dir --npoints ) و logs.

### A2. Quick EDA script

* الوصف: سكريبت صغير يعطينا counts per class, distribution, ويعمل preview لــ10 samples (render inline). (EDA — Exploration Descriptive)
* Outputs: `reports/eda_summary.md`, `reports/sample_images/*.png`

### A3. 3D Descriptors extraction (baseline)

* الوصف: حساب FPFH وVFH + global histograms لكل sample، وتخزينهم كـ`.npy` أو `.csv`.
* Outputs: `data/processed/descriptors/FPFH_<id>.npy`, `VFH_<id>.npy`, `descriptors_table.csv`
* ملاحظة: هاد الملفات كتسمح لعماد ولا لأي واحد يبدأ ويبني RandomForest/XGBoost بلا انتظار render views.

---

## جزء B — Multi-view & 2D pipelines (Owner: Imad)

### B1. Multi-view rendering (`render_views.py`)

* الوصف: Offscreen render V=12 views لكل point cloud (224x224)، وخزنهم فـ`data/processed/views/<id>/view_01.png`...
* Outputs: `data/processed/views/<id>/*.png`
* ملاحظة: إذا محمد خلّص A1 قبل B1، العماد يستعمل ملفات sampled; إلا ما كانوش موجودين يقدر يستعمل raw pcl مباشر مع render script.

### B2. Multi-view feature extraction (CNN backbone)

* الوصف: سكريبت لاستخراج features من كل view باستعمال ResNet50 pretrained (feature extraction mode—freeze backbone)، يخزن vectors.
* Outputs: `data/processed/mv_features/<id>.npy` (aggregation per object optional: mean, max)
* لماذا مهم: Mohamed يقدر يخدم على direct models في نفس الوقت بلا انتظار fine-tuning.

### B3. Multi-view fine-tuning & classifier

* الوصف: training script باش تدير fine-tune ديال ResNet head أو تدريب head فقط ثم unfreeze، مع logging.
* Outputs: `experiments/mv_resnet/checkpoint.pth`, `experiments/mv_resnet/metrics.csv`

---

## جزء C — Models training & Baselines (Parallel work possible)

### C1. Baseline descriptors + RandomForest (Owner: Imad)

* الوصف: يستعمل descriptors من A3، StandardScaler + RandomForest/XGBoost، يعطي baseline metrics.
* Inputs: `data/processed/descriptors/*`
* Outputs: `experiments/desc_rf/model.pkl`, `experiments/desc_rf/metrics.csv`
* ملاحظة: Independent — ما يحتاج render views.

### C2. PointNet training (Owner: Mohamed)

* الوصف: يستخدم sampled points من A1، يبني PointNet skeleton، يدير augmentations (rotation, jitter), training loop، save checkpoints.
* Inputs: `data/processed/sampled/*`
* Outputs: `experiments/pointnet/checkpoint.pth`, `experiments/pointnet/metrics.csv`
* ملاحظة: Mohamed يقدر يبدا يخدم على هاد المهمة مباشرة بعد A1.

### C3. Multi-view + classical stacking (Owner: Imad)

* الوصف: يستعمل `mv_features` + descriptors table للتدريب على XGBoost stacking أو SVM.
* Inputs: `data/processed/mv_features/*`, `data/processed/descriptors/*`
* Outputs: `experiments/mv_xgb/metrics.csv`

---

## جزء D — Evaluation, Analysis & Plots (Can be shared)

### D1. Unified evaluation script (`eval.py`) (Owner: Mohamed)

* الوصف: سكريبت واحد كيقرأ كل checkpoints ويحسب metrics على `test.csv` (Accuracy, Balanced Accuracy, Macro F1, confusion matrices), يخزن النتائج وplots.
* Inputs: checkpoints من `experiments/*`, `data/processed/*`, `data/test.csv` (ثابت)
* Outputs: `results/<method>/*` (metrics.json, confusion\_matrix.png)
* لماذا مهم: هذا هو اللي كيعرّض النتائج النهائية.

### D2. Comparative analysis (Owner: Imad)

* الوصف: script/jupyter notebook كيجمع metrics من كل تجربة ويرسم charts (accuracy vs time, confusion matrices grid), و يعطي interpretation notes.
* Outputs: `reports/comparison_plots/*.png`, `reports/comparison_notes.md`

---

## جزء E — DevOps, Docs & Delivery (Shared / split)

### E1. README & Usage examples (Owner: Mohamed)

* الوصف: تحديث README باش يعكس الأوامر الفعلية لتشغيل preprocessing, train, eval.
* Outputs: `README.md` محدث، `requirements.txt` مربوط بالإصدارات (pinned)

### E2. Slides & Final packaging (Owner: Imad)

* الوصف: تجهيز slides (≤20) يشرح الهدف، الطرق، أهم النتائج، and demo screenshots. Imad يجهز الslides وMohamed يعطيه الـplots وصور النتائج.
* Outputs: `slides/presentation.pdf` أو `slides/presentation.pptx`

### E3. CI / quick-tests (Owner: Shared)

* الوصف: إعداد simple GitHub Action أو scripts لتشغيل unit test صغير (preprocessing on 2 samples) + lint. كل واحد يساهم بقواعد التست.
* Outputs: `.github/workflows/ci.yml`، `tests/test_preprocess.py`

---

## قواعد التسليم بينكم (Delivery conventions)

* كل ما يسالي واحد مهمة ويعمل output، يعلن فـSlack/WhatsApp أو Issue comment مع path ديال الملفات. الطرف الآخر يقدر يستعمل Output مباشرة.
* Exceptions: الا ملفات كبيرة بزاف (مثلاً views كاملين)، نعتمدوا على archiving (zip) أو مشاركة عبر drive ونعطيو path في repo (مثال: `data/processed/views_archive.zip`).
* Logging: كل تجربة يخزن `metrics.csv` و `config.yaml` في `experiments/<id>/`.

---

## نصائح باش تبقى الخدمة غير متشابكة (Tips pour non-blocking work)

* استعملوا *small contracts*: مثال `prepare_all.py` خصو يعطي ملف `sampled/<id>.npy` و`descriptors_table.csv` — هادو هما الcontract بين Mohamed وImad.
* كل شيء واضح ومكتوب: path، format (numpy .npy / csv)، و dimension. هكا العماد ما يبقاش ينتظر.
* استعمل placeholder data (10–20 samples) باش تبداو التدريب/الـdebug قبل full run.

---

## خاتمة

هاد الفورمات مهيّأ باش كل واحد يخدم على المهام ديالو بلا ما يكون معطّل من الآخر. إلا بغيت نبدّل ownership على شي مهمة ولا نزيد تفاصيل (بحال commands مفصّلة لكل script) نقدر نكتبهم دابا.

*إذا موافق، نقدر نصدّر هاد TASKS.md فـrepo ولا ندير issue templates جاهزين لكل مهمة.*
