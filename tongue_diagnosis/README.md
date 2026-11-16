# Intelligent Tongue Diagnosis Model for Gastrointestinal Diseases

This project is a VTU final-year research prototype. The current local dataset is **not verified as a GI-disease dataset**: its source provenance, clinical labels, patient identifiers, and licence metadata are absent. In particular, positive filenames use the `diabetes_*` pattern. Therefore, the current checkpoint and UI must not be interpreted as GI-disease prediction or clinical diagnosis.

The legacy app remains available for demonstration, but new experiments are gated by the task configuration in `configs/task_config.json`.

## Installation

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset and reproducible split audit

Keep the raw dataset unchanged under:

- `data/raw/Normal/`
- `data/raw/GI_Disease/`

Generate a non-sensitive manifest and audit. It records requested image metadata, file hashes, and duplicate groups, but deliberately excludes EXIF values and other potentially identifying metadata.

```bash
python data/dataset_audit.py --task-config configs/task_config.json
```

Create a fixed, duplicate-aware 70/15/15 image-level split:

```bash
python data/generate_splits.py --manifest data/manifests/dataset_manifest.csv --seed 42
```

This produces:

- `data/manifests/dataset_manifest.csv`
- `data/manifests/dataset_audit_report.json`
- `data/manifests/split_manifest.csv`
- `data/manifests/split_summary.json`

Every image is assigned once and all detectable duplicate groups stay within one split. This does **not** guarantee patient-level independence because no patient identifiers are available.

## Baseline experiment status

`configs/baseline_config.json` defines the intended manifest-driven baseline and locks the test split. Training and evaluation have deliberately not been run or implemented in Stage 2. The existing commands below do **not** use the split manifest and must not be used to make a reproducible-baseline or GI-disease claim.

For an explicitly approved, non-clinical exploratory run on the current unverified labels:

```bash
python experiments/manifest_experiments.py --output-dir experiments/results/manifest_seed42 --allow-unverified-exploratory
# Compare validation results, then evaluate exactly one selected experiment once:
python experiments/evaluate_selected.py --run-dir experiments/results/manifest_seed42 --experiment <selected_experiment>
```

The final evaluator must run the locked test split once, after the model, preprocessing, threshold, and checkpoint have all been selected using training and validation data only.

## Optional Tongue-FLD image-only experiment

The Tongue-FLD image dataset is not included in this repository. The scripts below expect it locally under `data/external/tongue_fld/`; obtain it from the original authors and confirm its reuse terms before downloading or redistributing it. For the current app model, run the transfer pretraining and fine-tuning workflow:

```bash
python training/pretrain_finetune_current.py
```

This adapts MobileNetV2 on Tongue-FLD training images, then fine-tunes on the current local `Normal`/`GI_Disease` training split. The experimental candidate was not promoted because the existing app model scored better on the validation split; its candidate checkpoint is not included here. The active app checkpoint is `training/saved_model/best_model.h5`; its decision threshold is read from `training/saved_model/prediction_config.json` and was selected on the 45-image validation split. At that threshold, validation accuracy was 80.00% and balanced accuracy 78.15%. The folder labels remain unverified (positive filenames are diabetes-related), so these figures are not validated GI-disease accuracy. A separate FLD-only model was also evaluated, but is not used by the app or included here.

## Legacy commands

These preserve the existing prototype behavior but reuse a validation split and are not valid for the manifest-driven baseline:

```bash
python training/train.py
python evaluation/evaluate.py
```

## Launch the Streamlit UI

```bash
streamlit run app/streamlit_app.py
```
