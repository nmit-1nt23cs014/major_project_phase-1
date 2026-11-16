# Tongue-FLD dataset source and model notes

## Source

- Dataset/code release: [MFF-TDF](https://github.com/MLDMXM2017/MFF-TDF) and [HM-TDF](https://github.com/MLDMXM2017/HM-TDF).
- The release contains 5,717 tongue images and a CSV with FLD labels and eight physiological indicators.
- Labels are based on fatty liver disease (FLD) grades: `_0` means No FLD, `_1` mild FLD, and `_2` moderate/severe FLD. The binary training task maps `_0` to `No_FLD` and `_1`/`_2` to `FLD`.
- The dataset paper reports a multimodal approach using tongue images and physiological indicators. This project’s candidate uses tongue images only, so the paper’s reported multimodal results do not describe this model.

## Local layout

- Images: `images_full/Tongue_Images/` (5,717 JPG files)
- Labels and indicators: `raw_tar/Indicator_and_Label.csv`
- Training split: `../../manifests/tongue_fld_split.csv`
- FLD-only image model candidate: `../../../training/saved_model/fld_model.h5`
- FLD-only candidate metrics: `../../../training/saved_model/fld_metrics.json`
- Transfer pretraining and fine-tuning workflow for the current app model: `../../../training/pretrain_finetune_current.py`
- Transfer candidate validation metrics: `../../../training/saved_model/current_pretrained_candidate_metrics.json`
- Reproducible training entry point: `../../../training/train_tongue_fld.py`

## Important limitations

- FLD is a specific fatty-liver task; it is not a general GI-disease label and must not be mapped to the app’s `GI_Disease` class.
- The separate FLD-only image classifier achieved 58.97% held-out accuracy and 0.624 held-out ROC AUC on its recorded split, below the No-FLD majority baseline of 64.57%. That FLD-only classifier is not the app model.
- The MobileNetV2 transfer candidate reached 68.89% validation accuracy and 0.712 AUC on the local two-folder validation split, but was not selected because the backed-up app model scored better after threshold calibration. The active app still uses the previous checkpoint. These are exploratory results on unverified local labels, not validated GI-disease accuracy.
- The dataset release’s GitHub repositories do not declare a license file. Confirm reuse and redistribution terms with the authors before redistributing the images or a derived package.
- The released CSV does not establish patient-level identifiers for this training workflow, so independence between images from the same person cannot be confirmed.
- This is an academic experiment, not a diagnostic system. Model scores are not calibrated clinical probabilities.
