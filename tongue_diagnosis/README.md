# Intelligent Tongue Diagnosis Model for Gastrointestinal Diseases

This project is a VTU final year major project prototype for tongue image-based gastrointestinal disease screening.
Phase 1 focuses on preprocessing tongue images, segmenting the tongue region, classifying images as `Normal` or `GI Disease`, and presenting the result through a Streamlit interface.

## Installation

Install the project dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Download

Download the dataset by following the instructions from Section 1 of the project prompt.
After downloading, place the original images inside:

- `data/raw/Normal/`
- `data/raw/GI_Disease/`

Keep the raw dataset unchanged and use later pipeline stages to generate the `processed/` and `segmented/` outputs.

## Run Training

```bash
python training/train.py
```

## Run Evaluation

```bash
python evaluation/evaluate.py
```

## Launch the Streamlit UI

```bash
streamlit run app/streamlit_app.py
```
