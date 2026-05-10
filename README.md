<div align="center">

# ViMU: Benchmarking Video Metaphorical Understanding

[![Project Page](https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=googlechrome&logoColor=white)](https://liqiiiii.github.io/Video-Metaphorical-Understanding/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/xxxx.xxxxx)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Dataset-yellow?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/LIQIIIII/ViMU)


[Qi Li](https://liqiiiii.github.io/), [Xinchao Wang](https://sites.google.com/site/sitexinchaowang/)

[xML Lab](https://sites.google.com/view/xml-nus), National University of Singapore

</div>

This repository contains the evaluation scripts for ViMU, a benchmark for video metaphorical understanding. The code evaluates multimodal models on four tasks:

1. Open-ended interpretation (OE)
2. Evidence grounding (EG)
3. Rhetoric mechanism identification (RM)
4. Social value signal identification (SV)

## Directory Structure

Expected project structure:

```text
ViMU/
├── videos/
│   ├── vimu_000001.mp4
│   └── ...
├── metadata/
│   ├── vimu_oe.jsonl
│   ├── vimu_eg.jsonl
│   ├── vimu_ss.jsonl
│   ├── video_evidence.jsonl
│   └── cache/
├── scripts/
│   ├── 00-vimu_oe.py
│   ├── 01-vimu_oe_judge.py
│   ├── 02-vimu_oe_score.py
│   ├── 10-vimu_eg.py
│   ├── 11-vimu_eg_score.py
│   ├── 20-vimu_ss.py
│   ├── 21-vimu_ss_score.py
│   └── utils.py
└── output/
````

## Setup

Install dependencies:

```bash
pip install openai requests numpy pandas tqdm
```

Depending on the models used, additional API keys may be required.

Set API keys:

```bash
export OPENAI_API_KEY="your_openai_key"
export OPENROUTER_API_KEY="your_openrouter_key"
export GOOGLE_API_KEY="your_google_key"
```

Not all keys are required if you only run a subset of models.

## Path Configuration

Before running, edit each script and set:

```python
PROJECT_ROOT = "/Your/Path/To/ViMU"
```

For example:

```python
PROJECT_ROOT = "/Users/anonymous/ViMU"
```

## Recommended Running Order

For a full evaluation, run:

```bash
# Open-ended interpretation
python scripts/00-vimu_oe.py
python scripts/01-vimu_oe_judge.py
python scripts/02-vimu_oe_score.py

# Evidence grounding
python scripts/10-vimu_eg.py
python scripts/11-vimu_eg_score.py

# Structured subtext tasks without guidance
python scripts/20-vimu_ss.py --prompt_mode without_guidance
python scripts/21-vimu_ss_score.py --prompt_mode without_guidance

# Structured subtext tasks with guidance
python scripts/20-vimu_ss.py --prompt_mode with_guidance
python scripts/21-vimu_ss_score.py --prompt_mode with_guidance
```

## Model Configuration

Models are configured in the `MODEL_SPECS` list inside the inference scripts.

To enable or disable a model, edit:

```python
"enabled": True
```

or

```python
"enabled": False
```

For OpenRouter models, make sure the model ID and API key are valid.

## Output Files

The main output files are:

```text
output/vimu_oe_summary.json
output/vimu_eg_summary.json
output/vimu_ss_without_guidance_summary.json
output/vimu_ss_with_guidance_summary.json
```

These files contain aggregated evaluation results.

## Scoring Rules

### Open-ended Interpretation

Open-ended answers are evaluated using an LLM-as-a-judge protocol. The judge scores semantic understanding based on:

```text
core intent
implicit signal
target or social meaning
hallucination penalty
literal-only penalty
```

Evidence grounding is scored as a multi-label prediction problem. If the prediction contains any incorrect option, the score is 0. Otherwise, if the prediction is a subset of the gold answer, the score is: `score = number of correctly selected options / number of gold options`. Rhetoric and social value tasks use the same multi-label scoring rule. If no incorrect option is selected; otherwise: `score = 0`.

## Notes

The dataset contains socially sensitive video memes. The benchmark is intended for research use only.

