# SynthAlign evaluation and statistical analysis

Portable release of the evaluation/scoring helpers used in the September 2026
experiments. The default configuration is **vanilla LLaVA-1.5-7B plus the six
full-9K A–F checkpoints**. It does not refer to the earlier 2K or 12K models.
This folder contains code, not benchmark images, model weights, credentials,
private workstation configuration, or completed benchmark results.

## Protocol

| Task key | Benchmark/split | Samples per model | Maximum new tokens |
|---|---|---:|---:|
| `mmhal` | MMHal-Bench | 96 | 1,024 |
| `object_halbench` | Object HalBench **fixed 300-image protocol** | 300 | 1,024 |
| `amber` | AMBER generative | 1,004 | 2,048 |
| `amber_disc` | AMBER discriminative | 14,216 | 16 |
| `pope` | POPE random/popular/adversarial, 3,000 each | 9,000 | 128 |
| `scienceqa` | ScienceQA-IMG test | 2,017 | 16 |
| `mme` | MME perception and cognition | 2,374 questions | 16 |
| `mmmu` | MMMU validation | 900 | 128 |
| `seed_img` | SEED-Bench image-only test | 14,233 | 16 |

The first five tasks are enabled by default: 35 inference jobs and 172,312
predictions for seven models. The final four are optional full-split general
VQA tasks from the preceding evaluation pipeline; adding them is a separate
run, not a claim that the full-9K checkpoints have already completed them.
VizWiz and MM-Vet are not included in this release's reproduction runner.

All models use the same task-specific prompts, greedy decoding, one beam,
batch size one, and inference seed zero. LLaVA uses `vicuna_v1` and padded
images. POPE uses the frozen strict yes/no parser; AMBER discriminative accepts
a leading yes/no token and leaves other outputs invalid. MME reports both
perception and cognition; the analysis primary score sums both.

MMHal and Object HalBench use the fixed `gpt-4.1-mini-2025-04-14` judge snapshot
with the original upstream prompts, temperature zero, no model fallback, and
a shared conservative $10 budget ledger. This is **not** directly comparable
to historical results judged with other models. Pricing assumptions are $0.40/M
input and $1.60/M output tokens, checked 26 September 2026 against the
[official model page](https://developers.openai.com/api/docs/models/gpt-4.1-mini).
Recheck prices before a future run; set provider-side spending limits too.

The MMHal parser includes `explicit-final-rating-v1`: a unique explicit final
rating overrides a tentative rating; otherwise conflicting ratings fail.
Every returned API attempt is saved before parsing. Missing/failed samples are
not silently dropped. API keys are read only from `OPENAI_API_KEY` and must
never be placed in this repository.

## Installation (Linux/CUDA)

Use separate inference and scoring environments, **not the training environment**.
The workstation inference runtime used Python 3.11, PyTorch 2.6.0/CUDA 12.4,
Transformers 4.57.6 and lmms-eval 0.7.2 at the commit below. Scoring used the
pinned packages in `requirements-scoring.txt` and `en_core_web_lg` 3.7.1.

```bash
# In a fresh inference environment; install a matching torch/torchvision build.
pip install -r evaluation/requirements-inference.txt
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git /path/to/lmms-eval
git -C /path/to/lmms-eval checkout cb45ac4d4a667ea5ef89c7a148bff69b3489b981
pip install -e /path/to/lmms-eval
pip install -r evaluation/requirements-inference.txt
python evaluation/install_lmms_compat.py --lmms-root /path/to/lmms-eval
python evaluation/check_runtime.py --profile inference

# In a separate scoring environment:
pip install -r evaluation/requirements-scoring.txt
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl
python -m nltk.downloader -d /path/to/hallucination-suite/cache/nltk_data punkt wordnet averaged_perceptron_tagger
python evaluation/check_runtime.py --profile scoring
```

`compat/llava/` is the inference-only Transformers-4.57-compatible LLaVA copy
used by these runs. It is isolated by `PYTHONPATH`; the root training package
is not replaced. `install_lmms_compat.py` installs the matching lmms LLaVA
adapter into the explicitly supplied checkout, with a backup and a guard
against overwriting unrelated edits. See `THIRD_PARTY.md`.

## Benchmark assets

Acquire each benchmark from its upstream provider under its terms. Dataset
images and COCO annotations are **not redistributed here**. Clone the
following sources under your suite's `vendor/` and check out these revisions:

| Directory | Source | Revision |
|---|---|---|
| `mmhal-bench` | https://huggingface.co/datasets/Shengcao1006/MMHal-Bench | `f5f49a938f45ed99e235b8519ba28f76832a2add` |
| `rlhf-v` | https://github.com/RLHF-V/RLHF-V | `3863e39d5f541db7e3725acd8131ab99221455dc` |
| `amber` | https://github.com/junyangwang0410/AMBER | `534babf6bbfcce2e735c26289dedfb21cef3c939` |

Required layout (Git LFS files must contain real data, not pointer text):

```text
hallucination-suite/
  vendor/mmhal-bench/eval_gpt4.py
  vendor/rlhf-v/eval/eval_gpt_obj_halbench.py
  vendor/rlhf-v/eval/data/obj_halbench_300_with_image.jsonl
  vendor/amber/inference.py
  vendor/amber/data/                 # annotations, queries, safe words, relations
  assets/mmhal-bench/response_template.json
  assets/mmhal-bench/images/         # filenames from response_template image_src
  assets/amber/image/                # official 1,004 AMBER images
  assets/coco2014/annotations/       # captions+instances, train2014+val2014
  cache/nltk_data/
```

Preparation extracts the Object HalBench embedded images automatically.
ScienceQA, MME, MMMU, SEED and POPE are loaded from their revision-pinned
`lmms-lab` datasets. `prepare.py` records full-split IDs and metadata hashes.
It never samples an easier subset or uses previous model predictions.

## Configure and prepare

```bash
cp evaluation/config.example.json evaluation/config.json
# Edit config.json: absolute Python, suite, lmms and merged-checkpoint paths.
export SYNTHALIGN_EVAL_CONFIG=/absolute/path/to/repository/evaluation/config.json
python evaluation/prepare.py --seal
python evaluation/audit_images.py
```

Model paths must be **complete/merged checkpoints**, including the trained
multimodal projector, not bare LoRA adapters. For the original full-9K run:
A/B/C/E/F used 9,035 unique pairs, 9,040 exposures and 1,130 steps; D-small used
9,000 pairs and 1,125 steps. All were trained for one epoch with seed 42.

Set `training_images_json` to a JSON list of training-image paths if you want
the exact-pixel train/evaluation overlap check. Without it, image clustering
still runs, but training overlap is explicitly **not checked**. This audit is
not a semantic or near-duplicate contamination guarantee.

Preparation hashes all checkpoint shards and freezes config, task sources,
manifests and evaluator files. Changed inputs require a **new run directory**.
Use ordinary Python, never `python -O`: assertions are part of validation.

## Inference and scoring

```bash
# Each invocation evaluates one model sequentially; one worker per GPU.
python evaluation/worker.py --model A --gpu 0
python evaluation/worker.py --model B --gpu 1
# Repeat for vanilla, C, D_small, E and F. These two commands can be run
# concurrently in separate terminals. --tasks pope evaluates only that task.

# Use the scoring environment after inference has completed:
python evaluation/score.py --model A --task amber
# Set OPENAI_API_KEY privately in your environment before paid stages.
python evaluation/score.py --model A --task mmhal --allow-paid
python evaluation/score.py --model A --task object_halbench --allow-paid
```

Repeat scoring for each model. No paid API calls are made by preparation,
inference, tests or analysis. The high-level scorer uses an exclusive run lock
and verifies coverage before saving a completion marker. Re-running completed
work validates and reuses it. Use `score.py`, not the low-level judge modules,
to retain the shared budget guard. A failed/uncertain request retains its
conservative budget reservation; the cap is not automatically increased.

Generation is resumable by sample ID. An incomplete lmms task is archived and
rerun rather than silently counted as complete. Exact API responses can vary
even with a fixed snapshot and temperature zero.

## Analysis

```bash
# Latest full-9K hallucination analysis; requires all seven model labels.
python evaluation/analyze.py
# If you enabled the additional full general-VQA tasks before preparing:
python evaluation/analyze_full.py
```

`analyze.py` writes `report.json` and `report.md` under `run_root`. It includes
every condition, POPE subset metrics, AMBER generative/discriminative results,
MMHal question types and Object HalBench metrics. Primary outcomes are POPE
macro F1, AMBER Hal, MMHal mean score and Object response hallucination.
The optional `analyze_full.py` writes separate `report_full.json` and
`report_full.md` files, so it does not overwrite the hallucination report.

Uncertainty uses 10,000 **paired image-cluster bootstrap/permutation**
replicates, 95% intervals and prespecified Holm families: A versus B/C/D-small
(12 tests), A versus E/F (8), and variants versus vanilla (24). General-VQA
`analyze_full.py` retains the earlier eight-benchmark A–B/C/D and A–E/F families
(24 and 16 tests); do not merge these into the newer hallucination families.
Intervals measure evaluation-sample uncertainty, **not training-seed or
repeated-judge uncertainty**. Null results and regressions must be reported.
D-small also differs in question coverage, candidate-response count and reward
precision; it is not a pure image-filter-only causal control.

## Tests and release status

```bash
pip install -r evaluation/requirements-analysis.txt
PYTHONPATH=evaluation python -m unittest discover -s evaluation/tests -v
```

The portable packaging changes configuration/launch plumbing and adds input
guards; the scoring and metric implementations are derived from the production
scripts. CPU tests and import/CLI checks are reported in `VALIDATION.md`.
Publication does not itself rerun the full GPU benchmarks or certify every
clean-machine/CUDA combination. Existing workstation jobs are unaffected.
