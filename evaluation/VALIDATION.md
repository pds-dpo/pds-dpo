# Release validation — 26 September 2026

This is a portable code release of the production evaluation helpers, not a
new benchmark result release. Configuration, launch paths and artifact guards
were adapted for public use. Running jobs and their frozen source files were
not modified by this publication.

## Checks completed

- All Python sources parsed successfully.
- **20 offline unit tests passed** locally and in the workstation's Python 3.11
  scoring environment: MMHal rating parsing (9), sample-ID/output integrity
  (5), and statistical calculations/directionality (6).
- `worker.py --help`, `prepare.py --help` and `score.py --help` checked with
  the example configuration. No datasets or weights were loaded by these CLI
  checks.
- Inference runtime check passed: Transformers 4.57.6, datasets 4.4.2,
  PEFT 0.18.1, Accelerate 1.12.0, PyTorch 2.6.0+cu124. The bundled LLaVA
  loader imported successfully; CUDA was available. No inference was run.
- Scoring runtime check passed: NumPy 1.26.4, spaCy 3.7.2, NLTK 3.8.1,
  OpenAI SDK 3.1.0, and `en_core_web_lg` available. This environment is intended
  for CPU scoring; its separate Torch installation emitted a CUDA-driver
  compatibility warning. The dedicated inference environment passed its check.
- Release files scanned for known credential prefixes, private workstation
  paths and addresses. No credentials, user-specific configuration, model
  weights, benchmark images, runtime logs or generated outputs are included.

## Validation boundaries

The newly portable runner has **not** undergone a fresh end-to-end GPU run,
paid judge run or clean-machine dependency installation. Passing CPU tests is
not a claim of complete replication. Install into isolated environments, run
`check_runtime.py`, prepare/seal the configuration and perform a small smoke
check before scheduling the complete benchmarks. Do not reuse production
completion markers with a newly configured run directory.

The MMHal regression tests preserve explicit-final-rating handling and reject
ambiguous/conflicting outputs; they do not validate the judge's correctness.
Bootstrap intervals quantify evaluation-image uncertainty, not robustness to
training seeds or repeated API judging.
