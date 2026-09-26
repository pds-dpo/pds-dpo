# Third-party attribution

- `compat/llava/`: LLaVA by Haotian Liu and contributors, Apache-2.0;
  original notices are retained. This inference copy includes the experiment's
  Transformers 4.57 compatibility changes. See `compat/LICENSE.llava` and
  https://github.com/haotian-liu/LLaVA.
- `compat/lmms_llava.py`, and task helpers derived from lmms-eval/MMMU:
  LMMs-Lab lmms-eval revision `cb45ac4d4a667ea5ef89c7a148bff69b3489b981`.
  See `compat/LICENSE.lmms-eval`. MMMU parsing/aggregation follows the upstream
  evaluator contract; `mmmu_full.py` preserves open-answer candidate lists.
  https://github.com/EvolvingLMMs-Lab/lmms-eval
- MMHal-Bench, RLHF-V/Object HalBench and AMBER are **external dependencies**,
  not relicensed here. Their original evaluation prompts, data and scorers
  remain subject to upstream licenses and terms. Pinned sources are in README.
- Model weights, training datasets, benchmark images and COCO annotations are
  not bundled. Their original terms remain applicable independently of the
  repository's software license.
