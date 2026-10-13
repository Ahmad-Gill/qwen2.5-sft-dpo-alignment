# SFT + DPO Alignment of Qwen2.5-0.5B

**Student ID:** 25280065 · **Base model:** `Qwen/Qwen2.5-0.5B` · **Judge:** `google/gemini-3.6-flash` · **Report:** [`report.md`](report.md)

This repository implements the post-training pipeline from the assignment *Instruction Tuning and Direct Preference Optimization on a Small Language Model*:

1. **SFT:** own training loop, 30,000 UltraChat examples
2. **From-scratch DPO on HH-RLHF:** 15,000 pairs, evaluated on the synthetic helpful/harmless set
3. **From-scratch DPO on UltraFeedback:** 15,000 pairs, evaluated on the fixed AlpacaEval subset (300 prompts, seed 42)
4. **TRL `DPOTrainer` + LoRA persona run:** `HuggingFaceTB/SmolLM2-360M-Instruct` on `data/persona_lora/ramsay_style_preferences.json`
5. **Blind pairwise evaluation** with a Kaggle model judge (batches of 5, 5 s sleep, 3 retries, failure log, resume)

No Hugging Face `Trainer`, TRL or prebuilt DPO loop is used for SFT or the two from-scratch DPO runs. TRL and PEFT are used only for the LoRA persona task, as the assignment allows.

---

## Results

| Comparison (blind A/B judge) | Evaluation set | n | Win / Loss / Tie | Win rate (no ties) | Win rate (tie = ½) |
|---|---|---|---|---|---|
| UltraFeedback-DPO vs SFT | AlpacaEval, 300 prompts, seed 42 | 300 | 98 / 141 / 61 | **0.410** | 0.428 |
| HH-RLHF-DPO vs SFT | Synthetic HH safety set | 300 | 119 / 86 / 95 | **0.580** | 0.555 |
| LoRA-DPO vs base SmolLM2 | Persona style set | 12 | 6 / 5 / 1 | 0.545 | 0.542 |

Unknown judgments: 0. Failed batches: 0.

| Training run | Key metric |
|---|---|
| SFT | Validation loss per response token 1.3845 → **1.3512** (best = final, step 937) |
| HH-RLHF DPO | Best validation reward accuracy **0.629** (step 250) |
| UltraFeedback DPO | Best validation reward accuracy **0.742** (step 450) |
| LoRA-DPO | Final eval reward accuracy 1.0, margin 2.306 |

The full analysis is in [`report.md`](report.md): per-category safety results, length effects, examples and limitations.

---

## Repository layout

    qwen2.5-sft-dpo-alignment/
    ├── sft.py                    # SFT: prompt template, batching, response-only log-probs, SFT loss
    ├── dpo.py                    # DPO: batches, policy/reference log-probs, DPO loss
    ├── submission_adapter.py     # 6 grader functions -> thin wrappers around sft.py / dpo.py
    ├── report.md                 # final report (submitted as PDF)
    ├── requirements.txt, pyproject.toml
    ├── data/persona_lora/        # provided persona preference data (220 rows)
    ├── docs/assignment_manual.pdf
    ├── eval_sets/                # fixed evaluation prompts
    │   ├── alpaca_eval_300_seed42.json
    │   ├── hh_rlhf_eval_300_seed42.json
    │   └── persona_style_eval.json
    ├── notebooks/                # one Kaggle notebook per task (run in order)
    ├── results/                  # every config, metric, plot, generation and judgment
    ├── figures/                  # report figures + analysis.json
    ├── scripts/make_report_figures.py
    └── tests/                    # public adapter smoke tests (starter repo)

---

## Implementation notes

| Requirement | Where / how |
|---|---|
| Prompt format (SFT, DPO and evaluation) | `"### Instruction:\n{instruction}\n\n### Response:\n"`, no chat template (`sft.py`) |
| Response-only loss masking | Prompt and padding labels set to `-100` (`sft.build_sft_batch`) |
| EOS handling | EOS appended to each response **before** truncation to `max_length` |
| SFT loss reduction | **Mean over all unmasked response tokens in the batch** (`sft.compute_sft_loss`) |
| Response log-probs | Summed over response tokens only (`sft.sequence_logprobs`) |
| DPO loss | `-log σ(β[(log πθ(y_w) − log πθ(y_l)) − (log π_ref(y_w) − log π_ref(y_l))])`, batch mean (`dpo.dpo_loss`) |
| Frozen reference | Copy of SFT `best`, run under `torch.no_grad()` on `cuda:1`; policy on `cuda:0` |
| Mixed precision | fp32 trainable weights + fp16 autocast + `GradScaler`, gradient clipping 1.0 |
| Checkpoints | `best` + `final` for each run; DPO `best` chosen by validation reward accuracy |

---

## Setup (Kaggle, GPU T4 x2)

Kaggle's preinstalled CUDA PyTorch is kept; PyTorch is not reinstalled:

    python -m pip uninstall -y -q torchao || true
    python -m pip install -q "transformers>=4.43.0,<5" "datasets>=2.18,<5" \
      huggingface_hub accelerate tqdm pyyaml sentencepiece safetensors pytest
    python -m pip install -q -e . --no-deps

Kaggle secrets used by the notebooks: `GITHUB_TOKEN`, `GITHUB_REPO_URL`, `HF_TOKEN`.
Each notebook clones or pulls this repo, writes results under `results/`, and pushes them back. Model checkpoints go to private Hugging Face repos.

Environment used for all runs (`results/task1_setup/setup_info.json`): 2× Tesla T4, torch 2.11.0+cu128, transformers 4.57.6, datasets 4.8.5. The LoRA run used trl 0.25.1 and peft 0.17.1.

---

## Public adapter tests

    python -m pip install -q torch pytest
    PYTHONPATH=. python -m pytest -q tests/test_public_adapter_smoke.py

Result: **7 passed**. The tests run on CPU in seconds, with no model download.

---

## How to reproduce (run notebooks in order)

| Notebook | Task | Main outputs |
|---|---|---|
| `notebooks/25280065_02_baseline.ipynb` | Task 2: fixed AlpacaEval subset + base-model answers | `eval_sets/alpaca_eval_300_seed42.json`, `results/task2_baseline/` |
| `notebooks/25280065_03_sft.ipynb` | Task 3: SFT (30k train / 1k val, 1 epoch) | `sft.py`, `results/task3_sft/` |
| `notebooks/25280065_04_dpo.ipynb` | Task 4: HH-RLHF DPO and UltraFeedback DPO | `dpo.py`, `results/task4_dpo_hh/`, `results/task4_dpo_uf/` |
| `notebooks/25280065_05_lora.ipynb` | Task 5: TRL DPOTrainer + LoRA persona run | `eval_sets/persona_style_eval.json`, `results/task5_lora/` |
| `notebooks/25280065_06_generate.ipynb` | Task 6: SFT/DPO generations + blind pairwise files | `results/task6_generations/` |
| `notebooks/25280065_07_judge.ipynb` | Task 7: LLM-as-judge evaluation | `results/task7_judge/` |

Every notebook is restartable: it skips work whose outputs already exist (for example a pushed `model_location.json` or existing judgments).

Rebuild the report figures from the saved results (no GPU):

    python scripts/make_report_figures.py

---

## Configuration used

| Run | Trainable params | LR / schedule | Batch (device × accum) | Max len (prompt) | Epochs / steps | β |
|---|---|---|---|---|---|---|
| SFT | Full fine-tune | 2e-5, 3% warmup + cosine, WD 0.1 | 2 × 16 = 32 | 512 (≤256) | 1 / 937 | n/a |
| HH-RLHF DPO | Full fine-tune | 1e-6 fixed, AdamW, WD 0 | 1 × 32 = 32 | 512 (256) | 1 / 468 | 0.1 |
| UltraFeedback DPO | Full fine-tune | 1e-6 fixed, AdamW, WD 0 | 1 × 32 = 32 | 512 (256) | 1 / 468 | 0.1 |
| LoRA-DPO | LoRA r 16, α 32, dropout 0.05, all attention + MLP projections | 5e-5, 10% warmup + cosine | 4 × 2 = 8 | 512 (256) | 3 / 75 | 0.1 |

Generation for evaluation: greedy decoding, `max_new_tokens=256` (Qwen) or 200 (SmolLM2), `repetition_penalty=1.1`. The same settings are used for every model within a comparison. Deviations from the guide configuration are explained in §14.3 of the report.

---

## Model checkpoints (private Hugging Face repos)

| Run | Repository | Used for evaluation |
|---|---|---|
| SFT | `AhmaGill/qwen2.5-0.5b-sft` | `best` (step 937) |
| HH-RLHF DPO | `AhmaGill/qwen2.5-0.5b-dpo-hh` | `best` (step 250) |
| UltraFeedback DPO | `AhmaGill/qwen2.5-0.5b-dpo-uf` | `best` (step 450) |
| LoRA persona adapter | `AhmaGill/smollm2-360m-ramsay-lora` | final adapter |

The DPO repos also contain the exact train/validation pairs used (`data/train.json`, `data/val.json`).

---

## Evaluation protocol

- Pairwise files: `results/task6_generations/pairwise_dpo_uf_vs_sft.jsonl`, `results/task6_generations/pairwise_dpo_hh_vs_sft.jsonl`, `results/task7_judge/pairwise_persona_lora_vs_base.jsonl`
- Response order randomised (seed 42); the judge sees only "Response A / Response B". Model identity is stored as hidden metadata and mapped back after judging.
- Judge criteria: helpfulness, correctness, instruction-following, concision, formatting, and absence of hallucinations or unsafe content. For the safety set, the judge also rewards appropriate refusal, privacy protection, calibrated uncertainty and safe redirection.
- Reliability: batches of 5, ≥5 s sleep between attempts, up to 3 retries, unresolved batches logged to a failure JSONL file, resume from existing judgments.
- Outputs: `results/task7_judge/judged_*.jsonl` and `results/task7_judge/summary.json`