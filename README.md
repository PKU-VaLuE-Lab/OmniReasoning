<p align="center">
  <img src="omnireasoning.png" alt="OmniReasoning mascot" width="108">
</p>

<h1 align="center">OmniReasoning</h1>

<p align="center"><strong>Pushing the Limits of Audio-Visual Joint Reasoning</strong></p>

<div align="center">

<p>
  <a href="https://pku-value-lab.github.io/OmniReasoning-Homepage/">
    <img alt="Project Page" src="https://img.shields.io/badge/Project%20Page-Online-8a2be2.svg?style=for-the-badge&logo=googlechrome&logoColor=white">
  </a>
  <a href="https://arxiv.org/pdf/2609.39490">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2609.39490-b31b1b.svg?style=for-the-badge&logo=arxiv&logoColor=white">
  </a>
  <a href="#-benchmark">
    <img alt="Benchmark" src="https://img.shields.io/badge/OmniReasoningBench-Coming%20Soon-4c8bf5.svg?style=for-the-badge&logo=databricks&logoColor=white">
  </a>
  <a href="#-code--models">
    <img alt="Code" src="https://img.shields.io/badge/Code%20%26%20Models-Coming%20Soon-f59e0b.svg?style=for-the-badge&logo=github&logoColor=white">
  </a>
  <a href="#-license">
    <img alt="License" src="https://img.shields.io/badge/License-CC%20BY--NC%204.0-2ea44f.svg?style=for-the-badge">
  </a>
</p>

</div>


<div align="center">

**OmniQA** constructs audio-visual joint reasoning data. **OmniReasoningBench** evaluates
*reasoning over video* and *reasoning beyond video*.
The released datasets are **OmniReasoning-SFT-112K** (112,463 samples with synthesized thinking)
and **OmniReasoning-RL-19K** (18,991 questions with evidence annotations), retained from the
original 116,217 SFT and 19,849 RL samples; the benchmark has **1,150 questions**.

[<img src="assets/readme/123.png" alt="Training data statistics — data type distribution, disciplinary coverage, and task category" width="640">](assets/readme/123.png)

</div>

### OmniReasoningBench

<div align="center">

OmniReasoningBench contains 750 *reasoning over video* questions (375 multiple-choice, 375
open-ended) and 400 *reasoning beyond video* questions (250 multiple-choice, 150 open-ended;
288 video + figure and 112 video + text inputs). Each has a reference answer and an annotated
evidence chain.

[<img src="./assets/readme/benchmark.png" alt="Task taxonomy and examples from OmniReasoningBench" width="680">](./assets/readme/benchmark.png)

</div>

### Results on OmniReasoningBench

**OmniReasoningBench accuracy (%).** Our model is initialized by Qwen3-Omni-30B-A3B-Thinking.

| Model | Modality | Reasoning&nbsp;over&nbsp;video&nbsp;Multiple&#8209;choice | Reasoning&nbsp;over&nbsp;video&nbsp;Open&#8209;ended | Reasoning&nbsp;beyond&nbsp;video&nbsp;Multiple&#8209;choice | Reasoning&nbsp;beyond&nbsp;video&nbsp;Open&#8209;ended | Overall |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proprietary models** |  |  |  |  |  |  |
| <img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18"> Gemini-3.7-Flash | Omni | 57.6 | 47.7 | 72.0 | 58.0 | 57.6 |
| <img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18"> Gemini-3.8-Flash | Omni | 55.2 | 49.6 | 69.6 | 60.0 | 57.1 |
| <img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18"> Gemini-3.5-Flash | Omni | 53.5 | 47.2 | 66.4 | 51.3 | 54.0 |
| <img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18"> Gemini-3.1-Pro | Omni | 51.9 | 47.2 | 64.0 | 49.3 | 52.7 |
| <img src="assets/readme/model_logos/doubao.png" alt="doubao" height="18"> Doubao-2.0-Lite | Omni | 54.7 | 47.2 | 57.2 | 40.7 | 51.0 |
| <img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18"> Qwen3.5-Omni-Plus | Omni | 42.9 | 42.2 | 60.4 | 40.7 | 46.2 |
| <img src="assets/readme/model_logos/meta.png" alt="meta" height="18"> Muse-Spark-1.2 | Omni | 38.1 | 36.0 | 60.0 | 58.0 | 44.8 |
| <img src="assets/readme/model_logos/meta.png" alt="meta" height="18"> Muse-Spark-1.1 | Omni | 36.4 | 35.9 | 56.0 | 46.0 | 41.8 |
| **Open-source models** |  |  |  |  |  |  |
| <img src="assets/readme/model_logos/xiaomi.png" alt="xiaomi" height="18"> MiMo-V2.5 | Omni | 42.7 | 40.6 | 54.0 | 44.0 | 44.6 |
| <img src="assets/readme/model_logos/kimi.png" alt="kimi" height="18"> Kimi-K3 | Visual | 21.4 | 11.7 | 50.8 | 35.3 | 26.5 |
| <img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18"> Qwen3.8-Max | Visual | 19.7 | 10.5 | 46.8 | 34.7 | 24.6 |
| <img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18"> Qwen3.5-Plus | Visual | 17.2 | 8.7 | 46.0 | 30.0 | 22.5 |
| <img src="assets/readme/model_logos/nvidia.png" alt="nvidia" height="18"> Nemotron-3-Nano-Omni | Omni | 25.6 | 22.9 | 36.0 | 13.3 | 25.4 |
| <img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18"> Qwen3-Omni-30B-A3B-Thinking | Omni | 31.9 | 25.9 | 46.4 | 32.0 | 33.2 |
| **Ours** |  |  |  |  |  |  |
| <img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18"> OmniReasoning-30B-A3B | Omni | 48.0 | 30.1 | 58.8 | 32.7 | 42.5 |

### MFSD

<div align="center">

[<img src="./assets/readme/mfsd.png" alt="The MFSD training framework — paper figure" width="640">](./assets/readme/mfsd.png)

</div>

### Results of OmniReasoning-30B-A3B

#### Performance across benchmarks

| Model | OmniVideoBench | LVOmniBench | Video-MMMU | Video-MME-v2 |
| :---: | :---: | :---: | :---: | :---: |
| **Proprietary models** |  |  |  |  |
| <img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18"> Gemini-3.5-Flash | 66.5 | 66.9 | 86.1 | 67.6 |
| <img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18"> Gemini-3.1-Pro | 60.7 | 60.9 | 84.3 | 53.4 |
| <img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18"> Qwen3.5-Omni-Plus | 53.8 | 53.2 | 84.6 | 47.9 |
| <img src="assets/readme/model_logos/meta.png" alt="meta" height="18"> Muse-Spark-1.1 | 55.8 | 52.3 | 85.2 | 59.5 |
| <img src="assets/readme/model_logos/xiaomi.png" alt="xiaomi" height="18"> MiMo-V2.5 | 51.6 | 52.8 | 79.1 | 51.8 |
| **Open-source models** |  |  |  |  |
| <img src="assets/readme/model_logos/nvidia.png" alt="nvidia" height="18"> Nemotron-3-Nano-Omni | 42.1 | 32.1 | 73.1 | 38.4 |
| <img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18"> Qwen2.5-Omni-7B | 29.3 | 32.0 | 53.9 | 25.3 |
| <img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18"> Qwen3-Omni-30B-A3B-Thinking | 37.2 | 37.7 | 68.3 | 36.5 |
| **Ours** |  |  |  |  |
| <img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18"> OmniReasoning-30B-A3B | 50.0 | 44.7 | 74.4 | 44.0 |

#### Performance by audio type and video duration

Music, Sound, and Speech are audio types; duration intervals are in minutes.

| Model | Music | Sound | Speech | (0,1] minutes | (1,5] minutes | (5,10] minutes | (10,30] minutes | Average |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Proprietary models** |  |  |  |  |  |  |  |  |
| <img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18"> Gemini-3.5-Flash | 56.9 | 69.3 | 70.1 | 69.7 | 68.0 | 63.9 | 64.8 | 66.5 |
| <img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18"> Gemini-3.1-Pro | 51.1 | 65.8 | 61.9 | 67.3 | 64.2 | 59.5 | 53.0 | 60.7 |
| <img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18"> Gemini-3-Pro | 56.2 | 54.1 | 55.7 | 61.0 | 56.4 | 52.9 | 52.5 | 55.5 |
| <img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18"> Qwen3.5-Omni-Plus | 42.3 | 64.9 | 53.3 | 55.8 | 55.2 | 50.7 | 53.4 | 53.8 |
| <img src="assets/readme/model_logos/xiaomi.png" alt="xiaomi" height="18"> MiMo-V2.5 | 46.2 | 47.6 | 53.0 | 56.6 | 57.4 | 47.6 | 44.3 | 51.6 |
| <img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18"> Gemini-2.0-Flash | 29.7 | 40.3 | 43.2 | 49.4 | 43.2 | 41.1 | 34.9 | 41.5 |
| **Open-source models** |  |  |  |  |  |  |  |  |
| <img src="assets/readme/model_logos/nvidia.png" alt="nvidia" height="18"> Nemotron-3-Nano-Omni | 33.0 | 42.9 | 43.0 | 48.8 | 41.7 | 41.5 | 38.9 | 42.1 |
| <img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18"> Qwen2.5-Omni-7B | 23.1 | 25.3 | 30.7 | 41.6 | 27.4 | 25.3 | 26.7 | 29.3 |
| <img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18"> Qwen3-Omni-30B-A3B-Thinking | 26.4 | 37.2 | 38.5 | 46.8 | 35.6 | 35.5 | 35.2 | 37.2 |
| **Ours** |  |  |  |  |  |  |  |  |
| <img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18"> OmniReasoning-30B-A3B | 39.6 | 46.3 | 52.0 | 52.4 | 49.9 | 49.8 | 48.9 | 50.0 |

#### Training method ablation

| Method | OmniReasoningBench | OmniVideoBench | LVOmniBench | Video-MME-v2 | Video-MMMU |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3-Omni-30B-A3B-Thinking | 33.2 | 37.2 | 37.7 | 36.5 | 68.3 |
| + SFT | 36.1 | 47.0 | 40.2 | 40.3 | 73.7 |
| + GRPO | 38.3 | 44.8 | 39.8 | 39.1 | 72.9 |
| + RLSD | 39.2 | 45.4 | 40.1 | 39.8 | 72.0 |
| + MFSD | 42.5 | 50.0 | 44.7 | 44.0 | 74.4 |

## Installation

Run all commands from the repository root. Requires **Python 3.10+** and **ffmpeg / ffprobe**.

```bash
python -m pip install '.[engine]'
```

| Stage               | Service                                                      | Configuration                                                |
| :---: | :---: | :---: |
| 1 caption           | Gemini `generateContent`                                     | `GEMINI_API_KEY`, `GEMINI_API_BASE`, `GEMINI_AUTH_PREFIX`; `--gemini-model` |
| 2 generate / adapt  | OpenAI-compatible chat (+ `/images/generations` for RBY figures) | `OPENAI_API_KEY`, `OPENAI_API_BASE`; `--openai-model`, `GEN_MODEL`, `IMAGE_MODEL` |
| 3 verify / refine   | OpenAI-compatible chat                           | `DASHSCOPE_API_KEY`, optional `DASHSCOPE_BASE_URL`           |
| 3 gate              | text model + audiovisual model                               | `GATE_TEXT_*`, `GATE_AV_*`, `OPENAI_API_KEY`                 |
| 4 think             | OpenAI-compatible chat                                       | `OPENAI_API_KEY`, `OPENAI_API_BASE`, `SYNTH_MODEL`           |
| Benchmark inference | SGLang / OpenAI-compatible audiovisual server                | `SGLANG_BASE_URL`, `SGLANG_API_KEY`, `SGLANG_MODEL`          |
| Benchmark judge     | OpenAI-compatible chat                                       | `JUDGE_API_BASE`, `JUDGE_API_KEY`, `JUDGE_MODEL`             |

Place source videos in `data/videos/`.

## OmniQA Data Engine

<div align="center">

[<img src="./assets/readme/data_engine.png" alt="The OmniQA data engine — paper figure" width="560">](./assets/readme/data_engine.png)

</div>

### Stage 1: Caption Generation

```bash
omniqa caption --data-path data/videos --output-dir data/caption/videos \
  --gemini-model gemini-3.1-pro-preview \
  --skip-upload --max-inline-video-mb 20 --max-events 64
```



### Stage 2: QA Generation

Each route writes `outputs/run/generation/accepted.jsonl` with question, options, answer and a
`reasoning_steps` dependency chain.

**Reasoning over video (ROV):** task selection, question/answer/chain generation, structural and
grounding checks, a caption-conditioned solver, and a question-only (priors) blind gate.

```bash
omniqa generate --caption-raw-dir data/caption/videos/raw \
  --output-dir outputs/run/generation \
  --openai-api-base "$OPENAI_API_BASE" --openai-model deepseek-v4-flash \
  --questions-per-video 1
```

**Reasoning beyond video (RBY),** including generated question figures:

```bash
omniqa adapt --caption-dir data/caption/videos/raw --outdir outputs/run/generation --n 0
omniqa bridge-adaptation --outdir outputs/run/generation \
  --output outputs/run/generation/accepted.jsonl
```

**Direct video authoring (DYNAMIC):** the author watches the video; subtype weights must first be
mined from benchmark result files with `omniqa mine-queries` (see `--help`).

```bash
omniqa dynamic --caption-raw-dir data/caption/videos/raw --output-dir outputs/run/generation
```

### Stage 3: Verification

```bash
# 1. Verify timestamped clues against the actual clips (DashScope SDK).
omniqa verify --input outputs/run/generation/accepted.jsonl \
  --output-dir outputs/run/verification
# 2. Repair or reject questions against the video (DashScope SDK).
omniqa refine --input outputs/run/verification/accepted_verified.jsonl \
  --output-dir outputs/run/refinement
# 3. Modality shortcut screening.
omniqa gate --input outputs/run/refinement/refined_accepted.jsonl \
  --caption-raw-dir data/caption/videos/raw --video-root data/videos \
  --output-dir outputs/run/gate
```



### Stage 4: Thinking Generation

```bash
omniqa think --input outputs/run/refinement/refined_accepted.jsonl \
  --caption-root data/caption --output-dir outputs/run/thinking
```



## Evaluation

### 1. Prepare the benchmark and model service

Place the two benchmark files at `data/benchmark/over_video.jsonl` (750) and
`data/benchmark/beyond_video.jsonl` (400), with videos and figures under `data/media/`.
Start an audiovisual SGLang / OpenAI-compatible server and export `SGLANG_*`.

### 2. Run inference

```bash
for s in over_video beyond_video; do
  omnireasoning infer --benchmark $s \
    --data_json_file data/benchmark/$s.jsonl --video_dir data/media \
    --output_file outputs/$s.jsonl --model "$SGLANG_MODEL"
done
```

Defaults: 2 fps, at most 512 frames, audio kept, `--max_tokens 32768 --temperature 0.6 --top_p 0.95 --top_k 20`.

### 3. Score

Multiple-choice answers are scored by rule-based option-letter matching (no LLM fallback; an
output without a parseable letter is wrong). Open-ended answers are judged by deepseek-v4-flash.
Every benchmark question stays in the denominator: missing outputs, service errors and judge
failures count as wrong.

```bash
for s in over_video beyond_video; do
  omnireasoning score-judge --predictions outputs/$s.jsonl \
    --references data/benchmark/$s.jsonl --out_dir outputs/${s}_judge
done

# Table row: over-video MCQ / OE, beyond-video MCQ / OE, Overall (pooled counts).
omnireasoning report-table \
  --over_video_pred outputs/over_video_judge/predictions.jsonl --over_video_ref data/benchmark/over_video.jsonl \
  --beyond_video_pred outputs/beyond_video_judge/predictions.jsonl --beyond_video_ref data/benchmark/beyond_video.jsonl
```


## 📚 Citation

If you find **OmniReasoning** useful, please cite:

```bibtex
@misc{lin2026omnireasoningpushinglimitsaudiovisual,
      title={OmniReasoning: Pushing the Limits of Audio-Visual Joint Reasoning}, 
      author={Junming Lin and Yuxuan Wang and Zhenxin Lei and Yuxin Liu and Ruixun Liu and Yinsong Yan and Ling Wang and Minghao Han and Yunfei Chu and Shun Lei and Xueyao Zhang and Qize Yang and Jin Xu and Yiwu Zhong},
      year={2026},
      eprint={2609.39490},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2609.39490}, 
}
```

---



## 📄 License

The **OmniReasoning** data is released for **non-commercial research use** under **CC BY-NC 4.0**. Code and model licenses will accompany their respective releases.
