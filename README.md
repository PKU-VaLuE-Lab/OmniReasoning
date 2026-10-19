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
  <a href="https://qwen.ai/blog?id=qwen3.8-omni-flash">
    <img alt="Qwen3.8-Omni Blog" src="assets/readme/badges/qwen_blog.svg" height="28">
  </a>
  <a href="https://arxiv.org/pdf/2609.39490">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2609.39490-b31b1b.svg?style=for-the-badge&logo=arxiv&logoColor=white">
  </a>
  <a href="#-benchmark">
    <img alt="Benchmark" src="https://img.shields.io/badge/OmniReasoningBench-Coming%20Soon-4c8bf5.svg?style=for-the-badge&logo=databricks&logoColor=white">
  </a>
  <a href="#-code--models">
    <img alt="Code" src="https://img.shields.io/badge/Data-Coming%20Soon-f59e0b.svg?style=for-the-badge&logo=github&logoColor=white">
  </a>

</p>

</div>


<div align="center">


[<img src="assets/readme/teaser.png" alt="Training data statistics — data type distribution, disciplinary coverage, and task category" width="640">](assets/readme/teaser.png)

**OmniQA** constructs audio-visual joint reasoning data. **OmniReasoningBench** evaluates
*reasoning over video* and *reasoning beyond video*.
The released datasets are **OmniReasoning-SFT-112K** (112,463 samples with synthesized thinking)
and **OmniReasoning-RL-19K** (18,991 questions with evidence annotations),and the OmniReasoningBench has **1,150 questions**.


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

<table>
  <thead>
    <tr>
      <th align="center">Model</th>
      <th align="center">Modality</th>
      <th align="center">Reasoning over video<br>Multiple-choice</th>
      <th align="center">Reasoning over video<br>Open-ended</th>
      <th align="center">Reasoning beyond video<br>Multiple-choice</th>
      <th align="center">Reasoning beyond video<br>Open-ended</th>
      <th align="center">Overall</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th colspan="7" align="center"><strong>Proprietary models</strong></th>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Gemini&#8209;3.7&#8209;Flash</td>
      <td align="center">Omni</td>
      <td align="center">57.6</td>
      <td align="center">47.7</td>
      <td align="center">72.0</td>
      <td align="center">58.0</td>
      <td align="center">57.6</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Gemini&#8209;3.8&#8209;Flash</td>
      <td align="center">Omni</td>
      <td align="center">55.2</td>
      <td align="center">49.6</td>
      <td align="center">69.6</td>
      <td align="center">60.0</td>
      <td align="center">57.1</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Gemini&#8209;3.5&#8209;Flash</td>
      <td align="center">Omni</td>
      <td align="center">53.5</td>
      <td align="center">47.2</td>
      <td align="center">66.4</td>
      <td align="center">51.3</td>
      <td align="center">54.0</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Qwen3.8&#8209;Omni&#8209;Flash</td>
      <td align="center">Omni</td>
      <td align="center">50.4</td>
      <td align="center">42.1</td>
      <td align="center">69.2</td>
      <td align="center">57.3</td>
      <td align="center">52.7</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Gemini&#8209;3.1&#8209;Pro</td>
      <td align="center">Omni</td>
      <td align="center">51.9</td>
      <td align="center">47.2</td>
      <td align="center">64.0</td>
      <td align="center">49.3</td>
      <td align="center">52.7</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/doubao.png" alt="doubao" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Doubao&#8209;2.0&#8209;Lite</td>
      <td align="center">Omni</td>
      <td align="center">54.7</td>
      <td align="center">47.2</td>
      <td align="center">57.2</td>
      <td align="center">40.7</td>
      <td align="center">51.0</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Qwen3.5&#8209;Omni&#8209;Plus</td>
      <td align="center">Omni</td>
      <td align="center">42.9</td>
      <td align="center">42.2</td>
      <td align="center">60.4</td>
      <td align="center">40.7</td>
      <td align="center">46.2</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/meta.png" alt="meta" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Muse&#8209;Spark&#8209;1.2</td>
      <td align="center">Omni</td>
      <td align="center">38.1</td>
      <td align="center">36.0</td>
      <td align="center">60.0</td>
      <td align="center">58.0</td>
      <td align="center">44.8</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/meta.png" alt="meta" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Muse&#8209;Spark&#8209;1.1</td>
      <td align="center">Omni</td>
      <td align="center">36.4</td>
      <td align="center">35.9</td>
      <td align="center">56.0</td>
      <td align="center">46.0</td>
      <td align="center">41.8</td>
    </tr>
    <tr>
      <th colspan="7" align="center"><strong>Open-source models</strong></th>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/xiaomi.png" alt="xiaomi" height="18" align="absmiddle">&#8288;&nbsp;&#8288;MiMo&#8209;V2.5</td>
      <td align="center">Omni</td>
      <td align="center">42.7</td>
      <td align="center">40.6</td>
      <td align="center">54.0</td>
      <td align="center">44.0</td>
      <td align="center">44.6</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/kimi.png" alt="kimi" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Kimi&#8209;K3</td>
      <td align="center">Visual</td>
      <td align="center">21.4</td>
      <td align="center">11.7</td>
      <td align="center">50.8</td>
      <td align="center">35.3</td>
      <td align="center">26.5</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Qwen3.8&#8209;Max</td>
      <td align="center">Visual</td>
      <td align="center">19.7</td>
      <td align="center">10.5</td>
      <td align="center">46.8</td>
      <td align="center">34.7</td>
      <td align="center">24.6</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Qwen3.5&#8209;Plus</td>
      <td align="center">Visual</td>
      <td align="center">17.2</td>
      <td align="center">8.7</td>
      <td align="center">46.0</td>
      <td align="center">30.0</td>
      <td align="center">22.5</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/nvidia.png" alt="nvidia" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Nemotron&#8209;3&#8209;Nano&#8209;Omni</td>
      <td align="center">Omni</td>
      <td align="center">25.6</td>
      <td align="center">22.9</td>
      <td align="center">36.0</td>
      <td align="center">13.3</td>
      <td align="center">25.4</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Qwen3&#8209;Omni&#8209;30B&#8209;A3B&#8209;Thinking</td>
      <td align="center">Omni</td>
      <td align="center">31.9</td>
      <td align="center">25.9</td>
      <td align="center">46.4</td>
      <td align="center">32.0</td>
      <td align="center">33.2</td>
    </tr>
    <tr>
      <th colspan="7" align="center"><strong>Ours</strong></th>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18" align="absmiddle">&#8288;&nbsp;&#8288;OmniReasoning&#8209;30B&#8209;A3B</td>
      <td align="center">Omni</td>
      <td align="center">48.0</td>
      <td align="center">30.1</td>
      <td align="center">58.8</td>
      <td align="center">32.7</td>
      <td align="center">42.5</td>
    </tr>
  </tbody>
</table>

### MFSD

<div align="center">

[<img src="./assets/readme/mfsd.png" alt="The MFSD training framework — paper figure" width="640">](./assets/readme/mfsd.png)

</div>

### Results of OmniReasoning-30B-A3B

#### Performance across benchmarks

<table>
  <thead>
    <tr>
      <th align="center">Model</th>
      <th align="center">OmniVideoBench</th>
      <th align="center">LVOmniBench</th>
      <th align="center">Video&#8209;MMMU</th>
      <th align="center">Video&#8209;MME&#8209;v2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th colspan="5" align="center"><strong>Proprietary models</strong></th>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Gemini&#8209;3.5&#8209;Flash</td>
      <td align="center">66.5</td>
      <td align="center">66.9</td>
      <td align="center">86.1</td>
      <td align="center">67.6</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Gemini&#8209;3.1&#8209;Pro</td>
      <td align="center">60.7</td>
      <td align="center">60.9</td>
      <td align="center">84.3</td>
      <td align="center">53.4</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Qwen3.5&#8209;Omni&#8209;Plus</td>
      <td align="center">53.8</td>
      <td align="center">53.2</td>
      <td align="center">84.6</td>
      <td align="center">47.9</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/meta.png" alt="meta" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Muse&#8209;Spark&#8209;1.1</td>
      <td align="center">55.8</td>
      <td align="center">52.3</td>
      <td align="center">85.2</td>
      <td align="center">59.5</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/xiaomi.png" alt="xiaomi" height="18" align="absmiddle">&#8288;&nbsp;&#8288;MiMo&#8209;V2.5</td>
      <td align="center">51.6</td>
      <td align="center">52.8</td>
      <td align="center">79.1</td>
      <td align="center">51.8</td>
    </tr>
    <tr>
      <th colspan="5" align="center"><strong>Open-source models</strong></th>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/nvidia.png" alt="nvidia" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Nemotron&#8209;3&#8209;Nano&#8209;Omni</td>
      <td align="center">42.1</td>
      <td align="center">32.1</td>
      <td align="center">73.1</td>
      <td align="center">38.4</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Qwen2.5&#8209;Omni&#8209;7B</td>
      <td align="center">29.3</td>
      <td align="center">32.0</td>
      <td align="center">53.9</td>
      <td align="center">25.3</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Qwen3&#8209;Omni&#8209;30B&#8209;A3B&#8209;Thinking</td>
      <td align="center">37.2</td>
      <td align="center">37.7</td>
      <td align="center">68.3</td>
      <td align="center">36.5</td>
    </tr>
    <tr>
      <th colspan="5" align="center"><strong>Ours</strong></th>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18" align="absmiddle">&#8288;&nbsp;&#8288;OmniReasoning&#8209;30B&#8209;A3B</td>
      <td align="center">50.0</td>
      <td align="center">44.7</td>
      <td align="center">74.4</td>
      <td align="center">44.0</td>
    </tr>
  </tbody>
</table>

#### Performance by audio type and video duration

Music, Sound, and Speech are audio types; duration intervals are in minutes.

<table>
  <thead>
    <tr>
      <th align="center">Model</th>
      <th align="center">Music</th>
      <th align="center">Sound</th>
      <th align="center">Speech</th>
      <th align="center">(0,1] minutes</th>
      <th align="center">(1,5] minutes</th>
      <th align="center">(5,10] minutes</th>
      <th align="center">(10,30] minutes</th>
      <th align="center">Average</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th colspan="9" align="center"><strong>Proprietary models</strong></th>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Gemini&#8209;3.5&#8209;Flash</td>
      <td align="center">56.9</td>
      <td align="center">69.3</td>
      <td align="center">70.1</td>
      <td align="center">69.7</td>
      <td align="center">68.0</td>
      <td align="center">63.9</td>
      <td align="center">64.8</td>
      <td align="center">66.5</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Gemini&#8209;3.1&#8209;Pro</td>
      <td align="center">51.1</td>
      <td align="center">65.8</td>
      <td align="center">61.9</td>
      <td align="center">67.3</td>
      <td align="center">64.2</td>
      <td align="center">59.5</td>
      <td align="center">53.0</td>
      <td align="center">60.7</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Gemini&#8209;3&#8209;Pro</td>
      <td align="center">56.2</td>
      <td align="center">54.1</td>
      <td align="center">55.7</td>
      <td align="center">61.0</td>
      <td align="center">56.4</td>
      <td align="center">52.9</td>
      <td align="center">52.5</td>
      <td align="center">55.5</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Qwen3.5&#8209;Omni&#8209;Plus</td>
      <td align="center">42.3</td>
      <td align="center">64.9</td>
      <td align="center">53.3</td>
      <td align="center">55.8</td>
      <td align="center">55.2</td>
      <td align="center">50.7</td>
      <td align="center">53.4</td>
      <td align="center">53.8</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/xiaomi.png" alt="xiaomi" height="18" align="absmiddle">&#8288;&nbsp;&#8288;MiMo&#8209;V2.5</td>
      <td align="center">46.2</td>
      <td align="center">47.6</td>
      <td align="center">53.0</td>
      <td align="center">56.6</td>
      <td align="center">57.4</td>
      <td align="center">47.6</td>
      <td align="center">44.3</td>
      <td align="center">51.6</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/gemini.png" alt="gemini" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Gemini&#8209;2.0&#8209;Flash</td>
      <td align="center">29.7</td>
      <td align="center">40.3</td>
      <td align="center">43.2</td>
      <td align="center">49.4</td>
      <td align="center">43.2</td>
      <td align="center">41.1</td>
      <td align="center">34.9</td>
      <td align="center">41.5</td>
    </tr>
    <tr>
      <th colspan="9" align="center"><strong>Open-source models</strong></th>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/nvidia.png" alt="nvidia" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Nemotron&#8209;3&#8209;Nano&#8209;Omni</td>
      <td align="center">33.0</td>
      <td align="center">42.9</td>
      <td align="center">43.0</td>
      <td align="center">48.8</td>
      <td align="center">41.7</td>
      <td align="center">41.5</td>
      <td align="center">38.9</td>
      <td align="center">42.1</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Qwen2.5&#8209;Omni&#8209;7B</td>
      <td align="center">23.1</td>
      <td align="center">25.3</td>
      <td align="center">30.7</td>
      <td align="center">41.6</td>
      <td align="center">27.4</td>
      <td align="center">25.3</td>
      <td align="center">26.7</td>
      <td align="center">29.3</td>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18" align="absmiddle">&#8288;&nbsp;&#8288;Qwen3&#8209;Omni&#8209;30B&#8209;A3B&#8209;Thinking</td>
      <td align="center">26.4</td>
      <td align="center">37.2</td>
      <td align="center">38.5</td>
      <td align="center">46.8</td>
      <td align="center">35.6</td>
      <td align="center">35.5</td>
      <td align="center">35.2</td>
      <td align="center">37.2</td>
    </tr>
    <tr>
      <th colspan="9" align="center"><strong>Ours</strong></th>
    </tr>
    <tr>
      <td align="center" nowrap="nowrap"><img src="assets/readme/model_logos/qwen.png" alt="qwen" height="18" align="absmiddle">&#8288;&nbsp;&#8288;OmniReasoning&#8209;30B&#8209;A3B</td>
      <td align="center">39.6</td>
      <td align="center">46.3</td>
      <td align="center">52.0</td>
      <td align="center">52.4</td>
      <td align="center">49.9</td>
      <td align="center">49.8</td>
      <td align="center">48.9</td>
      <td align="center">50.0</td>
    </tr>
  </tbody>
</table>

#### Training method ablation

| Method | OmniReasoningBench | OmniVideoBench | LVOmniBench | Video&#8209;MME&#8209;v2 | Video&#8209;MMMU |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen3&#8209;Omni&#8209;30B&#8209;A3B&#8209;Thinking | 33.2 | 37.2 | 37.7 | 36.5 | 68.3 |
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
