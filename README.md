# EP-567-Project

Codebase for a **hybrid fake news detection** pipeline with two major components:

1. **Adversarial news generation** for robustness testing
2. **Hierarchical detection** using a fast lexical model, a deep semantic model, and an optional LLM-based verification path

This repository is notebook-first. The core implementation lives in:

- `generate_adversarial.ipynb`
- `detection.ipynb`

---

## What this project does

This project studies how well fake-news detectors hold up under **distribution shift** and **adversarial rewriting**.

The workflow is:

- build or download source datasets
- generate adversarial news variants with LLMs
- train a lightweight classifier and a transformer classifier
- route easy examples to the cheap model and hard examples to the stronger model
- optionally escalate ambiguous cases to a **ZoFia-style LLM verification path** with evidence retrieval and multi-agent debate
- evaluate the system on both standard test data and generated attack sets

---

## Repository structure

```text
.
├── README.md
├── detection.ipynb
└── generate_adversarial.ipynb
```

### `generate_adversarial.ipynb`

Builds adversarial evaluation sets from `GonzaloA/fake_news`.

Implemented attack families:

- **CoT**: generate fake articles from real articles while preserving real entities and mimicking newsroom style
- **AdSent**: rewrite fake articles into a more positive / neutral / professional tone
- **SheepDog**: style-transfer articles into publisher-specific voices
  - `CNN`
  - `The New York Times`
  - `National Enquirer`
  - `The Sun`

The notebook uses an OpenAI-compatible client pointed at **NVIDIA NIM** by default.

### `detection.ipynb`

Implements the detection and evaluation pipeline.

Main stages:

- dataset loading and preprocessing
- **TF-IDF + Passive Aggressive Classifier (PAC)** baseline
- **RoBERTa-Large** semantic classifier
- **hierarchical routing**: PAC for high-confidence examples, RoBERTa for low-confidence examples
- adversarial robustness evaluation on generated attack sets
- optional **ZoFia-style** LLM verification with retrieval + debate + judging

---

## Datasets used in the detection notebook

The detection notebook merges three sources:

### 1. `GonzaloA/fake_news`

Used as a binary fake/real dataset.

### 2. `chengxuphd/liar2`

Originally a 6-class LIAR-style dataset. The notebook collapses labels into binary classes:

- `0, 1 -> fake`
- `4, 5 -> real`
- `2, 3 -> dropped` as ambiguous middle classes

### 3. `ArkaMukherjee/Uni-Fakeddit-55k`

Loaded in streaming mode and parsed from records containing:

- `[TEXT]`
- `[OBJECTS]`
- `[LABEL]`

The notebook maps labels as:

- `0 -> fake`
- `1, 2 -> real`

and fuses text and object tokens into a single text field.

### Combined split sizes produced in the notebook run

- **Train:** 80,625
- **Validation:** 15,151
- **Test:** 15,152

---

## Models and methods

## 1. PAC baseline

The lexical baseline uses:

- `TfidfVectorizer`
  - lowercase
  - English stopword removal
  - 1–2 grams
  - `min_df=5`
  - `max_df=0.98`
  - `max_features=200000`
  - `sublinear_tf=True`
- `PassiveAggressiveClassifier`
  - class-balanced weights
  - online-style `partial_fit`
  - batch training over shuffled mini-batches

This stage is designed to be fast and cheap.

## 2. RoBERTa-Large

The semantic model uses:

- `roberta-large`
- `max_length=192`
- stratified downsampling for notebook-friendly runtime
  - train: 32,000
  - validation: 6,000
  - test: 6,000
- training settings in the notebook:
  - 3 epochs
  - learning rate `1e-5`
  - weight decay `0.03`
  - warmup ratio `0.06`
  - batch size `64`
  - eval batch size `512`
  - gradient checkpointing enabled

## 3. Hierarchical routing

The hybrid model computes PAC decision scores and chooses between models based on a validation-selected confidence threshold.

Conceptually:

- if PAC is confident, trust PAC
- otherwise, fall back to RoBERTa

This gives a cost/quality tradeoff while preserving a simple inference path.

## 4. ZoFia-style LLM verification path

For ambiguous samples, the notebook adds an optional LLM verification module using the **DeepSeek API**.

The implementation includes:

- entity extraction
  - DeepSeek-based NER when available
  - regex fallback when not available
- hierarchical salience scoring for candidate entities
- SC-MMR keyword selection
- evidence retrieval from:
  - Wikipedia
  - DuckDuckGo snippets
- collaborative analysis with multiple agents:
  - linguist agent
  - domain-expert agent
  - claim extractor / verifier
- final debate among:
  - proponent
  - skeptic
  - judge

This path is used only for ambiguous cases to control API cost.

---

## Adversarial attack generation

The generation notebook writes three attack sets:

- `cot_generated.csv`
- `adsent_generated.csv`
- `SheepDog_generated.csv`

### CoT attack

Starts from **real** articles and asks the model to produce a believable **fake** article while:

- keeping real people / organizations / locations
- mimicking journalistic voice
- introducing subtle fabrication such as invented statistics or misattributed quotes

### AdSent attack

Starts from **fake** articles and rewrites them to sound more neutral / professional while preserving the same underlying claims.

### SheepDog attack

Uses style transfer while preserving content. In the evaluation notebook, the label logic is:

- style transfer into `CNN` / `NYT` from fake articles -> still **fake**
- style transfer into `National Enquirer` / `The Sun` from real articles -> still **real**

This makes SheepDog the most distribution-shifting attack family in the repository.

---

## Important path mismatch to know before running

The two notebooks do **not** use the same output path by default.

### Generation notebook output

`generate_adversarial.ipynb` writes files to:

```text
data/generated_new/
```

### Detection notebook expectation

`detection.ipynb` reads adversarial CSVs from:

```text
generated/
```

and even includes a setup cell that expects a `generated.zip` archive to be unpacked there.

### Practical fix

Before running the robustness section of `detection.ipynb`, do one of the following:

1. copy the generated CSVs from `data/generated_new/` into `generated/`, or
2. zip them as `generated.zip` and let the notebook unpack them, or
3. edit the path in `detection.ipynb` to point directly to `data/generated_new/`

If you skip this step, the adversarial evaluation cells will fail or silently read the wrong location.

---

## Quick start

## 1. Create an environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

## 2. Install dependencies

There is no pinned `requirements.txt` in the repository, so install from the notebook imports:

```bash
pip install \
  numpy pandas torch datasets scikit-learn transformers joblib \
  matplotlib seaborn tqdm requests openai jupyter
```

Depending on your machine, you may also want a CUDA-enabled PyTorch build.

## 3. Generate adversarial datasets

Open and run:

```text
generate_adversarial.ipynb
```

### API note

The generation notebook currently sets:

```python
NIM_API_KEY = "YOUR_NVIDIA_API_KEY"
```

So you must either:

- edit that value directly in the notebook, or
- refactor it to use `os.getenv(...)`

The default model in the notebook is:

```text
qwen/qwen3.5-397b-a17b
```

served through the NVIDIA NIM OpenAI-compatible endpoint.

## 4. Move generated files into the location expected by detection

For example:

```bash
mkdir -p generated
cp data/generated_new/*.csv generated/
```

## 5. Run detection and evaluation

Open and run:

```text
detection.ipynb
```

For the optional LLM verification path, set:

```bash
export DEEPSEEK_API_KEY="your_key_here"
```

---

## Saved artifacts

The detection notebook writes artifacts to:

```text
artifacts/3_1_detection_phase/
```

including:

- `tfidf_vectorizer.joblib`
- `pac_classifier.joblib`
- `roberta_model/`
- `metrics_comparison.csv`
- `generated_attack_metrics.csv`
- `figures/`

RoBERTa training checkpoints are also written under:

```text
artifacts/roberta_large_detection/
```

---

## Results from the saved notebook run

## Standard evaluation

### PAC

- validation accuracy: **0.8517**
- test accuracy: **0.8545**
- test F1: **0.8448**

### RoBERTa-Large (subset evaluation)

- validation accuracy: **0.9218**
- test accuracy: **0.9197**
- test F1: **0.9159**

### Hybrid PAC -> RoBERTa-Large

- test accuracy: **0.9185**
- test F1: **0.9146**
- route-to-RoBERTa ratio on the subset: **0.6623**

### Hybrid + ZoFia(LLM)

- test accuracy: **0.9183**
- test F1: **0.9145**

Important: the RoBERTa / Hybrid / ZoFia comparisons are on the **downsampled shared subset**, while PAC is also reported once on the **full test set**. Keep that distinction in mind when comparing numbers.

## Adversarial robustness

From the saved run:

- **CoT** remains relatively easy for all models
  - RoBERTa / Hybrid accuracy around **0.9641**
  - Hybrid + ZoFia reaches **0.9833** on the sampled LLM-evaluated run
- **AdSent** is much harder
  - PAC accuracy **0.2600**
  - RoBERTa / Hybrid accuracy **0.1470**
- **SheepDog** is the hardest distribution shift in the notebook
  - PAC accuracy **0.1327**
  - RoBERTa accuracy **0.2603**
  - Hybrid accuracy **0.2479**

This suggests the repository is especially focused on **style-based attack robustness**, not just clean benchmark performance.

---

## Known limitations

- **Notebook-only project**: there is no script entrypoint, package layout, or reproducible pipeline runner.
- **No requirements file**: environment setup must be inferred from imports.
- **Path inconsistency** between generation and detection notebooks.
- **API key handling** in generation notebook is not production-safe because the key is stored as a literal variable placeholder.
- **LLM evaluation is capped** in the notebook:
  - `LLM_MAX_CALLS_TEST = 6`
  - `LLM_MAX_CALLS_PER_GENERATED_SET = 4`
    So the LLM-enhanced results are illustrative rather than exhaustive.
- **External retrieval dependency**: the ZoFia-style path depends on Wikipedia / DuckDuckGo access and may be brittle under network issues or API changes.
- **No license file** is present in the repository.

---

## Suggested next improvements

If you want to evolve this repo beyond the current coursework / research-notebook form, the highest-value changes would be:

1. add `requirements.txt` or `pyproject.toml`
2. move shared logic out of notebooks into `src/`
3. unify paths for generated data
4. replace inline API key placeholders with environment variables
5. add a small inference script for batch prediction
6. persist generated-sample metadata more explicitly so attack provenance is easier to trace
7. separate benchmark evaluation from expensive LLM-only evaluation

---

## In one sentence

This repository is a **fake-news robustness research prototype** that combines adversarial text generation, lexical and transformer-based detection, and an optional evidence-grounded LLM arbitration path in a notebook-centric workflow.
