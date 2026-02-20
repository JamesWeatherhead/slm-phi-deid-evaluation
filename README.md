# SLM De-identification Evaluation

Systematic evaluation of sub-4B parameter language models for clinical text de-identification.

## Study Design

4 models (Phi-4 Mini, Llama 3.2 3B, Qwen 3 4B, Gemma 3 4B) x 7 prompting strategies x 5 runs evaluated against the [ASQ-PHI benchmark](https://github.com/JamesWeatherhead/asq-phi) using three-tier scoring methodology.

### Models
| Model | Parameters | Company | Ollama Tag |
|-------|-----------|---------|------------|
| Phi-4 Mini Instruct | 3.8B | Microsoft | phi4-mini:latest |
| Llama 3.2 3B Instruct | 3.2B | Meta | llama3.2:3b-instruct-q4_K_M |
| Qwen 3 4B | 4.0B | Alibaba | qwen3:4b |
| Gemma 3 4B | 4.0B | Google | gemma3:4b |

### Prompting Strategies
| ID | Strategy | Description |
|----|----------|-------------|
| P1 | Zero-Shot Minimal | Baseline minimal instruction |
| P2 | Zero-Shot Structured | Detailed PHI definitions with conservative uncertainty handling |
| P2a | Structured + Aggressive | Ablation: aggressive tagging on uncertainty |
| P3 | Few-Shot (3 examples) | Three demonstrations including hard negative |
| P4-prog | Two-Pass (Programmatic) | LLM identifies, code replaces |
| P4-llm | Two-Pass (LLM) | LLM identifies, LLM replaces |
| P5 | Chain-of-Thought | Entity-by-entity reasoning before redaction |

### Evaluation (Three-Tier)
- **Tier 1:** String absence (did PHI disappear?)
- **Tier 2:** Position-aware tag matching (was a redaction tag placed correctly?)
- **Tier 3:** Output fidelity (was non-PHI text preserved without hallucination?)

## Quick Start

### Prerequisites
- Python 3.10+
- Ollama running with models pulled
- ASQ-PHI dataset (see data/README.md)

### Install
```
pip install -r requirements.txt
```

### Run Variance Pilot
```
python -m src.pilot --model phi4-mini --host http://localhost:11434
```

### Run Single Configuration
```
python -m src.runner --model phi4-mini --prompt zero-shot-minimal --run 1 --host http://localhost:11434
```

### Run Full Experiment (GPU Cluster)
```
bash run_full_experiment.sh
```

### Evaluate Results
```
python -m src.evaluator --all --output-dir .
python -m src.analyzer --output-dir .
```

## Repository Structure
```
configs/          Model, prompt, and inference configurations
src/              Evaluation harness source code
data/             Dataset documentation and links
scripts/          Data preparation utilities
raw/              Raw model outputs (generated at runtime)
processed/        Evaluated results (generated at runtime)
analysis/         Cross-model analysis (generated at runtime)
```

## Benchmark

This study uses the [ASQ-PHI](https://github.com/JamesWeatherhead/asq-phi) (Adversarial Synthetic Queries for PHI) benchmark:
- 1,051 clinical search queries (832 with PHI, 219 hard negatives)
- 2,973 PHI elements across 13 HIPAA Safe Harbor categories
- **Repository:** [github.com/JamesWeatherhead/asq-phi](https://github.com/JamesWeatherhead/asq-phi)
- **Dataset:** [Mendeley Data](https://data.mendeley.com/datasets/csz5dzp7nx/1)
- **Publication:** [Weatherhead, J. (2026). ASQ-PHI: Adversarial Synthetic Queries for Protected Health Information. *Data in Brief*.](https://www.sciencedirect.com/science/article/pii/S2352340926001393)

## Related Work

- **Local PHI Scrubber (Phi-3 Mini baseline):** [github.com/JamesWeatherhead/local-phi-scrubber](https://github.com/JamesWeatherhead/local-phi-scrubber)

## Citation

If you use this evaluation harness, please cite the accompanying manuscript:

Weatherhead, J. & McCaffrey, P. (2026). Evaluating Sub-4B Parameter Language Models for Clinical Text De-identification. *Frontiers in Digital Health*. (Manuscript in preparation)

## License

MIT
