# Data

This study uses the ASQ-PHI (Adversarial Synthetic Queries for PHI) benchmark dataset.

- **Repository:** [github.com/JamesWeatherhead/asq-phi](https://github.com/JamesWeatherhead/asq-phi)
- **Dataset (Mendeley Data):** [data.mendeley.com/datasets/csz5dzp7nx/1](https://data.mendeley.com/datasets/csz5dzp7nx/1)
- **Publication:** [Weatherhead, J. (2026). ASQ-PHI: Adversarial Synthetic Queries for Protected Health Information. *Data in Brief*.](https://www.sciencedirect.com/science/article/pii/S2352340926001393)

## Setup

Clone the ASQ-PHI repository:

```bash
git clone https://github.com/JamesWeatherhead/asq-phi.git
cp asq-phi/data/synthetic_clinical_queries.txt data/
```

Or download directly from Mendeley Data:
```bash
# Download from https://data.mendeley.com/datasets/csz5dzp7nx/1
cp ~/Downloads/synthetic_clinical_queries.txt data/
```

Then generate the JSON splits:
```bash
python scripts/split_dataset.py --input data/synthetic_clinical_queries.txt --output data/
```

This produces:
- `all_queries.json` (1,051 queries with metadata)
- `positive_queries.json` (832 queries with PHI)
- `negative_queries.json` (219 hard negatives)

## Dataset Statistics
- Total queries: 1,051
- Queries with PHI: 832
- Hard negatives (zero PHI): 219
- Total PHI elements: 2,973
- PHI categories: 13 of 18 HIPAA Safe Harbor types
