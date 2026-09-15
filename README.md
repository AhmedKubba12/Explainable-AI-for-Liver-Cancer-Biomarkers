# Explainable AI for Liver Cancer Biomarker Discovery

This repository holds the code and data for the paper *Potential of Artificial Intelligence Algorithms for Identification of Relevant Diagnostic and Prognostic Biomarkers of Early-Stage Liver Cancer*. A preprint is available on arXiv at [arXiv:2609.15638](https://arxiv.org/abs/2609.15638). The paper is currently under peer review. If you use this code or data, please cite the paper (see [Citation](#citation) below).

This project uses machine learning and explainable AI (XAI) to find gene expression biomarkers that separate the stages of liver disease progression toward hepatocellular carcinoma. The pipeline combines gradient boosted and tree based classifiers with SHAP for model interpretation, and it uses four metaheuristic search methods to reduce a large gene panel down to a small, informative subset.

## Background

The dataset covers five stages along the path of liver cancer development:

| Label  | Stage                         |
|--------|-------------------------------|
| `sl`   | Steatosis / normal liver      |
| `lgdn` | Low grade dysplastic nodule   |
| `hgdn` | High grade dysplastic nodule  |
| `ehcc` | Early hepatocellular carcinoma|
| `phcc` | Progressed hepatocellular carcinoma |

Each row is a sample and each column is the expression value of one gene. The starting panel holds 500 genes that were pre selected with an XGBoost ranking step. The goal is to shrink that panel to a handful of genes that still let a classifier tell the five stages apart, and then use SHAP to explain which genes drive each prediction.

## Approach

1. Train a base classifier (XGBoost or Random Forest) on the 500 gene panel.
2. Run SHAP to rank gene contributions per class.
3. Apply metaheuristic feature selection to search for a compact gene subset that keeps accuracy high.
4. Compare the gene subsets returned by each method and look at the overlap.
5. Retrain and save a final model on the top genes.

Four feature selection methods are included so their results can be compared:

- Ant Colony Optimization (ACO)
- Genetic Algorithm (GA)
- Particle Swarm Optimization (PSO)
- Grey Wolf Optimizer (GWO)

The `common_genes.py` script takes the gene lists produced by the different runs and reports the genes they agree on.

## Repository layout

```
xai-liver-cancer-biomarkers/
├── data/
│   ├── selected_genes_500_xg.csv      # 500 gene expression panel with stage labels
│   └── shap_values/                   # exported SHAP values, one file per class
├── feature_selection/
│   ├── ant_colony.py                  # ACO feature selection (XGBoost fitness)
│   ├── genetic_algorithm.py           # GA feature selection
│   ├── pso.py                         # PSO feature selection (Random Forest fitness)
│   ├── grey_wolf.py                   # GWO feature selection (SVM fitness)
│   └── common_genes.py               # intersection of gene lists across runs
├── notebooks/
│   ├── shap_baseline.ipynb            # first SHAP pass on the base model
│   ├── shap_results.ipynb             # SHAP value analysis and plots
│   ├── xai_with_shap.ipynb            # full SHAP workflow
│   ├── xai_with_shap_v0.2_ant_colony.ipynb
│   ├── xai_with_shap_v0.2_genetic.ipynb
│   └── xai_with_shap_v0.2_pso.ipynb   # SHAP applied to each selection method
├── models/
│   └── best_random_forest_model.pkl   # trained Random Forest on the top gene subset
├── results/
│   └── figures/                       # convergence curves and SHAP summary plots
├── docs/
│   └── references/                    # background papers
├── requirements.txt
└── README.md
```
selected_genes_500_xg.csv contains private data obtained from the University of Lübeck, please contact the authors for this file.

## Getting started

Clone the repository and install the dependencies. A virtual environment is recommended.

```bash
git clone https://github.com/<your-username>/xai-liver-cancer-biomarkers.git
cd xai-liver-cancer-biomarkers

python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

The feature selection scripts read `data/selected_genes_500_xg.csv` relative to the repository root, so run them from that root:

```bash
python feature_selection/ant_colony.py
python feature_selection/genetic_algorithm.py
python feature_selection/pso.py
python feature_selection/grey_wolf.py
```

Each script prints the gene subset it found along with the cross validation and test accuracy. The PSO script also saves the retrained model to `best_random_forest_model.pkl`.

To explore the SHAP analysis and the plots, open the notebooks:

```bash
jupyter notebook notebooks/
```

Note that the scripts expect the data file at `data/selected_genes_500_xg.csv`. Some of the original scripts point at the file by name only, so either run from the `data` folder or update the path near the top of each script to match your setup.

## Data

`selected_genes_500_xg.csv` holds the expression values for 500 genes across the samples, with the disease stage stored in the `Symbol` column. The `shap_values/` folder holds the per class SHAP values exported from the notebooks, one CSV per stage, which the analysis notebooks use to rank and plot gene importance.

## Results

The `results/figures/` folder holds the convergence curves for each search method and the SHAP summary plots for the XGBoost and Random Forest models. The written reports and slides in `docs/` walk through the findings in more detail.

## References

The `docs/references/` folder holds background papers on explainable AI for liver disease biomarkers. They are included for convenience and remain the property of their original authors and publishers.

## Citation

If you use this work, please cite the paper:

> Ali Bou Nassif, Darko Castven, Manar Abu Talib, Jibran Sualeh Muhammad, Ahmed Ammar Kubba, Jens Marquardt, and Abdalla Sayed Ali, "Potential of Artificial Intelligence Algorithms for Identification of Relevant Diagnostic and Prognostic Biomarkers of Early-Stage Liver Cancer," arXiv preprint arXiv:2609.15638, 2026.

BibTeX:

```bibtex
@article{liver_cancer_xai_2026,
  title   = {Potential of Artificial Intelligence Algorithms for Identification of Relevant Diagnostic and Prognostic Biomarkers of Early-Stage Liver Cancer},
  author  = {Nassif, Ali Bou and Castven, Darko and Abu Talib, Manar and Muhammad, Jibran Sualeh and Kubba, Ahmed Ammar and Marquardt, Jens and Ali, Abdalla Sayed},
  journal = {arXiv preprint arXiv:2609.15638},
  year    = {2026},
  url     = {https://arxiv.org/abs/2609.15638}
}
```
