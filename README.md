# Project Title

![Overview Figure](model.png)

A concise description of your project goes here. One or two sentences are enough to explain what the project is about and what problem it addresses.

---

## Datasets

We use four benchmark single-cell time-series datasets from four different species:

| Dataset            | Species                     | #Cells  | #Time Points | Source |
|---------------------|-------------------------------|---------|-------------|--------|
| Zebrafish Embryo    | *Danio rerio*               | 38,731  | 12          | [Single Cell Portal (SCP162)](https://singlecell.broadinstitute.org/single_cell/study/SCP162/single-cell-reconstruction-of-developmental-trajectories-during-zebrafish-embryogenesis) |
| Drosophila          | *Drosophila melanogaster*   | 27,386  | 11          | [GEO: GSE190147](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE190147) |
| Schiebinger2019     | *Mus musculus*             | 236,285 | 19          | [Broad Institute WOT Tutorial](https://broadinstitute.github.io/wot/tutorial/) |
| Veres               | *Homo sapiens*             | 51,274  | 8           | [GEO: GSE114412](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114412) |

Detailed dataset information and preprocessing steps can be found in the [`data_preprocessing`](./data_preprocessing) folder.

---

## Getting Started

### Requirements

To set up the environment, make sure you have the following:

- Python 3.8 or higher  
- [PyTorch](https://pytorch.org/) (with CUDA support if available)  
- Common scientific computing libraries: `numpy`, `scipy`, `pandas`, `matplotlib`, `scikit-learn`  
- For single-cell data processing: `scanpy`, `anndata`, `umap-learn`

You can install all dependencies using:

```bash
pip install -r requirements.txt
