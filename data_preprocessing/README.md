## Datasets and Preprocessing

The datasets used in this project are summarized below.

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Species</th>
      <th># cells</th>
      <th># TPs</th>
      <th>Source</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Zebrafish Embryo</td>
      <td><i>Danio rerio</i></td>
      <td>38 731</td>
      <td>12</td>
      <td>
        Single Cell Portal 
        (<a href="https://singlecell.broadinstitute.org/single_cell/study/SCP162/single-cell-reconstruction-of-developmental-trajectories-during-zebrafish-embryogenesis">SCP162</a>)
      </td>
    </tr>
    <tr>
      <td>Drosophila</td>
      <td><i>Drosophila melanogaster</i></td>
      <td>27 386</td>
      <td>11</td>
      <td>
        DEAP website 
        (<a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE190147">GSE190147</a>)
      </td>
    </tr>
    <tr>
      <td>Schiebinger2019</td>
      <td><i>Mus musculus</i></td>
      <td>236 285</td>
      <td>19</td>
      <td>
        Broad Institute 
        (<a href="https://broadinstitute.github.io/wot/tutorial/">Wot</a>)
      </td>
    </tr>
    <tr>
      <td>Veres</td>
      <td><i>Homo sapiens</i></td>
      <td>51 274</td>
      <td>8</td>
      <td>
        GEO 
        (<a href="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE114412">GSE114412</a>)
      </td>
    </tr>
  </tbody>
</table>

For **Zebrafish Embryo**, **Drosophila**, and **Schiebinger2019**, we followed the preprocessing pipeline provided in [scNODE](https://github.com/rsinghlab/scNODE).

For **Veres**, the preprocessing procedure was based on [PI-SDE](https://github.com/QiJiang-QJ/PI-SDE), with several adjustments to the original procedure.

## Requirements

### R
- R >= 4.0
- Packages:
  - Seurat
  - URD
  - stringr

### Python
- Python >= 3.7
- Packages:
  - scanpy
  - numpy
  - pandas
  - scikit-learn
  - umap-learn
  - scipy
  - torch

---

## How to Run Preprocessing

### Zebrafish Embryo / Drosophila datasets

#### Run the preprocessing script:

```bash
Rscript ZebrafishData_Processing.R
Rscript DrosophilaData_Processing.R
```

### Preprocessing overview
1. Split cells into training and testing sets based on time points.
2. Identify highly variable genes (HVGs) using training data.
3. Export HVG-filtered expression matrices and metadata for downstream analysis.

#### Customizing Train/Test Time Points

To use custom time-point splits, directly modify the following variables in the script, for example:

```R
train_tps <- c(1, 2, 4, 6, 8, 10)
test_tps <- c(3, 5, 7, 9, 11, 12)
```

#### Output
Running the preprocessing script generates the following files:

1. `-count_data-hvg.csv`  
   A gene expression matrix restricted to highly variable genes (HVGs),
   stored in cell × gene format.  
   This file is used as the direct input to the model training code.

2. `-var_genes_list.csv`  
   A list of selected highly variable genes (HVGs) identified from the
   training time points.

3. `meta_data.csv`  
   Cell-level metadata containing time-point annotations and other
   experimental information, shared across all split settings.

### Schiebinger2019 dataset

#### Run the preprocessing script:

```bash
python WOTData_Processing_Reduced.py
```

#### Preprocessing overview
1. Split time points into training and testing sets.
2. Select HVGs using training cells only.
3. Reduce the dataset to the selected genes for downstream modeling.

#### Customizing Train/Test Time Points

To use custom time-point splits, directly modify the following variables in the script, for example:

```python
train_tps = [0, 1, 2, 3, 4, 5]
test_tps  = [6, 7, 8, 9]
```

#### Output
The preprocessing script outputs:
- `-count_data-hvg.csv`:  
  HVG-filtered expression matrix (cell × gene).
- `-var_genes_list.csv`:  
  List of selected HVGs.
- `-meta_data.csv`:  
  Cell-level metadata with time-point annotations.


### Veres dataset

#### Run the preprocessing script:

```bash
python veres_processing.py
```

#### Preprocessing overview
1. Split cells into training and testing sets based on time points.
2. Select HVGs using training data.
3. Reduce data dimensionality for downstream analysis.

#### Customizing Train/Test Time Points

To use custom time-point splits, directly define the test time points via `ix_te` in the script, with all remaining cells used for training, for example:

```python
ix_te = (y == 2) | (y == 4) | (y == 6) | (y == 7)
ix_tr = ~ix_te
```

#### Output
The preprocessing script outputs:
- `fate_train.pt`:  
  A PyTorch data file containing HVG-filtered expression data, PCA/UMAP embeddings,
  time-point labels, cell-type annotations, and gene information for downstream
  training and visualization.
