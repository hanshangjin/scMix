# Modified from https://github.com/QiJiang-QJ/PI-SDE

import pandas as pd
import torch
import pandas as pd
import sklearn.preprocessing, sklearn.decomposition
import seaborn as sns
import umap
import scipy.stats

x = pd.read_csv("/media/udata/time/veres/raw/GSE114412_Stage_5.all.processed_counts.tsv.gz", sep="\t", index_col=0)
metadata = pd.read_csv("/media/udata/time/veres/raw/GSE114412_Stage_5.all.cell_metadata.tsv.gz", sep="\t", index_col=0)

scaler = sklearn.preprocessing.StandardScaler()
pca = sklearn.decomposition.PCA(n_components = 30, random_state=0)
um = umap.UMAP(n_components = 2, metric = 'euclidean', n_neighbors = 30, random_state=42, transform_seed=42)

y = metadata['CellWeek'].values.astype(int)

ix_te = (y == 2) | (y == 4) | (y == 6) | (y == 7) # for random recovery task
ix_tr = ~ix_te

x_tr = x[ix_tr]
x_te = x[ix_te]
y_tr = y[ix_tr]
y_te = y[ix_te]

meta_tr = metadata.loc[ix_tr]
meta_te = metadata.loc[ix_te]

# ===================== add：HVG =====================
n_hvg = 100
gene_vars = x_tr.var(axis=0)
hvg_genes = gene_vars.sort_values(ascending=False).head(n_hvg).index.tolist()
if 'TOP2A' not in hvg_genes and 'TOP2A' in x_tr.columns:
    hvg_genes = ['TOP2A'] + hvg_genes
x_tr = x_tr[hvg_genes]
x_te = x_te[hvg_genes]
# =====================================================================

normalized_x_tr = pd.DataFrame(scaler.fit_transform(x_tr),
                               index = x_tr.index, columns = x_tr.columns)

x1 = normalized_x_tr['TOP2A']
rs = {}
for g in normalized_x_tr.columns:
    x2 = normalized_x_tr[g]
    r, pval = scipy.stats.pearsonr(x1, x2)
    rs[g] = r
rs = pd.Series(rs)
(rs > 0.15).value_counts()

normalized_x_tr_f = normalized_x_tr.loc[:, rs < 0.15]
xp_tr = pca.fit_transform(normalized_x_tr_f)
xu_tr = um.fit_transform(xp_tr)

normalized_x_te = pd.DataFrame(scaler.transform(x_te),
                               index = x_te.index, columns = x_te.columns)
normalized_x_te_f = normalized_x_te.loc[:, rs < 0.15]
xp_te = pca.transform(normalized_x_te_f)
xu_te = um.transform(xp_te)

y = metadata['CellWeek'].values.astype(int)

x_l = [normalized_x_tr_f.loc[y_tr == 0,].values,
       normalized_x_tr_f.loc[y_tr == 1,].values,
       normalized_x_te_f.loc[y_te == 2,].values,
       normalized_x_tr_f.loc[y_tr == 3,].values,
       normalized_x_te_f.loc[y_te == 4,].values,
       normalized_x_tr_f.loc[y_tr == 5,].values,
       normalized_x_te_f.loc[y_te == 6,].values,
       normalized_x_te_f.loc[y_te == 7,].values]

xp_l = [xp_tr[y_tr == 0,],
        xp_tr[y_tr == 1,],
        xp_te[y_te == 2,],
        xp_tr[y_tr == 3,],
        xp_te[y_te == 4,],
        xp_tr[y_tr == 5,],
        xp_te[y_te == 6,],
        xp_te[y_te == 7,]]

xu_l = [xu_tr[y_tr == 0,],
        xu_tr[y_tr == 1,],
        xu_te[y_te == 2,],
        xu_tr[y_tr == 3,],
        xu_te[y_te == 4,],
        xu_tr[y_tr == 5,],
        xu_te[y_te == 6,],
        xu_te[y_te == 7,]]

x_l = [torch.from_numpy(a).float() for a in x_l]
xp_l = [torch.from_numpy(a).float() for a in xp_l]
xu_l = [torch.from_numpy(a).float() for a in xu_l]

celltype_ = [meta_tr['Assigned_cluster'][(y_tr == 0)],
             meta_tr['Assigned_cluster'][(y_tr == 1)],
             meta_te['Assigned_cluster'][(y_te == 2)],
             meta_tr['Assigned_cluster'][(y_tr == 3)],
             meta_te['Assigned_cluster'][(y_te == 4)],
             meta_tr['Assigned_cluster'][(y_tr == 5)],
             meta_te['Assigned_cluster'][(y_te == 6)],
             meta_te['Assigned_cluster'][(y_te == 7)]]

genes = normalized_x_tr_f.columns

unique_cell_types = metadata['Assigned_cluster'].unique()
default_colors = sns.color_palette("tab20", len(unique_cell_types))
color_palette_dict = {cell_type: color for cell_type, color in zip(unique_cell_types, default_colors)}


torch.save({
    'x': x_l,
    'xp': xp_l,
    'xu': xu_l,
    'y': [0,1,2,3,4,5,6,7],
    'celltype': celltype_,
    'genes': genes,
    'Types': unique_cell_types,
    'colors': color_palette_dict
}, 'data/Veres/rr/fate_train.pt')
