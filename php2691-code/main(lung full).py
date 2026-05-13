import os
import scanpy as sc
from matplotlib import pyplot as plt

#file_path = os.path.expanduser("~/Downloads/human_pancreas_norm_complexBatch (1).h5ad")
#adata = sc.read_h5ad(file_path)
#print(adata)

file_path1 = os.path.expanduser("~/Downloads/Lung_atlas_public.h5ad")
adata_lung = sc.read_h5ad(file_path1)
adata_lung

#######

sc.pl.violin(
    adata_lung,
    keys=["nUMI", "nGene", "percent.mito"],
    jitter=0.4,
    multi_panel=True
)

#######

adata_lung.obs["nGene"].hist(bins=50)
plt.title("nGene")
plt.show()

adata_lung.obs["nUMI"].hist(bins=50)
plt.title("nUMI")
plt.show()

adata_lung.obs["percent.mito"].hist(bins=50)
plt.title("percent.mito")
plt.show()

#######

sc.pl.scatter(adata_lung, x="nUMI", y="percent.mito")
sc.pl.scatter(adata_lung, x="nUMI", y="nGene")

#### QC for lung
adata_lung_qc = adata_lung[
    (adata_lung.obs["percent.mito"] < 0.08) &
    (adata_lung.obs["nGene"] >= 500) &
    (adata_lung.obs["nGene"] < 6000) &
    (adata_lung.obs["nUMI"] >= 1000) &
    (adata_lung.obs["nUMI"] < 25000),
    :
].copy()
sc.pp.filter_genes(adata_lung_qc, min_cells=3)

#desktop_path = os.path.expanduser("~/Desktop/adata_lung_qc.h5ad")
#adata_lung_qc.write(desktop_path)


adata_lung_scvi = adata_lung_qc.copy()
adata_lung_norm = adata_lung_qc.copy()
adata_lung_norm.X = adata_lung_norm.layers["counts"].copy()


# total count normalization
sc.pp.normalize_total(adata_lung_norm, target_sum=1e4)

# log-normalization
sc.pp.log1p(adata_lung_norm)

# highly variable genes
sc.pp.highly_variable_genes(
    adata_lung_norm,
    flavor="seurat",
    batch_key="batch",
    n_top_genes=2000
)


# Keep only HVGs
adata_lung_hvg = adata_lung_norm[:, adata_lung_norm.var["highly_variable"]].copy()

# PCA
sc.tl.pca(adata_lung_hvg)


################# Harmony

import harmonypy as hm

ho = hm.run_harmony(
    adata_lung_hvg.obsm["X_pca"],
    adata_lung_hvg.obs,
    vars_use=["batch"]
)


adata_lung_hvg.obsm["X_pca_harmony"] = ho.Z_corr

sc.pp.neighbors(adata_lung_hvg, use_rep="X_pca_harmony")
sc.tl.umap(adata_lung_hvg)
sc.tl.leiden(adata_lung_hvg, resolution=0.5)


# Plots
adata_lung_hvg.obs["batch"] = adata_lung_hvg.obs["batch"].astype("category")
adata_lung_hvg.obs["cell_type"] = adata_lung_hvg.obs["cell_type"].astype("category")

# Fix colors
batch_color_map = {
    "1":  "#1f77b4",
    "2":  "#ff7f0e",
    "3":  "#2ca02c",
    "4":  "#d62728",
    "5":  "#9467bd",
    "6":  "#8c564b",
    "A1": "#e377c2",
    "A2": "#bcbd22",
    "A3": "#17becf",
    "A4": "#aec7e8",
    "A5": "#ffbb78",
    "A6": "#98df8a",
    "B1": "#ff9896",
    "B2": "#c5b0d5",
    "B3": "#c49c94",
    "B4": "#f7b6d2"
}

celltype_color_map = {
    "Macrophage": "#1f77b4",
    "Type 2": "#ff7f0e",
    "Basal 2": "#2ca02c",
    "Ciliated": "#d62728",
    "Basal 1": "#9467bd",
    "Secretory": "#8c564b",
    "Neutrophil_CD14_high": "#e377c2",
    "Dendritic cell": "#7f7f7f",
    "T/NK cell": "#bcbd22",
    "Endothelium": "#17becf",
    "Fibroblast": "#aec7e8",
    "B cell": "#ffbb78",
    "Neutrophils_IL1R2": "#98df8a",
    "Mast cell": "#ff9896",
    "Lymphatic": "#c5b0d5",
    "Type 1": "#c49c94",
    "Ionocytes": "#f7b6d2"
}

adata_lung_hvg.uns["batch_colors"] = [
    batch_color_map[x] for x in adata_lung_hvg.obs["batch"].cat.categories
]
adata_lung_hvg.uns["cell_type_colors"] = [
    celltype_color_map[x] for x in adata_lung_hvg.obs["cell_type"].cat.categories
]


# batch UMAP

fig, ax = plt.subplots(figsize=(9, 7))

sc.pl.umap(
    adata_lung_hvg,
    color="batch",
    title="Harmony batch UMAP",
    ax=ax,
    show=False,
    frameon=True
)

legend = ax.get_legend()
handles = legend.legend_handles
labels = [t.get_text() for t in legend.get_texts()]

legend.remove()
ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=4,
    frameon=False
)

plt.tight_layout()
plt.show()



# cell umap

fig, ax = plt.subplots(figsize=(9, 7))

# Let scanpy create the legend first
sc.pl.umap(
    adata_lung_hvg,
    color="cell_type",
    title="Harmony cell type UMAP",
    ax=ax,
    show=False,
    frameon=True
)

legend = ax.get_legend()
handles = legend.legend_handles
labels = [t.get_text() for t in legend.get_texts()]

legend.remove()

ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.10),
    ncol=3,
    frameon=False
)

plt.tight_layout()
plt.show()

# Lung Harmony ARI & NMI

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ari_harmony = adjusted_rand_score(
    adata_lung_hvg.obs["cell_type"],
    adata_lung_hvg.obs["leiden"]
)

nmi_harmony = normalized_mutual_info_score(
    adata_lung_hvg.obs["cell_type"],
    adata_lung_hvg.obs["leiden"]
)

print("Harmony ARI:", ari_harmony)
print("Harmony NMI:", nmi_harmony)

# Lung Harmony iLISI

import scib_metrics as sm
from scib_metrics.nearest_neighbors import pynndescent

X_harmony = adata_lung_hvg.obsm["X_pca_harmony"]
batch_labels = adata_lung_hvg.obs["batch"].to_numpy()

nn_harmony = pynndescent(X_harmony, n_neighbors=30, random_state=0)

ilisi_harmony = sm.ilisi_knn(
    nn_harmony,
    batch_labels,
    scale=True
)

print("Harmony scaled iLISI:", ilisi_harmony)

# Lung Harmony scaled cLISI
cell_labels_harmony = adata_lung_hvg.obs["cell_type"].to_numpy()

clisi_harmony = sm.clisi_knn(
    nn_harmony,
    cell_labels_harmony,
    scale=True
)

print("Harmony scaled cLISI:", clisi_harmony)


###########scvi

import scvi

scvi.settings.seed = 0

scvi.model.SCVI.setup_anndata(
    adata_lung_scvi,
    layer="counts",
    batch_key="batch"
)

model = scvi.model.SCVI(
    adata_lung_scvi,
    n_layers=2,
    n_latent=30,
    gene_likelihood="nb"
)

model.train()

SCVI_LATENT_KEY = "X_scVI"
adata_lung_scvi.obsm[SCVI_LATENT_KEY] = model.get_latent_representation()

sc.pp.neighbors(adata_lung_scvi, use_rep=SCVI_LATENT_KEY)
sc.tl.leiden(adata_lung_scvi, resolution=0.5)
sc.tl.umap(adata_lung_scvi, min_dist=0.3)

sc.pl.umap(adata_lung_scvi, color="batch")
sc.pl.umap(adata_lung_scvi, color="cell_type")


# scvi batch umap

adata_lung_scvi.obs["batch"] = adata_lung_scvi.obs["batch"].astype("category")

adata_lung_scvi.uns["batch_colors"] = [
    batch_color_map[x] for x in adata_lung_scvi.obs["batch"].cat.categories
]

fig, ax = plt.subplots(figsize=(9, 7))

sc.pl.umap(
    adata_lung_scvi,
    color="batch",
    title="scVI batch UMAP",
    ax=ax,
    show=False,
    frameon=True
)

legend = ax.get_legend()
handles = legend.legend_handles
labels = [t.get_text() for t in legend.get_texts()]

legend.remove()

ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=4,
    frameon=False
)

plt.tight_layout()
plt.show()


# scvi cell type umap

adata_lung_scvi.obs["cell_type"] = adata_lung_scvi.obs["cell_type"].astype("category")

adata_lung_scvi.uns["cell_type_colors"] = [
    celltype_color_map[x] for x in adata_lung_scvi.obs["cell_type"].cat.categories
]

fig, ax = plt.subplots(figsize=(9, 7))

sc.pl.umap(
    adata_lung_scvi,
    color="cell_type",
    title="scVI cell type UMAP",
    ax=ax,
    show=False,
    frameon=True
)

legend = ax.get_legend()
handles = legend.legend_handles
labels = [t.get_text() for t in legend.get_texts()]

legend.remove()

ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=4,
    frameon=False
)

plt.tight_layout()
plt.show()


# Lung scvi ARI & NMI

ari_scvi = adjusted_rand_score(
    adata_lung_scvi.obs["cell_type"],
    adata_lung_scvi.obs["leiden"]
)

nmi_scvi = normalized_mutual_info_score(
    adata_lung_scvi.obs["cell_type"],
    adata_lung_scvi.obs["leiden"]
)

print("scVI ARI:", ari_scvi)
print("scVI NMI:", nmi_scvi)

# Lung scvi iLISI

X_scvi = adata_lung_scvi.obsm["X_scVI"]
batch_labels_scvi = adata_lung_scvi.obs["batch"].to_numpy()

nn_scvi = pynndescent(X_scvi, n_neighbors=30, random_state=0)

ilisi_scvi = sm.ilisi_knn(
    nn_scvi,
    batch_labels_scvi,
    scale=True
)

print("scVI scaled iLISI:", ilisi_scvi)

# Lung scvi cLISI

cell_labels_scvi = adata_lung_scvi.obs["cell_type"].to_numpy()

clisi_scvi = sm.clisi_knn(
    nn_scvi,
    cell_labels_scvi,
    scale=True
)

print("scVI scaled cLISI:", clisi_scvi)




# Plots for Seraut

file_path2 = os.path.expanduser("~/Desktop/lung_integrated.h5ad")
adata_lung_seurat = sc.read_h5ad(file_path2)
adata_lung_seurat


adata_lung_seurat.obs["batch"] = adata_lung_seurat.obs["batch"].astype("category")


adata_lung_seurat.uns["batch_colors"] = [
    batch_color_map[x] for x in adata_lung_seurat.obs["batch"].cat.categories
]


fig, ax = plt.subplots(figsize=(9, 7))

sc.pl.umap(
    adata_lung_seurat,
    color="batch",
    title="Seurat RPCA batch UMAP",
    ax=ax,
    show=False,
    frameon=True
)

legend = ax.get_legend()
handles = legend.legend_handles
labels = [t.get_text() for t in legend.get_texts()]
legend.remove()

ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=4,
    frameon=False
)

plt.tight_layout()
plt.show()


# cell type umap

adata_lung_seurat.obs["cell_type"] = adata_lung_seurat.obs["cell_type"].astype("category")

adata_lung_seurat.uns["cell_type_colors"] = [
    celltype_color_map[x] for x in adata_lung_seurat.obs["cell_type"].cat.categories
]

fig, ax = plt.subplots(figsize=(9, 7))

sc.pl.umap(
    adata_lung_seurat,
    color="cell_type",
    title="Seurat RPCA cell type UMAP",
    ax=ax,
    show=False,
    frameon=True
)

legend = ax.get_legend()
handles = legend.legend_handles
labels = [t.get_text() for t in legend.get_texts()]
legend.remove()

ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=3,
    frameon=False
)

plt.tight_layout()
plt.show()

# Seraut ARI/NMI
ari_rpca = adjusted_rand_score(
    adata_lung_seurat.obs["cell_type"],
    adata_lung_seurat.obs["seurat_clusters"]
)

nmi_rpca = normalized_mutual_info_score(
    adata_lung_seurat.obs["cell_type"],
    adata_lung_seurat.obs["seurat_clusters"]
)

print("RPCA ARI:", ari_rpca)
print("RPCA NMI:", nmi_rpca)

# serautRPCA iLISI

X_seurat = adata_lung_seurat.obsm["X_pca"]
batch_labels_seurat = adata_lung_seurat.obs["batch"].to_numpy()

nn_seurat = pynndescent(
    X_seurat,
    n_neighbors=30,
    random_state=0
)

ilisi_seurat = sm.ilisi_knn(
    nn_seurat,
    batch_labels_seurat,
    scale=True
)

print("Seurat RPCA scaled iLISI:", ilisi_seurat)

# Lung Seurat RPCA cLISI

cell_labels_seurat = adata_lung_seurat.obs["cell_type"].to_numpy()

clisi_seurat = sm.clisi_knn(
    nn_seurat,
    cell_labels_seurat,
    scale=True
)

print("Seurat RPCA scaled cLISI:", clisi_seurat)


###### Lung for fastMNN

file_path3 = os.path.expanduser("~/Desktop/lung_fastmnn.h5ad")

adata_lung_fastmnn = sc.read_h5ad(file_path3)
adata_lung_fastmnn


# batch UMAP

adata_lung_fastmnn.obs["batch"] = adata_lung_fastmnn.obs["batch"].astype("category")

adata_lung_fastmnn.uns["batch_colors"] = [
    batch_color_map[x] for x in adata_lung_fastmnn.obs["batch"].cat.categories
]

sc.pp.neighbors(adata_lung_fastmnn, use_rep="corrected")
sc.tl.umap(adata_lung_fastmnn)

fig, ax = plt.subplots(figsize=(9, 7))

sc.pl.umap(
    adata_lung_fastmnn,
    color="batch",
    title="fastMNN batch UMAP",
    ax=ax,
    show=False,
    frameon=True
)

legend = ax.get_legend()
handles = legend.legend_handles
labels = [t.get_text() for t in legend.get_texts()]
legend.remove()

ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=4,
    frameon=False
)

plt.tight_layout()
plt.show()

# fastMNN cell type UMAP

adata_lung_fastmnn.obs["cell_type"] = adata_lung_fastmnn.obs["cell_type"].astype("category")


adata_lung_fastmnn.uns["cell_type_colors"] = [
    celltype_color_map[x] for x in adata_lung_fastmnn.obs["cell_type"].cat.categories
]


fig, ax = plt.subplots(figsize=(9, 7))

sc.pl.umap(
    adata_lung_fastmnn,
    color="cell_type",
    title="fastMNN cell type UMAP",
    ax=ax,
    show=False,
    frameon=True
)

legend = ax.get_legend()
handles = legend.legend_handles
labels = [t.get_text() for t in legend.get_texts()]
legend.remove()

ax.legend(
    handles,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.08),
    ncol=3,
    frameon=False
)

plt.tight_layout()
plt.show()


# fastMNN ARI/NMI

sc.tl.leiden(adata_lung_fastmnn, resolution=0.5)

ari_fastmnn = adjusted_rand_score(
    adata_lung_fastmnn.obs["cell_type"],
    adata_lung_fastmnn.obs["leiden"]
)

nmi_fastmnn = normalized_mutual_info_score(
    adata_lung_fastmnn.obs["cell_type"],
    adata_lung_fastmnn.obs["leiden"]
)

print("fastMNN ARI:", ari_fastmnn)
print("fastMNN NMI:", nmi_fastmnn)


# fastMNN iLISI

X_fastmnn = adata_lung_fastmnn.obsm["corrected"]
batch_labels_fastmnn = adata_lung_fastmnn.obs["batch"].to_numpy()

nn_fastmnn = pynndescent(
    X_fastmnn,
    n_neighbors=30,
    random_state=0
)

ilisi_fastmnn = sm.ilisi_knn(
    nn_fastmnn,
    batch_labels_fastmnn,
    scale=True
)

print("fastMNN scaled iLISI:", ilisi_fastmnn)

# lung fastMNN scaled cLISI
cell_labels_fastmnn = adata_lung_fastmnn.obs["cell_type"].to_numpy()

clisi_fastmnn = sm.clisi_knn(
    nn_fastmnn,
    cell_labels_fastmnn,
    scale=True
)

print("fastMNN scaled cLISI:", clisi_fastmnn)













