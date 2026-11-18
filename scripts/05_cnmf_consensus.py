#!/usr/bin/env python3
"""Step 5 - cNMF consensus at the chosen K and projection of every cell onto it.

Input   results/cnmf_v3/ (from step 4), data/scvi_input_v3.h5ad, data/scvi_embedding_v3.parquet
Output  results/cnmf_v3_usage_k{K}_allcells.parquet   usage per cell, rows sum to 1
        results/cnmf_v3_gene_scores_k{K}.tsv          gene x programme scores
        results/cnmf_v3_top_genes_k{K}.tsv
        results/cnmf_v3_k{K}_usage_by_arm.tsv         lesional usage, CRC vs IBD
        results/cnmf_v3_k{K}_usage_by_cluster.tsv     mean usage per leiden_1.0 cluster

Why K = 17 (the thesis choice)
  Stability over the sweep: 7: 0.988, 9: 0.829, 11: 0.825, 13: 0.791, 15: 0.798,
  17: 0.812, 19: 0.804, 21: 0.801. K=17 is a local maximum, with 5% lower reconstruction
  error than K=9 for 2% less stability. K=7's 0.988 is the usual "too coarse is trivially
  stable" artefact and has the worst error of the sweep.

Projection
  The consensus runs on the stratified subsample. The spectra are fixed gene-space objects,
  so every cell is scored with NNLS against them after the same normalisation cNMF uses:
  TPM, then each gene divided by its standard deviation (no centring).

Programmes are named GEP1..GEP{K} in cNMF's output order.
"""
import argparse
import os
import time

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "4"

import anndata as ad
import numpy as np
import pandas as pd
from cnmf import cNMF
from sklearn.decomposition import non_negative_factorization

from _paths import DATA, RESULTS

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--k", type=int, default=17)
p.add_argument("--density-threshold", type=float, default=0.10)
a = p.parse_args()
K, DT = a.k, a.density_threshold

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

c = cNMF(output_dir=str(RESULTS / "cnmf_v3"), name="fib_v3")
log(f"consensus K={K}, density_threshold={DT}")
c.consensus(k=K, density_threshold=DT, show_clustering=True, close_clustergram_fig=True)
u, ss, st, tg = c.load_results(K=K, density_threshold=DT)
progs = [f"GEP{i+1}" for i in range(u.shape[1])]
for df in (u, ss, st, tg):
    df.columns = progs

A = ad.read_h5ad(DATA / "scvi_input_v3.h5ad")
X = A[:, pd.Index(st.index)].X.tocsr().astype(np.float32)
tot = np.asarray(X.sum(1)).ravel(); tot[tot == 0] = 1
T = X.multiply((1e6 / tot)[:, None]).tocsr()
var = np.asarray(T.power(2).mean(0)).ravel() - np.asarray(T.mean(0)).ravel() ** 2
D = np.maximum(np.asarray(T.todense()) / np.sqrt(np.maximum(var, 1e-12)), 0).astype(np.float32)
H = st.T.to_numpy(dtype=np.float32)
H = H / np.maximum(H.sum(1, keepdims=True), 1e-9)
log(f"NNLS projection of {D.shape[0]:,} cells")
W, _, _ = non_negative_factorization(D, H=H, n_components=K, init="custom", update_H=False,
                                     solver="cd", max_iter=400, random_state=0)
U = pd.DataFrame(W, index=A.obs_names, columns=progs)
U = U.div(U.sum(1).replace(0, np.nan), axis=0).fillna(0)
U.to_parquet(RESULTS / f"cnmf_v3_usage_k{K}_allcells.parquet")
ss.to_csv(RESULTS / f"cnmf_v3_gene_scores_k{K}.tsv", sep="\t")
tg.to_csv(RESULTS / f"cnmf_v3_top_genes_k{K}.tsv", sep="\t", index=False)

obs = A.obs.copy()
obs["cl"] = pd.read_parquet(DATA / "scvi_embedding_v3.parquet").loc[obs.index, "leiden_1.0"].astype(str).values
Uo = U.join(obs[["meta_disease", "involved", "cl"]])

les = Uo[Uo.involved == "Involved / lesion"]
t = les.groupby("meta_disease", observed=True)[progs].mean().T
t["log2_CRC_over_IBD"] = np.log2((t.CRC + 1e-6) / (t.IBD + 1e-6))
t.to_csv(RESULTS / f"cnmf_v3_k{K}_usage_by_arm.tsv", sep="\t")
Uo.groupby("cl", observed=True)[progs].mean().to_csv(
    RESULTS / f"cnmf_v3_k{K}_usage_by_cluster.tsv", sep="\t")

print("\ntop 15 genes per programme")
print(pd.DataFrame({q: ss[q].nlargest(15).index.tolist() for q in progs}).to_string(index=False))
print("\nlesional usage by arm")
print(t.round(4).sort_values("log2_CRC_over_IBD").to_string())

print("\nprogrammes loading non-fibroblast markers in their top 100 genes")
mark = {"epithelial": ["EPCAM","KRT8","KRT18","KRT19","CDH1"],
        "immune": ["PTPRC","CD3D","LYZ","C1QA","TYROBP"],
        "endothelial": ["PECAM1","VWF","CDH5","CLDN5","PLVAP"],
        "mural": ["MYH11","RGS5","NOTCH3","MYOCD"]}
for q in progs:
    top = set(ss[q].nlargest(100).index)
    hits = {k: sorted(top & set(v)) for k, v in mark.items() if top & set(v)}
    if hits:
        print(f"  {q}: {hits}")
log("done")
