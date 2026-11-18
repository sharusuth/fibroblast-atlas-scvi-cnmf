#!/usr/bin/env python3
"""Step 3 - neighbours, Leiden resolution sweep and UMAP on the scVI latent.

Input   data/scvi_latent_v3.parquet, data/scvi_input_v3.h5ad, data/qc_per_cell.parquet
Output  data/scvi_embedding_v3.parquet       latent + UMAP1/2 + leiden_<res> columns
        results/leiden_resolution_sweep.tsv

The neighbour graph is built once and every resolution reuses it, so the sweep is cheap.
Each resolution is scored on four things:
  n_clusters        granularity
  ari_within_study  ARI against the original authors' labels, computed within each study
                    (label vocabularies differ between studies, so a global ARI is
                    meaningless) and weighted by cells; studies with >= 3 labels and
                    >= 500 cells only
  study_dominated   fraction of clusters > 80% from one study; rises once clusters split
                    on batch
  depth_R2          variance in log10(nGene) explained by cluster; rises once clusters split
                    on sequencing depth
Take the ARI plateau before study_dominated and depth_R2 start climbing. Downstream steps
use leiden_1.0.
"""
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score

from _paths import DATA, RESULTS

LAT = DATA / "scvi_latent_v3.parquet"
SLIM = DATA / "scvi_input_v3.h5ad"
QC = DATA / "qc_per_cell.parquet"
OUT = DATA / "scvi_embedding_v3.parquet"
N_NEIGHBORS, MIN_DIST = 15, 0.3
RESOLUTIONS = [0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6, 2.0]

t0 = time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)

Z = pd.read_parquet(LAT)
A = ad.read_h5ad(SLIM, backed="r"); obs = A.obs.copy(); A.file.close()
assert Z.index.equals(obs.index), "cell order differs between latent and input"
obs = obs.join(pd.read_parquet(QC)[["nGene"]].reindex(obs.index))
log(f"latent {Z.shape}")

E = ad.AnnData(X=None, obs=obs, shape=(len(obs), 0))
E.obsm["X_scVI"] = Z.to_numpy(dtype="float32")
sc.pp.neighbors(E, use_rep="X_scVI", n_neighbors=N_NEIGHBORS, random_state=0)

vocab = obs.groupby("study", observed=True)["prior_label"].nunique()
size = obs.study.value_counts()
scorable = [s for s in vocab.index if vocab[s] >= 3 and size[s] >= 500]
log(f"ARI scored in {len(scorable)} studies ({int(size[scorable].sum()):,} cells)")

lg = np.log10(obs.nGene.clip(lower=1).to_numpy())
grand, ss_tot = lg.mean(), ((lg - lg.mean()) ** 2).sum()
rows = []
for r in RESOLUTIONS:
    key = f"leiden_{r}"
    sc.tl.leiden(E, resolution=r, key_added=key, flavor="igraph", n_iterations=2,
                 directed=False, random_state=0)
    cl = E.obs[key].astype(str)
    ct = pd.crosstab(cl, obs.study)
    dom = ((ct.max(1) / ct.sum(1)) > 0.8).mean()
    num = den = 0.0
    for s in scorable:
        m = (obs.study == s).to_numpy()
        num += adjusted_rand_score(obs.prior_label[m].astype(str), cl[m]) * m.sum()
        den += m.sum()
    ss_bet = sum(((lg[(cl == c).to_numpy()].mean() - grand) ** 2) * (cl == c).sum()
                 for c in cl.unique())
    rows.append(dict(resolution=r, n_clusters=cl.nunique(), ari_within_study=round(num/den, 4),
                     study_dominated=round(dom, 4), depth_R2=round(ss_bet/ss_tot, 4)))
    log(f"  res {r:<4} clusters {cl.nunique():3d}  ARI {num/den:.3f}  "
        f"study-dominated {dom:.3f}  depth R2 {ss_bet/ss_tot:.3f}")

pd.DataFrame(rows).to_csv(RESULTS / "leiden_resolution_sweep.tsv", sep="\t", index=False)

sc.tl.umap(E, min_dist=MIN_DIST, random_state=0)
out = pd.DataFrame(E.obsm["X_scVI"], index=obs.index,
                   columns=[f"scVI_{i}" for i in range(Z.shape[1])]).astype("float32")
out[["UMAP1", "UMAP2"]] = E.obsm["X_umap"].astype("float32")
for r in RESOLUTIONS:
    out[f"leiden_{r}"] = E.obs[f"leiden_{r}"].values
out.index.name = "cell"
out.to_parquet(OUT)
log(f"wrote {OUT}")
