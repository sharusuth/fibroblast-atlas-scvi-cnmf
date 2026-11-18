#!/usr/bin/env python3
"""Step 4 - cNMF: stratified subsample, K sweep, combine and K-selection statistics.

Input   data/scvi_input_v3.h5ad, data/scvi_embedding_v3.parquet
Output  data/cnmf_input_v3.h5ad               the subsample cNMF factorises
        results/cnmf_v3/                       cNMF working directory (name "fib_v3")
        results/cnmf_v3_k_selection.tsv       stability and error per K
        figures/cnmf_v3_k_selection.pdf

Parameters
  50,000 cells requested, stratified by leiden_1.0 (equal quota per cluster, smaller
    clusters taken whole), so small populations are not swamped by the large ones.
    Programmes are gene-space objects and are recoverable from a sample; step 5 projects
    every cell back onto them.
  Genes: the 3,000 HVGs of the scVI input, passed explicitly so programmes live in the same
    gene space as the embedding. Counts are raw integers; cNMF does its own TPM and
    variance scaling.
  K = 7..21, step 2, n_iter = 100 (cNMF's default), prepare seed 14.
  3 workers x 2 BLAS threads. Each worker holds its own dense copy of the matrix; six
    workers exhausted 16 GB of RAM. Raise --workers on a bigger machine.

Runs are resumable: factorize uses skip_completed_runs=True.
"""
import argparse
import glob
import re
import subprocess
import sys
import time
import os

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "2"

import anndata as ad
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from cnmf import cNMF

from _paths import DATA, FIGURES, RESULTS

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--workers", type=int, default=3)
p.add_argument("--n-cells", type=int, default=50_000)
p.add_argument("--n-iter", type=int, default=100)
a = p.parse_args()

SRC = DATA / "scvi_input_v3.h5ad"
EMB = DATA / "scvi_embedding_v3.parquet"
INP = DATA / "cnmf_input_v3.h5ad"
OUTD = str(RESULTS / "cnmf_v3")
NAME, KS, SEED = "fib_v3", list(range(7, 22, 2)), 0

t0 = time.time()
def log(m): print(f"[{(time.time()-t0)/60:6.1f}m] {m}", flush=True)

# 1. stratified subsample ------------------------------------------------------------
A = ad.read_h5ad(SRC)
A.obs["cl"] = pd.read_parquet(EMB).loc[A.obs_names, "leiden_1.0"].astype(str).values
rng = np.random.default_rng(SEED)
per = max(1, a.n_cells // A.obs.cl.nunique())
take = []
for _, idx in pd.Series(np.arange(A.n_obs), index=A.obs.cl.values).groupby(level=0):
    ii = idx.values
    take.append(ii if len(ii) <= per else rng.choice(ii, per, replace=False))
S = A[np.sort(np.concatenate(take))].copy()
del A
for c in S.obs.columns:
    if str(S.obs[c].dtype) == "category":
        S.obs[c] = S.obs[c].cat.remove_unused_categories()
X = S.X.tocsr()
assert X.data.min() >= 0 and (X.data == np.round(X.data)).all(), "cNMF needs raw integer counts"
S.write_h5ad(INP)
log(f"subsample {S.shape} | {S.obs.cl.nunique()} clusters | {S.obs.donor.nunique()} donors")

# 2. prepare ---------------------------------------------------------------------------
os.makedirs(OUTD, exist_ok=True)
genes_fn = os.path.join(OUTD, "hvg.txt")
with open(genes_fn, "w") as fh:
    fh.write("\n".join(map(str, S.var_names)))
c = cNMF(output_dir=OUTD, name=NAME)
c.prepare(counts_fn=str(INP), components=KS, n_iter=a.n_iter, seed=14,
          genes_file=genes_fn, densify=True)
log(f"prepared K={KS}, n_iter={a.n_iter} ({len(KS)*a.n_iter} factorisations)")

# 3. factorize in parallel workers -----------------------------------------------------
worker = ("import os\n"
          "for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS',"
          "'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[v]='2'\n"
          "from cnmf import cNMF\n"
          "cNMF(output_dir={o!r}, name={n!r}).factorize(worker_i={w}, total_workers={t}, "
          "skip_completed_runs=True)")
procs = [subprocess.Popen([sys.executable, "-c", worker.format(o=OUTD, n=NAME, w=w, t=a.workers)],
                          stdout=open(f"{OUTD}/w{w}.log", "w"), stderr=subprocess.STDOUT)
         for w in range(a.workers)]
pattern = f"{OUTD}/{NAME}/cnmf_tmp/{NAME}.spectra.k_*.iter_*.df.npz"
while any(pr.poll() is None for pr in procs):
    time.sleep(120)
    log(f"  {len(glob.glob(pattern))}/{len(KS)*a.n_iter} factorisations")
if any(pr.returncode for pr in procs):
    sys.exit(f"a factorize worker failed; see {OUTD}/w*.log, then rerun (completed runs are kept)")

# 4. combine + K-selection statistics --------------------------------------------------
have = sorted({int(re.search(r"k_(\d+)\.", f).group(1)) for f in glob.glob(pattern)})
c.combine(components=have, skip_missing_files=True)
norm = sc.read(c.paths["normalized_counts"])
rows = []
for k in have:
    st = c.consensus(k, skip_density_and_return_after_stats=True, show_clustering=False,
                     close_clustergram_fig=True, norm_counts=norm).stats
    rows.append(st)
    log(f"  K={k:2d}  stability {st.silhouette:.4f}  error {st.prediction_error:.0f}")
stats = pd.DataFrame(rows).reset_index(drop=True)
stats.to_csv(RESULTS / "cnmf_v3_k_selection.tsv", sep="\t", index=False)

# 5. K-selection figure: two stacked panels on a shared x, not a dual-axis plot ---------
fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.6, 4.8), sharex=True, gridspec_kw={"hspace": .16})
a1.plot(stats.k, stats.silhouette, "o-", color="#2a78d6", lw=2, ms=6)
a1.set_ylabel("stability (silhouette)")
a1.set_title(f"cNMF K selection, n_iter={a.n_iter}", loc="left")
a2.plot(stats.k, stats.prediction_error / 1e6, "o-", color="#eb6834", lw=2, ms=6)
a2.set_ylabel("reconstruction error (M)")
a2.set_xlabel("K (number of programmes)")
a2.set_xticks(stats.k)
for ax in (a1, a2):
    ax.grid(color="#e1e0d9", lw=.6); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
fig.savefig(FIGURES / "cnmf_v3_k_selection.pdf", bbox_inches="tight")
log("done; choose K from results/cnmf_v3_k_selection.tsv, then run step 5")
print(stats.to_string(index=False))
