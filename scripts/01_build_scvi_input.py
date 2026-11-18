#!/usr/bin/env python3
"""Step 1 - build the scVI input: QC- and lineage-filtered raw counts on 3,000 HVGs.

Input   data/merged_fibroblasts_v4_10x.h5ad   merged fibroblast counts (raw integer UMIs)
        data/qc_per_cell.parquet              per-cell nGene (index = cell id)
        resources/lineage_removed_cells_v2.txt.gz
Output  data/scvi_input_v3.h5ad               292,828 cells x 3,000 genes in the thesis run

Filters, in order
  1. Author labels Pericyte / Contractile muscle 1 / Contractile muscle 2 removed (mural).
  2. nGene >= 500. No %MT filter: five deposits stripped MT genes, so a %MT cut would only
     ever act on some studies.
  3. Non-fibroblast lineage clusters removed. These were Leiden clusters 10, 11 and 24 of an
     earlier scVI pass, called on ABSOLUTE marker expression rather than log2FC:
       cl 10  pericyte        DCN 24%, LUM 9%, PDGFRA 5%, NOTCH3 75%, ACTA2 87%
       cl 11  smooth muscle   PDGFRA 10%, MYH11 88% (borderline: LUM 66%)
       cl 24  no lineage      COL1A1 5%, DCN 3%, VIM 12%
     log2FC is enrichment relative to other clusters; where every cluster is fibroblast-like,
     collagen genes never surface as markers, so an absence from a top-50 list is not an
     absence from the cells. Checking absolute expression kept four clusters an earlier
     log2FC-based draft would have removed. The cell ids are shipped in resources/ so this
     step does not depend on the earlier pass.
  4. All nuclei (assay == "nucleus"): they cluster on intronic content.
  5. Donors with < 10 cells and studies with < 50 cells, iterated to a fixed point.

Gene space: MT, ribosomal and immediate-early/heat-shock stress genes are excluded from the
HVG pool, genes in < 10 cells dropped, then 3,000 seurat_v3 HVGs selected. The script tries
batch_key="study" first, then "meta_source", then global. In the thesis run the per-study
attempt failed and the HVGs were selected with batch_key="meta_source"; the key used is
recorded in uns["build"]["hvg_batch_key"].
"""
import gzip
import os
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from _paths import DATA, RESOURCES

V4 = DATA / "merged_fibroblasts_v4_10x.h5ad"
QC = DATA / "qc_per_cell.parquet"
REMOVED = RESOURCES / "lineage_removed_cells_v2.txt.gz"
OUT = DATA / "scvi_input_v3.h5ad"
TMP = OUT.with_suffix(".h5ad.building")

N_HVG, MIN_GENES = 3000, 500
MIN_CELLS_DONOR, MIN_CELLS_STUDY = 10, 50
MURAL = ["Pericyte", "Contractile muscle 1", "Contractile muscle 2"]
STRESS = ["FOS","FOSB","FOSL1","FOSL2","JUN","JUNB","JUND","EGR1","EGR2","EGR3","ATF3",
 "IER2","IER3","IER5","NR4A1","NR4A2","NR4A3","DUSP1","DUSP2","DUSP5","KLF2","KLF4","KLF6",
 "ZFP36","ZFP36L1","ZFP36L2","PPP1R15A","SOCS3","BTG1","BTG2","CYR61","CCN1","CCN2","MYC",
 "HSPA1A","HSPA1B","HSPA5","HSPA6","HSPA8","HSPB1","HSPB8","HSPH1","HSP90AA1","HSP90AB1",
 "DNAJA1","DNAJB1","DNAJB4","DNAJB6","BAG3","HSPE1","HSPD1","UBC","UBB","CEBPB","CEBPD",
 "MT1A","MT1E","MT1F","MT1G","MT1M","MT1X","MT2A","SOD2","NFKBIA","NFKBIZ","GEM","ICAM1",
 "CDKN1A","GADD45B","PLK2","RHOB","SGK1","TRIB1","MCL1","PIM1","SLC2A3","TNFAIP3"]
KEEP_OBS = ["study","donor","sample","meta_source","meta_disease","disease_detail","involved",
            "platform","chemistry","capture_end","assay","deposit_gene_filter","prior_label",
            "prior_scheme","mmr_final","tissue","treatment","fusobacterium","kras"]

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

log(f"reading {V4.name}")
A = ad.read_h5ad(V4, backed="r")
inall = np.flatnonzero(A.var["in_all_sources"].to_numpy().astype(bool))
A = A[:, inall].to_memory()
start = A.shape[0]
log(f"  {start:,} cells x {A.shape[1]:,} genes (genes present in every source)")

m = A.obs["prior_label"].astype(str).isin(MURAL).values
log(f"1. mural author labels: -{int(m.sum()):,}")
A = A[~m].copy()

ngene = pd.read_parquet(QC)["nGene"].reindex(A.obs_names)
ngene = ngene.fillna(pd.Series(np.diff(A.X.tocsr().indptr), index=A.obs_names))
k = (ngene >= MIN_GENES).to_numpy()
log(f"2. nGene >= {MIN_GENES}: -{int((~k).sum()):,}  "
    f"{A.obs.loc[~k, 'meta_disease'].value_counts().to_dict()}")
A = A[k].copy()

with gzip.open(REMOVED, "rt") as fh:
    removed = {l.strip() for l in fh if l.strip()}
drop = A.obs_names.isin(list(removed))
nuc = (A.obs["assay"].astype(str) == "nucleus").to_numpy()
log(f"3. lineage clusters: -{int(drop.sum()):,}")
log(f"4. nuclei: -{int((nuc & ~drop).sum()):,} additional")
A = A[~(drop | nuc)].copy()

for it in range(10):
    dc, sm = A.obs["donor"].value_counts(), A.obs["study"].value_counts()
    ok = ((A.obs["donor"].astype(str).map(dc).astype(float).to_numpy() >= MIN_CELLS_DONOR) &
          (A.obs["study"].astype(str).map(sm).astype(float).to_numpy() >= MIN_CELLS_STUDY))
    if ok.all():
        log(f"5. batch-size filter stable after {it} pass(es)")
        break
    log(f"5. pass {it+1}: -{int((~ok).sum()):,}")
    A = A[ok].copy()
for c in A.obs.columns:
    if str(A.obs[c].dtype) == "category":
        A.obs[c] = A.obs[c].cat.remove_unused_categories()
log(f"-> {A.shape[0]:,} cells ({100*A.shape[0]/start:.1f}%) | "
    f"{A.obs.donor.nunique()} donors | {A.obs.study.nunique()} studies")

sym = A.var_names.astype(str)
tech = (sym.str.startswith("MT-") | sym.str.match(r"^RP[LS]\d") |
        sym.str.match(r"^RP[LS]P\d") | sym.str.match(r"^RP[LS]A$"))
stress = sym.isin(STRESS)
log(f"excluding {int(tech.sum())} MT/ribosomal + {int(stress.sum())} stress genes from the HVG pool")
A = A[:, ~(tech | stress)].copy()
n0 = A.shape[1]
sc.pp.filter_genes(A, min_cells=10)
log(f"  dropped {n0-A.shape[1]} genes in <10 cells | pool {A.shape[1]:,}")

hvg = None
for key in ["study", "meta_source", None]:
    try:
        log(f"HVG seurat_v3 batch_key={key!r}")
        sc.pp.highly_variable_genes(A, n_top_genes=N_HVG, flavor="seurat_v3",
                                    batch_key=key, subset=True)
        hvg = key or "none (global)"
        break
    except Exception as e:
        log(f"  failed ({type(e).__name__})")
assert hvg, "HVG selection failed for every batch key"
log(f"  -> {A.shape} [hvg_batch={hvg}]")

A.obs = A.obs[[c for c in KEEP_OBS if c in A.obs.columns]].copy()
for c in A.obs.columns:
    A.obs[c] = A.obs[c].astype(str).astype("category")
X = A.X.tocsr()
assert X.data.min() >= 0 and (X.data == np.round(X.data)).all(), "expected raw integer counts"
A.uns["build"] = dict(version="v3", source=V4.name, n_hvg=N_HVG, hvg_batch_key=hvg,
    min_genes=MIN_GENES, mt_filter="none", stress_excluded_from_hvg=True,
    mural_removed=MURAL, lineage_cells_removed=REMOVED.name, nuclei_removed=True,
    min_cells_donor=MIN_CELLS_DONOR, min_cells_study=MIN_CELLS_STUDY)

if TMP.exists():
    TMP.unlink()
A.write_h5ad(TMP, compression="gzip")
os.replace(TMP, OUT)
log(f"wrote {OUT} ({OUT.stat().st_size/1e6:.0f} MB)")
print(A.obs.meta_disease.value_counts().to_string())
print("paired donors (lesional + uninvolved):",
      int((A.obs.groupby("donor", observed=True)["involved"].nunique() > 1).sum()))
