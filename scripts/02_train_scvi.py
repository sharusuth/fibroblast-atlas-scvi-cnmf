#!/usr/bin/env python3
"""Step 2 - train scVI and export the latent. Needs a GPU (~30-40 min on a T4).

Input   data/scvi_input_v3.h5ad
Output  data/scvi_latent_v3.parquet   cells x 30 latent dimensions
        data/scvi_model_v3/           trained model (reusable for scArches-style mapping)
        results/scvi_v3_elbo.tsv      training/validation ELBO per epoch

The thesis run used scvi-tools 1.5.0.post1 on Google Colab. UMAP and Leiden are not computed
here; they run on CPU in step 3.

Model choices
  batch_key="donor", not "sample". 206 donors contribute both lesional and uninvolved tissue.
    Batching on sample would treat those as different batches and regress out the
    lesion-vs-uninvolved difference, which is the contrast the atlas exists to measure.
    Batching on donor keeps it within batch, where no correction reaches it.
  No categorical covariates. chemistry, capture_end, assay, meta_disease and meta_source are
    each constant within donor, so they are deterministic functions of the batch.
  dispersion="gene", not "gene-batch". With 585 donors, gene-batch fits ~1.8M dispersion
    parameters, some from donors with only 10 cells.
  gene_likelihood="nb" on raw integer UMIs.
"""
import argparse
import time

import anndata as ad
import pandas as pd
import scvi
import torch

from _paths import DATA, RESULTS

p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
p.add_argument("--input", default=str(DATA / "scvi_input_v3.h5ad"))
p.add_argument("--max-epochs", type=int, default=200)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

scvi.settings.seed = a.seed
print(f"scvi-tools {scvi.__version__} | torch {torch.__version__} | CUDA {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("WARNING: no GPU visible; training on CPU will take many hours")

adata = ad.read_h5ad(a.input)
print(adata.shape, "| donors", adata.obs.donor.nunique(), "| studies", adata.obs.study.nunique())

scvi.model.SCVI.setup_anndata(adata, batch_key="donor")
model = scvi.model.SCVI(adata, n_latent=30, n_layers=2, n_hidden=128,
                        gene_likelihood="nb", dispersion="gene")

t0 = time.time()
model.train(max_epochs=a.max_epochs, batch_size=512, early_stopping=True,
            early_stopping_patience=20, check_val_every_n_epoch=1, plan_kwargs={"lr": 1e-3})
print(f"trained in {(time.time()-t0)/60:.1f} min")

h = model.history
pd.concat([h["elbo_train"], h["elbo_validation"]], axis=1).to_csv(
    RESULTS / "scvi_v3_elbo.tsv", sep="\t")

Z = model.get_latent_representation()
out = pd.DataFrame(Z, index=adata.obs_names,
                   columns=[f"scVI_{i}" for i in range(Z.shape[1])]).astype("float32")
out.index.name = "cell"
out.to_parquet(DATA / "scvi_latent_v3.parquet")
model.save(str(DATA / "scvi_model_v3"), overwrite=True, save_anndata=False)
print("latent", out.shape, "->", DATA / "scvi_latent_v3.parquet")
