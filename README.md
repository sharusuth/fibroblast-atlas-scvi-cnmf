# Fibroblast atlas: scVI integration and cNMF programmes

Code for two steps of a cross-disease fibroblast atlas built from public colorectal cancer
(CRC) and inflammatory bowel disease (IBD) single-cell data:

1. **scVI integration** of 292,828 fibroblasts from 585 donors and 44 studies.
2. **Consensus NMF (cNMF)** to find gene-expression programmes in the integrated object
   and compare their usage between CRC and IBD.

The upstream work (downloading the studies, harmonising metadata and merging them into one
object) is not part of this repository. Step 1 starts from that merged object.

## Pipeline

| Step | Script | Output | Runs on |
|---|---|---|---|
| 1 | `scripts/01_build_scvi_input.py` | `data/scvi_input_v3.h5ad` | CPU, ~32 GB RAM |
| 2 | `scripts/02_train_scvi.py` | `data/scvi_latent_v3.parquet`, `data/scvi_model_v3/` | GPU, ~30–40 min on a T4 |
| 3 | `scripts/03_cluster_latent.py` | `data/scvi_embedding_v3.parquet`, Leiden sweep table | CPU |
| 4 | `scripts/04_cnmf_factorize.py` | `results/cnmf_v3/`, K-selection table and figure | CPU, several hours |
| 5 | `scripts/05_cnmf_consensus.py --k 17` | programme usage for every cell, gene scores | CPU |

Each script's docstring explains the parameter choices.

```bash
conda env create -f environment.yml
conda activate fib-atlas
cd scripts
python 01_build_scvi_input.py
python 02_train_scvi.py
python 03_cluster_latent.py
python 04_cnmf_factorize.py
python 05_cnmf_consensus.py --k 17
```

By default, files are read from and written to `data/`, `results/` and `figures/` inside
the repository. To use another location, set `FIB_ATLAS_ROOT`:

```bash
export FIB_ATLAS_ROOT=/path/to/project
```

## Inputs

Place these in `data/` before step 1:

- **`merged_fibroblasts_v4_10x.h5ad`**: raw integer UMI counts for all fibroblasts, with
  - `var["in_all_sources"]` (bool): gene measured in every source study
  - `obs` columns `study`, `donor`, `sample`, `meta_source`, `meta_disease` (`CRC`/`IBD`),
    `involved` (`Involved / lesion` or uninvolved), `assay` (`cell`/`nucleus`) and
    `prior_label` (the original authors' cell label), plus optional covariates listed in
    step 1.
- **`qc_per_cell.parquet`**: per-cell `nGene`, indexed by cell id.

`resources/lineage_removed_cells_v2.txt.gz` lists the 4,723 cells that step 1 removes as
non-fibroblast lineages (pericyte, smooth muscle, unidentifiable). These were clusters from an
earlier integration pass. Shipping the ids keeps step 1 independent of that pass.

## Key choices

- **Batch is donor, not sample.** 206 donors contribute both lesional and uninvolved tissue.
  Treating samples as batches would remove the lesion-vs-uninvolved difference.
- **No %MT filter.** Several deposits removed mitochondrial genes, so a %MT cut would act
  on some studies only.
- **Stress genes are excluded from the HVG pool** (immediate-early and heat-shock genes),
  so dissociation stress does not drive the latent space or the programmes.
- **Nuclei are removed**, because they cluster on intronic content.
- **HVGs are batch-aware.** In the thesis run they were selected per `meta_source` (the
  per-study attempt failed and the script fell back).
- **cNMF K = 17**, a local stability maximum. The K = 7..21 sweep is in step 5's docstring.

## Software

The thesis run used scvi-tools 1.5.0.post1 (Google Colab, T4 GPU), cnmf 1.7.1,
scanpy 1.12.1 and anndata 0.12.16. Random seeds are fixed. Even so, GPU training is not
bit-for-bit deterministic, so a retrained latent will differ slightly from the one used in
the thesis.
