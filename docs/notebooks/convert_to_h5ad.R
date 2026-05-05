library(Seurat)

path_seurat <- "/exports/para-lipg-hpc/Julius_clean/File Objects/srt_2026.rds"

srt <- readRDS(path_seurat)

rna.data <- GetAssayData(srt, assay = "RNA", layer = "counts")  |> t() |> as.matrix()
data.table::fwrite(rna.data, "docs/notebooks/data/bcells_rna.csv", nThread=12)

data.table::fwrite(srt@meta.data, "docs/notebooks/data/bcells_obs.csv", nThread=12)
