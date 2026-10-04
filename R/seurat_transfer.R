# Seurat v3 transfer imputation, matching the totalVI paper's baseline
# (YosefLab/totalVI_reproducibility harmonization/R/impute_sln_seurat.R):
# LogNormalize RNA, FindTransferAnchors(dims 1:50, npcs 50, all exported genes),
# TransferData(refdata = ADT 'data' slot = raw counts, k.weight 50, l2.norm FALSE).
#
# Adds cross-fitted predictions for source cells (reference = other folds,
# query = held-out fold) so every method can be calibrated on out-of-sample
# source predictions.
#
# Usage:  Rscript R/seurat_transfer.R paper
# Input:  data/seurat/<benchmark>/ from src/export_for_seurat.py
# Output: results/seurat/<benchmark>/target_imputed.csv, source_crossfit_imputed.csv
#         (cells x proteins, imputed protein counts)

suppressPackageStartupMessages({
  library(Seurat)
  library(Matrix)
})
set.seed(1234)

args <- commandArgs(trailingOnly = TRUE)
benchmark <- if (length(args) >= 1) args[[1]] else "paper"
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_arg) != 1) stop("Run this file with Rscript")
root <- normalizePath(file.path(dirname(sub("^--file=", "", script_arg)), ".."))
inp <- file.path(root, "data", "seurat", benchmark)
out <- file.path(root, "results", "seurat", benchmark)
dir.create(out, recursive = TRUE, showWarnings = FALSE)

read_lines <- function(f) readLines(file.path(inp, f))
genes <- read_lines("genes.txt"); cells <- read_lines("cells.txt")
rna <- as(readMM(file.path(inp, "rna_counts.mtx")), "CsparseMatrix")
dimnames(rna) <- list(make.unique(genes), cells)
proteins <- read_lines("proteins.txt"); source_cells <- read_lines("source_cells.txt")
adt <- as(readMM(file.path(inp, "adt_counts.mtx")), "CsparseMatrix")
dimnames(adt) <- list(gsub("_", "-", proteins), source_cells)   # Seurat forbids "_" in feature names
target_cells <- read_lines("target_cells.txt")
folds <- read.csv(file.path(inp, "source_folds.csv"))

assay_data <- function(obj, assay) {
  out <- tryCatch(GetAssayData(obj, assay = assay, layer = "data"), error = function(e) NULL)
  if (is.null(out) || length(out) == 0) out <- tryCatch(GetAssayData(obj, assay = assay, slot = "data"), error = function(e) NULL)
  # Seurat v5 can leave the ADT data layer empty after CreateAssayObject(counts=).
  # The paper transfers raw ADT counts, so use the counts layer in that case.
  if (is.null(out) || length(out) == 0) out <- tryCatch(GetAssayData(obj, assay = assay, layer = "counts"), error = function(e) NULL)
  if (is.null(out) || length(out) == 0) out <- GetAssayData(obj, assay = assay, slot = "counts")
  out
}

make_obj <- function(cols, with_adt) {
  obj <- CreateSeuratObject(counts = rna[, cols, drop = FALSE])
  if (with_adt) obj[["ADT"]] <- CreateAssayObject(counts = adt[, cols, drop = FALSE])
  DefaultAssay(obj) <- "RNA"
  obj <- NormalizeData(obj, verbose = FALSE)
  ScaleData(obj, verbose = FALSE)
}

impute <- function(ref_cells, query_cells) {
  ref <- make_obj(ref_cells, TRUE)
  query <- make_obj(query_cells, FALSE)
  anchors <- FindTransferAnchors(reference = ref, query = query, features = rownames(ref),
                                 dims = 1:50, npcs = 50, verbose = FALSE)
  refdata <- assay_data(ref, "ADT")      # ADT never normalized -> 'data' equals raw counts, as in the paper
  # The authors' script sets slot = "counts". That option changes where the
  # transferred matrix is stored, not the transfer weights.
  imputed <- TransferData(anchorset = anchors, refdata = refdata, l2.norm = FALSE,
                          dims = 1:50, k.weight = 50, slot = "counts", verbose = FALSE)
  mat <- if (is(imputed, "Assay") || is(imputed, "Assay5")) as.matrix(assay_data_from(imputed)) else as.matrix(imputed)
  t(mat)                                 # cells x proteins
}
assay_data_from <- function(a) {
  x <- tryCatch(LayerData(a, layer = "counts"), error = function(e) NULL)
  if (is.null(x) || length(x) == 0) x <- tryCatch(GetAssayData(a, slot = "counts"), error = function(e) NULL)
  if (is.null(x) || length(x) == 0) x <- tryCatch(LayerData(a, layer = "data"), error = function(e) GetAssayData(a, slot = "data"))
  x
}

write_out <- function(m, name) {
  colnames(m) <- proteins[match(colnames(m), gsub("_", "-", proteins))]   # restore original names
  write.csv(m, file = file.path(out, name))
}

cat("Target transfer:", length(source_cells), "->", length(target_cells), "cells\n")
write_out(impute(source_cells, target_cells), "target_imputed.csv")

pieces <- list()
for (k in sort(unique(folds$fold))) {
  q <- folds$cell[folds$fold == k]; r <- folds$cell[folds$fold != k]
  cat("Source fold", k, ":", length(r), "->", length(q), "cells\n")
  pieces[[k]] <- impute(r, q)
}
write_out(do.call(rbind, pieces), "source_crossfit_imputed.csv")
writeLines(c(paste("Seurat", as.character(packageVersion("Seurat"))), paste("R", R.version.string)),
           file.path(out, "versions.txt"))
cat("Done:", out, "\n")
