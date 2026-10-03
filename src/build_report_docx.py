"""Build the consolidated Word report for all completed FusionVI experiments."""

from __future__ import annotations

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "FusionVI_All_Experiments_Technical_Report.docx"
SUMMARY = ROOT / "results" / "fusionvi_experiments_summary.json"
NAVY = "172554"
LIGHT_BLUE = "EFF6FF"
PALE_ORANGE = "FFF7ED"
BORDER = "D9D9D9"


def set_cell_fill(cell, color: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), color)


def set_cell_border(cell, color: str = BORDER) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = tc_pr.first_child_found_in("w:tcBorders")
    if borders is None:
        borders = OxmlElement("w:tcBorders")
        tc_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = f"w:{edge}"
        item = borders.find(qn(tag))
        if item is None:
            item = OxmlElement(tag)
            borders.append(item)
        item.set(qn("w:val"), "single")
        item.set(qn("w:sz"), "5")
        item.set(qn("w:color"), color)


def repeat_header(row) -> None:
    tr_pr = row._tr.get_or_add_trPr()
    item = OxmlElement("w:tblHeader")
    item.set(qn("w:val"), "true")
    tr_pr.append(item)


def style_table(table, widths: list[float]) -> None:
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    for r_idx, row in enumerate(table.rows):
        if r_idx == 0:
            repeat_header(row)
        for c_idx, cell in enumerate(row.cells):
            cell.width = Inches(widths[c_idx])
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            set_cell_border(cell)
            set_cell_fill(cell, NAVY if r_idx == 0 else (LIGHT_BLUE if r_idx % 2 == 0 else "FFFFFF"))
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(3)
                paragraph.paragraph_format.space_after = Pt(3)
                paragraph.paragraph_format.line_spacing = 1.0
                for run in paragraph.runs:
                    run.font.name = "Arial"
                    run.font.size = Pt(8.8)
                    if r_idx == 0:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor(255, 255, 255)


def add_table(doc: Document, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    for idx, text in enumerate(headers):
        table.rows[0].cells[idx].text = text
    for values in rows:
        cells = table.add_row().cells
        for idx, value in enumerate(values):
            cells[idx].text = str(value)
            cells[idx].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT if idx == 0 else WD_ALIGN_PARAGRAPH.CENTER
    style_table(table, widths)
    spacer = doc.add_paragraph()
    spacer.paragraph_format.space_after = Pt(1)


def add_bullet(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph(style="List Bullet")
    paragraph.add_run(text)


def add_figure(doc: Document, path: Path, caption: str, width: float = 6.65) -> None:
    if not path.exists():
        return
    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(path), width=Inches(width))
    cap = doc.add_paragraph(caption)
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(7)
    for run in cap.runs:
        run.italic = True
        run.font.size = Pt(8.5)


def add_result_lead(doc: Document, text: str) -> None:
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(7)
    run = paragraph.add_run(text)
    run.bold = True
    run.font.color.rgb = RGBColor(194, 65, 12)


def configure_document(doc: Document) -> None:
    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(0.62)
    section.bottom_margin = Inches(0.60)
    section.left_margin = Inches(0.72)
    section.right_margin = Inches(0.72)

    styles = doc.styles
    styles["Normal"].font.name = "Arial"
    styles["Normal"].font.size = Pt(10.2)
    styles["Normal"].paragraph_format.space_after = Pt(5)
    styles["Normal"].paragraph_format.line_spacing = 1.07
    for name, size in (("Title", 28), ("Heading 1", 16), ("Heading 2", 12.5)):
        style = styles[name]
        style.font.name = "Arial"
        style.font.size = Pt(size)
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.font.bold = True
    styles["Heading 1"].paragraph_format.space_before = Pt(10)
    styles["Heading 1"].paragraph_format.space_after = Pt(4)
    styles["Heading 2"].paragraph_format.space_before = Pt(7)
    styles["Heading 2"].paragraph_format.space_after = Pt(3)
    styles["List Bullet"].font.name = "Arial"
    styles["List Bullet"].font.size = Pt(10.2)
    styles["List Bullet"].paragraph_format.space_after = Pt(2)

    header = section.header.paragraphs[0]
    header.text = "FusionVI  |  Consolidated experimental report"
    header.runs[0].font.name = "Arial"
    header.runs[0].font.size = Pt(8.5)
    header.runs[0].font.color.rgb = RGBColor(71, 85, 105)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("Multimodal AI for therapeutic discovery")
    footer.runs[0].font.name = "Arial"
    footer.runs[0].font.size = Pt(8)
    footer.runs[0].font.color.rgb = RGBColor(100, 116, 139)


def main() -> None:
    lawlor, papalexi, targeted, paper = json.loads(SUMMARY.read_text())["experiments"]
    pm = papalexi["metrics"]
    relative = 100 * (paper["totalvi_mean_rmsle"] - paper["fusionvi_mean_rmsle"]) / paper["totalvi_mean_rmsle"]
    parameter_reduction = 100 * (paper["totalvi_parameters"] - paper["fusionvi_parameters"]) / paper["totalvi_parameters"]

    doc = Document()
    configure_document(doc)

    title = doc.add_paragraph(style="Title")
    title.add_run("FusionVI multimodal validation experiments")
    subtitle = doc.add_paragraph()
    subtitle.paragraph_format.space_after = Pt(12)
    run = subtitle.add_run("Technical report covering four biomarker recovery and perturbation transfer studies")
    run.font.name = "Arial"
    run.font.size = Pt(15.5)
    run.font.color.rgb = RGBColor(23, 37, 84)
    meta = doc.add_paragraph()
    meta.add_run("totalVI versus FusionVI  •  October 2026").bold = True
    meta.paragraph_format.space_after = Pt(14)

    doc.add_heading("Abstract", level=1)
    doc.add_paragraph(
        "FusionVI tests whether modality-specific encoding improves recovery of therapeutic surface biomarkers when protein "
        "measurements are hidden, discordant with RNA or observed under an unseen perturbation. Four completed experiments "
        "cover held-out human donors, held-out CRISPR targets, cross-mouse transfer and the full missing-protein-panel benchmark "
        "from the totalVI paper. The evidence is deliberately mixed: multimodal context is useful in the PD-L1 perturbation task, "
        "while targeted marker studies show that measured protein context often matters more than the learned latent. The apparent "
        "missing-panel benefit is small and still requires capacity controls. These results support a bounded biomarker-completion "
        "and perturbation-ranking contribution."
    )

    doc.add_heading("Project question", level=1)
    doc.add_paragraph(
        "Can modality-specific fusion improve recovery of therapeutic surface biomarkers when protein measurements are missing, "
        "discordant with RNA, or observed under an unseen molecular perturbation?"
    )

    doc.add_heading("Experiment overview", level=1)
    add_table(
        doc,
        ["Experiment", "Held-out unit", "Primary finding"],
        [
            ["Lawlor activation", "10 human donors", "Native encoders close; cross-modal context supplied most of the marker-recovery gain"],
            ["Papalexi PD-L1", "25 CRISPR targets", "FusionVI-X reached 0.879 effect Spearman and 84.0% direction accuracy"],
            ["SLN111 targeted markers", "2 mice", "Protein context + RNA reached 0.670; either latent added no more than 0.003"],
            ["SLN111 complete panel", "4 paired seeds", "RMSLE difference -0.0060; seed-level 95% CI crossed zero"],
        ],
        [1.55, 1.35, 3.85],
    )

    doc.add_heading("Model and evaluation framework", level=1)
    doc.add_paragraph(
        "totalVI is the published baseline. FusionVI retains its generative decoder and RNA and protein likelihoods, but replaces "
        "the joint encoder with separate RNA and protein branches combined by a learned cell-level gate. In the complete-panel "
        "benchmark, the gate follows an explicit availability rule: if every protein is absent, the cell is routed through RNA only."
    )
    doc.add_paragraph(
        "Experiments 1 to 3 also evaluated FusionVI-X, a nested cross-modal readout using the latent state, remaining measured "
        "proteins and a prespecified matching transcript. The identical readout applied to totalVI is totalVI-X. Native-decoder "
        "and X-readout results are reported separately so the encoder contribution is not confused with the supervised readout."
    )

    doc.add_page_break()
    doc.add_heading("Experiment 1 Lawlor donor held out activation", level=1)
    doc.add_heading("Biological question and design", level=2)
    doc.add_paragraph(
        f"Can hidden CD25, CD69 and HLA-DR be recovered in an unseen donor, particularly in cells where RNA and surface protein "
        f"disagree? The dataset contains {lawlor['cells']:,} PBMCs from 10 paired donors under baseline, LPS or anti-CD3/CD28 "
        "stimulation, with 4,000 genes and 39 antibody-derived tags. Each donor served once as the untouched test fold."
    )
    add_table(
        doc,
        ["Target", "totalVI", "FusionVI", "totalVI-X", "FusionVI-X"],
        [
            ["CD25 in T cells", f"{lawlor['native_totalvi_spearman'][0]:.3f}", f"{lawlor['native_fusionvi_spearman'][0]:.3f}", f"{lawlor['totalvi_xmodal_spearman'][0]:.3f}", f"{lawlor['fusionvi_xmodal_spearman'][0]:.3f}"],
            ["CD69 in T cells", f"{lawlor['native_totalvi_spearman'][1]:.3f}", f"{lawlor['native_fusionvi_spearman'][1]:.3f}", f"{lawlor['totalvi_xmodal_spearman'][1]:.3f}", f"{lawlor['fusionvi_xmodal_spearman'][1]:.3f}"],
            ["HLA-DR in monocytes", f"{lawlor['native_totalvi_spearman'][2]:.3f}", f"{lawlor['native_fusionvi_spearman'][2]:.3f}", f"{lawlor['totalvi_xmodal_spearman'][2]:.3f}", f"{lawlor['fusionvi_xmodal_spearman'][2]:.3f}"],
        ],
        [1.70, 1.15, 1.15, 1.15, 1.15],
    )
    add_result_lead(doc, "Result  The encoder change alone produced small and inconsistent differences.")
    doc.add_paragraph(
        "Both cross-modal readouts improved marker recovery, especially HLA-DR, but totalVI-X and FusionVI-X were nearly tied. "
        "The ablation supplies the biological insight: a matching transcript can become misleading when RNA and surface protein "
        "diverge, while the remaining protein panel restores predictive context."
    )
    add_figure(doc, ROOT / "results" / "figures" / "lawlor_marker_recovery.png", "Figure 1. Donor-paired native-decoder marker recovery. Large diamonds show means.", 6.55)

    doc.add_page_break()
    doc.add_heading("Experiment 2 Papalexi unseen CRISPR targets", level=1)
    doc.add_heading("Biological question and design", level=2)
    doc.add_paragraph(
        f"Can PD-L1 protein effects be predicted for a perturbation absent from training? The ECCITE-seq screen contains "
        f"{papalexi['cells']:,} IFN-gamma-treated THP-1 cells, 25 perturbation targets, three biological replicates and four surface "
        "proteins. Five outer folds held out complete CRISPR targets. Evaluation used 75 target-by-replicate effects."
    )
    add_table(
        doc,
        ["Model", "Effect Spearman", "Direction accuracy", "Effect MAE"],
        [
            ["CD274 RNA only", f"{pm['cd274_rna_only']['effect_spearman']:.3f}", f"{100*pm['cd274_rna_only']['direction_accuracy']:.1f}%", "—"],
            ["totalVI decoder", f"{pm['totalvi_decoder']['effect_spearman']:.3f}", f"{100*pm['totalvi_decoder']['direction_accuracy']:.1f}%", "—"],
            ["FusionVI decoder", f"{pm['fusionvi_decoder']['effect_spearman']:.3f}", f"{100*pm['fusionvi_decoder']['direction_accuracy']:.1f}%", "—"],
            ["totalVI-X", f"{pm['totalvi_xmodal']['effect_spearman']:.3f}", f"{100*pm['totalvi_xmodal']['direction_accuracy']:.1f}%", f"{pm['totalvi_xmodal']['effect_mae']:.3f}"],
            ["FusionVI-X", f"{pm['fusionvi_xmodal']['effect_spearman']:.3f}", f"{100*pm['fusionvi_xmodal']['direction_accuracy']:.1f}%", f"{pm['fusionvi_xmodal']['effect_mae']:.3f}"],
        ],
        [2.05, 1.55, 1.70, 1.35],
    )
    add_result_lead(doc, "Result  FusionVI-X ranked unseen perturbation effects best.")
    doc.add_paragraph(
        f"FusionVI-X reduced median gene-level absolute error by {papalexi['paired_gene_test']['median_absolute_error_reduction']:.4f} "
        f"relative to totalVI-X (paired Wilcoxon p={papalexi['paired_gene_test']['p_value']:.3f}, 25 targets). It recovered expected "
        "PD-L1 loss after IFNGR1, IFNGR2, JAK2 and STAT1 perturbation and gain after CUL3 or BRD4 perturbation."
    )
    doc.add_paragraph(
        "The model failed on CMTM6: measured PD-L1 decreased while CD274 RNA increased slightly. This is a useful boundary condition "
        "because CMTM6 regulates PD-L1 stability after translation, a mechanism that transcript-centered inference can miss."
    )
    add_figure(doc, ROOT / "results" / "figures" / "papalexi_pdl1_validation.png", "Figure 2. Prediction of PD-L1 effects for CRISPR targets excluded from training.", 6.55)

    doc.add_page_break()
    doc.add_heading("Experiment 3 original totalVI targeted marker transfer", level=1)
    doc.add_heading("Biological question and design", level=2)
    doc.add_paragraph(
        f"Can four hidden immune surface markers transfer to an unseen animal? The SLN111 object contains {targeted['cells']:,} "
        "mouse spleen and lymph-node cells, 4,000 genes and 110 proteins. CD20, CD28, CD4 and CD8a were masked together. "
        "Each model trained on one mouse and was evaluated on the other, then the direction was reversed."
    )
    add_table(
        doc,
        ["Readout", "totalVI", "FusionVI", "Difference"],
        [
            ["Native decoder mean Spearman", f"{targeted['native_totalvi_mean_spearman']:.3f}", f"{targeted['native_fusionvi_mean_spearman']:.3f}", f"{targeted['native_fusionvi_mean_spearman']-targeted['native_totalvi_mean_spearman']:+.3f}"],
            ["Fixed cross-modal mean Spearman", f"{targeted['totalvi_xmodal_mean_spearman']:.3f}", f"{targeted['fusionvi_xmodal_mean_spearman']:.3f}", f"{targeted['fusionvi_xmodal_mean_spearman']-targeted['totalvi_xmodal_mean_spearman']:+.3f}"],
            ["Protein context + matching RNA", f"{targeted['protein_context_plus_rna_mean_spearman']:.3f}", f"{targeted['protein_context_plus_rna_mean_spearman']:.3f}", "shared baseline"],
        ],
        [3.10, 1.20, 1.20, 1.20],
    )
    add_result_lead(doc, "Result  Most cross-mouse performance came from measured protein context, not either latent.")
    doc.add_paragraph(
        f"The remaining 106 proteins plus matching transcript reached {targeted['protein_context_plus_rna_mean_spearman']:.3f}. "
        f"Adding totalVI or FusionVI latents changed mean Spearman by only {targeted['totalvi_latent_increment']:.3f} and "
        f"{targeted['fusionvi_latent_increment']:.3f}. Because the dataset contains only two mice, it supports technical transfer "
        "rather than a population-level biological claim."
    )
    add_figure(doc, ROOT / "results" / "figures" / "totalvi_original_marker_recovery.png", "Figure 3. Hidden-marker recovery across the two mouse-held-out directions.", 6.60)

    doc.add_page_break()
    doc.add_heading("Experiment 4 paper aligned complete missing panel", level=1)
    doc.add_heading("Biological question and design", level=2)
    doc.add_paragraph(
        "Can RNA recover all 110 surface proteins in a batch where no protein measurements enter training? This experiment follows "
        "the totalVI paper's Figure 3 design. SLN111-D1 retained RNA and proteins; the entire D2 panel was hidden and preserved only "
        "for evaluation. Decoder, likelihoods, latent size, optimizer, split and prediction procedure were controlled."
    )
    add_table(
        doc,
        ["Metric", "totalVI", "FusionVI", "Difference"],
        [
            ["RMSLE primary", f"{paper['totalvi_mean_rmsle']:.4f}", f"{paper['fusionvi_mean_rmsle']:.4f}", f"{paper['fusionvi_minus_totalvi_rmsle']:+.4f}"],
            ["Proteins with lower RMSLE", "—", f"{paper['proteins_fusionvi_better']}/{paper['proteins_compared']}", "—"],
            ["Trainable parameters", f"{paper['totalvi_parameters']:,}", f"{paper['fusionvi_parameters']:,}", f"{parameter_reduction:.1f}% fewer"],
        ],
        [2.45, 1.40, 1.40, 1.45],
    )
    add_result_lead(doc, f"Result  FusionVI's mean RMSLE was {relative:.2f}% lower, but the seed-level interval included zero.")
    doc.add_paragraph(
        f"The seed-paired difference was {paper['fusionvi_minus_totalvi_rmsle']:+.4f} (95% CI "
        f"{paper['seed_level_rmsle_ci95'][0]:+.4f} to {paper['seed_level_rmsle_ci95'][1]:+.4f}; paired t p="
        f"{paper['seed_level_paired_t_p']:.2f}; exact p={paper['seed_level_exact_p']:.2f}). Three of four seeds favored FusionVI. "
        f"The protein-level p={paper['protein_level_wilcoxon_p']:.3g} is descriptive because proteins are repeated outcomes. "
        f"The {parameter_reduction:.1f}% parameter difference mostly reflects width 128 versus 256 and removal of totalVI's "
        "separate library encoder. Same-width, parameter-matched and missingness-aware controls are required to isolate fusion."
    )
    add_figure(doc, ROOT / "results" / "figures" / "paper_benchmark_totalvi_vs_fusionvi.png", "Figure 4. Paired seeds, per-protein RMSLE and paired protein differences.", 6.70)

    doc.add_page_break()
    doc.add_heading("Combined interpretation", level=1)
    doc.add_paragraph(
        "The experiments support a narrow and biologically coherent contribution. The complete-panel experiment suggests a small "
        "candidate benefit, but four seeds and unequal capacity do not yet identify gated fusion as its cause. When other proteins remain "
        "available, measured protein context often contributes more than the learned latent. The Papalexi result shows that multimodal context "
        "can rank unseen PD-L1 perturbation effects, while the CMTM6 failure exposes a post-transcriptional boundary."
    )
    add_bullet(doc, "Candidate result: lower mean full-panel reconstruction error, with uncertainty spanning zero.")
    add_bullet(doc, "Positive result: improved ranking and direction of unseen PD-L1 perturbation effects.")
    add_bullet(doc, "Negative result: the encoder alone did not consistently improve donor-held-out marker recovery.")
    add_bullet(doc, "Boundary condition: RNA-centered evidence can fail for protein-stability mechanisms such as CMTM6.")
    doc.add_paragraph(
        "The work is therefore a biomarker-completion and perturbation-ranking study for therapeutic discovery. It does not establish "
        "clinical utility, patient-level generalization or replacement of prospective protein measurements."
    )

    doc.add_heading("Reproducibility and limitations", level=1)
    add_bullet(doc, "Lawlor: 10 donors, one dose and one 24-hour timepoint.")
    add_bullet(doc, "Papalexi: one IFN-gamma-treated cell line and 25 molecular perturbations.")
    add_bullet(doc, "Targeted SLN111 transfer: two mice, sufficient only for a technical check.")
    add_bullet(doc, "Complete-panel benchmark: one source-target batch pair and four seeds rather than the paper's 30.")
    add_bullet(doc, "The original complete-panel comparison differs in encoder width and library-network design; control arms are running.")
    add_bullet(doc, "Seed-level inference is primary; per-protein tests are descriptive repeated-outcome summaries.")
    add_bullet(doc, "Native and X-readout results are separated throughout the report.")
    add_bullet(doc, "Consolidated metrics are stored in results/fusionvi_experiments_summary.json.")
    doc.add_paragraph(
        "The current full-panel benchmark is reproduced with .\\run_paper_benchmark.ps1. Earlier executed experiment outputs remain "
        "traceable in repository history; their consolidated metrics and figures are versioned with this report."
    )

    doc.add_heading("References", level=1)
    doc.add_paragraph(
        "Gayoso A, Steier Z, Lopez R, et al. Joint probabilistic modeling of single-cell multi-omic data with totalVI. "
        "Nature Methods. 2021;18:272–282. https://doi.org/10.1038/s41592-020-01050-x"
    )
    doc.add_paragraph(
        "Lawlor N, et al. Multiomic profiling identifies transcriptional and protein-level immune responses to stimulation. "
        "Frontiers in Immunology. 2021."
    )
    doc.add_paragraph(
        "Papalexi E, et al. Mapping and analysis of perturbation responses in single cells by pooled CRISPR screening. "
        "Nature Genetics. 2021."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
