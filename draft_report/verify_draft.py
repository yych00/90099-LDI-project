"""Check report evidence, LaTeX structure, PDF pagination and review previews."""
import csv
import hashlib
import json
import re
import statistics
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "my_report" / "submission"
OUT = ROOT / "tmp" / "pdfs"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def line_data(page):
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            yield "".join(s["text"] for s in line["spans"]), line["bbox"]

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (SOURCE / "report" / "main_results.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    runs = [json.loads(p.read_text(encoding="utf-8-sig")) for p in (SOURCE / "results").glob("main_*.json")]
    assert len(rows) == 16 and len(runs) == 48
    results_tex = (ROOT / "tables" / "results.tex").read_text(encoding="utf-8")
    checked_values = 0
    for row in rows:
        selected = [r for r in runs if r["dataset"] == row["dataset"] and r["model"] == row["model"]]
        assert sorted(r["seed"] for r in selected) == [42, 43, 44]
        for metric in ("mae", "rmse", "mase"):
            values = [r["metrics_24h"][metric] for r in selected]
            assert abs(statistics.mean(values) - float(row[metric])) < 1e-8
            assert abs(statistics.stdev(values) - float(row[metric + "_std"])) < 1e-8
            assert f"{float(row[metric]):,.3f}" in results_tex
            if row["model"] not in ("Persistence", "DailySeasonalNaive"):
                assert f"{float(row[metric + '_std']):.3f}" in results_tex
            checked_values += 2
    for name in ("data_overview.pdf", "hard_cases.pdf"):
        assert sha(ROOT / "figures" / name) == sha(SOURCE / "report" / "figures" / name)
    references = (ROOT / "references.tex").read_text(encoding="utf-8")
    refkeys = set(re.findall(r"\\bibitem\{([^}]+)\}", references))
    checks = {"source_main_runs": len(runs), "checked_metric_means_and_stds": checked_values,
              "figure_hashes_match": True, "new_training": False, "reports": {}}
    for stem, headings in (("report", ["Introduction", "Problem Definition and Scope", "Experiments", "Conclusion"]),
                           ("report_cn", ["引言", "问题定义与范围", "实验", "结论"])):
        tex = (ROOT / (stem + ".tex")).read_text(encoding="utf-8")
        aux = (ROOT / (stem + ".aux")).read_text(encoding="utf-8")
        log = (ROOT / (stem + ".log")).read_text(encoding="utf-8", errors="replace")
        assert len(re.findall(r"\\section\{", tex)) == 4
        assert len(re.findall(r"\\subsection\{", tex)) == 14
        assert len(re.findall(r"\\sectionauthor\s", tex)) == 4
        cites = set(k for group in re.findall(r"\\cite\{([^}]+)\}", tex) for k in group.split(","))
        assert cites == refkeys
        assert not re.search(r"Overfull|undefined on input|There were undefined|Missing character", log)
        assert len(re.findall(r"\\newlabel\{(?:tab|fig):", aux)) == 4
        body_pages = int(re.search(r"\\newlabel\{report:body-end\}\{\{[^}]+\}\{(\d+)\}", aux).group(1))
        assert 8 <= body_pages <= 10
        doc = fitz.open(ROOT / (stem + ".pdf"))
        assert len(doc) == body_pages + 3  # one title, one TOC, one bibliography page
        starts = []
        for i, (key, heading) in enumerate(zip(("introduction", "problem", "experiments", "conclusion"), headings), 1):
            p = int(re.search(r"\\newlabel\{sec:" + key + r"\}\{\{[^}]+\}\{(\d+)\}", aux).group(1))
            page = doc[p + 1]
            candidates = [(text, box) for text, box in line_data(page)
                          if re.sub(r"\s", "", re.sub(r"^[1-4]\s*", "", text)) == re.sub(r"\s", "", heading)]
            assert len(candidates) == 1, (stem, key, candidates)
            box = candidates[0][1]
            starts.append(p - 1 + (box[1] - 70.866) / (page.rect.height - 141.732))
        last = doc[body_pages + 1]
        last_bottom = max(box[3] for text, box in line_data(last) if box[1] < last.rect.height - 70.866)
        end = body_pages - 1 + (last_bottom - 70.866) / (last.rect.height - 141.732)
        allocations = [100 * (b - a) / (end - starts[0]) for a, b in zip(starts, starts[1:] + [end])]
        assert all(low <= value <= high for value, (low, high) in
                   zip(allocations, ((10, 20), (30, 40), (30, 40), (10, 20)))), (stem, allocations)
        checks["reports"][stem] = {"body_pages": body_pages, "total_pages": len(doc),
                                  "approx_section_percent": dict(zip(headings, [round(x, 1) for x in allocations])),
                                  "four_sections": True, "fourteen_subsections": True, "two_tables_two_figures": True,
                                  "all_nine_references_cited": True, "section_allocation_in_range": True}
        # Contact sheets supplement full-page previews; inspect both before delivery.
        for offset in range(0, len(doc), 6):
            sheet = Image.new("RGB", (1200, 1170), "#dddddd")
            draw = ImageDraw.Draw(sheet)
            for n in range(offset, min(offset + 6, len(doc))):
                page = doc[n]
                pix = page.get_pixmap(matrix=fitz.Matrix(400 / page.rect.width, 400 / page.rect.width), alpha=False)
                im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                col, rownum = (n - offset) % 3, (n - offset) // 3
                draw.text((col * 400 + 10, rownum * 585 + 5), f"{stem}, PDF page {n + 1}", fill="black")
                sheet.paste(im, (col * 400, rownum * 585 + 20))
            sheet.save(OUT / f"{stem}_contact_{offset // 6 + 1}.png")
    (ROOT / "checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
