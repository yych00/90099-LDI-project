# LDI 报告草稿

- `report.tex` / `report.pdf`：英文提交版。
- `report_cn.tex` / `report_cn.pdf`：中文对照版。
- 两份报告均为四个主章节、14 个子章节、两张表和两幅图；引言完整展开，其余章节采用简短说明、初步结果和后续任务要点。
- 封面、目录和参考文献各一页；正文目标各约九页，最终页数与章节比例记录在 `checks.json`。

## 编译

在本目录执行：

```powershell
pdflatex -interaction=nonstopmode -halt-on-error report.tex
pdflatex -interaction=nonstopmode -halt-on-error report.tex
xelatex -interaction=nonstopmode -halt-on-error report_cn.tex
xelatex -interaction=nonstopmode -halt-on-error report_cn.tex
```

`figures/`、`tables/` 和 `references.tex` 均位于本目录；复制报告时请一起复制。普通编译不依赖原实验目录。引用编号采用两份报告共用的文献列表。

## 材料与核对

结果取自相邻 `my_report/submission/report/main_results.csv`，两幅图取自其 `figures/`。`prepare_tables.py` 仅更新共用结果表并复制已有图，不修改正文、不运行训练。需保留原项目目录才能重新准备这些材料。

```powershell
python prepare_tables.py
python verify_draft.py
```

核对脚本依赖 PyMuPDF 和 Pillow；它将 48 条主实验 JSON 中的 MAE、RMSE、MASE 重新汇总，核对 96 个均值/标准差，检查图文件哈希、章节与引用、PDF 页数，以及近似章节篇幅比例。它不重新训练，也不重新运行全部历史实验。页面检查预览位于 `tmp/pdfs/`。

## 提交前填写

两位学生的姓名、真实学号和各章节实际作者仍保留待填写。请在两份 LaTeX 中同步修改后重新编译；本草稿未推断或虚构个人贡献。PDF 与课程要求的代码 ZIP 应分别提交，本目录未执行上传。
