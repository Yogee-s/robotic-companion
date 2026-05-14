# Documentation

## Reports

- **`index.html`** — Full technical report (self-contained HTML). Open in a browser.
- **`executive_summary.html`** — One-page executive summary.
- **`Robot_Companion_Report.pdf`** — PDF export of the full report.
- **`Executive_Summary.pdf`** — PDF export of the executive summary.

## Generating PDFs

Open the HTML file in Chromium → **Ctrl-P** → Paper A4, Default margins,
**Background graphics on** → Save as PDF. The print stylesheet hides the
sidebar TOC and adds page breaks between chapters.

## Assets

- `assets/figures/` — Hand-authored SVGs for the report
- `assets/images/` — Photos and logos
- `assets/css/` — Report stylesheet
- `assets/js/` — Report scripts (TOC generation, syntax highlighting)

## Local preview

```bash
cd docs && python3 -m http.server 8000
# Open http://localhost:8000
```
