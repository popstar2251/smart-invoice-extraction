# Smart Invoice & Document Extraction Pipeline

An end-to-end pipeline that extracts structured data from invoices in **PDF, PNG, JPG, WEBP, and JFIF** formats — turning messy real-world documents into clean, validated, spreadsheet-ready data.

Built as a portfolio project to demonstrate practical AI/ML engineering: not just "call an LLM API," but a full pipeline with document parsing, structured extraction, automated validation, and a second independent verification layer.

---

## 🎯 What it does

1. **Ingests** invoices in multiple formats (PDF, images, including uncommon formats like WEBP/JFIF)
2. **Extracts** raw text/layout using [Docling](https://github.com/docling-project/docling) (OCR + table structure detection)
3. **Structures** the extracted text into a strict schema (invoice number, vendor, line items, totals) using an LLM with Pydantic-enforced output
4. **Validates** the extracted data — checks that line items sum to the subtotal, that subtotal + tax + shipping = total due, and that required fields are present
5. **Cross-checks** the extraction against the original page image using a Vision-Language Model (VLM), as a second, independent layer of error detection
6. **Outputs** two linked CSVs — an invoice-level summary and a line-item-level detail table — every invoice included, flagged clearly if it needs human review

---

## 🏗️ Architecture

```
                 ┌─────────────┐
   PDF/Image  →  │   Docling   │  →  Markdown text (+ page image)
                 │ (OCR/layout)│
                 └──────┬──────┘
                        │
                        ▼
              ┌───────────────────┐
              │   LLM Extraction   │  →  Structured Invoice object
              │ (Groq + Pydantic)  │      (Pydantic-validated schema)
              └──────────┬─────────┘
                         │
           ┌─────────────┴─────────────┐
           ▼                           ▼
  ┌──────────────────┐       ┌──────────────────────┐
  │  Math Validation   │       │   VLM Verification    │
  │ (subtotal/total    │       │ (cross-check against  │
  │  consistency check)│       │  the original image)  │
  └─────────┬──────────┘       └───────────┬───────────┘
            │                              │
            └──────────────┬───────────────┘
                            ▼
                 ┌─────────────────────┐
                 │   CSV Output          │
                 │ (summary + line items,│
                 │  every invoice flagged)│
                 └─────────────────────┘
```

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Document parsing / OCR | [Docling](https://github.com/docling-project/docling) (RapidOCR backend) |
| LLM extraction | [Groq](https://groq.com/) (`openai/gpt-oss-120b`) via LangChain |
| Structured output | Pydantic + `PydanticOutputParser` |
| VLM verification | HuggingFace Inference Providers (`zai-org/GLM-4.5V`, `Qwen2.5-VL-3B-Instruct`) |
| Data output | pandas → CSV |
| Runtime | Google Colab (GPU) |

---

## 📂 Repo structure

```
├── notebooks/
│   └── Smart_Invoice_and_Document_Extraction_Pipeline.ipynb   # main pipeline notebook
├── sample_invoices/                                           # a few sample/dummy invoices to test with
│   ├── invoice_pdf_example.pdf
│   └── invoice_image_example.png
├── sample_output/                                             # example output from a sample run
│   ├── invoices_summary.csv
│   └── line_items_detail.csv
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/smart-invoice-extraction.git
cd smart-invoice-extraction
```

### 2. Get free API keys
- **Groq** (LLM extraction): [console.groq.com](https://console.groq.com) — free tier
- **HuggingFace** (VLM verification): [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) — free tier, "Read" access token

### 3. Run it in Google Colab (recommended)
1. Open `notebooks/invoice_extraction_pipeline.ipynb` in Colab
2. Add your API keys via Colab's Secrets manager (🔑 icon in the sidebar):
   - `GROQ_API_KEY`
   - `HF_TOKEN`
3. Upload your invoices to a folder (or use the provided `sample_invoices/`)
4. Update the `INVOICE_FOLDER` path in the notebook to point to your folder
5. Run all cells

### 4. Or run locally
```bash
pip install -r requirements.txt
```
Set the same two environment variables, then run the notebook via Jupyter, or adapt the pipeline into a `.py` script.

---

## 📊 Sample Output

The pipeline produces two linked CSVs:

**`invoices_summary.csv`** — one row per invoice:

| invoice_number | vendor_name | total_due | status | vlm_validated | vlm_issues |
|---|---|---|---|---|---|
| INV-000101 | Jennifer Thompson | 1095.0 | OK | Yes | None |
| BPXINV-00550 | Bioplex | 6610.95 | OK | Yes | None |

**`line_items_detail.csv`** — one row per line item, linked back by `invoice_number`.

Every invoice is saved, even ones that fail validation — flagged with `status: NEEDS REVIEW` and specific error/warning messages, rather than silently dropped.

---

## 🐛 Debugging Journey (the interesting part)

This project went through real, non-trivial debugging — documented here because the process is arguably more instructive than the final code:

- **Free-tier LLM reliability**: HuggingFace's free Inference API repeatedly returned `504 Gateway Timeout` and `model not supported by any provider` errors on multiple models. Switched to Groq for the core extraction step, which resolved reliability but required re-diagnosing an earlier bug (see below).
- **Missing line items on large invoices**: A 28-line-item invoice was consistently extracted with only 13-15 items. Initially suspected LLM context-window truncation — confirmed via token counting that the input was well within the model's context limit. Root cause turned out to be `max_new_tokens` being too low for the *output*, cutting the generation short before all items were written.
- **Hallucination on OCR failure**: An image with an unusual/corrupted format produced completely fabricated line item quantities. Root cause: OCR silently returned near-empty text, and the LLM filled in the required schema fields with guesses rather than failing loudly. Fixed with an explicit pre-LLM check on extracted text length — genuine OCR failures are now logged as failures, not passed through to the LLM.
- **File duplication in batch processing**: A file-format-conversion step was writing a converted copy back into the same folder being iterated over live, causing the batch loop to pick up and reprocess its own output as a "new" file. Fixed by snapshotting the file list before processing begins, and by performing format conversion entirely in-memory (no intermediate file ever written to disk).
- **VLM response parsing**: The vision model wrapped its JSON responses in markdown code fences, breaking naive `json.loads()` parsing. Fixed by switching to a Pydantic schema + `PydanticOutputParser` for the VLM response, matching the same robust pattern already used for the core invoice extraction.

---

## ⚠️ Known Limitations

- **Email attachment handling** is not yet implemented — deferred in favor of hardening PDF/image support first. The planned design checks whether an attachment is actually an invoice before running it through the full pipeline, rather than processing every attachment blindly.
- **VLM verification is flag-only, not auto-correcting.** When the VLM detects a mismatch, it's recorded for human review rather than automatically overwriting the extracted value — this was a deliberate choice to avoid compounding uncertainty from a second, smaller model without human oversight.
- **OCR accuracy varies with document design.** Invoices with small text over colored/patterned backgrounds are meaningfully harder to OCR accurately than plain black-text-on-white layouts. The validation and VLM layers exist specifically to catch these cases rather than assuming perfect OCR.

---

## 📄 License

MIT — see [LICENSE](LICENSE) for details.
