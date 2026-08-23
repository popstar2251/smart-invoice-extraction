"""
Smart Invoice & Document Extraction Pipeline
==============================================
Extracts structured data from invoices (PDF, PNG, JPG, WEBP, JFIF),
validates it, cross-checks it with a VLM, and outputs two linked CSVs.

This is a script version of the pipeline for reference / local use.
The primary, actively-developed version lives in notebooks/invoice_extraction_pipeline.ipynb.

Required environment variables:
    GROQ_API_KEY   - from console.groq.com (free tier)
    HF_TOKEN       - from huggingface.co/settings/tokens (free tier, "Read" access)
"""

import os
import base64
from pathlib import Path
from io import BytesIO
import pandas as pd
from PIL import Image
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat, DocumentStream
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional
from huggingface_hub import InferenceClient


# ── Schemas ──────────────────────────────
class LineItem(BaseModel):
    quantity: int
    description: str
    unit_price: float
    total: float


class Invoice(BaseModel):
    invoice_number: str
    date: str = Field(description="Invoice date in YYYY-MM-DD format, converted from whatever format appears on the original document")
    vendor_name: str
    vendor_address: Optional[str] = None
    vendor_phone: Optional[str] = None
    bill_to_name: str
    bill_to_address: Optional[str] = None
    ship_to_name: Optional[str] = None
    ship_to_address: Optional[str] = None
    salesperson: Optional[str] = None
    po_number: Optional[str] = None
    terms: Optional[str] = None
    line_items: List[LineItem]
    subtotal: float
    tax: float
    shipping: Optional[float] = None
    total_due: float


class VLMVerificationResult(BaseModel):
    matches: bool = Field(description="True if the extracted data matches the image, False if there are mismatches")
    issues: List[str] = Field(default_factory=list, description="Specific mismatches found, empty list if none")


# ── Docling converter (page images enabled for the VLM step) ──────────────────────────────
_pipeline_options = PdfPipelineOptions()
_pipeline_options.generate_page_images = True
_pipeline_options.images_scale = 2.0

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options)}
)

# ── LLM extraction chain ──────────────────────────────
chat_model = ChatGroq(model="openai/gpt-oss-120b", max_tokens=4096, temperature=0)
parser = PydanticOutputParser(pydantic_object=Invoice)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at extracting structured data from invoice documents. "
                "Always output the date in YYYY-MM-DD format, converting from whatever format appears on the original document. "
                "Some line items may be split across a page break — if you see an orphaned product code or partial line with no "
                "matching description, merge it with the line item immediately before it. "
                "Extract all fields accurately from the markdown below.\n\n{format_instructions}"),
    ("human", "{invoice_markdown}")
])
chain = prompt | chat_model | parser

# ── VLM verification ──────────────────────────────
vlm_parser = PydanticOutputParser(pydantic_object=VLMVerificationResult)
VLM_MODEL_CANDIDATES = ["zai-org/GLM-4.5V", "Qwen/Qwen2.5-VL-3B-Instruct"]


def verify_with_vlm(image_b64: str, extracted_invoice: Invoice) -> Optional[VLMVerificationResult]:
    client = InferenceClient(api_key=os.environ["HF_TOKEN"])
    extracted_json = extracted_invoice.model_dump_json(indent=2)
    prompt_text = (
        f"Here is data extracted from this invoice image:\n\n{extracted_json}\n\n"
        f"Carefully compare it against the image and check for mismatches.\n\n"
        f"{vlm_parser.get_format_instructions()}"
    )
    for model_name in VLM_MODEL_CANDIDATES:
        try:
            completion = client.chat.completions.create(
                model=model_name,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                }],
                max_tokens=1024,
            )
            return vlm_parser.parse(completion.choices[0].message.content)
        except Exception:
            continue
    return None  # every candidate failed — treated as "Not checked"


# ── Validation ──────────────────────────────
def validate_invoice(invoice: Invoice, tolerance: float = 0.01) -> dict:
    errors, warnings = [], []

    calculated_subtotal = sum(item.total for item in invoice.line_items)
    if abs(calculated_subtotal - invoice.subtotal) > tolerance:
        errors.append(f"Line items sum to {calculated_subtotal}, but subtotal says {invoice.subtotal}")

    shipping = invoice.shipping or 0
    calculated_total = invoice.subtotal + invoice.tax + shipping
    if abs(calculated_total - invoice.total_due) > tolerance:
        errors.append(f"Subtotal+tax+shipping = {calculated_total}, but total_due says {invoice.total_due}")

    for i, item in enumerate(invoice.line_items):
        expected = item.quantity * item.unit_price
        if abs(expected - item.total) > tolerance:
            warnings.append(f"Line item {i} ('{item.description}'): {item.quantity} x {item.unit_price} = {expected}, but total says {item.total}")

    for field_name, value in {
        "invoice_number": invoice.invoice_number,
        "total_due": invoice.total_due,
        "vendor_name": invoice.vendor_name,
    }.items():
        if not value:
            errors.append(f"Missing required field: {field_name}")

    if invoice.total_due <= 0:
        errors.append("total_due is zero or negative — likely extraction failure")
    for i, item in enumerate(invoice.line_items):
        if item.quantity <= 0:
            warnings.append(f"Line item {i} has non-positive quantity: {item.quantity}")

    return {"is_valid": len(errors) == 0, "errors": errors, "warnings": warnings}


# ── Format handling ──────────────────────────────
def get_docling_input(filepath: Path):
    """.jfif files are converted to JPEG entirely in memory — no disk writes."""
    if filepath.suffix.lower() == ".jfif":
        img = Image.open(filepath).convert("RGB")
        buffer = BytesIO()
        img.save(buffer, format="JPEG")
        buffer.seek(0)
        return DocumentStream(name=filepath.stem + ".jpg", stream=buffer)
    return filepath


def image_to_base64(pil_image: Image.Image) -> str:
    buffer = BytesIO()
    pil_image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# ── Row builders ──────────────────────────────
def invoice_to_summary_row(invoice: Invoice, validation_result: dict, vlm_result: Optional[VLMVerificationResult]) -> dict:
    if vlm_result is None:
        vlm_status, vlm_issues_text = "Not checked", "None"
    elif vlm_result.matches:
        vlm_status, vlm_issues_text = "Yes", "None"
    else:
        vlm_status = "No"
        vlm_issues_text = "; ".join(vlm_result.issues) if vlm_result.issues else "None"

    return {
        "invoice_number": invoice.invoice_number,
        "date": invoice.date,
        "vendor_name": invoice.vendor_name,
        "bill_to_name": invoice.bill_to_name,
        "subtotal": invoice.subtotal,
        "tax": invoice.tax,
        "shipping": invoice.shipping if invoice.shipping is not None else 0,
        "total_due": invoice.total_due,
        "line_item_count": len(invoice.line_items),
        "status": "OK" if validation_result["is_valid"] else "NEEDS REVIEW",
        "errors": "; ".join(validation_result["errors"]) if validation_result["errors"] else "None",
        "warnings": "; ".join(validation_result["warnings"]) if validation_result["warnings"] else "None",
        "vlm_validated": vlm_status,
        "vlm_issues": vlm_issues_text,
    }


def invoice_to_line_item_rows(invoice: Invoice) -> list:
    return [
        {
            "invoice_number": invoice.invoice_number,
            "description": item.description,
            "quantity": item.quantity,
            "unit_price": item.unit_price,
            "total": item.total,
        }
        for item in invoice.line_items
    ]


# ── Batch pipeline ──────────────────────────────
def run_pipeline(invoice_folder: str, min_markdown_length: int = 50):
    folder = Path(invoice_folder)
    all_summary_rows, all_line_item_rows, failed_files = [], [], []

    files_to_process = [
        f for f in folder.iterdir()
        if f.suffix.lower() in [".pdf", ".jpg", ".jpeg", ".png", ".webp", ".jfif"]
    ]

    for filepath in files_to_process:
        print(f"Processing: {filepath.name}")
        try:
            docling_input = get_docling_input(filepath)
            docling_result = converter.convert(docling_input)
            markdown_text = docling_result.document.export_to_markdown()

            if len(markdown_text.strip()) < min_markdown_length:
                print(f"  FAILED: insufficient OCR text ({len(markdown_text.strip())} chars)")
                failed_files.append({"filename": filepath.name, "error": "OCR extraction produced insufficient text"})
                continue

            result = chain.invoke({
                "invoice_markdown": markdown_text,
                "format_instructions": parser.get_format_instructions()
            })
            validation_result = validate_invoice(result)

            vlm_result = None
            try:
                if filepath.suffix.lower() == ".pdf":
                    first_page = next(iter(docling_result.document.pages.values()))
                    page_image = first_page.image.pil_image
                else:
                    page_image = Image.open(filepath).convert("RGB")
                image_b64 = image_to_base64(page_image)
                vlm_result = verify_with_vlm(image_b64, result)
            except Exception as vlm_error:
                print(f"  NOTE: VLM verification unavailable: {vlm_error}")

            for e in validation_result["errors"]:
                print(f"  ERROR: {e}")
            for w in validation_result["warnings"]:
                print(f"  WARNING: {w}")

            summary_row = invoice_to_summary_row(result, validation_result, vlm_result)
            summary_row["source_file"] = filepath.name
            all_summary_rows.append(summary_row)
            all_line_item_rows.extend(invoice_to_line_item_rows(result))

        except Exception as e:
            print(f"  FAILED to process {filepath.name}: {e}")
            failed_files.append({"filename": filepath.name, "error": str(e)})

    pd.DataFrame(all_summary_rows).to_csv("invoices_summary.csv", index=False)
    pd.DataFrame(all_line_item_rows).to_csv("line_items_detail.csv", index=False)
    if failed_files:
        pd.DataFrame(failed_files).to_csv("failed_files.csv", index=False)

    print(f"\nDone. Processed {len(all_summary_rows)} invoices successfully, {len(failed_files)} failed.")


if __name__ == "__main__":
    run_pipeline(invoice_folder="sample_invoices")
