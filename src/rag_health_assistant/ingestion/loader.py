"""
Document Loader for Diabetes and Hypertension Medical Guidelines & Patient Education Literature.
"""

from pathlib import Path
from typing import List, Dict, Any
from langchain_core.documents import Document
from ..config import RAW_DOCS_DIR


# Metadata mapping based on known guideline filenames
DOC_METADATA_MAP = {
    "2024 European Society of Hypertension clinical practice guidelines for the management of.pdf": {
        "title": "2024 ESH Clinical Practice Guidelines for Arterial Hypertension Management",
        "organization": "European Society of Hypertension (ESH)",
        "condition": "hypertension",
        "doc_type": "clinical_guideline",
        "target_audience": "Healthcare professionals & informed patients"
    },
    "GLOBAL REPORT ON DIABETES.pdf": {
        "title": "WHO Global Report on Diabetes",
        "organization": "World Health Organization (WHO)",
        "condition": "diabetes",
        "doc_type": "clinical_guideline",
        "target_audience": "Global health systems & clinicians"
    },
    "Guideline for the pharmacological treatment of hypertension in adults.pdf": {
        "title": "WHO Guideline for Pharmacological Treatment of Hypertension in Adults",
        "organization": "World Health Organization (WHO)",
        "condition": "hypertension",
        "doc_type": "clinical_guideline",
        "target_audience": "Clinicians & primary care providers"
    },
    "MANAGEMENT OF BLOOD PRESSURE.pdf": {
        "title": "Clinical Management of Blood Pressure",
        "organization": "Cardiovascular & Hypertension Working Group",
        "condition": "hypertension",
        "doc_type": "clinical_guideline",
        "target_audience": "Cardiovascular care & patients"
    },
    "Steps to Manage Your Diabetes for Life.pdf": {
        "title": "4 Steps to Manage Your Diabetes for Life",
        "organization": "Centers for Disease Control and Prevention (CDC) / NIDDK",
        "condition": "diabetes",
        "doc_type": "patient_education",
        "target_audience": "Patients with Type 1 & Type 2 Diabetes"
    }
}


def extract_text_from_pdf(pdf_path: Path) -> List[Document]:
    """
    Extracts text page by page from a PDF using PyMuPDF (fitz) or PyPDF fallback.
    """
    documents: List[Document] = []
    meta = DOC_METADATA_MAP.get(pdf_path.name, {
        "title": pdf_path.stem.replace("_", " ").title(),
        "organization": "Clinical Source",
        "condition": "both" if "diabetes" in pdf_path.name.lower() and "hypertension" in pdf_path.name.lower()
                     else "diabetes" if "diabetes" in pdf_path.name.lower()
                     else "hypertension" if ("hypertension" in pdf_path.name.lower() or "blood pressure" in pdf_path.name.lower())
                     else "general_health",
        "doc_type": "guideline",
        "target_audience": "Patients and Clinicians"
    })

    # Try PyMuPDF first for speed and layout quality
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        total_pages = len(doc)
        for page_idx in range(total_pages):
            page = doc[page_idx]
            text = page.get_text("text").strip()
            if text:
                # Basic cleaning of non-printable or null characters
                cleaned_text = text.replace("\x00", "").replace("\r\n", "\n")
                doc_obj = Document(
                    page_content=cleaned_text,
                    metadata={
                        "source": pdf_path.name,
                        "file_path": str(pdf_path),
                        "page": page_idx + 1,
                        "total_pages": total_pages,
                        **meta
                    }
                )
                documents.append(doc_obj)
        doc.close()
        return documents
    except ImportError:
        pass
    except Exception as e:
        print(f"PyMuPDF failed on {pdf_path.name}: {e}. Falling back to pypdf...")

    # PyPDF Fallback
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        total_pages = len(reader.pages)
        for page_idx, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                cleaned_text = text.replace("\x00", "").replace("\r\n", "\n")
                documents.append(
                    Document(
                        page_content=cleaned_text,
                        metadata={
                            "source": pdf_path.name,
                            "file_path": str(pdf_path),
                            "page": page_idx + 1,
                            "total_pages": total_pages,
                            **meta
                        }
                    )
                )
    except Exception as e:
        print(f"Error loading {pdf_path.name} with PyPDF: {e}")

    return documents


def load_raw_documents(docs_dir: Path = RAW_DOCS_DIR) -> List[Document]:
    """
    Loads all PDF documents found in the raw documents directory.
    """
    all_pages: List[Document] = []
    if not docs_dir.exists():
        print(f"Directory '{docs_dir}' does not exist.")
        return all_pages

    pdf_files = list(docs_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF documents in '{docs_dir}'. Loading...")

    for pdf_path in pdf_files:
        try:
            pages = extract_text_from_pdf(pdf_path)
            print(f"Loaded '{pdf_path.name}': {len(pages)} pages extracted.")
            all_pages.extend(pages)
        except Exception as e:
            print(f"Failed to load '{pdf_path.name}': {e}")

    return all_pages
