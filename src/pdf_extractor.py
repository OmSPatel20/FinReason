"""
pdf_extractor.py — Extract text and tables from any financial PDF.

Handles:
  - Text-based PDFs (direct extraction)
  - Scanned PDFs (OCR via Tesseract)
  - Mixed PDFs (tries text first, falls back to OCR)

Works with reports in any currency / any country.
"""
import os
import re
import sys
from typing import List, Dict, Optional

# =====================================================================
# PDF TEXT EXTRACTION
# =====================================================================

def extract_text_from_pdf(pdf_path: str, pages: Optional[List[int]] = None) -> List[Dict]:
    """
    Extract text from a PDF, page by page.
    Tries direct text extraction first. If empty, falls back to OCR.
    
    Returns: list of {"page": int, "text": str, "method": "direct"|"ocr"}
    """
    results = []
    
    # --- Try direct text extraction first (fast, works for digital PDFs) ---
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            target_pages = pages if pages else list(range(total_pages))
            
            for pg_num in target_pages:
                if pg_num >= total_pages:
                    continue
                page = pdf.pages[pg_num]
                text = page.extract_text() or ""
                
                if len(text.strip()) > 50:  # Got meaningful text
                    results.append({
                        "page": pg_num + 1,
                        "text": text.strip(),
                        "method": "direct",
                    })
                else:
                    # Page has no extractable text — need OCR
                    results.append({
                        "page": pg_num + 1,
                        "text": None,  # Will be filled by OCR
                        "method": "needs_ocr",
                    })
    except ImportError:
        print("pdfplumber not installed. Using OCR for all pages.")
        import subprocess
        result = subprocess.run(["pdfinfo", pdf_path], capture_output=True, text=True)
        total_pages = 1
        for line in result.stdout.split("\n"):
            if line.startswith("Pages:"):
                total_pages = int(line.split(":")[1].strip())
        target_pages = pages if pages else list(range(total_pages))
        for pg_num in target_pages:
            results.append({"page": pg_num + 1, "text": None, "method": "needs_ocr"})
    
    # --- OCR pages that need it ---
    pages_needing_ocr = [r["page"] for r in results if r["method"] == "needs_ocr"]
    
    if pages_needing_ocr:
        print(f"  OCR needed for {len(pages_needing_ocr)} pages...")
        ocr_texts = _ocr_pages(pdf_path, pages_needing_ocr)
        
        for r in results:
            if r["method"] == "needs_ocr" and r["page"] in ocr_texts:
                r["text"] = ocr_texts[r["page"]]
                r["method"] = "ocr"
    
    # Filter out pages with no text
    results = [r for r in results if r["text"] and len(r["text"].strip()) > 20]
    
    return results


def _ocr_pages(pdf_path: str, page_numbers: List[int]) -> Dict[int, str]:
    """OCR specific pages using Tesseract."""
    texts = {}
    
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except ImportError:
        print("  Install: pip install pdf2image pytesseract")
        print("  Also install Tesseract: sudo apt install tesseract-ocr")
        return texts
    
    for pg in page_numbers:
        try:
            images = convert_from_path(pdf_path, first_page=pg, last_page=pg, dpi=200)
            if images:
                text = pytesseract.image_to_string(images[0])
                if text.strip():
                    texts[pg] = text.strip()
        except Exception as e:
            print(f"  OCR failed for page {pg}: {e}")
    
    return texts


# =====================================================================
# TABLE DETECTION — find pages with financial tables
# =====================================================================

def find_table_pages(page_results: List[Dict]) -> List[Dict]:
    """
    Identify pages that contain financial tables (numbers in columns).
    Returns the subset of pages most likely to have useful financial data.
    """
    table_pages = []
    
    # Patterns that indicate financial table content
    number_pattern = re.compile(r'[\d,]+\.?\d*')
    currency_pattern = re.compile(
        r'(?:RM|USD|\$|€|£|¥|₹|SGD|HKD|AUD|CNY|JPY|INR|GBP|EUR)'
        r"|(?:\'000|million|billion|thousand)",
        re.IGNORECASE
    )
    
    for page in page_results:
        text = page["text"]
        lines = text.split("\n")
        
        # Count lines with multiple numbers (table rows)
        numeric_lines = 0
        for line in lines:
            numbers = number_pattern.findall(line)
            # A table row typically has 2+ numbers
            if len(numbers) >= 2:
                numeric_lines += 1
        
        has_currency = bool(currency_pattern.search(text))
        
        # If >3 lines with multiple numbers AND currency indicators → table page
        if numeric_lines >= 3 and has_currency:
            page["numeric_lines"] = numeric_lines
            page["has_currency"] = True
            table_pages.append(page)
    
    return table_pages


# =====================================================================
# CURRENCY DETECTION — auto-detect what currency the report uses
# =====================================================================

CURRENCY_MAP = {
    "RM": "Malaysian Ringgit (RM)",
    "USD": "US Dollar (USD)",
    "$": "US Dollar (USD)",
    "€": "Euro (EUR)",
    "EUR": "Euro (EUR)",
    "£": "British Pound (GBP)",
    "GBP": "British Pound (GBP)",
    "¥": "Japanese Yen (JPY) or Chinese Yuan (CNY)",
    "JPY": "Japanese Yen (JPY)",
    "CNY": "Chinese Yuan (CNY)",
    "₹": "Indian Rupee (INR)",
    "INR": "Indian Rupee (INR)",
    "SGD": "Singapore Dollar (SGD)",
    "HKD": "Hong Kong Dollar (HKD)",
    "AUD": "Australian Dollar (AUD)",
    "CHF": "Swiss Franc (CHF)",
    "KRW": "South Korean Won (KRW)",
    "BRL": "Brazilian Real (BRL)",
    "CAD": "Canadian Dollar (CAD)",
    "ZAR": "South African Rand (ZAR)",
    "THB": "Thai Baht (THB)",
    "IDR": "Indonesian Rupiah (IDR)",
    "PHP": "Philippine Peso (PHP)",
    "TWD": "Taiwan Dollar (TWD)",
    "SEK": "Swedish Krona (SEK)",
    "NOK": "Norwegian Krone (NOK)",
    "DKK": "Danish Krone (DKK)",
    "NZD": "New Zealand Dollar (NZD)",
    "MXN": "Mexican Peso (MXN)",
    "AED": "UAE Dirham (AED)",
    "SAR": "Saudi Riyal (SAR)",
}

# Scale keywords in multiple languages
SCALE_KEYWORDS = {
    "'000": 1000,
    "rm'000": 1000,
    "usd'000": 1000,
    "in thousands": 1000,
    "in millions": 1_000_000,
    "in billions": 1_000_000_000,
    "in rm'000": 1000,
    "in usd millions": 1_000_000,
    "in eur millions": 1_000_000,
    "(in thousands)": 1000,
    "(in millions)": 1_000_000,
}


def detect_currency(text: str) -> Dict:
    """
    Auto-detect currency and scale from financial report text.
    Returns {"currency": str, "scale": str, "scale_factor": int}
    """
    text_lower = text.lower()
    
    # Detect scale
    scale = "units"
    scale_factor = 1
    for keyword, factor in SCALE_KEYWORDS.items():
        if keyword in text_lower:
            scale = keyword
            scale_factor = factor
            break
    
    # Detect currency (check most specific first)
    detected_currency = "Unknown"
    # Check multi-char symbols first
    for symbol in ["RM", "SGD", "HKD", "AUD", "USD", "EUR", "GBP", 
                    "JPY", "CNY", "INR", "CHF", "KRW", "BRL", "CAD",
                    "ZAR", "THB", "IDR", "PHP", "TWD", "AED", "SAR"]:
        if symbol in text:
            detected_currency = CURRENCY_MAP.get(symbol, symbol)
            break
    
    # Check single-char symbols
    if detected_currency == "Unknown":
        for symbol in ["₹", "€", "£", "¥", "$"]:
            if symbol in text:
                detected_currency = CURRENCY_MAP.get(symbol, symbol)
                break
    
    return {
        "currency": detected_currency,
        "scale": scale,
        "scale_factor": scale_factor,
    }


# =====================================================================
# FORMAT FOR MODEL — prepare extracted text for the LLM
# =====================================================================

def format_page_for_model(page_text: str, max_chars: int = 1200) -> str:
    """
    Clean and format extracted page text for model input.
    Handles OCR artifacts, normalizes whitespace, preserves table structure.
    """
    # Clean common OCR artifacts
    text = page_text
    text = re.sub(r'\x0c', '', text)              # Form feeds
    text = re.sub(r'[|]{2,}', ' | ', text)        # Multiple pipes
    text = re.sub(r'[ \t]{3,}', '  |  ', text)    # Large gaps → pipe separators
    text = re.sub(r'\n{3,}', '\n\n', text)         # Excessive newlines
    text = re.sub(r'[^\S\n]+', ' ', text)          # Normalize spaces (keep newlines)
    
    # Truncate if needed
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[...truncated...]"
    
    return text.strip()


def extract_report_for_qa(pdf_path: str, max_pages: int = 20) -> Dict:
    """
    Full pipeline: PDF → extracted text → table pages → formatted for QA.
    
    Returns a dict with:
      - pages: list of extracted page dicts
      - table_pages: pages with financial tables
      - currency: detected currency info
      - ready_for_model: list of formatted text chunks ready for the LLM
    """
    print(f"Processing: {os.path.basename(pdf_path)}")
    
    # Get total page count
    try:
        import pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            total = len(pdf.pages)
    except Exception:
        import subprocess
        result = subprocess.run(["pdfinfo", pdf_path], capture_output=True, text=True)
        total = 1
        for line in result.stdout.split("\n"):
            if line.startswith("Pages:"):
                total = int(line.split(":")[1].strip())
    
    print(f"  Total pages: {total}")
    
    # Extract text from all pages (or first max_pages)
    target = list(range(min(total, max_pages)))
    pages = extract_text_from_pdf(pdf_path, target)
    print(f"  Extracted text from {len(pages)} pages")
    
    # Find table pages
    table_pages = find_table_pages(pages)
    print(f"  Found {len(table_pages)} pages with financial tables")
    
    # Detect currency from all text
    all_text = "\n".join(p["text"] for p in pages[:10])  # First 10 pages
    currency = detect_currency(all_text)
    print(f"  Currency: {currency['currency']} (scale: {currency['scale']})")
    
    # Format for model
    ready = []
    for page in table_pages:
        formatted = format_page_for_model(page["text"])
        ready.append({
            "page": page["page"],
            "text": formatted,
            "method": page["method"],
        })
    
    return {
        "filename": os.path.basename(pdf_path),
        "total_pages": total,
        "extracted_pages": len(pages),
        "table_pages": len(table_pages),
        "currency": currency,
        "pages": pages,
        "ready_for_model": ready,
    }


# =====================================================================
# CLI — Run directly to test on a PDF
# =====================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pdf_extractor.py <path_to_pdf>")
        print("Example: python pdf_extractor.py data/4Q2024-financial-report.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}")
        sys.exit(1)
    
    result = extract_report_for_qa(pdf_path)
    
    print(f"\n{'='*55}")
    print(f"  EXTRACTION SUMMARY")
    print(f"{'='*55}")
    print(f"  File:         {result['filename']}")
    print(f"  Total pages:  {result['total_pages']}")
    print(f"  Extracted:    {result['extracted_pages']}")
    print(f"  Table pages:  {result['table_pages']}")
    print(f"  Currency:     {result['currency']['currency']}")
    print(f"  Scale:        {result['currency']['scale']}")
    
    # Show first table page
    if result["ready_for_model"]:
        first = result["ready_for_model"][0]
        print(f"\n  --- Page {first['page']} (first table page) ---")
        print(f"  {first['text'][:500]}...")
    
    print(f"\n  Ready for model: {len(result['ready_for_model'])} chunks")
    print(f"{'='*55}")
