import os
import time
import base64
import re
from io import BytesIO
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import fitz  # PyMuPDF
from PIL import Image

from mistralai.client import Mistral
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

MODEL_ID = "pixtral-12b-2409"

def rate_limit_sleep():
    # Mistral Free Tier allows 1 request per second
    time.sleep(1.5)

def pil_to_base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{img_str}"

def clean_markdown(text: str) -> str:
    match = re.search(r'```(?:markdown)?(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=20)
)
def extract_text(client: Mistral, b64_img: str) -> str:
    prompt = """
You are a document reconstruction engine.

Your job is NOT OCR.

Your job is to reconstruct the page as a clean, structured document suitable for another LLM to learn from.

Analyze the entire page visually and preserve the document's meaning and hierarchy.

Requirements:

1. Extract ALL visible text.
2. Do NOT summarize.
3. Do NOT omit any content.
4. Do NOT paraphrase.
5. Ignore watermarks, timestamps, confidentiality banners, email overlays, and system-generated page stamps.
6. Preserve reading order.
7. Preserve visual hierarchy.

Formatting Rules:

* Main page titles → "#"
* Section headings → "##"
* Subsections → "###"
* Bullet lists → Markdown bullets
* Checklists → "- [ ]"
* Tables → Markdown tables
* Warnings / callout boxes → Separate sections
* Examples → Separate sections
* GOOD vs BAD comparisons → Separate sections

Very Important:

Many pages contain side-by-side columns.

Do NOT merge columns together.

If the page contains sections such as:

* WHAT GOES WRONG
* HOW TO FIX IT

extract the entire WHAT GOES WRONG section first, then the entire HOW TO FIX IT section.

Never interleave text from multiple columns.

For visual layouts:

Convert the visual structure into logical document structure.

Example:

WHAT GOES WRONG

[text]

HOW TO FIX IT

[text]

Output Requirements:

Return ONLY markdown.

No explanations.
No commentary.
No summaries.
No OCR confidence scores.

The markdown should be clean enough that another LLM could learn from it without seeing the original page.
"""
    response = client.chat.complete(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": b64_img}
                ]
            }
        ]
    )
    rate_limit_sleep()
    text = response.choices[0].message.content.strip()
    return clean_markdown(text)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=20)
)
def validate_text(client: Mistral, b64_img: str, previous_text: str) -> str:
    prompt = f"""
You are validating a document reconstruction.

You are given:

1. The original page image.
2. The reconstructed markdown.

Verify:

* No missing headings
* No missing bullets
* No missing checklist items
* No missing table rows
* No merged columns
* No missing examples
* No missing callout boxes

If any content is missing or incorrectly ordered, produce a corrected version.

Return only the corrected markdown.

Reconstructed markdown:
{previous_text}
"""
    response = client.chat.complete(
        model=MODEL_ID,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": b64_img}
                ]
            }
        ]
    )
    rate_limit_sleep()
    text = response.choices[0].message.content.strip()
    return clean_markdown(text)

def process_pdf(pdf_path: Path, md_dir: Path, client: Mistral):
    print(f"\nProcessing {pdf_path.name}...")
    doc = fitz.open(pdf_path)
    
    all_markdown = []
    
    for page_num in tqdm(range(len(doc)), desc="Pages"):
        page = doc.load_page(page_num)
        
        # Render at 300 DPI
        zoom = 300 / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        
        # Convert to PIL Image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        b64_img = pil_to_base64(img)
        
        try:
            md_text = extract_text(client, b64_img)
            md_text = validate_text(client, b64_img, md_text)
            all_markdown.append(f"<!-- Page {page_num + 1} -->\n\n{md_text}\n\n---\n")
        except Exception as e:
            print(f"Error processing page {page_num + 1}: {e}")
            all_markdown.append(f"<!-- Error processing Page {page_num + 1} -->\n\n")

    md_file_path = md_dir / f"{pdf_path.stem}.md"
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(all_markdown))
        
    print(f"Saved {md_file_path.name} to md/ directory.")

def main():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("Error: MISTRAL_API_KEY not found in .env file.")
        return

    client = Mistral(api_key=api_key)
    
    base_dir = Path(__file__).parent
    pdfs_dir = base_dir / "pdfs"
    md_dir = base_dir / "md"
    
    md_dir.mkdir(exist_ok=True)
    
    if not pdfs_dir.exists():
        print(f"Error: {pdfs_dir} directory not found.")
        return
        
    pdf_files = list(pdfs_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {pdfs_dir}.")
        return
        
    print(f"Found {len(pdf_files)} PDFs to process. Starting...")
    
    for pdf_file in pdf_files:
        process_pdf(pdf_file, md_dir, client)
        
    print("\nAll processing complete!")

if __name__ == "__main__":
    main()
