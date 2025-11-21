import fitz   # PyMuPDF
import cv2
import numpy as np
import os
import shutil
import re
import PyPDF2
from langchain_text_splitters import MarkdownHeaderTextSplitter


# ======================================================================
# 1️⃣ GRAPH + IMAGE EXTRACTION
# ======================================================================

def extract_graphs(pdf_path, output_folder="graphs_only", dpi=150):
    """
    Extract graph-like regions from the PDF.
    """

    # Reset folder
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    doc = fitz.open(pdf_path)
    print(f"[INFO] PDF Loaded: {len(doc)} pages")

    img_count = 0
    MIN_AREA = 40000
    MIN_WIDTH = 250
    MIN_HEIGHT = 180

    for page_no in range(len(doc)):
        page = doc[page_no]

        zoom = dpi / 72
        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.height, pix.width, 3
        )

        # Preprocess
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])  # top-bottom

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            if area < MIN_AREA or w < MIN_WIDTH or h < MIN_HEIGHT:
                continue

            graph = img[y:y+h, x:x+w]
            img_count += 1

            out_path = os.path.join(output_folder, f"graph_{page_no+1}_{img_count}.png")
            cv2.imwrite(out_path, graph)
            print(f"[GRAPH SAVED] {out_path}")

    print(f"[DONE] Total Graphs Extracted: {img_count}")
    return img_count



# ======================================================================
# 2️⃣ TEXT + MARKDOWN HEADINGS + CHUNKS EXTRACTION
# ======================================================================

def restore_newlines(text):
    keywords = [
        "CMP", "Rating", "Target Price", "Stock Info", "Shareholding",
        "Stock Performance", "Result update", "Key Highlights",
        "HFCL vs Nifty", "Outlook", "Valuation"
    ]

    for k in keywords:
        text = re.sub(rf"(?<!\n){k}\b", f"\n{k}", text)

    text = re.sub(r"(\d)([A-Za-z])", r"\1\n\2", text)

    return text


def convert_to_markdown(text):
    text = restore_newlines(text)

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    text = re.sub(r"(?m)^([A-Za-z0-9 /&()\-]+):$", r"## \1", text)

    forced = [
        "CMP", "Rating", "Target Price", "Stock Info",
        "Shareholding Pattern", "Stock Performance",
        "Key Highlights", "Result update", "HFCL vs Nifty",
    ]

    for h in forced:
        text = re.sub(rf"(?m)^{h}\b.*$", lambda m: "## " + m.group(0).strip(), text)

    # Title-case headings
    text = re.sub(
        r"(?m)^([A-Z][A-Za-z0-9 &,:()/\-]{4,})$",
        r"# \1",
    )

    return text


def extract_text_chunks(pdf_path, output_file="chunks.txt"):
    """
    Extract text from PDF → fix → convert to markdown → split to ordered chunks
    """
    full_text = ""

    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                full_text += t + "\n"

    markdown_text = convert_to_markdown(full_text)

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "Heading 1"),
            ("##", "Heading 2"),
            ("###", "Heading 3")
        ]
    )

    chunks = splitter.split_text(markdown_text)

    with open(output_file, "w", encoding="utf-8") as f:
        for i, c in enumerate(chunks, start=1):
            f.write(f"----- Heading Chunk {i} -----\n")
            f.write(c.page_content.strip() + "\n\n")

    print(f"[TEXT DONE] {len(chunks)} chunks saved → {output_file}")

    return len(chunks)



# ======================================================================
# 3️⃣ MERGED MAIN PIPELINE
# ======================================================================

def process_pdf(pdf_path):
    print("\n=========================")
    print(" EXTRACTING GRAPHS...")
    print("=========================")
    extract_graphs(pdf_path)

    print("\n=========================")
    print(" EXTRACTING TEXT...")
    print("=========================")
    extract_text_chunks(pdf_path)

    print("\n=========================")
    print(" ALL WORK COMPLETED")
    print("=========================\n")



# ======================================================================
# RUN
# ======================================================================

if __name__ == "__main__":
    process_pdf("sample1.pdf")
