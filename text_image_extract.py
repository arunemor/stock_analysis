import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import DoclingDocument

def extract_text_and_images_to_file(pdf_path: str, output_dir: str, font_size: int = 24) -> Dict[str, Any]:
    """
    Extracts text (as Markdown) and images (graphs) from a PDF using Docling, along with image captions (headlines).
    Stores everything in a single JSON file: text content and base64-encoded images with metadata.
    
    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Directory to save the output JSON file.
        font_size: Font size for the caption text (default: 24).
    
    Returns:
        Dictionary with 'text' (Markdown string) and 'images' (list of dicts).
    """
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    base_name = Path(pdf_path).stem
    output_file = os.path.join(output_dir, f"{base_name}_extracted_data.json")
    
    # Configure pipeline options for PDF processing
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True  # Enable image extraction
    pipeline_options.images_scale = 2.0  # Higher resolution (scale=1 is 72 DPI)
    # Optional: Enable OCR if needed for scanned PDFs
    # pipeline_options.do_ocr = True
    
    # Initialize the document converter with PDF options
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    # Convert the PDF to a DoclingDocument
    result = converter.convert(pdf_path)
    doc: DoclingDocument = result.document
    
    # Extract full text as Markdown
    try:
        full_text = doc.to_markdown()
    except Exception as e:
        print(f"Warning: Could not extract Markdown text from {pdf_path}: {e}")
        full_text = ""
    
    # Collect images and captions
    images_data = []
    if hasattr(doc, "pictures") and doc.pictures:
        for i, pic in enumerate(doc.pictures, 1):
            # Get caption (this acts as the "headline" for the image)
            try:
                caption = pic.caption_text(doc)
            except Exception:
                caption = f"Graph {i}"  # Fallback if no caption
            
            # Get PIL image using the proper method
            pil_image = None
            try:
                pil_image = pic.get_image(doc)
                if pil_image is None:
                    print(f"Warning: No image found for graph {i} in {pdf_path}")
                    continue
            except Exception as e:
                print(f"Warning: Could not get image for graph {i} in {pdf_path}: {e}")
                continue
            
            # Create a new image with caption on top
            # Assume a white background strip for caption
            caption_height = font_size + 20  # Padding
            new_width = pil_image.width
            new_height = pil_image.height + caption_height
            new_img = Image.new('RGB', (new_width, new_height), color='white')
            
            # Draw caption
            draw = ImageDraw.Draw(new_img)
            try:
                # Try to use a default font; fallback if not available
                font = ImageFont.truetype("arial.ttf", font_size)  # Assumes Arial; use default if not
            except:
                font = ImageFont.load_default()
            
            # Wrap caption if too long
            words = caption.split()
            lines = []
            current_line = []
            for word in words:
                test_line = ' '.join(current_line + [word])
                if draw.textlength(test_line, font=font) < new_width - 40:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
            if current_line:
                lines.append(' '.join(current_line))
            
            # Draw lines of text
            y_offset = 10
            for line in lines:
                draw.text((20, y_offset), line, fill='black', font=font)
                y_offset += font_size + 5
            
            # Paste the original image below
            new_img.paste(pil_image, (0, caption_height))
            
            # Convert image to base64
            buffer = BytesIO()
            new_img.save(buffer, format='PNG')
            image_b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            print(f"Processed graph {i} with headline: '{caption[:50]}...' from {pdf_path}")
            
            images_data.append({
                "index": i,
                "caption": caption,
                "image_b64": image_b64,
                "page": pic.prov[0].page_no if pic.prov else None
            })
    
    # Combine data
    extracted_data = {
        "source_pdf": pdf_path,
        "text": full_text,
        "images": images_data
    }
    
    # Save to single JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=4, ensure_ascii=False)
    print(f"Saved extracted text and {len(images_data)} images from {pdf_path} to '{output_file}'")
    
    if not images_data:
        print(f"No images found in {pdf_path}.")
    if not full_text.strip():
        print(f"No text extracted from {pdf_path}.")
    
    return extracted_data

# Batch usage: Input folder path with multiple PDFs
if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing PDFs: ").strip()
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
    else:
        pdf_files = list(Path(folder_path).glob("*.pdf")) + list(Path(folder_path).glob("*.PDF"))
        if not pdf_files:
            print(f"No PDF files found in '{folder_path}'.")
        else:
            print(f"Found {len(pdf_files)} PDF files. Processing...")
            all_extracted = []
            for pdf_file in pdf_files:
                pdf_path = str(pdf_file)
                data = extract_text_and_images_to_file(pdf_path, folder_path)  # Save under input folder
                all_extracted.append(data)
            
            # Optional: Save a summary of all extractions to a master file under input folder
            master_file = os.path.join(folder_path, "all_extracted_summary.json")
            with open(master_file, 'w', encoding='utf-8') as f:
                json.dump(all_extracted, f, indent=4, ensure_ascii=False)
            print(f"Batch complete! Individual JSONs saved per PDF under '{folder_path}'. Master summary: '{master_file}'")
            
            # Print summary
            total_pdfs = len(all_extracted)
            total_images = sum(len(d['images']) for d in all_extracted)
            total_text_chars = sum(len(d['text']) for d in all_extracted)
            print(f"Processed {total_pdfs} PDFs: {total_text_chars} total text chars, {total_images} total images.")