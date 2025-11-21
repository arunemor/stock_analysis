import os
import json
from pathlib import Path
from typing import List, Dict, Any
from PIL import Image, ImageDraw, ImageFont

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_core.types.doc import DoclingDocument

def extract_graphs_to_single_file(pdf_path: str, output_dir: str = "graphs_only", font_size: int = 24) -> List[Dict[str, Any]]:
    """
    Extracts images (graphs) from a PDF using Docling, along with their captions (headlines).
    Saves each image as a separate PNG file in the output directory and the metadata to a JSON file.
    
    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Directory to save the extracted images and JSON metadata (default: "graphs_only").
        font_size: Font size for the caption text (default: 24).
    
    Returns:
        List of dictionaries containing graph info: index, caption, image_path, page.
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
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
    
    # Collect images and captions
    extracted = []
    pic_index = 0
    
    # Iterate through document items to find pictures
    # iterate_items() returns tuples of (item, level)
    for item, level in doc.iterate_items():
        item_label = item.label.value if hasattr(item, 'label') else None
        
        if item_label == "picture":
            pic_index += 1
            
            # Get caption from the picture's markdown content
            try:
                caption = item.export_to_markdown().strip()
                if not caption:
                    caption = f"Graph {pic_index}"
            except Exception:
                caption = f"Graph {pic_index}"
            
            # Get PIL image using the proper method for docling_core
            pil_image = None
            image_path = None
            try:
                # Access image from the picture item - try different methods
                if hasattr(item, 'get_image'):
                    pil_image = item.get_image(doc)
                elif hasattr(item, 'image') and item.image is not None:
                    pil_image = item.image
                
                if pil_image is None:
                    print(f"Warning: No image found for graph {pic_index}")
                    continue
            except Exception as e:
                print(f"Warning: Could not get image for graph {pic_index}: {e}")
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
            
            # Save individual image as PNG
            safe_caption = "".join(c for c in caption if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{pic_index:03d}_{safe_caption[:50]}.png"
            image_path = os.path.join(output_dir, filename)
            new_img.save(image_path)
            
            print(f"Saved graph {pic_index} as '{filename}' with headline: '{caption[:50]}...'")
            
            extracted.append({
                "index": pic_index,
                "caption": caption,
                "image_path": image_path,
                "page": item.prov[0].page_no if hasattr(item, 'prov') and item.prov else None
            })
    
    if extracted:
        # Save metadata to JSON file
        json_path = os.path.join(output_dir, "graphs_only.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(extracted, f, indent=4, ensure_ascii=False)
        print(f"Saved metadata for {len(extracted)} graphs to '{json_path}'")
    else:
        print("No graphs found in the PDF.")
    
    return extracted

# Direct usage with hardcoded PDF file
if __name__ == "__main__":
    pdf_file = "sample1.pdf"  # Hardcoded to your sample PDF
    if not os.path.exists(pdf_file):
        print(f"Error: PDF file '{pdf_file}' not found. Make sure it's in the current directory.")
    else:
        graphs_info = extract_graphs_to_single_file(pdf_file)
        print(f"Extracted {len(graphs_info)} graphs with headlines into 'graphs_only'.")
        for info in graphs_info:
            print(f"- {info['caption'][:100]}... (from page {info['page']})")