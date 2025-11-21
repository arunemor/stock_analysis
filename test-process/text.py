# import re
# from pathlib import Path
# from docling.document_converter import DocumentConverter
# from docling.document_converter import PdfFormatOption
# from docling.datamodel.pipeline_options import PdfPipelineOptions
# from docling.datamodel.base_models import InputFormat


# def clean_text_block(text: str) -> str:
#     # Remove extra blank lines
#     text = re.sub(r"\n{3,}", "\n\n", text)

#     # Convert headings ending with ":" → proper markdown headings
#     text = re.sub(r"(?m)^([A-Za-z0-9 /&()\-]+):\s*$", r"## \1", text)

#     # Ensure tables have space above them
#     text = re.sub(r"(?m)([^|\n])\n\|", r"\1\n\n|", text)

#     # Normalize spaces
#     text = re.sub(r"[ \t]+", " ", text)

#     return text.strip()


# def extract_clean_text(pdf_path: str, output_txt: str):
#     pipeline = PdfPipelineOptions()
#     pipeline.generate_picture_images = False  # text only

#     converter = DocumentConverter(
#         format_options={
#             InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)
#         }
#     )

#     # Convert PDF → Docling document
#     result = converter.convert(pdf_path)
#     doc = result.document

#     # ⭐ Correct method for your version ⭐
#     raw_text = doc.export_to_text()    # <--- FIXED HERE

#     clean_text = clean_text_block(raw_text)

#     Path(output_txt).write_text(clean_text, encoding="utf-8")

#     print(f"[SAVED] Extracted clean text → {output_txt}")
#     return clean_text


# if __name__ == "__main__":
#     extract_clean_text("sample1.pdf", "sample1_clean.txt")




# Embedding and storing the text under weaviatedb.

import weaviate
from sentence_transformers import SentenceTransformer
from weaviate.classes.config import Configure


# ------------------------ LOAD CHUNKS ------------------------
def load_chunks(file_path):
    chunks = []
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    raw_chunks = content.split("----- Paragraph")

    for ch in raw_chunks:
        if "-----" in ch:
            clean = ch.split("-----", 1)[1].strip()
            if clean:
                chunks.append(clean)

    return chunks


# ------------------------ EMBEDDING MODEL ------------------------
def embed_chunks(chunks):
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)
    return embeddings


# ------------------------ WEAVIATE STORE ------------------------
def store_in_weaviate(chunks, embeddings):

    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051
    )

    try:
        class_name = "ParagraphChunk"

        existing = client.collections.list_all()

        # Create collection if not exists
        if class_name not in existing:
            client.collections.create(
                name=class_name,
                properties=[
                    weaviate.classes.config.Property(
                        name="text",
                        data_type=weaviate.classes.config.DataType.TEXT
                    )
                ],
                vectorizer_config=Configure.Vectorizer.none()  # Correct way for v4
            )
            print("Created class:", class_name)
        else:
            print("Class already exists:", class_name)

        collection = client.collections.get(class_name)

        # Insert embeddings
        for idx, (text, vector) in enumerate(zip(chunks, embeddings), start=1):
            collection.data.insert(
                properties={"text": text},
                vector=vector.tolist()
            )
            print(f"Inserted chunk {idx}/{len(chunks)}")

        print("\n⭐ All embeddings stored in Weaviate!")

    finally:
        client.close()
        print("Weaviate client closed.")


# ------------------------ RUN ------------------------
if __name__ == "__main__":
    chunks_file = "sample1_clean.txt"
    chunks = load_chunks(chunks_file)

    print(f"Loaded {len(chunks)} chunks")

    embeddings = embed_chunks(chunks)
    print("Embeddings generated!")

    store_in_weaviate(chunks, embeddings)