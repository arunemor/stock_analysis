import weaviate
from sentence_transformers import SentenceTransformer
import json

# ============================================================
# 1. Connect to Weaviate Local
# ============================================================

client = weaviate.connect_to_local(
    host="localhost",
    port=8081,
    grpc_port=50051,
    skip_init_checks=False
)

# ============================================================
# 2. Embedding Model
# ============================================================

text_model = SentenceTransformer("all-mpnet-base-v2")

def embed_text(text: str):
    return text_model.encode(text).tolist()


# ============================================================
# 3. Semantic Search Function (Weaviate v4 style)
# ============================================================

def search_weaviate(query: str, top_k: int = 5):
    vector = embed_text(query)

    collection = client.collections.get("PDFContent")

    response = collection.query.near_vector(
        near_vector=vector,
        limit=top_k,
        return_metadata=["distance", "certainty"]
    )

    return response


# ============================================================
# 4. User Interactive Test
# ============================================================

def run_user_query():
    while True:
        query = input("HFCL vs Nifty: ")

        if query.lower() == "exit":
            break

        print("\nSearching Weaviate... Please wait.\n")

        results = search_weaviate(query)

        print("\n================== RAW RESULT ==================\n")
        print(results)

        print("\n================== MATCHED RESULTS ==================\n")

        if len(results.objects) == 0:
            print("❌ No records found.")
            continue

        for idx, obj in enumerate(results.objects):
            props = obj.properties

            print(f"Result {idx + 1}:")
            print(f"Type      : {props.get('type')}")
            print(f"Content   : {props.get('content')[:150]} ...")
            print(f"Metadata  : {props.get('metadata')}")
            print(f"Distance  : {obj.metadata.distance:.4f}")
            print("--------------------------------------------------")

        print("\n====================================================\n")


# ============================================================
# 5. Run Script
# ============================================================

if __name__ == "__main__":
    print("Weaviate Search Test Ready.")
    print("Example queries:")
    print(" - shareholding pattern")
    print(" - promoters 31.58")
    print(" - HFCL vs nifty")
    print(" - revenue details")
    print(" - image page 1\n")

    run_user_query()

    client.close()
