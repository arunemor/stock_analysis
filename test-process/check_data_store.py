import weaviate
from sentence_transformers import SentenceTransformer


def search_weaviate(query_text):
    """
    Search Weaviate database with a text query
    """
    # Connect to Weaviate
    client = weaviate.connect_to_local(
        host="localhost",
        port=8080,
        grpc_port=50051
    )

    try:
        # Load embedding model
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Convert query to vector
        query_vector = model.encode(query_text)
        
        # Get collection
        collection = client.collections.get("ParagraphChunk")
        
        # Search for similar chunks
        response = collection.query.near_vector(
            near_vector=query_vector.tolist(),
            limit=3  # Return top 3 results
        )
        
        # Display results
        print(f"\n🔍 Search Query: '{query_text}'")
        print(f"📊 Found {len(response.objects)} results\n")
        print("="*80)
        
        for idx, obj in enumerate(response.objects, start=1):
            print(f"\n✅ Result {idx}:")
            print(f"{obj.properties['text']}")
            print("-"*80)
            
    finally:
        client.close()


# Run the search
if __name__ == "__main__":
    # Replace with your actual search query
    search_weaviate("Key Highlights")