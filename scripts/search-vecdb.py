from pymilvus import connections, Collection
from sentence_transformers import SentenceTransformer

connections.connect(host="localhost", port="19530")

collection = Collection("doc_sections")
collection.load()

model = SentenceTransformer("all-MiniLM-L6-v2")


def search_similar_docs(query_text, top_k=5):
    query_vector = model.encode(query_text)

    if len(query_vector.shape) == 2:
        query_vector = query_vector[0]

    print(f"Dimension of query vector: {len(query_vector)}")
    print(f"Shape of query vector: {query_vector.shape}")
    print(f"Type of query vector: {type(query_vector)}")

    query_vector_list = query_vector.tolist()
    print(f"Length of converted vector: {len(query_vector_list)}")

    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}

    try:
        results = collection.search(
            data=[query_vector_list],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["file_name", "content"],
        )

        print(f"\nQuery: {query_text}")
        print(f"\nFound {len(results[0])} similar documents:")

        for i, hit in enumerate(results[0]):
            print(f"\nResult {i+1}:")
            print(f"File name: {hit.entity.get('file_name')}")
            print(f"Similarity score: {hit.score}")
            print(f"Content fragment: {hit.entity.get('content')[:200]}...")

    except Exception as e:
        print(f"Error during search: {str(e)}")
        raise


if __name__ == "__main__":
    test_queries = [
        "Aysnc Support in Moonbit",
    ]

    for query in test_queries:
        search_similar_docs(query)
        print("\n" + "=" * 80 + "\n")
