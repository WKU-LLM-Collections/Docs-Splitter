import os
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility,
)
from sentence_transformers import SentenceTransformer
import glob

connections.connect(host="localhost", port="19530")

COLLECTION_NAME = "doc_sections"

if utility.has_collection(COLLECTION_NAME):
    utility.drop_collection(COLLECTION_NAME)
    print(f"Dropped old collection: {COLLECTION_NAME}")

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=200),
    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
]

schema = CollectionSchema(fields=fields, description="Document sections collection")
collection = Collection(name=COLLECTION_NAME, schema=schema)

index_params = {
    "metric_type": "L2",
    "index_type": "IVF_FLAT",
    "params": {"nlist": 1024},
}
collection.create_index(field_name="embedding", index_params=index_params)

model = SentenceTransformer("all-MiniLM-L6-v2")


def process_files():
    markdown_files = glob.glob("final_sections/*.md")

    file_names = []
    contents = []
    embeddings = []

    for file_path in markdown_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                embedding = model.encode(content)

                if len(embedding.shape) == 2:
                    embedding = embedding[0]

                file_names.append(os.path.basename(file_path))
                contents.append(content)
                embeddings.append(embedding.tolist())

                print(f"Processing file: {os.path.basename(file_path)}")

        except Exception as e:
            print(f"Error processing file {file_path}: {str(e)}")
            continue

    print(f"\nData statistics:")
    print(f"Number of files: {len(file_names)}")
    print(f"Number of contents: {len(contents)}")
    print(f"Number of vectors: {len(embeddings)}")
    print(f"Vector dimension: {len(embeddings[0]) if embeddings else 0}")

    try:
        entities = [file_names, contents, embeddings]

        collection.insert(entities)
        print(f"\nSuccessfully inserted {len(file_names)} documents into Milvus")

    except Exception as e:
        print(f"Error inserting data: {str(e)}")
        raise


if __name__ == "__main__":
    process_files()
    collection.flush()
    print("Vector database created successfully!")
