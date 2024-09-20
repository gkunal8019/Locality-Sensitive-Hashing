# Locality-Sensitive Hashing (LSH):

---

# LSH-based Document Search and Query Processing with Llama Index and SentenceTransformer

This repository implements a Locality Sensitive Hashing (LSH)-based search mechanism for document embeddings using `lshashpy3`. It allows users to query documents and retrieve relevant information, leveraging the `Ollama Llama` large language model and `SentenceTransformer` for embedding generation. This solution also includes querying via a `RouterQueryEngine` to integrate LSH and traditional query methods.

## Table of Contents

- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Code Explanation](#code-explanation)
  - [LSH Search Function](#lsh-search-function)
  - [Query Engine Creation](#query-engine-creation)
  - [Processing Queries](#processing-queries)
  - [Generating Responses](#generating-responses)
- [Mathematical Framework](#mathematical-framework)
- [Example Usage](#example-usage)
- [Future Work](#future-work)

## Requirements

Ensure the following Python libraries are installed:

```bash
pip install lshashpy3 llama_index sentence-transformers transformers ollama
```

## Setup Instructions

1. Clone the repository and navigate to the project directory:

   ```bash
   git clone https://github.com/loretoparisi/lshash.git
   ```

2. Install all necessary libraries (if not already installed):

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your environment has access to a CUDA-enabled GPU for efficient model inference (required for large models like `Ollama` and `SentenceTransformer`).

4. Place your documents in the designated directory (`/env/pdf_rag/integration_class_lsh/data`), as the `SimpleDirectoryReader` will load data from this path.

## Code Explanation

### 1. **LSH-based Search Function**

The core functionality of this project lies in using LSH (Locality Sensitive Hashing) to index and search document embeddings. The following code creates an LSH index, performs a query, and returns the closest matching documents based on the query embedding:

```python
from lshashpy3 import LSHash

# Initialize LSH
hash_size = 2
input_dim = 1024
num_hashtables = 1

lsh = LSHash(hash_size=hash_size, input_dim=input_dim, num_hashtables=num_hashtables,
             storage_config={'dict': None},
             matrices_filename='weights.npz',
             hashtable_filename='hash.npz',
             overwrite=True)

# Index document embeddings
for i, embedding in enumerate(doc_embeddings):
    lsh.index(embedding.tolist(), extra_data=i)

# Save LSH index
lsh.save()

# Query the LSH index with a query embedding
query_embedding = model.encode([query])[0]
top_n = 3
nn = lsh.query(query_embedding.tolist(), num_results=top_n, distance_func="euclidean")
```

### 2. **Query Engine Creation**

The project employs a `RouterQueryEngine` that uses both traditional indexing methods (`VectorStoreIndex`) and LSH-based document retrieval:

```python
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine

def create_query_engines(query, model, doc_texts, input_dim, weights_path, hash_path, query_engine_all):
    # Perform LSH search
    query_result = search_lsh(query, model, doc_texts, input_dim, weights_path, hash_path)

    # Create index from the LSH query result
    doc = Document(text=query_result)
    index = VectorStoreIndex.from_documents([doc], transformations=[text_splitter])

    # Initialize query engine for the LSH result
    query_engine_table = index.as_query_engine(similarity_top_k=3)

    # Set up tools for the router query engine
    query_engine_tools = [
        QueryEngineTool(query_engine=query_engine_all, metadata=ToolMetadata(name="contize data")),
        QueryEngineTool(query_engine=query_engine_table, metadata=ToolMetadata(name="all data")),
    ]

    # Create and return the router query engine
    router_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=query_engine_tools,
        verbose=True
    )

    return router_engine, query_engine_table
```

### 3. **Processing Queries**

The `process_query` function ties everything together by taking a query, generating embeddings, performing the LSH search, and then querying the appropriate engine:

```python
def process_query(query, model, doc_texts, input_dim, weights_path, hash_path, query_engine_all):
    router_engine, query_engine_table = create_query_engines(query, model, doc_texts, input_dim, weights_path, hash_path, query_engine_all)
    
    result = router_engine.query(query)
    
    return result, query_engine_table
```

### 4. **Generating Responses**

To integrate LSH with language model responses, we utilize the `Ollama Llama` model for generating responses based on the retrieved documents:

```python
from llama_index.core.llms import ChatMessage

def get_response_from_lsh(query, lsh, doc_texts, model, top_n=3):
    # Query the LSH index
    query_embedding = model.encode([query])[0]
    nn = lsh.query(query_embedding.tolist(), num_results=top_n, distance_func="euclidean")

    # Fetch the relevant document texts
    data = ""
    for result in nn:
        vec, doc_index = result[0]
        if doc_index is not None and doc_index < len(doc_texts):
            data += doc_texts[doc_index] + "\n"

    # Construct a language model query
    messages = [
        ChatMessage(role="system", content=f"Your task is to answer from the document: {data}"),
        ChatMessage(role="user", content=f"Give me the answer from the document. Question: {query}")
    ]

    resp = llm.chat(messages)
    return resp
```

## Mathematical Framework

### 1. **Distance Metrics**

#### a. **Euclidean Distance**
\[
d(x, y) = \|x - y\|_2
\]

#### b. **Cosine Similarity**
\[
d(x, y) = 1 - \frac{x \cdot y}{\|x\|\|y\|}
\]

### 2. **Hash Function in LSH**
For two points \(x\) and \(y\), the probability of collision under a hash function \(h\) is given by:
\[
P[h(x) = h(y)] = f(D(x, y))
\]
where \(D(x, y)\) is the distance metric (e.g., Euclidean or cosine), and \(f\) is a monotonically decreasing function.

### 3. **LSH for Euclidean Distance**
The hash function using random projections is defined as:
\[
h_{a,b}(x) = \left\lfloor \frac{a \cdot x + b}{w} \right\rfloor
\]
- \(a \sim \mathcal{N}(0, 1)\) (Gaussian distribution)
- \(b \sim U(0, w)\) (Uniform distribution)
- Probability of collision:
\[
P[h_{a,b}(x) = h_{a,b}(y)] = 1 - \frac{d(x, y)}{w}
\]

### 4. **LSH for Cosine Similarity**
The hash function using random hyperplanes is defined as:
\[
h_a(x) = \text{sign}(a \cdot x)
\]
- Probability of collision:
\[
P[h_a(x) = h_a(y)] = 1 - \frac{\theta(x, y)}{\pi}
\]
where \(\theta(x, y)\) is the angle between vectors \(x\) and \(y\).

### 5. **Multiple Hash Tables**
If using \(k\) hash functions in a single table and \(L\) tables, the overall probability of collision increases, enhancing the likelihood of finding similar items. 

- Each hash table offers a different projection of the data, improving accuracy while increasing memory and computation requirements.

## Example Usage

### Step 1: Running a query

You can call `process_query` to perform LSH-based retrieval and generate a response from the retrieved documents.

```python
query = "What is the Hillebrandt (1974), and Stone (1975)?"
result, query_engine_table = process_query(query, model, doc_texts, 1024, 'weights.npz', 'hash.npz', query_engine_all)
print(result.response)
```

### Step 2: Obtaining a language model response

```python
response = get_response_from_lsh(query, lsh, doc_texts, model)
print(response)
```

## Future Work

- Add more advanced query routing techniques.
- Implement more comprehensive LSH parameter tuning.
- Explore scaling the number of documents and embeddings.
- Integrate real-time document loading and query updates.

---

Feel free to modify any sections or add more details as needed!
