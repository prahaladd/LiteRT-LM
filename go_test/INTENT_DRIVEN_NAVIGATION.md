# Future Enhancement: Intent-Driven Navigation

This document outlines the conceptual design and research notes for migrating the web operator's vector storage from raw ARIA content indexing to **Intent-driven Navigation Indexing**. 

---

## 1. Core Concept

Instead of indexing the live page's ARIA accessibility tree to search for raw selectors on the fly, the vector database should index **user intent strings** and link them to pre-calibrated step sequences.

* **The Problem with ARIA Indexing**: Live page snapshots change dynamically and carry heavy overhead. Parsing selectors dynamically with an LLM on every page state introduces latency and layout sync bugs.
* **The Solution (Intent Mapping)**: Users may speak the same instruction in different ways:
  * *"Let's make this widescreen."*
  * *"Change the aspect ratio to 16 by 9."*
  * *"Ensure it is widescreen format."*
  
  By embedding these natural language variations into a local vector database, we can perform a similarity search to map the spoken voice segment to the pre-calibrated selector sequence:
  `[aria-label="Aspect Ratio"] -> [id$="--1"]`
  
  This eliminates the need for live LLM target-element reasoning for known actions and guarantees 100% selector execution accuracy.

---

## 2. In-Process Serverless Vector Databases (Research Notes)

For a local, self-contained Go/CGo application, standard client-server databases (like Chroma or Qdrant) carry too much overhead. We require serverless, in-process engines that operate directly on the local filesystem like SQLite (zero background daemons, zero network overhead, memory-mapped disk access).

Meta's **FAISS**, **LanceDB**, and **USearch** are the leading options:

### A. LanceDB (Recommended for Multi-Modal & Metadata)
[LanceDB](https://lancedb.github.io/lancedb/) is an open-source, serverless vector database designed to run fully embedded inside your process. It is built on top of the Lance columnar data format, making it incredibly fast for disk-based querying.

* **How it handles the file system**: It stores vectors, metadata, and actual documents in a single local directory. It utilizes memory-mapping (mmap) to query data directly from the disk without loading the entire dataset into RAM.
* **Why it fits**: It is completely serverless. You initialize it by passing a local file path, and it runs entirely within your application runtime.
* **Supported Languages**: Python, JavaScript/TypeScript, and Rust.

### B. FAISS (Facebook AI Similarity Search)
[FAISS](https://github.com/facebookresearch/faiss) is the industry standard for highly optimized, ultra-fast vector similarity searches. While traditionally used in-memory, it has native file system serialization.

* **How it handles the file system**: You build your vector index in-process, and with a single function call (`faiss.write_index`), it serializes the entire mathematical structure into a single compact file on your disk. To query it later, you simply boot your application and read the file back into the process.
* **Why it fits**: There is absolutely no network layer, no HTTP/gRPC overhead, and no server code. It is a raw, high-performance math library.
* **Supported Languages**: Python and C++ natively (with various community wrappers for other languages).

### C. USearch
[USearch](https://github.com/unum-cloud/usearch) is a smaller, hyper-lightweight alternative to FAISS. It is designed specifically for embedded, in-process execution with zero external dependencies.

* **How it handles the file system**: It can save and load indices to/from disk files and heavily utilizes memory-mapping. This allows multiple local processes to look at the same vector file on disk simultaneously without duplicating memory.
* **Why it fits**: It is a header-only library in C++, meaning its footprint inside your application process is minimal and highly efficient.
* **Supported Languages**: Python, JavaScript, Rust, C++, Java, and Go.

---

## 3. Direct Comparison

| Feature | LanceDB | FAISS | USearch |
| :--- | :--- | :--- | :--- |
| **Primary Focus** | Production RAG & Metadata | Maximum Search Speed | Lightweight & Low RAM |
| **Storage Structure** | Columnar directory | Single index file | Single index file |
| **Rich Metadata Filtering** | ✅ Native (SQL-like) | ❌ Complex / External | ❌ Basic |
| **Memory Management** | mmap (Disk-to-CPU) | Loads file into RAM | mmap (Disk-to-CPU) |

---

## 4. Implementation Reference (Python & Go Context)

### LanceDB (Python Reference)
```python
import lancedb

# 1. Connect directly to a local file system directory
db = lancedb.connect("./my_local_vectors")

# 2. Create a table (this automatically creates files on disk)
tbl = db.create_table("my_table", data=[
    {"vector": [0.1, 0.2, 0.3], "text": "Apply 16:9 aspect ratio", "id": 1},
    {"vector": [0.4, 0.5, 0.6], "text": "Set style to photographic", "id": 2}
])

# 3. Query fully in-process
results = tbl.search([0.1, 0.2, 0.3]).limit(1).to_list()
print(results)
```

### FAISS (Python Reference)
```python
import faiss
import numpy as np

dimension = 3
vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype='float32')

# 1. Create index in-process
index = faiss.IndexFlatL2(dimension)
index.add(vectors)

# 2. Serialize and save directly to a file
faiss.write_index(index, "./my_index.faiss")

# 3. Load it back later from the file system
local_index = faiss.read_index("./my_index.faiss")
```
