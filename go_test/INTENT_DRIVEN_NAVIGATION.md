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

---

## 5. Go In-Process Implementation with USearch (For MCP Server)

To build a fully in-process, file system-backed vector store in Go without any external database server, the two best options are USearch Go bindings or a pure Go library like `Phisat/Go-FAISS`.

For a Model Context Protocol (MCP) server, **USearch** is highly recommended. It uses memory-mapping (`mmap`), allowing your Go process to query multi-gigabyte vector files directly on disk with near-zero initialization time and minimal memory overhead.

Below is the design and production-ready Go implementation for an in-process, disk-backed vector search system tailored for an MCP server architecture.

### Architecture Overview

```text
[Claude / LLM Client] 
       │ (JSON-RPC via Standard I/O)
       ▼
┌──────────────────────────────── MCP Go Server ───────────────────────────────┐
│                                                                              │
│  1. Incoming Text ──► [Embedding API] ──► Float32 Array                      │
│                                                                              │
│  2. In-Process Indexing & Storage:                                           │
│     Vector Array  ──► [USearch Index (mmap)] ──► Saved to ./data/vectors.idx  │
│     Text/Metadata ──► [JSON / Key-Value]     ──► Saved to ./data/metadata.json│
└──────────────────────────────────────────────────────────────────────────────┘
```

Because native vector indexes (like FAISS or USearch) only map an ID (`uint64`) to a Vector, you must maintain a lightweight local sidecar file (like a JSON file or an embedded Key-Value store like BoltDB) to map that `uint64` ID back to your actual text content or metadata.

---

### Step-by-Step Implementation

#### 1. Project Setup
Initialize your Go module and fetch the required dependencies. We will use the official `usearch` bindings for high-performance vector operations.

```bash
mkdir mcp-vector-server
cd mcp-vector-server
go mod init mcp-vector-server
go get github.com/unum-cloud/usearch/golang
```
*(Note: USearch requires CGO to compile its high-performance C++ header core into Go).*

#### 2. The Core Vector Store Component
Create a file named `vector_store.go`. This struct handles initializing the index from a local file path, appending new vectors, saving to disk, and fetching corresponding text metadata.

```go
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"sync"

	usearch "github.com/unum-cloud/usearch/golang"
)

type Metadata struct {
	ID   uint64 `json:"id"`
	Text string `json:"text"`
}

type LocalVectorStore struct {
	mu          sync.RWMutex
	indexPath   string
	metaPath    string
	index       *usearch.Index
	metadata    map[uint64]string
	nextID      uint64
	dimensions  uint
}

func NewLocalVectorStore(dataDir string, dimensions uint) (*LocalVectorStore, error) {
	if err := os.MkdirAll(dataDir, 0755); err != nil {
		return nil, err
	}

	indexPath := fmt.Sprintf("%s/vectors.idx", dataDir)
	metaPath := fmt.Sprintf("%s/metadata.json", dataDir)

	// Configure USearch for Cosine similarity or L2Squared
	conf := usearch.NewIndexConfig()
	conf.SetMetric(usearch.MetricCosine)
	conf.SetDimensions(dimensions)
	conf.SetQuantization(usearch.ScalarKindFloat32)

	index := usearch.NewIndex(conf)

	store := &LocalVectorStore{
		indexPath:  indexPath,
		metaPath:   metaPath,
		index:      index,
		metadata:   make(map[uint64]string),
		nextID:     0,
		dimensions: dimensions,
	}

	// Load existing vector index from disk if it exists
	if _, err := os.Stat(indexPath); err == nil {
		if err := store.index.View(indexPath); err != nil {
			return nil, fmt.Errorf("failed to mmap index file: %w", err)
		}
		// Sync current size to determine next sequence ID
		store.nextID = uint64(store.index.Len())
	}

	// Load existing text metadata sidecar
	if _, err := os.Stat(metaPath); err == nil {
		metaDataBytes, err := os.ReadFile(metaPath)
		if err == nil {
			var savedMeta []Metadata
			if err := json.Unmarshal(metaDataBytes, &savedMeta); err == nil {
				for _, m := range savedMeta {
					store.metadata[m.ID] = m.Text
					if m.ID >= store.nextID {
						store.nextID = m.ID + 1
					}
				}
			}
		}
	}

	return store, nil
}

// AddVector inserts a vector and its text in-process, then flushes changes directly to disk
func (s *LocalVectorStore) AddVector(vector []float32, text string) (uint64, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if uint(len(vector)) != s.dimensions {
		return 0, fmt.Errorf("invalid vector dimension: got %d, want %d", len(vector), s.dimensions)
	}

	id := s.nextID
	
	// Add directly to in-process index
	if err := s.index.Add(id, vector); err != nil {
		return 0, fmt.Errorf("failed to add vector to index: %w", err)
	}

	s.metadata[id] = text
	s.nextID++

	// Atomically persist vector index back to local storage
	if err := s.index.Save(s.indexPath); err != nil {
		return 0, fmt.Errorf("failed to write index to disk: %w", err)
	}

	// Persist metadata sidecar
	var metaList []Metadata
	for k, v := range s.metadata {
		metaList = append(metaList, Metadata{ID: k, Text: v})
	}
	metaBytes, err := json.MarshalIndent(metaList, "", "  ")
	if err != nil {
		return 0, err
	}
	if err := os.WriteFile(s.metaPath, metaBytes, 0644); err != nil {
		return 0, fmt.Errorf("failed to write metadata to disk: %w", err)
	}

	return id, nil
}

// Search queries the file system-backed store and maps structural IDs back to text strings
func (s *LocalVectorStore) Search(queryVector []float32, limit uint) ([]Metadata, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	results, err := s.index.Search(queryVector, limit)
	if err != nil {
		return nil, fmt.Errorf("vector search failed: %w", err)
	}

	var matchedRecords []Metadata
	for _, id := range results.Keys {
		if text, exists := s.metadata[id]; exists {
			matchedRecords = append(matchedRecords, Metadata{
				ID:   id,
				Text: text,
			})
		}
	}

	return matchedRecords, nil
}
```

#### 3. Integrating with the MCP Server Loop
Create your `main.go` file. MCP servers natively use standard input (`os.Stdin`) and standard output (`os.Stdout`) to communicate with LLM clients like Claude Desktop. This loop wireframe showcases how your MCP tools tap directly into the file system store.

```go
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
)

// Simplified MCP structures for demo purposes
type MCPRequest struct {
	Method string          `json:"method"`
	Params json.RawMessage `json:"params"`
}

type StoreParams struct {
	Text   string    `json:"text"`
	Vector []float32 `json:"vector"`
}

type QueryParams struct {
	Vector []float32 `json:"vector"`
	Limit  uint      `json:"limit"`
}

func main() {
	// Initialize vector store in directory "./mcp_vault" with 1536 dimensions (e.g., OpenAI text-embedding-3-small)
	store, err := NewLocalVectorStore("./mcp_vault", 1536)
	if err != nil {
		log.Fatalf("Failed to spin up local vector database: %v", err)
	}

	// Standard MCP servers process continuous JSON-RPC requests via Stdin
	decoder := json.NewDecoder(os.Stdin)
	for {
		var req MCPRequest
		if err := decoder.Decode(&req); err != nil {
			break // Connection closed or invalid input
		}

		switch req.Method {
		case "tools/call/store_embedding":
			var params StoreParams
			json.Unmarshal(req.Params, &params)
			
			id, err := store.AddVector(params.Vector, params.Text)
			if err != nil {
				sendError(err.Error())
				continue
			}
			sendResponse(fmt.Sprintf("Successfully saved to disk with ID: %d", id))

		case "tools/call/search_embeddings":
			var params QueryParams
			json.Unmarshal(req.Params, &params)
			if params.Limit == 0 {
				params.Limit = 5
			}

			matches, err := store.Search(params.Vector, params.Limit)
			if err != nil {
				sendError(err.Error())
				continue
			}
			sendResponse(matches)
		}
	}
}

func sendResponse(data interface{}) {
	resp, _ := json.Marshal(data)
	fmt.Fprintln(os.Stdout, string(resp))
}

func sendError(msg string) {
	fmt.Fprintf(os.Stderr, "Error: %s\n", msg)
}
```

---

### Why This Design Fits Your Exact Requirement

* **Zero External Network Ports**: It compiles seamlessly into a single Go binary. No TCP connections, HTTP overhead, or Docker daemons are utilized.
* **Instant In-Process Reboots**: Because it uses `mmap` under the hood through USearch, if your MCP server crashes or restarts, it reopens the `vectors.idx` instantly without reading gigabytes of data into RAM at startup.
* **Complete Isolation**: All vectors and text contents reside predictably inside whatever path you give `NewLocalVectorStore` (like `./mcp_vault`), making it clean to deploy, backup, or wipe.

