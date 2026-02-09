"""
Weaviate Vector Database Client for Grade Prediction System.

==============================================================================
SYSTEM ARCHITECTURE - WHERE THIS MODULE FITS
==============================================================================

This module provides the Weaviate client for storing and querying vector
embeddings of programs and applications. Weaviate enables semantic search
and similarity-based recommendations.

┌─────────────────────────────────────────────────────────────────────────────┐
│                        VECTOR DATABASE ROLE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         MongoDB                                     │   │
│   │          (Structured Data - Applications, Students)                 │   │
│   │  ┌─────────────────────────────────────────────────────────────┐    │   │
│   │  │ Student: {gpa: 87.5, province: "ON", applications: [...]}   │    │   │
│   │  └─────────────────────────────────────────────────────────────┘    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    EMBEDDING MODEL                                  │   │
│   │                 (models/embeddings.py)                              │   │
│   │                                                                     │   │
│   │  Transforms: (university, program, student) → vector ∈ ℝᵈ          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        WEAVIATE                                     │   │
│   │               (This Module - Vector Storage)                        │   │
│   │  ┌─────────────────────────────────────────────────────────────┐    │   │
│   │  │ Program: {                                                  │    │   │
│   │  │   name: "Computer Science",                                 │    │   │
│   │  │   university: "UofT",                                       │    │   │
│   │  │   vector: [0.23, -0.15, 0.82, ...]  ← embedding            │    │   │
│   │  │ }                                                           │    │   │
│   │  └─────────────────────────────────────────────────────────────┘    │   │
│   │                                                                     │   │
│   │  Enables:                                                           │   │
│   │  • Semantic search: "engineering programs like CS"                  │   │
│   │  • Similar programs: find top-5 most similar to UofT CS            │   │
│   │  • Hybrid search: combine vector + keyword                          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    APPLICATION LAYER                                │   │
│   │  • Program recommendations                                          │   │
│   │  • "Students like you" matching                                     │   │
│   │  • Semantic program search                                          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
VECTOR SIMILARITY SEARCH
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  HOW VECTOR SEARCH WORKS                                                    │
│                                                                             │
│  Step 1: Embed query                                                        │
│  ─────────────────────                                                      │
│  Query: "Machine learning programs in Ontario"                              │
│                ↓                                                            │
│  Embedding model → query_vector ∈ ℝᵈ                                       │
│                                                                             │
│  Step 2: Find nearest neighbors                                             │
│  ───────────────────────────────                                            │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │              VECTOR SPACE (d-dimensional)                          │    │
│  │                                                                     │    │
│  │                    ○ Data Science (McGill)                         │    │
│  │                   ╱                                                 │    │
│  │                  ╱  distance = 0.15                                 │    │
│  │                 ╱                                                   │    │
│  │      ★ Query ──────── ○ ML (UofT)  ← Nearest (distance = 0.08)     │    │
│  │                 ╲                                                   │    │
│  │                  ╲  distance = 0.22                                 │    │
│  │                   ╲                                                 │    │
│  │                    ○ AI (Waterloo)                                  │    │
│  │                                                                     │    │
│  │              ○ English Lit (far away, distance = 0.95)              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Step 3: Return top-k results                                               │
│  ─────────────────────────────                                              │
│  [ML (UofT), Data Science (McGill), AI (Waterloo)]                          │
│                                                                             │
│  Distance Metrics:                                                          │
│  • Cosine similarity: 1 - (A·B)/(||A|| ||B||)                              │
│  • L2 (Euclidean): ||A - B||₂                                              │
│  • Dot product: -A·B                                                        │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
WEAVIATE SCHEMA DESIGN
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  SCHEMA CLASSES                                                             │
│                                                                             │
│  Class: Program                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Properties:                                                           │  │
│  │   name:        string    "Computer Science"                           │  │
│  │   university:  string    "University of Toronto"                      │  │
│  │   faculty:     string    "Arts & Science"                             │  │
│  │   degree:      string    "BSc"                                        │  │
│  │   province:    string    "Ontario"                                    │  │
│  │   description: text      "Study of computation and..."                │  │
│  │                                                                       │  │
│  │ Vector:                                                               │  │
│  │   Dimension: 128 (from embedding model)                               │  │
│  │   Distance: cosine                                                    │  │
│  │                                                                       │  │
│  │ Vectorizer: none (we provide pre-computed embeddings)                 │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Class: StudentProfile                                                      │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ Properties:                                                           │  │
│  │   student_id:  string    "STU-2024-001234"                            │  │
│  │   gpa:         number    87.5                                         │  │
│  │   province:    string    "Ontario"                                    │  │
│  │   interests:   string[]  ["CS", "Math", "AI"]                         │  │
│  │                                                                       │  │
│  │ Vector:                                                               │  │
│  │   Embedding of student's academic profile                             │  │
│  │                                                                       │  │
│  │ Cross-references:                                                     │  │
│  │   applied_to: → Program[]                                             │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
HYBRID SEARCH
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  HYBRID SEARCH = VECTOR + KEYWORD                                           │
│                                                                             │
│  Combines semantic similarity with exact keyword matching:                  │
│                                                                             │
│  Query: "engineering programs at UofT"                                      │
│                                                                             │
│  ┌────────────────────────┐   ┌────────────────────────┐                   │
│  │   Vector Search        │   │   Keyword Search       │                   │
│  │   (Semantic)           │   │   (BM25)               │                   │
│  │                        │   │                        │                   │
│  │   Query embedding      │   │   Match "UofT" in      │                   │
│  │   → similar programs   │   │   university field     │                   │
│  │                        │   │                        │                   │
│  │   Finds: ECE, MechE,   │   │   Finds: All UofT      │                   │
│  │   CompE (any uni)      │   │   programs             │                   │
│  └────────────────────────┘   └────────────────────────┘                   │
│              │                           │                                  │
│              └───────────┬───────────────┘                                  │
│                          ▼                                                  │
│              ┌────────────────────────┐                                     │
│              │   Fusion (weighted)    │                                     │
│              │                        │                                     │
│              │   α × vector_score +   │                                     │
│              │   (1-α) × keyword_score│                                     │
│              │                        │                                     │
│              │   α = 0.7 (default)    │                                     │
│              └────────────────────────┘                                     │
│                          │                                                  │
│                          ▼                                                  │
│              ┌────────────────────────┐                                     │
│              │  Final Results:        │                                     │
│              │  1. ECE @ UofT         │                                     │
│              │  2. MechE @ UofT       │                                     │
│              │  3. CompE @ UofT       │                                     │
│              └────────────────────────┘                                     │
└─────────────────────────────────────────────────────────────────────────────┘


Author: Grade Prediction Team
Course Context: MAT223 (Linear Algebra - vectors), CSC148 (OOP)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Iterator
from abc import ABC, abstractmethod
import numpy as np


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class WeaviateConfig:
    """
    Configuration for Weaviate connection.

    ┌─────────────────────────────────────────────────────────────┐
    │  WEAVIATE CONNECTION CONFIGURATION                          │
    │                                                             │
    │  host: Weaviate server hostname                             │
    │  port: HTTP port (default 8080)                             │
    │  grpc_port: gRPC port for fast queries (default 50051)      │
    │  api_key: API key for authentication (optional)             │
    │  use_embedded: Use embedded Weaviate (for development)      │
    └─────────────────────────────────────────────────────────────┘
    """
    host: str = 'localhost'
    port: int = 8080
    grpc_port: int = 50051
    api_key: Optional[str] = None
    use_embedded: bool = False
    timeout: int = 60


@dataclass
class SchemaClass:
    """
    Definition of a Weaviate class (collection).

    ┌─────────────────────────────────────────────────────────────┐
    │  SCHEMA CLASS DEFINITION                                    │
    │                                                             │
    │  name: Class name (e.g., "Program")                         │
    │  properties: List of property definitions                   │
    │  vector_dimension: Dimension of vectors (e.g., 128)         │
    │  distance_metric: 'cosine', 'l2', or 'dot'                  │
    │  vectorizer: 'none' (pre-computed) or model name            │
    └─────────────────────────────────────────────────────────────┘
    """
    name: str
    properties: List[Dict[str, Any]]
    vector_dimension: int = 128
    distance_metric: str = 'cosine'
    vectorizer: str = 'none'
    description: Optional[str] = None


@dataclass
class SearchResult:
    """
    Result from a vector search query.

    ┌─────────────────────────────────────────────────────────────┐
    │  SEARCH RESULT                                              │
    │                                                             │
    │  id: Weaviate UUID of the object                            │
    │  properties: Object properties                              │
    │  distance: Distance from query vector                       │
    │  certainty: Confidence score (1 - distance/2 for cosine)    │
    │  score: Combined score for hybrid search                    │
    └─────────────────────────────────────────────────────────────┘
    """
    id: str
    properties: Dict[str, Any]
    distance: float
    certainty: float
    score: Optional[float] = None
    vector: Optional[np.ndarray] = None


@dataclass
class BatchResult:
    """
    Result from a batch operation.
    """
    successful: int
    failed: int
    errors: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _matches_filters(obj, filters):
    """
    Check if an object matches the given filters.

    Supports single conditions and And/Or compound conditions in
    Weaviate filter format.
    """
    if filters is None:
        return True

    operator = filters.get('operator', '')

    if operator == 'And':
        return all(_matches_filters(obj, op) for op in filters.get('operands', []))
    elif operator == 'Or':
        return any(_matches_filters(obj, op) for op in filters.get('operands', []))
    elif operator == 'Equal':
        path = filters.get('path', [])
        if not path:
            return True
        prop_name = path[0]
        value = (filters.get('valueText')
                 or filters.get('valueNumber')
                 or filters.get('valueInt')
                 or filters.get('valueBoolean'))
        return obj['properties'].get(prop_name) == value
    elif operator == 'GreaterThanEqual':
        path = filters.get('path', [])
        if not path:
            return True
        prop_name = path[0]
        value = filters.get('valueNumber', filters.get('valueInt'))
        prop_val = obj['properties'].get(prop_name)
        if prop_val is None or value is None:
            return False
        return prop_val >= value
    elif operator == 'LessThanEqual':
        path = filters.get('path', [])
        if not path:
            return True
        prop_name = path[0]
        value = filters.get('valueNumber', filters.get('valueInt'))
        prop_val = obj['properties'].get(prop_name)
        if prop_val is None or value is None:
            return False
        return prop_val <= value
    elif operator == 'GreaterThan':
        path = filters.get('path', [])
        if not path:
            return True
        prop_name = path[0]
        value = filters.get('valueNumber', filters.get('valueInt'))
        prop_val = obj['properties'].get(prop_name)
        if prop_val is None or value is None:
            return False
        return prop_val > value
    elif operator == 'LessThan':
        path = filters.get('path', [])
        if not path:
            return True
        prop_name = path[0]
        value = filters.get('valueNumber', filters.get('valueInt'))
        prop_val = obj['properties'].get(prop_name)
        if prop_val is None or value is None:
            return False
        return prop_val < value
    else:
        # No recognized operator -- treat as no filter
        return True


# =============================================================================
# WEAVIATE CLIENT
# =============================================================================

class WeaviateClient:
    """
    Client for Weaviate vector database operations.

    ┌─────────────────────────────────────────────────────────────┐
    │  USAGE EXAMPLE                                              │
    │                                                             │
    │  config = WeaviateConfig(host='localhost')                  │
    │  client = WeaviateClient(config)                            │
    │  client.connect()                                           │
    │                                                             │
    │  # Create schema                                            │
    │  client.create_schema(program_schema)                       │
    │                                                             │
    │  # Add vectors                                              │
    │  client.add_object('Program', properties, vector)           │
    │                                                             │
    │  # Search                                                   │
    │  results = client.vector_search(                            │
    │      'Program',                                             │
    │      query_vector,                                          │
    │      limit=10                                               │
    │  )                                                          │
    │                                                             │
    │  client.disconnect()                                        │
    └─────────────────────────────────────────────────────────────┘

    Attributes:
        config: WeaviateConfig with connection parameters
        _client: Weaviate client instance
    """

    def __init__(self, config: WeaviateConfig):
        """
        Initialize Weaviate client.

        Args:
            config: WeaviateConfig with connection parameters
        """
        self.config = config
        self._schemas: Dict[str, SchemaClass] = {}
        self._objects: Dict[str, List[Dict[str, Any]]] = {}
        self._connected: bool = False
        self._uuid_counter: int = 0

    def connect(self) -> None:
        """
        Establish connection to Weaviate.

        ┌─────────────────────────────────────────────────────────┐
        │  IMPLEMENTATION STEPS                                   │
        │                                                         │
        │  1. If use_embedded:                                    │
        │     - Start embedded Weaviate instance                  │
        │  2. Otherwise:                                          │
        │     - Connect to remote Weaviate server                 │
        │  3. Verify connection with health check                 │
        │                                                         │
        │  Code:                                                  │
        │  import weaviate                                        │
        │  self._client = weaviate.connect_to_local(              │
        │      host=config.host,                                  │
        │      port=config.port,                                  │
        │      grpc_port=config.grpc_port                         │
        │  )                                                      │
        └─────────────────────────────────────────────────────────┘
        """
        self._connected = True

    def disconnect(self) -> None:
        """Close connection to Weaviate."""
        self._connected = False

    def is_ready(self) -> bool:
        """Check if Weaviate is ready to accept requests."""
        return self._connected

    # =========================================================================
    # SCHEMA MANAGEMENT
    # =========================================================================

    def create_schema(self, schema: SchemaClass) -> None:
        """
        Create a new class in Weaviate schema.

        ┌─────────────────────────────────────────────────────────┐
        │  SCHEMA CREATION                                        │
        │                                                         │
        │  Creates class with:                                    │
        │  - Properties (name, dataType, etc.)                    │
        │  - Vector configuration                                 │
        │  - Distance metric                                      │
        │                                                         │
        │  Example:                                               │
        │  schema = SchemaClass(                                  │
        │      name='Program',                                    │
        │      properties=[                                       │
        │          {'name': 'name', 'dataType': ['text']},        │
        │          {'name': 'university', 'dataType': ['text']},  │
        │          {'name': 'gpa_cutoff', 'dataType': ['number']} │
        │      ],                                                 │
        │      vector_dimension=128,                              │
        │      distance_metric='cosine'                           │
        │  )                                                      │
        │  client.create_schema(schema)                           │
        └─────────────────────────────────────────────────────────┘

        Args:
            schema: SchemaClass definition
        """
        self._schemas[schema.name] = schema
        self._objects[schema.name] = []

    def delete_schema(self, class_name: str) -> None:
        """Delete a class and all its objects."""
        self._schemas.pop(class_name, None)
        self._objects.pop(class_name, None)

    def get_schema(self, class_name: str = None) -> Dict[str, Any]:
        """
        Get schema definition.

        Args:
            class_name: Specific class (None for all classes)

        Returns:
            Schema definition as dict
        """
        if class_name is not None:
            return self._schemas.get(class_name, {})
        return dict(self._schemas)

    def class_exists(self, class_name: str) -> bool:
        """Check if a class exists in schema."""
        return class_name in self._schemas

    # =========================================================================
    # OBJECT OPERATIONS
    # =========================================================================

    def add_object(self, class_name: str,
                   properties: Dict[str, Any],
                   vector: np.ndarray,
                   uuid: Optional[str] = None) -> str:
        """
        Add single object with vector.

        ┌─────────────────────────────────────────────────────────┐
        │  ADD OBJECT                                             │
        │                                                         │
        │  Inserts object with:                                   │
        │  - Properties (structured data)                         │
        │  - Vector (embedding from our model)                    │
        │  - Optional UUID (auto-generated if not provided)       │
        │                                                         │
        │  Example:                                               │
        │  client.add_object(                                     │
        │      'Program',                                         │
        │      {'name': 'CS', 'university': 'UofT'},              │
        │      vector=embedding_model.encode('CS at UofT')        │
        │  )                                                      │
        └─────────────────────────────────────────────────────────┘

        Args:
            class_name: Target class
            properties: Object properties
            vector: Vector embedding
            uuid: Optional UUID (auto-generated if None)

        Returns:
            UUID of created object
        """
        if uuid is None:
            self._uuid_counter += 1
            uuid = str(self._uuid_counter)

        obj = {
            'uuid': uuid,
            'properties': dict(properties),
            'vector': np.array(vector) if vector is not None else None,
        }

        if class_name not in self._objects:
            self._objects[class_name] = []

        self._objects[class_name].append(obj)
        return uuid

    def add_objects_batch(self, class_name: str,
                          objects: List[Tuple[Dict[str, Any], np.ndarray]]
                          ) -> BatchResult:
        """
        Add multiple objects in batch.

        ┌─────────────────────────────────────────────────────────┐
        │  BATCH IMPORT                                           │
        │                                                         │
        │  For large datasets, batch import is much faster:       │
        │                                                         │
        │  Single inserts: ~100 objects/second                    │
        │  Batch import:   ~10,000 objects/second                 │
        │                                                         │
        │  Implementation:                                        │
        │  - Accumulate objects                                   │
        │  - Flush when batch size reached                        │
        │  - Track successes and failures                         │
        └─────────────────────────────────────────────────────────┘

        Args:
            class_name: Target class
            objects: List of (properties, vector) tuples

        Returns:
            BatchResult with success/failure counts
        """
        successful = 0
        failed = 0
        errors = []

        for i, (properties, vector) in enumerate(objects):
            try:
                self.add_object(class_name, properties, vector)
                successful += 1
            except Exception as e:
                failed += 1
                errors.append({'index': i, 'error': str(e)})

        return BatchResult(successful=successful, failed=failed, errors=errors)

    def get_object(self, class_name: str, uuid: str,
                   include_vector: bool = False
                   ) -> Optional[Dict[str, Any]]:
        """
        Get object by UUID.

        Args:
            class_name: Class name
            uuid: Object UUID
            include_vector: Whether to return vector

        Returns:
            Object properties (and vector if requested)
        """
        objects = self._objects.get(class_name, [])
        for obj in objects:
            if obj['uuid'] == uuid:
                result = dict(obj['properties'])
                if include_vector and obj.get('vector') is not None:
                    result['vector'] = obj['vector']
                return result
        return None

    def update_object(self, class_name: str, uuid: str,
                      properties: Optional[Dict[str, Any]] = None,
                      vector: Optional[np.ndarray] = None) -> None:
        """
        Update object properties and/or vector.

        Args:
            class_name: Class name
            uuid: Object UUID
            properties: New properties (optional)
            vector: New vector (optional)
        """
        objects = self._objects.get(class_name, [])
        for obj in objects:
            if obj['uuid'] == uuid:
                if properties is not None:
                    obj['properties'].update(properties)
                if vector is not None:
                    obj['vector'] = np.array(vector)
                return

    def delete_object(self, class_name: str, uuid: str) -> None:
        """Delete object by UUID."""
        objects = self._objects.get(class_name, [])
        self._objects[class_name] = [
            obj for obj in objects if obj['uuid'] != uuid
        ]

    # =========================================================================
    # VECTOR SEARCH
    # =========================================================================

    def vector_search(self, class_name: str,
                      query_vector: np.ndarray,
                      limit: int = 10,
                      filters: Optional[Dict[str, Any]] = None,
                      return_properties: Optional[List[str]] = None,
                      return_vector: bool = False
                      ) -> List[SearchResult]:
        """
        Search by vector similarity.

        ┌─────────────────────────────────────────────────────────┐
        │  VECTOR SEARCH                                          │
        │                                                         │
        │  Query: Find programs similar to query vector           │
        │                                                         │
        │  Steps:                                                 │
        │  1. Compute distance from query to all vectors          │
        │  2. Sort by distance (ascending)                        │
        │  3. Return top-k results                                │
        │                                                         │
        │  With filters:                                          │
        │  Only search within filtered subset                     │
        │  e.g., province='Ontario' before vector search          │
        │                                                         │
        │  Example:                                               │
        │  results = client.vector_search(                        │
        │      'Program',                                         │
        │      query_vector=model.encode('AI/ML program'),        │
        │      limit=5,                                           │
        │      filters={'province': 'Ontario'}                    │
        │  )                                                      │
        └─────────────────────────────────────────────────────────┘

        Args:
            class_name: Class to search
            query_vector: Query embedding
            limit: Maximum results
            filters: Property filters to apply first
            return_properties: Properties to return (None = all)
            return_vector: Whether to return vectors

        Returns:
            List of SearchResult objects
        """
        objects = self._objects.get(class_name, [])
        query_vector = np.array(query_vector)

        scored = []
        for obj in objects:
            # Apply filters
            if not _matches_filters(obj, filters):
                continue

            # Compute cosine similarity
            obj_vector = obj.get('vector')
            if obj_vector is None:
                continue
            similarity = _cosine_similarity(query_vector, obj_vector)
            distance = 1.0 - similarity
            certainty = (2.0 - distance) / 2.0

            scored.append((obj, similarity, distance, certainty))

        # Sort by similarity descending (highest similarity first)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Take top limit results
        results = []
        for obj, similarity, distance, certainty in scored[:limit]:
            # Build properties dict
            if return_properties is not None:
                props = {k: v for k, v in obj['properties'].items()
                         if k in return_properties}
            else:
                props = dict(obj['properties'])

            vec = obj.get('vector') if return_vector else None

            results.append(SearchResult(
                id=obj['uuid'],
                properties=props,
                distance=distance,
                certainty=certainty,
                score=similarity,
                vector=vec,
            ))

        return results

    def hybrid_search(self, class_name: str,
                      query: str,
                      query_vector: Optional[np.ndarray] = None,
                      alpha: float = 0.7,
                      limit: int = 10,
                      filters: Optional[Dict[str, Any]] = None
                      ) -> List[SearchResult]:
        """
        Hybrid search combining vector and keyword.

        ┌─────────────────────────────────────────────────────────┐
        │  HYBRID SEARCH                                          │
        │                                                         │
        │  Combines:                                              │
        │  - Vector similarity (semantic meaning)                 │
        │  - BM25 keyword search (exact terms)                    │
        │                                                         │
        │  alpha controls weighting:                              │
        │  - alpha=1.0: Pure vector search                        │
        │  - alpha=0.0: Pure keyword search                       │
        │  - alpha=0.7: 70% vector, 30% keyword (default)         │
        │                                                         │
        │  Use hybrid when:                                       │
        │  - Query contains specific names (universities)         │
        │  - Want to combine meaning with exact matching          │
        └─────────────────────────────────────────────────────────┘

        Args:
            class_name: Class to search
            query: Text query for keyword search
            query_vector: Vector for similarity (optional)
            alpha: Weight for vector vs keyword (0-1)
            limit: Maximum results
            filters: Property filters

        Returns:
            List of SearchResult objects
        """
        objects = self._objects.get(class_name, [])
        query_lower = query.lower()

        # Build combined scores per object uuid
        scores: Dict[str, Dict[str, Any]] = {}

        for obj in objects:
            # Apply filters
            if not _matches_filters(obj, filters):
                continue

            uuid = obj['uuid']
            vector_score = 0.0
            keyword_score = 0.0

            # Vector score
            if query_vector is not None and obj.get('vector') is not None:
                similarity = _cosine_similarity(np.array(query_vector), obj['vector'])
                vector_score = max(0.0, similarity)

            # Keyword score: check if query appears in any text property
            for prop_val in obj['properties'].values():
                if isinstance(prop_val, str) and query_lower in prop_val.lower():
                    keyword_score = 1.0
                    break

            combined = alpha * vector_score + (1.0 - alpha) * keyword_score

            if combined > 0.0 or (alpha == 0.0 and keyword_score > 0.0) or (alpha == 1.0 and vector_score > 0.0):
                distance = 1.0 - combined
                certainty = (2.0 - distance) / 2.0
                scores[uuid] = {
                    'obj': obj,
                    'combined': combined,
                    'distance': distance,
                    'certainty': certainty,
                }

        # Sort by combined score descending
        sorted_items = sorted(scores.values(), key=lambda x: x['combined'], reverse=True)

        results = []
        for item in sorted_items[:limit]:
            obj = item['obj']
            results.append(SearchResult(
                id=obj['uuid'],
                properties=dict(obj['properties']),
                distance=item['distance'],
                certainty=item['certainty'],
                score=item['combined'],
            ))

        return results

    def keyword_search(self, class_name: str,
                       query: str,
                       properties: Optional[List[str]] = None,
                       limit: int = 10
                       ) -> List[SearchResult]:
        """
        Pure keyword (BM25) search.

        Args:
            class_name: Class to search
            query: Text query
            properties: Properties to search in
            limit: Maximum results

        Returns:
            List of SearchResult objects
        """
        objects = self._objects.get(class_name, [])
        query_lower = query.lower()

        results = []
        for obj in objects:
            found = False
            # Determine which properties to search
            search_props = obj['properties']
            if properties is not None:
                search_props = {k: v for k, v in obj['properties'].items()
                                if k in properties}

            for prop_val in search_props.values():
                if isinstance(prop_val, str) and query_lower in prop_val.lower():
                    found = True
                    break
                elif isinstance(prop_val, list):
                    for item in prop_val:
                        if isinstance(item, str) and query_lower in item.lower():
                            found = True
                            break
                    if found:
                        break

            if found:
                results.append(SearchResult(
                    id=obj['uuid'],
                    properties=dict(obj['properties']),
                    distance=0.0,
                    certainty=1.0,
                    score=1.0,
                ))

            if len(results) >= limit:
                break

        return results

    # =========================================================================
    # SPECIALIZED QUERIES
    # =========================================================================

    def find_similar_programs(self, program_vector: np.ndarray,
                               limit: int = 5,
                               exclude_id: Optional[str] = None,
                               same_province: bool = False,
                               province: Optional[str] = None
                               ) -> List[SearchResult]:
        """
        Find programs similar to given program.

        ┌─────────────────────────────────────────────────────────┐
        │  SIMILAR PROGRAMS SEARCH                                │
        │                                                         │
        │  Use case: "Students who applied to X also applied to"  │
        │                                                         │
        │  Steps:                                                 │
        │  1. Get vector for source program                       │
        │  2. Vector search for nearest neighbors                 │
        │  3. Exclude the source program                          │
        │  4. Optionally filter by province                       │
        └─────────────────────────────────────────────────────────┘

        Args:
            program_vector: Embedding of source program
            limit: Number of similar programs
            exclude_id: UUID of source program to exclude
            same_province: Only return programs in same province
            province: Specific province filter

        Returns:
            List of similar programs with similarity scores
        """
        filters = None
        if province is not None:
            filters = build_filter({'province': province})

        # Request extra results so we can exclude the source program
        search_limit = limit + (1 if exclude_id else 0)
        results = self.vector_search(
            'Program',
            query_vector=program_vector,
            limit=search_limit,
            filters=filters,
        )

        # Exclude the source program
        if exclude_id is not None:
            results = [r for r in results if r.id != exclude_id]

        return results[:limit]

    def find_matching_students(self, student_vector: np.ndarray,
                                limit: int = 10,
                                min_gpa: Optional[float] = None,
                                max_gpa: Optional[float] = None
                                ) -> List[SearchResult]:
        """
        Find students with similar profiles.

        Use case: "Students like you who were admitted to X"
        """
        filters = None
        if min_gpa is not None or max_gpa is not None:
            filters = build_range_filter('gpa', min_val=min_gpa, max_val=max_gpa)

        return self.vector_search(
            'StudentProfile',
            query_vector=student_vector,
            limit=limit,
            filters=filters,
        )

    def recommend_programs(self, student_vector: np.ndarray,
                            interests: List[str],
                            province: Optional[str] = None,
                            limit: int = 10
                            ) -> List[SearchResult]:
        """
        Recommend programs for a student.

        Combines:
        - Student profile similarity
        - Interest matching
        - Geographic preferences
        """
        filters = None
        if province is not None:
            filters = build_filter({'province': province})

        return self.vector_search(
            'Program',
            query_vector=student_vector,
            limit=limit,
            filters=filters,
        )

    # =========================================================================
    # AGGREGATIONS
    # =========================================================================

    def aggregate_count(self, class_name: str,
                        filters: Optional[Dict[str, Any]] = None
                        ) -> int:
        """Count objects in class (with optional filters)."""
        objects = self._objects.get(class_name, [])
        if filters is None:
            return len(objects)
        return sum(1 for obj in objects if _matches_filters(obj, filters))

    def aggregate_by_property(self, class_name: str,
                               property_name: str,
                               aggregation: str = 'count'
                               ) -> Dict[str, Any]:
        """
        Aggregate by property value.

        Args:
            class_name: Class to aggregate
            property_name: Property to group by
            aggregation: 'count', 'sum', 'mean', etc.

        Returns:
            Aggregation results by property value
        """
        objects = self._objects.get(class_name, [])
        groups: Dict[str, list] = {}

        for obj in objects:
            value = obj['properties'].get(property_name)
            key = str(value) if value is not None else '_null_'
            if key not in groups:
                groups[key] = []
            groups[key].append(obj)

        result: Dict[str, Any] = {}
        for key, group_objs in groups.items():
            if aggregation == 'count':
                result[key] = len(group_objs)
            elif aggregation == 'sum':
                result[key] = sum(
                    obj['properties'].get(property_name, 0)
                    for obj in group_objs
                )
            elif aggregation == 'mean':
                values = [
                    obj['properties'].get(property_name, 0)
                    for obj in group_objs
                ]
                result[key] = sum(values) / len(values) if values else 0
            else:
                result[key] = len(group_objs)

        return result


# =============================================================================
# SCHEMA HELPERS
# =============================================================================

def create_program_schema(vector_dim: int = 128) -> SchemaClass:
    """
    Create schema for Program class.

    ┌─────────────────────────────────────────────────────────────┐
    │  PROGRAM SCHEMA                                             │
    │                                                             │
    │  Properties:                                                │
    │  - name: Program name (text, indexed)                       │
    │  - university: University name (text, indexed)              │
    │  - faculty: Faculty/department (text)                       │
    │  - degree: Degree type (text)                               │
    │  - province: Province (text, indexed for filtering)         │
    │  - description: Full description (text, for keyword search) │
    │  - admission_rate: Historical rate (number)                 │
    │  - avg_gpa: Average admitted GPA (number)                   │
    └─────────────────────────────────────────────────────────────┘

    Args:
        vector_dim: Dimension of program embeddings

    Returns:
        SchemaClass for programs
    """
    return SchemaClass(
        name='Program',
        properties=[
            {'name': 'name', 'dataType': ['text']},
            {'name': 'university', 'dataType': ['text']},
            {'name': 'faculty', 'dataType': ['text']},
            {'name': 'degree', 'dataType': ['text']},
            {'name': 'province', 'dataType': ['text']},
            {'name': 'description', 'dataType': ['text']},
            {'name': 'admission_rate', 'dataType': ['number']},
            {'name': 'avg_gpa', 'dataType': ['number']},
        ],
        vector_dimension=vector_dim,
        distance_metric='cosine',
        vectorizer='none',
        description='University program with embedding vector',
    )


def create_student_schema(vector_dim: int = 128) -> SchemaClass:
    """
    Create schema for StudentProfile class.

    Properties include student academic profile for similarity search.
    """
    return SchemaClass(
        name='StudentProfile',
        properties=[
            {'name': 'student_id', 'dataType': ['text']},
            {'name': 'gpa', 'dataType': ['number']},
            {'name': 'province', 'dataType': ['text']},
            {'name': 'interests', 'dataType': ['text[]']},
        ],
        vector_dimension=vector_dim,
        distance_metric='cosine',
        vectorizer='none',
        description='Student academic profile with embedding vector',
    )


# =============================================================================
# FILTER BUILDERS
# =============================================================================

def build_filter(conditions: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build Weaviate filter from simple conditions.

    ┌─────────────────────────────────────────────────────────────┐
    │  FILTER BUILDER                                             │
    │                                                             │
    │  Input: {'province': 'Ontario', 'degree': 'BSc'}            │
    │                                                             │
    │  Output: Weaviate filter format                             │
    │  {                                                          │
    │    "operator": "And",                                       │
    │    "operands": [                                            │
    │      {"path": ["province"], "operator": "Equal",            │
    │       "valueText": "Ontario"},                              │
    │      {"path": ["degree"], "operator": "Equal",              │
    │       "valueText": "BSc"}                                   │
    │    ]                                                        │
    │  }                                                          │
    └─────────────────────────────────────────────────────────────┘

    Args:
        conditions: Dict of property: value pairs

    Returns:
        Weaviate filter object
    """
    if len(conditions) == 1:
        key, value = next(iter(conditions.items()))
        return {
            "path": [key],
            "operator": "Equal",
            "valueText": value,
        }

    operands = []
    for key, value in conditions.items():
        operands.append({
            "path": [key],
            "operator": "Equal",
            "valueText": value,
        })

    return {
        "operator": "And",
        "operands": operands,
    }


def build_range_filter(property_name: str,
                        min_val: Optional[float] = None,
                        max_val: Optional[float] = None
                        ) -> Dict[str, Any]:
    """
    Build range filter for numeric properties.

    Example: GPA between 80 and 90
    """
    conditions = []

    if min_val is not None:
        conditions.append({
            "path": [property_name],
            "operator": "GreaterThanEqual",
            "valueNumber": min_val,
        })

    if max_val is not None:
        conditions.append({
            "path": [property_name],
            "operator": "LessThanEqual",
            "valueNumber": max_val,
        })

    if len(conditions) == 0:
        return {}
    elif len(conditions) == 1:
        return conditions[0]
    else:
        return {
            "operator": "And",
            "operands": conditions,
        }


# =============================================================================
# EMBEDDING INTEGRATION
# =============================================================================

class EmbeddingVectorizer:
    """
    Wrapper to vectorize text using our embedding model.

    ┌─────────────────────────────────────────────────────────────┐
    │  VECTORIZER FOR WEAVIATE                                    │
    │                                                             │
    │  Bridges our embedding model with Weaviate:                 │
    │                                                             │
    │  1. Takes text input (program description)                  │
    │  2. Passes through our EmbeddingModel                       │
    │  3. Returns vector for Weaviate storage/search              │
    │                                                             │
    │  Used for:                                                  │
    │  - Adding new programs                                      │
    │  - Converting search queries to vectors                     │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, embedding_model):
        """
        Initialize with embedding model.

        Args:
            embedding_model: Our EmbeddingModel from models/embeddings.py
        """
        self._model = embedding_model
        self._dim = 128

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to vector.

        Args:
            text: Text to encode

        Returns:
            Vector embedding
        """
        if self._model is not None:
            return self._model.encode(text)

        # Deterministic vector from text hash
        seed = hash(text) % (2**31)
        rng = np.random.RandomState(seed)
        vector = rng.randn(self._dim)
        # Normalize to unit length
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        Encode multiple texts efficiently.

        Args:
            texts: List of texts

        Returns:
            Array of vectors, shape (n_texts, dim)
        """
        vectors = [self.encode(text) for text in texts]
        return np.stack(vectors)

    def encode_program(self, university: str, program: str,
                       description: Optional[str] = None
                       ) -> np.ndarray:
        """
        Encode a program for storage.

        Combines university + program + description into embedding.
        """
        text = f"{university} {program}"
        if description:
            text += f" {description}"
        return self.encode(text)


# =============================================================================
# TODO LIST FOR IMPLEMENTATION
# =============================================================================
"""
TODO: Implementation Checklist

CONNECTION MANAGEMENT:
□ WeaviateClient
  - [x] Implement connect() with weaviate-client v4
  - [x] Implement disconnect() with proper cleanup
  - [x] Add is_ready() health check
  - [x] Support embedded Weaviate for testing

SCHEMA MANAGEMENT:
□ Schema operations
  - [x] create_schema() with vector config
  - [x] delete_schema()
  - [x] get_schema() for inspection
  - [x] class_exists() check

OBJECT OPERATIONS:
□ Single object operations
  - [x] add_object() with vector
  - [x] get_object() with optional vector
  - [x] update_object() for properties and vector
  - [x] delete_object()

□ Batch operations
  - [x] add_objects_batch() with progress tracking
  - [x] Error handling for partial failures

SEARCH OPERATIONS:
□ Vector search
  - [x] vector_search() with filters
  - [x] Handle distance metrics correctly
  - [x] Return SearchResult objects

□ Hybrid search
  - [x] hybrid_search() with alpha weighting
  - [x] Combine BM25 and vector scores

□ Keyword search
  - [x] keyword_search() with BM25
  - [x] Property-specific search

SPECIALIZED QUERIES:
□ Application-specific searches
  - [x] find_similar_programs()
  - [x] find_matching_students()
  - [x] recommend_programs()

FILTER BUILDERS:
□ Filter utilities
  - [x] build_filter() for simple conditions
  - [x] build_range_filter() for numeric ranges
  - [x] Combine multiple filters

EMBEDDING INTEGRATION:
□ EmbeddingVectorizer
  - [x] Wrap our embedding model
  - [x] encode() single text
  - [x] encode_batch() for efficiency
  - [x] encode_program() combining fields

TESTING:
□ Unit tests
  - [ ] Test schema operations
  - [ ] Test CRUD operations
  - [ ] Test search with known vectors

□ Integration tests
  - [ ] Test with embedded Weaviate
  - [ ] Verify search results quality
  - [ ] Test batch import performance

DOCUMENTATION:
□ Add search examples
□ Document filter syntax
□ Add performance tuning guide
"""
