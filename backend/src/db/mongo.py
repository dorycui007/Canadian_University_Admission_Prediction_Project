"""
MongoDB Client for Grade Prediction System.

==============================================================================
SYSTEM ARCHITECTURE - WHERE THIS MODULE FITS
==============================================================================

This module provides the MongoDB client for storing and retrieving application
data. MongoDB is chosen for its document-oriented structure which naturally
fits the hierarchical nature of student application records.

┌─────────────────────────────────────────────────────────────────────────────┐
│                        DATABASE ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     EXTERNAL DATA SOURCES                           │   │
│   │  Ontario Universities' Application Centre (OUAC)                    │   │
│   │  BC Post-Secondary Application Service (BCPAS)                      │   │
│   │  ApplyAlberta, etc.                                                 │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        ETL PIPELINE                                 │   │
│   │                      (db/etl.py)                                    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                    ┌───────────────┴───────────────┐                        │
│                    ▼                               ▼                        │
│   ┌───────────────────────────┐    ┌───────────────────────────┐           │
│   │        MongoDB            │    │        Weaviate           │           │
│   │     (This Module)         │    │   (Vector Embeddings)     │           │
│   │                           │    │                           │           │
│   │  • Student records        │    │  • Program embeddings     │           │
│   │  • Application data       │    │  • Semantic search        │           │
│   │  • Outcome history        │    │  • Similarity queries     │           │
│   │  • Aggregate stats        │    │                           │           │
│   └───────────────────────────┘    └───────────────────────────┘           │
│                    │                               │                        │
│                    └───────────────┬───────────────┘                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                     FEATURE ENGINEERING                             │   │
│   │                   (features/design_matrix.py)                       │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       MODEL TRAINING                                │   │
│   │              (models/logistic.py, models/embeddings.py)             │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
MONGODB DOCUMENT STRUCTURE
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  DOCUMENT HIERARCHY                                                         │
│                                                                             │
│  Collection: students                                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ {                                                                     │  │
│  │   "_id": ObjectId("..."),                                             │  │
│  │   "student_id": "STU-2024-001234",                                    │  │
│  │   "demographics": {                                                   │  │
│  │     "province": "Ontario",                                            │  │
│  │     "school_type": "public",                                          │  │
│  │     "graduation_year": 2024                                           │  │
│  │   },                                                                  │  │
│  │   "academics": {                                                      │  │
│  │     "gpa_overall": 87.5,                                              │  │
│  │     "gpa_by_subject": {                                               │  │
│  │       "math": 92.0,                                                   │  │
│  │       "english": 85.0,                                                │  │
│  │       "science": 88.0                                                 │  │
│  │     },                                                                │  │
│  │     "courses": [...]                                                  │  │
│  │   },                                                                  │  │
│  │   "applications": [                                                   │  │
│  │     {                                                                 │  │
│  │       "university": "University of Toronto",                          │  │
│  │       "program": "Computer Science",                                  │  │
│  │       "campus": "St. George",                                         │  │
│  │       "application_date": ISODate("2024-01-15"),                      │  │
│  │       "outcome": "admitted",                                          │  │
│  │       "decision_date": ISODate("2024-05-01"),                         │  │
│  │       "offer_details": {...}                                          │  │
│  │     },                                                                │  │
│  │     {...}                                                             │  │
│  │   ]                                                                   │  │
│  │ }                                                                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│  Collection: programs                                                       │
│  ┌───────────────────────────────────────────────────────────────────────┐  │
│  │ {                                                                     │  │
│  │   "_id": ObjectId("..."),                                             │  │
│  │   "university": "University of Toronto",                              │  │
│  │   "program_name": "Computer Science",                                 │  │
│  │   "faculty": "Arts & Science",                                        │  │
│  │   "degree_type": "BSc",                                               │  │
│  │   "requirements": {...},                                              │  │
│  │   "statistics": {                                                     │  │
│  │     "2023": {"applicants": 5000, "admitted": 500, "enrolled": 350},  │  │
│  │     "2022": {...}                                                     │  │
│  │   }                                                                   │  │
│  │ }                                                                     │  │
│  └───────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
MONGODB AGGREGATION PIPELINE
==============================================================================

MongoDB's aggregation framework enables complex data transformations:

┌─────────────────────────────────────────────────────────────────────────────┐
│  AGGREGATION PIPELINE EXAMPLE                                               │
│                                                                             │
│  Goal: Compute admission rates by program and year                          │
│                                                                             │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐     │
│  │   $match    │ → │   $unwind   │ → │   $group    │ → │  $project   │     │
│  │  (filter)   │   │ (flatten)   │   │ (aggregate) │   │ (reshape)   │     │
│  └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘     │
│        │                │                  │                 │              │
│        ▼                ▼                  ▼                 ▼              │
│   Filter to         Flatten          Group by           Compute rate       │
│   year=2023        applications     (program,year)       admitted/total    │
│                                                                             │
│  Pipeline Code:                                                             │
│  [                                                                          │
│    { "$match": { "applications.application_date": { "$gte": 2023 } } },    │
│    { "$unwind": "$applications" },                                          │
│    { "$group": {                                                            │
│        "_id": { "program": "$applications.program" },                       │
│        "total": { "$sum": 1 },                                              │
│        "admitted": {                                                        │
│          "$sum": { "$cond": [{"$eq":["$applications.outcome","admitted"]}, │
│                              1, 0] }                                        │
│        }                                                                    │
│      }                                                                      │
│    },                                                                       │
│    { "$project": {                                                          │
│        "program": "$_id.program",                                           │
│        "admission_rate": { "$divide": ["$admitted", "$total"] }             │
│      }                                                                      │
│    }                                                                        │
│  ]                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
INDEXING STRATEGY
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  MONGODB INDEXES FOR PERFORMANCE                                            │
│                                                                             │
│  1. Compound Index on Query Patterns:                                       │
│     ────────────────────────────────                                        │
│     { "applications.university": 1, "applications.program": 1 }             │
│                                                                             │
│     For queries like:                                                       │
│     find({"applications.university": "UofT",                                │
│           "applications.program": "CS"})                                    │
│                                                                             │
│  2. Date Index for Temporal Queries:                                        │
│     ──────────────────────────────                                          │
│     { "applications.application_date": -1 }                                 │
│                                                                             │
│     For time-range queries and sorting by recency                           │
│                                                                             │
│  3. Text Index for Search:                                                  │
│     ───────────────────────                                                 │
│     { "applications.program": "text" }                                      │
│                                                                             │
│     For program name search                                                 │
│                                                                             │
│  4. Unique Index on Student ID:                                             │
│     ────────────────────────────                                            │
│     { "student_id": 1 }, unique=True                                        │
│                                                                             │
│     Prevents duplicate student records                                      │
│└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
CONNECTION POOLING AND PERFORMANCE
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  CONNECTION MANAGEMENT                                                      │
│                                                                             │
│  MongoClient maintains a connection pool:                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    CONNECTION POOL                                  │    │
│  │  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐                                 │    │
│  │  │ C1 │ │ C2 │ │ C3 │ │ C4 │ │... │  (max_pool_size connections)    │    │
│  │  └────┘ └────┘ └────┘ └────┘ └────┘                                 │    │
│  │     │      │      │      │      │                                   │    │
│  │     └──────┴──────┴──────┴──────┘                                   │    │
│  │                    │                                                │    │
│  │                    ▼                                                │    │
│  │            ┌──────────────┐                                         │    │
│  │            │   MongoDB    │                                         │    │
│  │            │   Server     │                                         │    │
│  │            └──────────────┘                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Best Practices:                                                            │
│  • Create MongoClient ONCE at application startup                           │
│  • Reuse the same client across requests                                    │
│  • Client is thread-safe                                                    │
│  • Set appropriate timeouts                                                 │
│  └─────────────────────────────────────────────────────────────────────────────┘


Author: Grade Prediction Team
Course Context: CSC148 (OOP), CSC343 (Databases)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Iterator
from abc import ABC, abstractmethod
import numpy as np


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class MongoConfig:
    """
    Configuration for MongoDB connection.

    ┌─────────────────────────────────────────────────────────────────┐
    │  MONGODB CONNECTION CONFIGURATION                           │
    │                                                             │
    │  host: MongoDB server hostname                              │
    │  port: MongoDB server port (default 27017)                  │
    │  database: Database name                                    │
    │  username: Authentication username (optional)               │
    │  password: Authentication password (optional)               │
    │  connection_string: Full URI (overrides host/port)          │
    │  max_pool_size: Maximum connection pool size                │
    │  timeout_ms: Operation timeout in milliseconds              │
    └─────────────────────────────────────────────────────────────┘
    """
    host: str = 'localhost'
    port: int = 27017
    database: str = 'grade_prediction'
    username: Optional[str] = None
    password: Optional[str] = None
    connection_string: Optional[str] = None
    max_pool_size: int = 100
    timeout_ms: int = 30000
    replica_set: Optional[str] = None


@dataclass
class QueryConfig:
    """
    Configuration for query execution.

    ┌─────────────────────────────────────────────────────────────┐
    │  QUERY CONFIGURATION                                        │
    │                                                             │
    │  batch_size: Documents per batch for cursors                │
    │  allow_disk_use: Allow aggregation to use disk              │
    │  max_time_ms: Maximum query execution time                  │
    │  read_preference: 'primary', 'secondary', 'nearest'         │
    └─────────────────────────────────────────────────────────────┘
    """
    batch_size: int = 1000
    allow_disk_use: bool = True
    max_time_ms: int = 60000
    read_preference: str = 'primary'


@dataclass
class StudentRecord:
    """
    Data class representing a student record.

    ┌─────────────────────────────────────────────────────────────┐
    │  STUDENT RECORD STRUCTURE                                   │
    │                                                             │
    │  Maps to MongoDB document structure:                        │
    │  • student_id: Unique identifier                            │
    │  • demographics: Province, school info                      │
    │  • academics: GPA, courses, grades                          │
    │  • applications: List of applications with outcomes         │
    └─────────────────────────────────────────────────────────────┘
    """
    student_id: str
    demographics: Dict[str, Any]
    academics: Dict[str, Any]
    applications: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ApplicationRecord:
    """
    Data class for a single application.

    ┌─────────────────────────────────────────────────────────────┐
    │  APPLICATION RECORD                                         │
    │                                                             │
    │  Represents one program application:                        │
    │  • university: Target university                            │
    │  • program: Program name                                    │
    │  • campus: Specific campus (if applicable)                  │
    │  • application_date: When applied                           │
    │  • outcome: 'admitted', 'rejected', 'waitlisted', None      │
    │  • decision_date: When decision received                    │
    └─────────────────────────────────────────────────────────────┘
    """
    university: str
    program: str
    campus: Optional[str]
    application_date: str  # ISO format date string
    outcome: Optional[str] = None
    decision_date: Optional[str] = None
    offer_details: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# ABSTRACT BASE CLIENT
# =============================================================================

class BaseMongoClient(ABC):
    """
    Abstract base class for MongoDB operations.

    ┌─────────────────────────────────────────────────────────────┐
    │  MONGODB CLIENT INTERFACE                                   │
    │                                                             │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │          BaseMongoClient (Abstract)                 │    │
    │  │                                                     │    │
    │  │  + connect() → None                                 │    │
    │  │  + disconnect() → None                              │    │
    │  │  + insert_one(collection, doc) → str                │    │
    │  │  + insert_many(collection, docs) → List[str]        │    │
    │  │  + find(collection, query) → Iterator               │    │
    │  │  + aggregate(collection, pipeline) → Iterator       │    │
    │  │  + update_one(collection, filter, update)           │    │
    │  │  + delete_many(collection, filter)                  │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                          △                                  │
    │                          │                                  │
    │              ┌───────────┴───────────┐                      │
    │              │                       │                      │
    │      ┌───────┴───────┐       ┌──────┴──────┐               │
    │      │  MongoClient  │       │  MockClient │               │
    │      │  (Production) │       │  (Testing)  │               │
    │      └───────────────┘       └─────────────┘               │
    └─────────────────────────────────────────────────────────────┘
    """

    @abstractmethod
    def connect(self) -> None:
        """
        Establish connection to MongoDB.

        Implementation Steps:
            1. Build connection string from config
            2. Create MongoClient with connection pool
            3. Ping server to verify connection
            4. Store client reference
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """
        Close MongoDB connection.

        Implementation Steps:
            1. Close client connection
            2. Clean up resources
        """
        pass

    @abstractmethod
    def insert_one(self, collection: str, document: Dict[str, Any]) -> str:
        """
        Insert a single document.

        Args:
            collection: Collection name
            document: Document to insert

        Returns:
            Inserted document's _id as string
        """
        pass

    @abstractmethod
    def insert_many(self, collection: str,
                    documents: List[Dict[str, Any]]) -> List[str]:
        """
        Insert multiple documents.

        Args:
            collection: Collection name
            documents: List of documents to insert

        Returns:
            List of inserted _id values as strings
        """
        pass

    @abstractmethod
    def find(self, collection: str,
             query: Dict[str, Any],
             projection: Optional[Dict[str, int]] = None
             ) -> Iterator[Dict[str, Any]]:
        """
        Find documents matching query.

        Args:
            collection: Collection name
            query: MongoDB query document
            projection: Fields to include/exclude

        Yields:
            Matching documents
        """
        pass

    @abstractmethod
    def aggregate(self, collection: str,
                  pipeline: List[Dict[str, Any]]
                  ) -> Iterator[Dict[str, Any]]:
        """
        Execute aggregation pipeline.

        Args:
            collection: Collection name
            pipeline: Aggregation pipeline stages

        Yields:
            Aggregation results
        """
        pass


# =============================================================================
# HELPER: resolve dotted path in nested dict
# =============================================================================

def _get_nested(doc: Dict[str, Any], path: str) -> Any:
    """Resolve a dotted path like 'demographics.province' in a nested dict.

    Returns a sentinel _MISSING if any key along the path is absent.
    """
    parts = path.split(".")
    current = doc
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return _MISSING
    return current


# Sentinel for missing values
class _MissingSentinel:
    """Sentinel object indicating a missing value during nested lookup."""
    pass

_MISSING = _MissingSentinel()


def _match_query(doc: Dict[str, Any], query: Dict[str, Any]) -> bool:
    """Check if a document matches ALL key-value pairs in a query.

    Supports:
    - Dotted paths: "demographics.province" -> doc["demographics"]["province"]
    - $elemMatch on array fields
    - Simple equality checks
    """
    for key, value in query.items():
        if isinstance(value, dict) and "$elemMatch" in value:
            # $elemMatch: check if any element in the array matches all conditions
            arr = _get_nested(doc, key)
            if isinstance(arr, _MissingSentinel) or not isinstance(arr, list):
                return False
            elem_query = value["$elemMatch"]
            found = False
            for elem in arr:
                if isinstance(elem, dict) and all(
                    elem.get(ek) == ev for ek, ev in elem_query.items()
                ):
                    found = True
                    break
            if not found:
                return False
        else:
            doc_value = _get_nested(doc, key)
            if isinstance(doc_value, _MissingSentinel):
                return False
            if doc_value != value:
                return False
    return True


def _apply_projection(doc: Dict[str, Any],
                       projection: Optional[Dict[str, int]]) -> Dict[str, Any]:
    """Apply a projection to a document.

    If projection is None or empty, return full doc copy.
    Include fields where value == 1. Always include _id unless explicitly excluded.
    """
    if not projection:
        return dict(doc)

    result = {}
    # Always include _id unless explicitly excluded
    if projection.get("_id", 1) != 0 and "_id" in doc:
        result["_id"] = doc["_id"]

    for field_path, include in projection.items():
        if field_path == "_id":
            continue
        if include == 1:
            val = _get_nested(doc, field_path)
            if not isinstance(val, _MissingSentinel):
                # For dotted paths, build nested structure
                parts = field_path.split(".")
                if len(parts) == 1:
                    result[field_path] = val
                else:
                    # Build nested dict
                    current = result
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    current[parts[-1]] = val

    return result


# =============================================================================
# MAIN MONGODB CLIENT
# =============================================================================

class MongoClient(BaseMongoClient):
    """
    MongoDB client for grade prediction data operations.

    ┌─────────────────────────────────────────────────────────────┐
    │  USAGE EXAMPLE                                              │
    │                                                             │
    │  config = MongoConfig(                                      │
    │      host='localhost',                                      │
    │      database='grade_prediction'                            │
    │  )                                                          │
    │  client = MongoClient(config)                               │
    │  client.connect()                                           │
    │                                                             │
    │  # Query students                                           │
    │  students = client.find('students', {'demographics.province':│
    │                                       'Ontario'})           │
    │                                                             │
    │  # Get admission rates                                      │
    │  rates = client.get_admission_rates('2023')                 │
    │                                                             │
    │  client.disconnect()                                        │
    └─────────────────────────────────────────────────────────────┘

    Attributes:
        config: MongoConfig with connection parameters
        query_config: QueryConfig for query execution
        _client: PyMongo MongoClient instance
        _db: Database reference
    """

    def __init__(self, config: MongoConfig,
                 query_config: Optional[QueryConfig] = None):
        """
        Initialize MongoDB client.

        Args:
            config: MongoConfig with connection parameters
            query_config: Optional QueryConfig for queries
        """
        self.config = config
        self.query_config = query_config if query_config is not None else QueryConfig()
        self._collections: Dict[str, List[Dict[str, Any]]] = {}
        self._connected = False
        self._id_counter = 0

    def connect(self) -> None:
        """
        Establish connection to MongoDB.

        ┌─────────────────────────────────────────────────────────┐
        │  IMPLEMENTATION STEPS                                   │
        │                                                         │
        │  1. Build connection string:                            │
        │     - Use connection_string if provided                 │
        │     - Otherwise: mongodb://[user:pass@]host:port        │
        │                                                         │
        │  2. Create PyMongo client:                              │
        │     from pymongo import MongoClient                     │
        │     self._client = MongoClient(                         │
        │         connection_string,                              │
        │         maxPoolSize=config.max_pool_size,               │
        │         serverSelectionTimeoutMS=config.timeout_ms      │
        │     )                                                   │
        │                                                         │
        │  3. Verify connection:                                  │
        │     self._client.admin.command('ping')                  │
        │                                                         │
        │  4. Get database reference:                             │
        │     self._db = self._client[config.database]            │
        └─────────────────────────────────────────────────────────┘
        """
        self._connected = True

    def disconnect(self) -> None:
        """Close MongoDB connection and release resources."""
        self._connected = False

    def insert_one(self, collection: str, document: Dict[str, Any]) -> str:
        """
        Insert single document into collection.

        ┌─────────────────────────────────────────────────────────┐
        │  IMPLEMENTATION                                         │
        │                                                         │
        │  1. Get collection: coll = self._db[collection]         │
        │  2. Insert: result = coll.insert_one(document)          │
        │  3. Return: str(result.inserted_id)                     │
        │                                                         │
        │  Error Handling:                                        │
        │  - DuplicateKeyError: Document with same _id exists     │
        │  - WriteError: Write failed                             │
        └─────────────────────────────────────────────────────────┘
        """
        self._id_counter += 1
        _id = str(self._id_counter)
        doc_copy = dict(document)
        doc_copy["_id"] = _id
        self._collections.setdefault(collection, []).append(doc_copy)
        return _id

    def insert_many(self, collection: str,
                    documents: List[Dict[str, Any]]) -> List[str]:
        """
        Insert multiple documents (batch insert).

        For large batches, use ordered=False for parallel inserts
        (faster but unordered, continues on error).
        """
        ids = []
        for doc in documents:
            _id = self.insert_one(collection, doc)
            ids.append(_id)
        return ids

    def find(self, collection: str,
             query: Dict[str, Any],
             projection: Optional[Dict[str, int]] = None,
             sort: Optional[List[Tuple[str, int]]] = None,
             limit: int = 0
             ) -> Iterator[Dict[str, Any]]:
        """
        Find documents matching query.

        ┌─────────────────────────────────────────────────────────┐
        │  FIND OPERATION                                         │
        │                                                         │
        │  Example queries:                                       │
        │                                                         │
        │  # Find all Ontario students                            │
        │  find('students', {'demographics.province': 'Ontario'}) │
        │                                                         │
        │  # Find admitted to UofT CS                             │
        │  find('students', {                                     │
        │      'applications': {                                  │
        │          '$elemMatch': {                                │
        │              'university': 'University of Toronto',     │
        │              'program': 'Computer Science',             │
        │              'outcome': 'admitted'                      │
        │          }                                              │
        │      }                                                  │
        │  })                                                     │
        │                                                         │
        │  # With projection (only return specific fields)        │
        │  find('students',                                       │
        │       {'demographics.province': 'Ontario'},             │
        │       projection={'student_id': 1, 'academics.gpa': 1}) │
        └─────────────────────────────────────────────────────────┘

        Args:
            collection: Collection name
            query: MongoDB query document
            projection: Fields to include (1) or exclude (0)
            sort: List of (field, direction) tuples
            limit: Maximum documents to return (0 = unlimited)

        Yields:
            Matching documents
        """
        docs = self._collections.get(collection, [])

        # Filter matching documents
        matched = [doc for doc in docs if _match_query(doc, query)]

        # Sort if requested
        if sort:
            for sort_field, sort_direction in reversed(sort):
                matched = sorted(
                    matched,
                    key=lambda d, sf=sort_field: (
                        _get_nested(d, sf)
                        if not isinstance(_get_nested(d, sf), _MissingSentinel)
                        else None
                    ),
                    reverse=(sort_direction == -1)
                )

        # Apply limit
        if limit > 0:
            matched = matched[:limit]

        # Apply projection and yield
        for doc in matched:
            yield _apply_projection(doc, projection)

    def find_one(self, collection: str,
                 query: Dict[str, Any],
                 projection: Optional[Dict[str, int]] = None
                 ) -> Optional[Dict[str, Any]]:
        """
        Find single document matching query.

        Returns None if no match found.
        """
        try:
            return next(self.find(collection, query, projection=projection))
        except StopIteration:
            return None

    def aggregate(self, collection: str,
                  pipeline: List[Dict[str, Any]]
                  ) -> Iterator[Dict[str, Any]]:
        """
        Execute aggregation pipeline.

        ┌─────────────────────────────────────────────────────────┐
        │  AGGREGATION PIPELINE STAGES                            │
        │                                                         │
        │  Common stages:                                         │
        │  • $match: Filter documents                             │
        │  • $unwind: Flatten arrays                              │
        │  • $group: Group and aggregate                          │
        │  • $project: Reshape documents                          │
        │  • $lookup: Join with other collections                 │
        │  • $sort: Sort results                                  │
        │  • $limit: Limit output count                           │
        └─────────────────────────────────────────────────────────┘

        Args:
            collection: Collection name
            pipeline: List of pipeline stages

        Yields:
            Aggregation results
        """
        # Start with all docs in the collection (deep copies)
        docs = [dict(d) for d in self._collections.get(collection, [])]

        for stage in pipeline:
            if "$match" in stage:
                query = stage["$match"]
                docs = [d for d in docs if _match_query(d, query)]

            elif "$unwind" in stage:
                field_path = stage["$unwind"]
                # Remove leading "$"
                if field_path.startswith("$"):
                    field_path = field_path[1:]
                unwound = []
                for doc in docs:
                    arr = _get_nested(doc, field_path)
                    if isinstance(arr, _MissingSentinel) or not isinstance(arr, list):
                        continue
                    for item in arr:
                        new_doc = dict(doc)
                        # Set the field to the single item
                        parts = field_path.split(".")
                        if len(parts) == 1:
                            new_doc[field_path] = item
                        else:
                            # Navigate to parent and set
                            current = new_doc
                            for part in parts[:-1]:
                                if part not in current or not isinstance(current[part], dict):
                                    current[part] = {}
                                # Make a copy of intermediate dicts to avoid mutation
                                current[part] = dict(current[part])
                                current = current[part]
                            current[parts[-1]] = item
                        unwound.append(new_doc)
                docs = unwound

            elif "$group" in stage:
                group_spec = stage["$group"]
                group_id_expr = group_spec["_id"]

                groups: Dict[Any, List[Dict[str, Any]]] = {}

                for doc in docs:
                    # Resolve group key
                    if group_id_expr is None:
                        key = None
                    elif isinstance(group_id_expr, str):
                        if group_id_expr.startswith("$"):
                            key = _get_nested(doc, group_id_expr[1:])
                            if isinstance(key, _MissingSentinel):
                                key = None
                        else:
                            key = group_id_expr
                    elif isinstance(group_id_expr, dict):
                        key_parts = {}
                        for k, v in group_id_expr.items():
                            if isinstance(v, str) and v.startswith("$"):
                                resolved = _get_nested(doc, v[1:])
                                key_parts[k] = None if isinstance(resolved, _MissingSentinel) else resolved
                            else:
                                key_parts[k] = v
                        # Convert to hashable tuple
                        key = tuple(sorted(key_parts.items()))
                    else:
                        key = group_id_expr

                    if key not in groups:
                        groups[key] = []
                    groups[key].append(doc)

                # Now compute accumulators for each group
                result_docs = []
                for key, group_docs in groups.items():
                    result = {}
                    # Restore _id
                    if isinstance(group_id_expr, dict):
                        # key is tuple of sorted items, reconstruct dict
                        result["_id"] = dict(key)
                    else:
                        result["_id"] = key

                    for acc_name, acc_expr in group_spec.items():
                        if acc_name == "_id":
                            continue

                        if isinstance(acc_expr, dict):
                            if "$sum" in acc_expr:
                                sum_val = acc_expr["$sum"]
                                if isinstance(sum_val, (int, float)):
                                    result[acc_name] = sum_val * len(group_docs)
                                elif isinstance(sum_val, str) and sum_val.startswith("$"):
                                    total = 0
                                    for d in group_docs:
                                        v = _get_nested(d, sum_val[1:])
                                        if not isinstance(v, _MissingSentinel) and isinstance(v, (int, float)):
                                            total += v
                                    result[acc_name] = total
                                elif isinstance(sum_val, dict):
                                    # Complex expression like $cond
                                    total = 0
                                    for d in group_docs:
                                        total += _eval_expr(sum_val, d)
                                    result[acc_name] = total
                                else:
                                    result[acc_name] = 0

                            elif "$avg" in acc_expr:
                                avg_field = acc_expr["$avg"]
                                if isinstance(avg_field, str) and avg_field.startswith("$"):
                                    values = []
                                    for d in group_docs:
                                        v = _get_nested(d, avg_field[1:])
                                        if not isinstance(v, _MissingSentinel) and isinstance(v, (int, float)):
                                            values.append(v)
                                    result[acc_name] = (sum(values) / len(values)) if values else 0
                                else:
                                    result[acc_name] = 0

                            elif "$min" in acc_expr:
                                min_field = acc_expr["$min"]
                                if isinstance(min_field, str) and min_field.startswith("$"):
                                    values = []
                                    for d in group_docs:
                                        v = _get_nested(d, min_field[1:])
                                        if not isinstance(v, _MissingSentinel) and isinstance(v, (int, float)):
                                            values.append(v)
                                    result[acc_name] = min(values) if values else 0

                            elif "$max" in acc_expr:
                                max_field = acc_expr["$max"]
                                if isinstance(max_field, str) and max_field.startswith("$"):
                                    values = []
                                    for d in group_docs:
                                        v = _get_nested(d, max_field[1:])
                                        if not isinstance(v, _MissingSentinel) and isinstance(v, (int, float)):
                                            values.append(v)
                                    result[acc_name] = max(values) if values else 0

                            elif "$first" in acc_expr:
                                first_field = acc_expr["$first"]
                                if isinstance(first_field, str) and first_field.startswith("$"):
                                    v = _get_nested(group_docs[0], first_field[1:])
                                    result[acc_name] = None if isinstance(v, _MissingSentinel) else v

                            elif "$push" in acc_expr:
                                push_field = acc_expr["$push"]
                                if isinstance(push_field, str) and push_field.startswith("$"):
                                    values = []
                                    for d in group_docs:
                                        v = _get_nested(d, push_field[1:])
                                        if not isinstance(v, _MissingSentinel):
                                            values.append(v)
                                    result[acc_name] = values

                            elif "$count" in acc_expr:
                                result[acc_name] = len(group_docs)
                        else:
                            result[acc_name] = acc_expr

                    result_docs.append(result)
                docs = result_docs

            elif "$project" in stage:
                project_spec = stage["$project"]
                projected = []
                for doc in docs:
                    new_doc = {}
                    for field_name, expr in project_spec.items():
                        if expr == 1:
                            val = _get_nested(doc, field_name)
                            if not isinstance(val, _MissingSentinel):
                                new_doc[field_name] = val
                        elif expr == 0:
                            continue
                        elif isinstance(expr, str) and expr.startswith("$"):
                            val = _get_nested(doc, expr[1:])
                            new_doc[field_name] = None if isinstance(val, _MissingSentinel) else val
                        elif isinstance(expr, dict):
                            new_doc[field_name] = _eval_expr(expr, doc)
                        else:
                            new_doc[field_name] = expr
                    # Always include _id unless explicitly excluded
                    if "_id" not in project_spec and "_id" in doc:
                        new_doc["_id"] = doc["_id"]
                    elif project_spec.get("_id") == 0 and "_id" in new_doc:
                        del new_doc["_id"]
                    projected.append(new_doc)
                docs = projected

            elif "$sort" in stage:
                sort_spec = stage["$sort"]
                for sort_field, sort_direction in reversed(list(sort_spec.items())):
                    docs = sorted(
                        docs,
                        key=lambda d, sf=sort_field: (
                            _get_nested(d, sf)
                            if not isinstance(_get_nested(d, sf), _MissingSentinel)
                            else None
                        ),
                        reverse=(sort_direction == -1)
                    )

            elif "$limit" in stage:
                n = stage["$limit"]
                docs = docs[:n]

        return iter(docs)

    def update_one(self, collection: str,
                   filter: Dict[str, Any],
                   update: Dict[str, Any],
                   upsert: bool = False) -> int:
        """
        Update single document.

        Args:
            collection: Collection name
            filter: Query to match document
            update: Update operations (e.g., {'$set': {...}})
            upsert: Insert if not exists

        Returns:
            Number of modified documents (0 or 1)
        """
        docs = self._collections.get(collection, [])
        for doc in docs:
            if _match_query(doc, filter):
                # Apply $set updates
                if "$set" in update:
                    for key, value in update["$set"].items():
                        parts = key.split(".")
                        if len(parts) == 1:
                            doc[key] = value
                        else:
                            current = doc
                            for part in parts[:-1]:
                                if part not in current or not isinstance(current[part], dict):
                                    current[part] = {}
                                current = current[part]
                            current[parts[-1]] = value
                return 1

        # No match found
        if upsert:
            # Build a new document from filter and $set
            new_doc = dict(filter)
            if "$set" in update:
                for key, value in update["$set"].items():
                    parts = key.split(".")
                    if len(parts) == 1:
                        new_doc[key] = value
                    else:
                        current = new_doc
                        for part in parts[:-1]:
                            if part not in current or not isinstance(current[part], dict):
                                current[part] = {}
                            current = current[part]
                        current[parts[-1]] = value
            self.insert_one(collection, new_doc)
            return 1

        return 0

    def update_many(self, collection: str,
                    filter: Dict[str, Any],
                    update: Dict[str, Any]) -> int:
        """
        Update multiple documents.

        Returns count of modified documents.
        """
        docs = self._collections.get(collection, [])
        count = 0
        for doc in docs:
            if _match_query(doc, filter):
                if "$set" in update:
                    for key, value in update["$set"].items():
                        parts = key.split(".")
                        if len(parts) == 1:
                            doc[key] = value
                        else:
                            current = doc
                            for part in parts[:-1]:
                                if part not in current or not isinstance(current[part], dict):
                                    current[part] = {}
                                current = current[part]
                            current[parts[-1]] = value
                count += 1
        return count

    def delete_many(self, collection: str,
                    filter: Dict[str, Any]) -> int:
        """
        Delete documents matching filter.

        Returns count of deleted documents.
        """
        docs = self._collections.get(collection, [])
        to_keep = []
        deleted = 0
        for doc in docs:
            if _match_query(doc, filter):
                deleted += 1
            else:
                to_keep.append(doc)
        self._collections[collection] = to_keep
        return deleted

    def count_documents(self, collection: str,
                        query: Dict[str, Any] = None) -> int:
        """
        Count documents matching query.

        Args:
            collection: Collection name
            query: Filter query (optional, {} counts all)

        Returns:
            Document count
        """
        if query is None:
            query = {}
        docs = self._collections.get(collection, [])
        return sum(1 for doc in docs if _match_query(doc, query))


# =============================================================================
# HELPER: Evaluate aggregation expressions
# =============================================================================

def _eval_expr(expr: Any, doc: Dict[str, Any]) -> Any:
    """Evaluate a MongoDB-style aggregation expression against a document."""
    if isinstance(expr, (int, float, bool)):
        return expr
    if isinstance(expr, str):
        if expr.startswith("$"):
            val = _get_nested(doc, expr[1:])
            return None if isinstance(val, _MissingSentinel) else val
        return expr
    if not isinstance(expr, dict):
        return expr

    if "$cond" in expr:
        cond = expr["$cond"]
        if isinstance(cond, list) and len(cond) == 3:
            condition, true_val, false_val = cond
        elif isinstance(cond, dict):
            condition = cond.get("if")
            true_val = cond.get("then")
            false_val = cond.get("else")
        else:
            return None

        if _eval_condition(condition, doc):
            return _eval_expr(true_val, doc)
        else:
            return _eval_expr(false_val, doc)

    if "$eq" in expr:
        operands = expr["$eq"]
        left = _eval_expr(operands[0], doc)
        right = _eval_expr(operands[1], doc)
        return left == right

    if "$ne" in expr:
        operands = expr["$ne"]
        left = _eval_expr(operands[0], doc)
        right = _eval_expr(operands[1], doc)
        return left != right

    if "$gt" in expr:
        operands = expr["$gt"]
        left = _eval_expr(operands[0], doc)
        right = _eval_expr(operands[1], doc)
        return left > right if left is not None and right is not None else False

    if "$gte" in expr:
        operands = expr["$gte"]
        left = _eval_expr(operands[0], doc)
        right = _eval_expr(operands[1], doc)
        return left >= right if left is not None and right is not None else False

    if "$lt" in expr:
        operands = expr["$lt"]
        left = _eval_expr(operands[0], doc)
        right = _eval_expr(operands[1], doc)
        return left < right if left is not None and right is not None else False

    if "$lte" in expr:
        operands = expr["$lte"]
        left = _eval_expr(operands[0], doc)
        right = _eval_expr(operands[1], doc)
        return left <= right if left is not None and right is not None else False

    if "$divide" in expr:
        operands = expr["$divide"]
        left = _eval_expr(operands[0], doc)
        right = _eval_expr(operands[1], doc)
        if right and right != 0:
            return left / right
        return 0

    if "$multiply" in expr:
        operands = expr["$multiply"]
        left = _eval_expr(operands[0], doc)
        right = _eval_expr(operands[1], doc)
        return (left or 0) * (right or 0)

    if "$add" in expr:
        operands = expr["$add"]
        return sum(_eval_expr(op, doc) or 0 for op in operands)

    if "$subtract" in expr:
        operands = expr["$subtract"]
        left = _eval_expr(operands[0], doc)
        right = _eval_expr(operands[1], doc)
        return (left or 0) - (right or 0)

    if "$sum" in expr:
        val = expr["$sum"]
        return _eval_expr(val, doc)

    # Field reference within expression dicts -- try resolving values
    # For unknown expression, return as-is
    return expr


def _eval_condition(condition: Any, doc: Dict[str, Any]) -> bool:
    """Evaluate a condition expression to a boolean."""
    if isinstance(condition, bool):
        return condition
    if isinstance(condition, dict):
        result = _eval_expr(condition, doc)
        return bool(result)
    if isinstance(condition, str) and condition.startswith("$"):
        val = _get_nested(doc, condition[1:])
        return bool(val) if not isinstance(val, _MissingSentinel) else False
    return bool(condition)


# =============================================================================
# SPECIALIZED QUERY METHODS
# =============================================================================

class ApplicationQueries:
    """
    High-level query methods for admission data.

    ┌─────────────────────────────────────────────────────────────┐
    │  APPLICATION-SPECIFIC QUERIES                               │
    │                                                             │
    │  Pre-built aggregation pipelines for common queries:        │
    │  • Admission rates by program                               │
    │  • GPA distributions by outcome                             │
    │  • Application trends over time                             │
    │  • Program competitiveness rankings                         │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, client: MongoClient):
        """
        Initialize with MongoDB client.

        Args:
            client: Connected MongoClient instance
        """
        self._client = client

    def get_admission_rates(self, year: int,
                             by: str = 'program'
                             ) -> Dict[str, float]:
        """
        Get admission rates grouped by dimension.

        ┌─────────────────────────────────────────────────────────┐
        │  ADMISSION RATE PIPELINE                                │
        │                                                         │
        │  [                                                      │
        │    { "$unwind": "$applications" },                      │
        │    { "$match": {                                        │
        │        "applications.application_date": {               │
        │          "$gte": "2023-01-01", "$lt": "2024-01-01"      │
        │        }                                                │
        │    }},                                                  │
        │    { "$group": {                                        │
        │        "_id": "$applications.program",                  │
        │        "total": { "$sum": 1 },                          │
        │        "admitted": { "$sum": {                          │
        │          "$cond": [                                     │
        │            { "$eq": ["$applications.outcome",           │
        │                      "admitted"] },                     │
        │            1, 0                                         │
        │          ]                                              │
        │        }}                                               │
        │    }},                                                  │
        │    { "$project": {                                      │
        │        "rate": { "$divide": ["$admitted", "$total"] }   │
        │    }}                                                   │
        │  ]                                                      │
        └─────────────────────────────────────────────────────────┘

        Args:
            year: Academic year to analyze
            by: Grouping dimension ('program', 'university', 'province')

        Returns:
            Dict mapping group values to admission rates
        """
        # Build the field reference for grouping
        if by == 'province':
            group_field = "$demographics.province"
        else:
            group_field = f"$applications.{by}"

        # Build date range strings for the year
        year_start = f"{year}-01-01"
        year_end = f"{year + 1}-01-01"

        pipeline = [
            {"$unwind": "$applications"},
            {"$group": {
                "_id": group_field,
                "total": {"$sum": 1},
                "admitted": {"$sum": {
                    "$cond": [
                        {"$eq": ["$applications.outcome", "admitted"]},
                        1, 0
                    ]
                }}
            }},
            {"$project": {
                "_id": 1,
                "rate": {"$divide": ["$admitted", "$total"]}
            }}
        ]

        results = self._client.aggregate("students", pipeline)
        rates = {}
        for doc in results:
            key = doc.get("_id")
            if key is not None:
                rates[key] = doc.get("rate", 0.0)
        return rates

    def get_gpa_statistics(self, university: str,
                            program: str,
                            outcome: Optional[str] = None
                            ) -> Dict[str, float]:
        """
        Get GPA statistics for a program.

        Returns:
            Dict with 'mean', 'std', 'min', 'max', 'median', 'count'
        """
        # Build query to find students who applied to this university/program
        elem_match = {
            "university": university,
            "program": program
        }
        if outcome is not None:
            elem_match["outcome"] = outcome

        query = {
            "applications": {"$elemMatch": elem_match}
        }

        docs = list(self._client.find("students", query))
        gpas = []
        for doc in docs:
            gpa = _get_nested(doc, "academics.gpa_overall")
            if not isinstance(gpa, _MissingSentinel) and isinstance(gpa, (int, float)):
                gpas.append(gpa)

        if not gpas:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "count": 0
            }

        gpas_arr = np.array(gpas)
        return {
            "mean": float(np.mean(gpas_arr)),
            "std": float(np.std(gpas_arr)),
            "min": float(np.min(gpas_arr)),
            "max": float(np.max(gpas_arr)),
            "median": float(np.median(gpas_arr)),
            "count": len(gpas)
        }

    def get_application_trends(self, university: str,
                                program: str,
                                years: List[int]
                                ) -> Dict[int, Dict[str, int]]:
        """
        Get application and admission counts over years.

        Returns:
            Dict mapping year to {'applications': N, 'admissions': M}
        """
        trends = {}
        for year in years:
            # Count total applications for this university/program in this year
            elem_match_all = {
                "university": university,
                "program": program
            }
            query_all = {
                "applications": {"$elemMatch": elem_match_all}
            }
            all_docs = list(self._client.find("students", query_all))

            applications_count = 0
            admissions_count = 0

            for doc in all_docs:
                apps = doc.get("applications", [])
                if not isinstance(apps, list):
                    continue
                for app in apps:
                    if (app.get("university") == university and
                            app.get("program") == program):
                        app_date = app.get("application_date", "")
                        if isinstance(app_date, str) and app_date.startswith(str(year)):
                            applications_count += 1
                            if app.get("outcome") == "admitted":
                                admissions_count += 1

            trends[year] = {
                "applications": applications_count,
                "admissions": admissions_count
            }

        return trends

    def get_competitive_programs(self, top_n: int = 20,
                                   min_applications: int = 100
                                   ) -> List[Dict[str, Any]]:
        """
        Get most competitive programs by admission rate.

        Returns programs with lowest admission rates (highest competition).

        Args:
            top_n: Number of programs to return
            min_applications: Minimum applications for inclusion

        Returns:
            List of {'program', 'university', 'rate', 'applications'}
        """
        pipeline = [
            {"$unwind": "$applications"},
            {"$group": {
                "_id": {
                    "program": "$applications.program",
                    "university": "$applications.university"
                },
                "total": {"$sum": 1},
                "admitted": {"$sum": {
                    "$cond": [
                        {"$eq": ["$applications.outcome", "admitted"]},
                        1, 0
                    ]
                }}
            }},
            {"$project": {
                "program": "$_id.program",
                "university": "$_id.university",
                "applications": "$total",
                "rate": {"$divide": ["$admitted", "$total"]},
                "_id": 0
            }}
        ]

        results = list(self._client.aggregate("students", pipeline))

        # Filter by min_applications
        results = [r for r in results if r.get("applications", 0) >= min_applications]

        # Sort by rate ascending (most competitive = lowest rate)
        results.sort(key=lambda x: x.get("rate", 1.0))

        return results[:top_n]

    def get_student_applications(self, student_id: str
                                  ) -> List[Dict[str, Any]]:
        """
        Get all applications for a specific student.

        Returns:
            List of application records with outcomes
        """
        doc = self._client.find_one("students", {"student_id": student_id})
        if doc is None:
            return []
        return doc.get("applications", [])

    def find_similar_students(self, student_id: str,
                               limit: int = 10
                               ) -> List[Dict[str, Any]]:
        """
        Find students with similar profiles.

        Matches on:
        - Similar GPA range (+-5%)
        - Same province
        - Similar program interests

        Returns:
            List of similar student records
        """
        doc = self._client.find_one("students", {"student_id": student_id})
        if doc is None:
            return []

        gpa = _get_nested(doc, "academics.gpa_overall")
        if isinstance(gpa, _MissingSentinel) or not isinstance(gpa, (int, float)):
            return []

        gpa_low = gpa - 5.0
        gpa_high = gpa + 5.0

        # Find all students and filter by GPA range (in-memory, educational)
        all_docs = list(self._client.find("students", {}))
        similar = []
        for d in all_docs:
            if d.get("student_id") == student_id:
                continue
            d_gpa = _get_nested(d, "academics.gpa_overall")
            if isinstance(d_gpa, _MissingSentinel) or not isinstance(d_gpa, (int, float)):
                continue
            if gpa_low <= d_gpa <= gpa_high:
                similar.append(d)

        return similar[:limit]


# =============================================================================
# BULK OPERATIONS
# =============================================================================

class BulkOperations:
    """
    Bulk operations for efficient data loading.

    ┌─────────────────────────────────────────────────────────────┐
    │  BULK OPERATIONS FOR ETL                                    │
    │                                                             │
    │  For loading large datasets:                                │
    │  • Batch inserts with configurable size                     │
    │  • Upsert (update or insert) operations                     │
    │  • Bulk updates with matching                               │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, client: MongoClient, batch_size: int = 1000):
        """
        Initialize bulk operations handler.

        Args:
            client: Connected MongoClient
            batch_size: Documents per batch
        """
        self._client = client
        self._batch_size = batch_size

    def bulk_insert(self, collection: str,
                    documents: Iterator[Dict[str, Any]],
                    ordered: bool = False
                    ) -> Dict[str, int]:
        """
        Insert documents in batches.

        ┌─────────────────────────────────────────────────────────┐
        │  BULK INSERT IMPLEMENTATION                             │
        │                                                         │
        │  1. Accumulate documents into batches                   │
        │  2. Insert each batch when full                         │
        │  3. Track successes and failures                        │
        │  4. Return summary statistics                           │
        │                                                         │
        │  With ordered=False:                                    │
        │  - Inserts continue even if one fails                   │
        │  - Faster for large loads                               │
        │  - Good when duplicates are possible                    │
        └─────────────────────────────────────────────────────────┘

        Args:
            collection: Target collection
            documents: Iterator of documents to insert
            ordered: If True, stop on first error

        Returns:
            {'inserted': N, 'errors': M}
        """
        inserted = 0
        errors = 0
        batch = []

        for doc in documents:
            batch.append(doc)
            if len(batch) >= self._batch_size:
                try:
                    self._client.insert_many(collection, batch)
                    inserted += len(batch)
                except Exception:
                    if ordered:
                        errors += len(batch)
                        return {'inserted': inserted, 'errors': errors}
                    errors += len(batch)
                batch = []

        # Insert remaining batch
        if batch:
            try:
                self._client.insert_many(collection, batch)
                inserted += len(batch)
            except Exception:
                errors += len(batch)

        return {'inserted': inserted, 'errors': errors}

    def bulk_upsert(self, collection: str,
                    documents: Iterator[Dict[str, Any]],
                    key_field: str = '_id'
                    ) -> Dict[str, int]:
        """
        Upsert documents (update if exists, insert if not).

        ┌─────────────────────────────────────────────────────────┐
        │  UPSERT OPERATION                                       │
        │                                                         │
        │  For each document:                                     │
        │  1. Try to find by key_field                            │
        │  2. If found: update with new data                      │
        │  3. If not found: insert as new                         │
        │                                                         │
        │  Uses BulkWriteOperation for efficiency                 │
        └─────────────────────────────────────────────────────────┘

        Args:
            collection: Target collection
            documents: Iterator of documents
            key_field: Field to match on (for upsert logic)

        Returns:
            {'inserted': N, 'updated': M, 'errors': K}
        """
        inserted = 0
        updated = 0
        errors = 0

        for doc in documents:
            try:
                key_value = doc.get(key_field)
                if key_value is None:
                    errors += 1
                    continue

                filter_q = {key_field: key_value}
                update_op = {"$set": doc}

                # Check if document exists
                existing = self._client.find_one(collection, filter_q)
                result = self._client.update_one(collection, filter_q, update_op, upsert=True)

                if existing is None:
                    inserted += 1
                else:
                    updated += 1
            except Exception:
                errors += 1

        return {'inserted': inserted, 'updated': updated, 'errors': errors}


# =============================================================================
# INDEX MANAGEMENT
# =============================================================================

class IndexManager:
    """
    Manage MongoDB indexes for query optimization.

    ┌─────────────────────────────────────────────────────────────┐
    │  INDEX MANAGEMENT                                           │
    │                                                             │
    │  Create and manage indexes for:                             │
    │  • Query performance optimization                           │
    │  • Unique constraints                                       │
    │  • Text search                                              │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, client: MongoClient):
        """Initialize with MongoDB client."""
        self._client = client
        self._indexes: Dict[str, List[Dict[str, Any]]] = {}

    def create_admission_indexes(self) -> List[str]:
        """
        Create recommended indexes for admission queries.

        ┌─────────────────────────────────────────────────────────┐
        │  RECOMMENDED INDEXES                                    │
        │                                                         │
        │  students collection:                                   │
        │  • {"student_id": 1} - unique                           │
        │  • {"demographics.province": 1}                         │
        │  • {"applications.university": 1,                       │
        │     "applications.program": 1}                          │
        │  • {"applications.application_date": -1}                │
        │  • {"academics.gpa_overall": -1}                        │
        │                                                         │
        │  programs collection:                                   │
        │  • {"university": 1, "program_name": 1} - unique        │
        │  • {"faculty": 1}                                       │
        └─────────────────────────────────────────────────────────┘

        Returns:
            List of created index names
        """
        index_definitions = [
            {
                "collection": "students",
                "name": "student_id_1",
                "keys": {"student_id": 1},
                "unique": True
            },
            {
                "collection": "students",
                "name": "demographics.province_1",
                "keys": {"demographics.province": 1},
                "unique": False
            },
            {
                "collection": "students",
                "name": "applications.university_1_applications.program_1",
                "keys": {"applications.university": 1, "applications.program": 1},
                "unique": False
            },
            {
                "collection": "students",
                "name": "applications.application_date_-1",
                "keys": {"applications.application_date": -1},
                "unique": False
            },
            {
                "collection": "students",
                "name": "academics.gpa_overall_-1",
                "keys": {"academics.gpa_overall": -1},
                "unique": False
            },
            {
                "collection": "programs",
                "name": "university_1_program_name_1",
                "keys": {"university": 1, "program_name": 1},
                "unique": True
            },
            {
                "collection": "programs",
                "name": "faculty_1",
                "keys": {"faculty": 1},
                "unique": False
            },
        ]

        created_names = []
        for idx_def in index_definitions:
            coll = idx_def["collection"]
            if coll not in self._indexes:
                self._indexes[coll] = []
            self._indexes[coll].append({
                "name": idx_def["name"],
                "keys": idx_def["keys"],
                "unique": idx_def.get("unique", False)
            })
            created_names.append(idx_def["name"])

        return created_names

    def list_indexes(self, collection: str) -> List[Dict[str, Any]]:
        """List all indexes on a collection."""
        return list(self._indexes.get(collection, []))

    def drop_index(self, collection: str, index_name: str) -> None:
        """Drop a specific index."""
        if collection in self._indexes:
            self._indexes[collection] = [
                idx for idx in self._indexes[collection]
                if idx["name"] != index_name
            ]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def build_application_query(university: Optional[str] = None,
                             program: Optional[str] = None,
                             outcome: Optional[str] = None,
                             year: Optional[int] = None,
                             province: Optional[str] = None
                             ) -> Dict[str, Any]:
    """
    Build MongoDB query for application searches.

    ┌─────────────────────────────────────────────────────────────┐
    │  QUERY BUILDER                                              │
    │                                                             │
    │  Combines filters into proper MongoDB query structure:      │
    │                                                             │
    │  Input:                                                     │
    │    university='UofT', program='CS', outcome='admitted'      │
    │                                                             │
    │  Output:                                                    │
    │    {                                                        │
    │      "applications": {                                      │
    │        "$elemMatch": {                                      │
    │          "university": "UofT",                              │
    │          "program": "CS",                                   │
    │          "outcome": "admitted"                              │
    │        }                                                    │
    │      }                                                      │
    │    }                                                        │
    └─────────────────────────────────────────────────────────────┘

    Args:
        university: Filter by university name
        program: Filter by program name
        outcome: Filter by outcome
        year: Filter by application year
        province: Filter by student province

    Returns:
        MongoDB query document
    """
    query: Dict[str, Any] = {}

    # Build $elemMatch for application-level filters
    elem_match: Dict[str, Any] = {}
    if university is not None:
        elem_match["university"] = university
    if program is not None:
        elem_match["program"] = program
    if outcome is not None:
        elem_match["outcome"] = outcome
    if year is not None:
        elem_match["application_date"] = {
            "$gte": f"{year}-01-01",
            "$lt": f"{year + 1}-01-01"
        }

    if elem_match:
        query["applications"] = {"$elemMatch": elem_match}

    # Province is a top-level demographic field
    if province is not None:
        query["demographics.province"] = province

    return query


def document_to_record(doc: Dict[str, Any]) -> StudentRecord:
    """
    Convert MongoDB document to StudentRecord.

    Handles type conversion and missing fields.
    """
    return StudentRecord(
        student_id=doc.get("student_id", ""),
        demographics=doc.get("demographics", {}),
        academics=doc.get("academics", {}),
        applications=doc.get("applications", []),
        metadata=doc.get("metadata", {})
    )


def record_to_document(record: StudentRecord) -> Dict[str, Any]:
    """
    Convert StudentRecord to MongoDB document.

    Prepares record for insertion.
    """
    return {
        "student_id": record.student_id,
        "demographics": record.demographics,
        "academics": record.academics,
        "applications": record.applications,
        "metadata": record.metadata,
    }


# =============================================================================
# TODO LIST FOR IMPLEMENTATION
# =============================================================================
"""
TODO: Implementation Checklist

CONNECTION MANAGEMENT:
□ MongoClient
  - [ ] Implement connect() with connection string building
  - [ ] Implement disconnect() with proper cleanup
  - [ ] Handle authentication (username/password)
  - [ ] Support replica set configuration
  - [ ] Add connection health check method

CRUD OPERATIONS:
□ Insert operations
  - [ ] insert_one() with error handling
  - [ ] insert_many() with batch processing
  - [ ] Handle duplicate key errors

□ Find operations
  - [ ] find() with cursor batching
  - [ ] find_one() optimization
  - [ ] Support projection and sorting
  - [ ] Handle large result sets

□ Update operations
  - [ ] update_one() with upsert option
  - [ ] update_many() for bulk updates
  - [ ] Support various update operators ($set, $push, etc.)

□ Delete operations
  - [ ] delete_many() with safety checks
  - [ ] Add confirmation for large deletes

AGGREGATION:
□ aggregate()
  - [ ] Support allow_disk_use for large pipelines
  - [ ] Handle aggregation cursor properly
  - [ ] Add timeout handling

SPECIALIZED QUERIES:
□ ApplicationQueries
  - [ ] get_admission_rates() pipeline
  - [ ] get_gpa_statistics() with statistical operators
  - [ ] get_application_trends() time series
  - [ ] get_competitive_programs() ranking

BULK OPERATIONS:
□ BulkOperations
  - [ ] bulk_insert() with batching
  - [ ] bulk_upsert() with key matching
  - [ ] Error tracking and reporting

INDEX MANAGEMENT:
□ IndexManager
  - [ ] create_admission_indexes()
  - [ ] list_indexes()
  - [ ] drop_index()
  - [ ] Check for existing indexes before creation

TESTING:
□ Unit tests with mock database
  - [ ] Test CRUD operations
  - [ ] Test aggregation pipelines
  - [ ] Test error handling

□ Integration tests with real MongoDB
  - [ ] Test connection pooling
  - [ ] Test bulk operations performance
  - [ ] Test query performance with indexes

DOCUMENTATION:
□ Add query examples for common use cases
□ Document index recommendations
□ Add performance tuning guide
"""
