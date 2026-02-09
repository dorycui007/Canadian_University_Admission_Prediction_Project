"""
Database Package — Storage and ETL
====================================

MongoDB for structured application records, Weaviate for vector
embeddings and similarity search, and an ETL pipeline to orchestrate
data flow.

Modules:
    mongo           - MongoDB client, queries, and bulk operations
    weaviate_client - Weaviate vector DB client and schema management
    etl             - Extract-Transform-Load pipeline
"""

from .mongo import (
    MongoConfig,
    QueryConfig,
    StudentRecord,
    ApplicationRecord,
    BaseMongoClient,
    MongoClient,
    ApplicationQueries,
    BulkOperations,
    IndexManager,
    build_application_query,
    document_to_record,
    record_to_document,
)

from .weaviate_client import (
    WeaviateConfig,
    SchemaClass,
    SearchResult,
    BatchResult,
    WeaviateClient,
    EmbeddingVectorizer,
    create_program_schema,
    create_student_schema,
    build_filter,
    build_range_filter,
)

from .etl import (
    ETLConfig,
    DataSourceConfig,
    ValidationResult,
    ETLStats,
    BaseExtractor,
    BaseTransformer,
    BaseLoader,
    CSVExtractor,
    JSONExtractor,
    APIExtractor,
    CleaningTransformer,
    NormalizationTransformer,
    EnrichmentTransformer,
    ValidationTransformer,
    MongoLoader,
    WeaviateLoader,
    ETLPipeline,
    create_admission_etl,
    run_incremental_load,
)

__all__ = [
    # mongo
    "MongoConfig",
    "QueryConfig",
    "StudentRecord",
    "ApplicationRecord",
    "BaseMongoClient",
    "MongoClient",
    "ApplicationQueries",
    "BulkOperations",
    "IndexManager",
    "build_application_query",
    "document_to_record",
    "record_to_document",
    # weaviate_client
    "WeaviateConfig",
    "SchemaClass",
    "SearchResult",
    "BatchResult",
    "WeaviateClient",
    "EmbeddingVectorizer",
    "create_program_schema",
    "create_student_schema",
    "build_filter",
    "build_range_filter",
    # etl
    "ETLConfig",
    "DataSourceConfig",
    "ValidationResult",
    "ETLStats",
    "BaseExtractor",
    "BaseTransformer",
    "BaseLoader",
    "CSVExtractor",
    "JSONExtractor",
    "APIExtractor",
    "CleaningTransformer",
    "NormalizationTransformer",
    "EnrichmentTransformer",
    "ValidationTransformer",
    "MongoLoader",
    "WeaviateLoader",
    "ETLPipeline",
    "create_admission_etl",
    "run_incremental_load",
]
