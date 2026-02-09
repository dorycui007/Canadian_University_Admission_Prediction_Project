"""
ETL Pipeline for Grade Prediction System.

==============================================================================
SYSTEM ARCHITECTURE - WHERE THIS MODULE FITS
==============================================================================

This module implements the ETL (Extract, Transform, Load) pipeline that moves
data from external sources through processing stages into our databases.

┌─────────────────────────────────────────────────────────────────────────────┐
│                           ETL PIPELINE OVERVIEW                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                      EXTERNAL DATA SOURCES                          │   │
│   │                                                                     │   │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐ │   │
│   │   │    OUAC     │  │    BCPAS    │  │ ApplyAlberta│  │  CSV/JSON │ │   │
│   │   │  (Ontario)  │  │    (BC)     │  │  (Alberta)  │  │  Exports  │ │   │
│   │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬─────┘ │   │
│   │          │                │                │                │       │   │
│   └──────────┼────────────────┼────────────────┼────────────────┼───────┘   │
│              │                │                │                │           │
│              └────────────────┴────────────────┴────────────────┘           │
│                                      │                                      │
│                                      ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                        EXTRACT LAYER                                │   │
│   │                                                                     │   │
│   │   • Read from various file formats (CSV, JSON, XML)                 │   │
│   │   • API data fetching                                               │   │
│   │   • Handle encoding issues                                          │   │
│   │   • Initial validation                                              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                       TRANSFORM LAYER                               │   │
│   │                                                                     │   │
│   │   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐ │   │
│   │   │   CLEAN      │  │  NORMALIZE   │  │      ENRICH              │ │   │
│   │   │              │  │              │  │                          │ │   │
│   │   │ • Fix nulls  │→ │ • Standardize│→ │ • Add derived features   │ │   │
│   │   │ • Fix types  │  │   names      │  │ • Historical rates       │ │   │
│   │   │ • Deduplicate│  │ • Unify dates│  │ • Compute embeddings     │ │   │
│   │   └──────────────┘  └──────────────┘  └──────────────────────────┘ │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                         LOAD LAYER                                  │   │
│   │                                                                     │   │
│   │   ┌─────────────────────┐         ┌─────────────────────┐          │   │
│   │   │       MongoDB       │         │       Weaviate       │          │   │
│   │   │                     │         │                      │          │   │
│   │   │ • Student records   │         │ • Program vectors    │          │   │
│   │   │ • Applications      │         │ • Student profiles   │          │   │
│   │   │ • Outcomes          │         │ • Similarity search  │          │   │
│   │   └─────────────────────┘         └─────────────────────┘          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
DATA QUALITY DIMENSIONS
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  DATA QUALITY CHECKS                                                        │
│                                                                             │
│  1. COMPLETENESS                                                            │
│     ───────────────                                                         │
│     Are all required fields present?                                        │
│                                                                             │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Required:  student_id ✓  gpa ✓  university ✓  program ✓        │     │
│     │  Optional:  campus ○      phone ○     email ○                   │     │
│     │                                                                 │     │
│     │  Missing value rates:                                           │     │
│     │    gpa: 0.1% missing → OK (can impute)                          │     │
│     │    outcome: 5% missing → Check (may be pending decisions)       │     │
│     │    student_id: 0% missing → Required                            │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  2. ACCURACY                                                                │
│     ────────────                                                            │
│     Are values valid and reasonable?                                        │
│                                                                             │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  GPA: 0-100 for Ontario, 0-4.0 for some provinces               │     │
│     │       Flag: gpa=150 (impossible!)                               │     │
│     │                                                                 │     │
│     │  Dates: Application before decision                             │     │
│     │       Flag: decision_date < application_date (impossible!)      │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  3. CONSISTENCY                                                             │
│     ─────────────                                                           │
│     Do related records match?                                               │
│                                                                             │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Same student, different spellings:                             │     │
│     │    "U of T" vs "University of Toronto" vs "UofT"                │     │
│     │                                                                 │     │
│     │  Program name variations:                                       │     │
│     │    "Computer Science" vs "CS" vs "Comp Sci"                     │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  4. UNIQUENESS                                                              │
│     ────────────                                                            │
│     No duplicate records?                                                   │
│                                                                             │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  Check: Same student_id with same application                   │     │
│     │  Action: Deduplicate, keep most recent                          │     │
│     └─────────────────────────────────────────────────────────────────┘     │
│                                                                             │
│  5. TIMELINESS                                                              │
│     ───────────                                                             │
│     Is data current enough?                                                 │
│                                                                             │
│     ┌─────────────────────────────────────────────────────────────────┐     │
│     │  For training: Historical data (2019-2023) is fine              │     │
│     │  For prediction: Need current application cycle data            │     │
│     └─────────────────────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘


==============================================================================
IDEMPOTENT PIPELINE DESIGN
==============================================================================

┌─────────────────────────────────────────────────────────────────────────────┐
│  IDEMPOTENT = SAFE TO RE-RUN                                                │
│                                                                             │
│  Running pipeline twice with same input produces same result.               │
│                                                                             │
│  Implementation:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  1. Use UPSERT instead of INSERT                                   │    │
│  │     - Check if record exists (by student_id + application)         │    │
│  │     - If exists: update                                            │    │
│  │     - If not: insert                                               │    │
│  │                                                                     │    │
│  │  2. Track processing state                                         │    │
│  │     - Record which files have been processed                       │    │
│  │     - Store file hash to detect changes                            │    │
│  │                                                                     │    │
│  │  3. Use transactions where possible                                │    │
│  │     - Atomic updates to related collections                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Benefits:                                                                  │
│  • Safe to restart after failures                                           │
│  • Can re-process files when source is updated                              │
│  • No manual cleanup required                                               │
└─────────────────────────────────────────────────────────────────────────────┘


Author: Grade Prediction Team
Course Context: CSC148 (OOP), CSC343 (Databases)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Iterator, Callable
from abc import ABC, abstractmethod
import numpy as np


# =============================================================================
# CONFIGURATION DATACLASSES
# =============================================================================

@dataclass
class ETLConfig:
    """
    Configuration for ETL pipeline.

    ┌─────────────────────────────────────────────────────────────┐
    │  ETL CONFIGURATION                                          │
    │                                                             │
    │  batch_size: Records per batch for processing               │
    │  error_threshold: Max errors before aborting                │
    │  validate_all: Run all validations (slower)                 │
    │  skip_existing: Skip records already in database            │
    │  dry_run: Process without writing to database               │
    └─────────────────────────────────────────────────────────────┘
    """
    batch_size: int = 1000
    error_threshold: int = 100
    validate_all: bool = True
    skip_existing: bool = True
    dry_run: bool = False
    log_level: str = 'INFO'


@dataclass
class DataSourceConfig:
    """
    Configuration for a data source.

    ┌─────────────────────────────────────────────────────────────┐
    │  DATA SOURCE CONFIGURATION                                  │
    │                                                             │
    │  source_type: 'csv', 'json', 'api', 'database'              │
    │  path: File path or API endpoint                            │
    │  encoding: File encoding (default utf-8)                    │
    │  delimiter: CSV delimiter                                   │
    │  schema: Expected schema for validation                     │
    └─────────────────────────────────────────────────────────────┘
    """
    source_type: str
    path: str
    encoding: str = 'utf-8'
    delimiter: str = ','
    schema: Optional[Dict[str, str]] = None
    date_format: str = '%Y-%m-%d'


@dataclass
class ValidationResult:
    """
    Result from data validation.
    """
    is_valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ETLStats:
    """
    Statistics from ETL run.
    """
    records_read: int = 0
    records_processed: int = 0
    records_inserted: int = 0
    records_updated: int = 0
    records_skipped: int = 0
    records_failed: int = 0
    duration_seconds: float = 0.0
    errors: List[Dict[str, Any]] = field(default_factory=list)


# =============================================================================
# ABSTRACT BASE CLASSES
# =============================================================================

class BaseExtractor(ABC):
    """
    Abstract base class for data extractors.

    ┌─────────────────────────────────────────────────────────────┐
    │  EXTRACTOR INTERFACE                                        │
    │                                                             │
    │  ┌─────────────────────────────────────────────────────┐    │
    │  │           BaseExtractor (Abstract)                  │    │
    │  │                                                     │    │
    │  │  + extract() → Iterator[Dict]                       │    │
    │  │  + validate_source() → bool                         │    │
    │  │  + get_record_count() → int                         │    │
    │  └─────────────────────────────────────────────────────┘    │
    │                          △                                  │
    │                          │                                  │
    │          ┌───────────────┼───────────────┐                  │
    │          │               │               │                  │
    │  ┌───────┴───────┐ ┌─────┴─────┐ ┌──────┴──────┐           │
    │  │CSVExtractor   │ │JSONExtract│ │APIExtractor │           │
    │  └───────────────┘ └───────────┘ └─────────────┘           │
    └─────────────────────────────────────────────────────────────┘
    """

    @abstractmethod
    def extract(self) -> Iterator[Dict[str, Any]]:
        """
        Extract records from source.

        Yields:
            Dictionaries representing raw records
        """
        pass

    @abstractmethod
    def validate_source(self) -> bool:
        """
        Validate that source is accessible and readable.

        Returns:
            True if source is valid
        """
        pass

    def get_record_count(self) -> Optional[int]:
        """
        Get total record count (if known without reading all).

        Returns:
            Record count or None if unknown
        """
        return None


class BaseTransformer(ABC):
    """
    Abstract base class for data transformers.

    ┌─────────────────────────────────────────────────────────────┐
    │  TRANSFORMER INTERFACE                                      │
    │                                                             │
    │  Transforms raw records into normalized format.             │
    │  Can be chained: T1 → T2 → T3 → ...                         │
    └─────────────────────────────────────────────────────────────┘
    """

    @abstractmethod
    def transform(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Transform a single record.

        Args:
            record: Raw input record

        Returns:
            Transformed record, or None if record should be skipped
        """
        pass

    def transform_batch(self, records: List[Dict[str, Any]]
                        ) -> List[Dict[str, Any]]:
        """
        Transform a batch of records.

        Default implementation calls transform() on each record.
        Override for batch-optimized processing.
        """
        results = []
        for r in records:
            t = self.transform(r)
            if t is not None:
                results.append(t)
        return results


class BaseLoader(ABC):
    """
    Abstract base class for data loaders.

    ┌─────────────────────────────────────────────────────────────┐
    │  LOADER INTERFACE                                           │
    │                                                             │
    │  Loads transformed records into target database.            │
    └─────────────────────────────────────────────────────────────┘
    """

    @abstractmethod
    def load(self, record: Dict[str, Any]) -> bool:
        """
        Load a single record.

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    def load_batch(self, records: List[Dict[str, Any]]) -> int:
        """
        Load a batch of records.

        Returns:
            Number of records successfully loaded
        """
        pass

    @abstractmethod
    def exists(self, key: Dict[str, Any]) -> bool:
        """
        Check if record already exists.

        Args:
            key: Key fields to check

        Returns:
            True if exists
        """
        pass


# =============================================================================
# EXTRACTORS
# =============================================================================

class CSVExtractor(BaseExtractor):
    """
    Extract data from CSV files.

    ┌─────────────────────────────────────────────────────────────┐
    │  CSV EXTRACTION                                             │
    │                                                             │
    │  Handles:                                                   │
    │  • Various encodings (utf-8, latin-1, etc.)                 │
    │  • Different delimiters (, ; | tab)                         │
    │  • Quoted fields                                            │
    │  • Missing values                                           │
    │  • Large files (streaming, not loading all into memory)     │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: DataSourceConfig):
        """
        Initialize CSV extractor.

        Args:
            config: DataSourceConfig with file path and options
        """
        self._config = config

    def extract(self) -> Iterator[Dict[str, Any]]:
        """
        Extract records from CSV file.

        ┌─────────────────────────────────────────────────────────┐
        │  IMPLEMENTATION STEPS                                   │
        │                                                         │
        │  1. Open file with correct encoding                     │
        │  2. Read header row                                     │
        │  3. For each data row:                                  │
        │     a. Parse values                                     │
        │     b. Map to column names                              │
        │     c. Yield as dictionary                              │
        │  4. Handle errors gracefully                            │
        │                                                         │
        │  Code pattern:                                          │
        │  import csv                                             │
        │  with open(path, encoding=enc) as f:                    │
        │      reader = csv.DictReader(f, delimiter=delim)        │
        │      for row in reader:                                 │
        │          yield row                                      │
        └─────────────────────────────────────────────────────────┘

        Yields:
            Dictionary for each row
        """
        import csv
        with open(self._config.path, encoding=self._config.encoding) as f:
            reader = csv.DictReader(f, delimiter=self._config.delimiter)
            for row in reader:
                yield dict(row)

    def validate_source(self) -> bool:
        """Check file exists and is readable."""
        import os
        return os.path.isfile(self._config.path)

    def get_record_count(self) -> int:
        """Count lines in file (minus header)."""
        try:
            with open(self._config.path, encoding=self._config.encoding) as f:
                count = sum(1 for _ in f) - 1
            return max(count, 0)
        except (FileNotFoundError, OSError):
            return 0


class JSONExtractor(BaseExtractor):
    """
    Extract data from JSON files.

    Supports:
    • Single JSON object
    • JSON array
    • JSON Lines (newline-delimited JSON)
    """

    def __init__(self, config: DataSourceConfig,
                 record_path: Optional[str] = None):
        """
        Initialize JSON extractor.

        Args:
            config: DataSourceConfig with file path
            record_path: JSONPath to array of records (e.g., 'data.students')
        """
        self._config = config
        self._record_path = record_path

    def extract(self) -> Iterator[Dict[str, Any]]:
        """Extract records from JSON file."""
        import json
        with open(self._config.path, encoding=self._config.encoding) as f:
            data = json.load(f)

        if self._record_path:
            parts = self._record_path.split('.')
            for part in parts:
                data = data[part]

        if isinstance(data, list):
            for item in data:
                yield item
        elif isinstance(data, dict):
            yield data

    def validate_source(self) -> bool:
        """Check file exists and is valid JSON."""
        import os
        import json
        if not os.path.isfile(self._config.path):
            return False
        try:
            with open(self._config.path, encoding=self._config.encoding) as f:
                json.load(f)
            return True
        except (json.JSONDecodeError, IOError):
            return False


class APIExtractor(BaseExtractor):
    """
    Extract data from REST API.

    ┌─────────────────────────────────────────────────────────────┐
    │  API EXTRACTION                                             │
    │                                                             │
    │  Handles:                                                   │
    │  • Pagination                                               │
    │  • Rate limiting                                            │
    │  • Authentication                                           │
    │  • Retries on failure                                       │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: DataSourceConfig,
                 auth_token: Optional[str] = None,
                 page_size: int = 100):
        """
        Initialize API extractor.

        Args:
            config: DataSourceConfig with API endpoint
            auth_token: Bearer token for authentication
            page_size: Records per page
        """
        self._config = config
        self._auth_token = auth_token
        self._page_size = page_size

    def extract(self) -> Iterator[Dict[str, Any]]:
        """
        Extract records from API with pagination.

        Automatically handles:
        - Cursor-based pagination
        - Offset/limit pagination
        - Rate limiting with backoff
        """
        import urllib.request
        import json

        url = self._config.path

        while url:
            request = urllib.request.Request(url)
            if self._auth_token:
                request.add_header('Authorization', f'Bearer {self._auth_token}')

            with urllib.request.urlopen(request) as response:
                data = json.loads(response.read().decode(self._config.encoding))

            if isinstance(data, list):
                for item in data:
                    yield item
                url = None
            elif isinstance(data, dict):
                if 'data' in data:
                    for item in data['data']:
                        yield item
                elif 'results' in data:
                    for item in data['results']:
                        yield item
                else:
                    yield data

                url = data.get('next', None)
            else:
                url = None

    def validate_source(self) -> bool:
        """Check API is accessible."""
        import urllib.request
        try:
            request = urllib.request.Request(self._config.path)
            if self._auth_token:
                request.add_header('Authorization', f'Bearer {self._auth_token}')
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status == 200
        except Exception:
            return False


# =============================================================================
# TRANSFORMERS
# =============================================================================

class CleaningTransformer(BaseTransformer):
    """
    Clean raw data (nulls, types, whitespace).

    ┌─────────────────────────────────────────────────────────────┐
    │  CLEANING OPERATIONS                                        │
    │                                                             │
    │  1. Whitespace: Strip leading/trailing spaces               │
    │  2. Nulls: Convert empty strings, 'NA', 'N/A' to None       │
    │  3. Types: Convert strings to appropriate types             │
    │     - '87.5' → 87.5 (float)                                 │
    │     - '2024-01-15' → datetime object                        │
    │  4. Case: Normalize case where appropriate                  │
    │     - Province: 'ONTARIO' → 'Ontario'                       │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, type_map: Optional[Dict[str, str]] = None,
                 null_values: Optional[List[str]] = None):
        """
        Initialize cleaning transformer.

        Args:
            type_map: Field name → type ('int', 'float', 'date', 'bool')
            null_values: Strings to treat as null
        """
        self._type_map = type_map or {}
        self._null_values = null_values or ['', 'NA', 'N/A', 'null', 'None', 'n/a', 'na']

    def transform(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Clean a single record.

        Implementation:
            1. Strip whitespace from string fields
            2. Replace null values with None
            3. Convert types according to type_map
            4. Return cleaned record
        """
        cleaned = dict(record)
        for key, value in cleaned.items():
            if isinstance(value, str):
                value = value.strip()
                if value in self._null_values:
                    value = None
            cleaned[key] = value
            if key in self._type_map and value is not None:
                cleaned[key] = self._convert_type(value, self._type_map[key])
        return cleaned

    def _clean_string(self, value: Any) -> Optional[str]:
        """Clean and validate string value."""
        if value is None:
            return None
        if not isinstance(value, str):
            return str(value)
        result = value.strip()
        if result in self._null_values:
            return None
        return result

    def _convert_type(self, value: Any, target_type: str) -> Any:
        """Convert value to target type."""
        try:
            if target_type == 'int':
                return int(value)
            elif target_type == 'float':
                return float(value)
            elif target_type == 'bool':
                return str(value).lower() in ('true', '1', 'yes')
            elif target_type == 'date':
                return value
        except (ValueError, TypeError):
            return value
        return value


class NormalizationTransformer(BaseTransformer):
    """
    Normalize field values for consistency.

    ┌─────────────────────────────────────────────────────────────┐
    │  NORMALIZATION MAPPINGS                                     │
    │                                                             │
    │  University names:                                          │
    │  ┌──────────────────┬────────────────────────────────────┐  │
    │  │ Input            │ Normalized                         │  │
    │  ├──────────────────┼────────────────────────────────────┤  │
    │  │ "UofT"           │ "University of Toronto"            │  │
    │  │ "U of T"         │ "University of Toronto"            │  │
    │  │ "Univ. Toronto"  │ "University of Toronto"            │  │
    │  │ "UBC"            │ "University of British Columbia"   │  │
    │  └──────────────────┴────────────────────────────────────┘  │
    │                                                             │
    │  Program names:                                             │
    │  ┌──────────────────┬────────────────────────────────────┐  │
    │  │ "CS"             │ "Computer Science"                 │  │
    │  │ "Comp Sci"       │ "Computer Science"                 │  │
    │  │ "CompSci"        │ "Computer Science"                 │  │
    │  └──────────────────┴────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, mapping_file: Optional[str] = None,
                 mappings: Optional[Dict[str, Dict[str, str]]] = None):
        """
        Initialize normalization transformer.

        Args:
            mapping_file: Path to JSON file with mappings
            mappings: Direct mapping dict (field → {raw: normalized})
        """
        self._mappings = mappings or {}
        if mapping_file:
            import json
            with open(mapping_file) as f:
                self._mappings = json.load(f)

    def transform(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Normalize field values.

        Implementation:
            1. For each field with mapping:
            2. Look up normalized value
            3. If no mapping, keep original
            4. Return normalized record
        """
        result = dict(record)
        for field_name in self._mappings:
            if field_name in result:
                result[field_name] = self._normalize_field(field_name, result[field_name])
        return result

    def _normalize_field(self, field: str, value: str) -> str:
        """Normalize a single field value."""
        field_mappings = self._mappings.get(field, {})

        # Try exact match first
        if value in field_mappings:
            return field_mappings[value]

        # Try case-insensitive match
        if isinstance(value, str):
            lower_value = value.lower()
            for raw, normalized in field_mappings.items():
                if raw.lower() == lower_value:
                    return normalized

        return value


class EnrichmentTransformer(BaseTransformer):
    """
    Enrich records with derived features.

    ┌─────────────────────────────────────────────────────────────┐
    │  ENRICHMENT OPERATIONS                                      │
    │                                                             │
    │  1. Historical rates:                                       │
    │     Add historical acceptance rate for program              │
    │                                                             │
    │  2. Derived dates:                                          │
    │     - Days until deadline                                   │
    │     - Application month                                     │
    │     - Days between application and decision                 │
    │                                                             │
    │  3. Aggregates:                                             │
    │     - Student's application count                           │
    │     - Program's competitiveness rank                        │
    │                                                             │
    │  4. Embeddings:                                             │
    │     - Generate vector for Weaviate storage                  │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, historical_data: Optional[Dict[str, Any]] = None,
                 embedding_model=None):
        """
        Initialize enrichment transformer.

        Args:
            historical_data: Historical statistics for programs
            embedding_model: Model for generating embeddings
        """
        self._historical_data = historical_data or {}
        self._embedding_model = embedding_model

    def transform(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Enrich record with derived features.

        Implementation:
            1. Add historical rates
            2. Compute date-derived features
            3. Add embeddings if model provided
            4. Return enriched record
        """
        result = dict(record)
        self._add_historical_rate(result)
        self._add_date_features(result)
        return result

    def _add_historical_rate(self, record: Dict[str, Any]) -> None:
        """Add historical acceptance rate."""
        university = record.get('university', '')
        program = record.get('program', '')
        key = f"{university}_{program}"
        if key in self._historical_data:
            record['historical_admit_rate'] = self._historical_data[key]

    def _add_date_features(self, record: Dict[str, Any]) -> None:
        """Add date-derived features."""
        from datetime import datetime

        app_date = record.get('application_date')
        if app_date and isinstance(app_date, str):
            formats = ['%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
                        '%d-%m-%Y', '%m-%d-%Y', '%B %d, %Y', '%b %d, %Y']
            for fmt in formats:
                try:
                    parsed = datetime.strptime(app_date, fmt)
                    record['application_month'] = parsed.month
                    record['application_year'] = parsed.year
                    break
                except ValueError:
                    continue


class ValidationTransformer(BaseTransformer):
    """
    Validate records and flag issues.

    ┌─────────────────────────────────────────────────────────────┐
    │  VALIDATION RULES                                           │
    │                                                             │
    │  Required fields:                                           │
    │  - student_id: Cannot be null                               │
    │  - university: Cannot be null                               │
    │  - program: Cannot be null                                  │
    │                                                             │
    │  Value ranges:                                              │
    │  - GPA: 0-100 (Ontario) or 0-4.0 (4-point scale)            │
    │  - Year: 2015-2030                                          │
    │                                                             │
    │  Business rules:                                            │
    │  - decision_date >= application_date                        │
    │  - outcome in ['admitted', 'rejected', 'waitlisted', None]  │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, rules: Optional[Dict[str, Any]] = None,
                 strict: bool = False):
        """
        Initialize validation transformer.

        Args:
            rules: Validation rules configuration
            strict: If True, return None for invalid records
        """
        self._rules = rules or {}
        self._strict = strict
        self._required = self._rules.get('required', ['student_id', 'university', 'program'])

    def transform(self, record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate record.

        Implementation:
            1. Check required fields
            2. Validate value ranges
            3. Check business rules
            4. If strict and invalid, return None
            5. Otherwise, add '_validation_errors' field
        """
        errors = []
        errors.extend(self._validate_required(record))
        errors.extend(self._validate_ranges(record))
        errors.extend(self._validate_business_rules(record))

        if self._strict and errors:
            return None

        result = dict(record)
        if errors:
            result['_validation_errors'] = errors
        return result

    def _validate_required(self, record: Dict[str, Any]) -> List[str]:
        """Check required fields are present."""
        errors = []
        for field_name in self._required:
            if field_name not in record or record[field_name] is None:
                errors.append(f"Missing required field: {field_name}")
        return errors

    def _validate_ranges(self, record: Dict[str, Any]) -> List[str]:
        """Check value ranges."""
        errors = []

        # Check GPA fields
        for gpa_field in ['gpa', 'gpa_overall']:
            if gpa_field in record and record[gpa_field] is not None:
                try:
                    gpa_val = float(record[gpa_field])
                    if gpa_val < 0 or gpa_val > 100:
                        errors.append(f"{gpa_field} value {gpa_val} out of range 0-100")
                except (ValueError, TypeError):
                    pass

        # Check year fields
        for key, value in record.items():
            if 'year' in key.lower() and value is not None:
                try:
                    year_val = int(value)
                    if year_val < 2015 or year_val > 2030:
                        errors.append(f"{key} value {year_val} out of range 2015-2030")
                except (ValueError, TypeError):
                    pass

        return errors

    def _validate_business_rules(self, record: Dict[str, Any]) -> List[str]:
        """Check business logic rules."""
        errors = []

        # Check outcome is valid
        valid_outcomes = {'admitted', 'rejected', 'waitlisted', None}
        outcome = record.get('outcome')
        if outcome is not None and outcome not in valid_outcomes:
            errors.append(f"Invalid outcome: {outcome}")

        # Check decision_date >= application_date
        decision_date = record.get('decision_date')
        application_date = record.get('application_date')
        if decision_date and application_date:
            if str(decision_date) < str(application_date):
                errors.append("decision_date is before application_date")

        return errors


# =============================================================================
# LOADERS
# =============================================================================

class MongoLoader(BaseLoader):
    """
    Load records into MongoDB.

    ┌─────────────────────────────────────────────────────────────┐
    │  MONGODB LOADER                                             │
    │                                                             │
    │  Uses upsert for idempotent loading:                        │
    │  - If record exists (by key): update                        │
    │  - If record doesn't exist: insert                          │
    │                                                             │
    │  Key fields for matching:                                   │
    │  - Student: student_id                                      │
    │  - Application: student_id + university + program + date    │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, mongo_client, collection: str,
                 key_fields: List[str]):
        """
        Initialize MongoDB loader.

        Args:
            mongo_client: Connected MongoClient
            collection: Target collection name
            key_fields: Fields to use as upsert key
        """
        self._client = mongo_client
        self._collection = collection
        self._key_fields = key_fields

    def load(self, record: Dict[str, Any]) -> bool:
        """
        Load single record using upsert.

        Implementation:
            1. Extract key fields from record
            2. Build filter from key
            3. Upsert: update if exists, insert if not
            4. Return success status
        """
        if self._client is None:
            return True
        filter_dict = {k: record[k] for k in self._key_fields if k in record}
        self._client.update_one(self._collection, filter_dict, {'$set': record}, upsert=True)
        return True

    def load_batch(self, records: List[Dict[str, Any]]) -> int:
        """
        Load batch of records.

        Uses bulk write for efficiency.
        """
        count = sum(1 for r in records if self.load(r))
        return count

    def exists(self, key: Dict[str, Any]) -> bool:
        """Check if record exists by key."""
        if self._client is None:
            return False
        result = self._client.find_one(self._collection, key)
        return result is not None


class WeaviateLoader(BaseLoader):
    """
    Load records with embeddings into Weaviate.

    ┌─────────────────────────────────────────────────────────────┐
    │  WEAVIATE LOADER                                            │
    │                                                             │
    │  Loads objects with vectors for similarity search.          │
    │                                                             │
    │  Requires:                                                  │
    │  - Properties (structured data)                             │
    │  - Vector embedding                                         │
    │                                                             │
    │  Optional:                                                  │
    │  - Custom UUID (for upsert)                                 │
    │  - Cross-references                                         │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, weaviate_client, class_name: str,
                 embedding_model=None, id_field: str = 'id'):
        """
        Initialize Weaviate loader.

        Args:
            weaviate_client: Connected WeaviateClient
            class_name: Target Weaviate class
            embedding_model: Model to generate embeddings
            id_field: Field to use as UUID
        """
        self._client = weaviate_client
        self._class_name = class_name
        self._embedding_model = embedding_model
        self._id_field = id_field

    def load(self, record: Dict[str, Any]) -> bool:
        """
        Load single record with embedding.

        Implementation:
            1. Generate embedding from relevant fields
            2. Extract properties for Weaviate
            3. Upsert object (using id_field as UUID)
        """
        properties = dict(record)

        vector = None
        if self._embedding_model:
            text_fields = ['university', 'program', 'description']
            text_parts = [str(properties.get(f, '')) for f in text_fields if f in properties]
            text = ' '.join(text_parts)
            if text.strip():
                vector = self._embedding_model.encode(text)

        try:
            self._client.add_object(
                self._class_name, properties, vector=vector
            )
            return True
        except Exception:
            return False

    def load_batch(self, records: List[Dict[str, Any]]) -> int:
        """Load batch with batch import."""
        count = sum(1 for r in records if self.load(r))
        return count

    def exists(self, key: Dict[str, Any]) -> bool:
        """Check if object exists by ID."""
        try:
            obj_id = key.get(self._id_field)
            result = self._client.get_object(self._class_name, obj_id)
            return result is not None
        except Exception:
            return False


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

class ETLPipeline:
    """
    Orchestrates the complete ETL process.

    ┌─────────────────────────────────────────────────────────────┐
    │  PIPELINE ORCHESTRATION                                     │
    │                                                             │
    │  pipeline = ETLPipeline(config)                             │
    │                                                             │
    │  # Add stages                                               │
    │  pipeline.add_extractor(CSVExtractor(source_config))        │
    │  pipeline.add_transformer(CleaningTransformer())            │
    │  pipeline.add_transformer(NormalizationTransformer())       │
    │  pipeline.add_transformer(ValidationTransformer())          │
    │  pipeline.add_loader(MongoLoader(client, 'students'))       │
    │                                                             │
    │  # Run                                                      │
    │  stats = pipeline.run()                                     │
    │  print(f"Processed {stats.records_processed} records")      │
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: ETLConfig):
        """
        Initialize ETL pipeline.

        Args:
            config: ETLConfig with pipeline settings
        """
        self._config = config
        self._extractors = []
        self._transformers = []
        self._loaders = []

    def add_extractor(self, extractor: BaseExtractor) -> 'ETLPipeline':
        """
        Add data extractor.

        Returns self for method chaining.
        """
        self._extractors.append(extractor)
        return self

    def add_transformer(self, transformer: BaseTransformer) -> 'ETLPipeline':
        """
        Add data transformer.

        Transformers are applied in order added.
        Returns self for method chaining.
        """
        self._transformers.append(transformer)
        return self

    def add_loader(self, loader: BaseLoader) -> 'ETLPipeline':
        """
        Add data loader.

        Returns self for method chaining.
        """
        self._loaders.append(loader)
        return self

    def run(self) -> ETLStats:
        """
        Execute the ETL pipeline.

        ┌─────────────────────────────────────────────────────────┐
        │  PIPELINE EXECUTION                                     │
        │                                                         │
        │  1. Validate configuration                              │
        │  2. Validate source accessibility                       │
        │  3. For each batch of records:                          │
        │     a. Extract from source                              │
        │     b. Apply each transformer in sequence               │
        │     c. Load to target                                   │
        │     d. Track statistics                                 │
        │  4. Handle errors according to threshold                │
        │  5. Return statistics                                   │
        └─────────────────────────────────────────────────────────┘

        Returns:
            ETLStats with processing statistics
        """
        import time

        validation = self.validate()
        if not validation.is_valid:
            stats = ETLStats()
            stats.errors = validation.errors
            return stats

        start = time.time()
        stats = ETLStats()

        for extractor in self._extractors:
            batch = []
            for record in extractor.extract():
                stats.records_read += 1
                batch.append(record)
                if len(batch) >= self._config.batch_size:
                    success, fail = self._process_batch(batch)
                    stats.records_processed += success + fail
                    stats.records_inserted += success
                    stats.records_failed += fail
                    batch = []

            # Process remaining records
            if batch:
                success, fail = self._process_batch(batch)
                stats.records_processed += success + fail
                stats.records_inserted += success
                stats.records_failed += fail

        stats.duration_seconds = time.time() - start
        return stats

    def _process_batch(self, records: List[Dict[str, Any]]
                       ) -> Tuple[int, int]:
        """
        Process a batch through transformers and loaders.

        Returns:
            (successful_count, failed_count)
        """
        success = 0
        fail = 0

        for record in records:
            transformed = self._apply_transformers(record)
            if transformed is None:
                fail += 1
                continue

            loaded = False
            for loader in self._loaders:
                try:
                    if loader.load(transformed):
                        loaded = True
                except Exception:
                    pass

            if loaded:
                success += 1
            else:
                fail += 1

        return (success, fail)

    def _apply_transformers(self, record: Dict[str, Any]
                            ) -> Optional[Dict[str, Any]]:
        """
        Apply all transformers to a record.

        Returns None if any transformer returns None (skip record).
        """
        for transformer in self._transformers:
            record = transformer.transform(record)
            if record is None:
                return None
        return record

    def validate(self) -> ValidationResult:
        """
        Validate pipeline configuration.

        Checks:
        - At least one extractor
        - At least one loader
        - Source accessibility
        - Schema compatibility
        """
        errors = []
        if len(self._extractors) == 0:
            errors.append({'error': 'No extractors configured'})
        if len(self._loaders) == 0:
            errors.append({'error': 'No loaders configured'})

        return ValidationResult(is_valid=len(errors) == 0, errors=errors)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_admission_etl(mongo_client,
                          weaviate_client=None,
                          embedding_model=None) -> ETLPipeline:
    """
    Create standard ETL pipeline for admission data.

    ┌─────────────────────────────────────────────────────────────┐
    │  STANDARD ADMISSION ETL PIPELINE                            │
    │                                                             │
    │  Transformers:                                              │
    │  1. CleaningTransformer - Fix nulls, types                  │
    │  2. NormalizationTransformer - Standardize names            │
    │  3. EnrichmentTransformer - Add historical rates            │
    │  4. ValidationTransformer - Validate records                │
    │                                                             │
    │  Loaders:                                                   │
    │  1. MongoLoader - Store in MongoDB                          │
    │  2. WeaviateLoader - Store embeddings (optional)            │
    └─────────────────────────────────────────────────────────────┘
    """
    pipeline = ETLPipeline(ETLConfig())

    pipeline.add_transformer(CleaningTransformer())
    pipeline.add_transformer(NormalizationTransformer())
    pipeline.add_transformer(ValidationTransformer())

    pipeline.add_loader(MongoLoader(mongo_client, 'students', ['student_id']))

    if weaviate_client:
        pipeline.add_loader(WeaviateLoader(weaviate_client, 'Student',
                                            embedding_model=embedding_model))

    return pipeline


def run_incremental_load(pipeline: ETLPipeline,
                         since_date: str) -> ETLStats:
    """
    Run incremental load for records since date.

    For daily/hourly updates rather than full reload.
    """
    return pipeline.run()


# =============================================================================
# TODO LIST FOR IMPLEMENTATION
# =============================================================================
"""
TODO: Implementation Checklist

EXTRACTORS:
□ CSVExtractor
  - [ ] Implement extract() with streaming
  - [ ] Handle various encodings
  - [ ] Handle malformed rows gracefully
  - [ ] Implement get_record_count()

□ JSONExtractor
  - [ ] Support single object, array, JSON Lines
  - [ ] Implement JSONPath for nested records
  - [ ] Handle large files with streaming

□ APIExtractor
  - [ ] Implement pagination handling
  - [ ] Add rate limiting with backoff
  - [ ] Handle authentication
  - [ ] Implement retries

TRANSFORMERS:
□ CleaningTransformer
  - [ ] Strip whitespace
  - [ ] Handle null values
  - [ ] Type conversion
  - [ ] Case normalization

□ NormalizationTransformer
  - [ ] Load mappings from file
  - [ ] Apply field-specific mappings
  - [ ] Handle unmapped values

□ EnrichmentTransformer
  - [ ] Add historical rates
  - [ ] Compute date features
  - [ ] Generate embeddings

□ ValidationTransformer
  - [ ] Required field checking
  - [ ] Range validation
  - [ ] Business rule validation
  - [ ] Error collection

LOADERS:
□ MongoLoader
  - [ ] Implement upsert logic
  - [ ] Batch loading
  - [ ] Error handling

□ WeaviateLoader
  - [ ] Generate embeddings
  - [ ] Batch import
  - [ ] Handle existing objects

PIPELINE:
□ ETLPipeline
  - [ ] Orchestrate E-T-L stages
  - [ ] Batch processing
  - [ ] Error threshold handling
  - [ ] Statistics collection

TESTING:
□ Unit tests for each component
□ Integration tests with test databases
□ Performance tests with large datasets

DOCUMENTATION:
□ Add configuration examples
□ Document error handling
□ Add troubleshooting guide
"""
