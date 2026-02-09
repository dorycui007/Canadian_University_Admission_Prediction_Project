"""
Data Normalization Module for University Admissions Prediction System
======================================================================

This module implements normalization utilities for cleaning and standardizing
university and program names from raw CSV data sources.

================================================================================
                        SYSTEM ARCHITECTURE CONTEXT
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    WHERE THIS MODULE FITS                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Raw CSV ──► [THIS MODULE] ──► Cleaned CSV ──► MongoDB ──► Models      │
    │                 normalize.py                                             │
    │                     │                                                    │
    │                     ▼                                                    │
    │               ┌───────────────────────────────────────────────────┐     │
    │               │            NORMALIZATION PIPELINE                  │     │
    │               ├───────────────────────────────────────────────────┤     │
    │               │                                                    │     │
    │               │  1. UNIVERSITY NORMALIZATION                       │     │
    │               │     ┌────────────────────────────────────────┐    │     │
    │               │     │ Input: "UofT", "Waterloo", "Ryerson"   │    │     │
    │               │     │                  │                      │    │     │
    │               │     │      ┌───────────┴───────────┐          │    │     │
    │               │     │      ▼                       ▼          │    │     │
    │               │     │  Exact Match?           Fuzzy Match?    │    │     │
    │               │     │      │                       │          │    │     │
    │               │     │      ▼                       ▼          │    │     │
    │               │     │  "University of Toronto"  Score > 85?   │    │     │
    │               │     │                              │          │    │     │
    │               │     │                    Yes ──────┴── No     │    │     │
    │               │     │                     │             │     │    │     │
    │               │     │               Canonical       INVALID   │    │     │
    │               │     └────────────────────────────────────────┘    │     │
    │               │                                                    │     │
    │               │  2. PROGRAM NORMALIZATION                          │     │
    │               │     ┌────────────────────────────────────────┐    │     │
    │               │     │ Input: "BSc Honours: CS (Co-op)"       │    │     │
    │               │     │                  │                      │    │     │
    │               │     │          ┌───────┴────────┐             │    │     │
    │               │     │          ▼                ▼             │    │     │
    │               │     │    Extract Components   Fuzzy Match     │    │     │
    │               │     │    • degree: BSc        Base Name       │    │     │
    │               │     │    • honours: True                      │    │     │
    │               │     │    • coop: True                         │    │     │
    │               │     │    • base: "CS"                         │    │     │
    │               │     │          │                              │    │     │
    │               │     │          ▼                              │    │     │
    │               │     │  "Computer Science | BSc Honours | Co-op"   │     │
    │               │     └────────────────────────────────────────┘    │     │
    │               │                                                    │     │
    │               └───────────────────────────────────────────────────┘     │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                           KEY DESIGN DECISIONS
================================================================================

    1. UofT CAMPUSES: Kept as 3 separate entries
       ─────────────────────────────────────────
       - University of Toronto (St. George)
       - University of Toronto Mississauga
       - University of Toronto Scarborough

       Rationale: They have separate OUAC admission categories and different
       admission requirements/averages.

    2. TORONTO METROPOLITAN UNIVERSITY
       ────────────────────────────────
       All "Ryerson" references map to "Toronto Metropolitan University"
       (name officially changed April 26, 2022)

    3. FUZZY MATCHING THRESHOLD: 85
       ────────────────────────────
       After testing, 85 provides best balance:
       - Catches common typos: "UOFt" → "University of Toronto"
       - Avoids false positives: "Ontario" won't match "University of Ottawa"

    4. PROGRAM OUTPUT FORMAT
       ──────────────────────
       {Base Program} [{Specialization}] | {Degree} [Honours] | [Co-op]

       Examples:
       - Computer Science | BSc Honours | Co-op
       - Mechanical Engineering (Automotive) | BEng | Co-op

================================================================================
                        RAPIDFUZZ FUZZY MATCHING
================================================================================

    This module uses RapidFuzz (MIT license) for fuzzy string matching.
    RapidFuzz is faster than FuzzyWuzzy and provides various scorers:

    ┌────────────────────────────────────────────────────────────────────────┐
    │  SCORER COMPARISON                                                      │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  fuzz.ratio        - Simple Levenshtein ratio                          │
    │                      "UNIVERSITY OF TORONTO" vs "UOFT" → Low score     │
    │                                                                         │
    │  fuzz.partial_ratio - Best partial match                               │
    │                      "UNIVERSITY OF TORONTO" vs "TORONTO" → High score │
    │                                                                         │
    │  fuzz.WRatio       - Weighted combination (RECOMMENDED)                │
    │                      Handles partial, out-of-order, abbreviations      │
    │                                                                         │
    │  fuzz.token_set_ratio - Word-level matching (for programs)             │
    │                      "Computer Science Honours" vs                      │
    │                      "Honours Computer Science" → High score           │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    PROJECT USE:
    ────────────
    • UniversityNormalizer: Uses fuzz.WRatio for abbreviation handling
    • ProgramNormalizer: Uses fuzz.token_set_ratio for word reordering

================================================================================
                        COMPONENT INTERACTIONS
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  HOW NORMALIZATION FLOWS THROUGH THE SYSTEM                             │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  1. DATA INGESTION (scripts/clean_universities.py)                      │
    │     ┌─────────────────────────────────────────────────────────────┐     │
    │     │ Raw: uni="UofT", prog="BSc CS Co-op"                        │     │
    │     │                        │                                    │     │
    │     │                        ▼                                    │     │
    │     │ Clean: uni="University of Toronto"                          │     │
    │     │        prog="Computer Science | BSc | Co-op"                │     │
    │     └─────────────────────────────────────────────────────────────┘     │
    │                              │                                           │
    │                              ▼                                           │
    │  2. FEATURE ENCODING (src/features/encoders.py)                         │
    │     ┌─────────────────────────────────────────────────────────────┐     │
    │     │ Uses normalized names for consistent one-hot encoding       │     │
    │     │ "University of Toronto" → [1, 0, 0, ..., 0]                 │     │
    │     └─────────────────────────────────────────────────────────────┘     │
    │                              │                                           │
    │                              ▼                                           │
    │  3. MODEL TRAINING (src/models/)                                        │
    │     ┌─────────────────────────────────────────────────────────────┐     │
    │     │ Clean data → consistent features → better predictions       │     │
    │     └─────────────────────────────────────────────────────────────┘     │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                           EXTERNAL REFERENCES
================================================================================

    • Universities Canada: https://univcan.ca/universities/member-universities/
    • OUInfo Programs: https://www.ontariouniversitiesinfo.ca/programs
    • OUInfo Degrees: https://www.ontariouniversitiesinfo.ca/programs/degrees
    • RapidFuzz: https://github.com/maxbachmann/RapidFuzz

================================================================================
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

# Note: yaml import is deferred to avoid import errors if pyyaml not installed
# Import yaml in methods that need it: import yaml


@dataclass
class NormalizationStats:
    """
    Statistics container for normalization operations.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  NORMALIZATION STATISTICS                                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Tracks the outcome of each normalization attempt:                       │
    │                                                                          │
    │     matched: 450  ──► Exact case-insensitive matches                    │
    │     fuzzy:    45  ──► Fuzzy matches above threshold                     │
    │     invalid:  11  ──► No match found, marked as INVALID                 │
    │     ──────────────                                                       │
    │     total:   506                                                         │
    │                                                                          │
    │     match_rate: (450 + 45) / 506 = 97.8%                                │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Attributes:
        matched: Count of exact matches (case-insensitive)
        fuzzy: Count of fuzzy matches (above threshold)
        invalid: Count of failed matches
        invalid_values: List of unique values that could not be matched

    Example:
        >>> stats = NormalizationStats()
        >>> stats.matched = 100
        >>> stats.fuzzy = 10
        >>> stats.invalid = 5
        >>> stats.match_rate
        95.65
    """

    matched: int = 0
    fuzzy: int = 0
    invalid: int = 0
    invalid_values: List[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        """
        Calculate total number of normalization attempts.

        Returns:
            Sum of matched, fuzzy, and invalid counts
        """
        return self.matched + self.fuzzy + self.invalid

    @property
    def match_rate(self) -> float:
        """
        Calculate the percentage of successful matches.

        ┌────────────────────────────────────────────────────────────────────┐
        │  MATCH RATE FORMULA                                                 │
        ├────────────────────────────────────────────────────────────────────┤
        │                                                                     │
        │              (matched + fuzzy)                                      │
        │  rate = ─────────────────────── × 100                              │
        │                 total                                               │
        │                                                                     │
        │  Target: > 95% for production quality                              │
        │                                                                     │
        └────────────────────────────────────────────────────────────────────┘

        Returns:
            Match rate as percentage (0-100)
        """
        if self.total == 0:
            return 0.0
        return (self.matched + self.fuzzy) / self.total * 100

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert statistics to dictionary for reporting.

        Returns:
            Dictionary containing all statistics and unique invalid values
        """
        return {
            'total': self.total,
            'matched': self.matched,
            'exact_matches': self.matched,
            'fuzzy': self.fuzzy,
            'fuzzy_matches': self.fuzzy,
            'invalid': self.invalid,
            'match_rate': self.match_rate,
            'invalid_values': list(set(self.invalid_values)),
        }


@dataclass
class ProgramComponents:
    """
    Structured representation of extracted program components.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  PROGRAM COMPONENT EXTRACTION                                            │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  Input: "BSc Honours: Computer Science (Co-op, AI Specialization)"      │
    │                                                                          │
    │  ┌──────────────────────────────────────────────────────────────────┐   │
    │  │  original:       "BSc Honours: Computer Science (Co-op, AI...)"  │   │
    │  │  base_name:      "Computer Science"                              │   │
    │  │  degree:         "BSc"                                           │   │
    │  │  honours:        True                                            │   │
    │  │  coop:           True                                            │   │
    │  │  coop_modifier:  None                                            │   │
    │  │  specialization: "AI Specialization"                             │   │
    │  └──────────────────────────────────────────────────────────────────┘   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Attributes:
        original: The original program name before processing
        base_name: The core program name (e.g., "Computer Science")
        degree: Degree abbreviation (e.g., "BSc", "BEng", "BA")
        honours: Whether the program is an Honours program
        coop: Whether the program includes co-op
        coop_modifier: Optional modifier like "only", "available", "optional"
        specialization: Optional specialization/stream/option

    Example:
        >>> comp = ProgramComponents(
        ...     original="BSc Computer Science Co-op",
        ...     base_name="Computer Science",
        ...     degree="BSc",
        ...     honours=False,
        ...     coop=True
        ... )
        >>> comp.to_normalized_string()
        'Computer Science | BSc | Co-op'
    """

    original: str
    base_name: str
    degree: Optional[str] = None
    honours: bool = False
    coop: bool = False
    coop_modifier: Optional[str] = None
    specialization: Optional[str] = None

    def to_normalized_string(self) -> str:
        """
        Generate normalized program string in standard format.

        ┌────────────────────────────────────────────────────────────────────┐
        │  OUTPUT FORMAT                                                      │
        ├────────────────────────────────────────────────────────────────────┤
        │                                                                     │
        │  {Base Program} [{Specialization}] | {Degree} [Honours] | [Co-op]  │
        │                                                                     │
        │  Examples:                                                          │
        │  • "Computer Science | BSc Honours | Co-op"                        │
        │  • "Mechanical Engineering (Automotive) | BEng"                    │
        │  • "Biology | BSc Honours"                                         │
        │                                                                     │
        └────────────────────────────────────────────────────────────────────┘

        Returns:
            Normalized program string in standardized format
        """
        # Return base name only.
        # Degree, Honours, Co-op, and Specialization are self-reported modifiers
        # that fragment the same program into multiple entries — they are NOT
        # program differentiators for admission statistics purposes.
        return self.base_name

    def to_key(self) -> str:
        """
        Generate lowercase key for clustering and deduplication.

        ┌────────────────────────────────────────────────────────────────────┐
        │  KEY FORMAT                                                         │
        ├────────────────────────────────────────────────────────────────────┤
        │                                                                     │
        │  {base_name_snake_case}|{degree}|{honours}|{coop}                  │
        │                                                                     │
        │  Examples:                                                          │
        │  • "computer_science|bsc|honours|coop"                             │
        │  • "mechanical_engineering|beng||"                                 │
        │                                                                     │
        │  Use Case: Group similar programs for analysis                      │
        │                                                                     │
        └────────────────────────────────────────────────────────────────────┘

        Returns:
            Lowercase key string for clustering
        """
        return self.base_name.lower().replace(" ", "_")


class UniversityNormalizer:
    """
    Normalize university names using exact matching with fuzzy fallback.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  UNIVERSITY NORMALIZER                                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  PURPOSE:                                                                │
    │  ─────────                                                               │
    │  Transform raw university names from various data sources into          │
    │  canonical (official) names for consistent downstream processing.       │
    │                                                                          │
    │  NORMALIZATION STRATEGY:                                                 │
    │  ────────────────────────                                                │
    │                                                                          │
    │      Input: "UofT", "Waterloo", "Ryerson"                               │
    │               │                                                          │
    │               ▼                                                          │
    │      ┌─────────────────────────────────────────┐                        │
    │      │  1. Clean & lowercase input             │                        │
    │      │     "uoft"                              │                        │
    │      └─────────────────────────────────────────┘                        │
    │               │                                                          │
    │               ▼                                                          │
    │      ┌─────────────────────────────────────────┐                        │
    │      │  2. Exact match in reverse lookup?      │                        │
    │      │     "uoft" → "University of Toronto"    │ ──► YES → Return      │
    │      └─────────────────────────────────────────┘                        │
    │               │ NO                                                       │
    │               ▼                                                          │
    │      ┌─────────────────────────────────────────┐                        │
    │      │  3. Fuzzy match with RapidFuzz WRatio   │                        │
    │      │     Score >= 85? → Return canonical     │ ──► YES → Return      │
    │      └─────────────────────────────────────────┘                        │
    │               │ NO                                                       │
    │               ▼                                                          │
    │      ┌─────────────────────────────────────────┐                        │
    │      │  4. Return "INVALID"                    │                        │
    │      │     Log for manual review               │                        │
    │      └─────────────────────────────────────────┘                        │
    │                                                                          │
    │  CONFIGURATION:                                                          │
    │  ───────────────                                                         │
    │  • mapping_path: Path to universities.yaml                              │
    │  • fuzzy_threshold: Minimum score for fuzzy match (default: 85)         │
    │                                                                          │
    │  SPECIAL CASES:                                                          │
    │  ───────────────                                                         │
    │  • "Ryerson" → "Toronto Metropolitan University" (name changed 2022)    │
    │  • "UofT" → "University of Toronto" (main campus)                       │
    │  • "UTM" → "University of Toronto Mississauga" (separate campus)        │
    │  • "UTSC" → "University of Toronto Scarborough" (separate campus)       │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Attributes:
        mapping: Dictionary of canonical names to variations (from YAML)
        reverse_lookup: Dictionary of lowercase variations to canonical names
        fuzzy_threshold: Minimum fuzzy match score (0-100)
        stats: NormalizationStats tracking match outcomes

    Example:
        >>> normalizer = UniversityNormalizer('data/mappings/universities.yaml')
        >>> normalizer.normalize("UofT")
        'University of Toronto'
        >>> normalizer.normalize("Waterloo")
        'University of Waterloo'
        >>> normalizer.normalize("Ryerson")
        'Toronto Metropolitan University'
        >>> normalizer.normalize("Unknown University")
        'INVALID'
        >>> normalizer.get_report()
        {'total': 4, 'exact_matches': 2, 'fuzzy_matches': 1, 'invalid': 1, ...}

    IMPLEMENTATION NOTES:
    ─────────────────────
    • Uses RapidFuzz for fast fuzzy matching (install: pip install rapidfuzz)
    • YAML mapping loaded once at initialization
    • Thread-safe for read operations (stats tracking is not thread-safe)
    """

    # Default fuzzy matching threshold (0-100 scale)
    DEFAULT_THRESHOLD: int = 85

    def __init__(self, mapping_path: str, fuzzy_threshold: int = DEFAULT_THRESHOLD) -> None:
        """
        Initialize the UniversityNormalizer with a mapping file.

        Args:
            mapping_path: Path to YAML file containing university name mappings
            fuzzy_threshold: Minimum score for fuzzy matching (0-100, default: 85)

        Raises:
            FileNotFoundError: If mapping_path does not exist
            yaml.YAMLError: If YAML file is malformed

        Example:
            >>> normalizer = UniversityNormalizer(
            ...     'data/mappings/universities.yaml',
            ...     fuzzy_threshold=80
            ... )

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. Store fuzzy_threshold
        2. Load YAML mapping using _load_mapping()
        3. Build reverse lookup using _build_reverse_lookup()
        4. Initialize stats as NormalizationStats()
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.mapping = self._load_mapping(mapping_path)
        self.reverse_lookup = self._build_reverse_lookup()
        self.stats = NormalizationStats()

    def _load_mapping(self, path: str) -> Dict[str, List[str]]:
        """
        Load university name mapping from YAML file.

        ┌────────────────────────────────────────────────────────────────────┐
        │  YAML FORMAT                                                        │
        ├────────────────────────────────────────────────────────────────────┤
        │                                                                     │
        │  Canonical Name:                                                    │
        │    - variation1                                                     │
        │    - variation2                                                     │
        │    - ...                                                            │
        │                                                                     │
        │  Example:                                                           │
        │  University of Toronto:                                             │
        │    - University of Toronto                                          │
        │    - UofT                                                           │
        │    - U of T                                                         │
        │                                                                     │
        └────────────────────────────────────────────────────────────────────┘

        Args:
            path: Path to YAML mapping file

        Returns:
            Dictionary mapping canonical names to list of variations

        Raises:
            FileNotFoundError: If file does not exist
            yaml.YAMLError: If YAML is malformed

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. Open file with UTF-8 encoding
        2. Parse YAML using yaml.safe_load()
        3. Return parsed dictionary
        """
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _build_reverse_lookup(self) -> Dict[str, str]:
        """
        Build reverse lookup table from variations to canonical names.

        ┌────────────────────────────────────────────────────────────────────┐
        │  REVERSE LOOKUP CONSTRUCTION                                        │
        ├────────────────────────────────────────────────────────────────────┤
        │                                                                     │
        │  Input (self.mapping):                                              │
        │  {                                                                  │
        │      "University of Toronto": ["University of Toronto", "UofT"],   │
        │      "McMaster University": ["McMaster", "Mac"]                    │
        │  }                                                                  │
        │                                                                     │
        │  Output (reverse_lookup):                                           │
        │  {                                                                  │
        │      "university of toronto": "University of Toronto",             │
        │      "uoft": "University of Toronto",                              │
        │      "mcmaster": "McMaster University",                            │
        │      "mac": "McMaster University"                                  │
        │  }                                                                  │
        │                                                                     │
        │  Note: All keys are lowercase and stripped for case-insensitive    │
        │        matching.                                                    │
        │                                                                     │
        └────────────────────────────────────────────────────────────────────┘

        Returns:
            Dictionary mapping lowercase variations to canonical names

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. Initialize empty lookup dictionary
        2. For each canonical name and its variations in self.mapping:
           a. For each variation:
              - Convert to lowercase and strip whitespace
              - Map to canonical name
        3. Return lookup dictionary
        """
        lookup = {}
        for canonical, variations in self.mapping.items():
            for variation in variations:
                lookup[variation.strip().lower()] = canonical
        return lookup

    def normalize(self, name: str) -> str:
        """
        Normalize a university name to its canonical form.

        ┌────────────────────────────────────────────────────────────────────┐
        │  NORMALIZATION PROCESS                                              │
        ├────────────────────────────────────────────────────────────────────┤
        │                                                                     │
        │  Input: "UofT"                                                      │
        │           │                                                         │
        │           ▼                                                         │
        │  ┌───────────────────────────────────────────────────────────────┐ │
        │  │ Step 1: Validate input                                        │ │
        │  │   - If None, empty, or not string → return "INVALID"         │ │
        │  └───────────────────────────────────────────────────────────────┘ │
        │           │                                                         │
        │           ▼                                                         │
        │  ┌───────────────────────────────────────────────────────────────┐ │
        │  │ Step 2: Clean input                                           │ │
        │  │   - Strip whitespace                                          │ │
        │  │   - Convert to lowercase                                      │ │
        │  │   - "UofT" → "uoft"                                          │ │
        │  └───────────────────────────────────────────────────────────────┘ │
        │           │                                                         │
        │           ▼                                                         │
        │  ┌───────────────────────────────────────────────────────────────┐ │
        │  │ Step 3: Exact match                                           │ │
        │  │   - Look up in reverse_lookup                                 │ │
        │  │   - "uoft" → "University of Toronto" ✓                       │ │
        │  └───────────────────────────────────────────────────────────────┘ │
        │           │ (if not found)                                          │
        │           ▼                                                         │
        │  ┌───────────────────────────────────────────────────────────────┐ │
        │  │ Step 4: Fuzzy match with RapidFuzz                            │ │
        │  │   - Use process.extractOne with fuzz.WRatio                  │ │
        │  │   - Score cutoff: self.fuzzy_threshold                       │ │
        │  │   - If match found → return canonical                        │ │
        │  └───────────────────────────────────────────────────────────────┘ │
        │           │ (if not found)                                          │
        │           ▼                                                         │
        │  ┌───────────────────────────────────────────────────────────────┐ │
        │  │ Step 5: Return "INVALID"                                      │ │
        │  │   - Log to stats.invalid_values                              │ │
        │  └───────────────────────────────────────────────────────────────┘ │
        │                                                                     │
        └────────────────────────────────────────────────────────────────────┘

        Args:
            name: Raw university name to normalize

        Returns:
            Canonical university name or "INVALID" if no match found

        Example:
            >>> normalizer.normalize("UofT")
            'University of Toronto'
            >>> normalizer.normalize("Waterloo")
            'University of Waterloo'
            >>> normalizer.normalize("Unknown")
            'INVALID'

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. Validate input (None, empty, non-string → INVALID)
        2. Clean: strip() and lower()
        3. Try exact match in reverse_lookup
        4. If not found, try fuzzy match with RapidFuzz:
           - process.extractOne(cleaned, variations, scorer=fuzz.WRatio,
                               score_cutoff=self.fuzzy_threshold)
        5. If fuzzy match found, return canonical name
        6. Otherwise, record in stats and return "INVALID"
        """
        if not name or not isinstance(name, str):
            self.stats.invalid += 1
            self.stats.invalid_values.append(str(name))
            return "INVALID"

        cleaned = name.strip().lower()

        # Try exact match
        if cleaned in self.reverse_lookup:
            self.stats.matched += 1
            return self.reverse_lookup[cleaned]

        # Try fuzzy match against canonical names
        from rapidfuzz import process, fuzz
        result = process.extractOne(
            cleaned,
            list(self.mapping.keys()),
            scorer=fuzz.WRatio,
            score_cutoff=self.fuzzy_threshold,
        )
        if result is not None:
            canonical, score, idx = result
            self.stats.fuzzy += 1
            return canonical

        # No match
        self.stats.invalid += 1
        self.stats.invalid_values.append(name)
        return "INVALID"

    def normalize_series(self, series: 'pd.Series') -> 'pd.Series':
        """
        Normalize a pandas Series of university names.

        This is a convenience method for batch processing DataFrame columns.

        Args:
            series: Pandas Series containing university names

        Returns:
            New Series with normalized university names

        Example:
            >>> df['university_normalized'] = normalizer.normalize_series(
            ...     df['university']
            ... )

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. Apply self.normalize to each element
        2. Return resulting Series
        """
        return series.apply(self.normalize)

    def get_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive normalization report.

        ┌────────────────────────────────────────────────────────────────────┐
        │  REPORT FORMAT                                                      │
        ├────────────────────────────────────────────────────────────────────┤
        │                                                                     │
        │  {                                                                  │
        │      'total': 506,                                                 │
        │      'exact_matches': 450,                                         │
        │      'fuzzy_matches': 45,                                          │
        │      'invalid': 11,                                                │
        │      'match_rate': 97.8,                                           │
        │      'invalid_values': ['Unknown Uni', 'Some College', ...]        │
        │  }                                                                  │
        │                                                                     │
        └────────────────────────────────────────────────────────────────────┘

        Returns:
            Dictionary containing normalization statistics

        IMPLEMENTATION:
        ────────────────
        Return self.stats.to_dict()
        """
        return self.stats.to_dict()

    def get_canonical_names(self) -> List[str]:
        """
        Get list of all canonical university names.

        Returns:
            List of canonical university names from the mapping

        Example:
            >>> names = normalizer.get_canonical_names()
            >>> 'University of Toronto' in names
            True

        IMPLEMENTATION:
        ────────────────
        Return list(self.mapping.keys())
        """
        return list(self.mapping.keys())

    def get_variations(self, canonical_name: str) -> List[str]:
        """
        Get all known variations for a canonical university name.

        Args:
            canonical_name: The official university name

        Returns:
            List of known variations or empty list if not found

        Example:
            >>> normalizer.get_variations('University of Toronto')
            ['University of Toronto', 'UofT', 'U of T', ...]

        IMPLEMENTATION:
        ────────────────
        Return self.mapping.get(canonical_name, [])
        """
        return self.mapping.get(canonical_name, [])


class ProgramNormalizer:
    """
    Extract and normalize program components using regex patterns and fuzzy matching.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │  PROGRAM NORMALIZER                                                      │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │  PURPOSE:                                                                │
    │  ─────────                                                               │
    │  Transform raw program names from various formats into standardized     │
    │  components (base name, degree, honours, co-op, specialization).        │
    │                                                                          │
    │  NORMALIZATION STAGES:                                                   │
    │  ──────────────────────                                                  │
    │                                                                          │
    │      Input: "BSc Honours: Computer Science (Co-op, AI Stream)"          │
    │               │                                                          │
    │               ▼                                                          │
    │      ┌─────────────────────────────────────────┐                        │
    │      │  STAGE 1: REGEX EXTRACTION              │                        │
    │      │                                          │                        │
    │      │  • Extract degree: "BSc"                 │                        │
    │      │  • Extract honours: True                 │                        │
    │      │  • Extract co-op: True                   │                        │
    │      │  • Extract specialization: "AI Stream"   │                        │
    │      │  • Extract base: "Computer Science"      │                        │
    │      └─────────────────────────────────────────┘                        │
    │               │                                                          │
    │               ▼                                                          │
    │      ┌─────────────────────────────────────────┐                        │
    │      │  STAGE 2: BASE NAME NORMALIZATION       │                        │
    │      │                                          │                        │
    │      │  Fuzzy match "Computer Science" against  │                        │
    │      │  base_programs.yaml mappings             │                        │
    │      │  "comp sci" → "Computer Science"         │                        │
    │      └─────────────────────────────────────────┘                        │
    │               │                                                          │
    │               ▼                                                          │
    │      ┌─────────────────────────────────────────┐                        │
    │      │  STAGE 3: OUTPUT GENERATION             │                        │
    │      │                                          │                        │
    │      │  "Computer Science (AI Stream) |        │                        │
    │      │   BSc Honours | Co-op"                   │                        │
    │      └─────────────────────────────────────────┘                        │
    │                                                                          │
    │  REGEX PATTERNS (from OUInfo):                                          │
    │  ──────────────────────────────                                          │
    │                                                                          │
    │  DEGREE_PATTERNS:                                                        │
    │  ┌──────────────────────────────────────────────────────────────────┐   │
    │  │ 'BSc':  r'\\b(b\\.?sc\\.?|bachelor of science)\\b'               │   │
    │  │ 'BA':   r'\\b(b\\.?a\\.?|bachelor of arts)\\b'                   │   │
    │  │ 'BEng': r'\\b(b\\.?eng\\.?|bachelor of engineering)\\b'          │   │
    │  │ 'BCom': r'\\b(b\\.?com\\.?|bcomm|bachelor of commerce)\\b'       │   │
    │  │ 'BMath':r'\\b(bmath|bachelor of mathematics)\\b'                 │   │
    │  │ ... (see degree_abbreviations.yaml for full list)                │   │
    │  └──────────────────────────────────────────────────────────────────┘   │
    │                                                                          │
    │  COOP_PATTERN:                                                           │
    │  ┌──────────────────────────────────────────────────────────────────┐   │
    │  │ r'(?i)\\b(co-?op|cooperative|coop)\\b'                           │   │
    │  │      (\\s*\\(?(only|available|optional)\\)?)?'                   │   │
    │  └──────────────────────────────────────────────────────────────────┘   │
    │                                                                          │
    │  HONOURS_PATTERN:                                                        │
    │  ┌──────────────────────────────────────────────────────────────────┐   │
    │  │ r'(?i)\\b(honours?|hons?\\.?)\\b'                                │   │
    │  └──────────────────────────────────────────────────────────────────┘   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

    Attributes:
        base_mapping: Dictionary of canonical base names to variations
        degree_patterns: Dictionary of degree abbreviations to regex patterns
        reverse_lookup: Dictionary of lowercase base names to canonical names
        fuzzy_threshold: Minimum fuzzy match score (0-100)
        stats: NormalizationStats tracking match outcomes

    Example:
        >>> normalizer = ProgramNormalizer('data/mappings/base_programs.yaml')
        >>> normalizer.normalize("BSc Honours: Computer Science (Co-op)")
        'Computer Science | BSc Honours | Co-op'
        >>> components = normalizer.extract_components("BEng Mechanical Engineering Co-op")
        >>> components.degree
        'BEng'
        >>> components.base_name
        'Mechanical Engineering'
        >>> components.coop
        True

    IMPLEMENTATION NOTES:
    ─────────────────────
    • Uses regex for component extraction (compile patterns once)
    • Uses RapidFuzz with token_set_ratio for base name matching
    • Thread-safe for read operations (stats tracking is not thread-safe)
    """

    # Default fuzzy matching threshold for base program names
    DEFAULT_THRESHOLD: int = 80

    # Regex patterns for degree extraction (compiled at class level)
    DEGREE_PATTERNS: Dict[str, str] = {
        'BSc': r'\b(b\.?sc\.?|bachelor of science)\b',
        'BA': r'\b(b\.?a\.?|bachelor of arts)\b',
        'BASc': r'\b(b\.?a\.?sc\.?|bachelor of applied science)\b',
        'BEng': r'\b(b\.?eng\.?|bachelor of engineering)\b',
        'BBA': r'\b(bba|bachelor of business administration)\b',
        'BCom': r'\b(b\.?com\.?|bcomm|bachelor of commerce)\b',
        'BCS': r'\b(bcs|b\.?co\.?sc\.?|bachelor of computer science)\b',
        'BScN': r'\b(bscn|b\.?sc\.?n\.?|bachelor of science in nursing)\b',
        'BMath': r'\b(bmath|b\.? ?math\.?|bachelor of mathematics)\b',
        'BKin': r'\b(bkin|b\.? ?kin\.?|bachelor of kinesiology)\b',
        'BHSc': r'\b(bhsc|b\.?h\.?sc\.?|bachelor of health science)\b',
        'BES': r'\b(bes|b\.?e\.?s\.?|bachelor of environmental studies)\b',
        'BSW': r'\b(bsw|b\.?s\.?w\.?|bachelor of social work)\b',
        'BEd': r'\b(bed|b\.?ed\.?|bachelor of education)\b',
        'BFA': r'\b(bfa|b\.?f\.?a\.?|bachelor of fine arts)\b',
        'BMus': r'\b(bmus|b\.?mus\.?|bachelor of music)\b',
    }

    # Co-op detection pattern
    COOP_PATTERN: str = r'(?i)\b(co-?op|cooperative|coop)\b(\s*\(?(only|available|optional)\)?)?'

    # Honours detection pattern
    HONOURS_PATTERN: str = r'(?i)\b(honours?|hons?\.?)\b'

    # Specialization extraction pattern
    SPECIALIZATION_PATTERN: str = (
        r'\(([^)]*(?:stream|option|specialization|concentration)[^)]*)\)|'
        r'\(([^)]*(?:AI|ML|cyber|game|data|software)[^)]*)\)'
    )

    def __init__(
        self,
        base_mapping_path: str,
        degree_mapping_path: Optional[str] = None,
        fuzzy_threshold: int = DEFAULT_THRESHOLD
    ) -> None:
        """
        Initialize the ProgramNormalizer with mapping files.

        Args:
            base_mapping_path: Path to YAML file with base program name mappings
            degree_mapping_path: Optional path to degree abbreviations YAML
            fuzzy_threshold: Minimum score for fuzzy matching (0-100, default: 80)

        Raises:
            FileNotFoundError: If mapping_path does not exist
            yaml.YAMLError: If YAML file is malformed

        Example:
            >>> normalizer = ProgramNormalizer(
            ...     'data/mappings/base_programs.yaml',
            ...     'data/mappings/degree_abbreviations.yaml',
            ...     fuzzy_threshold=75
            ... )

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. Store fuzzy_threshold
        2. Load base program mapping using _load_mapping()
        3. If degree_mapping_path provided, load and update DEGREE_PATTERNS
        4. Build reverse lookup for base names
        5. Compile all regex patterns
        6. Initialize stats as NormalizationStats()
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.base_mapping = self._load_mapping(base_mapping_path)
        if degree_mapping_path:
            self._degree_data = self._load_mapping(degree_mapping_path)
        self.reverse_lookup = self._build_reverse_lookup()
        self.compiled_patterns = self._compile_patterns()
        self.stats = NormalizationStats()

    def _load_mapping(self, path: str) -> Dict[str, List[str]]:
        """
        Load mapping from YAML file.

        Args:
            path: Path to YAML mapping file

        Returns:
            Dictionary mapping canonical names to list of variations

        Raises:
            FileNotFoundError: If file does not exist
            yaml.YAMLError: If YAML is malformed

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. Open file with UTF-8 encoding
        2. Parse YAML using yaml.safe_load()
        3. Return parsed dictionary
        """
        import yaml
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def _build_reverse_lookup(self) -> Dict[str, str]:
        """
        Build reverse lookup table from base name variations to canonical names.

        Returns:
            Dictionary mapping lowercase variations to canonical names

        IMPLEMENTATION STEPS:
        ─────────────────────
        Same as UniversityNormalizer._build_reverse_lookup()
        """
        lookup = {}
        for canonical, variations in self.base_mapping.items():
            for variation in variations:
                lookup[variation.strip().lower()] = canonical
        return lookup

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """
        Compile all regex patterns for efficient repeated matching.

        Returns:
            Dictionary of pattern names to compiled regex Pattern objects

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. Compile each pattern in DEGREE_PATTERNS with re.IGNORECASE
        2. Compile COOP_PATTERN
        3. Compile HONOURS_PATTERN
        4. Compile SPECIALIZATION_PATTERN
        5. Return dictionary of compiled patterns
        """
        compiled = {}
        for degree, pattern in self.DEGREE_PATTERNS.items():
            compiled[f'degree_{degree}'] = re.compile(pattern, re.IGNORECASE)
        compiled['coop'] = re.compile(self.COOP_PATTERN)
        compiled['honours'] = re.compile(self.HONOURS_PATTERN)
        compiled['specialization'] = re.compile(self.SPECIALIZATION_PATTERN, re.IGNORECASE)
        return compiled

    def extract_components(self, name: str) -> ProgramComponents:
        """
        Extract structured components from a program name.

        ┌────────────────────────────────────────────────────────────────────┐
        │  COMPONENT EXTRACTION PROCESS                                       │
        ├────────────────────────────────────────────────────────────────────┤
        │                                                                     │
        │  Input: "BSc Honours: Computer Science (Co-op, AI Stream)"         │
        │                                                                     │
        │  Step 1: Store original                                             │
        │          "BSc Honours: Computer Science (Co-op, AI Stream)"        │
        │                                                                     │
        │  Step 2: Extract degree (and remove from working string)            │
        │          degree = "BSc"                                             │
        │          working = "Honours: Computer Science (Co-op, AI Stream)"  │
        │                                                                     │
        │  Step 3: Extract honours                                            │
        │          honours = True                                             │
        │          working = ": Computer Science (Co-op, AI Stream)"         │
        │                                                                     │
        │  Step 4: Extract co-op                                              │
        │          coop = True                                                │
        │          coop_modifier = None                                       │
        │          working = ": Computer Science (AI Stream)"                │
        │                                                                     │
        │  Step 5: Extract specialization                                     │
        │          specialization = "AI Stream"                               │
        │          working = ": Computer Science"                            │
        │                                                                     │
        │  Step 6: Clean base name                                            │
        │          base_name = "Computer Science"                            │
        │                                                                     │
        │  Output: ProgramComponents(...)                                     │
        │                                                                     │
        └────────────────────────────────────────────────────────────────────┘

        Args:
            name: Raw program name to parse

        Returns:
            ProgramComponents dataclass with extracted fields

        Example:
            >>> comp = normalizer.extract_components("BEng Mechanical Co-op")
            >>> comp.degree
            'BEng'
            >>> comp.coop
            True

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. Handle empty/None input → return empty ProgramComponents
        2. Store original, initialize working copy
        3. Extract degree using DEGREE_PATTERNS (return first match)
        4. Extract co-op using COOP_PATTERN (capture modifier if present)
        5. Extract honours using HONOURS_PATTERN
        6. Extract specialization using SPECIALIZATION_PATTERN
        7. Clean remaining string as base_name using _clean_base_name()
        8. Return ProgramComponents with all fields
        """
        if not name or not isinstance(name, str):
            return ProgramComponents(original=str(name) if name else "", base_name="")

        original = name
        working = name

        # Extract degree
        degree = None
        for deg_abbrev, pattern_str in self.DEGREE_PATTERNS.items():
            pattern = self.compiled_patterns[f'degree_{deg_abbrev}']
            match = pattern.search(working)
            if match:
                degree = deg_abbrev
                working = working[:match.start()] + working[match.end():]
                break

        # Extract co-op
        coop = False
        coop_modifier = None
        coop_match = self.compiled_patterns['coop'].search(working)
        if coop_match:
            coop = True
            if coop_match.group(3):
                coop_modifier = coop_match.group(3).strip()
            working = working[:coop_match.start()] + working[coop_match.end():]

        # Extract honours
        honours = False
        honours_match = self.compiled_patterns['honours'].search(working)
        if honours_match:
            honours = True
            working = working[:honours_match.start()] + working[honours_match.end():]

        # Extract specialization
        specialization = None
        spec_match = self.compiled_patterns['specialization'].search(working)
        if spec_match:
            specialization = spec_match.group(1) or spec_match.group(2)
            if specialization:
                specialization = specialization.strip()
            working = working[:spec_match.start()] + working[spec_match.end():]

        # Clean base name
        base_name = self._clean_base_name(working)

        return ProgramComponents(
            original=original,
            base_name=base_name,
            degree=degree,
            honours=honours,
            coop=coop,
            coop_modifier=coop_modifier,
            specialization=specialization,
        )

    def _clean_base_name(self, name: str) -> str:
        """
        Clean extracted base name by removing artifacts.

        ┌────────────────────────────────────────────────────────────────────┐
        │  BASE NAME CLEANING                                                 │
        ├────────────────────────────────────────────────────────────────────┤
        │                                                                     │
        │  Input: ": Computer Science - "                                    │
        │                                                                     │
        │  Cleaning steps:                                                    │
        │  1. Remove common prefixes: "B.", "Bachelor of"                    │
        │  2. Normalize separators: "-", ":", "," → " "                      │
        │  3. Collapse multiple spaces                                        │
        │  4. Strip whitespace                                                │
        │                                                                     │
        │  Output: "Computer Science"                                        │
        │                                                                     │
        └────────────────────────────────────────────────────────────────────┘

        Args:
            name: Raw base name with potential artifacts

        Returns:
            Cleaned base name string

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. Remove common prefixes with regex
        2. Replace separators with spaces
        3. Collapse whitespace with regex
        4. Strip and return
        """
        # Remove common degree-related prefixes
        cleaned = re.sub(r'(?i)^(bachelors?\s+of\s+)', '', name)
        # Remove degree remnants: (BAH/), (BScH), BAH, BScH
        cleaned = re.sub(r'(?i)\(?\b(bah|bsch|b\.?a\.?h\.?|b\.?sc\.?h\.?)\b\)?/?', '', cleaned)
        # Remove "Faculty of" prefix
        cleaned = re.sub(r'(?i)^faculty\s+of\s+', '', cleaned)
        # Remove campus suffixes
        cleaned = re.sub(r'(?i)\b(main\s+campus|st\.?\s*george\s+campus|waterloo\s+campus|'
                         r'main\s+site|durham\s+campus)\b', '', cleaned)
        # Remove "Regular" / "Regular only"
        cleaned = re.sub(r'(?i)\b(regular\s+only|regular)\b', '', cleaned)
        # Remove "including optional internship"
        cleaned = re.sub(r'(?i)\bincluding\s+optional\s+internship\b', '', cleaned)
        # Remove "(unavailable)"
        cleaned = re.sub(r'(?i)\(unavailable\)', '', cleaned)
        # Remove "(Undeclared)" / "(no concentration)" / "(undecided)"
        cleaned = re.sub(r'(?i)\(\s*(undeclared|no\s+concentration|undecided)\s*\)', '', cleaned)
        # Remove "( and regular)" / "( and Regular)"
        cleaned = re.sub(r'\(\s*and\s+regular\s*\)', '', cleaned)
        # Remove trailing "with" (e.g. "Managment With" → "Managment")
        cleaned = re.sub(r'(?i)\s+with\s*$', '', cleaned)
        # Remove "PEY" / "with PEY" / "with internship" / "with optional PEY"
        cleaned = re.sub(r'(?i)\b(with\s+)?(optional\s+)?(pey|internship)\b', '', cleaned)
        # Remove parenthesized content (specialization remnants after extraction)
        cleaned = re.sub(r'\([^)]*\)', '', cleaned)
        # Replace separators with spaces
        cleaned = re.sub(r'[-:,]', ' ', cleaned)
        # Remove empty parentheses
        cleaned = re.sub(r'\(\s*\)', '', cleaned)
        # Collapse whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        return cleaned.strip()

    def _normalize_base_name(self, name: str) -> str:
        """
        Normalize base program name using mapping and fuzzy matching.

        Args:
            name: Cleaned base program name

        Returns:
            Canonical base program name or original if no match

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. Try exact match in reverse_lookup
        2. If not found, try fuzzy match with token_set_ratio
        3. Return canonical name or original
        """
        if not name:
            return name
        cleaned = name.strip().lower()

        # Exact match
        if cleaned in self.reverse_lookup:
            self.stats.matched += 1
            return self.reverse_lookup[cleaned]

        # Fuzzy match
        from rapidfuzz import process, fuzz
        result = process.extractOne(
            cleaned,
            list(self.reverse_lookup.keys()),
            scorer=fuzz.token_set_ratio,
            score_cutoff=self.fuzzy_threshold,
        )
        if result is not None:
            match_key, score, idx = result
            self.stats.fuzzy += 1
            return self.reverse_lookup[match_key]

        # No match — title-case the original for consistent display
        return name.title()

    def normalize(self, name: str) -> str:
        """
        Normalize a program name to standardized format.

        ┌────────────────────────────────────────────────────────────────────┐
        │  FULL NORMALIZATION PIPELINE                                        │
        ├────────────────────────────────────────────────────────────────────┤
        │                                                                     │
        │  Input: "BSc Honours: Computer Science (Co-op)"                    │
        │           │                                                         │
        │           ▼                                                         │
        │  ┌───────────────────────────────────────────────────────────────┐ │
        │  │ Step 1: Extract components                                    │ │
        │  │   degree="BSc", honours=True, coop=True,                     │ │
        │  │   base_name="Computer Science"                                │ │
        │  └───────────────────────────────────────────────────────────────┘ │
        │           │                                                         │
        │           ▼                                                         │
        │  ┌───────────────────────────────────────────────────────────────┐ │
        │  │ Step 2: Normalize base name                                   │ │
        │  │   "Computer Science" → "Computer Science" (exact match)       │ │
        │  └───────────────────────────────────────────────────────────────┘ │
        │           │                                                         │
        │           ▼                                                         │
        │  ┌───────────────────────────────────────────────────────────────┐ │
        │  │ Step 3: Generate output format                                │ │
        │  │   "Computer Science | BSc Honours | Co-op"                   │ │
        │  └───────────────────────────────────────────────────────────────┘ │
        │                                                                     │
        └────────────────────────────────────────────────────────────────────┘

        Args:
            name: Raw program name to normalize

        Returns:
            Normalized program string in format:
            "{Base} [{Spec}] | {Degree} [Honours] | [Co-op]"

        Example:
            >>> normalizer.normalize("BSc Computer Science Co-op")
            'Computer Science | BSc | Co-op'

        IMPLEMENTATION STEPS:
        ─────────────────────
        1. Extract components using extract_components()
        2. Normalize base_name using _normalize_base_name()
        3. Build output using ProgramComponents.to_normalized_string()
        4. Update stats
        5. Return normalized string
        """
        components = self.extract_components(name)
        components.base_name = self._normalize_base_name(components.base_name)
        return components.to_normalized_string()

    def normalize_series(self, series: 'pd.Series') -> 'pd.Series':
        """
        Normalize a pandas Series of program names.

        Args:
            series: Pandas Series containing program names

        Returns:
            New Series with normalized program names

        Example:
            >>> df['program_normalized'] = normalizer.normalize_series(
            ...     df['program']
            ... )

        IMPLEMENTATION:
        ────────────────
        Return series.apply(self.normalize)
        """
        return series.apply(self.normalize)

    def to_key(self, name: str) -> str:
        """
        Generate a clustering key for a program name.

        This key is used for grouping similar programs during the
        clustering/deduplication phase.

        Args:
            name: Raw program name

        Returns:
            Lowercase clustering key

        Example:
            >>> normalizer.to_key("BSc Honours Computer Science Co-op")
            'computer_science|bsc|honours|coop'

        IMPLEMENTATION:
        ────────────────
        Return self.extract_components(name).to_key()
        """
        return self.extract_components(name).to_key()

    def get_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive normalization report.

        Returns:
            Dictionary containing normalization statistics

        IMPLEMENTATION:
        ────────────────
        Return self.stats.to_dict()
        """
        return self.stats.to_dict()


# =============================================================================
#                           MODULE FUNCTIONS
# =============================================================================

def create_university_normalizer(
    mapping_path: str = 'data/mappings/universities.yaml',
    fuzzy_threshold: int = 85
) -> UniversityNormalizer:
    """
    Factory function to create a UniversityNormalizer with default settings.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  FACTORY FUNCTION                                                       │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Provides a convenient way to create a normalizer with sensible        │
    │  defaults for the project.                                              │
    │                                                                         │
    │  Default mapping: data/mappings/universities.yaml                      │
    │  Default threshold: 85 (recommended for university names)              │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Args:
        mapping_path: Path to university mapping YAML file
        fuzzy_threshold: Minimum fuzzy match score (0-100)

    Returns:
        Configured UniversityNormalizer instance

    Example:
        >>> normalizer = create_university_normalizer()
        >>> normalizer.normalize("UofT")
        'University of Toronto'
    """
    return UniversityNormalizer(mapping_path, fuzzy_threshold=fuzzy_threshold)


def create_program_normalizer(
    base_mapping_path: str = 'data/mappings/base_programs.yaml',
    degree_mapping_path: str = 'data/mappings/degree_abbreviations.yaml',
    fuzzy_threshold: int = 80
) -> ProgramNormalizer:
    """
    Factory function to create a ProgramNormalizer with default settings.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  FACTORY FUNCTION                                                       │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Provides a convenient way to create a normalizer with sensible        │
    │  defaults for the project.                                              │
    │                                                                         │
    │  Default base mapping: data/mappings/base_programs.yaml                │
    │  Default degree mapping: data/mappings/degree_abbreviations.yaml       │
    │  Default threshold: 80 (lower than universities due to more variation) │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Args:
        base_mapping_path: Path to base program mapping YAML file
        degree_mapping_path: Path to degree abbreviations YAML file
        fuzzy_threshold: Minimum fuzzy match score (0-100)

    Returns:
        Configured ProgramNormalizer instance

    Example:
        >>> normalizer = create_program_normalizer()
        >>> normalizer.normalize("BSc CS Co-op")
        'Computer Science | BSc | Co-op'
    """
    return ProgramNormalizer(
        base_mapping_path,
        degree_mapping_path=degree_mapping_path,
        fuzzy_threshold=fuzzy_threshold,
    )


# =============================================================================
#                           TODO LIST
# =============================================================================
"""
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TODO                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HIGH PRIORITY:                                                              │
│  ─────────────                                                               │
│  [ ] Implement NormalizationStats.total property                            │
│  [ ] Implement NormalizationStats.match_rate property                       │
│  [ ] Implement NormalizationStats.to_dict()                                 │
│  [ ] Implement ProgramComponents.to_normalized_string()                     │
│  [ ] Implement ProgramComponents.to_key()                                   │
│  [ ] Implement UniversityNormalizer.__init__()                              │
│  [ ] Implement UniversityNormalizer._load_mapping()                         │
│  [ ] Implement UniversityNormalizer._build_reverse_lookup()                 │
│  [ ] Implement UniversityNormalizer.normalize()                             │
│                                                                              │
│  MEDIUM PRIORITY:                                                            │
│  ────────────────                                                            │
│  [ ] Implement ProgramNormalizer.__init__()                                 │
│  [ ] Implement ProgramNormalizer.extract_components()                       │
│  [ ] Implement ProgramNormalizer._clean_base_name()                         │
│  [ ] Implement ProgramNormalizer._normalize_base_name()                     │
│  [ ] Implement ProgramNormalizer.normalize()                                │
│                                                                              │
│  LOW PRIORITY:                                                               │
│  ─────────────                                                               │
│  [ ] Implement normalize_series() methods                                   │
│  [ ] Implement factory functions                                             │
│  [ ] Implement getter methods (get_report, get_canonical_names, etc.)       │
│                                                                              │
│  TESTING:                                                                    │
│  ────────                                                                    │
│  [ ] Write unit tests in tests/test_utils/test_normalize.py                │
│  [ ] Test edge cases: empty strings, None, unusual characters               │
│  [ ] Test fuzzy matching with various threshold values                       │
│  [ ] Verify all UofT campus names are handled correctly                     │
│  [ ] Verify Ryerson → Toronto Metropolitan mapping                          │
│                                                                              │
│  ENHANCEMENTS (OPTIONAL):                                                    │
│  ─────────────────────────                                                   │
│  [ ] Add caching for repeated normalizations                                │
│  [ ] Add thread-safe stats tracking                                          │
│  [ ] Add logging for debugging                                               │
│  [ ] Add confidence scores to output                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    # Quick self-test - uncomment after implementing functions
    print("Testing normalization utilities...")
    print("Implement the classes above and then run this test!")

    # Example test code (uncomment after implementation):
    #
    # # Test UniversityNormalizer
    # uni_norm = UniversityNormalizer('data/mappings/universities.yaml')
    # print(f"UofT → {uni_norm.normalize('UofT')}")
    # print(f"Waterloo → {uni_norm.normalize('Waterloo')}")
    # print(f"Ryerson → {uni_norm.normalize('Ryerson')}")
    # print(f"Unknown → {uni_norm.normalize('Unknown University')}")
    # print(f"\nReport: {uni_norm.get_report()}")
    #
    # # Test ProgramNormalizer
    # prog_norm = ProgramNormalizer('data/mappings/base_programs.yaml')
    # test_programs = [
    #     "BSc Honours: Computer Science (Co-op)",
    #     "Bachelor of Engineering - Mechanical",
    #     "BBA",
    #     "Software Engineering Co-op",
    # ]
    # for prog in test_programs:
    #     print(f"{prog} → {prog_norm.normalize(prog)}")
