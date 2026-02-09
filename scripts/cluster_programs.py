#!/usr/bin/env python3
"""
Program Clustering Script for University Admissions Prediction System
======================================================================

This script clusters similar program names using RapidFuzz fuzzy matching
to identify duplicates and variations that should be normalized together.

================================================================================
                        SYSTEM ARCHITECTURE CONTEXT
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    WHERE THIS SCRIPT FITS                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   clean_universities.py                                                  │
    │         │                                                                │
    │         ▼                                                                │
    │   [THIS SCRIPT]      ──────────────►    generate_review.py              │
    │   cluster_programs.py                          │                         │
    │         │                                      ▼                         │
    │         │                              Manual Review                     │
    │         │                                      │                         │
    │         ▼                                      ▼                         │
    │   ┌─────────────────────────────────────────────────────────────────┐   │
    │   │                    CLUSTERING PIPELINE                           │   │
    │   ├─────────────────────────────────────────────────────────────────┤   │
    │   │                                                                  │   │
    │   │  INPUT                     PROCESS               OUTPUT          │   │
    │   │  ─────                     ───────               ──────          │   │
    │   │                                                                  │   │
    │   │  Unique programs   ──►  ┌─────────────┐   ──►   Clusters         │   │
    │   │  (1,830 values)         │ RapidFuzz   │         (JSON/CSV)       │   │
    │   │                         │ Clustering  │                          │   │
    │   │  • BSc CS Co-op         │             │         Cluster 1:       │   │
    │   │  • Computer Science     │ token_set   │         • Computer Sci   │   │
    │   │  • B.Sc Comp Sci        │ _ratio      │         • BSc CS         │   │
    │   │                         │             │         • Comp Science   │   │
    │   │                         └─────────────┘                          │   │
    │   │                                                                  │   │
    │   └─────────────────────────────────────────────────────────────────┘   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                           CLUSTERING ALGORITHM
================================================================================

    ┌────────────────────────────────────────────────────────────────────────┐
    │  RAPIDFUZZ CLUSTERING APPROACH                                          │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  We use token_set_ratio for program matching because:                  │
    │                                                                         │
    │  • Handles word reordering:                                             │
    │    "Computer Science Honours" ≈ "Honours Computer Science"             │
    │                                                                         │
    │  • Handles word variations:                                             │
    │    "Bachelor of Science" ≈ "BSc"                                       │
    │                                                                         │
    │  • Handles extra words:                                                 │
    │    "Computer Science" ≈ "Computer Science Program"                     │
    │                                                                         │
    │  ALGORITHM:                                                             │
    │  ──────────                                                             │
    │                                                                         │
    │  1. Preprocess all program names                                        │
    │     ┌─────────────────────────────────────────────────────────────┐    │
    │     │ • Extract components (degree, honours, coop)                │    │
    │     │ • Generate clustering keys                                  │    │
    │     │ • Group by key                                              │    │
    │     └─────────────────────────────────────────────────────────────┘    │
    │                              │                                          │
    │                              ▼                                          │
    │  2. For each unique program (not yet clustered):                       │
    │     ┌─────────────────────────────────────────────────────────────┐    │
    │     │ matches = process.extract(                                  │    │
    │     │     program,                                                │    │
    │     │     all_programs,                                           │    │
    │     │     scorer=fuzz.token_set_ratio,                           │    │
    │     │     score_cutoff=threshold                                 │    │
    │     │ )                                                           │    │
    │     └─────────────────────────────────────────────────────────────┘    │
    │                              │                                          │
    │                              ▼                                          │
    │  3. Group matches into cluster                                          │
    │     ┌─────────────────────────────────────────────────────────────┐    │
    │     │ • First match (or most frequent) = canonical                │    │
    │     │ • All matches → cluster                                     │    │
    │     │ • Mark all as processed                                     │    │
    │     └─────────────────────────────────────────────────────────────┘    │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

================================================================================
                           USAGE
================================================================================

    Command Line:
    ─────────────
    $ python scripts/cluster_programs.py

    Options:
    ────────
    --input PATH      Path to CSV with programs (or cleaned data)
    --output PATH     Path for output clusters JSON/CSV
    --threshold INT   Fuzzy matching threshold (default: 80)
    --min-cluster N   Minimum cluster size to report (default: 2)
    --format FORMAT   Output format: json, csv, both (default: both)
    --verbose         Show detailed progress

    Example:
    ────────
    $ python scripts/cluster_programs.py --threshold 75 --format csv

================================================================================
                           OUTPUT FORMAT
================================================================================

    JSON Output (data/review/program_clusters.json):
    ─────────────────────────────────────────────────
    {
        "clusters": [
            {
                "id": 1,
                "canonical": "Computer Science | BSc Honours | Co-op",
                "members": [
                    {"name": "BSc Honours: Computer Science (Co-op)", "score": 100},
                    {"name": "B. Computer Science Honours, Co-op", "score": 92},
                    {"name": "CompSci Honours Co-op", "score": 85}
                ],
                "count": 3
            },
            ...
        ],
        "summary": {
            "total_programs": 1830,
            "total_clusters": 450,
            "singletons": 200,
            "largest_cluster": 15
        }
    }

    CSV Output (data/review/program_clusters.csv):
    ───────────────────────────────────────────────
    cluster_id,canonical_name,variant,similarity_score,frequency
    1,"Computer Science | BSc Honours | Co-op","BSc Honours: Computer Science (Co-op)",100,45
    1,"Computer Science | BSc Honours | Co-op","B. Computer Science Honours, Co-op",92,23

================================================================================
                           EXTERNAL REFERENCES
================================================================================

    • RapidFuzz: https://github.com/maxbachmann/RapidFuzz
    • OpenRefine Clustering: https://openrefine.org/docs/manual/cellediting#cluster-and-edit

================================================================================
"""

from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.utils.normalize import ProgramNormalizer


@dataclass
class ClusterMember:
    """
    Represents a single program name within a cluster.

    Attributes:
        name: The original program name
        score: Similarity score to the canonical name (0-100)
        frequency: Number of occurrences in the dataset

    Example:
        >>> member = ClusterMember(
        ...     name="BSc Honours: Computer Science (Co-op)",
        ...     score=100,
        ...     frequency=45
        ... )
    """
    name: str
    score: float
    frequency: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "score": self.score,
            "frequency": self.frequency,
        }


@dataclass
class ProgramCluster:
    """
    Represents a cluster of similar program names.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  CLUSTER STRUCTURE                                                      │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Cluster #1: Computer Science                                          │
    │  ─────────────────────────────                                          │
    │  canonical: "Computer Science | BSc Honours | Co-op"                   │
    │                                                                         │
    │  members:                                                               │
    │  ┌───────────────────────────────────────────────────────────────┐     │
    │  │ "BSc Honours: Computer Science (Co-op)"   score=100  freq=45 │     │
    │  │ "B. Computer Science Honours, Co-op"      score=92   freq=23 │     │
    │  │ "CompSci Honours Co-op"                   score=85   freq=12 │     │
    │  └───────────────────────────────────────────────────────────────┘     │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Attributes:
        id: Unique cluster identifier
        canonical: The canonical (representative) program name
        members: List of ClusterMember objects
        total_frequency: Sum of all member frequencies

    Example:
        >>> cluster = ProgramCluster(id=1, canonical="Computer Science")
        >>> cluster.add_member("CS", score=90, frequency=10)
    """
    id: int
    canonical: str
    members: List[ClusterMember] = field(default_factory=list)

    @property
    def size(self) -> int:
        """Return number of members in cluster."""
        return len(self.members)

    @property
    def total_frequency(self) -> int:
        """Return sum of all member frequencies."""
        return sum(m.frequency for m in self.members)

    def add_member(self, name: str, score: float, frequency: int = 1) -> None:
        """
        Add a member to the cluster.

        Args:
            name: Program name to add
            score: Similarity score to canonical (0-100)
            frequency: Number of occurrences in dataset
        """
        self.members.append(ClusterMember(name=name, score=score, frequency=frequency))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "canonical": self.canonical,
            "members": [m.to_dict() for m in self.members],
            "count": self.size,
        }


@dataclass
class ClusteringResult:
    """
    Container for the complete clustering output.

    Attributes:
        clusters: List of ProgramCluster objects
        singleton_count: Number of programs with no similar matches
        total_programs: Total unique programs processed

    Example:
        >>> result = cluster_programs(programs, threshold=80)
        >>> print(f"Found {len(result.clusters)} clusters")
    """
    clusters: List[ProgramCluster] = field(default_factory=list)
    singleton_count: int = 0
    total_programs: int = 0

    @property
    def cluster_count(self) -> int:
        """Return number of clusters (excluding singletons)."""
        return len(self.clusters)

    @property
    def largest_cluster_size(self) -> int:
        """Return size of the largest cluster."""
        if not self.clusters:
            return 0
        return max(c.size for c in self.clusters)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "clusters": [c.to_dict() for c in self.clusters],
            "summary": {
                "total_programs": self.total_programs,
                "total_clusters": self.cluster_count,
                "singletons": self.singleton_count,
                "largest_cluster": self.largest_cluster_size,
            },
        }

    def to_csv_rows(self) -> List[Dict[str, Any]]:
        """
        Convert to flat rows for CSV output.

        Returns:
            List of dictionaries with keys:
            cluster_id, canonical_name, variant, similarity_score, frequency
        """
        rows = []
        for cluster in self.clusters:
            for member in cluster.members:
                rows.append({
                    "cluster_id": cluster.id,
                    "canonical_name": cluster.canonical,
                    "variant": member.name,
                    "similarity_score": member.score,
                    "frequency": member.frequency,
                })
        return rows


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  COMMAND LINE ARGUMENTS                                                 │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  --input PATH       Path to input CSV with programs                    │
    │  --output PATH      Base path for output files (no extension)          │
    │  --threshold INT    Fuzzy matching threshold (0-100)                   │
    │  --min-cluster N    Minimum cluster size to include                    │
    │  --format FORMAT    Output format: json, csv, both                     │
    │  --verbose          Show detailed progress                              │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Returns:
        Parsed arguments namespace

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Create ArgumentParser with description
    2. Add required and optional arguments
    3. Set defaults
    4. Parse and return
    """
    parser = argparse.ArgumentParser(
        description="Cluster similar program names using RapidFuzz fuzzy matching."
    )
    parser.add_argument(
        "--input", default="data/cleaned/2024_2025_cleaned.csv",
        help="Path to input CSV with programs"
    )
    parser.add_argument(
        "--output", default="data/review/program_clusters",
        help="Base path for output files (no extension)"
    )
    parser.add_argument(
        "--threshold", type=int, default=80,
        help="Fuzzy matching threshold 0-100 (default: 80)"
    )
    parser.add_argument(
        "--min-cluster", type=int, default=2,
        help="Minimum cluster size to include (default: 2)"
    )
    parser.add_argument(
        "--format", choices=["json", "csv", "both"], default="both",
        help="Output format (default: both)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show detailed progress"
    )
    return parser.parse_args()


def load_programs(file_path: str, program_column: Optional[str] = None) -> List[Tuple[str, int]]:
    """
    Load unique programs with their frequencies from a CSV file.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  LOADING PROGRAMS                                                       │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Input CSV columns to try (in order):                                  │
    │  1. program_column (if specified)                                      │
    │  2. 'program_normalized'                                               │
    │  3. 'Program'                                                          │
    │  4. 'What program did you apply to?'                                   │
    │                                                                         │
    │  Output: List of (program_name, frequency) tuples                      │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Args:
        file_path: Path to CSV file
        program_column: Optional specific column name to use

    Returns:
        List of (program_name, frequency) tuples sorted by frequency desc

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Load CSV with pandas
    2. Find program column (try multiple names)
    3. Get value counts
    4. Return as list of tuples
    """
    import pandas as pd

    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin-1")

    candidate_columns = [
        program_column,
        "program_normalized",
        "Program",
        "What program did you apply to?",
    ]

    col = None
    for c in candidate_columns:
        if c is not None and c in df.columns:
            col = c
            break

    if col is None:
        raise ValueError(
            f"Could not find program column in {file_path}. "
            f"Columns: {list(df.columns)}"
        )

    counts = df[col].dropna().value_counts()
    return [(name, int(freq)) for name, freq in counts.items()]


def preprocess_programs(
    programs: List[Tuple[str, int]],
    normalizer: Optional['ProgramNormalizer'] = None
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Preprocess programs by generating clustering keys.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  PREPROCESSING                                                          │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Input:  [("BSc CS Co-op", 45), ("Computer Science", 30), ...]        │
    │                                                                         │
    │  Output (grouped by key):                                               │
    │  {                                                                      │
    │      "computer_science|bsc||coop": [("BSc CS Co-op", 45), ...],       │
    │      "computer_science|||": [("Computer Science", 30), ...],          │
    │      ...                                                                │
    │  }                                                                      │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Args:
        programs: List of (program_name, frequency) tuples
        normalizer: Optional ProgramNormalizer for key generation

    Returns:
        Dictionary mapping clustering keys to list of programs

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Initialize empty defaultdict(list)
    2. For each program:
       a. Generate key using normalizer.to_key() or lowercase
       b. Append to appropriate key group
    3. Return dictionary
    """
    groups: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for name, freq in programs:
        if normalizer is not None:
            key = normalizer.to_key(name)
        else:
            key = name.strip().lower()
        groups[key].append((name, freq))
    return dict(groups)


def cluster_programs(
    programs: List[Tuple[str, int]],
    threshold: int = 80,
    normalizer: Optional['ProgramNormalizer'] = None
) -> ClusteringResult:
    """
    Cluster similar programs using RapidFuzz fuzzy matching.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  CLUSTERING ALGORITHM                                                   │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  STEP 1: Initialize                                                     │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ processed = set()                                               │   │
    │  │ clusters = []                                                   │   │
    │  │ program_names = [p[0] for p in programs]                        │   │
    │  │ freq_map = {p[0]: p[1] for p in programs}                       │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                          │
    │                              ▼                                          │
    │  STEP 2: For each unprocessed program                                  │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ if prog in processed: continue                                  │   │
    │  │                                                                 │   │
    │  │ matches = process.extract(                                      │   │
    │  │     prog,                                                       │   │
    │  │     program_names,                                              │   │
    │  │     scorer=fuzz.token_set_ratio,                               │   │
    │  │     score_cutoff=threshold,                                    │   │
    │  │     limit=None  # Get all matches                              │   │
    │  │ )                                                               │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                              │                                          │
    │                              ▼                                          │
    │  STEP 3: Create cluster from matches                                   │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │ canonical = select_canonical(matches, freq_map)                │   │
    │  │ cluster = ProgramCluster(id=next_id, canonical=canonical)      │   │
    │  │                                                                 │   │
    │  │ for match_name, score, _ in matches:                           │   │
    │  │     if match_name not in processed:                            │   │
    │  │         cluster.add_member(match_name, score, freq_map[...])   │   │
    │  │         processed.add(match_name)                              │   │
    │  │                                                                 │   │
    │  │ clusters.append(cluster)                                       │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Args:
        programs: List of (program_name, frequency) tuples
        threshold: Minimum similarity score for clustering (0-100)
        normalizer: Optional ProgramNormalizer for preprocessing

    Returns:
        ClusteringResult containing all clusters

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Extract program names and build frequency map
    2. Initialize processed set and clusters list
    3. For each unprocessed program:
       a. Find all matches using process.extract()
       b. Select canonical name (most frequent or first)
       c. Create cluster and add all matches
       d. Mark all matches as processed
    4. Count singletons (clusters of size 1)
    5. Return ClusteringResult
    """
    from rapidfuzz import fuzz, process

    program_names = [p[0] for p in programs]
    freq_map = {p[0]: p[1] for p in programs}

    processed = set()
    clusters = []
    cluster_id = 1
    singleton_count = 0

    for prog in program_names:
        if prog in processed:
            continue

        matches = process.extract(
            prog,
            program_names,
            scorer=fuzz.token_set_ratio,
            score_cutoff=threshold,
            limit=None,
        )

        # Filter out already-processed matches
        unprocessed_matches = [
            (name, score, idx) for name, score, idx in matches
            if name not in processed
        ]

        if not unprocessed_matches:
            continue

        canonical = select_canonical(unprocessed_matches, freq_map)
        cluster = ProgramCluster(id=cluster_id, canonical=canonical)

        for match_name, score, _ in unprocessed_matches:
            cluster.add_member(match_name, score, freq_map.get(match_name, 1))
            processed.add(match_name)

        if cluster.size == 1:
            singleton_count += 1
        clusters.append(cluster)
        cluster_id += 1

    return ClusteringResult(
        clusters=clusters,
        singleton_count=singleton_count,
        total_programs=len(program_names),
    )


def select_canonical(
    matches: List[Tuple[str, float, int]],
    freq_map: Dict[str, int]
) -> str:
    """
    Select the canonical (representative) name for a cluster.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  CANONICAL SELECTION STRATEGY                                           │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Priority (in order):                                                   │
    │  1. Highest frequency (most common variation)                          │
    │  2. Longest name (more descriptive)                                    │
    │  3. Alphabetically first (deterministic tie-breaker)                   │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Args:
        matches: List of (name, score, index) tuples from RapidFuzz
        freq_map: Dictionary mapping program names to frequencies

    Returns:
        The canonical program name

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Sort matches by (-frequency, -length, name)
    2. Return first element
    """
    sorted_matches = sorted(
        matches,
        key=lambda m: (-freq_map.get(m[0], 0), -len(m[0]), m[0])
    )
    return sorted_matches[0][0]


def save_json(result: ClusteringResult, output_path: str) -> None:
    """
    Save clustering result to JSON file.

    Args:
        result: ClusteringResult to save
        output_path: Path for output file (will add .json extension)

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Convert result to dictionary
    2. Write to file with json.dump() and indent=2
    """
    path = Path(output_path).with_suffix(".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)


def save_csv(result: ClusteringResult, output_path: str) -> None:
    """
    Save clustering result to CSV file.

    Args:
        result: ClusteringResult to save
        output_path: Path for output file (will add .csv extension)

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Convert result to CSV rows
    2. Write using pandas DataFrame or csv module
    """
    import csv as csv_mod

    path = Path(output_path).with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = result.to_csv_rows()
    if not rows:
        return
    fieldnames = ["cluster_id", "canonical_name", "variant", "similarity_score", "frequency"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(result: ClusteringResult, verbose: bool = False) -> None:
    """
    Print clustering summary to stdout.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  SUMMARY OUTPUT                                                         │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  === Program Clustering Summary ===                                    │
    │  Total unique programs: 1,830                                          │
    │  Total clusters: 450                                                   │
    │  Singletons: 200                                                       │
    │  Largest cluster: 15 members                                           │
    │                                                                         │
    │  Top 10 largest clusters (if verbose):                                 │
    │  1. "Computer Science" (15 variations)                                 │
    │  2. "Mechanical Engineering" (12 variations)                           │
    │  ...                                                                    │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Args:
        result: ClusteringResult to summarize
        verbose: If True, show top clusters

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Print basic statistics
    2. If verbose, sort clusters by size and print top N
    """
    print("\n=== Program Clustering Summary ===")
    print(f"Total unique programs: {result.total_programs:,}")
    print(f"Total clusters: {result.cluster_count:,}")
    print(f"Singletons: {result.singleton_count:,}")
    print(f"Largest cluster: {result.largest_cluster_size} members")

    if verbose and result.clusters:
        sorted_clusters = sorted(result.clusters, key=lambda c: -c.size)
        top_n = min(10, len(sorted_clusters))
        print(f"\nTop {top_n} largest clusters:")
        for i, cluster in enumerate(sorted_clusters[:top_n], 1):
            print(f"  {i}. \"{cluster.canonical}\" ({cluster.size} variations)")


def main() -> int:
    """
    Main entry point for the clustering script.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  MAIN WORKFLOW                                                          │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  1. Parse arguments                                                     │
    │  2. Load programs from input file                                       │
    │  3. Optionally initialize ProgramNormalizer                            │
    │  4. Run clustering algorithm                                            │
    │  5. Save results (JSON and/or CSV)                                     │
    │  6. Print summary                                                       │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    args = parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    # Load programs
    try:
        programs = load_programs(str(input_path))
    except Exception as e:
        print(f"Error loading programs: {e}")
        return 1

    if not programs:
        print("Error: No programs found in input file.")
        return 1

    if args.verbose:
        print(f"Loaded {len(programs)} unique programs from {input_path}")

    # Optionally initialize normalizer
    normalizer = None
    try:
        from src.utils.normalize import ProgramNormalizer
        normalizer = ProgramNormalizer()
        if args.verbose:
            print("Using ProgramNormalizer for preprocessing")
    except (ImportError, Exception):
        if args.verbose:
            print("ProgramNormalizer not available, using simple lowercase keys")

    # Run clustering
    result = cluster_programs(programs, threshold=args.threshold, normalizer=normalizer)

    # Filter by min-cluster size
    if args.min_cluster > 1:
        result.clusters = [c for c in result.clusters if c.size >= args.min_cluster]

    # Save results
    output_path = args.output
    if args.format in ("json", "both"):
        save_json(result, output_path)
        if args.verbose:
            print(f"Saved JSON: {Path(output_path).with_suffix('.json')}")

    if args.format in ("csv", "both"):
        save_csv(result, output_path)
        if args.verbose:
            print(f"Saved CSV: {Path(output_path).with_suffix('.csv')}")

    # Print summary
    print_summary(result, verbose=args.verbose)

    return 0


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
│  [ ] Implement ClusterMember.to_dict()                                      │
│  [ ] Implement ProgramCluster.add_member()                                  │
│  [ ] Implement ProgramCluster.to_dict()                                     │
│  [ ] Implement ClusteringResult.to_csv_rows()                               │
│  [ ] Implement cluster_programs()                                           │
│  [ ] Implement select_canonical()                                           │
│                                                                              │
│  MEDIUM PRIORITY:                                                            │
│  ────────────────                                                            │
│  [ ] Implement parse_args()                                                 │
│  [ ] Implement load_programs()                                              │
│  [ ] Implement save_json() and save_csv()                                   │
│  [ ] Implement main()                                                       │
│                                                                              │
│  OPTIMIZATIONS:                                                              │
│  ──────────────                                                              │
│  [ ] Use process.cdist for batch pairwise comparison                        │
│  [ ] Add parallel processing for large datasets                             │
│  [ ] Implement incremental clustering                                       │
│                                                                              │
│  TESTING:                                                                    │
│  ────────                                                                    │
│  [ ] Test with sample program data                                          │
│  [ ] Test threshold edge cases                                              │
│  [ ] Verify JSON/CSV output format                                          │
│  [ ] Test canonical selection                                               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    sys.exit(main())
