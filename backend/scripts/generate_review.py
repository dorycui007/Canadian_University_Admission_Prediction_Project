#!/usr/bin/env python3
"""
Review File Generation Script for University Admissions Prediction System
==========================================================================

This script generates human-readable review files from clustering results,
enabling manual verification and refinement of the normalization mappings.

================================================================================
                        SYSTEM ARCHITECTURE CONTEXT
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    WHERE THIS SCRIPT FITS                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   cluster_programs.py                                                    │
    │         │                                                                │
    │         ▼                                                                │
    │   [THIS SCRIPT]          ──────────────►    Human Review                │
    │   generate_review.py                              │                      │
    │         │                                         ▼                      │
    │         ▼                                   Manual Corrections           │
    │   ┌─────────────────────────────────────────────────────────────────┐   │
    │   │                    REVIEW GENERATION PIPELINE                    │   │
    │   ├─────────────────────────────────────────────────────────────────┤   │
    │   │                                                                  │   │
    │   │  INPUT                     PROCESS               OUTPUT          │   │
    │   │  ─────                     ───────               ──────          │   │
    │   │                                                                  │   │
    │   │  Cluster JSON      ──►  ┌─────────────┐   ──►   Review CSV       │   │
    │   │  + Original data        │ Generate    │         + Summary        │   │
    │   │                         │ Review      │         + Statistics     │   │
    │   │                         │ Tables      │                          │   │
    │   │                         └─────────────┘                          │   │
    │   │                                                                  │   │
    │   │  Review CSV format for easy editing:                             │   │
    │   │  ┌──────────────────────────────────────────────────────────┐   │   │
    │   │  │ cluster_id | canonical | variant | score | freq | action │   │   │
    │   │  │ 1          | CS | BSc CS | 100 | 45 | KEEP                │   │   │
    │   │  │ 1          | CS | Comp Sci | 85 | 12 | KEEP               │   │   │
    │   │  │ 1          | CS | Chemistry | 40 | 3 | SPLIT              │   │   │
    │   │  └──────────────────────────────────────────────────────────┘   │   │
    │   │                                                                  │   │
    │   └─────────────────────────────────────────────────────────────────┘   │
    │                                                                          │
    │                                         │                                │
    │                                         ▼                                │
    │                               Apply corrections                          │
    │                               Generate final mapping                     │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                           REVIEW WORKFLOW
================================================================================

    ┌────────────────────────────────────────────────────────────────────────┐
    │  MANUAL REVIEW PROCESS                                                  │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  1. GENERATE REVIEW FILE                                                │
    │     ┌─────────────────────────────────────────────────────────────┐    │
    │     │ $ python scripts/generate_review.py                         │    │
    │     │                                                             │    │
    │     │ Creates: data/review/program_review.csv                     │    │
    │     └─────────────────────────────────────────────────────────────┘    │
    │                              │                                          │
    │                              ▼                                          │
    │  2. OPEN IN SPREADSHEET                                                 │
    │     ┌─────────────────────────────────────────────────────────────┐    │
    │     │ Open CSV in Excel/Google Sheets                             │    │
    │     │ Sort by cluster_id, then by score                          │    │
    │     │                                                             │    │
    │     │ Focus on:                                                   │    │
    │     │ • Low scores (< 80): Likely false positives               │    │
    │     │ • Large clusters (> 10): May contain errors               │    │
    │     │ • High frequency items: Important to get right            │    │
    │     └─────────────────────────────────────────────────────────────┘    │
    │                              │                                          │
    │                              ▼                                          │
    │  3. MARK ACTIONS                                                        │
    │     ┌─────────────────────────────────────────────────────────────┐    │
    │     │ action column options:                                      │    │
    │     │                                                             │    │
    │     │ KEEP    - Correct match, include in final mapping          │    │
    │     │ SPLIT   - Wrong cluster, create separate entry             │    │
    │     │ MERGE   - Merge with another cluster (specify target)      │    │
    │     │ RENAME  - Change canonical name                            │    │
    │     │ DELETE  - Invalid entry, exclude from mapping              │    │
    │     └─────────────────────────────────────────────────────────────┘    │
    │                              │                                          │
    │                              ▼                                          │
    │  4. GENERATE FINAL MAPPING                                              │
    │     ┌─────────────────────────────────────────────────────────────┐    │
    │     │ $ python scripts/generate_review.py --apply-corrections    │    │
    │     │                                                             │    │
    │     │ Creates: data/mappings/programs.yaml (final mapping)       │    │
    │     └─────────────────────────────────────────────────────────────┘    │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

================================================================================
                           USAGE
================================================================================

    Generate Review File:
    ─────────────────────
    $ python scripts/generate_review.py

    Options:
    ────────
    --clusters PATH     Path to clusters JSON file
    --output PATH       Path for review CSV output
    --include-stats     Include statistics columns
    --sort-by COLUMN    Sort by: cluster, score, frequency, canonical
    --min-score N       Only include items with score >= N
    --format FORMAT     Output format: csv, excel, html

    Apply Corrections:
    ──────────────────
    $ python scripts/generate_review.py --apply-corrections PATH

    Options:
    ────────
    --corrections PATH  Path to corrected review CSV
    --output PATH       Path for final YAML mapping
    --validate          Validate corrections before applying

================================================================================
                           OUTPUT FILES
================================================================================

    Review CSV (data/review/program_review.csv):
    ─────────────────────────────────────────────
    cluster_id,canonical,variant,score,frequency,action,notes
    1,"Computer Science | BSc","BSc CS",100,45,KEEP,
    1,"Computer Science | BSc","Comp Sci",85,12,KEEP,
    1,"Computer Science | BSc","Chemistry",40,3,SPLIT,"Different program"

    Statistics Summary (data/review/review_summary.txt):
    ────────────────────────────────────────────────────
    === Review Summary ===
    Total items: 1,830
    Total clusters: 450
    Items needing review (score < 80): 234
    High-priority items (freq > 50, score < 90): 45

    Final Mapping (data/mappings/programs.yaml):
    ─────────────────────────────────────────────
    Computer Science | BSc:
      - BSc CS
      - Comp Sci
      - Computer Science BSc
      ...

================================================================================
                           EXTERNAL REFERENCES
================================================================================

    • OpenRefine Manual: https://openrefine.org/docs/manual/cellediting
    • Data Cleaning Best Practices: https://www.kaggle.com/learn/data-cleaning

================================================================================
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

# Note: yaml import is deferred to save_mapping_yaml() to avoid import errors
# if pyyaml is not installed


# Review action options
class ReviewAction:
    """Valid actions for review items."""
    KEEP = "KEEP"       # Correct match, include in final mapping
    SPLIT = "SPLIT"     # Wrong cluster, create separate entry
    MERGE = "MERGE"     # Merge with another cluster
    RENAME = "RENAME"   # Change canonical name
    DELETE = "DELETE"   # Invalid entry, exclude from mapping


@dataclass
class ReviewItem:
    """
    Represents a single item in the review CSV.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  REVIEW ITEM STRUCTURE                                                  │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  cluster_id: 1                                                         │
    │  canonical: "Computer Science | BSc"                                   │
    │  variant: "BSc CS Co-op"                                               │
    │  score: 95.5                                                           │
    │  frequency: 45                                                         │
    │  action: "KEEP"                                                        │
    │  notes: ""                                                              │
    │  merge_target: None                                                    │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Attributes:
        cluster_id: ID of the cluster this item belongs to
        canonical: The canonical name of the cluster
        variant: The specific program name variant
        score: Similarity score (0-100)
        frequency: Number of occurrences in original data
        action: Review action (KEEP, SPLIT, MERGE, RENAME, DELETE)
        notes: Optional reviewer notes
        merge_target: If action=MERGE, the target cluster ID

    Example:
        >>> item = ReviewItem(
        ...     cluster_id=1,
        ...     canonical="Computer Science | BSc",
        ...     variant="BSc CS",
        ...     score=100,
        ...     frequency=45
        ... )
    """
    cluster_id: int
    canonical: str
    variant: str
    score: float
    frequency: int
    action: str = ReviewAction.KEEP
    notes: str = ""
    merge_target: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for CSV/JSON output."""
        d = {
            "cluster_id": self.cluster_id,
            "canonical": self.canonical,
            "variant": self.variant,
            "score": self.score,
            "frequency": self.frequency,
            "action": self.action,
            "notes": self.notes,
        }
        if self.merge_target is not None:
            d["merge_target"] = self.merge_target
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReviewItem':
        """Create ReviewItem from dictionary."""
        merge_target = data.get("merge_target")
        if merge_target is not None and merge_target != "":
            merge_target = int(merge_target)
        else:
            merge_target = None
        return cls(
            cluster_id=int(data["cluster_id"]),
            canonical=str(data["canonical"]),
            variant=str(data["variant"]),
            score=float(data["score"]),
            frequency=int(data["frequency"]),
            action=str(data.get("action", ReviewAction.KEEP)),
            notes=str(data.get("notes", "")),
            merge_target=merge_target,
        )


@dataclass
class ReviewStatistics:
    """
    Statistics about the review file for summary reporting.

    Attributes:
        total_items: Total number of review items
        total_clusters: Number of unique clusters
        needs_review: Items with score below threshold
        high_priority: High-frequency items needing review
        by_action: Count of items by action type

    Example:
        >>> stats = calculate_statistics(items)
        >>> print(f"Items needing review: {stats.needs_review}")
    """
    total_items: int = 0
    total_clusters: int = 0
    needs_review: int = 0
    high_priority: int = 0
    by_action: Dict[str, int] = field(default_factory=dict)

    def to_summary(self) -> str:
        """Generate human-readable summary text."""
        lines = [
            "=== Review Summary ===",
            f"Total items: {self.total_items:,}",
            f"Total clusters: {self.total_clusters:,}",
            f"Items needing review (score < threshold): {self.needs_review:,}",
            f"High-priority items (freq > 50, score < 90): {self.high_priority:,}",
        ]
        if self.by_action:
            lines.append("\nBy action:")
            for action, count in sorted(self.by_action.items()):
                lines.append(f"  {action}: {count:,}")
        return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  COMMAND LINE ARGUMENTS                                                 │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Generation Mode (default):                                            │
    │  --clusters PATH      Path to clusters JSON from cluster_programs.py   │
    │  --output PATH        Path for review CSV output                       │
    │  --include-stats      Add statistics columns                           │
    │  --sort-by COLUMN     Sort order for output                            │
    │  --min-score N        Filter by minimum score                          │
    │  --format FORMAT      Output format                                    │
    │                                                                         │
    │  Apply Mode (--apply-corrections):                                     │
    │  --corrections PATH   Path to corrected review CSV                     │
    │  --mapping-output     Path for final YAML mapping                      │
    │  --validate           Validate before applying                         │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Returns:
        Parsed arguments namespace

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Create ArgumentParser with description
    2. Add subparsers for 'generate' and 'apply' modes
    3. Add mode-specific arguments
    4. Parse and return
    """
    parser = argparse.ArgumentParser(
        description="Generate review files from clustering results and apply corrections."
    )
    subparsers = parser.add_subparsers(dest="mode", help="Operating mode")

    # Generate mode
    gen_parser = subparsers.add_parser("generate", help="Generate review CSV from clusters")
    gen_parser.add_argument(
        "--clusters", default="data/review/program_clusters.json",
        help="Path to clusters JSON file"
    )
    gen_parser.add_argument(
        "--output", default="data/review/program_review.csv",
        help="Path for review CSV output"
    )
    gen_parser.add_argument(
        "--include-stats", action="store_true",
        help="Include statistics columns"
    )
    gen_parser.add_argument(
        "--sort-by", choices=["cluster", "score", "frequency", "canonical"],
        default="cluster", help="Sort order for output"
    )
    gen_parser.add_argument(
        "--min-score", type=float, default=None,
        help="Only include items with score >= N"
    )
    gen_parser.add_argument(
        "--format", choices=["csv", "excel", "html"], default="csv",
        help="Output format"
    )

    # Apply mode
    apply_parser = subparsers.add_parser("apply", help="Apply corrections from review CSV")
    apply_parser.add_argument(
        "--corrections", default="data/review/program_review.csv",
        help="Path to corrected review CSV"
    )
    apply_parser.add_argument(
        "--mapping-output", default="data/mappings/programs.yaml",
        help="Path for final YAML mapping"
    )
    apply_parser.add_argument(
        "--validate", action="store_true",
        help="Validate corrections before applying"
    )

    return parser.parse_args()


def load_clusters(file_path: str) -> Dict[str, Any]:
    """
    Load clusters from JSON file produced by cluster_programs.py.

    Args:
        file_path: Path to clusters JSON file

    Returns:
        Dictionary with cluster data

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Open file and load JSON
    2. Validate structure (has 'clusters' key)
    3. Return data
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Clusters file not found: {file_path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "clusters" not in data:
        raise ValueError(f"Invalid clusters file: missing 'clusters' key in {file_path}")
    return data


def clusters_to_review_items(
    clusters_data: Dict[str, Any],
    min_score: Optional[float] = None
) -> List[ReviewItem]:
    """
    Convert cluster data to review items.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  CONVERSION PROCESS                                                     │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Input (from clusters JSON):                                           │
    │  {                                                                      │
    │      "clusters": [{                                                    │
    │          "id": 1,                                                      │
    │          "canonical": "Computer Science | BSc",                        │
    │          "members": [                                                  │
    │              {"name": "BSc CS", "score": 100, "frequency": 45},       │
    │              {"name": "Comp Sci", "score": 85, "frequency": 12}       │
    │          ]                                                             │
    │      }, ...]                                                           │
    │  }                                                                      │
    │                                                                         │
    │  Output:                                                                │
    │  [                                                                      │
    │      ReviewItem(cluster_id=1, canonical="CS", variant="BSc CS", ...),  │
    │      ReviewItem(cluster_id=1, canonical="CS", variant="Comp Sci", ...) │
    │  ]                                                                      │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Args:
        clusters_data: Dictionary from load_clusters()
        min_score: Optional minimum score filter

    Returns:
        List of ReviewItem objects

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Initialize empty list
    2. For each cluster in clusters_data['clusters']:
       a. For each member in cluster['members']:
          - If min_score and score < min_score: skip
          - Create ReviewItem with cluster info
          - Set action based on score threshold
          - Append to list
    3. Return list
    """
    items = []
    for cluster in clusters_data.get("clusters", []):
        cluster_id = cluster["id"]
        canonical = cluster["canonical"]
        for member in cluster.get("members", []):
            score = float(member.get("score", 0))
            if min_score is not None and score < min_score:
                continue
            action = ReviewAction.KEEP if score >= 80 else ReviewAction.KEEP
            items.append(ReviewItem(
                cluster_id=cluster_id,
                canonical=canonical,
                variant=member["name"],
                score=score,
                frequency=int(member.get("frequency", 1)),
                action=action,
            ))
    return items


def sort_review_items(
    items: List[ReviewItem],
    sort_by: str = 'cluster'
) -> List[ReviewItem]:
    """
    Sort review items by specified column.

    Args:
        items: List of ReviewItem objects
        sort_by: Sort key: 'cluster', 'score', 'frequency', 'canonical'

    Returns:
        Sorted list of ReviewItem objects

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Define sort key functions for each option
    2. Sort items using appropriate key
    3. Return sorted list
    """
    sort_keys = {
        "cluster": lambda item: (item.cluster_id, -item.score),
        "score": lambda item: (item.score, item.cluster_id),
        "frequency": lambda item: (-item.frequency, item.cluster_id),
        "canonical": lambda item: (item.canonical, -item.score),
    }
    key_func = sort_keys.get(sort_by, sort_keys["cluster"])
    return sorted(items, key=key_func)


def calculate_statistics(
    items: List[ReviewItem],
    review_threshold: float = 80
) -> ReviewStatistics:
    """
    Calculate statistics about review items.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  STATISTICS CALCULATION                                                 │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Metrics:                                                               │
    │  • total_items: len(items)                                             │
    │  • total_clusters: len(unique cluster_ids)                             │
    │  • needs_review: count where score < threshold                         │
    │  • high_priority: count where freq > 50 AND score < 90                 │
    │  • by_action: Counter of action values                                  │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Args:
        items: List of ReviewItem objects
        review_threshold: Score below which items need review

    Returns:
        ReviewStatistics object

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Count total items
    2. Count unique cluster IDs
    3. Count items needing review (score < threshold)
    4. Count high priority (high freq, low score)
    5. Count by action
    6. Return ReviewStatistics
    """
    total_items = len(items)
    cluster_ids = set(item.cluster_id for item in items)
    total_clusters = len(cluster_ids)
    needs_review = sum(1 for item in items if item.score < review_threshold)
    high_priority = sum(
        1 for item in items if item.frequency > 50 and item.score < 90
    )
    by_action: Dict[str, int] = {}
    for item in items:
        by_action[item.action] = by_action.get(item.action, 0) + 1
    return ReviewStatistics(
        total_items=total_items,
        total_clusters=total_clusters,
        needs_review=needs_review,
        high_priority=high_priority,
        by_action=by_action,
    )


def save_review_csv(
    items: List[ReviewItem],
    output_path: str,
    include_stats: bool = False
) -> None:
    """
    Save review items to CSV file.

    Args:
        items: List of ReviewItem objects
        output_path: Path for output CSV
        include_stats: If True, add statistics columns

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Define column headers
    2. Open file for writing
    3. Write header row
    4. Write item rows using csv.DictWriter
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["cluster_id", "canonical", "variant", "score", "frequency", "action", "notes"]
    if include_stats:
        fieldnames.append("merge_target")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in items:
            row = item.to_dict()
            # Only keep fields in fieldnames
            row = {k: v for k, v in row.items() if k in fieldnames}
            writer.writerow(row)


def save_summary(stats: ReviewStatistics, output_path: str) -> None:
    """
    Save statistics summary to text file.

    Args:
        stats: ReviewStatistics object
        output_path: Path for output text file

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Generate summary text using stats.to_summary()
    2. Write to file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(stats.to_summary())
        f.write("\n")


def load_corrections(file_path: str) -> List[ReviewItem]:
    """
    Load corrected review items from CSV.

    Args:
        file_path: Path to corrected review CSV

    Returns:
        List of ReviewItem objects with user corrections

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Open CSV file
    2. Read rows using csv.DictReader
    3. Convert to ReviewItem objects
    4. Return list
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Corrections file not found: {file_path}")
    items = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append(ReviewItem.from_dict(row))
    return items


def validate_corrections(items: List[ReviewItem]) -> List[str]:
    """
    Validate corrected review items.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  VALIDATION CHECKS                                                      │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  1. Action values are valid                                            │
    │  2. MERGE actions have valid merge_target                              │
    │  3. RENAME actions have new canonical in notes                         │
    │  4. No duplicate variants across clusters                              │
    │  5. All cluster_ids are consistent                                     │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Args:
        items: List of corrected ReviewItem objects

    Returns:
        List of error messages (empty if valid)

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Check each validation rule
    2. Collect error messages
    3. Return list of errors
    """
    errors = []
    valid_actions = {ReviewAction.KEEP, ReviewAction.SPLIT, ReviewAction.MERGE,
                     ReviewAction.RENAME, ReviewAction.DELETE}
    seen_variants = {}
    cluster_ids = set(item.cluster_id for item in items)

    for i, item in enumerate(items):
        # Check valid action
        if item.action not in valid_actions:
            errors.append(
                f"Row {i + 1}: Invalid action '{item.action}' for variant '{item.variant}'"
            )

        # MERGE must have valid merge_target
        if item.action == ReviewAction.MERGE:
            if item.merge_target is None:
                errors.append(
                    f"Row {i + 1}: MERGE action requires merge_target for '{item.variant}'"
                )
            elif item.merge_target not in cluster_ids:
                errors.append(
                    f"Row {i + 1}: MERGE target {item.merge_target} not found for '{item.variant}'"
                )

        # RENAME must have new name in notes
        if item.action == ReviewAction.RENAME:
            if not item.notes.strip():
                errors.append(
                    f"Row {i + 1}: RENAME action requires new canonical in notes for '{item.variant}'"
                )

        # Check duplicate variants
        if item.variant in seen_variants:
            prev_cluster = seen_variants[item.variant]
            if prev_cluster != item.cluster_id:
                errors.append(
                    f"Row {i + 1}: Duplicate variant '{item.variant}' in clusters "
                    f"{prev_cluster} and {item.cluster_id}"
                )
        else:
            seen_variants[item.variant] = item.cluster_id

    return errors


def apply_corrections(items: List[ReviewItem]) -> Dict[str, List[str]]:
    """
    Apply corrections to generate final mapping.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  CORRECTION APPLICATION                                                 │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  For each item by action:                                              │
    │                                                                         │
    │  KEEP:   Add variant to canonical's list                               │
    │  SPLIT:  Create new entry with variant as canonical                    │
    │  MERGE:  Add variant to merge_target's canonical                       │
    │  RENAME: Use notes field as new canonical                              │
    │  DELETE: Skip (don't add to mapping)                                   │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Args:
        items: List of corrected ReviewItem objects

    Returns:
        Dictionary mapping canonical names to list of variants

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Initialize empty mapping dict
    2. Process KEEP items first (add to existing canonical)
    3. Process SPLIT items (create new entries)
    4. Process MERGE items (add to target canonical)
    5. Process RENAME items (update canonical)
    6. Skip DELETE items
    7. Return mapping
    """
    mapping: Dict[str, List[str]] = {}

    # Build cluster_id -> canonical lookup for MERGE targets
    canonical_by_id: Dict[int, str] = {}
    for item in items:
        canonical_by_id[item.cluster_id] = item.canonical

    # Process RENAME first to update canonical names
    renamed: Dict[int, str] = {}
    for item in items:
        if item.action == ReviewAction.RENAME and item.notes.strip():
            renamed[item.cluster_id] = item.notes.strip()

    for item in items:
        if item.action == ReviewAction.DELETE:
            continue

        if item.action == ReviewAction.KEEP:
            canonical = renamed.get(item.cluster_id, item.canonical)
            mapping.setdefault(canonical, [])
            if item.variant != canonical:
                mapping[canonical].append(item.variant)

        elif item.action == ReviewAction.SPLIT:
            # Create new entry with variant as its own canonical
            mapping.setdefault(item.variant, [])

        elif item.action == ReviewAction.MERGE:
            if item.merge_target is not None:
                target_canonical = renamed.get(
                    item.merge_target,
                    canonical_by_id.get(item.merge_target, str(item.merge_target))
                )
                mapping.setdefault(target_canonical, [])
                if item.variant != target_canonical:
                    mapping[target_canonical].append(item.variant)

        elif item.action == ReviewAction.RENAME:
            canonical = renamed.get(item.cluster_id, item.canonical)
            mapping.setdefault(canonical, [])
            if item.variant != canonical:
                mapping[canonical].append(item.variant)

    return mapping


def save_mapping_yaml(mapping: Dict[str, List[str]], output_path: str) -> None:
    """
    Save final mapping to YAML file.

    Args:
        mapping: Dictionary mapping canonical names to variants
        output_path: Path for output YAML file

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Import yaml
    2. Write mapping with yaml.dump()
    3. Use default_flow_style=False for readability
    """
    import yaml

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(mapping, f, default_flow_style=False, allow_unicode=True, sort_keys=True)


def generate_review(args: argparse.Namespace) -> int:
    """
    Main function for generating review file.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code: 0 for success, 1 for failure

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Load clusters from JSON
    2. Convert to review items
    3. Sort items
    4. Calculate statistics
    5. Save review CSV
    6. Save summary
    7. Print completion message
    """
    try:
        clusters_data = load_clusters(args.clusters)
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        print(f"Error: {e}")
        return 1

    min_score = getattr(args, "min_score", None)
    items = clusters_to_review_items(clusters_data, min_score=min_score)

    sort_by = getattr(args, "sort_by", "cluster")
    items = sort_review_items(items, sort_by=sort_by)

    stats = calculate_statistics(items)

    include_stats = getattr(args, "include_stats", False)
    save_review_csv(items, args.output, include_stats=include_stats)

    summary_path = str(Path(args.output).with_suffix(".txt").parent / "review_summary.txt")
    save_summary(stats, summary_path)

    print(f"Review CSV written to: {args.output}")
    print(f"Summary written to: {summary_path}")
    print(stats.to_summary())

    return 0


def apply_review(args: argparse.Namespace) -> int:
    """
    Main function for applying corrections.

    Args:
        args: Parsed command line arguments

    Returns:
        Exit code: 0 for success, 1 for failure

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Load corrections from CSV
    2. If validate flag, run validation
    3. Apply corrections to generate mapping
    4. Save mapping YAML
    5. Print completion message
    """
    try:
        items = load_corrections(args.corrections)
    except (FileNotFoundError, Exception) as e:
        print(f"Error loading corrections: {e}")
        return 1

    if getattr(args, "validate", False):
        errors = validate_corrections(items)
        if errors:
            print("Validation errors found:")
            for err in errors:
                print(f"  - {err}")
            return 1
        print("Validation passed.")

    mapping = apply_corrections(items)

    output_path = getattr(args, "mapping_output", "data/mappings/programs.yaml")
    save_mapping_yaml(mapping, output_path)

    print(f"Final mapping written to: {output_path}")
    print(f"Total canonical programs: {len(mapping)}")
    total_variants = sum(len(v) for v in mapping.values())
    print(f"Total variants mapped: {total_variants}")

    return 0


def main() -> int:
    """
    Main entry point for the review generation script.

    Returns:
        Exit code: 0 for success, 1 for failure
    """
    args = parse_args()

    if args.mode == "generate":
        return generate_review(args)
    elif args.mode == "apply":
        return apply_review(args)
    else:
        print("Please specify a mode: 'generate' or 'apply'")
        print("Usage: python scripts/generate_review.py generate --clusters PATH")
        print("       python scripts/generate_review.py apply --corrections PATH")
        return 1


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
│  [ ] Implement ReviewItem.to_dict() and from_dict()                         │
│  [ ] Implement ReviewStatistics.to_summary()                                │
│  [ ] Implement clusters_to_review_items()                                   │
│  [ ] Implement save_review_csv()                                            │
│  [ ] Implement apply_corrections()                                          │
│                                                                              │
│  MEDIUM PRIORITY:                                                            │
│  ────────────────                                                            │
│  [ ] Implement parse_args()                                                 │
│  [ ] Implement load_clusters()                                              │
│  [ ] Implement calculate_statistics()                                       │
│  [ ] Implement validate_corrections()                                       │
│  [ ] Implement save_mapping_yaml()                                          │
│                                                                              │
│  LOW PRIORITY:                                                               │
│  ─────────────                                                               │
│  [ ] Add Excel output format                                                │
│  [ ] Add HTML output format with highlighting                               │
│  [ ] Add diff view for corrections                                          │
│                                                                              │
│  TESTING:                                                                    │
│  ────────                                                                    │
│  [ ] Test with sample cluster data                                          │
│  [ ] Test correction validation                                             │
│  [ ] Test YAML output format                                                │
│  [ ] Test round-trip (generate → edit → apply)                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    sys.exit(main())
