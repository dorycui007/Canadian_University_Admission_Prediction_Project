#!/usr/bin/env python3
"""
University Data Cleaning Script for University Admissions Prediction System
============================================================================

This script cleans and normalizes university names across all raw CSV files
in the data pipeline.

================================================================================
                        SYSTEM ARCHITECTURE CONTEXT
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    WHERE THIS SCRIPT FITS                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   [THIS SCRIPT]                                                          │
    │   clean_universities.py                                                  │
    │         │                                                                │
    │         ▼                                                                │
    │   ┌─────────────────────────────────────────────────────────────────┐   │
    │   │                    DATA CLEANING PIPELINE                        │   │
    │   ├─────────────────────────────────────────────────────────────────┤   │
    │   │                                                                  │   │
    │   │  INPUT                     PROCESS               OUTPUT          │   │
    │   │  ─────                     ───────               ──────          │   │
    │   │                                                                  │   │
    │   │  data/raw/                 ┌─────────────┐       data/processed/ │   │
    │   │  ├── 2024_2025_*.csv  ──► │ Normalizer  │ ──►   ├── 2024_*.csv  │   │
    │   │  ├── 2023_2024_*.csv  ──► │   - Exact   │ ──►   ├── 2023_*.csv  │   │
    │   │  └── 2022_2023_*.csv  ──► │   - Fuzzy   │ ──►   └── 2022_*.csv  │   │
    │   │                           │   - Report  │                        │   │
    │   │                           └─────────────┘                        │   │
    │   │                                 │                                │   │
    │   │                                 ▼                                │   │
    │   │                           Normalization                          │   │
    │   │                           Report (stdout)                        │   │
    │   │                                                                  │   │
    │   └─────────────────────────────────────────────────────────────────┘   │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                           RAW DATA STRUCTURE
================================================================================

    The raw CSV files have DIFFERENT column names for university across years:

    ┌────────────────────────────────────────────────────────────────────────┐
    │  COLUMN NAME VARIATIONS BY YEAR                                         │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  2024_2025:  Column "University" (index 2)                             │
    │              Example: "UofT", "Waterloo", "McMaster"                    │
    │                                                                         │
    │  2023_2024:  Column "What university did you apply to?" (index 3)      │
    │              Example: "University of Toronto", "UW"                     │
    │                                                                         │
    │  2022_2023:  Column "Which university was this program from?" (idx 2)  │
    │              Example: "Ryerson", "Western"                              │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

================================================================================
                           NORMALIZATION PROCESS
================================================================================

    ┌────────────────────────────────────────────────────────────────────────┐
    │  FOR EACH CSV FILE                                                      │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  1. LOAD                                                                │
    │     ┌─────────────────────────────────────────────────────────────┐    │
    │     │ df = pd.read_csv(raw_path)                                  │    │
    │     │ uni_col = COLUMN_MAPPING[year]                              │    │
    │     └─────────────────────────────────────────────────────────────┘    │
    │                              │                                          │
    │                              ▼                                          │
    │  2. NORMALIZE                                                           │
    │     ┌─────────────────────────────────────────────────────────────┐    │
    │     │ df['university_normalized'] = df[uni_col].apply(            │    │
    │     │     normalizer.normalize                                    │    │
    │     │ )                                                           │    │
    │     └─────────────────────────────────────────────────────────────┘    │
    │                              │                                          │
    │                              ▼                                          │
    │  3. SAVE                                                                │
    │     ┌─────────────────────────────────────────────────────────────┐    │
    │     │ df.to_csv(output_path, index=False)                         │    │
    │     │ print(f"Saved: {output_path}")                              │    │
    │     └─────────────────────────────────────────────────────────────┘    │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

================================================================================
                           USAGE
================================================================================

    Command Line:
    ─────────────
    $ python scripts/clean_universities.py

    Options:
    ────────
    --mapping PATH    Path to universities.yaml (default: data/mappings/...)
    --threshold INT   Fuzzy matching threshold (default: 85)
    --dry-run         Preview without saving
    --verbose         Show detailed progress

    Example:
    ────────
    $ python scripts/clean_universities.py --threshold 80 --verbose

================================================================================
                           OUTPUT
================================================================================

    Files Created:
    ──────────────
    data/processed/2024_2025_cleaned.csv
    data/processed/2023_2024_cleaned.csv
    data/processed/2022_2023_cleaned.csv

    Console Output:
    ───────────────
    === Processing 2024_2025 ===
    Loaded: 1,234 rows from data/raw/2024_2025_Canadian_University_Results.csv
    Saved: data/processed/2024_2025_cleaned.csv

    === Normalization Report ===
    Total: 3,456
    Exact matches: 3,100
    Fuzzy matches: 300
    Invalid: 56
    Match rate: 98.4%

    Invalid values:
    - "Unknown University" (12 occurrences)
    - "Some College" (8 occurrences)
    ...

================================================================================
                           EXTERNAL REFERENCES
================================================================================

    • Universities Canada: https://univcan.ca/universities/member-universities/
    • RapidFuzz: https://github.com/maxbachmann/RapidFuzz

================================================================================
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    from src.utils.normalize import UniversityNormalizer


# Column name mapping by academic year
# These vary because the survey questions changed between years
COLUMN_MAPPING: Dict[str, str] = {
    '2024_2025': 'University',
    '2023_2024': 'What university did you apply to?',
    '2022_2023': 'Which university was this program from?',
}

# Default paths
DEFAULT_MAPPING_PATH = 'data/mappings/universities.yaml'
DEFAULT_RAW_DIR = 'data/raw'
DEFAULT_OUTPUT_DIR = 'data/processed'
DEFAULT_FUZZY_THRESHOLD = 85


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  COMMAND LINE ARGUMENTS                                                 │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  --mapping PATH     Path to universities.yaml mapping file             │
    │  --raw-dir PATH     Directory containing raw CSV files                 │
    │  --output-dir PATH  Directory for cleaned output files                 │
    │  --threshold INT    Fuzzy matching threshold (0-100)                   │
    │  --dry-run          Preview changes without saving                     │
    │  --verbose          Show detailed progress messages                    │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Returns:
        Parsed arguments namespace

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Create ArgumentParser with description
    2. Add arguments for mapping, raw-dir, output-dir, threshold
    3. Add flags for dry-run and verbose
    4. Parse and return arguments
    """
    parser = argparse.ArgumentParser(
        description='Clean and normalize university names across raw CSV files.'
    )
    parser.add_argument(
        '--mapping', default=DEFAULT_MAPPING_PATH,
        help=f'Path to universities.yaml mapping file (default: {DEFAULT_MAPPING_PATH})'
    )
    parser.add_argument(
        '--raw-dir', default=DEFAULT_RAW_DIR,
        help=f'Directory containing raw CSV files (default: {DEFAULT_RAW_DIR})'
    )
    parser.add_argument(
        '--output-dir', default=DEFAULT_OUTPUT_DIR,
        help=f'Directory for cleaned output files (default: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--threshold', type=int, default=DEFAULT_FUZZY_THRESHOLD,
        help=f'Fuzzy matching threshold 0-100 (default: {DEFAULT_FUZZY_THRESHOLD})'
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview changes without saving files'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Show detailed progress messages'
    )
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> bool:
    """
    Validate that required files and directories exist.

    Args:
        args: Parsed command line arguments

    Returns:
        True if all paths are valid, False otherwise

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Check mapping file exists
    2. Check raw directory exists
    3. Create output directory if it doesn't exist
    4. Return True if all valid, print errors and return False otherwise
    """
    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        print(f"Error: Mapping file not found: {mapping_path}")
        return False

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        print(f"Error: Raw data directory not found: {raw_dir}")
        return False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return True


def load_csv(file_path: str, year: str) -> Optional['pd.DataFrame']:
    """
    Load a raw CSV file for a specific year.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  CSV LOADING                                                            │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  Handles:                                                               │
    │  • Different encodings (UTF-8, Latin-1)                                │
    │  • Missing files (returns None)                                        │
    │  • Column name verification                                            │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Args:
        file_path: Path to CSV file
        year: Academic year key (e.g., '2024_2025')

    Returns:
        DataFrame if successful, None if file not found or invalid

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Check if file exists
    2. Try loading with UTF-8, fallback to Latin-1
    3. Verify required column exists using COLUMN_MAPPING
    4. Return DataFrame or None
    """
    import pandas as pd

    path = Path(file_path)
    if not path.exists():
        print(f"Warning: File not found: {path}")
        return None

    # Try UTF-8, fallback to Latin-1
    df = None
    for encoding in ('utf-8', 'latin-1'):
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        print(f"Error: Could not read {file_path} with any supported encoding")
        return None

    # Verify required column exists
    required_col = COLUMN_MAPPING.get(year)
    if required_col and required_col not in df.columns:
        print(f"Warning: Column '{required_col}' not found in {file_path}")
        print(f"  Available columns: {list(df.columns)}")
        return None

    return df


def process_year(
    year: str,
    normalizer: 'UniversityNormalizer',
    raw_dir: str,
    output_dir: str,
    dry_run: bool = False,
    verbose: bool = False
) -> Dict[str, int]:
    """
    Process a single year's CSV file.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  PROCESSING STEPS                                                       │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  1. Construct file paths                                                │
    │     raw_path = f"{raw_dir}/{year}_Canadian_University_Results.csv"     │
    │     output_path = f"{output_dir}/{year}_cleaned.csv"                   │
    │                                                                         │
    │  2. Load raw data                                                       │
    │     df = load_csv(raw_path, year)                                      │
    │                                                                         │
    │  3. Get university column name                                          │
    │     uni_col = COLUMN_MAPPING[year]                                     │
    │                                                                         │
    │  4. Normalize university names                                          │
    │     df['university_normalized'] = df[uni_col].apply(                   │
    │         normalizer.normalize                                           │
    │     )                                                                   │
    │                                                                         │
    │  5. Save cleaned data (unless dry_run)                                  │
    │     df.to_csv(output_path, index=False)                                │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Args:
        year: Academic year key (e.g., '2024_2025')
        normalizer: Configured UniversityNormalizer instance
        raw_dir: Directory containing raw CSV files
        output_dir: Directory for output files
        dry_run: If True, don't save files
        verbose: If True, print detailed progress

    Returns:
        Dictionary with row counts: {'loaded': N, 'processed': N}

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Construct input/output paths
    2. Load CSV using load_csv()
    3. Get column name from COLUMN_MAPPING
    4. Apply normalizer to create university_normalized column
    5. If not dry_run, save to output_path
    6. Return statistics dictionary
    """
    raw_path = f"{raw_dir}/{year}_Canadian_University_Results.csv"
    output_path = f"{output_dir}/{year}_cleaned.csv"

    if verbose:
        print(f"\n=== Processing {year} ===")

    df = load_csv(raw_path, year)
    if df is None:
        return {'loaded': 0, 'processed': 0}

    loaded = len(df)
    if verbose:
        print(f"Loaded: {loaded:,} rows from {raw_path}")

    uni_col = COLUMN_MAPPING[year]
    df['university_normalized'] = df[uni_col].apply(normalizer.normalize)

    processed = len(df)

    if not dry_run:
        df.to_csv(output_path, index=False)
        if verbose:
            print(f"Saved: {output_path}")
    elif verbose:
        print(f"Dry run: would save to {output_path}")

    return {'loaded': loaded, 'processed': processed}


def print_report(normalizer: 'UniversityNormalizer', verbose: bool = False) -> None:
    """
    Print the normalization report to stdout.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  REPORT FORMAT                                                          │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  === Normalization Report ===                                          │
    │  Total: 3,456                                                          │
    │  Exact matches: 3,100                                                  │
    │  Fuzzy matches: 300                                                    │
    │  Invalid: 56                                                           │
    │  Match rate: 98.4%                                                     │
    │                                                                         │
    │  Invalid values (if verbose):                                          │
    │  - "Unknown University" (12 occurrences)                               │
    │  - "Some College" (8 occurrences)                                      │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Args:
        normalizer: UniversityNormalizer with populated stats
        verbose: If True, show full list of invalid values

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. Get report from normalizer.get_report()
    2. Print summary statistics
    3. If verbose, print invalid values (truncate if many)
    """
    report = normalizer.get_report()

    print("\n=== Normalization Report ===")
    print(f"Total: {report['total']:,}")
    print(f"Exact matches: {report['exact_matches']:,}")
    print(f"Fuzzy matches: {report['fuzzy_matches']:,}")
    print(f"Invalid: {report['invalid']:,}")
    print(f"Match rate: {report['match_rate']:.1f}%")

    if verbose and report['invalid_values']:
        from collections import Counter
        counts = Counter(report['invalid_values'])
        print("\nInvalid values:")
        for value, count in counts.most_common(20):
            print(f'  - "{value}" ({count} occurrences)')
        if len(counts) > 20:
            print(f"  ... and {len(counts) - 20} more")


def main() -> int:
    """
    Main entry point for the cleaning script.

    ┌────────────────────────────────────────────────────────────────────────┐
    │  MAIN WORKFLOW                                                          │
    ├────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  1. Parse arguments                                                     │
    │  2. Validate paths                                                      │
    │  3. Initialize UniversityNormalizer                                    │
    │  4. Process each year's CSV                                             │
    │  5. Print report                                                        │
    │  6. Return exit code (0 = success, 1 = failure)                        │
    │                                                                         │
    └────────────────────────────────────────────────────────────────────────┘

    Returns:
        Exit code: 0 for success, 1 for failure

    IMPLEMENTATION STEPS:
    ─────────────────────
    1. args = parse_args()
    2. if not validate_paths(args): return 1
    3. normalizer = UniversityNormalizer(args.mapping, args.threshold)
    4. for year in COLUMN_MAPPING:
           process_year(year, normalizer, args.raw_dir, args.output_dir,
                       args.dry_run, args.verbose)
    5. print_report(normalizer, args.verbose)
    6. return 0 if match_rate > 95% else 1
    """
    args = parse_args()

    if not validate_paths(args):
        return 1

    from src.utils.normalize import UniversityNormalizer
    normalizer = UniversityNormalizer(args.mapping, fuzzy_threshold=args.threshold)

    for year in COLUMN_MAPPING:
        process_year(
            year, normalizer, args.raw_dir, args.output_dir,
            args.dry_run, args.verbose
        )

    print_report(normalizer, args.verbose)

    report = normalizer.get_report()
    return 0 if report['match_rate'] > 95.0 else 1


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
│  [ ] Implement parse_args()                                                 │
│  [ ] Implement validate_paths()                                             │
│  [ ] Implement load_csv()                                                   │
│  [ ] Implement process_year()                                               │
│  [ ] Implement main()                                                       │
│                                                                              │
│  MEDIUM PRIORITY:                                                            │
│  ────────────────                                                            │
│  [ ] Implement print_report()                                               │
│  [ ] Add progress bars with tqdm                                            │
│  [ ] Add logging support                                                    │
│                                                                              │
│  TESTING:                                                                    │
│  ────────                                                                    │
│  [ ] Test with sample CSV files                                             │
│  [ ] Test with missing files                                                │
│  [ ] Test dry-run mode                                                      │
│  [ ] Verify output format matches expected                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
"""


if __name__ == "__main__":
    sys.exit(main())
