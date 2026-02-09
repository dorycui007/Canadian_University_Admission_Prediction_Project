"""
Program Analytics Service
=========================

Reads historical CSV data and computes per-program admission analytics
for the Program Detail page.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from src.utils.normalize import ProgramNormalizer, UniversityNormalizer


DATA_DIR = Path(__file__).parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
MAPPINGS_DIR = DATA_DIR / "mappings"

# Module-level cache
_cached_df: Optional[pd.DataFrame] = None


# =============================================================================
#                          GRADE PARSING
# =============================================================================

_DESCRIPTIVE_GRADE_MAP = {
    "low 70": 72, "low 70s": 72, "low-70s": 72,
    "mid 70": 75, "mid 70s": 75, "mid-70s": 75,
    "high 70": 78, "high 70s": 78, "high-70s": 78,
    "low 80": 82, "low 80s": 82, "low-80s": 82,
    "low-mid 80": 82, "low-mid 80s": 82,
    "mid 80": 85, "mid 80s": 85, "mid-80s": 85,
    "mid-high 80": 87, "mid-high 80s": 87,
    "high 80": 88, "high 80s": 88, "high-80s": 88,
    "low 90": 91, "low 90s": 91, "low-90s": 91,
    "low-mid 90": 92, "low-mid 90s": 92,
    "mid 90": 95, "mid 90s": 95, "mid-90s": 95,
    "mid-high 90": 97, "mid-high 90s": 97,
    "high 90": 98, "high 90s": 98, "high-90s": 98,
}


def parse_grade(raw: str) -> Optional[float]:
    """Parse messy grade strings into float values."""
    if not isinstance(raw, str):
        return None
    s = raw.strip().rstrip("%").strip()
    if not s:
        return None

    # Try direct float conversion
    try:
        val = float(s)
        if 0 <= val <= 100:
            return val
        return None
    except ValueError:
        pass

    # Try range like "85-90" or "85 - 90"
    range_match = re.match(r"^(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?)$", s)
    if range_match:
        lo, hi = float(range_match.group(1)), float(range_match.group(2))
        mid = (lo + hi) / 2
        if 0 <= mid <= 100:
            return mid
        return None

    # Try descriptive grades
    normalized = s.lower().strip()
    if normalized in _DESCRIPTIVE_GRADE_MAP:
        return float(_DESCRIPTIVE_GRADE_MAP[normalized])

    # Try partial match: "low 90s" within a longer string
    for desc, val in _DESCRIPTIVE_GRADE_MAP.items():
        if desc in normalized:
            return float(val)

    return None


# =============================================================================
#                          DECISION NORMALIZATION
# =============================================================================

def _load_decision_map() -> dict[str, str]:
    """Build a lookup from decision variation -> canonical name."""
    path = MAPPINGS_DIR / "decisions.yaml"
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    lookup: dict[str, str] = {}
    for canonical, variations in raw.items():
        lookup[canonical.lower()] = canonical
        for v in variations:
            lookup[v.lower().strip()] = canonical
    return lookup


_DECISION_MAP: Optional[dict[str, str]] = None


def normalize_decision(raw: str) -> Optional[str]:
    """Normalize a decision string to its canonical form."""
    global _DECISION_MAP
    if _DECISION_MAP is None:
        _DECISION_MAP = _load_decision_map()
    if not isinstance(raw, str):
        return None
    key = raw.strip().lower()
    # Handle compound decisions like "Deferred to General Science"
    # by checking the first word
    if key in _DECISION_MAP:
        return _DECISION_MAP[key]
    first_word = key.split()[0] if key else ""
    if first_word in _DECISION_MAP:
        return _DECISION_MAP[first_word]
    return None


# =============================================================================
#                          CSV LOADING
# =============================================================================

# Column mappings per CSV file
_COL_MAP_2022 = {
    "Which university was this program from? ": "university",
    "Which university was this program from?": "university",
    "What program did you apply to?": "program",
    "What was your average when accepted? (average for particular program)": "top_6_average",
    "Accepted, rejected, waitlisted, or deferred? (if deffered let us know to which program)": "decision",
    "Are you a 101 or 105 applicant?": "applicant_type",
    "What date was the offer/rejection/deferral received?": "decision_date_raw",
}

_COL_MAP_2023 = {
    "What university did you apply to?": "university",
    "What program did you apply to?": "program",
    "Acceptance Average": "top_6_average",
    "Were you accepted, rejected, waitlisted or deferred?": "decision",
    "Province/State/Territory of Residence": "province",
    "Country of Citizenship": "citizenship",
    "What date did you receive the offer/rejection/deferral?": "decision_date_raw",
    "Grade 11 Final Average": "g11_avg_raw",
    "Grade 12 Midterm Average": "g12_midterm_raw",
    "Grade 12 Final Average": "g12_final_raw",
}

_COL_MAP_2024 = {
    "University": "university",
    "Program name": "program",
    "Top 6 Average": "top_6_average",
    "Decision": "decision",
    "Province": "province",
    "Citizenship": "citizenship",
    "Date of decision": "decision_date_raw",
}


def _load_csv(filename: str, col_map: dict[str, str], cycle_year: str) -> pd.DataFrame:
    """Load a single CSV, rename columns, add cycle_year."""
    path = RAW_DIR / filename
    df = pd.read_csv(path, encoding="utf-8-sig")
    # Rename only columns that exist
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=rename)
    df["cycle_year"] = cycle_year
    # Keep only the columns we care about
    keep = [
        "university", "program", "top_6_average", "decision", "province",
        "citizenship", "applicant_type", "cycle_year",
        "decision_date_raw", "g11_avg_raw", "g12_midterm_raw", "g12_final_raw",
    ]
    for col in keep:
        if col not in df.columns:
            df[col] = None
    return df[keep]


_MONTH_ABBR = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "january": 1, "february": 2, "march": 3, "april": 4,
    "june": 6, "july": 7, "august": 8, "september": 9,
    "october": 10, "november": 11, "december": 12,
}

_MONTH_NAMES = [
    "", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
]


def _parse_decision_date(raw: str) -> Optional[int]:
    """Parse messy decision date strings into a month number (1-12).

    Handles: "November 26", "Dec 1", "11/9/2024", "29 Nov 2023",
    "december 7", "6 Oct 2023", "2024-02-15".
    """
    if not isinstance(raw, str):
        return None
    s = raw.strip()
    if not s:
        return None

    # Try M/D/YYYY or MM/DD/YYYY
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{2,4})$", s)
    if m:
        month = int(m.group(1))
        if 1 <= month <= 12:
            return month

    # Try YYYY-MM-DD
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m:
        month = int(m.group(2))
        if 1 <= month <= 12:
            return month

    # Try "Month Day" or "Month Day, Year" ("November 26", "Dec 1, 2023")
    m = re.match(r"^([a-zA-Z]+)\s+\d", s)
    if m:
        month_str = m.group(1).lower()
        if month_str in _MONTH_ABBR:
            return _MONTH_ABBR[month_str]

    # Try "Day Month Year" ("29 Nov 2023", "6 Oct 2023")
    m = re.match(r"^\d+\s+([a-zA-Z]+)", s)
    if m:
        month_str = m.group(1).lower()
        if month_str in _MONTH_ABBR:
            return _MONTH_ABBR[month_str]

    return None


def _normalize_applicant_type(row: pd.Series) -> Optional[str]:
    """Derive '101' or '105' from available columns."""
    # 2022-2023: has explicit applicant_type column
    at = row.get("applicant_type")
    if pd.notna(at):
        s = str(at).strip()
        if "101" in s:
            return "101"
        if "105" in s:
            return "105"

    # 2023-2024 / 2024-2025: derive from citizenship
    cit = row.get("citizenship")
    if pd.notna(cit):
        s = str(cit).strip().lower()
        if s in ("canada", "canadian"):
            return "101"
        if s and s not in ("none", "nan", "n/a", ""):
            return "105"

    return None


def load_all_records() -> pd.DataFrame:
    """Load and normalize all CSV files. Cached at module level."""
    global _cached_df
    if _cached_df is not None:
        return _cached_df

    dfs = [
        _load_csv("2022_2023_Canadian_University_Results.csv", _COL_MAP_2022, "2022-2023"),
        _load_csv("2023_2024_Canadian_University_Results.csv", _COL_MAP_2023, "2023-2024"),
        _load_csv("2024_2025_Canadian_University_Results.csv", _COL_MAP_2024, "2024-2025"),
    ]
    df = pd.concat(dfs, ignore_index=True)

    # Parse grades
    df["grade"] = df["top_6_average"].apply(
        lambda x: parse_grade(str(x)) if pd.notna(x) else None
    )

    # Normalize decisions
    df["decision_canonical"] = df["decision"].apply(
        lambda x: normalize_decision(str(x)) if pd.notna(x) else None
    )

    # Clean university and program: strip whitespace
    df["university"] = df["university"].apply(
        lambda x: str(x).strip() if pd.notna(x) else None
    )
    df["program"] = df["program"].apply(
        lambda x: str(x).strip() if pd.notna(x) else None
    )

    # Normalize university names to canonical form
    uni_normalizer = UniversityNormalizer(str(MAPPINGS_DIR / "universities.yaml"))
    df["university"] = df["university"].apply(
        lambda x: uni_normalizer.normalize(x) if pd.notna(x) and x else None
    )
    df = df[df["university"] != "INVALID"]

    # Strip university name prefix from program field (users sometimes type
    # "University of Toronto Computer Science" in the program field)
    def _strip_uni_prefix(row):
        prog = row["program"]
        uni = row["university"]
        if pd.notna(prog) and pd.notna(uni) and isinstance(prog, str) and isinstance(uni, str):
            if prog.startswith(uni):
                prog = prog[len(uni):].lstrip(" -:")
        return prog
    df["program"] = df.apply(_strip_uni_prefix, axis=1)

    # Normalize program names to canonical form
    prog_normalizer = ProgramNormalizer(
        str(MAPPINGS_DIR / "base_programs.yaml"),
        degree_mapping_path=str(MAPPINGS_DIR / "degree_abbreviations.yaml"),
    )
    df["program"] = df["program"].apply(
        lambda x: prog_normalizer.normalize(x) if pd.notna(x) and x else None
    )

    # University-specific program overrides — resolves cases where the same
    # name means different things at different universities
    _UNIVERSITY_PROGRAM_OVERRIDES = {
        # Western: "Management" is BMOS (the actual program name)
        ("Western University", "Management"): "Management and Organizational Studies",
        # Waterloo: plain "Accounting" is AFM
        ("University of Waterloo", "Accounting"): "Accounting and Financial Management",
        ("University of Waterloo", "Accounting/Finance"): "Accounting and Financial Management",
        ("University of Waterloo", "Mathematics/Business"): "Science and Business",
        # TMU: Management/Commerce → Business Administration (Ted Rogers)
        ("Toronto Metropolitan University", "Management"): "Business Administration",
        ("Toronto Metropolitan University", "Commerce"): "Business Administration",
        # Brock/Trent/Dalhousie: "Medical Sciences" is Biomedical Sciences
        ("Brock University", "Medical Sciences"): "Biomedical Sciences",
        ("Trent University", "Medical Sciences"): "Biomedical Sciences",
        ("Dalhousie University", "Medical Sciences"): "Biomedical Sciences",
        # Trent: streams → parent programs
        ("Trent University", "Medical Professional Stream"): "Biomedical Sciences",
        ("Trent University", "Science Teacher Ed Stream"): "Education",
        # Ontario Tech: Commerce → Business Administration
        ("Ontario Tech University", "Commerce"): "Business Administration",
    }
    def _apply_overrides(row):
        key = (row["university"], row["program"])
        return _UNIVERSITY_PROGRAM_OVERRIDES.get(key, row["program"])
    df["program"] = df.apply(_apply_overrides, axis=1)

    # Filter out garbage program names
    _GARBAGE_PROGRAMS = {
        "none", "idk", "n/a", "nan", "non", "co op", "queens", "western",
        "fgg", "ff", "hh", "ba&", "hb1", "laurier", "(ubcv)",
        "science", "arts", "engineering",  # too generic — not a program name
        "multiple leadership roles internship sports teams",
        "multiple leadership roles sports teams",
        "faculty of science", "faculty of arts",
        "bachelors of science", "bachelor of science",
        "honors sci", "honors science",
        "health sci on", "on",
        "with", "managment with", "with major",
        "arts with", "arts university",
        "bba and co op",
        "western science",
        "arts honors",
        "/cs dd",
        "life sci (neuroscience0",
        "laurier /fmath + laurier bba",
        # Catch-all / placeholder entries
        "all other majors",
        "university of ottawa uo",
        "undeclared arts",
        "b. arts all subjects available",
        "b. science all subjects available",
        "bgins",
        "arts programs all arts majors",
        "arts programs all other arts majors",
        # Combined degree descriptors (not real program names)
        "arts degree and mba",
        "arts degree and master's degree",
        "arts degree and master's degree global studies",
        "law degree and arts degree",
        "and juris doctor",
        "in ophthalmic medical technology",
        "double degree bba and u of t scarborough",
        "integrated media non co op",
        # Slash-based combined degrees (not standalone programs)
        "uo political science/juris doctor",
        "public administration/political science",
        "trent/swansea dual degree criminology/ ll.b. law",
    }
    def _is_valid_program(name):
        if not name or not isinstance(name, str):
            return False
        cleaned = name.strip()
        if len(cleaned) < 3:
            return False
        if cleaned.lower() in _GARBAGE_PROGRAMS:
            return False
        # Filter out entries that are mostly punctuation/numbers
        alpha_chars = sum(1 for c in cleaned if c.isalpha())
        if alpha_chars < 3:
            return False
        # Filter out entries starting with "/"
        if cleaned.startswith("/"):
            return False
        # Filter out entries with unbalanced/trailing parentheses
        if cleaned.endswith("(") or cleaned.endswith(")"):
            return False
        # Filter out overly long free-text entries (> 6 words, likely not a program)
        if len(cleaned.split()) > 6:
            return False
        return True
    df = df[df["program"].apply(_is_valid_program)]

    # Clean province
    df["province"] = df["province"].apply(
        lambda x: str(x).strip() if pd.notna(x) else None
    )

    # Parse decision month
    df["decision_month"] = df["decision_date_raw"].apply(
        lambda x: _parse_decision_date(str(x)) if pd.notna(x) else None
    )

    # Parse grade progression columns (2023-24 only, others will be NaN)
    for col in ["g11_avg_raw", "g12_midterm_raw", "g12_final_raw"]:
        parsed_col = col.replace("_raw", "")
        df[parsed_col] = df[col].apply(
            lambda x: parse_grade(str(x)) if pd.notna(x) else None
        )

    # Normalize applicant type (101/105)
    df["applicant_type_norm"] = df.apply(_normalize_applicant_type, axis=1)

    _cached_df = df
    return df


# =============================================================================
#                          ANALYTICS COMPUTATION
# =============================================================================

def _compute_grade_stats(grades: pd.Series) -> Optional[dict]:
    """Compute grade statistics for a series of grades."""
    valid = grades.dropna()
    if len(valid) == 0:
        return None
    return {
        "mean": round(float(valid.mean()), 1),
        "median": round(float(valid.median()), 1),
        "p25": round(float(np.percentile(valid, 25)), 1),
        "p75": round(float(np.percentile(valid, 75)), 1),
        "min": round(float(valid.min()), 1),
        "max": round(float(valid.max()), 1),
        "std": round(float(valid.std()), 1) if len(valid) > 1 else 0.0,
        "n": int(len(valid)),
    }


def _compute_distribution(accepted_grades: pd.Series, rejected_grades: pd.Series) -> dict:
    """Compute histogram bins matching DistributionData format.

    Bins are 2-point intervals derived from the actual grade range,
    so programs with grades only in 90-100 won't have empty 60-88 bins.
    """
    all_grades = pd.concat([accepted_grades, rejected_grades]).dropna()
    if len(all_grades) == 0:
        return {"bins": [], "counts_accepted": [], "counts_rejected": []}

    step = 2
    # Round min down and max up to nearest step boundary
    lo_bound = int(all_grades.min() // step * step)
    hi_bound = int(-(-all_grades.max() // step) * step)  # ceil to step
    # Clamp to 0-100
    lo_bound = max(0, lo_bound)
    hi_bound = min(100, hi_bound)
    if hi_bound <= lo_bound:
        hi_bound = lo_bound + step

    bins = list(range(lo_bound, hi_bound + 1, step))
    if bins[-1] < hi_bound:
        bins.append(hi_bound)

    counts_accepted = []
    counts_rejected = []

    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        if i == len(bins) - 2:
            # Last bin is inclusive on both ends
            acc = int(((accepted_grades >= lo) & (accepted_grades <= hi)).sum())
            rej = int(((rejected_grades >= lo) & (rejected_grades <= hi)).sum())
        else:
            acc = int(((accepted_grades >= lo) & (accepted_grades < hi)).sum())
            rej = int(((rejected_grades >= lo) & (rejected_grades < hi)).sum())
        counts_accepted.append(acc)
        counts_rejected.append(rej)

    return {
        "bins": bins,
        "counts_accepted": counts_accepted,
        "counts_rejected": counts_rejected,
    }


# =============================================================================
#                  COMPETITIVENESS HELPERS
# =============================================================================


def _difficulty_label(median_accepted: Optional[float]) -> str:
    """Classify program difficulty from the median accepted grade."""
    if median_accepted is None:
        return "Unknown"
    if median_accepted >= 93:
        return "Very Competitive"
    if median_accepted >= 88:
        return "Competitive"
    if median_accepted >= 82:
        return "Moderate"
    return "Accessible"


def _confidence_level(n: int) -> str:
    """Classify sample size into a confidence tier."""
    if n >= 30:
        return "high"
    if n >= 10:
        return "moderate"
    return "low"


def _admitted_grade_range(grades: pd.Series) -> Optional[dict]:
    """Compute the grade range for admitted students."""
    valid = grades.dropna()
    if len(valid) == 0:
        return None
    return {
        "min": round(float(valid.min()), 1),
        "p25": round(float(np.percentile(valid, 25)), 1),
        "median": round(float(valid.median()), 1),
        "p75": round(float(np.percentile(valid, 75)), 1),
        "max": round(float(valid.max()), 1),
        "n": int(len(valid)),
    }


def _compute_offer_timeline(subset: pd.DataFrame) -> dict:
    """Compute when offers/decisions come out, by month."""
    months = subset["decision_month"].dropna().astype(int)
    if len(months) == 0:
        return {"by_month": [], "total_with_dates": 0}

    # Count per month, ordered chronologically
    counts = months.value_counts().sort_index()
    by_month = [
        {"month": _MONTH_NAMES[int(m)], "count": int(c)}
        for m, c in counts.items() if 1 <= m <= 12
    ]

    sorted_months = sorted(months.tolist())
    median_idx = len(sorted_months) // 2

    return {
        "earliest_month": _MONTH_NAMES[sorted_months[0]],
        "median_month": _MONTH_NAMES[sorted_months[median_idx]],
        "latest_month": _MONTH_NAMES[sorted_months[-1]],
        "by_month": by_month,
        "total_with_dates": int(len(months)),
    }


def _compute_grade_progression(subset: pd.DataFrame) -> Optional[dict]:
    """Compute grade trajectory (G11 → G12 midterm → final) for accepted students."""
    acc = subset[subset["decision_canonical"] == "Accepted"]

    g11 = acc["g11_avg"].dropna()
    g12_mid = acc["g12_midterm"].dropna()
    g12_final = acc["g12_final"].dropna()

    # Need at least some data for this to be meaningful
    if len(g11) < 2 and len(g12_mid) < 2:
        return None

    result: dict = {"n": 0}
    if len(g11) > 0:
        result["g11_avg"] = round(float(g11.mean()), 1)
        result["g11_median"] = round(float(g11.median()), 1)
        result["g11_n"] = int(len(g11))
        result["n"] = max(result["n"], int(len(g11)))
    if len(g12_mid) > 0:
        result["g12_midterm_avg"] = round(float(g12_mid.mean()), 1)
        result["g12_midterm_median"] = round(float(g12_mid.median()), 1)
        result["g12_midterm_n"] = int(len(g12_mid))
        result["n"] = max(result["n"], int(len(g12_mid)))
    if len(g12_final) > 0:
        result["g12_final_avg"] = round(float(g12_final.mean()), 1)
        result["g12_final_median"] = round(float(g12_final.median()), 1)
        result["g12_final_n"] = int(len(g12_final))
        result["n"] = max(result["n"], int(len(g12_final)))

    return result if result["n"] > 0 else None


def _compute_applicant_type_breakdown(subset: pd.DataFrame) -> dict:
    """Count 101 vs 105 applicants."""
    types = subset["applicant_type_norm"].dropna()
    counts = types.value_counts().to_dict()
    total_known = len(types)
    total = len(subset)
    return {
        "101": int(counts.get("101", 0)),
        "105": int(counts.get("105", 0)),
        "unknown": total - total_known,
    }


def _fuzzy_match(value: str, target: str) -> bool:
    """Case-insensitive substring match for university/program filtering."""
    v = value.lower().strip()
    t = target.lower().strip()
    return t in v or v in t or v == t


def get_program_analytics(university: str, program: str) -> Optional[dict]:
    """Compute comprehensive analytics for a (university, program) pair."""
    df = load_all_records()

    # Try exact match first (data is now normalized)
    mask = (df["university"] == university) & (df["program"] == program)
    subset = df[mask]

    # Fall back to fuzzy matching for old bookmarked URLs
    if len(subset) == 0:
        mask = df["university"].apply(
            lambda x: _fuzzy_match(str(x), university) if pd.notna(x) else False
        ) & df["program"].apply(
            lambda x: _fuzzy_match(str(x), program) if pd.notna(x) else False
        )
        subset = df[mask]

    if len(subset) == 0:
        return None

    total = len(subset)
    cycle_years = sorted(subset["cycle_year"].dropna().unique().tolist())

    # --- Decision breakdown ---
    decisions = subset["decision_canonical"].value_counts().to_dict()

    # --- Competitiveness (grade-based, not acceptance-rate-based) ---
    accepted_mask = subset["decision_canonical"] == "Accepted"
    rejected_mask = subset["decision_canonical"] == "Rejected"
    accepted_grades = subset.loc[accepted_mask, "grade"].dropna()
    rejected_grades = subset.loc[rejected_mask, "grade"].dropna()

    admitted_range = _admitted_grade_range(accepted_grades)
    median_acc = admitted_range["median"] if admitted_range else None

    # Per-year admitted grade ranges
    grades_by_year = []
    for cy in cycle_years:
        year_subset = subset[subset["cycle_year"] == cy]
        year_acc_grades = year_subset.loc[
            year_subset["decision_canonical"] == "Accepted", "grade"
        ].dropna()
        year_range = _admitted_grade_range(year_acc_grades)
        year_total = len(year_subset)
        grades_by_year.append({
            "year": cy,
            "admitted_range": year_range,
            "total_reports": year_total,
            "confidence_level": _confidence_level(year_total),
        })

    # --- Grade statistics ---
    all_grades = subset["grade"].dropna()

    grade_stats = {
        "all": _compute_grade_stats(all_grades),
        "accepted": _compute_grade_stats(accepted_grades),
        "rejected": _compute_grade_stats(rejected_grades),
    }

    # --- Distribution (histogram) ---
    distribution = _compute_distribution(accepted_grades, rejected_grades)
    dist_all_stats = grade_stats["all"] or {
        "mean": 0, "median": 0, "p25": 0, "p75": 0, "min": 0, "max": 0, "n": 0
    }
    distribution["statistics"] = dist_all_stats

    # --- Year trend ---
    year_trend = []
    for cy in cycle_years:
        year_subset = subset[subset["cycle_year"] == cy]
        year_grades = year_subset["grade"].dropna()
        year_acc_grades = year_subset.loc[
            year_subset["decision_canonical"] == "Accepted", "grade"
        ].dropna()
        # Extract the start year from "2022-2023" -> 2022
        start_year = int(cy.split("-")[0])
        year_trend.append({
            "year": start_year,
            "median_grade": round(float(year_grades.median()), 1) if len(year_grades) > 0 else 0,
            "median_accepted": round(float(year_acc_grades.median()), 1) if len(year_acc_grades) > 0 else 0,
            "count": int(len(year_subset)),
        })

    # --- Province breakdown ---
    province_counts = {}
    for _, row in subset.iterrows():
        prov = row.get("province")
        if pd.isna(prov) or not str(prov).strip() or str(prov).strip().lower() in ("none", "nan", "n/a"):
            prov_key = "Unknown"
        else:
            prov_key = str(prov).strip()
        province_counts[prov_key] = province_counts.get(prov_key, 0) + 1

    # --- Offer timeline ---
    offer_timeline = _compute_offer_timeline(subset)

    # --- Grade progression (G11 → G12) ---
    grade_progression = _compute_grade_progression(subset)

    # --- Applicant type (101 vs 105) ---
    applicant_type = _compute_applicant_type_breakdown(subset)

    # --- Data quality ---
    missing_grade_pct = round(1.0 - (len(all_grades) / total), 3) if total > 0 else 1.0

    sparse_warning = None
    if total < 3:
        sparse_warning = f"Very limited data ({total} record{'s' if total != 1 else ''}). Only basic counts are shown."
    elif total < 10:
        sparse_warning = f"This program has limited data ({total} records). Statistics may not be reliable."

    return {
        "university": university,
        "program": program,
        "total_records": total,
        "cycle_years": cycle_years,
        "competitiveness": {
            "difficulty": _difficulty_label(median_acc),
            "admitted_range": admitted_range,
            "sample_size": total,
            "confidence_level": _confidence_level(total),
            "by_year": grades_by_year,
        },
        "decision_breakdown": {k: int(v) for k, v in decisions.items() if k is not None},
        "grade_statistics": grade_stats,
        "distribution": distribution,
        "year_trend": year_trend,
        "offer_timeline": offer_timeline,
        "grade_progression": grade_progression,
        "applicant_type": applicant_type,
        "province_breakdown": province_counts,
        "data_quality": {
            "missing_grade_pct": missing_grade_pct,
            "earliest_cycle": cycle_years[0] if cycle_years else None,
            "latest_cycle": cycle_years[-1] if cycle_years else None,
            "sparse_warning": sparse_warning,
        },
    }


def get_program_listing() -> list[dict]:
    """Return all unique (university, program) pairs with summary stats."""
    df = load_all_records()

    # Group by university + program
    groups = df.groupby(["university", "program"], dropna=False)

    listing = []
    for (uni, prog), group in groups:
        if pd.isna(uni) or pd.isna(prog):
            continue
        uni_str = str(uni).strip()
        prog_str = str(prog).strip()
        if not uni_str or not prog_str:
            continue

        total = len(group)
        acc_grades = group.loc[group["decision_canonical"] == "Accepted", "grade"].dropna()
        median_acc = round(float(acc_grades.median()), 1) if len(acc_grades) > 0 else None
        avg_acc = round(float(acc_grades.mean()), 1) if len(acc_grades) > 0 else 0.0

        listing.append({
            "university": uni_str,
            "program": prog_str,
            "total_records": total,
            "difficulty": _difficulty_label(median_acc),
            "median_grade_accepted": median_acc,
            "confidence_level": _confidence_level(total),
            "avg_grade_accepted": avg_acc,
        })

    # Sort by total records descending
    listing.sort(key=lambda x: x["total_records"], reverse=True)
    return listing
