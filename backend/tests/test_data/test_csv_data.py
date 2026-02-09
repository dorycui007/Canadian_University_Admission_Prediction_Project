"""
Tests for raw CSV data files.

Validates the structure and content of the CSV files in data/raw/
independently of any source code that consumes them.
"""

import os
import glob
import pytest
import pandas as pd


# ---------------------------------------------------------------------------
#  CSV discovery and metadata
# ---------------------------------------------------------------------------

# Column that holds the university name differs by survey year.
UNIVERSITY_COLUMN = {
    "2024_2025": "University",
    "2023_2024": "What university did you apply to?",
    "2022_2023": "Which university was this program from? ",
}

# Column that holds the admission decision differs by survey year.
DECISION_COLUMN = {
    "2024_2025": "Decision",
    "2023_2024": "Were you accepted, rejected, waitlisted or deferred?",
    "2022_2023": "Accepted, rejected, waitlisted, or deferred? (if deffered let us know to which program)",
}

# Column that holds the program name differs by survey year.
PROGRAM_COLUMN = {
    "2024_2025": "Program name",
    "2023_2024": "What program did you apply to?",
    "2022_2023": "What program did you apply to?",
}

# Average / grade column that should contain numeric-like values.
AVERAGE_COLUMN = {
    "2024_2025": "Top 6 Average",
    "2023_2024": "Grade 12 Final Average",
    "2022_2023": "What was your average when accepted? (average for particular program)",
}

# Minimum expected row counts per file (conservative lower bounds).
MIN_ROW_COUNTS = {
    "2022_2023": 50,
    "2023_2024": 50,
    "2024_2025": 50,
}


def _year_tag(csv_path):
    """Extract the year tag like '2024_2025' from the file name."""
    basename = os.path.basename(csv_path)
    # e.g. "2024_2025_Canadian_University_Results.csv" -> "2024_2025"
    parts = basename.split("_")
    return f"{parts[0]}_{parts[1]}"


# ---------------------------------------------------------------------------
#  Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def csv_files(raw_data_dir):
    """Return sorted list of CSV file paths in data/raw/."""
    pattern = os.path.join(raw_data_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    return files


def _get_csv_files():
    """Standalone helper used for parametrize (runs at collection time)."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(project_root, "data", "raw")
    pattern = os.path.join(raw_dir, "*.csv")
    return sorted(glob.glob(pattern))


def _csv_id(path):
    """Readable test ID from a CSV path."""
    return os.path.basename(path).replace(".csv", "")


# ===========================================================================
#  TestCSVStructure
# ===========================================================================

class TestCSVStructure:
    """Structural validation of each raw CSV file."""

    # --- file-level tests (parametrized) ---

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_file_exists(self, csv_path):
        assert os.path.isfile(csv_path), f"CSV not found: {csv_path}"

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_file_is_not_empty(self, csv_path):
        assert os.path.getsize(csv_path) > 0, f"CSV is empty: {csv_path}"

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_parseable_by_pandas(self, csv_path):
        df = pd.read_csv(csv_path)
        assert isinstance(df, pd.DataFrame)

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_has_rows(self, csv_path):
        df = pd.read_csv(csv_path)
        assert len(df) > 0, f"CSV has no data rows: {csv_path}"

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_minimum_row_count(self, csv_path):
        tag = _year_tag(csv_path)
        df = pd.read_csv(csv_path)
        min_rows = MIN_ROW_COUNTS.get(tag, 10)
        assert len(df) >= min_rows, (
            f"{os.path.basename(csv_path)} has {len(df)} rows, "
            f"expected at least {min_rows}"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_has_multiple_columns(self, csv_path):
        df = pd.read_csv(csv_path)
        assert len(df.columns) >= 5, (
            f"Expected at least 5 columns, got {len(df.columns)}"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_has_university_column(self, csv_path):
        tag = _year_tag(csv_path)
        expected_col = UNIVERSITY_COLUMN.get(tag)
        assert expected_col is not None, (
            f"No university column mapping for year tag '{tag}'"
        )
        df = pd.read_csv(csv_path)
        assert expected_col in df.columns, (
            f"Missing university column '{expected_col}' in "
            f"{os.path.basename(csv_path)}. Columns: {list(df.columns)}"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_has_decision_column(self, csv_path):
        tag = _year_tag(csv_path)
        expected_col = DECISION_COLUMN.get(tag)
        assert expected_col is not None, (
            f"No decision column mapping for year tag '{tag}'"
        )
        df = pd.read_csv(csv_path)
        assert expected_col in df.columns, (
            f"Missing decision column '{expected_col}' in "
            f"{os.path.basename(csv_path)}. Columns: {list(df.columns)}"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_has_program_column(self, csv_path):
        tag = _year_tag(csv_path)
        expected_col = PROGRAM_COLUMN.get(tag)
        assert expected_col is not None, (
            f"No program column mapping for year tag '{tag}'"
        )
        df = pd.read_csv(csv_path)
        assert expected_col in df.columns, (
            f"Missing program column '{expected_col}' in "
            f"{os.path.basename(csv_path)}. Columns: {list(df.columns)}"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_has_average_column(self, csv_path):
        tag = _year_tag(csv_path)
        expected_col = AVERAGE_COLUMN.get(tag)
        assert expected_col is not None, (
            f"No average column mapping for year tag '{tag}'"
        )
        df = pd.read_csv(csv_path)
        assert expected_col in df.columns, (
            f"Missing average column '{expected_col}' in "
            f"{os.path.basename(csv_path)}. Columns: {list(df.columns)}"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_has_timestamp_column(self, csv_path):
        df = pd.read_csv(csv_path)
        assert "Timestamp" in df.columns, (
            f"Missing 'Timestamp' column in {os.path.basename(csv_path)}"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_no_fully_empty_columns(self, csv_path):
        """No column should be 100% NaN (indicates a structural problem)."""
        df = pd.read_csv(csv_path)
        fully_empty = [col for col in df.columns if df[col].isna().all()]
        # Allow unnamed columns (often artifacts of trailing commas)
        meaningful_empty = [
            c for c in fully_empty if not c.startswith("Unnamed")
        ]
        assert meaningful_empty == [], (
            f"Fully empty columns in {os.path.basename(csv_path)}: "
            f"{meaningful_empty}"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_no_duplicate_column_names(self, csv_path):
        df = pd.read_csv(csv_path)
        cols = list(df.columns)
        dupes = [c for c in cols if cols.count(c) > 1]
        assert dupes == [], (
            f"Duplicate column names in {os.path.basename(csv_path)}: "
            f"{set(dupes)}"
        )

    # --- aggregate test (non-parametrized) ---

    def test_at_least_three_csv_files_present(self, csv_files):
        assert len(csv_files) >= 3, (
            f"Expected at least 3 CSV files in data/raw/, found {len(csv_files)}"
        )

    def test_each_csv_covers_distinct_year(self, csv_files):
        tags = [_year_tag(f) for f in csv_files]
        assert len(tags) == len(set(tags)), (
            f"Duplicate year tags found: {tags}"
        )

    def test_csv_files_follow_naming_convention(self, csv_files):
        for path in csv_files:
            basename = os.path.basename(path)
            assert "Canadian_University_Results" in basename, (
                f"CSV file '{basename}' does not follow expected naming convention "
                f"'YYYY_YYYY_Canadian_University_Results.csv'"
            )

    def test_all_csvs_have_timestamp(self, csv_files):
        for path in csv_files:
            df = pd.read_csv(path)
            assert "Timestamp" in df.columns, (
                f"Missing 'Timestamp' column in {os.path.basename(path)}"
            )


# ===========================================================================
#  TestCSVValues
# ===========================================================================

class TestCSVValues:
    """Value-level validation of each raw CSV file."""

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_university_names_non_empty(self, csv_path):
        tag = _year_tag(csv_path)
        col = UNIVERSITY_COLUMN[tag]
        df = pd.read_csv(csv_path)
        non_null = df[col].dropna()
        assert len(non_null) > 0, (
            f"University column '{col}' is entirely empty in "
            f"{os.path.basename(csv_path)}"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_university_names_are_strings(self, csv_path):
        tag = _year_tag(csv_path)
        col = UNIVERSITY_COLUMN[tag]
        df = pd.read_csv(csv_path)
        non_null = df[col].dropna()
        all_strings = non_null.apply(lambda v: isinstance(v, str)).all()
        assert all_strings, (
            f"Non-string values in university column '{col}'"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_university_names_not_blank(self, csv_path):
        """Non-null university names should not be blank strings."""
        tag = _year_tag(csv_path)
        col = UNIVERSITY_COLUMN[tag]
        df = pd.read_csv(csv_path)
        non_null = df[col].dropna()
        blank = non_null[non_null.str.strip() == ""]
        assert len(blank) == 0, (
            f"Found {len(blank)} blank university names in "
            f"{os.path.basename(csv_path)}"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_decision_values_non_empty(self, csv_path):
        tag = _year_tag(csv_path)
        col = DECISION_COLUMN[tag]
        df = pd.read_csv(csv_path)
        non_null = df[col].dropna()
        assert len(non_null) > 0, (
            f"Decision column '{col}' is entirely empty in "
            f"{os.path.basename(csv_path)}"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_decision_values_are_strings(self, csv_path):
        tag = _year_tag(csv_path)
        col = DECISION_COLUMN[tag]
        df = pd.read_csv(csv_path)
        non_null = df[col].dropna()
        all_strings = non_null.apply(lambda v: isinstance(v, str)).all()
        assert all_strings, (
            f"Non-string values in decision column '{col}'"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_decision_values_recognizable(self, csv_path):
        """At least 80% of non-null decisions should contain a known keyword."""
        known_keywords = {
            "accepted", "rejected", "waitlisted", "deferred", "pending",
            "withdrawn", "offer", "denied", "declined", "admit",
        }
        tag = _year_tag(csv_path)
        col = DECISION_COLUMN[tag]
        df = pd.read_csv(csv_path)
        non_null = df[col].dropna()
        if len(non_null) == 0:
            pytest.skip("No decision values to validate")

        def is_recognizable(val):
            lower = str(val).lower()
            return any(kw in lower for kw in known_keywords)

        recognised = non_null.apply(is_recognizable).sum()
        pct = recognised / len(non_null)
        assert pct >= 0.80, (
            f"Only {pct:.1%} of decisions are recognizable in "
            f"{os.path.basename(csv_path)} (expected >= 80%)"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_average_column_has_some_numeric_values(self, csv_path):
        """The average column should contain values convertible to numbers."""
        tag = _year_tag(csv_path)
        col = AVERAGE_COLUMN[tag]
        df = pd.read_csv(csv_path)
        raw = df[col].dropna().astype(str)
        # Strip %-signs and whitespace before conversion attempt
        cleaned = raw.str.replace("%", "", regex=False).str.strip()
        numeric = pd.to_numeric(cleaned, errors="coerce").dropna()
        assert len(numeric) > 0, (
            f"No numeric values found in average column '{col}' of "
            f"{os.path.basename(csv_path)}"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_averages_in_plausible_range(self, csv_path):
        """Numeric averages should fall between 0 and 100."""
        tag = _year_tag(csv_path)
        col = AVERAGE_COLUMN[tag]
        df = pd.read_csv(csv_path)
        raw = df[col].dropna().astype(str)
        cleaned = raw.str.replace("%", "", regex=False).str.strip()
        numeric = pd.to_numeric(cleaned, errors="coerce").dropna()
        if len(numeric) == 0:
            pytest.skip("No numeric averages to validate")
        out_of_range = numeric[(numeric < 0) | (numeric > 100)]
        pct_ok = 1 - len(out_of_range) / len(numeric)
        assert pct_ok >= 0.95, (
            f"Only {pct_ok:.1%} of averages are in [0, 100] in "
            f"{os.path.basename(csv_path)} (expected >= 95%)"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_program_names_non_empty(self, csv_path):
        tag = _year_tag(csv_path)
        col = PROGRAM_COLUMN[tag]
        df = pd.read_csv(csv_path)
        non_null = df[col].dropna()
        assert len(non_null) > 0, (
            f"Program column '{col}' is entirely empty in "
            f"{os.path.basename(csv_path)}"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_program_names_are_strings(self, csv_path):
        tag = _year_tag(csv_path)
        col = PROGRAM_COLUMN[tag]
        df = pd.read_csv(csv_path)
        non_null = df[col].dropna()
        all_strings = non_null.apply(lambda v: isinstance(v, str)).all()
        assert all_strings, (
            f"Non-string values in program column '{col}'"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_multiple_distinct_universities(self, csv_path):
        """Each CSV should reference more than one university."""
        tag = _year_tag(csv_path)
        col = UNIVERSITY_COLUMN[tag]
        df = pd.read_csv(csv_path)
        nunique = df[col].dropna().nunique()
        assert nunique > 1, (
            f"Only {nunique} distinct university value(s) in "
            f"{os.path.basename(csv_path)}"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_multiple_distinct_programs(self, csv_path):
        """Each CSV should reference more than one program."""
        tag = _year_tag(csv_path)
        col = PROGRAM_COLUMN[tag]
        df = pd.read_csv(csv_path)
        nunique = df[col].dropna().nunique()
        assert nunique > 1, (
            f"Only {nunique} distinct program value(s) in "
            f"{os.path.basename(csv_path)}"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_multiple_distinct_decisions(self, csv_path):
        """Each CSV should contain more than one decision type."""
        tag = _year_tag(csv_path)
        col = DECISION_COLUMN[tag]
        df = pd.read_csv(csv_path)
        nunique = df[col].dropna().nunique()
        assert nunique > 1, (
            f"Only {nunique} distinct decision value(s) in "
            f"{os.path.basename(csv_path)}"
        )

    @pytest.mark.parametrize("csv_path", _get_csv_files(), ids=_csv_id)
    def test_no_row_entirely_empty(self, csv_path):
        """No row should be 100% NaN (indicates a structural problem)."""
        df = pd.read_csv(csv_path)
        fully_empty_rows = df.isna().all(axis=1).sum()
        assert fully_empty_rows == 0, (
            f"Found {fully_empty_rows} fully empty row(s) in "
            f"{os.path.basename(csv_path)}"
        )
