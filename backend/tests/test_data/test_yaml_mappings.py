"""
Tests for YAML mapping data files.

Validates the structure and content of the YAML mapping files in
data/mappings/ independently of any source code that consumes them.
"""

import os
import pytest
import yaml


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path):
    """Load a YAML file and return its parsed contents."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ===========================================================================
#  TestUniversitiesYAML
# ===========================================================================

class TestUniversitiesYAML:
    """Validate data/mappings/universities.yaml structure and content."""

    @pytest.fixture(autouse=True)
    def _setup(self, mappings_dir):
        self.path = os.path.join(mappings_dir, "universities.yaml")

    # 1 ---------------------------------------------------------------
    def test_file_exists(self):
        assert os.path.isfile(self.path), (
            f"universities.yaml not found at {self.path}"
        )

    # 2 ---------------------------------------------------------------
    def test_valid_yaml(self):
        data = _load_yaml(self.path)
        assert isinstance(data, dict), (
            "universities.yaml should parse to a dict"
        )

    # 3 ---------------------------------------------------------------
    def test_all_values_are_lists(self):
        data = _load_yaml(self.path)
        for uni, aliases in data.items():
            assert isinstance(aliases, list), (
                f"Expected list of aliases for '{uni}', got {type(aliases).__name__}"
            )

    # 4 ---------------------------------------------------------------
    def test_no_empty_alias_lists(self):
        data = _load_yaml(self.path)
        for uni, aliases in data.items():
            assert len(aliases) > 0, (
                f"University '{uni}' has an empty alias list"
            )

    # 5 ---------------------------------------------------------------
    def test_no_duplicate_variations_across_universities(self):
        """Every alias (lowered) must belong to exactly one canonical name."""
        data = _load_yaml(self.path)
        seen = {}
        duplicates = []
        for uni, aliases in data.items():
            for alias in aliases:
                key = alias.strip().lower()
                if key in seen and seen[key] != uni:
                    duplicates.append(
                        f"'{alias}' appears under both '{seen[key]}' and '{uni}'"
                    )
                else:
                    seen[key] = uni
        assert duplicates == [], (
            "Duplicate alias variations found:\n" + "\n".join(duplicates)
        )


# ===========================================================================
#  TestBaseProgramsYAML
# ===========================================================================

class TestBaseProgramsYAML:
    """Validate data/mappings/base_programs.yaml structure and content."""

    @pytest.fixture(autouse=True)
    def _setup(self, mappings_dir):
        self.path = os.path.join(mappings_dir, "base_programs.yaml")

    # 1 ---------------------------------------------------------------
    def test_file_exists(self):
        assert os.path.isfile(self.path), (
            f"base_programs.yaml not found at {self.path}"
        )

    # 2 ---------------------------------------------------------------
    def test_valid_yaml(self):
        data = _load_yaml(self.path)
        assert isinstance(data, dict), (
            "base_programs.yaml should parse to a dict"
        )

    # 3 ---------------------------------------------------------------
    def test_all_values_are_lists(self):
        data = _load_yaml(self.path)
        for program, variations in data.items():
            assert isinstance(variations, list), (
                f"Expected list of variations for '{program}', "
                f"got {type(variations).__name__}"
            )

    # 4 ---------------------------------------------------------------
    def test_no_empty_variation_lists(self):
        data = _load_yaml(self.path)
        for program, variations in data.items():
            assert len(variations) > 0, (
                f"Program '{program}' has an empty variation list"
            )


# ===========================================================================
#  TestDecisionsYAML
# ===========================================================================

class TestDecisionsYAML:
    """Validate data/mappings/decisions.yaml structure and content."""

    @pytest.fixture(autouse=True)
    def _setup(self, mappings_dir):
        self.path = os.path.join(mappings_dir, "decisions.yaml")

    # 1 ---------------------------------------------------------------
    def test_file_exists(self):
        assert os.path.isfile(self.path), (
            f"decisions.yaml not found at {self.path}"
        )

    # 2 ---------------------------------------------------------------
    def test_valid_yaml(self):
        data = _load_yaml(self.path)
        assert isinstance(data, dict), (
            "decisions.yaml should parse to a dict"
        )

    # 3 ---------------------------------------------------------------
    def test_contains_expected_decisions(self):
        data = _load_yaml(self.path)
        expected = {"Accepted", "Rejected", "Waitlisted"}
        actual = set(data.keys())
        missing = expected - actual
        assert not missing, (
            f"decisions.yaml is missing expected decisions: {missing}"
        )

    # 4 ---------------------------------------------------------------
    def test_all_values_are_lists(self):
        data = _load_yaml(self.path)
        for decision, variations in data.items():
            assert isinstance(variations, list), (
                f"Expected list of variations for '{decision}', "
                f"got {type(variations).__name__}"
            )


# ===========================================================================
#  TestDegreeAbbreviationsYAML
# ===========================================================================

class TestDegreeAbbreviationsYAML:
    """Validate data/mappings/degree_abbreviations.yaml structure and content."""

    @pytest.fixture(autouse=True)
    def _setup(self, mappings_dir):
        self.path = os.path.join(mappings_dir, "degree_abbreviations.yaml")

    # 1 ---------------------------------------------------------------
    def test_file_exists(self):
        assert os.path.isfile(self.path), (
            f"degree_abbreviations.yaml not found at {self.path}"
        )

    # 2 ---------------------------------------------------------------
    def test_valid_yaml(self):
        data = _load_yaml(self.path)
        assert isinstance(data, dict), (
            "degree_abbreviations.yaml should parse to a dict"
        )

    # 3 ---------------------------------------------------------------
    def test_entries_have_full_name(self):
        """Every degree abbreviation entry must have a 'full_name' field."""
        data = _load_yaml(self.path)
        missing = [
            abbrev for abbrev, info in data.items()
            if not isinstance(info, dict) or "full_name" not in info
        ]
        assert missing == [], (
            f"Degree entries missing 'full_name': {missing}"
        )
