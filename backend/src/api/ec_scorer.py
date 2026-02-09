"""
EC Scorer — scores extracurriculars by semantic similarity against historical admission data.

Embeds historical EC descriptions from CSV admission data using sentence-transformers,
then at prediction time finds the most similar past applicants and scores based on
their outcomes.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class HistoricalRecord:
    """A single historical applicant record with EC text."""
    ec_text: str
    university: str
    program: str
    grade: Optional[float]
    decision: str  # "accepted", "rejected", "waitlisted", "deferred"
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class ECAssessment:
    """Result of EC scoring."""
    score: float          # 0-20 scale
    tier: int             # 1-4
    tier_label: str
    category_breakdown: List[Dict[str, Any]]
    activity_tiers: List[int]


@dataclass
class SimilarCase:
    """A similar historical applicant for the scatter plot."""
    grade: float
    ec_score: float
    ec_tier: int
    outcome: str          # "accepted", "rejected", "waitlisted"
    university: str
    program: str


# ---------------------------------------------------------------------------
# Trivial-text filter
# ---------------------------------------------------------------------------

_TRIVIAL_PATTERNS = re.compile(
    r"^(no|nope|n/a|na|none|nothing|nil|nah|not really|"
    r"didn'?t|did not|i did not|i didn'?t|no i didn'?t|"
    r"no extra|no ec|no extracurriculars?|"
    r"not applicable|not applicable\.?|"
    r"no\.?|nope\.?)$",
    re.IGNORECASE,
)


def _is_trivial(text: str) -> bool:
    """Return True if EC text is empty, too short, or a trivial response."""
    text = text.strip()
    if len(text) < 10:
        return True
    if _TRIVIAL_PATTERNS.match(text):
        return True
    return False


# ---------------------------------------------------------------------------
# Decision normalization
# ---------------------------------------------------------------------------

def _normalize_decision(raw: str) -> str:
    """Normalize decision strings to accepted/rejected/waitlisted/deferred."""
    low = raw.strip().lower()
    if "accept" in low or "admit" in low or "offer" in low:
        return "accepted"
    if "reject" in low or "denied" in low or "refus" in low:
        return "rejected"
    if "waitlist" in low or "wait list" in low or "wait-list" in low:
        return "waitlisted"
    if "defer" in low:
        return "deferred"
    return "rejected"  # default fallback


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def _parse_grade(value: str) -> Optional[float]:
    """Try to parse a grade string into a float."""
    if not value:
        return None
    # Remove common suffixes and non-numeric chars
    cleaned = re.sub(r"[%\s]", "", value)
    try:
        g = float(cleaned)
        if 0 <= g <= 100:
            return g
    except ValueError:
        pass
    return None


def _load_csv_records(csv_dir: str) -> List[HistoricalRecord]:
    """Load historical records from all three CSV files."""
    records: List[HistoricalRecord] = []

    # --- 2022-2023 ---
    path_22 = os.path.join(csv_dir, "2022_2023_Canadian_University_Results.csv")
    if os.path.exists(path_22):
        with open(path_22, encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                for row in reader:
                    if len(row) < 8:
                        continue
                    ec_text = row[5].strip()
                    if _is_trivial(ec_text):
                        continue
                    records.append(HistoricalRecord(
                        ec_text=ec_text,
                        university=row[2].strip(),
                        program=row[1].strip(),
                        grade=_parse_grade(row[3]),
                        decision=_normalize_decision(row[7]),
                    ))

    # --- 2023-2024 ---
    path_23 = os.path.join(csv_dir, "2023_2024_Canadian_University_Results.csv")
    if os.path.exists(path_23):
        with open(path_23, encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                for row in reader:
                    if len(row) < 13:
                        continue
                    ec_text = row[11].strip()
                    if _is_trivial(ec_text):
                        continue
                    # Use acceptance average first, fallback to gr12 final
                    grade = _parse_grade(row[9]) or _parse_grade(row[8])
                    records.append(HistoricalRecord(
                        ec_text=ec_text,
                        university=row[3].strip(),
                        program=row[2].strip(),
                        grade=grade,
                        decision=_normalize_decision(row[4]),
                    ))

    # --- 2024-2025 ---
    path_24 = os.path.join(csv_dir, "2024_2025_Canadian_University_Results.csv")
    if os.path.exists(path_24):
        with open(path_24, encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                for row in reader:
                    if len(row) < 14:
                        continue
                    # Column 13 = "Notable info from supp app"
                    ec_text = row[13].strip()
                    if _is_trivial(ec_text):
                        continue
                    records.append(HistoricalRecord(
                        ec_text=ec_text,
                        university=row[2].strip(),
                        program=row[4].strip(),
                        grade=_parse_grade(row[6]),
                        decision=_normalize_decision(row[5]),
                    ))

    return records


# ---------------------------------------------------------------------------
# ECScorer
# ---------------------------------------------------------------------------

class ECScorer:
    """Scores ECs by semantic similarity against historical admission data."""

    def __init__(self) -> None:
        self._model = None                                # SentenceTransformer
        self._index: List[HistoricalRecord] = []          # Historical records with embeddings
        self._embedding_matrix: Optional[np.ndarray] = None  # (N, 384) matrix for fast cosine sim
        self._category_embeddings: Dict[str, np.ndarray] = {}
        self._category_names: List[str] = []

    # ── Lazy initialization ──────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        """Lazy-load model and build index from CSV data on first call."""
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("all-MiniLM-L6-v2")
        self._build_index()
        self._embed_categories()

    def _build_index(self) -> None:
        """Load CSVs, filter trivial EC entries, embed remaining."""
        csv_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
        csv_dir = os.path.normpath(csv_dir)
        self._index = _load_csv_records(csv_dir)

        if not self._index:
            self._embedding_matrix = np.empty((0, 384))
            return

        # Batch-encode all EC texts
        texts = [r.ec_text for r in self._index]
        embeddings = self._model.encode(texts, show_progress_bar=False, batch_size=64)
        embeddings = np.array(embeddings, dtype=np.float32)

        # Normalize for cosine similarity (dot product on unit vectors = cosine sim)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        embeddings = embeddings / norms

        for i, rec in enumerate(self._index):
            rec.embedding = embeddings[i]

        self._embedding_matrix = embeddings

    def _embed_categories(self) -> None:
        """Embed EC category descriptions for classification."""
        proto_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "mappings", "ec_prototypes.yaml"
        )
        proto_path = os.path.normpath(proto_path)
        try:
            with open(proto_path) as f:
                data = yaml.safe_load(f) or {}
        except FileNotFoundError:
            return

        categories = data.get("categories", {})
        if not categories:
            return

        self._category_names = list(categories.keys())
        descriptions = list(categories.values())
        embs = self._model.encode(descriptions, show_progress_bar=False)
        embs = np.array(embs, dtype=np.float32)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / np.maximum(norms, 1e-8)

        for name, emb in zip(self._category_names, embs):
            self._category_embeddings[name] = emb

    # ── Core scoring ─────────────────────────────────────────────────────

    def score(
        self,
        descriptions: List[str],
        university: str,
        program: str,
        grade: float,
        top_k: int = 20,
    ) -> ECAssessment:
        """
        Score EC descriptions against historical data.

        Args:
            descriptions: List of free-text EC descriptions from the user
            university: Target university
            program: Target program
            grade: Student's top-6 average
            top_k: Number of most similar historical records to consider

        Returns:
            ECAssessment with score, tier, and category breakdown
        """
        self._ensure_loaded()

        if not descriptions or not self._index:
            return self._empty_assessment()

        # Combine all user descriptions into one text for embedding
        combined = " | ".join(d.strip() for d in descriptions if d.strip())
        if not combined:
            return self._empty_assessment()

        user_emb = self._model.encode([combined], show_progress_bar=False)
        user_emb = np.array(user_emb, dtype=np.float32)
        user_emb = user_emb / np.maximum(np.linalg.norm(user_emb), 1e-8)
        user_emb = user_emb.flatten()  # (384,)

        # Cosine similarity against all historical records
        sims = self._embedding_matrix @ user_emb  # (N,)

        # Boost scores for same university/program and similar grade
        uni_low = university.lower()
        prog_low = program.lower()

        boost = np.ones(len(self._index), dtype=np.float32)
        for i, rec in enumerate(self._index):
            # Program match bonus
            if rec.program.lower() == prog_low:
                boost[i] += 0.15
            elif prog_low in rec.program.lower() or rec.program.lower() in prog_low:
                boost[i] += 0.05
            # University match bonus
            if rec.university.lower() == uni_low:
                boost[i] += 0.10
            # Grade proximity bonus (within ±5%)
            if rec.grade is not None and abs(rec.grade - grade) <= 5:
                boost[i] += 0.10

        weighted_sims = sims * boost

        # Get top-k indices
        k = min(top_k, len(self._index))
        top_indices = np.argpartition(weighted_sims, -k)[-k:]
        top_indices = top_indices[np.argsort(weighted_sims[top_indices])[::-1]]

        # Compute acceptance rate among top-k
        matched_records = [self._index[i] for i in top_indices]
        accepted = sum(1 for r in matched_records if r.decision == "accepted")
        total = len(matched_records)
        acceptance_rate = accepted / total if total > 0 else 0.0

        # Score and tier
        ec_score = acceptance_rate * 20.0  # 0-20 scale

        if acceptance_rate >= 0.75:
            tier, tier_label = 1, "Exceptional"
        elif acceptance_rate >= 0.50:
            tier, tier_label = 2, "High"
        elif acceptance_rate >= 0.25:
            tier, tier_label = 3, "Moderate"
        else:
            tier, tier_label = 4, "General"

        # Classify categories from user text
        category_breakdown = self._classify_categories(user_emb)

        # Assign a tier per description (based on individual similarity quality)
        activity_tiers = []
        for desc in descriptions:
            if desc.strip():
                activity_tiers.append(tier)  # simplified: same tier for all
            else:
                activity_tiers.append(4)

        return ECAssessment(
            score=round(ec_score, 1),
            tier=tier,
            tier_label=tier_label,
            category_breakdown=category_breakdown,
            activity_tiers=activity_tiers,
        )

    def find_similar_cases(
        self,
        descriptions: List[str],
        university: str,
        program: str,
        grade: float,
        k: int = 6,
    ) -> List[SimilarCase]:
        """Return the k most similar historical applicants for the scatter plot."""
        self._ensure_loaded()

        if not descriptions or not self._index:
            return []

        combined = " | ".join(d.strip() for d in descriptions if d.strip())
        if not combined:
            return []

        user_emb = self._model.encode([combined], show_progress_bar=False)
        user_emb = np.array(user_emb, dtype=np.float32)
        user_emb = user_emb / np.maximum(np.linalg.norm(user_emb), 1e-8)
        user_emb = user_emb.flatten()

        sims = self._embedding_matrix @ user_emb

        # Filter to records with grades
        valid_mask = np.array([r.grade is not None for r in self._index])
        sims_filtered = np.where(valid_mask, sims, -1.0)

        actual_k = min(k, int(valid_mask.sum()))
        if actual_k == 0:
            return []

        top_indices = np.argpartition(sims_filtered, -actual_k)[-actual_k:]
        top_indices = top_indices[np.argsort(sims_filtered[top_indices])[::-1]]

        cases: List[SimilarCase] = []
        for idx in top_indices:
            rec = self._index[idx]
            sim_score = float(sims[idx])
            # Map cosine similarity to a 0-20 EC score
            ec_approx = max(0.0, sim_score) * 20.0

            outcome = rec.decision
            if outcome == "deferred":
                outcome = "waitlisted"

            # Determine tier from similarity
            if sim_score >= 0.7:
                ec_tier = 1
            elif sim_score >= 0.5:
                ec_tier = 2
            elif sim_score >= 0.3:
                ec_tier = 3
            else:
                ec_tier = 4

            cases.append(SimilarCase(
                grade=rec.grade,
                ec_score=round(ec_approx, 1),
                ec_tier=ec_tier,
                outcome=outcome,
                university=rec.university,
                program=rec.program,
            ))

        return cases

    # ── Helpers ───────────────────────────────────────────────────────────

    def _classify_categories(self, user_emb: np.ndarray) -> List[Dict[str, Any]]:
        """Classify which EC categories the user's text matches."""
        if not self._category_embeddings:
            return []

        result = []
        for name in self._category_names:
            cat_emb = self._category_embeddings[name]
            sim = float(np.dot(user_emb, cat_emb))
            result.append({
                "category": name,
                "active": sim >= 0.35,  # threshold for category match
            })

        return result

    @staticmethod
    def _empty_assessment() -> ECAssessment:
        """Return a default empty assessment."""
        return ECAssessment(
            score=0.0,
            tier=4,
            tier_label="General",
            category_breakdown=[],
            activity_tiers=[],
        )


# ---------------------------------------------------------------------------
# Module-level singleton (shared across requests)
# ---------------------------------------------------------------------------

_scorer: Optional[ECScorer] = None


def get_ec_scorer() -> ECScorer:
    """Get or create the singleton ECScorer instance."""
    global _scorer
    if _scorer is None:
        _scorer = ECScorer()
    return _scorer
