"""
Utilities Package for University Admissions Prediction System
==============================================================

This package provides utility modules for data cleaning, normalization,
and preprocessing tasks essential to the admission prediction pipeline.

================================================================================
                        SYSTEM ARCHITECTURE CONTEXT
================================================================================

    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    WHERE THIS PACKAGE FITS                               │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   Raw CSV Data ──► [THIS PACKAGE] ──► MongoDB ──► Models ──► Predictions │
    │                      src/utils/                                          │
    │                          │                                               │
    │                          ▼                                               │
    │                    ┌─────────────┐                                       │
    │                    │  Utilities: │                                       │
    │                    │ • normalize │                                       │
    │                    │   ├ Unis    │                                       │
    │                    │   └ Progs   │                                       │
    │                    └─────────────┘                                       │
    │                          │                                               │
    │                          ▼                                               │
    │                    Cleaned Data                                          │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

================================================================================
                              MODULES
================================================================================

normalize
    Data normalization utilities for universities and programs.

    Classes:
        - UniversityNormalizer: Normalize university names using fuzzy matching
        - ProgramNormalizer: Extract and normalize program components

================================================================================
                           USAGE EXAMPLE
================================================================================

    >>> from src.utils.normalize import UniversityNormalizer, ProgramNormalizer
    >>>
    >>> # University normalization
    >>> uni_normalizer = UniversityNormalizer('data/mappings/universities.yaml')
    >>> uni_normalizer.normalize("UofT")
    'University of Toronto'
    >>>
    >>> # Program normalization
    >>> prog_normalizer = ProgramNormalizer('data/mappings/base_programs.yaml')
    >>> prog_normalizer.normalize("BSc Honours: Computer Science (Co-op)")
    'Computer Science | BSc Honours | Co-op'

================================================================================
"""

from .normalize import (
    UniversityNormalizer,
    ProgramNormalizer,
    NormalizationStats,
    ProgramComponents,
    create_university_normalizer,
    create_program_normalizer,
)

__all__ = [
    'UniversityNormalizer',
    'ProgramNormalizer',
    'NormalizationStats',
    'ProgramComponents',
    'create_university_normalizer',
    'create_program_normalizer',
]
