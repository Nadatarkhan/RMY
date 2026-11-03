"""Audit EPW files for invalid values and missing data.

    python -m rmy.audit data/RMY/*.epw

Exits with status 1 if any file holds invalid values. See rmy.validation.
"""
import sys

from .validation import main

if __name__ == '__main__':
    sys.exit(main())
