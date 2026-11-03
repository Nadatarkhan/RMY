"""RMY: Representative Meteorological Years with embedded extreme events.

Importing the package has no side effects. Typical use:

    from rmy import run_full_rmy_pipeline
    run_full_rmy_pipeline('data/EPWs', 'data/base', 'output')
"""
from .rmy_generation import construct_final_rmy, run_full_rmy_pipeline
from .utils import read_epw_file

__all__ = ['construct_final_rmy', 'run_full_rmy_pipeline', 'read_epw_file']
