#!/usr/bin/env python3
"""
Generate a TMY-style EPW from a set of AMY EPWs using monthly Finkelstein-Schafer selection.

Inputs:
  --amy-dir      Directory containing AMY EPW files.
  --header-epw   EPW whose first 8 header lines should be used in the output.
  --output       Output EPW path.

The weather data rows are selected month-by-month from the AMY set. The output header is always
copied from --header-epw, so this is suitable when you want a generated TMY/FRMY with the original
site/location metadata.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# Column index constants (0-based, per EPW)
YEAR_COL  = 0
MONTH_COL = 1
DAY_COL   = 2
HOUR_COL  = 3
TEMP_COL  = 6   # Dry bulb temperature (°C)
RH_COL    = 8   # Relative humidity (%)
GHI_COL   = 13  # Global horizontal radiation (Wh/m2)
DNI_COL   = 14  # Direct normal radiation (Wh/m2)
DHI_COL   = 15  # Diffuse horizontal radiation (Wh/m2)
WS_COL    = 21  # Wind speed (m/s)

DEFAULT_SMOOTH_COLS = [TEMP_COL, RH_COL, WS_COL]


def read_epw_header(epw_path: Path) -> List[str]:
    """Return the first 8 EPW header lines from an EPW file."""
    with epw_path.open("r", encoding="utf-8", errors="ignore") as f:
        return [next(f) for _ in range(8)]


def read_epw_data(epw_path: Path) -> pd.DataFrame:
    """Read the EPW hourly data section as a DataFrame."""
    return pd.read_csv(epw_path, skiprows=8, header=None)


def find_epw_files(amy_dir: Path, pattern: str = "*.epw", recursive: bool = False) -> List[Path]:
    """Find EPW files in the AMY directory."""
    iterator = amy_dir.rglob(pattern) if recursive else amy_dir.glob(pattern)
    files = sorted([p for p in iterator if p.is_file() and p.suffix.lower() == ".epw"])
    return files


def _choose_solar_column(df: pd.DataFrame) -> int:
    """Pick the best available solar column in EPW order: GHI -> DNI -> DHI."""
    for c in (GHI_COL, DNI_COL, DHI_COL):
        if c < df.shape[1]:
            col = pd.to_numeric(df.iloc[:, c], errors="coerce")
            if col.notna().any():
                return c
    return GHI_COL


def _ecdf(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Empirical CDF of values evaluated on sorted grid."""
    v = np.sort(values)
    idx = np.searchsorted(v, grid, side="right")
    return idx / max(len(v), 1)


def _fs_distance(candidate_values: np.ndarray, ref_values: np.ndarray) -> float:
    """
    Finkelstein-Schafer distance: sum |F_candidate - F_reference| over the reference grid.
    Works for different sample sizes and avoids trimming.
    """
    candidate_values = np.asarray(candidate_values, dtype=float)
    ref_values = np.asarray(ref_values, dtype=float)
    candidate_values = candidate_values[~np.isnan(candidate_values)]
    ref_values = ref_values[~np.isnan(ref_values)]

    if len(candidate_values) == 0 or len(ref_values) == 0:
        return float("inf")

    grid = np.sort(ref_values)
    f_candidate = _ecdf(candidate_values, grid)
    f_reference = _ecdf(ref_values, grid)
    return float(np.sum(np.abs(f_candidate - f_reference)))


def _daily_mean(df: pd.DataFrame, col: int) -> np.ndarray:
    """Daily mean for a given EPW column index within a single-month DataFrame."""
    s = pd.to_numeric(df.iloc[:, col], errors="coerce")
    return s.groupby(df.iloc[:, DAY_COL]).mean().dropna().values


def _daily_sum(df: pd.DataFrame, col: int) -> np.ndarray:
    """Daily sum for a given EPW column index within a single-month DataFrame."""
    s = pd.to_numeric(df.iloc[:, col], errors="coerce")
    return s.groupby(df.iloc[:, DAY_COL]).sum().dropna().values


def _monthly_mean(df: pd.DataFrame, col: int) -> float:
    """Monthly mean for a given EPW column index."""
    s = pd.to_numeric(df.iloc[:, col], errors="coerce")
    return float(np.nanmean(s))


def _safe_abs_diff(a: float, b: float) -> float:
    if np.isnan(a) or np.isnan(b):
        return float("inf")
    return abs(a - b)


def _smooth_month_boundary(
    final_df: pd.DataFrame,
    end_prev: int,
    start_next: int,
    cols: Iterable[int],
    span: int = 8,
) -> None:
    """
    Blend the last span hours of one month with the first span hours of the next month.
    Also supports the December -> January wrap-around boundary.
    """
    n = len(final_df)
    if span <= 0 or n == 0:
        return

    for c in cols:
        if c >= final_df.shape[1]:
            continue

        final_df.iloc[:, c] = pd.to_numeric(final_df.iloc[:, c], errors="coerce").astype(float)

        a_idx = np.arange(max(end_prev - span + 1, 0), end_prev + 1)
        b_idx = np.arange(start_next, min(start_next + span, n))

        span_eff = min(len(a_idx), len(b_idx))
        if span_eff == 0:
            continue

        a_idx = a_idx[-span_eff:]
        b_idx = b_idx[:span_eff]

        a = pd.to_numeric(final_df.iloc[a_idx, c], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(final_df.iloc[b_idx, c], errors="coerce").to_numpy(dtype=float)

        w = np.linspace(span_eff / (span_eff + 1), 1 / (span_eff + 1), span_eff)

        blended_a = w * a + (1 - w) * b
        blended_b = (1 - w) * a + w * b

        col_vals = pd.to_numeric(final_df.iloc[:, c], errors="coerce").to_numpy(dtype=float)
        col_vals[a_idx] = blended_a
        col_vals[b_idx] = blended_b
        final_df.iloc[:, c] = col_vals


def _rank_rows(rows: List[dict], key: str) -> None:
    """Add rank field for a score key to row dictionaries. Rank 1 is best/lowest."""
    scores = np.array([r[key] for r in rows], dtype=float)
    # stable argsort helps when there are ties
    ranks = np.argsort(np.argsort(scores, kind="mergesort"), kind="mergesort") + 1
    for r, rank in zip(rows, ranks):
        r[f"{key}_rank"] = int(rank)


def generate_tmy_from_amys(
    amy_dir: Path | str,
    output_epw: Path | str,
    header_epw: Path | str,
    sources_csv: Optional[Path | str] = None,
    pattern: str = "*.epw",
    recursive: bool = False,
    smooth_span: int = 8,
    set_year: Optional[int] = None,
    verbose: bool = True,
) -> Dict[int, str]:
    """
    Generate a TMY-style EPW from AMYs.

    Parameters
    ----------
    amy_dir:
        Directory containing AMY EPW files.
    output_epw:
        Path where generated EPW should be written.
    header_epw:
        EPW file whose first 8 header lines should be copied to the generated output.
    sources_csv:
        Optional CSV path logging selected month source files and scores.
    pattern:
        File glob pattern, default "*.epw".
    recursive:
        Whether to search amy_dir recursively.
    smooth_span:
        Number of hours to blend across month boundaries. Set 0 to disable.
    set_year:
        Optional year value to write into EPW data column 0 for all output rows.
    verbose:
        Print progress and selected month sources.

    Returns
    -------
    Dict[int, str]
        Mapping from month number to selected source file name.
    """
    amy_dir = Path(amy_dir)
    output_epw = Path(output_epw)
    header_epw = Path(header_epw)
    sources_csv = Path(sources_csv) if sources_csv is not None else output_epw.with_suffix(".month_sources.csv")

    if not amy_dir.exists():
        raise FileNotFoundError(f"AMY directory does not exist: {amy_dir}")
    if not header_epw.exists():
        raise FileNotFoundError(f"Header EPW does not exist: {header_epw}")

    files = find_epw_files(amy_dir, pattern=pattern, recursive=recursive)
    if verbose:
        print(f"Found {len(files)} EPW files in {amy_dir}")

    if not files:
        raise FileNotFoundError(f"No EPW files found in {amy_dir} using pattern={pattern!r}, recursive={recursive}")

    output_header = read_epw_header(header_epw)
    expected_cols: Optional[int] = None
    valid_months: Dict[int, List[Tuple[pd.DataFrame, str]]] = defaultdict(list)

    for epw_path in files:
        try:
            df = read_epw_data(epw_path)
        except Exception as exc:
            print(f"Skipping {epw_path.name}: could not read EPW data ({exc})")
            continue

        if expected_cols is None:
            expected_cols = df.shape[1]

        if df.shape[1] != expected_cols:
            print(f"Skipping {epw_path.name}: column mismatch ({df.shape[1]} vs {expected_cols})")
            continue

        for month in range(1, 13):
            month_df = df[df.iloc[:, MONTH_COL] == month].copy()
            if len(month_df) in (672, 696, 720, 744):
                valid_months[month].append((month_df, epw_path.name))

    if expected_cols is None:
        raise RuntimeError("No valid EPW files could be read.")

    missing_months = [m for m in range(1, 13) if len(valid_months[m]) == 0]
    if missing_months:
        raise RuntimeError(f"No valid candidate months found for: {missing_months}")

    tmy_blocks: List[pd.DataFrame] = []
    month_sources: Dict[int, str] = {}
    source_rows: List[dict] = []

    for month in range(1, 13):
        candidates = valid_months[month]

        solar_col = _choose_solar_column(candidates[0][0])

        ref_temp = np.concatenate([_daily_mean(df, TEMP_COL) for df, _ in candidates])
        ref_hum = np.concatenate([_daily_mean(df, RH_COL) for df, _ in candidates])
        ref_sol = np.concatenate([_daily_sum(df, solar_col) for df, _ in candidates])
        ref_ws_mean = float(np.nanmean([_monthly_mean(df, WS_COL) for df, _ in candidates]))

        rows: List[dict] = []
        for df_candidate, fname in candidates:
            rows.append({
                "df": df_candidate,
                "fname": fname,
                "fs_temp": _fs_distance(_daily_mean(df_candidate, TEMP_COL), ref_temp),
                "fs_hum": _fs_distance(_daily_mean(df_candidate, RH_COL), ref_hum),
                "fs_sol": _fs_distance(_daily_sum(df_candidate, solar_col), ref_sol),
                "ws_mean": _monthly_mean(df_candidate, WS_COL),
            })

        for score_key in ("fs_temp", "fs_hum", "fs_sol"):
            _rank_rows(rows, score_key)

        for row in rows:
            row["rank_sum"] = row["fs_temp_rank"] + row["fs_hum_rank"] + row["fs_sol_rank"]
            row["ws_abs_diff"] = _safe_abs_diff(row["ws_mean"], ref_ws_mean)

        rows.sort(key=lambda x: (x["rank_sum"], x["ws_abs_diff"], x["fname"]))
        top3 = rows[:3]
        selected = min(top3, key=lambda x: (x["ws_abs_diff"], x["rank_sum"], x["fname"]))

        selected_df = selected["df"].copy()
        if int(selected_df.iloc[0, MONTH_COL]) == 2 and len(selected_df) == 696:
            selected_df = selected_df[selected_df.iloc[:, DAY_COL] != 29].copy()

        tmy_blocks.append(selected_df)
        month_sources[month] = selected["fname"]

        source_rows.append({
            "month": month,
            "selected_file": selected["fname"],
            "n_candidates": len(candidates),
            "solar_column_index": solar_col,
            "fs_temp": selected["fs_temp"],
            "fs_hum": selected["fs_hum"],
            "fs_sol": selected["fs_sol"],
            "fs_temp_rank": selected["fs_temp_rank"],
            "fs_hum_rank": selected["fs_hum_rank"],
            "fs_sol_rank": selected["fs_sol_rank"],
            "rank_sum": selected["rank_sum"],
            "ws_mean": selected["ws_mean"],
            "ref_ws_mean": ref_ws_mean,
            "ws_abs_diff": selected["ws_abs_diff"],
        })

    tmy_blocks = sorted(tmy_blocks, key=lambda x: int(x.iloc[0, MONTH_COL]))
    final_df = pd.concat(tmy_blocks, axis=0).iloc[:, :expected_cols].copy()

    if set_year is not None:
        final_df.iloc[:, YEAR_COL] = int(set_year)

    lengths = [len(block) for block in tmy_blocks]
    starts = np.cumsum([0] + lengths[:-1])
    ends = np.cumsum(lengths) - 1

    solar_col_final = _choose_solar_column(tmy_blocks[0]) if tmy_blocks else GHI_COL
    smooth_cols = [TEMP_COL, RH_COL, solar_col_final, WS_COL]

    if smooth_span > 0:
        final_df.iloc[:, smooth_cols] = (
            final_df.iloc[:, smooth_cols]
            .apply(pd.to_numeric, errors="coerce")
            .astype("float64")
        )

        for i in range(len(tmy_blocks) - 1):
            _smooth_month_boundary(final_df, int(ends[i]), int(starts[i + 1]), smooth_cols, span=smooth_span)
        _smooth_month_boundary(final_df, len(final_df) - 1, 0, smooth_cols, span=smooth_span)

    if final_df.shape[1] != expected_cols:
        raise AssertionError(f"Expected {expected_cols} columns, got {final_df.shape[1]}")
    if final_df.shape[0] != 8760:
        raise AssertionError(f"Expected 8760 rows, got {final_df.shape[0]}")

    output_epw.parent.mkdir(parents=True, exist_ok=True)
    with output_epw.open("w", encoding="utf-8", newline="") as out:
        out.writelines(output_header)
        final_df.astype(str).to_csv(out, index=False, header=False)

    sources_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(source_rows).to_csv(sources_csv, index=False)

    if verbose:
        print("Month sources:")
        for month in range(1, 13):
            print(f"  {month:02d} -> {month_sources[month]}")
        print(f"TMY written to: {output_epw}")
        print(f"Month-source log written to: {sources_csv}")
        print(f"Final shape: {final_df.shape}")

    return month_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a TMY-style EPW from a folder of AMY EPWs, using a separate EPW for the output header."
    )
    parser.add_argument("--amy-dir", required=True, type=Path, help="Directory containing AMY EPW files.")
    parser.add_argument("--header-epw", required=True, type=Path, help="EPW file whose first 8 header lines should be used.")
    parser.add_argument("--output", required=True, type=Path, help="Output generated TMY EPW path.")
    parser.add_argument("--sources-csv", type=Path, default=None, help="Optional month-source CSV output path.")
    parser.add_argument("--pattern", default="*.epw", help="Glob pattern for EPW files. Default: *.epw")
    parser.add_argument("--recursive", action="store_true", help="Search AMY directory recursively.")
    parser.add_argument("--smooth-span", type=int, default=8, help="Hours to blend across month boundaries. Use 0 to disable.")
    parser.add_argument("--set-year", type=int, default=None, help="Optional year value to write into EPW column 0 for all output rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_tmy_from_amys(
        amy_dir=args.amy_dir,
        output_epw=args.output,
        header_epw=args.header_epw,
        sources_csv=args.sources_csv,
        pattern=args.pattern,
        recursive=args.recursive,
        smooth_span=args.smooth_span,
        set_year=args.set_year,
        verbose=True,
    )


if __name__ == "__main__":
    main()
