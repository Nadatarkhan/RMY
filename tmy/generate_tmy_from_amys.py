#!/usr/bin/env python3
"""
Generate a TMY-style EPW from a set of AMY EPWs using monthly Finkelstein-Schafer selection.

Inputs:
  --amy-dir      Directory containing AMY EPW files.
  --header-epw   EPW whose first 8 header lines should be used in the output.
  --output       Output EPW path.

The weather data rows are selected month-by-month from the AMY set. The output header is copied
from --header-epw so the generated TMY/FRMY retains the original site/location metadata. By default,
year ranges in COMMENT header lines are updated using years detected from AMY filenames.
"""

from __future__ import annotations

import argparse
import re
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
DEWPOINT_COL = 7 # Dew point temperature (°C)
RH_COL    = 8   # Relative humidity (%)
GHI_COL   = 13  # Global horizontal radiation (Wh/m2)
DNI_COL   = 14  # Direct normal radiation (Wh/m2)
DHI_COL   = 15  # Diffuse horizontal radiation (Wh/m2)
GHI_ILLUM_COL = 16 # Global horizontal illuminance (lux)
DNI_ILLUM_COL = 17 # Direct normal illuminance (lux)
DHI_ILLUM_COL = 18 # Diffuse horizontal illuminance (lux)
ZENITH_LUM_COL = 19 # Zenith luminance (Cd/m2)
WS_COL    = 21  # Wind speed (m/s)

# Columns blended at stitched month boundaries. Wind direction is deliberately omitted
# because it is circular; blending 350° and 10° naively would be wrong.
DEFAULT_SMOOTH_COLS = [
    TEMP_COL, DEWPOINT_COL, RH_COL,
    GHI_COL, DNI_COL, DHI_COL,
    GHI_ILLUM_COL, DNI_ILLUM_COL, DHI_ILLUM_COL, ZENITH_LUM_COL,
    WS_COL,
]


def read_epw_header(epw_path: Path) -> List[str]:
    """Return the first 8 EPW header lines from an EPW file."""
    with epw_path.open("r", encoding="utf-8", errors="ignore") as f:
        return [next(f) for _ in range(8)]


def _extract_years_from_filenames(files: Iterable[Path]) -> List[int]:
    """
    Extract plausible calendar years from EPW file names.

    This intentionally looks only at file names, not EPW data columns, because future
    morphed EPWs often keep a conventional EPW year column while the intended source
    year is encoded in the file name, e.g. famy_2043_ssp126_CanESM5.epw.
    """
    years = []
    for path in files:
        for match in re.findall(r"(?<!\d)((?:19|20|21)\d{2})(?!\d)", path.name):
            year = int(match)
            if 1900 <= year <= 2199:
                years.append(year)
    return sorted(set(years))


def _format_year_range(years: Iterable[int]) -> Optional[str]:
    """Return a compact year range like '2009-2023', or None if no years were detected."""
    years = sorted(set(int(y) for y in years))
    if not years:
        return None
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}-{years[-1]}"


def update_header_year_range(header: List[str], years: Iterable[int], add_generation_note: bool = True) -> List[str]:
    """
    Replace year ranges in the EPW header with the AMY filename-derived year range.

    The function preserves the 8-line EPW header length. It replaces patterns such as
    2009-2023, 2009–2023, 2009 to 2023, and 2009/2023 when they appear in COMMENT
    lines. If no existing range is found, it updates/uses COMMENTS 2 with a concise
    generation note.
    """
    year_range = _format_year_range(years)
    if year_range is None:
        return list(header)

    updated = list(header)
    # Prefer replacing in COMMENT lines so LOCATION/DESIGN CONDITION metadata remain intact.
    comment_indices = [i for i, line in enumerate(updated) if line.upper().startswith("COMMENTS")]
    target_indices = comment_indices if comment_indices else list(range(len(updated)))

    # Common header formats: 2009-2023, 2009–2023, 2009 to 2023, 2009/2023.
    range_patterns = [
        re.compile(r"(?<!\d)((?:19|20|21)\d{2})\s*[-–—]\s*((?:19|20|21)\d{2})(?!\d)"),
        re.compile(r"(?<!\d)((?:19|20|21)\d{2})\s+to\s+((?:19|20|21)\d{2})(?!\d)", re.IGNORECASE),
        re.compile(r"(?<!\d)((?:19|20|21)\d{2})\s*/\s*((?:19|20|21)\d{2})(?!\d)"),
    ]

    replaced = False
    for i in target_indices:
        line = updated[i]
        new_line = line
        for pattern in range_patterns:
            new_line = pattern.sub(year_range, new_line)
        if new_line != line:
            updated[i] = new_line
            replaced = True

    if add_generation_note:
        note = f"COMMENTS 2,Generated TMY from AMY files spanning {year_range}. Original site/location metadata retained from header EPW.\n"
        if not replaced:
            # Use existing COMMENTS 2 if available; otherwise replace the second comment line if possible;
            # otherwise replace the final EPW header line to keep exactly 8 header lines.
            comments2_idx = next((i for i, line in enumerate(updated) if line.upper().startswith("COMMENTS 2")), None)
            if comments2_idx is not None:
                updated[comments2_idx] = note
            elif len(comment_indices) >= 2:
                updated[comment_indices[1]] = note
            else:
                updated[-1] = note
        else:
            # If COMMENTS 2 exists and does not mention the new range, append a compact marker.
            comments2_idx = next((i for i, line in enumerate(updated) if line.upper().startswith("COMMENTS 2")), None)
            if comments2_idx is not None and year_range not in updated[comments2_idx]:
                line = updated[comments2_idx].rstrip("\n")
                updated[comments2_idx] = f"{line}; generated from AMY files spanning {year_range}\n"

    return updated


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


def _available_cols(df: pd.DataFrame, cols: Iterable[int]) -> List[int]:
    """Return only column indices that exist in the DataFrame."""
    return [c for c in cols if c < df.shape[1]]


def _apply_physical_consistency_checks(final_df: pd.DataFrame) -> Dict[str, int]:
    """
    Apply small physical consistency corrections after month-boundary smoothing.

    These mainly prevent smoothing artifacts such as dew point exceeding dry bulb or
    DHI exceeding GHI near stitched month boundaries.
    """
    corrections = {
        "dew_point_gt_dry_bulb_clipped": 0,
        "rh_outside_0_100_clipped": 0,
        "negative_radiation_clipped": 0,
        "dhi_gt_ghi_clipped": 0,
        "negative_wind_speed_clipped": 0,
        "negative_illuminance_clipped": 0,
    }

    # Ensure relevant columns are numeric floats first.
    relevant_cols = _available_cols(
        final_df,
        [TEMP_COL, DEWPOINT_COL, RH_COL, GHI_COL, DNI_COL, DHI_COL,
         GHI_ILLUM_COL, DNI_ILLUM_COL, DHI_ILLUM_COL, ZENITH_LUM_COL, WS_COL],
    )
    for c in relevant_cols:
        final_df.iloc[:, c] = pd.to_numeric(final_df.iloc[:, c], errors="coerce").astype(float)

    if TEMP_COL < final_df.shape[1] and DEWPOINT_COL < final_df.shape[1]:
        dry = final_df.iloc[:, TEMP_COL].to_numpy(dtype=float)
        dew = final_df.iloc[:, DEWPOINT_COL].to_numpy(dtype=float)
        mask = np.isfinite(dry) & np.isfinite(dew) & (dew > dry)
        corrections["dew_point_gt_dry_bulb_clipped"] = int(mask.sum())
        dew[mask] = dry[mask]
        final_df.iloc[:, DEWPOINT_COL] = dew

    if RH_COL < final_df.shape[1]:
        rh = final_df.iloc[:, RH_COL].to_numpy(dtype=float)
        mask = np.isfinite(rh) & ((rh < 0) | (rh > 100))
        corrections["rh_outside_0_100_clipped"] = int(mask.sum())
        final_df.iloc[:, RH_COL] = np.clip(rh, 0, 100)

    for c in _available_cols(final_df, [GHI_COL, DNI_COL, DHI_COL]):
        vals = final_df.iloc[:, c].to_numpy(dtype=float)
        mask = np.isfinite(vals) & (vals < 0)
        corrections["negative_radiation_clipped"] += int(mask.sum())
        vals[mask] = 0.0
        final_df.iloc[:, c] = vals

    if GHI_COL < final_df.shape[1] and DHI_COL < final_df.shape[1]:
        ghi = final_df.iloc[:, GHI_COL].to_numpy(dtype=float)
        dhi = final_df.iloc[:, DHI_COL].to_numpy(dtype=float)
        mask = np.isfinite(ghi) & np.isfinite(dhi) & (dhi > ghi)
        corrections["dhi_gt_ghi_clipped"] = int(mask.sum())
        dhi[mask] = ghi[mask]
        final_df.iloc[:, DHI_COL] = dhi

    if WS_COL < final_df.shape[1]:
        ws = final_df.iloc[:, WS_COL].to_numpy(dtype=float)
        mask = np.isfinite(ws) & (ws < 0)
        corrections["negative_wind_speed_clipped"] = int(mask.sum())
        ws[mask] = 0.0
        final_df.iloc[:, WS_COL] = ws

    for c in _available_cols(final_df, [GHI_ILLUM_COL, DNI_ILLUM_COL, DHI_ILLUM_COL, ZENITH_LUM_COL]):
        vals = final_df.iloc[:, c].to_numpy(dtype=float)
        mask = np.isfinite(vals) & (vals < 0)
        corrections["negative_illuminance_clipped"] += int(mask.sum())
        vals[mask] = 0.0
        final_df.iloc[:, c] = vals

    return corrections


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
    update_header_years: bool = True,
    add_header_generation_note: bool = True,
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
    update_header_years:
        If True, detect years from AMY file names and update year ranges in COMMENT header lines.
    add_header_generation_note:
        If True, add or append a concise COMMENT note about the detected AMY year range.
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

    detected_years = _extract_years_from_filenames(files)
    detected_year_range = _format_year_range(detected_years)

    output_header = read_epw_header(header_epw)
    if update_header_years:
        output_header = update_header_year_range(
            output_header,
            detected_years,
            add_generation_note=add_header_generation_note,
        )

    if verbose and detected_year_range:
        print(f"Detected AMY filename year range: {detected_year_range}")

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

    smooth_cols = _available_cols(final_df, DEFAULT_SMOOTH_COLS)

    if smooth_span > 0 and smooth_cols:
        # Cast one column at a time to avoid pandas dtype-assignment warnings.
        for c in smooth_cols:
            final_df.iloc[:, c] = pd.to_numeric(final_df.iloc[:, c], errors="coerce").astype("float64")

        for i in range(len(tmy_blocks) - 1):
            _smooth_month_boundary(final_df, int(ends[i]), int(starts[i + 1]), smooth_cols, span=smooth_span)
        _smooth_month_boundary(final_df, len(final_df) - 1, 0, smooth_cols, span=smooth_span)

    corrections = _apply_physical_consistency_checks(final_df)

    if final_df.shape[1] != expected_cols:
        raise AssertionError(f"Expected {expected_cols} columns, got {final_df.shape[1]}")
    if final_df.shape[0] != 8760:
        raise AssertionError(f"Expected 8760 rows, got {final_df.shape[0]}")

    output_epw.parent.mkdir(parents=True, exist_ok=True)
    with output_epw.open("w", encoding="utf-8", newline="") as out:
        out.writelines(output_header)
        final_df.astype(str).to_csv(out, index=False, header=False)

    sources_csv.parent.mkdir(parents=True, exist_ok=True)
    source_log = pd.DataFrame(source_rows)
    if detected_year_range:
        source_log["detected_amy_year_range"] = detected_year_range
    for key, value in corrections.items():
        source_log[key] = value
    source_log.to_csv(sources_csv, index=False)

    if verbose:
        print("Month sources:")
        for month in range(1, 13):
            print(f"  {month:02d} -> {month_sources[month]}")
        if detected_year_range and update_header_years:
            print(f"Header year range updated using AMY filenames: {detected_year_range}")
        print("Post-smoothing consistency corrections:")
        for key, value in corrections.items():
            print(f"  {key}: {value}")
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
    parser.add_argument(
        "--no-update-header-years",
        action="store_true",
        help="Disable automatic replacement of year ranges in COMMENT header lines using years detected from AMY filenames.",
    )
    parser.add_argument(
        "--no-header-generation-note",
        action="store_true",
        help="Do not add/append a COMMENT note with the detected AMY year range.",
    )
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
        update_header_years=not args.no_update_header_years,
        add_header_generation_note=not args.no_header_generation_note,
        verbose=True,
    )


if __name__ == "__main__":
    main()
