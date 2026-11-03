from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rmy import run_full_rmy_pipeline
from rmy.utils import list_epws, read_epw_file, write_epw, extract_original_header
from rmy.validation import audit_epws, check_epw_values, fill_missing_codes, mask_invalid

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / 'data'


def _frame(**cols):
    n = len(next(iter(cols.values())))
    return pd.DataFrame({'year': [2020] * n, **cols})


def test_flags_fill_values_out_of_range_and_inf():
    df = _frame(ghi=[100.0, 9.96921e36, 9999.0],
                wind_speed=[3.0, 447.0, 999.0],
                relative_humidity=[50.0, np.inf, 999.0])
    rep = check_epw_values(df).set_index(['field', 'problem'])['hours']
    assert rep[('ghi', 'invalid')] == 1 and rep[('ghi', 'missing')] == 1
    assert rep[('wind_speed', 'invalid')] == 1 and rep[('wind_speed', 'missing')] == 1
    assert rep[('relative_humidity', 'invalid')] == 1


def test_mask_then_fill_writes_each_fields_missing_code():
    df = _frame(temp_air=[20.0, 99.9, 1e37], ghi=[5.0, 9.96921e36, 9999.0],
                wind_speed=[2.0, 447.0, 1.0])
    masked = mask_invalid(df)
    assert masked['temp_air'].isna().tolist() == [False, True, True]
    out = fill_missing_codes(masked)
    assert out['temp_air'].tolist() == [20.0, 99.9, 99.9]
    assert out['ghi'].tolist() == [5.0, 9999.0, 9999.0]
    assert out['wind_speed'].tolist() == [2.0, 999.0, 1.0]


def test_audit_traces_phoenix_fill_values_to_source_years():
    rep = audit_epws([DATA / 'RMY' / 'Phoenix_RMY.epw'])
    ghi = rep[(rep.field == 'ghi') & (rep.problem == 'invalid')].iloc[0]
    assert ghi.hours == 4320
    assert ghi.source_years == '2009, 2011, 2014, 2020, 2021, 2023'


def test_write_epw_round_trip_keeps_missing_codes(tmp_path):
    src = list_epws(DATA / 'EPWs')[0]
    out = write_epw(read_epw_file(src), extract_original_header(src), tmp_path / 'x.epw')
    a, b = read_epw_file(src, mask=False), read_epw_file(out, mask=False)
    for col in ('ghi_infrared', 'total_sky_cover', 'temp_air', 'ghi'):
        pd.testing.assert_series_equal(a[col], b[col])
    # The 447 m/s wind speeds in the source become the EPW missing code.
    assert (a.wind_speed == 447).any() and not (b.wind_speed == 447).any()
    assert ((b.wind_speed == 999) == ((a.wind_speed == 447) | (a.wind_speed == 999))).all()


def _copy_inputs_with_fill(dst):
    """Copy the Boston inputs and put a netCDF fill value into every AMY's GHI."""
    for sub in ('EPWs', 'base'):
        (dst / sub).mkdir(parents=True)
        for f in list_epws(DATA / sub):
            lines = f.read_text().splitlines(True)
            if sub == 'EPWs':
                for i in range(8, len(lines)):
                    fields = lines[i].split(',')
                    fields[13] = '9.96921e+36'
                    lines[i] = ','.join(fields)
            (dst / sub / f.name).write_text(''.join(lines))


def test_fill_values_never_reach_the_rmy(tmp_path):
    _copy_inputs_with_fill(tmp_path)
    rmy_path = run_full_rmy_pipeline(tmp_path / 'EPWs', tmp_path / 'base', tmp_path / 'out')
    rmy = read_epw_file(rmy_path, mask=False)
    assert check_epw_values(rmy).query("problem == 'invalid'").empty
    base = read_epw_file(list_epws(tmp_path / 'base')[0], mask=False)
    changed = rmy['ghi'] != base['ghi']
    assert changed.sum() > 1000 and (rmy.loc[changed, 'ghi'] == 9999).all()


def test_invalid_values_error_mode_stops(tmp_path):
    _copy_inputs_with_fill(tmp_path)
    with pytest.raises(ValueError, match='invalid values'):
        run_full_rmy_pipeline(tmp_path / 'EPWs', tmp_path / 'base', tmp_path / 'out',
                              invalid_values='error')
