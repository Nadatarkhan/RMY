"""Section 4.6 and Table 10: effect of recording CAVOK/NSC/SKC/NCD/CLR reports without cloud
groups as clear, and of using only 1-6 h precipitation reports, at the ten international
stations (2022). 'Before' is the method as of commit 7b6d07e, loaded from the git history."""
import io, contextlib, importlib.util, subprocess, tempfile
import numpy as np, pandas as pd, requests
from common import REPO, work_dir, stations
import generate_amys as new

OLD_COMMIT = '7b6d07e'


def load_old():
    src = subprocess.run(['git', '-C', str(REPO), 'show', f'{OLD_COMMIT}:amy_sourcing/generate_amys.py'],
                         check=True, capture_output=True, text=True).stdout
    f = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False); f.write(src); f.close()
    spec = importlib.util.spec_from_file_location('generate_amys_old', f.name); m = importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(m)
    return m


def main():
    work = work_dir(); old = load_old(); sess = requests.Session(); rows = []
    for x in stations('INT'):
        rec = {'city': x['city'], 'id': x['id']}
        for tag, g in (('old', old), ('new', new)):
            with contextlib.redirect_stdout(io.StringIO()):
                raw = g.required_surface(sess, x['id'], 2022, str(work / 'cache'), x['tz'])
            rec[f'{tag}_cover_missing'] = int(raw.TC.isna().sum()); rec[f'{tag}_cover_mean'] = float(raw.TC.mean())
            rec[f'{tag}_precip_mm'] = float(np.nansum(raw.PC)); rec[f'{tag}_precip_hours'] = int(raw.PC.notna().sum())
        csv = pd.read_csv(work / 'cache' / 'noaa_s3' / x['id'] / '2022.csv', dtype=str, keep_default_na=False,
                          usecols=lambda c: c in ('REPORT_TYPE', 'REM'))
        m = csv[csv.REPORT_TYPE.str.strip() == 'FM-15']
        rec['metar_reports'] = len(m)
        rec['cavok_share'] = float(m.REM.str.contains(r'\b(?:CAVOK|NSC|NCD)\b', regex=True).mean()) if len(m) else float('nan')
        rows.append(rec); print(x['city'], flush=True)
    pd.DataFrame(rows).to_csv(work / 'results' / 'clear_sky_reports.csv', index=False)


if __name__ == '__main__':
    main()
