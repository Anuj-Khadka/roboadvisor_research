"""
fix_all.py
Applies the correct CAGR and Sortino formulas to:
  - build_comparison.py  (so future regenerations are clean)
  - notebooks 02–08      (standalone notebooks)
  - notebook 09          (already fixed, re-confirmed)
Run: python fix_all.py
"""
import sys, json, re
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path('.')

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

CAGR_PATTERNS = [
    # pattern                                   replacement
    ('annualized_return     = monthly_returns.mean() * 12 * 100',
     'annualized_return     = ((1 + monthly_returns).cumprod().iloc[-1] ** (12 / len(monthly_returns)) - 1) * 100  # geometric CAGR'),

    ('annualized_return = monthly_returns.mean() * 12 * 100',
     'annualized_return = ((1 + monthly_returns).cumprod().iloc[-1] ** (12 / len(monthly_returns)) - 1) * 100  # geometric CAGR'),

    # generic in compute_stats / compute_performance_stats
    ('    ar   = ret.mean() * 12 * 100',
     '    ar   = ((1 + ret).cumprod().iloc[-1] ** (12 / len(ret)) - 1) * 100  # geometric CAGR'),

    ('    ar  = ret.mean()*12*100',
     '    ar  = ((1 + ret).cumprod().iloc[-1] ** (12 / len(ret)) - 1) * 100  # geometric CAGR'),
]

SORTINO_PATTERN_08 = (
    # notebook 08 uses these variable names
    '    downside_returns      = monthly_returns[monthly_returns < avg_monthly_risk_free]\n'
    '    downside_deviation    = downside_returns.std() * np.sqrt(12) * 100\n'
    '    sortino_ratio         = (\n'
    '        (monthly_returns.mean() - avg_monthly_risk_free) * 12\n'
    '        / (downside_returns.std() * np.sqrt(12))\n'
    '        if len(downside_returns) > 0 else np.nan\n'
    '    )',
    # replacement
    '    # Sortino: downside deviation across ALL periods [Sortino & van der Meer, 1991]\n'
    '    shortfalls_08         = np.minimum(monthly_returns - avg_monthly_risk_free, 0)\n'
    '    downside_deviation    = np.sqrt((shortfalls_08**2).mean()) * np.sqrt(12) * 100\n'
    '    sortino_ratio         = (\n'
    '        (monthly_returns.mean() - avg_monthly_risk_free) * 12\n'
    '        / (np.sqrt((shortfalls_08**2).mean()) * np.sqrt(12))\n'
    '        if (shortfalls_08**2).mean() > 0 else np.nan\n'
    '    )'
)

# ─────────────────────────────────────────────────────────────────────────────
# Fix build_comparison.py
# ─────────────────────────────────────────────────────────────────────────────
bp_path = ROOT / 'build_comparison.py'
bp = bp_path.read_text(encoding='utf-8')

# CAGR in compute_stats inside the script (it's embedded as a string)
old_cagr = '"    ar   = ret.mean() * 12 * 100\\n"'
new_cagr = '"    ar   = ((1 + ret).cumprod().iloc[-1] ** (12 / len(ret)) - 1) * 100  # geometric CAGR\\n"'
if old_cagr in bp:
    bp = bp.replace(old_cagr, new_cagr)
    print('build_comparison.py: CAGR fixed.')
else:
    print('build_comparison.py: CAGR pattern not found — check manually.')

# Sortino in compute_stats inside the script
old_so = (
    '"    dn   = ret[ret < rf_m]\\n"\n'
    '"    so   = ((ret.mean()-rf_m)*12/(dn.std()*np.sqrt(12)) if len(dn)>1 else np.nan)\\n"'
)
new_so = (
    '"    shortfalls = np.minimum(ret - rf_m, 0)\\n"\n'
    '"    dd_ann     = np.sqrt((shortfalls**2).mean()) * np.sqrt(12)\\n"\n'
    '"    so   = ((ret.mean() - rf_m) * 12 / dd_ann if dd_ann > 0 else np.nan)\\n"'
)
if old_so in bp:
    bp = bp.replace(old_so, new_so)
    print('build_comparison.py: Sortino fixed.')
else:
    # Try alternate spacing
    old_so2 = (
        '"    dn   = ret[ret < rf_m]\\n"\n'
        '"    so   = ((ret.mean() - rf_m) * 12 / (dn.std() * np.sqrt(12))\\n"\n'
        '"            if len(dn) > 1 else np.nan)\\n"'
    )
    new_so2 = (
        '"    shortfalls = np.minimum(ret - rf_m, 0)\\n"\n'
        '"    dd_ann     = np.sqrt((shortfalls**2).mean()) * np.sqrt(12)\\n"\n'
        '"    so   = ((ret.mean() - rf_m) * 12 / dd_ann if dd_ann > 0 else np.nan)\\n"'
    )
    if old_so2 in bp:
        bp = bp.replace(old_so2, new_so2)
        print('build_comparison.py: Sortino fixed (alt pattern).')
    else:
        print('build_comparison.py: Sortino pattern not found — will fix via regex.')
        # Regex fallback for the Sortino block
        bp = re.sub(
            r'"    dn   = ret\[ret < rf_m\]\\n"\n"    so   = .*?\\n"',
            ('"    shortfalls = np.minimum(ret - rf_m, 0)\\\\n"\n'
             '"    dd_ann     = np.sqrt((shortfalls**2).mean()) * np.sqrt(12)\\\\n"\n'
             '"    so   = ((ret.mean() - rf_m) * 12 / dd_ann if dd_ann > 0 else np.nan)\\\\n"'),
            bp, flags=re.DOTALL
        )
        print('build_comparison.py: Sortino fixed via regex.')

bp_path.write_text(bp, encoding='utf-8')
print()

# ─────────────────────────────────────────────────────────────────────────────
# Fix notebooks 02–08
# ─────────────────────────────────────────────────────────────────────────────
notebooks = [
    'src/models/02_baseline_charts.ipynb',
    'src/models/03_risk_parity_model.ipynb',
    'src/models/04_markowitz_model.ipynb',
    'src/models/05_black_litterman_model.ipynb',
    'src/models/06_sample_portfolio_model.ipynb',
    'src/models/07_factor_investing_model.ipynb',
    'src/models/08_full_scale_optimization_model.ipynb',
]

for nb_path in notebooks:
    path = ROOT / nb_path
    with open(path, encoding='utf-8') as f:
        nb = json.load(f)

    changed = False
    for cell in nb['cells']:
        if cell['cell_type'] != 'code':
            continue
        src = ''.join(cell['source'])
        orig = src

        # ── Fix CAGR ──────────────────────────────────────────────────────────
        for old, new in CAGR_PATTERNS:
            if old in src:
                src = src.replace(old, new)
                changed = True

        # ── Fix Sortino — notebook 08 only ───────────────────────────────────
        if '08_full_scale' in nb_path:
            old_s, new_s = SORTINO_PATTERN_08
            if old_s in src:
                src = src.replace(old_s, new_s)
                changed = True

        if src != orig:
            cell['source'] = [src]

    if changed:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f'Fixed: {nb_path}')
    else:
        print(f'No changes needed: {nb_path}')

print()
print('Done. All files updated.')
