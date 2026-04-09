import sys, json
sys.stdout.reconfigure(encoding='utf-8')

with open('src/models/09_comparison.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    changed = False

    # ── FIX 1: CAGR — arithmetic mean*12 → true geometric CAGR ──────────────
    if 'def compute_stats(' in src:
        old = '    ar   = ret.mean() * 12 * 100\n'
        new = '    ar   = ((1 + ret).cumprod().iloc[-1] ** (12 / len(ret)) - 1) * 100  # true geometric CAGR\n'
        if old in src:
            src = src.replace(old, new)
            print('FIX 1: CAGR formula corrected.')
            changed = True

        # ── FIX 2: Sortino — full-sample downside deviation ──────────────────
        old2 = (
            '    dn   = ret[ret < rf_m]\n'
            '    so   = ((ret.mean() - rf_m) * 12 / (dn.std() * np.sqrt(12))\n'
            '            if len(dn) > 1 else np.nan)\n'
        )
        new2 = (
            '    # Sortino: downside deviation uses ALL periods (not just down months)\n'
            '    # Formula: sqrt(mean(min(r - rf, 0)^2)) * sqrt(12)  [Sortino & van der Meer, 1991]\n'
            '    shortfalls = np.minimum(ret - rf_m, 0)\n'
            '    dd_ann     = np.sqrt((shortfalls**2).mean()) * np.sqrt(12)\n'
            '    so   = ((ret.mean() - rf_m) * 12 / dd_ann if dd_ann > 0 else np.nan)\n'
        )
        if old2 in src:
            src = src.replace(old2, new2)
            print('FIX 2: Sortino formula corrected.')
            changed = True

    # ── FIX 3: Print note — remove false 'arithmetic=CAGR' claim ─────────────
    if "Note: Ann. Return is geometric CAGR." in src:
        old3 = "print('Note: Ann. Return is geometric CAGR. Sortino uses correct downside deviation.')"
        new3 = (
            "print('Note: Ann. Return is true geometric CAGR: (product(1+r_t))^(12/T) - 1.')\n"
            "print('      Sortino uses full-sample downside deviation [Sortino & van der Meer, 1991].')"
        )
        if old3 in src:
            src = src.replace(old3, new3)
            print('FIX 3: Print note corrected.')
            changed = True

    # ── FIX 4: ERC — add convergence check before returning weights ───────────
    if 'def erc_weights(' in src:
        old4 = '    return pd.Series(r.x / r.x.sum(), index=rw.columns)\n'
        new4 = (
            '    if not r.success:\n'
            '        return pd.Series(np.ones(n) / n, index=rw.columns)\n'
            '    return pd.Series(r.x / r.x.sum(), index=rw.columns)\n'
        )
        if old4 in src:
            src = src.replace(old4, new4)
            print('FIX 4: ERC convergence fallback added.')
            changed = True

    if changed:
        cell['source'] = [src]

with open('src/models/09_comparison.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print()
print('All fixes applied. Notebook saved.')
