"""Key the d1/d2/d3 verification ledger by batch arm so different arms verify separately."""
from pathlib import Path

P = Path('paper/scripts/verify_batch.py')
s = P.read_text(encoding='utf-8')

old = (
    "        if iid in combined and combined[iid].get('resolution', {}).get('resolved') is not None:"
    + chr(10)
    + "            print(f'[skip] {iid}: already verified')"
    + chr(10)
    + "            continue")
new = (
    "        ledger_key = batch_dir.name + '/' + iid"
    + chr(10)
    + "        if ledger_key in combined and combined[ledger_key].get('resolution', {}).get('resolved') is not None:"
    + chr(10)
    + "            print(f'[skip] {iid}: already verified ({ledger_key})')"
    + chr(10)
    + "            continue")
assert old in s, 'skip anchor missing'
s = s.replace(old, new, 1)
s = s.replace('combined[iid] = {', 'combined[ledger_key] = {')
P.write_text(s, encoding='utf-8')
compile(s, 'x', 'exec')
print('ledger keyed by arm')
