"""Fail on missing/mismatched locked versions; record actual runtime, not intended versions."""
from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
expected = {}
for line in (ROOT/'requirements.D.lock.txt').read_text().splitlines():
    if line.strip() and not line.startswith('#'):
        name, version = line.strip().split('==')
        expected[name] = version
if sys.platform != 'win32':
    expected.update(pexpect='4.9.0', ptyprocess='0.7.0')
actual = {}
issues = []
for name, wanted in expected.items():
    try:
        actual[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        actual[name] = None
    if actual[name] != wanted:
        issues.append(f'{name}: expected {wanted}, got {actual[name]}')
if sys.version_info[:2] != (3,12):
    issues.append('CPython 3.12.x required')
report = {'checked_utc':datetime.now(timezone.utc).isoformat(), 'python':platform.python_version(),
          'platform':platform.platform(), 'entry_point':'requirements.current.txt',
          'packages':actual,'passed':not issues,'issues':issues}
(ROOT/'outputs/integration').mkdir(parents=True,exist_ok=True)
(ROOT/'outputs/integration/environment.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
sys.exit(1 if issues else 0)
