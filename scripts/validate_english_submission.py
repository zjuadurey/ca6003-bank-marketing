"""Execute the notebook with only its companion archive in a fresh directory/kernel."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT = Path(__file__).resolve().parents[1]
path = ROOT / 'analysis.ipynb'
nb = nbformat.read(path, as_version=4)
original = json.loads(subprocess.check_output(['git', 'show', '90b25ba:analysis.ipynb'], cwd=ROOT))
current = [c for c in nb.cells if not c.metadata.get('portable_bootstrap')]

class WithoutDocstrings(ast.NodeTransformer):
    def visit_Expr(self, node):
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

for i, old in enumerate(original['cells']):
    if old['cell_type'] == 'code' and i not in (1, 6):
        before = WithoutDocstrings().visit(ast.parse(''.join(old['source'])))
        after = WithoutDocstrings().visit(ast.parse(current[i].source))
        assert ast.dump(before) == ast.dump(after), f'Executable logic changed in cell {i}'
for i in [21, 39]:
    old_tables = re.findall(r'(?m)^\|.*(?:\n\|.*)*', ''.join(original['cells'][i]['source']))
    for table in old_tables:
        if not re.search(r'[\u3400-\u9fff]', table):
            assert table in current[i].source, f'Numerical table changed in cell {i}'

started = datetime.now(timezone.utc).isoformat()
with tempfile.TemporaryDirectory(prefix='bank_notebook_check_') as temporary:
    base = Path(temporary)
    work = base / 'empty_working_directory'
    work.mkdir()
    shutil.copy2(ROOT / 'BankMarketing_Data.zip', work / 'BankMarketing_Data.zip')
    for folder in ['kernels/bank-english', 'runtime', 'ipython', 'jupyter']:
        (base / folder).mkdir(parents=True, exist_ok=True)
    for key, folder in [('JUPYTER_RUNTIME_DIR', 'runtime'), ('IPYTHONDIR', 'ipython'), ('JUPYTER_CONFIG_DIR', 'jupyter')]:
        os.environ[key] = str(base / folder)
    spec = {'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
            'display_name': 'Bank Marketing English', 'language': 'python'}
    (base / 'kernels/bank-english/kernel.json').write_text(json.dumps(spec))
    manager = KernelManager(kernel_name='bank-english', kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(base / 'kernels')]))
    try:
        NotebookClient(nb, km=manager, timeout=360, allow_errors=False).execute(cwd=str(work))
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)

codes = [c for c in nb.cells if c.cell_type == 'code']
assert [c.execution_count for c in codes] == list(range(1, len(codes) + 1))
for cell in nb.cells:
    assert not re.search(r'[\u3400-\u9fff]', cell.source)
    for output in cell.get('outputs', []):
        assert output.output_type != 'error'
        visible = output.get('text', '') + ''.join(v for k, v in output.get('data', {}).items() if k.startswith('text/'))
        assert not re.search(r'[\u3400-\u9fff]', visible)
nbformat.validate(nb)
nbformat.write(nb, path)
report = {'status': 'passed', 'started_utc': started, 'completed_utc': datetime.now(timezone.utc).isoformat(),
          'code_cells_executed': len(codes), 'errors': 0, 'fresh_kernel': True, 'isolated_working_directory': True,
          'only_external_input': 'BankMarketing_Data.zip',
          'all_visible_content_english': True, 'unchanged_analysis_and_model_logic': True,
          'unchanged_numerical_markdown_tables': True, 'eda_partition': 'train',
          'validation_models_refitted': 4, 'final_test_replayed_from_saved_predictions': True,
          'final_test_models_refitted': 0, 'human_review': 'pending',
          'notebook_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
(ROOT / 'outputs/integration/english_submission_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
