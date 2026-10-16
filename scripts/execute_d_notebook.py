"""Execute D notebook in a fresh project-local kernel, with no test evaluation."""
from pathlib import Path
import json
import os
import sys
from datetime import datetime, timezone
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]
local=ROOT/'.local'
for folder in ['kernels/ca6003-d','runtime','ipython','jupyter']:
    (local/folder).mkdir(parents=True,exist_ok=True)
os.environ['JUPYTER_RUNTIME_DIR']=str(local/'runtime')
os.environ['IPYTHONDIR']=str(local/'ipython')
os.environ['JUPYTER_CONFIG_DIR']=str(local/'jupyter')
spec={'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'CA6003 D local','language':'python'}
(local/'kernels/ca6003-d/kernel.json').write_text(json.dumps(spec),encoding='utf-8')
manager=KernelManager(kernel_name='ca6003-d',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(local/'kernels')]))
path=ROOT/'notebooks/D_evaluation.ipynb'
notebook=nbformat.read(path,as_version=4)
for cell in notebook.cells:
    if cell.cell_type=='code':
        cell.outputs=[];cell.execution_count=None
started=datetime.now(timezone.utc).isoformat()
try:
    NotebookClient(notebook,km=manager,timeout=240,allow_errors=False).execute(cwd=str(ROOT))
finally:
    if manager.has_kernel:
        manager.shutdown_kernel(now=True)
nbformat.validate(notebook)
cells=[c for c in notebook.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in cells)
assert not any(o.output_type=='error' for c in cells for o in c.outputs)
nbformat.write(notebook,path)
report={'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),'status':'passed',
        'fresh_kernel':True,'code_cells_executed':len(cells),'errors':0,'test_evaluated':False}
(ROOT/'outputs/D/notebook_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
