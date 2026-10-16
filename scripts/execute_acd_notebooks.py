"""Fresh local kernel for the C chapter and combined review draft; no real test."""
from pathlib import Path
from datetime import datetime, timezone
import json
import os
import sys
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

ROOT=Path(__file__).resolve().parents[1]
local=ROOT/'.local'
for folder in ['kernels/ca6003-acd','runtime','ipython','jupyter']:
    (local/folder).mkdir(parents=True,exist_ok=True)
os.environ['JUPYTER_RUNTIME_DIR']=str(local/'runtime')
os.environ['IPYTHONDIR']=str(local/'ipython')
os.environ['JUPYTER_CONFIG_DIR']=str(local/'jupyter')
spec={'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'CA6003 ACD review','language':'python'}
(local/'kernels/ca6003-acd/kernel.json').write_text(json.dumps(spec),encoding='utf-8')
records=[]
for rel in ['notebooks/C_preprocessing.ipynb','analysis.ipynb']:
    manager=KernelManager(kernel_name='ca6003-acd',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(local/'kernels')]))
    path=ROOT/rel
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
    codes=[c for c in notebook.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in codes)
    assert not any(o.output_type=='error' for c in codes for o in c.outputs)
    nbformat.write(notebook,path)
    records.append({'path':rel,'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),
                    'code_cells_executed':len(codes),'errors':0,'fresh_kernel':True,'real_test_evaluated':False})
report={'status':'passed','notebooks':records,'B_discovery':'not_received','human_review':'pending'}
(ROOT/'outputs/C/local_notebook_validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps(report,indent=2))
