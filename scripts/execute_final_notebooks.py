"""Execute B and integrated notebook with the current project interpreter."""
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
for folder in ['kernels/ca6003-final','runtime','ipython','jupyter']:
    (local/folder).mkdir(parents=True,exist_ok=True)
os.environ['JUPYTER_RUNTIME_DIR']=str(local/'runtime')
os.environ['IPYTHONDIR']=str(local/'ipython')
os.environ['JUPYTER_CONFIG_DIR']=str(local/'jupyter')
spec={'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'CA6003 final','language':'python'}
(local/'kernels/ca6003-final/kernel.json').write_text(json.dumps(spec))
records=[]
for rel in ['notebooks/B_discovery.ipynb','analysis.ipynb']:
    manager=KernelManager(kernel_name='ca6003-final',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(local/'kernels')]))
    path=ROOT/rel
    notebook=nbformat.read(path,as_version=4)
    started=datetime.now(timezone.utc).isoformat()
    try:
        NotebookClient(notebook,km=manager,timeout=360,allow_errors=False).execute(cwd=str(ROOT))
    finally:
        if manager.has_kernel:manager.shutdown_kernel(now=True)
    nbformat.validate(notebook)
    codes=[c for c in notebook.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in codes)
    assert not any(o.output_type=='error' for c in codes for o in c.outputs)
    nbformat.write(notebook,path)
    records.append({'path':rel,'started_utc':started,'completed_utc':datetime.now(timezone.utc).isoformat(),
                    'code_cells_executed':len(codes),'errors':0,'fresh_kernel':True})
report={'status':'passed','notebooks':records,'B_eda_partition':'train','final_test_replayed_from_saved_predictions':True,'human_review':'pending'}
(ROOT/'outputs/integration/notebook_validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
