"""Execute the Phase 1 notebook on verified real data using the current interpreter, then save outputs."""
from pathlib import Path
import ast
import json
import os
import sys
import uuid
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    os.environ['MPLCONFIGDIR']=str(ROOT/'.venv'/'mplconfig')
    os.environ['JUPYTER_RUNTIME_DIR']=str(ROOT/'.venv'/'jupyter-runtime')
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['IPYTHONDIR']=str(ROOT/'.venv'/'ipython')
    Path(os.environ['IPYTHONDIR']).mkdir(parents=True,exist_ok=True)
    path=ROOT/'notebooks'/'01_MovieLens_Small_EDA.ipynb'
    nb=nbformat.read(path,as_version=4)
    nbformat.validate(nb)
    for index,cell in enumerate(nb.cells,1):
        if cell.cell_type=='code':ast.parse(cell.source,filename=f'EDA cell {index}')
    # Use a KernelManager with an in-memory kernel spec override: no global kernel registration.
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(nb,km=manager,timeout=180,resources={'metadata':{'path':str(ROOT)}})
    try:
        client.execute(cleanup_kc=True)
    except Exception:
        failure=ROOT/'outputs'/'eda'/('execution_failed_'+uuid.uuid4().hex+'.ipynb')
        failure.parent.mkdir(parents=True,exist_ok=True)
        nbformat.write(nb,failure)
        print('Partial execution retained at',failure)
        raise
    errors=[o for c in nb.cells for o in c.get('outputs',[]) if o.output_type=='error']
    pngs=sum('image/png' in o.get('data',{}) for c in nb.cells for o in c.get('outputs',[]))
    expected_figures=sum('plt.subplots(' in c.source for c in nb.cells if c.cell_type=='code')
    if errors or pngs!=expected_figures:
        raise RuntimeError(f'Notebook verification failed: errors={len(errors)}, inline PNG figures={pngs}')
    nbformat.validate(nb)
    nbformat.write(nb,path)
    print(json.dumps(dict(notebook=str(path.relative_to(ROOT)),code_cells=sum(c.cell_type=='code' for c in nb.cells),
                         inline_figures=pngs,errors=len(errors),python=sys.version.split()[0]),indent=2))
