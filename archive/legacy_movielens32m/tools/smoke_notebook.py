"""Execute a new notebook's smoke cells in a fresh process, changing only its project path.

Usage: PYTHONPATH=src python tools/smoke_notebook.py notebooks/03_Popularity_Genre.ipynb
Uses the current Python environment; installs nothing, mounts no Drive locally, and trains only synthetic data.
"""
import ast
from pathlib import Path
import sys
import os
for _key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
    os.environ[_key] = '1'
import nbformat

ROOT=Path(__file__).resolve().parents[1]


def main(path):
    notebook=nbformat.read(path,as_version=4)
    namespace={'__name__':'__main__'}
    executed=0
    for index,cell in enumerate(notebook.cells,1):
        if cell.cell_type!='code':continue
        tree=ast.parse(cell.source,filename=f'{Path(path).name} cell {index}')
        if cell.id=='paths-and-mode':
            mode_assignments = [st for st in tree.body if isinstance(st,ast.Assign) and
                                any(isinstance(t,ast.Name) and t.id=='RUN_MODE' for t in st.targets)]
            if len(mode_assignments)!=1 or not isinstance(mode_assignments[0].value,ast.Constant) or mode_assignments[0].value.value!='smoke_test':
                raise ValueError('Refusing to execute a notebook configured for real-data training/evaluation')
            for statement in tree.body:
                if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='PROJECT_DIR' for t in statement.targets):
                    statement.value=ast.Call(func=ast.Name(id='Path',ctx=ast.Load()),args=[ast.Constant(str(ROOT))],keywords=[])
            ast.fix_missing_locations(tree)
        exec(compile(tree,f'{Path(path).name} cell {index}','exec'),namespace)
        executed+=1
    if namespace['RUN_MODE']!='smoke_test':raise ValueError('Smoke validation must not switch to a real-data mode')
    print(f'NOTEBOOK SMOKE OK: {Path(path).name}; {executed} code cells; project path override only')


if __name__=='__main__':
    main(sys.argv[1])
