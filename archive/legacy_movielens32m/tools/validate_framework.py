"""Validate all notebooks with nbformat and syntax-check readable Python/source imports."""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib
import json
import sys
import nbformat

ROOT=Path(__file__).resolve().parents[1]
NEW_NOTEBOOKS=['03_Popularity_Genre.ipynb','04_ItemCF.ipynb','05_Implicit_ALS.ipynb',
               '06_LightGCN.ipynb','07_Two_Tower.ipynb','08_Retrieval_Fusion.ipynb',
               '09_LightGBM_Ranking.ipynb','10_Final_Evaluation.ipynb']


def validate():
    for path in sorted((ROOT/'notebooks').glob('*.ipynb')):
        # Validate read-only; nbformat.read may normalize in memory, never save it back.
        notebook=nbformat.read(path,as_version=4)
        nbformat.validate(notebook)
        new=path.name in NEW_NOTEBOOKS
        for i,cell in enumerate(notebook.cells,1):
            if cell.cell_type=='code':
                source=cell.source
                if new:
                    assert cell.execution_count is None and not cell.outputs,path
                    assert 'SOURCE_BUNDLE_B64' not in source and 'base64' not in source and 'zipfile' not in source
                    ast.parse(source,filename=f'{path.name} cell {i}')
                else:
                    # Preserved Colab notebooks can contain IPython shell/magic syntax.
                    try:ast.parse(source)
                    except SyntaxError:
                        try:
                            from IPython.core.interactiveshell import InteractiveShell
                        except ImportError:
                            # Those legacy cells were not edited; do not relabel them as Python syntax passes.
                            print(f'LEGACY IPython syntax requires IPython: {path.name} cell {i}')
                        else:ast.parse(InteractiveShell.instance().input_transformer_manager.transform_cell(source))
        if new:
            joined='\n'.join(c.source for c in notebook.cells if c.cell_type=='code')
            assert 'RUN_MODE = "smoke_test"' in joined
            assert 'SRC_DIR = PROJECT_DIR / "src"' in joined
        print(f'nbformat OK: {path.name} ({len(notebook.cells)} cells)')
    sys.path.insert(0,str(ROOT/'src'))
    for path in sorted((ROOT/'src'/'recsys').rglob('*.py')):
        ast.parse(path.read_text(),filename=str(path))
        name='.'.join(path.relative_to(ROOT/'src').with_suffix('').parts)
        if name.endswith('.__init__'):name=name[:-9]
        importlib.import_module(name)
    for folder in ['tests','tools']:
        for path in (ROOT/folder).glob('*.py'):ast.parse(path.read_text(),filename=str(path))
    print('All readable module imports and Python syntax checks passed.')


def preserved(hash_path):
    before=json.loads(Path(hash_path).read_text())
    for filename,sha in before.items():
        path = Path(filename) if Path(filename).is_absolute() else ROOT/filename
        if hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            raise AssertionError(f'Preserved notebook changed: {filename}')
    print(f'Preserved notebook byte checks passed: {len(before)} files')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--preserved-hashes')
    args=parser.parse_args()
    validate()
    if args.preserved_hashes:preserved(args.preserved_hashes)
