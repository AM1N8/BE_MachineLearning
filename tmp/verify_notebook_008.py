import ast
import io
import json
import os
import tokenize
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'
root = Path(__file__).resolve().parents[1]
notebook = json.loads((root / 'datamanip/008_optimisation_cv.ipynb').read_text(encoding='utf-8'))
env = {'__name__': '__main__'}
for i, cell in enumerate(notebook['cells']):
    code = ''.join(cell['source'])
    if cell['cell_type'] == 'markdown':
        assert len(code.strip().splitlines()) == 1
        continue
    ast.parse(code)
    assert not any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(code).readline))
    print(f'Cell {i}', flush=True)
    exec(compile(code, f'notebook:cell{i}', 'exec'), env)
    if code.startswith('DATA_DIR ='):
        env['VALEURS_C'] = [3.0, 20.0]
    if code.startswith('def trouver_fichier'):
        env['SORTIE'] = root / 'tmp/verification_008'
        env['SORTIE'].mkdir(exist_ok=True)
    if code.startswith('def ouvrir'):
        train = env['train_df']
        counts = train.util.value_counts()
        selected = counts[counts >= 4].sort_values().head(12).index
        env['train_df'] = train[train.util.isin(selected)].reset_index(drop=True)
        env['test_df'] = env['test_df'].iloc[:3].copy()
        env['exemple'] = env['exemple'].iloc[:3].copy()
        print('Smoke subset:', len(env['train_df']), 'sessions', flush=True)
assert not (env['SORTIE'] / 'submission_008.csv').exists()
assert all(c.get('execution_count') is None for c in notebook['cells'])
print('PASS: cells execute, no comments, no submission CSV, notebook remains unexecuted.')
