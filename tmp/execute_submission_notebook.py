"""Exécute les cellules Python du notebook et conserve leurs vraies sorties."""
from pathlib import Path
import contextlib
import json
import platform
import sys
import traceback

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
path = Path(__file__).resolve().parents[1] / 'datamanip/007_soumission_kaggle.ipynb'
notebook = json.loads(path.read_text(encoding='utf-8'))
notebook['metadata']['language_info']['version'] = platform.python_version()
notebook['metadata']['execution_method'] = 'Python cell execution; captured stdout and display outputs'
environment = {'__name__': '__main__'}
count = 0


def save():
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')


for index, cell in enumerate(notebook['cells']):
    if cell['cell_type'] != 'code':
        continue
    count += 1
    cell['execution_count'] = count
    cell['outputs'] = []
    print(f'CELLULE {index} — exécution {count}', flush=True)

    class Capture:
        def write(self, value):
            sys.__stdout__.write(value)
            sys.__stdout__.flush()
            if not value:
                return 0
            outputs = cell['outputs']
            if outputs and outputs[-1].get('output_type') == 'stream':
                outputs[-1]['text'] += value
            else:
                outputs.append(dict(output_type='stream', name='stdout', text=value))
            return len(value)

        def flush(self):
            sys.__stdout__.flush()

    def display(value):
        data = {'text/plain': repr(value)}
        if hasattr(value, '_repr_html_'):
            html = value._repr_html_()
            if html:
                data['text/html'] = html
        cell['outputs'].append(dict(output_type='display_data', metadata={}, data=data))

    try:
        with contextlib.redirect_stdout(Capture()), contextlib.redirect_stderr(Capture()):
            exec(compile(''.join(cell['source']), f'{path.name}:cellule-{index}', 'exec'), environment)
        environment['display'] = display
    except Exception as error:
        cell['outputs'].append(dict(output_type='error', ename=type(error).__name__,
                                    evalue=str(error), traceback=traceback.format_exc().splitlines()))
        save()
        raise
    save()
print('Notebook exécuté entièrement, sorties sauvegardées.', flush=True)
