import multiprocessing as mp
from flask import Flask, request, jsonify, send_from_directory
import solver

app = Flask(__name__, static_folder='static', static_url_path='/static')

def _work(q, out):
    try: out.put(('ok', solver.solve(q)))
    except ValueError as e: out.put(('err', str(e)))
    except Exception: out.put(('err', 'Could not solve that one. Try rewording it or simplifying the expression.'))

def run(q, timeout=20):
    out = mp.Queue(); p = mp.Process(target=_work, args=(q, out)); p.start()
    try: kind, val = out.get(timeout=timeout)
    except Exception:
        p.terminate(); return 'err', 'That took too long to solve. Try a simpler form.'
    p.join(); return kind, val

@app.route('/')
def home(): return send_from_directory('static', 'index.html')

@app.post('/api/solve')
def api_solve():
    q = str((request.get_json(silent=True) or {}).get('q', ''))[:300]
    kind, val = run(q)
    return (jsonify(val), 200) if kind == 'ok' else (jsonify({'error': val}), 400)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
