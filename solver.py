
"""Rule-based calculus engine: SymPy computes, this module explains the steps. No AI involved."""
import re
import sympy as sp
from sympy.parsing.sympy_parser import (parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor)
 
x, y, z, t = sp.symbols('x y z t')
L = sp.latex
TR = standard_transformations + (implicit_multiplication_application, convert_xor)
LOC = {'x': x, 'y': y, 'z': z, 't': t, 'e': sp.E, 'pi': sp.pi, 'ln': sp.log, 'inf': sp.oo, 'oo': sp.oo}
ALLOWED = set(LOC) | {'sin','cos','tan','cot','sec','csc','asin','acos','atan','sinh','cosh','tanh',
    'exp','log','sqrt','abs','Abs','YP','YPP'}
 
def parse(s, extra=None):
    s = s.replace('×', '*').replace('÷', '/').replace('π', 'pi').replace('√', 'sqrt').replace('−', '-').strip()
    if '=' in s:
        lhs, rhs = s.split('=', 1)
        if re.fullmatch(r'\s*[fgyh]x?\s*(\([^)]*\))?\s*', lhs): s = rhs.strip()
        else: raise ValueError('Write just the expression, for example 8/sqrt(x), or f(x) = 8/sqrt(x).')
    s = re.sub(r'\b(sqrt|asin|acos|atan|sinh|cosh|tanh|sin|cos|tan|cot|sec|csc|ln|log|exp)\s*(\d*\.?\d*[xyzt])\b', r'\1(\2)', s)
    if not s or len(s) > 200: raise ValueError('Expression is empty or too long.')
    if re.search(r'[^A-Za-z0-9+\-*/^().,\s]', s) or re.search(r'(?<!\d)\.(?!\d)', s):
        raise ValueError('Unsupported characters in: ' + s)
    for w in re.findall(r'[A-Za-z]+', s):
        if w not in ALLOWED: raise ValueError(f'Unknown name "{w}". Use x, sin(x), e^x, ln(x), sqrt(x), pi.')
    loc = dict(LOC); loc.update(extra or {})
    try: return parse_expr(s, local_dict=loc, transformations=TR)
    except Exception: raise ValueError('Could not read the expression: ' + s)
 
def clean(e):
    e = re.sub(r'\bwith respect to\s+\w+|\bwrt\s+\w+', '', e, flags=re.I)
    e = re.sub(r'\s*\bd[xyzt]\b\s*[?.]*$', '', e.strip(), flags=re.I)
    return re.sub(r'[?.]+$', '', e).strip()
 
def pick_var(text, expr):
    m = re.search(r'(?:with respect to|wrt|d/d|∂/∂)\s*([xyzt])', text, re.I)
    if m: return sp.Symbol(m.group(1).lower())
    fs = expr.free_symbols
    return x if x in fs or not fs else sorted(fs, key=str)[0]
 
class Steps:
    def __init__(s): s.list = []
    def add(s, title, math=None, depth=0): s.list.append({'title': title, 'math': math, 'depth': depth})
 
# ---------- derivatives ----------
FT = {sp.sin: lambda u: sp.cos(u), sp.cos: lambda u: -sp.sin(u), sp.tan: lambda u: 1/sp.cos(u)**2,
      sp.exp: lambda u: sp.exp(u), sp.log: lambda u: 1/u, sp.asin: lambda u: 1/sp.sqrt(1-u**2),
      sp.acos: lambda u: -1/sp.sqrt(1-u**2), sp.atan: lambda u: 1/(1+u**2),
      sp.sinh: lambda u: sp.cosh(u), sp.cosh: lambda u: sp.sinh(u), sp.tanh: lambda u: 1/sp.cosh(u)**2}
 
def dsteps(e, v, st, d=0):
    D = lambda s_: f"\\frac{{d}}{{d{L(v)}}}\\left[{s_}\\right]"
    if not e.has(v): st.add('Constant rule', f"{D(L(e))}=0", d); return sp.Integer(0)
    if e == v: st.add('Variable rule', f"{D(L(v))}=1", d); return sp.Integer(1)
    if e.is_Add:
        st.add('Sum rule: differentiate each term', f"{D(L(e))}=" + '+'.join(D(L(a)) for a in e.args), d)
        return sp.Add(*[dsteps(a, v, st, d+1) for a in e.args])
    n, den = sp.fraction(e)
    if e.is_Mul and den != 1 and den.has(v):
        st.add('Quotient rule', f"\\left(\\frac{{f}}{{g}}\\right)'=\\frac{{f'g-fg'}}{{g^2}},\\ f={L(n)},\\ g={L(den)}", d)
        dn = dsteps(n, v, st, d+1); dd = dsteps(den, v, st, d+1)
        return (dn*den - n*dd)/den**2
    if e.is_Mul:
        c, r = e.as_independent(v)
        if c != 1:
            st.add('Constant multiple rule', f"({L(c)}\\cdot f)'={L(c)}\\cdot f',\\ f={L(r)}", d)
            return c*dsteps(r, v, st, d+1)
        a, b = e.args[0], sp.Mul(*e.args[1:])
        st.add('Product rule', f"(fg)'=f'g+fg',\\ f={L(a)},\\ g={L(b)}", d)
        return dsteps(a, v, st, d+1)*b + a*dsteps(b, v, st, d+1)
    if e.is_Pow:
        b, p = e.as_base_exp()
        if not p.has(v):
            st.add('Power rule' + (' with chain rule' if b != v else ''), f"(u^n)'=n\\,u^{{n-1}}u',\\ u={L(b)},\\ n={L(p)}", d)
            return p*b**(p-1)*(dsteps(b, v, st, d+1) if b != v else 1)
        if not b.has(v):
            st.add('Exponential rule', f"(a^u)'=a^u\\ln a\\cdot u',\\ a={L(b)},\\ u={L(p)}", d)
            return e*sp.log(b)*dsteps(p, v, st, d+1)
        st.add('Logarithmic differentiation', f"(f^g)'=f^g\\left(g'\\ln f+\\frac{{g f'}}{{f}}\\right),\\ f={L(b)},\\ g={L(p)}", d)
        return e*(dsteps(p, v, st, d+1)*sp.log(b) + p*dsteps(b, v, st, d+1)/b)
    if e.func in FT and len(e.args) == 1:
        u = e.args[0]; outer = FT[e.func](u)
        st.add(f'Derivative of {e.func.__name__}' + (' with chain rule' if u != v else ''),
               f"{D(L(e))}={L(outer)}" + (f"\\cdot u',\\ u={L(u)}" if u != v else ''), d)
        return outer*(dsteps(u, v, st, d+1) if u != v else 1)
    r = sp.diff(e, v); st.add('Differentiate', f"{D(L(e))}={L(r)}", d); return r
 
def derivative(expr, v, order=1):
    st = Steps(); cur = expr
    for k in range(order):
        if order > 1: st.add(f'Derivative number {k+1}', f"f{'^{('+str(k)+')}' if k else ''}={L(cur)}")
        raw = dsteps(cur, v, st); truth = sp.diff(cur, v)
        cur = sp.simplify(raw) if sp.simplify(raw - truth) == 0 else sp.simplify(truth)
        st.add('Simplify', f"{L(cur)}")
    return {'type': 'Derivative' if order == 1 else f'Derivative of order {order}',
            'steps': st.list, 'answer': L(cur), 'plot': [expr, cur], 'var': v}
 
# ---------- integrals ----------
NAMES = {'ConstantRule': 'Integral of a constant', 'ConstantTimesRule': 'Pull the constant out',
    'PowerRule': 'Power rule', 'AddRule': 'Split the sum and integrate each term', 'URule': 'Substitution',
    'PartsRule': 'Integration by parts', 'ExpRule': 'Exponential rule', 'ReciprocalRule': 'Integral of 1/x',
    'TrigRule': 'Standard trig integral', 'RewriteRule': 'Rewrite the integrand', 'DontKnowRule': 'No elementary rule found'}
 
def rule_steps(r, st, d=0):
    if r is None: return
    n = type(r).__name__; v = getattr(r, 'variable', None); ig = getattr(r, 'integrand', None)
    m = f"\\int {L(ig)}\\,d{L(v)}" if ig is not None else None
    if n == 'URule': m += f",\\quad u={L(r.u_func)}"
    if n == 'PartsRule': m += f",\\quad u={L(r.u)},\\ dv={L(r.dv)}\\,d{L(v)}"
    st.add(NAMES.get(n, n), m, d)
    for k in (getattr(r, '_fields', None) or getattr(r, '__dataclass_fields__', {})):
        val = getattr(r, k)
        for item in (val if isinstance(val, (list, tuple)) else [val]):
            if hasattr(item, 'integrand') and hasattr(item, 'variable'): rule_steps(item, st, d+1)
 
def antiderivative(expr, v, st):
    try:
        from sympy.integrals.manualintegrate import integral_steps
        rule_steps(integral_steps(expr, v), st)
    except Exception:
        st.add('Integrate with the symbolic engine', f"\\int {L(expr)}\\,d{L(v)}")
    F = sp.integrate(expr, v)
    return F
 
def integral(expr, v):
    st = Steps(); F = antiderivative(expr, v, st)
    if F.has(sp.Integral): raise ValueError('No closed-form antiderivative exists for this integrand (or it is beyond this engine).')
    ok = sp.simplify(sp.diff(F, v) - expr) == 0
    st.add('Check: differentiating the result gives the integrand back' if ok else 'Result', f"{L(F)}+C")
    return {'type': 'Indefinite integral', 'steps': st.list, 'answer': L(F) + '+C', 'plot': [expr, F], 'var': v}
 
def definite(expr, v, a, b):
    st = Steps(); F = antiderivative(expr, v, st)
    val = sp.integrate(expr, (v, a, b))
    if not F.has(sp.Integral) and not any(q in (sp.oo, -sp.oo) for q in (a, b)):
        st.add('Fundamental theorem of calculus', f"\\int_{{{L(a)}}}^{{{L(b)}}}f\\,d{L(v)}=F({L(b)})-F({L(a)}),\\ F={L(F)}")
        st.add('Substitute the bounds', f"\\left({L(sp.simplify(F.subs(v,b)))}\\right)-\\left({L(sp.simplify(F.subs(v,a)))}\\right)")
    else: st.add('Evaluate as a limit (improper integral)', f"\\int_{{{L(a)}}}^{{{L(b)}}}{L(expr)}\\,d{L(v)}")
    if val.has(sp.Integral): val = sp.Integral(expr, (v, a, b)).evalf()
    ans = L(sp.simplify(val)) + (f"\\approx {sp.N(val, 8)}" if not val.is_Integer else '')
    return {'type': 'Definite integral', 'steps': st.list, 'answer': ans, 'plot': [expr], 'var': v, 'shade': [a, b]}
 
# ---------- limits ----------
def limit(expr, v, p):
    st = Steps(); st.add('Goal', f"\\lim_{{{L(v)}\\to {L(p)}}} {L(expr)}")
    try: direct = expr.subs(v, p)
    except Exception: direct = sp.nan
    if direct.is_finite: st.add('Direct substitution works (the function is continuous there)', f"{L(sp.simplify(direct))}")
    else:
        st.add('Direct substitution is indeterminate or undefined', f"f({L(p)})={L(direct)}")
        n, den = sp.fraction(sp.together(expr)); cur = (n, den); done = False
        for k in range(3):
            if den == 1: break
            ln, ld = sp.limit(cur[0], v, p), sp.limit(cur[1], v, p)
            if (ln == 0 and ld == 0) or (ln.is_infinite and ld.is_infinite):
                cur = (sp.diff(cur[0], v), sp.diff(cur[1], v))
                st.add("L'Hôpital's rule: differentiate top and bottom", f"\\lim\\frac{{{L(cur[0])}}}{{{L(cur[1])}}}"); done = True
            else: break
        if not done:
            s2 = sp.simplify(expr); st.add('Simplify first, then substitute', f"{L(s2)}")
    ans = sp.limit(expr, v, p)
    if p.is_finite:
        l, r = sp.limit(expr, v, p, '-'), sp.limit(expr, v, p, '+')
        if l != r:
            st.add('One-sided limits disagree', f"\\text{{left}}={L(l)},\\ \\text{{right}}={L(r)}")
            return {'type': 'Limit', 'steps': st.list, 'answer': '\\text{does not exist}', 'plot': [expr], 'var': v, 'center': p}
    st.add('Result', L(ans))
    return {'type': 'Limit', 'steps': st.list, 'answer': L(ans), 'plot': [expr], 'var': v, 'center': p}
 
# ---------- series ----------
def series(expr, v, a, n):
    st = Steps(); st.add('Taylor formula', f"f(x)\\approx\\sum_{{k=0}}^{{{n}}}\\frac{{f^{{(k)}}({L(a)})}}{{k!}}(x-{L(a)})^k")
    rows, cur = [], expr
    for k in range(n+1):
        rows.append(f"f^{{({k})}}(x)={L(sp.simplify(cur))}\\ \\Rightarrow\\ f^{{({k})}}({L(a)})={L(sp.simplify(cur.subs(v, a)))}")
        cur = sp.diff(cur, v)
    st.add('Derivatives and their values at the centre', "\\begin{aligned}" + "\\\\".join(rows) + "\\end{aligned}")
    poly = sp.series(expr, v, a, n+1).removeO()
    st.add('Assemble the polynomial', L(poly))
    return {'type': f'Taylor series (degree {n})', 'steps': st.list, 'answer': L(poly), 'plot': [expr, poly], 'var': v}
 
# ---------- differential equations ----------
def ode(q):
    yf = sp.Function('y')
    ics = {}
    for m in re.finditer(r"y(')?\s*\(\s*([^)]+?)\s*\)\s*=\s*([-+\w./^*]+)", q):
        pt, val = parse(m.group(2)), parse(m.group(3))
        ics[(yf(x).diff(x).subs(x, pt) if m.group(1) else yf(pt))] = val
    body = re.split(r'\b(?:with|given|where|subject to)\b|[,;]', q, flags=re.I)[0]
    body = re.sub(r'^\s*(solve|find|the|differential equation|ode)\b', '', body, flags=re.I)
    body = re.sub(r"^\s*(solve|the|differential equation|ode|:)\s*", '', body, flags=re.I)
    body = body.replace("y''", ' YPP ').replace("y'", ' YP ').replace('dy/dx', ' YP ').replace('d2y/dx2', ' YPP ')
    if '=' not in body: raise ValueError("Write the equation with an equals sign, e.g. dy/dx = 2y")
    l, r = body.split('=', 1)
    loc = {'y': yf(x), 'YP': yf(x).diff(x), 'YPP': yf(x).diff(x, 2)}
    eq = sp.Eq(parse(l, loc), parse(r, loc))
    st = Steps(); st.add('The equation', L(eq))
    hints = sp.classify_ode(eq, yf(x)); h = hints[0] if hints else 'unknown'
    st.add('Type of equation: ' + h.replace('_', ' '), None)
    gen = sp.dsolve(eq, yf(x)); st.add('General solution (C₁, C₂ are constants)', L(gen))
    sol = gen
    if ics:
        sol = sp.dsolve(eq, yf(x), ics=ics)
        st.add('Use the initial conditions to find the constants', ",\\ ".join(L(sp.Eq(k, v_)) for k, v_ in ics.items()))
        st.add('Particular solution', L(sol))
    try:
        if sp.checkodesol(eq, sol)[0]: st.add('Check: substituting back satisfies the equation', None)
    except Exception: pass
    rhs = sol.rhs if not isinstance(sol, (list, tuple)) else sol[0].rhs
    rhs = rhs.subs({s: 1 for s in rhs.free_symbols if str(s).startswith('C')})
    return {'type': 'Differential equation', 'steps': st.list, 'answer': L(sol if not isinstance(sol, list) else sol[0]), 'plot': [rhs], 'var': x}
 
# ---------- dispatcher ----------
def curve(expr, v, lo, hi, n=300):
    f = sp.lambdify(v, expr, 'math'); xs, ys = [], []
    for i in range(n+1):
        xv = lo + (hi-lo)*i/n
        try: yv = float(f(xv))
        except Exception: yv = None
        xs.append(xv); ys.append(yv if yv is not None and abs(yv) < 1e9 and yv == yv else None)
    return {'x': xs, 'y': ys, 'label': L(expr)}
 
def solve(q):
    q = q.strip().replace('∫', ' integral ').replace('→', '->')
    if not q: raise ValueError('Type a question first.')
    ql = q.lower(); res = None
    if re.search(r"\bdy/dx\b|y'|differential equation|\bode\b", ql): res = ode(q)
    elif re.search(r'\blim\b|\blimit\b|->', ql):
        m = re.search(r'(?:limit(?: of)?|lim)\s*(.+?)\s*(?:as|when)\s*([a-z])\s*(?:->|approaches|tends to|goes to)\s*(\S+)', q, re.I)
        if m: f, vv, p = m.group(1), m.group(2), m.group(3)
        else:
            m = re.search(r'lim\w*\s*([a-z])\s*->\s*(\S+)\s+(.+)', q, re.I)
            if not m: raise ValueError('Write limits like: limit of sin(x)/x as x approaches 0')
            vv, p, f = m.groups()
        v = sp.Symbol(vv.lower()); res = limit(parse(clean(f)), v, parse(p)); 
    elif re.search(r'taylor|maclaurin|series', ql):
        n = 5; a = sp.Integer(0)
        m = re.search(r'(?:up to|to|degree|order)\s*(?:x\^)?(\d+)', ql)
        if m: n = min(int(m.group(1)), 10); q = q[:m.start()] + q[m.end():]
        m = re.search(r'(?:about|around|at)\s*(?:[a-z]\s*=\s*)?(-?[\w.]+)', q, re.I)
        if m: a = parse(m.group(1)); q = q[:m.start()] + q[m.end():]
        m = re.search(r'(?:taylor|maclaurin)(?: series| polynomial| expansion)?(?: of)?\s*(.+)', q, re.I) or re.search(r'series(?: of)?\s*(.+)', q, re.I)
        if not m: raise ValueError('Write it like: Maclaurin series of e^x up to 4')
        e = parse(clean(m.group(1))); res = series(e, pick_var(q, e), a, n)
    elif re.search(r'integral|integrate|antiderivative', ql):
        m = re.search(r'(?:integral|integrate)(?: of)?\s*(.+?)\s+from\s+(\S+)\s+to\s+(\S+)', q, re.I)
        if m:
            e = parse(clean(m.group(1))); res = definite(e, pick_var(q, e), parse(m.group(2)), parse(m.group(3).rstrip('.?')))
        else:
            m = re.search(r'(?:integral|integrate|antiderivative)(?: of)?\s*(.+)', q, re.I)
            e = parse(clean(m.group(1))); res = integral(e, pick_var(q, e))
    elif re.search(r'deriv|differentiate|d/d[xyzt]|∂|partial', ql):
        order = 1
        for w, k in (('second', 2), ('third', 3), ('fourth', 4)):
            if w in ql: order = k
        m = re.search(r'(\d+)(?:st|nd|rd|th)\s+deriv', ql)
        if m: order = min(int(m.group(1)), 5)
        m = re.search(r'(?:derivative|differentiate|d/d[xyzt]|∂/∂[xyzt])(?: of)?\s*(.+)', q, re.I)
        if not m: raise ValueError('Write it like: derivative of x^3*sin(x)')
        e = parse(clean(m.group(1))); res = derivative(e, pick_var(q, e), order)
    else: raise ValueError('Start with what to do, e.g. "derivative of …", "integrate … from 0 to 3", "limit of … as x approaches 0", "Maclaurin series of …", or "dy/dx = …".')
    v = res['var']; lo, hi = -6, 6
    if 'shade' in res:
        fin = [float(sp.N(q_)) for q_ in res['shade'] if q_.is_finite]
        if len(fin) == 2: lo, hi = min(fin) - 2, max(fin) + 2
        elif len(fin) == 1: lo, hi = fin[0] - 2, fin[0] + 8
    if res.get('center') is not None and res['center'].is_finite: lo, hi = float(res['center']) - 6, float(res['center']) + 6
    out = {'type': res['type'], 'steps': res['steps'], 'answer': res['answer'], 'plot': []}
    for e in res['plot']:
        try:
            if sp.sympify(e).free_symbols <= {v}: out['plot'].append(curve(sp.sympify(e), v, lo, hi))
        except Exception: pass
    if 'shade' in res and all(s.is_finite for s in res['shade']): out['shade'] = [float(sp.N(s)) for s in res['shade']]
    return out