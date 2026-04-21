import ast
from pathlib import Path
from translate import load_model, neural_translate, hybrid_postprocess


def extract_tests(path):
    src = Path(path).read_text(encoding='utf-8')
    tree = ast.parse(src)
    tests = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == 'tests':
                    for elt in node.value.elts:
                        fr = elt.elts[0].value
                        exp_node = elt.elts[1]
                        expected = None if isinstance(exp_node, ast.Constant) and exp_node.value is None else exp_node.value
                        tests.append((fr, expected, Path(path).name))
    return tests

all_tests = extract_tests('test_hybrid.py') + extract_tests('test_hybrid2.py')

print('Loading 20k model...')
tok20, mdl20, t520 = load_model('model_fr_gloss_20k/final')
print('Loading mega model...')
tokM, mdlM, t5M = load_model('model_fr_gloss_mega/final')

rows = []
for fr, expected, suite in all_tests:
    n20 = neural_translate(fr, tok20, mdl20, t520)
    h20 = hybrid_postprocess(fr, n20)
    nM = neural_translate(fr, tokM, mdlM, t5M)
    hM = hybrid_postprocess(fr, nM)

    ok20 = None if expected is None else (h20.strip().upper() == expected.strip().upper())
    okM = None if expected is None else (hM.strip().upper() == expected.strip().upper())
    rows.append((suite, fr, expected, h20, hM, ok20, okM))

# Summary
from collections import Counter
c = Counter()
for _, _, expected, _, _, ok20, okM in rows:
    if expected is None:
        continue
    if ok20 and okM:
        c['same_ok'] += 1
    elif (not ok20) and (not okM):
        c['same_fail'] += 1
    elif (not ok20) and okM:
        c['improved'] += 1
    elif ok20 and (not okM):
        c['regressed'] += 1

print('\n=== SUMMARY ===')
for k in ['same_ok','same_fail','improved','regressed']:
    print(f'{k}: {c[k]}')

print('\n=== IMPROVED ===')
for suite, fr, expected, h20, hM, ok20, okM in rows:
    if expected is not None and (not ok20) and okM:
        print(f'[{suite}] {fr}')
        print(f'  20k:  {h20}')
        print(f'  mega: {hM}')
        print(f'  exp:  {expected}')

print('\n=== REGRESSED ===')
for suite, fr, expected, h20, hM, ok20, okM in rows:
    if expected is not None and ok20 and (not okM):
        print(f'[{suite}] {fr}')
        print(f'  20k:  {h20}')
        print(f'  mega: {hM}')
        print(f'  exp:  {expected}')

print('\n=== LIMITS CHANGED (expected=None, outputs differ) ===')
for suite, fr, expected, h20, hM, ok20, okM in rows:
    if expected is None and h20.strip().upper() != hM.strip().upper():
        print(f'[{suite}] {fr}')
        print(f'  20k:  {h20}')
        print(f'  mega: {hM}')
