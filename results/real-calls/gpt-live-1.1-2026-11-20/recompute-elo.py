"""Reproduce the reported ELO fit from numeric scenario-score vectors.

Run from any directory: python recompute-elo.py [--bootstrap]. No credentials needed.
"""

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / 's2s-human-bench' / 'src'))
from s2s_human_bench import rating


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap', action='store_true', help='Also reproduce the 1,000-draw confidence intervals.')
    args = parser.parse_args()
    doc = json.loads((HERE / 'elo.json').read_text())
    method = doc['method']
    ranked = doc['ranked_models']
    references = doc['reference_models']
    scores = doc['scenario_scores']
    for group, codes in doc['groups'].items():
        subset = {code: scores[code] for code in codes}
        result = rating.ratings(subset, ranked, references, method['tie_margin'], method['iterations'])
        expected = doc['group_ratings'][group]
        assert result.keys() == expected.keys(), group
        assert all(abs(result[m] - expected[m]) < 1e-8 for m in result), group
    if args.bootstrap:
        ci = rating.bootstrap_ci(scores, ranked, method['tie_margin'], method['bootstrap_draws'], method['bootstrap_seed'], method['bootstrap_iterations'])
        assert ci == doc['confidence_intervals']
    ranked_order = sorted(ranked, key=lambda m: doc['group_ratings']['all'][m], reverse=True)
    for place, model in enumerate(ranked_order, 1):
        print(f"{place:2} {model:26} {round(doc['group_ratings']['all'][model])}")
    print('All suite/language fits match' + ('; bootstrap intervals match.' if args.bootstrap else '.'))


if __name__ == '__main__':
    main()
