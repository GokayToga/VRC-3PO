"""Fig. 1 for the first revision (benchmark v1.1 counts, repeated-CV endpoint).

Reuses the layout code of ``generate_ieee_figures.figure_study_design`` and
substitutes only the box text, so the geometry stays identical.

    python -m analysis.figure_study_design_r1 --output-dir figures
"""
import argparse
import inspect
from pathlib import Path

from analysis import generate_ieee_figures as g

REPLACE = {
    '"25 people; 2,983 windows': '"25 people; 2,980 windows',
    '"22 people; 1,197 windows': '"22 people; 1,159 windows',
    '"6,442 windows; 84 identifiers"': '"6,401 windows (v1.1)\\n84 identifiers"',
    '"9 held-out people\\n768 windows; AUC 0.757\\nCluster CI 0.696-0.814"':
        '"47 IDs; 5x5 repeated CV\\nAUC 0.634; env.-only 0.677\\nWithin environment 0.487"',
    'output_dir, "fig1_study_design"': 'output_dir, "fig1_study_design_r1"',
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", type=Path, required=True)
    a = p.parse_args()
    src = inspect.getsource(g.figure_study_design)
    for old, new in REPLACE.items():
        assert src.count(old) == 1, old
        src = src.replace(old, new)
    ns = dict(vars(g))
    exec(src, ns)
    g.set_style()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    ns["figure_study_design"](a.output_dir)

if __name__ == "__main__":
    main()
