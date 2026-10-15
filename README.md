# MBE-8705 public reproducibility deposit

This package contains only the research code and numerical evidence for the article “Thresholds and Control in a Multiscale Epidemic Model.” It does not include referee correspondence or personal data.

The model analysis is `code/revised_analysis.py`; the additional minor-revision comparison is `code/minor_revision_analysis.py`. The notebook launcher runs from this directory. The `data/` folder includes all archived trajectories, fitted data, sampling vectors, baseline cost decomposition, comparison results, perturbation draws, policies and optimizer logs; the six figures are in `figures/`. Version information, seeds and method settings appear in `data/run_manifest.json` and `data/minor_revision_manifest.json`. The requirements are pinned in `requirements.txt`.

Run `python code/minor_revision_analysis.py` from the unzipped root to recompute the weight-zero optimization and paired maximal-control comparisons. It verifies the original 108 heuristic heatmap results before replacing the two-comparator figures. This can take several minutes. To rerun the original model experiments from scratch, run `python code/revised_analysis.py` first, then the minor-revision script. Numerical optimization is finite and does not prove global optimality.

No public DOI is assigned to this local ZIP. The authors must publish it through their public repository account and place that record’s DOI in the article before publication.
