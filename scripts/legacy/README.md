# Campaign-specific plotting

These implementations use fixed paths for earlier result sets:

| Script | Inputs |
| --- | --- |
| `visualization/analysis_plot2.py` | Combined DFT, TDDFT, MD, CO₂RR, and `results/neb_iddp` analysis |
| `visualization/plot_md_comparison.py` | `results/my_analysis/md/ZnO` |
| `visualization/plot_co2rr_pathway.py` | `results/my_analysis/co2rr_pathway_summary.csv` |

The corresponding paths in `scripts/visualization/` are compatibility wrappers.
The DFT and optical plots in `analysis_plot2.py` include the manuscript styling
changes. See [figure sources](../../docs/figures.md).

These scripts are separate from `run_all.sh`. Historical directory spellings
are retained because saved records refer to them.
