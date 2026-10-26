#!/usr/bin/env python3
"""Single entry point of the pipeline.

    python run_pipeline.py                      # full run: sample, analyses in parallel, then collect, inference,
                                                # figures and manifest
    python run_pipeline.py --stage rq2          # one stage (sample | rq1 | rq1_rf | rq1b | rq1b_rf | rq2 | rq3 |
                                                #   task2 | task2b | robustness | collect | inference | figures |
                                                #   manifest)
    python run_pipeline.py --stage rq1b_rf,collect,inference,figures,manifest   # several stages, in this order
    python run_pipeline.py --quick              # smoke test on a few windows (minutes instead of hours)
    python run_pipeline.py --sequential         # analyses one after another (machines with few cores or little RAM)

All parameters are in config.yaml; all outputs are written to outputs/ together with manifest.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from leakaware.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
