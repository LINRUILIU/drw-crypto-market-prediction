from __future__ import annotations

import sys

from run_beta7_mlp_signal import main


if __name__ == "__main__":
    if "--config" not in sys.argv:
        sys.argv.extend(["--config", "configs/01_main_sprint_b_mlp_seed_stability.yaml"])
    raise SystemExit(main())
