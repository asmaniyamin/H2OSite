# Copyright 2026 Asmani Yamin
# SPDX-License-Identifier: LicenseRef-H2OSite-Noncommercial-Research-1.0

"""Run the 5hxb_production example with the shared CLI."""

from pathlib import Path
import sys

from H2OSite.cli import main


if __name__ == "__main__":
    main(args=["run", "--config", str(Path(__file__).parent / "configs" / "5hxb_production.yaml"),
               *sys.argv[1:]])
