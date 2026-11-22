#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Render README.md from README_template.md + reports via cv_metrics.json.

Every number in the README comes from a report file, so the README cannot drift from
the actual runs. Unknown placeholders are left visible as ⟦key⟧.

    python scripts/collect_metrics.py
    python scripts/render_readme.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
TPL = HERE / "README_template.md"
METRICS = Path(r"E:\xz\dsh work place\resume_work\cv_metrics.json")
OUT = HERE / "README.md"


def main() -> int:
    if not TPL.exists():
        print(f"[ERR] missing {TPL}", file=sys.stderr)
        return 1
    metrics = json.loads(METRICS.read_text(encoding="utf-8")) if METRICS.exists() else {}
    text = TPL.read_text(encoding="utf-8")
    rendered = re.sub(
        r"\{\{([a-zA-Z0-9_]+)\}\}",
        lambda mo: str(metrics.get(mo.group(1)) or f"⟦{mo.group(1)}⟧"),
        text,
    )
    OUT.write_text(rendered, encoding="utf-8")
    missing = sorted(set(re.findall(r"⟦([a-zA-Z0-9_]+)⟧", rendered)))
    print(f"[OK] {OUT}")
    if missing:
        print(f"[WARN] missing metrics: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
