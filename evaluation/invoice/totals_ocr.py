#!/usr/bin/env python3
"""Compatibility wrapper for the SDK's printed-total OCR module.

The implementation lives in ``sdk/tinydoc/ocr_totals.py`` so the measured
evaluation path and the product path share one parser. Keep this file thin:
adding a second implementation here is exactly how the evaluation and the
shipped SDK drift apart.
"""

from __future__ import annotations

import sys
from pathlib import Path

SDK = Path(__file__).resolve().parent.parent.parent / "sdk"
if str(SDK) not in sys.path:
    sys.path.insert(0, str(SDK))

from tinydoc.ocr_totals import (  # noqa: E402,F401
    AMOUNT,
    CONFUSABLE_ONE,
    CURRENCY_CODE,
    SUBTOTAL,
    SYMBOL,
    TAX,
    TOTAL,
    _amount_after,
    _candidates,
    _is_table_header,
    _symbol,
    _tolerance,
    apply_totals,
    main,
    ocr_lines,
    parse_amount,
    parse_totals_lines,
    read_totals,
    totals_are_consistent,
)

if __name__ == "__main__":
    raise SystemExit(main())
