#!/usr/bin/env python3
"""Convert an HDF5/H5 file to one or more Excel sheets.

Tries pandas (PyTables) first; falls back to h5py for plain datasets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _try_pandas(h5_path: Path, xlsx_path: Path) -> bool:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return False

    try:
        store = pd.HDFStore(h5_path)
    except Exception:
        return False

    with store:
        keys = store.keys()
        if not keys:
            print("No keys found in HDF5 store.")
            return True

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for key in keys:
                try:
                    obj = store.get(key)
                except Exception as exc:
                    print(f"Skip key {key!r}: {exc}")
                    continue

                sheet = key.strip("/") or "data"
                sheet = sheet.replace("/", "_")[:31]

                if hasattr(obj, "to_frame") and not hasattr(obj, "to_excel"):
                    obj = obj.to_frame()

                if hasattr(obj, "to_excel"):
                    obj.to_excel(writer, sheet_name=sheet)
                else:
                    # Fallback: try to wrap array-like into DataFrame
                    try:
                        df = pd.DataFrame(obj)
                        df.to_excel(writer, sheet_name=sheet)
                    except Exception as exc:
                        print(f"Skip key {key!r}: cannot convert to DataFrame ({exc})")
                        continue

    return True


def _try_h5py(h5_path: Path, xlsx_path: Path) -> bool:
    try:
        import h5py  # type: ignore
        import pandas as pd  # type: ignore
    except Exception:
        return False

    datasets = []

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets.append((name, obj))

    with h5py.File(h5_path, "r") as f:
        f.visititems(_visit)
        if not datasets:
            print("No datasets found in HDF5 file.")
            return True

        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            for name, ds in datasets:
                sheet = name.replace("/", "_")[:31] or "data"
                try:
                    data = ds[()]
                    df = pd.DataFrame(data)
                    df.to_excel(writer, sheet_name=sheet)
                except Exception as exc:
                    print(f"Skip dataset {name!r}: {exc}")
                    continue

    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert .h5/.hdf5 to .xlsx")
    parser.add_argument("h5_path", help="Path to .h5/.hdf5 file")
    parser.add_argument(
        "--out",
        dest="xlsx_path",
        help="Output .xlsx path (default: same name)",
        default=None,
    )

    args = parser.parse_args()
    h5_path = Path(args.h5_path).expanduser().resolve()
    if not h5_path.exists():
        print(f"File not found: {h5_path}")
        return 1

    xlsx_path = (
        Path(args.xlsx_path).expanduser().resolve()
        if args.xlsx_path
        else h5_path.with_suffix(".xlsx")
    )

    if _try_pandas(h5_path, xlsx_path):
        print(f"Wrote {xlsx_path}")
        return 0

    if _try_h5py(h5_path, xlsx_path):
        print(f"Wrote {xlsx_path}")
        return 0

    print("Missing dependencies. Please install at least one of these options:")
    print("  - pandas + pytables + openpyxl")
    print("  - h5py + pandas + openpyxl")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
