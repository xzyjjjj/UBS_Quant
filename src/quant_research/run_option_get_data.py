#!/usr/bin/env python3
from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch vanilla option data from Futu OpenD: snapshot, expiries, chain, and quotes."
    )
    p.add_argument("--host", default=os.getenv("FUTU_HOST", "127.0.0.1"), help="OpenD host")
    p.add_argument("--port", type=int, default=int(os.getenv("FUTU_PORT", "11111")), help="OpenD port")
    p.add_argument("--code", default="HK.00700", help="Underlying code (default: HK.00700)")
    p.add_argument(
        "--out-dir",
        default="output/futu",
        help="Output directory (default: output/futu)",
    )
    p.add_argument(
        "--expiry",
        default="",
        help="Filter option chain by expiry date (YYYY-MM-DD). Default: no filter",
    )
    p.add_argument(
        "--right",
        default="ALL",
        choices=("ALL", "CALL", "PUT"),
        help="Filter option chain by CALL/PUT (default: ALL)",
    )
    p.add_argument(
        "--cond",
        default="ALL",
        help="Filter option chain condition when supported (e.g. ALL/ITM/OTM; default: ALL)",
    )
    p.add_argument(
        "--max-contracts",
        type=int,
        default=120,
        help="Max contracts to fetch quotes for (default: 120)",
    )
    p.add_argument(
        "--save-merged",
        action="store_true",
        help="Also save a merged chain+quotes table (default: off)",
    )
    return p.parse_args()


def _require_futu() -> "object":
    try:
        import futu  # type: ignore

        return futu
    except Exception as e:
        raise RuntimeError(
            "Missing dependency: futu-api. Install it with `pip install futu-api` and keep OpenD running."
        ) from e


def _safe_float(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        v = float(x)
        return v if math.isfinite(v) else None
    s = str(x).strip()
    if not s or s.upper() == "N/A":
        return None
    try:
        v = float(s)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def _print_preview(title: str, obj: "object", *, max_rows: int = 5) -> None:
    print(f"\n== {title} ==")
    try:
        shape = getattr(obj, "shape", None)
        cols = getattr(obj, "columns", None)
        print(f"shape={shape}")
        if cols is not None:
            cols_list = list(cols)
            print(f"columns={cols_list[:12]}{' ...' if len(cols_list) > 12 else ''}")
        if hasattr(obj, "head"):
            print(obj.head(max_rows))
        else:
            print(obj)
    except Exception:
        print(obj)


def _save_csv(df: "object", path: Path) -> None:
    try:
        import pandas as pd  # type: ignore

        if isinstance(df, pd.DataFrame):
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            print(f"saved: {path}")
    except Exception:
        return


def _try_call(ctx: Any, method_names: tuple[str, ...], *args: Any, **kwargs: Any) -> tuple[Any, Any] | None:
    last: Exception | None = None
    for name in method_names:
        if not hasattr(ctx, name):
            continue
        fn = getattr(ctx, name)
        call_kwargs = kwargs
        try:
            sig = inspect.signature(fn)
            allowed = set(sig.parameters.keys())
            call_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        except Exception:
            call_kwargs = kwargs
        try:
            return fn(*args, **call_kwargs)
        except TypeError as e:
            last = e
            continue
        except Exception:
            raise
    if last is not None:
        raise last
    return None


def _enum_or_none(obj: Any, enum_name: str, member: str) -> Any:
    try:
        enum = getattr(obj, enum_name)
        return getattr(enum, member)
    except Exception:
        return None


def _fetch_option_expiries(quote_ctx: Any, futu: Any, underlying: str) -> tuple[bool, Any, str | None]:
    out = _try_call(quote_ctx, ("get_option_expiration_date",), underlying)
    if out is None:
        return False, None, "method get_option_expiration_date not found"
    ret, df = out
    if ret == futu.RET_OK:
        return True, df, None
    return False, df, str(df)


def _fetch_option_chain(
    quote_ctx: Any,
    futu: Any,
    underlying: str,
    *,
    start: str | None,
    end: str | None,
    right: str,
    cond: str,
) -> tuple[bool, Any, str | None]:
    option_type = None
    if str(right).upper() in ("CALL", "PUT"):
        option_type = _enum_or_none(futu, "OptionType", str(right).upper())
    if str(right).upper() == "ALL":
        option_type = _enum_or_none(futu, "OptionType", "ALL")

    option_cond_type = _enum_or_none(futu, "OptionCondType", str(cond).upper()) or _enum_or_none(
        futu, "OptionCondType", "ALL"
    )

    kwargs: dict[str, Any] = {
        "start": start,
        "end": end,
        "option_type": option_type,
        "option_cond_type": option_cond_type,
    }

    idx_type = _enum_or_none(futu, "IndexOptionType", "NORMAL")
    if idx_type is not None:
        kwargs["index_option_type"] = idx_type

    out = _try_call(quote_ctx, ("get_option_chain",), underlying, **kwargs)
    if out is None:
        return False, None, "method get_option_chain not found"
    ret, df = out
    if ret == futu.RET_OK:
        return True, df, None
    return False, df, str(df)


def _build_option_universe(chain_df: Any, *, expiry: str, right: str, limit: int) -> tuple[list[str], Any]:
    try:
        import pandas as pd  # type: ignore

        if not isinstance(chain_df, pd.DataFrame) or chain_df.empty:
            return [], chain_df

        df = chain_df.copy()

        if expiry:
            for col in ("strike_time", "expiry_date", "maturity_date"):
                if col in df.columns:
                    df = df[df[col].astype(str) == str(expiry)]
                    break

        if str(right).upper() in ("CALL", "PUT"):
            for col in ("option_type", "type", "call_put", "option_side"):
                if col in df.columns:
                    df = df[df[col].astype(str).str.upper().str.contains(str(right).upper())]
                    break

        code_col = None
        for col in ("code", "option_code", "contract_code", "security_code"):
            if col in df.columns:
                code_col = col
                break
        if code_col is None:
            return [], df

        codes = [str(x) for x in df[code_col].head(int(limit)).tolist()]
        return codes, df
    except Exception:
        return [], chain_df


def _fetch_option_quotes(quote_ctx: Any, futu: Any, option_codes: list[str]) -> tuple[bool, Any, str | None]:
    if not option_codes:
        return False, None, "empty option_codes"
    try:
        out = _try_call(quote_ctx, ("get_option_quote", "request_option_quote"), option_codes)
        if out is not None:
            ret, df = out
            if ret == futu.RET_OK:
                return True, df, None
            return False, df, str(df)
    except Exception:
        pass

    ret, df = quote_ctx.get_market_snapshot(option_codes)
    if ret == futu.RET_OK:
        return True, df, None
    return False, df, str(df)


def _merge_chain_and_quotes(chain_df: Any, quote_df: Any) -> Any:
    try:
        import pandas as pd  # type: ignore

        if not isinstance(chain_df, pd.DataFrame) or chain_df.empty:
            return chain_df
        if not isinstance(quote_df, pd.DataFrame) or quote_df.empty:
            return chain_df

        chain = chain_df.copy()
        quotes = quote_df.copy()

        chain_code_col = "code" if "code" in chain.columns else ("option_code" if "option_code" in chain.columns else None)
        quote_code_col = "code" if "code" in quotes.columns else ("option_code" if "option_code" in quotes.columns else None)
        if chain_code_col is None or quote_code_col is None:
            return chain_df

        dup_cols = [c for c in quotes.columns if c in chain.columns and c != quote_code_col]
        if dup_cols:
            quotes = quotes.drop(columns=dup_cols)

        return chain.merge(quotes, how="left", left_on=chain_code_col, right_on=quote_code_col)
    except Exception:
        return chain_df


@dataclass(frozen=True)
class RunPaths:
    out_dir: Path
    snapshot: Path
    expiries: Path
    chain: Path
    quotes: Path
    merged: Path
    metrics: Path


def _paths(out_dir: Path, code: str, *, expiry: str, right: str) -> RunPaths:
    out_dir = out_dir
    safe_code = code.replace(".", "_")
    suffix = []
    if expiry:
        suffix.append(expiry)
    if right and right.upper() != "ALL":
        suffix.append(right.upper())
    suf = "_".join(suffix) if suffix else "all"
    return RunPaths(
        out_dir=out_dir,
        snapshot=out_dir / "snapshot.csv",
        expiries=out_dir / f"option_expiries_{safe_code}.csv",
        chain=out_dir / f"option_chain_{safe_code}_{suf}.csv",
        quotes=out_dir / f"option_quotes_{safe_code}_{suf}.csv",
        merged=out_dir / f"option_chain_quotes_{safe_code}_{suf}.csv",
        metrics=out_dir / "metrics_options.json",
    )


def main() -> int:
    args = _parse_args()
    futu = _require_futu()

    out_dir = Path(args.out_dir)
    run_paths = _paths(out_dir, str(args.code), expiry=str(args.expiry).strip(), right=str(args.right).strip())

    quote_ctx = None
    metrics: dict[str, Any] = {
        "asof": datetime.now().isoformat(timespec="seconds"),
        "underlying": str(args.code),
        "expiry_filter": str(args.expiry).strip(),
        "right_filter": str(args.right),
        "cond_filter": str(args.cond),
        "max_contracts": int(args.max_contracts),
        "ok": False,
    }

    try:
        quote_ctx = futu.OpenQuoteContext(host=str(args.host), port=int(args.port))

        ret, snap = quote_ctx.get_market_snapshot([str(args.code)])
        if ret == futu.RET_OK:
            _print_preview("market_snapshot", snap)
            _save_csv(snap, run_paths.snapshot)
            metrics["S0"] = _safe_float(snap.loc[0, "last_price"]) if hasattr(snap, "loc") else None
        else:
            print(f"snapshot error: {snap}")

        ok_e, exp_df, exp_err = _fetch_option_expiries(quote_ctx, futu, str(args.code))
        if ok_e:
            _print_preview("option_expiration_date", exp_df, max_rows=12)
            _save_csv(exp_df, run_paths.expiries)
            metrics["expiries_rows"] = int(getattr(exp_df, "shape", [0])[0])
        else:
            print(f"option_expiration_date error: {exp_err}")
            metrics["expiries_error"] = exp_err

        start = str(args.expiry).strip() or None
        end = str(args.expiry).strip() or None
        ok_c, chain_df, chain_err = _fetch_option_chain(
            quote_ctx,
            futu,
            str(args.code),
            start=start,
            end=end,
            right=str(args.right),
            cond=str(args.cond),
        )
        if not ok_c:
            print(f"option_chain error: {chain_err}")
            metrics["chain_error"] = chain_err
        else:
            _print_preview("option_chain", chain_df, max_rows=10)
            _save_csv(chain_df, run_paths.chain)
            metrics["chain_rows"] = int(getattr(chain_df, "shape", [0])[0])

            option_codes, filtered_chain = _build_option_universe(
                chain_df,
                expiry=str(args.expiry).strip(),
                right=str(args.right),
                limit=int(args.max_contracts),
            )
            ok_q, qdf, qerr = _fetch_option_quotes(quote_ctx, futu, option_codes)
            if not ok_q:
                print(f"option_quotes error: {qerr}")
                metrics["quotes_error"] = qerr
            else:
                _print_preview("option_quotes", qdf, max_rows=5)
                _save_csv(qdf, run_paths.quotes)
                metrics["quotes_rows"] = int(getattr(qdf, "shape", [0])[0])

                if bool(args.save_merged):
                    merged = _merge_chain_and_quotes(filtered_chain, qdf)
                    _print_preview("option_chain_quotes", merged, max_rows=10)
                    _save_csv(merged, run_paths.merged)
                    metrics["merged_rows"] = int(getattr(merged, "shape", [0])[0])

        metrics["ok"] = True
        out_dir.mkdir(parents=True, exist_ok=True)
        run_paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"\nsaved: {run_paths.metrics}")
        return 0
    except Exception as e:
        print(f"fatal: {e}")
        return 2
    finally:
        if quote_ctx is not None:
            try:
                quote_ctx.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())

