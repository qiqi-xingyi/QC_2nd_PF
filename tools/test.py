# --*-- conding:utf-8 --*--
# @time:9/3/25 15:46
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:test.py

"""
Probe a BioLib app's CLI outputs and save logs locally.

- Tries both "DTU/NetSurfP-3.0" and "DTU/NetSurfP-3"
- For each app id, runs CLI with:
    1) --help
    2) -h
    3) (no args)
- Prints stdout/stderr to terminal and saves all job files under biolib_probe/<app_id>/<mode>/

Run this file directly in your IDE. No arguments needed.
"""

from __future__ import annotations
from pathlib import Path
import biolib
import traceback

APP_IDS = ["DTU/NetSurfP-3.0", "DTU/NetSurfP-3"]  # try both variants
MODES = [
    ("help", "--help"),
    ("h", "-h"),
    ("noargs", None),
]

OUT_ROOT = Path(__file__).resolve().parent / "biolib_probe"

def run_and_dump(app_id: str, mode_name: str, args: str | None) -> None:
    print(f"\n=== [{app_id}] mode={mode_name} args={args!r} ===")
    try:
        app = biolib.load(app_id)
        if args is None:
            job = app.cli()
        else:
            job = app.cli(args=args)

        # Be safe: wait until the job is finished (usually quick for help/usage)
        try:
            job.wait()
        except Exception:
            # some implementations may already be blocking; ignore
            pass

        status = None
        try:
            status = job.get_status()
            print(f"[status] {status}")
        except Exception as e:
            print(f"[status] unavailable: {e}")

        # Try to print stdout/stderr to terminal
        try:
            out = job.get_stdout()
            if out:
                print("----- STDOUT -----")
                print(out)
        except Exception as e:
            print(f"[stdout] unavailable: {e}")

        try:
            err = job.get_stderr()
            if err:
                print("----- STDERR -----")
                print(err)
        except Exception as e:
            print(f"[stderr] unavailable: {e}")

        # Save all files produced by the job for offline inspection
        out_dir = OUT_ROOT / app_id.replace("/", "_") / mode_name
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            job.save_files(str(out_dir))
            print(f"[saved] files saved to: {out_dir}")
            try:
                print("[files]", job.list_output_files())
            except Exception:
                pass
        except Exception as e:
            print(f"[saved] failed: {e}")

    except Exception as e:
        print(f"[ERROR] app.cli failed for {app_id} mode={mode_name}: {e}")
        traceback.print_exc()

def main():
    print(f"[probe] output root: {OUT_ROOT}")
    for app_id in APP_IDS:
        for mode_name, args in MODES:
            run_and_dump(app_id, mode_name, args)

if __name__ == "__main__":
    main()
