# --*-- conding:utf-8 --*--
# @time:9/3/25 15:46
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:test.py

from pathlib import Path
import biolib

if __name__ == "__main__":
    app = biolib.load("DTU/NetSurfP-3")
    job = app.cli(args="--help", blocking=False)
    job.wait()

    print("status:", job.get_status())

    outdir = Path("biolib_help_dump")
    outdir.mkdir(exist_ok=True)
    job.save_files(str(outdir))
    print("saved:", job.list_output_files())

    try:
        print(job.get_stdout())
    except Exception:
        pass
