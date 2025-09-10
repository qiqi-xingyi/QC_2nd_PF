# --*-- coding:utf-8 --*--
# @time: 8/28/25 23:38
# @Author : Yuqi Zhang
# @Email  : yzhan135@kent.edu
# @File   : online_model_client.py

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import shutil
import time
import json
import traceback
import shlex  # for safe shell quoting

# Third-party: pybiolib provides `import biolib`
try:
    import biolib
    from biolib._internal.http_client import HttpError  # for error pattern detection
except ImportError as e:
    raise ImportError(
        "pybiolib is required. Install with: pip install -U pybiolib"
    ) from e


# ---------- Data structures passed between steps ----------

@dataclass
class RawArtifact:
    """A single model run output for one input FASTA."""
    seq_id: str
    input_fasta: Path
    output_dir: Path            # local copy of BioLib job outputs
    main_table: Optional[Path]  # TSV/CSV path if detected
    meta: Dict

@dataclass
class RawArtifactIndex:
    """Collection of raw outputs for downstream post-processing."""
    items: List[RawArtifact]
    meta: Dict                  # run-wide metadata (app id, version, timings, failures)


# ---------- Client for NetSurfP-3 over BioLib ----------

class OnlineModelClient:
    """
    Client for submitting FASTA files to NetSurfP-3 via BioLib,
    saving the raw outputs locally, and returning an index.

    IMPORTANT:
      - Use AppID 'DTU/NetSurfP-3' (NOT 'DTU/NSP3').
      - The app expects CLI args: -i <input_fasta>  -o <output_dir>.
        Pass them as a SINGLE CLI STRING via `app.cli(args="...")`.
    """

    def __init__(
        self,
        app_id: str = "DTU/NetSurfP-3",
        timeout_s: int = 3600,         # kept for API symmetry; cli() blocks until done
        retries: int = 2,              # total attempts = retries + 1
        rate_limit_s: float = 2.0,     # gentle between jobs
        overwrite: bool = False,
        verbose: bool = True,
        min_len: int = 10,             # NetSurfP online typically expects len >= 10
        long_backoff_s: int = 60,      # initial backoff for compute_limit_exceeded
    ):
        self.app_id = app_id
        self.timeout_s = timeout_s
        self.retries = retries
        self.rate_limit_s = rate_limit_s
        self.overwrite = overwrite
        self.verbose = verbose
        self.min_len = min_len
        self.long_backoff_s = long_backoff_s

        # Preload the BioLib application once
        self._app = biolib.load(self.app_id)

    # -------------------------- Public API ---------------------------------

    def predict(self, dataset, out_dir: Path) -> RawArtifactIndex:
        """
        Submit each FASTA in `dataset.records[*].fasta_path` to NetSurfP-3 and
        persist raw outputs under:
            <out_dir>/<seq_id>/

        Returns a RawArtifactIndex listing all successful outputs.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        items: List[RawArtifact] = []
        failures: List[Dict] = []
        t0 = time.time()

        total = len(dataset.records)

        for i, rec in enumerate(dataset.records):
            seq_id = rec.seq_id
            fasta_path = Path(rec.fasta_path)

            print(f"[{i+1}/{total}] Running NetSurfP-3 for: {seq_id}", flush=True)

            # Per-record output folder (local)
            rec_out = out_dir / self._safe(seq_id)
            if rec_out.exists() and self.overwrite:
                shutil.rmtree(rec_out, ignore_errors=True)
            rec_out.mkdir(parents=True, exist_ok=True)

            start_t = time.time()
            ok = False
            last_err = None

            for attempt in range(self.retries + 1):
                try:
                    # Validate FASTA early to avoid opaque remote failures
                    self._validate_fasta(fasta_path, self.min_len)

                    # Build CLI string exactly as the app expects: -i <file> -o out
                    args = f"-i {shlex.quote(str(fasta_path))} -o out"

                    print(f"[{seq_id}] submitting job (attempt {attempt+1}) ...", flush=True)
                    job = self._app.cli(args=args)  # blocking call
                    status = str(job.get_status()).upper()
                    print(f"[{seq_id}] job returned with status={status}", flush=True)

                    # Always save remote artifacts (incl. stdout/stderr) locally
                    csv_path, json_path = self._save_job_outputs(job, rec_out)

                    # Mirror a short preview of stdout to console
                    try:
                        out_txt = job.get_stdout()
                        if out_txt:
                            preview = out_txt[:200]
                            if isinstance(preview, bytes):
                                preview = preview.decode(errors="ignore")
                            print(f"[BioLib][{seq_id}] stdout[0:200]: {preview}", flush=True)
                    except Exception:
                        pass

                    # --- SUCCESS STATES ---
                    SUCCESS = {"SUCCEEDED", "COMPLETED", "COMPLETED_WITH_WARNINGS"}
                    if status not in SUCCESS:
                        raise RuntimeError(f"BioLib job status={status}")

                    # Prefer CSV as main table if we grabbed it; else try to locate any table
                    main_table = csv_path or self._find_main_table(rec_out)

                    items.append(RawArtifact(
                        seq_id=seq_id,
                        input_fasta=fasta_path,
                        output_dir=rec_out,
                        main_table=main_table,
                        meta={
                            "attempt": attempt,
                            "app_id": self.app_id,
                            "saved_dir": str(rec_out),
                            "status": status,
                            "csv": (str(csv_path) if csv_path else None),
                            "json": (str(json_path) if json_path else None),
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                    ))

                    elapsed = time.time() - start_t
                    print(f"[DONE] {seq_id}  status={status}  saved={rec_out}  elapsed={elapsed:.1f}s",
                          flush=True)

                    ok = True
                    break

                except Exception as e:
                    last_err = e
                    print(f"  -> attempt {attempt+1} failed: {e}", flush=True)
                    traceback.print_exc()

                    # Handle compute_limit_exceeded with longer backoff
                    sleep_s = min(5.0 * (attempt + 1), 15.0)  # default short backoff
                    if isinstance(e, HttpError) and "compute_limit_exceeded" in str(e):
                        sleep_s = self.long_backoff_s * (attempt + 1)
                        print(f"  -> server under high load; backing off for {sleep_s}s", flush=True)

                    time.sleep(sleep_s)

            if not ok:
                failures.append({
                    "seq_id": seq_id,
                    "input_fasta": str(fasta_path),
                    "error": repr(last_err),
                    "saved_dir": str(rec_out),
                })
                elapsed = time.time() - start_t
                print(f"[FAIL] {seq_id}  saved={rec_out}  elapsed={elapsed:.1f}s  error={last_err}",
                      flush=True)

            # polite rate limit between jobs if requested
            if self.rate_limit_s > 0:
                time.sleep(self.rate_limit_s)

        meta = {
            "app_id": self.app_id,
            "timeout_s": self.timeout_s,
            "retries": self.retries,
            "rate_limit_s": self.rate_limit_s,
            "overwrite": self.overwrite,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_s": round(time.time() - t0, 3),
            "success": len(items),
            "failures": failures,
        }

        # Write a small run index for reproducibility
        (out_dir / "raw_index.json").write_text(
            json.dumps({
                "meta": meta,
                "items": [
                    {
                        **asdict(it),
                        "input_fasta": str(it.input_fasta),
                        "output_dir": str(it.output_dir),
                        "main_table": (str(it.main_table) if it.main_table else None),
                    } for it in items
                ]
            }, ensure_ascii=False, indent=2)
        )

        print(f"[OK] NetSurfP-3 runs completed: {len(items)} success, {len(failures)} failed", flush=True)
        if failures:
            print("     See failures in raw_index.json", flush=True)

        return RawArtifactIndex(items=items, meta=meta)

    # -------------------------- Internals ----------------------------------

    @staticmethod
    def _safe(text: str) -> str:
        """Make a string safe for directory names."""
        out = []
        for ch in text:
            if ch.isalnum() or ch in ("-", "_", "."):
                out.append(ch)
            else:
                out.append("_")
        return "".join(out)

    @staticmethod
    def _find_main_table(out_dir: Path) -> Optional[Path]:
        """
        Try to locate a primary TSV/CSV file in the saved outputs directory.
        """
        candidates = list(out_dir.glob("*.tsv")) + list(out_dir.glob("*.csv"))
        if not candidates:
            for sub in out_dir.glob("*"):
                if sub.is_dir():
                    candidates.extend(sub.glob("*.tsv"))
                    candidates.extend(sub.glob("*.csv"))
                    if candidates:
                        break
        return candidates[0] if candidates else None

    def _save_job_outputs(self, job, dst_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
        """
        Persist all remote artifacts locally (incl. stdout/stderr and default files),
        and explicitly pull results.csv / results.json if present.

        Returns:
            (csv_path, json_path) — paths under dst_dir if files were fetched, else None.
        """
        dst_dir.mkdir(parents=True, exist_ok=True)
        csv_path: Optional[Path] = None
        json_path: Optional[Path] = None

        # 1) Save default artifacts (output.md, images, stdout/stderr, etc.)
        try:
            # allow overwriting to avoid duplicate-file errors on retries
            job.save_files(str(dst_dir), overwrite=True)
        except Exception as e:
            raise RuntimeError(f"Failed to save BioLib job files: {e}")

        # 2) Some BioLib apps expose additional downloadable files not included by default
        #    (as links in output.md). Try to fetch them explicitly.
        try:
            files = job.get_files()  # dict-like; keys are filenames available for download
            csv_name = "results.csv" if "results.csv" in files else next((k for k in files if k.lower().endswith(".csv")), None)
            json_name = "results.json" if "results.json" in files else next((k for k in files if k.lower().endswith(".json")), None)

            if csv_name:
                target = dst_dir / Path(csv_name).name
                job.save_file(csv_name, str(target), overwrite=True)
                csv_path = target

            if json_name:
                target = dst_dir / Path(json_name).name
                job.save_file(json_name, str(target), overwrite=True)
                json_path = target

        except Exception:
            # If the API does not expose get_files/save_file for this app, just ignore.
            pass

        return csv_path, json_path

    @staticmethod
    def _validate_fasta(fasta_path: Path, min_len: int = 10) -> None:
        """Basic FASTA sanity checks before sending to BioLib."""
        txt = Path(fasta_path).read_text().splitlines()
        if not txt or not txt[0].startswith(">"):
            raise ValueError(f"FASTA header must start with '>' ({fasta_path})")
        seq = "".join([ln.strip() for ln in txt[1:] if ln and not ln.startswith(">")]).upper().replace(" ", "")
        if len(seq) < min_len:
            raise ValueError(f"Sequence length {len(seq)} < {min_len} in {fasta_path}")
        allowed = set("ACDEFGHIKLMNPQRSTVWYBXZOU")  # be tolerant for rare letters
        bad = sorted({ch for ch in seq if ch not in allowed})
        if bad:
            raise ValueError(f"Illegal amino-acid letters in {fasta_path}: {bad}")
