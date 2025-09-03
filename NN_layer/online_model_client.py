# --*-- coding:utf-8 --*--
# @time: 8/28/25 23:38
# @Author : Yuqi Zhang
# @Email  : yzhan135@kent.edu
# @File   : online_model_client.py

from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional
import shutil
import time
import json
import traceback

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
    output_dir: Path           # local copy of BioLib job outputs
    main_table: Optional[Path] # TSV/CSV path if detected
    meta: Dict

@dataclass
class RawArtifactIndex:
    """Collection of raw outputs for downstream post-processing."""
    items: List[RawArtifact]
    meta: Dict                # run-wide metadata (app id, version, timings, failures)


# ---------- Client for NetSurfP-3 over BioLib ----------

class OnlineModelClient:
    """
    Minimal client for submitting standardized FASTA files to NetSurfP-3 via BioLib,
    saving the raw outputs locally, and returning an index.

    IMPORTANT:
      - Use AppID 'DTU/NetSurfP-3' (NOT 'DTU/NetSurfP-3.0'); the latter returns 400.
      - This app requires CLI args: -i <input_fasta>  -o <output_dir>.
        Therefore we must call .start(i=..., o="out"); .wait(); then save files.
    """

    def __init__(
        self,
        app_id: str = "DTU/NetSurfP-3",
        timeout_s: int = 3600,         # kept for API symmetry; not used by biolib.wait()
        retries: int = 2,              # total attempts = retries + 1
        rate_limit_s: float = 2.0,     # be gentle between jobs
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

        for i, rec in enumerate(dataset.records):
            seq_id = rec.seq_id
            fasta_path = Path(rec.fasta_path)
            if self.verbose:
                print(f"[{i+1}/{len(dataset.records)}] Running NetSurfP-3 for: {seq_id}")

            # Per-record output folder (local)
            rec_out = out_dir / self._safe(seq_id)
            if rec_out.exists() and self.overwrite:
                shutil.rmtree(rec_out, ignore_errors=True)
            rec_out.mkdir(parents=True, exist_ok=True)

            # Run job with retries
            ok = False
            last_err = None
            for attempt in range(self.retries + 1):
                try:
                    # Validate FASTA early to avoid opaque remote failures
                    self._validate_fasta(fasta_path, self.min_len)

                    # ---- KEY CHANGE: use start + wait (no timeout kw) ----
                    job = self._app.start(i=str(fasta_path), o="out")
                    job.wait()  # don't pass timeout; this pybiolib doesn't accept it
                    status = str(job.get_status()).upper()

                    # Always save remote artifacts (incl. stdout/stderr) locally
                    self._save_job_outputs(job, rec_out)

                    if self.verbose:
                        try:
                            out_txt = job.get_stdout()
                            if out_txt:
                                print(f"[BioLib][{seq_id}] status={status}  stdout[0:200]: {out_txt[:200]}")
                        except Exception:
                            pass

                    if status != "SUCCEEDED":
                        raise RuntimeError(f"BioLib job status={status}")

                    # Try to locate a main TSV/CSV for downstream parsing
                    main_table = self._find_main_table(rec_out)

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
                            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        }
                    ))
                    ok = True
                    break

                except Exception as e:
                    last_err = e
                    if self.verbose:
                        print(f"  -> attempt {attempt+1} failed: {e}")
                        traceback.print_exc()

                    # Handle compute_limit_exceeded with longer backoff
                    sleep_s = min(5.0 * (attempt + 1), 15.0)  # default short backoff
                    if isinstance(e, HttpError) and "compute_limit_exceeded" in str(e):
                        sleep_s = self.long_backoff_s * (attempt + 1)
                        if self.verbose:
                            print(f"  -> server under high load; backing off for {sleep_s}s")

                    time.sleep(sleep_s)

            if not ok:
                failures.append({
                    "seq_id": seq_id,
                    "input_fasta": str(fasta_path),
                    "error": repr(last_err),
                    "saved_dir": str(rec_out),
                })

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

        if self.verbose:
            print(f"[OK] NetSurfP-3 runs completed: {len(items)} success, {len(failures)} failed")
            if failures:
                print("     See failures in raw_index.json")

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
        Try to locate the primary TSV/CSV file produced by NetSurfP-3.
        Real runs typically emit a tabular summary either at the root or inside the declared output dir.
        """
        candidates = list(out_dir.glob("*.tsv")) + list(out_dir.glob("*.csv"))
        if not candidates:
            # also check nested dirs (e.g., 'out/')
            for sub in out_dir.glob("*"):
                if sub.is_dir():
                    candidates.extend(sub.glob("*.tsv"))
                    candidates.extend(sub.glob("*.csv"))
                    if candidates:
                        break
        return candidates[0] if candidates else None

    @staticmethod
    def _save_job_outputs(job, dst_dir: Path) -> None:
        """Persist all remote artifacts locally (incl. stdout/stderr and files under -o)."""
        dst_dir.mkdir(parents=True, exist_ok=True)
        try:
            job.save_files(str(dst_dir))
        except Exception as e:
            raise RuntimeError(f"Failed to save BioLib job files: {e}")

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
