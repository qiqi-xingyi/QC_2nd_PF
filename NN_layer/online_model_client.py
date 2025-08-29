# --*-- conding:utf-8 --*--
# @time:8/28/25 23:38
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:online_model_client.py

# NN_layer/online_model_client.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional
import shutil
import time
import json
import traceback

# Third-party: pip install biolib
try:
    import biolib
except ImportError as e:
    raise ImportError(
        "biolib is required for OnlineModelClient. "
        "Install with: pip install biolib"
    ) from e


# ---------- Data types passed between steps ----------

@dataclass
class RawArtifact:
    """A single model run output for one input FASTA."""
    seq_id: str
    input_fasta: Path
    output_dir: Path          # local copy of BioLib job outputs
    main_table: Optional[Path]  # TSV/CSV path if detected
    meta: Dict

@dataclass
class RawArtifactIndex:
    """Collection of raw outputs for downstream post-processing."""
    items: List[RawArtifact]
    meta: Dict                # run-wide metadata (app id, version, timings, failures)


# ---------- Client for NetSurfP-3.0 over BioLib ----------

class OnlineModelClient:
    """
    Minimal client for submitting standardized FASTA files to NetSurfP-3.0
    through BioLib, saving the raw outputs locally, and returning an index.
    """

    def __init__(
        self,
        app_id: str = "DTU/NetSurfP-3",
        timeout_s: int = 3600,
        retries: int = 1,
        rate_limit_s: float = 0.0,   # sleep between jobs to be gentle with the service
        overwrite: bool = False,
        verbose: bool = True,
    ):
        self.app_id = app_id
        self.timeout_s = timeout_s
        self.retries = retries
        self.rate_limit_s = rate_limit_s
        self.overwrite = overwrite
        self.verbose = verbose

        # Preload the BioLib application once
        self._app = biolib.load(self.app_id)

    # Public API -------------------------------------------------------------

    def predict(self, dataset, out_dir: Path) -> RawArtifactIndex:
        """
        Submit each FASTA in `dataset.records[*].fasta_path` to NetSurfP-3.0
        and persist raw outputs under:
            result/NN_layer/runs/<run_id>/raw/<seq_id>/

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

            # Per-record output folder
            rec_out = out_dir / self._safe(seq_id)
            if rec_out.exists() and self.overwrite:
                shutil.rmtree(rec_out, ignore_errors=True)
            rec_out.mkdir(parents=True, exist_ok=True)

            # Run job with basic retry
            ok = False
            last_err = None
            for attempt in range(self.retries + 1):
                try:
                    job = self._run_job(fasta_path)
                    # Copy BioLib job outputs to our rec_out
                    self._sync_outputs(job.output_path, rec_out)
                    main_table = self._find_main_table(rec_out)
                    items.append(RawArtifact(
                        seq_id=seq_id,
                        input_fasta=fasta_path,
                        output_dir=rec_out,
                        main_table=main_table,
                        meta={
                            "attempt": attempt,
                            "job_output_path": str(job.output_path),
                            "app_id": self.app_id,
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
                    time.sleep(min(2.0 * (attempt + 1), 5.0))  # simple backoff

            if not ok:
                failures.append({
                    "seq_id": seq_id,
                    "input_fasta": str(fasta_path),
                    "error": repr(last_err),
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

    # Internals --------------------------------------------------------------

    def _run_job(self, fasta_path: Path):
        """Run a blocking BioLib job with timeout semantics."""
        # `run` blocks until completion (BioLib handles server-side queueing).
        # If you prefer async: start() + wait(timeout=...) and check status.
        job = self._app.run(input_file=str(fasta_path))
        # Basic client-side timeout guard (BioLib handles most of it)
        # Here we just assert job finished by existence of output_path.
        if not Path(job.output_path).exists():
            raise RuntimeError("BioLib job finished without a valid output_path")
        return job

    @staticmethod
    def _sync_outputs(src_dir: str | Path, dst_dir: Path) -> None:
        """Copy job outputs into our project folder."""
        src = Path(src_dir)
        if not src.exists():
            raise FileNotFoundError(f"BioLib output directory not found: {src}")
        # Copytree-like sync; for small outputs a shallow copy is fine.
        for p in src.iterdir():
            target = dst_dir / p.name
            if p.is_dir():
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
                shutil.copytree(p, target)
            else:
                shutil.copy2(p, target)

    @staticmethod
    def _find_main_table(out_dir: Path) -> Optional[Path]:
        """Try to locate the primary TSV/CSV file produced by NetSurfP-3.0."""
        # Many deployments emit a single TSV/CSV results file; search it.
        candidates = list(out_dir.glob("*.tsv")) + list(out_dir.glob("*.csv"))
        if not candidates:
            # Also check nested dirs (some BioLib apps store in a subdir)
            for sub in out_dir.iterdir():
                if sub.is_dir():
                    candidates.extend(sub.glob("*.tsv"))
                    candidates.extend(sub.glob("*.csv"))
                    if candidates:
                        break
        return candidates[0] if candidates else None

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

