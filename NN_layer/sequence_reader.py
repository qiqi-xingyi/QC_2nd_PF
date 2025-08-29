# --*-- conding:utf-8 --*--
# @time:8/29/25 00:27
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:sequence_reader.py

# NN_layer/sequence_reader.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import hashlib
import json
import time

# Allowed amino acids (20 canonical residues)
ALLOWED_AA = set("ACDEFGHIKLMNPQRSTVWY")

# ---------------------------
# Data structures
# ---------------------------

@dataclass
class SeqRecord:
    """One sequence record (raw or windowed) that can be fed to the model."""
    seq_id: str
    seq: str
    start: int
    end: int
    fasta_path: Path
    sha1: str
    length: int

@dataclass
class Dataset:
    """Dataset produced by SequenceReader, consumed by the model client."""
    records: List[SeqRecord]
    meta: Dict

# ---------------------------
# Utilities
# ---------------------------

def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def _read_fasta(path: Path) -> List[Tuple[str, str]]:
    """Minimal FASTA reader: returns list of (id, sequence)."""
    items: List[Tuple[str, str]] = []
    seq_id: Optional[str] = None
    buf: List[str] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if seq_id is not None:
                items.append((seq_id, "".join(buf)))
            seq_id = line[1:].strip() or f"unnamed_{len(items)}"
            buf = []
        else:
            buf.append(line)
    if seq_id is not None:
        items.append((seq_id, "".join(buf)))
    return items

def _clean_sequence(raw: str, *, upper: bool = True) -> str:
    """Keep only allowed amino acids, uppercase if required."""
    s = raw.upper() if upper else raw
    return "".join([c for c in s if c in ALLOWED_AA])

# ---------------------------
# Main class: SequenceReader
# ---------------------------

class SequenceReader:
    """
    Responsibilities:
      1. Load FASTA file(s)
      2. Clean sequences (remove invalid residues)
      3. (Optional) slice into windows with given stride
      4. Write standardized FASTA files for model input
      5. Return Dataset with SeqRecord objects
    """
    def __init__(
        self,
        window_size: Optional[int] = None,
        stride: int = 1,
        dedup: bool = True,
        min_len: int = 3,
    ):
        assert stride >= 1, "stride must be >= 1"
        if window_size is not None:
            assert window_size >= 1, "window_size must be positive"
        self.window_size = window_size
        self.stride = stride
        self.dedup = dedup
        self.min_len = min_len

    def load(self, src: Path | str) -> List[Tuple[str, str]]:
        """
        Load sequences from a FASTA file or a directory of FASTA files.
        Return list of (seq_id, cleaned_seq).
        """
        src_path = Path(src)
        pairs: List[Tuple[str, str]] = []
        fasta_files: List[Path] = []

        if src_path.is_dir():
            for p in sorted(src_path.iterdir()):
                if p.suffix.lower() in {".fa", ".fasta", ".faa"}:
                    fasta_files.append(p)
        else:
            fasta_files.append(src_path)

        for fp in fasta_files:
            for sid, raw in _read_fasta(fp):
                seq = _clean_sequence(raw)
                if len(seq) < self.min_len:
                    continue
                pairs.append((sid, seq))

        if self.dedup:
            seen = set()
            uniq: List[Tuple[str, str]] = []
            for sid, seq in pairs:
                h = _sha1(seq)
                if h in seen:
                    continue
                seen.add(h)
                uniq.append((sid, seq))
            pairs = uniq

        return pairs

    def save_fasta(self, dataset: Dataset, out_dir: Path) -> Dataset:
        """
        Write each record in Dataset as a separate FASTA file.
        Also create an index.json for reproducibility.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for rec in dataset.records:
            name = f"{self._safe(rec.seq_id)}_{rec.start}_{rec.end}_{rec.sha1[:8]}.fasta"
            fpath = out_dir / name
            with fpath.open("w") as f:
                header = f">{rec.seq_id}|start={rec.start}|end={rec.end}|sha1={rec.sha1}"
                f.write(header + "\n")
                for i in range(0, len(rec.seq), 60):
                    f.write(rec.seq[i:i+60] + "\n")
            rec.fasta_path = fpath

        index = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "count": len(dataset.records),
            "records": [
                {**asdict(rec), "fasta_path": str(rec.fasta_path)} for rec in dataset.records
            ],
            "meta": dataset.meta,
        }
        (out_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2))
        return dataset

    def build_dataset(self, pairs: List[Tuple[str, str]]) -> Dataset:
        """
        Convert list of (seq_id, seq) into a Dataset object.
        Apply optional window slicing if window_size is set.
        """
        records: List[SeqRecord] = []
        for sid, seq in pairs:
            if self.window_size is None or len(seq) <= self.window_size:
                h = _sha1(seq)
                records.append(
                    SeqRecord(seq_id=sid, seq=seq, start=0, end=len(seq),
                              fasta_path=Path(), sha1=h, length=len(seq))
                )
            else:
                w = self.window_size
                st = self.stride
                for start in range(0, max(1, len(seq) - w + 1), st):
                    frag = seq[start:start + w]
                    if len(frag) < self.min_len:
                        continue
                    h = _sha1(frag)
                    records.append(
                        SeqRecord(seq_id=f"{sid}_win{start}_{start+len(frag)}",
                                  seq=frag, start=start, end=start+len(frag),
                                  fasta_path=Path(), sha1=h, length=len(frag))
                    )
        meta = {
            "window_size": self.window_size,
            "stride": self.stride,
            "dedup": self.dedup,
            "min_len": self.min_len,
            "total_input": len(pairs),
            "total_records": len(records),
        }
        return Dataset(records=records, meta=meta)

    @staticmethod
    def _safe(text: str) -> str:
        """Make a string safe for use in filenames."""
        out = []
        for ch in text:
            if ch.isalnum() or ch in ("-", "_", "."):
                out.append(ch)
            else:
                out.append("_")
        return "".join(out)
