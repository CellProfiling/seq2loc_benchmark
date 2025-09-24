import os
import time
import random
from io import BytesIO
import threading

import pandas as pd
from dask import delayed, compute
from dask.threaded import get as threaded_get
from dask.diagnostics import ProgressBar
from tqdm import tqdm
from Bio.ExPASy import ScanProsite

# Global rate limiter shared across threads
_REQUEST_LOCK = threading.Lock()
_LAST_REQUEST_TS = 0.0
_MIN_SECONDS_BETWEEN_REQUESTS = 1.0  # adjust if server continues rate limiting


def _enforce_rate_limit():
    """Ensure a minimum gap between starting successive remote requests."""
    global _LAST_REQUEST_TS
    with _REQUEST_LOCK:
        now = time.time()
        elapsed = now - _LAST_REQUEST_TS
        if elapsed < _MIN_SECONDS_BETWEEN_REQUESTS:
            time.sleep(_MIN_SECONDS_BETWEEN_REQUESTS - elapsed)
        _LAST_REQUEST_TS = time.time()

def run_ps_scan(sequence: str, max_retries: int = 7, initial_backoff_sec: float = 2.0):
    """
    Call ExPASy's ScanProsite service via Biopython for a single sequence.
    Adds retry with exponential backoff and detects server-side rate limiting.

    Returns a Bio.ExPASy.ScanProsite.Record or None on persistent failure.
    """
    if not sequence:
        return None

    rate_limit_markers = (
        b"Too many user pattern scan jobs",
        b"Please retry later",
        b"<center>ERROR",
    )

    # Avoid Python version-specific typing here to be maximally compatible
    last_exception = None
    for attempt_index in range(max_retries):
        try:
            _enforce_rate_limit()
            handle = ScanProsite.scan(seq=sequence, output='xml')
            raw_bytes: bytes = handle.read()
            handle.close()

            # Check for server-side rate limit HTML response and retry if seen
            if any(marker in raw_bytes for marker in rate_limit_markers):
                raise RuntimeError("ScanProsite rate limit: server asked to retry later")

            # Parse the XML content from memory to avoid re-hitting the server
            memory_handle = BytesIO(raw_bytes)
            record = ScanProsite.read(memory_handle)
            memory_handle.close()
            return record
        except Exception as exc:  # noqa: BLE001 - broad to capture remote HTML/parse errors
            last_exception = exc

            # Backoff and retry except on final attempt
            if attempt_index < max_retries - 1:
                # Exponential backoff with jitter
                sleep_seconds = (initial_backoff_sec * (2 ** attempt_index)) + random.uniform(0, initial_backoff_sec)
                time.sleep(sleep_seconds)
                continue
            # Exhausted retries
            print(f"Error calling ScanProsite: {exc}")
            return None

def parse_scanprosite_record(record, protein_id):
    """Parse a ScanProsite Record into our motif dict format."""
    motifs = []
    if not record:
        return motifs
    for hit in record:
        try:
            start = int(hit.get('start')) if isinstance(hit, dict) else int(getattr(hit, 'start'))
            stop = int(hit.get('stop')) if isinstance(hit, dict) else int(getattr(hit, 'stop'))
            signature_ac = hit.get('signature_ac') if isinstance(hit, dict) else getattr(hit, 'signature_ac')
            level = hit.get('level') if isinstance(hit, dict) else getattr(hit, 'level', None)
            motifs.append({
                'protein_id': protein_id,
                'motif_ac': signature_ac,
                'start_pos': start,
                'end_pos': stop,
                'score': level if level is not None else None,
                'length': stop - start + 1,
            })
        except Exception:
            continue
    return motifs

def _scan_one_protein(protein_id: str, sequence: str):
    """Top-level helper for parallel execution. Returns list of motif dicts."""
    if not sequence:
        return []
    # Small jitter to avoid bursty simultaneous requests
    time.sleep(random.uniform(0.0, 0.5))
    record = run_ps_scan(sequence)
    if not record:
        return []
    return parse_scanprosite_record(record, protein_id)

def run_ps_scan_motif_finder(hou):
    print(f"Starting ScanProsite analysis (ExPASy) for {len(hou)} proteins...")

    # Initialize list to store all motif results
    all_motifs = []

    # Process each protein sequence
    for idx, row in tqdm(hou.iterrows(), total=len(hou), desc="Scanning proteins"):
        protein_id = row.get('uniprot_id', f'protein_{idx}')
        sequence = row['sequence']
        
        # Skip empty sequences
        if not sequence or len(sequence) == 0:
            continue
        
        # Run ScanProsite on the sequence
        record = run_ps_scan(sequence)
        
        if record:
            # Parse the record and extract motif information
            motifs = parse_scanprosite_record(record, protein_id)
            all_motifs.extend(motifs)
        
        # Print progress every 100 proteins
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1} proteins, found {len(all_motifs)} motifs so far...")

    print(f"\nCompleted scanning {len(hou)} proteins.")
    print(f"Total motifs found: {len(all_motifs)}")

    # Create the hou_prosite_motifs dataframe
    hou_prosite_motifs = pd.DataFrame(all_motifs)

    if len(hou_prosite_motifs) > 0:
        print(f"\nhou_prosite_motifs dataframe created with {len(hou_prosite_motifs)} rows and {len(hou_prosite_motifs.columns)} columns.")
        print("\nColumns:", list(hou_prosite_motifs.columns))
        print("\nFirst few rows:")
        print(hou_prosite_motifs.head())
        
        # Summary statistics
        print(f"\nSummary:")
        print(f"- Unique proteins with motifs: {hou_prosite_motifs['protein_id'].nunique()}")
        print(f"- Unique motif types found: {hou_prosite_motifs['motif_ac'].nunique()}")
        print(f"- Average motifs per protein: {len(hou_prosite_motifs) / hou_prosite_motifs['protein_id'].nunique():.2f}")
        
        # Most common motifs
        print("\nTop 10 most frequent motifs:")
        print(hou_prosite_motifs['motif_ac'].value_counts().head(10))
        
    else:
        print("\nNo motifs were found in any of the sequences.")
        hou_prosite_motifs = pd.DataFrame(columns=['protein_id', 'motif_ac', 'start_pos', 'end_pos', 'score', 'length'])

def run_ps_scan_motif_finder_dask(hou, max_workers: int = 3):
    """
    Parallel version using Dask (local processes) with a diagnostics progress bar,
    calling ExPASy's ScanProsite for each sequence.
    """
    # Limit concurrency to avoid overwhelming the remote ScanProsite service.
    max_workers = max(1, int(max_workers))
    print(f"Starting ScanProsite (Dask) for {len(hou)} proteins with up to {max_workers} concurrent workers...")

    delayed_tasks = []
    for idx, row in hou.iterrows():
        protein_id = row.get('uniprot_id', f'protein_{idx}')
        sequence = row['sequence']
        task = delayed(_scan_one_protein)(protein_id, sequence)
        delayed_tasks.append(task)

    # Compute with threaded scheduler to share a process-level rate limiter and restrict concurrency
    with ProgressBar():
        results = compute(*delayed_tasks, scheduler=threaded_get, num_workers=max_workers)

    # Flatten and aggregate
    all_motifs = [motif for sublist in results for motif in sublist]

    hou_prosite_motifs = pd.DataFrame(all_motifs) if all_motifs else pd.DataFrame(
        columns=['protein_id', 'motif_ac', 'start_pos', 'end_pos', 'score', 'length']
    )

    print(f"\nCompleted scanning {len(hou)} proteins (Dask).")
    print(f"Total motifs found: {len(hou_prosite_motifs)}")

    if len(hou_prosite_motifs) > 0:
        print(f"\nhou_prosite_motifs dataframe created with {len(hou_prosite_motifs)} rows and {len(hou_prosite_motifs.columns)} columns.")
        print("\nColumns:", list(hou_prosite_motifs.columns))
        print("\nFirst few rows:")
        print(hou_prosite_motifs.head())
        print(f"\nSummary:")
        print(f"- Unique proteins with motifs: {hou_prosite_motifs['protein_id'].nunique()}")
        print(f"- Unique motif types found: {hou_prosite_motifs['motif_ac'].nunique()}")
        print(f"- Average motifs per protein: {len(hou_prosite_motifs) / hou_prosite_motifs['protein_id'].nunique():.2f}")
        print("\nTop 10 most frequent motifs:")
        print(hou_prosite_motifs['motif_ac'].value_counts().head(10))

    return hou_prosite_motifs