import os
import csv
import json
import argparse
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import pandas as pd
import requests
from requests.exceptions import RequestException

# -----------------------
# Defaults (override via CLI)
# -----------------------
DEFAULT_INPUT_DIR = "input"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_COLUMN_NAME = "track-visit-website href"
DEFAULT_CHECKPOINT = "validate-url-checkpoint.json"
DEFAULT_TIMEOUT = 30  # seconds; handles sites that take >10s

# -----------------------
# Utility helpers
# -----------------------
def utc_now_iso() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

def parse_iso(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ")

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def load_checkpoint(checkpoint_file: str) -> Dict[str, Any]:
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict) and "files" in data and isinstance(data["files"], list):
                    return data
        except Exception as e:
            print(f"[warn] Failed to load checkpoint: {e}")
    # fresh skeleton
    return {
        "version": 1,
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "files": []
    }

def save_checkpoint(checkpoint_file: str, state: Dict[str, Any]) -> None:
    state["updated_at"] = utc_now_iso()
    tmp = checkpoint_file + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
        os.replace(tmp, checkpoint_file)
    except Exception as e:
        print(f"[warn] Failed to save checkpoint: {e}")

def find_file_state(state: Dict[str, Any], input_file: str) -> Optional[Dict[str, Any]]:
    for f in state["files"]:
        if f.get("input_file") == input_file:
            return f
    return None

def csv_output_paths(base_output_dir: str, input_csv_path: str) -> Tuple[str, str]:
    base_name = os.path.basename(input_csv_path)
    stem, _ = os.path.splitext(base_name)
    per_file_folder = os.path.join(base_output_dir, stem)
    safe_mkdir(per_file_folder)
    output_csv = os.path.join(per_file_folder, base_name)
    return per_file_folder, output_csv

def infer_progress_from_output(output_csv: str) -> int:
    """
    Infer last_processed_index from an existing output file:
    returns -1 if no rows (only header or file missing),
    otherwise returns index of last processed input row (0-based).
    """
    if not os.path.exists(output_csv):
        return -1
    try:
        with open(output_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
            if not rows:
                return -1
            # first row is header; remaining are processed rows
            processed_count = max(0, len(rows) - 1)
            return processed_count - 1
    except Exception as e:
        print(f"[warn] Could not infer progress from existing output '{output_csv}': {e}")
        return -1

# -----------------------
# Networking
# -----------------------
SESSION = requests.Session()
SESSION.headers.update({
    "User-Agent": "URL-Accessibility-Checker/1.0 (+https://example.local)"
})

def is_website_accessible(url: Any, timeout: int = DEFAULT_TIMEOUT) -> bool:
    if not isinstance(url, str):
        return False
    url = url.strip()
    if not url:
        return False
    try:
        resp = SESSION.get(url, timeout=timeout, allow_redirects=True)
        return 200 <= resp.status_code < 300
    except RequestException:
        return False

# -----------------------
# Core processing
# -----------------------
def process_file(
    input_file: str,
    column_name: str,
    base_output_dir: str,
    checkpoint_file: str,
    state: Dict[str, Any],
    timeout: int,
) -> None:
    print(f"[info] Processing: {input_file}")

    # Load CSV (let pandas infer encoding; you can pass encoding via CLI if needed)
    df = pd.read_csv(input_file)
    total_rows = len(df)

    if column_name not in df.columns:
        print(f"[warn] Column '{column_name}' not found in {input_file}. Skipping.")
        return

    # Prepare per-file checkpoint entry
    per_file_folder, output_csv = csv_output_paths(base_output_dir, input_file)
    file_state = find_file_state(state, input_file)

    if file_state is None:
        # New entry (or checkpoint lost); try to infer from existing output
        inferred_last_idx = infer_progress_from_output(output_csv)
        file_state = {
            "input_file": input_file,
            "output_folder": per_file_folder,
            "output_file": output_csv,
            "column_name": column_name,
            "total_rows": total_rows,
            "last_processed_index": inferred_last_idx,
            "processed_rows": inferred_last_idx + 1 if inferred_last_idx >= 0 else 0,
            "status": "in_progress",
            "started_at": utc_now_iso(),
            "updated_at": utc_now_iso(),
            "completed_at": None
        }
        state["files"].append(file_state)
        save_checkpoint(checkpoint_file, state)
    else:
        # Keep totals in sync
        if file_state.get("total_rows") != total_rows:
            file_state["total_rows"] = total_rows

    # Decide whether to write header
    write_header = not os.path.exists(output_csv)
    if write_header:
        # (Re)create file with header
        with open(output_csv, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out)
            header = list(df.columns) + ["Accessibility"]
            writer.writerow(header)

    # Determine starting index (resume point)
    start_idx = file_state.get("last_processed_index", -1) + 1
    if start_idx >= total_rows:
        # Already done—ensure completed metadata & duration set
        file_state["status"] = "completed"
        file_state["completed_at"] = file_state.get("completed_at") or utc_now_iso()
        try:
            start_dt = parse_iso(file_state["started_at"])
            try:
                end_dt = parse_iso(file_state["completed_at"])
            except Exception:
                end_dt = datetime.utcnow()
            file_state["duration_minutes"] = round((end_dt - start_dt).total_seconds() / 60.0, 2)
        except Exception:
            pass
        file_state["updated_at"] = utc_now_iso()
        save_checkpoint(checkpoint_file, state)
        print(f"[info] Already completed: {input_file}")
        return

    # Open output in append mode, stream one row at a time
    with open(output_csv, "a", newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out)

        for idx in range(start_idx, total_rows):
            row = df.iloc[idx]
            url = row[column_name] if column_name in df.columns else ""
            accessible = "Accessible" if is_website_accessible(url, timeout=timeout) else "Not Accessible"

            writer.writerow(row.tolist() + [accessible])
            f_out.flush()  # persist each line

            # Update checkpoint live after every row
            file_state["last_processed_index"] = idx
            file_state["processed_rows"] = idx + 1
            file_state["updated_at"] = utc_now_iso()
            save_checkpoint(checkpoint_file, state)

            print(f"[row {idx+1}/{total_rows}] {url} -> {accessible}")

    # Mark file complete + duration
    file_state["status"] = "completed"
    file_state["completed_at"] = utc_now_iso()
    try:
        start_dt = parse_iso(file_state["started_at"])
        end_dt = parse_iso(file_state["completed_at"])
        file_state["duration_minutes"] = round((end_dt - start_dt).total_seconds() / 60.0, 2)
    except Exception:
        pass
    file_state["updated_at"] = utc_now_iso()
    save_checkpoint(checkpoint_file, state)
    print(f"[done] Output: {output_csv}")

# -----------------------
# Main
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Check URL accessibility in CSV files with resume support.")
    parser.add_argument("--input", default=DEFAULT_INPUT_DIR, help="Input folder containing CSV files")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_DIR, help="Base output folder (per-file subfolders will be created)")
    parser.add_argument("--column", default=DEFAULT_COLUMN_NAME, help="CSV column name containing URLs")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Checkpoint JSON path")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds for each URL")
    args = parser.parse_args()

    safe_mkdir(args.input)
    safe_mkdir(args.output)

    # Create or load checkpoint immediately (file exists from the start)
    state = load_checkpoint(args.checkpoint)
    save_checkpoint(args.checkpoint, state)

    try:
        files = [f for f in os.listdir(args.input) if f.lower().endswith(".csv")]
        if not files:
            print(f"[info] No CSV files found in '{args.input}'.")
            return

        for name in files:
            input_file = os.path.join(args.input, name)
            try:
                process_file(
                    input_file=input_file,
                    column_name=args.column,
                    base_output_dir=args.output,
                    checkpoint_file=args.checkpoint,
                    state=state,
                    timeout=args.timeout,
                )
            except KeyboardInterrupt:
                print("\n[keyboard] Interrupted. State saved to checkpoint. Rerun to resume.")
                return
            except Exception as e:
                print(f"[error] Failed while processing '{input_file}': {e}")
                # mark file as error (but keep progress)
                file_state = find_file_state(state, input_file)
                if file_state:
                    file_state["status"] = "error"
                    file_state["updated_at"] = utc_now_iso()
                    save_checkpoint(args.checkpoint, state)
    finally:
        save_checkpoint(args.checkpoint, state)

if __name__ == "__main__":
    main()
