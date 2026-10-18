"""Throwaway feasibility-probe downloader; preserve receipts for reproduction.

Only the three interaction CSVs are copied from regular archive members.
No downloaded code or side-information/statistics file is executed or extracted.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
RECORD_URL = "https://zenodo.org/api/records/10439422"
NAME = "KuaiRand-Pure.tar.gz"
WANTED = {
    "log_standard_4_08_to_4_21_pure.csv",
    "log_standard_4_22_to_5_08_pure.csv",
    "log_random_4_22_to_5_08_pure.csv",
}


def request(url):
    return urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "ResearchDatasetAudit/1.0"}),
        timeout=60,
    )


def digests(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    return md5.hexdigest(), sha.hexdigest()


def main():
    started = time.perf_counter()
    RAW.mkdir(parents=True, exist_ok=True)
    with request(RECORD_URL) as response:
        record = json.load(response)
    (ROOT / "zenodo_record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    source = next(item for item in record["files"] if item["key"] == NAME)
    if source["size"] > 100_000_000:
        raise RuntimeError("Archive exceeds preflight scope of 100 MB")
    if shutil.disk_usage(RAW).free < 1_000_000_000:
        raise RuntimeError("Less than 1 GB free for this bounded probe")
    archive = RAW / NAME
    download_seconds = 0.0
    if not archive.exists():
        download_start = time.perf_counter()
        partial = RAW / (NAME + ".partial")
        total, last_notice = 0, 0
        with request(source["links"]["self"]) as response, partial.open("wb") as output:
            for block in iter(lambda: response.read(1024 * 1024), b""):
                output.write(block)
                total += len(block)
                if total - last_notice >= 10_000_000:
                    print(json.dumps({"downloaded_bytes": total, "expected_bytes": source["size"]}), flush=True)
                    last_notice = total
        if total != source["size"]:
            raise RuntimeError(f"Size mismatch: {total} != {source['size']}")
        md5, sha = digests(partial)
        if "md5:" + md5 != source["checksum"]:
            raise RuntimeError("Published checksum mismatch")
        partial.replace(archive)
        download_seconds = time.perf_counter() - download_start
    md5, sha = digests(archive)
    if archive.stat().st_size != source["size"] or "md5:" + md5 != source["checksum"]:
        raise RuntimeError("Existing archive does not match published release")
    files, seen = [], set()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle:
            base = Path(member.name).name
            if base not in WANTED:
                continue
            if not member.isfile() or base in seen:
                raise RuntimeError("Unsafe or duplicate selected archive member")
            seen.add(base)
            if member.size > 150_000_000:
                raise RuntimeError("Selected CSV exceeds expected scope")
            destination = (RAW / base).resolve()
            if destination.parent != RAW.resolve():
                raise RuntimeError("Extraction escaped raw directory")
            stream = bundle.extractfile(member)
            if stream is None:
                raise RuntimeError("Selected member has no data stream")
            with stream, destination.open("wb") as output:
                shutil.copyfileobj(stream, output)
            files.append({"archive_member": member.name, "path": str(destination), "bytes": destination.stat().st_size, "sha256": digests(destination)[1]})
    if seen != WANTED:
        raise RuntimeError(f"Missing interaction logs: {sorted(WANTED - seen)}")
    receipt = {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "record": record["id"], "doi": record.get("doi"),
        "license": record["metadata"].get("license"),
        "archive": {"name": NAME, "bytes": archive.stat().st_size, "published_checksum": source["checksum"], "verified_md5": md5, "sha256": sha, "url": source["links"]["self"]},
        "files": files, "download_seconds": download_seconds,
        "total_seconds": time.perf_counter() - started,
    }
    (ROOT / "release_receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
