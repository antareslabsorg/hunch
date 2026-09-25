"""The sha256 a data manifest records for a split.

`hunch.data.build` writes a manifest whose per-split entries are mappings carrying `path`, `sha256` and `bytes`; the
slimmer manifests the later data builds wrote store a bare hex digest instead. Every tool in this directory that
verifies a source against its manifest has to accept both, so the normalisation lives here rather than in three
copies that can drift.
"""
from __future__ import annotations


def manifest_digest(entry) -> str:
    """The sha256 recorded for one split, whichever manifest shape produced it."""
    return entry["sha256"] if isinstance(entry, dict) else str(entry)
