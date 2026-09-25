"""Tokenize records into candidate paths and pack complete questions into token-budgeted microbatches."""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import pad_paths
from .schema import Record
from .serialize import SERIALIZATION_VERSION, build_paths_batched


def cache_name(data_dir: str, split: str, tokenizer_name: str, max_path_tokens: int) -> str:
    """Deterministic on-disk cache path for a tokenized split."""
    import hashlib
    from pathlib import Path

    key = hashlib.sha1(f"{SERIALIZATION_VERSION}|{tokenizer_name}|{max_path_tokens}".encode()).hexdigest()[:12]
    return str(Path(data_dir) / ".tokcache" / f"{split}.{key}.pkl")


@dataclass
class TokenizedQuestion:
    record: Record
    paths: list[list[int]]  # one per candidate, in candidate order
    target: list[float]

    @property
    def k(self) -> int:
        return len(self.paths)

    @property
    def padded_tokens(self) -> int:
        return self.k * max(len(p) for p in self.paths)


def tokenize_records(records: list[Record], tokenizer, max_path_tokens: int, cache_path: str | None = None) -> tuple[list[TokenizedQuestion], int]:
    """Returns (questions, n_skipped). Questions whose longest path exceeds the limit are skipped whole and counted.
    With cache_path, results are pickled to disk (keyed by the caller) and reused. The cache is a trusted local file
    written by this function on the training host; it is never loaded from an untrusted source."""
    import pickle
    from pathlib import Path

    if cache_path and Path(cache_path).exists():
        with Path(cache_path).open("rb") as f:
            return pickle.load(f)
    out, skipped = [], 0
    for r, p in zip(records, build_paths_batched(records, tokenizer, max_path_tokens=max_path_tokens)):
        if p is None:
            skipped += 1
            continue
        out.append(TokenizedQuestion(r, p.flat(), list(r.target.probs)))
    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        tmp = Path(cache_path).with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump((out, skipped), f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(cache_path)
    return out, skipped


def subsample_candidates(q: TokenizedQuestion, max_k: int, g: torch.Generator) -> TokenizedQuestion:
    """Training-time candidate capping for large-K Choice questions (sampled softmax): keep the candidates carrying
    target mass (top by target, at least the argmax) plus uniformly sampled distractors, renormalize the kept target.
    Never used at evaluation, where the full candidate set is scored."""
    if q.k <= max_k or q.record.question.type != "choice":
        return q
    t = torch.tensor(q.target)
    order = torch.argsort(t, descending=True)
    n_pos = max(1, int((t > 1e-6).sum()))
    keep = order[: min(n_pos, max_k - 1)].tolist()
    rest = [i for i in range(q.k) if i not in set(keep)]
    n_neg = max_k - len(keep)
    if n_neg > 0 and rest:
        perm = torch.randperm(len(rest), generator=g)[:n_neg].tolist()
        keep += [rest[i] for i in perm]
    keep.sort()
    sub_t = t[keep]
    sub_t = (sub_t / sub_t.sum()).tolist() if float(sub_t.sum()) > 0 else [1.0 / len(keep)] * len(keep)
    return TokenizedQuestion(q.record, [q.paths[i] for i in keep], sub_t)


def pack_questions(questions: list[TokenizedQuestion], token_budget: int) -> list[list[TokenizedQuestion]]:
    """Greedy packing of complete questions; the padded footprint of a microbatch is (sum of paths) * (longest path).
    A single question above the budget raises: it must be filtered upstream, never split."""
    groups, cur, paths, width = [], [], 0, 0
    for q in sorted(questions, key=lambda q: max(len(p) for p in q.paths)):
        longest = max(len(p) for p in q.paths)
        if q.k * longest > token_budget:
            raise ValueError(f"{q.record.id}: needs {q.k * longest} padded tokens > budget {token_budget}")
        if cur and (paths + q.k) * max(width, longest) > token_budget:
            groups.append(cur)
            cur, paths, width = [], 0, 0
        cur.append(q)
        paths += q.k
        width = max(width, longest)
    if cur:
        groups.append(cur)
    return groups


def collate(group: list[TokenizedQuestion], pad_id: int, device) -> dict:
    flat = [p for q in group for p in q.paths]
    ids, mask, ro = pad_paths(flat, pad_id, device)
    kmax = max(q.k for q in group)
    target = torch.zeros((len(group), kmax), dtype=torch.float32)
    for i, q in enumerate(group):
        target[i, : q.k] = torch.tensor(q.target)
    return {"input_ids": ids, "attention_mask": mask, "readout_idx": ro, "group_sizes": [q.k for q in group],
            "target": target.to(device), "n_tokens": int(mask.sum())}


if __name__ == "__main__":  # note: packing invariants
    from .schema import Candidate, Question, Target, one_hot

    class Tok:  # deterministic fake tokenizer: one token per character; supports single and batched calls
        def encode(self, s, add_special_tokens=False):
            return [ord(c) % 100 + 3 for c in s]

        def __call__(self, texts, add_special_tokens=False):
            return {"input_ids": [self.encode(t) for t in texts]}

    recs = []
    for i in range(6):
        q = Question(id="q", type="choice", instruction="x" * (5 + i), candidates=[Candidate(str(j), "c" * (j + 1)) for j in range(2 + i % 3)])
        recs.append(Record(id=str(i), source="s", source_group=str(i), family="f", split="train", license_class="A", state="s" * (10 * i + 1), question=q,
                           target=Target("label_distribution", "deterministic_truth", one_hot(q.k, 0), 0)))
    qs, skipped = tokenize_records(recs, Tok(), max_path_tokens=200)
    assert skipped == 0 and len(qs) == 6
    groups = pack_questions(qs, token_budget=600)
    assert sum(len(g) for g in groups) == 6
    for g in groups:
        assert sum(q.k for q in g) * max(len(p) for q in g for p in q.paths) <= 600
    b = collate(groups[0], pad_id=0, device="cpu")
    assert b["input_ids"].shape[0] == sum(b["group_sizes"]) and torch.allclose(b["target"].sum(-1), torch.ones(len(groups[0])))
    big = Question(id="q", type="choice", instruction="big", candidates=[Candidate(str(j), f"c{j}") for j in range(40)])
    rec = Record(id="big", source="s", source_group="b", family="f", split="train", license_class="A", state="s", question=big,
                 target=Target("label_distribution", "deterministic_truth", one_hot(40, 17), 17))
    tq, _ = tokenize_records([rec], Tok(), max_path_tokens=200)
    sub = subsample_candidates(tq[0], 8, torch.Generator().manual_seed(0))
    assert sub.k == 8 and abs(sum(sub.target) - 1) < 1e-9 and max(sub.target) == 1.0, (sub.k, sub.target)
    assert any(p == tq[0].paths[17] for p in sub.paths), "gold candidate must be kept"
    print("batching ok", [len(g) for g in groups], "capped K", sub.k)
