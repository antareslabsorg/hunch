"""On-device formats of the Hunch scorer: GGUF (llama.cpp) and MLX (Apple silicon).

Hunch is a scorer, not a causal LM. Each converter ships the Qwen3 backbone in the target format and carries the
fp32 readout (RMSNorm + Linear(hidden -> 1)) and the release temperature as metadata; each runtime applies the readout
in fp32 and the softmax per question. `equivalence_test.py` is the deliverable: it measures, on the 6,000 archived
held-out predictions, how far every artifact is from the PyTorch reference and prints the failures too.
"""
