"""The floor rule of FORMATS.md on the 0.6B release's own rows, plus an at-floor case.  python -m hunch.formats.test_floor_gate"""
from hunch.formats.equivalence_test import floor_gate


def test_floor_gate():
    row = lambda n, tv, d: {"n": n, "n_nonfinite": 0, "tv_max": tv, "n_argmax_diff": d}
    floor = row(6000, 7.677e-02, 28)                                            # bf16 against fp32, the 0.6B release
    gguf = floor_gate(row(6000, 4.880e-03, 1), floor)                           # its GGUF f16
    assert gguf["verdict"] == "pass" and abs(gguf["z"] + 5.02) < 0.05, gguf
    assert floor_gate(row(6000, 5.691e-03, 0), floor)["verdict"] == "pass"      # its MLX f16
    at = floor_gate(row(6000, 1.44e-02, 36), row(2531, 1.68e-02, 15))           # an earlier 1.7B GGUF f16 read on CUDA
    assert at["verdict"] == "at floor" and abs(at["z"] - 0.04) < 0.05, at
    assert floor_gate(row(6000, 8.0e-02, 20), floor)["verdict"] == "no"         # outside the TV floor


if __name__ == "__main__":
    test_floor_gate(); print("floor gate ok")
