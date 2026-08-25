"""
model.py - the capped 1D CNN, and the receptive-field proof that goes with it.

THE CAP IS ENFORCED BY THE INPUT, NOT BY THE ARCHITECTURE.

The network is fed a window of exactly `cap_symbols * SPS` samples and asked to
label the symbol in it. It cannot see more context than that because no more
context exists in its input. This is a stronger statement than "we chose kernel
sizes so the receptive field works out", because it survives any later change to
the layer stack, and it is trivial for a reviewer to verify.

Why the caps are what they are (handbook Section 11.1): the board's sequence is
a 63-bit order-6 m-sequence, so any 6 consecutive bits determine the 7th exactly
via s[i+6] = s[i] XOR s[i+5]. Any 5 do not. So as long as the network sees at
most 5 bits, predicting the next symbol from context is not merely unlikely -
it is impossible, and the network must read the waveform.

    BPSK   5 symbols x 1 bit = 5 bits   100 samples
    QPSK   2 symbols x 2 bit = 4 bits    40 samples
    8-QAM  1 symbol  x 3 bit = 3 bits    20 samples
"""

from __future__ import annotations

import torch
import torch.nn as nn

MAX_BITS_VISIBLE = 5          # 6 would let the LFSR recursion close


def conv_receptive_field(kernels, strides) -> int:
    """Receptive field of a stack of 1-D convolutions, in input samples."""
    rf, jump = 1, 1
    for k, s in zip(kernels, strides):
        rf += (k - 1) * jump
        jump *= s
    return rf


class CappedCNN(nn.Module):
    """Small 1-D CNN over the raw PASSBAND waveform.

    Two heads, per handbook Section 11.2:
      * `sym`  - class logits for the constellation point
      * `iq`   - a latent (I, Q) used by the constellation loss

    Amplitude normalisation is an AGC over the whole capture, applied in
    data.py. Section 11.3's per-window normalisation is applied to the
    PREDICTED (I, Q) inside the constellation loss, where it belongs - not to
    the input, where it would erase the 8-QAM ring.
    """

    # strides are chosen only to keep the tensor small; they cannot widen the
    # receptive field beyond the input window, which is the actual cap
    _STRIDES = {20: (1, 1, 1, 1), 40: (2, 1, 1, 1), 100: (2, 2, 2, 1)}

    def __init__(self, n_classes: int, window: int, channels: int = 32,
                 kernel: int = 5):
        super().__init__()
        if window not in self._STRIDES:
            raise ValueError(f"no stride plan for a {window}-sample window")
        strides = self._STRIDES[window]
        self.window = window
        self.n_classes = n_classes
        self.kernels = (kernel,) * 4
        self.strides = strides

        layers, c_in, length = [], 1, window
        for s in strides:
            layers += [nn.Conv1d(c_in, channels, kernel, stride=s), nn.ReLU()]
            length = (length - kernel) // s + 1
            c_in = channels
        if length < 1:
            raise ValueError(f"window {window} too short for this stack")
        self.features = nn.Sequential(*layers)
        # Full-width learned combiner instead of global average pooling.
        # Average pooling would make the output depend on every position
        # equally, so the honest receptive field would be the whole window
        # while the printed conv-stack number said something smaller. A conv
        # spanning exactly the remaining length keeps one number that is both
        # true and informative, and lets the network weight the target symbol's
        # position rather than smearing it into its neighbours.
        self.combine = nn.Sequential(
            nn.Conv1d(channels, channels, length), nn.ReLU())
        self.kernels = self.kernels + (length,)
        self.strides = self.strides + (1,)
        self.sym = nn.Linear(channels, n_classes)
        self.iq = nn.Linear(channels, 2)
        self.out_length = length

    def forward(self, x):
        # x: (batch, 1, window), already AGC-normalised in data.py by the
        # capture's RMS. Do NOT re-normalise per window here: for 8-QAM the
        # window is a single symbol, and forcing it to unit energy erases the
        # inner/outer ring, which is 1 of the 3 bits per symbol.
        h = self.combine(self.features(x)).flatten(1)
        return self.sym(h), self.iq(h)

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def receptive_field(self) -> int:
        """Input samples one output actually depends on.

        Capped by the window whatever the stack does: the input physically
        contains nothing else.
        """
        return min(conv_receptive_field(self.kernels, self.strides), self.window)


def build(modulation: str, sps: int, cap_symbols: int, **kw) -> CappedCNN:
    """Build a model and PROVE its cap before returning it."""
    import mcm31 as M

    allowed = M.MAX_RECEPTIVE_FIELD_SYMBOLS[modulation]
    bits = cap_symbols * M.BITS_PER_SYMBOL[modulation]
    window = cap_symbols * sps

    # the assertions the paper quotes
    assert cap_symbols <= allowed, (
        f"{modulation}: {cap_symbols} symbols exceeds the cap of {allowed}")
    assert bits <= MAX_BITS_VISIBLE, (
        f"{modulation}: {cap_symbols} symbols exposes {bits} bits; "
        f"{MAX_BITS_VISIBLE} is the limit before the PRBS63 recursion closes")

    net = CappedCNN(M.n_classes(modulation), window, **kw)
    assert net.receptive_field() <= window
    return net


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import mcm31 as M

    SPS = 20
    print("RECEPTIVE-FIELD PROOF  (assert-backed, printed for the paper)\n")
    print(f"  PRBS63 is an order-6 m-sequence: any 6 bits determine the 7th.")
    print(f"  A network that sees <= {MAX_BITS_VISIBLE} bits cannot close that "
          f"recursion.\n")
    print(f"  {'mod':<6} {'symbols':>8} {'bits':>5} {'window':>8} {'conv RF':>9} "
          f"{'effective':>10} {'params':>8} {'safe':>6}")
    print("  " + "-" * 70)
    for mod, cap in M.MAX_RECEPTIVE_FIELD_SYMBOLS.items():
        net = build(mod, SPS, cap)
        raw = conv_receptive_field(net.kernels, net.strides)
        print(f"  {mod:<6} {cap:>8} {cap * M.BITS_PER_SYMBOL[mod]:>5} "
              f"{net.window:>8} {raw:>9} {net.receptive_field():>10} "
              f"{net.n_parameters():>8,} {'YES':>6}")
        assert raw <= net.window, "conv stack reaches outside the window"
        x = torch.randn(4, 1, net.window)
        s, iq = net(x)
        assert s.shape == (4, net.n_classes) and iq.shape == (4, 2)

    print("\n  A window of N symbols contains N symbols. The network cannot")
    print("  attend to a 6th bit because no 6th bit is present in its input.")
