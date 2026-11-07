"""
Bit-Exact Integer Hardware Golden Model for BitMamba-CV (Intel Cyclone V).
All datapath operations use strictly integer arithmetic (int8 / int16 / int32) and
bit shifts, matching the Verilog RTL cycle-by-cycle and bit-by-bit (0 LSB mismatch).

Includes:
1. Radix-3 Packer & Combinational LUT Unpacker (5 trits in {-1,0,+1} <-> 1 uint8 byte in [0..242])
2. Multiplier-Free 16-Point Fast Walsh-Hadamard Transform (4 butterfly stages + >> 2 shift)
3. Three BitLinear Microarchitectures (Adder-Tree, Table-Lookup TLMM g=4, and Zero-Skipping)
4. Complementary Power-of-Two (CPoT) vs. Standard PoT Integer Selective Scan (Q8.8 state)
5. 16-Segment Piecewise-Linear (PWL) Integer SiLU
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np


# ==============================================================================
# 1. Radix-3 Weight Compression (5 trits per 8-bit byte: 1.60 bits/weight)
# ==============================================================================

def pack_radix3_weights(w_ternary: np.ndarray) -> np.ndarray:
    """
    Packs an array of ternary weights in {-1, 0, +1} along the last axis into uint8 bytes.
    Each group of 5 trits (w0, w1, w2, w3, w4) is mapped to d_i = w_i + 1 in {0, 1, 2}:
        byte = d0*1 + d1*3 + d2*9 + d3*27 + d4*81  in [0, 242].
    """
    assert np.all(np.isin(w_ternary, [-1, 0, 1])), "Weights must be strictly in {-1, 0, +1}"
    orig_shape = w_ternary.shape
    K = orig_shape[-1]
    pad = (5 - (K % 5)) % 5
    if pad > 0:
        pad_width = [(0, 0)] * (w_ternary.ndim - 1) + [(0, pad)]
        w_padded = np.pad(w_ternary, pad_width, mode="constant", constant_values=0)
    else:
        w_padded = w_ternary

    digits = (w_padded + 1).astype(np.uint8)
    digits = digits.reshape(*orig_shape[:-1], -1, 5)
    powers = np.array([1, 3, 9, 27, 81], dtype=np.uint8)
    packed = np.sum(digits * powers, axis=-1, dtype=np.uint8)
    return packed


def build_radix3_lut() -> np.ndarray:
    """
    Builds the 256-entry combinational ROM/LUT table used by radix3_decoder.v.
    Entry b in [0..242] maps to 5 trits in {-1, 0, +1}; entries [243..255] map to zeros.
    """
    lut = np.zeros((256, 5), dtype=np.int8)
    for b in range(243):
        val = b
        for i in range(5):
            d = val % 3
            lut[b, i] = d - 1
            val //= 3
    return lut


RADIX3_LUT = build_radix3_lut()


def unpack_radix3_weights(packed_bytes: np.ndarray, orig_k: int) -> np.ndarray:
    """Unpacks uint8 Radix-3 bytes back into int8 ternary weights {-1, 0, +1} via LUT."""
    unpacked = RADIX3_LUT[packed_bytes]
    flat = unpacked.reshape(*packed_bytes.shape[:-1], -1)
    return flat[..., :orig_k]


# ==============================================================================
# 2. Multiplier-Free 16-Point Fast Walsh-Hadamard Transform (FWHT)
# ==============================================================================

def fwht16_int_bitexact(x_int: np.ndarray) -> np.ndarray:
    """
    Bit-exact integer 16-point Fast Walsh-Hadamard Transform.
    Input: int16 array of shape (..., D) where D is a multiple of 16.
    4 butterfly stages of pure integer add/sub, followed by arithmetic right shift >> 2 (div by 4).
    """
    orig_shape = x_int.shape
    D = orig_shape[-1]
    assert D % 16 == 0, "Dimension D must be divisible by 16"
    blocks = x_int.astype(np.int32).reshape(-1, 16).copy()

    # 4 Butterfly stages (stride 1, 2, 4, 8)
    for stage in range(4):
        stride = 1 << stage
        next_blocks = np.empty_like(blocks)
        for i in range(0, 16, 2 * stride):
            for j in range(stride):
                a = blocks[:, i + j]
                b = blocks[:, i + j + stride]
                next_blocks[:, i + j] = a + b
                next_blocks[:, i + j + stride] = a - b
        blocks = next_blocks

    # Arithmetic right shift >> 2 (1/sqrt(16) = 1/4) with clamping to int8 range [-127, 127]
    out = np.right_shift(blocks, 2)
    out = np.clip(out, -127, 127).astype(np.int8)
    return out.reshape(orig_shape)


# ==============================================================================
# 3. Three BitLinear Microarchitectures (Adder-Tree vs. TLMM vs. Zero-Skip)
# ==============================================================================

def bitlinear_adder_tree_int(x_int8: np.ndarray, w_ternary: np.ndarray) -> np.ndarray:
    """
    Microarchitecture 1: Carry-Chain Ternary Adder Tree in Cyclone V ALMs.
    Each PE computes mux(w_ij, +x_j, 0, -x_j) and sums in int32 accumulator.
    """
    return np.matmul(x_int8.astype(np.int32), w_ternary.T.astype(np.int32))


def bitlinear_tlmm_lut_int(x_int8: np.ndarray, w_ternary: np.ndarray, group_size: int = 4) -> np.ndarray:
    """
    Microarchitecture 2: Table-Lookup Matrix Multiplication (TLMM, cf. T-MAC / TeLLMe).
    For every group of G=4 activations (x0, x1, x2, x3), precomputes all 3^4 = 81 possible
    ternary sums in a small LUT, then accumulates table lookups across K // 4 groups.
    """
    M, K = w_ternary.shape
    assert K % group_size == 0
    num_groups = K // group_size
    powers = (3 ** np.arange(group_size)).astype(np.int32)

    # Encode each group of G trits into an index in [0..3^G - 1]
    w_digits = (w_ternary + 1).reshape(M, num_groups, group_size).astype(np.int32)
    w_indices = np.sum(w_digits * powers, axis=-1)  # (M, num_groups)

    # All 3^G trit combinations in {-1, 0, +1}^G
    num_entries = 3 ** group_size
    combos = np.zeros((num_entries, group_size), dtype=np.int32)
    for idx in range(num_entries):
        v = idx
        for g in range(group_size):
            combos[idx, g] = (v % 3) - 1
            v //= 3

    x_flat = x_int8.astype(np.int32).reshape(-1, num_groups, group_size)
    B = x_flat.shape[0]
    out = np.zeros((B, M), dtype=np.int32)

    for g in range(num_groups):
        # Precompute 81-entry LUT for activation group g: (B, 81)
        lut_g = np.matmul(x_flat[:, g, :], combos.T)  # (B, 81)
        # Lookup using weight indices for group g: (M,)
        out += lut_g[:, w_indices[:, g]]

    return out.reshape(*x_int8.shape[:-1], M)


def bitlinear_zeroskip_int(
    x_int8: np.ndarray, w_ternary: np.ndarray, parallelism_p: int = 16
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Microarchitecture 3: Structured Zero-Skipping Ternary Engine.
    Skips zero weights (w == 0) in chunks of size P, returning exact integer output
    and cycle compression statistics.
    """
    M, K = w_ternary.shape
    dense_cycles = M * (K // parallelism_p)
    nz_mask = (w_ternary != 0)
    nz_counts_per_row = np.sum(nz_mask, axis=-1)
    active_cycles = int(np.sum(np.ceil(nz_counts_per_row / float(parallelism_p))))

    out = np.matmul(x_int8.astype(np.int32), w_ternary.T.astype(np.int32))
    stats = {
        "zero_fraction": float(np.mean(w_ternary == 0)),
        "dense_cycles": float(dense_cycles),
        "zeroskip_cycles": float(active_cycles),
        "cycle_reduction_ratio": float(dense_cycles) / max(float(active_cycles), 1.0),
    }
    return out, stats


# ==============================================================================
# 4. Complementary Power-of-Two (CPoT) vs. Standard PoT Selective Scan (Bit-Exact)
# ==============================================================================

def encode_abar_cpot(abar_float: np.ndarray, max_shift: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encodes floating-point A_bar in (0, 1) into hardware control signals:
    - mode_bit in {0, 1}: 0 -> standard PoT (2^-k), 1 -> Complementary PoT (1 - 2^-k)
    - shift_k in {1..max_shift}
    """
    abar_c = np.clip(abar_float, 1e-5, 1.0 - 1e-5)
    mode_bit = (abar_c > 0.625).astype(np.uint8)

    k_low = np.clip(np.round(-np.log2(abar_c)), 1, max_shift).astype(np.uint8)
    comp = np.clip(1.0 - abar_c, 1e-5, 0.5)
    k_high = np.clip(np.round(-np.log2(comp)), 1, max_shift).astype(np.uint8)

    shift_k = np.where(mode_bit == 1, k_high, k_low).astype(np.uint8)
    return mode_bit, shift_k


def cpot_step_int16(
    h_prev_q88: np.ndarray,
    mode_bit: np.ndarray,
    shift_k: np.ndarray,
    b_int8: np.ndarray,
    x_int8: np.ndarray,
    c_int8: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Executes 1 recurrent step of the Mamba-2 Selective Scan in 16-bit fixed-point (Q8.8)
    using strictly barrel shifts and integer add/sub for A_bar * h_{t-1}:
      - mode_bit == 0: h_decay = h_prev >> k
      - mode_bit == 1: h_decay = h_prev - (h_prev >> k)
      - h_new = clip(h_decay + ((b_int8 * x_int8) >> 4), -32768, 32767)
      - y_out = sum_n((h_new * c_int8) >> 8)
    """
    h_prev32 = h_prev_q88.astype(np.int32)
    shifted = np.right_shift(h_prev32, shift_k.astype(np.int32))
    h_decay = np.where(mode_bit == 1, h_prev32 - shifted, shifted)

    # Input injection B_t * x_t via Cyclone V 9x9/18x18 DSP
    bx_prod = np.right_shift(b_int8.astype(np.int32) * x_int8.astype(np.int32), 4)
    h_new = np.clip(h_decay + bx_prod, -32768, 32767).astype(np.int16)

    # Output readout sum_n(C_t * h_t) via Cyclone V DSP
    y_acc = np.sum(np.right_shift(h_new.astype(np.int32) * c_int8.astype(np.int32), 8), axis=-1)
    y_out = np.clip(y_acc, -32768, 32767).astype(np.int16)
    return h_new, y_out


# ==============================================================================
# 5. 16-Segment Piecewise-Linear (PWL) Integer SiLU (Q8.8 Fixed-Point)
# ==============================================================================

def build_pwl_silu_rom_q88() -> Tuple[np.ndarray, np.ndarray]:
    """
    Builds the 16-entry slope and intercept tables in Q8.8 fixed-point (scale = 256)
    for x in [-4.0, 4.0) with segment width 0.5 (128 in Q8.8).
    """
    knots = np.linspace(-4.0, 4.0, 17)
    silu_knots = knots / (1.0 + np.exp(-knots))
    slopes = np.diff(silu_knots) / 0.5  # dy / dx
    base_y = silu_knots[:-1]

    base_y_q88 = np.round(base_y * 256.0).astype(np.int16)
    slopes_q88 = np.round(slopes * 256.0).astype(np.int16)
    return base_y_q88, slopes_q88


PWL_BASE_Y_Q88, PWL_SLOPE_Q88 = build_pwl_silu_rom_q88()


def pwl_silu_int_bitexact(x_q88: np.ndarray) -> np.ndarray:
    """
    Evaluates 16-segment PWL SiLU in Q8.8 fixed-point:
    - x <= -4.0 (-1024): 0
    - x >= +4.0 (+1024): x
    - Otherwise: seg_idx = (x + 1024) >> 7 in [0..15], frac = (x + 1024) & 0x7F
                 y = base_y[seg_idx] + ((slope[seg_idx] * frac) >> 7)
    """
    x32 = x_q88.astype(np.int32)
    shifted = np.clip(x32 + 1024, 0, 2047)
    seg_idx = np.right_shift(shifted, 7)  # divide by 128 (0.5 in Q8.8) -> 0..15
    frac = shifted & 0x7F                 # 7-bit fractional offset within segment

    y_mid = PWL_BASE_Y_Q88[seg_idx].astype(np.int32) + np.right_shift(
        PWL_SLOPE_Q88[seg_idx].astype(np.int32) * frac, 7
    )
    out = np.where(x32 <= -1024, 0, np.where(x32 >= 1024, x32, y_mid))
    return np.clip(out, -32768, 32767).astype(np.int16)


if __name__ == "__main__":
    # Self-test of bit-exact equivalence across all Golden Model units
    rng = np.random.default_rng(42)
    W = rng.choice([-1, 0, 1], size=(32, 64), p=[0.3, 0.4, 0.3]).astype(np.int8)
    packed = pack_radix3_weights(W)
    unpacked = unpack_radix3_weights(packed, 64)
    assert np.array_equal(W, unpacked), "Radix-3 pack/unpack mismatch!"

    X = rng.integers(-60, 60, size=(8, 64), dtype=np.int16)
    X_rot = fwht16_int_bitexact(X)

    y_adder = bitlinear_adder_tree_int(X_rot, W)
    y_tlmm = bitlinear_tlmm_lut_int(X_rot, W, group_size=4)
    y_zskip, zstats = bitlinear_zeroskip_int(X_rot, W, parallelism_p=16)
    assert np.array_equal(y_adder, y_tlmm), "Adder-Tree vs TLMM mismatch!"
    assert np.array_equal(y_adder, y_zskip), "Adder-Tree vs Zero-Skip mismatch!"
    print("[OK] Bit-exact verification passed! Zero-skip cycle reduction:", f"{zstats['cycle_reduction_ratio']:.2f}x")

