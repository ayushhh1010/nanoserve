"""INT8 weight-only quantization: round-to-nearest, symmetric, group size 128.

Weights are stored as int8 with one fp32 scale per group of 128 input channels,
and dequantized to the activation dtype inside the forward pass. Activations
stay in bf16 -- this is W8A16, not W8A8.

**This is a memory optimisation, not a compute one, and saying otherwise would
be dishonest.** A real INT8 speedup needs a kernel that multiplies int8 by int8
and accumulates in int32. PyTorch has no such path for this shape without
writing one, and this project deliberately dropped GPU-kernel work, so the
weights are dequantized before the matmul. The arithmetic is identical to bf16;
what changes is that the stored weights are half the size.

Three choices, each from published practice rather than taste:

**Group size 128, symmetric.** Group sizes of 64-128 are the documented
quality/latency sweet spot for weight-only quantization. Symmetric (zero-point
fixed at 0) halves the bookkeeping and costs almost nothing at 8 bits, where
the range is wide enough that an asymmetric offset buys little.

**Round-to-nearest, no calibration.** RTN is the recommended algorithm at INT8:
at 8 bits the quantization grid is fine enough that data-aware methods like AWQ
or GPTQ have almost nothing left to recover. They earn their complexity at 4
bits and below.

**The LM head is not quantized.** Standard practice, and here it is not
optional: this model ties its output head to its input embedding, so they are
one tensor. Quantizing it would quantize every token lookup as well, and an
embedding table is a gather -- there is no matmul to accelerate and nothing to
gain.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

#: Documented sweet spot for weight-only quantization.
DEFAULT_GROUP_SIZE = 128

#: int8 range. Symmetric quantization uses -127..127 rather than -128..127 so
#: the grid is centred: keeping -128 would make the negative side one step
#: wider than the positive one and bias every quantized tensor slightly low.
QMAX = 127


def quantize_tensor(
    weight: Tensor, group_size: int = DEFAULT_GROUP_SIZE
) -> tuple[Tensor, Tensor]:
    """(int8 weights, fp32 scales) for a 2-D weight of shape (out, in).

    Grouping runs along the *input* dimension, which is the one summed over in
    the matmul. Each group of 128 input channels gets its own scale, so a
    single large channel only inflates the scale of its own group rather than
    crushing the resolution of the whole row.
    """
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} is not divisible by group_size={group_size}"
        )

    w = weight.detach().to(torch.float32)
    grouped = w.reshape(out_features, in_features // group_size, group_size)

    # Symmetric: the scale is set by the largest magnitude in the group.
    absmax = grouped.abs().amax(dim=-1, keepdim=True)
    # A group of exact zeros would divide by zero; its quantized values are
    # zero regardless, so any positive scale is correct.
    scales = torch.where(absmax > 0, absmax / QMAX, torch.ones_like(absmax))

    q = torch.round(grouped / scales).clamp(-QMAX, QMAX).to(torch.int8)
    return q.reshape(out_features, in_features), scales.squeeze(-1)


def dequantize_tensor(q: Tensor, scales: Tensor, group_size: int) -> Tensor:
    """Reconstruct an approximate float weight from int8 + scales."""
    out_features, in_features = q.shape
    grouped = q.reshape(out_features, in_features // group_size, group_size).to(scales.dtype)
    return (grouped * scales.unsqueeze(-1)).reshape(out_features, in_features)


class QuantizedLinear(nn.Module):
    """A Linear whose weights live as int8 and are dequantized per forward.

    Drop-in for `nn.Linear` with `bias=False`, which is every linear in this
    model. Bias is supported anyway so the class is not quietly wrong if one
    appears later.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        group_size: int = DEFAULT_GROUP_SIZE,
        bias: bool = False,
        device=None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.compute_dtype = dtype

        self.register_buffer(
            "qweight", torch.zeros((out_features, in_features), dtype=torch.int8, device=device)
        )
        self.register_buffer(
            "scales",
            torch.ones((out_features, in_features // group_size), dtype=torch.float32, device=device),
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype)) if bias else None
        )

    @classmethod
    def from_linear(
        cls, linear: nn.Linear, group_size: int = DEFAULT_GROUP_SIZE
    ) -> QuantizedLinear:
        q, scales = quantize_tensor(linear.weight, group_size)
        out = cls(
            linear.in_features, linear.out_features, group_size,
            bias=linear.bias is not None,
            device=linear.weight.device, dtype=linear.weight.dtype,
        )
        out.qweight.copy_(q)
        out.scales.copy_(scales)
        if linear.bias is not None:
            out.bias.data.copy_(linear.bias.data)
        return out

    def dequantized_weight(self) -> Tensor:
        return dequantize_tensor(
            self.qweight, self.scales.to(torch.float32), self.group_size
        ).to(self.compute_dtype)

    def forward(self, x: Tensor) -> Tensor:
        # Dequantize then matmul. Not free, and not pretending to be: without
        # an int8 GEMM this buys storage, not speed.
        return torch.nn.functional.linear(x, self.dequantized_weight(), self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"group_size={self.group_size}, int8"
        )


@dataclass
class QuantStats:
    layers_quantized: int
    layers_skipped: int
    original_bytes: int
    quantized_bytes: int

    @property
    def compression(self) -> float:
        return self.original_bytes / self.quantized_bytes if self.quantized_bytes else 1.0

    @property
    def saved_bytes(self) -> int:
        return self.original_bytes - self.quantized_bytes

    def to_dict(self) -> dict:
        mb = 1024 * 1024
        return {
            "layers_quantized": self.layers_quantized,
            "layers_skipped": self.layers_skipped,
            "original_mb": round(self.original_bytes / mb, 2),
            "quantized_mb": round(self.quantized_bytes / mb, 2),
            "saved_mb": round(self.saved_bytes / mb, 2),
            "compression": round(self.compression, 3),
        }


def quantize_model(
    model: nn.Module,
    group_size: int = DEFAULT_GROUP_SIZE,
    skip: tuple[str, ...] = ("lm_head",),
) -> QuantStats:
    """Replace eligible `nn.Linear` modules with int8 equivalents, in place.

    `skip` names modules to leave alone. `lm_head` is excluded by default: it
    is standard practice, and in this model it is tied to the embedding table,
    so quantizing it would quantize every token lookup -- a gather, with no
    matmul to accelerate.
    """
    original = quantized = 0
    n_quant = n_skip = 0

    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{child_name}" if name else child_name

            if any(s in full for s in skip):
                n_skip += 1
                original += child.weight.numel() * child.weight.element_size()
                quantized += child.weight.numel() * child.weight.element_size()
                continue
            if child.in_features % group_size != 0:
                # Leave it rather than silently changing the grouping: a layer
                # quantized at a different group size than reported is worse
                # than one not quantized at all.
                n_skip += 1
                original += child.weight.numel() * child.weight.element_size()
                quantized += child.weight.numel() * child.weight.element_size()
                continue

            original += child.weight.numel() * child.weight.element_size()
            q = QuantizedLinear.from_linear(child, group_size)
            quantized += q.qweight.numel() + q.scales.numel() * 4
            setattr(module, child_name, q)
            n_quant += 1

    return QuantStats(n_quant, n_skip, original, quantized)


def quantization_error(weight: Tensor, group_size: int = DEFAULT_GROUP_SIZE) -> dict:
    """How much a single tensor loses. Useful for locating a bad layer."""
    q, scales = quantize_tensor(weight, group_size)
    recon = dequantize_tensor(q, scales, group_size)
    w = weight.detach().to(torch.float32)
    err = (w - recon).abs()
    return {
        "max_abs_error": float(err.max()),
        "mean_abs_error": float(err.mean()),
        "relative_error": float(err.norm() / w.norm()) if w.norm() > 0 else 0.0,
    }
