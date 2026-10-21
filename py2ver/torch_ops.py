# torch_ops.py
"""
PyTorch code in the loop translator (numpy_loops.py).

A torch model is Python like a numpy one: its forward() and the code around it
run at translation time, tensors kept symbolic as the translator's arrays, with
torch's meanings where they differ from numpy's (transpose(d0, d1), max(dim)
giving values and indices, dim / keepdim):

  - calling a module: nn.Sequential calls its children in order; the layers
    below are built from the translator's operations; a module of your own
    runs its forward() as written
      Conv2d          im2col: the padded input copied into its patches' order
                      (sliding_window_view, a view, then a copy), one matrix
                      product with the filters (groups=1, dilation=1)
      BatchNorm1d/2d  in eval mode a scale and a shift per channel, computed
                      from the running statistics when translating; in
                      training mode the batch's mean and variance, the running
                      statistics updated (state)
      MaxPool2d       a maximum over each window (a view)
      Embedding       its weight's rows the indices pick (a gather)
      LayerNorm       the mean and the variance of each row (its last
                      dimensions), the row normalized, scaled and shifted
      GELU            a table of the function (erf's, or tanh's approximation)
      SiLU            a table, as GELU (SwiGLU's gate)
      Linear, ReLU, Flatten, Dropout (eval: nothing; training: a mask from
      the generator in hardware), Identity, Softmax, LogSoftmax, Tanh, Sigmoid
  - matrix products broadcast their leading dimensions (attention's heads)
  - masked_fill(mask, -inf) before a softmax (causal attention): the mask, a
    constant, stays a mask, and the softmax's exponentials are multiplied by
    it, so the masked entries are 0, as torch's, without infinities
  - torch / torch.nn.functional functions: the same operations (relu,
    softmax, log_softmax, cross_entropy, conv2d, linear, max_pool2d, dropout,
    flatten, cat, stack, argmax, exp, log, tanh, sigmoid, sqrt, rsqrt, pow,
    silu, ...); scaled_dot_product_attention as torch's reference (q k^T times
    the scale, the causal mask a constant as masked_fill's, softmax, times v);
    repeat_interleave a view (a dimension of stride 0, merged)
  - parameters and buffers are arrays like numpy attributes: tables, or state
    when changed (a training step)

Autograd (loss.backward()), optimizers and learning-rate schedulers: see
autograd.py and torch_optim.py.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Dict, List, Optional, Tuple

from .numpy_fe import NotTranslatable


def _torch():
    import torch
    return torch


def is_module(v: Any) -> bool:
    import sys
    return "torch" in sys.modules and isinstance(v, _torch().nn.Module)


def is_torch_function(fn: Any) -> bool:
    mod = getattr(fn, "__module__", None) or ""
    return mod == "torch" or mod.startswith("torch.nn.functional") or mod.startswith("torch._C") \
        or (mod.startswith("torch.nn.utils") and getattr(fn, "__name__", "") in UTILS)


UTILS = {"clip_grad_norm_", "clip_grad_norm"}


def _pair(v: Any) -> Tuple[int, int]:
    return (int(v), int(v)) if isinstance(v, int) else (int(v[0]), int(v[1]))


def _dim(d: int, nd: int) -> int:
    return d % nd if nd else 0


class NegInf:
    """x with -inf where mask (a constant) is true: only a softmax takes it."""

    def __init__(self, x: Any, keep: Any) -> None:
        self.x, self.keep = x, keep          # (keep: 1.0 where x is kept, a numpy array)
        self.shape = tuple(x.shape)


def const_values(it, a) -> Any:
    """A constant table's view as a numpy array (its values when translating), or None."""
    np = it.np
    from .numpy_loops import Lazy
    if isinstance(a, Lazy) or not getattr(a, "const", False) or a.name not in it.tables:
        return None
    flat = np.asarray(it.tables[a.name], dtype=float)
    idx = np.zeros(a.shape, dtype=np.int64) + a.offset
    for d, (n, st) in enumerate(zip(a.shape, a.strides)):
        shape = [1] * len(a.shape)
        shape[d] = n
        idx = idx + (np.arange(n) * st).reshape(shape)
    return flat[idx] if a.shape else flat[a.offset]


def masked_fill(it, x, mask, value):
    np = it.np
    if isinstance(value, (int, float)) and math.isinf(value) and value < 0:
        m = const_values(it, it.to_arr(mask)) if not isinstance(mask, np.ndarray) else mask
        if m is None:
            raise NotTranslatable("masked_fill(mask, -inf): the mask must be a constant")
        keep = np.broadcast_to(1.0 - (np.asarray(m) != 0), tuple(np.broadcast_shapes(
            np.shape(m), tuple(x.shape)))).copy()
        return NegInf(x, keep)
    if isinstance(value, (int, float)) and math.isinf(value):
        raise NotTranslatable("masked_fill with +inf is not supported")
    # (a finite value: x where the mask is false, value where it is true)
    return it.elementwise(lambda xv, mv: ast.IfExp(test=mv, body=ast.Constant(float(value)),
                                                   orelse=xv),
                          x, mask, op=("where", float(value)))


# ==================================================================== views

def transpose(a, axes):
    return a.view(tuple(a.shape[x] for x in axes), tuple(a.strides[x] for x in axes))


def unsqueeze(a, d: int):
    nd = len(a.shape) + 1
    d = d % nd
    return a.view(a.shape[:d] + (1,) + a.shape[d:], a.strides[:d] + (0,) + a.strides[d:])


def squeeze(a, d: Optional[int] = None):
    keep = [i for i in range(len(a.shape))
            if not (a.shape[i] == 1 and (d is None or i == d % len(a.shape)))]
    return a.view(tuple(a.shape[i] for i in keep), tuple(a.strides[i] for i in keep))


def windows(it, x, k: Tuple[int, int], stride: Tuple[int, int]):
    """x (N, C, H, W) as its k windows (N, C, Ho, Wo, kh, kw), every stride-th: a view."""
    win = it.numpy("sliding_window_view", [it.to_arr(x), k], {"axis": (2, 3)})
    if stride != (1, 1):
        shape, strides = list(win.shape), list(win.strides)
        for d, s in ((2, stride[0]), (3, stride[1])):
            shape[d] = (shape[d] - 1) // s + 1
            strides[d] *= s
        win = win.view(tuple(shape), tuple(strides))
    return win


# ==================================================================== layers

def conv2d(it, x, w, b, stride=1, padding=0, dilation=1, groups=1):
    if groups != 1 or _pair(dilation) != (1, 1):
        raise NotTranslatable("conv2d: only groups=1, dilation=1 are supported")
    w = it.to_arr(w)
    F, C, kh, kw = w.shape
    x = it.to_arr(x)
    N = x.shape[0]
    if isinstance(padding, str):
        if padding == "valid":
            padding = 0
        elif padding == "same":
            padding = (kh // 2, kw // 2)
        else:
            raise NotTranslatable(f"conv2d: padding={padding!r}")
    ph, pw = _pair(padding)
    if ph or pw:
        x = it.numpy("pad", [x, ((0, 0), (0, 0), (ph, ph), (pw, pw))], {})
    win = windows(it, x, (kh, kw), _pair(stride))               # N, C, Ho, Wo, kh, kw
    Ho, Wo = win.shape[2], win.shape[3]
    # im2col: each output pixel's patch a row (a copy), the filters a matrix
    cols = it.method(transpose(win, (0, 2, 3, 1, 4, 5)), "reshape", [(N * Ho * Wo, C * kh * kw)], {})
    filters = it.method(w, "reshape", [(F, C * kh * kw)], {})
    out = it.matmul(cols, transpose(filters, (1, 0)))         # N * Ho * Wo, F
    if b is not None:
        out = it.binop(ast.Add(), out, b)
    out = it.method(it.to_arr(out), "reshape", [(N, Ho, Wo, F)], {})
    return transpose(out, (0, 3, 1, 2))


def linear(it, x, w, b=None):
    w = it.to_arr(w)
    out = it.matmul(x, transpose(w, (1, 0)) if len(w.shape) == 2 else w)
    return it.binop(ast.Add(), out, b) if b is not None else out


def relu(it, x):
    return it.numpy("maximum", [x, 0.0], {})


def max_pool2d(it, x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False,
               return_indices=False):
    if _pair(padding) != (0, 0) or _pair(dilation) != (1, 1) or ceil_mode or return_indices:
        raise NotTranslatable("max_pool2d: only padding=0, dilation=1 are supported")
    k = _pair(kernel_size)
    win = windows(it, x, k, _pair(stride if stride is not None else kernel_size))
    # (torch's gradient goes to each window's first maximum, not to every element
    # equal to it: its own operation for autograd)
    was = it.recording
    it.recording = False
    out = it.reduce(win, (4, 5), False, "max")
    it.recording = was
    it.record("maxpool", out, win)
    return out


def softmax(it, x, dim):
    if isinstance(x, NegInf):
        # exp((x - max) * keep) * keep / sum: the masked entries 0, as -inf's would be.
        # The maximum is the kept entries' (the masked ones replaced by a column kept
        # in every row: causal attention's first), and the exponential's argument 0
        # where masked, so the exp table covers only what is kept
        np = it.np
        keep = x.keep
        x = it.to_arr(x.x)
        nd = len(x.shape)
        d = dim % nd
        moved = np.moveaxis(keep, d, -1).reshape(-1, keep.shape[d])
        common = np.flatnonzero(moved.all(axis=0))
        if common.size:
            j = int(common[0])
            idx = [slice(None)] * nd
            idx[d] = slice(j, j + 1)
            shape, strides = list(x.shape), list(x.strides)
            shape[d], strides[d] = 1, 0
            col = x.view(tuple(shape), tuple(strides), x.offset + j * x.strides[d])
            kept = it.binop(ast.Add(), it.binop(ast.Mult(), x, keep),
                            it.binop(ast.Mult(), col, 1.0 - keep))
            m = it.reduce(it.to_arr(kept), dim, True, "max")
        else:
            m = it.reduce(x, dim, True, "max")
        arg = it.binop(ast.Mult(), it.binop(ast.Sub(), x, m), keep)
        e = it.binop(ast.Mult(), it.numpy("exp", [arg], {}), keep)
        return it.binop(ast.Div(), e, it.reduce(it.to_arr(e), dim, True, "sum"))
    x = it.to_arr(x)
    m = it.reduce(x, dim, True, "max")
    e = it.numpy("exp", [it.binop(ast.Sub(), x, m)], {})
    return it.binop(ast.Div(), e, it.reduce(it.to_arr(e), dim, True, "sum"))


def log_softmax(it, x, dim):
    if isinstance(x, NegInf):
        raise NotTranslatable("log_softmax of -inf values is not supported")
    x = it.to_arr(x)
    z = it.binop(ast.Sub(), x, it.reduce(x, dim, True, "max"))
    s = it.reduce(it.to_arr(it.numpy("exp", [z], {})), dim, True, "sum")
    return it.binop(ast.Sub(), z, it.numpy("log", [s], {}))


def cross_entropy(it, logits, target, reduction="mean", **kw):
    """-log_softmax(logits)[i, target[i]], averaged (a target of class indices)."""
    for k, v in kw.items():
        if k == "weight" and v is None or k == "ignore_index" and v == -100 \
                or k == "label_smoothing" and v == 0.0 or k in ("size_average", "reduce") and v is None:
            continue
        raise NotTranslatable(f"cross_entropy: {k}={v!r} is not supported")
    lp = it.to_arr(log_softmax(it, logits, 1))
    np = it.np
    rows = np.arange(lp.shape[0])
    picked = it.fancy(lp, [rows, target], False)
    nll = it.elementwise(lambda v: ast.UnaryOp(op=ast.USub(), operand=v), picked, op=("neg",))
    if reduction == "none":
        return nll
    return it.reduce(it.to_arr(nll), None, False, "mean" if reduction == "mean" else "sum")


def dropout(it, x, p, training):
    if not training or p == 0:
        return x
    u = it.random(None, [tuple(x.shape)], {})          # (the generator in hardware)
    keep = it.elementwise(lambda v: ast.Compare(left=v, ops=[ast.Gt()],
                                                comparators=[ast.Constant(float(p))]), u,
                          base="m", op=("cmp",))
    mask = it.binop(ast.Mult(), keep, 1.0 / (1.0 - p))
    return it.binop(ast.Mult(), x, mask)


def flatten(it, x, start_dim=0, end_dim=-1):
    nd = len(x.shape)
    s, e = _dim(start_dim, nd), _dim(end_dim, nd)
    shape = x.shape[:s] + (math.prod(x.shape[s:e + 1]),) + x.shape[e + 1:]
    return it.method(it.to_arr(x), "reshape", [shape], {})


def batch_norm(it, mod, x):
    """BatchNorm1d / 2d: x of (N, C) or (N, C, H, W)."""
    np = it.np
    C = x.shape[1]
    extra = (1,) * (len(x.shape) - 2)
    if not mod.training and mod.track_running_stats:
        # eval: a scale and a shift per channel (constants)
        from .numpy_loops import _values
        rm, rv = _values(mod.running_mean), _values(mod.running_var)
        w = _values(mod.weight) if mod.weight is not None else np.ones(C)
        b = _values(mod.bias) if mod.bias is not None else np.zeros(C)
        scale = w / np.sqrt(rv + mod.eps)
        shift = b - rm * scale
        y = it.binop(ast.Mult(), x, scale.reshape((C,) + extra))
        return it.binop(ast.Add(), y, shift.reshape((C,) + extra))
    from .torch_optim import batch_norm_train
    return batch_norm_train(it, mod, x)


def clip_grad_norm(it, params, max_norm, norm_type=2.0, error_if_nonfinite=False):
    """torch.nn.utils.clip_grad_norm_: the gradients' total 2-norm, each gradient
    scaled by min(1, max_norm / (norm + 1e-6)), as torch does."""
    from .numpy_loops import Sym
    if float(norm_type) != 2.0:
        raise NotTranslatable("clip_grad_norm_: only the 2-norm is supported")
    if _torch().is_tensor(params):
        params = [params]
    grads = []
    for p in list(params):
        g = it.grads.get(it.adopt(p).name)
        if g is not None:
            grads.append(g)
    if not grads:
        return 0.0
    total = None
    for g in grads:
        s = it.reduce(it.to_arr(it.binop(ast.Mult(), g, g)), None, False, "sum")
        total = s if total is None else it.binop(ast.Add(), total, s)
    norm = it.rename(it.numpy("sqrt", [it.rename(total, "gn2")], {}), "gnorm")
    coef = it.binop(ast.Div(), float(max_norm), it.binop(ast.Add(), norm, 1e-6))
    coef = it.rename(it.elementwise(lambda c: ast.Call(func=ast.Name(id="min", ctx=ast.Load()),
                                                       args=[c, ast.Constant(1.0)], keywords=[]),
                                    coef, op=("min",)), "clip")
    for g in grads:
        it.copy_into(g, it.binop(ast.Mult(), g, coef))
    return norm


def embedding(it, idx, weight):
    w = it.to_arr(weight)
    return it.fancy(w, [idx], False)


def layer_norm(it, x, shape, weight=None, bias=None, eps=1e-5):
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    x = it.to_arr(x)
    nd = len(x.shape)
    axes = tuple(range(nd - len(shape), nd))
    mean = it.reduce(x, axes, True, "mean")
    d = it.binop(ast.Sub(), x, mean)
    var = it.reduce(it.to_arr(it.binop(ast.Mult(), d, d)), axes, True, "mean")
    inv = it.to_arr(it.binop(ast.Div(), 1.0, it.numpy("sqrt", [it.binop(ast.Add(), var, float(eps))],
                                                      {})))
    y = it.binop(ast.Mult(), d, inv)
    if weight is not None:
        y = it.binop(ast.Mult(), y, weight)
    if bias is not None:
        y = it.binop(ast.Add(), y, bias)
    return y


def group_norm(it, x, groups, weight=None, bias=None, eps=1e-5):
    """GroupNorm: each sample's channels in `groups` groups, each group (its
    channels and positions) normalized as layer_norm's rows, then a scale and a
    shift per channel."""
    x = it.to_arr(x)
    N, C, rest = x.shape[0], x.shape[1], tuple(x.shape[2:])
    if C % groups:
        raise NotTranslatable(f"group_norm: {C} channels don't divide into {groups} groups")
    rows = it.method(x, "reshape", [(N, groups, (C // groups) * math.prod(rest))], {})
    y = it.method(it.to_arr(layer_norm(it, rows, (rows.shape[-1],), eps=eps)), "reshape",
                  [tuple(x.shape)], {})
    per_channel = (C,) + (1,) * len(rest)
    if weight is not None:
        y = it.binop(ast.Mult(), y, it.method(it.to_arr(weight), "reshape", [per_channel], {}))
    if bias is not None:
        y = it.binop(ast.Add(), y, it.method(it.to_arr(bias), "reshape", [per_channel], {}))
    return y


def gelu(it, x, approximate="none"):
    """GELU as a table of the function (with its derivative, for autograd)."""
    from .numpy_fe import TABLES_KEY
    name = GELU_TANH if approximate == "tanh" else GELU
    it.ns[name] = _GELU_FUNCTIONS[name]
    it.ns[name + "_grad"] = _GELU_FUNCTIONS[name + "_grad"]
    it.ns.setdefault(TABLES_KEY, set()).update({name, name + "_grad"})
    return it.elementwise(lambda v: ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=[v],
                                             keywords=[]), x, op=("table", name))


def silu(it, x):
    """SiLU (x * sigmoid(x), SwiGLU's gate) as a table of the function, as GELU."""
    from .numpy_fe import TABLES_KEY
    it.ns[SILU] = _silu
    it.ns[SILU + "_grad"] = _silu_grad
    it.ns.setdefault(TABLES_KEY, set()).update({SILU, SILU + "_grad"})
    return it.elementwise(lambda v: ast.Call(func=ast.Name(id=SILU, ctx=ast.Load()), args=[v],
                                             keywords=[]), x, op=("table", SILU))


def rsqrt(it, x):
    return it.binop(ast.Div(), 1.0, it.numpy("sqrt", [x], {}))


def power(it, x, exponent):
    """x ** exponent: a whole power as products, 0.5 / -0.5 as sqrt / rsqrt."""
    if not isinstance(exponent, (int, float)) or isinstance(exponent, bool):
        raise NotTranslatable("pow: only a number as the exponent is supported")
    if exponent in (0.5, -0.5):
        return it.numpy("sqrt", [x], {}) if exponent > 0 else rsqrt(it, x)
    if not (isinstance(exponent, int) or float(exponent).is_integer()) or int(exponent) < 1:
        raise NotTranslatable(f"pow({exponent!r}): only a whole power, 0.5 or -0.5 is supported")
    out = x
    for _ in range(int(exponent) - 1):
        out = it.binop(ast.Mult(), out, x)
    return out


def stack(it, tensors, dim=0):
    """Each tensor with a new dimension dim, then concatenated along it."""
    arrs = [it.to_arr(t) for t in tensors]
    nd = len(arrs[0].shape) + 1
    return it.numpy("concatenate", [[unsqueeze(a, dim % nd) for a in arrs]], {"axis": dim % nd})


def repeat_interleave(it, a, repeats, dim=None):
    """Each element along dim `repeats` times in a row (a view: a new dimension
    of stride 0 after dim, then merged with it)."""
    if not isinstance(repeats, int) or dim is None:
        raise NotTranslatable("repeat_interleave: only a whole number of repeats along a dim")
    a = it.to_arr(a)
    nd = len(a.shape)
    d = _dim(dim, nd)
    v = a.view(a.shape[:d + 1] + (repeats,) + a.shape[d + 1:],
               a.strides[:d + 1] + (0,) + a.strides[d + 1:])
    return flatten(it, v, d, d + 1)


def interpolate(it, x, size=None, scale_factor=None, mode="nearest", align_corners=None,
                recompute_scale_factor=None, antialias=False):
    """F.interpolate, mode 'nearest' by a whole factor: each element repeated
    along each spatial dimension (output i reads input i // factor, as torch's)."""
    x = it.to_arr(x)
    spatial = len(x.shape) - 2
    if mode != "nearest" or antialias:
        raise NotTranslatable(f"interpolate: only mode='nearest' is supported (not {mode!r})")
    if size is not None:
        size = (size,) * spatial if isinstance(size, int) else tuple(size)
        factors = [s // n for s, n in zip(size, x.shape[2:])]
        if any(s != f * n for s, f, n in zip(size, factors, x.shape[2:])):
            raise NotTranslatable("interpolate: only a size a whole multiple of the input's")
    else:
        sf = (scale_factor,) * spatial if isinstance(scale_factor, (int, float)) \
            else tuple(scale_factor)
        if any(float(f) != int(f) or f < 1 for f in sf):
            raise NotTranslatable("interpolate: only a whole scale_factor is supported")
        factors = [int(f) for f in sf]
    for d, f in enumerate(factors):
        if f > 1:
            x = it.to_arr(repeat_interleave(it, x, f, d + 2))
    return x


def scaled_dot_product_attention(it, q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False,
                                 scale=None, enable_gqa=False):
    """softmax(q k^T * scale) v, as torch's reference: the scale 1 / sqrt(E) unless
    given; is_causal: the mask torch makes (ones(L, S).tril()), a constant, as
    masked_fill(-inf)'s."""
    np = it.np
    if attn_mask is not None or enable_gqa:
        raise NotTranslatable("scaled_dot_product_attention: attn_mask / enable_gqa are not "
                              "supported")
    q, k = it.to_arr(q), it.to_arr(k)
    nd = len(k.shape)
    L, S, E = q.shape[-2], k.shape[-2], q.shape[-1]
    axes = list(range(nd))
    axes[-2], axes[-1] = axes[-1], axes[-2]
    w = it.binop(ast.Mult(), it.matmul(q, transpose(k, axes)),
                 float(scale) if scale is not None else 1.0 / math.sqrt(E))
    if is_causal:
        keep = np.broadcast_to(np.tril(np.ones((L, S))), tuple(w.shape)).copy()
        w = NegInf(w, keep)
    w = softmax(it, w, -1)
    w = dropout(it, w, dropout_p, True)
    return it.matmul(w, v)


def _silu(x):
    import numpy as np
    e = np.exp if not isinstance(x, (int, float)) else math.exp
    return x / (1.0 + e(-x))


def _silu_grad(x):
    import numpy as np
    e = np.exp if not isinstance(x, (int, float)) else math.exp
    s = 1.0 / (1.0 + e(-x))
    return s * (1.0 + x * (1.0 - s))


_silu.__py2ver_vectorized__ = True
_silu_grad.__py2ver_vectorized__ = True
SILU = "__py2ver_silu"
GELU = "__py2ver_gelu"
GELU_TANH = "__py2ver_gelu_tanh"


def _erf(x: Any) -> Any:
    import numpy as np
    if isinstance(x, np.ndarray):
        return np.vectorize(math.erf, otypes=[float])(x)
    return math.erf(x)


def _gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def _gelu_grad(x):
    import numpy as np
    e = np.exp if not isinstance(x, (int, float)) else math.exp
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0))) + x * e(-0.5 * x * x) / math.sqrt(2 * math.pi)


def _gelu_tanh(x):
    import numpy as np
    t = np.tanh if not isinstance(x, (int, float)) else math.tanh
    return 0.5 * x * (1.0 + t(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


def _gelu_tanh_grad(x):
    import numpy as np
    t = np.tanh if not isinstance(x, (int, float)) else math.tanh
    c = math.sqrt(2.0 / math.pi)
    u = c * (x + 0.044715 * x * x * x)
    th = t(u)
    return 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * c * (1.0 + 3 * 0.044715 * x * x)


for _f in (_gelu, _gelu_grad, _gelu_tanh, _gelu_tanh_grad):
    _f.__py2ver_vectorized__ = True          # (takes numpy arrays: vectorize.py may call it so)
_GELU_FUNCTIONS = {GELU: _gelu, GELU + "_grad": _gelu_grad,
                   GELU_TANH: _gelu_tanh, GELU_TANH + "_grad": _gelu_tanh_grad}


# ==================================================================== calls

_ACTIVATIONS = {"Tanh": "tanh", "Sigmoid": "sigmoid"}


def module_call(it, mod, args: List[Any], kwargs: Dict[str, Any]) -> Any:
    """Calling a torch module (its __call__): a layer, or its own forward()."""
    torch = _torch()
    nn = torch.nn
    cls = type(mod)
    if cls.__module__.startswith("torch.nn"):
        name = cls.__name__
        x = args[0] if args else None
        if isinstance(mod, nn.Sequential):
            for m in mod:
                x = it.call_value(m, [x], {})
            return x
        if name == "Conv2d":
            if mod.padding_mode != "zeros":
                raise NotTranslatable("Conv2d: only padding_mode='zeros' is supported")
            return conv2d(it, x, it.param(mod, "weight"), it.param(mod, "bias"),
                          mod.stride, mod.padding, mod.dilation, mod.groups)
        if name == "Linear":
            return linear(it, x, it.param(mod, "weight"), it.param(mod, "bias"))
        if name in ("BatchNorm1d", "BatchNorm2d"):
            return batch_norm(it, mod, x)
        if name == "ReLU":
            return relu(it, x)
        if name == "MaxPool2d":
            return max_pool2d(it, x, mod.kernel_size, mod.stride, mod.padding, mod.dilation,
                              mod.ceil_mode, mod.return_indices)
        if name == "Flatten":
            return flatten(it, x, mod.start_dim, mod.end_dim)
        if name == "Dropout":
            return dropout(it, x, mod.p, mod.training)
        if name == "Identity":
            return x
        if name == "Embedding":
            if mod.padding_idx is not None or mod.max_norm is not None:
                raise NotTranslatable("Embedding(padding_idx / max_norm) is not supported")
            return embedding(it, x, it.param(mod, "weight"))
        if name == "LayerNorm":
            return layer_norm(it, x, mod.normalized_shape, it.param(mod, "weight"),
                              it.param(mod, "bias") if mod.elementwise_affine else None,
                              mod.eps)
        if name == "GELU":
            return gelu(it, x, mod.approximate)
        if name == "GroupNorm":
            return group_norm(it, x, mod.num_groups,
                              it.param(mod, "weight") if mod.affine else None,
                              it.param(mod, "bias") if mod.affine else None, mod.eps)
        if name == "SiLU":
            return silu(it, x)
        if name == "Softmax":
            return softmax(it, x, mod.dim)
        if name == "LogSoftmax":
            return log_softmax(it, x, mod.dim)
        if name in _ACTIVATIONS:
            return it.numpy(_ACTIVATIONS[name], [x], {})
        raise NotTranslatable(f"The torch module {name} is not supported")
    # A module of your own: its forward, as written
    return it.call(cls.forward, args, kwargs, obj=mod)


def function(it, fn, args: List[Any], kwargs: Dict[str, Any]) -> Any:
    """A torch / torch.nn.functional function on arrays computed in hardware."""
    name = getattr(fn, "__name__", "")

    def arg(i, key, default=None):
        return args[i] if len(args) > i else kwargs.get(key, default)
    if name in UTILS:
        return clip_grad_norm(it, arg(0, "parameters"), arg(1, "max_norm"), arg(2, "norm_type", 2.0),
                              arg(3, "error_if_nonfinite", False))
    if name in ("relu", "relu_"):
        return relu(it, args[0])
    if name == "gelu":
        return gelu(it, args[0], kwargs.get("approximate", "none"))
    if name == "silu":
        return silu(it, args[0])
    if name == "rsqrt":
        return rsqrt(it, args[0])
    if name == "pow":
        return power(it, args[0], arg(1, "exponent"))
    if name == "stack":
        return stack(it, list(arg(0, "tensors")), arg(1, "dim", 0))
    if name == "repeat_interleave":
        return repeat_interleave(it, args[0], arg(1, "repeats"), arg(2, "dim"))
    if name == "scaled_dot_product_attention":
        return scaled_dot_product_attention(
            it, args[0], arg(1, "key"), arg(2, "value"), arg(3, "attn_mask"),
            arg(4, "dropout_p", 0.0), arg(5, "is_causal", False), kwargs.get("scale"),
            kwargs.get("enable_gqa", False))
    if name == "layer_norm":
        return layer_norm(it, args[0], arg(1, "normalized_shape"), arg(2, "weight"),
                          arg(3, "bias"), arg(4, "eps", 1e-5))
    if name == "interpolate":
        return interpolate(it, args[0], arg(1, "size"), arg(2, "scale_factor"),
                           arg(3, "mode", "nearest"), arg(4, "align_corners"),
                           arg(5, "recompute_scale_factor"), arg(6, "antialias", False))
    if name == "group_norm":
        return group_norm(it, args[0], arg(1, "num_groups"), arg(2, "weight"), arg(3, "bias"),
                          arg(4, "eps", 1e-5))
    if name == "embedding":
        return embedding(it, args[0], arg(1, "weight"))
    if name == "masked_fill":
        return masked_fill(it, args[0], arg(1, "mask"), arg(2, "value"))
    if name == "softmax":
        return softmax(it, args[0], arg(1, "dim"))
    if name == "log_softmax":
        return log_softmax(it, args[0], arg(1, "dim"))
    if name == "cross_entropy":
        rest = {k: v for k, v in kwargs.items() if k not in ("input", "target", "reduction")}
        return cross_entropy(it, args[0], arg(1, "target"), arg(5, "reduction", "mean"), **rest)
    if name == "conv2d":
        return conv2d(it, args[0], arg(1, "weight"), arg(2, "bias"), arg(3, "stride", 1),
                      arg(4, "padding", 0), arg(5, "dilation", 1), arg(6, "groups", 1))
    if name == "linear":
        return linear(it, args[0], arg(1, "weight"), arg(2, "bias"))
    if name == "max_pool2d":
        return max_pool2d(it, args[0], arg(1, "kernel_size"), arg(2, "stride"),
                          arg(3, "padding", 0), arg(4, "dilation", 1), arg(5, "ceil_mode", False),
                          arg(6, "return_indices", False))
    if name == "dropout":
        return dropout(it, args[0], arg(1, "p", 0.5), arg(2, "training", True))
    if name == "flatten":
        return flatten(it, args[0], arg(1, "start_dim", 0), arg(2, "end_dim", -1))
    if name == "cat":
        return it.numpy("concatenate", [list(args[0])], {"axis": arg(1, "dim", 0)})
    if name in ("argmax", "sum", "mean", "max", "amax"):
        return tensor_method(it, it.to_arr(args[0]), name, list(args[1:]), kwargs)
    if name in ("exp", "log", "tanh", "sigmoid", "sqrt", "abs", "maximum", "minimum", "sin",
                "cos"):
        return it.numpy(name, list(args), kwargs)
    if name in ("matmul", "mm"):
        return it.matmul(args[0], args[1])
    if name in ("flatten", "reshape"):
        return it.method(it.to_arr(args[0]), "reshape", [arg(1, "shape")], {})
    raise NotTranslatable(f"torch.{name} is not supported")


def tensor_method(it, a, name: str, args: List[Any], kwargs: Dict[str, Any]) -> Any:
    """A tensor's method, with torch's meaning; NotImplemented: as numpy's."""
    from .numpy_loops import Sym
    if name == "backward":
        if args or kwargs:
            raise NotTranslatable("backward(gradient) is not supported")
        from .autograd import backward
        backward(it, a)
        return None
    if isinstance(a, Sym):                             # (a single value)
        if name in ("item", "detach", "to", "cpu", "float", "clone", "squeeze"):
            return Sym(a.expr) if name in ("item", "detach") else a
        raise NotTranslatable(f"A single value's .{name}() is not supported")
    nd = len(a.shape)

    def arg(i, key, default=None):
        return args[i] if len(args) > i else kwargs.get(key, default)
    if name in ("view", "reshape"):
        shape = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else tuple(args)
        return it.method(it.to_arr(a), "reshape", [tuple(shape)], {})
    if name == "flatten":
        return flatten(it, a, arg(0, "start_dim", 0), arg(1, "end_dim", -1))
    if name == "transpose":
        d0, d1 = _dim(args[0], nd), _dim(args[1], nd)
        axes = list(range(nd))
        axes[d0], axes[d1] = axes[d1], axes[d0]
        return transpose(it.to_arr(a), axes)
    if name == "permute":
        dims = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list)) else args
        return transpose(it.to_arr(a), [_dim(d, nd) for d in dims])
    if name == "t":
        return transpose(it.to_arr(a), (1, 0))
    if name == "unsqueeze":
        return unsqueeze(it.to_arr(a), arg(0, "dim"))
    if name == "squeeze":
        return squeeze(it.to_arr(a), arg(0, "dim"))
    if name in ("to", "cpu", "cuda", "float", "detach", "contiguous", "requires_grad_",
                "type_as"):
        return a
    if name == "pow":
        return power(it, it.to_arr(a), arg(0, "exponent"))
    if name == "rsqrt":
        return rsqrt(it, it.to_arr(a))
    if name == "repeat_interleave":
        return repeat_interleave(it, a, arg(0, "repeats"), arg(1, "dim"))
    if name == "clone":
        return it.method(it.to_arr(a), "copy", [], {})
    if name in ("split", "chunk"):
        dim = _dim(arg(1, "dim", 0), nd)
        n = a.shape[dim]
        if name == "chunk":
            size = -(-n // int(args[0]))
            sizes = [size] * (n // size) + ([n % size] if n % size else [])
        elif isinstance(args[0], int):
            sizes = [args[0]] * (n // args[0]) + ([n % args[0]] if n % args[0] else [])
        else:
            sizes = list(args[0])
        a = it.to_arr(a)
        out, at = [], 0
        for k in sizes:
            shape = list(a.shape)
            shape[dim] = k
            out.append(a.view(tuple(shape), a.strides, a.offset + at * a.strides[dim]))
            at += k
        return tuple(out)
    if name == "unbind":                               # (each slice along dim: a view)
        dim = _dim(arg(0, "dim", 0), nd)
        a = it.to_arr(a)
        shape = tuple(a.shape[:dim] + a.shape[dim + 1:])
        strides = tuple(a.strides[:dim] + a.strides[dim + 1:])
        return tuple(a.view(shape, strides, a.offset + i * a.strides[dim])
                     for i in range(a.shape[dim]))
    if name == "masked_fill":
        return masked_fill(it, a, arg(0, "mask"), arg(1, "value"))
    if name == "size":
        d = arg(0, "dim")
        return tuple(a.shape) if d is None else a.shape[_dim(d, nd)]
    if name == "dim":
        return nd
    if name == "numel":
        return math.prod(a.shape)
    if name in ("sum", "mean", "argmax", "max", "amax", "min"):
        dim = arg(0, "dim")
        keep = bool(arg(1, "keepdim", False))
        if name == "max" and dim is not None:          # (values, indices)
            return (it.reduce(it.to_arr(a), dim, keep, "max"),
                    it.reduce(it.to_arr(a), dim, keep, "argmax"))
        if name == "min" and dim is not None:
            raise NotTranslatable("min(dim) is not supported")
        return it.method(it.to_arr(a), "max" if name == "amax" else name,
                         [], {"axis": dim, "keepdims": keep})
    if name == "softmax":
        return softmax(it, a, arg(0, "dim"))
    if name == "log_softmax":
        return log_softmax(it, a, arg(0, "dim"))
    if name in ("relu", "relu_"):
        return relu(it, a)
    if name in ("exp", "log", "tanh", "sigmoid", "sqrt", "abs", "sin", "cos"):
        return it.numpy(name, [a], {})
    if name in ("add", "sub", "mul", "div"):
        op = {"add": ast.Add(), "sub": ast.Sub(), "mul": ast.Mult(), "div": ast.Div()}[name]
        return it.binop(op, a, args[0])
    if name in ("matmul", "mm"):
        return it.matmul(a, args[0])
    if name == "item":
        from .numpy_loops import Sym
        if math.prod(a.shape) != 1:
            raise NotTranslatable(".item() of an array of more than one value")
        return Sym(it.to_arr(a).at([ast.Constant(0)] * len(a.shape)))
    return NotImplemented
