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
      Linear, ReLU, Flatten, Dropout (eval: nothing; training: a mask from
      the generator in hardware), Identity, Softmax, LogSoftmax, Tanh, Sigmoid
  - torch / torch.nn.functional functions: the same operations (relu,
    softmax, log_softmax, cross_entropy, conv2d, linear, max_pool2d, dropout,
    flatten, cat, argmax, exp, log, tanh, sigmoid, sqrt, ...)
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
    return mod == "torch" or mod.startswith("torch.nn.functional") or mod.startswith("torch._C")


def _pair(v: Any) -> Tuple[int, int]:
    return (int(v), int(v)) if isinstance(v, int) else (int(v[0]), int(v[1]))


def _dim(d: int, nd: int) -> int:
    return d % nd if nd else 0


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
    x = it.to_arr(x)
    m = it.reduce(x, dim, True, "max")
    e = it.numpy("exp", [it.binop(ast.Sub(), x, m)], {})
    return it.binop(ast.Div(), e, it.reduce(it.to_arr(e), dim, True, "sum"))


def log_softmax(it, x, dim):
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
    if name in ("relu", "relu_"):
        return relu(it, args[0])
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
    if name in ("exp", "log", "tanh", "sigmoid", "sqrt", "abs", "maximum", "minimum"):
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
    if name in ("to", "cpu", "cuda", "float", "detach", "contiguous", "requires_grad_"):
        return a
    if name == "clone":
        return it.method(it.to_arr(a), "copy", [], {})
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
    if name in ("exp", "log", "tanh", "sigmoid", "sqrt", "abs"):
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
