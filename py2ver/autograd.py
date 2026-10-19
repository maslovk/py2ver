# autograd.py
"""
loss.backward() in the loop translator: reverse-mode gradients, as loops.

While a torch method runs (numpy_loops.py), each operation on values that
depend on a parameter (requires_grad) is recorded: elementwise operations
(with what they compute: add, mul, exp, ...), reductions, matrix products,
copies into lists (views, im2col, padding) and gathers (a[rows, labels]).
backward() goes through them in reverse and writes the gradient code:

  - a list's gradient is a list of its length: a view's contribution (a
    transpose, a window of a convolution, a broadcast) is added at the view's
    positions, element by element, so overlapping windows add up as they do
    in col2im; along a dimension of stride 0 (an element repeated, as by
    repeat_interleave) the contributions are summed first
  - an operand broadcast in an operation gets the sum of its contributions
    over the broadcast dimensions
  - elementwise: the derivative of each operation (relu's, as torch's, is 0
    at 0), from the operands' values; max over a window: the elements equal to
    the maximum; a matrix product: the two products with the other operand;
    a gather: added back at the indices

Parameters' gradients are then their .grad (torch.optim reads them, see
torch_optim.py).
"""

from __future__ import annotations

import ast
import math
from typing import Any, Dict, Optional, Tuple

from .numpy_fe import NotTranslatable


def _bin(op: ast.operator):
    return lambda x, y: ast.BinOp(left=x, op=op, right=y)


def _cmp(op: ast.cmpop):
    return lambda x, y: ast.Compare(left=x, ops=[op], comparators=[y])


def _c(v: float) -> ast.Constant:
    return ast.Constant(float(v))


class _Backward:
    def __init__(self, it) -> None:
        from .numpy_loops import Arr, Lazy, Sym, _Scatter
        self.it, self.Arr, self.Lazy, self.Sym, self.Scatter = it, Arr, Lazy, Sym, _Scatter
        self.G: Dict[Tuple, Any] = {}

    # ------------------------------------------------------------ gradients' storage
    def key(self, v: Any) -> Optional[Tuple]:
        if isinstance(v, self.Lazy):
            return ("list", v.real.name) if v.real is not None else ("lazy", id(v))
        if isinstance(v, self.Arr):
            return ("list", v.name)
        if isinstance(v, self.Sym):
            return ("list", v.arr.name) if v.arr is not None else ("sym", id(v))
        return None

    def geometry(self, v: Any):
        """(shape, strides, offset) of v in its list."""
        if isinstance(v, self.Sym):
            return (), (), v.arr.offset
        a = v.real if isinstance(v, self.Lazy) else v
        return tuple(a.shape), tuple(a.strides), a.offset

    def grad_list(self, name: str):
        """The gradient list of a program list (zeros at first)."""
        k = ("list", name)
        if k not in self.G:
            n = self.it.lists.get(name) or len(self.it.tables.get(name, ()))
            if not n:
                raise NotTranslatable(f"Can't hold the gradient of {name}")
            self.G[k] = self.it.new_list((n,), "g")
        return self.G[k]

    def grad(self, v: Any) -> Any:
        """v's gradient so far (its shape), or None."""
        k = self.key(v)
        if k is None or k not in self.G:
            return None
        if k[0] != "list":
            return self.G[k]
        shape, strides, offset = self.geometry(v)
        g = self.Arr(self.G[k].name, shape, strides, offset)
        if isinstance(v, self.Sym):
            return self.Sym(self.Arr(g.name, (1,), (1,), offset).at([ast.Constant(0)]),
                            arr=g)
        return g

    def add(self, v: Any, contrib: Any, shape: Tuple[int, ...]) -> None:
        """v's gradient += contrib (of shape, v broadcast to it)."""
        it = self.it
        if not it.requires(v):
            return
        vshape = () if isinstance(v, self.Sym) else tuple(v.shape)
        contrib = self.unbroadcast(contrib, shape, vshape)
        k = self.key(v)
        if k[0] == "list":
            gl = self.grad_list(k[1])
            vs, st, off = self.geometry(v)
            # (a dimension of stride 0 - a broadcast view, repeat_interleave's -
            #  holds one element: it gets the sum of the contributions along it,
            #  added once, not each written over the last)
            zero = tuple(d for d, (n, s) in enumerate(zip(vs, st)) if s == 0 and n > 1)
            if zero:
                if isinstance(contrib, (int, float)):
                    contrib = contrib * math.prod(vs[d] for d in zero)
                else:
                    contrib = it.reduce(it.to_arr(contrib), zero, True, "sum")
                vs = tuple(1 if d in zero else n for d, n in enumerate(vs))
            gv = self.Arr(gl.name, vs, st, off)
            it.copy_into(gv, it.binop(ast.Add(), gv, contrib))
        elif k[0] == "lazy":
            c = it.to_arr(contrib)
            self.G[k] = c if k not in self.G else it.to_arr(it.binop(ast.Add(), self.G[k], c))
        else:
            c = contrib if isinstance(contrib, self.Sym) else self.Sym(it.elem(contrib, (), []))
            if k in self.G:
                c = it.binop(ast.Add(), self.G[k], c)
            self.G[k] = it.rename(c, "gs") if isinstance(c, self.Sym) else c

    def unbroadcast(self, c: Any, shape: Tuple[int, ...], vshape: Tuple[int, ...]) -> Any:
        """c (shape) summed over the dimensions v (vshape) was broadcast along."""
        if tuple(shape) == tuple(vshape) or not shape:
            return c
        it = self.it
        n = len(shape) - len(vshape)
        axes = tuple(range(n)) + tuple(n + d for d, s in enumerate(vshape)
                                        if s == 1 and shape[n + d] != 1)
        if not axes:
            return c
        r = it.reduce(it.to_arr(c), axes, False, "sum")
        if not vshape or isinstance(r, self.Sym):
            return r
        # (the kept dimensions, v's ones of size 1 put back as broadcasts)
        strides, k = [], 0
        for d, s in enumerate(vshape):
            if s == 1 and shape[n + d] != 1:
                strides.append(0)
            else:
                strides.append(r.strides[k])
                k += 1
        return r.view(vshape, tuple(strides))

    # ------------------------------------------------------------ the operations
    def run(self, root: Any) -> None:
        it = self.it
        k = self.key(root)
        if k is None or not it.requires(root):
            raise NotTranslatable("backward() of a value that doesn't depend on a parameter")
        if (math.prod(root.shape) if not isinstance(root, self.Sym) else 1) != 1:
            raise NotTranslatable("backward() of an array: only of a single value")
        self.add(root, 1.0, () if isinstance(root, self.Sym) else tuple(root.shape))
        for what, out, ins, info in reversed(it.tape):
            g = self.grad(out)
            if g is None:
                continue
            getattr(self, "on_" + what)(out, g, ins, info)

    def on_copy(self, out, g, ins, info) -> None:
        src = ins[0]
        if self.key(src) == self.key(out):
            return                                    # (a value made a list: the same)
        self.add(src, g, () if isinstance(out, self.Sym) else tuple(out.shape))

    def on_rename(self, out, g, ins, info) -> None:
        self.add(ins[0], g, ())

    def on_gather(self, out, g, ins, info) -> None:
        base, ops = ins[0], ins[1:]
        it = self.it
        if not it.requires(base):
            return
        gl = self.grad_list(base.name)
        target = self.Arr(gl.name, base.shape, base.strides, base.offset)
        it.scatter(self.Scatter(target, ops, tuple(out.shape)),
                   lambda old, new: ast.BinOp(left=old, op=ast.Add(), right=new), g)

    def on_matmul(self, out, g, ins, info) -> None:
        from .torch_ops import transpose
        it = self.it
        va, vb = ins

        def t(v):                                     # (the last two dimensions swapped)
            axes = list(range(len(v.shape)))
            axes[-2], axes[-1] = axes[-1], axes[-2]
            return transpose(v, axes)
        if it.requires(va):
            self.add(va, it.matmul(g, t(vb)), tuple(it.to_arr(g).shape[:-1]) + (va.shape[-1],))
        if it.requires(vb):
            self.add(vb, it.matmul(t(va), g), tuple(it.to_arr(g).shape[:-2]) + tuple(vb.shape[-2:]))

    def on_reduce(self, out, g, ins, info) -> None:
        it = self.it
        a = it.to_arr(ins[0])
        axes, kind, result = info["axes"], info["kind"], info["result"]
        gl = self.grad_list(result.name)
        # The result's gradient and value broadcast back over the reduced axes (the
        # result's list holds the kept ones, in order)
        kept = list(result.strides) if len(axes) < len(a.shape) else []
        strides, k = [], 0
        for d in range(len(a.shape)):
            if d in axes:
                strides.append(0)
            else:
                strides.append(kept[k])
                k += 1
        gb = self.Arr(gl.name, tuple(a.shape), tuple(strides))
        if kind == "sum":
            c = gb
        elif kind == "mean":
            n = math.prod(a.shape[d] for d in axes)
            c = it.elementwise(lambda x: ast.BinOp(left=x, op=ast.Mult(), right=_c(1.0 / n)), gb)
        elif kind == "max":
            rb = self.Arr(result.name, tuple(a.shape), tuple(strides))
            c = it.elementwise(lambda x, y, z: ast.BinOp(left=x, op=ast.Mult(), right=ast.Compare(
                left=y, ops=[ast.Eq()], comparators=[z])), gb, a, rb)
        else:
            raise NotTranslatable(f"The gradient of {kind} is not supported")
        self.add(ins[0], c, tuple(a.shape))

    def on_maxpool(self, out, g, ins, info) -> None:
        """torch's max pooling: each window's gradient to its first maximum (in the
        window's order), as torch's index of the maximum."""
        it = self.it
        win = ins[0]
        if not it.requires(win):
            return
        from .numpy_loops import _assign, _load, _store
        gl = self.grad_list(win.name)
        gwin = self.Arr(gl.name, win.shape, win.strides, win.offset)
        g = it.to_arr(g)
        res = it.to_arr(out)
        nout = len(res.shape)

        def body(oidx):
            done = it.fresh("dn")
            it.emit(_assign(done, _c(0)))

            def inner(widx):
                idx = list(oidx) + list(widx)
                hit = it.fresh("ht")
                it.emit(_assign(hit, ast.BinOp(
                    left=ast.Compare(left=win.at(idx), ops=[ast.Eq()], comparators=[res.at(list(oidx))]),
                    op=ast.Mult(), right=ast.BinOp(left=_c(1), op=ast.Sub(), right=_load(done)))))
                it.emit(ast.Assign(targets=[_store(gwin.at(idx))], value=ast.BinOp(
                    left=gwin.at(idx), op=ast.Add(),
                    right=ast.BinOp(left=g.at(list(oidx)), op=ast.Mult(), right=_load(hit)))))
                it.emit(_assign(done, ast.BinOp(left=_load(done), op=ast.Add(), right=_load(hit))))
            it.loops(tuple(win.shape[nout:]), inner)
        it.loops(tuple(res.shape), body)

    def on_ew(self, out, g, ins, info) -> None:
        it = self.it
        op = info.get("op")
        shape = () if isinstance(out, self.Sym) else tuple(out.shape)
        req = [it.requires(v) for v in ins]
        if not any(req):
            return
        if op is None:
            raise NotTranslatable("Can't differentiate an operation (no derivative known)")
        kind = op[0]
        ew = it.elementwise
        mul, div, sub, add = _bin(ast.Mult()), _bin(ast.Div()), _bin(ast.Sub()), _bin(ast.Add())

        def neg(x):
            return ast.UnaryOp(op=ast.USub(), operand=x)

        def give(i, fn, *vals):
            if req[i]:
                self.add(ins[i], ew(fn, g, *vals), shape)
        if kind in ("cmp", "rnd", "int"):
            return
        if kind == "add":
            give(0, lambda d: d)
            give(1, lambda d: d)
        elif kind == "sub":
            give(0, lambda d: d)
            give(1, neg)
        elif kind == "mul":
            give(0, lambda d, y: mul(d, y), ins[1])
            give(1, lambda d, x: mul(d, x), ins[0])
        elif kind == "div":
            give(0, lambda d, y: div(d, y), ins[1])
            give(1, lambda d, x, y: neg(div(mul(d, x), mul(y, y))), ins[0], ins[1])
        elif kind == "neg":
            give(0, neg)
        elif kind == "pow":
            k = op[1]
            give(0, lambda d, x: mul(mul(d, _c(k)), _power(x, k - 1)), ins[0])
        elif kind in ("max", "min"):
            first, second = (ast.Gt(), ast.GtE()) if kind == "max" else (ast.Lt(), ast.LtE())
            give(0, lambda d, x, y: mul(d, _cmp(first)(x, y)), ins[0], ins[1])
            give(1, lambda d, x, y: mul(d, _cmp(second)(y, x)), ins[0], ins[1])
        elif kind == "exp":
            give(0, lambda d, o: mul(d, o), out)
        elif kind == "log":
            give(0, lambda d, x: div(d, x), ins[0])
        elif kind == "tanh":
            give(0, lambda d, o: mul(d, sub(_c(1), mul(o, o))), out)
        elif kind == "sqrt":
            give(0, lambda d, o: div(mul(d, _c(0.5)), o), out)
        elif kind == "table":                          # (a function as a table: its derivative's)
            grad = op[1] + "_grad"
            if grad not in it.ns:
                raise NotTranslatable(f"Can't differentiate {op[1]}")
            give(0, lambda d, x: mul(d, ast.Call(func=ast.Name(id=grad, ctx=ast.Load()),
                                                 args=[x], keywords=[])), ins[0])
        elif kind == "where":                          # (masked_fill: x where not masked)
            give(0, lambda d, m: mul(d, sub(_c(1), m)), ins[1])
        elif kind == "recip":
            give(0, lambda d, y: neg(div(d, mul(y, y))), ins[0])
        elif kind == "abs":
            give(0, lambda d, x: mul(d, sub(_cmp(ast.Gt())(x, _c(0)), _cmp(ast.Lt())(x, _c(0)))),
                 ins[0])
        else:
            raise NotTranslatable(f"Can't differentiate {kind}")


def _power(x: ast.expr, k: int) -> ast.expr:
    if k == 0:
        return _c(1)
    out = x
    for _ in range(k - 1):
        out = ast.BinOp(left=out, op=ast.Mult(), right=x)
    return out


def backward(it, root: Any) -> None:
    """root.backward(): the gradients of the parameters, written as loops; they
    become the parameters' .grad (it.grads)."""
    from .numpy_loops import Arr, _contiguous
    was = it.recording
    it.recording = False
    try:
        b = _Backward(it)
        b.run(root)
    finally:
        it.recording = was
    for name in it.req_lists:
        real = it.real_of.get(name)
        if getattr(real, "requires_grad", False) and ("list", name) in b.G:
            shape = tuple(real.shape)
            it.grads[name] = Arr(b.G[("list", name)].name, shape, _contiguous(shape))
    it.tape.clear()
