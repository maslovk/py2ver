# torch_optim.py
"""
torch's training machinery in the loop translator (numpy_loops.py):

  - methods of torch's own objects that only set Python values
    (model.train(), model.eval(), optimizer.zero_grad()) run as they are, on
    the translation's copy of the objects
  - optimizer.step() for torch.optim.Adam / AdamW / SGD, as torch computes it
    (the single-tensor algorithm): the parameters' .grad from loss.backward()
    (autograd.py); the optimizer's state for a parameter (exp_avg, exp_avg_sq,
    momentum_buffer), made on its first step as zeros, is state the hardware
    keeps, its step count an integer
  - scheduler.step() for OneCycleLR: the learning rate (and Adam's beta1, the
    momentum it cycles) of each step depends only on the step's number, so
    torch's values for all of them are tables, read at the scheduler's counter
    (state); they are the optimizer's group['lr'] / group['betas'][0], state too
  - BatchNorm in training mode: the batch's mean and variance (biased) per
    channel, the running statistics updated with the unbiased variance, and
    num_batches_tracked counted
"""

from __future__ import annotations

import ast
import inspect
from typing import Any, Dict, List

from .numpy_fe import NotTranslatable


def _torch():
    import torch
    return torch


def method_call(it, fn: Any, args: List[Any], kwargs: Dict[str, Any]) -> Any:
    """A method of one of torch's objects; NotImplemented for others."""
    owner = getattr(fn, "__self__", None)
    func = getattr(fn, "__func__", None)
    if owner is None or func is None or not inspect.ismethod(fn):
        return NotImplemented
    if not (getattr(func, "__module__", "") or "").startswith("torch"):
        return NotImplemented
    torch = _torch()
    name = fn.__name__
    if isinstance(owner, torch.optim.Optimizer) and name == "step":
        if args or kwargs:
            raise NotTranslatable("optimizer.step(closure) is not supported")
        return optimizer_step(it, owner)
    if isinstance(owner, torch.optim.lr_scheduler.LRScheduler) and name == "step":
        if args or kwargs:
            raise NotTranslatable("scheduler.step(epoch) is not supported")
        return scheduler_step(it, owner)
    if name == "zero_grad":
        it.grads.clear()
    from .numpy_loops import Arr, Sym, _arrays_in
    if _arrays_in(list(args) + list(kwargs.values())) or any(
            isinstance(v, (Arr, Sym)) for v in list(args) + list(kwargs.values())):
        raise NotTranslatable(f"{type(owner).__name__}.{name}() on values computed "
                              "in hardware is not supported")
    return fn(*args, **kwargs)


# ==================================================================== helpers

def _const(v: float) -> ast.Constant:
    return ast.Constant(float(v))


def _bin(a: Any, op: ast.operator, b: Any) -> ast.expr:
    return ast.BinOp(left=a, op=op, right=b)


def _sym_expr(it, v: Any) -> ast.expr:
    from .numpy_loops import Sym
    if isinstance(v, Sym):
        return v.expr
    return _const(v)


def _scalar(it, expr: ast.expr, base: str):
    return it.scalar(expr, base)


# ==================================================================== optimizers

def optimizer_step(it, opt) -> None:
    torch = _torch()
    kind = type(opt).__name__
    if kind not in ("Adam", "AdamW", "SGD"):
        raise NotTranslatable(f"The optimizer {kind} is not supported (Adam, AdamW, SGD)")
    for group in opt.param_groups:
        for flag in ("amsgrad", "maximize", "nesterov", "capturable", "differentiable"):
            if group.get(flag):
                raise NotTranslatable(f"{kind}({flag}=True) is not supported")
        lr = it.item_value(group, "lr")
        for p in group["params"]:
            pa = it.adopt(p)
            grad = it.grads.get(pa.name)
            if grad is None:
                continue                                  # (p.grad is None: skipped)
            state = opt.state[p]                          # (the copy's: made on the first step)
            if kind == "SGD":
                _sgd(it, group, state, p, pa, grad, lr)
            else:
                _adam(it, group, state, p, pa, grad, lr, decoupled=kind == "AdamW" or bool(
                    group.get("decoupled_weight_decay")))
    return None


def _adam(it, group, state, p, pa, grad, lr, decoupled: bool) -> None:
    torch = _torch()
    if not state:
        state["step"] = torch.tensor(0.0)
        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
    beta1 = it.item_value(group, "betas", 0)
    beta2 = group["betas"][1]
    eps, wd = float(group["eps"]), float(group["weight_decay"])
    m, v = it.adopt(state["exp_avg"]), it.adopt(state["exp_avg_sq"])
    step = it.int_value(state["step"])
    step = it.set_int(state["step"], _bin(_sym_expr(it, step), ast.Add(), ast.Constant(1)))
    mul, add, sub, div = ast.Mult(), ast.Add(), ast.Sub(), ast.Div()
    if wd:
        if decoupled:                                    # param.mul_(1 - lr * wd)
            f = it.rename(it.binop(sub, 1.0, it.binop(mul, lr, wd)), "wdf")
            it.copy_into(pa, it.binop(mul, pa, f))
        else:                                            # grad = grad + wd * param
            grad = it.to_arr(it.binop(add, grad, it.binop(mul, pa, wd)))
    # exp_avg.lerp_(grad, 1 - beta1); exp_avg_sq.mul_(beta2).addcmul_(grad, grad, 1 - beta2)
    one_m_b1 = it.rename(it.binop(sub, 1.0, beta1), "omb")
    it.copy_into(m, it.binop(add, m, it.binop(mul, it.binop(sub, grad, m), one_m_b1)))
    it.copy_into(v, it.binop(add, it.binop(mul, v, beta2),
                             it.binop(mul, it.binop(mul, grad, grad), 1.0 - beta2)))
    bc1 = it.rename(it.binop(sub, 1.0, it.power_of(beta1, step)), "bc1")
    bc2 = it.rename(it.binop(sub, 1.0, it.power_of(beta2, step)), "bc2")
    step_size = it.rename(it.binop(div, lr, bc1), "ss")
    bc2_sqrt = it.rename(it.numpy("sqrt", [bc2], {}), "bcs")
    # denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
    denom = it.binop(add, it.binop(div, it.numpy("sqrt", [v], {}), bc2_sqrt), eps)
    # param.addcdiv_(exp_avg, denom, value=-step_size)
    it.copy_into(pa, it.binop(sub, pa, it.binop(mul, it.binop(div, m, denom), step_size)))


def _sgd(it, group, state, p, pa, grad, lr) -> None:
    torch = _torch()
    mom, damp, wd = float(group.get("momentum", 0)), float(group.get("dampening", 0)), \
        float(group.get("weight_decay", 0))
    mul, add, sub = ast.Mult(), ast.Add(), ast.Sub()
    if wd:
        grad = it.to_arr(it.binop(add, grad, it.binop(mul, pa, wd)))
    if mom:
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(p)
        if damp:
            raise NotTranslatable("SGD(dampening=...) is not supported")
        buf = it.adopt(state["momentum_buffer"])        # (zeros: the first step's is the gradient)
        it.copy_into(buf, it.binop(add, it.binop(mul, buf, mom), it.binop(mul, grad, 1.0 - damp)))
        grad = buf
    it.copy_into(pa, it.binop(sub, pa, it.binop(mul, grad, lr)))


# ==================================================================== schedulers

def scheduler_step(it, sched) -> None:
    kind = type(sched).__name__
    if kind != "OneCycleLR":
        raise NotTranslatable(f"The scheduler {kind} is not supported (OneCycleLR)")
    np = it.np
    total = int(sched.total_steps)
    momentum = bool(sched.cycle_momentum)
    groups = sched.optimizer.param_groups
    count = it.attr_value(sched, "last_epoch")
    count = it.set_attr(sched, "last_epoch", _bin(_sym_expr(it, count), ast.Add(), ast.Constant(1)))
    # (the step's number: the tables' index, at most the last step)
    index = it.rename(it.elementwise(lambda c: ast.Call(func=ast.Name(id="min", ctx=ast.Load()),
                                                        args=[c, ast.Constant(total)], keywords=[]),
                                     count, op=("int",)), "sn")
    for g, group in enumerate(groups):
        lrs, moms = [], []
        for step_num in range(total + 1):
            lr, mo = _one_cycle(sched, group, step_num)
            lrs.append(lr)
            moms.append(mo)
        lr_table = it.const_array(None, f"lr_schedule{g}", np.array(lrs), hint="lr_schedule")
        it.set_item(group, "lr", _int_read(lr_table, index))
        if momentum:
            if "betas" not in group:
                raise NotTranslatable("OneCycleLR's momentum: only Adam's beta1 is supported")
            m_table = it.const_array(None, f"beta1_schedule{g}", np.array(moms),
                                     hint="beta1_schedule")
            it.set_item(group, "betas", _int_read(m_table, index), 0)
    return None


def _int_read(table, index) -> ast.expr:
    from .numpy_loops import _int_index
    return table.at([_int_index(index.expr)])


def _one_cycle(sched, group, step_num: int):
    """OneCycleLR.get_lr's values for one step (and its momentum)."""
    start_step = 0.0
    phases = sched._schedule_phases
    for i, phase in enumerate(phases):
        end_step = phase["end_step"]
        if step_num <= end_step or i == len(phases) - 1:
            pct = (step_num - start_step) / (end_step - start_step)
            lr = sched._anneal_func(group[phase["start_lr"]], group[phase["end_lr"]], pct)
            mo = (sched._anneal_func(group[phase["start_momentum"]], group[phase["end_momentum"]],
                                     pct) if sched.cycle_momentum else None)
            return lr, mo
        start_step = phase["end_step"]
    raise AssertionError


# ==================================================================== BatchNorm

def batch_norm_train(it, mod, x):
    """BatchNorm in training mode: normalized with the batch's statistics, the running
    ones updated (state)."""
    torch = _torch()
    if mod.momentum is None:
        raise NotTranslatable("BatchNorm(momentum=None) is not supported")
    nd = len(x.shape)
    axes = (0,) + tuple(range(2, nd))
    C = x.shape[1]
    n = 1
    for d in axes:
        n *= x.shape[d]
    extra = (1,) * (nd - 2)
    mul, add, sub, div = ast.Mult(), ast.Add(), ast.Sub(), ast.Div()
    x = it.to_arr(x)
    mean = it.reduce(x, axes, True, "mean")                 # (1, C, 1, 1)
    d = it.binop(sub, x, mean)
    var = it.reduce(it.to_arr(it.binop(mul, d, d)), axes, True, "mean")
    inv = it.to_arr(it.binop(div, 1.0, it.numpy("sqrt", [it.binop(add, var, float(mod.eps))], {})))
    y = it.binop(mul, d, inv)
    if mod.affine:
        w = it.method(it.param(mod, "weight"), "reshape", [(C,) + extra], {})
        b = it.method(it.param(mod, "bias"), "reshape", [(C,) + extra], {})
        y = it.binop(add, it.binop(mul, y, w), b)
    if mod.track_running_stats:
        m = float(mod.momentum)
        was = it.recording
        it.recording = False                                # (no gradients through them)
        rm, rv = it.param(mod, "running_mean"), it.param(mod, "running_var")
        flat_mean = mean.view((C,), (mean.strides[1],))
        flat_var = var.view((C,), (var.strides[1],))
        it.copy_into(rm, it.binop(add, it.binop(mul, rm, 1.0 - m), it.binop(mul, flat_mean, m)))
        it.copy_into(rv, it.binop(add, it.binop(mul, rv, 1.0 - m),
                                  it.binop(mul, flat_var, m * n / max(n - 1, 1))))
        nbt = mod.num_batches_tracked
        if nbt is not None:
            c = it.int_value(nbt)
            it.set_int(nbt, _bin(_sym_expr(it, c), ast.Add(), ast.Constant(1)))
        it.recording = was
    return y
