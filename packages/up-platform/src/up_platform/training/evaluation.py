"""Evaluation on the existing training model without selecting inference kernels."""
import contextlib
import functools

import torch


@contextlib.contextmanager
def evaluation_context(model):
    from up_platform import get_platform
    platform = get_platform()
    if platform.purpose != "training":
        raise RuntimeError("Training evaluation requires the training platform context")
    states = [(module, module.training) for module in model.modules()]
    gradients = [(parameter, parameter.requires_grad) for parameter in model.parameters()]
    try:
        model.eval()
        with torch.no_grad():
            yield model
    finally:
        for module, training in states:
            module.training = training
        for parameter, requires_grad in gradients:
            parameter.requires_grad_(requires_grad)


def preserve_evaluation_state(function):
    @functools.wraps(function)
    def invoke(self, *args, **kwargs):
        from up_platform import get_platform
        if get_platform().purpose != "training":
            return function(self, *args, **kwargs)
        with contextlib.ExitStack() as stack:
            stack.enter_context(evaluation_context(self.model))
            ema = getattr(self, "ema", None)
            if ema is not None and ema.model is not self.model:
                stack.enter_context(evaluation_context(ema.model))
            return function(self, *args, **kwargs)
    return invoke
