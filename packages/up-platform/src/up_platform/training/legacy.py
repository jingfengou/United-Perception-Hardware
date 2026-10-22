"""Original optional linklink parameter broadcast and gradient synchronization."""
from up_platform.shared.layers import link


def wrap_model(model, sync=True, master_rank=0):
    def _register_hooks(self):
        for i, (name, p) in enumerate(self.named_parameters()):
            if p.requires_grad:
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                grad_acc.register_hook(self._make_hook(name, p, i))
                self._grad_accs.append(grad_acc)

    def _make_hook(name, p, i):
        def hook(*ignore):
            link.allreduce_async(name, p.grad.data)
        return hook

    for name, p in model.state_dict().items():
        link.broadcast(p, master_rank)
    if not sync:
        model._grad_accs = []
        model._register_hooks = _register_hooks
        model._make_hook = _make_hook
        model._register_hooks(model)
    return model


def reduce_gradients(model, sync=True, allow_dead_parameter=False):
    if sync:
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                try:
                    link.allreduce(param.grad.data)
                except AttributeError as e:
                    warning = (f"The AttributeError above was probably caused by {name}.grad being None "
                               f"but the gradient w.r.t {name} is required. "
                               f"Please check your model to make sure that {name} is always used in your "
                               "forward pass if it is learnable, otherwise, set it's requrires_grad flag to False. "
                               "Another temporary workaround, you may add 'and param.grad is not None' to "
                               "the conditional above only if you know exactly the grad is needless.")
                    if not allow_dead_parameter:
                        raise AttributeError('This line is not an error, just warning: ' + warning) from e
    else:
        link.synchronize()
