import pytest
import torch
import torch.nn.functional as F

from up.accelerator import configure_accelerator, BackendConfigurationError, CapabilityError
from up.accelerator.numerics import PortableNumerics


@pytest.fixture(autouse=True)
def restore_runtime():
    configure_accelerator(backend="cpu", reset=True)
    yield
    configure_accelerator(backend="cpu", reset=True)


def test_native_profile_does_not_enter_pytorch_modes():
    from torch.overrides import _get_current_function_mode
    from torch.utils._python_dispatch import _get_current_dispatch_mode
    before = (_get_current_function_mode(), _get_current_dispatch_mode(), torch.backends.mha.get_fastpath_enabled())
    provider = configure_accelerator(backend="nvidia", reset=True)
    assert provider.numerics_state()["profile"] == "native"
    assert before == (_get_current_function_mode(), _get_current_dispatch_mode(), torch.backends.mha.get_fastpath_enabled())


def test_portable_profile_reset_restores_modes_and_mha_policy():
    from torch.overrides import _get_current_function_mode
    from torch.utils._python_dispatch import _get_current_dispatch_mode
    original = torch.backends.mha.get_fastpath_enabled()
    provider = configure_accelerator({"runtime": {"accelerator": {"backend": "metax", "numerics": "portable_v1"}}}, reset=True)
    assert _get_current_function_mode() is provider._numerics
    assert _get_current_dispatch_mode() is provider._numerics.random_mode
    assert not torch.backends.mha.get_fastpath_enabled()
    configure_accelerator(backend="cpu", reset=True)
    assert _get_current_function_mode() is None and _get_current_dispatch_mode() is None
    assert torch.backends.mha.get_fastpath_enabled() == original


def test_invalid_profile_and_unsupported_device_fail_explicitly():
    with pytest.raises(BackendConfigurationError, match="Unknown numerical profile"):
        configure_accelerator({"runtime": {"accelerator": {"backend": "cpu", "numerics": "unknown"}}}, reset=True)
    with pytest.raises(CapabilityError, match="CUDA-shaped"):
        configure_accelerator({"runtime": {"accelerator": {"backend": "cpu", "numerics": "portable_v1"}}}, reset=True)


def test_portable_state_is_versioned_and_restores_native_rng_atomically(monkeypatch):
    native = torch.arange(40, dtype=torch.uint8)
    observed = []
    monkeypatch.setattr(torch.cuda, "get_rng_state", lambda: native.clone())
    monkeypatch.setattr(torch.cuda, "set_rng_state", lambda value: observed.append(value.clone()))
    provider = configure_accelerator({"runtime": {"accelerator": {"backend": "metax", "numerics": "portable_v1"}}}, reset=True)
    provider._numerics.reseed(123)
    provider._numerics.offset = 2 ** 32 + 7
    state = provider.get_rng_state()
    provider._numerics.reseed(456)
    provider.set_rng_state(state)
    assert provider.numerics_state()["seed"] == 123 and provider.numerics_state()["offset"] == 2 ** 32 + 7
    assert torch.equal(observed[-1], native)
    with pytest.raises(BackendConfigurationError, match="do not mix"):
        provider.set_rng_state(native)
    with pytest.raises(CapabilityError, match="autocast"):
        provider.autocast()


@pytest.mark.parametrize("kind", ["linear", "conv", "layer_norm", "batch_norm", "silu"])
def test_widened_forward_and_backward_preserve_fp32_contract(kind):
    g = torch.Generator().manual_seed(11)
    x = torch.randn(2, 3, 4, 4, generator=g, requires_grad=True)
    weight = torch.randn(3, generator=g, requires_grad=True)
    bias = torch.randn(3, generator=g, requires_grad=True)
    running = torch.zeros(3), torch.ones(3)
    if kind == "linear":
        fn = lambda v,w,b:F.linear(v,w.expand(4,3),None)
        x = x.transpose(1,3).contiguous().detach().requires_grad_()
    elif kind == "conv":
        fn = lambda v,w,b:F.conv2d(v,w.reshape(1,3,1,1),None)
    elif kind == "layer_norm":
        fn = lambda v,w,b:F.layer_norm(v,(4,4))
    elif kind == "batch_norm":
        fn = lambda v,w,b:F.batch_norm(v,running[0],running[1],w,b,training=True)
    else:
        fn = lambda v,w,b:F.silu(v)
    with PortableNumerics(device_type="cpu"):
        output = fn(x,weight,bias)
        loss = output.sum()
    gradients = torch.autograd.grad(loss,(x,weight,bias),allow_unused=True)
    assert output.dtype == torch.float32
    assert all(t is None or t.dtype == torch.float32 and torch.isfinite(t).all() for t in gradients)
    assert torch.isfinite(output).all()


def test_nested_mode_cannot_be_popped_by_reset():
    from torch.overrides import TorchFunctionMode
    class Outer(TorchFunctionMode):
        def __torch_function__(self, function, types, args=(), kwargs=None):
            return function(*args, **(kwargs or {}))
    configure_accelerator({"runtime": {"accelerator": {"backend": "metax", "numerics": "portable_v1"}}}, reset=True)
    with Outer():
        with pytest.raises(BackendConfigurationError, match="nested PyTorch modes"):
            configure_accelerator(backend="cpu", reset=True)


def test_portable_semantic_operations_are_shared_but_native_owner_is_unchanged():
    from up.accelerator import ops
    config = {"runtime": {"accelerator": {"backend": "nvidia", "numerics": "portable_v1"}}}
    provider = configure_accelerator(config, reset=True)
    assert provider.capabilities()["operations"]["naive_nms"]["owner"] == "portable_v1"
    assert torch.equal(ops.torchvision_nms(torch.tensor([[0.,0.,3.,3.],[0.,0.,3.,3.]]),
                                         torch.ones(2),.5), torch.tensor([0]))
    provider = configure_accelerator(backend="nvidia", reset=True)
    assert provider.capabilities()["operations"]["naive_nms"]["owner"] == "nvidia"


def test_batch_norm_running_state_and_gradients_match_widened_reference():
    g = torch.Generator().manual_seed(11)
    values = [torch.randn(3,4,5,6,generator=g), torch.randn(4,generator=g), torch.randn(4,generator=g)]
    inputs = [[value.clone().requires_grad_() for value in values] for _ in range(2)]
    stats = [(torch.zeros(4), torch.ones(4)), (torch.zeros(4,dtype=torch.float64), torch.ones(4,dtype=torch.float64))]
    upstream = torch.randn(3,4,5,6,generator=g)
    with PortableNumerics(device_type="cpu"):
        actual = F.batch_norm(inputs[0][0],*stats[0],*inputs[0][1:],training=True,momentum=.2,eps=1e-4)
    expected = F.batch_norm(inputs[1][0].double(),*stats[1],*[t.double() for t in inputs[1][1:]],
                           training=True,momentum=.2,eps=1e-4).float()
    assert torch.equal(actual,expected)
    a, b = [torch.autograd.grad(output, state, upstream) for output,state in zip((actual,expected),inputs)]
    assert all(torch.equal(x,y) for x,y in zip(a,b))
    assert all(torch.equal(x,y.float()) for x,y in zip(*stats))


def test_inplace_silu_preserves_alias_and_gradient():
    source = torch.tensor([-2.,.3,4.],requires_grad=True)
    value = source * 1
    with PortableNumerics(device_type="cpu"):
        result = F.silu(value,inplace=True)
    assert result is value
    result.sum().backward()
    reference = source.detach().double().requires_grad_()
    F.silu(reference).sum().backward()
    assert torch.equal(source.grad,reference.grad.float())


def test_nested_dispatch_mode_cannot_be_popped_by_reset():
    from torch.utils._python_dispatch import TorchDispatchMode
    class Other(TorchDispatchMode):
        def __torch_dispatch__(self,function,types,args=(),kwargs=None):
            return function(*args,**(kwargs or {}))
    configure_accelerator({"runtime": {"accelerator": {"backend": "metax", "numerics": "portable_v1"}}},reset=True)
    with Other():
        with pytest.raises(BackendConfigurationError,match="nested PyTorch dispatch modes"):
            configure_accelerator(backend="cpu",reset=True)


def test_new_plan_keeps_existing_models_budgets_seeds_and_threshold():
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1] / "tests/fixtures/protocols"
    old = json.loads((root / "cross_vendor_diagnosis_20261021/plan.json").read_text())
    new = json.loads((root / "cross_vendor_fix_20261021/plan.json").read_text())
    assert old["models"] == new["models"]
    assert old["equivalence"] == new["equivalence"]
    assert new["numerical_profile"] == "portable_v1"
    fixed=json.loads((root / "cross_vendor_fix_20261021/fixed-order/plan.json").read_text())
    assert fixed["models"]==old["models"] and fixed["equivalence"]==old["equivalence"]


def test_profile_convolution_policy_covers_backward_and_restores():
    observed=[]
    class FlagEcho(torch.autograd.Function):
        @staticmethod
        def forward(ctx,value):
            return value*2
        @staticmethod
        def backward(ctx,gradient):
            observed.append(torch.backends.cudnn.enabled)
            return gradient*2
    original=torch.backends.cudnn.enabled
    provider=configure_accelerator({"runtime": {"accelerator": {"backend": "metax", "numerics": "portable_v1"}}},reset=True)
    assert torch.backends.cudnn.enabled is False
    provider.configure_convolution(True)
    with provider.convolution_flags(True):
        assert torch.backends.cudnn.enabled is False
        FlagEcho.apply(torch.tensor(1.,requires_grad=True)).backward()
    assert observed==[False] and torch.backends.cudnn.enabled is False
    configure_accelerator(backend="cpu",reset=True)
    assert torch.backends.cudnn.enabled==original


def test_dropout_backward_uses_shared_widened_scaling():
    gradient=torch.tensor([.1,-.3,1.2345678,0.],dtype=torch.float32)
    mask=torch.tensor([True,False,True,True])
    scale=1./.9
    with PortableNumerics(device_type="cpu").random_mode:
        actual=torch.ops.aten.native_dropout_backward.default(gradient,mask,scale)
    expected=(gradient.double()*mask*scale).float()
    assert torch.equal(actual,expected)
    assert actual.dtype==gradient.dtype


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("affine", [True, False])
def test_shared_batch_norm_formula_and_complete_backward(monkeypatch, training, affine):
    from up.accelerator.backends import portable_normalization as implementation
    monkeypatch.setattr(implementation,"reduce_sum",lambda x,dims=None,keepdim=False:x.sum(dim=dims,keepdim=keepdim))
    g=torch.Generator().manual_seed(11)
    x=torch.randn(4,3,5,7,generator=g,dtype=torch.float64).requires_grad_()
    w=torch.randn(3,generator=g,dtype=torch.float64).requires_grad_() if affine else None
    b=torch.randn(3,generator=g,dtype=torch.float64).requires_grad_() if affine else None
    mean=torch.randn(3,generator=g,dtype=torch.float64)
    variance=torch.rand(3,generator=g,dtype=torch.float64)+1
    expected_mean,expected_variance=mean.clone(),variance.clone()
    reference=F.batch_norm(x,expected_mean,expected_variance,w,b,training=training,momentum=.2,eps=1e-3)
    upstream=torch.randn(reference.shape,generator=g,dtype=torch.float64)
    parameters=(x,w,b) if affine else (x,)
    gradients=torch.autograd.grad(reference,parameters,upstream)
    with torch.no_grad():
        output,saved_mean,saved_inverse=implementation.batch_norm(x,w,b,mean,variance,training,.2,1e-3)
        actual=implementation.batch_norm_backward(upstream,x,w,mean,variance,saved_mean,saved_inverse,
                                                   training,1e-3,(True,affine,affine))
    torch.testing.assert_close(output,reference,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(mean,expected_mean,rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(variance,expected_variance,rtol=1e-12,atol=1e-12)
    for value,expected in zip(actual,gradients):
        torch.testing.assert_close(value,expected,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize("shape,normalized", [((4,5,7),(7,)),((4,5,7),(5,7)),((5,7),(5,7))])
def test_shared_layer_norm_formula_and_complete_backward(monkeypatch,shape,normalized):
    from up.accelerator.backends import portable_normalization as implementation
    monkeypatch.setattr(implementation,"reduce_sum",lambda x,dims=None,keepdim=False:x.sum(dim=dims,keepdim=keepdim))
    g=torch.Generator().manual_seed(11)
    x=torch.randn(shape,generator=g,dtype=torch.float64).requires_grad_()
    w=torch.randn(normalized,generator=g,dtype=torch.float64).requires_grad_()
    b=torch.randn(normalized,generator=g,dtype=torch.float64).requires_grad_()
    reference=F.layer_norm(x,normalized,w,b,1e-5)
    upstream=torch.randn(shape,generator=g,dtype=torch.float64)
    expected=torch.autograd.grad(reference,(x,w,b),upstream)
    with torch.no_grad():
        output,mean,inverse=implementation.layer_norm(x,normalized,w,b,1e-5)
        actual=implementation.layer_norm_backward(upstream,x,normalized,mean,inverse,w,b,(True,True,True))
    for a,b in zip((output,*actual),(reference,*expected)):
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize("shapes", [((3,4),(7,4)),((2,3,4),(1,7,4)),((3,4),(1,7,4))])
def test_shared_l1_distance_and_both_operand_gradients(monkeypatch,shapes):
    from up.accelerator.backends import portable_distance as implementation
    monkeypatch.setattr(implementation,"reduce_sum",lambda x,dims=None,keepdim=False:x.sum(dim=dims,keepdim=keepdim))
    g=torch.Generator().manual_seed(11)
    a,b=[torch.randn(shape,generator=g,dtype=torch.float64).requires_grad_() for shape in shapes]
    reference=torch.cdist(a,b,p=1)
    upstream=torch.randn(reference.shape,generator=g,dtype=torch.float64)
    expected=torch.autograd.grad(reference,(a,b),upstream)
    with torch.no_grad():
        output=implementation.l1_distance(a,b)
        actual=(implementation.l1_backward(upstream,a,b,1,output),
                implementation.l1_backward(upstream.transpose(-1,-2),b,a,1,output.transpose(-1,-2)))
    for a,b in zip((output,*actual),(reference,*expected)):
        torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)


def test_shared_l2_norm_preserves_dim_keepdim_and_dtype(monkeypatch):
    from up.accelerator.backends import portable_distance as implementation
    monkeypatch.setattr(implementation,"reduce_sum",lambda x,dims=None,keepdim=False:x.sum(dim=dims,keepdim=keepdim))
    x=torch.arange(24,dtype=torch.float64).reshape(2,3,4)
    actual=implementation.vector_norm(x,ord=2,dim=(1,2),keepdim=True,dtype=torch.float64)
    torch.testing.assert_close(actual,torch.linalg.vector_norm(x,dim=(1,2),keepdim=True))


def test_missing_profile_dependency_names_backend_operation_and_reason(monkeypatch):
    mode=PortableNumerics(device_type="cpu",backend_name="example_vendor")
    def unavailable(*args,**kwargs):
        raise ImportError("vendor Triton unavailable")
    monkeypatch.setattr(mode,"uniform",unavailable)
    with pytest.raises(CapabilityError,match="backend='example_vendor'.*aten.rand.*vendor Triton unavailable"):
        with mode.random_mode:
            torch.rand(2)
