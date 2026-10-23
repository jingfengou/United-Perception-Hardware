"""Load only the requested training implementation, without importing inference."""
from up_platform.errors import CapabilityError
from up_platform.registry_factory import resolve


def load_backend(spec):
    if spec.training_factory is None:
        raise CapabilityError("backend={!r}, purpose='training': no execution backend registered".format(spec.name))
    return resolve(spec.training_factory)


def get_backend():
    from up_platform import get_platform
    return get_platform().training
