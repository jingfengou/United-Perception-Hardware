"""Load only the requested inference implementation, without importing training."""
from up_platform.errors import CapabilityError
from up_platform.registry_factory import resolve


def load_backend(spec):
    if spec.inference_factory is None:
        raise CapabilityError("backend={!r}, purpose='inference': no execution backend registered".format(spec.name))
    return resolve(spec.inference_factory)


def get_backend():
    from up_platform import get_platform
    return get_platform().inference
