"""Complete operator templates exist; this backend is not enabled or validated."""
from up_platform.contracts import UnsupportedOperation


def build(config, purpose):
    raise UnsupportedOperation("ascend", purpose, "configure",
                               "Template only: integrate device/SDK and pass the platform acceptance tests first")
