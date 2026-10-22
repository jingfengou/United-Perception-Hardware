from up_platform import DeviceBackend


class ExampleDevice(DeviceBackend):
    name = "example_vendor"
    device_type = "cpu"  # Demo only; use the namespace provided by the vendor SDK.
    communication_backend = "gloo"

    def __init__(self, device=None):
        # Import the real vendor SDK here if required, never at package import time.
        super().__init__(device)
