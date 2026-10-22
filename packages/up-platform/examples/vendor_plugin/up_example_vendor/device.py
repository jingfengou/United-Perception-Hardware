from up_platform import DeviceBackend


class ExampleDevice(DeviceBackend):
    name = "example_vendor"
    device_type = "cpu"  # Demo only; use the namespace provided by the vendor SDK.
    communication_backend = "gloo"
