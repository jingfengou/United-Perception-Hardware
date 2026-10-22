"""Neutral lifecycle hooks for optional numerical policies; native is a no-op."""


class NativePolicy:
    profile = "native"
    provider = None
    mode = None
    operations = {}

    def __init__(self, runtime):
        self.runtime = runtime

    def start(self):
        pass

    def close(self):
        pass

    def reseed(self, seed):
        pass

    def wrap_rng_state(self, state):
        return state

    def restore_rng_state(self, state, restore_native):
        restore_native(state)

    def record_operation(self, name):
        pass

    def validate_autocast(self, enabled):
        pass

    def convolution_enabled(self, requested):
        return requested

    def state(self):
        return {"profile": self.profile, "rng": "vendor-native", "accumulation": "vendor-native",
                "seed": None, "offset": None, "operations": {},
                "convolution_library_enabled": getattr(self.runtime, "_convolution_library_enabled", None)}
