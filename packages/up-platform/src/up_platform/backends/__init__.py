"""Backend metadata is registered lazily, without importing SDKs."""

from up_platform.registry import register_numerical_policy

register_numerical_policy("native", "up_platform.common.policy:NativePolicy")
register_numerical_policy("portable_v1", "up_platform.experimental.numerics.policy:PortablePolicy")
