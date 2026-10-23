# up-platform

Independent hardware runtime and explicit training/inference contracts for UP.
The LightX2V-style layout separates `base/` device backends, category-first
`ops/<category>/<vendor>/` implementations, `registry_factory.py`, and explicit
initialization in `set_ai_device.py`. Training has its own runtime backend
factory; inference has a separate factory. Compatible implementations are shared
inside each operator category, not copied into each vendor.
Importing `up_platform` requires neither UP nor a GPU SDK. Install the vendor's
matching PyTorch before executing tensor operations. Public builds do not compile
GPU extensions. `UP_BUILD_BACKEND=nvidia` explicitly enables NVIDIA compilation.

See the repository's `docs/platform/` for architecture, the complete interface
catalog, vendor templates and acceptance status.
