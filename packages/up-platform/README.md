# up-platform

Independent hardware runtime and explicit training/inference contracts for UP.
Importing `up_platform` requires neither UP nor a GPU SDK. Install the vendor's
matching PyTorch before executing tensor operations. Public builds do not compile
GPU extensions. `UP_BUILD_BACKEND=nvidia` explicitly enables NVIDIA compilation.

See the repository's `docs/platform/` for architecture, the complete interface
catalog, vendor templates and acceptance status.
