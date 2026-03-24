"""Custom exceptions used by the bridge assessment backend."""


class BridgeAppError(Exception):
    """Base class for application-specific failures."""


class AssetValidationError(BridgeAppError):
    """Raised when required model or preprocessing assets are missing."""


class InputValidationError(BridgeAppError):
    """Raised when an uploaded file or JSON payload is invalid."""


class InferenceExecutionError(BridgeAppError):
    """Raised when model inference cannot be completed."""
