class ApplicationError(Exception):
    """Base domain exception."""


class CustomerNotFoundError(ApplicationError):
    pass


class ModelNotReadyError(ApplicationError):
    pass


class DataValidationError(ApplicationError):
    pass


class AgentProviderError(ApplicationError):
    pass
