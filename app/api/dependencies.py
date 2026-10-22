from app.core.container import ApplicationContainer, get_container


def container_dependency() -> ApplicationContainer:
    return get_container()
