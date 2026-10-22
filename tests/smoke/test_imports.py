def test_core_imports() -> None:
    from app.api.main import app
    from app.core.container import get_container

    assert app.title
    assert get_container()
