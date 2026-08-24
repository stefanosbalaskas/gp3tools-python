import gp3tools as gp3


def test_frozen_public_api_has_278_names():
    assert len(gp3.R_EXPORTS) == 278
    assert len(set(gp3.R_EXPORTS)) == 278
    assert all(callable(getattr(gp3, name)) for name in gp3.R_EXPORTS)


def test_all_exports_have_native_or_adapted_python_implementation():
    status = gp3.api_status()
    assert len(status) == 278
    assert not status["status"].eq("r-bridge").any()
    assert set(status["status"]) <= {"native", "native-adapted", "native-adapter"}
