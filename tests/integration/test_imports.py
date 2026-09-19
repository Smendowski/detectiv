def test_public_scenario_and_callback_namespaces_import_together() -> None:
    import detectiv.callbacks
    import detectiv.scenarios

    assert detectiv.callbacks.BaseCallback.__name__ == "BaseCallback"
    assert (
        detectiv.scenarios.ReconstructionScenario.__name__ == "ReconstructionScenario"
    )
