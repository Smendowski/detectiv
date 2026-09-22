import random

import numpy as np
import torch

from detectiv.runs import ReproducibilitySettings, configure_reproducibility


def test_reproducibility_settings_seed_supported_random_generators() -> None:
    settings = ReproducibilitySettings(seed=42, deterministic_algorithms=True)

    settings.apply()
    first = (random.random(), np.random.random(), torch.rand(1).item())
    settings.apply()
    second = (random.random(), np.random.random(), torch.rand(1).item())

    assert first == second
    assert torch.are_deterministic_algorithms_enabled()


def test_reproducibility_settings_record_execution_configuration() -> None:
    settings = ReproducibilitySettings(seed=42, deterministic_algorithms=True)

    record = settings.record(device="cpu")

    assert record["seed"] == 42
    assert record["deterministic_algorithms"] is True
    assert record["device"] == "cpu"


def test_reproducibility_settings_apply_is_seeded_end_to_end() -> None:
    settings = ReproducibilitySettings(seed=7)

    settings.apply()
    first = torch.rand(3)
    settings.apply()
    second = torch.rand(3)

    torch.testing.assert_close(first, second)


def test_configure_reproducibility_seeds_model_initialization() -> None:
    settings = configure_reproducibility(seed=7)
    first = torch.nn.Linear(3, 2)

    configure_reproducibility(seed=7)
    second = torch.nn.Linear(3, 2)

    assert settings == ReproducibilitySettings(seed=7)
    torch.testing.assert_close(first.weight, second.weight)
    torch.testing.assert_close(first.bias, second.bias)
