from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from detectiv.scenarios.reconstruction import ReconstructionScenarioResult


class ReconstructionCallback:
    @property
    def name(self) -> str:
        raise NotImplementedError

    def on_run_started(self) -> None:
        pass

    def on_epoch_finished(self, epoch: int, loss: float) -> None:
        pass

    def on_run_finished(self, result: "ReconstructionScenarioResult") -> None:
        pass

    def on_run_failed(self, error: BaseException) -> None:
        pass
