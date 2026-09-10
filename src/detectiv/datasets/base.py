from abc import ABC, abstractmethod
from collections.abc import Mapping


class Dataset[K, T](ABC):
    def __init__(
        self,
        dataset_id: str,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if not dataset_id:
            raise ValueError("dataset_id must not be empty")

        self.dataset_id = dataset_id
        self.metadata = dict(metadata or {})

    @abstractmethod
    def __len__(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, key: K) -> T:
        raise NotImplementedError
