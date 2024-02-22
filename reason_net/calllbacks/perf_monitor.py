from __future__ import annotations
import time

from pydantic import PrivateAttr

from lightning.pytorch.utilities.rank_zero import rank_zero_only
from lightning.pytorch.callbacks import Callback

from typing import TYPE_CHECKING, Any, Type
from reason_net.data import BatchDataPoint

from reason_net.pydantic_conf import Config

if TYPE_CHECKING:
    from reason_net.module import LLaMaModule
    from lightning.pytorch import Trainer


class PerfMonitor(Callback):
    def __init__(self, config: PerfMonitorConfig) -> None:
        super().__init__()
        self.config = config
        self.num_batch_seen = 0
        self.total_sample_seen = 0
        self.total_token_seen = 0

        self.start_time: float | None = None

    @rank_zero_only
    def on_train_batch_end(
        self,
        trainer: "Trainer",
        pl_module: "LLaMaModule",
        _outputs: Any,
        batch: BatchDataPoint,
        batch_idx: int,
        *args,
        **kwargs,
    ) -> None:
        if self.num_batch_seen % self.config.log_every_n_batchs == 0:
            if self.start_time is None:
                self.start_time = time.time()
            else:
                end = time.time()

                time_elapsed = end - self.start_time
                logs = {
                    "perf/sample_per_sec": self.total_sample_seen / time_elapsed,
                    "perf/batch_per_sec": self.num_batch_seen / time_elapsed,
                    "perf/token_per_sec": self.total_token_seen / time_elapsed,
                }

                for metric_name, metric_val in list(logs.items()):
                    logs[f"{metric_name}_per_device"] = metric_val / trainer.world_size

                self.start_time = end

                self.log_dict(logs)

                self.num_batch_seen = 0
                self.total_sample_seen = 0
                self.total_token_seen = 0

        self.num_batch_seen += 1
        self.total_sample_seen += len(batch[0])
        self.total_token_seen += len(batch[0][0])


class PerfMonitorConfig(Config):
    log_every_n_batchs: int
    _target_: Type[Callback] = PrivateAttr(PerfMonitor)
