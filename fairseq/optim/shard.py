# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict

from fairseq.distributed import utils


# try:
#     from fairscale.optim import OSS

#     _has_fairscale = True
# except ImportError:
#     _has_fairscale = False
from torch.distributed.optim import ZeroRedundancyOptimizer


def shard_(optimizer, group):
    # if not _has_fairscale:
    #     raise ImportError(
    #         "\n\nPlease install the fairscale package:" "\n\n  pip install fairscale"
    #     )

    class FairseqOSS(ZeroRedundancyOptimizer):
        @property
        def disable_mem_eff_fp16_loading_hack(self):
            return True

        def __getattr__(self, name):
            if name.startswith("supports") and hasattr(self.optim, name):
                return getattr(self.optim, name)
            raise AttributeError(
                "'FairseqOSS' object has no attribute {0!r}".format(name)
            )

        def broadcast_global_state_dict(
            self, state_dict: Dict[str, Any]
        ) -> Dict[str, Any]:
            """
            Broadcasts the entire state_dict to all other ranks
            each rank is responsible to load their own partition of data
            """
            return utils.broadcast_object(
                state_dict,
                src_rank=0,
                group=self.process_group,
            )
        
        def state_dict(self) -> Dict[str, Any]:
            state_dicts = {
                "local_optim_state": self.optim.state_dict(),
                "global_optim_state": super(ZeroRedundancyOptimizer, self).state_dict(),
                "_partition_parameters_cache": self._partition_parameters_cache,
                "_param_to_index": self._param_to_index,
            }
            return state_dicts

    torch_optimizer = optimizer.optimizer
    optim_cls = type(torch_optimizer)

    optimizer.optimizer = FairseqOSS(
        torch_optimizer.param_groups,
        optim_cls,
        process_group=group,
        **optimizer.optimizer_config
    )
