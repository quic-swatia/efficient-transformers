# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

import random
from itertools import islice

import numpy as np
import torch

class LengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, batch_size: int, num_replicas: int, drop_last: bool, shuffle: bool = True) -> None:
        if isinstance(next(iter(data_source)), dict):
            first_key = next(iter(next(iter(data_source)).keys()))
            self.lengths = [len(d[first_key]) for d in data_source]
        else:
            self.lengths = [len(d) for d in data_source]
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.data_source = data_source

    def __iter__(self):
        ids = np.argsort(self.lengths, kind="mergesort")
        if self.drop_last:
            ids = ids[: len(ids) // self.batch_size * self.batch_size]
        else:
            padding_size = self.num_replicas - len(ids) % self.num_replicas
            padded_indices = [i for i in range(len(self.lengths), len(self.lengths)+ padding_size)]
            ids = np.concatenate((ids, padded_indices))
            
            
            dummy_row = next(iter(self.data_source))
            for i in range(len(padded_indices)):
                self.lengths.append(len(dummy_row['labels']))
            dummy_row['labels'] = [-100]*len(dummy_row['labels'])
            dummy_row = {k: v.tolist() if isinstance(v, torch.Tensor) else v for k, v in dummy_row.items()}
            
            
            self.data_source.add_item(dummy_row)


        batches = [ids[i : i + self.batch_size] for i in range(0, len(ids), self.batch_size)]

        # if self.shuffle:
        #     random.shuffle(batches)

        for b in batches:
            yield b

    def __len__(self):
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        else:
            return len(self.lengths) // self.batch_size + (len(self.lengths) % self.batch_size > 0)

class DistributedLengthBasedBatchSampler(torch.utils.data.BatchSampler):
    def __init__(
        self, data_source, batch_size: int, num_replicas: int, rank: int, shuffle: bool = True, seed: int = 0
    ) -> None:
        random.seed(seed)
        self.batch_sampler = LengthBasedBatchSampler(
            data_source, batch_size=batch_size, num_replicas = num_replicas, drop_last=False, shuffle=shuffle
        )
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        max_length = len(self.batch_sampler) // self.num_replicas * self.num_replicas
        return islice(self.batch_sampler, self.rank, max_length, self.num_replicas)

    def __len__(self):
        return len(self.batch_sampler) // self.num_replicas
