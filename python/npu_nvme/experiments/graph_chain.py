"""MindSpore graph-native auxiliary chain; imported only inside the NPU worker."""
import math
import numpy as np
import mindspore as ms
from mindspore import ops, nn, Parameter, ParameterTuple, Tensor
from mindspore.common.initializer import initializer
from mindformers.wrapper.wrapper import MFTrainOneStepCell, _grad_scale
from mindspore.ops import functional as F

from npu_nvme.experiments.graph_geometry import geometry, summarize


class ParameterScan(nn.Cell):
    def __init__(self, row, level, full_scan):
        super().__init__(auto_prefix=False)
        self.unit = row['unit']
        self.count = row['block_count']
        self.level = level
        self.full_scan = full_scan
        self.present = bool(row['elements'])
        self.tiles = Tensor(np.asarray(row['tile_indices'], np.int32))
        self.segments = Tensor(np.asarray(row['segment_ids'], np.int32))
        self.zero = Tensor(np.zeros(self.count, np.float32))
        self.reduce = ops.ReduceSum()
        self.gather = ops.Gather()
        self.segment = ops.UnsortedSegmentSum()

    def construct(self, weight, reference):
        if not self.present:
            return self.zero
        w = ops.reshape(weight, (-1, self.unit))
        r = ops.reshape(reference, (-1, self.unit))
        if not self.full_scan:
            w = self.gather(w, self.tiles, 0)
            r = self.gather(r, self.tiles, 0)
        if self.level == 2:
            partial = self.reduce(w, 1) + self.reduce(r, 1)
        else:
            difference = w - r
            if self.level >= 4:
                difference = difference * difference
            partial = self.reduce(difference, 1)
        return self.segment(partial, self.segments, self.count)


class DetectionChain(nn.Cell):
    def __init__(self, weights, schema, rank, level, fraction, ratio):
        super().__init__(auto_prefix=False)
        if level not in range(1, 7):
            raise ValueError('this detection implementation covers G1 through G6')
        self.level = level
        self.rows = geometry(schema, rank, fraction) if level >= 2 else []
        registry = {p.name: p for p in weights}
        self.first = weights[0]
        selected = [registry[r['name']] for r in self.rows]
        if any(tuple(p.shape) != tuple(r['local_shape']) for p, r in zip(selected, self.rows)):
            raise ValueError('auxiliary weights are not the expected physical TP shards')
        self.sources = ParameterTuple(selected)
        self.references = ParameterTuple([
            Parameter(initializer('zeros', p.shape, ms.float32), name=f'graph_reference_{i}', requires_grad=False)
            for i, p in enumerate(selected)])
        self.scanners = nn.CellList([ParameterScan(r, level, fraction == 1.0) for r in self.rows], auto_prefix=False)
        self.count = sum(r['block_count'] for r in self.rows) if self.rows else 1
        self.k = max(1, math.ceil(self.count * ratio)) if level >= 6 else 1
        self.scores = Parameter(initializer('zeros', (self.count,), ms.float32),
                                name='graph_last_scores', requires_grad=False)
        self.indices = Parameter(initializer('zeros', (self.k,), ms.int32),
                                 name='graph_last_indices', requires_grad=False)
        self.values = Parameter(initializer('zeros', (self.k,), ms.float32),
                                name='graph_last_values', requires_grad=False)
        self.version = Parameter(Tensor(0, ms.int32), name='graph_detection_version', requires_grad=False)
        self.allreduce = ops.AllReduce(ops.ReduceOp.SUM, group='graph_topk_aux_tp4') if level >= 5 else None
        self.select = ops.TopK(sorted=True)
        self.reduce = ops.ReduceSum()
        self.zero_index = Tensor([0], ms.int32)

    def construct(self):
        if self.level == 1:
            scores = ops.reshape(self.reduce(ops.reshape(self.first, (-1,))[:16]), (1,))
        else:
            pieces = ()
            for i in range(len(self.scanners)):
                pieces += (self.scanners[i](self.sources[i], self.references[i]),)
            scores = ops.concat(pieces)
        if self.level >= 5:
            scores = self.allreduce(scores)
        if self.level >= 6:
            values, indices = self.select(scores, self.k)
        else:
            values, indices = ops.reshape(self.reduce(scores), (1,)), self.zero_index
        token = F.assign(self.scores, scores)
        token = F.depend(token, F.assign(self.indices, indices))
        token = F.depend(token, F.assign(self.values, values))
        token = F.depend(token, F.assign_add(self.version, Tensor(1, ms.int32)))
        return self.reduce(token)

    def reset_reference(self):
        # Initialization only, never called in the measured interval.
        for row, weight, reference in zip(self.rows, self.sources, self.references):
            array = weight.asnumpy().astype(np.float32, copy=True)
            array *= np.float32(.99)
            array += np.float32((row['parameter_index'] % 17 + 1) * .00001)
            reference.set_data(Tensor(array))
        self.version.set_data(Tensor(0, ms.int32))


def install_wrapper(options, schema, rank, holder):
    """Override wrapper construction locally; keep original G0 implementation."""
    from mindformers.trainer.base_trainer import BaseTrainer
    original = BaseTrainer.create_model_wrapper

    class GraphTrain(MFTrainOneStepCell):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.use_legacy or self.local_norm or self.dump_device_local_norm or self.use_graceful_exit or self.use_skip_data_by_global_norm:
                raise ValueError('unsupported training wrapper feature; do not silently change training')
            self.chain = DetectionChain(self.weights, schema, rank, options['level'],
                                        options['scan_fraction'], options['ratio'])
            self.serial_aux = options['layout'] == 'serial'
            self.skip_aux_step = options['warmup_steps']
            holder['chain'] = self.chain

        def construct(self, *inputs):
            scaling_sens = self.scale_sense
            # A_(t-1) reads weights from the preceding completed update.
            # Skip only the first formal update; its A is in the next graph.
            aux = Tensor(0.0, ms.float32)
            if self.optimizer.global_step != self.skip_aux_step:
                aux = self.chain()
            if self.serial_aux:
                scaling_sens = F.depend(scaling_sens, aux)
                inputs = F.depend(inputs, aux)
            loss, grads, grad_scale_factor = self.grads_for_mcore(scaling_sens, *inputs)
            status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
            grads = self.hyper_map(F.partial(_grad_scale, scaling_sens * grad_scale_factor), grads)
            grads = self.grad_reducer(grads)
            cond = self.get_overflow_status(status, grads)
            overflow = self.process_loss_scale(cond)
            global_norm = None
            if self.use_clip_grad:
                grads, global_norm = self.clip_grad_norm(grads)
            learning_rate = self.learning_rate
            if self.optimizer.dynamic_lr:
                if self.optimizer.is_group_lr:
                    learning_rate = self.learning_rate[-1](self.optimizer.global_step).reshape(())
                else:
                    learning_rate = self.learning_rate(self.optimizer.global_step).reshape(())
            # The only requested completion edge for the parallel variant:
            # A_(t-1) -> U_t. No added auxiliary -> forward/backward edge.
            grads = F.depend(grads, aux)
            if not overflow:
                loss = F.depend(loss, self.optimizer(grads))
            return loss, overflow, scaling_sens, learning_rate, global_norm

    def create(trainer, network, optimizer):
        if options['level'] == 0:
            wrapper = original(trainer, network, optimizer)
        else:
            from mindformers.tools.register import MindFormerRegister, MindFormerModuleType
            from importlib import import_module
            builder = import_module('mindformers.wrapper.build_wrapper')
            MindFormerRegister.register_cls(GraphTrain, MindFormerModuleType.WRAPPER)
            builder.WRAPPERS_MINDFORMERS_DEFINED.append('GraphTrain')
            trainer.config.runner_wrapper.type = 'GraphTrain'
            wrapper = original(trainer, network, optimizer)
        holder['wrapper'] = wrapper
        return wrapper
    BaseTrainer.create_model_wrapper = create
