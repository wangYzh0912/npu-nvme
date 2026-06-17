#!/usr/bin/env python3
"""Phase 1a inject test — small param counts to find the safe ceiling.
Usage: python phase1a_inject.py --inject <N> --label <label>
"""
import os, sys, time, json, math, argparse
REPO = "/home/user7/npu-nvme"
sys.path.insert(0, os.path.join(REPO, "python"))
import numpy as np
import mindspore as ms
from mindspore import nn, Tensor, ops
ms.set_recursion_limit(10000)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inject", type=int, default=50)
    parser.add_argument("--label", default="A2")
    parser.add_argument("--sink", type=int, default=4)
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args()

    DEVICE_ID, SEQ_LEN = 1, 1024
    EPOCHS = args.steps // args.sink

    ms.context.set_context(mode=ms.GRAPH_MODE, device_target="Ascend", device_id=DEVICE_ID)
    ms.common.set_seed(42)

    from mindformers import AutoModel, AutoConfig
    cfg = AutoConfig.from_pretrained("gpt2_xl")
    cfg.seq_length = SEQ_LEN; cfg.max_position_embeddings = SEQ_LEN
    model = AutoModel.from_config(cfg)

    ds = ms.dataset.MindDataset(
        REPO + "/dataset_prepare/gpt2/wikitext2_data/gpt2_train_1025.mindrecord",
        shuffle=True)
    ds = ds.batch(1, drop_remainder=True).take(args.steps)

    opt = nn.AdamWeightDecay(model.trainable_params(), learning_rate=1e-5)
    all_params = list(model.trainable_params())
    n_total = len(all_params)
    covered = all_params[:args.inject] if args.inject > 0 else []
    n_inject = len(covered)

    num_groups = max(1, min(math.ceil(n_inject / 100), 10)) if n_inject > 0 else 0
    param_groups = []; fp16_needed = []
    if n_inject > 0:
        gs = max(1, math.ceil(n_inject / max(num_groups, 1)))
        for g in range(num_groups):
            s = g * gs; e = min(s + gs, n_inject)
            if s < n_inject:
                pg = covered[s:e]
                param_groups.append(pg)
                fp16_needed.append([
                    hasattr(p, 'dtype') and p.dtype != ms.float16 for p in pg
                ])

    total_elems = sum(int(np.prod(p.shape)) for p in covered) if args.inject else 0
    print("[{}] Total={} params, Inject={}, {:.2f}B elems, {} groups".format(
        args.label, n_total, n_inject, total_elems/1e9, num_groups), flush=True)

    class ProfiledCell(nn.Cell):
        def __init__(self, network, optimizer, pg, fn, inj):
            super().__init__(auto_prefix=False)
            self.network = network; self.network.set_grad()
            self.optimizer = optimizer
            self.grad_fn = ops.value_and_grad(self.network, grad_position=None,
                                               weights=self.optimizer.parameters)
            self.depend = ops.Depend()
            self.pg = pg; self.fn = fn; self.inj = inj
        def construct(self, *inputs):
            loss, grads = self.grad_fn(*inputs)
            if self.inj:
                acc = Tensor([0.0], dtype=ms.float16)
                for gi, group in enumerate(self.pg):
                    flags = self.fn[gi]
                    flat_parts = []
                    for pi, p in enumerate(group):
                        pv = ops.Cast()(p, ms.float16) if flags[pi] else p
                        flat_parts.append(ops.Reshape()(pv, (-1,)))
                    flat = flat_parts[0] if len(flat_parts)==1 else ops.Concat()(tuple(flat_parts))
                    delta = ops.Sub()(flat, ops.ZerosLike()(flat))
                    red = ops.ReduceSum()(delta)
                    c32 = ops.Cast()(red, ms.float32)
                    c16 = ops.Cast()(c32, ms.float16)
                    acc = ops.Add()(acc, c16)
                loss = self.depend(loss, acc)
            opt_res = self.optimizer(grads)
            return self.depend(loss, opt_res)

    t_build = time.perf_counter()
    cell = ProfiledCell(model, opt, param_groups, fp16_needed, args.inject > 0)
    ms_model = ms.Model(cell)
    build_s = time.perf_counter() - t_build
    print("[{}] Build={:.1f}s".format(args.label, build_s), flush=True)

    epoch_times_ms = []
    class CB(ms.Callback):
        def on_train_epoch_begin(self, rc):
            self.t0 = time.perf_counter()
        def on_train_epoch_end(self, rc):
            epoch_times_ms.append((time.perf_counter() - self.t0) * 1000)

    print("[{}] Training {} steps...".format(args.label, args.steps), flush=True)
    t_total = time.perf_counter()
    compiled_ok = True; error_msg = None
    try:
        ms_model.train(epoch=EPOCHS, train_dataset=ds, callbacks=[CB()],
                       dataset_sink_mode=True, sink_size=args.sink)
    except Exception as e:
        compiled_ok = False; error_msg = str(e)[:300]
        print("[{}] FAILED: {}".format(args.label, error_msg), flush=True)

    total_s = time.perf_counter() - t_total
    compile_epoch = epoch_times_ms[0] if epoch_times_ms else 0
    warm_epochs = epoch_times_ms[1:] if len(epoch_times_ms) > 1 else []
    avg_step = sum(warm_epochs)/len(warm_epochs)/args.sink if warm_epochs else 0

    print("[{}] compile={:.0f}ms  warm={}  avg_step={:.0f}ms".format(
        args.label, compile_epoch, [round(e,0) for e in warm_epochs], avg_step), flush=True)

    result = {
        "test": args.label, "total_params": n_total, "inject_params": args.inject,
        "inject_elems_B": round(total_elems/1e9, 3), "num_groups": num_groups,
        "sink_size": args.sink, "total_steps": args.steps, "epochs": EPOCHS,
        "compiled_ok": compiled_ok, "error": error_msg,
        "build_s": round(build_s, 1), "total_wall_s": round(total_s, 1),
        "compile_epoch_ms": round(compile_epoch, 0),
        "warm_epochs_ms": [round(et, 0) for et in warm_epochs],
        "avg_step_ms": round(avg_step, 1),
    }
    os.makedirs(REPO + "/experiments/output", exist_ok=True)
    out = REPO + "/experiments/output/phase1a_{}.json".format(args.label.lower())
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print("[{}] -> {}".format(args.label, os.path.basename(out)), flush=True)

if __name__ == "__main__":
    main()
