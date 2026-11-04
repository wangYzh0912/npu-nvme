"""Export task captures after model workers have terminated."""
import argparse
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trace',type=Path,required=True)
    args=p.parse_args()
    from torch_npu.profiler.profiler import analyse
    traces=[args.trace] if args.trace.name=='trace' else sorted(args.trace.glob('*/trace'))
    if not traces:raise ValueError('no task captures')
    for trace in traces:
        analyse(str(trace))


if __name__=='__main__':main()
