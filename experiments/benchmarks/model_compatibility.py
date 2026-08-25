#!/usr/bin/env python3
"""Record local large-model compatibility for the WP1 runner.

This deliberately separates model-format support from I/O performance.  A
model is only promoted to E2/E4 after its configuration is accepted by the
active MindFormers environment; unsupported HuggingFace layouts are recorded
as blockers instead of being silently treated as a successful benchmark.
"""

import argparse
import json
import os
import platform
import time
from pathlib import Path


def probe(name, local_path=None):
    result = {"name": name, "local_path": local_path,
              "path_exists": bool(local_path and Path(local_path).exists())}
    try:
        from mindformers import AutoConfig
        config = AutoConfig.from_pretrained(local_path or name)
        result.update({"status": "config_supported",
                       "config_type": type(config).__name__,
                       "model_type": getattr(config, "model_type", None),
                       "vocab_size": getattr(config, "vocab_size", None),
                       "hidden_size": getattr(config, "hidden_size", None),
                       "num_layers": getattr(config, "num_layers", None),
                       "num_hidden_layers": getattr(config, "num_hidden_layers", None)})
    except Exception as error:  # compatibility probing must be reportable
        result.update({"status": "unsupported", "error": repr(error)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--qwen-path", default="/models/Qwen3-8B")
    args = parser.parse_args()
    result = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "mindspore": None,
        "mindformers": None,
        "models": [],
    }
    try:
        import mindspore
        import mindformers
        result["mindspore"] = getattr(mindspore, "__version__", None)
        result["mindformers"] = getattr(mindformers, "__version__", None)
    except Exception as error:
        result["import_error"] = repr(error)
    result["models"].append(probe("qwen3_local", args.qwen_path))
    result["models"].append(probe("llama2_7b"))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
