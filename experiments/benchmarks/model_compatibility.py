#!/usr/bin/env python3
"""Record local large-model compatibility for the WP1 runner.

This probes configuration only. Loading, training, full restart and live I/O
remain untested until their independent hardware gates have actually run.
"""

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path


def probe(name, local_path=None):
    result = {"name": name, "local_path": local_path,
              "path_exists": None, "environment_id": os.environ.get("NPU_NVME_ENVIRONMENT_ID"),
              "stages": {stage: "not_run" for stage in (
                  "config", "load", "train", "full_restart", "live")}}
    try:
        result["path_exists"] = bool(local_path and Path(local_path).exists())
    except OSError as error:
        result["path_error"] = repr(error)
    # A local HuggingFace directory can be inspected even when the active
    # MindFormers release cannot construct the model.  Keep these facts
    # separate so a format/config match is never reported as trainability.
    if local_path:
        config_path = Path(local_path) / "config.json"
        try:
            raw = config_path.read_bytes()
            config = json.loads(raw.decode("utf-8"))
            result["local_config"] = {
                key: config.get(key) for key in (
                    "model_type", "architectures", "hidden_size",
                    "num_hidden_layers", "num_attention_heads",
                    "num_key_value_heads", "vocab_size", "torch_dtype")
            }
            result["local_config_sha256"] = hashlib.sha256(raw).hexdigest()
            result["local_config_status"] = "parsed"
        except FileNotFoundError:
            result["local_config_status"] = "missing"
        except (OSError, ValueError) as error:
            result["local_config_status"] = "unreadable"
            result["local_config_error"] = repr(error)
    try:
        from mindformers import AutoConfig
        try:
            config = AutoConfig.from_pretrained(local_path or name)
        except Exception:
            # MindFormers 1.7 experimental mode does not accept a raw
            # HuggingFace directory.  Qwen3Config is the supported native
            # entry point; translate the local JSON into that config while
            # retaining the raw-file digest above.
            if local_path and result.get("local_config", {}).get("model_type") == "qwen3":
                from mindformers.models.qwen3 import Qwen3Config
                config = Qwen3Config.from_pretrained(str(Path(local_path)))
            else:
                raise
        result["stages"]["config"] = "pass"
        result.update({"status": "config_supported",
                       "runtime_config_status": "supported",
                       "config_type": type(config).__name__,
                       "model_type": getattr(config, "model_type", None),
                       "vocab_size": getattr(config, "vocab_size", None),
                       "hidden_size": getattr(config, "hidden_size", None),
                       "num_layers": getattr(config, "num_layers", None),
                       "num_hidden_layers": getattr(config, "num_hidden_layers", None)})
    except Exception as error:  # compatibility probing must be reportable
        result["stages"]["config"] = "fail"
        result.update({"status": "config_failed",
                       "runtime_config_status": "failed",
                       "error": repr(error)})
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
