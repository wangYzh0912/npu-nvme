"""Pinned GPT-2 model; no global MindFormers registry changes."""
import json
from pathlib import Path

def configuration(name):
    from .gpt2_config import GPT2Config
    if name not in ('gpt2', 'gpt2_xl'):
        raise ValueError('unsupported pinned model')
    values=json.loads((Path(__file__).with_name('gpt2.json')).read_text())
    for key in ('mindformers_version', 'model_type'):
        values.pop(key, None)
    if name == 'gpt2_xl':
        values.update(hidden_size=1600, num_layers=48, num_heads=25)
    return GPT2Config(**values)

def model(config):
    from .gpt2 import GPT2LMHeadModel
    return GPT2LMHeadModel(config)
