from pathlib import Path
import json
from mindspore.parallel import build_searched_strategy
from google.protobuf.json_format import MessageToDict
root=Path('/models/npu_nvme_exp/user7-stack/qwen3-v13-source-002/training/strategy')
result={}
for rank in range(4):
    mapping=build_searched_strategy(str(root/f'ckpt_strategy_rank_{rank}.ckpt'))
    result[str(rank)]={name:MessageToDict(value,preserving_proto_field_name=True) for name,value in mapping.items()}
Path('/tmp/v13-strategy-layouts.json').write_text(json.dumps(result,indent=2)+'\n')
print({k:len(v) for k,v in result.items()})
print(list(result['0'].items())[:3])
