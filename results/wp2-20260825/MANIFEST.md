# WP2 evidence manifest

生成基线：`exp/wp2-ablation`，证据包提交时的代码基线为 `5503fdd`；后续如继续
补跑，必须在对应结果目录中记录新的 commit。

| 目录 | 内容 |
|---|---|
| `evidence/model_13b/` | GPT-2 13B：A1 P2 FS、A2 P3/P4、A3 scalar、A4 depth=1/2/4/8/16、A5 1/16 MiB、A10 node2/node4 |
| `evidence/model_xl/` | GPT-2 XL：A1、A2、A3 scalar、A7 FaF；另含 A8 正式协议结果 |
| `evidence/synthetic/` | A3、A4、A5、A6、A9 的结果子集；`a6_formal/` 为 sync API/request-ring 对照，`a9_formal_slots_*` 为 host slot 生命周期 |
| `evidence/failures/` | A4 depth=16 修复前读失败；A4 depth=2 的 npu-smi 误判 |

证据筛选规则：只复制结构化结果和时间线，不复制模型 payload、SPDK hugepage
临时文件或未审阅的全量原始目录；失败样本不进入 PASS 统计，但保留失败 JSON
用于解释修复和重跑关系。
