队友上手：代码协作版
====================

先按根目录README配置Python3.12和requirements-core.txt，然后运行::

    python scripts/run_core_tests.py
    python scripts/smoke_pipeline.py

核心测试和随机初始化的合成数据smoke不需要训练数据或checkpoint。
smoke只验证coarse->fine->坐标的接口，不代表蛋白生成质量。

先看进度
--------

当前完整说明是 RESEARCH_STATUS.rst（英文）和 PROGRESS_2026-11-24_ZH.rst
（中文）；progress_2026-11-24.json 保存选定的汇总数字和来源哈希。

原模型上不改权重的 fine-token 搜索有独立键角收益。自训多尺度表示和 PAR
迁移还在原型阶段；最新手性辅助损失复验一好一坏，没有采用候选权重。
11 月 24 日核对时没有新训练正在跑，也没有新实验已提交。

当前与历史代码
--------------

* sequence_multiscale_ensemble.py：当前物理 coarse 中心 M×3、连续 fine N×8。
* block_stable_frame_tokenizer.py：coarse 导出的局部标架、fine 与刚体骨架输出。
* hierarchical_ensemble_flow.py / explicit_coarse_fine_flow.py：coarse 先采样，
  fine 在同一个 coarse 条件下采样；当前生成路径不经过 fine VQ/snap。
* par_coarse_backbone.py：公开 PAR 的 coarse 接口与 AA adapter，尚非完整
  L/4→L/2→L 序列条件 PAR。初始化依赖和运行边界见 PAR_REPRODUCTION.rst。
* par_coarse_capacity.py、audit_par_coarse_chirality.py、par_signed_volume.py、
  par_signed_volume_replication.py：最近已完成的容量/手性检查与负向复验。
* hierarchical_rigid_vq.py：自训结构encoder/decoder、EMA码本、刚体N/CA/C输出。
* hierarchical_latent_contract.py：每4残基分块、中心池化和局部偏移组合。
* mixed_hierarchical_generator.py：早期连续 16D coarse 和条件 fine mask 生成。
* fixed_pool_geometry_search.py：冻结decoder上的局部token搜索。
* upstream/experiments：已有训练、诊断、评估、分析和绘图脚本。
* upstream/tests：全部现有测试。部分历史测试需要本地实验数据；根目录
  scripts/run_core_tests.py挑选了不依赖实验缓存的核心测试。

这次更新同步代码、测试、环境说明和经用户授权的进度汇总。权重、AA 特征、
训练数据、逐样本结果、稿件、图表和集群日志留在本地/共享目录。
原作者公开权重/数据入口在 upstream/README.md；PAR 权重入口见 PAR 说明。

使用共享模型和数据
------------------

有本地访问权限后，用scripts/export_inference_checkpoint.py从可信训练
checkpoint导出精简推理模型。它只写本地文件，默认产物不进Git。之后可用
scripts/demo_representation.py做结构重建，用scripts/sample_generator.py
做序列特征条件采样。这些便携入口对应旧的 16D coarse / 离散 fine 模型，
不是当前连续 fine/PAR 分支。具体参数见 README；原始权重和缓存不在仓库。

旧实验依赖完整的版本化protocol和哈希，有些包含原机器路径。新实验创建
新版本、明确修改数据路径和参数。不要删除哈希检查后称为原实验复现。
当前 coarse 为真实块中心坐标，fine 为连续局部表示；旧 fine 码本 1024，
作者原路径 fine 码本 4096。它们的语义、信息率、指标和 checkpoint 不能混用。
当前新生成器仍有质量/泛化问题，两个训练结构上的 coarse 拟合不算系综确认。

可选外部项目以固定commit的submodule记录，按需初始化；frame-flow兼容
patch在docs/patches，原始版权/数据条款随各项目保留。
