队友上手：代码协作版
====================

先按根目录README配置Python3.12和requirements-core.txt，然后运行::

    python scripts/run_core_tests.py
    python scripts/smoke_pipeline.py

核心测试和随机初始化的合成数据smoke不需要训练数据或checkpoint。
smoke只验证coarse->fine->坐标的接口，不代表蛋白生成质量。

当前代码
--------

* hierarchical_rigid_vq.py：自训结构encoder/decoder、EMA码本、刚体N/CA/C输出。
* hierarchical_latent_contract.py：每4残基分块、中心池化和局部偏移组合。
* mixed_hierarchical_generator.py：连续coarse flow和条件fine mask生成。
* fixed_pool_geometry_search.py：冻结decoder上的局部token搜索。
* upstream/experiments：已有训练、诊断、评估、分析和绘图脚本。
* upstream/tests：全部现有测试。部分历史测试需要本地实验数据；根目录
  scripts/run_core_tests.py挑选了不依赖实验缓存的核心测试。

这个公开版本上传代码、测试、环境和运行说明；没有上传自训权重、AA特征、
训练数据、内部结果、稿件、图表或集群日志。这些研究产物的公开发布另需明确授权。
原作者公开权重/数据入口在upstream/README.md。

使用共享模型和数据
------------------

有本地访问权限后，用scripts/export_inference_checkpoint.py从可信训练
checkpoint导出精简推理模型。它只写本地文件，默认产物不进Git。之后可用
scripts/demo_representation.py做结构重建，用scripts/sample_generator.py
做序列特征条件采样。具体参数见README；原始大模型权重和完整缓存不在仓库。

旧实验依赖完整的版本化protocol和哈希，有些包含原机器路径。新实验创建
新版本、明确修改数据路径和参数。不要删除哈希检查后称为原实验复现。
当前自训结构表示的coarse保持连续，fine码本1024；作者原路径fine4096，
两条路线的token语义和结果不能直接混用。当前新生成器仍有质量/泛化问题。

可选外部项目以固定commit的submodule记录，按需初始化；frame-flow兼容
patch在docs/patches，原始版权/数据条款随各项目保留。
