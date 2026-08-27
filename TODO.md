- [x] Zero-2 All-Gather 与数据加载操作重叠
    - Megatron 默认逻辑
  
- [ ] Zero-2 Reduce-Scatter 基于优先级排序
    - 这个很麻烦。不同的 Reduce-Scatter，得再调研一下

- [x] PP 异步 Send/Resv
    - 默认开启，有一个 --no-p2p-comm-overlap 参数显式指定关闭

- [ ] TP GEMM 与 All-Gather 重叠
    - 这个要重构一下 Megatron 的 PP schedule.py 和 TP 的逻辑

- [ ] 监控全局通信模式，定位通信瓶颈
    - 想一下咋做。任务书没有要求运行时完成，可以训练后分析一下，主要瓶颈应该还是在跨节点的通信上。

- [x] 对通信瓶颈进行通信压缩
    - 主要就是 Reduce-Scatter 和 All-Gather 两个操作。

- [x] PP 提前对完成 Backward 的 Param Chunk 进行 DP
    - 这是 PP 的默认逻辑。对于任意一个 PP 节点，只要本地的 Backward 完成，就会对完成的 Param Chunk 进行 DP。

- [x] 根据统计特征进行等概率分布通信量化
    - 没有说要在线完成。之后分析一下统计特征，调整量化策略

- [x] 对关键信息进行更精细的压缩
    - 稀疏化+量化

- [x] 误差反馈