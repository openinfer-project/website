# Green Contexts

[https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/green-contexts.html)

在 CUDA 12.4 的时候， Green Contexts 发布，作用就是切分 GPU 的资源，然后应用层面仍然通过 stream/kernel 去工作，区别只是提交的 stream 由 Green Context 创建，此时是在 CUDA Driver API。在 CUDA 13.1，在 Runtime API 中暴露，可以简单理解为 Green Context 逐步稳定，并开始提供更方便的接口。在 CUDA 13.3，cuBLAS 开始适配 Green Context，会考虑 Green Context 切分 SM 后的限制。

它能切分两种 GPU 资源，一种是 SM 另一种是 WQ，前者可以简单理解为 GPU 核心，限制某个负载能使用的核心数，给其他负载腾出空间处理，避免因为类似大任务吃满 GPU，小任务无法执行的情况。

![image.png](/blog/green-ctx/image.png)

NVIDIA 这个图很直观，因为有天生物理限制，任务 B 到来的时候能有资源执行，加快完成时间。

听到这里读者可能好奇了，看起来 Green Context 是用来隔离多个负载公平性的。用在推理里，也许它能隔离 Decode 和 Prefill，让 Decode 稳定。但看上图，这样造成了 SM 空置，总体性能会回退。我们怎么用 Green Context 完成“吞吐提升”，吞吐提升来自于什么呢？

### Green Contexts: 非常易用

Green Context 的 API 复杂度主要在创建时候，创建流程大致就是，配置 SM 切分，比如一组可以用 16 个 SM，另一组只可以用 8 个，分配，然后创建 GreenCtx，最后基于 GreenCtx 创建 Stream 即可，比如这里创建了两组，我们就得到两个 CUDA stream，一个 stream 只能用 16 个 SM，另一个 stream 只能用 8 个 SM，这个 stream 和普通 stream 没有其他区别，可以正常提交 kernel，可以捕获 CUDA Graph，所以对于应用层是完全需要其他额外适配的。唯一需要做的事情就是适配创建分配流程，大概七八行代码。

适配点也不是完全没有，文档给了第一个，Thread Block Clusters。这个从 Hopper 架构开始引入的特性，新增了一个 cluster 的层级，同一个 cluster 里的几个 block 保证被同时调度，并且保证物理上落在相邻的 SM 上，这样可以用 DSMEM。到了 Blackwell，引入了 TMEM 和 `tcgen05.mma`，一个 cluster 一般等于 2，也就是两个 block。当然可能有些 kernel 会把 cluster 开得很大，然后 Green Context 又是边角料切出来的，这里可能会存在一个 corner case，要用 `cudaOccupancyMaxPotentialClusterSize` double check。

同时一个程序可以拥有多个 Green Context。

写到最后，NVIDIA 给了 Green Context 能带来性能提升的例子。baseline 是 two stream，先在 stream1 启动一个长时间计算的内核 kernel1，然后在 stream2 启动一个短但是延迟敏感的内核 kernel2，然后对 Green Context 切分（16 SM 给 stream2，112 SM 给 stream1）后的两个 stream 进行相同压测。

普通 two stream，kernel1 执行了 10 ms，kernel2 等待了 1 ms 才执行，执行时间只需要 50 us。

Green Context 切分后，kernel1 需要执行 12 ms（因为 SM 变少了），但 kernel2 端到端只需要 95 us 就结束了。

### 推理视角的思考: Prefill vs Decode

干扰，干扰还是干扰。P/D 分离的出发点就是让 Prefill 不打断 Decode，从而获得稳定的 TPOT，用户输出不会因为其他人的长请求进入，而造成输出短暂中断。

这不就是 Green Context 所强调的场景吗，我可以按照 NV 那个实验一样，把大部分 SM 给 Prefill，剩下一些给 Decode，这样不就既能保证 TPOT 稳定了吗，虽然 TTFT 可能会变差（但 P/D 分离也会啊，要传输，要重新调度，P/D 分离也有不小的 TTFT overhead）。

但问题在于，就像之前那个图那样，这样会不会导致 SM 空置，总吞吐下降呢。

我们从一个实验开始。和 NVIDIA 的实验一样，我们创建两个 stream，一个用于做 Prefill 的计算密集负载，另一个用来做 Decode 的内存密集型负载，然后向这两个 stream 各自提交 100 个 kernel。对比 two stream vs Green Context two stream vs 单 stream 两个 kernel 交替执行。

我们分别在 5070 Ti 和 H200 上进行测试（两者 kernel 负载稍微需要调整一下）

首先是 5070 Ti 我的消费卡的性能：

1. 单 stream 两个 kernel 交替执行：耗费 1860 ms
2. naive two stream: 1812 ms，快了 3%
3. Green Context two stream: 1173 ms，比单 stream 快了接近 60%

H200 上的性能

1. 单流：4717 ms
2. naive two stream: 4446 ms，快了 6%
3. Green Context: 3522 ms，快了 34%

在 naive two stream 下，如果给 memory bound kernel 的 stream 开启高优先级调度，调优 workload 构造，能再快百分之几的幅度，但还是够不到 Green Context.

直接看 profile，单 stream 下是这样的，两个 kernel 交替执行，符合预期

![image.png](/blog/green-ctx/image%201.png)

naive two stream，也是两个 kernel 交替执行，

![image.png](/blog/green-ctx/image%202.png)

来到 Green Context，两个 kernel 就变成了并发执行，同时 compute kernel 因为受到 SM 变少的限制，耗时上涨 7ms → 11ms。

![image.png](/blog/green-ctx/image%203.png)

为什么两个 kernel 并行执行就能带来性能提升呢？

two stream 的 kernel 为什么没有并行执行？猜想可能是大家默认写 kernel 的时候 grid 很大，直接打满了 GPU 的调度队列，而 GPU 调度队列是 FIFO 的，没有抢占，也没有 work stealing 什么的。实际上也可以压低 kernel 的 SM 占用，算是某种“软分区“，这样会让 two stream 收益稍好一些，但依旧够不到 Green Context。

再激进一点就是 Megakernel，启动一个大 Kernel 调度所有 SM 的任务，可以实现自己的调度逻辑，比如 DeepSeek 的 MegaMOE。实际上社区里 NV 的人也给了用 Green Context 的 MegaMOE 版本，基于 Blackwell：[https://github.com/deepseek-ai/DeepGEMM/pull/357](https://github.com/deepseek-ai/DeepGEMM/pull/357), 性能并不比 mega 的差很多，甚至数据量大的时候有一些加速。

站在某种角度思考，Green Context 是另一种形式的 Megakernel?

### Green Context 性能提升的本质： SM 经济学

要解释 Green Context 为什么能带来性能提升，我们需要分别看 Decode kernel 和 compute kernel 随 SM 扩展时，执行时间的变化曲线。

![kernel_sm_scaling.png](/blog/green-ctx/kernel_sm_scaling.png)

H200 上会稍微变化一下，需要更多 SM（也许是我 kernel 写的一般）。

![kernel_sm_scaling_h200_gpu7.png](/blog/green-ctx/kernel_sm_scaling_h200_gpu7.png)

在 5070 Ti 下，memory bound 的 kernel，在 16 个 SM 后就不随 SM 增加而扩展了，已经打到理论带宽的 80%+（当然这个 case 有点极端），而 compute kernel 几乎是随着 SM 增长，执行时间按照理想预期那样减少。H200 则需要更多 SM，48 个 SM 才接近某个平台期，后续增加 SM 还能有细微提升。

我们形式化定义这个问题， 

 输入

- GPU 有 S 个 SM，
- 有 n 种 workload，每种 workload 有一个先验知识，t_i(s)（给 workload i 分配 s 个 SM 时，单个 kernel 的执行时间）
- 每个 workload 有 N_i 个 kernel，

输出

- 一个**分配方案** {*s*1,*s*2,…,*sn*}，满足 ∑*i*=1*nsi*≤*S*，以及每个 *si* 满足对齐/粒度约束
- 一个**执行顺序**（或并发编排），决定哪些 kernel 同时运行

目标最小化 makespan C_max = max_i Ci，其中 Ci 是 kernel i 的完成时间

这是一个什么类型的问题呢？我们使用 OpenRouter 的 fusion 功能，让 opus, GPT, Gemini 各自最强模型共同讨论得出一个结论。

在看 LLM 老师们的结论之前，我们先过一个叫做 PARTITION（划分问题）的数学问题。假设我和你去搬运箱子，箱子有大有小，需要的时间不同，但一个箱子只能一个人搬，所有箱子都搬完后我俩一起下班。

直觉上很简单，我俩干差不多一样的活就好。比如 10 个一样大的箱子，我 5 个，你 5 个，谁也没有浪费时间，一起下班。但如果箱子多了呢？另一个直觉是，我按箱子大小排个序，累计工作量，直到刚好跨过总工作的二分之一就好。但问题是，这个解可能也不错，但它不是最优解。

比如  5 5 4 4 3 3，总工作量是 24，理论上我 12，你 12 就好，也就是你我各自推一个 3,4,5 的箱子。如果按照排序来，则会分配成，3,3,4,4，我 14，你 10 了，虽然也不差，但不是**最优解**。当然还有各种启发式的贪心策略，包括人数从 2 变成 3。两个人可能还是个跷跷板，三个人开始就变成了一个三角形的木皮了，问题就变成了 strongly NP-hard 了。

BTW，这是数学家的视角。程序员的视角我们怎么看待这个问题，或者说我们其实也需要处理这种问题，我们是怎么 handle 的呢，我们有时候连箱子大小都不知道，我们怎么做的平衡呢，work stealing。

回到我们的问题，“如何编排 SM 使所有 kernel 最快完成”，LLM 老师们达成共识，这是运筹学里的可塑任务并行调度（Moldable Parallel Task Scheduling)，是一个二维的 strip packing：每个 kernel 是宽 =si(SM 数)、高 =ti(si)(时间)的矩形，要在宽 S 的带子里无重叠堆放，最小化总高度。它一般是 NP-hard。

静态的话，需要用 ILP 等价去求解，是不是听着很高端，实际上是分组背包问题。

有 n 组物品（每个 kernel 是一组），背包容量 V = S（总 SM 数）。第 i 组里的第 s 个物品，体积是 s，代价是 `c_{i,s}`。每组必须恰好选一个物品，求塞进背包后总代价最小。

时间复杂度就是 kernel 数量 * SM 分配方案的平方。

我们继续回到 Green Context 上。我们现在找到了一个合适的 SM 切分比例（或者暴力搜的），那我们理想的时间就是最慢那个，ideal = max(solo_decode@memSM, solo_prefill@compSM)，但实际上是多少呢？

在 compute kernel 计算密度高的时候，影响不大，基本上就是 ideal 值，当 compute kernel 计算密度下降时，开始与理想值拉开最多 10% 的差值。这是为什么呢？

我们 profile 了两个 kernel，指标类似取了两者之间的平均值，比如 compute kernel 单独跑 SM 利用率 90%，memory SM 20% 左右，两者并发跑 85% 左右；DRAM、L2 hit 都有下降，说不准谁是影响更大的那位。

单独跑 compute kernel 时，计算密度高的时候 L2 hit 更低，计算密度低的时候 L2 hit 更高，也就意味着计算密度低的时候，compute kernel 的时间可能和 L2 hit 存在某种正向相关性。

我们做一个对比实验，还是 compute kernel 和一个 memory kernel，只不过我们把 memory kernel 的 working set 缩小（I/O 范围减少），从扫 4GB 变成扫 4MB。

1. 低计算密度 compute kernel： 当 Decode 在扫 4GB 的时候，16.6ms，只扫 4MB 的时候，12.99 ms，接近理想值。
2. 而高计算密度的 compute kernel 在两种形态下都不受影响。

所以按照这个对比实验，我们可以证明一个点：Decode 因为在读取很多数据，冲刷了 L2 cache，导致低计算密度 compute kernel 性能下降很多（也许 NVIDIA 需要做个 LRU-K、TinyLFU 什么的）。

但现实问题是，我们不可能这么缩减 Decode kernel 的扫描量，模型 Decode 一个 token，就是需要读取完整权重。

在 CPU 侧，有时候我们也希望读取大块数据的时候不影响正常运作的 CPU cache。`PREFETCHNTA`（Prefetch Non-Temporal Access）的设计目标就是：我只读这块数据一次，别给 CPU cache 大幅度污染了。本质上是个 hint，CPU 怎么干就不知道了。

[https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#cache-operators) 与之相对的，NV 的 PTX 也给了几种，

加载侧

1. .ca： cache at all levels, likely to be accessed again，默认的
2. .cg: cache at global level, cache in L2 and below not L1，不污染 L1
3. .cs: cache streaming: likely to be accessed once，在 L1 和 L2 中 evict first。
4. .lu: last use. 
5. .cv: don’t cache and fetch again. 听着像是驱逐 L2 cache 的.

Store 侧大致也类似，只不过没有.lu。

在 SM70，可以给 L1 缓存提示，evict normal, evict first, evict last, evict unchanged, no allocate（不经过缓存），其实就是普通、高优、低优、不过缓存。叫 hint 说明，和 CPU 一样，硬件怎么干的无从得知。Support for `.level2::eviction_priority` qualifier and `.v8.b32`/`.v4.b64` require `sm_100` or higher，在 Blackwell 才可以 L2 的 evict first。

对 Decode 的 cache policy 做枚举后可以看到：并发时 compute 从默认 .ca 的 24.2ms 出发，.cg 恶化到 25.8ms，.cs、L1::no_allocate / L1::evict_first 和 Blackwell L2 evict_first 都能改善到约 22.3-22.5ms，而
L2 evict_last 退化到 24.7ms，说明 Decode 这种流式扫描应当被标成 streaming/evict-first，避免把一次性数据留在 cache 里污染 compute 的热数据。

做了这个优化后

range                                 time    DRAM%    L2 hit      SM%
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ━━━━━━━━━━━  ━━━━━━━  ━━━━━━━━  ━━━━━━━
solo_compute                     15.525 ms    23.14     20.49    83.40
───────────────────────────────  ───────────  ───────  ────────  ───────
concurrent_normal                16.892 ms    32.39     25.54    76.65
───────────────────────────────  ───────────  ───────  ────────  ───────
concurrent_cs (__ldcs/__stcs)    16.798 ms    34.57     23.32    77.08

### OpenInfer 适配 Green Context

经常我提到 Green Context 的时候，大家常问的一个问题，如果给 Decode 20%， Prefill 80% 的 SM 分配比例。那如果某个时刻，系统只有 Prefill 或者 Decode 了，那不是浪费了 SM。即使 Prefill 和 Decode 一直稳定来，那也只能 overlap 一个 batch，Prefill 可能几百 ms，而 Decode 只有几十 ms，收益可能有几 ms，没什么必要。

OpenInfer 设计如下，首先我们 SM 切分是静态的，比如 P:D 是 8:2（不同 workload，不同硬件下有不同的比例）。系统启动后会有三个 stream

1. full-SM stream: 拥有完整 SM 的 stream，用于纯 Decode 和 Prefill batch，这意味着即使系统里没有 Prefill 和 Decode 混合，也不会损失任何性能。
2. 经过 Green Context 切分过的 Prefill-SM stream：用于专门承载 Prefill kernel。
3. 经过 Green Context 切分过的 Decode-SM stream: 承载 Decode kernel。

我们为 full-SM stream 和 Decode-SM stream 捕获 CUDA Graph。

naive 的调度算法很简单：如果系统是纯 P 或 D，就告诉 worker 提交到 full-SM stream。否则如果是混合的 batch，就分别提交到 P stream 和 D stream，然后等待它们完成。

这样的问题就是，Prefill 可能会持续比较久，Decode 很快结束，造成一定的 SM 空置。

解决方法也很简单，Scheduler 引入了 two inflight batches，或者更简单的说，我们让这种 case 的 Prefill batch 异步化，提交完 Prefill kernel 后再提交一个 cuEvent，然后 Scheduler 拿到这个 event 简单非阻塞查询就好。这也是 Rust 给的一个好处，worker 和 scheduler 在一个进程，传递这些东西非常便捷。

当某个 scheduler loop 观测到 event 完成后，让 worker 真正触发 sync，然后触发采样，像一个正常 Prefill 一样处理好剩下工作，比如提交 KV block，提交 offloading 任务等。如果这个请求结束了 Prefill，就进入 active（也就是 Decode）队列，等待调度。

最头疼的接入问题在于 GPU 内存管理是异步的，平常在一个 stream 上，分配 → Kernel → 释放，都在一个 stream 上，安全。但现在如果 kernel 在另一条 stream 跑的话，启动之前需要 sync 第一次分配的 stream，否则有可能出现 Xid。包括释放时间点。当然这可能是 OpenInfer 还没有给出好的 stream 切换方案，但总的来说，切换 stream 最头疼点在于内存。CPU RAII 和 GPU RAII 直接有个 gap。其实好像 io_uring 一大痛点也是这个生命周期的问题？ 这个话题值得出另一篇博客了，如果在 Rust 里抽象好 GPU 资源（Rust 语言社区里还有几个大佬，做的就是这类事情，也有人已经创业了）。

我们直接上压测，压测我们选择用 vllm-bench（Rust 的模拟多轮对话，符合现在的 agent workload），压测分两组，重 Prefill （首轮 2k，后续轮次 512)，轻 Prefill（首轮 512，后续轮次 256)，输出都是 256 token，并发 16。结果如下

重 Prefill 负载下，图中的 off 是 OpenInfer 默认流，stream 就是 naive two stream，gc x% 是指 Decode 用 x% 的 SM。

![qwen3_4b_multiturn_2k512_overall.png](/blog/green-ctx/qwen3_4b_multiturn_2k512_overall.png)

轻 Prefill 负载下

![qwen3_4b_multiturn_512256_overall.png](/blog/green-ctx/qwen3_4b_multiturn_512256_overall.png)

图中已经把该说的说得足够清楚了。我们再看一个足够吸引人的图表，就是 ITL 分布图。开启 Green Context 后，能获得极其稳定的 ITL，意味着输出速度更加平稳。

![qwen3_4b_multiturn_itl_two_workloads.png](/blog/green-ctx/qwen3_4b_multiturn_itl_two_workloads.png)

当然重 Prefill 负载下性能退化很可能和我们之前测量的 L2 有关，我们 Decode kernel 暂时还没有维护新的一套 kernel。

如果重 Prefill 比较难处理的话，我们依旧可以用 P/D 分离的路子解决，就像 Mooncake 后面发那篇，Prefill As a Service，简单来说就是把长 Prefill 摘出去。普通节点只处理短 Prefill 和 Decode，这点在 Together 博客里也有提及，[https://www.together.ai/blog/cache-aware-disaggregated-inference](https://www.together.ai/blog/cache-aware-disaggregated-inference) ，这个架构叫 CPD，是 P/D 的改善。而 Green Context 能在 CPD 的基础上再次改进。

很有意思的一点是，上了 Green Context 在 5090 上经常打到功耗墙，不知道商业卡会不会稍微好一些。

至于 CPD 怎么做，其实我在写 OpenInfer 的时候，我就有一个想法，路由-引擎-KV cache 三位一体的 co-design，现如今后半部分我应该是比较熟悉了。引擎部分还有一些坑要继续填，推测解码，然后是大 MOE 的推理建模。

下一篇写推测解码，其实 draft 已经有了，DFlash，不过下周再写博客吧。
