# Funnel_Hashing-GPU-CPU-
This project demonstrates the comparison of time it takes for the Funnel Hashing algorithm in both CPU &amp; GPU.


# Funnel_Hashing-GPU-CPU-
This project demonstrates the time it takes for the algorithm in both CPU &amp; GPU.


Abstract:
This project explores the GPU-based implementation of the Funnel Hashing algorithm, focusing on high throughput insertions using CUDA. Funnel Hashing is a multi-level open addressing scheme designed to 
improve insertion performance by reducing collisions and supporting parallelism. The algorithm 
partitions the hash table into geometrically decreasing levels and utilizes a special overflow array to 
guarantee successful insertions even under high load. This report details how the algorithm was 
parallelized to fully leverage the thread-level concurrency and memory bandwidth of modern GPUs. 
Techniques such as shared memory buffering, atomic operations for thread-safe access, and level-wise 
probing were employed to optimize performance. Experimental results show that the GPU 
implementation achieves significant speedup over a sequential CPU counterpart, scaling efficiently with 
table size. These findings highlight Funnel Hashing's practical viability for real-time data-intensive GPU 
applications.
1. Introduction
As data volumes continue to grow, efficient and scalable data structures become essential for high performance applications. Hash tables are a fundamental tool for constant-time lookups, but their efficiency 
can degrade significantly under high load factors or when insertions trigger collisions. Funnel Hashing is a 
multi-level hashing strategy designed to mitigate these challenges by spreading insertions across 
geometrically decreasing arrays, combined with a small overflow area. This structure reduces collision 
probability and balances load across memory levels.
This project focuses on the GPU-accelerated implementation of Funnel Hashing using CUDA. The work is 
based on the "Implementation of Elastic Hash" paper, which presents a practical algorithm for achieving 
optimal probe complexity without reordering keys. Our goal was to parallelize this insertion process to fully 
leverage the massive thread-level parallelism offered by modern GPUs.


To accelerate the insertion process, the algorithm was implemented in CUDA using a parallel design that 
takes advantage of GPU thread concurrency. Key techniques include dividing the work across threads for 
simultaneous insertions, managing collisions with atomic operations, and organizing the hash table into 
multiple geometric levels to reduce contention. To further improve performance, shared memory is used to 
locally buffer key-value pairs during processing, which helps reduce global memory access latency. These 
combined strategies allow for high-throughput insertions while preserving the structure and behavior of the 
original Funnel Hashing algorithm.
2. Background and Related Work
Hash tables are foundational data structures used to support constant-time access to key-value pairs. Classic 
approaches, like linear probing and double hashing, can suffer from performance degradation at high load 
factors due to clustering and long probe sequences. These limitations have led to the development of 
advanced hashing schemes that aim to maintain low probe complexity even under heavy use.
One notable breakthrough is the Funnel Hashing technique. Funnel Hashing arranges the hash table into 
multiple geometrically shrinking levels, each of which is probed in order until an empty slot is found. This 
structure achieves a worst-case expected probe complexity of 𝑂(log2
(1𝛿)), where 𝛿 represents the fraction 
of empty slots, offering strong theoretical guarantees on insertion performance.
Building on this, the paper Implementation of Elastic Hash (2025) [1] presents a practical realization of 
Funnel Hashing and Elastic Hashing, tailored for efficient, reorder-free insertions. It introduces a multi level array layout where insertions are attempted level by level, eventually defaulting to a special overflow 
array when necessary. The paper outlines key algorithmic strategies such as array partitioning, structured 
probe sequences, and batch insertions, all designed to achieve optimal space utilization and low probe 
complexity in practice.
This project adopts the Funnel Hashing portion of that work and translates it into a parallel GPU 
implementation using CUDA. While the original paper is CPU-centric, our version adapts the algorithm for 
parallel thread execution, making use of CUDA primitives like thread blocks, atomic operations, and shared 
memory. This bridges the gap between theory and practical high-throughput applications, demonstrating 
how modern GPU architectures can accelerate advanced hashing schemes.
3. Funnel Hashing Algorithm Design
Funnel Hashing is a multi-level open addressing scheme designed to support efficient insertions while 
avoiding key reordering. The algorithm partitions the hash table into 𝛼 levels 𝐴1, 𝐴2, . . . , 𝐴𝛼, each 
composed of multiple buckets of size 𝛽. The size of each level decreases geometrically according to the 
recurrence:
|𝐴𝑖+1
| ≈ ⌊
3
4
|𝐴𝑖
|⌋ ± 1
The total number of entries in all levels is 𝑛
′ = 𝑛 − |𝐴𝛼+1
|, where 𝐴𝛼+1 is a special overflow area. Its 
size is constrained within:
⌈𝛿𝑛2⌉ ≤ |𝐴𝛼+1| ≤ ⌊3𝛿𝑛4⌋3
The parameters 𝛼 and 𝛽 are functions of the desired slack 𝛿 ∈ (0,1), chosen to balance probe complexity 
and memory efficiency:
𝛼 = ⌈4 log2 (𝛿1) + 10⌉ , 𝛽 = ⌈2 log2 (𝛿1)⌉
Each level 𝐴𝑖
is made of 𝑎𝑖 buckets, where:|𝐴𝑖| = 𝛽 ∗ 𝑎𝑖and
𝑎𝑖+1 = ⌊34𝑎𝑖⌋ ± 1
Insertion Procedure
Given a key 𝑥 and value 𝑣, the insertion proceeds as follows:
1. Level-wise Attempt:
For each level 𝑖 ∈ [1, 𝛼] , compute the bucket index using a hash function salted with a level specific value 𝑠𝑖:𝑏𝑢𝑐𝑘𝑒𝑡 = |ℎ𝑎𝑠ℎ(𝑥 ⊕ 𝑠𝑖)| 𝑚𝑜𝑑 𝑎𝑖
The algorithm then probes all 𝛽 slots within that bucket. If an empty slot or a duplicate key is found, the 
key-value pair is inserted.
2. Fallback to Overflow Array:
If all main levels fail, the key is inserted into the overflow array 𝐴𝛼+1 using uniform probing. The 
probe limit is bounded by:
max 𝑝𝑟𝑜𝑏𝑒𝑠 = max(1,⌈𝑙𝑜𝑔(𝑙𝑜𝑔(𝑛 + 1))⌉)
Two fallback probes (into indices ℎ 𝑚𝑜𝑑 𝑠 and (ℎ + 1) ) are attempted if the standard probe sequence fails.
3. Probe Complexity:
The worst-case expected probe complexity is 𝑂 (log2(1)),while the overflow array guarantees 
bounded insertion time even in highly loaded scenarios.
Design Properties
• Geometric Decrease: Ensures early levels are larger, reducing collisions when load is light, while 
deeper levels handle overflow gracefully.
• Level Salting: Random salts per level help decorrelate probe sequences and distribute keys evenly.
• Overflow Guarantee: The overflow array ensures that every insertion eventually succeeds, even 
under worst-case fill conditions.
This design provides strong theoretical performance guarantees while remaining practical for parallel GPU 
implementation, as no element reordering or dynamic reallocation is required after initialization.
4
To better illustrate the algorithm's structure and provide a foundation for our CUDA implementation, we 
include a high-level pseudocode summary adapted from the Implementation of Elastic Hash paper [1]. 
These routines formally define the process for dividing the hash table into geometric levels (Algorithm 8), 
assigning bucket structure within levels (Algorithm 9), performing single-level insertions (Algorithm 10), 
and executing the full funnel insertion sequence (Algorithm 11). These algorithms encapsulate the key 
operations underlying the Funnel Hashing method and served as a direct basis for the design of our CUDA 
parallel version.
𝐴𝑙𝑔𝑜𝑟𝑖𝑡ℎ𝑚 8: 𝑆𝑝𝑙𝑖𝑡 𝐴𝑟𝑟𝑎𝑦 𝑖𝑛𝑡𝑜 𝐿𝑒𝑣𝑒𝑙𝑠.
𝐹𝑢𝑛𝑐𝑡𝑖𝑜𝑛 𝑆𝑝𝑙𝑖𝑡𝐼𝑛𝑡𝑜𝐿𝑒𝑣𝑒𝑙𝑠(𝐴, 𝛿)
𝛼 ← 𝑐𝑒𝑖𝑙(4 ∗ 𝑙𝑜𝑔2(1/𝛿) + 10)
𝛽 ← 𝑐𝑒𝑖𝑙(2 ∗ 𝑙𝑜𝑔2(1/𝛿))
𝐹𝑜𝑟 𝑠𝑝𝑒𝑐𝑖𝑎𝑙_𝑠𝑖𝑧𝑒 𝑓𝑟𝑜𝑚 𝑐𝑒𝑖𝑙(𝛿 ∗ |𝐴| / 2) 𝑡𝑜 𝑓𝑙𝑜𝑜𝑟(3 ∗ 𝛿 ∗ |𝐴| / 4):
𝑝𝑟𝑖𝑚𝑎𝑟𝑦_𝑠𝑖𝑧𝑒 ← |𝐴| − 𝑠𝑝𝑒𝑐𝑖𝑎𝑙_𝑠𝑖𝑧𝑒
𝑝𝑟𝑖𝑚𝑎𝑟𝑦_𝑠𝑖𝑧𝑒 ← 𝑓𝑙𝑜𝑜𝑟(𝑝𝑟𝑖𝑚𝑎𝑟𝑦_𝑠𝑖𝑧𝑒 / 𝛽) ∗ 𝛽
𝐼𝑓 |𝐴| − 𝑝𝑟𝑖𝑚𝑎𝑟𝑦_𝑠𝑖𝑧𝑒 == 𝑠𝑝𝑒𝑐𝑖𝑎𝑙_𝑠𝑖𝑧𝑒:
𝑏𝑟𝑒𝑎𝑘
𝐼𝑓 𝑚𝑖𝑠𝑚𝑎𝑡𝑐ℎ: 𝑠𝑝𝑒𝑐𝑖𝑎𝑙_𝑠𝑖𝑧𝑒 ← |𝐴| − 𝑝𝑟𝑖𝑚𝑎𝑟𝑦_𝑠𝑖𝑧𝑒
𝑡𝑜𝑡𝑎𝑙_𝑏𝑢𝑐𝑘𝑒𝑡𝑠 ← 𝑝𝑟𝑖𝑚𝑎𝑟𝑦_𝑠𝑖𝑧𝑒 / 𝛽
𝑎1 ← 𝑡𝑜𝑡𝑎𝑙_𝑏𝑢𝑐𝑘𝑒𝑡𝑠 / (4 ∗ (1 − (0.75)^𝛼))
|𝐴1| ← 𝛽 ∗ 𝑎1
𝐹𝑜𝑟 𝑖 = 2 𝑡𝑜 𝛼:
𝑎𝑖 ← 𝑓𝑙𝑜𝑜𝑟((3 ∗ 𝑎_{𝑖 − 1}) / 4 ± 1)
|𝐴𝑖| ← 𝛽 ∗ 𝑎𝑖
𝑅𝑒𝑡𝑢𝑟𝑛 𝑙𝑒𝑣𝑒𝑙𝑠 𝐴1 𝑡𝑜 𝐴𝛼 𝑎𝑛𝑑 𝑜𝑣𝑒𝑟𝑓𝑙𝑜𝑤 𝑎𝑟𝑟𝑎𝑦 𝐴_{𝛼 + 1}
𝐸𝑛𝑑 𝐹𝑢𝑛𝑐𝑡𝑖𝑜𝑛
This function determines the number and size of levels in the table, ensuring that the overflow area satisfies 
the required slack conditions.
𝐴𝑙𝑔𝑜𝑟𝑖𝑡ℎ𝑚 9: 𝑆𝑢𝑏𝑑𝑖𝑣𝑖𝑑𝑒 𝐸𝑎𝑐ℎ 𝐿𝑒𝑣𝑒𝑙 𝑖𝑛𝑡𝑜 𝐵𝑢𝑐𝑘𝑒𝑡𝑠.
𝐹𝑢𝑛𝑐𝑡𝑖𝑜𝑛 𝑆𝑢𝑏𝑑𝑖𝑣𝑖𝑑𝑒𝐿𝑒𝑣𝑒𝑙𝑠(𝐴𝑖, 𝛽)
𝐹𝑜𝑟 𝑒𝑎𝑐ℎ 𝐴𝑖 𝑖𝑛 𝑙𝑒𝑣𝑒𝑙𝑠:
𝐹𝑜𝑟 𝑗 𝑓𝑟𝑜𝑚 1 𝑡𝑜 |𝐴𝑖| / 𝛽:
𝐴𝑖,𝑗 ← 𝑆𝑢𝑏𝑎𝑟𝑟𝑎𝑦 𝑜𝑓 𝑠𝑖𝑧𝑒 𝛽
𝑅𝑒𝑡𝑢𝑟𝑛 {𝐴𝑖,𝑗}
𝐸𝑛𝑑 𝐹𝑢𝑛𝑐𝑡𝑖𝑜𝑛
Each level is further partitioned into fixed-size buckets, where each bucket can hold up to 𝛽 keys.
𝐴𝑙𝑔𝑜𝑟𝑖𝑡ℎ𝑚 10: 𝐴𝑡𝑡𝑒𝑚𝑝𝑡𝑒𝑑 𝐼𝑛𝑠𝑒𝑟𝑡𝑖𝑜𝑛 𝑖𝑛𝑡𝑜 𝑎 𝐿𝑒𝑣𝑒𝑙.
5
𝐹𝑢𝑛𝑐𝑡𝑖𝑜𝑛 𝐴𝑡𝑡𝑒𝑚𝑝𝑡𝐼𝑛𝑠𝑒𝑟𝑡(𝑘𝑒𝑦, 𝐴𝑖, 𝛽)
𝑏𝑢𝑐𝑘𝑒𝑡_𝑖𝑛𝑑𝑒𝑥 ← 𝐻𝑎𝑠ℎ(𝑘𝑒𝑦) 𝑚𝑜𝑑 (|𝐴𝑖| / 𝛽)
𝐹𝑜𝑟 𝑠𝑙𝑜𝑡 𝑖𝑛 0 𝑡𝑜 𝛽 − 1:
𝐼𝑓 𝐴𝑖[𝑏𝑢𝑐𝑘𝑒𝑡_𝑖𝑛𝑑𝑒𝑥][𝑠𝑙𝑜𝑡] 𝑖𝑠 𝑒𝑚𝑝𝑡𝑦:
𝐴𝑖[𝑏𝑢𝑐𝑘𝑒𝑡_𝑖𝑛𝑑𝑒𝑥][𝑠𝑙𝑜𝑡] ← 𝑘𝑒𝑦
𝑅𝑒𝑡𝑢𝑟𝑛 𝑆𝑢𝑐𝑐𝑒𝑠𝑠
𝑅𝑒𝑡𝑢𝑟𝑛 𝐹𝑎𝑖𝑙
𝐸𝑛𝑑 𝐹𝑢𝑛𝑐𝑡𝑖𝑜𝑛
This logic governs how a key attempts to insert itself into a specific level using linear probing within its 
target bucket.
𝐴𝑙𝑔𝑜𝑟𝑖𝑡ℎ𝑚 11: 𝐹𝑢𝑙𝑙 𝐹𝑢𝑛𝑛𝑒𝑙 𝐼𝑛𝑠𝑒𝑟𝑡𝑖𝑜𝑛.
𝐹𝑢𝑛𝑐𝑡𝑖𝑜𝑛 𝐼𝑛𝑠𝑒𝑟𝑡𝐾𝑒𝑦(𝑘𝑒𝑦,{𝐴1,𝐴2, . . . , 𝐴𝛼}, 𝐴_{𝛼 + 1}, 𝛽)
𝐹𝑜𝑟 𝑖 𝑓𝑟𝑜𝑚 1 𝑡𝑜 𝛼:
𝐼𝑓 𝐴𝑡𝑡𝑒𝑚𝑝𝑡𝐼𝑛𝑠𝑒𝑟𝑡(𝑘𝑒𝑦, 𝐴𝑖, 𝛽) == 𝑆𝑢𝑐𝑐𝑒𝑠𝑠:
𝑅𝑒𝑡𝑢𝑟𝑛
𝐼𝑛𝑠𝑒𝑟𝑡 𝑘𝑒𝑦 𝑖𝑛𝑡𝑜 𝑜𝑣𝑒𝑟𝑓𝑙𝑜𝑤 𝑎𝑟𝑟𝑎𝑦 𝐴_{𝛼 + 1}
𝐸𝑛𝑑 𝐹𝑢𝑛𝑐𝑡𝑖𝑜𝑛
This function coordinates the complete insertion process by cascading the key through all levels before 
using the overflow array as a last resort.
4. CUDA Implementation
The Funnel Hashing algorithm was implemented on the GPU using NVIDIA’s CUDA programming model 
to leverage high concurrency and memory bandwidth. This section outlines the key design aspects, followed 
by the optimization techniques applied to maximize performance and scalability.
4.1 Memory Management
Each level of the Funnel Hash Table, including the special overflow array, was allocated in global GPU 
memory using cudaMalloc. After allocation, all key and value arrays were initialized in parallel with a 
sentinel value (EMPTY_SLOT) using a custom CUDA kernel, initialize_memory. This avoids the cost of 
host-side sequential initialization and ensures consistent state across device memory.
A simplified overview of the memory flow is as follows:
• On the CPU side, the FunnelHashTable structure is initialized, including all levels and associated 
arrays.
• Then, device-side memory is allocated using cudaMalloc, and the structure is copied to the GPU.
• Once setup is complete, the GPU takes over via the insert_keys_kernel where each thread inserts 
a single key-value pair.
• Insertions proceed level by level; if no spot is found in any level, the key is inserted into the 
overflow array.
• After all insertions are completed, data is copied back from the GPU using cudaMemcpy, and 
results (e.g., contents, timing) are printed by the host.


This flow illustrates a clean division of responsibilities: the host sets up and wraps up, while the device 
performs the high-throughput insertions:
4.2 CUDA-Based Pseudocode and Optimization Techniques:
To implement Funnel Hashing on the GPU, the original algorithm was divided into a host-side 
preprocessing phase and a device-side parallel insertion phase. The following CUDA-based pseudocode 
captures the device-side parallel logic and highlights the optimization techniques applied. Each routine 
corresponds to a core part of the Funnel Hashing algorithm, adapted to leverage thread-level parallelism, 
shared memory, and atomic operations.
Before describing the GPU execution phase, it's important to note that Algorithms 8 and 9 are preserved 
and executed on the host (CPU). These algorithms are implemented in the initialize_table() function:
• Algorithm 8 (SplitIntoLevels) prepares the structure of the hash table by partitioning it into 
multiple geometric levels and an overflow array.
• Algorithm 9 (SubdivideIntoBuckets) defines how each level is logically divided into buckets of 
size β. While buckets are not physically separated in memory, their boundaries are accessed through 
index arithmetic during insertion.
These setup steps remain sequential and are executed before launching any CUDA kernels. The CUDA specific logic begins after this initialization, where parallel insertions are performed by thousands of threads 
using the memory structure defined by Algorithms 8 and 9.
1. 𝐶𝑈𝐷𝐴 𝐼𝑛𝑠𝑒𝑟𝑡𝑖𝑜𝑛 𝐾𝑒𝑟𝑛𝑒𝑙 (𝑃𝑎𝑟𝑎𝑙𝑙𝑒𝑙 𝐷𝑖𝑠𝑝𝑎𝑡𝑐ℎ + 𝑆ℎ𝑎𝑟𝑒𝑑 𝑀𝑒𝑚𝑜𝑟𝑦):
𝑃𝑟𝑜𝑐𝑒𝑑𝑢𝑟𝑒 𝑖𝑛𝑠𝑒𝑟𝑡_𝑘𝑒𝑦𝑠_𝑘𝑒𝑟𝑛𝑒𝑙(𝑡𝑎𝑏𝑙𝑒, 𝑘𝑒𝑦𝑠[], 𝑣𝑎𝑙𝑢𝑒𝑠[], 𝑛)
𝑆ℎ𝑎𝑟𝑒𝑑 𝑖𝑛𝑡 𝑠ℎ𝑎𝑟𝑒𝑑_𝑘𝑒𝑦𝑠[𝑏𝑙𝑜𝑐𝑘𝐷𝑖𝑚. 𝑥]
𝑆ℎ𝑎𝑟𝑒𝑑 𝑖𝑛𝑡 𝑠ℎ𝑎𝑟𝑒𝑑_𝑣𝑎𝑙𝑢𝑒𝑠[𝑏𝑙𝑜𝑐𝑘𝐷𝑖𝑚. 𝑥]
𝑡𝑖𝑑 ← 𝑏𝑙𝑜𝑐𝑘𝐼𝑑𝑥. 𝑥 ∗ 𝑏𝑙𝑜𝑐𝑘𝐷𝑖𝑚. 𝑥 + 𝑡ℎ𝑟𝑒𝑎𝑑𝐼𝑑𝑥. 𝑥
𝑖𝑓 𝑡𝑖𝑑 < 𝑛:
𝑠ℎ𝑎𝑟𝑒𝑑_𝑘𝑒𝑦𝑠[𝑡ℎ𝑟𝑒𝑎𝑑𝐼𝑑𝑥. 𝑥] ← 𝑘𝑒𝑦𝑠[𝑡𝑖𝑑]
𝑠ℎ𝑎𝑟𝑒𝑑_𝑣𝑎𝑙𝑢𝑒𝑠[𝑡ℎ𝑟𝑒𝑎𝑑𝐼𝑑𝑥. 𝑥] ← 𝑣𝑎𝑙𝑢𝑒𝑠[𝑡𝑖𝑑]
𝑠𝑦𝑛𝑐_𝑡ℎ𝑟𝑒𝑎𝑑𝑠()
𝑖𝑓 𝑡𝑖𝑑 < 𝑛:
𝐼𝑛𝑠𝑒𝑟𝑡𝐾𝑒𝑦(𝑠ℎ𝑎𝑟𝑒𝑑𝑘𝑒𝑦𝑠[𝑡ℎ𝑟𝑒𝑎𝑑𝐼𝑑𝑥.𝑥]
, 𝑠ℎ𝑎𝑟𝑒𝑑𝑣𝑎𝑙𝑢𝑒𝑠[𝑡ℎ𝑟𝑒𝑎𝑑𝐼𝑑𝑥.𝑥]
,𝑡𝑎𝑏𝑙𝑒)
𝐸𝑛𝑑 𝐹𝑢𝑛𝑐𝑡𝑖𝑜𝑛



This kernel distributes the insertion workload across threads. Each thread loads one key-value pair into 
shared memory and synchronizes with its block before insertion. This technique improves runtime by 
reducing global memory accesses and allowing massively parallel insertion of independent elements.
2. 𝐴𝑡𝑡𝑒𝑚𝑝𝑡𝑒𝑑 𝐵𝑢𝑐𝑘𝑒𝑡 𝐼𝑛𝑠𝑒𝑟𝑡𝑖𝑜𝑛 𝑤𝑖𝑡ℎ 𝐴𝑡𝑜𝑚𝑖𝑐 𝑆𝑎𝑓𝑒𝑡𝑦:
𝐹𝑢𝑛𝑐𝑡𝑖𝑜𝑛 𝐴𝑡𝑡𝑒𝑚𝑝𝑡𝐼𝑛𝑠𝑒𝑟𝑡(𝑙𝑒𝑣𝑒𝑙, 𝑘𝑒𝑦, 𝑣𝑎𝑙𝑢𝑒, 𝑠𝑎𝑙𝑡, 𝛽)
𝑏𝑢𝑐𝑘𝑒𝑡 ← ℎ𝑎𝑠ℎ(𝑘𝑒𝑦 ⊕ 𝑠𝑎𝑙𝑡) 𝑚𝑜𝑑 𝑛𝑢𝑚_𝑏𝑢𝑐𝑘𝑒𝑡𝑠
𝑠𝑡𝑎𝑟𝑡 ← 𝑏𝑢𝑐𝑘𝑒𝑡 ∗ 𝛽
𝑓𝑜𝑟 𝑖 𝑖𝑛 [𝑠𝑡𝑎𝑟𝑡, 𝑠𝑡𝑎𝑟𝑡 + 𝛽):
𝑖𝑓 𝑎𝑡𝑜𝑚𝑖𝑐𝐶𝐴𝑆(𝑙𝑒𝑣𝑒𝑙. 𝑘𝑒𝑦𝑠[𝑖], 𝐸𝑀𝑃𝑇𝑌_𝑆𝐿𝑂𝑇, 𝑘𝑒𝑦) ∈ {𝐸𝑀𝑃𝑇𝑌_𝑆𝐿𝑂𝑇, 𝑘𝑒𝑦}:
𝑙𝑒𝑣𝑒𝑙. 𝑣𝑎𝑙𝑢𝑒𝑠[𝑖] ← 𝑣𝑎𝑙𝑢𝑒
𝑟𝑒𝑡𝑢𝑟𝑛 𝑆𝑈𝐶𝐶𝐸𝑆𝑆
𝑟𝑒𝑡𝑢𝑟𝑛 𝐹𝐴𝐼𝐿
𝐸𝑛𝑑 𝐹𝑢𝑛𝑐𝑡𝑖𝑜𝑛
This function reflects the insertion logic within a single bucket (CUDA version of Algorithm 10). By 
using atomic operations, it ensures thread-safe insertion when multiple threads target overlapping buckets. 
This approach avoids the need for explicit locking while preserving correctness.
3. 𝐹𝑢𝑛𝑛𝑒𝑙 𝐼𝑛𝑠𝑒𝑟𝑡𝑖𝑜𝑛 𝐶𝑎𝑠𝑐𝑎𝑑𝑒 𝑤𝑖𝑡ℎ 𝑂𝑣𝑒𝑟𝑓𝑙𝑜𝑤 𝐹𝑎𝑙𝑙𝑏𝑎𝑐𝑘:
𝐹𝑢𝑛𝑐𝑡𝑖𝑜𝑛 𝐼𝑛𝑠𝑒𝑟𝑡𝐾𝑒𝑦(𝑘𝑒𝑦, 𝑣𝑎𝑙𝑢𝑒,𝑡𝑎𝑏𝑙𝑒)
𝑓𝑜𝑟 𝑙𝑒𝑣𝑒𝑙 𝑖𝑛 𝑡𝑎𝑏𝑙𝑒. 𝑙𝑒𝑣𝑒𝑙𝑠:
𝑖𝑓 𝐴𝑡𝑡𝑒𝑚𝑝𝑡𝐼𝑛𝑠𝑒𝑟𝑡(𝑙𝑒𝑣𝑒𝑙, 𝑘𝑒𝑦, 𝑣𝑎𝑙𝑢𝑒, 𝑙𝑒𝑣𝑒𝑙. 𝑠𝑎𝑙𝑡, 𝛽) == 𝑆𝑈𝐶𝐶𝐸𝑆𝑆:
𝑟𝑒𝑡𝑢𝑟𝑛
// 𝑂𝑣𝑒𝑟𝑓𝑙𝑜𝑤 𝑓𝑎𝑙𝑙𝑏𝑎𝑐𝑘
𝑓𝑜𝑟 𝑗 𝑖𝑛 0 𝑡𝑜 𝑚𝑎𝑥_𝑝𝑟𝑜𝑏𝑒𝑠:
𝑖𝑑𝑥 ← (ℎ𝑎𝑠ℎ(𝑘𝑒𝑦 ⊕ 𝑠𝑎𝑙𝑡) + 𝑗) 𝑚𝑜𝑑 𝑜𝑣𝑒𝑟𝑓𝑙𝑜𝑤_𝑠𝑖𝑧𝑒
𝑖𝑓 𝑎𝑡𝑜𝑚𝑖𝑐𝐶𝐴𝑆(𝑜𝑣𝑒𝑟𝑓𝑙𝑜𝑤_𝑘𝑒𝑦𝑠[𝑖𝑑𝑥], 𝐸𝑀𝑃𝑇𝑌_𝑆𝐿𝑂𝑇, 𝑘𝑒𝑦) ∈ {𝐸𝑀𝑃𝑇𝑌_𝑆𝐿𝑂𝑇, 𝑘𝑒𝑦}:
𝑜𝑣𝑒𝑟𝑓𝑙𝑜𝑤_𝑣𝑎𝑙𝑢𝑒𝑠[𝑖𝑑𝑥] ← 𝑣𝑎𝑙𝑢𝑒
𝑟𝑒𝑡𝑢𝑟𝑛
𝑇𝑟𝑦 𝑡𝑤𝑜 𝑓𝑖𝑥𝑒𝑑 𝑓𝑎𝑙𝑙𝑏𝑎𝑐𝑘 𝑝𝑜𝑠𝑖𝑡𝑖𝑜𝑛𝑠 𝑢𝑠𝑖𝑛𝑔 𝑎𝑡𝑜𝑚𝑖𝑐𝐶𝐴𝑆
𝐸𝑛𝑑 𝐹𝑢𝑛𝑐𝑡𝑖𝑜𝑛
This is the full cascade insertion routine (CUDA version of Algorithm 11). A key is inserted level by level; 
if all levels are full, it is placed into the overflow array using capped uniform probing. Final fallback 
positions ensure that every key is eventually inserted. This guarantees performance stability even at high 
load factors.
4.3 Justification of CUDA Optimization Techniques
To achieve high throughput in the GPU implementation of the Funnel Hashing algorithm, several CUDA 
optimization techniques were integrated. These techniques are widely adopted in GPU computing and are 
supported by performance analyses found in related literature [2]. Each strategy addresses specific 
bottlenecks such as memory latency, thread utilization, and synchronization overhead, and contributes to a 
measurable reduction in execution time.
4.3.1 Thread-Level Parallelism for Concurrent Insertions
The most fundamental optimization in the implementation is the use of thread-level parallelism, where each 
CUDA thread is responsible for inserting a single key-value pair. This embarrassingly parallel structure 
allows thousands of insertions to be processed simultaneously, leveraging the massive thread concurrency 
of modern GPUs. As demonstrated in [2], this model leads to high instruction throughput and better resource 
utilization, especially for data structures like hash tables where insertions can proceed independently.
4.3.2 Shared Memory Buffering to Minimize Global Memory Latency
In the insert_keys_kernel, key-value pairs are first loaded into on-chip shared memory before any 
processing is performed. Shared memory offers significantly lower latency than global memory, which 
makes it ideal for temporary buffering of frequently accessed data. By reducing redundant global memory 
accesses and increasing memory access locality, this technique improves warp execution efficiency and 
overall kernel performance, consistent with shared memory usage strategies discussed in [2].
4.3.3 Atomic Operations for Safe Concurrent Writes
Concurrent insertions into a shared hash table pose potential consistency issues. To handle this safely 
without serializing the entire process, the implementation uses atomic compare-and-swap (atomicCAS) 
operations to manage concurrent writes. These operations ensure that multiple threads can safely attempt 
insertions into overlapping regions of memory without conflicts. Although atomic operations carry a 
synchronization cost, they are essential for correctness and have been shown to scale well in memory-bound 
scenarios, as supported by the results in [2].
4.3.4 Register Usage and Occupancy-Aware Kernel Design
Frequently accessed loop indices and hash values are stored in registers, the fastest memory available on a 
CUDA device. Efficient use of registers helps reduce instruction latency and improves warp execution 
throughput. At the same time, the kernel was designed to limit shared memory usage per block, preserving 
high occupancy (the number of active warps per streaming multiprocessor) an important factor in hiding 
global memory latency. These trade-offs are in line with the kernel-level optimization guidelines found in 
[2].
4.3.5 Indexing Strategy and Load Balancing
Global thread indices are computed using standard grid-stride patterns (blockIdx.x * blockDim.x + 
threadIdx.x), which allows the kernel to scale regardless of the total number of insertions. This ensures that 
all data is processed, even when the input size exceeds the number of threads per block. Such an approach 
promotes effective load balancing across thread blocks, improving both device occupancy and execution 
consistency, as recommended in parallel workload distribution strategies [2].
4.3.6 Parallel Memory Initialization for Scalable Setup
Before kernel execution, all table arrays are initialized using a parallel kernel (initialize_memory), replacing 
slower CPU-side loops. This ensures that device memory is clean and ready for use across thousands of 
concurrent threads. Performing this initialization in parallel reduces setup overhead and aligns with best 
practices for scalable GPU preprocessing outlined in [2].
8
9
5. Experimental Setup
The experimental setup was designed to evaluate the performance of the GPU-based Funnel Hashing 
implementation. The following hardware and software configurations were used.
Hardware:
 GPU: GAIVI
Software:
 Operating System: Windows 11
 CUDA Toolkit: CUDA 12.2
 Compiler: nvcc (NVIDIA CUDA Compiler)
 Development Environment: Visual Studio Code with CUDA Extension
Experimental Parameters:
 Block Size: 256 threads per block
 Grid Size: Calculated based on number of insertions ((NUM_INSERTIONS+255)/256)
 Shared Memory Usage: Each block uses 2 × 256 × sizeof(int) bytes for storing keys and values 
temporarily.
Performance Measurement:
⚫ Timing Tool: CUDA Events (cudaEventRecord, cudaEventElapsedTime)
⚫ Measured Section: Only the insertion kernel execution time (excluding memory allocation, data 
transfer, and printing overhead).
All keys and values were initialized in host (CPU) memory and then copied to device (GPU) memory before 
starting the insertion kernel. This setup ensured that the performance evaluation focused on the 
computational efficiency of Funnel Hashing itself, minimizing external factors like host-device memory 
transfer times.
6. Results and Evaluation
To evaluate the performance of the Funnel Hashing implementation, we compared the execution time of 
the CPU and GPU versions across varying hash table capacities and insertion load factors. For each 
capacity, four insertion volumes were tested: 25%, 50%, 75%, and 90% of total capacity.
CPU vs GPU Execution Time:
Figure 1. Execution Time for CPU Implementation.
This figure presents the running time of the CPU implementation as the number of insertions increases. A 
near-linear growth in execution time is observed, especially at higher load factors (75% and 90%). This 
behavior is expected in sequential hashing algorithms, where insertions become increasingly expensive due 
to collision resolution and longer probe sequences in a densely filled table. As load increases, the likelihood 
of probing deeper into multiple levels or falling back into the overflow array rises, contributing to the 
steeper slope of the curves.
Figure 2. Execution Time for CUDA Implementation
In contrast, the GPU-based implementation maintains relatively flat execution times across all capacities 
and insertion volumes. This highlights the core strength of the CUDA implementation: thousands of 
insertions are processed in parallel, and the cost per insertion remains nearly constant. The marginal 
increase seen at 90% load factor is due to increased contention and atomic operations, but the impact is 
minimal compared to the CPU implementation. The use of shared memory and the division of workload 
across threads allow the GPU to absorb the increased load without significant degradation in 
performance.


Figure 3. Speedup of CUDA Implementation over CPU
This figure quantifies the performance gain of the CUDA implementation over the CPU baseline. At low 
table sizes, the speedup is modest (approximately 3×), primarily due to the fixed overhead of kernel 
launches and memory transfers. However, as the dataset scales, the speedup becomes increasingly 
pronounced, reaching over 50× for the largest capacity and highest load factor tested. This result 
confirms that the CUDA optimization techniques (particularly thread-level parallelism, shared memory 
buffering, and atomic insertions) pay off substantially when enough computational work is available to 
offset the GPU overhead.
7. Conclusion 
The experimental evaluation demonstrates that Funnel Hashing is highly amenable to GPU acceleration, 
offering both efficiency and scalability under increasing workloads. The CUDA implementation 
consistently outperforms its CPU counterpart, particularly as table capacity and insertion volume grow. 
While the CPU version exhibits near-linear performance degradation due to sequential insertions and 
collision resolution, the GPU version maintains a nearly flat execution profile—underscoring the 
advantage of massively parallel execution.
A core reason for this scalability lies in the structure of the algorithm itself. Funnel Hashing distributes 
insertions across multiple geometrically shrinking levels, balancing load and minimizing contention. The 
fallback mechanism into the overflow array further ensures successful insertions even under high 
occupancy, without introducing runtime bottlenecks. This hierarchical design maps effectively GPU 
hardware, where thread independence allows thousands of insertions to be handled simultaneously.
Several CUDA-specific optimizations contribute directly to the observed performance gains. Thread-level 
parallelism enables each thread to independently process a key-value pair, and the use of atomicCAS
ensures correctness during concurrent writes without the need for costly synchronization primitives. 
Shared memory further improves performance by temporarily buffering data within each block, reducing 
reliance on high-latency global memory. Although shared memory is limited in size, its strategic use 
significantly lowers per-thread access delays during the critical insertion phase.


The results also confirm that GPU efficiency improves with scale. Initial speedups are modest—around 
3× for small tables (where launch overheads and memory transfer costs dominate). However, as problem 
size increases, the benefits compound, with speedups approaching 50× for large capacities and high load 
factors. This aligns with the general principle that GPUs are most effective when fully utilized, and that 
parallel workloads amortize fixed costs over a larger number of operations.
Nevertheless, limitations remain. When the table size is small, or when latency is a strict requirement, the 
overhead of kernel launches and synchronization can outweigh the benefits of parallelism. Additionally, 
the current implementation assumes a static table size. Extending it to support deletions or dynamic 
resizing would require more complex memory management and may impact performance consistency.
Overall, the CUDA-based Funnel Hashing solution delivers substantial speedups while preserving 
algorithmic correctness and structural robustness. It scales gracefully with workload size and proves wellsuited for high-throughput applications such as dynamic hash tables, real-time data ingestion, and GPUaccelerated indexing systems. These results validate Funnel Hashing as a compelling candidate for 
modern data infrastructure where fast, concurrent insertions are essential.
