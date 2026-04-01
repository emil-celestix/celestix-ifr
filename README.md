# Induced-Fit Retrieval (IFR)

## Description
Induced-Fit Retrieval (IFR) is an information retrieval system designed to solve the multi-hop reasoning problem by treating query retrieval as a dynamic graph traversal process. The architecture is inspired by Daniel Koshland's 1958 "induced fit" model of enzyme-substrate binding from biochemistry. 

Unlike traditional Retrieval-Augmented Generation (RAG) which uses a static query (a "lock and key" approach), IFR mutates the query vector at each hop based on the visited node's embedding. This allows the query to adapt as it encounters new information, moving along the embedding space's curved manifolds to discover semantically distant but logically connected concepts.

---

## Executive Summary & The "One Number"
**IFR-hybrid+CE nDCG@10: 0.367 vs RAG-rerank 0.321 (+14.3%)**

Our test suite (30 queries, 10 methods, multiple graph sizes) evaluated the prototype across Retrieval, Scaling, and End-to-End LLM generation. 

* **Retrieval (PASS):** IFR successfully finds multi-hop targets that are entirely invisible to RAG.
* **Scaling (PASS):** Sub-linear O(1) latency scaling was confirmed (<5ms at 10K atoms).
* **End-to-End (FAIL):** Catastrophic drift during query mutation degrades the final context provided to the LLM, lowering generation quality.
* **Verdict:** CONDITIONAL PASS. The retrieval mechanism is proven, but requires ranking and drift-damping fixes for v2.

---

## Empirical Testing & Results

### 1. Multi-Hop Discovery Advantage
All tested traditional RAG methods scored **0% Hit@20** on complex multi-hop queries. In contrast, IFR successfully discovered targets that were ranked deep in the baseline RAG results (e.g., rank 22–665), achieving a **15% Hit@20** on multi-hop evaluation. 

### 2. Algorithmic Ablations
* **Induced Fit is Necessary:** Setting the query mutation rate ($\alpha$) to zero resulted in a 0% multi-hop hit rate and a near-zero MRR. Dynamic query mutation is strictly required for success.
* **Beam Search vs. Greedy:** At high data scales (10K+ atoms), Beam Search ($k=5$) with a novelty exploration bonus outperformed greedy traversal (15% vs 0%, p=0.037).
* **Trail Learning:** Applying Ant Colony-style "pheromone trails" to reinforce successful cross-cluster edges yielded strong generalization on fresh test sets (Hit@20 improved 70% $\rightarrow$ 80%), proving that the graph can "learn" structural market patterns.

### 3. O(1) Traversal Scaling
Latency tests prove that IFR converts retrieval from a geometric nearest-neighbor problem to a topological traversal problem.
* **100 Atoms:** 1.50ms median
* **1,000 Atoms:** 1.64ms median 
* **10,000 Atoms:** 1.64ms median (P99 = 4.46ms)
* **Conclusion:** 100x data growth resulted in only 1.1x latency growth.

### 4. Failure Analysis & The "Drift" Problem
While IFR beats RAG in pure retrieval recall, it struggles in End-to-End LLM testing (RAG Token F1 = 0.089 vs IFR Token F1 = 0.040).
* **Catastrophic Drift:** Detailed failure analysis revealed that 67% of IFR's failures were due to "catastrophic drift." The query mutated too aggressively at intermediate hops, losing over 80% of its original intent.
* **Context U-Curve:** While RAG context quality peaks at $k=3$, IFR's context quality degrades at $k=5$ due to noise dilution and drift.

---

## Recommended Architecture for v2
Based on the data, pure IFR loses to RAG on overall metrics, but a hybrid approach excels. The optimal configuration for future iterations is a **Hybrid Fusion Pipeline**:

```text
Query -> [RAG top-20] + [IFR-beam traverse] -> RRF fusion -> Cross-encoder re-rank -> LLM
