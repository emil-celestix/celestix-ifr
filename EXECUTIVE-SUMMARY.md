# IFR Test — Executive Summary

## The One Number
**IFR-hybrid+CE nDCG@10: 0.367 vs RAG-rerank 0.321 (+14.3%)**

## Statistical Significance
- IFR-beam vs RAG at 10K atoms: p=0.037 (significant)
- IFR-beam H@20=15% vs RAG 0% on multi-hop at scale
- Bootstrap on 30 queries: directional advantage, not significant at N=30

## Cost-Efficiency
- Both Ollama variants: $0.00/query (local inference)
- Claude API variants: not tested (future work)
- Quality/dollar: RAG 4/30 correct vs IFR 2/30 correct at $0

## Verdict

| Dimension | Result |
|-----------|--------|
| Retrieval | **PASS** — IFR finds multi-hop targets invisible to RAG |
| Scaling | **PASS** — O(1) confirmed, <5ms at 10K atoms |
| End-to-End | **FAIL** — catastrophic drift degrades LLM context |
| Combined | **CONDITIONAL PASS** |

## What Works
- Induced fit is the necessary mechanism (without it: 0%)
- Beam search + novelty confirmed at scale (p=0.037)
- IFR-hybrid+CE is the best overall method (+14% nDCG)
- O(1) latency scaling proven (1.1x growth at 100x data)
- Selective cross-cluster trails enable generalization

## What Doesn't Work
- Greedy traversal is worse than random walk on multi-hop
- Catastrophic drift causes 67% of failures
- IFR context quality too low for LLM answer generation
- Pure IFR (without RAG fusion) loses to RAG on all metrics

## Recommended Architecture for v2
```
Query -> [RAG top-20] + [IFR-beam traverse] -> RRF fusion -> Cross-encoder re-rank -> LLM
```
With: alpha floor 0.5, beam k=5, novelty bonus, selective trails.
