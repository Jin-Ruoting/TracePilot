from __future__ import annotations

from .schemas import ExperimentContext, PaperCandidate


PAPER_FIXTURES: tuple[PaperCandidate, ...] = (
    PaperCandidate(
        paper_id="graphreg-2020",
        title="GraphReg: modeling gene expression with regulatory graphs",
        year=2020,
        tags=("gene_expression", "graph", "enhancer_promoter", "biology_prior"),
        claim="Graph-structured regulatory context can help connect distal elements to gene expression.",
        risk="May require high-quality regulatory edges that are unavailable or inconsistent across tissues.",
        source_url="fixture://papers/graphreg-2020",
        venue="fixture",
    ),
    PaperCandidate(
        paper_id="tabr-2023",
        title="TabR: retrieval-augmented tabular prediction",
        year=2023,
        tags=("retrieval", "tabular", "nearest_neighbors", "conditioning"),
        claim="Neighbor retrieval can improve prediction when similar samples expose local structure.",
        risk="Naive retrieval can leak split information if neighbors are selected across evaluation boundaries.",
        source_url="fixture://papers/tabr-2023",
        venue="fixture",
    ),
    PaperCandidate(
        paper_id="flex-moe-2024",
        title="Flexible mixture-of-experts routing for heterogeneous inputs",
        year=2024,
        tags=("moe", "routing", "heterogeneous", "conditional_compute"),
        claim="Routing can specialize computation for heterogeneous sample groups.",
        risk="Router collapse and weak supervision can hide regressions in minority groups.",
        source_url="fixture://papers/flex-moe-2024",
        venue="fixture",
    ),
    PaperCandidate(
        paper_id="rank-n-contrast-2024",
        title="Rank-N-Contrast for representation learning",
        year=2024,
        tags=("contrastive_learning", "ranking", "representation"),
        claim="Ranking-aware contrastive objectives can improve ordered representation quality.",
        risk="Benefit depends on label scale reliability and well-designed positive/negative construction.",
        source_url="fixture://papers/rank-n-contrast-2024",
        venue="fixture",
    ),
)


DEFAULT_CONTEXT = ExperimentContext(
    project_name="EpiCETRA",
    trusted_split="chromosome-aware split",
    diagnostic_split="sample-random split",
    baseline="40kb promoter + distal-k8 lifted metadata + FiLM conditioning",
    metrics=("Pearson", "Spearman", "MSE", "MAE", "R2"),
    constraints=(
        "main claims must use chromosome-aware split",
        "sample-random split is diagnostic only",
        "new modules need ablation against the trusted baseline",
        "promotion requires better Pearson without worse MSE/R2 consistency",
    ),
)
