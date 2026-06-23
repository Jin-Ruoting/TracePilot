# TracePilot Decision Packet

## Decision Summary
- Question: Should we try retrieval-aware conditioning for promoter expression prediction?
- Recommendation: `promote to small controlled screen`
- Review status: `requires_approval`
- Audit decision: `ready_for_human_review`
- Audit passed: `True`
- Eval passed: `True`
- Trace tools: search_literature, read_experiment_context, score_transfer_fit, draft_experiment_plan, create_review_checkpoint

## Evidence Table
| Evidence | Fit | Relevance | Source | Support | Caution |
| --- | --- | ---: | --- | --- | --- |
| E1 / tabr-2023 | high | 1.00 | fixture://papers/tabr-2023; venue:fixture | Retrieval-style conditioning maps cleanly to tissue/sample-neighbor hypotheses. | Neighbor selection must respect chromosome-aware and gene-level boundaries. Context gate: chromosome-aware split. |
| E2 / graphreg-2020 | medium | 0.90 | fixture://papers/graphreg-2020; venue:fixture | Graph priors align with enhancer-promoter regulation and distal context. | Requires reliable edge construction; start with diagnostic ablation before scaling. Context gate: chromosome-aware split. |

## Cited Claims
- Recommendation 'promote to small controlled screen' is grounded in the transfer fit of TabR: retrieval-augmented tabular prediction.
  - Evidence ids: E1
  - Source refs: fixture://papers/tabr-2023
- Retrieval-style methods need split-aware neighbor controls before promotion.
  - Evidence ids: E1
  - Source refs: fixture://papers/tabr-2023

## Experiment Matrix
- control: 40kb promoter + distal-k8 lifted metadata + FiLM conditioning (roadmap_hg38_40kb_distal_lifted_film; Pearson=0.612, Spearman=0.598, MSE=0.741, MAE=0.631, R2=0.381 on chromosome-aware split)
- variant A: add retrieval features with train-split-only neighbor index
- variant B: retrieval features with shuffled-neighbor negative control
- variant C: remove new module while keeping preprocessing identical
- stratify: report by chromosome, EID/tissue, and expression band

## Promotion Gate
- Pearson improves on chromosome-aware split
- MSE and R2 remain scale-consistent with the trusted baseline
- sample-random gains are not used as main evidence
- no neighbor or metadata lookup crosses evaluation boundaries

## Risks
- Neighbor selection must respect chromosome-aware and gene-level boundaries. Context gate: chromosome-aware split.
- Requires reliable edge construction; start with diagnostic ablation before scaling. Context gate: chromosome-aware split.
- If gains only appear on sample-random split, mark as leakage-risk diagnostic.

## Review Checkpoint
- Checkpoint id: `approval:promotion-gate`
- Status: `requires_approval`
- Reason: Promotion requires human approval before experiment execution.
- Required actions:
  - review cited claims and source refs
  - confirm main evaluation uses chromosome-aware split
  - confirm no retrieval, metadata, or graph lookup crosses evaluation boundaries

## Eval Checks
- Overall: `True`
- `PASS` `tool_trace_complete`: all required tools were traced
- `PASS` `evidence_nonempty`: evidence contains support and caution text
- `PASS` `evidence_provenance_traceable`: every evidence item maps back to candidate source metadata
- `PASS` `claim_citations_bound`: plan claims cite evidence ids and source refs
- `PASS` `source_refs_valid`: source URLs and DOI refs are parseable
- `PASS` `review_checkpoint_valid`: review checkpoint matches recommendation risk
- `PASS` `split_gate_present`: plan gates main and diagnostic split claims
- `PASS` `promotion_supported`: promotion backed by high-fit evidence
- `PASS` `source_poisoning_guarded`: no instruction-like source text detected

## Decision Audit
- Overall: `True`
- Decision: `ready_for_human_review`
- Score: `1.000`
- Summary: Promotion is evidence-grounded but still requires human approval before experiment execution.
- `PASS` `promotion_evidence` [critical]: promotion is grounded in cited high-fit top evidence
- `PASS` `abstention_quality` [info]: run does not abstain
- `PASS` `split_controls` [critical]: promotion gate separates main split, diagnostic split, and leakage boundary
- `PASS` `experiment_matrix` [major]: matrix has baseline/control, variants, and reporting stratification
- `PASS` `review_checkpoint` [critical]: promotion is gated behind specific human approval actions
- `PASS` `source_grounding` [major]: plan claims are tied to evidence ids and source refs
- `PASS` `source_poisoning` [info]: no instruction-like source text detected

## Tool Trace
| Step | Tool | Inputs | Output summary |
| ---: | --- | --- | --- |
| 1 | search_literature | {"query":"Should we try retrieval-aware conditioning for promoter expression prediction?","top_k":4} | {"count":2,"items":[{"paper_id":"tabr-2023","tags":["retrieval","tabular","nearest_neighbors","conditioning"],"title":"TabR: retrieval-augmented tabular prediction","year":2023},{"paper_id":"graphreg-2020","tags":["gene_expression","graph","enhancer_promoter","biology_prior"],"title":"GraphReg: modeling gene expression with regulatory graphs","year":2020}],"type":"list"} |
| 2 | read_experiment_context | {"project_name":"EpiCETRA"} | {"type":"ExperimentContext","value":{"baseline":"40kb promoter + distal-k8 lifted metadata + FiLM conditioning (roadmap_hg38_40kb_distal_lifted_film; Pearson=0.612, Spearman=0.598, MSE=0.741, MAE=0.631, R2=0.381 on chromosome-aware split)","diagnostic_split":"sample-random split","project_name":"EpiCETRA","trusted_split":"chromosome-aware split"}} |
| 3 | score_transfer_fit | {"candidates":[{"citation_count":null,"claim":"Neighbor retrieval can improve prediction when similar samples expose local structure.","doi":"","paper_id":"tabr-2023","risk":"Naive retrieval can leak split information if neighbors are selected across evaluation boundaries.","source_url":"fixture://papers/tabr-2023","tags":["retrieval","tabular","nearest_neighbors","conditioning"],"title":"TabR: retrieval-augmented tabular prediction","venue":"fixture","year":2023},{"citation_count":null,"claim":"Graph-structured regulatory context can help connect distal elements to gene expression.","doi":"","paper_id":"graphreg-2020","risk":"May require high-quality regulatory edges that are unavailable or inconsistent across tissues.","source_url":"fixture://papers/graphreg-2020","tags":["gene_expression","graph","enhancer_promoter","biology_prior"],"title":"GraphReg: modeling gene expression with regulatory graphs","venue":"fixture","year":2020}],"context":{"baseline":"40kb promoter + distal-k8 lifted metadata + FiLM conditioning (roadmap_hg38_40kb_distal_lifted_film; Pearson=0.612, Spearman=0.598, MSE=0.741, MAE=0.631, R2=0.381 on chromosome-aware split)","constraints":["main claims must use chromosome-aware split","sample-random split is diagnostic only","new modules need ablation against the trusted baseline","promotion requires better Pearson without worse MSE/R2 consistency"],"diagnostic_split":"sample-random split","metrics":["Pearson","Spearman","MSE","MAE","R2"],"project_name":"EpiCETRA","trusted_split":"chromosome-aware split"}} | {"count":2,"items":[{"evidence_id":"E1","paper_id":"tabr-2023","relevance":1.0,"title":"TabR: retrieval-augmented tabular prediction"},{"evidence_id":"E2","paper_id":"graphreg-2020","relevance":0.9,"title":"GraphReg: modeling gene expression with regulatory graphs"}],"type":"list"} |
| 4 | draft_experiment_plan | {"context":{"baseline":"40kb promoter + distal-k8 lifted metadata + FiLM conditioning (roadmap_hg38_40kb_distal_lifted_film; Pearson=0.612, Spearman=0.598, MSE=0.741, MAE=0.631, R2=0.381 on chromosome-aware split)","constraints":["main claims must use chromosome-aware split","sample-random split is diagnostic only","new modules need ablation against the trusted baseline","promotion requires better Pearson without worse MSE/R2 consistency"],"diagnostic_split":"sample-random split","metrics":["Pearson","Spearman","MSE","MAE","R2"],"project_name":"EpiCETRA","trusted_split":"chromosome-aware split"},"evidence":[{"cautions":"Neighbor selection must respect chromosome-aware and gene-level boundaries. Context gate: chromosome-aware split.","citation_count":null,"doi":"","evidence_id":"E1","paper_id":"tabr-2023","relevance":1.0,"source_url":"fixture://papers/tabr-2023","supports":"Retrieval-style conditioning maps cleanly to tissue/sample-neighbor hypotheses.","title":"TabR: retrieval-augmented tabular prediction","transfer_fit":"high","venue":"fixture"},{"cautions":"Requires reliable edge construction; start with diagnostic ablation before scaling. Context gate: chromosome-aware split.","citation_count":null,"doi":"","evidence_id":"E2","paper_id":"graphreg-2020","relevance":0.9,"source_url":"fixture://papers/graphreg-2020","supports":"Graph priors align with enhancer-promoter regulation and distal context.","title":"GraphReg: modeling gene expression with regulatory graphs","transfer_fit":"medium","venue":"fixture"}]} | {"type":"ExperimentPlan","value":{"matrix":["control: 40kb promoter + distal-k8 lifted metadata + FiLM conditioning (roadmap_hg38_40kb_distal_lifted_film; Pearson=0.612, Spearman=0.598, MSE=0.741, MAE=0.631, R2=0.381 on chromosome-aware split)","variant A: add retrieval features with train-split-only neighbor index","variant B: retrieval features with shuffled-neighbor negative control","variant C: remove new module while keeping preprocessing identical","stratify: report by chromosome, EID/tissue, and expression band"],"promotion_gate":["Pearson improves on chromosome-aware split","MSE and R2 remain scale-consistent with the trusted baseline","sample-random gains are not used as main evidence","no neighbor or metadata lookup crosses evaluation boundaries"],"risks":["Neighbor selection must respect chromosome-aware and gene-level boundaries. Context gate: chromosome-aware split.","Requires reliable edge construction; start with diagnostic ablation before scaling. Context gate: chromosome-aware split.","If gains only appear on sample-random split, mark as leakage-risk diagnostic."],"summary":"Start with retrieval-aware conditioning because it has the clearest transfer path and testable leakage controls."}} |
| 5 | create_review_checkpoint | {"context":{"baseline":"40kb promoter + distal-k8 lifted metadata + FiLM conditioning (roadmap_hg38_40kb_distal_lifted_film; Pearson=0.612, Spearman=0.598, MSE=0.741, MAE=0.631, R2=0.381 on chromosome-aware split)","constraints":["main claims must use chromosome-aware split","sample-random split is diagnostic only","new modules need ablation against the trusted baseline","promotion requires better Pearson without worse MSE/R2 consistency"],"diagnostic_split":"sample-random split","metrics":["Pearson","Spearman","MSE","MAE","R2"],"project_name":"EpiCETRA","trusted_split":"chromosome-aware split"},"evidence":[{"cautions":"Neighbor selection must respect chromosome-aware and gene-level boundaries. Context gate: chromosome-aware split.","citation_count":null,"doi":"","evidence_id":"E1","paper_id":"tabr-2023","relevance":1.0,"source_url":"fixture://papers/tabr-2023","supports":"Retrieval-style conditioning maps cleanly to tissue/sample-neighbor hypotheses.","title":"TabR: retrieval-augmented tabular prediction","transfer_fit":"high","venue":"fixture"},{"cautions":"Requires reliable edge construction; start with diagnostic ablation before scaling. Context gate: chromosome-aware split.","citation_count":null,"doi":"","evidence_id":"E2","paper_id":"graphreg-2020","relevance":0.9,"source_url":"fixture://papers/graphreg-2020","supports":"Graph priors align with enhancer-promoter regulation and distal context.","title":"GraphReg: modeling gene expression with regulatory graphs","transfer_fit":"medium","venue":"fixture"}],"plan":{"cited_claims":[{"claim":"Recommendation 'promote to small controlled screen' is grounded in the transfer fit of TabR: retrieval-augmented tabular prediction.","evidence_ids":["E1"],"source_refs":["fixture://papers/tabr-2023"]},{"claim":"Retrieval-style methods need split-aware neighbor controls before promotion.","evidence_ids":["E1"],"source_refs":["fixture://papers/tabr-2023"]}],"matrix":["control: 40kb promoter + distal-k8 lifted metadata + FiLM conditioning (roadmap_hg38_40kb_distal_lifted_film; Pearson=0.612, Spearman=0.598, MSE=0.741, MAE=0.631, R2=0.381 on chromosome-aware split)","variant A: add retrieval features with train-split-only neighbor index","variant B: retrieval features with shuffled-neighbor negative control","variant C: remove new module while keeping preprocessing identical","stratify: report by chromosome, EID/tissue, and expression band"],"promotion_gate":["Pearson improves on chromosome-aware split","MSE and R2 remain scale-consistent with the trusted baseline","sample-random gains are not used as main evidence","no neighbor or metadata lookup crosses evaluation boundaries"],"recommendation":"promote to small controlled screen","risks":["Neighbor selection must respect chromosome-aware and gene-level boundaries. Context gate: chromosome-aware split.","Requires reliable edge construction; start with diagnostic ablation before scaling. Context gate: chromosome-aware split.","If gains only appear on sample-random split, mark as leakage-risk diagnostic."],"summary":"Start with retrieval-aware conditioning because it has the clearest transfer path and testable leakage controls."}} | {"type":"ReviewCheckpoint","value":{"checkpoint_id":"approval:promotion-gate","reason":"Promotion requires human approval before experiment execution.","required_actions":["review cited claims and source refs","confirm main evaluation uses chromosome-aware split","confirm no retrieval, metadata, or graph lookup crosses evaluation boundaries"],"status":"requires_approval"}} |
