# Project Portfolio Series Design

## Goal

Turn `HistoryDocent` and `Insurance_PF` into a job-application portfolio series that can be scanned by a recruiter and audited by an AI/RAG interviewer. The series must explain the author's decisions and failures without inflating dev-only evidence into production claims.

## Audience and reading paths

The primary audience is an AI backend or RAG engineering hiring team.

| Reader | Recommended path | What the path must answer |
| --- | --- | --- |
| Recruiter | Each project's `01` post | What was built, why it matters, what the author did |
| Technical interviewer | Data, retrieval, evaluation posts | Why each architecture and model decision was made |
| Interview preparation | Failure and retrospective posts | What failed, what changed, what remains unverified |
| Full reviewer | Entire series | Whether the claims match the recorded evidence |

## Series architecture

`HistoryDocent` has enough evidence for seven posts.

1. Project landing and role
2. Product scope and architecture
3. Parser output, provenance, and citation-ready corpus
4. Chunking baseline and failure-driven reevaluation
5. Dense, hybrid, reranker, and query rewrite comparison
6. GraphRAG-lite, RAPTOR-lite, HyDE, routing, and locked comparison
7. Citation generation, local voice demo, and retrospective

`Insurance_PF` has enough defensible evidence for five posts.

1. Project landing and role
2. Policy-aware preprocessing and hierarchical chunking
3. Tokenizer and BM25 retrieval comparison
4. QLoRA and RAG comparison
5. Negative-clause failures and retrospective

## Evidence hierarchy

When sources disagree, use this order.

1. Reproducible report or current repository artifact
2. Current project README and final decision document
3. Stage-specific engineering log
4. Notion page and other historical planning material

The Notion `역사 RAG 프로젝트` page is historical evidence. It records an earlier BGE-M3, RRF hybrid, fine-tuned reranker, and Gemini-based pipeline. The current `HistoryDocent` repository uses a later evaluation contract and a different submission baseline. Do not merge the two states into one architecture. Use the Notion page only to explain how the project direction changed.

## Voice

- Use Korean `한다` style with a technical-blog register.
- Start with a concrete failure, decision, or observed result.
- Use first person only for actions and judgments supported by project records.
- Prefer `처음에는 A를 선택했다. B에서 문제가 드러났다. 그래서 C로 바꿨다.` over a feature inventory.
- Vary paragraph length. Do not give every post the same `문제/해결/성과/배운 점` skeleton.
- Use tables only for real comparisons. Use Mermaid only when a flow is easier to understand visually.
- Remove emoji headings, hype, generic conclusions, and unsupported causal claims.

## Claim boundaries

### HistoryDocent

Allowed claims include dev-only model comparisons, current non-rerank candidate selection, citation recoverability, contract smoke results, and local voice demo candidate decisions.

Do not claim production performance, a completed production voice app, final STT/TTS provider selection, or improvement from GraphRAG-lite, RAPTOR-lite, HyDE, or active routing.

### Insurance_PF

Allowed claims include the 6,402-chunk preprocessing result, the logged 30- and 50-question retrieval comparisons, and qualitative Base/Fine-Tuned/RAG comparisons.

Do not convert qualitative review into an accuracy percentage. Do not claim that planned Ragas evaluation or v4 negative-case training was completed. Do not treat `96% (48/50)` Top-5 retrieval as general production accuracy.

## Navigation and publication

- Store posts under `_posts/projects/history-docent/` and `_posts/projects/insurance-pf/`.
- Use category `2-2. History_Docent` for HistoryDocent and `2-8. Insurance_PF` for Insurance_PF.
- Use `{% post_url projects/<project>/<date-slug> %}` for series links.
- Keep the existing parser comparison post as historical research. Do not rename it because the global permalink and existing references must remain stable.

## Review gates

1. Every number appears in a referenced local source.
2. Historical and current HistoryDocent architectures are not collapsed into one timeline.
3. Each post adds a distinct decision or evidence set.
4. Series links resolve during Jekyll build.
5. No raw copyrighted corpus, private query, answer, evidence, audio, secret, or private path is published.
6. AI-writing audit finds no emoji headings, chatbot phrases, generic future closers, or repeated template paragraphs.
