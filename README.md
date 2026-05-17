# Scientific Paper Knowledge Graph with Embeddings and Local LLMs

Учебный Python-проект для курсовой работы на тему построения предметно-ориентированного графа знаний по научным статьям.

Проект строит граф по PDF-документам: каждая вершина соответствует научной статье, а ребро описывает тематическую или смысловую связь между двумя статьями. В основном pipeline используются только локальные модели: `sentence-transformers` для эмбеддингов и `transformers` для LLM-классификации отношений. OpenAI API и Hugging Face Inference API не используются.

## Pipeline

```text
PDF papers
-> text extraction
-> preprocessing
-> sentence embeddings
-> candidate pair generation
-> local LLM relation classification
-> LLM graph
-> weighted hybrid graph
-> graph analysis
```

## Graphs

### 1. Embedding Similarity Graph

Baseline-граф, который строится по cosine similarity между sentence embeddings статей. Он дает широкое покрытие корпуса и показывает тематическую близость даже тогда, когда тип связи между статьями явно не классифицирован.

### 2. LLM Relation Graph

Интерпретируемый граф, построенный по классификации candidate pairs локальной LLM. Поддерживаемые типы отношений:

- `BASED_ON`
- `EXTENDS`
- `COMPARES_WITH`
- `SIMILAR_TO`
- `SAME_TOPIC`

Связи `NO_RELATION` и `ERROR` удаляются. Для `SAME_TOPIC` и `SIMILAR_TO` используется порог `confidence >= 0.7`; для `BASED_ON`, `EXTENDS`, `COMPARES_WITH` используется `confidence >= 0.5`. Такой граф обычно более разреженный, но его ребра легче защищать и объяснять.

### 3. Weighted Hybrid Knowledge Graph

Итоговый предметно-ориентированный граф знаний. LLM-рёбра имеют приоритет над embedding-рёбрами. Если LLM уже дала связь между двумя статьями, embedding edge не дублируется. Остальные embedding-рёбра добавляются как `SIMILAR_TO`.

Вес LLM-ребра:

```text
w_ij = alpha * sim_ij + (1 - alpha) * conf_ij
```

где `sim_ij` это embedding similarity, а `conf_ij` это confidence локальной LLM.

## Analysis

Для графов считаются:

- density
- average degree
- clustering coefficient
- connected components
- communities через `networkx.algorithms.community.greedy_modularity_communities`
- degree centrality
- betweenness centrality
- PageRank
- shortest path explanation между выбранными статьями

Главная интерпретация:

- Embedding Graph дает широкое покрытие и связность.
- LLM Relation Graph дает типизированные интерпретируемые связи.
- Weighted Hybrid Graph объединяет оба подхода и является итоговым графом знаний.

## Project Structure

```text
knowledge_graph/
├── src/
│   ├── pdf_parser.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── llm_relation_extractor.py
│   ├── graph_builder.py
│   ├── graph_metrics.py
│   ├── visualization.py
│   └── utils.py
├── data/
│   ├── papers/
│   └── results/
├── notebooks/
│   └── main_pipeline.ipynb
├── requirements.txt
└── README.md
```

## Run

1. Put PDF papers into `data/papers/`.
2. Open `notebooks/main_pipeline.ipynb` in Google Colab or locally.
3. Install dependencies if needed.
4. Configure:

```python
PAPERS_PATH = PROJECT_ROOT / "data" / "papers"
RESULTS_PATH = PROJECT_ROOT / "data" / "results"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SIMILARITY_THRESHOLD = 0.55
TOP_K = 5
USE_LLM = True
```

If `USE_LLM=False`, the project skips local LLM inference and builds the embedding graph plus `hybrid_graph = embedding_graph.copy()`.

## Outputs

After running the notebook, the main results are saved to `data/results/`:

```text
papers.csv
candidate_pairs.csv
llm_relations_raw.csv
llm_relations_filtered.csv
graph_metrics.csv
communities.csv
community_summary.csv
centrality.csv
path_example.csv
hybrid_nodes.csv
hybrid_edges.csv
embedding_graph.graphml
llm_graph.graphml
hybrid_graph.graphml
```

Optional HTML visualizations and metric plots are also saved when `pyvis` and `matplotlib` are available.

## Notes

- The LLM is loaded locally through `transformers`.
- The project does not call paid APIs.
- The project does not use Hugging Face Inference API.
- PDF title and abstract extraction use heuristic parsing, so `TITLE_MAP` in the notebook can be used to fix titles before graph construction.
