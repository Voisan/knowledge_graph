# Сравнительный анализ методов построения графов научных знаний

Учебный Python-проект для курсовой работы на тему: **"Построение графов знаний с помощью LLM"**.

В проекте вершина графа соответствует научной статье, а ребро описывает связь между двумя статьями. Главный акцент сделан не только на построении графов, но и на сравнении того, как разные способы формирования ребер меняют структуру графа.

## Цель

Построить несколько вариантов графа научных статей и сравнить их по структурным и семантическим характеристикам: плотности, связности, центральности, PageRank и community structure.

## Архитектура

```text
project/
├── data/
│   ├── papers/
│   ├── processed/
│   └── results/
├── src/
│   ├── pdf_parser.py
│   ├── preprocessing.py
│   ├── embeddings.py
│   ├── citation_extractor.py
│   ├── llm_relation_extractor.py
│   ├── graph_builder.py
│   ├── graph_metrics.py
│   ├── visualization.py
│   └── utils.py
├── notebooks/
│   └── main_pipeline.ipynb
├── requirements.txt
└── README.md
```

## Методы построения графа

1. **Embedding Similarity Graph**
   Статьи кодируются моделью `sentence-transformers/all-MiniLM-L6-v2`. Ребра создаются между top-k наиболее похожими статьями, если cosine similarity выше заданного порога.

2. **Citation Graph**
   Из секции References извлекаются библиографические записи. Если название статьи из корпуса встречается в списке литературы другой статьи, создается направленное ребро `PaperA -> PaperB` с отношением `CITES`.

3. **LLM Relation Graph**
   Локальная HuggingFace LLM, по умолчанию `Qwen/Qwen2.5-3B-Instruct`, классифицирует отношения только для candidate pairs из embedding graph. Разрешенные отношения: `SIMILAR_TO`, `BASED_ON`, `EXTENDS`, `COMPARES_WITH`, `SAME_TOPIC`, `NO_RELATION`.

4. **Hybrid Graph**
   Объединяет similarity edges, citation edges и LLM edges в `networkx.MultiDiGraph`. Это позволяет сравнить комбинированный подход с каждым отдельным источником связей.

## Запуск на Kaggle

1. Создайте Kaggle Dataset с PDF-файлами и подключите его к ноутбуку так, чтобы статьи были доступны в `/kaggle/input/papers`.
2. Загрузите файлы проекта в Kaggle Notebook или добавьте их в `/kaggle/working`.
3. Откройте `notebooks/main_pipeline.ipynb`.
4. При необходимости выполните ячейку установки зависимостей.
5. Проверьте конфигурацию:

```python
PAPERS_PATH = "/kaggle/input/papers"
RESULTS_PATH = "/kaggle/working/results"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
SIMILARITY_THRESHOLD = 0.55
TOP_K = 5
USE_LLM = False
```

По умолчанию `USE_LLM=False`, поэтому проект строит embedding, citation и hybrid graph без загрузки тяжелой LLM. Чтобы включить LLM-классификацию, установите `USE_LLM=True` и используйте Kaggle GPU.

## Метрики

Для каждого графа считаются:

- number of nodes
- number of edges
- density
- number of connected components для неориентированной версии
- average degree
- average clustering для неориентированной версии
- degree centrality
- PageRank для направленных графов
- top-10 nodes by degree
- top-10 nodes by PageRank
- community detection через `networkx.greedy_modularity_communities`

Сравнительная таблица сохраняется в:

```text
/kaggle/working/results/graph_metrics_comparison.csv
```

Полные метрики сохраняются в:

```text
/kaggle/working/results/graph_metrics_full.json
```

## Пример результатов

Ожидаемая интерпретация:

- **Similarity graph** часто получается более плотным, потому что семантически похожие статьи находятся даже без явных ссылок.
- **Citation graph** обычно более разреженный и направленный, но отражает реальные библиографические зависимости.
- **LLM graph** добавляет тип связи и объяснение, но требует больше вычислительных ресурсов.
- **Hybrid graph** объединяет разные источники связей и обычно дает более информативную структуру для анализа центральных статей и сообществ.

## Ограничения

- Извлечение title, abstract и references основано на простых эвристиках и может ошибаться на плохо распознанных PDF.
- Citation matching ищет совпадения названий статей в references, поэтому не покрывает сокращенные, неполные или сильно переформатированные ссылки.
- LLM запускается только на candidate pairs, иначе вычислительная стоимость быстро становится слишком высокой.
- Качество LLM-классификации зависит от выбранной модели, GPU-памяти и качества извлеченного текста.
- Проект не использует OpenAI API, платные API или внешние закрытые сервисы.

