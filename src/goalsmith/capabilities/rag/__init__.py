"""Document search: ingest files, index them in Qdrant, retrieve with citations.

files --ingest--> chunks --embed + BM25--> Qdrant
query --dense + BM25--> RRF fusion --> rerank --> top k chunks with source ids
"""
