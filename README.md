# mlx7-week2-search-engine-on-MS-marco

Week 2 project at the ml.institute programme.
Building a search engine which takes in queries and produces a list of relevant documents.

We will build an appropriate architecture and train it on data from Microsoft Machine Reading Comprehension (MS MARCO), which is a collection of datasets for deep learning related to search tasks. <br> Datasets v1.1(102k rows) and v2.1(1.01M rows) from: https://huggingface.co/datasets/microsoft/ms_marco/viewer/v1.1/train?views%5B%5D=v11_train

Instructions:

- combine MS MARCO splits into one file
  python s00_combine_data.py
- Create triplets (query, positive, random negative)
  python s01_create_triplets.py
- tokenise queries & passages using GloVe vocab
  python s02_tkn_ms_marco.py
- build vocab embeddins:
  python s03_build_embedding_matrix.py
- train initial Two-Tower model on random triplets
  python s04_train_tower.py
- save encodings to chromadb
  python s05_encode_docs_to_chromadb.py
- mine hard negatives using ChromaDB + trained model
  python s06_mine_hard_negatives.py
- train Two-Tower model again with hard negatives
  python s07_train_w_hard_negatives.py
- re-encode passages using the improved model
  python s08_reencode_all.py
- run real user queries against ChromaDB
  python s09_query_retrieve.py

What do we need to do tomorrow?

- re-encode passages with the improved model
  run s08_encode_w_hard_model.py

- run user queries
  python s09_query_retrieve.py

- try to implement RNN-based encoder
  replace the average pooling in the current Two-Tower model with a GRU or LSTM.
  update the TwoTowerModel to use an RNN encoder for both query and passage towers.

- train the RNN version
  adapt the training script (e.g. 02_train_tower.py) to train the RNN-based model.
  log and compare performance with the baseline using Weights & Biases.
