# mlx7-week2-search-engine-on-MS-marco

Week 2 project at the ml.institute programme.
Building a search engine which takes in queries and produces a list of relevant documents.

We will build an appropriate architecture and train it on data from Microsoft Machine Reading Comprehension (MS MARCO), which is a collection of datasets for deep learning related to search tasks. <br> Datasets v1.1(102k rows) and v2.1(1.01M rows) from: https://huggingface.co/datasets/microsoft/ms_marco/viewer/v1.1/train?views%5B%5D=v11_train

Instructions:
1. combine MS MARCO splits into one file
   - python s00_combine_data.py
2. Create triplets (query, positive, random negative)
   - python s01_create_triplets.py
3. tokenise queries & passages using GloVe vocab
   - python s02_tkn_ms_marco.py
4. build vocab embeddins:
   - python s03_build_embedding_matrix.py
5. train initial Two-Tower model on random triplets
   - python s04_train_tower.py
6. save encodings to chromadb
   - python s05_encode_docs_to_chromadb.py
7. mine hard negatives using ChromaDB + trained model
   - python s06_mine_hard_negatives.py
8. train Two-Tower model again with hard negatives
   - python s07_train_w_hard_negatives.py
9. re-encode passages using the improved model
   - python s08_reencode_all.py
10. run real user queries against ChromaDB
    - python s09_query_retrieve.py

What have I achieved in this week? Reflections and future ideas for improvements:

- combined MS MARCO splits into a single Parquet for end-to-end processing
- generated query–positive–negative triplets with random sampling
- built GloVe-based vocabulary and embedding matrix aligned to our tokens
- trained an initial Two-Tower model on random negatives
- indexed passage embeddings into ChromaDB for fast retrieval
- mined hard negatives via ChromaDB and updated our triplet set -> this doesn't work well!
- retrained the Two-Tower model with hard negatives, reducing triplet loss -> this doesn't work well!
- re-encoded all passages with the improved checkpoint into ChromaDB
- validated the full query→embedding -> search pipeline / python s09_query_retrieve.py prompts the user to "ask" a query

What we could have done better?

- python s09_query_retrieve.py
  --> the results which I am getting are terrible (in terms of real connection to the query)
  --> I understand how everything works and is supposed to though!

- try to implement RNN-based encoder - did not have time for that...
  replace the average pooling in the current Two-Tower model with a GRU or LSTM.
  update the TwoTowerModel to use an RNN encoder for both query and passage towers.

- train the RNN version - did not have time for that...
  adapt the training script (e.g. 02_train_tower.py) to train the RNN-based model.
  log and compare performance with the baseline using Weights & Biases.
