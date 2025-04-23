# mlx7-week2-search-engine-on-MS-marco

Week 2 project at the ml.institute programme.
Building a search engine which takes in queries and produces a list of relevant documents.

We will build an appropriate architecture and train it on data from Microsoft Machine Reading Comprehension (MS MARCO), which is a collection of datasets for deep learning related to search tasks. <br> Datasets v1.1(102k rows) and v2.1(1.01M rows) from: https://huggingface.co/datasets/microsoft/ms_marco/viewer/v1.1/train?views%5B%5D=v11_train

STEPS:

0. Combine MS Marco splits into one file

   - python combine_data.py

1. Create random triplets

   - python create_triplets.py
   - length of positive samples = length of negative samples

2. Load - Pretrained GloVe
   builds:

   - vocab_to_int → {word: id}
   - int_to_vocab → {id: word}
   - embedding_matrix → torch.FloatTensor(n_vocab x 300)

   Save: - vocab_to_int.pkl, int_to_vocab.pkl - glove.6B.300d.npy (the matrix)

3. Preprocess & Tokenise MS MARCO v1.1 using GloVe vocab

   - create Triplets (Query, Positive, Negatives)
   - start with: Random negative sampling
     for every query:
     - 1 positive passage (from MS MARCO labels)
     - 1 random document as negative (not in its label set)
   - train the model on the random negatives saved in train_tokenised.pkl as training base
   - BM25-ranked negatives (Hard negatives) - too slow and unneficient

4. Build embedding_matrix.npy - 01_build_embedding_matrix.py

5. Train Initial Two-Tower Model on Random Triplets - 02_train_tower.py

   - use train_tokenised.pkl (random negatives)
   - save a checkpoint after a few epochs (enough for "confusion" to emerge)
   - freeze or continue training later = the goal is to get a model that starts to "believe" which documents are relevant — and get confused by tricky ones => then stop and mine hard negatives

6. Encode & Retrieve top-k + store in ChromaDB

   - encode all passages and save their vectors to ChromaDB:

   ```
     collection = chroma_client.get_or_create_collection(
         name="document",
         metadata={"hnsw:space": "cosine"}
     )
     collection.add(documents=[...], embeddings=[...])

   ```

   - encode a query using the trained model
   - etrieve nearest embeddings from ChromaDB:

   ```
   collection.query(query_embeddings=[...], n_results=5)
   ```

Instructions to run what I have:

- combine the files by running
  python combine_data.py
- create triplets for the individual splits by running
  python create_triplets.py
- tokenise:
  python tkn_ms_marco.py
- build vocab embeddins:
  python 01_build_embedding_matrix.py
- train initial Two-Tower model on random triplets
  python 02_train_tower.py

What do we need to do tomorrow?
