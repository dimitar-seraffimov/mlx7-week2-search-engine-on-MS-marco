# mlx7-week2-search-engine-on-MS-marco

Week 2 project at the ml.institute programme.
Building a search engine which takes in queries and produces a list of relevant documents.

We will build an appropriate architecture and train it on data from Microsoft Machine Reading Comprehension (MS MARCO), which is a collection of datasets for deep learning related to search tasks. <br> Datasets v1.1(102k rows) and v2.1(1.01M rows) from: https://huggingface.co/datasets/microsoft/ms_marco/viewer/v1.1/train?views%5B%5D=v11_train

What do we need to do tomorrow?

1. Load Pretrained GloVe
   Builds:
   - vocab_to_int → {word: id}
   - int_to_vocab → {id: word}
   - embedding_matrix → torch.FloatTensor(n_vocab x 300)

Save: - vocab_to_int.pkl, int_to_vocab.pkl - glove.6B.300d.npy (the matrix)

2. Preprocess & Tokenise MS MARCO v1.1 (the large dataset) using GloVe vocab

- Save as triplets_glove_tokenised.pkl

3. Create Triplets (Query, Positive, Negatives) on the MS Marco v2.2

- Start with: Random negative sampling
  For every query:

  - 1 positive passage (from MS MARCO labels)
  - 1 random document as negative (not in its label set)

- Upgrade to: BM25-ranked negatives (Hard negatives) - do this directly?
  This step gives much stronger negatives and helps model generalise better.

4. Generate Embeddings (for Pos/Neg only)

5. Store in ChromaDB

6. Build Two-Tower Model - have 1-5 done on wednesday, implement this step on Thursday

- having the BM25-ranked negatives (Hard negatives) will help the model massively (or at least I hope so)
