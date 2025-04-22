# mlx7-week2-search-engine-on-MS-marco

Week 2 project at the ml.institute programme.
Building a search engine which takes in queries and produces a list of relevant documents.

We will build an appropriate architecture and train it on data from Microsoft Machine Reading Comprehension (MS MARCO), which is a collection of datasets for deep learning related to search tasks. <br> Datasets v1.1(102k rows) and v2.1(1.01M rows) from: https://huggingface.co/datasets/microsoft/ms_marco/viewer/v1.1/train?views%5B%5D=v11_train

What do we need to do tomorrow?

- add MS Marco vocabulary to the Wiki one
  -- upload to hugging face
- train the word2vec on the new vocabulary - consider the impact of negative_passage when using the CBOW
  -- upload to weights and biases