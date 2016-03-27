# Search-Query-Results-Evaluation
Evaluation of the results returned by a search query, using natural language processing techniques

Various approaches adopted for feature engineering:

1. Bag of Words using the term frequencies in documents i.e.the search queries, product titles, descriptions and attributes
2. Bag of word term frequencies weighted by the term inverse document frequencies
3. Document similarity measures on the tf-idf transformed vector space such as cosine similarity and Jaccard's coefficient ,
   BM 25 distance 
4. Latent Dirichlet Allocation for extracting topic distribution of the documents , and subsequently computing document         distance measures on the topic space , such as Bhattacharya Distance and Kullback Lielber Divergence.
5. Latent Semantic Analysis and Non Negative Matrix Factorization , followed by  distance measures such as cosine similarity 
6. Word embedding using the Word2Vec algorithm implemented in Gensim Python package, followed by computing cosine similarity    measure abnd the word mover's distance between documents


