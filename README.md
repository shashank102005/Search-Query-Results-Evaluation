# Search-Query-Results-Evaluation
Evaluation of the results returned by a search query, using natural language processing techniques

Various approaches adopted for feature engineering:

1. Bag of Words using the term frequencies in documents i.e.the search queries, product titles, descriptions and attributes
2. Bag of word term frequencies weighted by the term inverse document frequencies
3. Document similarity measures on the tf-idf transformed vestor space such as cosine similarity and Jaccard's coefficient ,
   BM 25 distance 
4. Latent Dirichlet Allocation for extracting topic distribution of the documents , and subsequently computing document distance measures on the topic space , such as Bhattacharya Distance and Kullback Lielber Divergence 

