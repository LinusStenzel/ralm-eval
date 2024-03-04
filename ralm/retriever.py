
from faiss import IDSelectorArray, SearchParameters
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import jensenshannon
import torch

def softmax_(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

class Retriever:
    """
    Handles the retrieval of relevant documents from a pre-built FAISS index.
    Enables querying with sentence transformers embeddings & diverse retrieval using Maximal Marginal Relevance.

    Attributes:
        index (faiss.Index): FAISS index for fast similarity search.
        doc_info (pd.DataFrame): DataFrame containing detailed information about documents.
        documents (list of str): List of original documents.
        embedding_model (SentenceTransformer): Model used for embedding the documents and queries.
    """

    def __init__(self, index, doc_info, embedding_model_name, model_loader_seq2seq, index_titles, model_loader_classification=None):
        """Initializes the Retriever class with necessary components.

        Args:
            index: FAISS index for fast retrieval.
            doc_info (DataFrame): DataFrame containing info about embedded document; aligned indices with index embeddings.
            documents (list): List of original documents.
            embedding_model_name (str): Name of the sentence transformer model.
        """
        self.index = index
        self.doc_info = doc_info
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_model = SentenceTransformer(embedding_model_name).to(self.device)

        self.model_seq2seq = model_loader_seq2seq.model
        self.tokenizer_seq2seq = model_loader_seq2seq.tokenizer
        self.text_query_pairs = [
            {"text": "Mitochondria play a crucial role in cellular respiration and energy production within human cells.", "query": "Cell Biology, Mitochondria, Energy Metabolism"},
            {"text": "The Treaty of Versailles had significant repercussions that contributed to the onset of World War II.", "query": "World History, Treaty of Versailles, World War II"},
            {"text": "What are the implications of the Higgs boson discovery for particle physics and the Standard Model?", "query": "Particle Physics, Higgs Boson, Standard Model"},
            {"text": "How did the Silk Road influence cultural and economic interactions during the Middle Ages?", "query": "Silk Road, Middle Ages, Cultural Exchange"}
        ]
        self.index_titles = index_titles

        self.model_classification = model_loader_classification.model if model_loader_classification else None
        self.tokenizer_classification = model_loader_classification.tokenizer if model_loader_classification else None


    def retrieve(self, query_batch, k, mmr, lambda_, set_multiplier, expand_query):
        """
        Retrieves the top-k most similar documents for each query in a batch of queries.
        Optionally applies Maximal Marginal Relevance for diverse retrieval.

        Args:
            query_batch (list of str): List of query strings.
            k (int): Number of documents to retrieve.
            mmr (bool): Use MMR for diverse retrieval.
            lambda_ (float): Balance parameter for MMR.
            set_multiplier (int): Multiplier to increase the initial retrieval set size when using MMR

        Returns:
            List[List[dict]]: List of lists containing formatted results of retrieved documents for each query.
        """

        if k == 0:
            return [[] for _ in query_batch]

        if expand_query:
            # TODO put into separate function
            eq_prompt_batch_str = []
            for query in query_batch:
                examples = self.text_query_pairs.copy()
                examples.append({"text": query, "query": ""})
                eq_prompt = "\n".join([f"Question: {example['text']}\nQuery Keywords: {example['query']}" for example in examples])
                eq_prompt_batch_str.append(eq_prompt)

            eq_prompt_batch_enc = self.tokenizer_seq2seq(eq_prompt_batch_str, return_tensors='pt', padding=True).to(self.device)
            eq_batch_enc = self.model_seq2seq.generate(**eq_prompt_batch_enc, max_length=25, num_return_sequences=1)
            eq_batch = self.tokenizer_seq2seq.batch_decode(eq_batch_enc, skip_special_tokens=True)
            eq_batch = [eq.split(", ") for eq in eq_batch]  # TODO what if expanded queries are not in corcect format for this splitting -> make more general

            eq_batch_indexed = [(eq, i) for i, eqs in enumerate(eq_batch) for eq in eqs]
            eq_batch_flat = [eq for eq, _ in eq_batch_indexed]
            eq_embeddings = self.embedding_model.encode(eq_batch_flat, show_progress_bar=False)
            _, indices_eq = self.index_titles.search(np.array(eq_embeddings), 7)

            # Reconstruct the original 2D array
            indices_eq_batch = [[] for _ in range(len(query_batch))]
            for ids, (_, i) in zip(indices_eq, eq_batch_indexed):
                indices_eq_batch[i].append(self.doc_info[self.doc_info['org_doc_id'].isin(ids)].index.tolist())
        else:
            indices_eq_batch = [[] for _ in range(len(query_batch))]


        # Batch encode the queries
        query_embeddings = self.embedding_model.encode(query_batch, show_progress_bar=False)

        # Process each query separately
        results_batch = []
        for query_embedding, ids_filter in zip(query_embeddings, indices_eq_batch):
            ids_filter = ids_filter if ids_filter else [list(range(self.index.ntotal))]

            id_filter_set = set()
            for id_filter in ids_filter:
                id_filter_set.update(id_filter)

            id_filter = list(id_filter_set)
            id_selector = IDSelectorArray(id_filter)
            # Search the index for similar documents, retrieve a larger set of documents if MMR is enabled
            similarities, indices = self.index.search(np.array([query_embedding]), k*set_multiplier if mmr else k, params=SearchParameters(sel=id_selector))
            indices, similarities = indices[0], similarities[0]

            if mmr:
                # Apply MMR for diverse retrieval
                doc_embeddings = np.vstack(self.doc_info.loc[indices]['embedding'].values)
                doc_embeddings = doc_embeddings.reshape(k*set_multiplier, -1)
                doc_topic_dist = np.vstack(self.doc_info.loc[indices]['topic_distribution'].values)
                mmr_indices, similarities = self._mmr(doc_embeddings, query_embedding, doc_topic_dist, k, lambda_)
                indices = [indices[idx] for idx in mmr_indices]

            results_batch.append([self._create_result(idx, sim) for idx, sim in zip(indices[:k], similarities)])

        return results_batch

    def _mmr(self, doc_embeddings, query_embedding, doc_topic_dist, k, lambda_):
        """
        Applies Maximal Marginal Relevance to select a diverse set of documents from the retrieved set.

        Args:
            doc_embeddings: Embeddings of the documents.
            query_embedding: Embedding of the query.
            doc_topic_dist: Topic distributions of the documents.
            lambda_param (float): Balances relevance and diversity.
            k (int): Number of documents to retrieve.

        Returns:
            list: Indices of selected documents in the order of their selection.
        """
        # Compute similarity scores
        similarities = softmax_(np.dot(doc_embeddings, query_embedding.T).flatten())

        # Initialize tracking lists
        selected_idx, scores, unselected_idx = [], [], set(range(len(similarities)))

        for _ in range(k):
            remaining_idx = np.array(list(unselected_idx))
            diversity_scores = np.zeros(len(remaining_idx))

            # Compute diversity scores for unselected documents
            for i, rem_idx in enumerate(remaining_idx):
                # Calculate Jensen-Shannon divergences between this unselected document and all selected documents
                diversities = [
                    jensenshannon(doc_topic_dist[rem_idx], doc_topic_dist[sel_idx])
                    for sel_idx in selected_idx
                ] if selected_idx else [0]

                # The average of these diversities represents how distinct this document is from the already selected set
                diversity_scores[i] = np.mean(diversities)

            # Calculate MMR scores and select document
            mmr_scores = lambda_ * similarities[remaining_idx] + (1 - lambda_) * diversity_scores
            next_index = remaining_idx[np.argmax(mmr_scores)]
            selected_idx.append(next_index)
            unselected_idx.remove(next_index)
            scores.append(np.max(mmr_scores))

        return selected_idx, scores


    def _create_result(self, idx, score):
        """
        Creates/builds a result dictionary of the retrieved document.

        Args:
            idx (int): Index of the result/document in doc_info.
            score (float): Similarity (& Diversity) score of document.

        Returns:
            dict: Dictionary containing the document text and additional information.
        """
        doc = self.doc_info.iloc[idx]
        # Format and return the result as a dictionary
        return {
            "text": doc["text"],
            "metadata": doc["metadata"],
            "doc_id": doc["org_doc_id"],
            "score": score
        }