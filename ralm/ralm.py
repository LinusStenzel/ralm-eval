import numpy as np
import pandas as pd
import re
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

def softmax_(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)

def robust_mean(data, trim_factor=0.05):
    """Calculate a weighted mean where weights are lower for points further from the median."""
    if len(data) == 0:
        return np.nan
    if len(np.unique(data)) == 1:
        return data[0]
    if len(np.unique(data)) == 2:
        return np.mean(data)

    sorted_data = np.sort(data)
    median = np.median(sorted_data)
    distances = np.abs(sorted_data - median)
    max_distance = np.quantile(distances, 1 - trim_factor)
    weights = np.clip(1 - distances / max_distance, 0, 1)
    return np.sum(sorted_data * weights) / np.sum(weights)

class RALM:
    """
    Integrates a retriever and a language model to generate responses based on retrieved information.
    This class uses a retriever to fetch relevant document (chunks) based on a query and employs a language model
    to synthesize responses incorporating the information from these documents and the original query.
    Its main functionality is exposed throug the evaluate method, which assesses the system's performance using test data.

    Attributes:
        retriever (Retriever): Instance of Retriever class for fetching documents.
        language_model (LanguageModel): Instance of LanguageModel class for generating text.
        system_prompt (str): A predefined prompt to prepend to each input.
    """

    def __init__(self, retriever, language_model, system_prompt="", repeat_system_prompt=True, top_k_docs=2, mmr=False, lambda_=0.5, set_multiplier=5, expand_query=False, ensemble_perplexity=False, ensemble_generation=False, stride=-1, query_len=200, do_sample=False, temperature=1, top_p=0, num_beams=1, max_new_tokens=50, batch_size=4):
        self.retriever = retriever
        self.language_model = language_model

        self.system_prompt = system_prompt
        self.repeat_system_prompt = repeat_system_prompt

        self.retrieval_kwargs = {
            "k": top_k_docs,
            "mmr": mmr,
            "lambda_": lambda_,
            "set_multiplier": set_multiplier,
            "expand_query": expand_query
        }

        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.ensemble_perplexity = ensemble_perplexity
        self.ensemble_generation = ensemble_generation
        self.stride = stride
        self.query_len = query_len
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

    def _prompt_template(self, query, docs):
        if docs:
            repeat_prompt = self.system_prompt + ": " if self.repeat_system_prompt else ""
            docs_str = "\n".join("- " + re.sub(r'[\t\n\r\f\v]', ' ', doc) for doc in docs) + "\n---"
            ralm_prompt =  f"{self.system_prompt} considering these information:\n{docs_str}\n{repeat_prompt}{query}\nAnswer: "
        else:
            ralm_prompt = f"{self.system_prompt}: {query}\nAnswer: "
        return self.language_model.instruct_start + ralm_prompt + self.language_model.instruct_end


    def evaluate(self, test_data):
        """
        Evaluates the RALM instance using the provided test data. It processes the data in batches, computing metrics
        such as perplexity and generating responses.

        Args:
            test_data (DataFrame): DataFrame containing query and expected response pairs.
            batch_size (int): Size of the batch for processing.

        Returns:
            DataFrame: DataFrame containing test data, generated responses, perplexities, and evaluation scores.
        """
        # Prepare batches
        test_samples_perplexity = self._prepare_batches_perplexity(test_data)
        test_batches_perplexity = [test_samples_perplexity[i:i + self.batch_size] for i in range(0, len(test_samples_perplexity), self.batch_size)]

        test_samples_generation = [{'query_id': idx, 'query': query} for idx, query in test_data['question'].items()]
        test_batches_generation = [test_samples_generation[i:i + self.batch_size] for i in range(0, len(test_samples_generation), self.batch_size)]

        # Evaluate perplexity and generate responses for each batch
        results_perplexity = self._evaluate_perplexity(test_batches_perplexity)
        results_generation = self._evaluate_generation(test_batches_generation)

        # Combine and format results
        return self._compute_evaluation_results(test_data, results_perplexity, results_generation)

    def _prepare_batches_perplexity(self, test_data):
        """
        Prepares batches of test data for perplexity evaluation.

        Args:
            test_data (DataFrame): DataFrame containing test data.

        Returns:
            List[List[Dict]]: List of batches, each batch being a list of query-response dictionaries.
        """
        entries = []
        for i in range(0, len(test_data), self.batch_size):
            # Construct a batch by pairing each query with all types of available answers
            batch = [
                {'query_id': idx,'query': row['question'], 'response': answer, 'type': response_type}
                for idx, row in test_data.iloc[i:i + self.batch_size].iterrows()
                for response_type, answers in [('best', [row['best_answer']]),
                                              ('correct', row['correct_answers']),
                                              ('incorrect', row['incorrect_answers'])]
                for answer in answers
            ]
            entries.extend(batch)
        return entries

    def _evaluate_perplexity(self, test_batches):
        """
        Evaluates perplexity for each batch in the test data.

        Args:
            test_batches (List[List[Dict]]): List of test batches.

        Returns:
            Dict: Results with perplexity values for each test entry.
        """
        results_perplexity = {}
        for test_batch in tqdm(test_batches, desc="Calculating Perplexity"):
            context_batch = [item['query'] for item in test_batch]
            target_batch = [item['response'] for item in test_batch]

            target_perplexity_batch = self._calculate_perplexity(context_batch, target_batch)

            for idx, item in enumerate(test_batch):
                query_id = item['query_id']
                if query_id not in results_perplexity:
                    results_perplexity[query_id] = {'perplexity_best': None, 'perplexities_correct': [], 'perplexities_incorrect': []}

                result = results_perplexity[query_id]
                if item['type'] == 'best':
                    result['perplexity_best'] = target_perplexity_batch[idx]
                elif item['type'] == 'correct':
                    result['perplexities_correct'].append(target_perplexity_batch[idx])
                elif item['type'] == 'incorrect':
                    result['perplexities_incorrect'].append(target_perplexity_batch[idx])
        return results_perplexity

    def _calculate_perplexity(self, query_batch, target_batch):
        """
        Calculates perplexity of target responses for given queries, indicating the model's confidence.

        Args:
            queries (list[str]): The input queries.
            target_responses (list[str]): Target responses for each query.
            top_k_docs (int): Number of top relevant documents to retrieve.

        Returns:
            list[float]: Perplexity values for each query-response pair.
        """
        docs_batch = self.retriever.retrieve(query_batch, **self.retrieval_kwargs)
        docs_str_batch = [[doc['text'] for doc in docs] for docs in docs_batch]

        if not self.ensemble_perplexity:
            context_batch = self._format_context(query_batch, docs_str_batch)
            return self.language_model.calculate_perplexity(context_batch, target_batch)
        else:
            docs_score_batch = [[doc['score'] for doc in docs] for docs in docs_batch]

            perplexities = []
            # Process each query-response-documents tuple for marginalizing over documents
            for query, target, docs_str, docs_scores in zip(query_batch, target_batch, docs_str_batch, docs_score_batch):

                contexts = [self._format_context([query], [[doc]])[0] for doc in docs_str]
                targets = [target] * len(docs_str)

                perplexities_sample = []
                # Process in batches
                for i in range(0, len(contexts), self.batch_size):
                    context_batch = contexts[i:i + self.batch_size]
                    target_batch = targets[i:i + self.batch_size]
                    perplexity_batch = self.language_model.calculate_perplexity(context_batch, target_batch)
                    perplexities_sample.extend(perplexity_batch)

                # Normalize scores to probabilities
                scores_normalized = softmax_(np.array(docs_scores))

                # Calculate weighted perplexity
                average_perplexity = np.dot(perplexities_sample, scores_normalized)
                perplexities.append(average_perplexity)

            return perplexities


    def _evaluate_generation(self, test_batches):
        """
        Generates responses for each batch in the test data.

        Args:
            test_batches (List[List[Dict]]): List of test batches.

        Returns:
            Dict: Generated responses for each test entry.
        """
        results_gen = {}
        for batch in tqdm(test_batches, desc="Calculating Generation"):
            batch_queries = [item['query'] for item in batch]
            input_texts, generated_responses = self._generate(batch_queries)

            for i, item in enumerate(batch):
                query_id = item['query_id']
                results_gen[query_id] = {'input_text': input_texts[i], 'generated_response': generated_responses[i]}
        return results_gen


    def _generate(self, query_batch):
        """
        Generates responses to queries by first retrieving relevant information and then synthesizing answers.

        Args:
            query_batch (list[str]): The input queries.
            top_k_docs (int): Number of top relevant documents to retrieve.
            max_new_tokens (int): Maximum token length of the response.
            query_length is measured in chars

        Returns:
            list[str]: Generated responses.
        """
        stride = self.stride if self.stride > 0 else self.max_new_tokens

        docs_batch = self.retriever.retrieve(query_batch, **self.retrieval_kwargs)
        docs_str_batch = [[doc['text'] for doc in docs] for docs in docs_batch]
        context_batch = self._format_context(query_batch, docs_str_batch)

        responses_enc = [[] for _ in query_batch]
        if not self.ensemble_generation:

            done_batch = [False for _ in query_batch]
            for i in range(0, self.max_new_tokens, stride):
                query_reponse_batch = [(q+" "+self.language_model.tokenizer.decode(r))[-self.query_len:] for q, r in zip(query_batch, responses_enc)]
                docs_batch = self.retriever.retrieve(query_reponse_batch, **self.retrieval_kwargs)
                docs_str_batch = [[doc['text'] for doc in docs] for docs in docs_batch]

                _context_batch = self._format_context(query_batch, docs_str_batch)
                running_context_batch = [c+self.language_model.tokenizer.decode(r) for c, r in zip(_context_batch, responses_enc)]

                new_responses_str, _done_batch = self.language_model.generate(running_context_batch, self.do_sample, self.temperature, self.top_p, self.num_beams, max_new_tokens=stride)
                done_batch = [bool(d + _d) for d, _d in zip(done_batch, _done_batch)]

                for idx, (done, new_response_str, running_context, response_enc) in enumerate(zip(done_batch, new_responses_str, running_context_batch, responses_enc)):
                    if not done:
                        new_response_str = new_response_str[len(running_context):]
                        new_response_enc = self.language_model.tokenizer.encode(new_response_str, add_special_tokens=False)
                        responses_enc[idx] = response_enc + new_response_enc

        else:

            done_idx = set([])
            for i in range(0, self.max_new_tokens, stride):

                query_reponse_batch = [(q+" "+self.language_model.tokenizer.decode(r))[-self.query_len:] for q, r in zip(query_batch, responses_enc)]
                docs_batch = self.retriever.retrieve(query_reponse_batch, **self.retrieval_kwargs)
                docs_str_batch = [[doc['text'] for doc in docs] for docs in docs_batch]
                docs_score_batch = [[doc['score'] for doc in docs] for docs in docs_batch]

                new_responses_enc = []
                for idx, (query, docs_str, docs_score, response_enc) in enumerate(zip(query_batch, docs_str_batch, docs_score_batch, responses_enc)):
                    if not idx in done_idx:
                        response_str = self.language_model.tokenizer.decode(response_enc)
                        if docs_str:
                            _contexts = [self._format_context([query], [[doc]])[0] + response_str for doc in docs_str]
                        else:
                            _contexts = [self._format_context([query], [[]])[0] + response_str]

                        new_response_enc = self.language_model.generate_greedy_ensemble(_contexts, docs_score, max_new_tokens=stride, batch_size=self.batch_size)
                    else:
                        new_response_enc = []

                    new_responses_enc.append(new_response_enc)

                for idx, new_response_enc in enumerate(new_responses_enc):
                      if new_response_enc: # Disclaimer: phenomena when doing stride with greedy ensemble: when sequence was done in last stride call, the current call will produce no tokens meaning empty list
                          responses_enc[idx] += new_response_enc
                      else:
                          done_idx.add(idx)

        responses_str = [self.language_model.tokenizer.decode(r, add_special_tokens=False) for r in responses_enc]
        return context_batch, responses_str

    def _format_context(self, queries, retrieved_docs):
        """
        Formats the input for the language model by combining retrieved docuemnts with queries.

        Args:
            queries (list[str]): A list of query dictionaries.
            retrieved_docs (list[list[str]]): A list containing lists of retrieved docs for each query.

        Returns:
            list[str]: Formatted input texts for the language model.
        """
        input_texts = []
        for docs, query in zip(retrieved_docs, queries):
            formatted_input = self._prompt_template(query, docs)
            input_texts.append(formatted_input)
        return input_texts


    def _compute_evaluation_results(self, test_data, results_perp, results_gen):
        """
        Formats and combines evaluation results into a single DataFrame.

        Args:
            test_data (DataFrame): Original test data.
            results_perp (Dict): Perplexity results.
            results_gen (Dict): Generated response results.

        Returns:
            DataFrame: Combined evaluation results.
        """
        data_and_evaluation = []
        for qid in test_data.index:
            result = test_data.loc[qid].to_dict()
            result.update({
                'perplexity_correct': robust_mean(results_perp[qid]['perplexities_correct'] + [results_perp[qid]['perplexity_best']]*2),
                'perplexity_incorrect': robust_mean(results_perp[qid]['perplexities_incorrect']),
                'input_text': results_gen[qid]['input_text'],
                'generated_response': results_gen[qid]['generated_response']
            })
            result.update(self._calculate_metrics(result))

            data_and_evaluation.append(result)

        return pd.DataFrame(data_and_evaluation)

    def _calculate_metrics(self, result):
        """
        Calculates and aggregates various metrics, including F1 scores and cosine similarities, for evaluation results.

        Args:
            result (dict): A dictionary representing the evaluation data and result for a single query.

        Returns:
            dict: A dictionary containing aggregated F1 scores and cosine similarities for different types of answers.
        """
        metrics = {}
        f1_scores_dict = {}
        similarities_dict = {}

        for answer_type in ['correct', 'incorrect']:
            answers = result[f'{answer_type}_answers']
            if answer_type == 'correct':
                answers = answers.tolist() + [result['best_answer']]*2

            # Calculate F1 scores and similarities
            f1_scores = self._calculate_f1_score([result['generated_response']], answers)
            similarities = self._calculate_cosine_similarity([result['generated_response']], answers)

            # Compute the gaussian weighted mean of the scores
            f1_scores_dict[f'f1_{answer_type}'] = np.mean(f1_scores)
            similarities_dict[f'similarity_{answer_type}'] = np.mean(similarities)

        metrics.update(f1_scores_dict)
        metrics.update(similarities_dict)
        return metrics


    def _calculate_f1_score(self, generated_answers, gold_answers, n='1'):
        """
        Calculates the F1 score based on the overlap between two lists of strings.

        Args:
            generated_answers (list[str]): Generated answers.
            gold_answers (list[str]): Gold standard answers.
            n (string): N-gram length for ROUGE-N calculation. Default is 1 (ROUGE-1).

        Returns:
            float: F1 score based on the overlap of the answers.
        """
        scorer = rouge_scorer.RougeScorer([f'rouge{n}'], use_stemmer=True)

        # Calculate f1 score for each pair
        scores = []
        for generated in generated_answers:
            for gold in gold_answers:
                score = scorer.score(gold, generated)
                rouge_n_f1 = score[f'rouge{n}'].fmeasure
                scores.append(rouge_n_f1)

        return scores

    def _calculate_cosine_similarity(self, generated_answers, gold_answers):
        """
        Calculates cosine similarity between the generated and gold answers.

        Args:
            generated_answers (list[str]): Generated answers.
            gold_answers (list[str]): Gold standard answers.

        Returns:
            float: Cosine similarity between the answers.
        """
         # Generate embeddings
        generated_embeddings = self.retriever.embedding_model.encode(generated_answers)
        gold_embeddings = self.retriever.embedding_model.encode(gold_answers)

        # Calculate cosine similarity for each pair
        similarities = []
        for gen_emb in generated_embeddings:
            for gold_emb in gold_embeddings:
                similarity = cosine_similarity([gen_emb], [gold_emb])[0][0]
                similarities.append(similarity)

        return similarities