import gc
import json
import os
from datetime import datetime

import pandas as pd
import torch
from datasets import load_dataset

from ralm.utils import compare_metrics
from ralm.index_builder import IndexBuilder
from ralm.language_model import LanguageModel
from ralm.model_loader import ModelLoader
from ralm.ralm import RALM
from ralm.retriever import Retriever

def initialize_index_builder(knowledge_base, config):
    index_builder = IndexBuilder(knowledge_base, config['embedding_model_name'], **config['index_builder'])
    return index_builder.initialize_components()

def initialize_ralm(knowledge_base, config, model_loader_generation, model_loader_seq2seq):
    index, index_titles, doc_info = initialize_index_builder(knowledge_base, config)
    retriever = Retriever(index, doc_info, config['embedding_model_name'], model_loader_seq2seq, index_titles)
    language_model = LanguageModel(model_loader_generation, config['is_chat_model'], config['instruct_tokens'])
    del index, index_titles
    gc.collect()
    return RALM(retriever, language_model, **config['ralm'])

knowledge_base = pd.read_pickle('./resources/articles_l3.pkl')

truthful_qa = load_dataset("truthful_qa", "generation", split='validation').to_pandas()
test_data = truthful_qa[['question', 'best_answer', 'correct_answers', 'incorrect_answers']]

from config import control_configs, experiment_configs

for control_config, experiment_config in zip(control_configs, experiment_configs):

    time = datetime.now().strftime("%m-%d_%H-%M")
    results_dir = f'./results/{time}'
    os.makedirs(results_dir, exist_ok=True)

    model_loader_generation_control = ModelLoader(control_config['generation_model_name'], 'causal', quant_type='4bit')
    model_loader_seq2seq_control = ModelLoader(control_config['seq2seq_model_name'], 'seq2seq', quant_type='4bit')

    ralm_control = initialize_ralm(knowledge_base, control_config, model_loader_generation_control, model_loader_seq2seq_control)
    evaluation_control = ralm_control.evaluate(test_data)

    del ralm_control
    del model_loader_generation_control
    del model_loader_seq2seq_control
    gc.collect()
    torch.cuda.empty_cache()

    model_loader_generation_experiment =  ModelLoader(experiment_config['generation_model_name'], 'causal', quant_type='4bit')
    model_loader_seq2seq_experiment = ModelLoader(experiment_config['seq2seq_model_name'], 'seq2seq', quant_type='4bit')

    ralm_experiment = initialize_ralm(knowledge_base, experiment_config, model_loader_generation_experiment, model_loader_seq2seq_experiment)
    evaluation_experiment = ralm_experiment.evaluate(test_data)

    del ralm_experiment
    del model_loader_generation_experiment
    del model_loader_seq2seq_experiment
    gc.collect()
    torch.cuda.empty_cache()

    results, fig = compare_metrics(evaluation_control, evaluation_experiment)

    with open(os.path.join(results_dir, f'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    fig.savefig(os.path.join(results_dir, f'comparison.png'))

    with open(os.path.join(results_dir, f'config_control.json'), 'w') as f:
        json.dump(control_config, f, indent=4)
    with open(os.path.join(results_dir, f'config_experiment.json'), 'w') as f:
        json.dump(experiment_config, f, indent=4)

    evaluation_control.to_pickle(os.path.join(results_dir, f'evaluation_control.pkl'))
    evaluation_experiment.to_pickle(os.path.join(results_dir, f'evaluation_experiment.pkl'))