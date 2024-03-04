control_configs = [
    {"generation_model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "seq2seq_model_name": "google/flan-t5-large",
    "is_chat_model": True,
    "instruct_tokens": ("[INST]","[/INST]"),
    "index_builder": {
        "metadata_cols": [],
        "tokenizer_model_name": None,
        "chunk_size": 64,
        "overlap": 8,
        "num_topics": 5,
        "passes": 10
        },
    "ralm": {
        "mmr": False,
        "lambda_": 0.5,
        "set_multiplier": 5,
        "expand_query": False,
        "top_k_docs": 3,
        "system_prompt": "You are a truthful expert question-answering bot and should correctly and concisely answer the following question",
        "repeat_system_prompt": False,
        "ensemble_perplexity": False,
        "ensemble_generation": False,
        "stride": -1,
        "query_len": 200,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 0.1,
        "num_beams": 2,
        "max_new_tokens": 50,
        "batch_size": 16
        }
    }
]
for config in control_configs:
    config["index_builder"]["tokenizer_model_name"] = config["generation_model_name"]


experiment_configs = [
    {"generation_model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "seq2seq_model_name": "google/flan-t5-large",
    "is_chat_model": True,
    "instruct_tokens": ("[INST]","[/INST]"),
    "index_builder": {
        "metadata_cols": ["title_en", "description_en"],
        "tokenizer_model_name": None,
        "chunk_size": 64,
        "overlap": 8,
        "num_topics": 5,
        "passes": 10
        },
    "ralm": {
        "mmr": False,
        "lambda_": 0.5,
        "set_multiplier": 5,
        "expand_query": False,
        "top_k_docs": 3,
        "system_prompt": "You are a truthful expert question-answering bot and should correctly and concisely answer the following question",
        "repeat_system_prompt": True,
        "ensemble_perplexity": False,
        "ensemble_generation": False,
        "stride": -1,
        "query_len": 200,
        "do_sample": False,
        "temperature": 1.0,
        "top_p": 0.1,
        "num_beams": 2,
        "max_new_tokens": 50,
        "batch_size": 16
        }
    }
]
for config in experiment_configs:
    config["index_builder"]["tokenizer_model_name"] = config["generation_model_name"]