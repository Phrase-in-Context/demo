""" A flask-based web server

Usage:
    python server.py --config ../config/dev.json


"""
import sys
sys.path.append("../")

import os
import time
import json

import torch
import numpy as np
from scipy.special import softmax
from nltk.util import ngrams

from datasets import load_dataset
from retrieval_ranking import CreateLogger
from retrieval_ranking import ROOT_DIR
from system import System

from transformers import (
    AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)

from flask import Flask
from flask import request, render_template, jsonify
app = Flask(__name__, static_folder="static")


@app.route('/')
@app.route('/index')
def index():
    global datasets
    return render_template('index.html', examples=datasets["PR-pass"])


def pick_top_n(results, top_n):
    """ """
    new_results = []
    closed = set()
    for result in results:
        # if the higher-scored phrases are not a substring of result['phrase']
        if all([x not in result['phrase'][0] for x in closed]):
            new_results.append(result)
            closed.add(result['phrase'][0])

        if len(new_results) >= top_n:
            break

    return new_results


def prepare_phrase_list(context):
    global tokenizer

    tokenized_examples = tokenizer(
        context.replace("</br></br>", "  "),
        truncation="only_second",
        return_tensors="pt",
        return_offsets_mapping=True,
    )

    offset_mapping = tokenized_examples["offset_mapping"].numpy()[0].tolist()
    list_tokens = tokenizer.convert_ids_to_tokens(tokenized_examples["input_ids"][0])
    word_dict = {}

    # [1: -1]: Ignore <s> and </s> tokens
    for word_id, token, offset in zip(tokenized_examples.word_ids()[1:-1], list_tokens[1:-1], offset_mapping[1:-1]):
        if word_id not in word_dict:
            word_dict[word_id] = {"tokens": [], "offsets": []}

        word_dict[word_id]["tokens"].append(token)
        word_dict[word_id]["offsets"].append(offset)

    list_words, list_offsets = [], []
    for word_id, values in word_dict.items():
        list_words.append("".join(values["tokens"]))
        list_offsets.append([values["offsets"][0][0], values["offsets"][-1][1]])

    list_words = ["".join(word_dict[word_id]["tokens"]) for word_id in word_dict.keys()]
    list_words_processed = [token.replace("Ġ", "") for token in list_words]
    words_indices = [(word, i) for i, word in enumerate(list_words_processed)]
    list_ngram = []

    for n in range(1, 4):
        ngrams_indices = [grams for grams in ngrams(words_indices, n)]
        for grams_indices in ngrams_indices:
            phrase, indices = [], []
            for gram, idx in grams_indices:
                phrase.append(gram)
                indices.append(idx)

            list_ngram.append((" ".join(phrase).lower(), indices))

    return list_ngram, list_words, list_offsets


@app.route('/ranking_search', methods=['GET', 'POST'])
def ranking_search():
    """ """
    global logger
    global model_phrasebert

    start_time = time.time()

    text = request.form.get('paragraph_text', '').strip()
    query = request.form.get('query_text', '').strip()
    answer = request.form.get('answer_text', '').strip()
    extractor_name = request.form.get('extractor', '').strip()
    scorer = request.form.get('scorer', '').strip()
    answer_start_offset = int(request.form.get('start_index', -1))
    
    logger.debug("extractor_name: %s", extractor_name)
    logger.debug("scorer: %s", scorer)

    phrases_ngrams, list_tokens, offset_mapping = prepare_phrase_list(text)
    results = model_phrasebert.search_with_indices(query, phrases=phrases_ngrams, top_n=3)

    highlight_dict = {}
    answer_start_index, answer_end_index = 0, 0

    # context_start_index follows subwords but list_tokens follows words. Fix it tomorrow.
    for idx, offset in enumerate(offset_mapping):
        if offset[0] == answer_start_offset:
            answer_start_index = idx
        elif offset[1] == answer_start_offset + len(answer):
            answer_end_index = idx

    for idx, token in enumerate(list_tokens):
        if idx not in highlight_dict:
            highlight_dict[idx] = []

        if answer_start_index <= idx <= answer_end_index:
            highlight_dict[idx].append("ground-truth")

        for rank, result in enumerate(results, 1):
            pred_start_index = result["phrase"][1][0]
            pred_end_index = result["phrase"][1][-1]
            if pred_start_index <= idx <= pred_end_index:
                highlight_dict[idx].append("top-{}".format(rank))

    for idx, class_value in highlight_dict.items():
        if len(class_value) > 0:
            token_to_be_replaced = list_tokens[idx].replace("Ġ", "")
            list_tokens[idx] = list_tokens[idx].replace(token_to_be_replaced,'<span class="{}">{}</span>'.format(" ".join(class_value), token_to_be_replaced))

    end_time = time.time()
    diff_time = round((end_time - start_time)*1000)

    info = "[info] number of candidates: " + str(len(phrases_ngrams)) + '<br>'
    info = info + "[info] processing time (ms): " + str(diff_time) + '<br><br>'

    html_response_text = tokenizer.convert_tokens_to_string(list_tokens).replace("Ġ", " ")

    html_predictions = '<label for="result-sentences-container2"><b>Top {} predictions:</b></label><table>'.format(len(results))
    for idx, result in enumerate(results, 1):
        html_predictions += '<tr><td>{:.3f}</td><td width="25px"></td><td><span id="top-{}-phrase" class="top-{}">{}</span></td></tr>'.format(result['score'], idx, idx, result['phrase'][0])
    html_predictions += "</table>"

    return jsonify({'html': info + html_response_text.replace("  ", "</br></br>"), 'predictions': html_predictions})


@app.route('/qa_search', methods=['GET', 'POST'])
def qa_search():
    """ """
    global logger
    global model_longformer
    global tokenizer

    start_time = time.time()

    text = request.form.get('paragraph_text', '').strip().replace("</br></br>", "  ")
    query = request.form.get('query_text', '').strip()
    answer = request.form.get('answer_text', '').strip()
    answer_start_offset = int(request.form.get('start_index', -1))

    tokenized_examples = tokenizer(
        query, text,
        truncation="only_second",
        max_length=4096, stride=128,
        padding="max_length",
        return_tensors="pt",
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
    )

    offset_mapping = tokenized_examples["offset_mapping"].numpy()[0].tolist()
    for i in range(len(tokenized_examples["input_ids"])):
        # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
        # position is part of the context or not.
        sequence_ids = tokenized_examples.sequence_ids(i)
        offset_mapping[i] = [(o if sequence_ids[k] == 1 else None) for k, o in enumerate(offset_mapping[i])]

    tokenized_examples.to("cuda")

    with torch.no_grad():
        outputs = model_longformer(tokenized_examples.data["input_ids"], tokenized_examples.data["attention_mask"])

    start_logits = outputs.start_logits.cpu().data.numpy()[0]
    end_logits = outputs.end_logits.cpu().data.numpy()[0]

    start_scores = softmax(start_logits)
    end_scores = softmax(end_logits)

    start_indexes = np.argsort(start_logits)[-1 : -10 - 1 : -1].tolist()
    end_indexes = np.argsort(end_logits)[-1: -10 - 1: -1].tolist()

    n_best = 3
    max_answer_length = 10
    results = []

    for start_index in start_indexes:
        for end_index in end_indexes:
            # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
            # to part of the input_ids that are not in the context.
            if (start_index >= len(offset_mapping) or end_index >= len(offset_mapping) or
                offset_mapping[start_index] is None or len(offset_mapping[start_index]) < 2 or
                offset_mapping[end_index] is None or len(offset_mapping[end_index]) < 2):
                continue

            # Don't consider answers with a length that is either < 0 or > max_answer_length.
            if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                continue

            offsets = (offset_mapping[start_index][0], offset_mapping[end_index][1])

            results.append(
                {
                    "phrase": text[offsets[0]: offsets[1]],
                    "offsets": offsets,
                    "score": (start_scores[start_index] + end_scores[end_index]) / 2,
                    "start_logit": start_logits[start_index],
                    "end_logit": end_logits[end_index],
                    "start_index": start_index,
                    "end_index": end_index,
                }
            )

    results = sorted(results, key=lambda d: d['score'], reverse=True)[:n_best]
    list_tokens = tokenizer.convert_ids_to_tokens(tokenized_examples["input_ids"][0])
    list_tokens = [token for token in list_tokens if token != '<pad>']
    highlight_dict = {}

    context_start_index = [i for i, x in enumerate(list_tokens) if x == "</s>"][1] + 1
    answer_start_index, answer_end_index = 0, 0

    for idx, offset in enumerate(offset_mapping[context_start_index: -1]):
        if offset[0] == answer_start_offset:
            answer_start_index = idx + context_start_index
        elif offset[1] == answer_start_offset + len(answer):
            answer_end_index = idx + context_start_index

    for idx, token in enumerate(list_tokens):
        if idx not in highlight_dict:
            highlight_dict[idx] = []

        if answer_start_index <= idx <= answer_end_index:
            highlight_dict[idx].append("ground-truth")

        for rank, result in enumerate(results, 1):
            if result["start_index"] <= idx <= result["end_index"]:
                highlight_dict[idx].append("top-{}".format(rank))

    for idx, class_value in highlight_dict.items():
        if len(class_value) > 0:
            token_to_be_replaced = list_tokens[idx].replace("Ġ", "") if list_tokens[idx].startswith("Ġ") else list_tokens[idx]
            list_tokens[idx] = list_tokens[idx].replace(token_to_be_replaced, '<span class="{}">{}</span>'.format(" ".join(class_value), token_to_be_replaced))

    end_time = time.time()
    diff_time = round((end_time - start_time)*1000)

    info = "[info] processing time (ms): " + str(diff_time) + '<br><br>'
    html_response_text = tokenizer.convert_tokens_to_string(list_tokens[context_start_index: -1]).replace("Ġ", " ")

    html_predictions = '<label for="result-sentences-container2"><b>Top {} predictions:</b></label><table>'.format(len(results))
    for idx, result in enumerate(results, 1):
        html_predictions += '<tr><td>{:.3f}</td><td width="25px"></td><td><span id="top-{}-phrase" class="top-{}">{}</span></td></tr>'.format(result['score'], idx, idx, result['phrase'])
    html_predictions += "</table>"

    del outputs
    del tokenized_examples
    torch.cuda.empty_cache()

    return jsonify({'html': info + html_response_text.replace("  ", "</br></br>"), 'predictions': html_predictions})


@app.route('/dataset_select', methods=['GET', 'POST'])
def dataset_select():
    """ """
    global datasets
    global dataset_selected

    dataset_selected = request.form.get('new_dataset', '').strip()
    return jsonify({'examples': datasets[dataset_selected]})


def load_model(scorer):
    config_fpath = os.path.join(ROOT_DIR, "./model_config.json")
    with open(config_fpath) as f:
        config = json.load(f)

    scorer_name = scorer.split(':')[0].strip()
    scorer_type = scorer.split(':')[1].strip()

    scorers = [x for x in config if (x['scorer'] == scorer_name and x['scorer_type'] == scorer_type)]
    model_fpath = scorers[-1]['model_fpath'] if len(scorers) > 0 else ""
    tokenizer = None

    if scorer_name == "PhraseBERT":
        model = System()
        model.set_ss_scorer(scorer_name, model_fpath, scorer_type)
    else:
        model_name_or_path = model_fpath if model_fpath else scorer_type
        config = AutoConfig.from_pretrained(model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name_or_path, from_tf=False, config=config)
        model.to("cuda")
        model.eval()

    return model, tokenizer


def load_datasets():
    # Prepare examples loaded from Huggingface
    pr_pass = load_dataset("PiC/phrase_retrieval", "PR-pass")
    pr_page = load_dataset("PiC/phrase_retrieval", "PR-page")
    psd = load_dataset("PiC/phrase_sense_disambiguation")

    datasets = {
        "PR-pass": [example for example in pr_pass["test"]][:100],
        "PR-page": [example for example in pr_page["test"]][:100],
        "PSD": [example for example in psd["test"]][:100]
    }

    with open("resources/wiki_url_title.json", "r") as input_file:
        title_dict = json.load(input_file)

    for dataset, examples in datasets.items():
        for example in examples:
            if dataset == "PSD":
                example["title"] = title_dict[example["title"].split(",")[0]]
            else:
                example["title"] = title_dict[example["title"]]

            if "\n" in example["context"]:
                example["context"] = example["context"].replace("\n", "</br>")

    return datasets


if __name__ == '__main__':

    logger = CreateLogger()

    datasets = load_datasets()

    model_phrasebert, _ = load_model('PhraseBERT:phrase-bert-qa')
    model_longformer, tokenizer = load_model('Longformer:allenai/longformer-base-4096')

    # Other arguments: use_reloader, debug
    app.run(debug=True, host='0.0.0.0', port=5007)
