# Project:
#   Localized Questions in VQA
# Description:
#   NLP functions and classes
# Author:
#   Sergio Tascon-Morales, Ph.D. Student, ARTORG Center, University of Bern

from collections import Counter 
import re
from tqdm import tqdm
from os.path import join as jp
import itertools
import json
import torch
import pickle
import os

def get_top_answers(answers, nans=2000):
    counts = Counter(answers).most_common() # get counts
    if len(set(answers)) == nans: # for binary case, return both answers
        return [e[0] for e in counts] # in this case return all answers, ordered from most to least frequent
    top_answers = [elem[0] for elem in counts[:nans]]
    return top_answers

def clean_text(text):
    text = text.lower().replace("\n", " ").replace("\r", " ")
    # replace numbers and punctuation with space
    punc_list = '!"#$%&()*+,-./:;<=>?@[\]^_{|}~' # * Leave numbers because some questions use coordinates+ '0123456789'
    t = str.maketrans(dict.fromkeys(punc_list, " "))
    text = text.translate(t)

    # replace single quote with empty character
    t = str.maketrans(dict.fromkeys("'`", ""))
    text = text.translate(t)

    # remove double spaces
    text = text.replace("  ", " ")

    return text


def tokenizer_nltk(text, tokenizer):
    text = clean_text(text)
    tokens = tokenizer(text)
    return tokens

def tokenizer_spacy(text, tokenizer):
    text = clean_text(text)
    tokens = list(tokenizer(text))
    tokens_list_of_strings = [str(token) for token in tokens]
    return tokens_list_of_strings

def tokenizer_re(text):
    WORD = re.compile(r'\w+')
    text = clean_text(text)
    tokens = WORD.findall(text)
    return tokens

def add_tokens(qa_samples, tokenizer_name):
    """Function to add tokens to data

    Parameters
    ----------
    qa_samples : _type_
        _description_
    tokenizer_name : _type_
        _description_

    Returns
    -------
    list
        original list of samples with each sample having a new field for the tokens

    Raises
    ------
    ValueError
        _description_
    """
    if tokenizer_name == 'nltk':
        from nltk import word_tokenize
    elif tokenizer_name == 'spacy':
        from spacy.tokenizer import Tokenizer
        from spacy.lang.en import English
        lang = English()
        tokenizer = Tokenizer(lang.vocab)
        

    for elem in tqdm(qa_samples):
        question_text = elem['question']
        if tokenizer_name == 'nltk':
            elem['question_tokens'] = tokenizer_nltk(question_text, word_tokenize)
        elif tokenizer_name == 'spacy':
            elem['question_tokens'] = tokenizer_spacy(question_text, tokenizer)
        elif tokenizer_name == 're':
            elem['question_tokens'] = tokenizer_re(question_text)
        else:
            raise ValueError('Unknown tokenizer')
    
    return qa_samples


def add_UNK_token_and_build_word_maps(data, min_word_frequency):
    # function to build vocabulary from question words and then build maps to indexes
    all_words_in_all_questions = list(itertools.chain.from_iterable(elem['question_tokens'] for elem in data))
    # count and sort 
    counts = Counter(all_words_in_all_questions).most_common()
    # get list of words (vocabulary for questions)
    vocab_words_in_questions = [elem[0] for elem in counts if elem[1] > min_word_frequency]
    # add_entry for tokens with UNK to data
    for elem in tqdm(data):
        elem['question_tokens_with_UNK'] = [w if w in vocab_words_in_questions else 'UNK' for w in elem['question_tokens']]
    # build maps
    vocab_words_in_questions.append('UNK') # Add UNK to the vocabulary
    map_word_index = {elem:i+1 for i,elem in enumerate(vocab_words_in_questions)} #*  +1  to avoid same symbol of padding
    map_index_word = {v:k for k,v in map_word_index.items()} 
    return data, map_word_index, map_index_word


def add_UNK_token(data, vocab):
    for elem in tqdm(data):
        elem['question_tokens_with_UNK'] = [w if w in vocab else 'UNK' for w in elem['question_tokens']]
    return data



def encode_questions(data, map_word_index, question_vector_length):
    for elem in tqdm(data):
        # add question length
        elem['question_length'] = min(question_vector_length, len(elem['question_tokens_with_UNK']))
        elem['question_word_indexes'] = [0]*question_vector_length # create list with question_vector_length zeros
        for i, word in enumerate(elem['question_tokens_with_UNK']):
            if i < question_vector_length:
                # using padding to the right. Add padding left?
                elem['question_word_indexes'][i] = map_word_index[word] # replace word with index in vocabulary
    return data



def encode_answers(data, map_answer_index):
    # function to encode answers. If they are not in the answer vocab, they are mapped to -1
    if 'answers_occurence' in data[0]: # if there are multiple answers (VQA2 dataset)
        for i, elem in enumerate(data):
            answers = []
            answers_indexes = []
            answers_count = []
            unknown_answer_symbol = map_answer_index['UNK']
            elem['answer_index'] = map_answer_index.get(elem['answer'], unknown_answer_symbol) # unknown_answer_symbol for unknown answers
            for answer in elem['answers_occurence']:
                answer_index = map_answer_index.get(answer[0], unknown_answer_symbol)
                #if answer_index != unknown_answer_symbol:
                answers += answer[1]*[answer[0]] # add all answers
                answers_indexes += answer[1]*[answer_index]
                answers_count.append(answer[1])
            elem['answers'] = answers 
            elem['answers_indexes'] = answers_indexes
            elem['answers_counts'] = answers_count
    else:
        for elem in tqdm(data):
            unknown_answer_symbol = map_answer_index['UNK']
            elem['answer_index'] = map_answer_index.get(elem['answer'], unknown_answer_symbol) # unknown_answer_symbol for unknown answers
    return data







def process_qa(config, data_train, data_val, data_test, data_testdev = None):

    if config['alt_questions']:
        max_question_len = config['max_question_length_alt']
        # for each subset, swap question and question_alt fields. This allows alternative questions to be treated as normal questions, which is useful in the end
        for d in [data_train, data_val, data_test]:
            for elem in d:
                elem['question'], elem['question_alt'] = elem['question_alt'], elem['question']
        if data_testdev is not None:
            for elem in data_testdev:
                elem['question'], elem['question_alt'] = elem['question_alt'], elem['question']
    else:
        max_question_len = config['max_question_length']

    # function to process questions and answers using functions from nlp.py This function can be used on other datasets
    all_answers = [elem['answer'] for elem in data_train]

    # get top answers
    top_answers = get_top_answers(all_answers, config['num_answers'])

    # get maps for answers
    top_answers.append('UNK') # add unknown symbol answer
    map_answer_index = {elem:i for i, elem in enumerate(top_answers)}
    map_index_answer = top_answers.copy()

    # remove examples for which answer is not in top answers
    # data_train = nlp.remove_examples_if_answer_not_common(data_train, top_answers)

    # tokenize questions for each subset
    print('Tokenizing questions...')
    data_train = add_tokens(data_train, config['tokenizer'])
    data_val = add_tokens(data_val, config['tokenizer'])
    data_test = add_tokens(data_test, config['tokenizer'])
    if data_testdev is not None:
        data_testdev = add_tokens(data_testdev, config['tokenizer'])

    # insert UNK tokens and build word maps
    print("Adding UNK tokens...")
    data_train, map_word_index, map_index_word = add_UNK_token_and_build_word_maps(data_train, config['min_word_frequency'])
    words_vocab_list = list(map_index_word.values())
    data_val = add_UNK_token(data_val, words_vocab_list)
    data_test = add_UNK_token(data_test, words_vocab_list)
    if data_testdev is not None:
        data_testdev = add_UNK_token(data_testdev, words_vocab_list)

    # encode questions
    print("Encoding questions...")
    data_train = encode_questions(data_train, map_word_index, max_question_len)
    data_val = encode_questions(data_val, map_word_index, max_question_len)
    data_test = encode_questions(data_test, map_word_index, max_question_len)
    if data_testdev is not None:
        data_testdev = encode_questions(data_testdev, map_word_index, max_question_len)

    # encode answers
    print("Encoding answers...")
    data_train = encode_answers(data_train, map_answer_index)
    data_val = encode_answers(data_val, map_answer_index)
    if 'answer' in data_test[0]: # if test set has answers
        data_test = encode_answers(data_test, map_answer_index)

    # build return dictionaries
    if data_testdev is not None:
        sets = {'trainset': data_train, 'valset': data_val, 'testset': data_test, 'testdevset': data_testdev}
    else:
        sets = {'trainset': data_train, 'valset': data_val, 'testset': data_test}
    maps = {'map_index_word': map_index_word, 'map_word_index': map_word_index, 'map_index_answer': map_index_answer, 'map_answer_index': map_answer_index}

    # sets: {'trainset': trainset, 'valset': valset, 'testset': testset, 'testdevset': testdevset}
    # maps: {'map_index_word': map_index_word, 'map_word_index': map_word_index, 'map_index_answer': map_index_answer, 'map_answer_index': map_answer_index}
    return sets, maps 