from typing import Iterable
import torch
import json
from rouge_score import rouge_scorer
import spacy
from bs4 import BeautifulSoup
import numpy as np
import nltk
import transformers
import torch.nn.functional as F
import lrqa.preproc.simple as simple
from tqdm import auto as tqdm_lib


def tqdm(iterable=None, desc=None, total=None, initial=0):
    return tqdm_lib.tqdm(
        iterable=iterable,
        desc=desc,
        total=total,
        initial=initial,
    )

def maybe_tqdm(iterable=None, desc=None, total=None, initial=0, verbose=True):
    if verbose:
        return tqdm(iterable=iterable, desc=desc, total=total, initial=initial)
    else:
        return iterable

def read_jsonl(path):
    # Manually open because .splitlines is different from iterating over lines
    ls = []
    with open(path, "r") as f:
        for idx,line in enumerate(f):
            ls.append(json.loads(line))
#             if idx==0:
#                 print(json.loads(line))
#                 print(line)
    
    print(len(ls),"file_length for path = ", path)
    return ls

def write_file(data, path, mode="w", **kwargs):
    with open(path, mode=mode, **kwargs) as f:
        f.write(data)

def write_jsonl(data, path):
    assert isinstance(data, list)
    lines = [
        to_jsonl(elem)
        for elem in data
    ]
    print(len(lines)," for file to be written at path = ", path)
    write_file("\n".join(lines), path)

def to_jsonl(data):
    return json.dumps(data).replace("\n", "")

def get_clean_text(str_obj):
    if str_obj is None:
        return ""
    return " ".join(str(str_obj).strip().split())

def format_nice_text(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    p_list = soup.findAll('p')
    if len(p_list) == 0:
        # Fall-back for if we have no <p> tags to work off
        return " ".join(soup.get_text().strip().split())
    else:
        text_list = []
        header = get_clean_text(p_list[0].prev_sibling)
        if header:
            text_list.append(header)
        for p_elem in p_list:
            clean_p_text = get_clean_text(p_elem.get_text())
            if clean_p_text:
                text_list.append(clean_p_text)
            clean_p_suffix = get_clean_text(p_elem.next_sibling)
            if clean_p_suffix:
                text_list.append(clean_p_suffix)
        return "\n\n".join(text_list)

class SimpleScorer:
    def __init__(self, metrics=(("rouge1", "r"),), use_stemmer=True):
        self.metrics = metrics
        self.scorer = rouge_scorer.RougeScorer(
            [metric[0] for metric in self.metrics],
            use_stemmer=use_stemmer,
        )

    def score(self, reference: str, target: str):
        scores = self.scorer.score(reference, target)
        sub_scores = []
        for metric, which_score in self.metrics:
            score = scores[metric]
            if which_score == "p":
                score_value = score.precision
            elif which_score == "r":
                score_value = score.recall
            elif which_score == "f":
                score_value = score.fmeasure
            else:
                raise KeyError(which_score)
            sub_scores.append(score_value)
        return np.mean(sub_scores)


class FastTextScorer:
    def __init__(self, data, use_cache=True, verbose=True):
        if isinstance(data, str):
            data = torch.load(data)
        self.data_dict = {k: data["arr_data"][i] for i, k in enumerate(data["keys"])}
        self.nlp = spacy.load('en_core_web_sm', disable=['ner', 'tagger', "lemmatizer", "attribute_ruler"])
        self.use_cache = use_cache
        if use_cache:
            self.cache = {}
        else:
            self.cache = None
        self.verbose = verbose
        self.unk_set = set()

    def _embed_single(self, string: str):
        token_list = [str(token) for token in self.nlp(string)]
        token_embeds = []
        for token in token_list:
            if token in self.data_dict:
                token_embeds.append(self.data_dict[token])
            else:
                if self.verbose and token not in self.unk_set:
                    print(f"Verbose: Did not find '{token}'")
                    self.unk_set.add(token)
        if not token_embeds:
            return np.zeros(300)
        token_embeds = np.array(token_embeds)
        return token_embeds.mean(0)

    def score(self, reference: str, target: str):
        if self.use_cache:
            if reference not in self.cache:
                self.cache[reference] = self._embed_single(reference)
            if target not in self.cache:
                self.cache[target] = self._embed_single(target)
            ref_embed = self.cache[reference]
            tgt_embed = self.cache[target]
        else:
            ref_embed = self._embed_single(reference)
            tgt_embed = self._embed_single(target)
        return cosine_similarity(ref_embed, tgt_embed)


class DPRScorer:
    def __init__(self,
                 context_encoder_name="facebook/dpr-ctx_encoder-multiset-base",
                 question_encoder_name="facebook/dpr-question_encoder-multiset-base",
                 device=None,
                 use_cache=True, verbose=True):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = transformers.DPRQuestionEncoderTokenizer.from_pretrained(question_encoder_name)
        self.context_encoder = transformers.DPRContextEncoder.from_pretrained(context_encoder_name).to(device)
        self.question_encoder = transformers.DPRQuestionEncoder.from_pretrained(question_encoder_name).to(device)
        self.device = device
        self.use_cache = use_cache
        if use_cache:
            self.cache = {}
        else:
            self.cache = None
        self.verbose = verbose
        self.unk_set = set()

    def _convert_to_batch(self, string):
        return {k: torch.tensor([v]).to(self.device) for k, v in self.tokenizer(string).items()}

    def _embed_context(self, context: str):
        context_batch = self._convert_to_batch(context)
        with torch.no_grad():
            out = self.context_encoder(**context_batch)
        return out.pooler_output[0].cpu().numpy()

    def _embed_question(self, question: str):
        query_batch = self._convert_to_batch(question)
        with torch.no_grad():
            out = self.question_encoder(**query_batch)
        return out.pooler_output[0].cpu().numpy()

    def score(self, reference: str, target: str):
        # Reference <- question
        # Target <- context
        if self.use_cache:
            if reference not in self.cache:
                self.cache[reference] = self._embed_question(reference)
            if target not in self.cache:
                self.cache[target] = self._embed_context(target)
            ref_embed = self.cache[reference]
            tgt_embed = self.cache[target]
        else:
            ref_embed = self._embed_question(reference)
            tgt_embed = self._embed_context(target)
        return -np.linalg.norm(ref_embed - tgt_embed)
    
    
def load_fasttext_vectors(fname, max_lines, dim=300):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    arr_data = np.zeros([max_lines, dim])
    keys = []
    for i, line in enumerate(fin):
        if i == max_lines:
            break
        tokens = line.rstrip().split(' ')
        arr_data[i] = np.array(list(map(float, tokens[1:])))
        keys.append(tokens[0])
    return {
        "keys": keys,
        "arr_data": arr_data,
    }


def cosine_similarity(arr1, arr2):
    return F.cosine_similarity(
        torch.from_numpy(arr1.reshape(1, 300)),
        torch.from_numpy(arr2.reshape(1, 300)),
    )[0]


def get_sent_data(raw_text, clean_text=True):
    """Given a passage, return sentences and word counts."""
    nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])
    if clean_text:
        if isinstance(raw_text, list):
            raw_text = "\n".join(raw_text)
        context = simple.format_nice_text(raw_text)
    else:
        assert isinstance(raw_text, str)
        context = raw_text
    sent_data = []
    for sent_obj in nlp(context).sents:
        sent_data.append({
            "text": str(sent_obj).strip(),
            "word_count": len(sent_obj),
        })
    return sent_data


def get_top_sentences(query: str, sent_data: list, max_word_count: int, scorer: SimpleScorer):
    scores = []
    for sent_idx, sent_dict in enumerate(sent_data):
        scores.append((sent_idx, scorer.score(query, sent_dict["text"])))

    # Sort by rouge score, in descending order
    sorted_scores = sorted(scores, key=lambda _: _[1], reverse=True)

    # Choose highest scoring sentences
    chosen_sent_indices = []
    total_word_count = 0
    for sent_idx, score in sorted_scores:
        sent_word_count = sent_data[sent_idx]["word_count"]
        if total_word_count + sent_word_count > max_word_count:
            break
        chosen_sent_indices.append(sent_idx)
        total_word_count += sent_word_count

    # Re-condense article
    shortened_article = " ".join(sent_data[sent_idx]["text"] for sent_idx in sorted(chosen_sent_indices))
    return shortened_article


    
    
# def cosine_similarity(arr1, arr2):
#     return F.cosine_similarity(
#         torch.from_numpy(arr1.reshape(1, 300)),
#         torch.from_numpy(arr2.reshape(1, 300)),
#     )[0]


# def get_sent_data(raw_text, clean_text=True):
#     """Given a passage, return sentences and word counts."""
#     nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])
#     if clean_text:
#         if isinstance(raw_text, list):
#             raw_text = "\n".join(raw_text)
#         context = format_nice_text(raw_text)
#     else:
#         assert isinstance(raw_text, str)
#         context = raw_text
#     sent_data = []
#     for sent_obj in nlp(context).sents:
#         sent_data.append({
#             "text": str(sent_obj).strip(),
#             "word_count": len(sent_obj),
#         })
#     return sent_data


# def get_top_sentences(query: str, sent_data: list, max_word_count: int, scorer: SimpleScorer):
#     scores = []
#     for sent_idx, sent_dict in enumerate(sent_data):
#         scores.append((sent_idx, scorer.score(query, sent_dict["text"])))

#     # Sort by rouge score, in descending order
#     sorted_scores = sorted(scores, key=lambda _: _[1], reverse=True)

#     # Choose highest scoring sentences
#     chosen_sent_indices = []
#     total_word_count = 0
#     for sent_idx, score in sorted_scores:
#         sent_word_count = sent_data[sent_idx]["word_count"]
#         if total_word_count + sent_word_count > max_word_count:
#             break
#         chosen_sent_indices.append(sent_idx)
#         total_word_count += sent_word_count

#     # Re-condense article
#     shortened_article = " ".join(sent_data[sent_idx]["text"] for sent_idx in sorted(chosen_sent_indices))
#     return shortened_article


def process_file(input_path, output_path, scorer: SimpleScorer, query_type="question", max_word_count=500,
                 verbose=False, clean_text=True, ):
    data = read_jsonl(input_path)
    out = []
#     print(data[0].keys())
    for idx,row in enumerate(data):
        questions = row["questions"]
        for idx2,item in enumerate(questions):
            item["question"+str(idx2+1)] = item["question"]
            del item["question"]
        row["questions"] = questions
        data[idx] = row
    
#     print(data[0]["questions"][0])
       
    for rowx in maybe_tqdm(data, verbose=verbose):
        sent_data = get_sent_data(rowx["article"], clean_text=clean_text)
#         print(sent_data)
        i = 1
        
        while True:
            if i-1 == len(rowx["questions"]):
                break
            row = rowx["questions"][i-1]
            if f"question{i}" not in row:
                print("row[questions] = ",row)
                print("*"*100)
#                 if i==1:
                print("I am breaking at i = ", i)
#                 print(f"question{i}")
#                 print("*"*100)
#                 print(row)
                break
            if query_type == "question":
                query = row[f"question{i}"].strip()
            elif query_type == "question_answer":
                true_label = row["gold_label"] - 1
                query = row[f"question{i}"].strip() + " " + row["options"][true_label].strip()
                if i==1:
                    print(query)
            elif query_type == "answer_only":
                try:
                    true_label = row["gold_label"] - 1
                    query = row["options"][true_label].strip()
                except:
                    break
            elif query_type == "oracle_answer":
                query = row[f"question{i}option{row[f'question{i}_gold_label']}"].strip()
            elif query_type == "oracle_question_answer":
                query = (
                    row[f"question{i}"].strip()
                    + " " + row[f"question{i}option{row['question{i}_gold_label']}"].strip()
                )
            else:
                raise KeyError(query_type)
            shortened_article = get_top_sentences(
                query=query,
                sent_data=sent_data,
                max_word_count=max_word_count,
                scorer=scorer,
            )
            if "gold_label" not in row.keys():
                row["gold_label"] = -1
               
            out.append({
                "context": shortened_article,
                "query": " " + row[f"question{i}"].strip(),
                "option_0": " " + row["options"][0].strip(),
                "option_1": " " + row["options"][1].strip(),
                "option_2": " " + row["options"][2].strip(),
                "option_3": " " + row["options"][3].strip(),
                "label": row["gold_label"] - 1,
#                 "option_0": " " + row[f"question{i}option1"].strip(),
#                 "option_1": " " + row[f"question{i}option2"].strip(),
#                 "option_2": " " + row[f"question{i}option3"].strip(),
#                 "option_3": " " + row[f"question{i}option4"].strip(),
#                 "label": row[f"question{i}_gold_label"] - 1,
            })
            print(len(out)," this is the output_length at iteration - ",i)
            i += 1
    print(len(out)," this is the output_length")
    write_jsonl(out, output_path)
