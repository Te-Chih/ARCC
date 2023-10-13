import os
import sys
# 解决linux下无法导入自己的包的问题
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
import json
import torch
from tqdm import tqdm

import os

from src.utils import parseJson, saveJson
import random
import pandas as pd
from transformers import BertTokenizer,BertModel
torch.cuda.set_device(0)
device = torch.device("cuda")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
current_path = os.path.abspath(os.path.dirname(__file__)) + '/'
bert = BertModel.from_pretrained('bert-base-uncased')
bert.to(device)
# tokenizer.to(device)
def produce_bert_corpus_TKVB():
    res = {}
    papers_path = current_path + "../../data/aminerData/pubs_raw.json"
    papers = parseJson(papers_path)
    for pid,paper in papers.items():
        title = str(paper.get("title","")) + "."
        abstract = str(paper.get("abstract","")) + "."
        venue = str(paper.get("venue", ""))
        keywords = paper.get("keywords","")

        if isinstance(keywords,list):
            kw = " ".join(keywords)
            kw = kw + "."
            corpus = title + " " + abstract + " " + kw + " " + venue
        else:
            corpus = title + " " + abstract  + " " + venue
        res[pid] = corpus
    saveJson( current_path + "../../data/aminerDataProcess/bert_corpus_TKVB.json", res)

def produce_bert_corpus_TKV():
    res = {}
    papers_path = current_path + "../../data/aminerData/pubs_raw.json"
    papers = parseJson(papers_path)
    for pid, paper in papers.items():
        title = str(paper.get("title", "")) + "."
        abstract = str(paper.get("abstract", "")) + "."
        venue = str(paper.get("venue", ""))
        keywords = paper.get("keywords", "")

        if isinstance(keywords, list):
            kw = " ".join(keywords)
            kw = kw + "."
            corpus = title + " " + kw + " " + venue
        else:
            corpus = title + " " + venue
        res[pid] = corpus
    saveJson(current_path + "../../data/aminerDataProcess/bert_corpus_TKV.json",res )
    # return current_path + "../../data/aminerDataProcess/bert_courpus_TKV.json"

def produce_bert_corpus_v2():
    # res = {}
    # papers_path = current_path + "../data/aminerData/pubs_raw.json"
    # papers = parseJson(papers_path)
    # for pid,paper in papers.items():
    #     title = str(paper.get("title","")) + "."
    #     abstract = str(paper.get("abstract","")) + "."
    #     venue = str(paper.get("venue", ""))
    #     keywords = paper.get("keywords","")
    #
    #     if isinstance(keywords,list):
    #         kw = " ".join(keywords)
    #         kw = kw + "."
    #         corpus = title + " " + abstract + " " + kw + " " + venue
    #     else:
    #         corpus = title + " " + abstract  + " " + venue
    #     res[pid] = corpus
    out_model_path = current_path + "../../bert_pretrain/pre/bert-base-uncased-epoch-5"

    tokenizer = BertTokenizer.from_pretrained(out_model_path)

    res =  parseJson( current_path + "../data/aminerDataProcess/bert_courpus.json")
    bert_tokenizer = {}
    for pid, sent in res.items():
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=256,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        # out = bert(**encoded_dict)
        # Add the encoded sentence to the list.
        bert_tokenizer[pid] = {}
        bert_tokenizer[pid]["input_ids"] = torch.squeeze(encoded_dict['input_ids']).tolist()
        bert_tokenizer[pid]["attention_masks"] =  torch.squeeze(encoded_dict['attention_mask']).tolist()
    saveJson(current_path + "../data/aminerDataProcess/bert_tokenizer_v2.json", bert_tokenizer)


def produce_bert_embeding():
    # res = {}
    # papers_path = current_path + "../data/aminerData/pubs_raw.json"
    # papers = parseJson(papers_path)
    # for pid, paper in papers.items():
    #     title = str(paper.get("title", "")) + "."
    #     abstract = str(paper.get("abstract", "")) + "."
    #     venue = str(paper.get("venue", ""))
    #     keywords = paper.get("keywords", "")
    #
    #     if isinstance(keywords, list):
    #         kw = " ".join(keywords)
    #         kw = kw + "."
    #         corpus = title + " " + abstract + " " + kw + " " + venue
    #     else:
    #         corpus = title + " " + abstract + " " + venue
    #     res[pid] = corpus
    res =  parseJson(current_path + "../data/aminerDataProcess/bert_courpus.json")
    bert_tokenizer = {}
    for pid, sent in res.items():
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=256,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        encoded_dict =encoded_dict.to(device)
        out = bert(**encoded_dict)
        bert_tokenizer[pid] = torch.squeeze(out['pooler_output']).to("cpu").tolist()
        # Add the encoded sentence to the list.
        # bert_tokenizer[pid] = {}
        # bert_tokenizer[pid]["input_ids"] = torch.squeeze(encoded_dict['input_ids']).tolist()
        # bert_tokenizer[pid]["attention_masks"] = torch.squeeze(encoded_dict['attention_mask']).tolist()
    saveJson(current_path + "../data/aminerEmbeding/bert_embeding.json", bert_tokenizer)


def produce_fix_bert_mask_epoch20_embeding():

    out_model_path = current_path + "../bert_pretrain/pre/bert-base-uncased-epoch-20"

    bert1 = BertModel.from_pretrained(out_model_path, local_files_only=True)
    bert1.cuda()
    res = parseJson(current_path + "../data/aminerDataProcess/bert_courpus.json")
    bert_tokenizer = parseJson(current_path + "../data/aminerDataProcess/bert_tokenizer.json")
    bert_embeding = {}

    for paperId, sent in tqdm(res.items()):
        bert_input_ids = bert_tokenizer[paperId].get("input_ids")
        bert_attention_masks = bert_tokenizer[paperId].get("attention_masks")
        bert_input_ids = torch.tensor(bert_input_ids, dtype=torch.int64).cuda()
        bert_attention_masks = torch.tensor(bert_attention_masks, dtype=torch.int64).cuda()
        h_0 = torch.unsqueeze(bert_input_ids, dim=0).cuda()
        h_1 = torch.unsqueeze(bert_attention_masks, dim=0).cuda()
        out = bert1(input_ids=h_0, attention_mask=h_1)
        bert_embeding[paperId] = torch.squeeze(out['pooler_output']).to("cpu").tolist()
        # Add the encoded sentence to the list.
        # bert_tokenizer[pid] = {}
        # bert_tokenizer[pid]["input_ids"] = torch.squeeze(encoded_dict['input_ids']).tolist()
        # bert_tokenizer[pid]["attention_masks"] = torch.squeeze(encoded_dict['attention_mask']).tolist()
    saveJson(current_path + "../data/aminerEmbeding/bert_mask_embeding_epoch20.json", bert_embeding)


def produce_bert_embeding_v2():
    # res = {}
    # papers_path = current_path + "../data/aminerData/pubs_raw.json"
    # papers = parseJson(papers_path)
    # for pid, paper in papers.items():
    #     title = str(paper.get("title", "")) + "."
    #     abstract = str(paper.get("abstract", "")) + "."
    #     venue = str(paper.get("venue", ""))
    #     keywords = paper.get("keywords", "")
    #
    #     if isinstance(keywords, list):
    #         kw = " ".join(keywords)
    #         kw = kw + "."
    #         corpus = title + " " + abstract + " " + kw + " " + venue
    #     else:
    #         corpus = title + " " + abstract + " " + venue
    #     res[pid] = corpus
    # out_model_path = current_path + "../bert_pretrain/pre/bert-base-uncased-epoch-5"
    out_model_path = current_path + "../bert_pretrain/pre/bert-base-uncased"

    bert1 = BertModel.from_pretrained(out_model_path,local_files_only=True)
    bert1.cuda()
    res =  parseJson(current_path + "../data/aminerDataProcess/bert_courpus.json")
    bert_tokenizer = parseJson(current_path + "../data/aminerDataProcess/bert_tokenizer.json")
    bert_embeding = {}

    for paperId, sent in tqdm(res.items()):
        # encoded_dict = tokenizer.encode_plus(
        #     sent,  # Sentence to encode.
        #     add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        #     max_length=256,  # Pad & truncate all sentences.
        #     pad_to_max_length=True,
        #     return_attention_mask=True,  # Construct attn. masks.
        #     return_tensors='pt',  # Return pytorch tensors.
        # )
        # encoded_dict =encoded_dict.to(device)
        bert_input_ids = bert_tokenizer[paperId].get("input_ids")
        bert_attention_masks = bert_tokenizer[paperId].get("attention_masks")
        bert_input_ids = torch.tensor(bert_input_ids, dtype=torch.int64).cuda()
        bert_attention_masks = torch.tensor(bert_attention_masks, dtype=torch.int64).cuda()
        h_0 = torch.unsqueeze(bert_input_ids, dim=0).cuda()
        h_1 = torch.unsqueeze(bert_attention_masks, dim=0)
        out = bert1(input_ids=h_0, attention_mask=h_1)
        bert_embeding[paperId] = torch.squeeze(out['pooler_output']).to("cpu").tolist()
        # Add the encoded sentence to the list.
        # bert_tokenizer[pid] = {}
        # bert_tokenizer[pid]["input_ids"] = torch.squeeze(encoded_dict['input_ids']).tolist()
        # bert_tokenizer[pid]["attention_masks"] = torch.squeeze(encoded_dict['attention_mask']).tolist()
    saveJson(current_path + "../data/aminerEmbeding/bert_embeding_finetune_epoch2.json", bert_embeding)


def produce_bert_embeding_epoch20():
    # res = {}
    # papers_path = current_path + "../data/aminerData/pubs_raw.json"
    # papers = parseJson(papers_path)
    # for pid, paper in papers.items():
    #     title = str(paper.get("title", "")) + "."
    #     abstract = str(paper.get("abstract", "")) + "."
    #     venue = str(paper.get("venue", ""))
    #     keywords = paper.get("keywords", "")
    #
    #     if isinstance(keywords, list):
    #         kw = " ".join(keywords)
    #         kw = kw + "."
    #         corpus = title + " " + abstract + " " + kw + " " + venue
    #     else:
    #         corpus = title + " " + abstract + " " + venue
    #     res[pid] = corpus
    # out_model_path = current_path + "../bert_pretrain/pre/bert-base-uncased-epoch-5"
    out_model_path = current_path + "../bert_pretrain/pre/bert-base-uncased-epoch-20"

    bert1 = BertModel.from_pretrained(out_model_path,local_files_only=True)
    bert1.cuda()
    res =  parseJson(current_path + "../data/aminerDataProcess/bert_courpus.json")
    bert_tokenizer = parseJson(current_path + "../data/aminerDataProcess/bert_tokenizer.json")
    bert_embeding = {}

    for paperId, sent in tqdm(res.items()):
        # encoded_dict = tokenizer.encode_plus(
        #     sent,  # Sentence to encode.
        #     add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        #     max_length=256,  # Pad & truncate all sentences.
        #     pad_to_max_length=True,
        #     return_attention_mask=True,  # Construct attn. masks.
        #     return_tensors='pt',  # Return pytorch tensors.
        # )
        # encoded_dict =encoded_dict.to(device)
        bert_input_ids = bert_tokenizer[paperId].get("input_ids")
        bert_attention_masks = bert_tokenizer[paperId].get("attention_masks")
        bert_input_ids = torch.tensor(bert_input_ids, dtype=torch.int64).cuda()
        bert_attention_masks = torch.tensor(bert_attention_masks, dtype=torch.int64).cuda()
        h_0 = torch.unsqueeze(bert_input_ids, dim=0).cuda()
        h_1 = torch.unsqueeze(bert_attention_masks, dim=0)
        out = bert1(input_ids=h_0, attention_mask=h_1)
        bert_embeding[paperId] = torch.squeeze(out['pooler_output']).to("cpu").tolist()
        # Add the encoded sentence to the list.
        # bert_tokenizer[pid] = {}
        # bert_tokenizer[pid]["input_ids"] = torch.squeeze(encoded_dict['input_ids']).tolist()
        # bert_tokenizer[pid]["attention_masks"] = torch.squeeze(encoded_dict['attention_mask']).tolist()
    saveJson(current_path + "../data/aminerEmbeding/bert_embeding_finetune_epoch20.json", bert_embeding)



def produce_bert_embeding_cls_epoch50(courpus_path,bert_emb_path):
    # res = {}
    # papers_path = current_path + "../data/aminerData/pubs_raw.json"
    # papers = parseJson(papers_path)
    # for pid, paper in papers.items():
    #     title = str(paper.get("title", "")) + "."
    #     abstract = str(paper.get("abstract", "")) + "."
    #     venue = str(paper.get("venue", ""))
    #     keywords = paper.get("keywords", "")
    #
    #     if isinstance(keywords, list):
    #         kw = " ".join(keywords)
    #         kw = kw + "."
    #         corpus = title + " " + abstract + " " + kw + " " + venue
    #     else:
    #         corpus = title + " " + abstract + " " + venue
    #     res[pid] = corpus
    # out_model_path = current_path + "../bert_pretrain/pre/bert-base-uncased-epoch-5"
    out_model_path = current_path + "../../bert_pretrain/pre/bert-base-uncased-epoch-50"

    bert1 = BertModel.from_pretrained(out_model_path,output_hidden_states=True,output_attentions=True,local_files_only=True)
    bert1.to(device)
    res =  parseJson(courpus_path)
    # bert_tokenizer = parseJson(current_path + "../../data/aminerDataProcess/bert_tokenizer.json")
    bert_embeding = {}

    for pid, sent in tqdm(res.items()):
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=256,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        encoded_dict = encoded_dict.to(device)
        # encoded_dict = encoded_dict
        # out = bert(**encoded_dict)
        # Add the encoded sentence to the list.
        # bert_tokenizer[pid] = {}
        # bert_tokenizer[pid]["input_ids"] = torch.squeeze(encoded_dict['input_ids']).tolist()
        # bert_tokenizer[pid]["attention_masks"] =  torch.squeeze(encoded_dict['attention_mask']).tolist()



    # for paperId, sent in tqdm(res.items()):
        # encoded_dict = tokenizer.encode_plus(
        #     sent,  # Sentence to encode.
        #     add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        #     max_length=256,  # Pad & truncate all sentences.
        #     pad_to_max_length=True,
        #     return_attention_mask=True,  # Construct attn. masks.
        #     return_tensors='pt',  # Return pytorch tensors.
        # )
        # encoded_dict =encoded_dict.to(device)
        # bert_input_ids = bert_tokenizer[paperId].get("input_ids")
        # bert_attention_masks = bert_tokenizer[paperId].get("attention_masks")
        # bert_input_ids = torch.tensor(bert_input_ids, dtype=torch.int64).cuda()
        # bert_attention_masks = torch.tensor(bert_attention_masks, dtype=torch.int64).cuda()
        # h_0 = torch.unsqueeze(bert_input_ids, dim=0).cuda()
        # h_1 = torch.unsqueeze(bert_attention_masks, dim=0)
        # out = bert1(input_ids=h_0, attention_mask=h_1)
        # last_hidden_state, pooler_output, all_hidden_states, all_attentions = bert1(**encoded_dict)
        outputs  = bert1(**encoded_dict)
        bert_embeding[pid] = torch.squeeze(outputs['pooler_output']).to("cpu").tolist()
        # bert_embeding[pid] = torch.squeeze(outputs.last_hidden_state.mean(dim=1)).to("cpu").tolist()
        # Add the encoded sentence to the list.
        # bert_tokenizer[pid] = {}
        # bert_tokenizer[pid]["input_ids"] = torch.squeeze(encoded_dict['input_ids']).tolist()
        # bert_tokenizer[pid]["attention_masks"] = torch.squeeze(encoded_dict['attention_mask']).tolist()
    saveJson(bert_emb_path, bert_embeding)


def produce_bert_embeding_hiddenstate_epoch50(courpus_path,bert_emb_path):
    # res = {}
    # papers_path = current_path + "../data/aminerData/pubs_raw.json"
    # papers = parseJson(papers_path)
    # for pid, paper in papers.items():
    #     title = str(paper.get("title", "")) + "."
    #     abstract = str(paper.get("abstract", "")) + "."
    #     venue = str(paper.get("venue", ""))
    #     keywords = paper.get("keywords", "")
    #
    #     if isinstance(keywords, list):
    #         kw = " ".join(keywords)
    #         kw = kw + "."
    #         corpus = title + " " + abstract + " " + kw + " " + venue
    #     else:
    #         corpus = title + " " + abstract + " " + venue
    #     res[pid] = corpus
    # out_model_path = current_path + "../bert_pretrain/pre/bert-base-uncased-epoch-5"
    out_model_path = current_path + "../../bert_pretrain/pre/bert-base-uncased-epoch-50"

    bert1 = BertModel.from_pretrained(out_model_path,output_hidden_states=True,output_attentions=True,local_files_only=True)
    bert1.to(device)
    res =  parseJson(courpus_path)
    # bert_tokenizer = parseJson(current_path + "../../data/aminerDataProcess/bert_tokenizer.json")
    bert_embeding = {}

    for pid, sent in tqdm(res.items()):
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=256,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        encoded_dict = encoded_dict.to(device)
        # encoded_dict = encoded_dict
        # out = bert(**encoded_dict)
        # Add the encoded sentence to the list.
        # bert_tokenizer[pid] = {}
        # bert_tokenizer[pid]["input_ids"] = torch.squeeze(encoded_dict['input_ids']).tolist()
        # bert_tokenizer[pid]["attention_masks"] =  torch.squeeze(encoded_dict['attention_mask']).tolist()



    # for paperId, sent in tqdm(res.items()):
        # encoded_dict = tokenizer.encode_plus(
        #     sent,  # Sentence to encode.
        #     add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        #     max_length=256,  # Pad & truncate all sentences.
        #     pad_to_max_length=True,
        #     return_attention_mask=True,  # Construct attn. masks.
        #     return_tensors='pt',  # Return pytorch tensors.
        # )
        # encoded_dict =encoded_dict.to(device)
        # bert_input_ids = bert_tokenizer[paperId].get("input_ids")
        # bert_attention_masks = bert_tokenizer[paperId].get("attention_masks")
        # bert_input_ids = torch.tensor(bert_input_ids, dtype=torch.int64).cuda()
        # bert_attention_masks = torch.tensor(bert_attention_masks, dtype=torch.int64).cuda()
        # h_0 = torch.unsqueeze(bert_input_ids, dim=0).cuda()
        # h_1 = torch.unsqueeze(bert_attention_masks, dim=0)
        # out = bert1(input_ids=h_0, attention_mask=h_1)
        # last_hidden_state, pooler_output, all_hidden_states, all_attentions = bert1(**encoded_dict)
        outputs  = bert1(**encoded_dict)
        # bert_embeding[pid] = torch.squeeze(out['pooler_output']).to("cpu").tolist()
        bert_embeding[pid] = torch.squeeze(outputs.last_hidden_state.mean(dim=1)).to("cpu").tolist()
        # Add the encoded sentence to the list.
        # bert_tokenizer[pid] = {}
        # bert_tokenizer[pid]["input_ids"] = torch.squeeze(encoded_dict['input_ids']).tolist()
        # bert_tokenizer[pid]["attention_masks"] = torch.squeeze(encoded_dict['attention_mask']).tolist()
    saveJson(bert_emb_path, bert_embeding)




def produce_bert_embeding_epoch50():
    # res = {}
    # papers_path = current_path + "../data/aminerData/pubs_raw.json"
    # papers = parseJson(papers_path)
    # for pid, paper in papers.items():
    #     title = str(paper.get("title", "")) + "."
    #     abstract = str(paper.get("abstract", "")) + "."
    #     venue = str(paper.get("venue", ""))
    #     keywords = paper.get("keywords", "")
    #
    #     if isinstance(keywords, list):
    #         kw = " ".join(keywords)
    #         kw = kw + "."
    #         corpus = title + " " + abstract + " " + kw + " " + venue
    #     else:
    #         corpus = title + " " + abstract + " " + venue
    #     res[pid] = corpus

    # out_model_path = current_path + "../bert_pretrain/pre/bert-base-uncased-epoch-5"
    out_model_path = current_path + "../../bert_pretrain/pre/bert-base-uncased-epoch-50"

    bert1 = BertModel.from_pretrained(out_model_path,local_files_only=True)
    bert1.cuda()
    res =  parseJson(courpus_path)
    bert_tokenizer = parseJson(current_path + "../../data/aminerDataProcess/bert_tokenizer.json")
    bert_embeding = {}

    for paperId, sent in tqdm(res.items()):
        # encoded_dict = tokenizer.encode_plus(
        #     sent,  # Sentence to encode.
        #     add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        #     max_length=256,  # Pad & truncate all sentences.
        #     pad_to_max_length=True,
        #     return_attention_mask=True,  # Construct attn. masks.
        #     return_tensors='pt',  # Return pytorch tensors.
        # )
        # encoded_dict =encoded_dict.to(device)
        bert_input_ids = bert_tokenizer[paperId].get("input_ids")
        bert_attention_masks = bert_tokenizer[paperId].get("attention_masks")
        bert_input_ids = torch.tensor(bert_input_ids, dtype=torch.int64).cuda()
        bert_attention_masks = torch.tensor(bert_attention_masks, dtype=torch.int64).cuda()
        h_0 = torch.unsqueeze(bert_input_ids, dim=0).cuda()
        h_1 = torch.unsqueeze(bert_attention_masks, dim=0)
        out = bert1(input_ids=h_0, attention_mask=h_1)
        bert_embeding[paperId] = torch.squeeze(out['pooler_output']).to("cpu").tolist()
        # Add the encoded sentence to the list.
        # bert_tokenizer[pid] = {}
        # bert_tokenizer[pid]["input_ids"] = torch.squeeze(encoded_dict['input_ids']).tolist()
        # bert_tokenizer[pid]["attention_masks"] = torch.squeeze(encoded_dict['attention_mask']).tolist()
    saveJson(current_path + "../../data/aminerEmbeding/bert_mask_50_embeding_new.json", bert_embeding)


if __name__ == '__main__':
    # produce_bert_corpus()
    # produce_bert_embeding()
    # produce_bert_corpus_v2()
    # produce_bert_embeding_epoch20()
    # produce_fix_bert_mask_epoch20_embeding()
    # produce_bert_embeding_epoch50()
    print("TKVB corpus...")
    produce_bert_corpus_TKVB()
    print("TKV corpus...")
    produce_bert_corpus_TKV()

    # cls tkvb
    print("start cls ktvb................")
    courpus_path = current_path + "../../data/aminerDataProcess/bert_corpus_TKVB.json"
    bert_emb_path = current_path + "../../data/aminerEmbeding/bert_mask_50_embeding_new_cls_TKVB.json"
    produce_bert_embeding_cls_epoch50(courpus_path, bert_emb_path)

    print("start cls ktv................")
    #cls tkv
    courpus_path = current_path + "../../data/aminerDataProcess/bert_corpus_TKV.json"
    bert_emb_path = current_path + "../../data/aminerEmbeding/bert_mask_50_embeding_new_cls_TKV.json"
    produce_bert_embeding_cls_epoch50(courpus_path,bert_emb_path)

    print("start lhs ktvb................")
    # last hidden state  tkvb
    courpus_path = current_path + "../../data/aminerDataProcess/bert_corpus_TKVB.json"
    bert_emb_path = current_path + "../../data/aminerEmbeding/bert_mask_50_embeding_new_hiddenState_TKVB.json"
    produce_bert_embeding_hiddenstate_epoch50(courpus_path,bert_emb_path)

    print("start lhs ktv................")
    # last hidden state tkv
    courpus_path = current_path + "../../data/aminerDataProcess/bert_corpus_TKV.json"
    bert_emb_path = current_path + "../../data/aminerEmbeding/bert_mask_50_embeding_new_hiddenState_TKV.json"
    produce_bert_embeding_hiddenstate_epoch50(courpus_path,bert_emb_path)





