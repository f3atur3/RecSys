from transformers import AutoModel, AutoTokenizer
import torch

import numpy as np
import pandas as pd
from tqdm.notebook import tqdm

class TextEmbeddings:
    """
    Пример использования:
        text_embeddings = TextEmbeddings(True, True)
        data = text_embeddings.add_many_embeddings(data, 'text', models)
    """
    def __init__(self, add_cls_embeddings=True, add_mean_embeddings=False, device: torch.device=torch.device("cpu")):
        self.add_mean_embeddings = add_mean_embeddings
        self.add_cls_embeddings = add_cls_embeddings
        self.device = device
        if add_cls_embeddings is False and add_mean_embeddings is False:
            raise 'Error: you should select at least one type of embeddings to be computed'

    def mean_pooling(self, hidden_state, attention_mask):
        """
        Возвращает усредненный с учетом attention_mask hidden_state.
        """
        token_embeddings = hidden_state.detach().cpu() 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        return sum_embeddings / attention_mask.sum()

    def extract_embeddings(self, texts, model_name, max_len):
        """
        Возвращает значения, посчитанные данной моделью - эмбеддинги для всех текстов из texts.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(self.device)
        text_features = []
        for sentence in tqdm(texts):
            encoded_input = tokenizer([sentence],
                                      padding='max_length',
                                      truncation=True,
                                      max_length=max_len,
                                      return_tensors='pt')
            with torch.no_grad():
                hidden_state, cls_head = model(input_ids=encoded_input['input_ids'].to(self.device), return_dict=False)
                sentence_embeddings = self.mean_pooling(hidden_state, encoded_input['attention_mask'])
            
            now_emb = []
            if self.add_cls_embeddings:
                now_emb.append(cls_head.detach().cpu().numpy().flatten())
            
            if self.add_mean_embeddings:
                now_emb.append(sentence_embeddings.detach().cpu().numpy().flatten())
            
            text_features.append(np.concatenate(now_emb, axis=0))
        return text_features

    def add_many_embeddings(self, df: pd.DataFrame, text_col: str, models: list[str]) -> pd.DataFrame:
        """"
        Добавляет в качестве признаков эмбеддинги для колонки text_col.
        В качестве моделей и максимальных длин используются models.
        """
        text_features = []
        for model_name, max_len in models:
            print(model_name)
            model_text_features = self.extract_embeddings(df[text_col], model_name, max_len)
            if len(text_features) == 0:
                text_features = model_text_features
            else:
                for i in range(len(model_text_features)):
                    text_features[i] = np.concatenate((text_features[i], model_text_features[i]))
        
        df[f'{text_col}_features'] = text_features
        return df

# Полный список поддерживаемых моделей можно найти на https://huggingface.co/models
models = [
        #   ('cointegrated/LaBSE-en-ru', 512),
        #   ('sberbank-ai/ruRoberta-large', 512),
        #   ('sberbank-ai/sbert_large_nlu_ru', 512),
        #   ('sberbank-ai/sbert_large_mt_nlu_ru', 512),
        #   ('sberbank-ai/ruBert-large', 512),
          ('sberbank-ai/ruBert-base', 512),
        #   ('cointegrated/rubert-tiny2', 2048),
        #   ('DeepPavlov/rubert-base-cased-conversational', 512),
        #   ('microsoft/mdeberta-v3-base', 512),
        #   ('vicgalle/xlm-roberta-large-xnli-anli', 512),
        #   ('MoritzLaurer/mDeBERTa-v3-base-mnli-xnli', 512),
        #   ('facebook/bart-large-mnli', 1024)
]