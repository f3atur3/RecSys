import math
from typing import Optional

import numpy as np  # noqa: F401
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class RecSysModel(nn.Module):
    def __init__(self, vocab_size: int, device: torch.device=torch.device("cpu")):
        super(RecSysModel, self).__init__()
        self.embedding_dim = 100
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.act_func = nn.ReLU()
        self.fc1 = nn.Linear(vocab_size * 2, vocab_size)
        self.fc2 = nn.Linear(vocab_size, vocab_size // 2)
        self.fc3 = nn.Linear(vocab_size // 2, 2)
        self.sm = nn.Softmax(dim=-1)
        self.device = device

    def Attn(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        E = Q.size(-1)
        W = torch.bmm(Q, K.transpose(-2, -1)) / math.sqrt(E)
        W = nn.functional.softmax(W, dim = -1)
        return (torch.bmm(W, V), W)

    def forward(self, matrixs: torch.Tensor, vectors: torch.Tensor) -> tuple[torch.Tensor, Optional[torch.Tensor]]:        
        attn_output, attn_output_weights = self.Attn(vectors.unsqueeze(1), matrixs, matrixs)

        attn_output.squeeze_(1)
        attn_output_weights.squeeze_(1)

        concat_output = torch.cat((attn_output, vectors), dim=1)

        output = self.fc1(concat_output)
        output = self.act_func(output)
        output = self.fc2(output)
        output = self.act_func(output)
        output = self.fc3(output)

        return (output, attn_output_weights)
    
    def inference(self, matrixs: torch.Tensor, vectors: torch.Tensor):
        x, w = self.forward(matrixs, vectors)
        x = self.sm(x)
        return x, w

    def fit(self, data: pd.DataFrame, movies: pd.DataFrame, num_epochs=10) -> None:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)
        for _ in tqdm(range(num_epochs)):
            for index, row in data.iterrows():
                split_vectors = [(row["combined_features"][i:9+i+1], row["combined_features"][9+i+1]) for i in range(10)]
                split_df = pd.DataFrame(split_vectors, columns=["history", "next_movie"])
                split_df["label"] = 1
                negative_df = split_df.copy(deep=True)
                negative_df["next_movie"] = movies["combined_features"].sample(n=split_df.shape[0], random_state=index).reset_index(drop=True)
                negative_df["label"] = 0
                split_df = pd.concat([split_df, negative_df], axis=0).reset_index(drop=True)
                optimizer.zero_grad()
                outputs, _ = self.forward(torch.Tensor(split_df["history"]).to(self.device).float(), torch.Tensor(split_df["next_movie"]).to(self.device).float())
                loss = criterion(outputs.to(self.device), torch.Tensor([split_df["label"]]).to(self.device).long().squeeze(0))
                loss.backward()
                optimizer.step()
                if index % 1000 == 0:
                    print(f"Прошел {index} ряд")
        print("Обучение завершено")

    def predict(self, history: list, movies: pd.DataFrame, num_movies=10) -> tuple[list, list, list]:
        history: torch.Tensor = torch.Tensor(history).to(self.device).float().unsqueeze(0)
        with torch.no_grad():
            outputs = list()
            weights = list()
            for _, row in movies.iterrows():
                t_movie = torch.Tensor(row["combined_features"]).to(self.device).float().unsqueeze(0)
                o, w = self.inference(history, t_movie)
                outputs.extend(o.tolist())
                weights.extend(w.tolist())
            scores = [i + j + [k] for i, j, k in zip(outputs, weights, movies["id"].tolist())]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[:num_movies]
        return [i[:2] for i in scores], [j[2:-1] for j in scores], [k[-1] for k in scores]