#!/usr/bin/env python3

import torch
import pandas as pd
import os
import chess
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

DATASET_DIR = os.path.join("..", "dataset")
MAIN_DATASET = os.path.join(DATASET_DIR, "chessData.csv")
df = pd.read_csv(MAIN_DATASET)

df["Evaluation"] = pd.to_numeric(df["Evaluation"], downcast="integer", errors="coerce")

MODEL_PATH = os.path.join("..", "catfish_eval.pt")

class ChessDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fen = self.df.iloc[idx]["FEN"]
        eval_score = self.df.iloc[idx]["Evaluation"]
        board = fen_to_tensor(fen)
        target = torch.tensor([eval_score], dtype=torch.float32)
        active_player_str = fen.split(" ")[1]
        active_player = 1 if active_player_str == 'w' else 0
        halfmove = fen.split(" ")[5]
        data = {
            "board_tensor": board,
            "evaluation": target,
            "active_player": torch.tensor(active_player, dtype=torch.long),
            "halfmove": torch.tensor(int(halfmove), dtype=torch.float32),    
        }
        return data
    
dataset = ChessDataset(df)
loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)

def get_en_passant(fen):
    matrix = torch.zeros((8,8), dtype=torch.float32)
    fen_en_passant = fen.split(" ")[3]
    if fen_en_passant != "-":
        square = chess.parse_square(fen_en_passant)
        matrix[square % 8][square % 8] = 1
    return matrix

def get_piece_matrix(board, piece_type, color):
    matrix = torch.zeros((8,8), dtype=torch.float32)
    for sq in board.pieces(piece_type, color):
        r = 7 - chess.square_rank(sq)
        c = chess.square_file(sq)
        matrix[r, c] = 1
    return matrix

def fen_to_tensor(fen):
    board = chess.Board(fen)
    piece_index = {
        chess.PAWN : 1,
        chess.ROOK : 2,
        chess.KNIGHT : 3,
        chess.BISHOP : 4,
        chess.QUEEN : 5,
        chess.KING : 6,
    }
    fen_tensor = torch.zeros((13,8,8), dtype=torch.float32)
    fen_tensor[0] = get_en_passant(fen)
    for piece, val in piece_index.items():
        fen_tensor[val] = get_piece_matrix(board, piece, chess.BLACK)
        fen_tensor[val + 6] = get_piece_matrix(board, piece, chess.WHITE)
    return fen_tensor

class ConditionalBatchNorm2d(nn.Module):
    def __init__(self, num_features, num_conditions):
        super(ConditionalBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.bn = nn.BatchNorm2d(num_features, affine=False)
        self.gamma = nn.Embedding(num_conditions, num_features)
        self.beta = nn.Embedding(num_conditions, num_features)
        nn.init.ones_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)

    def forward(self, x, condition):
        out = self.bn(x)
        gamma = self.gamma(condition).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(condition).unsqueeze(-1).unsqueeze(-1)
        return gamma * out + beta

class Catfish(nn.Module):
    def __init__(self, num_piece_channels=13, num_classes=1, num_conditions=2):
        super(Catfish, self).__init__()
        self.conv1 = nn.Conv2d(num_piece_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.cbn1 = ConditionalBatchNorm2d(64, num_conditions)
        self.cbn2 = ConditionalBatchNorm2d(128, num_conditions)
        self.cbn3 = ConditionalBatchNorm2d(256, num_conditions)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024 + 1, num_classes)

    def forward(self, board_tensor, active_player, halfmove_clock):
        x = self.conv1(board_tensor)
        x = self.cbn1(x, active_player)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.cbn2(x, active_player)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.cbn3(x, active_player)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        halfmove_clock = halfmove_clock.float()
        x = torch.cat([x, halfmove_clock.unsqueeze(1)], dim=1)
        output = self.fc2(x)
        return output


model = Catfish()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    model.train()
    for i, data in enumerate(loader):
        predictions = model(data["board_tensor"], data["active_player"], data["halfmove"])
        loss = criterion(predictions, data["evaluation"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch:", epoch, "loss:", loss.item())
    
torch.save(model.state_dict(), MODEL_PATH)

train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# def test_model():
#     model = Catfish()
#     model.eval()
#     predictions = []
#     targets = []
#     with torch.no_grad():
#         for i in range(len(test_df)):
#             board = fen_to_tensor(test_df.iloc[i]["FEN"]).unsqueeze(0)
#             target = test_df.iloc[i]["Evaluation"]
#             pred = model(board).item()
#             predictions.append(pred)
#             targets.append(target)
#     return predictions, targets

def evaluate_board(fen):
    model = Catfish()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    board = fen_to_tensor(fen)
    board = board.unsqueeze(0)
    with torch.no_grad():
        evaluation = model(board)
    return evaluation.item()

# print(test_model())
evaluate_board("r1bqkb1r/pp1n1ppp/2n1p3/2ppP3/3P1P2/2P5/PP1N2PP/R1BQKBNR w KQkq - 1 7") #+86
