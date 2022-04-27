from chess import Board
from data import get_chess
import cv2


# from imutils.perspective import four_point_transform
from math import floor, ceil


# def find_chessboard(frame):
#     chessboard_flags = (
#         cv2.CALIB_CB_ADAPTIVE_THRESH
#         + cv2.CALIB_CB_FAST_CHECK
#         + cv2.CALIB_CB_NORMALIZE_IMAGE
#     )
#     small_frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
#     return (
#         cv2.findChessboardCorners(small_frame, (9, 6), chessboard_flags)[0]
#         and cv2.findChessboardCorners(frame, (9, 6), chessboard_flags)[0]
#     )


def split_squares(img, GRID_R, GRID_C):
    # Define the window size
    windowsize_r = img.shape[0] / GRID_R
    windowsize_c = img.shape[1] / GRID_C

    for r in range(0, GRID_R):
        for c in range(0, GRID_C):
            subimg = img[
                ceil(r * windowsize_r) : floor((r + 1) * windowsize_r),
                ceil(c * windowsize_c) : floor((c + 1) * windowsize_c),
            ]

            yield (r, c), subimg


def split_online_chess(fen, img):
    GRID_R = 8
    GRID_C = 8
    board = Board(fen)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for pos, subimg in split_squares(img, GRID_R=GRID_R, GRID_C=GRID_C):
        r, c = pos
        square = (GRID_R - 1 - r) * GRID_C + c
        item = board.piece_at(square)
        yield item, subimg

import numpy as np

def symbol_to_int(symbol):
    if symbol is None:
        return 0
    else:
        return 1 + "PNBRQKpnbrqk".index(symbol.symbol())


def int_to_symbol(n):
    if n == 0:
        return None
    else:
        return 1 + "PNBRQKpnbrqk"[n]


import torch
import torch.nn as nn
import torch.nn.functional as F


class data_randomized_loader:
    def __init__(
        self, lookahead=40, batch_size=32, img_size=(24, 24), subsample_empty=0
    ):
        self.gen = get_chess()
        self.lookahead = lookahead
        self.batch_size = batch_size
        self.img_size = img_size
        self.subsample_empty = subsample_empty
        self.batches = -1
        self.items = None
        self.imgs = None
        self.index = None
        self.batch_start = None
        self.empty_count = 0

    def set_batches(self, batches):
        self.batches = batches

    def __preload__(self):
        batch = []
        for board in self.gen:
            batch.append(board)
            if len(batch) == self.lookahead:
                break

        items = []
        imgs = []
        for fen, img in batch:
            for item, subimg in split_online_chess(fen, img):
                item = symbol_to_int(item)
                if item == 0:
                    if self.empty_count != self.subsample_empty:
                        self.empty_count += 1
                    else:
                        self.empty_count = 0
                        items.append(item)
                        imgs.append(cv2.resize(subimg, self.img_size))
        self.items = np.array(items)
        self.imgs = np.array(imgs)

        assert (
            len(self.items) == len(self.imgs) and len(self.items) <= self.lookahead * 64
        )

        self.index = np.arange(len(self.items))
        np.random.shuffle(self.index)

        self.batch_start = 0
        return

    def __iter__(self):
        return self

    def __next__(self):
        if self.batches == 0:
            raise StopIteration()
        self.batches -= 1

        if self.batch_start is None or self.batch_start >= len(self.index):
            self.__preload__()

        batch_index = self.index[
            self.batch_start : max(self.batch_start + self.batch_size, len(self.index))
        ]
        batch_imgs = self.imgs[batch_index]
        batch_imgs = np.moveaxis(batch_imgs, 3, 1)
        batch_imgs = batch_imgs / 255
        return (
            torch.FloatTensor(batch_imgs),
            torch.LongTensor(self.items[batch_index]),
        )


class OCR_online_chess(nn.Module):
    def __init__(self):
        super().__init__()
        # 24x24x3
        self.conv1 = nn.Conv2d(3, 5, 5)
        # 20x20x5
        self.pool = nn.MaxPool2d(2, 2)
        # 10x10x5
        self.conv2 = nn.Conv2d(5, 10, 5)
        # 6x6x10
        # 3x3x10
        self.fc1 = nn.Linear(3 * 3 * 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1 + 6 + 6)  # maybe classify board color as well?

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = OCR_online_chess()
net.train()
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

data_loader = data_randomized_loader(batch_size=128, lookahead=100, subsample_empty=10)

data_loader.set_batches(1000)
updates = 300
running_loss = 0.0
for i, data in enumerate(data_loader, 0):
    inputs, labels = data
    optimizer.zero_grad()
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    running_loss += loss.item()

    if i % updates == updates - 1:
        net.eval()
        with torch.no_grad():
            inputs, labels = next(data_loader)
            correct = (torch.argmax(net(inputs), dim=1) == labels).numpy().mean()
        net.train()
        print(f"[{i + 1:5d}] val: {correct:1.4f} loss: {running_loss / updates:.8f}")
        running_loss = 0.0

from pathlib import Path
path = Path(__file__)
save_loc = path.parent.absolute() / 'models' / 'chess_online_square.txt'

torch.save(net.state_dict(), str(save_loc))
print("Finished Training")