from chess import Board, Piece
from data import get_chess
import cv2
from chess_engine_interface import get_move

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


def split_squares(img, GRID_R, GRID_C, subimg_size=(24,24)):
    # Define the window size
    windowsize_r = img.shape[0] / GRID_R
    windowsize_c = img.shape[1] / GRID_C

    for r in range(0, GRID_R):
        for c in range(0, GRID_C):
            subimg = img[
                ceil(r * windowsize_r) : floor((r + 1) * windowsize_r),
                ceil(c * windowsize_c) : floor((c + 1) * windowsize_c),
            ]

            subimg = cv2.resize(subimg, subimg_size)

            yield (r, c), subimg


def split_online_chess(fen, img):
    GRID_R = 8
    GRID_C = 8
    board = Board(fen)

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
        return "PNBRQKpnbrqk"[n-1]


import torch
import torch.nn as nn
import torch.nn.functional as F

def batch_patches_to_torch(batch_imgs):
    batch_imgs = np.moveaxis(batch_imgs, 3, 1)
    batch_imgs = batch_imgs / 255
    return torch.FloatTensor(batch_imgs)

class data_randomized_loader:
    def __init__(
        self, lookahead=40, batch_size=32, subsample_empty=6
    ):
        self.gen = get_chess()
        self.lookahead = lookahead
        self.batch_size = batch_size
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
                        imgs.append(subimg)
                else:
                    items.append(item)
                    imgs.append(subimg)
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
        batch_imgs = batch_patches_to_torch(batch_imgs)
        return (
            batch_imgs,
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

from pathlib import Path
path = Path(__file__)
save_loc = path.parent.absolute() / 'models' / 'chess_online_square.txt'

train = True
new = False
if train:
    OCR_model = OCR_online_chess()
    if not new:
        OCR_model.load_state_dict(torch.load(save_loc))
    OCR_model.train()
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(OCR_model.parameters(), lr=0.001)

    data_loader = data_randomized_loader(batch_size=128, lookahead=100, subsample_empty=10)

    data_loader.set_batches(500)
    updates = 100
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = OCR_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % updates == updates - 1:
            OCR_model.eval()
            with torch.no_grad():
                inputs, labels = next(data_loader)
                correct = (torch.argmax(OCR_model(inputs), dim=1) == labels).numpy().mean()
            OCR_model.train()
            print(f"[{i + 1:5d}] val: {correct:1.4f} loss: {running_loss / updates:.8f}")
            running_loss = 0.0

    torch.save(OCR_model.state_dict(), str(save_loc))
    print("Finished Training")
else:
    OCR_model = OCR_online_chess()
    OCR_model.load_state_dict(torch.load(save_loc))
    OCR_model.eval()

def estimate_online_chess(img):
    GRID_R = 8
    GRID_C = 8
    board = Board()

    batch_imgs = []
    positions = []
    for pos, subimg in split_squares(img, GRID_R=GRID_R, GRID_C=GRID_C):
        batch_imgs.append(subimg)
        positions.append(pos)
        # r, c = pos
        # square = (GRID_R - 1 - r) * GRID_C + c
    batch_imgs = np.array(batch_imgs)
    batch_imgs = batch_patches_to_torch(batch_imgs)
    # TODO perhaps calc. confidence and throw warning if not confident!
    OCR_model.eval()
    estimate = OCR_model(batch_imgs)
    estimate = torch.argmax(estimate, dim=1)
    for piece, pos in zip(estimate, positions):
        # Squares go from bottom left to right to top, is range(64)
        r, c = pos
        square = (GRID_R - 1 - r) * GRID_C + c
        piece = int_to_symbol(piece)
        if piece is not None:
            piece = Piece.from_symbol(piece)
        board.set_piece_at(square, piece)
    return board

for fen, img in get_chess(n=10):
    board = estimate_online_chess(img)
    print(board.fen())
    print(fen)
    assert board.fen().split(" ")[0] == fen