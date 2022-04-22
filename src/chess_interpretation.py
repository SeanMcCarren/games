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

    for r in range(0,GRID_R):
        for c in range(0,GRID_C):
            subimg = img[
                ceil(r * windowsize_r): floor((r+1) * windowsize_r),
                ceil(c * windowsize_c): floor((c+1) * windowsize_c)
            ]

            yield (r, c), subimg


def split_online_chess(fen, img):
    GRID_R = 8
    GRID_C = 8
    board = Board(fen)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    for pos, subimg in split_squares(img, GRID_R=GRID_R, GRID_C=GRID_C):
        r, c = pos
        square = (GRID_R-1-r) * GRID_C + c
        item = board.piece_at(square)
        yield item, subimg

def get_grid_cells(n=None):
    for fen, img in get_chess(n=n):
        for item, subimg in split_online_chess(fen, img):
            resized = cv2.resize(subimg, (24,24))
            cv2.imshow("original", subimg)
            cv2.imshow("resized", resized)
            cv2.waitKey(0)

get_grid_cells()

# import torch.nn as nn
# import torch.nn.functional as F

# def model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 5, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x


