import cv2
import numpy as np
from pathlib import Path
path = Path(__file__)
local_location = path.parent.parent.absolute() / 'data'
store_location = Path('D:\Data\games')

print(f"Data location: {local_location}")
print(f"Store location: {store_location}")

def get_sudoku(n=None):
    dir = local_location / 'sudoku' / 'wichtounet'
    files = [x.stem for x in dir.glob('**/*') if x.is_file()]
    files = list(set(files))
    if n is None:
        n = len(files)
    for i in range(n):
        file = files[i]
        img = cv2.imread(str(dir / (file + '.jpg')))
        data = np.loadtxt(str(dir / (file + '.dat')), skiprows=2, dtype=int)
        assert data.shape == (9,9)
        yield img, data

def get_chess(n=None):
    dir = store_location / 'chess' / 'koryakinp' / 'archive' / 'dataset'
    gen = dir.glob('**/*.jpeg')
    if n is None:
        n = -1
    for path in gen:
        n -= 1
        fen = path.stem.replace('-', '/')
        img = cv2.imread(str(path))
        yield fen, img
        if n == 0:
            break
