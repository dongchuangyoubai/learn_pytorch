import torch
import random
import zipfile

with zipfile.ZipFile('jaychou_lyrics.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
print(corpus_chars[:40])
print(1)