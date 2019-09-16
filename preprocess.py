#! /usr/bin/python

from __future__ import unicode_literals, print_function, division
import glob
import os
import unicodedata
import string
import torch

def findpath(path):
    return glob.glob(path)   #Returns a list of file paths that match with the criterion argument.

all_letters = string.ascii_letters + ",.;'"
n_letters = len(all_letters)


# https://stackoverflow.com/a/518232/2809427
# Turning unicode to ascii characters (to take care of accents)
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Build the category_lines dictionary, a list of names per language
category_lines = {}
all_categories = []


# Read a file and split into lines
def readlines(filename):
  return os.path.splitext(os.path.basename(filename))[0]


# Convert names in contained in a file to an ascii list.
def readlines(filename):
  lines = open(filename, encoding='utf-8').read().strip().split('\n')
  return [unicodeToAscii(name) for name in lines]


# Extracting all categories
for file in findpath('/Users/karanpraharaj/nameclassify/data/names/*.txt'):
  filename = os.path.basename(file)
  categ = os.path.splitext(filename)[0]
  all_categories.append(categ)
  categ_names = readlines(file)
  category_lines[categ] = categ_names

n_categories = len(all_categories)

# Converting Names to Tensors

def lettertoindex(c):
  return all_letters.find(c)

# Letter to tensor (1 x n_letters)

def lettertotensor(c):
  a = torch.zeros(1,n_letters)
  a[0][lettertoindex(c)] = 1
  return a


def linetotensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for i, c in enumerate(line):
        tensor[i][0][lettertoindex(c)] = 1
    return tensor


