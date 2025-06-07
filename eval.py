import os
import sys
import pandas as pd
from nn import eval

if len(sys.argv) != 2:
    print('Usage: python eval.py <uuid>')
    sys.exit(1)

uuid = sys.argv[1]

df = pd.read_csv('dataset.csv')

eval(df, uuid)