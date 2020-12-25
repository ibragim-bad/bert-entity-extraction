import string

punc_set = {k:v for v,k in enumerate(string.punctuation)}

pnc = ',.!?'
punc_set = {k:v for v,k in zip(list(range(1,len(pnc))), pnc)}
extra_punc = list(set(string.punctuation) - set(pnc))
sps = [' ' + p for p in pnc]

def clean_extra(s):
    s = s.replace('...', '.')
    s = s.replace(':', ',')
    s = s.replace(';', ',')
    for i in extra_punc:
        s = s.replace(i, '')
    l = s.split()
    for i,w in enumerate(l):
      if l[i] in pnc:
        p = l.pop(i)
        l[i-1] = l[i-1] + p
    return " ".join(l)

def get_targets(s):
    ids = []
    for w in s.split():
        p = w[-1] if w[-1] in pnc else '_'
        ids.append(p)
    return ids

def clean_targets(s):
    s = s.lower()
    for i in pnc:
        s = s.replace(i, '')
    return " ".join(s.split())

import pandas as pd

# Load the dataset into a pandas dataframe.
df = pd.read_csv("data/short_jokes75k.csv")
#df = pd.read_csv("apho_pack.csv")
df.head()

df = df.dropna()
df['punc'] = df.text.apply(clean_extra)
df['targets'] = df.punc.apply(get_targets)
df['words'] = df.punc.apply(clean_targets).str.split()
df = df[(df.words.str.len() == df.targets.str.len())]

def get_explode(adf):
  tdf = adf.copy()
  tdf = tdf.explode('words')
  tdf['labels'] = adf.labels.explode().values
  return tdf

labels = list('_' + pnc)
df[~(df.words.str.len() == df.targets.str.len())]

from sklearn.model_selection import train_test_split

aldf = df[['words', 'targets']]
aldf = pd.DataFrame(aldf.reset_index().values,columns=["sentence_id", "words", "labels"])
train, test = train_test_split(aldf, test_size=0.05)
train = get_explode(train)
test = get_explode(test)

train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)