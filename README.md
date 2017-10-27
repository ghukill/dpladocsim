# dpladocsim
Attemp fuzzy match of unseen/unprocessed record against DPLA datasets

## download DPLA datasets

For examples to run, download [Michigan data](https://dpla-provider-export.s3.amazonaws.com/2017/10/michigan.json.gz) from DPLA and unzip to `/data` directory.

## Character Vector approach

One approach has been to count each character in a document's metadata, `a-z0-9`, as a "fingerprint" of the document.  Then, calculate the Euclidean distance of that fingerprint against all other document fingerprints as represented in a Pandas dataframe.

### example of poor, but working model

```
from console import *
dsm = DocSimModel()
dsm.load_model('models/small_test.df')
tr = pd.read_pickle('models/altered_record')
tr.to_pickle('models/altered_record')
dsm.get_similar_records(tr)

In [9]: dsm.get_similar_records(tr)
Out[9]: 
[('8486c3e1ace17a216166a16841c026db', 6.5574385243020004),
 ('fb114973179d1d53fd41cb1218ea0f3d', 14.317821063276353),
 ('19a09791aaf9a2cb7e2468a6c1529fba', 14.7648230602334),
 ('0ff0e9ebd5555f429d16071a3cf8590d', 14.933184523068078),
 ('168b4d038f9e5ac9d477dc81068a5070', 15.033296378372908),
 ('c09ede896f2523c67c873fa82b52865d', 15.0996688705415),
 ('947572b34c703f190fca8f3e4d9cf0af', 15.427248620541512),
 ('97c1f9f0328a5b3427c2ef3b652494e3', 15.524174696260024),
 ('f502f90440e04794c60b50145f4baf8c', 16.093476939431081),
 ('b6d5a70525f377a65d19060cdfa90269', 17.0)]
```

### testing with raw MODS

Once you have a model created, we can shoehorn a test with a raw MODS record (make sure record is at same level of transformation as what was used to train the model).  The goal is to create a pandas Series object and pass this to the model for similarity checking.

```
# get MODS record as string
%cpaste
mods_string = '''[PASTE MODS HERE]'''

# get instance of RawRecord by passing an identifier and metadata string to parse
r = RawRecord('mystery_record', mods_string)

In [9]: type(r)
Out[9]: pydocsim.RawRecord

In [10]: r.m_as_char_vect_series()
Out[10]: 
0     29
1     31
2     11
3     21
4      1
5      4
6      0
7      0
8      2
9     34
a    127
b     25
c     63
d     38
e    130
f      9
g     43
h     68
i    112
j      0
k     18
l     59
m     35
n     94
o    108
p     42
q      0
r    106
s     68
t    157
u     26
v      9
w     96
x     12
y     28
z      1
Name: mystery_record, dtype: int64

# instantiate and load pre-made DocSimModel
dsm = DocSimModel
dsm.load_model('models/michigan.df')

# pass record for similarity check
dsm.get_similar_records(r.m_as_char_vect_series())

Out[16]: 
[('de78b3f09ab94100368e535c97c07ac7', 47.275786614291256),
 ('4f8d879eb8100f194c5d2c46f742958b', 48.959166659574592),
 ('1d3f74ae4f35101006bc84939dd92b0b', 50.695167422546305),
 ('c6ab93584bb99b61521be43eadce8ee1', 51.273774973177076),
 ('a226ce25f6fa746af36743e9a71c8ec8', 51.691391933280343),
 ('8758a248900c6912922c29d15f047e61', 51.913389409669641),
 ('0ea41add149ed2591a78db2c78fee692', 51.980765673468106),
 ('5761537507647801ed1b0d3542ba082b', 52.163205422979907),
 ('bad1a186e81f4c55cc1f4e6ebada6ce2', 52.249401910452526),
 ('702247799a87d13b1e921b09ba6bfee2', 52.268537381487917)]
```

Should have been:
[1905 Packard Model N engine-front elevation](https://dp.la/item/3fae4c81a975d5fd7c44a4ddbeac3af8)

Best guess at 47.275:
[1958 Packard station wagon, three-quarter front left view](https://dp.la/item/de78b3f09ab94100368e535c97c07ac7)



## LDA

Another approach is using GenSim and an LDA model to try and generate topics across a dataset that may be useful for finding that same record, even if slightly different.

### Train a model

Assuming already acquired a dataset as explained above.

```
from pydocsim import *

# get a reader
rr = ReaderRaw('data/michigan.json')

# instantiate and train model (where each stage is automatically saved to /models/[NAME].*)
dsm = DocSimModelLDA(reader=rr, name='michfull')
dsm.gen_corpora()
dsm.gen_lda()
dsm.gen_similarity_index()
```

### Load a record and check for similarity
```
from pydocsim import *

# if neccessary, reload dsm model
dsm = DocSimModelLDA(name='michfull')
dsm.load_corpora()
dsm.load_lda()
dsm.load_similarity_index()

# get reader and document to check
rr = ReaderRaw('data/michigan.json')
r = rr.get_next_dpla_record()

In [9]: dsm.get_similar_records(r)
Out[9]: 
[(0, 0.99999994),
 (17, 0.87552118),
 (18, 0.87552118),
 (33, 0.87552118),
 (16, 0.85225111),
 (5, 0.84634936),
 (15, 0.84494537),
 (28, 0.84336436),
 (12, 0.8423329),
 (19, 0.83408332),
 (41, 0.83208382),
 (30, 0.83183944),
 (44, 0.82675105),
 (46, 0.82443923),
 (38, 0.82100642),
 (13, 0.81981587),
 (11, 0.81447327),
 (39, 0.81388229),
 (37, 0.8057099),
 (36, 0.80004919)]
```

## Discussion

Still just exploratory at this point.  Both approaches have some interesting characteristics, pros, cons, etc.

Another interesting avenue might be to query against an ElasticSearch (ES) index, with the DPLA docs already indexed.  Because we won't have the metadata parsed as a DPLA doc yet, we can't query field-by-field, but we could query block of text by block of text.  Loop through extracted tokens, fire queries for each, save to dataframe, and calculate most probable document. 

## ElasticSearch

Assumes indexing has taken place (also method from this dsm)

```
from dpladocsim import *

# get subset of records
rr = ReaderRaw('data/michigan.json')
records = []
count = 0
for r in rr.dpla_record_generator():
    records.append(r)
    count += 1
    if count >= 1000:
        break

# load model
dsm = DocSimModelES(name='michfull')
misses = []
for i, record in enumerate(records):
    top_match = dsm.get_similar_records(record)[0]
    print(record.dpla_id, top_match)
    if record.dpla_id != top_match[0]:
        misses.append((i, record))
```

Accuracy, with not much tweaking, 223 misses / 1000, ~79%.






