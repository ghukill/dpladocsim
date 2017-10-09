# dpladocsim
Compare two JSON records from DPLA 

# download DPLA datasets

For examples to run, download [Michigan data](https://dpla-provider-export.s3.amazonaws.com/2017/10/michigan.json.gz) from DPLA and unzip to `/data` directory.

# example of poor, but working model

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

# testing with raw MODS

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

Not quite.  Work to be done.



