# dpladocsim
Compare two JSON records from DPLA 

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
