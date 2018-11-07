# q2-extractor

This small utility exists to wrangle QIIME data and metadata out of artifacts and into formats needed for other applications.

### Installation

```bash

git clone https://github.com/beiko-lab/q2-extractor.git
cd q2-extractor
pip install .
```

### Quickstart

```python

import q2_extractor as q2e

extractor = q2e.Extractor.q2Extractor("/path/to/artifact.qza")

#Get some basic information on the object
print(str(extractor))

#Pull out the data
data = extractor.extract_data()

mh_extractor = q2e.MetaHCRExtractor.MetaHCRExtractor("/path/to/artifact.qza")

mh_data = mh_extractor.extract_data()
```
