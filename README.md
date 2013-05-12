This collection contains several functions and class utilities that I find useful for data analysis in Python, particularly with pandas and sklearn.

Here's an example of using `StringEncoder` with data from [cms.gov](http://www.cms.gov/Research-Statistics-Data-and-Systems/Statistics-Trends-and-Reports/Medicare-Provider-Charge-Data/index.html).
```
>>> import pandas as pd
>>> import datatools as dt
>>> from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
>>> from sklearn.cross_validation import ShuffleSplit
>>> from sklearn.metrics import confusion_matrix
>>> df = pd.read_csv('Medicare_Provider_Charge_Inpatient_DRG100_FY2011.csv')
>>> df.columns = dt.under_col(df)
>>> df.head()
                                      DRG_Definition  Provider_Id  \
0           039 - EXTRACRANIAL PROCEDURES W/O CC/MCC        10001
1  057 - DEGENERATIVE NERVOUS SYSTEM DISORDERS W/...        10001
2  064 - INTRACRANIAL HEMORRHAGE OR CEREBRAL INFA...        10001
3  065 - INTRACRANIAL HEMORRHAGE OR CEREBRAL INFA...        10001
4  066 - INTRACRANIAL HEMORRHAGE OR CEREBRAL INFA...        10001

                      Provider_Name Provider_Street_Address Provider_City  \
0  SOUTHEAST ALABAMA MEDICAL CENTER  1108 ROSS CLARK CIRCLE        DOTHAN
1  SOUTHEAST ALABAMA MEDICAL CENTER  1108 ROSS CLARK CIRCLE        DOTHAN
2  SOUTHEAST ALABAMA MEDICAL CENTER  1108 ROSS CLARK CIRCLE        DOTHAN
3  SOUTHEAST ALABAMA MEDICAL CENTER  1108 ROSS CLARK CIRCLE        DOTHAN
4  SOUTHEAST ALABAMA MEDICAL CENTER  1108 ROSS CLARK CIRCLE        DOTHAN

...
```
Some columns above have data as strings, which breaks a lot of sklearn's algorithms, since it is not able to use factors. This can be shown by trying to fit on a new column which indicates whether an entry is more or less expensive than the median:
```
>>> df['Expensive'] = (df.Average_Total_Payments > df.Average_Total_Payments.median()).astype(int)
>>> xcols = df.columns[:-2]
>>> xcols.tolist()
['DRG_Definition', 'Provider_Id', 'Provider_Name', 'Provider_Street_Address', 'Provider_City', 'Provider_State', 'Provider_Zip_Code', 'Hospital_Referral_Region_Description', 'Total_Discharges', 'Average_Covered_Charges']
>>> Y = df.Expensive
>>> X = df[xcols]
>>> try:
...     et = ExtraTreesClassifier(n_jobs=-1, bootstrap=1, oob_score=1, compute_importances=1).fit(X, Y)
... except ValueError, e:
...     print 'ValueError:', e
...
ValueError: could not convert string to float: TX - Houston
```
To get around this, we can create a new string encoder and apply it in place to the data frame, which will convert all of the text column values to integers:
```
>>> se = dt.StringEncoder(df, null_fills=False).apply(df)
>>> se
StringEncoder({'Provider_City', 'Provider_Street_Address', ...})
>>> df.head()
   DRG_Definition  Provider_Id  Provider_Name  Provider_Street_Address  \
0               0        10001              0                        0
1               1        10001              0                        0
2               2        10001              0                        0
3               3        10001              0                        0
4               4        10001              0                        0

...
```

Now we can train a random forest classifier, and see how well its predictions are on the test set.
```
>>> train_ix, test_ix = iter(ShuffleSplit(len(df), n_iter=1, test_size=.25)).next()
>>> Y = df.Expensive
>>> X = df[xcols]
>>> et = ExtraTreesClassifier().fit(X.ix[train_ix], Y.ix[train_ix])
>>> pred = et.predict(X.ix[test_ix])
>>> confusion_matrix(Y.ix[test_ix], pred)
array([[18859,  1529],
       [ 3159, 17220]])
>>> (Y.ix[test_ix] == pred).mean()
0.88500502857703534
```

To get the original data frame values back, just use the `unapply` method on the string encoder object:

```
>>> se.unapply(df)
StringEncoder({'Provider_City', 'Provider_Street_Address', 'Hospital_Referral_Region_Description', 'Provider_State', 'Provider_Name', 'DRG_Definition'})
>>> df.head()
                                      DRG_Definition  Provider_Id  \
0           039 - EXTRACRANIAL PROCEDURES W/O CC/MCC        10001
1  057 - DEGENERATIVE NERVOUS SYSTEM DISORDERS W/...        10001
2  064 - INTRACRANIAL HEMORRHAGE OR CEREBRAL INFA...        10001
3  065 - INTRACRANIAL HEMORRHAGE OR CEREBRAL INFA...        10001
4  066 - INTRACRANIAL HEMORRHAGE OR CEREBRAL INFA...        10001

                      Provider_Name Provider_Street_Address Provider_City  \
0  SOUTHEAST ALABAMA MEDICAL CENTER  1108 ROSS CLARK CIRCLE        DOTHAN
1  SOUTHEAST ALABAMA MEDICAL CENTER  1108 ROSS CLARK CIRCLE        DOTHAN
2  SOUTHEAST ALABAMA MEDICAL CENTER  1108 ROSS CLARK CIRCLE        DOTHAN
3  SOUTHEAST ALABAMA MEDICAL CENTER  1108 ROSS CLARK CIRCLE        DOTHAN
4  SOUTHEAST ALABAMA MEDICAL CENTER  1108 ROSS CLARK CIRCLE        DOTHAN
...
```