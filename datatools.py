from __future__ import division
import itertools
import numpy as np
import pandas as pd
import re
from operator import itemgetter
import sys
from collections import defaultdict


def getsize(obj):
    if type(obj) not in (pd.DataFrame, np.ndarray):
        return sys.getsizeof(obj)
    try:
        return obj.values.nbytes * 8
    except AttributeError:
        return obj.nbytes * 8


def sizeof_fmt(num):
    for x in ['bytes', 'KB', 'MB', 'GB']:
        if num < 1024.0:
            return "%3.1f %s" % (num, x)
        num /= 1024.0
    return "%3.1f %s" % (num, 'TB')


def get_sizes(locs, n=30):
    """Returns DataFrame with sizes of objects in namespace.
    Call with sz = get_sizes(locals())
    """
    objs = {k: getsize(v) for k, v in locs.iteritems()}
    sizes = sorted(objs.iteritems(), key=itemgetter(1), reverse=1)
    sizes = pd.DataFrame(sizes, columns=['Vars', 'Size'])
    sizes['Human'] = sizes.Size.map(sizeof_fmt)
    return sizes if not n else sizes[:n]


def under_col(df):
    "Return column names of a dataframe, with spaces replaced by underscores"
    return ['_'.join(col.split()) for col in df.columns]


class StringEncoder(dict):
    """Class to encode strings with integers, with ability to
    retrieve them. Useful for encoding features from Pandas
    DataFrames for use with sklearn.

    Usage:
    To convert `df` in place:
    >>> se = StringEncoder(df).apply(df)

    To revert back to `df`'s original values:
    >>> se = se.unapply(df)
    """

    def __init__(self, df=None, dtypes=['object'], override=True, columns=None,
                 null_fills=None, verbose=False):
        class _ReverseEncoder(dict):
            pass
        self.names = {}
        self.override = override
        self.inv = _ReverseEncoder()
        self.dtypes = [np.dtype(dt) for dt in dtypes]
        self.verbose = verbose
        if null_fills is None:
            self.null_fills = defaultdict(int)
            self.null_fills.update({np.dtype('object'): ''})
        else:
            self.null_fills = null_fills

        if df is not None:
            if self.null_fills != False:
                null_cols = df.columns[df.apply(lambda s: s.isnull().sum() != 0)]
                for col in null_cols:
                    dt = df[col].dtype
                    df[col] = df[col].fillna(self.null_fills[dt])

            cols = columns or df.dtypes[df.dtypes.isin(self.dtypes)].index
            self.add_columns(df, cols)

    def _safe_setattr(self, obj, attr, val):
        "Tries to set attribute, but prints error if not possible."
        pat = re.compile(r"^[A-Za-z_][A-Za-z\d_]*")
        if getattr(obj, attr, None) is not None:
            print "Error: Object already has attribute {}".format(attr)
        elif not pat.search(attr):
            print "Error: Attribute {} contains invalid sequence".format(attr)
        else:
            setattr(obj, attr, val)

    def __repr__(self):
        return "StringEncoder({{'{}'}})".format("', '".join(self))

    def add_column(self, df, col, override=False):
        "Takes dataframe and column name(string)"
        if col in self.names:
            print "Column already in use"
            if not (self.override or override):
                return
        #self.names.add(col)
        dct = dict(zip(df[col].unique(), itertools.count()))
        rev = {v: k for k, v in dct.iteritems()}
        self.names.update({col: dct})
        self[col] = dct
        self._safe_setattr(self, col, dct)
        self.inv[col] = rev
        self._safe_setattr(self.inv, col, rev)
        return self

    def add_columns(self, df, cols, **kwargs):
        for col in cols:
            self.add_column(df, col, **kwargs)
        return self

    def is_converted(self, df):
        N = len(self)
        inv = []
        orig = []
        ambig = []
        self.check_extras(df)
        for col in filter(lambda x: x in df, self.columns):
            if set(df[col]) == set(self[col]):
                orig.append(col)
            elif set(df[col]) == set(self[col].values()):
                inv.append(col)
            else:
                ambig.append(col)
        if len(orig) == N:
            return False
        elif len(inv) == N:
            return True
        else:
            print "Warning, ambiguous columns '{}'".format("', '".join(ambig))
            return True

    def apply(self, df, inv=False):
        "In-place modification of df"
        self.check_extras(df)
        for col in filter(lambda x: x in df, self.columns):
            dct = self.inv[col] if inv else self[col]
            df[col] = df[col].map(dct)
        return self

    def unapply(self, df):
        "In-place restoring of df's original values."
        return self.apply(df, inv=True)

    def check_extras(self, df):
        extras = set(self.columns) - set(df.columns)
        if extras != set() and self.verbose:
            print "Warning, columns '{}' not in Data Frame".format("', '".join(extras))

    def fillna():
        pass

    @property
    def columns(self):
        return self.keys()
