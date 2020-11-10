#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 20:23:15 2020

@author: AJ
"""
import matplotlib
import matplotlib.pyplot as plt
import sympy as sym
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.interpolate import lagrange
import math
import re
from IPython.display import HTML
from math import exp
from math import factorial


def frange(start=0, stop=1, jump=0.1):
    return np.linspace(start, stop, num=int((stop-start)/jump+1))

from locale import localeconv
def dlen(foo):
    if isinstance(foo, int):
        return 0
    foo = str(foo)
    dec_pt = localeconv()['decimal_point']
    decrgx = re.compile("\d+(%s\d+)?e(-|\+)(\d+)" % dec_pt)
    if decrgx.search(foo):
        # still figuring this out
        raise NotImplementedError("e notation not implemented")
    else:
        digits = len(foo.split(dec_pt)[-1])
        return digits

def transpose(mat):
    mato = []
    for i in range(len(mat[0])):
        mato.append([])
        for j in range(len(mat)):
            mato[-1].append(mat[j][i])
    return mato


def print_Table_V(array, dec=[], title=''):
    if isinstance(dec, list):
        display(HTML(
            '<table><caption>' + title + '</caption><tr>{}</tr></table>'.format(
                '</tr>'.join('<td style="min-width:50px"><nobr>{}</nobr></td>'.format('</td><td style="min-width:50px">'.join(("{:,." + str(dec[r]) + "f}").format(round(_, dec[r])) if not isinstance(_, str) else _ for  _ in row)) for (r, row) in enumerate(array))
                )
            )
        )
    else:
        display(HTML(
            '<table><caption>' + title + '</caption><tr>{}</tr></table>'.format(
                '</tr>'.join('<td style="min-width:50px"><nobr>{}</nobr></td>'.format('</td><td style="min-width:50px">'.join(_ if isinstance(_, str) else ("{:,." + str(dec)+ "f}").format(_) for _ in row)) for (r, row) in enumerate(array))
                )
            )
        )
