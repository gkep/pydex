# -*- coding: utf-8 -*-
"""
Created on Thu May 24 09:42:08 2018

@author: GKep
"""

# Pydex can be run in severeal ways 

# Work with certain Data frame
import pandas as pd
import numpy as np
import arcelormittal as am

df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
am.dex(df)

# If you directly run explore.py script (ex. in Anaconda) you will already see the dataframes from environment

# In other case you can run explore.py script and load xlsx, pkl files when needed





