# -*- coding: utf-8 -*-
"""
Created on Thu May 24 09:42:08 2018

@author: GKep
"""

# Pydex can be run in severeal ways 

# 1) Work with certain Data frame
import pandas as pd
import numpy as np
import arcelormittal as am

df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
am.dex(df)

# 2) Work with multiple Data frames

import pandas as pd
import numpy as np
import arcelormittal as am

df = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
df2 = pd.DataFrame(np.random.randn(100, 4), columns=list('ABCD'))
dic = {'df':df, 'df2':df2}
am.dex(dic)

# If you directly run explore.py script (ex. in Anaconda) you will already see the dataframes from environment

# In other cases you can run explore.py script and load xlsx, pkl files when needed





