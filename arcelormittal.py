# -*- coding: utf-8 -*-
"""
ArcelorMittal Functions

2018-02-08 - luc.vandeputte@arcelormittal.com
"""

#%% Import statements
#from data import data
from explore import explore
import settings
import pandas as pd
import inspect

def dex(data):
    # Check if 'dataframes' attribute exists and initialize if necessary
    if not hasattr(settings, 'dataframes'):
        settings.init()
    # Handle different types of data argument
    if isinstance(data, dict):
        settings.dataframes = {k: v for k, v in data.items() if isinstance(eval(k), pd.core.frame.DataFrame)}
    elif isinstance(data, pd.core.frame.DataFrame):
        settings.dataframes[retrieve_name(data)] = data
    # Run data explorer
    app = explore() # <-- data
    app.master.title('PyExplore - Python Data Exploration Tool - version 0.1')
    app.mainloop()

def retrieve_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var][0]